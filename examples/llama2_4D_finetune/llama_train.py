################################################################################
#
# Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################

import argparse
import os
import math
import inspect

import torch
import torch.distributed as dist

from vescale.dmodule import parallelize_module
from vescale.ddp.distributed_data_parallel import DistributedDataParallel as DDP
from vescale.optim.distributed_optimizer import DistributedOptimizer
from vescale.optim.base_optimizer import BasicOptimizer
from vescale.devicemesh_api import VESCALE_DEVICE_MESH
from vescale.dtensor.random import manual_seed

from transformers import LlamaForCausalLM

from sharding_plan import llama2_plan

from data_loader import DataLoader


def estimate_llama2(config, bsz, sqence_length):
    embed = 4 * bsz * sqence_length * config.hidden_size
    ff = 3 * 2 * config.hidden_size * config.intermediate_size * bsz * sqence_length
    attn_qkv = 2 * bsz * sqence_length * config.hidden_size * 3 * config.hidden_size
    attn_mask = 2 * sqence_length * config.hidden_size
    attn_proj = 2 * config.hidden_size * config.intermediate_size * bsz * sqence_length
    attn = attn_qkv + attn_mask + attn_proj
    return embed + (ff + attn) * config.num_hidden_layers


def run_llama2(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        device = f"cuda:{rank}"
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
        VESCALE_DEVICE_MESH.init_device_mesh(device, (args.dp, args.tp), mesh_dim_names=["DP", "TP"])
        device_mesh = VESCALE_DEVICE_MESH.get()
        dp_rank = dist.get_rank() // args.tp
        torch.random.manual_seed(0)
        torch.cuda.random.manual_seed_all(0)
        manual_seed(0, device_mesh)
    else:
        local_rank = 0
        rank = 0
        device = f"cuda:{0}"
        device_mesh = None
        torch.cuda.set_device(device)
        dp_rank = 0
        torch.random.manual_seed(0)
        torch.cuda.random.manual_seed_all(0)

    ptdtype = {
        "float32": torch.float,
        "bfloat16": torch.bfloat16,
    }[args.dtype]

    model = LlamaForCausalLM.from_pretrained("openlm-research/open_llama_3b", torch_dtype=ptdtype)
    llama_config = model.config
    if rank == 0:
        print(model)
        print(llama_config)
        print(ptdtype)
    model.to(ptdtype)

    if world_size > 1:
        model = parallelize_module(
            model,
            VESCALE_DEVICE_MESH["TP"],
            llama2_plan,
            factory=True,
        )

        model = DDP(
            model,
            VESCALE_DEVICE_MESH["DP"],
            accumulate_allreduce_grads_in_fp32=False,
            overlap_grad_reduce=True,
            use_distributed_optimizer=args.use_DO,
        )
    else:
        model.to(device)

    def configure_optimizers(model, weight_decay, learning_rate, betas):
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and (world_size == 1 or device_mesh.device_type == "cuda")
        extra_args = dict(fused=True) if use_fused else dict()
        base_optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        if world_size == 1 or dist.get_rank() == 0:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
            print(f"using fused AdamW: {use_fused}")
        # + + + Initialize a ZeRO-2 optimizer using veScale API
        if args.use_DO and world_size > 1:
            optimizer = DistributedOptimizer(
                base_optimizer,
                models=[model],
                clip_grad=args.grad_clip,
                grad_to_fp32=False,
                overlap_param_gather=False,
            )
        elif world_size > 1:
            optimizer = BasicOptimizer(base_optimizer, models=model)
        else:
            optimizer = base_optimizer
        return optimizer

    doptimizer = configure_optimizers(model, args.weight_decay, args.lr, (0.9, 0.95))

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < args.warmup_iters:
            return args.lr * it / args.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > args.lr_decay_iters:
            return args.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - args.warmup_iters) / (args.lr_decay_iters - args.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return args.min_lr + coeff * (args.lr - args.min_lr)

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ["train", "val"]:
            factor = 1
            losses = torch.zeros(args.eval_iters // factor).to(device)
            for k in range(args.eval_iters // factor):
                X, Y = data_loader.get_batch(split, args.bsz * factor, factor * args.bsz // args.dp)
                loss = model(X, labels=Y).loss
                if world_size > 1:
                    losses[k] = loss.to_local().item()
                else:
                    losses[k] = loss.item()
            if world_size > 1:
                dist.all_reduce(losses)
            out[split] = losses.mean() / world_size
        model.train()
        return out

    data_loader = DataLoader(args.dataset, args.seqlen, device_mesh, dp_rank)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    model.train()
    for iter in range(args.max_iters):
        if iter % args.eval_interval == 0:
            out = estimate_loss()
            if world_size == 1 or dist.get_rank() == 0:
                print(f"iter {iter} train_loss: {out['train']:.6f} val_loss: {out['val']:.6f}")
        # determine and set the learning rate for this iteration
        lr = get_lr(iter) if args.decay_lr else args.lr
        for param_group in doptimizer.param_groups if world_size == 1 else doptimizer.optimizer.param_groups:
            param_group["lr"] = lr
        # load a batch of training data
        X, Y = data_loader.get_batch("train", args.bsz, args.bsz // args.dp)

        start_epoch = torch.cuda.Event(enable_timing=True)
        end_epoch = torch.cuda.Event(enable_timing=True)
        start_epoch.record()
        if world_size > 1:
            model.zero_grad_buffer()
        loss = model(X, labels=Y).loss
        loss.backward()
        grad_norm = -1
        if world_size == 1 and args.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if world_size > 1:
            model.finish_grad_sync()
        if world_size > 1 and args.grad_clip > 0:
            grad_norm = doptimizer.step()
        else:
            doptimizer.step()
        doptimizer.zero_grad(set_to_none=True)
        end_epoch.record()
        torch.cuda.synchronize()
        epoch_t = start_epoch.elapsed_time(end_epoch)
        if world_size > 1:
            loss_val = loss.to_local()
            dist.all_reduce(loss_val)
            loss_val = loss_val.item() / world_size
        else:
            loss_val = loss.item()
        if world_size == 1 or dist.get_rank() == 0:
            print(f"iter {iter} loss {loss_val:.6f} |g| {grad_norm:.6f} lr {lr:.6f} fwd/bwd_t {epoch_t:.2f}ms")
    end.record()
    torch.cuda.synchronize()
    exec_t = start.elapsed_time(end) / 1000 / args.max_iters
    # masure mfu
    if rank == 0:
        total_flops = {
            "A100": {
                "bfloat16": 312 * (10**12),
                "float32": 19.5 * (10**12),
            },
            "H100": {
                "bfloat16": 1000 * (10**12),
                "float32": 312 * (10**12),
            },
        }["A100"][args.dtype]
        if world_size > 1:
            total_flops *= world_size
        print(f"1 iter time: {exec_t}")
        llama2_flops = estimate_llama2(llama_config, args.bsz, args.seqlen)
        print(f"fwd llama2 flops: {llama2_flops}")
        # bwd ~= fwd * 2
        print("mfu:", llama2_flops * 3 * 100 / exec_t / total_flops)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser()
    # Training Meta
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--max_iters", type=int, default=2)
    parser.add_argument("--bsz", type=int, default=16)
    parser.add_argument("--seqlen", type=int, default=256)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--dataset", type=str, default="shakespeare")
    parser.add_argument("--eval_iters", type=int, default=1)
    parser.add_argument("--eval_interval", type=int, default=400)

    # Optimizer related
    parser.add_argument("--use_DO", type=bool, default=True)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--decay_lr", type=bool, default=False)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--lr_decay_iters", type=int, default=5000)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--grad_clip", type=float, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    return parser


if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()
    run_llama2(args)
