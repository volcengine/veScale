################################################################################
#
# Copyright 2023 ByteDance Ltd. and/or its affiliates. All rights reserved.
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
from vescale.dtensor.placement_types import InterleavedShard
from vescale.moe import parallelize_experts, MoEOptimizer
from vescale.ddp.distributed_data_parallel import DistributedDataParallel as DDP
from vescale.optim.distributed_optimizer import DistributedOptimizer
from vescale.optim.base_optimizer import BasicOptimizer
from vescale.devicemesh_api import VESCALE_DEVICE_MESH
from vescale.dtensor.random import manual_seed
from vescale import DTensor

from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM, MixtralSparseMoeBlock, MixtralModel
from transformers.models.mixtral.configuration_mixtral import MixtralConfig
from sharding_plan import mixtral_plan

from data_loader import DataLoader


class Net(torch.nn.Module):
    def __init__(self, mixtral_config):
        super().__init__()
        self.mixtral_model = MixtralForCausalLM(mixtral_config)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, labels):
        logits = self.mixtral_model(input_ids).logits
        logits = logits.flatten(end_dim=-2)
        labels = labels.flatten()
        loss = self.loss_fn(logits, labels)
        return loss


def wrap_moe_block(forward_func):
    old_func_dict = {}

    def _pre_forward_overload():
        nonlocal old_func_dict
        old_func_dict = {}
        old_func_dict["where"] = torch.where
        old_func_dict["index_select"] = DTensor.__getitem__

        def local_where(*args, **kwargs):
            output = old_func_dict["where"](*args, **kwargs)
            if isinstance(output, DTensor):
                return output.to_local()
            elif isinstance(output, torch.Tensor):
                return output
            elif isinstance(output, tuple):
                output_list = []
                for t in output:
                    if isinstance(t, DTensor):
                        output_list.append(t.to_local())
                    else:
                        output_list.append(t)
                return tuple(output_list)
            else:
                raise NotImplementedError

        def local_index_select(*args, **kwargs):
            return old_func_dict["index_select"](args[0].to_local(), *args[1:], **kwargs)

        torch.where = local_where
        DTensor.__getitem__ = local_index_select

    def _post_forward_overload():
        nonlocal old_func_dict
        torch.where = old_func_dict["where"]
        DTensor.__getitem__ = old_func_dict["index_select"]

    def forward(*args, **kwargs):
        _pre_forward_overload()
        output = forward_func(*args, **kwargs)
        _post_forward_overload()
        return output

    return forward


def estimate_mixtral(config, bsz, sqence_length):
    embed = 4 * bsz * sqence_length * config.hidden_size
    # MixtralMoE consists of 3 linear layers.
    ff = 3 * 2 * config.num_experts_per_tok * config.hidden_size * config.intermediate_size * bsz * sqence_length
    # GQA
    head_size = config.hidden_size // config.num_attention_heads
    attn_q = 2 * bsz * sqence_length * config.hidden_size * config.hidden_size
    attn_kv = 2 * 2 * bsz * sqence_length * config.hidden_size * config.num_key_value_heads * head_size
    attn_mask = 2 * sqence_length * config.hidden_size
    attn_proj = 2 * config.hidden_size * config.hidden_size * bsz * sqence_length
    attn = attn_q + attn_kv + attn_mask + attn_proj
    return embed + (ff + attn) * config.num_hidden_layers


def run_mixtral(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        device = f"cuda:{local_rank}"
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

    mixtral_config = MixtralConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
    )

    if world_size > 1:
        model = Net(mixtral_config)
        model.to(ptdtype)

        experts_module_name = r"mixtral_model.model.layers.\d+.block_sparse_moe.experts"

        factory = {
            MixtralSparseMoeBlock: {torch.zeros: [InterleavedShard(0, args.bsz // args.dp)]},
            MixtralModel: True,
        }
        MixtralSparseMoeBlock.forward = wrap_moe_block(MixtralSparseMoeBlock.forward)

        model = parallelize_module(
            model,
            VESCALE_DEVICE_MESH["TP"],
            mixtral_plan,
            factory=factory,
        )

        param_to_ignore = [param_name for param_name, _ in model.named_parameters() if "experts" in param_name]

        model = DDP(
            model,
            VESCALE_DEVICE_MESH["DP"],
            accumulate_allreduce_grads_in_fp32=False,
            use_distributed_optimizer=args.use_DO,
            param_to_ignore=param_to_ignore,
        )

        moe_config = {
            "num_layers": mixtral_config.num_hidden_layers,
            "num_experts": mixtral_config.num_local_experts,
            "num_devices": torch.distributed.get_world_size(),
        }

        model = parallelize_experts(
            model,
            experts_module_name,
            config=moe_config,
        )
    else:
        model = Net(mixtral_config).to(device)
        model.to(ptdtype)
    print(f"rank {rank} cuda.rng_state {torch.cuda.get_rng_state().view(torch.int64)}")

    def configure_optimizers(model, weight_decay, learning_rate, betas):
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2 and "experts" not in n]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2 and "experts" not in n]
        moe_params = [p for n, p in param_dict.items() if "experts" in n]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        num_moe_params = sum(p.numel() for p in moe_params)
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and (world_size == 1 or device_mesh.device_type == "cuda")
        extra_args = dict(fused=True) if use_fused else dict()
        base_optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        if world_size == 1 or dist.get_rank() == 0:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
            print(f"experts parameter tensors: {len(moe_params)}, with {num_moe_params:,} parameters")
            print(f"using fused AdamW: {use_fused}")
        # + + + Initialize a ZeRO-2 optimizer using veScale API
        if args.use_DO and world_size > 1:
            optimizer = DistributedOptimizer(
                base_optimizer,
                models=[model],
                clip_grad=args.grad_clip,
                grad_to_fp32=False,
            )
            moe_optimizer = MoEOptimizer(
                torch.optim.AdamW,
                clip_grad=args.grad_clip,
                param_buffer=model.moe_param_buffer,
                lr=learning_rate,
                betas=betas,
                weight_decay=weight_decay,
                **extra_args,
            )
        elif world_size > 1:
            optimizer = BasicOptimizer(base_optimizer, models=model)
            moe_optimizer = MoEOptimizer(
                torch.optim.AdamW,
                clip_grad=args.grad_clip,
                param_buffer=model.moe_param_buffer,
                lr=learning_rate,
                betas=betas,
                weight_decay=weight_decay,
                **extra_args,
            )
        else:
            optimizer = base_optimizer
            moe_optim_groups = [
                {"params": moe_params, "weight_decay": weight_decay},
            ]
            moe_optimizer = torch.optim.AdamW(moe_optim_groups, lr=learning_rate, betas=betas, **extra_args)
        return optimizer, moe_optimizer

    # TODO: wrap up them into a single optimizer
    doptimizer, moe_optimizer = configure_optimizers(model, args.weight_decay, args.lr, (0.9, 0.95))

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < args.warmup_iters:
            return args.lr * it / args.warmup_iters
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - args.warmup_iters) / (args.max_iters - args.warmup_iters)
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
                loss = model(X, Y)
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
        # determine and set the learning rate for this iteration
        lr = get_lr(iter) if args.decay_lr else args.lr
        for param_group in doptimizer.param_groups if world_size == 1 else doptimizer.optimizer.param_groups:
            param_group["lr"] = lr
        for param_group in moe_optimizer.param_groups:
            param_group["lr"] = lr
        # load a batch of training data
        X, Y = data_loader.get_batch("train", args.bsz, args.bsz // args.dp)

        start_epoch = torch.cuda.Event(enable_timing=True)
        end_epoch = torch.cuda.Event(enable_timing=True)
        start_epoch.record()
        if world_size > 1:
            model.zero_grad_buffer()
        loss = model(X, Y)
        loss.backward()
        grad_norm = -1
        if world_size == 1 and args.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if world_size > 1:
            model.finish_grad_sync()
        if world_size > 1 and args.grad_clip > 0:
            grad_norm = doptimizer.step()
            moe_optimizer.step()
        else:
            doptimizer.step()
            moe_optimizer.step()
        doptimizer.zero_grad(set_to_none=True)
        moe_optimizer.zero_grad(set_to_none=True)
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
        mixtral_flops = estimate_mixtral(mixtral_config, args.bsz, args.seqlen)
        print(f"fwd llama2 flops: {mixtral_flops}")
        # bwd ~= fwd * 2
        print("mfu:", mixtral_flops * 3 * 100 / exec_t / total_flops)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser()
    # Training Meta
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--max_iters", type=int, default=2)
    parser.add_argument("--bsz", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=256)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--dataset", type=str, default="shakespeare")
    parser.add_argument("--eval_iters", type=int, default=1)
    parser.add_argument("--eval_interval", type=int, default=400)

    # Model config
    parser.add_argument("--vocab_size", type=int, default=50304)
    parser.add_argument("--hidden_size", type=int, default=384)
    parser.add_argument("--intermediate_size", type=int, default=1536)
    parser.add_argument("--num_hidden_layers", type=int, default=2)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--num_key_value_heads", type=int, default=8)

    # Optimizer related
    parser.add_argument("--use_DO", type=bool, default=True)
    parser.add_argument("--decay_lr", type=bool, default=True)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--grad_clip", type=float, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    return parser


if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()
    run_mixtral(args)
