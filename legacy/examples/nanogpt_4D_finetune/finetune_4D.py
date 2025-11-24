################################################################################
# Copyright (c) 2022 Andrej Karpathy

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

import os
import time
import math
import pickle
import inspect

import numpy as np
import torch
from torch.distributed import broadcast, all_reduce, barrier, init_process_group, destroy_process_group, get_rank

from model import GPTConfig, GPT
from vescale.devicemesh_api import VESCALE_DEVICE_MESH

from vescale import distribute_tensor
from vescale.dmodule.api import parallelize_module
from vescale.dtensor.placement_types import Replicate
from vescale.ddp.distributed_data_parallel import DistributedDataParallel as DDP
from vescale.optim.distributed_optimizer import DistributedOptimizer
from vescale.optim.base_optimizer import BasicOptimizer, GradOptimizerHookBase
from sharding_plan import nanoGPT_plan, nanoGPT_plan_dist_dropout
import vescale
from vescale.dtensor.random import manual_seed

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = "out"
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
init_from = "scratch"  # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False  # disabled by default
wandb_project = "owt"
wandb_run_name = "gpt2"  # 'run' + str(time.time())
# data
dataset = "openwebtext"
gradient_accumulation_steps = 5 * 8  # used to simulate larger batch sizes
batch_size = 12  # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4  # max learning rate
max_iters = 600000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 2000  # how many steps to warm up for
lr_decay_iters = 600000  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = "nccl"  # 'nccl', 'gloo', etc.
# system
device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = (
    "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
)  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True  # use PyTorch 2.0 to compile the model to be faster
use_DO = True
dp_size = 4
tp_size = 1
DDP_grads_in_fp32 = True
save_checkpoint_path = "./nanogpt_checkpoint_dir"
load_checkpoint_path = ""
use_dist_dropout = True
async_checkpoint = False
broadcast_checkpoint = False
config = {}


def main():
    world_size = dp_size * tp_size
    local_batch_size = batch_size

    wandb_run_name = f"{world_size}gpu-dp{dp_size}-tp{tp_size}"

    # various inits, derived attributes, I/O setup
    # ddp = world_size > 1
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        init_process_group(backend=backend, world_size=world_size, rank=rank)
        # + + + VeScale API below
        VESCALE_DEVICE_MESH.init_device_mesh(device, (dp_size, tp_size), mesh_dim_names=["DP", "TP"])
        mesh = VESCALE_DEVICE_MESH.get()
        # + + + VeScale API above
        ddp_rank = get_rank() // tp_size
    else:
        rank = 0
        ddp_rank = 0
        device = f"cuda:{rank}"
        torch.cuda.set_device(device)
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    master_process = rank == 0  # this process will do logging, checkpointing etc.
    assert batch_size % dp_size == 0
    local_batch_size = batch_size // dp_size
    tokens_per_iter = gradient_accumulation_steps * dp_size * local_batch_size * block_size
    if master_process:
        print(f"tokens per iteration will be: {tokens_per_iter:,}")
        print(f"Use new distributed random: {os.environ.get('VESCALE_SINGLE_DEVICE_RAND', '1')}")

    if master_process:
        os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(1337)
    # + + + VeScale API below
    manual_seed(1337, mesh)
    # + + + VeScale API above
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]

    # poor man's data loader
    data_dir = os.path.join("data", dataset)

    """
    Deterministic data loader for loss match:
    This data loader ensures that the mini-batch sampling has identical behavior no matter how many GPUs are used.
    In particular, at each training iteration, each rank samples a batch of indices under the identical RNG state.
    Then, each Data Parallelism (DP) rank takes the corresponding subset of indices and fetches the corresponding sequences from the dataset.
    """

    def get_batch(split, bsz=batch_size, lbsz=local_batch_size):
        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        if split == "train":
            data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
        else:
            data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
        ix = torch.randint(len(data) - block_size, (bsz,)).to(device)
        if world_size > 1:
            broadcast(ix, src=0, async_op=False)
        ix = torch.split(ix, lbsz)[ddp_rank]
        x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix])
        if device_type == "cuda":
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        # + + + VeScale API below
        if ddp:
            x = distribute_tensor(x, VESCALE_DEVICE_MESH["TP"], [Replicate()])
            y = distribute_tensor(y, VESCALE_DEVICE_MESH["TP"], [Replicate()])
        # + + + VeScale API above
        return x, y

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9

    # attempt to derive vocab_size from the dataset
    meta_path = os.path.join(data_dir, "meta.pkl")
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        meta_vocab_size = meta["vocab_size"]
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

    # model init
    model_args = dict(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        bias=bias,
        vocab_size=None,
        dropout=dropout,
    )  # start with model_args from command line
    if init_from == "scratch":
        # init a new model from scratch
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        if meta_vocab_size is None:
            print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
        model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 50304
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif init_from.startswith("gpt2"):
        print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
        # initialize from OpenAI GPT-2 weights
        override_args = dict(dropout=dropout)
        if init_from.startswith("gpt2"):
            model = GPT.from_pretrained(init_from, override_args)
        else:
            model = GPT.from_pretrained("gpt2", override_args)
        # read off the created config params, so we can store them into checkpoint correctly
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            model_args[k] = getattr(model.config, k)
    elif init_from == "resume":
        print(f"Resuming the training process from: {load_checkpoint_path}")
        # determine the vocab size we'll use for from-scratch training
        if meta_vocab_size is None:
            print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
        model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 50304
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    # crop down the model block size if desired, using model surgery
    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_args["block_size"] = block_size  # so that the checkpoint will have the right value
    model.to(ptdtype)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=True) if dtype == "float16" else None

    # compile the model
    if compile:
        print("WARNING: veScale does not support model compilation")
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)  # requires PyTorch 2.0

    # + + + parallelize the model and wrap it with DDP using veScale APIs
    if ddp:
        model = parallelize_module(
            model, VESCALE_DEVICE_MESH["TP"], nanoGPT_plan_dist_dropout if use_dist_dropout else nanoGPT_plan
        )
        model = DDP(
            model,
            data_pg_or_device_mesh=VESCALE_DEVICE_MESH["DP"],
            accumulate_allreduce_grads_in_fp32=DDP_grads_in_fp32,
            overlap_grad_reduce=False,
            use_distributed_optimizer=use_DO,
        )

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
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        base_optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        # + + + Initialize a ZeRO-2 optimizer using veScale API
        if use_DO and ddp:
            optimizer = DistributedOptimizer(
                base_optimizer,
                models=[model],
                clip_grad=grad_clip,
                overlap_param_gather=False,
                grad_to_fp32=DDP_grads_in_fp32,
            )
        elif ddp:
            optimizer = BasicOptimizer(base_optimizer, models=model, grad_hook=GradOptimizerHookBase)
        else:
            optimizer = base_optimizer

        return optimizer

    # optimizer
    optimizer = configure_optimizers(model, weight_decay, learning_rate, (beta1, beta2))

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ["train", "val"]:
            factor = 1
            losses = torch.zeros(eval_iters // factor).to(device)
            for k in range(eval_iters // factor):
                X, Y = get_batch(split, batch_size * factor, local_batch_size * factor)
                logits, loss = model(X, Y)
                if ddp:
                    losses[k] = loss.to_local().item() / world_size
                else:
                    losses[k] = loss.item() / world_size
            if ddp:
                all_reduce(losses)
            out[split] = losses.mean()
        model.train()
        return out

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)

    # logging
    if wandb_log and master_process:
        import wandb

        wandb_run_name = f"vescale-{wandb_run_name}"
        global config
        wandb.init(project=wandb_project, name=wandb_run_name, config=config)

    # Load checkpoint
    # + + + VeScale Load checkpoint
    if load_checkpoint_path:
        checkpoint_state = {"model": model, "optimizer": optimizer}
        vescale.checkpoint.load(load_checkpoint_path, checkpoint_state, broadcast_checkpoint=broadcast_checkpoint)
    # + + + VeScale API above
    # training loop
    X, Y = get_batch("train")  # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0  # number of iterations in the lifetime of this process
    raw_model = model.module if ddp else model  # unwrap DDP container if needed
    running_mfu = -1.0
    while True:
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0:
            losses = estimate_loss()
        if iter_num % eval_interval == 0:
            if master_process:
                print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                if wandb_log:
                    wandb.log(
                        {
                            "iter": iter_num,
                            "train/loss": losses["train"],
                            "val/loss": losses["val"],
                            "lr": lr,
                            "mfu": running_mfu * 100,  # convert to percentage
                        }
                    )
            if iter_num > 0:
                # When iter_num == 0, the training does not start sotoptimizer state is empty,
                # Don't save checkpoint
                # + + + VeScale API below
                checkpoint_state = {"model": model, "optimizer": optimizer}
                vescale.checkpoint.save(
                    os.path.join(save_checkpoint_path, f"iter_{iter_num}"),
                    checkpoint_state,
                    async_checkpoint=async_checkpoint,
                )
                # + + + VeScale API above
        if iter_num == 0 and eval_only:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # + + + VeScale API below
        if ddp:
            model.zero_grad_buffer()
        # + + + VeScale API above
        for micro_step in range(gradient_accumulation_steps):
            # with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps  # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch("train")
            # backward pass
            loss.backward()
        # + + + VeScale API below
        if ddp:
            model.finish_grad_sync()
        # + + + VeScale API above
        optimizer.step()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        if ddp:
            lossf = loss.to_local() * gradient_accumulation_steps / world_size
            all_reduce(lossf)
        else:
            lossf = loss * gradient_accumulation_steps / world_size
        lossf = lossf.item()
        if iter_num % log_interval == 0 and master_process:
            if local_iter_num >= 5:  # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(local_batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > max_iters:
            break

    if ddp:
        barrier()
        destroy_process_group()


if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    config_keys = [k for k, v in globals().items() if not k.startswith("_") and isinstance(v, (int, float, bool, str))]
    exec(open("configurator.py").read())  # overrides from command line or config file
    config = {k: globals()[k] for k in config_keys}  # will be useful for logging
    # -----------------------------------------------------------------------------
    main()
