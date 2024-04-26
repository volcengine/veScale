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

import torch
import torch.distributed as dist

from vescale.dtensor.device_mesh import DeviceMesh
from vescale.dmodule import parallelize_module
from vescale.ddp.distributed_data_parallel import DistributedDataParallel as DDP
from vescale.optim.distributed_optimizer import DistributedOptimizer
from vescale.initialize.deferred_init import deferred_init, is_deferred

from transformers.models.mixtral.modeling_mixtral import MixtralModel
from transformers.models.mixtral.configuration_mixtral import MixtralConfig
from sharding_plan import mixtral_plan

local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
rank = int(os.environ["RANK"])


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
    torch.random.manual_seed(777)
    device_list = [
        list(range(i * args.tp, min((i + 1) * args.tp, world_size))) for i in range(max(world_size // args.tp, 1))
    ]
    device_mesh = DeviceMesh("cuda", device_list, mesh_dim_names=("DP", "TP"))
    torch.cuda.set_device(local_rank)

    mixtral_config = MixtralConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
    )

    model_deferred = deferred_init(MixtralModel, mixtral_config)

    mixtral_model = parallelize_module(
        model_deferred,
        device_mesh["TP"],
        mixtral_plan,
        factory=True,
    )

    assert not is_deferred(mixtral_model)

    ddp_mixtral_model = DDP(
        mixtral_model,
        device_mesh["DP"],
        accumulate_allreduce_grads_in_fp32=True,
        overlap_grad_reduce=False,
        use_distributed_optimizer=True,
    )

    doptim = DistributedOptimizer(
        torch.optim.Adam(mixtral_model.parameters(), lr=0.01),
        models=[ddp_mixtral_model],
        overlap_param_gather=True,
    )

    dataloader = []
    for iter in range(args.iter):
        data = torch.randint(0, args.vocab_size, (args.bsz, args.seqlen)).cuda()
        dist.all_reduce(data, op=dist.ReduceOp.MAX)
        dataloader.append(data)

    # =----- warmup -----= #
    for _ in range(args.warmup):
        data = torch.randint(0, args.vocab_size, (args.bsz, args.seqlen)).cuda()
        doptim.zero_grad()
        ddp_mixtral_model(data).last_hidden_state.to_local().sum().backward()
        ddp_mixtral_model.finish_grad_sync()
        doptim.step()

    # =----- training ----= #
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for iter in range(args.iter):
        doptim.zero_grad()
        x = dataloader[iter]
        ddp_mixtral_model(x).last_hidden_state.to_local().sum().backward()
        ddp_mixtral_model.finish_grad_sync()
        doptim.step()
    end.record()
    torch.cuda.synchronize()
    exec_t = start.elapsed_time(end) / 1000 / args.iter
    # masure mfu
    if local_rank == 0:
        # Note we are using FP32. The peak FLOPs of H100 is 59 TFLOPs.
        total_flops = 59 * (10**12) * device_mesh.ndevice
        print(f"1 iter time: {exec_t}")
        mixtral_flops = estimate_mixtral(mixtral_config, args.bsz, args.seqlen)
        print(f"fwd mixtral flops: {mixtral_flops}")
        # bwd ~= fwd * 2
        print("mfu:", mixtral_flops * 3 * args.dp * 100 / exec_t / total_flops)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iter", type=int, default=10)
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--hidden_size", type=int, default=4096)
    parser.add_argument("--intermediate_size", type=int, default=14336)
    parser.add_argument("--num_hidden_layers", type=int, default=16)
    parser.add_argument("--num_attention_heads", type=int, default=32)
    parser.add_argument("--num_key_value_heads", type=int, default=8)
    parser.add_argument("--bsz", type=int, default=16)
    parser.add_argument("--seqlen", type=int, default=256)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--tp", type=int, default=8)
    return parser


if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()
    run_mixtral(args)
