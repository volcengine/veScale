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

# Define functions which are commonly used for nano_gpt checkpointing test
import torch
import math

from vescale.dtensor.placement_types import Replicate, Shard
from vescale.dmodule.api import parallelize_module
from vescale.ddp.distributed_data_parallel import DistributedDataParallel as DDP
from vescale.optim.distributed_optimizer import DistributedOptimizer
from transformers import AutoModelForCausalLM
from .nano_gpt import GPT, GPTConfig
import os


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


fwd_plan = {
    "transformer.wte.input": [[Replicate()]],
    "transformer.wte.output": [[Replicate()]],
    "transformer.wpe.input": [[Replicate()]],
    "transformer.wpe.output": [[Replicate()]],
    r"transformer.h.\d+.input": [[Shard(1)]],
    r"transformer.h.\d+.attn.input": [[Replicate()]],
    r"transformer.h.\d+.attn.c_proj.output": [[Replicate()]],
    r"transformer.h.\d+.attn.output": [[Shard(1)]],
    r"transformer.h.\d+.mlp.c_fc.input": [[Replicate()]],
    r"transformer.h.\d+.mlp.c_proj.output": [[Replicate()]],
    r"transformer.h.\d+.mlp.output": [[Shard(1)]],
    "transformer.ln_f.input": [[Shard(1)]],
    "lm_head.input": [[Shard(2)]],
    "lm_head.output": [[Replicate()]],
}

params_plan = {
    "transformer.wte.weight": [Shard(1)],
    "transformer.wpe.weight": [Shard(1)],
    r"transformer.h.\d+.attn.q_proj.weight": [Shard(0)],
    r"transformer.h.\d+.attn.q_proj.bias": [Shard(0)],
    r"transformer.h.\d+.attn.k_proj.weight": [Shard(0)],
    r"transformer.h.\d+.attn.k_proj.bias": [Shard(0)],
    r"transformer.h.\d+.attn.v_proj.weight": [Shard(0)],
    r"transformer.h.\d+.attn.v_proj.bias": [Shard(0)],
    r"transformer.h.\d+.attn.c_proj.weight": [Shard(1)],
    r"transformer.h.\d+.attn.c_proj.bias": [Replicate()],
    r"transformer.h.\d+.mlp.c_fc.weight": [Shard(0)],
    r"transformer.h.\d+.mlp.c_fc.bias": [Shard(0)],
    r"transformer.h.\d+.mlp.c_proj.weight": [Shard(1)],
    r"transformer.h.\d+.mlp.c_proj.bias": [Replicate()],
    "lm_head.weight": [Shard(1)],
}

nanoGPT_plan = {"parameter": params_plan, "forward": fwd_plan}


def build_gpt_model_optimizer_and_dataset(init_method, dp_size=1, tp_size=1):
    # -----------------------------------------------------------------------------
    num_iters = 1
    # data
    batch_size = 4
    block_size = 8
    vocab_size = 32
    # model
    n_layer = 2
    n_head = 4
    n_embd = 16
    dropout = 0.1  # for pretraining 0 is good, for finetuning try 0.1+
    bias = True  # do we use bias inside LayerNorm and Linear layers?
    # system
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(999)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    # -----------------------------------------------------------------------------
    # fake data loader
    data_set = []
    for _ in range(num_iters):
        idx = torch.randint(0, vocab_size, (batch_size, block_size), dtype=torch.int64).cuda()
        target = torch.randint(0, vocab_size, (batch_size, block_size), dtype=torch.int64).cuda()
        data_set.append((idx, target))

    # model config
    model_args = dict(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        bias=bias,
    )
    # DP=2 TP=2
    gptconf = GPTConfig(**model_args)
    if init_method == "scratch":
        gpt = GPT(gptconf).bfloat16()
    else:
        gpt = GPT.from_pretrained(init_method, dict(dropout=0.0)).bfloat16()

    open_source = False
    try:
        from vescale.devicemesh_api import VESCALE_DEVICE_MESH
    except ImportError:
        open_source = True
    VESCALE_DEVICE_MESH.init_device_mesh(
        device_type="cuda",
        mesh_shape=(dp_size, tp_size),
        mesh_dim_names=("DP", "TP"),
    )

    # Enable tensor Parallel
    tp_gpt = parallelize_module(gpt, VESCALE_DEVICE_MESH["TP"], nanoGPT_plan)

    # Enable data Parallel
    ddp_gpt = DDP(
        tp_gpt,
        data_pg_or_device_mesh=VESCALE_DEVICE_MESH["DP"],
        accumulate_allreduce_grads_in_fp32=True,
        overlap_grad_reduce=False,
        use_distributed_optimizer=True,
    )

    # Build distributed optimizer
    dist_optimizer = DistributedOptimizer(
        torch.optim.Adam(ddp_gpt.parameters(), lr=0.01),
        clip_grad=0.0,
        overlap_param_gather=False,
        models=[ddp_gpt],
    )
    return ddp_gpt, dist_optimizer, data_set


def merge_optimizer_states(states):
    merged_kvs = {}
    # Use length directly instead of world size
    # Because we may merge it on CPU + memory using one process
    state_length = len(states)
    for s_dict in states:
        s_dict[torch.float32] = flatten_dict(s_dict[torch.float32])

    for s_dict in states:
        for k, v in s_dict[torch.float32].items():
            if "step" not in k:
                cross_dp = False
                for rank in range(state_length):
                    if k in states[rank][torch.float32] and states[rank][torch.float32][k].dp_ranks_ranges:
                        cross_dp = True
                        break

                if not cross_dp:
                    assert v.dp_ranks_ranges is None
                    if k not in merged_kvs:
                        merged_kvs[k] = torch.zeros(v.global_shape, dtype=v.local_tensor.dtype)

                    if len(v.global_shape) == 1:
                        merged_kvs[k][v.global_offset[0] : v.global_offset[0] + v.local_shape[0],] = (
                            v.local_tensor.view(v.local_shape)
                        )
                    elif len(v.global_shape) == 2:
                        merged_kvs[k][
                            v.global_offset[0] : v.global_offset[0] + v.local_shape[0],
                            v.global_offset[1] : v.global_offset[1] + v.local_shape[1],
                        ] = v.local_tensor.view(v.local_shape)
                else:
                    if k not in merged_kvs:
                        # Two stage merging:
                        # Stage 1: merge tensors with different dp and same tp

                        # Create tp sharded tensors
                        # Key: global offset
                        # Value: tensor after tp sharding
                        tp_offset_shape = {}
                        tp_sharded_tensors = {}
                        for rank in range(state_length):
                            if k in states[rank][torch.float32]:
                                state_on_dp = states[rank][torch.float32][k]
                                range_1d = state_on_dp.dp_ranks_ranges[rank]

                                if state_on_dp.global_offset not in tp_sharded_tensors:
                                    tp_sharded_tensors[state_on_dp.global_offset] = torch.zeros(
                                        (math.prod(state_on_dp.local_shape),), dtype=state_on_dp.local_tensor.dtype
                                    )
                                    tp_offset_shape[state_on_dp.global_offset] = state_on_dp.local_shape

                                tp_sharded_tensors[state_on_dp.global_offset][range_1d.start : range_1d.end] = (
                                    state_on_dp.local_tensor
                                )

                        # Stage 2: merge tensors with different tp
                        merged_kvs[k] = torch.zeros(v.global_shape, dtype=v.local_tensor.dtype)

                        for offset, tensor in tp_sharded_tensors.items():
                            shape = tp_offset_shape[offset]
                            if len(v.global_shape) == 1:
                                merged_kvs[k][offset[0] : offset[0] + shape[0]] = tensor.view(shape)
                            elif len(v.global_shape) == 2:
                                merged_kvs[k][offset[0] : offset[0] + shape[0], offset[1] : offset[1] + shape[1]] = (
                                    tensor.view(shape)
                                )

    return merged_kvs


def get_open_llama_model(layer_number=None):
    if layer_number is None:
        model = AutoModelForCausalLM.from_pretrained("openlm-research/open_llama_7b")
    else:
        model = AutoModelForCausalLM.from_pretrained("openlm-research/open_llama_7b", num_hidden_layers=layer_number)
    docoder = model.model
    return docoder, model.config


# forward resharding plan for a single open llama decoder
_decoder_fwd_resharding_plan = {
    "input": {"hidden_states": [Shard(1)], "attention_mask": [Replicate()], "position_ids": [Replicate()]},
    # atten
    "self_attn.input": {"hidden_states": [Replicate()], "attention_mask": [Replicate()], "position_ids": [Replicate()]},
    "self_attn.o_proj.output": [[Shard(1)]],
    "self_attn.output": [[Shard(1)], None, None],
    # feedforward(mlp)
    "mlp.input": [[Replicate()]],
    "mlp.output": [[Shard(1)]],
    "output": [[Shard(1)], None],
}

# parameter sharding plan for a single open llama decoder
_decoder_param_sharding_plan = {
    # atten weight, no bias
    "self_attn.q_proj.weight": [Shard(0)],
    "self_attn.k_proj.weight": [Shard(0)],
    "self_attn.v_proj.weight": [Shard(0)],
    "self_attn.o_proj.weight": [Shard(1)],
    # feedforward(mlp)
    "mlp.up_proj.weight": [Shard(0)],
    "mlp.gate_proj.weight": [Shard(0)],
    "mlp.down_proj.weight": [Shard(1)],
}

# forward resharding plan for the whole open llama model
model_fwd_resharding_plan = {
    ".input": [[Replicate()]],
    "embed_tokens.output": [[Shard(1)]],
    "norm.input": [[Shard(1)]],
    ".output": {
        "last_hidden_state": [Replicate()],
    },
    **{rf"layers.\d+.{k}": v for k, v in _decoder_fwd_resharding_plan.items()},
}

# model parameter sharding plan for the whole open llama model
model_param_sharding_plan = {
    "embed_tokens.weight": [Shard(1)],
    **{rf"layers.\d+.{k}": v for k, v in _decoder_param_sharding_plan.items()},
}

sharding_plan = {"parameter": model_param_sharding_plan, "forward": model_fwd_resharding_plan}


def get_open_llama_model_optimizer(dp_size, tp_size, layer_number=None):
    from vescale.devicemesh_api import VESCALE_DEVICE_MESH

    VESCALE_DEVICE_MESH.init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("DP", "TP"), check_uniqueness=True)
    # Set 4 layers to avoid timeout on CI
    # Use 32 layers when running on training platform
    vescale_decoder, config = get_open_llama_model(layer_number=layer_number)

    vescale_decoder = parallelize_module(
        vescale_decoder,
        VESCALE_DEVICE_MESH["TP"],
        sharding_plan,
    )

    ddp_decoder = DDP(
        vescale_decoder,
        data_pg_or_device_mesh=VESCALE_DEVICE_MESH["DP"],
        accumulate_allreduce_grads_in_fp32=True,
        overlap_grad_reduce=False,
        use_distributed_optimizer=True,
    )

    ve_optimizer = DistributedOptimizer(
        torch.optim.Adam(ddp_decoder.parameters(), lr=0.01),
        clip_grad=0.0,
        overlap_param_gather=False,
        models=[ddp_decoder],
    )
    return ddp_decoder, ve_optimizer, config
