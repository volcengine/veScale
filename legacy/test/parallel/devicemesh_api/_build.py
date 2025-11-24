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
import torch
import os
from parallel.devicemesh_api._model import GPT
from vescale.optim.distributed_optimizer import DistributedOptimizer
from vescale.ddp.distributed_data_parallel import DistributedDataParallel as DDP
from vescale.dmodule.api import parallelize_module
from vescale.devicemesh_api import VESCALE_DEVICE_MESH


def system_setup():
    # system
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(999)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul


def prepare_config_and_data():
    # -----------------------------------------------------------------------------
    num_iters = 1
    # data
    batch_size = 4
    block_size = 8
    vocab_size = 32
    # model
    n_layer = 12
    n_head = 4
    n_embd = 16
    dropout = 0.1  # for pretraining 0 is good, for finetuning try 0.1+
    bias = True  # do we use bias inside LayerNorm and Linear layers?
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
    return model_args, data_set


def build_gpt_model_and_optimizer(
    gptconf, init_method, dp_size, tp_size, sharding_plan, use_dist_optimizer=False, device_type="cuda"
):
    if init_method == "scratch":
        model = GPT(gptconf).bfloat16()
    else:
        model = GPT.from_pretrained(init_method, dict(dropout=0.0)).bfloat16()

    VESCALE_DEVICE_MESH.init_device_mesh(
        device_type,
        mesh_shape=(dp_size, tp_size),
        mesh_dim_names=("DP", "TP"),
    )
    if tp_size > 1:
        # Enable tensor parallelism
        model = parallelize_module(model, VESCALE_DEVICE_MESH["TP"], sharding_plan)
    elif device_type == "cuda":
        model.to("cuda")

    if dp_size > 1:
        # Enable data Parallel
        dp_comm = (
            VESCALE_DEVICE_MESH["DP"]
            if VESCALE_DEVICE_MESH.ndim > 1
            else VESCALE_DEVICE_MESH.get_data_parallel_dim_groups()
        )
        model = DDP(
            model,
            data_pg_or_device_mesh=dp_comm,
            accumulate_allreduce_grads_in_fp32=True,
            overlap_grad_reduce=False,
            use_distributed_optimizer=True,
        )

    # Build base optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Build distributed optimizer
    if use_dist_optimizer and tp_size > 1:
        dp_comm = (
            VESCALE_DEVICE_MESH["DP"]
            if VESCALE_DEVICE_MESH.ndim > 1
            else VESCALE_DEVICE_MESH.get_data_parallel_dim_groups()
        )
        optimizer = DistributedOptimizer(
            optimizer,
            clip_grad=0.0,
            fp16=False,
            bf16=True,
            params_dtype=torch.bfloat16,
            grad_scaler=None,
            log_num_zeros_in_grad=False,
            overlap_param_gather=False,
            data_parallel_group=dp_comm,
            models=[model],
        )

    return model, optimizer, VESCALE_DEVICE_MESH.get()


def prepare_data(bsz, hidden_dim, dtype=torch.float, device_type="cuda"):
    x1, y1 = torch.rand(bsz, hidden_dim, dtype=dtype), torch.rand(bsz, hidden_dim, dtype=dtype)
    x2, y2 = torch.rand(bsz, hidden_dim, dtype=dtype), torch.rand(bsz, hidden_dim, dtype=dtype)
    x3, y3 = torch.rand(bsz, hidden_dim, dtype=dtype), torch.rand(bsz, hidden_dim, dtype=dtype)
    x4, y4 = torch.rand(bsz, hidden_dim, dtype=dtype), torch.rand(bsz, hidden_dim, dtype=dtype)
    if device_type == "cuda":
        x1, y1 = x1.cuda(), y1.cuda()
        x2, y2 = x2.cuda(), y2.cuda()
        x3, y3 = x3.cuda(), y3.cuda()
        x4, y4 = x4.cuda(), y4.cuda()
    return [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
