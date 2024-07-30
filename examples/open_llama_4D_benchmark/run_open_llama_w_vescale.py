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
import os
import torch
import argparse

os.environ["VESCALE_DISABLE_RUN_CHECK"] = "1"

from vescale.dtensor.device_mesh import init_device_mesh
from vescale.ddp.distributed_data_parallel import DistributedDataParallel as DDP
from vescale.optim.distributed_optimizer import DistributedOptimizer
from vescale.dmodule.api import parallelize_module
from sharding_plan import sharding_plan

from transformers import AutoModelForCausalLM, AutoConfig, LlamaModel

from llama_mfu_calculator import estimate_llama

local_rank = int(os.environ["LOCAL_RANK"])
parser = argparse.ArgumentParser()
parser.add_argument("--total_bsz", type=int, default=16)
parser.add_argument("--dp", type=int)
parser.add_argument("--tp", type=int)
parser.add_argument("--warmup", type=int, default=5)
parser.add_argument("--iter", type=int, default=10)
parser.add_argument("--no-ckpt", action="store_true")

args = parser.parse_args()

assert args.total_bsz % args.dp == 0, f"total batch size {args.total_bsz} is not divisiable by dp size {args.dp}"
bsz = args.total_bsz // args.dp
s = 2048

# init model
if args.no_ckpt:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    config = AutoConfig.from_pretrained(os.path.join(dir_path, "config.json"))
    model = LlamaModel(config)
else:
    model = AutoModelForCausalLM.from_pretrained("openlm-research/open_llama_7b")
    model = model.model
    config = model.config
assert s <= config.max_position_embeddings

# --------  training config --------
device_mesh = init_device_mesh(
    "cuda",
    (
        args.dp,
        args.tp,
    ),
    mesh_dim_names=("DP", "TP"),
)

input = torch.randint(low=0, high=config.vocab_size, size=(bsz, s)).cuda()

model = model.cuda().bfloat16()
vescale_model = parallelize_module(model, device_mesh["TP"], sharding_plan)

ddp_model = DDP(
    vescale_model,
    data_pg_or_device_mesh=device_mesh["DP"],
    use_distributed_optimizer=True,
)
orig_optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.01)

ve_optimizer = DistributedOptimizer(
    orig_optimizer,
    overlap_param_gather=True,
    models=[ddp_model],
)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

# --------  warm up --------
for _ in range(args.warmup):
    ve_optimizer.zero_grad()
    vescale_output = ddp_model(input).last_hidden_state
    vescale_loss = vescale_output.mean()
    vescale_loss.backward()
    ve_optimizer.step()

# --------  training loop --------
start.record()
for _ in range(args.iter):
    ve_optimizer.zero_grad()
    vescale_output = ddp_model(input).last_hidden_state
    vescale_loss = vescale_output.mean()
    vescale_loss.backward()
    ve_optimizer.step()
end.record()
torch.cuda.synchronize()
exec_t = start.elapsed_time(end) / 1000 / args.iter

if local_rank == 0:
    flops_dict = {
        "A100": 312,
        "H100": 1000,
    }
    d_name = torch.cuda.get_device_name()
    total_flops = flops_dict["A100"] * (10**12) * device_mesh.ndevice
    for k, v in flops_dict.items():
        if k in d_name:
            total_flops = v * (10**12) * device_mesh.ndevice
            break
    print(f"1 iter time: {exec_t}")
    # fwd + bwd =3
    print("mfu:", estimate_llama(config, bsz, s) * 3 * args.dp * 100 / exec_t / total_flops)
