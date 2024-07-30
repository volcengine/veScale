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

from vescale.dtensor.placement_types import Replicate, Shard

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

fwd_plan_dist_dropout = {
    "transformer.wte.input": [[Replicate()]],
    "transformer.wte.output": [[Replicate()]],
    "transformer.wpe.input": [[Replicate()]],
    "transformer.wpe.output": [[Replicate()]],
    r"transformer.h.\d+.input": [[Shard(1)]],
    r"transformer.h.\d+.attn.input": [[Replicate()]],
    r"transformer.h.\d+.attn.c_proj.output": [[Shard(1)]],
    r"transformer.h.\d+.attn.output": [[Shard(1)]],
    r"transformer.h.\d+.mlp.c_fc.input": [[Replicate()]],
    r"transformer.h.\d+.mlp.c_proj.output": [[Shard(1)]],
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

nanoGPT_plan_dist_dropout = {"parameter": params_plan, "forward": fwd_plan_dist_dropout}
