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

from vescale.dtensor.placement_types import Replicate, Shard

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
    "model.input": [[Replicate()]],
    "model.embed_tokens.output": [[Shard(1)]],
    "model.norm.input": [[Shard(1)]],
    "model.output": {
        "last_hidden_state": [Replicate()],
    },
    **{rf"model.layers.\d+.{k}": v for k, v in _decoder_fwd_resharding_plan.items()},
}

# model parameter sharding plan for the whole open llama model
model_param_sharding_plan = {
    "model.embed_tokens.weight": [Shard(1)],
    **{rf"model.layers.\d+.{k}": v for k, v in _decoder_param_sharding_plan.items()},
}

llama2_plan = {"parameter": model_param_sharding_plan, "forward": model_fwd_resharding_plan}
