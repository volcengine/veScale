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

"""This file contain TP/SP sharding plans for Mixtral example code."""

from vescale.dtensor.placement_types import Replicate, Shard

param_sharding_plan = {
    "mixtral_model.model.embed_tokens.weight": [Replicate()],
    r"mixtral_model.model.layers.\d+.input_layernorm.weight": [Replicate()],  # MixtralRMSNorm
    r"mixtral_model.model.layers.\d+.self_attn.q_proj.weight": [Shard(0)],
    r"mixtral_model.model.layers.\d+.self_attn.k_proj.weight": [Shard(0)],
    r"mixtral_model.model.layers.\d+.self_attn.v_proj.weight": [Shard(0)],
    # TODO: buggy, cos_cached or sin_cached can be updated or recreated if seqlen exceeds the max seqlen.
    r"mixtral_model.model.layers.\d+.self_attn.rotary_emb.cos_cached": [Replicate()],
    r"mixtral_model.model.layers.\d+.self_attn.rotary_emb.sin_cached": [Replicate()],
    r"mixtral_model.model.layers.\d+.self_attn.o_proj.weight": [Shard(1)],
    r"mixtral_model.model.layers.\d+.post_attention_layernorm.weight": [Replicate()],
    r"mixtral_model.model.layers.\d+.block_sparse_moe.gate.weight": [Replicate()],
    r"mixtral_model.model.layers.\d+.block_sparse_moe.experts.\d+.w1.weight": [Shard(0)],
    r"mixtral_model.model.layers.\d+.block_sparse_moe.experts.\d+.w3.weight": [Shard(0)],
    r"mixtral_model.model.layers.\d+.block_sparse_moe.experts.\d+.w2.weight": [Shard(1)],
    "mixtral_model.model.norm.weight": [Replicate()],
}

fwd_resharding_plan = {
    "mixtral_model.model.embed_tokens.output": [[Shard(1)]],
    r"mixtral_model.model.layers.\d+.self_attn.input": [[Replicate()]],
    r"mixtral_model.model.layers.\d+.self_attn.o_proj.output": [[Shard(1)]],
    "mixtral_model.lm_head.input": [[Replicate()]],
}

mixtral_plan = {"parameter": param_sharding_plan, "forward": fwd_resharding_plan}
