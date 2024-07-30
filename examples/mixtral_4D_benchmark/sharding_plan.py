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

"""This file contain TP/SP sharding plans for Mixtral example code."""

from vescale.dtensor.placement_types import Replicate, Shard


param_sharding_plan = {
    "embed_tokens.weight": [Replicate()],
    r"layers.\d+.input_layernorm.weight": [Replicate()],  # MixtralRMSNorm
    r"layers.\d+.self_attn.q_proj.weight": [Shard(0)],
    r"layers.\d+.self_attn.k_proj.weight": [Shard(0)],
    r"layers.\d+.self_attn.v_proj.weight": [Shard(0)],
    # TODO: buggy, cos_cached or sin_cached can be updated or recreated if seqlen exceeds the max seqlen.
    r"layers.\d+.self_attn.rotary_emb.layers.\d+.cos_cached": [Replicate()],
    r"layers.\d+.self_attn.rotary_emb.layers.\d+.sin_cached": [Replicate()],
    r"layers.\d+.self_attn.o_proj.weight": [Shard(1)],
    r"layers.\d+.post_attention_layernorm.weight": [Replicate()],
    r"layers.\d+.block_sparse_moe.gate.weight": [Replicate()],
    r"layers.\d+.block_sparse_moe.experts.\d+.w1.weight": [Shard(0)],
    r"layers.\d+.block_sparse_moe.experts.\d+.w3.weight": [Shard(0)],
    r"layers.\d+.block_sparse_moe.experts.\d+.w2.weight": [Shard(1)],
    "norm.weight": [Replicate()],
}

fwd_resharding_plan = {
    # TODO: buggy: attn mask is torch.Tensor, in training, it's a None
    r".input": {"input_ids": [Replicate()], "attention_mask": [Replicate()]},
    "embed_tokens.input": [[Replicate()]],
    # No SP
    # r"layers.\d+.input_layernorm.input": [[Replicate()]],
    # r"layers.\d+.input_layernorm.output": [[Replicate()]],
    # SP
    r"layers.\d+.input_layernorm.input": [[Shard(1)]],
    r"layers.\d+.input_layernorm.output": [[Shard(1)]],
    r"layers.\d+.self_attn.input": [[Replicate()]],
    r"layers.\d+.self_attn.output": {"attn_output": [Replicate()], "attn_weights": None, "past_key_value": None},
    r"layers.\d+.self_attn.o_proj.output": [[Replicate()]],
    # No SP
    # r"layers.\d+.post_attention_layernorm.input": [[Replicate()]],
    # r"layers.\d+.post_attention_layernorm.output": [[Replicate()]],
    # SP
    r"layers.\d+.post_attention_layernorm.input": [[Shard(1)]],
    r"layers.\d+.post_attention_layernorm.output": [[Shard(1)]],
    r"layers.\d+.block_sparse_moe.input": [[Replicate()]],
    r"layers.\d+.block_sparse_moe.gate.output": [[Replicate()]],
    r"layers.\d+.block_sparse_moe.output": {"final_hidden_states": [Replicate()], "router_logits": [Replicate()]},
    r"layers.\d+.block_sparse_moe.experts.\d+.w1.input": [[Replicate()]],
    r"layers.\d+.block_sparse_moe.experts.\d+.w3.input": [[Replicate()]],
    r"layers.\d+.block_sparse_moe.experts.\d+.w2.output": [[Replicate()]],
    "norm.input": [[Replicate()]],
}

mixtral_plan = {"parameter": param_sharding_plan, "forward": fwd_resharding_plan}
