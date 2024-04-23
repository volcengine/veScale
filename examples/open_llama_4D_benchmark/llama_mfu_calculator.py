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

# reference: https://www.adamcasson.com/posts/transformer-flops
# reference: https://arxiv.org/pdf/2001.08361.pdf


def estimate_llama(config, bsz, sqence_length):
    embed = 4 * bsz * sqence_length * config.hidden_size
    ff = 3 * 2 * config.hidden_size * config.intermediate_size * bsz * sqence_length
    attn_qkv = 2 * bsz * sqence_length * config.hidden_size * 3 * config.hidden_size
    attn_mask = 2 * sqence_length * config.hidden_size
    attn_proj = 2 * config.hidden_size * config.intermediate_size * bsz * sqence_length
    attn = attn_qkv + attn_mask + attn_proj
    return embed + (ff + attn) * config.num_hidden_layers
