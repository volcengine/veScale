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

import re
from typing import Sequence


from vescale import Shard, Replicate, Placement


class LLMHandWriteFormat:
    def __init__(self, params_sharding_plan):
        super().__init__()
        self.default_params_sharding_plan = params_sharding_plan

    def get_tensor_sharding_plan_by_name(self, name: str) -> Sequence[Placement]:
        for pattern, placements in self.default_params_sharding_plan.items():
            if re.search(pattern, name):
                return placements
        return [Replicate()]


MEGATRON_GPT_RULES = {
    r"model.gpt_model.language_model.embedding.word_embeddings.weight": [Shard(0)],
    r"model.gpt_model.language_model.encoder.layers.\d+.mlp.dense_h_to_4h.weight": [Shard(0)],
    r"model.gpt_model.language_model.encoder.layers.\d+.mlp.dense_h_to_4h_lora.weight": [Shard(0)],
    r"model.gpt_model.language_model.encoder.layers.\d+.mlp.dense_4h_to_h.weight": [Shard(1)],
    r"model.gpt_model.language_model.encoder.layers.\d+.mlp.dense_4h_to_h_lora.weight": [Shard(1)],
    r"model.gpt_model.language_model.encoder.layers.\d+.self_attention.query_key_value.weight": [Shard(0)],
    r"model.visual_encoder.blocks.\d+.attn.qkv.weight": [Shard(0)],
    r"model.visual_encoder.blocks.\d+.attn.proj.weight": [Shard(1)],
    r"model.visual_encoder.blocks.\d+.mlp.fc1.weight": [Shard(0)],
    r"model.visual_encoder.blocks.\d+.mlp.fc2.weight": [Shard(1)],
}
