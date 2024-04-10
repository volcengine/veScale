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
from torch import nn
from vescale.dtensor.placement_types import Replicate, Shard


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 16)
        self.gelu = torch.nn.GELU()
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(4, bias=False)
        self.mlp = MLP()

    def forward(self, x):
        return self.mlp(self.ln(x))


param_sharding_plan = {
    "fc1.weight": [Shard(0)],
    "fc1.bias": [Shard(0)],
    "fc2.weight": [Shard(1)],
    "fc2.bias": [Replicate()],
}

fwd_resharding_plan = {
    "fc1.input": [[Replicate()]],
    "fc2.output": [[Replicate()]],
}

sharding_plan = {"parameter": param_sharding_plan, "forward": fwd_resharding_plan}
