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

import torch.nn as nn
from vescale.dtensor.placement_types import Shard, Replicate


class MLP(nn.Module):
    def __init__(self, features_in, features_out, value):
        super().__init__()
        self.value = value
        self.fc1 = nn.Linear(features_in, 16, bias=False)
        self.fc1.weight.data.fill_(value)
        self.fc2 = nn.Linear(16, features_out, bias=False)
        self.fc2.weight.data.fill_(value * 2)
        self.gelu = nn.GELU()

    def forward(self, x):
        t = self.fc1(x)
        t = self.gelu(t)
        t = self.fc2(t)
        return t


class FourMLP(nn.Module):
    def __init__(self, hidden=64, fixed_size=True):
        super().__init__()
        if fixed_size:
            self.mlp1 = MLP(hidden, hidden, 1)
            self.mlp2 = MLP(hidden, hidden, 2)
            self.mlp3 = MLP(hidden, hidden, 3)
            self.mlp4 = MLP(hidden, hidden, 4)
        else:
            self.mlp1 = MLP(hidden * 1, hidden * 2, 1)
            self.mlp2 = MLP(hidden * 2, hidden * 3, 2)
            self.mlp3 = MLP(hidden * 3, hidden * 4, 3)
            self.mlp4 = MLP(hidden * 4, hidden * 5, 4)

    def forward(self, x):
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = self.mlp4(x)
        return x


sharding_plan = {
    "forward": {
        ".input": [[Replicate()]],
        r"mlp\d.fc1.input": [[Replicate()]],
        r"mlp\d.fc2.output": [[Replicate()]],
    },
    "parameter": {
        r"mlp\d.fc1.weight": [Shard(0)],
        r"mlp\d.fc2.weight": [Shard(1)],
    },
}
