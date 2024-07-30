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
import torch.nn as nn

from vescale.dtensor.placement_types import Shard, Replicate


class MLP(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))


HIDDEN_DIM = 4
SEQ_LEN = 7
BSZ = 3

MLP_PAIRWISE_PARAM_SHARDING_PLAN = {
    "fc1.weight": [Shard(0)],
    "fc1.bias": [Shard(0)],
    "fc2.weight": [Shard(1)],
    "fc2.bias": [Replicate()],
}

MLP_FWD_RESAHRDING_PLAM = {
    "fc1.input": [[Replicate()]],
    "fc2.output": [[Replicate()]],
}
