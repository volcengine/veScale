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
import os


class MLP(nn.Module):
    def __init__(self, features_in, feature_middle, features_out, value):
        super().__init__()
        self.value = value
        self.counter = 0
        self.fc1 = nn.Linear(1024, 1024, bias=False)
        self.fc1.weight.data.fill_(value)
        self.fc2 = nn.Linear(1024, 1024, bias=False)
        self.fc2.weight.data.fill_(value * 2)
        self.gelu = nn.GELU()

    def forward(self, x):
        t = self.fc1(x)
        t = self.gelu(t)
        t = self.fc2(t)
        torch.save(t, f"{os.environ['model_name']}_mlp{self.value}_fwd{self.counter}_out_tensor.pt")
        self.counter += 1
        return t


class FourMLP(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.mlp1 = MLP(hidden * 1, hidden * 2, hidden * 3, 0)
        self.mlp2 = MLP(hidden * 3, hidden * 4, hidden * 5, 1)
        self.mlp3 = MLP(hidden * 5, hidden * 6, hidden * 7, 2)
        self.mlp4 = MLP(hidden * 7, hidden * 8, hidden * 9, 3)
        self.sequence = nn.Sequential(self.mlp1, self.mlp2, self.mlp3, self.mlp4)

    def forward(self, x):
        return self.sequence(x)
