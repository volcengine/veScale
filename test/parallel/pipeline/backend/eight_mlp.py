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


class Embed(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 64)

    def forward(self, x):
        return self.embedding(x)

    def get_word_embeddings_weight(self):
        return self.embedding.weight


class EmbedTwo(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 64)

    def forward(self, x):
        return self.embedding(x)

    def get_word_embeddings_weight(self):
        return self.embedding.weight


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


class SmallMLP(nn.Module):
    def __init__(self, features_in, features_out, value):
        super().__init__()
        self.value = value
        self.fc1 = nn.Linear(features_in, features_out, bias=False)
        self.fc1.weight.data.fill_(value)
        self.gelu = nn.GELU()

    def forward(self, x):
        t = self.fc1(x)
        t = self.gelu(t)
        return t


class HierachicalMLP(nn.Module):
    def __init__(self, features_in, features_out, value):
        super().__init__()
        self.value = value
        self.fc0 = SmallMLP(features_in, features_in, value)
        self.fc1 = nn.Linear(features_in, 16, bias=False)
        self.fc2 = nn.Linear(16, features_out, bias=False)
        self.fc3 = SmallMLP(features_out, features_out, value)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = x + x
        x = self.fc0(x)
        t = self.fc1(x)
        t = self.gelu(t)
        t = self.fc2(t)
        t = self.fc3(t)
        return t


class EightMLP(nn.Module):
    def __init__(self, hidden=64, fixed_size=True, embedded_module=False):
        super().__init__()
        module = HierachicalMLP if embedded_module else MLP
        if fixed_size:
            self.mlp1 = module(hidden, hidden, 0)
            self.mlp2 = module(hidden, hidden, 1)
            self.mlp3 = module(hidden, hidden, 2)
            self.mlp4 = module(hidden, hidden, 3)
            self.mlp5 = module(hidden, hidden, 4)
            self.mlp6 = module(hidden, hidden, 5)
            self.mlp7 = module(hidden, hidden, 6)  # tranformerlayer7 = TransformerLayer(hidden)
            self.mlp8 = module(hidden, hidden, 7)  # tranformerlayer8 = TransformerLayer(hidden)
        else:
            self.mlp1 = module(hidden * 1, hidden * 2, 0)
            self.mlp2 = module(hidden * 2, hidden * 3, 1)
            self.mlp3 = module(hidden * 3, hidden * 4, 2)
            self.mlp4 = module(hidden * 4, hidden * 5, 3)
            self.mlp5 = module(hidden * 5, hidden * 6, 4)
            self.mlp6 = module(hidden * 6, hidden * 7, 5)
            self.mlp7 = module(hidden * 7, hidden * 8, 6)
            self.mlp8 = module(hidden * 8, hidden * 9, 7)

    def forward(self, x):
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = self.mlp4(x)
        x = self.mlp5(x)
        x = self.mlp6(x)
        x = self.mlp7(x)
        x = self.mlp8(x)
        return x


class EightMLPDiffNames(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.mlp1 = MLP(hidden, hidden, 0)
        self.mlp2 = MLP(hidden, hidden, 1)
        self.mlp3 = MLP(hidden, hidden, 2)
        self.layer1 = MLP(hidden, hidden, 3)
        self.layer2 = MLP(hidden, hidden, 4)
        self.layer3 = MLP(hidden, hidden, 5)
        self.layer4 = MLP(hidden, hidden, 6)
        self.more_layer1 = MLP(hidden, hidden, 7)

    def forward(self, x):
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.more_layer1(x)
        return x


class EightMLPWithOps(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.mlp1 = MLP(hidden, hidden, 0)
        self.mlp2 = MLP(hidden, hidden, 1)
        self.mlp3 = MLP(hidden, hidden, 2)
        self.mlp4 = MLP(hidden, hidden, 3)
        self.mlp5 = MLP(hidden, hidden, 4)
        self.mlp6 = MLP(hidden, hidden, 5)
        self.mlp7 = MLP(hidden, hidden, 6)
        self.mlp8 = MLP(hidden, hidden, 7)

    def forward(self, x):
        x = x + x
        x = self.mlp1(x)
        x = x * 2
        x = self.mlp2(x)
        x = x * 2
        x = x * 2
        x = x * 2
        x = self.mlp3(x)
        x = self.mlp4(x)
        x = self.mlp5(x)
        x = self.mlp6(x)
        x = self.mlp7(x)
        x = self.mlp8(x)
        return x


class EightMLPWithOpsTail(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.mlp1 = MLP(hidden, hidden, 0)
        self.mlp2 = MLP(hidden, hidden, 1)
        self.mlp3 = MLP(hidden, hidden, 2)
        self.mlp4 = MLP(hidden, hidden, 3)
        self.mlp5 = MLP(hidden, hidden, 4)
        self.mlp6 = MLP(hidden, hidden, 5)
        self.mlp7 = MLP(hidden, hidden, 6)
        self.mlp8 = MLP(hidden, hidden, 7)

    def forward(self, x):
        x = x + x
        x = self.mlp1(x)
        x = x * 2
        x = self.mlp2(x)
        x = x * 2
        x = self.mlp3(x)
        x = self.mlp4(x)
        x = self.mlp5(x)
        x = self.mlp6(x)
        x = self.mlp7(x)
        x = self.mlp8(x)
        x = x * 2
        x = x * 4
        x = x + 4
        return x


class EightMLPSharedEmbed(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.embed1 = Embed()
        self.mlp1 = MLP(hidden, hidden, 0)
        self.mlp2 = MLP(hidden, hidden, 1)
        self.mlp3 = MLP(hidden, hidden, 2)
        self.mlp4 = MLP(hidden, hidden, 3)
        self.mlp5 = MLP(hidden, hidden, 4)
        self.mlp6 = MLP(hidden, hidden, 5)
        self.mlp7 = MLP(hidden, hidden, 6)
        self.mlp8 = MLP(hidden, hidden, 7)
        self.embed2 = EmbedTwo()

    def forward(self, x):
        x = self.embed1(x).float()
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = self.mlp4(x)
        x = self.mlp5(x)
        x = self.mlp6(x)
        x = self.mlp7(x)
        x = self.mlp8(x).long()
        x = self.embed2(x)
        return x


sharding_plan = {
    "forward": {
        r"mlp\d.input": [[Replicate()]],
        r"mlp\d.output": [[Replicate()]],
    },
    "parameter": {
        r"mlp\d.fc1.weight": [Shard(0)],
        r"mlp\d.fc2.weight": [Shard(1)],
    },
}

sharding_plan_two = {
    "forward": {
        r"mlp\d.input": [[Replicate()]],
        r"mlp\d.output": [[Replicate()]],
    },
    "parameter": {
        r"mlp\d.weight": [Shard(1)],
    },
}

sharding_plan_combo = {
    "forward": {
        r"mlp\d.input": [[Replicate()]],
        r"mlp\d.output": [[Replicate()]],
        r"layer\d.input": [[Replicate()]],
        r"layer\d.output": [[Replicate()]],
        r"more_layer\d.input": [[Replicate()]],
        r"more_layer\d.output": [[Replicate()]],
    },
    "parameter": {
        r"mlp\d.weight": [Shard(1)],
        r"layer\d.weight": [[Replicate()]],
    },
}

sharding_plan_fc = {
    "forward": {
        r"mlp\d.fc\d.input": [[Replicate()]],
        r"mlp\d.fc\d.output": [[Replicate()]],
    },
    "parameter": {
        r"mlp\d.fc1.weight": [Shard(0)],
        r"mlp\d.fc2.weight": [Shard(1)],
    },
}
