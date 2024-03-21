################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

from torch.testing._internal.common_utils import run_tests

from common_dtensor import with_comms, DTensorTestBase

import torch
import torch.nn as nn

from vescale.dtensor.placement_types import Replicate, Shard, Partial
from vescale.dtensor import DTensor
from vescale.dtensor.device_mesh import DeviceMesh
from vescale.dmodule.api import parallelize_module
from vescale.dmodule.api import PlacementsInterface as PI


class MLP(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size * 4, bias=False)
        self.fc2 = nn.Linear(hidden_size * 4, hidden_size, bias=False)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class PTBMLP(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.m1 = MLP(hidden_size)
        self.m2 = MLP(hidden_size)

    def forward(self, x):
        x1 = self.m1(x)
        x2 = self.m2(x)
        assert x1._spec.placements[0].is_partial()
        partial_m = x1 + x2
        assert partial_m._spec.placements[0].is_replicate()
        return x + partial_m


class PTBFunctionalMLP(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.m1 = MLP(hidden_size)
        self.partial_bias = nn.Parameter(torch.empty((hidden_size,)))

    def forward(self, x):
        x1 = self.m1(x)
        assert self.partial_bias._spec.placements[0].is_partial()
        x2 = x1 + self.partial_bias
        assert x2._spec.placements[0].is_replicate()
        return x2


class TestDeferResharding(DTensorTestBase):
    @with_comms
    def test_dtensor_resharding(self):
        mesh = DeviceMesh("cuda", list(range(self.world_size)))
        tensor = torch.randn((4, 16, 16))
        dtensor = DTensor.from_local(tensor, mesh, [Replicate()])

        ptb_mlp = PTBMLP(16)

        fwd_sharding_plan = {
            "m1.input": [[Replicate()]],
            "m1.output": [PI([Replicate()], defer_reshard=True)],
            "m2.input": [[Replicate()]],
            "m2.output": [PI([Replicate()], defer_reshard=True)],
        }

        param_sharding_plan = {
            "m1.fc1.weight": [Shard(0)],
            "m1.fc2.weight": [Shard(1)],
            "m2.fc1.weight": [Shard(0)],
            "m2.fc2.weight": [Shard(1)],
        }

        ptb_mlp = parallelize_module(ptb_mlp, mesh, {"parameter": param_sharding_plan, "forward": fwd_sharding_plan})

        out = ptb_mlp(dtensor)

    @with_comms
    def test_dtensor_resharding_function(self):
        mesh = DeviceMesh("cuda", list(range(self.world_size)))
        tensor = torch.randn((4, 16, 16))
        dtensor = DTensor.from_local(tensor, mesh, [Replicate()])

        ptb_mlp = PTBFunctionalMLP(16)

        fwd_sharding_plan = {
            "m1.input": [[Replicate()]],
            "m1.output": [PI([Replicate()], defer_reshard=True)],
        }

        param_sharding_plan = {
            "m1.fc1.weight": [Shard(0)],
            "m1.fc2.weight": [Shard(1)],
            "partial_bias": [Partial()],
        }

        ptb_mlp = parallelize_module(ptb_mlp, mesh, {"parameter": param_sharding_plan, "forward": fwd_sharding_plan})

        out = ptb_mlp(dtensor)


if __name__ == "__main__":
    run_tests()
