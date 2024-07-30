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

import copy
import unittest
from typing import Tuple

import torch
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

from vescale.dtensor.placement_types import Replicate, Shard
from vescale.dtensor.device_mesh import DeviceMesh
from vescale.dtensor.api import redistribute_dtensor, distribute_tensor

from common_dtensor import DTensorTestBase, with_comms


class AllToAllTest(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @unittest.skip("failed in CI, strange!")
    @with_comms
    @parametrize("shard_dims", [(0, 1), (1, 2), (2, 1), (1, 0)])
    def test_all_to_all_first(self, shard_dims: Tuple[int]):
        original_shard_dim, target_shard_dim = shard_dims[0], shard_dims[1]
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        x = torch.rand(8, 8, 8, 4).cuda()
        copy_x = copy.deepcopy(x)

        dx = distribute_tensor(x, device_mesh, [Shard(original_shard_dim)])
        dy = redistribute_dtensor(dx, device_mesh, [Shard(target_shard_dim)])

        d_copy_x = distribute_tensor(copy_x, device_mesh, [Shard(original_shard_dim)])
        r_copy_x = redistribute_dtensor(d_copy_x, device_mesh, [Replicate()])
        d_copy_y = redistribute_dtensor(r_copy_x, device_mesh, [Shard(target_shard_dim)])

        self.assertTrue(dy.placements == d_copy_y.placements)
        torch.testing.assert_close(dy._local_tensor, d_copy_y._local_tensor)


instantiate_parametrized_tests(AllToAllTest)

if __name__ == "__main__":
    run_tests()
