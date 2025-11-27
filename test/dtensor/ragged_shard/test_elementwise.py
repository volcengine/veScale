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
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor
from torch.testing._internal.common_utils import run_tests
from torch.distributed._tensor.experimental import implicit_replication

from vescale.dtensor import distribute_tensor
from vescale.dtensor.placement_types import RaggedShard

from common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from dtensor.ragged_shard.utils import make_strided_shard


class TestElementwise(DTensorTestBase):
    @property
    def world_size(self):
        return 8

    def _run_add(self, dtensor: DTensor):
        add_tensor = torch.arange(0, dtensor.numel(), device="cuda").view(dtensor.shape)
        with implicit_replication():
            dtensor.add_(add_tensor)
        result = dtensor.full_tensor()
        self.assertEqual(result, add_tensor)

    @with_comms
    def test_ragged_shard_element_add(self):
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        tensor = torch.zeros(32, 128, dtype=torch.int64, device="cuda")
        placement = (RaggedShard((0,), (0, 1, 5, 2, 0, 0, 0, 0)),)
        dtensor = distribute_tensor(tensor.clone(), device_mesh, placement)
        self._run_add(dtensor)
        placement = (RaggedShard((0, 1), (0, 1, 5, 2, 0, 0, 0, 0)),)
        dtensor = distribute_tensor(tensor.clone(), device_mesh, placement)
        self._run_add(dtensor)
        placement = (RaggedShard((0,), (0, 1, 0, 0, 0, 0, 0, 0)),)
        dtensor = distribute_tensor(tensor.clone(), device_mesh, placement)
        self._run_add(dtensor)
        placement = (RaggedShard((0, 1), (455, 1000, 2000, 641, 0, 0, 0, 0)),)
        dtensor = distribute_tensor(tensor.clone(), device_mesh, placement)
        self._run_add(dtensor)
        tensor = tensor.view(16, 4, 64)
        placement = (RaggedShard((0,), (0, 1, 5, 2, 0, 0, 0, 0)),)
        dtensor = distribute_tensor(tensor.clone(), device_mesh, placement)
        self._run_add(dtensor)
        placement = (RaggedShard((0, 1), (0, 1, 5, 2, 0, 0, 0, 0)),)
        dtensor = distribute_tensor(tensor.clone(), device_mesh, placement)
        self._run_add(dtensor)
        placement = (RaggedShard((0,), (0, 1, 0, 0, 0, 0, 0, 0)),)
        dtensor = distribute_tensor(tensor.clone(), device_mesh, placement)
        self._run_add(dtensor)

    @with_comms
    def test_strided_ragged_shard_element_add(self):
        device_mesh = init_device_mesh(self.device_type, (self.world_size // 2, 2), mesh_dim_names=("fsdp", "ep"))
        tensor = torch.zeros(64, 128, dtype=torch.int64, device="cuda")
        dtensor = make_strided_shard(tensor.clone(), device_mesh, dims=(0,), local_units=(0, 1, 5, 2))
        self._run_add(dtensor)
        dtensor = make_strided_shard(tensor.clone(), device_mesh, dims=(0, 1), local_units=(0, 1, 5, 2))
        self._run_add(dtensor)
        dtensor = make_strided_shard(tensor.clone(), device_mesh, dims=(0,), local_units=(0, 1, 0, 0))
        self._run_add(dtensor)
        dtensor = make_strided_shard(tensor.clone(), device_mesh, dims=(0, 1), local_units=(455, 1000, 2000, 641))
        self._run_add(dtensor)
        tensor = tensor.view(16, 8, 64)
        dtensor = make_strided_shard(tensor.clone(), device_mesh, dims=(0,), local_units=(0, 1, 5, 2))
        self._run_add(dtensor)
        dtensor = make_strided_shard(tensor.clone(), device_mesh, dims=(0, 1), local_units=(0, 1, 5, 2))
        self._run_add(dtensor)
        dtensor = make_strided_shard(tensor.clone(), device_mesh, dims=(0,), local_units=(0, 1, 0, 0))
        self._run_add(dtensor)


if __name__ == "__main__":
    run_tests()
