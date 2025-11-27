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
from torch.testing._internal.common_utils import run_tests

from vescale.dtensor import distribute_tensor
from vescale.dtensor.placement_types import RaggedShard

from common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from dtensor.ragged_shard.utils import make_strided_shard


# TODO(jiacheng) test
# TODO(jiacheng) test adam
# TODO(jiacheng) test sgd
# TODO(jiacheng) test all reduce ops
# TODO(jiacheng) test view ops
# TODO(jiacheng) test index ops
# TODO(jiacheng) test element-wise op
# TODO(jiacheng) make a list of ops we do not support


# TODO(jiacheng) test foreach norm
# TODO(jiacheng) test compute_local_shape_and_global_offset


class TestNorm(DTensorTestBase):
    @property
    def world_size(self):
        return 8

    def _run_norm(self, dtensor, local_tensor, *, norm_ord, norm_dim):
        full_norm = torch.linalg.vector_norm(local_tensor, ord=norm_ord, dim=norm_dim)
        dtensor_partial_norm = torch.linalg.vector_norm(dtensor, ord=norm_ord, dim=norm_dim)
        dtensor_full_norm = dtensor_partial_norm.full_tensor()
        self.assertAllClose(full_norm, dtensor_full_norm, atol=1e-06, rtol=1e-06)

    @with_comms
    def test_ragged_shard_norm_no_dim(self):
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        tensor = torch.arange(0, 32 * 128, dtype=torch.float32, device="cuda").view(32, 128)
        placement = (RaggedShard((0,), (0, 1, 5, 2, 0, 0, 0, 0)),)
        dtensor = distribute_tensor(tensor, device_mesh, placement)
        self._run_norm(dtensor, tensor, norm_ord=2, norm_dim=None)
        placement = (RaggedShard((0, 1), (0, 1, 5, 2, 0, 0, 0, 0)),)
        dtensor = distribute_tensor(tensor, device_mesh, placement)
        self._run_norm(dtensor, tensor, norm_ord=2, norm_dim=None)
        placement = (RaggedShard((0,), (0, 1, 0, 0, 0, 0, 0, 0)),)
        dtensor = distribute_tensor(tensor, device_mesh, placement)
        self._run_norm(dtensor, tensor, norm_ord=2, norm_dim=None)
        placement = (RaggedShard((0, 1), (455, 1000, 2000, 641, 0, 0, 0, 0)),)
        dtensor = distribute_tensor(tensor, device_mesh, placement)
        self._run_norm(dtensor, tensor, norm_ord=2, norm_dim=None)
        tensor = tensor.view(16, 4, 64)
        placement = (RaggedShard((0,), (0, 1, 5, 2, 0, 0, 0, 0)),)
        dtensor = distribute_tensor(tensor, device_mesh, placement)
        self._run_norm(dtensor, tensor, norm_ord=2, norm_dim=None)
        placement = (RaggedShard((0, 1), (0, 1, 5, 2, 0, 0, 0, 0)),)
        dtensor = distribute_tensor(tensor, device_mesh, placement)
        self._run_norm(dtensor, tensor, norm_ord=2, norm_dim=None)
        placement = (RaggedShard((0,), (0, 1, 0, 0, 0, 0, 0, 0)),)
        dtensor = distribute_tensor(tensor, device_mesh, placement)
        self._run_norm(dtensor, tensor, norm_ord=2, norm_dim=None)

    @with_comms
    def test_ragged_shard_norm_dim0(self):
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        tensor = torch.arange(0, 32 * 128, dtype=torch.float32, device="cuda").view(32, 128)
        placement = (RaggedShard((0,), (0, 1, 5, 2, 0, 0, 0, 0)),)
        dtensor = distribute_tensor(tensor, device_mesh, placement)
        self._run_norm(dtensor, tensor, norm_ord=2, norm_dim=0)
        placement = (RaggedShard((0, 1), (0, 1, 5, 2, 0, 0, 0, 0)),)
        dtensor = distribute_tensor(tensor, device_mesh, placement)
        self._run_norm(dtensor, tensor, norm_ord=2, norm_dim=0)
        placement = (RaggedShard((0,), (0, 1, 0, 0, 0, 0, 0, 0)),)
        dtensor = distribute_tensor(tensor, device_mesh, placement)
        self._run_norm(dtensor, tensor, norm_ord=2, norm_dim=0)
        placement = (RaggedShard((0, 1), (455, 1000, 2000, 641, 0, 0, 0, 0)),)
        dtensor = distribute_tensor(tensor, device_mesh, placement)
        self._run_norm(dtensor, tensor, norm_ord=2, norm_dim=0)
        tensor = tensor.view(16, 4, 64)
        placement = (RaggedShard((0,), (0, 1, 5, 2, 0, 0, 0, 0)),)
        dtensor = distribute_tensor(tensor, device_mesh, placement)
        self._run_norm(dtensor, tensor, norm_ord=2, norm_dim=0)
        placement = (RaggedShard((0, 1), (0, 1, 5, 2, 0, 0, 0, 0)),)
        dtensor = distribute_tensor(tensor, device_mesh, placement)
        self._run_norm(dtensor, tensor, norm_ord=2, norm_dim=0)
        placement = (RaggedShard((0,), (0, 1, 0, 0, 0, 0, 0, 0)),)
        dtensor = distribute_tensor(tensor, device_mesh, placement)
        self._run_norm(dtensor, tensor, norm_ord=2, norm_dim=0)

    @with_comms
    def test_ragged_shard_norm_dim1(self):
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        tensor = torch.arange(0, 32 * 128, dtype=torch.float32, device="cuda").view(32, 128)
        placement = (RaggedShard((0,), (0, 1, 5, 2, 0, 0, 0, 0)),)
        dtensor = distribute_tensor(tensor, device_mesh, placement)
        self._run_norm(dtensor, tensor, norm_ord=2, norm_dim=1)
        placement = (RaggedShard((0, 1), (0, 1, 5, 2, 0, 0, 0, 0)),)
        dtensor = distribute_tensor(tensor, device_mesh, placement)
        self._run_norm(dtensor, tensor, norm_ord=2, norm_dim=1)
        placement = (RaggedShard((0,), (0, 1, 0, 0, 0, 0, 0, 0)),)
        dtensor = distribute_tensor(tensor, device_mesh, placement)
        self._run_norm(dtensor, tensor, norm_ord=2, norm_dim=1)
        placement = (RaggedShard((0, 1), (455, 1000, 2000, 641, 0, 0, 0, 0)),)
        dtensor = distribute_tensor(tensor, device_mesh, placement)
        self._run_norm(dtensor, tensor, norm_ord=2, norm_dim=1)
        tensor = tensor.view(16, 4, 64)
        placement = (RaggedShard((0,), (0, 1, 5, 2, 0, 0, 0, 0)),)
        dtensor = distribute_tensor(tensor, device_mesh, placement)
        self._run_norm(dtensor, tensor, norm_ord=2, norm_dim=1)
        placement = (RaggedShard((0, 1), (0, 1, 5, 2, 0, 0, 0, 0)),)
        dtensor = distribute_tensor(tensor, device_mesh, placement)
        self._run_norm(dtensor, tensor, norm_ord=2, norm_dim=1)
        placement = (RaggedShard((0,), (0, 1, 0, 0, 0, 0, 0, 0)),)
        dtensor = distribute_tensor(tensor, device_mesh, placement)
        self._run_norm(dtensor, tensor, norm_ord=2, norm_dim=1)

    @with_comms
    def test_ragged_shard_norm_dim2(self):
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        tensor = torch.arange(0, 32 * 128, dtype=torch.float32, device="cuda").view(16, 4, 64)
        placement = (RaggedShard((0,), (0, 1, 5, 2, 0, 0, 0, 0)),)
        dtensor = distribute_tensor(tensor, device_mesh, placement)
        self._run_norm(dtensor, tensor, norm_ord=2, norm_dim=2)
        placement = (RaggedShard((0, 1), (0, 1, 5, 2, 0, 0, 0, 0)),)
        dtensor = distribute_tensor(tensor, device_mesh, placement)
        self._run_norm(dtensor, tensor, norm_ord=2, norm_dim=2)
        placement = (RaggedShard((0,), (0, 1, 0, 0, 0, 0, 0, 0)),)
        dtensor = distribute_tensor(tensor, device_mesh, placement)
        self._run_norm(dtensor, tensor, norm_ord=2, norm_dim=2)

    @with_comms
    def test_strided_ragged_shard_norm_no_dim(self):
        device_mesh = init_device_mesh(self.device_type, (self.world_size // 2, 2), mesh_dim_names=("fsdp", "ep"))
        tensor = torch.arange(0, 64 * 128, dtype=torch.float32, device="cuda").view(64, 128)
        dtensor = make_strided_shard(tensor, device_mesh, dims=(0,), local_units=(0, 1, 5, 2))
        self._run_norm(dtensor, tensor, norm_ord=2, norm_dim=None)
        dtensor = make_strided_shard(tensor, device_mesh, dims=(0, 1), local_units=(0, 1, 5, 2))
        self._run_norm(dtensor, tensor, norm_ord=2, norm_dim=None)
        dtensor = make_strided_shard(tensor, device_mesh, dims=(0,), local_units=(0, 1, 0, 0))
        self._run_norm(dtensor, tensor, norm_ord=2, norm_dim=None)
        dtensor = make_strided_shard(tensor, device_mesh, dims=(0, 1), local_units=(455, 1000, 2000, 641))
        self._run_norm(dtensor, tensor, norm_ord=2, norm_dim=None)
        tensor = tensor.view(16, 8, 64)
        dtensor = make_strided_shard(tensor, device_mesh, dims=(0,), local_units=(0, 1, 5, 2))
        self._run_norm(dtensor, tensor, norm_ord=2, norm_dim=None)
        dtensor = make_strided_shard(tensor, device_mesh, dims=(0, 1), local_units=(0, 1, 5, 2))
        self._run_norm(dtensor, tensor, norm_ord=2, norm_dim=None)
        dtensor = make_strided_shard(tensor, device_mesh, dims=(0,), local_units=(0, 1, 0, 0))
        self._run_norm(dtensor, tensor, norm_ord=2, norm_dim=None)

    @with_comms
    def test_strided_ragged_shard_norm_dim0(self):
        device_mesh = init_device_mesh(self.device_type, (self.world_size // 2, 2), mesh_dim_names=("fsdp", "ep"))
        tensor = torch.arange(0, 64 * 128, dtype=torch.float32, device="cuda").view(64, 128)
        dtensor = make_strided_shard(tensor, device_mesh, dims=(0,), local_units=(0, 1, 5, 2))
        self._run_norm(dtensor, tensor, norm_ord=2, norm_dim=0)
        dtensor = make_strided_shard(tensor, device_mesh, dims=(0, 1), local_units=(0, 1, 5, 2))
        self._run_norm(dtensor, tensor, norm_ord=2, norm_dim=0)
        dtensor = make_strided_shard(tensor, device_mesh, dims=(0,), local_units=(0, 1, 0, 0))
        self._run_norm(dtensor, tensor, norm_ord=2, norm_dim=0)
        dtensor = make_strided_shard(tensor, device_mesh, dims=(0, 1), local_units=(455, 1000, 2000, 641))
        self._run_norm(dtensor, tensor, norm_ord=2, norm_dim=0)
        tensor = tensor.view(16, 8, 64)
        dtensor = make_strided_shard(tensor, device_mesh, dims=(0,), local_units=(0, 1, 5, 2))
        self._run_norm(dtensor, tensor, norm_ord=2, norm_dim=0)
        dtensor = make_strided_shard(tensor, device_mesh, dims=(0, 1), local_units=(0, 1, 5, 2))
        self._run_norm(dtensor, tensor, norm_ord=2, norm_dim=0)
        dtensor = make_strided_shard(tensor, device_mesh, dims=(0,), local_units=(0, 1, 0, 0))
        self._run_norm(dtensor, tensor, norm_ord=2, norm_dim=0)

    @with_comms
    def test_strided_ragged_shard_norm_dim1(self):
        device_mesh = init_device_mesh(self.device_type, (self.world_size // 2, 2), mesh_dim_names=("fsdp", "ep"))
        tensor = torch.arange(0, 64 * 128, dtype=torch.float32, device="cuda").view(64, 128)
        dtensor = make_strided_shard(tensor, device_mesh, dims=(0,), local_units=(0, 1, 5, 2))
        self._run_norm(dtensor, tensor, norm_ord=2, norm_dim=1)
        dtensor = make_strided_shard(tensor, device_mesh, dims=(0, 1), local_units=(0, 1, 5, 2))
        self._run_norm(dtensor, tensor, norm_ord=2, norm_dim=1)
        dtensor = make_strided_shard(tensor, device_mesh, dims=(0,), local_units=(0, 1, 0, 0))
        self._run_norm(dtensor, tensor, norm_ord=2, norm_dim=1)
        dtensor = make_strided_shard(
            tensor,
            device_mesh,
            dims=(0, 1),
            local_units=(
                455,
                1000,
                2000,
                641,
            ),
        )
        self._run_norm(dtensor, tensor, norm_ord=2, norm_dim=1)
        tensor = tensor.view(16, 8, 64)
        dtensor = make_strided_shard(tensor, device_mesh, dims=(0,), local_units=(0, 1, 5, 2))
        self._run_norm(dtensor, tensor, norm_ord=2, norm_dim=1)
        dtensor = make_strided_shard(tensor, device_mesh, dims=(0, 1), local_units=(0, 1, 5, 2))
        self._run_norm(dtensor, tensor, norm_ord=2, norm_dim=1)
        dtensor = make_strided_shard(tensor, device_mesh, dims=(0,), local_units=(0, 1, 0, 0))
        self._run_norm(dtensor, tensor, norm_ord=2, norm_dim=1)

    @with_comms
    def test_strided_ragged_shard_norm_dim2(self):
        device_mesh = init_device_mesh(self.device_type, (self.world_size // 2, 2), mesh_dim_names=("fsdp", "ep"))
        tensor = torch.arange(0, 64 * 128, dtype=torch.float32, device="cuda").view(16, 8, 64)
        dtensor = make_strided_shard(tensor, device_mesh, dims=(0,), local_units=(0, 1, 5, 2))
        self._run_norm(dtensor, tensor, norm_ord=2, norm_dim=2)
        dtensor = make_strided_shard(tensor, device_mesh, dims=(0, 1), local_units=(0, 1, 5, 2))
        self._run_norm(dtensor, tensor, norm_ord=2, norm_dim=2)
        dtensor = make_strided_shard(tensor, device_mesh, dims=(0,), local_units=(0, 1, 0, 0))
        self._run_norm(dtensor, tensor, norm_ord=2, norm_dim=2)


if __name__ == "__main__":
    run_tests()
