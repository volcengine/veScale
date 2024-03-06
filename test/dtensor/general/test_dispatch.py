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

from common_dtensor import DTensorTestBase, with_comms, skip_unless_torch_gpu

import torch
from torch.testing._internal.common_utils import run_tests
from vescale import DeviceMesh, DTensor
from vescale.dtensor.placement_types import Shard, Replicate, Partial

aten = torch.ops.aten


class TestBypassOps(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @skip_unless_torch_gpu
    @with_comms
    def test_is_same_size(self):
        device_mesh = DeviceMesh("cuda", list(range(self.world_size)))

        local_tensor1 = torch.ones((2, 8), device="cuda")
        dtensor1 = DTensor.from_local(local_tensor1, device_mesh, [Shard(0)])

        local_tensor2 = torch.ones((2, 8), device="cuda")
        dtensor2 = DTensor.from_local(local_tensor2, device_mesh, [Shard(0)])
        self.assertTrue(aten.is_same_size(dtensor1, dtensor2) is True)

        local_tensor3 = torch.ones((2, 16), device="cuda")
        dtensor3 = DTensor.from_local(local_tensor3, device_mesh, [Shard(0)])
        self.assertTrue(aten.is_same_size(dtensor1, dtensor3) is False)

    @skip_unless_torch_gpu
    @with_comms
    def test_to_copy(self):
        device_mesh = DeviceMesh("cuda", list(range(self.world_size)))
        local_tensor = torch.randn((2, 8), device="cuda", dtype=torch.float32)
        gpu_dtensor = DTensor.from_local(local_tensor, device_mesh, [Shard(0)])
        cpu_dtensor = aten._to_copy(gpu_dtensor, device="cpu", dtype=torch.bfloat16)
        self.assertTrue(cpu_dtensor.device.type == "cpu")
        self.assertTrue(cpu_dtensor.placements == gpu_dtensor.placements)
        self.assertTrue(cpu_dtensor._spec.tensor_meta.shape == gpu_dtensor._spec.tensor_meta.shape)
        self.assertTrue(cpu_dtensor._spec.tensor_meta.stride == gpu_dtensor._spec.tensor_meta.stride)
        self.assertTrue(cpu_dtensor._spec.tensor_meta.dtype == torch.bfloat16)

    @skip_unless_torch_gpu
    @with_comms
    def test_equal(self):
        device_mesh = DeviceMesh("cuda", list(range(self.world_size)))

        local_tensor1 = torch.ones((2, 8), dtype=torch.float32, device="cuda")
        dtensor1 = DTensor.from_local(local_tensor1, device_mesh, [Shard(0)])

        local_tensor2 = torch.ones((2, 8), dtype=torch.float32, device="cuda")
        dtensor2 = DTensor.from_local(local_tensor2, device_mesh, [Shard(0)])
        self.assertTrue(aten.equal(dtensor1, dtensor2) is True)

        local_tensor3 = torch.zeros((2, 8), dtype=torch.float32, device="cuda")
        dtensor3 = DTensor.from_local(local_tensor3, device_mesh, [Shard(0)])
        self.assertTrue(aten.equal(dtensor1, dtensor3) is False)

    @skip_unless_torch_gpu
    @with_comms
    def test_local_scalar_dense(self):
        device_mesh = DeviceMesh("cuda", list(range(self.world_size)))
        local_tensor = torch.ones((2, 8), dtype=torch.float32, device="cuda")
        dtensor = DTensor.from_local(local_tensor, device_mesh, [Shard(0)])
        self.assertTrue(aten._local_scalar_dense(dtensor) == torch.tensor(1.0, dtype=torch.float32, device="cuda"))


class TestTupleStrategyDispatch(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @skip_unless_torch_gpu
    @with_comms
    def test_foreach_add(self):
        device_mesh = DeviceMesh("cuda", list(range(self.world_size)))

        local_tensor1 = torch.ones((2, 8), device="cuda")
        dtensor1 = DTensor.from_local(local_tensor1, device_mesh, [Shard(0)])

        local_tensor2 = torch.ones((8, 2), device="cuda")
        dtensor2 = DTensor.from_local(local_tensor2, device_mesh, [Shard(1)])

        local_tensor3 = torch.ones((8, 8), device="cuda")
        dtensor3 = DTensor.from_local(local_tensor3, device_mesh, [Replicate()])

        local_tensor4 = torch.ones((8, 8), device="cuda")
        dtensor4 = DTensor.from_local(local_tensor4, device_mesh, [Partial()])

        torch.ops.aten._foreach_add_([dtensor1, dtensor2, dtensor3, dtensor4], 1.0)

        self.assertTrue(dtensor1.to_local().mean().item() == 2.0)
        self.assertTrue(dtensor2.to_local().mean().item() == 2.0)
        self.assertTrue(dtensor3.to_local().mean().item() == 2.0)
        self.assertTrue(dtensor4.to_local().mean().item() == 2.0)

        self.assertTrue(dtensor1.placements[0].is_shard(0))
        self.assertTrue(dtensor2.placements[0].is_shard(1))
        self.assertTrue(dtensor3.placements[0].is_replicate())
        self.assertTrue(dtensor4.placements[0].is_partial())


if __name__ == "__main__":
    run_tests()
