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

import unittest
from common_dtensor import skip_unless_torch_gpu, with_comms, DTensorTestBase
from torch.testing._internal.common_utils import run_tests

import torch
from torch import nn
from torch.cuda import empty_cache, memory_reserved, memory_stats, reset_peak_memory_stats, synchronize
from torchdistx.fake import is_fake

from vescale.dtensor.placement_types import Replicate, Shard
from vescale.dtensor.api import distribute_tensor
from vescale.dtensor.dtensor import DTensor
from vescale.dtensor.device_mesh import DeviceMesh
from vescale.dtensor import randn
from vescale.initialize.deferred_init import deferred_init, is_deferred, materialize_dtensor, materialize_dparameter
from vescale.dmodule.api import parallelize_module


class TestDeferInitDTensor(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    def _test_accuracy_base(self, op_call, global_shape, sharding, mesh):
        torch.use_deterministic_algorithms(True)

        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        tensor_golden = op_call(global_shape)
        dtensor_golden = distribute_tensor(tensor_golden, mesh, sharding)

        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        tensor_defer = deferred_init(op_call, global_shape)
        dtensor_defer = materialize_dtensor(tensor_defer, mesh, sharding)
        self.assertTrue(
            torch.equal(dtensor_defer._local_tensor, dtensor_golden._local_tensor),
            msg=f"{op_call.__name__}({global_shape}), not match: {dtensor_defer} vs {dtensor_golden}!",
        )

    @skip_unless_torch_gpu
    @with_comms
    def test_accuracy(self):
        mesh = DeviceMesh("cuda", list(range(self.world_size)))
        for op in (torch.ones, torch.zeros):
            for global_shape in [(4, 16, 16), (4, 5, 16)]:
                for shard in ([Replicate()], [Shard(1)]):
                    self._test_accuracy_base(op, global_shape, shard, mesh)

    @unittest.skip("FIXME!")
    @skip_unless_torch_gpu
    @with_comms
    def test_accuracy_random(self):
        mesh = DeviceMesh("cuda", list(range(self.world_size)))
        for op in (torch.randn, torch.rand):
            for global_shape in [(4, 16, 16), (4, 5, 16)]:
                for shard in ([Replicate()], [Shard(1)]):
                    self._test_accuracy_base(op, global_shape, shard, mesh)

    def _assert_eq_empty(self, x: torch.Tensor, y: torch.Tensor):
        # self.assertTrue(torch.equal(x, y))
        self.assertTrue(x.shape == y.shape)
        self.assertTrue(x.dtype == y.dtype)
        self.assertTrue(x.device.type == y.device.type)
        self.assertTrue(x.layout == y.layout)
        self.assertTrue(x.requires_grad == y.requires_grad)

    @skip_unless_torch_gpu
    @with_comms
    def test_accuracy_random2(self):
        mesh = DeviceMesh("cuda", list(range(self.world_size)))

        torch.use_deterministic_algorithms(True)

        # replicate
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        defer_dtensor = deferred_init(torch.randn, (4, 16, 16))
        dtensor_replicate = materialize_dtensor(defer_dtensor, mesh, [Replicate()])

        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        expected_tensor = torch.randn((4, 16, 16), device=mesh.device_type)

        self.assertTrue(torch.equal(dtensor_replicate._local_tensor, expected_tensor))

        # shard
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        defer_dtensor = deferred_init(torch.randn, (4, 16, 16))
        dtensor = materialize_dtensor(defer_dtensor, mesh, [Shard(1)])

        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        dtensor_rand = randn((4, 16, 16), device_mesh=mesh, placements=[Shard(1)])

        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        expected_tensor = torch.randn((4, 16 // self.world_size, 16), device=mesh.device_type)

        self._assert_eq_empty(dtensor._local_tensor, expected_tensor)
        self._assert_eq_empty(dtensor_rand._local_tensor, expected_tensor)
        self._assert_eq_empty(dtensor_rand._local_tensor, dtensor._local_tensor)

    @skip_unless_torch_gpu
    @with_comms
    def test_local_shape(self):
        mesh = DeviceMesh("cuda", list(range(self.world_size)))

        # even shard
        tensor = deferred_init(torch.empty, (4, 16, 16))
        dtensor = materialize_dtensor(tensor, mesh, [Shard(1)])
        self.assertEqual(dtensor._local_tensor.shape, (4, 4, 16))

        # uneven shard
        tensor = deferred_init(torch.empty, (4, 6, 16))
        dtensor = materialize_dtensor(tensor, mesh, [Shard(1)])
        if self.rank in (0, 1, 2):
            self.assertEqual(dtensor._local_tensor.shape, (4, 2, 16))
        else:
            self.assertEqual(dtensor._local_tensor.shape, (4, 0, 16))

    def test_meta_device(self):
        with torch.device("meta"):
            tensor = deferred_init(torch.empty, (4, 16, 16))
        self.assertTrue(is_deferred(tensor))
        self.assertTrue(is_fake(tensor))
        self.assertEqual(tensor.device, torch.device("meta"))

    @skip_unless_torch_gpu
    @with_comms
    def test_on_device(self):
        with torch.device("cpu"):
            tensor = deferred_init(torch.empty, (4, 16, 16))
        self.assertTrue(is_deferred(tensor))
        self.assertTrue(is_fake(tensor))
        self.assertEqual(tensor.device, torch.device("cpu"))

        mesh = DeviceMesh("cuda", list(range(self.world_size)))
        dtensor = materialize_dtensor(tensor, mesh, [Shard(1)])
        # self.assertTrue(not is_deferred(dtensor))
        # self.assertTrue(not is_fake(dtensor))
        self.assertTrue(isinstance(dtensor, DTensor))
        self.assertEqual(dtensor.device.type, torch.device(mesh.device_type).type)

    @skip_unless_torch_gpu
    @with_comms
    def test_memory(self):
        global_shape = (1024 * 1024, self.world_size)  # even shard only
        placements = [Shard(1)]

        empty_cache()
        reset_peak_memory_stats()
        synchronize()
        self.assertEqual(memory_reserved(), 0)

        mesh = DeviceMesh("cuda", list(range(self.world_size)))

        synchronize()
        m1 = memory_stats()

        tensor = deferred_init(torch.zeros, global_shape)

        synchronize()
        m2 = memory_stats()
        self.assertEqual(m1, m2, msg="`deferred_init` should have zero cuda allocation!")

        dtensor = materialize_dtensor(tensor, mesh, placements)

        synchronize()
        m3 = memory_stats()

        self.assertEqual(
            m3["allocated_bytes.large_pool.allocated"] - m2["allocated_bytes.large_pool.allocated"],
            dtensor._local_tensor.numel() * 4,
            msg="`materialize_dtensor` should only allocate should be local tensor!",
        )
        self.assertEqual(
            m3["allocation.large_pool.allocated"] - m2["allocation.large_pool.allocated"],
            1,
            msg="`materialize_dtensor` should only allocate should be one tensor!",
        )


class TestDeferInitDParameter(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @skip_unless_torch_gpu
    @with_comms
    def test_dparameter(self):
        mesh = DeviceMesh("cuda", list(range(self.world_size)))

        fc = deferred_init(nn.Linear, 4, 4)
        self.assertTrue(is_deferred(fc))
        self.assertTrue(is_deferred(fc.weight))
        self.assertTrue(is_fake(fc.weight))
        self.assertTrue(not is_deferred(fc.weight.data))  # NOTE
        self.assertTrue(is_fake(fc.weight.data))
        self.assertTrue(fc.weight.requires_grad)
        self.assertTrue(not fc.weight.data.requires_grad)

        if self.rank == 0:
            print("*** BEFORE ***", fc.weight)

        dparam = materialize_dparameter(fc.weight, mesh, [Shard(0)])

        if self.rank == 0:
            print("*** AFTER ***", dparam)

        # self.assertTrue(not is_deferred(dparam))
        # self.assertTrue(not is_fake(dparam))
        # self.assertTrue(not is_deferred(dparam.data))
        # self.assertTrue(not is_fake(dparam.data))
        self.assertTrue(isinstance(dparam, nn.Parameter))
        self.assertTrue(dparam.requires_grad)
        self.assertTrue(isinstance(dparam.data, DTensor))
        self.assertTrue(not dparam.data.requires_grad)
        self.assertTrue(isinstance(dparam.data._local_tensor, torch.Tensor))
        self.assertEqual(dparam.data._local_tensor.shape, (1, 4))
        self.assertTrue(dparam.data._local_tensor.is_cuda)
        self.assertTrue(not dparam.data._local_tensor.requires_grad)


class MLP(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.register_buffer("buffer", torch.ones(hidden_size, requires_grad=False))

    def forward(self, x):
        return self.fc2(self.fc1(x)) + self.buffer


class TestDeferInitDModule(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @skip_unless_torch_gpu
    @with_comms
    def test_dmodule(self):
        mlp = deferred_init(MLP, 8)
        self.assertTrue(is_deferred(mlp))

        mesh = DeviceMesh("cuda", list(range(self.world_size)))
        tensor = torch.randn((8, 8))
        fwd_sharding_plan = {
            "input": [[Replicate()]],
            "fc2.output": [[Replicate()]],
        }
        param_sharding_plan = {
            "fc1.weight": [Shard(0)],
            "fc2.weight": [Shard(1)],
            "buffer": [Replicate()],
        }
        mlp = parallelize_module(mlp, mesh, {"parameter": param_sharding_plan, "forward": fwd_sharding_plan})
        out = mlp(tensor)
        self.assertTrue(isinstance(out, DTensor))
        self.assertTrue(out.placements[0].is_replicate())


if __name__ == "__main__":
    run_tests()
