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

from vescale import distribute_tensor
from vescale.dtensor.placement_types import Replicate, Shard
from vescale.dtensor.dtensor import DTensor
from vescale.dtensor.device_mesh import DeviceMesh
from vescale.initialize.deferred_init import deferred_init, is_deferred, materialize_dtensor, materialize_dparameter
from vescale.dmodule.api import parallelize_module
from vescale.dtensor.random import manual_seed


class TestDeferInitDTensor(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    def _test_accuracy_base(self, op_call, global_shape, sharding, mesh):
        torch.use_deterministic_algorithms(True)

        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        tensor_golden = op_call(global_shape, device=self.device_type)
        dist_golden = distribute_tensor(tensor_golden, mesh, sharding)

        manual_seed(0, mesh)
        tensor_defer = deferred_init(op_call, global_shape)
        dtensor_defer = materialize_dtensor(tensor_defer, mesh, sharding)

        self.assertTrue(
            torch.equal(dtensor_defer.to_local(), dist_golden.to_local()),
            msg=f"{op_call.__name__}({global_shape}), local tensors don't match: {dtensor_defer.to_local()} vs {dist_golden.to_local()}!",
        )
        global_dtensor = dtensor_defer.full_tensor()
        self.assertTrue(
            torch.equal(global_dtensor, tensor_golden),
            msg=f"{op_call.__name__}({global_shape}), global tensors don't match: {global_dtensor} vs {tensor_golden}!",
        )

    @skip_unless_torch_gpu
    @with_comms
    def test_accuracy(self):
        mesh = DeviceMesh("cuda", list(range(self.world_size)))
        for op in (torch.ones, torch.zeros):
            for global_shape in [(4, 16, 16), (4, 5, 16)]:
                for shard in ([Replicate()], [Shard(1)]):
                    self._test_accuracy_base(op, global_shape, shard, mesh)

    @skip_unless_torch_gpu
    @with_comms
    def test_accuracy_random(self):
        mesh = DeviceMesh("cuda", torch.arange(self.world_size))
        for op in (torch.randn, torch.rand):
            for global_shape in [(9, 7), (4, 16, 16), (4, 5, 16)]:
                for shard in ([Replicate()], [Shard(0)], [Shard(1)]):
                    self._test_accuracy_base(op, global_shape, shard, mesh)
        mesh = DeviceMesh("cuda", torch.arange(self.world_size).reshape(self.world_size // 2, 2))
        for op in (torch.randn, torch.rand):
            for global_shape in [(9, 7), (4, 16, 16), (4, 5, 16)]:
                for shard in ([Replicate(), Replicate()], [Shard(0), Shard(1)], [Shard(1), Shard(0)]):
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
    @unittest.skip(
        "torchdistx.deferred_init._C.is_gen_by_random_op doesn't know that nn.Linear is randomly initialized"
    )
    def test_dparameter(self):
        mesh = DeviceMesh("cuda", list(range(self.world_size)))
        # all_shapes = [(4, 4), (5, 9), (13, 7)]
        # all_placesments = [[Shard(0)], [Shard(1)], [Replicate()]]
        all_shapes = [(4, 4)]
        all_placesments = [[Shard(0)]]
        for shape in all_shapes:
            for placements in all_placesments:
                torch.cuda.manual_seed_all(0)
                expected_fc = nn.Linear(*shape, device=self.device_type)
                dist_fc_wgt = distribute_tensor(expected_fc.weight, mesh, placements)
                if mesh.get_rank() == 0:
                    print(f"expected_fc.weight {expected_fc.weight}")

                manual_seed(0, mesh)
                fc = deferred_init(nn.Linear, *shape)
                self.assertTrue(is_deferred(fc))
                self.assertTrue(is_deferred(fc.weight))
                self.assertTrue(is_fake(fc.weight))
                self.assertTrue(not is_deferred(fc.weight.data))  # NOTE
                self.assertTrue(is_fake(fc.weight.data))
                self.assertTrue(fc.weight.requires_grad)
                self.assertTrue(not fc.weight.data.requires_grad)

                dparam = materialize_dparameter(fc.weight, mesh, placements)
                print(f"rank {mesh.get_rank()} dparam.data {dparam.data._local_tensor}")
                self.assertTrue(isinstance(dparam, nn.Parameter))
                self.assertTrue(dparam.requires_grad)
                self.assertTrue(isinstance(dparam.data, DTensor))
                self.assertTrue(not dparam.data.requires_grad)
                self.assertTrue(isinstance(dparam.data._local_tensor, torch.Tensor))

                self.assertEqual(dparam.data._local_tensor, dist_fc_wgt._local_tensor, atol=0.0, rtol=0.0)
                full_dparam = dparam.data.full_tensor()
                self.assertEqual(full_dparam, expected_fc.weight, atol=0.0, rtol=0.0)


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
