################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################


import torch
import torch.nn as nn
from torch.cuda import memory_stats
from torch.testing._internal.common_utils import run_tests

import vescale
from vescale import Partial, Replicate, Shard, DeviceMesh, DTensor, distribute_tensor

from common_dtensor import DTensorTestBase, with_comms, skip_unless_torch_version_bigger_than

_512 = 512
_1K = 1 * 1024
_10K = 10 * 1024
_100K = 100 * 1024
_1M = 1 * 1024 * 1024
_10M = 10 * 1024 * 1024
_100M = 100 * 1024 * 1024


class DTensorTestCuda(DTensorTestBase):
    def _match_meta_dtensor(self, create_fn, global_shape, local_shape, placements):
        self.assertTrue(create_fn in (distribute_tensor, DTensor.from_local))
        # test meta tensor
        init_shape = global_shape if create_fn is distribute_tensor else local_shape
        meta_tensor = torch.randn(init_shape, dtype=torch.float, requires_grad=True, device="meta")
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        meta_dtensor = create_fn(meta_tensor, device_mesh, placements)
        self.assertTrue(meta_dtensor.is_meta)
        self.assertEqual(meta_dtensor.shape, global_shape)
        self.assertEqual(meta_dtensor.dtype, torch.float)
        self.assertEqual(meta_dtensor.requires_grad, True)

        meta_local_tensor = meta_dtensor.to_local()
        self.assertTrue(meta_local_tensor.is_meta)
        self.assertEqual(meta_local_tensor.shape, local_shape)
        self.assertEqual(meta_local_tensor.dtype, torch.float)
        self.assertEqual(meta_local_tensor.requires_grad, True)

        # 1. test materialization using meta_local_tensor
        local_tensor = torch.empty_like(meta_local_tensor, device=self.device_type).fill_(1.2)
        local_tensor.requires_grad_(True)
        self.assertFalse(local_tensor.is_meta)
        self.assertEqual(local_tensor.device.type, self.device_type)

        the_dtensor = DTensor.from_local(local_tensor, device_mesh, placements)
        self.assertFalse(the_dtensor.is_meta)
        self.assertEqual(the_dtensor.device.type, self.device_type)
        self.assertEqual(the_dtensor.shape, global_shape)
        self.assertEqual(the_dtensor.dtype, torch.float)
        self.assertEqual(the_dtensor.requires_grad, local_tensor.requires_grad)

        value_tensor = torch.empty_like(the_dtensor.to_local()).fill_(1.2).requires_grad_(True)
        self.assertEqual(the_dtensor.to_local(), value_tensor)

        # 2. test materialization using meta_dtensor with prop_op # TODO: fix red testing
        # global_tensor = torch.empty_like(meta_dtensor, device=self.device_type).fill_(1.2)
        # global_tensor.requires_grad_(True)
        # if self.rank == 0: print(f"{global_tensor}")
        # self.assertTrue(vescale.allclose(global_tensor, the_dtensor))

    @with_comms
    @skip_unless_torch_version_bigger_than(torch_version="2.2")
    def test_distribute_tensor(self):
        global_shape = (self.world_size * _1K, self.world_size * _1K)
        all_placements = [[Shard(0)], [Shard(1)], [Replicate()], [Partial()]]
        all_local_shape = [
            (global_shape[0] // self.world_size, global_shape[1]),
            (global_shape[0], global_shape[1] // self.world_size),
            global_shape,
            global_shape,
        ]
        for placements, local_shape in zip(all_placements, all_local_shape):
            self._match_meta_dtensor(distribute_tensor, global_shape, local_shape, placements)

    @with_comms
    @skip_unless_torch_version_bigger_than(torch_version="2.2")
    def test_from_local(self):
        local_shape = (self.world_size * _1K, self.world_size * _1K)
        all_placements = [[Shard(0)], [Shard(1)], [Replicate()], [Partial()]]
        all_global_shape = [
            (local_shape[0] * self.world_size, local_shape[1]),
            (local_shape[0], local_shape[1] * self.world_size),
            local_shape,
            local_shape,
        ]
        for placements, global_shape in zip(all_placements, all_global_shape):
            self._match_meta_dtensor(DTensor.from_local, global_shape, local_shape, placements)

    # @with_comms
    # def test_nn_module_with_meta_dtensor(self):
    #     # create nn.Module with meta Tensor
    #     m1 = memory_stats(self.rank)
    #     with torch.device("meta"): # any tensor created within this torch.device context manager will be on the meta device.
    #         model = DummyMLP()
    #     # replace nn.Module's meta Tensor with meta DTensor
    #     model = parallelize_mlp(model, "net1", "net2", device_mesh)
    #     m2 = memory_stats(self.rank)
    #     self.assertEqual(m1, m2) # nn.Module has no .is_meta
    #     # materialize nn.Module's meta DTensor on GPU
    #     model.to_empty(device=self.device_type) # TODO: fix
    #     model.requires_grad_(True) # TODO: fix
    #     model.reset_parameters() # TODO: fix
    #     optim = torch.optim.SGD(model.parameters(), lr=0.1)

    @with_comms
    def test_nn_module_with_meta_init(self):
        class DummyMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.net1 = nn.Linear(5, 1024, bias=True)
                self.relu = nn.ReLU()
                self.net2 = nn.Linear(1024, 4, bias=False)

            def forward(self, x):
                return self.net2(self.relu(self.net1(x)))

            def reset_parameters(self, *args, **kwargs):
                with torch.no_grad():
                    self.net1.weight.fill_(0.5)
                    self.net1.bias.fill_(1.5)
                    self.net2.weight.fill_(1)

        def replicate_input(input: torch.Tensor, device_mesh: DeviceMesh) -> DTensor:
            return DTensor.from_local(input, device_mesh, [Replicate()], run_check=False)

        def replicate_output(output: DTensor, device_mesh: DeviceMesh) -> DTensor:
            return output.redistribute(device_mesh, [Replicate()])

        def parallelize_mlp(create_fn, mlp: nn.Module, linear1: str, linear2: str, device_mesh: DeviceMesh):
            first_linear = mlp._modules[linear1]
            for name, param in first_linear.named_parameters():
                placements = [Shard(0)]
                dist_param = nn.Parameter(create_fn(param, device_mesh, placements))
                first_linear.register_parameter(name, dist_param)

            second_linear = mlp._modules[linear2]
            for name, param in second_linear.named_parameters():
                placements = [Shard(1)] if name == "weight" else [Partial()]
                dist_param = nn.Parameter(create_fn(param, device_mesh, placements))
                second_linear.register_parameter(name, dist_param)

            return mlp

        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        device_mesh_gold = DeviceMesh(self.device_type, list(range(self.world_size)))
        # create nn.Module with meta Tensor
        m1 = memory_stats(self.rank) if self.device_type == "cuda" else 0
        with torch.device(
            "meta"
        ):  # any tensor created within this torch.device context manager will be on the meta device.
            model = DummyMLP()
        m2 = memory_stats(self.rank) if self.device_type == "cuda" else 0
        self.assertEqual(m1, m2)  # nn.Module has no .is_meta
        # materialize nn.Module's meta Tensor with DTensor
        model = parallelize_mlp(
            lambda t, d, p: vescale.empty(
                t.shape, dtype=t.dtype, requires_grad=t.requires_grad, device_mesh=d, placements=p
            ),
            model,
            "net1",
            "net2",
            device_mesh,
        )
        model.reset_parameters()
        optim = torch.optim.SGD(model.parameters(), lr=0.1)

        # materialize golden
        with torch.device(self.device_type):
            model_gold = DummyMLP()
        model_gold = parallelize_mlp(
            lambda t, d, p: distribute_tensor(t, d, p), model_gold, "net1", "net2", device_mesh_gold
        )
        model_gold.reset_parameters()
        optim_gold = torch.optim.SGD(model_gold.parameters(), lr=0.1)

        # create data
        torch.manual_seed(0)
        data = torch.randn(20, 5, device=self.device_type).requires_grad_(True)
        data_gold = data.detach().clone().requires_grad_(True)
        self.assertEqual(data, data_gold)

        # match forward
        output = replicate_output(model(replicate_input(data, device_mesh)), device_mesh).to_local()
        output_gold = replicate_output(model_gold(replicate_input(data_gold, device_mesh_gold)), device_mesh).to_local()
        self.assertEqual(output, output_gold)

        # match backward
        output.sum().backward()
        output_gold.sum().backward()
        self.assertTrue(data.grad is not None and data_gold.grad is not None)
        self.assertEqual(data.grad, data_gold.grad)

        # match optimizer step
        optim.step()
        optim_gold.step()
        for (n, p), (n_gold, p_gold) in zip(model.named_parameters(), model_gold.named_parameters()):
            self.assertEqual(n, n_gold)
            self.assertEqual(p.to_local(), p_gold.to_local())

        # match next forward
        torch.manual_seed(1)
        data = torch.randn(20, 5, device=self.device_type)
        data_gold = data.detach().clone()
        self.assertEqual(data, data_gold)

        output = replicate_output(model(replicate_input(data, device_mesh)), device_mesh).to_local()
        output_gold = replicate_output(model_gold(replicate_input(data_gold, device_mesh_gold)), device_mesh).to_local()
        self.assertEqual(output, output_gold)


if __name__ == "__main__":
    run_tests()
