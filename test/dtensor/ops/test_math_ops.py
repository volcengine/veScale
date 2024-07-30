################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2024 ByteDance Ltd. and/or its affiliates.
################################################################################

import itertools
from unittest import skip
from common_dtensor import (
    DTensorTestBase,
    skip_unless_torch_gpu,
    with_comms,
)
from unittest import skip

import torch
from torch.testing._internal.common_utils import run_tests, instantiate_parametrized_tests, parametrize
from vescale import distribute_tensor
from vescale.dtensor.placement_types import Replicate, Shard


class DistMathOpsTest(DTensorTestBase):
    def linear_op_reductions(self, op_str):
        device_mesh = self.build_device_mesh()
        shard_spec = [Shard(0)]

        tensor = torch.randn(12, 8, 8)
        dtensor = distribute_tensor(tensor, device_mesh, shard_spec)

        op = getattr(tensor, op_str)
        op_dt = getattr(dtensor, op_str)

        keep_dim_or_not = [True, False, None]
        for dim in range(tensor.ndim):
            for keep_dim in keep_dim_or_not:
                args = (dim, keep_dim) if keep_dim is not None else (dim,)
                if op_str in ("max", "min"):
                    # min and max return a tuple when dim specified
                    dim_reduced_tensor, _ = op(*args)
                    dt_reduced, _ = op_dt(*args)
                else:
                    dim_reduced_tensor = op(*args)
                    dt_reduced = op_dt(*args)
                dt_dim_reduced_tensor = dt_reduced.full_tensor()
                self.assertEqual(dt_dim_reduced_tensor, dim_reduced_tensor)

        full_reduced_tensor = op()
        dt_full_reduced = op_dt().full_tensor()
        self.assertEqual(dt_full_reduced, full_reduced_tensor)

    @with_comms
    def test_linear_op_reductions(self):
        for op_str in ("all", "sum", "prod", "max", "min"):
            self.linear_op_reductions(op_str)

    @with_comms
    @skip_unless_torch_gpu
    def test_mean(self):
        self.linear_op_reductions("mean")

    # TODO: forward test can be removed once test_softmax_with_bwd passes on CPU
    @with_comms
    @skip("failed")
    def test_softmax_fwd(self):
        device_mesh = self.build_device_mesh()

        x = torch.rand(8, 12, 16, device=self.device_type)
        dims = range(3)  # used to convert -1 to the actual dim
        softmax_dims = [-1, 0, 1, 2]
        shard_dims = [-1, 0, 1, 2]
        test_list = list(itertools.product(softmax_dims, shard_dims))

        for softmax_dim, shard_dim in test_list:
            local_y = torch.nn.functional.softmax(x, dim=softmax_dim, dtype=torch.float32)
            dist_x = distribute_tensor(x, device_mesh, [Shard(shard_dim)])
            if dims[shard_dim] == dims[softmax_dim]:
                with self.assertRaisesRegex(Exception, "Cannot run .* on sharding dimension!$"):
                    dist_y = torch.nn.functional.softmax(dist_x, dim=softmax_dim, dtype=torch.float32)
            else:
                dist_y = torch.nn.functional.softmax(dist_x, dim=softmax_dim, dtype=torch.float32)
                self.assertTrue(dist_y.placements[0].is_shard(dim=shard_dim))
                dist_y = dist_y.redistribute(device_mesh, [Replicate()])
                self.assertEqual(dist_y.to_local(), local_y)

    @with_comms
    @parametrize("func", [torch.argmax, torch.argmin])
    def test_arg_max_arg_min(self, func):
        device_mesh = self.build_device_mesh()
        shard_spec = [Shard(0)]
        tensor = torch.randn(12, 8, 8)
        dtensor = distribute_tensor(tensor, device_mesh, shard_spec)

        keep_dim_or_not = [True, False, None]
        for dim in range(1, tensor.ndim):
            for keep_dim in keep_dim_or_not:
                args = (dim, keep_dim) if keep_dim is not None else (dim,)
                dt_result = func(dtensor, *args)
                t_result = func(tensor, *args)
                self.assertEqual(dt_result.full_tensor(), t_result)

        shard_spec = [Replicate()]
        tensor = torch.randn(12, 8, 8)
        dtensor = distribute_tensor(tensor, device_mesh, shard_spec)
        dt_result = func(dtensor)
        t_result = func(tensor)
        self.assertEqual(dt_result.full_tensor(), t_result)

    # TODO: get test_softmax_with_bwd pass on CPU
    # DTensor's _softmax_backward_data produces wrong result on CPU on certain dimension.
    # fail_on_cpu_list = [(0, -1), (1, -1)]
    @with_comms
    @skip_unless_torch_gpu
    @skip("failed")
    def test_softmax_with_bwd(self):
        device_mesh = self.build_device_mesh()

        dims = range(3)  # used to convert -1 to the actual dim
        softmax_dims = [-1, 0, 1, 2]
        shard_dims = [-1, 0, 1, 2]
        test_list = list(itertools.product(softmax_dims, shard_dims))

        for params in test_list:
            softmax_dim, shard_dim = params
            x = torch.rand(8, 12, 16, device=self.device_type, requires_grad=True)
            self.assertTrue(x.requires_grad)
            local_y = torch.nn.functional.softmax(x, dim=softmax_dim, dtype=torch.float32).sum()
            local_y.backward()

            dist_x = distribute_tensor(x, device_mesh, [Shard(shard_dim)])
            self.assertTrue(dist_x.requires_grad)
            if dims[softmax_dim] == dims[shard_dim]:
                with self.assertRaisesRegex(Exception, "Cannot run .* on sharding dimension!$"):
                    dist_softmax = dist_x.softmax(dim=softmax_dim)
            else:
                dist_softmax = dist_x.softmax(dim=softmax_dim)
                self.assertTrue(dist_softmax.placements[0].is_shard(dim=shard_dim))
                dist_y = dist_softmax.sum()
                dist_y = dist_y.redistribute(device_mesh, [Replicate()])
                self.assertEqual(dist_y.to_local(), local_y)
                self.assertIsNone(dist_x.grad)
                dist_y.backward()
                self.assertIsNotNone(dist_x.grad)
                dist_x_grad = dist_x.grad.redistribute(device_mesh, [Replicate()])
                self.assertEqual(dist_x_grad.to_local(), x.grad)

    @with_comms
    def test_onehot_replicate(self):
        device_mesh = self.build_device_mesh()
        tensor = torch.randint(0, 8, (8, 8))
        dtensor = distribute_tensor(tensor, device_mesh, [Replicate()])
        out = torch.nn.functional.one_hot(tensor, 8)
        d_out = torch.nn.functional.one_hot(dtensor, 8)
        self.assertTrue(d_out.placements[0].is_replicate())
        self.assertEqual(d_out.to_local(), out)

    @with_comms
    def test_onehot_sharded(self):
        device_mesh = self.build_device_mesh()
        tensor = torch.randint(0, 8, (8, 8))
        dtensor = distribute_tensor(tensor, device_mesh, [Shard(0)])
        out = torch.nn.functional.one_hot(tensor, 8)
        d_out = torch.nn.functional.one_hot(dtensor, 8)
        self.assertTrue(d_out.placements[0].is_shard(0))
        self.assertEqual(d_out.full_tensor(), out)

    @with_comms
    def test_mse_loss(self):
        device_mesh = self.build_device_mesh()
        tensor = torch.rand((8, 8), requires_grad=True)
        dtensor = distribute_tensor(tensor, device_mesh, [Replicate()])
        loss = torch.nn.MSELoss()

        label = torch.rand((8, 8))
        d_label = distribute_tensor(label, device_mesh, [Replicate()])

        local_loss = loss(tensor, label)
        d_loss = loss(dtensor, d_label)
        local_loss.backward()
        d_loss.backward()

        self.assertEqual(tensor.grad, dtensor.grad.to_local())

    @with_comms
    def test_topk(self):
        device_mesh = self.build_device_mesh()
        tensor = torch.randn(8, 8)
        topk_dim = 0
        shard_dim = 1
        dtensor = distribute_tensor(tensor, device_mesh, [Shard(shard_dim)])
        local_result = torch.topk(tensor, 2, topk_dim)
        d_result = torch.topk(dtensor, 2, topk_dim)
        self.assertTrue(d_result.values.placements[0].is_shard(dim=shard_dim))
        self.assertEqual(d_result.values.full_tensor(), local_result.values)

    @with_comms
    def test_topk_no_dim(self):
        device_mesh = self.build_device_mesh()
        tensor = torch.randn(8, 8)
        dtensor = distribute_tensor(tensor, device_mesh, [Shard(1)])
        topk_no_dim = torch.topk(tensor, 2)
        dtopk_no_dim = torch.topk(dtensor, 2)

    @with_comms
    def test_topk_backward(self):
        device_mesh = self.build_device_mesh()
        tensor = torch.randn((8, 8), requires_grad=True)
        topk_dim = 0
        shard_dim = 1
        dtensor = distribute_tensor(tensor, device_mesh, [Replicate()])
        local_result = torch.topk(tensor, 2, topk_dim)
        d_result = torch.topk(dtensor, 2, topk_dim)
        self.assertTrue(d_result.values.placements[0].is_replicate())
        self.assertEqual(d_result.values.to_local(), local_result.values)

        loss = local_result.values.sum()
        d_loss = d_result.values.sum()

        loss.backward()
        d_loss.backward()

        self.assertEqual(tensor.grad, dtensor.grad.to_local())

    @with_comms
    def test_topk_backward_shard(self):
        device_mesh = self.build_device_mesh()
        tensor = torch.randn((8, 8), requires_grad=True)
        topk_dim = 0
        shard_dim = 1
        dtensor = distribute_tensor(tensor, device_mesh, [Shard(1)])
        local_result = torch.topk(tensor, 2, topk_dim)
        d_result = torch.topk(dtensor, 2, topk_dim)
        self.assertTrue(d_result.values.placements[0].is_shard(dim=1))
        self.assertEqual(d_result.values.full_tensor(), local_result.values)

        loss = local_result.values.sum()
        d_loss = d_result.values.full_tensor().sum()
        loss.backward()
        d_loss.backward()

        self.assertEqual(tensor.grad, dtensor.grad.redistribute(device_mesh, [Replicate()]).to_local())

    @with_comms
    def test_onehot(self):
        device_mesh = self.build_device_mesh()
        tensor = torch.randint(0, 8, (8, 8))
        dtensor = distribute_tensor(tensor, device_mesh, [Replicate()])
        out = torch.nn.functional.one_hot(tensor, 8)
        d_out = torch.nn.functional.one_hot(dtensor, 8)
        self.assertTrue(d_out.placements[0].is_replicate())
        self.assertEqual(d_out.to_local(), out)

    @with_comms
    def test_where(self):
        device_mesh = self.build_device_mesh()
        tensor = torch.rand((8, 8))
        y = torch.ones((8, 8))
        dtensor = distribute_tensor(tensor, device_mesh, [Replicate()])
        d_y = distribute_tensor(y, device_mesh, [Replicate()])
        out = torch.where(tensor > 0, tensor, y)
        d_out = torch.where(dtensor > 0, dtensor, d_y)
        self.assertTrue(d_out.placements[0].is_replicate())
        self.assertEqual(d_out.to_local(), out)

        dtensor = distribute_tensor(tensor, device_mesh, [Shard(0)])
        d_y = distribute_tensor(y, device_mesh, [Shard(0)])

        out = torch.where(tensor > 0, tensor, y)
        d_out = torch.where(dtensor > 0, dtensor, d_y)
        self.assertTrue(d_out.placements[0].is_shard(dim=0))
        self.assertEqual(d_out.full_tensor(), out)

    @with_comms
    def test_where_backward(self):
        device_mesh = self.build_device_mesh()
        tensor = torch.rand((8, 8), requires_grad=True)
        y = torch.ones((8, 8), requires_grad=True)
        dtensor = distribute_tensor(tensor, device_mesh, [Replicate()])
        d_y = distribute_tensor(y, device_mesh, [Replicate()])
        out = torch.where(tensor > 0, tensor, y)
        d_out = torch.where(dtensor > 0, dtensor, d_y)
        self.assertTrue(d_out.placements[0].is_replicate())
        self.assertEqual(d_out.to_local(), out)
        loss = out.sum()
        loss.backward()
        d_loss = d_out.sum()
        d_loss.backward()
        self.assertTrue(dtensor.grad.placements[0].is_replicate())
        self.assertEqual(dtensor.grad.to_local(), tensor.grad)

    @with_comms
    def test_where_backward_shard(self):
        device_mesh = self.build_device_mesh()
        tensor = torch.rand((8, 8), requires_grad=True)
        y = torch.ones((8, 8), requires_grad=True)
        dtensor = distribute_tensor(tensor, device_mesh, [Shard(0)])
        d_y = distribute_tensor(y, device_mesh, [Shard(0)])
        out = torch.where(tensor > 0, tensor, y)
        d_out = torch.where(dtensor > 0, dtensor, d_y)
        loss = out.sum()
        loss.backward()
        d_loss = d_out.redistribute(device_mesh, [Replicate()]).sum()
        d_loss.backward()
        self.assertTrue(dtensor.grad.placements[0].is_shard(dim=0))
        self.assertEqual(dtensor.grad.full_tensor(), tensor.grad)

    @with_comms
    def test_unique(self):
        # TODO: support specifying dim, and it should be implemented in aten.unique_dim
        device_mesh = self.build_device_mesh()
        tensor = torch.randn(8, 8)
        dtensor = distribute_tensor(tensor, device_mesh, [Replicate()])
        local_result = torch.unique(tensor)
        d_result = torch.unique(dtensor)
        self.assertEqual(d_result.to_local(), local_result)

        local_result, local_inverse = torch.unique(tensor, return_inverse=True)
        d_result, d_inverse = torch.unique(dtensor, return_inverse=True)
        self.assertEqual(d_result.to_local(), local_result)
        self.assertEqual(d_inverse.to_local(), local_inverse)

        local_result, local_counts = torch.unique(tensor, return_counts=True)
        d_result, d_counts = torch.unique(dtensor, return_counts=True)
        self.assertEqual(d_result.to_local(), local_result)
        self.assertEqual(d_counts.to_local(), local_counts)


instantiate_parametrized_tests(DistMathOpsTest)


if __name__ == "__main__":
    run_tests()
