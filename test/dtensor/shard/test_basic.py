################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

"""Test sharding for operators."""

import unittest
from common_dtensor import DTensorTestBase, with_comms

import torch
from torch.distributed.distributed_c10d import ReduceOp
from torch.testing._internal.common_utils import run_tests

from vescale import DeviceMesh, Partial, Replicate, Shard, distribute_tensor
from vescale.dtensor._collective_utils import mesh_all_reduce

CHECK_RTOL = 1e-4
CHECK_ATOL = 1e-4


class ShardingBasicTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        else:
            return 8

    @with_comms
    def test_2dmatmul_shard(self):
        batch_size = 8
        in_channel = 16
        out_channel = 8
        device_mesh_1d = DeviceMesh(self.device_type, torch.arange(self.world_size))
        device_mesh_2d = DeviceMesh(self.device_type, torch.arange(self.world_size).reshape(self.world_size // 2, 2))

        lhs_tensor = torch.randn(batch_size, in_channel)
        rhs_tensor = torch.randn(in_channel, out_channel)
        # local cal
        local_output = torch.matmul(lhs_tensor, rhs_tensor)

        # mini case1: 1d sharding oc,TP
        lhs_dtensor_1d = distribute_tensor(lhs_tensor, device_mesh=device_mesh_1d, placements=[Replicate()])
        rhs_dtensor_1d = distribute_tensor(rhs_tensor, device_mesh=device_mesh_1d, placements=[Shard(1)])
        output = torch.matmul(lhs_dtensor_1d, rhs_dtensor_1d)
        # spec check
        self.assertTrue(repr(output.placements) == "(Shard(dim=1),)")
        # correctness check
        local_output_shard_1d = distribute_tensor(
            local_output, device_mesh=device_mesh_1d, placements=output.placements
        )
        self.assertTrue(
            torch.allclose(local_output_shard_1d.to_local(), output.to_local(), rtol=CHECK_RTOL, atol=CHECK_ATOL)
        )

        # mini case2: sharding batch and oc, DP+TP
        lhs_dtensor_2d = distribute_tensor(lhs_tensor, device_mesh=device_mesh_2d, placements=[Shard(0), Replicate()])
        rhs_dtensor_2d = distribute_tensor(rhs_tensor, device_mesh=device_mesh_2d, placements=[Replicate(), Shard(1)])
        output = torch.matmul(lhs_dtensor_2d, rhs_dtensor_2d)
        # spec check
        self.assertTrue(repr(output.placements) == "(Shard(dim=0), Shard(dim=1))")
        # correctness check
        local_output_shard_2d = distribute_tensor(
            local_output, device_mesh=device_mesh_2d, placements=output.placements
        )
        self.assertTrue(
            torch.allclose(local_output_shard_2d.to_local(), output.to_local(), rtol=CHECK_RTOL, atol=CHECK_ATOL)
        )

    @with_comms
    def test_batchmatmul_shard(self):
        config = {
            "seq_length": 8,
            "head_size": 4,
            "hidden_size": 4 * 4,
            "n_head": max(4, self.world_size),
            "batch_size": 1,
        }
        device_mesh_1d = DeviceMesh(self.device_type, list(range(self.world_size)))

        lhs_tensor = torch.randn(config["batch_size"], config["n_head"], config["seq_length"], config["head_size"])
        rhs_tensor = torch.randn(config["batch_size"], config["n_head"], config["head_size"], config["seq_length"])

        # mini case1: 1d sharding oc,TP
        lhs_dtensor_1d = distribute_tensor(lhs_tensor, device_mesh=device_mesh_1d, placements=[Shard(1)])
        rhs_dtensor_1d = distribute_tensor(rhs_tensor, device_mesh=device_mesh_1d, placements=[Shard(1)])
        output = torch.matmul(lhs_dtensor_1d, rhs_dtensor_1d)
        # spec check
        self.assertTrue(repr(output.placements) == "(Shard(dim=1),)")
        # correctness check
        local_output = torch.matmul(lhs_tensor, rhs_tensor)
        local_output_shard = distribute_tensor(local_output, device_mesh=device_mesh_1d, placements=output.placements)
        self.assertTrue(
            torch.allclose(local_output_shard.to_local(), output.to_local(), rtol=CHECK_RTOL, atol=CHECK_ATOL)
        )


class PartialBasicTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        else:
            return 8

    @with_comms
    @unittest.skip("this test may cause ci hang")
    def test_basic_partial_distribution(self):
        mesh = self.build_device_mesh()
        # NOTE: make sure invoke construtor of `DeviceMesh` first, it will defaultly
        # initialize distribute process environment.
        bias = torch.rand(10).to(mesh.device_type)
        d_bias = distribute_tensor(bias, mesh, [Partial()])

        if mesh.get_rank() == 0:
            res = torch.allclose(bias, d_bias.to_local(), rtol=CHECK_RTOL, atol=CHECK_ATOL)
        else:
            zero = bias.zero_()
            res = torch.allclose(zero, d_bias.to_local(), rtol=CHECK_RTOL, atol=CHECK_ATOL)
        out = mesh_all_reduce(torch.tensor(res).to(mesh.device_type).to(torch.int), mesh, ReduceOp.SUM, 0)
        self.assertTrue(out.sum() == mesh.size())


if __name__ == "__main__":
    run_tests()
