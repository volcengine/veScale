################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

"""Test collective operators"""

from common_dtensor import DTensorTestBase, with_comms
from typing import cast

import torch
import torch.distributed._functional_collectives as funcol
from torch.testing._internal.common_utils import run_tests

from vescale import DeviceMesh, distribute_tensor
from vescale.dtensor._collective_utils import mesh_all_gather, mesh_all_reduce, mesh_reduce_scatter
from vescale.dtensor.api import vescale_all_gather, vescale_all_reduce, vescale_reduce_scatter
from vescale.dtensor.placement_types import Partial, Shard


class TensorCommBasicTest(DTensorTestBase):
    @property
    def world_size(self):
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        else:
            return 8

    @with_comms
    def test_reduce_scatter(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        partial_spec = [Partial()]

        origin_tensor = torch.randn(3 * self.world_size, 3)
        partial_tensor = distribute_tensor(origin_tensor, device_mesh, partial_spec)
        partial_placement = cast(Partial, partial_tensor.placements[0])

        sharded_tensor = mesh_reduce_scatter(
            partial_tensor._local_tensor, device_mesh, partial_placement.reduce_op, scatter_dim=0, mesh_dim=0
        )
        chunk_size = origin_tensor.size()[0] // self.world_size

        rank = device_mesh.get_rank()
        torch.testing.assert_close(origin_tensor[rank * chunk_size : (rank + 1) * chunk_size, :], sharded_tensor.cpu())

    @with_comms
    def test_all_reduce(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        partial_spec = [Partial()]

        origin_tensor = torch.randn(3 * self.world_size, 3)
        partial_tensor = distribute_tensor(origin_tensor, device_mesh, partial_spec)
        partial_placement = cast(Partial, partial_tensor.placements[0])

        reduced_tensor = mesh_all_reduce(
            partial_tensor._local_tensor, device_mesh, partial_placement.reduce_op, mesh_dim=0
        )
        torch.testing.assert_close(origin_tensor, reduced_tensor.cpu())

    @with_comms
    def test_all_gather(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]

        origin_tensor = torch.rand(3 * self.world_size, 3)
        sharded_tensor = distribute_tensor(origin_tensor, device_mesh, shard_spec)
        gathered_tensor = mesh_all_gather(
            sharded_tensor._local_tensor, origin_tensor.size(), device_mesh, scatter_dim=0, mesh_dim=0
        )

        torch.testing.assert_close(origin_tensor, gathered_tensor.cpu())


class DTensorCommBasicTest(DTensorTestBase):
    @property
    def world_size(self):
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        else:
            return 8

    @with_comms
    def test_all_gather(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        origin_tensor = torch.randn(3 * self.world_size, 3)
        shard_spec = [Shard(0)]

        sharded_tensor = distribute_tensor(origin_tensor, device_mesh=device_mesh, placements=shard_spec)

        for async_op in [True, False]:
            gathered_tensor = vescale_all_gather(sharded_tensor, mesh_dims=0, async_op=async_op)
            if async_op:
                self.assertTrue(isinstance(gathered_tensor._local_tensor, funcol.AsyncCollectiveTensor))

            torch.testing.assert_close(origin_tensor, gathered_tensor._local_tensor.cpu())

    @with_comms
    def test_all_reduce(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        partial_spec = [Partial()]

        origin_tensor = torch.randn(3 * self.world_size, 3)
        partial_tensor = distribute_tensor(origin_tensor, device_mesh, partial_spec)

        for async_op in [True, False]:
            reduced_tensor = vescale_all_reduce(partial_tensor, mesh_dims=0, async_op=async_op)
            if async_op:
                self.assertTrue(isinstance(reduced_tensor._local_tensor, funcol.AsyncCollectiveTensor))
            torch.testing.assert_close(origin_tensor, reduced_tensor._local_tensor.cpu())

    @with_comms
    def test_reduce_scatter(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        partial_spec = [Partial()]

        origin_tensor = torch.randn(3 * self.world_size, 3)
        partial_tensor = distribute_tensor(origin_tensor, device_mesh, partial_spec)
        for async_op in [True, False]:
            sharded_tensor = vescale_reduce_scatter(
                partial_tensor, reduce_mesh_dims=0, scatter_dims=0, mesh_dims=0, async_op=async_op
            )
            if async_op:
                self.assertTrue(isinstance(sharded_tensor._local_tensor, funcol.AsyncCollectiveTensor))

            chunk_size = origin_tensor.size()[0] // self.world_size
            rank = device_mesh.get_rank()
            torch.testing.assert_close(
                origin_tensor[rank * chunk_size : (rank + 1) * chunk_size], sharded_tensor._local_tensor.cpu()
            )
        pass


if __name__ == "__main__":
    run_tests()
