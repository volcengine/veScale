################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

from common_dtensor import (
    DTensorTestBase,
    with_comms,
)

import torch
from torch.testing._internal.common_utils import run_tests
from vescale.dtensor._utils import (
    compute_local_shape,
    compute_local_shape_and_global_offset,
    compute_global_tensor_info,
    gather_local_tensor_shape,
)
from vescale.dtensor.api import distribute_tensor
from vescale.dtensor.device_mesh import DeviceMesh
from vescale.dtensor.placement_types import Replicate, Shard, Partial


class UtilTest(DTensorTestBase):
    @property
    def world_size(self):
        return 8

    @with_comms
    def test_compute_local_offset_and_global_offset_1d(self):
        # mesh: 8 * 1
        mesh_tensor = torch.arange(self.world_size)
        device_mesh = DeviceMesh(self.device_type, mesh_tensor)
        my_rank = device_mesh.get_rank()
        size = torch.Size([10])

        placement = [Shard(0)]
        local_size = compute_local_shape(size, device_mesh, placement)
        _, local_offset = compute_local_shape_and_global_offset(size, device_mesh, placement)

        tensor = torch.ones([10])
        tensor_lists = list(torch.chunk(tensor, self.world_size, dim=0))
        # chunk_sizes = [2, 2, 2, 2, 2, 0, 0, 0]
        chunk_sizes = [
            tensor_lists[idx].size(dim=0) if idx < len(tensor_lists) else 0
            for idx, tensor in enumerate(range(self.world_size))
        ]

        self.assertEqual(local_size[0], chunk_sizes[my_rank])
        # Offset for empty shard on the current dimension is equal to
        # global tensor dim size on the current dimension.
        self.assertEqual(local_offset[0], sum(chunk_sizes[:my_rank]))

    @with_comms
    def test_compute_local_shape_2d(self):
        # mesh: 4 * 2
        mesh_tensor = torch.arange(self.world_size).reshape(4, 2)
        mesh = DeviceMesh(self.device_type, mesh_tensor)
        size = torch.Size([8, 6])

        # replicate, replicate
        placements1 = [Replicate(), Replicate()]
        local_size1 = compute_local_shape(size, mesh, placements1)
        self.assertEqual(local_size1, torch.Size([8, 6]))

        # replicate, shard
        placements2 = [Replicate(), Shard(0)]
        local_size2 = compute_local_shape(size, mesh, placements2)
        self.assertEqual(local_size2, torch.Size([4, 6]))

        # shard, shard
        placements3 = [Shard(0), Shard(1)]
        local_size3 = compute_local_shape(size, mesh, placements3)
        self.assertEqual(local_size3, torch.Size([2, 3]))

    @with_comms
    def test_compute_local_shape_2d_uneven(self):
        # mesh: 4 * 2
        mesh_tensor = torch.arange(self.world_size).reshape(4, 2)
        mesh = DeviceMesh(self.device_type, mesh_tensor)
        size = torch.Size([7, 7])
        rank_coordinates = mesh.get_coordinate()

        # replicate, shard
        placements2 = [Replicate(), Shard(0)]
        local_size2 = compute_local_shape(size, mesh, placements2)
        if rank_coordinates[1] < 1:
            self.assertEqual(local_size2, torch.Size([4, 7]))
        else:
            self.assertEqual(local_size2, torch.Size([3, 7]))

        # shard, shard
        placements3 = [Shard(0), Shard(1)]
        local_size3 = compute_local_shape(size, mesh, placements3)
        # first dim
        if rank_coordinates[0] < 3:
            self.assertEqual(local_size3[0], 2)
        else:
            self.assertEqual(local_size3[0], 1)
        # second dim
        if rank_coordinates[1] < 1:
            self.assertEqual(local_size3[1], 4)
        else:
            self.assertEqual(local_size3[1], 3)


class UtilTest2(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @with_comms
    def test_compute_global_tensor_info(self):
        # Even Shard 1D
        global_tensor = torch.randn((4, 8, 3))
        placements = [Shard(1)]
        device_mesh = DeviceMesh(self.device_type, list(range(4)))
        dist_tensor = distribute_tensor(global_tensor, device_mesh, placements)
        local_tensor = dist_tensor._local_tensor

        global_shape, global_stride = compute_global_tensor_info(local_tensor, device_mesh, placements)

        self.assertTrue(global_shape == list(global_tensor.shape))
        self.assertTrue(
            global_stride == list(global_tensor.stride()),
            msg=f"[rank{self.rank}] {global_stride} vs {global_tensor.stride()}",
        )

        # Even Shard 2D
        global_tensor = torch.randn((4, 8, 4))
        placements = [Shard(2), Shard(1)]
        device_mesh = DeviceMesh(self.device_type, [[0, 1], [2, 3]])
        dist_tensor = distribute_tensor(global_tensor, device_mesh, placements)
        local_tensor = dist_tensor._local_tensor

        global_shape, global_stride = compute_global_tensor_info(local_tensor, device_mesh, placements)

        self.assertTrue(global_shape == list(global_tensor.shape))
        self.assertTrue(
            global_stride == list(global_tensor.stride()),
            msg=f"[rank{self.rank}] {global_stride} vs {global_tensor.stride()}",
        )

    @with_comms
    def test_compute_global_tensor_info_uneven_shard(self):
        # Uneven Shard 1D
        global_tensor = torch.randn((4, 5, 3))
        placements = [Shard(1)]
        device_mesh = DeviceMesh(self.device_type, list(range(4)))
        dist_tensor = distribute_tensor(global_tensor, device_mesh, placements)
        local_tensor = dist_tensor._local_tensor

        meshdim_localtensor_shape = gather_local_tensor_shape(local_tensor, device_mesh, placements)

        global_shape, global_stride = compute_global_tensor_info(
            local_tensor, device_mesh, placements, meshdim_localtensor_shape
        )

        self.assertTrue(global_shape == list(global_tensor.shape))
        self.assertTrue(
            global_stride == list(global_tensor.stride()),
            msg=f"[rank{self.rank}] {global_stride} vs {global_tensor.stride()}",
        )

        # Uneven Shard 2D
        global_tensor = torch.randn((4, 7, 3))
        placements = [Shard(2), Shard(1)]
        device_mesh = DeviceMesh(self.device_type, [[0, 1], [2, 3]])
        dist_tensor = distribute_tensor(global_tensor, device_mesh, placements)
        local_tensor = dist_tensor._local_tensor

        meshdim_localtensor_shape = gather_local_tensor_shape(local_tensor, device_mesh, placements)

        global_shape, global_stride = compute_global_tensor_info(
            local_tensor, device_mesh, placements, meshdim_localtensor_shape
        )

        self.assertTrue(global_shape == list(global_tensor.shape))
        self.assertTrue(
            global_stride == list(global_tensor.stride()),
            msg=f"[rank{self.rank}] {global_stride} vs {global_tensor.stride()}",
        )

        # Hybrid Uneven Shard 2D
        for placements in (
            [Replicate(), Shard(1)],
            [Shard(1), Replicate()],
            [Partial(), Shard(1)],
            [Shard(1), Partial()],
        ):
            global_tensor = torch.randn((4, 3, 4))
            dist_tensor = distribute_tensor(global_tensor, device_mesh, placements)
            local_tensor = dist_tensor._local_tensor

            meshdim_localtensor_shape = gather_local_tensor_shape(local_tensor, device_mesh, placements)

            global_shape, global_stride = compute_global_tensor_info(
                local_tensor, device_mesh, placements, meshdim_localtensor_shape
            )

            self.assertTrue(global_shape == list(global_tensor.shape))
            self.assertTrue(
                global_stride == list(global_tensor.stride()),
                msg=f"[rank{self.rank}] {global_stride} vs {global_tensor.stride()}",
            )


if __name__ == "__main__":
    run_tests()
