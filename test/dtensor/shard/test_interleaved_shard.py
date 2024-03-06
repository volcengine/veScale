################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

"""Testing InterleavedShard"""

from copy import deepcopy
from common_dtensor import DTensorTestBase, with_comms

import torch
import torch.distributed._functional_collectives as funcol
from torch.testing._internal.common_utils import run_tests

from vescale.dtensor.api import distribute_tensor, from_local
from vescale.dtensor.device_mesh import DeviceMesh
from vescale.dtensor.placement_types import InterleavedShard, Partial, Replicate, Shard


class InterleavedShardBasicTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        else:
            return 8

    @with_comms
    def test_construct_interleaved_sharded_dtensor(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        interleaved_size = 4
        shard_spec = [InterleavedShard(dim=0, interleaved_size=interleaved_size)]
        for require_grad in [True, False]:
            tensor_to_distribute = torch.randn(
                2 * interleaved_size * self.world_size, 3, requires_grad=require_grad, device=self.device_type
            )
            dist_tensor = distribute_tensor(tensor_to_distribute, device_mesh, shard_spec)

            self.assertEqual(dist_tensor.size(), torch.Size([2 * interleaved_size * self.world_size, 3]))
            local_tensor = dist_tensor.to_local()
            self.assertEqual(local_tensor.size(), torch.Size([2 * interleaved_size, 3]))
            if require_grad:
                self.assertTrue(dist_tensor.requires_grad)
                self.assertTrue(local_tensor.requires_grad)

            # content comparasion
            directly_split_tensor_list = list(torch.chunk(tensor_to_distribute, self.world_size, dim=0))
            self.assertNotEqual(local_tensor, directly_split_tensor_list[self.rank])
            reshaped_tensor = tensor_to_distribute.reshape(interleaved_size, -1, 3)
            split_tensor_list = list(torch.chunk(reshaped_tensor, self.world_size, dim=1))
            split_tensor_list = [t.reshape(-1, 3) for t in split_tensor_list]
            self.assertEqual(local_tensor, split_tensor_list[self.rank])

    @with_comms
    def test_from_local_interleaved_shard(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        interleaved_size = 4
        shard_spec = [InterleavedShard(dim=0, interleaved_size=interleaved_size)]

        for requires_grad in [True, False]:
            # invalid
            with self.assertRaises(ValueError):
                t = torch.rand(3, 3, device=self.device_type, requires_grad=requires_grad)
                dt = from_local(t, device_mesh, shard_spec, run_check=True)

            # valid
            t = torch.rand(interleaved_size * 3, 3, device=self.device_type, requires_grad=requires_grad)
            dt = from_local(t, device_mesh, shard_spec, run_check=True)
            t = t.reshape(interleaved_size, -1, 3)
            global_tensor = funcol.all_gather_tensor(t, gather_dim=1, group=device_mesh._dim_group_infos[0][1])
            global_tensor = global_tensor.reshape(-1, 3)
            dt = dt.redistribute(device_mesh, [Replicate()])
            self.assertEqual(global_tensor, dt._local_tensor)

    @with_comms
    def test_comm_from_interleaved_shard(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        interleaved_size = 4
        shard_spec = [InterleavedShard(dim=0, interleaved_size=interleaved_size)]

        t = torch.arange(0, interleaved_size * self.world_size * 2 * 3, device=self.device_type).view(
            interleaved_size * self.world_size * 2, 3
        )
        dt = distribute_tensor(t, device_mesh, shard_spec)

        # IS -> R
        dt1 = dt.redistribute(device_mesh, [Replicate()])
        self.assertEqual(t, dt1._local_tensor)

        # IS -> P
        dt2 = dt.redistribute(device_mesh, [Partial()])
        if self.rank == 0:
            self.assertEqual(t, dt2._local_tensor)
        else:
            self.assertEqual(torch.zeros_like(t), dt2._local_tensor)

        # IS -> IS
        with self.assertRaises(NotImplementedError):
            dt.redistribute(device_mesh, [InterleavedShard(dim=0, interleaved_size=2)])

        # IS -> S
        with self.assertRaises(NotImplementedError):
            dt.redistribute(device_mesh, [Shard(dim=0)])

    @with_comms
    def test_comm_to_interleaved_shard(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        interleaved_size = 4

        t = torch.arange(0, interleaved_size * self.world_size * 2 * 3, device=self.device_type).view(
            interleaved_size * self.world_size * 2, 3
        )
        reshape_t = t.clone().reshape(interleaved_size, self.world_size * 2, 3)
        split_tensor_list = list(torch.chunk(reshape_t, chunks=self.world_size, dim=1))

        # R -> IS
        dt = distribute_tensor(t, device_mesh, [Replicate()])
        dt1 = dt.redistribute(device_mesh, [InterleavedShard(dim=0, interleaved_size=interleaved_size)])
        self.assertEqual(split_tensor_list[self.rank].reshape(-1, 3), dt1._local_tensor)

        # P -> IS
        dt = distribute_tensor(t, device_mesh, [Partial()])
        dt2 = dt.redistribute(device_mesh, [InterleavedShard(dim=0, interleaved_size=interleaved_size)])
        self.assertEqual(split_tensor_list[self.rank].reshape(-1, 3), dt2._local_tensor)

        # S -> IS
        dt = distribute_tensor(t, device_mesh, [Shard(0)])
        with self.assertRaises(NotImplementedError):
            dt.redistribute(device_mesh, [InterleavedShard(dim=0, interleaved_size=interleaved_size)])

        # IS -> IS
        dt = distribute_tensor(t, device_mesh, [InterleavedShard(dim=0, interleaved_size=2)])
        with self.assertRaises(NotImplementedError):
            dt.redistribute(device_mesh, [InterleavedShard(dim=0, interleaved_size=interleaved_size)])


class InterleavedShardViewLikeOperatorTest(DTensorTestBase):
    @property
    def world_size(self):
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        return 8

    @with_comms
    def test_view_op(self):
        device_mesh = self.build_device_mesh()
        bsz = interleaved_size = 4
        trailing_dim_size = 5 * 6
        t = torch.rand((bsz, 6 * self.world_size, trailing_dim_size), device=self.device_type)
        dt = distribute_tensor(t, device_mesh, [Shard(1)])

        reshaped_dt = dt.reshape((-1, trailing_dim_size))
        self.assertTrue(reshaped_dt.placements[0].is_interleaved_shard())

        # case 1: IS -> S
        back_dt = reshaped_dt.reshape((bsz, -1, trailing_dim_size))
        self.assertTrue(back_dt.placements[0].is_shard())
        self.assertEqual(back_dt.placements[0].dim, 1)

        # case 2: IS -> S
        with self.assertRaisesRegex(RuntimeError, "Vescale not support auto resharding DTensor."):
            back_dt = reshaped_dt.reshape((8, -1, trailing_dim_size))
            back_dt = reshaped_dt.reshape((3, -1, trailing_dim_size))

        # case 3: IS -> IS
        back_dt = reshaped_dt.reshape((2, -1, trailing_dim_size))
        self.assertTrue(back_dt.placements[0].is_interleaved_shard())
        self.assertEqual(back_dt.placements[0].interleaved_size, 2)

        # case 3:
        back_dt = reshaped_dt.reshape((-1, 5, 6))
        self.assertTrue(back_dt.placements[0].is_interleaved_shard())

    @with_comms
    def test_uncontiguous_view_op(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        interleaved_size = 2
        bsz = 4
        trailing_dim_size = 5 * 6
        t = torch.rand((interleaved_size, bsz, 3 * self.world_size, trailing_dim_size), device=self.device_type)
        dt = distribute_tensor(t, device_mesh, [Shard(2)])
        # make the dt not contiguous
        dt = torch.permute(dt, (1, 0, 2, 3))

        reshaped_dt = dt.reshape((bsz, -1, trailing_dim_size))
        self.assertTrue(reshaped_dt.placements[0].is_interleaved_shard())

        # content comparasion
        replicated_dt = reshaped_dt.redistribute(device_mesh, [Replicate()])
        new_t = torch.permute(
            replicated_dt._local_tensor.view((bsz, interleaved_size, -1, trailing_dim_size)), (1, 0, 2, 3)
        )
        self.assertEqual(t, new_t)


class InterleavedShardMatmulTest(DTensorTestBase):
    @property
    def world_size(self):
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        return 8

    @with_comms
    def test_mm_op(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        bsz = 4
        shard_spec = [Shard(1)]

        lhs = torch.rand((bsz, 3 * self.world_size, 5), device=self.device_type)
        rhs = torch.rand((5, 7), device=self.device_type)
        d_lhs = distribute_tensor(lhs, device_mesh, shard_spec)
        d_rhs = distribute_tensor(rhs, device_mesh, [Replicate()])

        d_lhs = d_lhs.reshape((-1, 5))
        d_out = torch.mm(d_lhs, d_rhs)
        d_out = d_out.reshape((bsz, -1, 7))

        self.assertTrue(d_out.placements[0].is_shard())
        self.assertEqual(d_out.placements[0].dim, 1)
        self.assertFalse(d_out.placements[0].is_interleaved_shard())

    @with_comms
    def test_addmm_op(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        bsz = 4
        shard_spec = [Shard(1)]

        lhs = torch.rand((bsz, 3 * self.world_size, 5), device=self.device_type)
        rhs = torch.rand((5, 7), device=self.device_type)
        bias = torch.rand((7,), device=self.device_type)
        d_lhs = distribute_tensor(lhs, device_mesh, shard_spec)
        d_rhs = distribute_tensor(rhs, device_mesh, [Replicate()])
        d_bias = distribute_tensor(bias, device_mesh, [Replicate()])

        d_lhs = d_lhs.reshape((-1, 5))
        d_out = torch.addmm(d_bias, d_lhs, d_rhs)
        d_out = d_out.reshape((bsz, -1, 7))

        self.assertTrue(d_out.placements[0].is_shard())
        self.assertEqual(d_out.placements[0].dim, 1)
        self.assertFalse(d_out.placements[0].is_interleaved_shard())

    @with_comms
    def test_mm_backward(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        bsz = 4
        k = 5
        n = 7
        shard_spec = [Shard(1)]
        test_lhs = torch.rand((bsz, 3 * self.world_size, k), device=self.device_type, requires_grad=True)
        test_rhs = torch.rand((k, n), device=self.device_type, requires_grad=True)
        base_lhs = deepcopy(test_lhs)
        base_rhs = deepcopy(test_rhs)

        # DTensor operation
        d_lhs = distribute_tensor(test_lhs, device_mesh, shard_spec)
        d_rhs = distribute_tensor(test_rhs, device_mesh, [Replicate()])
        d_reshaped_lhs = d_lhs.reshape((-1, k))
        d_out = torch.mm(d_reshaped_lhs, d_rhs)
        d_out = d_out.reshape((bsz, -1, n)).redistribute(device_mesh, [Replicate()])
        d_out.sum().to_local().backward()

        # Tensor operation
        base_out = torch.mm(base_lhs.reshape((-1, k)), base_rhs)
        base_out.reshape((bsz, -1, n)).sum().backward()

        # compare grad of lhs, note that dtensor's grad is Sharded
        torch.testing.assert_close(d_lhs.grad.redistribute(device_mesh, [Replicate()])._local_tensor, base_lhs.grad)
        # compare grad of rhs, note that dtensor's grad is Partial
        torch.testing.assert_close(d_rhs.grad.redistribute(device_mesh, [Replicate()])._local_tensor, base_rhs.grad)


if __name__ == "__main__":
    run_tests()
