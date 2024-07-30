################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

import unittest
from common_dtensor import (
    DTensorTestBase,
    skip_if_lt_x_gpu,
    skip_unless_torch_gpu,
    with_comms,
)
from torch.testing._internal.common_utils import run_tests

import torch
import torch.distributed._functional_collectives as funcol
from torch.distributed.distributed_c10d import broadcast_object_list

from vescale import DeviceMesh, DTensor, Shard, Partial, Replicate, distribute_tensor
import vescale.dtensor.random as random
from vescale.dtensor.random import is_rng_supported_mesh, manual_seed
from vescale.dtensor import empty as dempty


class DTensorRandomInitTest(DTensorTestBase):
    def _run_init_op(self, init_op, *args, **kwargs):
        all_mesh_shapes = [
            torch.arange(self.world_size),
            torch.arange(self.world_size).reshape(self.world_size // 2, 2),
        ]
        for mesh_shape in all_mesh_shapes:
            mesh_dim = mesh_shape.dim()
            device_mesh = DeviceMesh(self.device_type, mesh_shape)
            all_shapes = [(8, 4), (4, 4, 4), (8, 8, 4, 4), (5, 6, 7, 8, 9)]
            for global_shape in all_shapes:
                all_placements = [Replicate(), Partial()] + [Shard(d) for d in range(len(global_shape))]
                from itertools import product

                all_placements = [list(placements) for placements in product(all_placements, repeat=mesh_dim)]

                for placements in all_placements:
                    sharded_dims = [placement.dim for placement in placements if placement.is_shard()]
                    if len(sharded_dims) > len(set(sharded_dims)):
                        # Skip the placements that shard along the same dim more than once
                        continue
                    # NOTE: currently random initialization on cuda device has different
                    # behavior from other devices. Unify the test once the behavior is unified.
                    if not is_rng_supported_mesh(device_mesh):
                        input_tensor = torch.randn(*global_shape, device=self.device_type)
                        dtensor = DTensor.from_local(input_tensor, device_mesh, [Shard(0)])
                        local_tensor_clone = torch.clone(input_tensor)
                        torch.manual_seed(self.rank)
                        local_tensor_clone = init_op(local_tensor_clone, *args, **kwargs)
                        torch.manual_seed(self.rank)
                        dtensor = init_op(dtensor, *args, **kwargs)
                        self.assertEqual(local_tensor_clone, dtensor.to_local())
                    else:
                        torch.cuda.manual_seed_all(0)
                        expected_tensor = init_op(torch.empty(*global_shape, device="cuda"), *args, **kwargs)
                        dist_expected = distribute_tensor(expected_tensor.detach().clone(), device_mesh, placements)
                        manual_seed(0, device_mesh)
                        dtensor = init_op(
                            dempty(*global_shape, device_mesh=device_mesh, placements=placements), *args, **kwargs
                        )
                        if any(p.is_partial() for p in placements):
                            self.assertTrue(all(not p.is_partial() for p in dtensor._spec.placements))
                        else:
                            self.assertTrue(list(dtensor._spec.placements) == placements)
                            self.assertEqual(dtensor._local_tensor, dist_expected._local_tensor, atol=0.0, rtol=0.0)
                        full_tensor = dtensor.full_tensor()
                        self.assertEqual(full_tensor, expected_tensor, atol=0.0, rtol=0.0)

    @with_comms
    @skip_unless_torch_gpu
    def test_init_ops(self):
        self._run_init_op(torch.nn.init.kaiming_uniform_, a=0, mode="fan_in", nonlinearity="leaky_relu")
        self._run_init_op(torch.nn.init.normal_, mean=1.5, std=0.8)
        self._run_init_op(torch.nn.init.uniform_, a=0, b=1.2)

        for dtype in (torch.float32, torch.float16):
            self._run_init_op(torch.rand_like, dtype=dtype)
            self._run_init_op(torch.randn_like, dtype=dtype)
            self._run_init_op(torch.randint_like, low=0, high=100, dtype=dtype)


class DTensorRandomOpTest(DTensorTestBase):
    @with_comms
    @skip_unless_torch_gpu
    def test_rng_tracker_init(self):
        torch.cuda.manual_seed(self.rank)
        object_list = [torch.cuda.initial_seed()]
        broadcast_object_list(object_list)
        seed_from_rank_0 = int(object_list[0])

        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        # seed synchronization happens after the first `distribute_tensor` call
        dtensor = distribute_tensor(torch.empty([self.world_size], device="cuda"), device_mesh, [Shard(0)])
        self.assertEqual(seed_from_rank_0, random._rng_tracker.get_seed("parallel-rng"))

    @with_comms
    @skip_unless_torch_gpu
    def test_manual_seed(self):
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        manual_seed(1234, device_mesh)
        self.assertEqual(1234, random._rng_tracker.get_seed("parallel-rng"))
        with self.assertRaisesRegex(RuntimeError, "different seed values"):
            manual_seed(self.rank, device_mesh)

    def run_dropout(self, global_shape, mesh, placements, inplace):
        torch.cuda.manual_seed_all(0)
        dropout = torch.nn.Dropout(p=0.2, inplace=inplace)
        expected_tensor = dropout(torch.ones(global_shape, device=self.device_type))
        dist_expected = distribute_tensor(expected_tensor.detach().clone(), mesh, placements)

        manual_seed(0, mesh)
        dtensor = distribute_tensor(torch.ones(global_shape, device=self.device_type), mesh, placements)
        dtensor = dropout(dtensor)

        if all(not p.is_partial() for p in placements):
            self.assertEqual(dtensor._local_tensor, dist_expected._local_tensor, atol=0.0, rtol=0.0)
        full_tensor = dtensor.full_tensor()
        self.assertEqual(full_tensor, expected_tensor, atol=0.0, rtol=0.0)

    @with_comms
    @skip_unless_torch_gpu
    def test_deterministic_dropout_1d(self):
        # test suite sets each rank's seed to the same value
        shapes = [(9, 7), (4, 16, 16), (7, 5, 16)]
        mesh = DeviceMesh("cuda", torch.arange(self.world_size))
        for global_shape in shapes:
            for placements in [[Replicate()], [Partial()], [Shard(0)], [Shard(1)]]:
                self.run_dropout(global_shape, mesh, placements, inplace=True)
                self.run_dropout(global_shape, mesh, placements, inplace=False)
        mesh = DeviceMesh("cuda", torch.arange(self.world_size).reshape(self.world_size // 2, 2))
        for global_shape in shapes:
            for placements in [
                [Shard(0), Shard(1)],
                [Shard(0), Replicate()],
                [Shard(0), Partial()],
                [Shard(1), Shard(0)],
                [Shard(1), Replicate()],
                [Shard(1), Partial()],
                [Replicate(), Shard(0)],
                [Replicate(), Shard(1)],
                [Replicate(), Partial()],
                [Replicate(), Replicate()],
                [Partial(), Shard(0)],
                [Partial(), Shard(1)],
                [Partial(), Partial()],
                [Partial(), Replicate()],
            ]:
                self.run_dropout(global_shape, mesh, placements, inplace=True)
                self.run_dropout(global_shape, mesh, placements, inplace=False)

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_deterministic_uniform_2d(self):
        mesh = torch.arange(self.world_size).reshape(2, 2)
        device_mesh = DeviceMesh(self.device_type, mesh)
        dtensor = distribute_tensor(
            torch.empty(*[self.world_size for _ in mesh.size()], device=self.device_type),
            device_mesh,
            [Replicate(), Replicate()],
        )

        placements_list = [  # this list of placements should be enough to cover
            [Shard(0), Shard(1)],
            [Shard(0), Replicate()],
            [Shard(0), Partial()],
            [Shard(1), Shard(0)],
            [Shard(1), Replicate()],
            [Shard(1), Partial()],
            [Replicate(), Shard(0)],
            [Replicate(), Shard(1)],
            [Replicate(), Partial()],
            [Replicate(), Replicate()],
            [Partial(), Shard(0)],
            [Partial(), Shard(1)],
            [Partial(), Partial()],
            [Partial(), Replicate()],
        ]

        for placements in placements_list:
            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            golden = torch.empty(*[self.world_size for _ in mesh.size()], device=self.device_type)
            golden.uniform_(0, 1)
            dist_golden = distribute_tensor(golden.detach().clone(), device_mesh, placements)

            manual_seed(0, device_mesh)
            dtensor = distribute_tensor(
                torch.empty(*[self.world_size for _ in mesh.size()], device=self.device_type),
                device_mesh,
                placements,
            )
            dtensor.uniform_(0, 1)

            if any(p.is_partial() for p in placements):
                self.assertTrue(all(not p.is_partial() for p in dtensor._spec.placements))
            else:
                self.assertTrue(list(dtensor._spec.placements) == placements)
                self.assertEqual(dtensor._local_tensor, dist_golden._local_tensor, atol=0.0, rtol=0.0)
            full_tensor = dtensor.full_tensor()
            self.assertEqual(full_tensor, golden, atol=0.0, rtol=0.0)

    @with_comms
    @skip_if_lt_x_gpu(4)
    @unittest.skip("Meta tensor broadcast is not implemented")
    def test_meta_tensor_init(self):
        # TODO: Fix this
        # test suite sets each rank's seed to the same value but in actual
        # execution the default random seed will be different (a random value).
        torch.cuda.manual_seed(self.rank)
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        size = [1024, 2048]
        meta_dtensor = distribute_tensor(torch.empty(*size, device="meta"), device_mesh, [Replicate()])
        self.assertTrue(meta_dtensor.is_meta)
        dtensor = torch.empty_like(meta_dtensor, device=self.device_type)

        # disable the distribute region for RNG
        random._rng_tracker.distribute_region_enabled = False
        dtensor.uniform_()

        # allgather the local tensors
        local_tensor = funcol.all_gather_tensor(
            dtensor.to_local(), gather_dim=0, group=device_mesh._dim_group_infos[0][1]
        )

        # compare with local tensors from other ranks
        self_slice = slice(1024 * self.rank, 1024 * self.rank + 1024)
        for other_rank in range(self.world_size):
            # the RNG result on each rank differs even they're supposed
            # to be replicated
            if self.rank != other_rank:
                other_slice = slice(1024 * other_rank, 1024 * other_rank + 1024)
                self.assertNotEqual(local_tensor[self_slice, :], local_tensor[other_slice, :])

        # enable the distribute region for RNG
        random._rng_tracker.distribute_region_enabled = True
        self.assertTrue(meta_dtensor.is_meta)
        dtensor = torch.empty_like(meta_dtensor, device=self.device_type)
        dtensor.uniform_()

        # allgather the local tensors
        local_tensor = funcol.all_gather_tensor(
            dtensor.to_local(), gather_dim=0, group=device_mesh._dim_group_infos[0][1]
        )

        # compare with local tensors from other ranks
        for other_rank in range(self.world_size):
            # the RNG result on each rank are the same because they're replicated
            if self.rank != other_rank:
                # other rank should have an identical local tensor
                other_slice = slice(1024 * other_rank, 1024 * other_rank + 1024)
                self.assertEqual(local_tensor[self_slice, :], local_tensor[other_slice, :])


if __name__ == "__main__":
    run_tests()
