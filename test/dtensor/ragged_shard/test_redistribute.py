################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

import numpy as np
import math

import torch
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.testing._internal.common_utils import run_tests

from vescale.dtensor import distribute_tensor, DTensor
from vescale.dtensor.debug import CommDebugMode
from vescale.dtensor.placement_types import Replicate, Shard, RaggedShard, _StridedRaggedShard, TensorMeta, DTensorSpec

from common_dtensor import (
    DTensorTestBase,
    with_comms,
)


funcol = torch.ops.c10d_functional


class RedistributeTest(DTensorTestBase):
    @property
    def world_size(self):
        return 8

    def assert_ragged_1d(self, rank: int, dt: DTensor, local_units: tuple, expected_tensor: torch.Tensor):
        # if local_shard is not empty, assert it's the same with expected_tensor slice.
        local_slice = dt._local_tensor
        local_length = expected_tensor.numel() // sum(local_units) * local_units[rank]
        offset = expected_tensor.numel() // sum(local_units) * sum(local_units[:rank])
        self.assertEqual(local_slice.view(-1), expected_tensor.view(-1)[offset : offset + local_length])

    @with_comms
    def test_ragged_to_replicate_forward_backward(self):
        # 1) test ragged shard -> replicate forward
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        replica_spec = [Replicate()]

        input_sizes = [
            (self.world_size * 3, 3),
            (self.world_size * 3 + 1, 3),
            (self.world_size * 3 + 2, 3),
            (3, self.world_size * 3),
            (3, self.world_size * 3 + 1),
            (3, self.world_size * 3 + 2),
        ]

        # Return k non-negative integers that sum to N.
        def gen_ragged_local_units(N: int, k: int):
            rng = np.random.default_rng(42)
            probs = np.full(k, 1.0 / k)
            return tuple(rng.multinomial(N, probs))

        comm_mode = CommDebugMode()
        for input_size in input_sizes:
            expected_tensor = torch.randn(input_size, device=self.device_type, requires_grad=True)
            assert device_mesh.ndim == 1 and device_mesh.size() == self.world_size
            dims = tuple(range(expected_tensor.ndim))
            local_units = gen_ragged_local_units(expected_tensor.numel(), device_mesh.size())
            shard_spec = [RaggedShard(dims, local_units)]

            dtensor = distribute_tensor(expected_tensor, device_mesh, shard_spec)
            with comm_mode:
                reshard_dtensor = dtensor.redistribute(device_mesh, replica_spec)
            self.assertEqual(reshard_dtensor.size(), torch.Size(input_size))
            self.assertEqual(expected_tensor, reshard_dtensor.to_local())

            # 2) test ragged shard -> replicate backward:
            # should give gradient as shard
            grad_output = torch.ones_like(reshard_dtensor)
            with comm_mode:
                reshard_dtensor.backward(grad_output)
            grad_input = dtensor.grad
            self.assertEqual(grad_input.placements, shard_spec)
            self.assertEqual(grad_input.to_local(), torch.ones(dtensor.to_local().size()))
            self.assertEqual(comm_mode.get_total_counts(), 0)

    @with_comms
    def test_stride_ragged_to_replicate_forward_backward(self):
        return  # TODO(jiacheng) reenable
        # 1) test ragged shard -> replicate forward
        device_mesh = init_device_mesh(self.device_type, (self.world_size // 2, 2), mesh_dim_names=("fsdp", "ep"))
        replica_spec = [Replicate()]

        input_sizes = [
            (self.world_size * 3, 3),
            (self.world_size * 3 + 1, 3),
            (self.world_size * 3 + 2, 3),
            (3, self.world_size * 3),
            (3, self.world_size * 3 + 1),
            (3, self.world_size * 3 + 2),
        ]

        # Return k non-negative integers that sum to N.
        def gen_ragged_local_units(N: int, k: int):
            rng = np.random.default_rng(42)
            probs = np.full(k, 1.0 / k)
            return tuple(rng.multinomial(N, probs))

        comm_mode = CommDebugMode()
        for input_size in input_sizes:
            expected_tensor = torch.randn(input_size, device=self.device_type, requires_grad=True)
            assert device_mesh.ndim == 1 and device_mesh.size() == self.world_size
            dims = tuple(range(expected_tensor.ndim))
            local_units = gen_ragged_local_units(expected_tensor.numel(), device_mesh["fsdp"].size())
            shard_spec = [RaggedShard(dims, local_units)]

            dtensor = distribute_tensor(expected_tensor, device_mesh, shard_spec)
            with comm_mode:
                reshard_dtensor = dtensor.redistribute(device_mesh, replica_spec)
            self.assertEqual(reshard_dtensor.size(), torch.Size(input_size))
            self.assertEqual(expected_tensor, reshard_dtensor.to_local())

            # 2) test ragged shard -> replicate backward:
            # should give gradient as shard
            grad_output = torch.ones_like(reshard_dtensor)
            with comm_mode:
                reshard_dtensor.backward(grad_output)
            grad_input = dtensor.grad
            self.assertEqual(grad_input.placements, shard_spec)
            self.assertEqual(grad_input.to_local(), torch.ones(dtensor.to_local().size()))
            self.assertEqual(comm_mode.get_total_counts(), 0)

    @with_comms
    def test_ragged_to_shard_forward(self):
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        # 1) test ragged shard -> ragged shard forward
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        assert device_mesh.ndim == 1 and device_mesh.size() == self.world_size

        # Return k non-negative integers that sum to N.
        def gen_ragged_local_units(N: int, k: int):
            # explicit test for Muon case.
            if np.random.randint(1, 5) == 1:
                rt = np.random.randint(0, k)
                return tuple([1 if i == rt else 0 for i in range(k)])
            probs = np.full(k, 1.0 / k)
            return tuple(np.random.multinomial(N, probs))

        for _ in range(10):
            nums = [np.random.randint(16, 128) for _ in range(3)]
            # enumerate dims in 0/1/2.
            for input_size in [tuple(nums[:1]), tuple(nums[:2]), tuple(nums[:3])]:
                ndim = len(input_size)
                expected_tensor = torch.randn(input_size, device=self.device_type, requires_grad=True)
                for src_dim, dst_shard_dim in [(a, b) for a in range(ndim) for b in range(ndim)]:
                    src_dims = tuple(range(src_dim + 1))
                    src_local_units = gen_ragged_local_units(
                        math.prod(expected_tensor.shape[: src_dim + 1]), device_mesh.size()
                    )
                    src_shard_spec = [RaggedShard(src_dims, src_local_units)]
                    dst_shard_spec = [Shard(dst_shard_dim)]
                    src_dtensor = distribute_tensor(expected_tensor, device_mesh, src_shard_spec)
                    dst_dtensor = src_dtensor.redistribute(device_mesh, dst_shard_spec)
                    golden_dtensor = distribute_tensor(expected_tensor, device_mesh, dst_shard_spec)
                    self.assertEqual(dst_dtensor.shape, golden_dtensor.shape)
                    self.assertEqual(dst_dtensor._local_tensor, golden_dtensor._local_tensor)

    @with_comms
    def test_ragged_to_ragged_forward(self):
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        # 1) test ragged shard -> ragged shard forward
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        assert device_mesh.ndim == 1 and device_mesh.size() == self.world_size

        # Return k non-negative integers that sum to N.
        def gen_ragged_local_units(N: int, k: int):
            # explicit test for Muon case.
            if np.random.randint(1, 5) == 1:
                rt = np.random.randint(0, k)
                return tuple([1 if i == rt else 0 for i in range(k)])
            probs = np.full(k, 1.0 / k)
            return tuple(np.random.multinomial(N, probs))

        for _ in range(10):
            nums = [np.random.randint(16, 128) for _ in range(3)]
            # enumerate dims in 0/1/2.
            for input_size in [tuple(nums[:1]), tuple(nums[:2]), tuple(nums[:3])]:
                ndim = len(input_size)
                expected_tensor = torch.randn(input_size, device=self.device_type, requires_grad=True)
                for src_dim, dst_dim in [(a, b) for a in range(ndim) for b in range(ndim)]:
                    src_dims = tuple(range(src_dim + 1))
                    dst_dims = tuple(range(dst_dim + 1))
                    src_local_units = gen_ragged_local_units(
                        math.prod(expected_tensor.shape[: src_dim + 1]), device_mesh.size()
                    )
                    dst_local_units = gen_ragged_local_units(
                        math.prod(expected_tensor.shape[: dst_dim + 1]), device_mesh.size()
                    )
                    src_shard_spec = [RaggedShard(src_dims, src_local_units)]
                    dst_shard_spec = [RaggedShard(dst_dims, dst_local_units)]

                    rank = device_mesh.get_coordinate()[0]
                    src_dtensor = distribute_tensor(expected_tensor, device_mesh, src_shard_spec)
                    self.assert_ragged_1d(rank, src_dtensor, src_local_units, expected_tensor)
                    dst_dtensor = src_dtensor.redistribute(device_mesh, dst_shard_spec)
                    self.assert_ragged_1d(rank, dst_dtensor, dst_local_units, expected_tensor)

    @with_comms
    def test_ragged_shard_2d(self):
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        # Return k non-negative integers that sum to N.
        def gen_ragged_local_units(N: int, k: int):
            # explicit test for Muon case.
            if np.random.randint(1, 5) == 1:
                rt = np.random.randint(0, k)
                return tuple([1 if i == rt else 0 for i in range(k)])
            probs = np.full(k, 1.0 / k)
            return tuple(np.random.multinomial(N, probs))

        def test_ragged_shard0_2d(
            global_tensor: torch.Tensor, device_mesh: DeviceMesh, ragged_dim: int, shard_dim: int
        ):
            # Shard() dtensor.
            dt_shard = distribute_tensor(global_tensor, device_mesh["other"], [Shard(shard_dim)])
            shard_tensor = dt_shard._local_tensor
            assert shard_tensor.ndim == global_tensor.ndim

            # RaggedShard based on Shard() dtensor's local_tensor.
            ragged_dims = tuple(range(ragged_dim + 1))
            ragged_lst = gen_ragged_local_units(
                math.prod(shard_tensor.shape[: ragged_dim + 1]), device_mesh["ragged"].size()
            )
            assert sum(ragged_lst) != 0, f"ragged_lst: {ragged_lst}"
            dt_2d = distribute_tensor(shard_tensor, device_mesh["ragged"], [RaggedShard(ragged_dims, ragged_lst)])

            if shard_dim == 0:
                _sharding_spec = DTensorSpec(
                    device_mesh,
                    [
                        _StridedRaggedShard(
                            dims=ragged_dims, local_units=ragged_lst, split_factor=device_mesh["other"].size()
                        ),
                        Shard(shard_dim),
                    ],
                    tensor_meta=TensorMeta(dt_shard.size(), dt_shard.stride(), dt_shard.dtype),
                )
            else:
                _sharding_spec = DTensorSpec(
                    device_mesh,
                    [RaggedShard(dims=ragged_dims, local_units=ragged_lst), Shard(shard_dim)],
                    tensor_meta=TensorMeta(dt_shard.size(), dt_shard.stride(), dt_shard.dtype),
                )
            dt_2d = DTensor(
                dt_2d._local_tensor,
                _sharding_spec,
                requires_grad=True,
            )
            rank = device_mesh["ragged"].get_coordinate()[0]
            self.assert_ragged_1d(rank, dt_2d, ragged_lst, shard_tensor)

            full_tensor = dt_2d.full_tensor()
            self.assertEqual(full_tensor.shape, global_tensor.shape)
            self.assertEqual(full_tensor, global_tensor)

        for mesh_size in [(1, 8), (2, 4), (4, 2), (8, 1)]:
            device_mesh = init_device_mesh(self.device_type, mesh_size, mesh_dim_names=["ragged", "other"])
            assert device_mesh.ndim == 2 and device_mesh.size() == self.world_size

            for _ in range(10):
                nums = [np.random.randint(16, 128) * mesh_size[1] for _ in range(3)]
                for input_size in [tuple(nums[:1]), tuple(nums[:2]), tuple(nums[:3])]:
                    ndim = len(input_size)
                    global_tensor = torch.randn(input_size, device=self.device_type, requires_grad=True)

                    # [_StridedRaggedShard, Shard(0)]
                    for src_dim in range(ndim):
                        test_ragged_shard0_2d(global_tensor, device_mesh, src_dim, 0)

                    # [RaggedShard, Shard(1)]
                    if ndim > 1:
                        test_ragged_shard0_2d(global_tensor, device_mesh, 0, 1)


if __name__ == "__main__":
    run_tests()
