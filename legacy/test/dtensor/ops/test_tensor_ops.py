################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

from common_dtensor import DTensorConverter, DTensorTestBase, with_comms
from unittest import skip

import torch
from torch.testing._internal.common_utils import run_tests
from unittest import skip

from vescale import DeviceMesh, DTensor, distribute_tensor
from vescale.dtensor.placement_types import Partial, Replicate, Shard, InterleavedShard
from vescale.dtensor import empty as dempty


class DistTensorOpsTest(DTensorTestBase):
    @with_comms
    def test_aten_contiguous(self):
        # this op not covered by dtensor_ops
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        self._test_op(
            mesh,
            lambda x: torch.ops.aten.contiguous(x),
            torch.randn(16, 32),
        )

    @with_comms
    def test_detach(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]

        tensor_to_detach = torch.randn(12, 8, requires_grad=True)
        mat = distribute_tensor(tensor_to_detach, device_mesh, shard_spec)
        detached_mat = mat.detach()
        self.assertFalse(detached_mat is mat)

    @with_comms
    def test_clone(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        specs = [[Replicate()], [Shard(0)]]
        tensor_to_clone = torch.randn(12, 8, requires_grad=True)
        for spec in specs:
            mat = distribute_tensor(tensor_to_clone, device_mesh, spec)
            cloned_mat = mat.clone()
            self.assertFalse(cloned_mat is mat)
            self.assertEqual(cloned_mat.to_local(), mat.to_local())

    @with_comms
    def test_contiguous(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        tensor = torch.rand(3, 5, 6, requires_grad=True)
        sharding = [Shard(0)]
        dist_tensor = DTensor.from_local(tensor, device_mesh, sharding)
        self.assertTrue(dist_tensor.is_contiguous())
        # shard on dim 0 should not change stride (30, 6, 1)
        self.assertEqual(dist_tensor.stride(), tensor.stride())

        new_dt = dist_tensor.transpose(0, 2)
        self.assertFalse(new_dt.is_contiguous())
        self.assertFalse(new_dt.to_local().is_contiguous())
        # check stride
        self.assertEqual(new_dt.stride(), (1, 6, 30))

        new_dt = new_dt.contiguous()
        self.assertTrue(new_dt.is_contiguous())
        self.assertTrue(new_dt.to_local().is_contiguous())
        # check stride
        self.assertEqual(dist_tensor.stride(), tensor.stride())

        # check backward
        new_dt.to_local().sum().backward()
        self.assertEqual(tensor.grad, torch.ones(3, 5, 6))

    @with_comms
    @skip("fail")
    def test_inplace_op(self):
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        input_tensor = torch.randn((12, 3), device=self.device_type)
        dt_to_add = distribute_tensor(input_tensor, mesh, [Shard(0)])
        dt_to_mul = dt_to_add.clone()
        expected_add_dt = dt_to_add.clone() + 3
        add_res = dt_to_add.add_(3)
        expected_mul_dt = dt_to_mul.clone() * 3
        mul_res = dt_to_mul.mul_(3)
        # inplace op should be the same instance before and after
        self.assertTrue(add_res is dt_to_add)
        self.assertEqual(add_res.to_local(), expected_add_dt.to_local())

        self.assertTrue(mul_res is dt_to_mul)
        self.assertEqual(mul_res.to_local(), expected_mul_dt.to_local())

        # test inplace op self and other dtensor with other specs
        # and make sure out spec not change
        shard_spec = [Shard(0)]
        partial_spec = [Partial()]
        dt_to_inplace_add = distribute_tensor(input_tensor, mesh, shard_spec)
        partial_grad = DTensor.from_local(torch.randn(12, 3), mesh, partial_spec)
        res = dt_to_inplace_add.add_(partial_grad)
        self.assertTrue(res is dt_to_inplace_add)
        self.assertTrue(res.placements == shard_spec)

    @with_comms
    @skip("failed")
    def test_op_out_variant(self):
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        input_tensor = torch.randn((12, 3), device=self.device_type)
        sharded_dt_input = distribute_tensor(input_tensor, mesh, [Shard(0)])
        expected_dt = sharded_dt_input.clone() + 3
        sharded_dt_out = sharded_dt_input.clone()
        res = torch.add(sharded_dt_input, 3, out=sharded_dt_out)
        # op out variant should be the same instance before and after
        self.assertTrue(res is sharded_dt_out)
        self.assertEqual(sharded_dt_out.to_local(), expected_dt.to_local())

        # test op out variant with other spec and make sure out spec not change
        replica_spec = [Replicate()]
        replicate_out = distribute_tensor(input_tensor, mesh, replica_spec)
        expected_dt = replicate_out.clone() + 3
        res = torch.add(sharded_dt_input, 3, out=replicate_out)
        self.assertTrue(res is replicate_out)
        self.assertTrue(res.placements == replica_spec)
        self.assertEqual(replicate_out.to_local(), expected_dt.to_local())

    @with_comms
    def test_fill_inplace(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]

        input_tensor = torch.randn(4, 8, requires_grad=True)
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        full_like_dt = torch.fill_(dist_tensor, 42.0)
        full_expected = torch.full((4, 8), 42.0)
        self.assertEqual(full_expected, full_like_dt.to_local())
        self.assertEqual(full_expected, dist_tensor.to_local())

    def _run_xxx_like(self, xxx_like_op, *args, **kwargs):
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
                    expected_tensor = xxx_like_op(torch.empty(*global_shape, device="cuda"), *args, **kwargs)
                    dist_expected = distribute_tensor(expected_tensor.detach().clone(), device_mesh, placements)
                    dtensor = xxx_like_op(
                        dempty(*global_shape, device_mesh=device_mesh, placements=placements), *args, **kwargs
                    )
                    self.assertEqual(dtensor._local_tensor.shape, dist_expected._local_tensor.shape, atol=0.0, rtol=0.0)
                    if any(p.is_partial() for p in placements):
                        self.assertTrue(all(not p.is_partial() for p in dtensor._spec.placements))
                    else:
                        self.assertTrue(list(dtensor._spec.placements) == placements)
                        if xxx_like_op != torch.empty_like:
                            self.assertEqual(dtensor._local_tensor, dist_expected._local_tensor, atol=0.0, rtol=0.0)
                    full_tensor = dtensor.full_tensor()
                    self.assertEqual(full_tensor.shape, expected_tensor.shape, atol=0.0, rtol=0.0)
                    if xxx_like_op != torch.empty_like:
                        self.assertEqual(full_tensor, expected_tensor, atol=0.0, rtol=0.0)

    @with_comms
    def test_xxx_like(self):
        self._run_xxx_like(torch.empty_like)
        self._run_xxx_like(torch.ones_like)
        self._run_xxx_like(torch.zeros_like)
        self._run_xxx_like(torch.full_like, fill_value=42.0)

    @with_comms
    @skip("failed")
    def test_fill_inplace_partial_sum(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Partial()]

        input_tensor = torch.randn(4, 8, requires_grad=True)
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        assert dist_tensor.shape == (4, 8)

        torch.fill_(dist_tensor, 42)
        fill_expected = torch.full(dist_tensor.shape, 42, dtype=input_tensor.dtype)
        self.assertEqual(
            fill_expected,
            dist_tensor.redistribute(device_mesh, [Replicate()]).to_local(),
        )

    @with_comms
    def test_zero_inplace(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]

        input_tensor = torch.randn(4, 8, requires_grad=True)
        dist_tensor = DTensor.from_local(input_tensor, device_mesh, shard_spec)
        zeros_like_dt = torch.zero_(dist_tensor)
        zeros_expected = torch.zeros(4, 8)
        self.assertEqual(zeros_expected, zeros_like_dt.to_local())
        self.assertEqual(zeros_expected, dist_tensor.to_local())

    @with_comms
    def test_equal(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]

        input_tensor_1 = torch.ones(4, 4)
        dist_tensor_1 = DTensor.from_local(input_tensor_1, device_mesh, shard_spec)

        # tensors are equal
        input_tensor_2 = torch.ones(4, 4)
        dist_tensor_2 = DTensor.from_local(input_tensor_2, device_mesh, shard_spec)

        eq_result = dist_tensor_1.equal(dist_tensor_2)
        self.assertTrue(eq_result)

        # tensors are different on some shards
        if self.rank == 0:
            input_tensor_2 = torch.ones(4, 4)
        else:
            input_tensor_2 = torch.randn(4, 4)
        dist_tensor_2 = DTensor.from_local(input_tensor_2, device_mesh, shard_spec)

        eq_result = dist_tensor_1.equal(dist_tensor_2)
        # equal op all reduces each shard's local result
        self.assertFalse(eq_result)

    def _test_op(self, mesh, op_call, *args, **kwargs):
        out = op_call(*args, **kwargs)
        dtc = DTensorConverter(mesh, args, kwargs)
        for d_args, d_kwargs in dtc:
            self.assertTrue(dtc.successful())
            d_out = op_call(*d_args, **d_kwargs)
            self.assertEqual(
                d_out.redistribute(mesh, [Replicate()] * mesh.ndim).to_local(),
                out,
            )

    @with_comms
    def test_select(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        shard_spec_1 = [Shard(1)]
        input_tensor_1 = torch.rand(4, 8)
        dist_tensor_1 = distribute_tensor(input_tensor_1, device_mesh, shard_spec_1)
        dist_result_1 = dist_tensor_1[1]
        self.assertEqual(dist_result_1.redistribute(device_mesh, [Replicate()]).to_local(), input_tensor_1[1])

        shard_spec_2 = [Shard(0)]
        input_tensor_2 = torch.rand(4, 8)
        dist_tensor_2 = distribute_tensor(input_tensor_2, device_mesh, shard_spec_2)
        dist_result_2 = dist_tensor_2[:, 1]
        self.assertEqual(dist_result_2.redistribute(device_mesh, [Replicate()]).to_local(), input_tensor_2[:, 1])

    @with_comms
    @skip("failed")
    def test_index_select(self):
        meshes = [
            DeviceMesh(self.device_type, list(range(self.world_size))),  # 1D mesh
            # TODO(@azzolini): un-comment when DTensorConverter supports N-D mesh
            # DeviceMesh(self.device_type, torch.arange(self.world_size).reshape(2, -1)), # 2D mesh
        ]
        for mesh in meshes:
            self._test_op(
                mesh,
                lambda x, y: x[y],
                torch.randn(16, 32, 16),
                torch.randint(5, (4, 8)),
            )
            self._test_op(
                mesh,
                lambda x, y: x.index_select(1, y),
                torch.randn(16, 32, 16),
                torch.randint(5, (4,)),
            )
            self._test_op(
                mesh,
                lambda x, y: x.index_select(0, y),
                torch.randn(16, 32, 16),
                torch.randint(5, (4,)),
            )
            self._test_op(
                mesh,
                lambda x, y: x[y],
                torch.randn(16, 32, 16),
                torch.randint(5, (12,)),
            )
            self._test_op(
                mesh,
                lambda x, y: x[:, y],
                torch.randn(16, 32, 16),
                torch.randint(5, (4, 8)),
            )
            self._test_op(
                mesh,
                lambda x, y: x[..., y],
                torch.randn(16, 32, 16),
                torch.randint(5, (4, 12)),
            )
            self._test_op(
                mesh,
                lambda x, y: x[..., y],
                torch.randn(16, 32, 16),
                torch.randint(5, (4, 8, 16)),
            )
            self._test_op(
                mesh,
                lambda x, y, z: x[z, y],
                torch.randn(16, 32, 16),
                torch.randint(5, (12, 8, 12)),
                torch.randint(2, (12, 8, 12)),
            )
            self._test_op(
                mesh,
                lambda x, y, z: x[z, :, y],
                torch.randn(16, 32, 16),
                torch.randint(5, (12, 8, 12)),
                torch.randint(2, (12, 8, 12)),
            )
            self._test_op(
                mesh,
                lambda x, y, z: x[:, z, :, y],
                torch.randn(16, 32, 16, 12),
                torch.randint(5, (12, 8, 12)),
                torch.randint(2, (12, 8, 12)),
            )
            # broadcast in inner dimensions
            self._test_op(
                mesh,
                lambda x, y, z: x[:, z, :, y],
                torch.randn(16, 32, 16, 12),
                torch.randint(5, (12, 8, 12)),
                torch.randint(2, (12, 1, 12)),
            )
            # implicit (left-padded) broadcast
            self._test_op(
                mesh,
                lambda x, y, z: x[:, z, :, y],
                torch.randn(16, 32, 16, 12),
                torch.randint(5, (12, 8, 12)),
                torch.randint(2, (8, 12)),
            )
            self._test_op(
                mesh,
                lambda x, y, z: x[z, y, :, :],
                torch.randn(16, 32, 16, 12),
                torch.randint(2, (8, 12)),
                torch.randint(5, (12, 8, 12)),
            )
            self._test_op(
                mesh,
                lambda x, y, z: x[z, :, y, :],
                torch.randn(16, 32, 16, 12),
                torch.randint(2, (8, 12)),
                torch.randint(5, (12, 8, 12)),
            )
            self._test_op(
                mesh,
                lambda x, y, z: x[z, :, :, y],
                torch.randn(16, 32, 16, 12),
                torch.randint(2, (8, 1)),
                torch.randint(5, (12, 8, 12)),
            )

    @with_comms
    def test_index_put_(self):
        device_mesh = DeviceMesh(self.device_type, [0, 1, 2, 3])
        inout_sharding = [Replicate()]
        partial_sharding = [Partial()]
        x = torch.rand(16, 8)
        idx = torch.tensor([0, 1, 2, 3])
        src = torch.rand(4, 8)
        src2 = torch.rand(4, 8)
        x_residual = torch.rand(16, 8)
        dx = distribute_tensor(x, device_mesh, partial_sharding)
        dx_residual = distribute_tensor(x_residual, device_mesh, partial_sharding)
        didx = distribute_tensor(idx, device_mesh, inout_sharding)
        dsrc1 = distribute_tensor(src, device_mesh, partial_sharding)
        dsrc2 = distribute_tensor(src2, device_mesh, partial_sharding)
        dsrc1.requires_grad_(True)
        dsrc2.requires_grad_(True)
        dsrc = dsrc1 + dsrc2
        # out = torch.ops.aten.index_put_(dx, [didx], dsrc)
        dx[didx] = dsrc
        out = dx + dx_residual
        out.redistribute(device_mesh, inout_sharding)
        loss = out.mean()
        loss.backward()

    @with_comms
    def test_scatter(self):
        device_mesh = self.build_device_mesh()
        tensor = torch.zeros(3, 4)
        index = torch.tensor([[1], [2]])
        dtensor = distribute_tensor(tensor, device_mesh, [Replicate()])
        dindex = distribute_tensor(index, device_mesh, [Replicate()])

        local_result = torch.scatter(tensor, 0, index, 1)
        d_result = torch.scatter(dtensor, 0, dindex, 1)
        self.assertEqual(d_result.to_local(), local_result)

    @with_comms
    def test_expand_with_broadcast(self):
        device_mesh = self.build_device_mesh()
        tensor = torch.randn((4,))
        matrix = torch.randn((2, 3, 4))
        dtensor = distribute_tensor(tensor, device_mesh, [Replicate()])
        dmatrix_shard = distribute_tensor(matrix, device_mesh, [Shard(0)])
        dout1 = dtensor.expand_as(dmatrix_shard)
        assert dout1._spec.placements[0] == Shard(0), f"sharding error {dout1._spec}"
        dmatrix_partial = dmatrix_shard.redistribute(device_mesh, [Partial()])
        dout2 = dtensor.expand_as(dmatrix_partial)
        assert dout2._spec.placements[0] == Replicate(), f"sharding error {dout2._spec}"

    @with_comms
    @skip("failed")
    def test_stack(self):
        device_mesh = DeviceMesh(
            self.device_type,
            [0, 1],
        )
        x = torch.rand([4, 2, 4, 8])
        # y = torch.rand([32, 2, 4, 128])
        dx = distribute_tensor(x, device_mesh, [Replicate()]).requires_grad_(True)
        # dy = distribute_tensor(y, device_mesh, [Shard(1)]).requires_grad_(True)
        dx = torch.chunk(dx.transpose(1, 2).float(), 2, dim=-1)
        dx = torch.stack(dx)
        # torch.autograd.backward(dout, torch.ones_like(dout))

    @with_comms
    def test_nonzero(self):
        device_mesh = self.build_device_mesh()
        x = torch.randint(0, 1, (4, 5, 6))
        out = torch.nonzero(x)

        d_x = distribute_tensor(x, device_mesh, [Replicate()])
        d_out = torch.nonzero(d_x)

        self.assertEqual(d_out.to_local(), out)
        self.assertEqual(d_out.size(), d_out._local_tensor.size())

    @with_comms
    def test_unbind(self):
        device_mesh = self.build_device_mesh()
        x = torch.randint(0, 1, (4, 5, 6))
        d_x = distribute_tensor(x, device_mesh, [Replicate()])
        for dim in range(3):
            out = torch.unbind(x, dim)
            d_out = torch.unbind(d_x, dim)
            for d_r, r in zip(d_out, out):
                self.assertEqual(d_r.to_local(), r)

    @with_comms
    def test_split_interleaved_shard_dim(self):
        device_mesh = self.build_device_mesh()
        x = torch.arange(0, 1024)
        d_x = distribute_tensor(x, device_mesh, [InterleavedShard(0, 2)])
        d_out_0, d_out_1 = torch.split(d_x, 512, 0)
        frag_size = 1024 // self.world_size // 2
        local_res_0 = torch.arange(self.rank * frag_size, (self.rank + 1) * frag_size)
        local_res_1 = torch.arange(512 + self.rank * frag_size, 512 + (self.rank + 1) * frag_size)
        self.assertEqual(d_out_0.to_local(), local_res_0)
        self.assertEqual(d_out_1.to_local(), local_res_1)

    @with_comms
    def test_cat_shard(self):
        device_mesh = self.build_device_mesh()
        x_0 = torch.arange(0, 1024).cuda()
        x_1 = torch.arange(1024, 2048).cuda()
        d_x_0 = distribute_tensor(x_0, device_mesh, [Shard(0)])
        d_x_1 = distribute_tensor(x_1, device_mesh, [Shard(0)])
        d_res = torch.cat([d_x_0, d_x_1], 0)
        local_res = torch.cat(
            [
                torch.arange(self.rank * 256, (self.rank + 1) * 256),
                1024 + torch.arange(self.rank * 256, (self.rank + 1) * 256),
            ],
            0,
        ).cuda()
        self.assertEqual(d_res.to_local(), local_res)

    @with_comms
    def test_cat_interleaved_shard(self):
        device_mesh = self.build_device_mesh()
        x_0 = torch.arange(0, 1024).cuda()
        x_1 = torch.arange(1024, 2048).cuda()
        d_x_0 = distribute_tensor(x_0, device_mesh, [InterleavedShard(0, 2)])
        d_x_1 = distribute_tensor(x_1, device_mesh, [InterleavedShard(0, 2)])
        d_res = torch.cat([d_x_0, d_x_1], 0)
        local_res = torch.cat(
            [i * 512 + torch.arange(self.rank * 128, (self.rank + 1) * 128) for i in range(4)], 0
        ).cuda()
        self.assertEqual(d_res.to_local(), local_res)
        self.assertEqual(d_res.placements[0].interleaved_size, 4)


if __name__ == "__main__":
    run_tests()
