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
from common_dtensor import DTensorTestBase, with_comms

import torch
import torch.distributed as dist
import torch.nn.functional as F
from numpy.testing import assert_array_equal
from torch.distributed._functional_collectives import AsyncCollectiveTensor
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed.fake_pg import FakeStore

from vescale import DeviceMesh, DTensor, distribute_tensor
from vescale.dtensor.placement_types import Partial, Replicate, Shard


class DummyMLP(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.net1 = torch.nn.Linear(5, 1024, device=device)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(1024, 4, device=device)

    def forward(self, x):
        return self.net2(F.relu(self.net1(x)))

    def reset_parameters(self, *args, **kwargs):
        with torch.no_grad():
            self.net1.weight.fill_(0.5)
            self.net2.weight.fill_(1)
            self.net1.bias.fill_(1.5)
            self.net2.bias.fill_(1.2)


class DTensorTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    def test_dtensor_constructor(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]
        local_tensor = torch.randn(3, 3, requires_grad=True)
        dist_tensor_shape = torch.Size([self.world_size * 3, 3])
        dist_tensor = DTensor(
            local_tensor,
            device_mesh,
            shard_spec,
            shape=dist_tensor_shape,
            dtype=local_tensor.dtype,
            requires_grad=True,
            stride=local_tensor.stride(),
        )
        self.assertEqual(dist_tensor.size(), torch.Size((self.world_size * 3, 3)))

        with self.assertWarnsRegex(UserWarning, "To construct"):
            DTensor(
                local_tensor,
                device_mesh,
                shard_spec,
                shape=dist_tensor_shape,
                dtype=local_tensor.dtype,
                requires_grad=False,
                stride=local_tensor.stride(),
            )

        local_tensor = torch.randn(3, 3, requires_grad=False)
        with self.assertWarnsRegex(UserWarning, "To construct"):
            dist_tensor = DTensor(
                local_tensor,
                device_mesh,
                shard_spec,
                shape=dist_tensor_shape,
                dtype=local_tensor.dtype,
                requires_grad=True,
                stride=local_tensor.stride(),
            )

    @with_comms
    def test_dtensor_stride(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard0_spec = [Shard(0)]
        local_tensor = torch.randn(4, 8)
        global_shape = torch.Size([self.world_size * 4, 8])
        dist_tensor = DTensor.from_local(local_tensor, device_mesh, shard0_spec)
        # won't affect stride
        self.assertEqual(dist_tensor.stride(), (8, 1))

        shard1_spec = [Shard(1)]
        local_tensor = torch.randn(8, 4)
        global_shape = torch.Size([8, self.world_size * 4])
        dist_tensor = DTensor.from_local(local_tensor, device_mesh, shard1_spec)
        # will affect stride after DT initialized
        self.assertEqual(dist_tensor.stride(), (4 * self.world_size, 1))

        # if initialized from a transposed mat (Even Sharding)
        local_tensor = torch.randn(8, 4, 8)
        local_tensor_t = local_tensor.permute(1, 2, 0)
        global_shape = torch.Size([4, self.world_size * 8, 8])
        self.assertEqual(local_tensor_t.stride(), (8, 1, 32))
        dist_tensor = DTensor.from_local(local_tensor_t, device_mesh, shard1_spec, run_check=False)
        global_stride = (8 * self.world_size, 1, 32 * self.world_size)
        self.assertEqual(dist_tensor.stride(), global_stride)

    @with_comms
    def test_from_local(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]
        local_tensor = torch.randn(3, 3)
        sharded_tensor = DTensor.from_local(local_tensor, device_mesh, shard_spec)
        self.assertEqual(sharded_tensor.size(), torch.Size([self.world_size * 3, 3]))

        replica_spec = [Replicate()]
        ddp_tensor = DTensor.from_local(local_tensor, device_mesh, replica_spec)
        self.assertEqual(ddp_tensor.size(), local_tensor.size())

        partial_spec = [Partial()]
        partial_tensor = DTensor.from_local(local_tensor, device_mesh, partial_spec)
        self.assertEqual(partial_tensor.size(), local_tensor.size())

        # test dist tensor works with torch.Tensor during backwards
        local_tensor_with_grad = torch.randn(3, 3, requires_grad=True)
        # do some operations on local tensor
        local_tensor_temp = local_tensor_with_grad * 3
        # create the dist tensor with non leaf local tensor, dist tensor created
        # should also be non leaf node
        dist_tensor = DTensor.from_local(local_tensor_temp, device_mesh, shard_spec)
        self.assertFalse(dist_tensor.is_leaf)
        # do some random operations on dist tensor
        output = dist_tensor * 3
        self.assertIsInstance(output, DTensor)
        # trigger .backward() on dist tensor directly
        local_grad = torch.ones(3, 3)
        grad_output = DTensor.from_local(local_grad, device_mesh, shard_spec)
        # run backward directly on dist tensor
        output.backward(grad_output)
        # check it gradients flow back to original torch.Tensor
        self.assertIsNotNone(local_tensor_with_grad.grad)
        expected_grad = torch.ones(3, 3) * 9
        self.assertEqual(local_tensor_with_grad.grad, expected_grad)

    @with_comms
    def test_from_local_with_given_shape_stride(self):
        torch.manual_seed(0)

        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        # Assert given shape and stride
        local_tensor = torch.randn(3 * self.world_size, 3)
        global_tensor = local_tensor
        shard_spec = (Replicate(),)
        with self.assertRaisesRegex(ValueError, "Please pass both shape and stride at the same time."):
            DTensor.from_local(local_tensor, device_mesh, shard_spec, shape=global_tensor.size())
        with self.assertRaisesRegex(ValueError, "Please pass both shape and stride at the same time."):
            DTensor.from_local(local_tensor, device_mesh, shard_spec, stride=global_tensor.stride())

        # Replicate
        local_tensor = torch.randn(3 * self.world_size, 3)
        global_tensor = local_tensor
        dist_tensor = DTensor.from_local(
            local_tensor, device_mesh, [Replicate()], shape=global_tensor.shape, stride=global_tensor.stride()
        )
        self.assertEqual(dist_tensor.size(), global_tensor.shape)
        self.assertEqual(dist_tensor.stride(), global_tensor.stride())
        dist_tensor = dist_tensor.redistribute(placements=[Replicate()])
        self.assertEqual(dist_tensor._local_tensor, global_tensor, atol=0, rtol=0)

        # Partial
        local_tensor = torch.randn(3 * self.world_size, 3)
        global_tensor = local_tensor
        dist_tensor = DTensor.from_local(
            local_tensor, device_mesh, [Partial()], shape=global_tensor.shape, stride=global_tensor.stride()
        )
        self.assertEqual(dist_tensor.size(), global_tensor.shape)
        self.assertEqual(dist_tensor.stride(), global_tensor.stride())
        dist_tensor = dist_tensor.redistribute(placements=[Replicate()])
        self.assertEqual(dist_tensor._local_tensor, global_tensor * self.world_size, atol=0, rtol=0)

        # Even Shard(0)
        local_tensors = [torch.randn(3, 3) for _ in range(self.world_size)]
        global_tensor = torch.concat(local_tensors, dim=0)
        dist_tensor = DTensor.from_local(
            local_tensors[self.rank], device_mesh, [Shard(0)], shape=global_tensor.shape, stride=global_tensor.stride()
        )
        self.assertEqual(dist_tensor.size(), global_tensor.shape)
        self.assertEqual(dist_tensor.stride(), global_tensor.stride())
        dist_tensor = dist_tensor.redistribute(placements=[Replicate()])
        self.assertEqual(dist_tensor._local_tensor, global_tensor, atol=0, rtol=0)

        # Uneven Shard(0) without pad
        global_tensor = torch.randn(self.world_size + 1, 2)
        local_tensors, _ = Shard(0)._split_tensor(
            global_tensor,
            device_mesh.size(dim=0),
            with_padding=False,
            contiguous=True,
        )
        dist_tensor = DTensor.from_local(
            local_tensors[self.rank],
            device_mesh,
            (Shard(0),),
            shape=global_tensor.size(),
            stride=global_tensor.stride(),
        )
        self.assertEqual(dist_tensor.size(), global_tensor.size())
        self.assertEqual(dist_tensor.stride(), global_tensor.stride())
        dist_tensor = dist_tensor.redistribute(placements=[Replicate()])
        self.assertEqual(dist_tensor._local_tensor, global_tensor, atol=0, rtol=0)

        # # Uneven Shard(0) with pad # TODO
        # global_tensor = torch.randn(self.world_size + 1, 2)
        # local_tensors, _ = Shard(0)._split_tensor(
        #     global_tensor,
        #     device_mesh.size(dim=0),
        #     with_padding=True,
        #     contiguous=True,
        # )
        # with self.assertRaisesRegex(
        #     ValueError, "Given global shape and stride does not match local shape and stride!"
        # ):
        #     DTensor.from_local(local_tensors[self.rank], device_mesh, (Shard(0),),
        #                         shape=global_tensor.size(), stride=global_tensor.stride())

    @with_comms
    def test_from_local_without_given_shape_stride(self):
        torch.manual_seed(0)

        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        # Replicate
        local_tensor = torch.randn(3 * self.world_size, 3)
        global_tensor = local_tensor
        dist_tensor = DTensor.from_local(local_tensor, device_mesh, [Replicate()])
        self.assertEqual(dist_tensor.size(), global_tensor.shape)
        self.assertEqual(dist_tensor.stride(), global_tensor.stride())
        dist_tensor = dist_tensor.redistribute(placements=[Replicate()])
        self.assertEqual(dist_tensor._local_tensor, global_tensor, atol=0, rtol=0)

        # Partial
        local_tensor = torch.randn(3 * self.world_size, 3)
        global_tensor = local_tensor
        dist_tensor = DTensor.from_local(local_tensor, device_mesh, [Partial()])
        self.assertEqual(dist_tensor.size(), global_tensor.shape)
        self.assertEqual(dist_tensor.stride(), global_tensor.stride())
        dist_tensor = dist_tensor.redistribute(placements=[Replicate()])
        self.assertEqual(dist_tensor._local_tensor, global_tensor * self.world_size, atol=0, rtol=0)

        # Even Shard(0)
        local_tensors = [torch.randn(3, 3) for _ in range(self.world_size)]
        global_tensor = torch.concat(local_tensors, dim=0)
        dist_tensor = DTensor.from_local(local_tensors[self.rank], device_mesh, [Shard(0)])
        self.assertEqual(dist_tensor.size(), global_tensor.shape)
        self.assertEqual(dist_tensor.stride(), global_tensor.stride())
        dist_tensor = dist_tensor.redistribute(placements=[Replicate()])
        self.assertEqual(dist_tensor._local_tensor, global_tensor, atol=0, rtol=0)

        # Uneven Shard(0) without pad
        global_tensor = torch.randn(self.world_size + 1, 2)
        local_tensors, _ = Shard(0)._split_tensor(
            global_tensor,
            device_mesh.size(dim=0),
            with_padding=False,
            contiguous=True,
        )
        dist_tensor = DTensor.from_local(local_tensors[self.rank], device_mesh, (Shard(0),))
        self.assertEqual(dist_tensor.size(), global_tensor.size())
        self.assertEqual(dist_tensor.stride(), global_tensor.stride())
        dist_tensor = dist_tensor.redistribute(placements=[Replicate()])
        self.assertEqual(dist_tensor._local_tensor, global_tensor, atol=0, rtol=0)

    @with_comms
    def test_from_local_without_run_check(self):
        torch.manual_seed(0)

        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        # Replicate
        local_tensor = torch.randn(3 * self.world_size, 3)
        global_tensor = local_tensor
        dist_tensor = DTensor.from_local(local_tensor, device_mesh, [Replicate()], run_check=False)
        self.assertEqual(dist_tensor.size(), global_tensor.shape)
        self.assertEqual(dist_tensor.stride(), global_tensor.stride())
        dist_tensor = dist_tensor.redistribute(placements=[Replicate()])
        self.assertEqual(dist_tensor._local_tensor, global_tensor, atol=0, rtol=0)

        # Partial
        local_tensor = torch.randn(3 * self.world_size, 3)
        global_tensor = local_tensor
        dist_tensor = DTensor.from_local(local_tensor, device_mesh, [Partial()], run_check=False)
        self.assertEqual(dist_tensor.size(), global_tensor.shape)
        self.assertEqual(dist_tensor.stride(), global_tensor.stride())
        dist_tensor = dist_tensor.redistribute(placements=[Replicate()])
        self.assertEqual(dist_tensor._local_tensor, global_tensor * self.world_size, atol=0, rtol=0)

        # Even Shard(0)
        local_tensors = [torch.randn(3, 3) for _ in range(self.world_size)]
        global_tensor = torch.concat(local_tensors, dim=0)
        dist_tensor = DTensor.from_local(local_tensors[self.rank], device_mesh, [Shard(0)], run_check=False)
        self.assertEqual(dist_tensor.size(), global_tensor.shape)
        self.assertEqual(dist_tensor.stride(), global_tensor.stride())
        dist_tensor = dist_tensor.redistribute(placements=[Replicate()])
        self.assertEqual(dist_tensor._local_tensor, global_tensor, atol=0, rtol=0)

        # Uneven Shard(0) without pad
        global_tensor = torch.randn(self.world_size + 1, 2)
        local_tensors, _ = Shard(0)._split_tensor(
            global_tensor,
            device_mesh.size(dim=0),
            with_padding=False,
            contiguous=True,
        )
        dist_tensor = DTensor.from_local(
            local_tensors[self.rank],
            device_mesh,
            (Shard(0),),
            run_check=False,
            shape=global_tensor.size(),
            stride=global_tensor.stride(),
        )
        self.assertEqual(dist_tensor.size(), global_tensor.size())
        self.assertEqual(dist_tensor.stride(), global_tensor.stride())
        dist_tensor = dist_tensor.redistribute(placements=[Replicate()])
        self.assertEqual(dist_tensor._local_tensor, global_tensor, atol=0, rtol=0)

    @with_comms
    def test_to_local(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]
        dist_tensor_shape = torch.Size([self.world_size * 3, 3])
        local_tensor_with_grad = torch.randn(3, 3, device=self.device_type, requires_grad=True)

        sharded_tensor = DTensor(
            local_tensor_with_grad,
            device_mesh,
            shard_spec,
            shape=dist_tensor_shape,
            dtype=local_tensor_with_grad.dtype,
            requires_grad=True,
            stride=local_tensor_with_grad.stride(),
        )
        self.assertEqual(sharded_tensor.size(), dist_tensor_shape)
        self.assertEqual(sharded_tensor.to_local(), local_tensor_with_grad)

        # test dist tensor works with torch.Tensor during backwards
        # dist tensor created is a leaf node, do some operation on dist tensor
        temp_st = sharded_tensor * 3

        # do some operation on local tensor of the dist tensor
        new_tensor_with_grad = torch.randn(3, 3, device=self.device_type, requires_grad=True)
        res = temp_st.to_local() + new_tensor_with_grad
        # call backward directly on torch.Tensor, and see if it works by
        # propagating through dist tensor
        res.sum().backward()
        self.assertIsNotNone(sharded_tensor.grad)

        self.assertEqual(sharded_tensor.grad.to_local(), torch.ones(3, 3) * 3)

        # test the case when grad stride is different from fwd input.
        res = sharded_tensor.to_local()
        model = torch.nn.ReLU()
        res.register_hook(lambda grad: grad.t())
        target = torch.randn(3, 3, device=self.device_type)
        mae_loss = torch.nn.L1Loss()
        output = mae_loss(model(res), target)
        # The manual change to grad stride leads to the failure of the copy op afterwards.
        # so that we need a try-catch here.
        try:
            output.backward()
        except RuntimeError:
            self.assertEqual(sharded_tensor.grad.stride(), [1, 3 * self.world_size])

    @with_comms
    def test_dtensor_async_output(self):
        # Tests that if the output of some dtensor operations isn't used in any compute,
        # the output should be an AsyncCollectiveTensor (representing the fact that
        # we haven't synced the collective yet).
        from torch.distributed._functional_collectives_impl import _tensor_needs_wait

        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size), _validate_mesh=False)

        def fn(dt, async_output):
            dt_out_redistribute = dt.redistribute(mesh, [Replicate()])
            # Make sure we haven't synced yet
            # TODO: figure out why this is returning None
            # self.assertTrue(_tensor_needs_wait(dt_out_redistribute))
            dt_out_redistribute_view = dt_out_redistribute.view(dt_out_redistribute.shape)
            local_tensor = dt_out_redistribute_view.to_local(async_output=async_output)
            return local_tensor

        #### When to_local's async_output is True ####
        x = torch.ones((4, 2), device=self.device_type)
        dt = distribute_tensor(x, mesh, [Shard(0)])
        out = fn(dt, async_output=True)
        # Make sure we haven't synced yet
        self.assertEqual(type(out), AsyncCollectiveTensor)
        self.assertTrue(_tensor_needs_wait(out.elem))
        out_view = out.view(-1)

        # Assert that output is a `AsyncCollectiveTensor`
        self.assertEqual(type(out_view), AsyncCollectiveTensor)
        self.assertTrue(_tensor_needs_wait(out_view.elem))

        # Use the data, requiring a sync
        ref = torch.ones((4, 2), device=self.device_type) + 1
        ref = ref.view(-1)
        out_data = out_view + 1
        self.assertEqual(type(out_data), torch.Tensor)
        self.assertEqual(out_data, ref)

        #### When to_local's async_output is False ####
        x = torch.ones((4, 2), device=self.device_type)
        dt = distribute_tensor(x, mesh, [Shard(0)])
        out = fn(dt, async_output=False)
        # Make sure we have synced already
        self.assertEqual(type(out), torch.Tensor)

    @with_comms
    def test_dtensor_async_input_grad(self):
        # Tests that if the input grad of some dtensor operations isn't used in any compute,
        # the input grad should be an AsyncCollectiveTensor (representing the fact that
        # we haven't synced the collective yet).

        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size), _validate_mesh=False)

        def fn(t, async_input):
            # forward
            t_view = t.view(t.shape)
            dt = DTensor.from_local(t_view, mesh, [Shard(1)], async_input=async_input)
            dt_view = dt.view(dt.shape)
            dt_redistribute = dt_view.redistribute(mesh, [Replicate()])
            # backward
            local_grad = torch.ones(dt_redistribute.shape, device=self.device_type)
            dt_grad = distribute_tensor(local_grad, mesh, [Replicate()])
            dt_redistribute.backward(dt_grad)
            return t.grad

        #### When from_local's async_input is True ####
        # x = torch.ones((4, 2), device=self.device_type, requires_grad=True)
        # x_grad = fn(x, async_input=True)
        # # Make sure we haven't synced yet
        # self.assertTrue(x_grad is not None)
        # self.assertEqual(type(x_grad), AsyncCollectiveTensor)
        # self.assertTrue(_tensor_needs_wait(x_grad.elem))

        #### When from_local's async_input is False ####
        x = torch.ones((4, 2), device=self.device_type, requires_grad=True)
        x_grad = fn(x, async_input=False)
        # Make sure we haven't synced yet
        self.assertTrue(x_grad is not None)
        self.assertEqual(type(x_grad), torch.Tensor)

    @with_comms
    def test_from_local_then_to_local(self):
        # this test ensure end to end from torch.Tensor -> dist tensor -> torch.Tensor works
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]

        # step 1. construct from construct local tensor
        local_tensor_with_grad = torch.randn(3, 3, device=self.device_type, requires_grad=True)
        # do some operations on local tensor
        local_tensor_temp = local_tensor_with_grad + 8
        # step 2. create the dist tensor with non leaf local tensor, dist tensor
        # created should also be non leaf node
        dist_tensor = DTensor.from_local(local_tensor_temp, device_mesh, shard_spec)
        self.assertFalse(dist_tensor.is_leaf)
        # do some random operations on dist tensor
        output = dist_tensor * 6
        self.assertIsInstance(output, DTensor)

        # step 3. do some operation on local tensor of the dist tensor
        new_tensor_with_grad = torch.randn(3, 3, device=self.device_type, requires_grad=True)
        res = output.to_local() + new_tensor_with_grad
        # call backward directly on torch.Tensor, and see if it works by
        # propagating all the way back to the original torch.Tensor
        res.sum().backward()
        self.assertIsNotNone(local_tensor_with_grad.grad)

        expected_grad = torch.ones(3, 3) * 6
        self.assertEqual(local_tensor_with_grad.grad, expected_grad)

    @with_comms
    def test_dtensor_spec_read_only_after_set(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]
        local_tensor = torch.randn(3, 3)
        sharded_tensor = DTensor.from_local(local_tensor, device_mesh, shard_spec)

        # modify shard_spec, and dist_tensor's spec should not be changed
        shard_spec[0] = Replicate()
        self.assertTrue(sharded_tensor.placements is not shard_spec)
        self.assertNotEqual(sharded_tensor.placements, shard_spec)

    @with_comms
    def test_dtensor_spec_hash(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]
        local_tensor = torch.randn(3, 3)
        local_tensor2 = torch.randn(3, 3)
        sharded_tensor = DTensor.from_local(local_tensor, device_mesh, shard_spec)
        sharded_tensor2 = DTensor.from_local(local_tensor2, device_mesh, shard_spec)
        # note that DTensorSpec without real tensor data, so the hash would be the same
        # as long as the mesh, placements and tensor properties are the same
        self.assertEqual(hash(sharded_tensor._spec), hash(sharded_tensor2._spec))

        # change the placements would change the hash
        local_tensor3 = torch.ones(3, 3)
        replica_spec = [Replicate()]
        replica_tensor = DTensor.from_local(local_tensor3, device_mesh, replica_spec, run_check=False)
        self.assertNotEqual(hash(sharded_tensor._spec), hash(replica_tensor._spec))

    @with_comms
    def test_dtensor_properties(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]
        local_tensor = torch.randn(3, 3)
        sharded_tensor = DTensor.from_local(local_tensor, device_mesh, shard_spec)
        self.assertEqual(sharded_tensor.device.type, self.device_type)


class DTensorMeshTest(DTensorTestBase):
    @property
    def world_size(self):
        return 8

    def sub_mesh_assert_equal(self, mesh, exp_in_mesh, exp_out_of_mesh, tensor):
        if self.rank in mesh:
            self.assertEqual(tensor, exp_in_mesh)
        else:
            self.assertEqual(tensor, exp_out_of_mesh)

    @with_comms
    def test_dtensor_device_mesh_device_conversion(self):
        # construct a cuda device mesh
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # construct from a cpu local tensor with cuda device mesh
        # should automatically convert the dist tensor to cuda
        shard_spec = [Shard(0)]
        local_tensor = torch.randn(3, 3)
        dist_tensor = DTensor.from_local(local_tensor, mesh, shard_spec)
        self.assertEqual(dist_tensor.device.type, self.device_type)
        self.assertEqual(dist_tensor.to_local().device.type, self.device_type)

    @with_comms
    def test_dtensor_api_device_mesh_context_manager(self):
        with DeviceMesh(self.device_type, list(range(self.world_size))) as mesh:
            shard_spec = [Shard(0)]
            local_tensor = torch.randn(3, 3)
            sharded_tensor = DTensor.from_local(local_tensor, device_mesh=mesh, placements=shard_spec)

        with DeviceMesh(self.device_type, list(range(self.world_size))):
            shard_spec = [Shard(0)]
            local_tensor = torch.randn(3, 3)
            sharded_tensor = DTensor.from_local(local_tensor, placements=shard_spec)
            replica_spec = [Replicate()]
            replica_tensor = sharded_tensor.redistribute(placements=replica_spec)
            self.assertEqual(replica_tensor.size(), torch.Size([3 * self.world_size, 3]))

        with DeviceMesh(self.device_type, torch.arange(self.world_size)):
            shard_spec = [Shard(0)]
            global_shape = torch.Size([3 * self.world_size, 3])
            global_tensor = torch.randn(global_shape)
            sharded_tensor = distribute_tensor(global_tensor, placements=shard_spec)
            self.assertEqual(sharded_tensor.to_local().shape, torch.Size([3, 3]))

            mesh_2d = DeviceMesh(self.device_type, torch.arange(self.world_size).reshape(2, 4))

            with mesh_2d:
                shard_2d_spec = [Shard(0), Replicate()]
                tensor_2d = distribute_tensor(global_tensor, placements=shard_2d_spec)

                self.assertEqual(tensor_2d.to_local().shape, torch.Size([3 * 4, 3]))

            sharded_after_2d = distribute_tensor(global_tensor, placements=shard_spec)
            self.assertEqual(sharded_after_2d.to_local().shape, torch.Size([3, 3]))

    @with_comms
    def test_dtensor_2d_mesh(self):
        mesh_tensor = torch.arange(self.world_size).reshape(2, 4)
        # construct a cuda device mesh
        mesh = DeviceMesh(self.device_type, mesh_tensor)

        # construct a dist tensor on 2d device mesh and test if works
        shard_spec = [Shard(0), Shard(1)]
        local_tensor = torch.randn(3, 3)
        dist_tensor = DTensor.from_local(local_tensor, mesh, shard_spec)
        self.assertEqual(dist_tensor.size(), torch.Size([3 * mesh.size(0), 3 * mesh.size(1)]))
        self.assertEqual(dist_tensor.device.type, self.device_type)
        self.assertEqual(dist_tensor.to_local().device.type, self.device_type)

        # if shard on the same tensor dimension
        # we should correctly construct the global tensor size
        shard_same_dim_spec = [Shard(0), Shard(0)]
        local_tensor = torch.randn(3, 3)
        dist_tensor = DTensor.from_local(local_tensor, mesh, shard_same_dim_spec)
        self.assertEqual(dist_tensor.size(), torch.Size([3 * self.world_size, 3]))

    @with_comms
    def test_device_mesh_nd(self):
        # construct a cuda device mesh
        mesh_tensor = torch.arange(self.world_size).reshape(2, 2, 2)
        mesh = DeviceMesh(self.device_type, mesh_tensor)
        # construct a dist tensor on 3d device mesh and test if works
        shard_spec = [Shard(0), Shard(1), Shard(2)]
        local_tensor = torch.randn(3, 3, 3)
        dist_tensor = DTensor.from_local(local_tensor, mesh, shard_spec)
        self.assertEqual(dist_tensor.size(), torch.Size([6, 6, 6]))
        self.assertEqual(dist_tensor.device.type, self.device_type)
        self.assertEqual(dist_tensor.to_local().device.type, self.device_type)

        # construct a dist tensor on 3d device mesh with some shards on same dim
        shard_spec = [Shard(0), Shard(0), Shard(2)]
        local_tensor = torch.randn(3, 3, 3)
        dist_tensor = DTensor.from_local(local_tensor, mesh, shard_spec)
        self.assertEqual(dist_tensor.size(), torch.Size([12, 3, 6]))
        self.assertEqual(dist_tensor.device.type, self.device_type)
        self.assertEqual(dist_tensor.to_local().device.type, self.device_type)

    @with_comms
    def test_dtensor_spec_local_shard_offset(self):
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size).reshape(2, 4))
        tensor_shape = (3 * self.world_size, 3 * self.world_size)
        # sharding specs and its corresponding local shard offsets
        shard_spec_and_offsets = [
            (
                [Shard(0), Replicate()],
                (3 * (self.world_size // 2) * (self.rank // 4), 0),
            ),
            (
                [Shard(1), Replicate()],
                (0, 3 * (self.world_size // 2) * (self.rank // 4)),
            ),
            (
                [Replicate(), Shard(0)],
                (3 * (self.world_size // 4) * (self.rank % 4), 0),
            ),
            (
                [Replicate(), Shard(1)],
                (0, 3 * (self.world_size // 4) * (self.rank % 4)),
            ),
        ]

        from vescale.dtensor._utils import compute_local_shape_and_global_offset

        # loop through all sharding specs and check local shard offsets
        logical_tensor = torch.randn(tensor_shape)
        for shard_spec, expected_shard_offsets in shard_spec_and_offsets:
            dtensor = distribute_tensor(logical_tensor, device_mesh, shard_spec)
            _, offset = compute_local_shape_and_global_offset(dtensor.shape, device_mesh, dtensor.placements)
            self.assertEqual(expected_shard_offsets, offset)

    @with_comms
    def test_from_local_sub_mesh(self):
        mesh = DeviceMesh(self.device_type, [0, 2])
        local_tensor = torch.ones(3, 4)

        dtensor = DTensor.from_local(local_tensor, mesh, [Shard(0)])
        self.assertEqual(dtensor.size(), torch.Size([6, 4]))

        self.sub_mesh_assert_equal(
            mesh.mesh,
            torch.ones(3, 4),
            torch.tensor([]),
            dtensor.to_local(),
        )

        # test dtensor created in submesh, the operation should only
        # be applied to the local shard inside the mesh, not the whole
        # world, so only 0/2 really run the computation
        dtensor = dtensor + 2

        self.sub_mesh_assert_equal(
            mesh.mesh,
            torch.ones(3, 4) + 2,
            torch.tensor([]),
            dtensor.to_local(),
        )

    @with_comms
    def test_default_value_sub_mesh(self):
        mesh = DeviceMesh(self.device_type, [0, 2])

        # test scalar return value
        local_tensor1 = torch.ones(4, 3)
        local_tensor2 = torch.ones(4, 3)
        dtensor1 = DTensor.from_local(local_tensor1, mesh, [Shard(0)])
        dtensor2 = DTensor.from_local(local_tensor2, mesh, [Shard(0)])
        local_res = dtensor1.to_local().equal(dtensor2.to_local())
        self.sub_mesh_assert_equal(
            mesh.mesh,
            True,
            True,
            local_res,
        )

        # test 0-d tensor return value
        local_tensor = torch.ones(4, 3)
        dtensor = DTensor.from_local(local_tensor, mesh, [Shard(0)]).sum()
        self.sub_mesh_assert_equal(
            mesh.mesh,
            torch.tensor(12.0),
            torch.tensor(0.0),
            dtensor.to_local(),
        )

        # test List[torch.Tensor] return value
        local_tensor = torch.ones(3, 4)
        dtensor = DTensor.from_local(local_tensor, mesh, [Shard(0)])
        dtensor_list = dtensor.split([2, 2], dim=1)
        self.sub_mesh_assert_equal(
            mesh.mesh,
            [torch.ones(3, 2)] * 2,
            [torch.tensor([])] * 2,
            [dt.to_local() for dt in dtensor_list],
        )

    @with_comms
    def test_redistribute_sub_mesh(self):
        mesh = DeviceMesh(self.device_type, [0, 2])

        # test redistribute on a submesh
        local_tensor1 = torch.ones(4, 3)
        sharded_dtensor = DTensor.from_local(local_tensor1, mesh, [Shard(0)])
        replicated_dtensor = sharded_dtensor.redistribute(placements=[Replicate()])
        self.sub_mesh_assert_equal(mesh.mesh, torch.ones(8, 3), torch.tensor([]), replicated_dtensor.to_local())
        sharded_again = replicated_dtensor.redistribute(placements=[Shard(0)])
        self.sub_mesh_assert_equal(mesh.mesh, torch.ones(4, 3), torch.tensor([]), sharded_again.to_local())


class TestDTensorPlacementTypes(DTensorTestBase):
    @property
    def world_size(self):
        return 8

    def _create_tensor(self, size):
        # Keep everything deterministic.
        torch.manual_seed(0)
        tensor = torch.rand(size)
        if self.device_type == "cuda":
            return tensor.cuda()
        else:
            return tensor

    @with_comms
    def test_split_tensor(self) -> None:
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        shard_placement = Shard(0)

        for size in range(8):
            tensor = self._create_tensor(size)
            if size == 0:
                with self.assertRaisesRegex(
                    Exception,
                    "Tensor size along dim0 is 0. There is nothing to be sharded.",
                ):
                    _, _ = shard_placement._split_tensor(
                        tensor,
                        mesh.size(),
                        with_padding=True,
                        contiguous=True,
                    )
            else:
                splitted_tensor_list, pad_sizes = shard_placement._split_tensor(
                    tensor,
                    mesh.size(),
                    with_padding=True,
                    contiguous=True,
                )
                expected_pad_sizes = [0 if idx < size else 1 for idx, _ in enumerate(range(dist.get_world_size()))]
                assert_array_equal(expected_pad_sizes, pad_sizes)

                unpadded_list = [
                    shard_placement._unpad_tensor(tensor, pad_sizes[i]) if pad_sizes[i] > 0 else tensor
                    for i, tensor in enumerate(splitted_tensor_list)
                ]
                expected_is_tensor_empty = [
                    False if idx < size else True for idx, _ in enumerate(range(dist.get_world_size()))
                ]
                is_tensor_empty = [False if unpadded_tensor.numel() > 0 else True for unpadded_tensor in unpadded_list]
                assert_array_equal(expected_is_tensor_empty, is_tensor_empty)


class TestDynamoDTensor(torch._dynamo.test_case.TestCase):
    def setUp(self):
        super().setUp()
        fake_store = FakeStore()
        dist.init_process_group("fake", store=fake_store, rank=0, world_size=self.world_size)

    def tearDown(self):
        super().tearDown()
        dist.destroy_process_group()

    @property
    def device_type(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def world_size(self) -> int:
        return 2

    @unittest.skip("FIXME!")
    def test_fakify_dtensor(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # pass in DTensor as inputs/outputs to the function
        def fn(x):
            return x

        x = DTensor.from_local(torch.rand(1), mesh, [Shard(0)], run_check=False)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @unittest.skip("FIXME!")
    def test_dynamo_dtensor(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # test passing in DTensor as inputs/outputs and run some tensor computation
        def fn(x):
            return x * x + 2

        x = DTensor.from_local(torch.rand(1), mesh, [Shard(0)], run_check=False)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @unittest.skip("Not support yet!")
    def test_dynamo_dtensor_from_local(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # create DTensor inside fn and run some compute
        def fn(x):
            dt = DTensor.from_local(x, mesh, [Replicate()], run_check=False)
            return dt.to_local() + 2

        # below is the op approach for reference
        # from vescale.dtensor.api import _FromTorchTensor
        # def from_local_tensor(x):
        #     return _FromTorchTensor.apply(x, mesh, [Replicate()], False)

        # _dt_lib_def = torch.library.Library("dtensor", "DEF")
        # _dt_lib_def.define("from_local(Tensor self) -> Tensor")

        # _dt_lib_impl = torch.library.Library("dtensor", "IMPL")
        # _dt_lib_impl.impl("from_local", from_local_tensor, "Autograd")

        x = torch.ones(1)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)

    @unittest.skip("Not support yet!")
    def test_dynamo_dtensor_from_local_redistribute(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))

        # pass in tensor as inputs/outputs, create DTensor and run redistribute
        # (allgather collective) inside the fn
        def fn(x):
            dt = DTensor.from_local(x, mesh, [Shard(0)], run_check=False)
            return dt.redistribute(mesh, [Replicate()]).to_local() + 2

        x = torch.ones(1)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(res, ref)


if __name__ == "__main__":
    run_tests()
