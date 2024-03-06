################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

from common_dtensor import DTensorTestBase, with_comms, skip_unless_torch_gpu

import torch
from torch.testing._internal.common_utils import run_tests
import vescale
from vescale import Shard, DeviceMesh, distribute_tensor, from_local


class DTensorTest(DTensorTestBase):
    @with_comms
    def test_equal(self):
        global_shape = (8, 8)
        for device in (self.device_type, "meta"):  # TODO: add explict cpu, cuda, meta by torch.run
            global_tensor = torch.ones(global_shape, dtype=torch.float, device=device, requires_grad=False)
            device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
            placements = [Shard(1)]
            dt1 = distribute_tensor(global_tensor, device_mesh, placements)
            self.assertTrue(vescale.equal(dt1, dt1))

            global_tensor2 = global_tensor.detach().clone().requires_grad_(True)
            dt2 = distribute_tensor(global_tensor2, device_mesh, placements)
            self.assertFalse(vescale.equal(dt1, dt2))

            device_mesh2 = DeviceMesh(self.device_type, torch.arange(self.world_size // 2))
            dt2 = distribute_tensor(global_tensor, device_mesh2, placements)
            self.assertFalse(vescale.equal(dt1, dt2))

            placements2 = [Shard(0)]
            dt2 = distribute_tensor(global_tensor, device_mesh, placements2)
            self.assertFalse(vescale.equal(dt1, dt2))

            dt2 = distribute_tensor(global_tensor.reshape(self.world_size, -1), device_mesh, placements)
            self.assertFalse(vescale.equal(dt1, dt2))

            dt2 = distribute_tensor(global_tensor + 1e-06, device_mesh, placements)
            if device != "meta":
                self.assertFalse(torch.equal(dt1._local_tensor, dt2._local_tensor))
                self.assertFalse(vescale.equal(dt1, dt2))
            else:
                self.assertTrue(vescale.equal(dt1, dt2))

            dt2 = distribute_tensor(global_tensor, device_mesh, placements)
            if device != "meta":
                self.assertTrue(torch.equal(dt1._local_tensor, dt2._local_tensor))
            self.assertTrue(vescale.equal(dt1, dt2))

    @with_comms
    def test_allclose(self):
        global_shape = (8, 8)
        for device in (self.device_type, "meta"):
            global_tensor = torch.ones(global_shape, dtype=torch.float, device=device, requires_grad=False)
            device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
            placements = [Shard(1)]
            dt1 = distribute_tensor(global_tensor, device_mesh, placements)
            self.assertTrue(vescale.allclose(dt1, dt1))

            global_tensor2 = global_tensor.detach().clone().requires_grad_(True)
            dt2 = distribute_tensor(global_tensor2, device_mesh, placements)
            self.assertFalse(vescale.allclose(dt1, dt2))

            device_mesh2 = DeviceMesh(self.device_type, torch.arange(self.world_size // 2))
            dt2 = distribute_tensor(global_tensor, device_mesh2, placements)
            self.assertFalse(vescale.allclose(dt1, dt2))

            placements2 = [Shard(0)]
            dt2 = distribute_tensor(global_tensor, device_mesh, placements2)
            self.assertFalse(vescale.allclose(dt1, dt2))

            dt2 = distribute_tensor(global_tensor.reshape(self.world_size, -1), device_mesh, placements)
            self.assertFalse(vescale.allclose(dt1, dt2))

            dt2 = distribute_tensor(global_tensor + 1e-01, device_mesh, placements)
            if device != "meta":
                self.assertFalse(torch.allclose(dt1._local_tensor, dt2._local_tensor))
                self.assertFalse(vescale.allclose(dt1, dt2))
            else:
                self.assertTrue(vescale.allclose(dt1, dt2))

            dt2 = distribute_tensor(global_tensor + 1e-06, device_mesh, placements)
            if device != "meta":
                self.assertTrue(torch.allclose(dt1._local_tensor, dt2._local_tensor))
            self.assertTrue(vescale.allclose(dt1, dt2))

            dt2 = distribute_tensor(global_tensor, device_mesh, placements)
            if device != "meta":
                self.assertTrue(torch.allclose(dt1._local_tensor, dt2._local_tensor))
            self.assertTrue(vescale.allclose(dt1, dt2))

    @skip_unless_torch_gpu
    @with_comms
    def test_nonexact_device(self):
        global_shape = (8, 8)
        device_mesh = DeviceMesh("cuda", torch.arange(self.world_size))
        placements = [Shard(1)]

        # meta vs meta
        global_tensor = torch.ones(global_shape, dtype=torch.float, device="meta", requires_grad=False)
        dt1 = distribute_tensor(global_tensor, device_mesh, placements)

        global_tensor2 = torch.ones(global_shape, dtype=torch.float, device="meta", requires_grad=False)
        dt2 = distribute_tensor(global_tensor2, device_mesh, placements)
        self.assertTrue(vescale.equal(dt1, dt2, exact_device=False))
        self.assertTrue(vescale.allclose(dt1, dt2, exact_device=False))

        # cpu/gpu vs cpu/gpu
        global_tensor = torch.ones(global_shape, dtype=torch.float, device="cuda", requires_grad=False)
        dt1 = distribute_tensor(global_tensor, device_mesh, placements)

        local_tensor_cpu = dt1.to_local().cpu()
        dt2 = from_local(local_tensor_cpu, device_mesh, placements, run_check=False)
        self.assertTrue(vescale.equal(dt1, dt2, exact_device=False))
        self.assertTrue(vescale.allclose(dt1, dt2, exact_device=False))

        local_tensor_cpu = dt1.to_local().cpu()
        device_mesh_cpu = DeviceMesh(
            "cpu", torch.arange(self.world_size), _init_process_groups=False, _validate_mesh=False
        )
        dt2 = from_local(
            local_tensor_cpu, device_mesh_cpu, placements, run_check=False, shape=dt1.shape, stride=dt1.stride()
        )
        self.assertEqual(dt2.device.type, "cpu")
        self.assertEqual(dt2._spec.mesh.device_type, "cpu")
        self.assertEqual(dt2._local_tensor.device.type, "cpu")
        self.assertTrue(vescale.equal(dt1, dt2, exact_device=False))
        self.assertTrue(vescale.allclose(dt1, dt2, exact_device=False))

        # cpu/gpu vs meta
        # --> must error at dispatcher!


if __name__ == "__main__":
    run_tests()
