################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

import os

import numpy as np
from common_dtensor import (
    DTensorTestBase,  # skip_unless_torch_gpu,
    with_comms,
)
from typing import List, cast

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch.testing._internal.common_utils import run_tests

import vescale
from vescale.dtensor.dtensor import DTensor
from vescale.dtensor.placement_types import Placement, Replicate, Shard

from vescale.emulator.device_mesh import dump_nccl_graph_for_mesh
from vescale.emulator.distributed import ProcessGroup, dump_nccl_graph_for_pg
from vescale.emulator.comm_api import distribute_tensor, redistribute_dtensor
from vescale.emulator.device_mesh import DeviceMesh
from vescale.emulator.emulator_instrumentation import EmulatorInstrumentation
from emulator.common_emulator import with_comms_emulator


class DistMatrixOpsTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    def init_emulator_pg(self):
        torch.manual_seed(0)
        backend = "nccl"
        world_size = self.world_size

        vescale.emulator.distributed.init_process_group(backend=backend, world_size=world_size, rank=0)
        vescale.emulator.distributed.set_rank(0)
        # dump default process group
        self.pg: ProcessGroup = vescale.emulator.distributed._world.default_pg
        self.torch_pg = torch.distributed.distributed_c10d._get_default_group()
        dump_nccl_graph_for_pg(self.pg, self.torch_pg, self.rank)

        # dump for other process groups
        mesh_tensor = list(range(world_size))
        self.vescale_mesh = vescale.dtensor.device_mesh.DeviceMesh(self.device_type, mesh_tensor)
        self.mesh = DeviceMesh(self.device_type, mesh_tensor)
        dump_nccl_graph_for_mesh(self.mesh, self.vescale_mesh)

    def destroy_emulator_pg(self):
        vescale.emulator.distributed.destroy_process_group()

    @with_comms
    @with_comms_emulator
    def test_mm(self):
        device_mesh = self.mesh
        vescale_device_mesh = vescale.dtensor.device_mesh.DeviceMesh(self.device_type, list(range(self.world_size)))
        device = f"cuda:{self.rank}"
        replica_spec = Replicate()

        input_file = "input_dtensors.pt"
        if self.rank == 0:
            t1 = torch.randn(12, 8, requires_grad=True).cuda()
            t2 = torch.randn(8, 12, requires_grad=True).cuda()
            torch.save((t1, t2), input_file)
        dist.barrier()

        t1, t2 = torch.load(input_file)
        t1 = t1.to(device)
        t2 = t2.to(device)
        t1_list = [t1.clone().detach().requires_grad_() for _ in range(self.world_size)]
        t2_list = [t2.clone().detach().requires_grad_() for _ in range(self.world_size)]

        def test_placement_comb(placements1: List[Placement], placements2: List[Placement]) -> None:
            dt1_list = distribute_tensor(t1_list, device_mesh, placements1)
            dt2_list = distribute_tensor(t2_list, device_mesh, placements2)

            # Emulator replace the given pytorch function to accpet lists of tensors as input
            func_list = ["mm"]
            indices = [(0, 1)]
            with EmulatorInstrumentation(torch, func_list, indices):
                dist_res_list = torch.mm(dt1_list, dt2_list)
                dist_res_list = redistribute_dtensor(dist_res_list, device_mesh, [replica_spec])

            dt1 = vescale.distribute_tensor(t1.clone().detach().requires_grad_(), vescale_device_mesh, placements1)
            dt2 = vescale.distribute_tensor(t2.clone().detach().requires_grad_(), vescale_device_mesh, placements2)
            dist_res: DTensor = cast(DTensor, torch.mm(dt1, dt2)).redistribute(vescale_device_mesh, [replica_spec])

            for dist_res_emu in dist_res_list:
                self.assertTrue(torch.equal(dist_res.to_local(), dist_res_emu.to_local()))

        shard_specs_comb = [
            (Shard(dim=0), Replicate()),
            (Shard(dim=1), Shard(dim=0)),
            (Replicate(), Shard(dim=1)),
            (Replicate(), Replicate()),
        ]

        for spec in shard_specs_comb:
            test_placement_comb([spec[0]], [spec[1]])

        if self.rank == 0:
            if os.path.exists(input_file):
                os.remove(input_file)


if __name__ == "__main__":
    run_tests()
