################################################################################
#
# Copyright 2023 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################


import os
import torch
import torch.distributed as dist
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

import vescale
from vescale.emulator.device_mesh import DeviceMesh, dump_nccl_graph_for_mesh
from vescale.emulator.distributed import ProcessGroup, dump_nccl_graph_for_pg
from vescale.emulator.reduce_kernel import ReduceOp
from vescale.emulator.mesh_collectives import mesh_all_gather, mesh_all_reduce, mesh_reduce_scatter, mesh_all_to_all
from common_dtensor import DTensorTestBase, with_comms
from emulator.common_emulator import with_comms_emulator
from vescale.emulator.utils import emulator_reduce_op_to_torch


class TestMeshCollectives(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    def init_emulator_pg(self):
        torch.manual_seed(0)
        backend = "nccl"
        world_size = self.world_size
        dp_size = 2
        tp_size = 2

        vescale.emulator.distributed.init_process_group(backend=backend, world_size=world_size, rank=0)
        vescale.emulator.distributed.set_rank(0)
        self.pg: ProcessGroup = vescale.emulator.distributed._world.default_pg
        self.torch_pg = torch.distributed.distributed_c10d._get_default_group()
        # dump default process group
        dump_nccl_graph_for_pg(self.pg, self.torch_pg, self.rank)

        mesh_tensor = torch.tensor(list(range(world_size))).view(dp_size, tp_size)
        self.vescale_mesh = vescale.dtensor.device_mesh.DeviceMesh(self.device_type, mesh_tensor)
        self.mesh = DeviceMesh(self.device_type, mesh_tensor)
        # dump for other process groups
        dump_nccl_graph_for_mesh(self.mesh, self.vescale_mesh)

    def destroy_emulator_pg(self):
        vescale.emulator.distributed.destroy_process_group()

    @with_comms
    @with_comms_emulator
    @parametrize("mesh_dim", [0, 1])
    @parametrize("scatter_dim", [0])
    @parametrize("nelement", [1, 1024, 1024 * 1024])
    def test_mesh_all_gather(self, mesh_dim, scatter_dim, nelement):
        nranks = self.pg.size()
        torch_rank = self.rank
        device = f"cuda:{torch_rank}"

        input_file = "input_mesh_coll.pt"
        if self.rank == 0:
            tensor_list = [torch.randn((nelement,)).to(device) for _ in range(nranks)]
            torch.save(tensor_list, input_file)
        dist.barrier()

        tensor_list = torch.load(input_file)
        tensor_list = [tensor.to(device) for tensor in tensor_list]

        local_tensor = tensor_list[torch_rank]
        group = self.vescale_mesh.get_dim_groups(mesh_dim)
        group_world_size = torch.distributed.get_world_size(group)

        ground_truth = vescale.dtensor._collective_utils.mesh_all_gather(
            local_tensor, [nelement * group_world_size], self.vescale_mesh, scatter_dim, mesh_dim
        )
        result = mesh_all_gather(tensor_list, self.mesh, scatter_dim, mesh_dim)

        self.assertTrue(torch.equal(result[torch_rank], ground_truth))

        if self.rank == 0:
            if os.path.exists(input_file):
                os.remove(input_file)

    @with_comms
    @with_comms_emulator
    # @parametrize("reduce_op", [ReduceOp.SUM, ReduceOp.PRODUCT, ReduceOp.MAX, ReduceOp.MIN])
    @parametrize("reduce_op", [ReduceOp.SUM])
    @parametrize("mesh_dim", [0, 1])
    @parametrize("nelement", [1, 1024, 1024 * 1024])
    def test_mesh_all_reduce(self, reduce_op, mesh_dim, nelement):
        nranks = self.pg.size()
        torch_rank = self.rank
        device = f"cuda:{torch_rank}"
        tree_structure = [[0, 1, 2, 3]]

        input_file = "input_mesh_coll.pt"
        if self.rank == 0:
            # To ensure all ranks have the same input
            input_list = []
            for i in range(nranks):
                input_list.append(torch.randn((nelement,), device="cuda"))
            torch.save(input_list, input_file)
        dist.barrier()

        tensor_list = torch.load(input_file)
        tensor_list = [data.to(device) for data in tensor_list]
        ground_truth = [tensor_list[rank].clone().to(device) if rank == torch_rank else [] for rank in range(nranks)]
        torch_reduce_op = emulator_reduce_op_to_torch(reduce_op)

        ground_truth[torch_rank] = vescale.dtensor._collective_utils.mesh_all_reduce(
            ground_truth[torch_rank], self.vescale_mesh, torch_reduce_op, mesh_dim
        )
        result = mesh_all_reduce(tensor_list, self.mesh, reduce_op, mesh_dim, tree_structure=tree_structure)

        self.assertTrue(torch.equal(result[torch_rank], ground_truth[torch_rank]))

        if self.rank == 0:
            if os.path.exists(input_file):
                os.remove(input_file)

    @with_comms
    @with_comms_emulator
    # @parametrize("reduce_op", [ReduceOp.SUM, ReduceOp.PRODUCT, ReduceOp.MAX, ReduceOp.MIN])
    @parametrize("reduce_op", [ReduceOp.SUM])
    @parametrize("mesh_dim", [0, 1])
    @parametrize("scatter_dim", [0])
    @parametrize("nelement", [1024, 1024 * 1024])
    def test_mesh_reduce_scatter(self, reduce_op, mesh_dim, scatter_dim, nelement):
        nranks = self.pg.size()
        torch_rank = self.rank
        device = f"cuda:{torch_rank}"

        input_file = "input_mesh_coll.pt"
        if self.rank == 0:
            # To ensure all ranks have the same input
            input_list = []
            for i in range(nranks):
                input_list.append(torch.randn((nelement,), device="cuda"))
            torch.save(input_list, input_file)
        dist.barrier()

        tensor_list = torch.load(input_file)
        tensor_list = [data.to(device) for data in tensor_list]
        ground_truth = [tensor_list[rank].clone().to(device) if rank == torch_rank else [] for rank in range(nranks)]
        torch_reduce_op = emulator_reduce_op_to_torch(reduce_op)

        ground_truth[torch_rank] = vescale.dtensor._collective_utils.mesh_reduce_scatter(
            ground_truth[torch_rank], self.vescale_mesh, torch_reduce_op, scatter_dim, mesh_dim
        )
        result = mesh_reduce_scatter(tensor_list, self.mesh, reduce_op, scatter_dim, mesh_dim)

        self.assertTrue(torch.equal(result[torch_rank], ground_truth[torch_rank]))

        if self.rank == 0:
            if os.path.exists(input_file):
                os.remove(input_file)

    @with_comms
    @with_comms_emulator
    @parametrize("mesh_dim", [0, 1])
    @parametrize("nelement", [1, 1024, 1024 * 1024])
    def test_mesh_all_to_all(self, mesh_dim, nelement):
        nranks = self.pg.size()
        torch_rank = self.rank
        device = f"cuda:{torch_rank}"

        input_file = "input_mesh_coll.pt"
        if self.rank == 0:
            # To ensure all ranks have the same input
            input_list = []
            for i in range(nranks):
                input_list.append([])
                for j in range(nranks):
                    input_list[i].append(torch.randn((nelement,), device="cuda"))
            torch.save(input_list, input_file)
        dist.barrier()

        data_list = torch.load(input_file)
        outputs_list = []
        for i in range(nranks):
            outputs_list.append([])
            for j in range(nranks):
                data_list[i][j] = data_list[i][j].to(device)
                outputs_list[i].append((torch.zeros(nelement)).to(device))

        local_tensor_list = data_list[torch_rank]
        group = self.vescale_mesh.get_dim_groups(mesh_dim)
        group_world_size = torch.distributed.get_world_size(group)
        ground_truth_list = [torch.zeros(nelement).to(device) for _ in range(nranks)]

        ground_truth_list = [torch.cat(ground_truth_list, dim=0)]
        ground_truth_list = list(torch.chunk(ground_truth_list[0], group_world_size, dim=0))
        local_tensor_list = [torch.cat(local_tensor_list, dim=0)]
        local_tensor_list = list(torch.chunk(local_tensor_list[0], group_world_size, dim=0))

        vescale.dtensor._collective_utils.mesh_all_to_all(
            ground_truth_list, local_tensor_list, self.vescale_mesh, mesh_dim
        )
        mesh_all_to_all(outputs_list, data_list, self.mesh, mesh_dim)

        local_output = outputs_list[torch_rank]
        local_output = torch.cat(local_output, dim=0)
        ground_truth = torch.cat(ground_truth_list, dim=0)
        self.assertTrue(torch.equal(local_output, ground_truth))

        if self.rank == 0:
            if os.path.exists(input_file):
                os.remove(input_file)


instantiate_parametrized_tests(TestMeshCollectives)

if __name__ == "__main__":
    run_tests()
