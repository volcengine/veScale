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
from vescale.emulator.distributed import ProcessGroup, dump_nccl_graph_for_pg
from vescale.emulator.reduce_kernel import ReduceOp

from vescale.emulator.all_gather import expand_tensor_list
from vescale.emulator.reduce_scatter import contract_tensor_list
from common_dtensor import DTensorTestBase, with_comms
from emulator.common_emulator import with_comms_emulator
from vescale.emulator.utils import emulator_reduce_op_to_torch


class TestDistributed(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    def init_emulator_pg(self):
        torch.manual_seed(0)
        backend = "nccl"
        world_size = self.world_size

        vescale.emulator.distributed.init_process_group(backend=backend, world_size=world_size, rank=0)
        vescale.emulator.distributed.set_rank(0)
        self.pg: ProcessGroup = vescale.emulator.distributed._world.default_pg
        self.torch_pg = torch.distributed.distributed_c10d._get_default_group()
        dump_nccl_graph_for_pg(self.pg, self.torch_pg, self.rank)

    def destroy_emulator_pg(self):
        vescale.emulator.distributed.destroy_process_group()

    @with_comms
    @with_comms_emulator
    def test_process_group(self):
        ground_truth_pg_group_ranks = [{0: 0, 1: 1, 2: 2, 3: 3}, {0: 0, 2: 1}, {1: 0, 3: 1}, {0: 0, 1: 1}, {2: 0, 3: 1}]
        for count, value in enumerate(vescale.emulator.distributed._world.pg_group_ranks.values()):
            self.assertEqual(value, ground_truth_pg_group_ranks[count])

    @with_comms
    @with_comms_emulator
    # @parametrize("reduce_op", [ReduceOp.SUM, ReduceOp.PRODUCT, ReduceOp.MAX, ReduceOp.MIN])
    @parametrize("reduce_op", [ReduceOp.SUM])
    @parametrize("nelement", [1, 1024, 1024 * 1024])
    def test_all_reduce(self, nelement, reduce_op):
        nranks = self.pg.size()
        tree_structure = [[0, 1], [2, 3]]
        torch_rank = self.rank
        device = f"cuda:{torch_rank}"

        input_file = "input_distributed.pt"
        if self.rank == 0:
            # To ensure all ranks have the same input
            input_list = []
            for i in range(nranks):
                input_list.append(torch.randn((nelement,), device="cuda"))
            torch.save(input_list, input_file)
        dist.barrier()

        data_list = torch.load(input_file)
        data_list = [data.to(device) for data in data_list]
        ground_truth = [data_list[rank].clone().to(device) if rank == torch_rank else [] for rank in range(nranks)]
        torch_reduce_op = emulator_reduce_op_to_torch(reduce_op)

        torch.distributed.all_reduce(ground_truth[torch_rank], torch_reduce_op)
        self.pg.all_reduce(data_list, op=reduce_op, tree_structure=tree_structure)

        self.assertTrue(torch.equal(data_list[torch_rank], ground_truth[torch_rank]))

        if self.rank == 0:
            if os.path.exists(input_file):
                os.remove(input_file)

    @with_comms
    @with_comms_emulator
    @parametrize("nelement", [1, 1024, 1024 * 1024])
    def test_all_gather(self, nelement):
        nranks = self.pg.size()
        torch_rank = self.rank
        device = f"cuda:{torch_rank}"

        input_file = "input_distributed.pt"
        if self.rank == 0:
            # To ensure all ranks have the same input
            input_list = []
            for i in range(nranks):
                input_list.append(torch.randn((nelement,), device="cuda"))
            torch.save(input_list, input_file)
        dist.barrier()

        data_list = torch.load(input_file)
        data_list = [data.to(device) for data in data_list]
        ground_truth_list = [torch.zeros(nelement).to(device) for _ in range(nranks)]
        output_list = expand_tensor_list(data_list)

        torch.distributed.all_gather(ground_truth_list, data_list[torch_rank])
        self.pg.all_gather(output_list, data_list)

        for gt, data in zip(ground_truth_list, data_list):
            self.assertTrue(torch.equal(gt, data))

        if self.rank == 0:
            if os.path.exists(input_file):
                os.remove(input_file)

    @with_comms
    @with_comms_emulator
    # @parametrize("reduce_op", [ReduceOp.SUM, ReduceOp.PRODUCT, ReduceOp.MAX, ReduceOp.MIN])
    @parametrize("reduce_op", [ReduceOp.SUM])
    @parametrize("nelement", [1, 1024, 1024 * 1024])
    def test_reduce_scatter(self, nelement, reduce_op):
        nranks = self.pg.size()
        torch_rank = self.rank
        device = f"cuda:{torch_rank}"

        input_file = "input_distributed.pt"
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
        data_list = [[elem.to(device) for elem in data] for data in data_list]
        ground_truth = torch.zeros(nelement).to(device)
        outputs = contract_tensor_list(data_list)
        torch_reduce_op = emulator_reduce_op_to_torch(reduce_op)

        torch.distributed.reduce_scatter(ground_truth, data_list[torch_rank], torch_reduce_op)

        self.pg.reduce_scatter(outputs, data_list, op=reduce_op)

        result = outputs[torch_rank]
        self.assertTrue(torch.equal(result, ground_truth))

        if self.rank == 0:
            if os.path.exists(input_file):
                os.remove(input_file)

    @with_comms
    @with_comms_emulator
    @parametrize("nelement", [1, 1024, 1024 * 1024])
    def test_all_to_all(self, nelement):
        nranks = self.pg.size()
        torch_rank = self.rank
        device = f"cuda:{torch_rank}"

        input_file = "input_distributed.pt"
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
        ground_truth_list = []
        for i in range(nranks):
            outputs_list.append([])
            for j in range(nranks):
                data_list[i][j] = data_list[i][j].to(device)
                outputs_list[i].append((torch.zeros(nelement)).to(device))
            ground_truth_list.append((torch.zeros(nelement)).to(device))

        torch.distributed.all_to_all(ground_truth_list, data_list[torch_rank])
        self.pg.all_to_all(outputs_list, data_list)

        for gt, output in zip(ground_truth_list, outputs_list[torch_rank]):
            self.assertTrue(torch.equal(gt, output))

        if self.rank == 0:
            if os.path.exists(input_file):
                os.remove(input_file)


instantiate_parametrized_tests(TestDistributed)

if __name__ == "__main__":
    run_tests()
