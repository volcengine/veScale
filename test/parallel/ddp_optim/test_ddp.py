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

import copy

import torch
import torch.distributed as dist
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

from vescale.dtensor.placement_types import Replicate
from vescale.dtensor.device_mesh import init_device_mesh
from vescale.dtensor.api import redistribute_dtensor
from vescale.dmodule.api import parallelize_module
from vescale.ddp.distributed_data_parallel import DistributedDataParallel as DDP
from vescale.optim.base_optimizer import BasicOptimizer, BaseOptimizerHook

from common_dtensor import DTensorTestBase, with_comms
from test.test_models.mlp import (
    MLP,
    HIDDEN_DIM,
    BSZ,
    MLP_PAIRWISE_PARAM_SHARDING_PLAN as PAIRWISE_PARAM_SHARDING_PLAN,
    MLP_FWD_RESAHRDING_PLAM as FWD_RESAHRDING_PLAM,
)


def get_unfied_param_and_data(bsz, hidden_dim):
    fc1_weight = torch.rand(hidden_dim * 4, hidden_dim).cuda()
    fc1_bias = torch.rand(hidden_dim * 4).cuda()
    fc2_weight = torch.rand(hidden_dim, hidden_dim * 4).cuda()
    fc2_bias = torch.rand(hidden_dim).cuda()

    batch1_epoch1 = torch.rand(bsz, hidden_dim).cuda()
    batch2_epoch1 = torch.rand(bsz, hidden_dim).cuda()
    batch1_epoch2 = torch.rand(bsz, hidden_dim).cuda()
    batch2_epoch2 = torch.rand(bsz, hidden_dim).cuda()

    # allreduce parameter and batches to make sure they are same at all ranks
    torch.distributed.all_reduce(fc1_weight)
    torch.distributed.all_reduce(fc1_bias)
    torch.distributed.all_reduce(fc2_weight)
    torch.distributed.all_reduce(fc2_bias)
    torch.distributed.all_reduce(batch1_epoch1)
    torch.distributed.all_reduce(batch2_epoch1)
    torch.distributed.all_reduce(batch1_epoch2)
    torch.distributed.all_reduce(batch2_epoch2)

    params_and_inputs = {
        "fc1.weight": fc1_weight,
        "fc1.bias": fc1_bias,
        "fc2.weight": fc2_weight,
        "fc2.bias": fc2_bias,
        "batch1_epoch1": batch1_epoch1,
        "batch2_epoch1": batch2_epoch1,
        "batch1_epoch2": batch1_epoch2,
        "batch2_epoch2": batch2_epoch2,
    }

    return params_and_inputs


class VeScaleDDPTest(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @with_comms
    @parametrize("use_device_mesh", [True, False])
    def test_ddp_basic(self, use_device_mesh: bool):
        tp_parallel_size = 2

        dp_size = self.world_size // tp_parallel_size
        device_mesh = init_device_mesh(self.device_type, (dp_size, tp_parallel_size), mesh_dim_names=("DP", "TP"))
        tp_sub_mesh = device_mesh["TP"]
        dp_pg = device_mesh.get_dim_groups(0)

        ve_model = MLP(HIDDEN_DIM).cuda(self.rank)

        ve_model = parallelize_module(
            ve_model, tp_sub_mesh, {"parameter": PAIRWISE_PARAM_SHARDING_PLAN, "forward": FWD_RESAHRDING_PLAM}
        )

        ve_model = DDP(
            ve_model,
            data_pg_or_device_mesh=device_mesh if use_device_mesh else dp_pg,
            accumulate_allreduce_grads_in_fp32=True,
            overlap_grad_reduce=False,
            use_distributed_optimizer=False,
        )

        if self.rank == 0 or self.rank == 2:
            self.assertEqual(ve_model.data_parallel_ranks, [0, 2])
        if self.rank == 1 or self.rank == 3:
            self.assertEqual(ve_model.data_parallel_ranks, [1, 3])

    def gen_golden_output(self, params_and_inputs):
        m = MLP(HIDDEN_DIM).cuda()
        m.fc1.weight = torch.nn.Parameter(params_and_inputs["fc1.weight"])
        m.fc1.bias = torch.nn.Parameter(params_and_inputs["fc1.bias"])
        m.fc2.weight = torch.nn.Parameter(params_and_inputs["fc2.weight"])
        m.fc2.bias = torch.nn.Parameter(params_and_inputs["fc2.bias"])

        optimizer = torch.optim.Adam(m.parameters(), lr=0.01)

        # epoch 1
        optimizer.zero_grad()
        output = m(params_and_inputs["batch1_epoch1"])
        output.sum().backward()
        output = m(params_and_inputs["batch2_epoch1"])
        output.sum().backward()

        # manually reduce-mean the grad
        for p in m.parameters():
            p.grad /= 2

        optimizer.step()

        # epoch 2
        optimizer.zero_grad()
        output = m(params_and_inputs["batch1_epoch2"])
        output.sum().backward()
        output = m(params_and_inputs["batch2_epoch2"])
        output.sum().backward()

        # manually reduce-mean the grad
        for p in m.parameters():
            p.grad /= 2

        optimizer.step()

        return m.fc1.weight.data, m.fc1.bias.data, m.fc2.weight.data, m.fc2.bias.data

    @with_comms
    @parametrize("overlap_grad_reduce", [True, False])
    def test_ddp_e2e(self, overlap_grad_reduce: bool):
        tp_parallel_size = 2

        dp_size = self.world_size // tp_parallel_size
        device_mesh = init_device_mesh(self.device_type, (dp_size, tp_parallel_size), mesh_dim_names=("DP", "TP"))
        tp_sub_mesh = device_mesh["TP"]
        dp_pg = device_mesh.get_dim_groups(0)

        params_and_inputs = get_unfied_param_and_data(BSZ, HIDDEN_DIM)
        new_params_and_inputs = copy.deepcopy(params_and_inputs)

        ve_model = MLP(HIDDEN_DIM).cuda(self.rank)
        ve_model.fc1.weight = torch.nn.Parameter(params_and_inputs["fc1.weight"])
        ve_model.fc1.bias = torch.nn.Parameter(params_and_inputs["fc1.bias"])
        ve_model.fc2.weight = torch.nn.Parameter(params_and_inputs["fc2.weight"])
        ve_model.fc2.bias = torch.nn.Parameter(params_and_inputs["fc2.bias"])

        ve_model = parallelize_module(
            ve_model, tp_sub_mesh, {"parameter": PAIRWISE_PARAM_SHARDING_PLAN, "forward": FWD_RESAHRDING_PLAM}
        )

        ve_model = DDP(
            ve_model,
            data_pg_or_device_mesh=dp_pg,
            accumulate_allreduce_grads_in_fp32=True,
            overlap_grad_reduce=overlap_grad_reduce,
            use_distributed_optimizer=False,
        )

        ve_optimizer = torch.optim.Adam(ve_model.parameters(), lr=0.01)
        ve_optimizer = BasicOptimizer(ve_optimizer, models=ve_model, grad_hook=BaseOptimizerHook)

        # epoch 1
        ve_optimizer.zero_grad()
        ve_model.zero_grad_buffer()
        x = params_and_inputs["batch1_epoch1"]
        if dist.get_rank() == 2 or dist.get_rank() == 3:
            x = params_and_inputs["batch2_epoch1"]
        ve_model(x).to_local().sum().backward()
        # trigger grad all reducing synchronously if not overlap_grad_reduce,
        # or wait asynchronous grad reduce finish.
        ve_model.finish_grad_sync()
        ve_optimizer.step()

        # epoch 2
        ve_optimizer.zero_grad()
        ve_model.zero_grad_buffer()
        x = params_and_inputs["batch1_epoch2"]
        if dist.get_rank() == 2 or dist.get_rank() == 3:
            x = params_and_inputs["batch2_epoch2"]
        ve_model(x).to_local().sum().backward()
        # trigger grad all reducing synchronously if not overlap_grad_reduce,
        # or wait asynchronous grad reduce finish.
        ve_model.finish_grad_sync()
        ve_optimizer.step()

        fc1_weight, fc1_bias, fc2_weight, fc2_bias = self.gen_golden_output(new_params_and_inputs)
        # all gather parameter split by TP
        ve_fc1_weight = redistribute_dtensor(ve_model.module.fc1.weight, tp_sub_mesh, [Replicate()])
        ve_fc1_bias = redistribute_dtensor(ve_model.module.fc1.bias, tp_sub_mesh, [Replicate()])
        ve_fc2_weight = redistribute_dtensor(ve_model.module.fc2.weight, tp_sub_mesh, [Replicate()])
        ve_fc2_bias = redistribute_dtensor(ve_model.module.fc2.bias, tp_sub_mesh, [Replicate()])

        torch.testing.assert_close(ve_fc1_weight._local_tensor, fc1_weight)
        torch.testing.assert_close(ve_fc1_bias._local_tensor, fc1_bias)
        torch.testing.assert_close(ve_fc2_weight._local_tensor, fc2_weight)
        torch.testing.assert_close(ve_fc2_bias._local_tensor, fc2_bias)


instantiate_parametrized_tests(VeScaleDDPTest)

if __name__ == "__main__":
    run_tests()
