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
from common_dtensor import DTensorTestBase, with_comms
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from vescale.dtensor.api import redistribute_dtensor
from vescale.dmodule.api import parallelize_module
from vescale.dtensor.device_mesh import init_device_mesh
from vescale.dtensor.placement_types import Replicate
from vescale.ddp.distributed_data_parallel import DistributedDataParallel as DDP
from vescale.optim.distributed_optimizer import DistributedOptimizer

from parallel.ddp_optim.test_ddp import get_unfied_param_and_data
from test.test_models.mlp import (
    MLP,
    HIDDEN_DIM,
    BSZ,
    MLP_PAIRWISE_PARAM_SHARDING_PLAN as PAIRWISE_PARAM_SHARDING_PLAN,
    MLP_FWD_RESAHRDING_PLAM as FWD_RESAHRDING_PLAM,
)


class VeScaleDOptimizerTest(DTensorTestBase):
    @property
    def world_size(self):
        return 4

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
    @parametrize("use_distributed_optimizer", [True, False])
    @parametrize("overlap_param_gather", [True, False])
    @parametrize("use_optimizer_class", [True, False])
    def test_distributed_optimizer(
        self,
        overlap_grad_reduce: bool,
        overlap_param_gather: bool,
        use_distributed_optimizer: bool,
        use_optimizer_class: bool,
    ):
        tp_parallel_size = 2
        dp_size = self.world_size // tp_parallel_size
        device_mesh = init_device_mesh(self.device_type, (dp_size, tp_parallel_size), mesh_dim_names=("DP", "TP"))

        params_and_inputs = get_unfied_param_and_data(BSZ, HIDDEN_DIM)
        new_params_and_inputs = copy.deepcopy(params_and_inputs)
        tp_sub_mesh = device_mesh["TP"]
        dp_pg = device_mesh.get_dim_groups(0)
        tp_pg = device_mesh.get_dim_groups(1)

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
            use_distributed_optimizer=use_distributed_optimizer,
        )

        if not use_optimizer_class:
            orig_optimizer = torch.optim.Adam(ve_model.parameters(), lr=0.01)
            ve_optimizer = DistributedOptimizer(
                orig_optimizer,
                models=[ve_model],
                overlap_param_gather=overlap_param_gather,
            )
        else:
            ve_optimizer = DistributedOptimizer(
                torch.optim.Adam,
                models=[ve_model],
                overlap_param_gather=overlap_param_gather,
                optimizer_kwargs={"lr": 0.01},
            )

        # epoch 1
        # NOTE: we can't invoke optimizer.zero_grad here. Because if overlap_param_gather is True,
        # DOptimizer will try to all gather parameters.
        for m in ve_optimizer.models:
            m.zero_grad_buffer()

        x = params_and_inputs["batch1_epoch1"]
        if dist.get_rank() == 2 or dist.get_rank() == 3:
            x = params_and_inputs["batch2_epoch1"]
        ve_model(x).to_local().sum().backward()
        ve_model.finish_grad_sync()
        ve_optimizer.step()

        # epoch 2
        ve_optimizer.zero_grad()
        x = params_and_inputs["batch1_epoch2"]
        if dist.get_rank() == 2 or dist.get_rank() == 3:
            x = params_and_inputs["batch2_epoch2"]
        ve_model(x).to_local().sum().backward()
        ve_model.finish_grad_sync()
        ve_optimizer.step()

        # perform extra forward to trigger parameter all gather.
        if overlap_param_gather:
            ve_optimizer.zero_grad()
            ve_model(x)

        fc1_weight, fc1_bias, fc2_weight, fc2_bias = self.gen_golden_output(new_params_and_inputs)
        ve_fc1_weight = redistribute_dtensor(
            ve_model.module.fc1.weight,
            tp_sub_mesh,
            [Replicate()],
            async_op=False,
        )
        ve_fc1_bias = redistribute_dtensor(
            ve_model.module.fc1.bias,
            tp_sub_mesh,
            [Replicate()],
            async_op=False,
        )
        ve_fc2_weight = redistribute_dtensor(
            ve_model.module.fc2.weight,
            tp_sub_mesh,
            [Replicate()],
            async_op=False,
        )
        ve_fc2_bias = redistribute_dtensor(
            ve_model.module.fc2.bias,
            tp_sub_mesh,
            [Replicate()],
            async_op=False,
        )

        torch.testing.assert_close(ve_fc1_weight._local_tensor, fc1_weight)
        torch.testing.assert_close(ve_fc1_bias._local_tensor, fc1_bias)
        torch.testing.assert_close(ve_fc2_weight._local_tensor, fc2_weight)
        torch.testing.assert_close(ve_fc2_bias._local_tensor, fc2_bias)


instantiate_parametrized_tests(VeScaleDOptimizerTest)

if __name__ == "__main__":
    run_tests()
