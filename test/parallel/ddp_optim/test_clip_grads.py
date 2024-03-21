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
from vescale.dtensor.api import distribute_tensor
from vescale.dmodule.api import parallelize_module
from vescale.dtensor.device_mesh import init_device_mesh
from vescale.ddp.distributed_data_parallel import DistributedDataParallel as DDP
from vescale.optim.distributed_optimizer import DistributedOptimizer

from parallel.ddp_optim.test_ddp import get_unfied_param_and_data
from test.test_models.mlp import MLP, HIDDEN_DIM, BSZ, MLP_PAIRWISE_PARAM_SHARDING_PLAN, MLP_FWD_RESAHRDING_PLAM


class VeScaleClipGradsTest(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    def golden_run(self, params_and_inputs, max_norm):
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
        torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm, norm_type=2.0)

        return m

    @with_comms
    @parametrize("max_norm", [2.0])
    def test_clip_grad(self, max_norm):
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
            ve_model, tp_sub_mesh, {"parameter": MLP_PAIRWISE_PARAM_SHARDING_PLAN, "forward": MLP_FWD_RESAHRDING_PLAM}
        )

        ve_model = DDP(
            ve_model,
            data_pg_or_device_mesh=dp_pg,
            accumulate_allreduce_grads_in_fp32=True,
            overlap_grad_reduce=True,
            use_distributed_optimizer=True,
        )

        orig_optimizer = torch.optim.Adam(ve_model.parameters(), lr=0.01)
        ve_optimizer = DistributedOptimizer(
            orig_optimizer,
            models=[ve_model],
            overlap_param_gather=False,
            clip_grad=max_norm,
        )

        # epoch 1
        ve_optimizer.zero_grad()
        x = params_and_inputs["batch1_epoch1"]
        if dist.get_rank() == 2 or dist.get_rank() == 3:
            x = params_and_inputs["batch2_epoch1"]
        ve_model(x).to_local().sum().backward()
        ve_model.finish_grad_sync()

        # copy main grad in grad buffer to sharded parameters' grad field.
        ve_optimizer._copy_model_grads_to_main_grads()
        # do the grad norm clipping
        ve_optimizer.clip_grad_norm(ve_optimizer.clip_grad)

        golden_mlp = self.golden_run(new_params_and_inputs, max_norm=max_norm)
        golden_fc1_weight_grad = distribute_tensor(
            golden_mlp.fc1.weight.grad.data, tp_sub_mesh, MLP_PAIRWISE_PARAM_SHARDING_PLAN["fc1.weight"]
        )._local_tensor
        golden_fc1_bias_grad = distribute_tensor(
            golden_mlp.fc1.bias.grad.data, tp_sub_mesh, MLP_PAIRWISE_PARAM_SHARDING_PLAN["fc1.bias"]
        )._local_tensor
        golden_fc2_weight_grad = distribute_tensor(
            golden_mlp.fc2.weight.grad.data, tp_sub_mesh, MLP_PAIRWISE_PARAM_SHARDING_PLAN["fc2.weight"]
        )._local_tensor
        golden_fc2_bias_grad = distribute_tensor(
            golden_mlp.fc2.bias.grad.data, tp_sub_mesh, MLP_PAIRWISE_PARAM_SHARDING_PLAN["fc2.bias"]
        )._local_tensor

        if self.rank in [0, 1]:
            optimizer_params = ve_optimizer.get_parameters()
            ve_fc2_bias_grad = optimizer_params[0].grad
            ve_fc2_weight_grad = optimizer_params[1].grad
            ve_fc1_bias_head_2_grad = optimizer_params[2].grad
            torch.testing.assert_close(golden_fc2_bias_grad, ve_fc2_bias_grad)
            torch.testing.assert_close(golden_fc2_weight_grad.flatten(), ve_fc2_weight_grad)
            torch.testing.assert_close(golden_fc1_bias_grad[:2,], ve_fc1_bias_head_2_grad)
        if self.rank in [2, 3]:
            optimizer_params = ve_optimizer.get_parameters()
            ve_fc1_bias_tail_6_grad = optimizer_params[0].grad
            ve_fc1_weight_grad = optimizer_params[1].grad
            torch.testing.assert_close(golden_fc1_bias_grad[2:], ve_fc1_bias_tail_6_grad)
            torch.testing.assert_close(golden_fc1_weight_grad.flatten(), ve_fc1_weight_grad)

        """
        The above check may seem strange. We do some explanations here.
        The parallel mesh is like
            0  |  1
            -------
            2  |  3
        Ranks in horizontal direction construct tensor parallelism, while ones in the vertical direction construct data parallelism
        After splitting the model weights in tensor parallel, the weights and sizes on each rank are as follows:
            0(fc1.weight^1[8x4], fc1.bias^1[8], fc2.weight^1[4x8], fc2.bias^1[4]) | 1(fc1.weight^2[8x4], fc1.bias^2[8], fc2.weight^2[4x8], fc2.bias^2[4])
            ---------------------------------------------------------------------------------------------------------------------------------------------
            2(fc1.weight^1[8x4], fc1.bias^1[8], fc2.weight^1[4x8], fc2.bias^1[4]) | 3(fc1.weight^2[8x4], fc1.bias^2[8], fc2.weight^2[4x8], fc2.bias^2[4])
        Different superscripts represent different parts of the model parameters

        The next step is DOptimizer will take split model weights, flatten them into a 1 rank tensor, and further split them across data parallel. As
        a result, rank 0's optimizer get the whole fc2.bias^1, fc2.weight^1 and the first 2 elements of fc1.bias^1, while rank 2's optimizer get the last 6
        elements of fc1.bias^1 and whole fc1.weight^1. Note that, DDP or DOptimzer traverse model weights in a reverse order.
        """


instantiate_parametrized_tests(VeScaleClipGradsTest)

if __name__ == "__main__":
    run_tests()
