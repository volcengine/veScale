################################################################################
#
# Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
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
    run_tests,
)

from vescale.dtensor.placement_types import Replicate, Shard
from vescale.dtensor.device_mesh import init_device_mesh
from vescale.dtensor.api import redistribute_dtensor
from vescale.dmodule.api import parallelize_module
from vescale.ddp.distributed_data_parallel import DistributedDataParallel as DDP
from vescale.optim.base_optimizer import BasicOptimizer, BasicOptimizerHook

from common_dtensor import DTensorTestBase, with_comms
from test_models.mlp import (
    MLP,
    BSZ,
)

HIDDEN_DIM = 512

PAIRWISE_PARAM_SHARDING_PLAN = {
    r"moe.experts.\d+.fc1.weight": [Shard(0)],
    r"moe.experts.\d+.fc1.bias": [Shard(0)],
    r"moe.experts.\d+.fc2.weight": [Shard(1)],
    r"moe.experts.\d+.fc2.bias": [Replicate()],
    r"ln.weight": [Replicate()],
    r"ln.bias": [Replicate()],
}

FWD_RESAHRDING_PLAM = {
    r".input": [[Replicate()]],
    r"moe.experts.\d+.fc1.input": [[Replicate()]],
    r"moe.experts.\d+.fc2.output": [[Replicate()]],
}


def get_unfied_param_and_data(bsz, hidden_dim, dtype=torch.float):
    fc1_weights = torch.rand(8, hidden_dim * 4, hidden_dim, dtype=dtype).cuda()
    fc1_biases = torch.rand(8, hidden_dim * 4, dtype=dtype).cuda()

    fc2_weights = torch.rand(8, hidden_dim, hidden_dim * 4, dtype=dtype).cuda()
    fc2_biases = torch.rand(8, hidden_dim, dtype=dtype).cuda()

    ln_weight = torch.rand(hidden_dim).cuda()
    ln_bias = torch.rand(hidden_dim).cuda()

    batch1_epoch1 = torch.rand(bsz, hidden_dim, dtype=dtype).cuda()
    batch2_epoch1 = torch.rand(bsz, hidden_dim, dtype=dtype).cuda()
    batch1_epoch2 = torch.rand(bsz, hidden_dim, dtype=dtype).cuda()
    batch2_epoch2 = torch.rand(bsz, hidden_dim, dtype=dtype).cuda()

    # allreduce parameter and batches to make sure they are same at all ranks
    torch.distributed.all_reduce(fc1_weights)
    torch.distributed.all_reduce(fc1_biases)
    torch.distributed.all_reduce(fc2_weights)
    torch.distributed.all_reduce(fc2_biases)
    torch.distributed.all_reduce(ln_weight)
    torch.distributed.all_reduce(ln_bias)
    torch.distributed.all_reduce(batch1_epoch1)
    torch.distributed.all_reduce(batch2_epoch1)
    torch.distributed.all_reduce(batch1_epoch2)
    torch.distributed.all_reduce(batch2_epoch2)

    params_and_inputs = {
        "fc1.weights": torch.unbind(fc1_weights, 0),
        "fc1.biases": torch.unbind(fc1_biases, 0),
        "fc2.weights": torch.unbind(fc2_weights, 0),
        "fc2.biases": torch.unbind(fc2_biases, 0),
        "ln.weight": ln_weight,
        "ln.bias": ln_bias,
        "batch1_epoch1": batch1_epoch1,
        "batch2_epoch1": batch2_epoch1,
        "batch1_epoch2": batch1_epoch2,
        "batch2_epoch2": batch2_epoch2,
    }

    return params_and_inputs


class MoEBlock(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.experts = torch.nn.ModuleList(
            [
                MLP(hidden_dim),
                MLP(hidden_dim),
                MLP(hidden_dim),
                MLP(hidden_dim),
                MLP(hidden_dim),
                MLP(hidden_dim),
                MLP(hidden_dim),
                MLP(hidden_dim),
            ]
        )

    def forward(self, x):
        # we simulate a sparse MoE by only invoking some of the experts.
        output = torch.zeros_like(x)
        for i in range(0, 4):
            output += self.experts[i](x)
        return output


class Net(torch.nn.Module):
    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.moe = MoEBlock(hidden_dim)
        self.ln = torch.nn.LayerNorm(hidden_dim)

    def forward(self, x):
        return self.moe(self.ln(x))


class VeScaleDDPTest(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    def gen_golden_output(self, params_and_inputs):
        m = Net(HIDDEN_DIM).cuda()
        m.ln.weight = torch.nn.Parameter(params_and_inputs["ln.weight"])
        m.ln.bias = torch.nn.Parameter(params_and_inputs["ln.bias"])
        for i in range(8):
            m.moe.experts[i].fc1.weight = torch.nn.Parameter(params_and_inputs["fc1.weights"][i])
            m.moe.experts[i].fc1.bias = torch.nn.Parameter(params_and_inputs["fc1.biases"][i])
            m.moe.experts[i].fc2.weight = torch.nn.Parameter(params_and_inputs["fc2.weights"][i])
            m.moe.experts[i].fc2.bias = torch.nn.Parameter(params_and_inputs["fc2.biases"][i])

        optimizer = torch.optim.Adam(m.parameters(), lr=0.01)

        # epoch 1
        optimizer.zero_grad()
        output = m(params_and_inputs["batch1_epoch1"])
        output.sum().backward()
        output = m(params_and_inputs["batch2_epoch1"])
        output.sum().backward()

        # manually reduce-mean the grad
        for p in m.parameters():
            if p.grad is not None:
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
            if p.grad is not None:
                p.grad /= 2

        optimizer.step()

        return m

    @with_comms
    def test_ddp_moe(self):
        tp_parallel_size = 2

        dp_size = self.world_size // tp_parallel_size
        device_mesh = init_device_mesh(self.device_type, (dp_size, tp_parallel_size), mesh_dim_names=("DP", "TP"))
        tp_sub_mesh = device_mesh["TP"]
        dp_pg = device_mesh.get_dim_groups(0)

        params_and_inputs = get_unfied_param_and_data(BSZ, HIDDEN_DIM)
        new_params_and_inputs = copy.deepcopy(params_and_inputs)

        ve_model = Net(HIDDEN_DIM).cuda(self.rank)
        ve_model.ln.weight = torch.nn.Parameter(params_and_inputs["ln.weight"])
        ve_model.ln.bias = torch.nn.Parameter(params_and_inputs["ln.bias"])
        for i in range(8):
            ve_model.moe.experts[i].fc1.weight = torch.nn.Parameter(params_and_inputs["fc1.weights"][i])
            ve_model.moe.experts[i].fc1.bias = torch.nn.Parameter(params_and_inputs["fc1.biases"][i])
            ve_model.moe.experts[i].fc2.weight = torch.nn.Parameter(params_and_inputs["fc2.weights"][i])
            ve_model.moe.experts[i].fc2.bias = torch.nn.Parameter(params_and_inputs["fc2.biases"][i])

        ve_model = parallelize_module(
            ve_model, tp_sub_mesh, {"parameter": PAIRWISE_PARAM_SHARDING_PLAN, "forward": FWD_RESAHRDING_PLAM}
        )

        ve_model = DDP(
            ve_model,
            data_pg_or_device_mesh=dp_pg,
            accumulate_allreduce_grads_in_fp32=True,
            overlap_grad_reduce=True,
            use_distributed_optimizer=False,
            bucket_size=2000000,
            whitelist_module_types=[MoEBlock],
        )

        ve_optimizer = torch.optim.Adam(ve_model.parameters(), lr=0.01)
        ve_optimizer = BasicOptimizer(ve_optimizer, models=ve_model, grad_hook=BasicOptimizerHook)

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

        golden_module = self.gen_golden_output(new_params_and_inputs)
        for name, golden_param in golden_module.named_parameters():
            param = ve_model.module.get_parameter(name)
            replicate_param = redistribute_dtensor(param.data, tp_sub_mesh, [Replicate()])
            torch.testing.assert_close(golden_param.data, replicate_param._local_tensor)


if __name__ == "__main__":
    run_tests()
