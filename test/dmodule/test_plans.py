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

import torch
import torch.distributed as dist
from torch import nn
from torch.testing._internal.common_utils import run_tests

from common_dtensor import DTensorTestBase, with_comms_device

from vescale.dmodule.api import parallelize_module
from vescale.dtensor.device_mesh import DeviceMesh
from vescale.dtensor.placement_types import Replicate, Shard


config = {"seq_length": 8, "head_size": 4, "hidden_size": 4 * 4, "n_head": 4, "batch_size": 4}


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config["hidden_size"], config["hidden_size"] * 4)
        self.gelu = torch.nn.GELU()
        self.fc2 = nn.Linear(config["hidden_size"] * 4, config["hidden_size"])

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x


param_sharding_plan1 = {
    "fc1.weight": [Shard(0)],
    "fc1.bias": [Shard(0)],
    "fc2.weight": [Shard(1)],
    "fc2.bias": [Replicate()],
}

fwd_resharding_plan1 = {
    "fc1.input": [[Replicate()]],
    "fc2.output": [[Replicate()]],
}

param_sharding_plan2 = {
    "fc1.weight": [Shard(0)],
    "fc1.bias": [Replicate()],
    "fc2.weight": [Shard(0)],
    "fc2.bias": [Replicate()],
}

fwd_resharding_plan2 = {
    "fc1.input": [[Replicate()]],
    "fc1.weight": [Replicate()],
    "fc2.weight": [Replicate()],
    "fc2.output": [[Replicate()]],
}


class DMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config["hidden_size"], config["hidden_size"] * 4)
        self.gelu = torch.nn.GELU()
        self.fc2 = nn.Linear(config["hidden_size"] * 4, config["hidden_size"])

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x


class DModuleTestPlans(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    def _run_plan(self, param_sharding_plan, fwd_resharding_plan, devce_type):
        device_mesh = DeviceMesh(devce_type, list(range(self.world_size)))

        # create golden model (local replicate)
        mlp_golden = MLP(config)
        mlp_golden.to(devce_type)
        for name, param in mlp_golden.named_parameters():
            dist.all_reduce(param, async_op=False)

        # create dmodule (by plans)
        dmlp = DMLP(config)
        dmlp.to(devce_type)
        dmlp.load_state_dict(mlp_golden.state_dict())
        parallelize_module(dmlp, device_mesh, {"parameter": param_sharding_plan, "forward": fwd_resharding_plan})

        # create data (local replicate)
        input_golden = torch.randn(
            config["batch_size"] * config["seq_length"], config["hidden_size"], device=devce_type, requires_grad=False
        )
        dist.all_reduce(input_golden, async_op=False)
        input_tensor = input_golden.detach().clone()

        # match forward
        output_tensor = dmlp(input_tensor).to_local()
        output_golden = mlp_golden(input_golden)
        self.assertTrue(torch.allclose(output_tensor, output_golden, rtol=1e-4, atol=1e-5))

        # match backward
        output_tensor.sum().backward()
        output_golden.sum().backward()
        for n, p in dmlp.named_parameters():
            if n.endswith("bias") and any(place.is_partial() for place in p.placements):
                continue  # vescalized linear
            self.assertTrue(p.grad is not None)
            with torch.no_grad():
                grad_dmlp = p.grad.redistribute(placements=[Replicate()] * device_mesh.ndim).to_local()
            grad_golden = mlp_golden.get_parameter(n).grad
            self.assertTrue(torch.allclose(grad_dmlp, grad_golden, rtol=1e-4, atol=1e-5))

    @with_comms_device(device_type="cpu")
    def test_cpu(self):
        torch.manual_seed(42)
        self._run_plan(param_sharding_plan1, fwd_resharding_plan1, "cpu")
        self._run_plan(param_sharding_plan2, fwd_resharding_plan2, "cpu")

    @with_comms_device(device_type="cuda")
    def test_cuda(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        self._run_plan(param_sharding_plan1, fwd_resharding_plan1, "cuda")
        self._run_plan(param_sharding_plan2, fwd_resharding_plan2, "cuda")

    @with_comms_device(device_type="cuda")
    def test_wrong_plan(self):
        device_mesh = DeviceMesh("cuda", list(range(self.world_size)))
        # create dmodule (by plans)
        dmlp = DMLP(config)
        with self.assertRaises(KeyError):
            parallelize_module(dmlp, device_mesh, {"parameters": param_sharding_plan1, "forward": fwd_resharding_plan1})
        with self.assertRaises(KeyError):
            parallelize_module(dmlp, device_mesh, {"parameter": param_sharding_plan1, "forwards": fwd_resharding_plan1})


if __name__ == "__main__":
    run_tests()
