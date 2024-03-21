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

from vescale import DeviceMesh, DTensor, distribute_tensor
from vescale.dmodule.api import parallelize_module
from vescale.dtensor.placement_types import Partial, Replicate, Shard
from vescale.initialize.deferred_init import deferred_init

config = {"seq_length": 8, "head_size": 4, "hidden_size": 4 * 4, "n_head": 4, "batch_size": 4}


def replicate_input(input: torch.Tensor, device_mesh: DeviceMesh) -> DTensor:
    return DTensor.from_local(input, device_mesh, [Replicate()] * device_mesh.ndim)


def replicate_output(output: DTensor, device_mesh: DeviceMesh) -> DTensor:
    return output.redistribute(device_mesh, [Replicate()] * device_mesh.ndim)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.register_buffer("buf", torch.ones(config["hidden_size"], requires_grad=False))
        self.fc1 = nn.Linear(config["hidden_size"], config["hidden_size"] * 4, bias=True)
        self.gelu = torch.nn.GELU()
        self.fc2 = nn.Linear(config["hidden_size"] * 4, config["hidden_size"], bias=False)

    def forward(self, x):
        x = x + self.buf
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x

    def reset_parameters(self, *args, **kwargs):
        with torch.no_grad():
            self.buf.fill_(1)
            self.fc1.weight.fill_(0.5)
            self.fc1.bias.fill_(1.5)
            self.fc2.weight.fill_(1)

    def parallelize(self, device_mesh: DeviceMesh):
        self.register_buffer("buf", distribute_tensor(self.buf, device_mesh, [Replicate()]))

        for name, param in self.fc1.named_parameters():
            placements = [Shard(0)]
            dist_param = nn.Parameter(distribute_tensor(param.data, device_mesh, placements))
            self.fc1.register_parameter(name, dist_param)

        for name, param in self.fc2.named_parameters():
            placements = [Shard(1)] if name == "weight" else [Partial()]
            dist_param = nn.Parameter(distribute_tensor(param.data, device_mesh, placements))
            self.fc2.register_parameter(name, dist_param)

        return self

    def parallel_forward(self, x: torch.Tensor, device_mesh: DeviceMesh):
        x = replicate_input(x, device_mesh)
        x = self.forward(x)
        return replicate_output(x, device_mesh)


param_sharding_plan = {
    "buf": [Replicate()],  # should work
    "fc1.weight": [Shard(0)],
    "fc1.bias": [Shard(0)],
    "fc2.weight": [Shard(1)],
}

fwd_resharding_plan = {
    "input": [[Replicate()]],
    "output": [[Replicate()]],
}


class DMLP(nn.Module):
    def __init__(self, config, is_sharded=False, world_size=4):
        super().__init__()
        self.register_buffer("buf", torch.ones(config["hidden_size"], requires_grad=False))
        self.fc1 = nn.Linear(
            config["hidden_size"],
            config["hidden_size"] * 4 if not is_sharded else config["hidden_size"] * 4 // world_size,
            bias=True,
        )
        self.gelu = torch.nn.GELU()
        self.fc2 = nn.Linear(
            config["hidden_size"] * 4 if not is_sharded else config["hidden_size"] * 4 // world_size,
            config["hidden_size"],
            bias=False,
        )

    def forward(self, x):
        x = x + self.buf
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x

    def reset_parameters(self, *args, **kwargs):
        with torch.no_grad():
            self.buf.fill_(1)
            self.fc1.weight.fill_(0.5)
            self.fc1.bias.fill_(1.5)
            self.fc2.weight.fill_(1)


class DModuleTestInit(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    def __compare(self, dmlp, optim, mlp_golden, optim_golden, device_mesh):
        # create data
        input_golden = torch.randn(
            config["batch_size"] * config["seq_length"],
            config["hidden_size"],
            device=device_mesh.device_type,
            requires_grad=False,
        )
        dist.all_reduce(input_golden, async_op=False)
        input_tensor = input_golden.detach().clone()

        # match forward
        output_tensor = dmlp(input_tensor).to_local()
        output_golden = mlp_golden.parallel_forward(input_golden, device_mesh).to_local()
        self.assertTrue(torch.allclose(output_tensor, output_golden))

        # match backward
        output_tensor.sum().backward()
        output_golden.sum().backward()
        for (n, p), (n_golden, p_golden) in zip(dmlp.named_parameters(), mlp_golden.named_parameters()):
            self.assertEqual(n, n_golden)
            self.assertTrue(isinstance(p.grad, DTensor) and isinstance(p_golden.grad, DTensor))
            self.assertTrue(torch.allclose(p.grad.to_local(), p_golden.grad.to_local()))

        # match optimizer step
        optim.step()
        optim_golden.step()
        for (n, p), (n_golden, p_golden) in zip(dmlp.named_parameters(), mlp_golden.named_parameters()):
            self.assertEqual(n, n_golden)
            self.assertTrue(torch.allclose(p.to_local(), p_golden.to_local()))

        # match next forward
        input_golden = torch.randn(
            config["batch_size"] * config["seq_length"],
            config["hidden_size"],
            device=device_mesh.device_type,
            requires_grad=False,
        )
        dist.all_reduce(input_golden, async_op=False)
        input_tensor = input_golden.detach().clone()

        output_tensor = dmlp(input_tensor).to_local()
        output_golden = mlp_golden.parallel_forward(input_golden, device_mesh).to_local()
        self.assertTrue(torch.allclose(output_tensor, output_golden))

    def _run_parallelize_not_meta_not_sharded(self, device_type):
        device_mesh = DeviceMesh(device_type, list(range(self.world_size)))

        # create golden model
        mlp_golden = MLP(config)
        mlp_golden.to(device_type)
        mlp_golden.reset_parameters()
        mlp_golden.parallelize(device_mesh)
        optim_golden = torch.optim.SGD(mlp_golden.parameters(), lr=0.1)

        # parallelize dmodule
        with torch.device("cpu"):
            dmlp = DMLP(config)
        sharding_plan = {"parameter": param_sharding_plan, "forward": fwd_resharding_plan}

        parallelize_module(dmlp, device_mesh, sharding_plan)
        dmlp.reset_parameters()
        optim = torch.optim.SGD(dmlp.parameters(), lr=0.1)

        # compare both
        self.__compare(dmlp, optim, mlp_golden, optim_golden, device_mesh)

    def _run_parallelize_not_meta_sharded(self, device_type):
        device_mesh = DeviceMesh(device_type, list(range(self.world_size)))

        # create golden model
        mlp_golden = MLP(config)
        mlp_golden.to(device_type)
        mlp_golden.reset_parameters()
        mlp_golden.parallelize(device_mesh)
        optim_golden = torch.optim.SGD(mlp_golden.parameters(), lr=0.1)

        # parallelize dmodule
        with torch.device("cpu"):
            dmlp = DMLP(config, is_sharded=True, world_size=self.world_size)

        sharding_plan = {"parameter": param_sharding_plan, "forward": fwd_resharding_plan}

        parallelize_module(dmlp, device_mesh, sharding_plan, is_model_sharded=True)
        dmlp.reset_parameters()
        optim = torch.optim.SGD(dmlp.parameters(), lr=0.1)

        # compare both
        self.__compare(dmlp, optim, mlp_golden, optim_golden, device_mesh)

    def _run_parallelize_meta_not_sharded(self, device_type):
        device_mesh = DeviceMesh(device_type, list(range(self.world_size)))

        # create golden model
        mlp_golden = MLP(config)
        mlp_golden.to(device_type)
        mlp_golden.reset_parameters()
        mlp_golden.parallelize(device_mesh)
        optim_golden = torch.optim.SGD(mlp_golden.parameters(), lr=0.1)

        sharding_plan = {"parameter": param_sharding_plan, "forward": fwd_resharding_plan}

        # parallelize dmodule
        dmlp = deferred_init(DMLP, config)
        parallelize_module(dmlp, device_mesh, sharding_plan)
        dmlp.reset_parameters()
        optim = torch.optim.SGD(dmlp.parameters(), lr=0.1)

        # compare both
        self.__compare(dmlp, optim, mlp_golden, optim_golden, device_mesh)

    @with_comms_device(device_type="cpu")
    def test_initialize_cpu(self):
        self._run_parallelize_not_meta_not_sharded("cpu")
        self._run_parallelize_not_meta_sharded("cpu")
        self._run_parallelize_meta_not_sharded("cpu")

    @with_comms_device(device_type="cuda")
    def test_initialize_cuda(self):
        self._run_parallelize_not_meta_not_sharded("cuda")
        self._run_parallelize_not_meta_sharded("cuda")
        self._run_parallelize_meta_not_sharded("cuda")


if __name__ == "__main__":
    run_tests()
