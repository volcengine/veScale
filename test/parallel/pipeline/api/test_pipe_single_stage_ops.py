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
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests
from vescale.devicemesh_api import VESCALE_DEVICE_MESH
from vescale.plan import PipelineScheduleType, PipelineParallelPlan, ModeType, PipelineSplitMethodType
from vescale.pipe.pipe_stage import PipeModule, construct_stage_modules
from vescale.engine import PipeEngine
from common_dtensor import DTensorTestBase, with_comms
from torch.optim import SGD

microbatch_size = 16
factor = 8
batch_size = microbatch_size * factor
RANDOM_SEED = 9999


class MLP(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value
        self.counter = 0
        self.fc1 = nn.Linear(32, 32, bias=False)
        self.fc1.weight.data.fill_(value)
        self.fc2 = nn.Linear(32, 32, bias=False)
        self.fc2.weight.data.fill_(value * 2)
        self.gelu = nn.GELU()

    def forward(self, x):
        t = self.fc1(x)
        t = self.gelu(t)
        t = self.fc2(t)
        torch.save(t, f"{os.environ['model_name']}_mlp{self.value}_fwd{self.counter}_out_tensor.pt")
        self.counter += 1
        return t


class MLPWithForwardUtil(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value
        self.counter = 0
        self.fc1 = nn.Linear(32, 32, bias=False)
        self.fc1.weight.data.fill_(value)
        self.fc2 = nn.Linear(32, 32, bias=False)
        self.fc2.weight.data.fill_(value * 2)
        self.gelu = nn.GELU()

    def forward(self, x):
        t = self.fc1(x)
        t = self.gelu(t)
        t = self.fc2(t)
        torch.save(t, f"{os.environ['model_name']}_mlp{self.value}_fwd{self.counter}_out_tensor.pt")
        self.counter += 1
        return t

    def forward_util(self, p2p_input, local_input=None):
        print("This is an auxilary forward_util() provided by the user")
        if p2p_input is not None:
            print("Modified p2p_input value!")
            p2p_input *= 2
        else:
            print("Load local input as p2p input")
            p2p_input = local_input
        if local_input is not None:
            print("Handling local inputs")
        return [p2p_input]


class EightMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp1 = MLPWithForwardUtil(0)
        self.mlp2 = MLP(1)
        self.mlp3 = MLP(2)
        self.mlp4 = MLP(3)
        self.mlp5 = MLPWithForwardUtil(3)
        self.mlp6 = MLP(3)
        self.mlp7 = MLP(3)
        self.mlp8 = MLP(3)
        self.sequence = nn.Sequential(
            self.mlp1,
            self.mlp2,
            self.mlp3,
            self.mlp4,
            self.mlp5,
            self.mlp6,
            self.mlp7,
            self.mlp8,
        )

    def forward(self, x):
        return self.sequence(x)


class PipelineSingleStageOpsTest(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @staticmethod
    def loss_fn(x):
        return x.mean()

    def test_stage_forward(self):
        """
        Test single stage forward.
        """
        if self.rank == 0:
            self._run_no_pp_model()
        n_gpus = torch.cuda.device_count()
        assert n_gpus >= 2, "Requires at least 2 GPUs to run model with pp engine"
        self._run_stage_forward()

    def _run_no_pp_model(self):
        os.environ["model_name"] = "golden"
        model = EightMLP().to("cuda:0")
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.01, momentum=0, dampening=0, weight_decay=0, nesterov=False
        )
        torch.manual_seed(9999)
        batch = [torch.ones(microbatch_size, 128, 32, dtype=torch.float32).to("cuda:0") for _ in range(factor)]
        for mb in batch:
            out = model(mb)

    @with_comms
    def _run_stage_forward(self):
        os.environ["model_name"] = "pp"
        device = f"cuda:{self.rank}"
        torch.cuda.set_device(device)
        model = EightMLP().cuda()

        num_layers = 8
        config = PipelineParallelPlan(
            mode=ModeType.GRAPH_EAGER,
            split_method=PipelineSplitMethodType.MANUAL,
            num_stages=4,
            virtual_chunks=2,
            smallest_unsplittable_units=[f"mlp{i + 1}" for i in range(num_layers)],
            split_points=["mlp2", "mlp4", "mlp6", "mlp8"],
            batch_p2p_comm=False,
            overlap_p2p_comm=True,
            schedule_type=PipelineScheduleType.INTERLEAVED_1F1B,
        )

        VESCALE_DEVICE_MESH.init_device_mesh(
            device_type="cuda",
            mesh_shape=(4, 1, 1),
            mesh_dim_names=("PP", "DP", "TP"),
        )

        stage_modules, stage_dependency, p2p_index_mapping = construct_stage_modules(
            model,
            config,
            VESCALE_DEVICE_MESH,
            update_split_points=True,
        )

        optimizer_fn_kwargs = {
            "lr": 0.01,
            "momentum": 0,
            "dampening": 0,
            "weight_decay": 0,
            "nesterov": False,
            "maximize": False,
            "foreach": None,
            "differentiable": False,
        }
        _parameters = list(stage_modules[0].parameters()) + list(stage_modules[1].parameters())
        optimizer = SGD(_parameters, **optimizer_fn_kwargs)
        pipe_module = PipeModule(stage_modules, optimizer, None, stage_dependency, p2p_index_mapping, config)

        engine = PipeEngine(
            pipe_module,
            VESCALE_DEVICE_MESH,
            self.loss_fn,
            config,
        )
        torch.manual_seed(9999)
        batch = [torch.ones(microbatch_size, 128, 32, dtype=torch.float32).to(device) for _ in range(factor)]
        if self.rank == 0:
            # first stage only receives inputs from dataloader
            chunk_id = 0
            print(f"Chunk ID: {chunk_id}")
            output_chunk_one = engine.module(None, local_inputs=batch[0], chunk_id=chunk_id)
            chunk_id = 1
            print(f"Chunk ID: {chunk_id}")
            output_chunk_two = engine.module(batch[1], local_inputs=None, chunk_id=chunk_id)
            assert not torch.equal(output_chunk_one, output_chunk_two)
        if self.rank == 2:
            # other stages can receive inputs communicated by their peers
            chunk_id = 0
            print(f"Chunk ID: {chunk_id}")
            output_chunk_three = engine.module(batch[2], local_inputs=None, chunk_id=chunk_id)
            chunk_id = 1
            print(f"Chunk ID: {chunk_id}")
            output_chunk_four = engine.module(batch[3], local_inputs=None, chunk_id=chunk_id)
            assert not torch.equal(output_chunk_three, output_chunk_four)


if __name__ == "__main__":
    run_tests()
