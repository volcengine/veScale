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
from vescale.plan import (
    PipelineParallelPlan,
    PipelineScheduleType,
    ModeType,
    PipelineSplitMethodType,
)
from vescale.pipe import PipeModule, construct_stage_modules
from vescale.engine import PipeEngine
from common_dtensor import DTensorTestBase, with_comms
from vescale.devicemesh_api import VESCALE_DEVICE_MESH

microbatch_size = 2
factor = 8
batch_size = microbatch_size * factor
stage = 4
RANDOM_SEED = 9999


class MLP(nn.Module):
    def __init__(self, features_in, feature_middle, features_out, value, idx=1):
        super().__init__()
        self.value = value
        self.idx = idx
        self.counter = 0
        self.fc1 = nn.Linear(features_in, feature_middle, bias=False)
        self.fc2 = nn.Linear(feature_middle, features_out, bias=False)
        self.gelu = nn.GELU()

    def forward(self, x):
        t = self.fc1(x)
        t = self.gelu(t)
        t = self.fc2(t)
        torch.save(t, f"{os.environ['model_name']}_mlp{self.value}_fwd{self.counter}_out_tensor.pt")
        self.counter += 1
        return t


class EightMLP(nn.Module):
    def __init__(self, hidden=1024, fixed_size=True):
        super().__init__()
        self.mlp1 = MLP(hidden, hidden, hidden, 1, 1)
        self.mlp2 = MLP(hidden, hidden, hidden, 2, 2)
        self.mlp3 = MLP(hidden, hidden, hidden, 1, 3)
        self.mlp4 = MLP(hidden, hidden, hidden, 2, 4)
        self.mlp5 = MLP(hidden, hidden, hidden, 1, 5)
        self.mlp6 = MLP(hidden, hidden, hidden, 2, 6)
        self.mlp7 = MLP(hidden, hidden, hidden, 1, 7)
        self.mlp8 = MLP(hidden, hidden, hidden, 2, 8)

    def forward(self, x):
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = self.mlp4(x)
        x = self.mlp5(x)
        x = self.mlp6(x)
        x = self.mlp7(x)
        x = self.mlp8(x)
        return x


class PipelineAccuracyAlignmentTest(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @staticmethod
    def loss_fn(x):
        return x.mean()

    @staticmethod
    def save_mlp_parameter(model: MLP, f_name):
        torch.save(model.fc1.weight, f"{f_name}.fc1")
        torch.save(model.fc2.weight, f"{f_name}.fc2")

    @staticmethod
    def load_mlp_parameter(f_prefix):
        fc1_weight = torch.load(f"{f_prefix}.fc1").to("cuda:0")
        fc2_weight = torch.load(f"{f_prefix}.fc2").to("cuda:0")
        return (fc1_weight, fc2_weight)

    def check_model_weight_diff(self, f_prefix):
        def helper(f1, f2):
            golden_weights = self.load_mlp_parameter(f1)
            pp_weights = self.load_mlp_parameter(f2)
            torch.testing.assert_close(golden_weights[0], pp_weights[0])
            torch.testing.assert_close(golden_weights[1], pp_weights[1])

        helper(f"golden_mlp{self.rank + 1}", f"{f_prefix}_mlp{self.rank + 1}")

    def check_out_tensors(self, model_name):
        def helper(f1, f2):
            golden_out = torch.load(f1).to("cuda:0")
            pp_out = torch.load(f2).to("cuda:0")
            torch.testing.assert_close(golden_out, pp_out)

        for i in range(1, 3):
            for j in range(8):
                helper(f"golden_mlp{i}_fwd{j}_out_tensor.pt", f"{model_name}_mlp{i}_fwd{j}_out_tensor.pt")
                torch.cuda.synchronize()

    def test_accuracy_alignment(self, fixed_size=True):
        """
        Tests alignment of updated parameter and output activations of single device model and
        the model partitioned into four stages with pipeline parallelism API.
        """
        if self.rank == 0:
            self._run_no_pp_model(fixed_size=fixed_size)
        torch.cuda.synchronize()
        n_gpus = torch.cuda.device_count()
        assert n_gpus >= 2, "Requires at least 2 GPUs to run model with pp engine"
        self._run_engine_with_1f1b(fixed_size=fixed_size)
        if self.rank == 0:
            self.check_out_tensors("pp")
        self.check_model_weight_diff("engine_1f1b")

    def _run_no_pp_model(self, fixed_size=True):
        os.environ["model_name"] = "golden"
        model = EightMLP(16, fixed_size=fixed_size).to("cuda:0")
        torch.save(model.state_dict(), "baseline_model.pt")
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0,
            dampening=0,
            weight_decay=0,
            nesterov=False,
            maximize=False,
            foreach=None,
            differentiable=False,
        )
        torch.manual_seed(9999)
        batch = [torch.ones(microbatch_size, 128, 16, dtype=torch.float32).to("cuda:0") for _ in range(factor)]
        for mb in batch:
            out = model(mb)
        loss = self.loss_fn(out)
        loss.backward()
        optimizer.step()
        torch.save(out, "golden_out.pt")
        torch.save(loss, "golden_loss.pt")
        self.save_mlp_parameter(model.mlp1, "golden_mlp1")
        self.save_mlp_parameter(model.mlp2, "golden_mlp2")
        self.save_mlp_parameter(model.mlp3, "golden_mlp3")
        self.save_mlp_parameter(model.mlp4, "golden_mlp4")
        self.save_mlp_parameter(model.mlp5, "golden_mlp5")
        self.save_mlp_parameter(model.mlp6, "golden_mlp6")
        self.save_mlp_parameter(model.mlp7, "golden_mlp7")
        self.save_mlp_parameter(model.mlp8, "golden_mlp8")

    @with_comms
    def _run_engine_with_1f1b(self, fixed_size=True):
        os.environ["model_name"] = "pp"
        device = f"cuda:{self.rank}"
        # must do this: https://pytorch.org/docs/stable/distributed.html
        torch.cuda.set_device(device)
        model = EightMLP(16, fixed_size=fixed_size).cuda()
        model.load_state_dict(torch.load("baseline_model.pt"))

        pipe_config = PipelineParallelPlan(
            mode=ModeType.GRAPH_EAGER,
            split_method=PipelineSplitMethodType.MANUAL,
            num_stages=4,
            virtual_chunks=2,
            smallest_unsplittable_units=["mlp1", "mlp2", "mlp3", "mlp4", "mlp5", "mlp6", "mlp7", "mlp8"],
            split_points=["mlp2", "mlp4", "mlp6", "mlp8"],
            batch_p2p_comm=False,
            overlap_p2p_comm=True,
            schedule_type=PipelineScheduleType.INTERLEAVED_1F1B,
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

        torch.manual_seed(9999)
        with torch.no_grad():
            batch = [torch.ones(microbatch_size, 128, 16, dtype=torch.float32).to(device) for _ in range(factor)]

        VESCALE_DEVICE_MESH.init_device_mesh(
            device_type="cuda",
            mesh_shape=(4, 1, 1),
            mesh_dim_names=["PP", "DP", "TP"],
        )
        stage_modules, stage_dependency, p2p_index_mapping = construct_stage_modules(
            model,
            pipe_config,
            VESCALE_DEVICE_MESH,
            update_split_points=True,
        )
        _parameters = list(stage_modules[0].parameters()) + list(stage_modules[1].parameters())
        optimizer = torch.optim.SGD(_parameters, **optimizer_fn_kwargs)
        pipe_module = PipeModule(stage_modules, optimizer, None, stage_dependency, p2p_index_mapping, pipe_config)
        engine = PipeEngine(
            pipe_module,
            VESCALE_DEVICE_MESH,
            self.loss_fn,
            pipe_config,
        )

        engine(batch)
        optimizer = engine.get_optimizer
        optimizer.step()

        if self.rank == 0:
            self.save_mlp_parameter(engine.module[0].get_submodule("mlp1"), "engine_1f1b_mlp1")
            self.save_mlp_parameter(engine.module[1].get_submodule("mlp5"), "engine_1f1b_mlp5")
        if self.rank == 1:
            self.save_mlp_parameter(engine.module[0].get_submodule("mlp2"), "engine_1f1b_mlp2")
            self.save_mlp_parameter(engine.module[1].get_submodule("mlp6"), "engine_1f1b_mlp6")
        if self.rank == 2:
            self.save_mlp_parameter(engine.module[0].get_submodule("mlp3"), "engine_1f1b_mlp3")
            self.save_mlp_parameter(engine.module[1].get_submodule("mlp7"), "engine_1f1b_mlp7")
        if self.rank == 3:
            self.save_mlp_parameter(engine.module[0].get_submodule("mlp4"), "engine_1f1b_mlp4")
            self.save_mlp_parameter(engine.module[1].get_submodule("mlp8"), "engine_1f1b_mlp8")


if __name__ == "__main__":
    run_tests()
