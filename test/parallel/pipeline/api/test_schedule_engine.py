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

import os
import torch
from common_dtensor import DTensorTestBase, with_comms
from torch.testing._internal.common_utils import run_tests
from vescale.devicemesh_api import VESCALE_DEVICE_MESH
from vescale.pipe.pipe_stage import PipeModule, construct_stage_modules
from vescale.pipe._schedules.instruction_base import StageDeps
from vescale.pipe.pipe_emmiter import ScheduleEngine
from vescale.plan.spec import PipelineScheduleType, ModeType, PipelineSplitMethodType
from vescale.plan.pipeline_parallel import PipelineParallelPlan
from four_mlp import FourMLP
from torch.optim import SGD


class ScheduleEngineRuntimeTest(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    @staticmethod
    def loss_fn(x):
        return x.mean()

    def _setup(self):
        os.environ["model_name"] = "pp"
        global local_rank
        local_rank = self.rank
        device = f"cuda:{local_rank}"
        # must do this: https://pytorch.org/docs/stable/distributed.html
        torch.cuda.set_device(device)

        torch.manual_seed(9999)
        microbatch_size = 2
        factor = 4
        batch = [torch.ones(microbatch_size, 128, 1024, dtype=torch.float32).to(device) for _ in range(factor)]
        return batch, microbatch_size

    @with_comms
    def test_simple_1f1b(self):
        """
        Test simple 1f1b schedule with schedule runtime.
        """
        batch, microbatch_size = self._setup()

        VESCALE_DEVICE_MESH.init_device_mesh(
            device_type="cuda",
            mesh_shape=(2, 1, 1),
            mesh_dim_names=("PP", "DP", "TP"),
        )

        model = FourMLP(1024).cuda()
        num_layers = 4

        config = PipelineParallelPlan(
            mode=ModeType.GRAPH_EAGER,
            split_method=PipelineSplitMethodType.UNIFORM,
            num_stages=2,
            virtual_chunks=1,
            smallest_unsplittable_units=[f"mlp{i + 1}" for i in range(num_layers)],
            batch_p2p_comm=False,
            overlap_p2p_comm=True,
            schedule_type=PipelineScheduleType.SIMPLE_1F1B,
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
        _parameters = list(stage_modules[0].parameters())
        optimizer = SGD(_parameters, **optimizer_fn_kwargs)
        pipe_module = PipeModule(stage_modules, optimizer, None, stage_dependency, p2p_index_mapping, config)

        dep = pipe_module.stage_deps
        device_mesh_list = VESCALE_DEVICE_MESH.get_global_tensor_parallel_meshes()
        stage_deps = StageDeps(dep, device_mesh_list, pipe_module)

        pipe_engine = ScheduleEngine(
            stage_deps,
            meshes=VESCALE_DEVICE_MESH.get_global_tensor_parallel_meshes(),
            schedule=PipelineScheduleType.SIMPLE_1F1B,
            batches=len(batch),
            data_iterator=iter(batch),
            stage_id=VESCALE_DEVICE_MESH.get_pipeline_parallel_rank(),
            shape=(microbatch_size, 128, 1024),
            dtype=torch.float32,
        )
        minibatch_loss, all_forward_outputs = ScheduleEngine.execute(pipe_engine)


if __name__ == "__main__":
    run_tests()
