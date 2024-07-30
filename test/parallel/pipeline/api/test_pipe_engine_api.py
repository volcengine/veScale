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
from common_dtensor import DTensorTestBase, with_comms
import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests
from vescale.optim.base_optimizer import BasicOptimizer
from vescale.pipe.pipe_stage import PipeModule, construct_stage_modules
from vescale.devicemesh_api import VESCALE_DEVICE_MESH
from vescale.engine import PipeEngine
from vescale.plan import (
    PipelineParallelPlan,
    PipelineScheduleType,
    ModeType,
    PipelineSplitMethodType,
)


class MLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.fc1 = nn.Linear(n_features, n_features * 2, bias=False)
        torch.nn.init.uniform_(self.fc1.weight, 0, 1)
        self.fc2 = nn.Linear(n_features * 2, n_features)
        torch.nn.init.uniform_(self.fc2.weight, 0, 1)
        self.gelu = nn.GELU()

    def forward(self, x):
        t = self.fc1(x)
        t = self.gelu(t)
        t = self.fc2(t)
        return t


class FourMLP(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.mlp1 = MLP(hidden)
        self.mlp2 = MLP(hidden)
        self.mlp3 = MLP(hidden)
        self.mlp4 = MLP(hidden)
        self.sequence = nn.Sequential(self.mlp1, self.mlp2, self.mlp3, self.mlp4)

    def forward(self, x):
        return self.sequence(x)


class EightMLP(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.mlp1 = MLP(hidden)
        self.mlp2 = MLP(hidden)
        self.mlp3 = MLP(hidden)
        self.mlp4 = MLP(hidden)
        self.mlp5 = MLP(hidden)
        self.mlp6 = MLP(hidden)
        self.mlp7 = MLP(hidden)
        self.mlp8 = MLP(hidden)

    def forward(self, x):
        x = self.mlp1(x)
        x.retain_grad()
        x = self.mlp2(x)
        x.retain_grad()
        x = self.mlp3(x)
        x.retain_grad()
        x = self.mlp4(x)
        x.retain_grad()
        x = self.mlp5(x)
        x.retain_grad()
        x = self.mlp6(x)
        x.retain_grad()
        x = self.mlp7(x)
        x.retain_grad()
        x = self.mlp8(x)
        return x


class ScheduleTest(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @staticmethod
    def loss_fn(x):
        return torch.sum(x)

    def _prepare_runtime_engine(self, model, forward_only: bool = False):
        pipe_config = PipelineParallelPlan(
            mode=ModeType.GRAPH_EAGER,
            split_method=PipelineSplitMethodType.MANUAL,
            num_stages=4,
            virtual_chunks=1,
            smallest_unsplittable_units=["mlp1", "mlp2", "mlp3", "mlp4"],
            split_points=["mlp1", "mlp2", "mlp3", "mlp4"],
            batch_p2p_comm=False,
            overlap_p2p_comm=True,
            schedule_type=PipelineScheduleType.SIMPLE_1F1B,
            forward_only=forward_only,
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
        _parameters = list(stage_modules[0].parameters())
        optimizer = torch.optim.SGD(_parameters, **optimizer_fn_kwargs)
        basic_optimizer = BasicOptimizer(optimizer, models=stage_modules)
        pipe_module = PipeModule(stage_modules, optimizer, None, stage_dependency, p2p_index_mapping, pipe_config)
        engine = PipeEngine(
            pipe_module,
            VESCALE_DEVICE_MESH,
            self.loss_fn,
            pipe_config,
        )

        return engine, optimizer

    def _prepare_runtime_interleaved_engine(self, model, forward_only: bool = False):
        num_layer = 8
        pipe_config = PipelineParallelPlan(
            mode=ModeType.GRAPH_EAGER,
            split_method=PipelineSplitMethodType.MANUAL,
            num_stages=4,
            virtual_chunks=2,
            smallest_unsplittable_units=[f"mlp{i + 1}" for i in range(num_layer)],
            split_points=["mlp2", "mlp4", "mlp6", "mlp8"],
            batch_p2p_comm=True,
            overlap_p2p_comm=False,
            schedule_type=PipelineScheduleType.INTERLEAVED_1F1B,
            forward_only=forward_only,
        )

        VESCALE_DEVICE_MESH.init_device_mesh(
            device_type="cuda",
            mesh_shape=(4, 1, 1),
            mesh_dim_names=["PP", "DP", "TP"],
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
        return engine, optimizer

    @with_comms
    def test_runtime_engine(self):
        """
        Tests pipeline engine.
        """
        local_rank = self.rank
        device = f"cuda:{local_rank}"
        # must do this: https://pytorch.org/docs/stable/distributed.html
        torch.cuda.set_device(device)
        os.environ["LOCAL_RANK"] = str(local_rank)
        n_hidden = 3
        batches = 8
        model = FourMLP(n_hidden).cuda()

        all_batches_out = []
        if local_rank == 3:
            for i in range(batches):
                print(f" ===========batch: {i}================= ")
                data = torch.zeros(1, 1, n_hidden) + i
                data = data.float().cuda(3)
                model.cuda(3)
                out = model(data)
                loss = out.sum()
                all_batches_out.append(loss)
                loss.backward(create_graph=True)
                print(loss)
                print(" ====================================== ")

        engine, optimizer = self._prepare_runtime_engine(model)

        data_iterator = []
        for i in range(batches):
            data = torch.zeros(1, 1, n_hidden) + i
            data_iterator.append(data.to(device))

        minibatch_loss, _ = engine(data_iterator)

        if local_rank == 3:
            self.assertEqual(minibatch_loss, sum(all_batches_out))

    @with_comms
    def test_simple_inference_schedule(self):
        """
        Tests pipeline engine's inference mode.
        """
        local_rank = self.rank
        device = f"cuda:{local_rank}"
        # must do this: https://pytorch.org/docs/stable/distributed.html
        torch.cuda.set_device(device)
        os.environ["LOCAL_RANK"] = str(local_rank)
        n_hidden = 3
        batches = 8
        model = FourMLP(n_hidden).cuda()

        all_batches_out = []
        if local_rank == 3:
            for i in range(batches):
                print(f" ===========batch: {i}================= ")
                data = torch.zeros(1, 1, n_hidden) + i
                data = data.float().cuda(3)
                model.cuda(3)
                out = model(data)
                loss = out.sum()
                all_batches_out.append(loss)
                loss.backward(create_graph=True)
                print(loss)
                print(" ====================================== ")

        engine, optimizer = self._prepare_runtime_engine(model, forward_only=True)

        data_iterator = []
        for i in range(batches):
            data = torch.zeros(1, 1, n_hidden) + i
            data_iterator.append(data.to(device))

        minibatch_loss, _ = engine(data_iterator)

        if local_rank == 3:
            self.assertEqual(minibatch_loss, sum(all_batches_out))

    @with_comms
    def test_runtime_interleaved_1f1b_engine_batch(self):
        """
        Tests pipeline engine with interleaved 1f1b schedule under
        batch p2p communication.
        """
        global local_rank
        local_rank = self.rank
        device = f"cuda:{local_rank}"
        # must do this: https://pytorch.org/docs/stable/distributed.html
        torch.cuda.set_device(device)
        os.environ["LOCAL_RANK"] = str(local_rank)
        n_hidden = 3
        batches = 8
        model = EightMLP(n_hidden).cuda()
        single_model_data = []
        all_batches_out = []
        if local_rank == 3:
            true_model = model
            true_model = true_model.cuda()
            true_model.train()
            for i in range(batches):
                print(f" ===========batch: {i}================= ")
                data = torch.zeros(1, 1, n_hidden) + i % 8
                data = data.float().cuda(3)
                single_model_data.append(data)
                out = true_model(data)
                loss = out.sum()
                all_batches_out.append(loss)
                loss.backward(create_graph=True)
                print(" ====================================== ")

        pipe_engine, optimizer = self._prepare_runtime_interleaved_engine(model)

        data_iterator = []
        for j in range(batches):
            data = torch.zeros(1, 1, n_hidden) + j
            data_iterator.append(data.to(device))

        minibatch_loss, _ = pipe_engine(data_iterator)

        if local_rank == 3:
            ground_truth_loss = sum(all_batches_out)
            self.assertEqual(minibatch_loss, ground_truth_loss)

    @with_comms
    def test_runtime_interleaved_1f1b_engine_p2p(self):
        """
        Tests pipeline engine with interleaved 1f1b schedule under
        overlapped p2p communication.
        """
        global local_rank
        local_rank = self.rank
        device = f"cuda:{local_rank}"
        # must do this: https://pytorch.org/docs/stable/distributed.html
        torch.cuda.set_device(device)
        os.environ["LOCAL_RANK"] = str(local_rank)
        n_hidden = 3
        batches = 8
        model = EightMLP(n_hidden).cuda()
        single_model_data = []
        all_batches_out = []
        if local_rank == 3:
            true_model = model
            true_model.train()
            for i in range(batches):
                print(f" ===========batch: {i}================= ")
                data = torch.zeros(1, 1, n_hidden) + i % 8  # + i
                data = data.float().cuda(3)
                single_model_data.append(data)
                out = true_model(data)
                loss = out.sum()
                all_batches_out.append(loss)
                loss.backward(create_graph=True)
                print(" ====================================== ")

        num_layer = 8
        pipe_config = PipelineParallelPlan(
            mode=ModeType.GRAPH_EAGER,
            split_method=PipelineSplitMethodType.MANUAL,
            num_stages=4,
            virtual_chunks=2,
            smallest_unsplittable_units=[f"mlp{i + 1}" for i in range(num_layer)],
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
        basic_optimizer = BasicOptimizer(optimizer, models=stage_modules)
        pipe_module = PipeModule(stage_modules, optimizer, None, stage_dependency, p2p_index_mapping, pipe_config)
        engine = PipeEngine(
            pipe_module,
            VESCALE_DEVICE_MESH,
            self.loss_fn,
            pipe_config,
        )

        data_iterator = []
        for j in range(batches):
            data = torch.zeros(1, 1, n_hidden) + j
            data_iterator.append(data.to(device))

        minibatch_loss, _ = engine.forward_backward(data_iterator)

        if local_rank == 3:
            ground_truth_loss = sum(all_batches_out)
            self.assertEqual(minibatch_loss, ground_truth_loss)


if __name__ == "__main__":
    run_tests()
