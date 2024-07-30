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

import torch
import numpy as np
import torch.fx as fx
import re
from torch.testing._internal.common_utils import run_tests
from common_dtensor import DTensorTestBase, with_comms
from vescale.pipe import PipeModule, construct_stage_modules, construct_pipeline_split_graph
from vescale.plan import (
    PipelineParallelPlan,
    PipelineScheduleType,
    PipelineSplitMethodType,
    ModeType,
    TracerType,
)
from vescale.initialize.deferred_init import deferred_init, is_deferred
from eight_mlp import EightMLP, sharding_plan, sharding_plan_fc
from vescale.dmodule._dmodule import DModule
from vescale.dmodule.api import parallelize_module
from vescale.devicemesh_api import VESCALE_DEVICE_MESH
import torch.distributed as dist
from vescale.ddp.distributed_data_parallel import DistributedDataParallel as DDP
from vescale.optim.distributed_optimizer import DistributedOptimizer
from vescale.dtensor.api import distribute_tensor
from vescale.dtensor.placement_types import Replicate
from torch.fx.passes.split_utils import split_by_tags


class PipeModuleTest(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @staticmethod
    def loss_fn(x):
        return x.mean()

    def _setup(self, pp_size: int = 2, dp_size: int = 1, tp_size: int = 2, virtual_chunks: int = 1):
        num_layers = 8
        VESCALE_DEVICE_MESH.init_device_mesh("cuda", (pp_size, dp_size, tp_size), mesh_dim_names=("PP", "DP", "TP"))
        deferred_mlp = deferred_init(EightMLP, hidden=8)
        pipe_config = PipelineParallelPlan(
            mode=ModeType.GRAPH_EAGER,
            split_method=PipelineSplitMethodType.UNIFORM,
            num_stages=2,
            virtual_chunks=virtual_chunks,
            smallest_unsplittable_units=[f"mlp{i + 1}" for i in range(num_layers)],
            batch_p2p_comm=False,
            overlap_p2p_comm=True,
            schedule_type=PipelineScheduleType.SIMPLE_1F1B
            if virtual_chunks == 1
            else PipelineScheduleType.INTERLEAVED_1F1B,
        )
        return deferred_mlp, pipe_config

    @with_comms
    def test_generate_stage_dependency(self):
        """
        Tests PipeModule's ability to generate inter-stage dependency.
        """
        deferred_mlp, config = self._setup()
        num_stages = 2

        _, stage_dependency, p2p_index_mapping = construct_stage_modules(
            deferred_mlp, config, VESCALE_DEVICE_MESH, update_split_points=True
        )

        target_deps = np.zeros((num_stages, num_stages))
        target_deps[0, 1] = 1
        target_p2p_mapping = {0: [(0, 0)], 1: [(0, 0)]}
        self.assertEqual(stage_dependency, target_deps)
        flattened_index_mapping = {
            i: [(spec[0].peer_stage_idx, spec[0].peer_output_idx)] for i, spec in p2p_index_mapping.items()
        }
        self.assertEqual(flattened_index_mapping, target_p2p_mapping)

    @with_comms
    def test_generate_stage_dependency_four_stages(self):
        """
        Tests PipeModule's ability to generate inter-stage dependency among four pipeline stages.
        """
        deferred_mlp, config = self._setup(pp_size=4, dp_size=1, tp_size=1, virtual_chunks=1)
        num_stages = 4
        config.num_stages = num_stages

        _, stage_dependency, p2p_index_mapping = construct_stage_modules(
            deferred_mlp, config, VESCALE_DEVICE_MESH, update_split_points=True
        )

        target_deps = np.zeros((num_stages, num_stages))
        target_deps[0, 1] = 1
        target_deps[1, 2] = 1
        target_deps[2, 3] = 1
        target_p2p_mapping = {0: [(0, 0)], 1: [(0, 0)], 2: [(1, 0)], 3: [(2, 0)]}
        self.assertEqual(stage_dependency, target_deps)
        flattened_index_mapping = {
            i: [(spec[0].peer_stage_idx, spec[0].peer_output_idx)] for i, spec in p2p_index_mapping.items()
        }
        self.assertEqual(flattened_index_mapping, target_p2p_mapping)

    @with_comms
    def test_forward(self):
        """
        Tests PipeModule's forward function.
        """
        deferred_mlp, _ = self._setup(virtual_chunks=2)
        num_layers = 8
        pipe_config = PipelineParallelPlan(
            mode=ModeType.GRAPH_EAGER,
            split_method=PipelineSplitMethodType.UNIFORM,
            num_stages=4,
            virtual_chunks=2,
            smallest_unsplittable_units=[f"mlp{i + 1}" for i in range(num_layers)],
            batch_p2p_comm=False,
            overlap_p2p_comm=True,
            schedule_type=PipelineScheduleType.SIMPLE_1F1B,
        )

        VESCALE_DEVICE_MESH.init_device_mesh(
            device_type="cuda",
            mesh_shape=(4, 1, 1),
            mesh_dim_names=["PP", "DP", "TP"],
        )
        tp_mesh = VESCALE_DEVICE_MESH["TP"]

        stage_modules, stage_dependency, p2p_index_mapping = construct_stage_modules(
            deferred_mlp,
            pipe_config,
            VESCALE_DEVICE_MESH,
            update_split_points=True,
        )
        for i in range(len(stage_modules)):
            parallelized_module = parallelize_module(
                stage_modules[i],
                tp_mesh,
                sharding_plan,
                factory=False,
            )
            stage_modules[i] = parallelized_module

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
        optimizer = torch.optim.SGD(_parameters, **optimizer_fn_kwargs)
        pipe_module = PipeModule(stage_modules, optimizer, None, stage_dependency, p2p_index_mapping, pipe_config)

        model_chunk_one = pipe_module[0]
        model_chunk_two = pipe_module[1]
        assert DModule.is_dmodule(pipe_module.stage_modules[0])
        assert DModule.is_dmodule(pipe_module.stage_modules[1])
        input = torch.randn((3, 8))
        out_chunk_one = pipe_module(input, chunk_id=0)
        out_chunk_two = pipe_module(input, chunk_id=1)
        assert torch.equal(out_chunk_one, model_chunk_one(input))
        assert torch.equal(out_chunk_two, model_chunk_two(input))


class PipeModuleTraceTest(DTensorTestBase):
    @with_comms
    def test_compile_mode(self):
        """
        Tests correctness of registering hooks on partitioned model graphs.
        """
        model = EightMLP(8)

        def hook(sel, args):
            print(f"{torch.distributed.get_rank()}: call hook")
            return args

        graph = fx.symbolic_trace(model)
        input = torch.randn((3, 8))
        rule = r"mlp\d+.*"
        for node in graph.graph.nodes:
            if re.match(rule, node.name):
                if int(node.name[3]) <= 4:
                    node.tag = "stage0"
                else:
                    node.tag = "stage1"
        global_graph = split_by_tags(graph, ["stage0", "stage1"])
        splited_module = global_graph.get_submodule("stage0")
        splited_module.mlp1.fc1.register_forward_pre_hook(hook)
        splited_module.mlp1.gelu.register_forward_pre_hook(hook)
        splited_module.mlp1.fc2.register_forward_pre_hook(hook)
        splited_module.mlp2.fc1.register_forward_pre_hook(hook)
        splited_module.mlp2.gelu.register_forward_pre_hook(hook)
        splited_module.mlp2.fc2.register_forward_pre_hook(hook)
        splited_module.mlp3.fc1.register_forward_pre_hook(hook)
        splited_module.mlp3.gelu.register_forward_pre_hook(hook)
        splited_module.mlp3.fc2.register_forward_pre_hook(hook)
        splited_module.mlp4.fc1.register_forward_pre_hook(hook)
        splited_module.mlp4.gelu.register_forward_pre_hook(hook)
        splited_module.mlp4.fc2.register_forward_pre_hook(hook)
        splited_module(input)

    @with_comms
    def test_compile_equivalent(self):
        """
        Tests correctness of registering hooks on partitioned model graphs.
        """
        model = EightMLP(8)

        def hook(sel, args):
            print(f"{torch.distributed.get_rank()}: call hook")
            return args

        graph = fx.symbolic_trace(model)
        input = torch.randn((3, 8))
        rule = r"mlp\d+.*"
        for node in graph.graph.nodes:
            if re.match(rule, node.name):
                if int(node.name[3]) <= 4:
                    node.tag = "stage0"
                else:
                    node.tag = "stage1"
        global_graph = split_by_tags(graph, ["stage0", "stage1"])
        splited_module = global_graph.get_submodule("stage0")
        call_modules_fqns = [node.target for node in splited_module.graph.nodes if node.op == "call_module"]
        for submodule_path in call_modules_fqns:
            splited_module.get_submodule(submodule_path).register_forward_pre_hook(hook)
        splited_module(input)

    @with_comms
    def test_decomposable_5d_parallelization(self):
        """
        Tests decomposable API of writing 5D parallelization from plan to parallelization.
        """
        # build device mesh
        device_mesh = VESCALE_DEVICE_MESH.init_device_mesh(
            device_type="cuda", mesh_shape=(2, 1, 2), mesh_dim_names=["PP", "DP", "TP"]
        )
        # deferred init mlp module
        deferred_mlp = deferred_init(EightMLP, hidden=8)
        # pipe module config
        boundaries = ["mlp4", "mlp8"]
        num_layers = 8
        pipe_config = PipelineParallelPlan(
            num_stages=2,
            split_method=PipelineSplitMethodType.MANUAL,
            smallest_unsplittable_units=[f"mlp{i + 1}" for i in range(num_layers)],
            split_points=boundaries,
            tracer_type=TracerType.TORCH_FX,
            tracer_kwargs={"shard_plan": sharding_plan},
        )
        split_graph = construct_pipeline_split_graph(deferred_mlp, pipe_config, update_split_points=True)

        # parallelize and materialize module
        model_chunks = []
        for i in range(pipe_config.num_stages):
            stage = getattr(split_graph, f"stage{i}")
            stage = parallelize_module(
                stage, VESCALE_DEVICE_MESH.get_tensor_parallel_mesh(), sharding_plan, factory=False
            )
            assert not is_deferred(stage)
            model_chunks.append(stage)
        if dist.get_rank() == 0:
            assert model_chunks[0].mlp1.fc1.weight._spec.placements[0].is_shard()

        # make ddp module
        ddp_models = []
        for model_chunk in model_chunks:
            ddp_models.append(
                DDP(
                    model_chunk,
                    VESCALE_DEVICE_MESH.get_data_parallel_mesh(),
                    accumulate_allreduce_grads_in_fp32=True,
                    overlap_grad_reduce=True,
                    use_distributed_optimizer=True,
                )
            )

        if dist.get_rank() == 0:
            assert model_chunks[0].mlp1.fc1.weight._spec.placements[0].is_shard()

        # make optimizer
        doptim = DistributedOptimizer(
            torch.optim.Adam(split_graph.parameters(), lr=0.01),
            models=ddp_models,
            overlap_param_gather=False,
        )
        tp_mesh = VESCALE_DEVICE_MESH.get_tensor_parallel_mesh()
        stage_id = VESCALE_DEVICE_MESH.get_pipeline_parallel_rank()

        num_layers = 8
        dataloader = [distribute_tensor(torch.zeros((5, 8)), tp_mesh, [Replicate()]) * i for i in range(num_layers)]
        for sample in dataloader:
            doptim.zero_grad()
            output = ddp_models[stage_id](sample)
            loss = output.mean()
            loss.backward()
            doptim.step()

    @with_comms
    def test_manual_split_various_boundary_level(self):
        """
        Tests PipeModule's ability to split stage by boundaries of various depths.
        """
        VESCALE_DEVICE_MESH.init_device_mesh("cuda", (2, 1, 2), mesh_dim_names=("PP", "DP", "TP"))
        deferred_mlp = deferred_init(EightMLP, hidden=8)
        pipe_config = PipelineParallelPlan(
            num_stages=2,
            split_method=PipelineSplitMethodType.MANUAL,
            smallest_unsplittable_units=["mlp7", "mlp8"],
            split_points=["mlp4.fc1", "mlp8"],
            tracer_type=TracerType.TORCH_FX,
            tracer_kwargs={"partition_units": ["mlp7", "mlp8"]},
        )

        split_graph = construct_pipeline_split_graph(deferred_mlp, pipe_config, update_split_points=True)
        for i in range(pipe_config.num_stages):
            stage = getattr(split_graph, f"stage{i}")
            stage = parallelize_module(
                stage, VESCALE_DEVICE_MESH.get_tensor_parallel_mesh(), sharding_plan_fc, factory=False
            )
            assert not is_deferred(stage)


if __name__ == "__main__":
    run_tests()
