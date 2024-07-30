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
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests
from vescale.devicemesh_api import VESCALE_DEVICE_MESH
from common_dtensor import DTensorTestBase, with_comms
from vescale.dtensor.api import distribute_tensor
from vescale.optim.base_optimizer import BasicOptimizer
from vescale.initialize import materialize_module
from vescale.ddp.distributed_data_parallel import DistributedDataParallel as DDP
from vescale.plan import (
    PipelineParallelPlan,
    PipelineSplitMethodType,
    PipelineScheduleType,
    ModeType,
)
from vescale.pipe import PipeModule, build_shared_module_group, construct_stage_modules, construct_pipeline_split_graph
from vescale.initialize.deferred_init import deferred_init
from eight_mlp import sharding_plan, EightMLPSharedEmbed
from vescale.dtensor.placement_types import Replicate
from vescale.dmodule.api import parallelize_module


microbatch_size = 16
factor = 16
batch_size = microbatch_size * factor
RANDOM_SEED = 9999


class SharedParamsTest(DTensorTestBase):
    @property
    def world_size(self):
        return 8

    @with_comms
    def test_sync_embedding_weights_two_stages(self):
        """
        Test correctness of synchronizing "shared_units" (embedding)
        weights upon engine initialization.
        """
        pp_size = 2
        dp_size = 2
        tp_size = 2
        deferred_mlp = deferred_init(EightMLPSharedEmbed, hidden=8)
        partition_units = [f"mlp{i + 1}" for i in range(8)] + ["embed1", "embed2"]
        pp_plan = PipelineParallelPlan(
            mode=ModeType.GRAPH_EAGER,
            split_method=PipelineSplitMethodType.UNIFORM,
            num_stages=2,
            virtual_chunks=1,
            smallest_unsplittable_units=partition_units,
            schedule_type=PipelineScheduleType.SIMPLE_1F1B,
            shared_modules=[
                ["embed1", "embed2"]
            ],  # each sublist represents a group of modules to synchronize params/grads
        )
        split_graph = construct_pipeline_split_graph(deferred_mlp, pp_plan, update_split_points=True)
        VESCALE_DEVICE_MESH.init_device_mesh(
            device_type="cuda",
            mesh_shape=(pp_size, dp_size, tp_size),
            mesh_dim_names=["PP", "DP", "TP"],
        )

        stage_modules, stage_dependency, p2p_index_mapping = construct_stage_modules(
            deferred_mlp,
            pp_plan,
            VESCALE_DEVICE_MESH,
            update_split_points=True,
        )
        for module in stage_modules:
            materialize_module(module)
            module.cuda()

        combined_parameters = list(stage_modules[0].parameters())
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
        optimizer = torch.optim.SGD(combined_parameters, **optimizer_fn_kwargs)
        pipe_module = PipeModule(stage_modules, optimizer, None, stage_dependency, p2p_index_mapping, pp_plan)

        build_shared_module_group(
            pipe_module,
            split_graph,
            pp_plan.num_stages,
            pp_plan.virtual_chunks,
            pp_plan.shared_modules,
            VESCALE_DEVICE_MESH,
        )
        if VESCALE_DEVICE_MESH.get_pipeline_parallel_rank() == 0:
            embedding = pipe_module[0].get_submodule("embed1").get_word_embeddings_weight().data
        else:
            embedding = pipe_module[0].get_submodule("embed2").get_word_embeddings_weight().data
        pipe_module.sync_shared_params(VESCALE_DEVICE_MESH, group_id=0, share_params=True, chunk_id=0)
        if VESCALE_DEVICE_MESH.get_pipeline_parallel_rank() == 0:
            sync_embedding = pipe_module[0].get_submodule("embed1").get_word_embeddings_weight().data
        else:
            sync_embedding = pipe_module[0].get_submodule("embed2").get_word_embeddings_weight().data
        assert not torch.testing.assert_close(embedding, sync_embedding)

    @with_comms
    def test_sync_embedding_weights_four_stages(self):
        """
        Test correctness of synchronizing "shared_units" (embedding)
        weights given four stages partitioned.
        """
        pp_size = 4
        dp_size = 2
        tp_size = 1
        model = EightMLPSharedEmbed(hidden=8).cuda()
        partition_units = [f"mlp{i + 1}" for i in range(8)] + ["embed1", "embed2"]
        pp_plan = PipelineParallelPlan(
            mode=ModeType.GRAPH_EAGER,
            split_method=PipelineSplitMethodType.MANUAL,
            num_stages=4,
            virtual_chunks=1,
            smallest_unsplittable_units=partition_units,
            schedule_type=PipelineScheduleType.SIMPLE_1F1B,
            split_points=["mlp2", "mlp5", "mlp7", "embed2"],
            shared_modules=[
                ["embed1", "embed2"]
            ],  # each sublist represents a group of modules to synchronize params/grads
        )

        split_graph = construct_pipeline_split_graph(model, pp_plan, update_split_points=True)
        VESCALE_DEVICE_MESH.init_device_mesh(
            device_type="cuda",
            mesh_shape=(pp_size, dp_size, tp_size),
            mesh_dim_names=["PP", "DP", "TP"],
        )

        stage_modules, stage_dependency, p2p_index_mapping = construct_stage_modules(
            model,
            pp_plan,
            VESCALE_DEVICE_MESH,
            update_split_points=True,
        )
        combined_parameters = list(stage_modules[0].parameters())
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
        optimizer = torch.optim.SGD(combined_parameters, **optimizer_fn_kwargs)
        basic_optimizer = BasicOptimizer(optimizer, models=stage_modules)
        pipe_module = PipeModule(stage_modules, optimizer, None, stage_dependency, p2p_index_mapping, pp_plan)

        build_shared_module_group(
            pipe_module,
            split_graph,
            pp_plan.num_stages,
            pp_plan.virtual_chunks,
            pp_plan.shared_modules,
            VESCALE_DEVICE_MESH,
        )
        if VESCALE_DEVICE_MESH.get_pipeline_parallel_rank() == 0:
            embedding = pipe_module[0].get_submodule("embed1").get_word_embeddings_weight().data
        elif VESCALE_DEVICE_MESH.get_pipeline_parallel_rank() == 3:
            embedding = pipe_module[0].get_submodule("embed2").get_word_embeddings_weight().data
        else:
            embedding = None
        if VESCALE_DEVICE_MESH.get_pipeline_parallel_rank() in [0, 3]:
            pipe_module.sync_shared_params(VESCALE_DEVICE_MESH, group_id=0, share_params=True, chunk_id=0)
        if VESCALE_DEVICE_MESH.get_pipeline_parallel_rank() == 0:
            sync_embedding = pipe_module[0].get_submodule("embed1").get_word_embeddings_weight().data
            assert not torch.testing.assert_close(embedding, sync_embedding)
        elif VESCALE_DEVICE_MESH.get_pipeline_parallel_rank() == 3:
            sync_embedding = pipe_module[0].get_submodule("embed2").get_word_embeddings_weight().data
            assert not torch.testing.assert_close(embedding, sync_embedding)

    @with_comms
    def test_sync_embedding_gradients(self):
        """
        Test correctness of synchronizing "shared_units" (embedding)
        weights given uniform partition results.
        """
        pp_size = 2
        dp_size = 4
        tp_size = 1
        model = EightMLPSharedEmbed(hidden=8).cuda()
        partition_units = [f"mlp{i + 1}" for i in range(8)] + ["embed1", "embed2"]

        pp_plan = PipelineParallelPlan(
            mode=ModeType.GRAPH_EAGER,
            split_method=PipelineSplitMethodType.UNIFORM,
            num_stages=2,
            virtual_chunks=1,
            smallest_unsplittable_units=partition_units,
            schedule_type=PipelineScheduleType.SIMPLE_1F1B,
            shared_modules=[
                ["embed1", "embed2"]
            ],  # each sublist represents a group of modules to synchronize params/grads
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

        split_graph = construct_pipeline_split_graph(model, pp_plan, update_split_points=True)
        VESCALE_DEVICE_MESH.init_device_mesh(
            device_type="cuda",
            mesh_shape=(pp_size, dp_size, tp_size),
            mesh_dim_names=["PP", "DP", "TP"],
        )
        tp_mesh = VESCALE_DEVICE_MESH["TP"]
        dp_mesh = VESCALE_DEVICE_MESH["DP"]

        stage_modules, stage_dependency, p2p_index_mapping = construct_stage_modules(
            model,
            pp_plan,
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
            ddp_module = DDP(
                parallelized_module,
                dp_mesh,
                accumulate_allreduce_grads_in_fp32=True,
                overlap_grad_reduce=True,
                use_distributed_optimizer=False,
                disable_bucketing=False,
                bucket_size=40000000,
            )
            stage_modules[i] = ddp_module
        combined_parameters = list(stage_modules[0].parameters())
        optimizer = torch.optim.SGD(combined_parameters, **optimizer_fn_kwargs)
        basic_optimizer = BasicOptimizer(optimizer, models=stage_modules)
        pipe_module = PipeModule(stage_modules, optimizer, None, stage_dependency, p2p_index_mapping, pp_plan)

        build_shared_module_group(
            pipe_module,
            split_graph,
            pp_plan.num_stages,
            pp_plan.virtual_chunks,
            pp_plan.shared_modules,
            VESCALE_DEVICE_MESH,
        )
        loss_fn = nn.MSELoss()
        input_tensor = distribute_tensor(torch.ones(3).long().cuda(), tp_mesh, [Replicate()])

        if VESCALE_DEVICE_MESH.get_pipeline_parallel_rank() == 0:
            embed = pipe_module[0].module.embed1
        else:
            embed = pipe_module[0].module.embed2
        output = embed(input_tensor)
        target = torch.zeros_like(output)
        target = distribute_tensor(target, tp_mesh, [Replicate()])
        losses = loss_fn(output, target)
        losses.backward()
        old_grad = embed.embedding.weight.main_grad.clone()
        pipe_module.sync_shared_params(VESCALE_DEVICE_MESH, group_id=0, share_params=False, chunk_id=0)
        if VESCALE_DEVICE_MESH.get_pipeline_parallel_rank() == 0:
            embed = pipe_module[0].module.embed1
        else:
            embed = pipe_module[0].module.embed2
        new_grad = embed.embedding.weight.main_grad.clone()
        assert not torch.equal(old_grad, new_grad)


if __name__ == "__main__":
    run_tests()
