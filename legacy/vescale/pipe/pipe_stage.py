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

"""
This `PipeModule` Class is the abstraction of a pipeline stage.

PipeModule takes as microbatch input 1). List of data per microbatch, 2). Dictionary of data per microbatch, 3). torch.Tensor

PipeModule takes both 1). p2p transmitted data from incoming stages, and 2). local data inputs

Each Pipeline stage can run single batch data forward, just like nn.Modules, we can
use forward functions and new p2p ops to replement pipeline forward and backward

For Example 1.
    ```python
        stage: PipeModule = ...
        single_data = ... # a single microbatch of data
        fwd = stage(single_data)
        p2p_send_recv( ... )
    ```

For Example 2.
    ```python
        stage: PipeModule = ...
        p2p_data = ... # a torch.Tensor from last stage
        local_data = Dict(...) # a single microbatch of data
        fwd = stage(p2p_data, local_inputs=local_data)
        p2p_send_recv( ... )
    ```

"""

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
import inspect
import re
from typing import Dict, List, Tuple, Union, Optional, Sequence, Callable, Any
from vescale.optim.base_optimizer import BasicOptimizer
from vescale.optim.distributed_optimizer import DistributedOptimizer
from vescale.devicemesh_api.api import VeDeviceMesh
from vescale.ddp.distributed_data_parallel import DistributedDataParallel as DDP
from vescale.dtensor.dtensor import DTensor
from vescale.plan import PipelineParallelPlan, PipelineP2PSpec
from vescale.pipe.pipe_parser import construct_pipeline_split_graph
from collections import defaultdict


class PipeModule(nn.Module):
    def __init__(
        self,
        module: Union[nn.Module, List],
        doptimizer: Union[BasicOptimizer, DistributedOptimizer],
        lr_scheduler: Callable,
        stage_deps: np.ndarray,
        p2p_index_mapping: Dict,
        plan: PipelineParallelPlan,
    ):
        super().__init__()
        self.stage_modules = {}
        if isinstance(module, List):
            for i in range(len(module)):
                self.stage_modules[i] = module[i]
        else:
            self.stage_modules[0] = module
        self.doptimizer = doptimizer
        self.lr_scheduler = lr_scheduler
        self.shared_module_process_groups = defaultdict()
        self.sync_chunk_ids = set()
        self.shared_path_this_stage = {}
        self.shared_module_mapping = {}
        self.plan = plan
        self.num_stages = self.plan.num_stages
        self.virtual_chunks = self.plan.virtual_chunks
        self.stage_deps = stage_deps
        self.p2p_index_mapping = p2p_index_mapping

    def forward(
        self,
        inputs: Union[torch.Tensor, List, Dict],
        local_inputs: Union[torch.Tensor, List, Dict] = None,
        chunk_id: int = 0,
    ):
        """
        Forward propagation function of a pipeline stage. This function processes inputs to model chunks from p2p data transfers
        and local dataloaders.

        Note:
            - inputs (Union[torch.Tensor, List, Dict]): transmitted data received from another pipeline stage
            - local_inputs (Union[torch.Tensor, List, Dict]): optional input of local data
            - chunk_id (int): identifier of dictating what virtual model chunk to execute in interleaved 1f1b schedule.
              If it is the simple 1f1b schedule, chunk_id=0.

        Args:
            inputs (torch.Tensor, list, dict): inputs fed into model partition module.
            local_inputs (torch.Tensor, list, dict): local inputs from dataloaders, used when executing pipeline schedule.

        Returns:
            Output activations.

        """
        chunk_module = self.stage_modules[chunk_id]
        if local_inputs is None:
            if isinstance(inputs, list):
                return chunk_module(*inputs)
            elif isinstance(inputs, dict):
                return chunk_module(**inputs)
            elif inputs is None:
                return chunk_module()
            else:
                return chunk_module(inputs)
        else:
            combined_data = self._prepare_inputs(chunk_module, inputs, local_inputs)
            return chunk_module(**combined_data)

    def _prepare_inputs(self, module, inputs, local_inputs=None):
        fwd = module.module.forward if isinstance(module, DDP) else module.forward
        sig = inspect.signature(fwd)
        arguments = list(sig.parameters.keys())
        dict_inputs = self._prepare_data_formats(arguments, inputs)
        dict_local_inputs = self._prepare_data_formats(arguments, local_inputs)
        final_inputs = {}
        for key in arguments:
            input_val, local_val = dict_inputs.get(key), dict_local_inputs.get(key)
            if input_val is not None:
                final_inputs[key] = input_val
            elif local_val is not None:
                final_inputs[key] = local_val
            elif sig.parameters[key].default is not inspect.Parameter.empty:
                final_inputs[key] = sig.parameters[key].default.default
        return final_inputs

    def _prepare_data_formats(self, keys, data):
        if data is None or isinstance(data, Sequence) and len(data) == 1 and data[0] is None:
            if keys:
                return {keys[0]: None}
            return None
        if isinstance(data, torch.Tensor):
            data = [data]
        if isinstance(data, Sequence):
            args_length = min(len(data), len(keys))
            data = {keys[i]: data[i] for i in range(args_length)}
        return data

    def __getitem__(self, module_chunk_id: int):
        assert module_chunk_id in self.stage_modules, "Virtual chunk id not existed!"
        return self.stage_modules[module_chunk_id]

    @property
    def get_optimizer(self):
        return self.doptimizer

    @property
    def get_lr_scheduler(self):
        return self.lr_scheduler

    def parameters(self):
        parameters = []
        for chunk_id in range(self.virtual_chunks):
            parameters += list(self.stage_modules[chunk_id].parameters())
        return parameters

    def has_shared_params(self, global_mesh: VeDeviceMesh, group_id: int, tp_rank: int) -> bool:
        """
        Checks whether this stage has submodules to synchronize parameters or gradients.
        An additional use case of this function is to dictate if a submodule's shared parameter
        (invoked by self.get_shared_module()) participates in grad norm clipping.

        Args:
            global_mesh (VeDeviceMesh): global DeviceMesh with which one looks up communication information.
            group_id (int): specify groups of modules across stages to synchronize. Default by 0.
            tp_rank (int): tensor model parallel rank of current stage.

        Returns:
            whether a stage contains sharable parameters

        """
        local_rank = global_mesh.get_local_rank()
        return not (
            not self.shared_module_process_groups
            or tp_rank not in self.shared_module_process_groups[group_id]
            or local_rank not in dist.get_process_group_ranks(self.shared_module_process_groups[group_id][tp_rank])
        )

    def sync_shared_params(
        self, global_mesh: VeDeviceMesh, group_id: int = 0, share_params: bool = True, chunk_id: int = 0
    ):
        """
        Synchronize parameters of reused modules e.g.
        Embedding. This function is invoked in each run of PP schedule.

        Args:
            global_mesh (VeDeviceMesh): global DeviceMesh with which one looks up communication information.
            group_id (int): specify groups of modules across stages to synchronize. Default by 0.
            share_params (bool): if True, sync weight parameters; otherwise, share gradients.
            chunk_id (int): identify if current virtual model chunk in this stage has any module to synchronize.

        """
        tp_rank = global_mesh.get_tensor_parallel_rank()
        if (
            not self.has_shared_params(global_mesh, group_id=group_id, tp_rank=tp_rank)
            or chunk_id not in self.sync_chunk_ids
        ):
            return
        # assume that each model chunk has at most 1 sharable sub-module per shared group
        shared_submodule_path = self.shared_path_this_stage[(group_id, chunk_id)]
        model_chunk = self.stage_modules[chunk_id]
        if isinstance(model_chunk, DDP):
            model_chunk = model_chunk.module
        target_module = model_chunk.get_submodule(shared_submodule_path)
        if getattr(target_module, "get_word_embeddings_weight", None):
            target_module = target_module.get_word_embeddings_weight()

        # assume tp coordinate is always the last dimension
        sync_group = self.shared_module_process_groups[group_id][tp_rank]
        group_size = dist.get_world_size(group=sync_group)

        if share_params:
            if isinstance(target_module.data, DTensor):
                dist.all_reduce(target_module.data._local_tensor, group=sync_group)
            else:
                dist.all_reduce(target_module.data, group=sync_group)
            target_module.data /= group_size
        else:
            # if type is DTensr, then do local_tensor.grad
            if target_module.grad is not None:
                target_module.grad.data /= group_size
                dist.all_reduce(target_module.grad.data, group=sync_group)
            else:  # DDP Module
                target_module.main_grad /= group_size
                dist.all_reduce(target_module.main_grad, group=sync_group)


def construct_stage_modules(
    model: nn.Module,
    plan: PipelineParallelPlan,
    global_mesh: VeDeviceMesh,
    update_split_points: bool = False,
):
    """
    Pipeline Parallelism API that constructs ingredients for building PipelineModule.

    Args:
        model (nn.Module): raw model
        plan (PipelineParallelPlan): configuration of pipeline paralellism API.
        update_split_points (bool): set this switch on to update pipeline split points in-place.

    Returns:
        Triplet of 1). list of modules in a pipeline stage, 2). abstraction of send-receive dependency relationship
            among stages, 3). P2P input index mapping.

    """
    num_stages = plan.num_stages
    virtual_chunks = plan.virtual_chunks
    split_graph = construct_pipeline_split_graph(model, plan, update_split_points=update_split_points)

    # assign modules to stage, establish stage dependency and input mapping
    stage_modules, stage_dependency, p2p_index_mapping = build_stage_module_and_dependency(
        split_graph,
        num_stages,
        virtual_chunks,
        stage_id=global_mesh.get_pipeline_parallel_rank(),
    )
    submodules_this_stage = []
    for chunk_id in range(len(stage_modules)):
        submodules_this_stage.append(stage_modules[chunk_id])
    return submodules_this_stage, stage_dependency, p2p_index_mapping


def construct_pipeline_stage(
    model: nn.Module,
    plan: PipelineParallelPlan,
    global_mesh: VeDeviceMesh,
    lr_scheduler: Optional[Union[Callable, Tuple[Callable, Any]]] = None,
    update_split_points: bool = False,
):
    """
    Pipeline Parallelism API that constructs PipeModule from the raw model.

    Args:
        model (nn.Module): raw model.
        plan (PipelineParallelPlan): configuration of pipeline paralellism API.
        lr_scheduler (Optional[Union[Callable, Tuple[Callable, Any]]]): learning rate scheduler.
        update_split_points (bool): set this switch on to update pipeline split points in-place.

    Returns:
        Pipeline stage.

    """
    stage_modules, stage_dependency, p2p_index_mapping = construct_stage_modules(
        model, plan, global_mesh, update_split_points
    )
    return PipeModule(stage_modules, None, lr_scheduler, stage_dependency, p2p_index_mapping, plan)


def build_shared_module_group(
    pipe_module: PipeModule,
    split_graph: torch.fx.GraphModule,
    num_stages: int,
    virtual_chunks: int,
    shared_module_path_groups: List[List],
    global_mesh: VeDeviceMesh,
):
    """
    Pipeline Parallelism API that establishes groups of modules which
    synchronize parameters or gradients amongst one another.

    Args:
        pipe_module (PipeModule): pipeline stage to assign synchronzied mapping.
        split_graph (torch.fx.GraphModule): the global model graph split into stages.
        num_stages (int): number of pipeline stages.
        virtual_chunks (int): number of virtual pipeline stage chunks in a stage.
        shared_module_path_groups (List[List]): list of groups of module fully qualified names,
            where modules in the same group synchronizes parameters or gradients.
        global_mesh (VeDeviceMesh): global DeviceMesh with which one looks up communication information.

    Returns:
        Tuple of 1). a dictionary of shared group items, 2). a dictionary of shared group this stage is involved
            3). synchronized model chunk ids, and 4). path to the shared submodule, if applicable.

    """
    shared_module_process_groups = defaultdict()
    shared_module_mapping = {}
    sync_chunk_ids = set()
    shared_path_this_stage = {}
    module_partition_names_by_stage = [[] for _ in range(num_stages)]
    num_model_partitions = num_stages * virtual_chunks
    for j in range(num_model_partitions):
        module_partition_names_by_stage[j % num_stages].append(f"stage{j}")
    stage_id = global_mesh.get_pipeline_parallel_rank()
    # establish process groups of synchronizing shared embeddings
    if shared_module_path_groups:
        shared_module_process_groups, shared_module_mapping, shared_info = _establish_shared_module_groups(
            num_stages,
            virtual_chunks,
            module_partition_names_by_stage,
            split_graph,
            shared_module_path_groups,
            global_mesh,
        )
        for group_id, group in enumerate(shared_info):
            for _stage_id, chunk_id, path in group:
                if _stage_id == stage_id:
                    sync_chunk_ids.add(chunk_id)
                    shared_path_this_stage[(group_id, chunk_id)] = path
    pipe_module.shared_module_process_groups = shared_module_process_groups
    pipe_module.shared_module_mapping = shared_module_mapping
    pipe_module.sync_chunk_ids = sync_chunk_ids
    pipe_module.shared_path_this_stage = shared_path_this_stage
    return shared_module_process_groups, shared_module_mapping, sync_chunk_ids, shared_path_this_stage


def build_stage_module_and_dependency(
    split_graph: torch.fx.GraphModule,
    num_stages: int,
    virtual_chunks: int,
    stage_id: int,
):
    """
    Establishes sub-modules of the same stage as well as the send-receive relationship among stages.

    Args:
        split_graph (torch.fx.GraphModule): the global model graph split into stages.
        num_stages (int): number of pipeline stages.
        virtual_chunks (int): number of virtual pipeline stage chunks in a stage.
        stage_id (int): pipeline stage id.

    Returns:
        Submodules of a pipeline stage, inter-stage dependency, and P2P input mapping.

    """
    # generate inter-stage communication dependency and communication mapping
    stage_dependency, p2p_index_mapping = _generate_stage_dependencies(split_graph, num_stages, virtual_chunks)
    # build sub-modules belonging to the current pipeline stage
    stage_modules = _build_module(split_graph, num_stages, virtual_chunks, stage_id)
    return stage_modules, stage_dependency, p2p_index_mapping


def _generate_stage_dependencies(graph: torch.fx.GraphModule, num_stage: int, virtual_chunks: int):
    """
    Generates inter-stage dependency and P2P index mapping across stages.

    Args:
        graph (torch.fx.GraphModule): the whole trace graph of the model.

    Returns:
        Mapping of inter-stage dependency and p2p index mapping.

    """
    stage_to_chunk_mapping = _get_stage_to_chunk_mapping(virtual_chunks, num_stage)
    _stage_to_chunk_mapping = {}
    for stage_id, partition_ids in stage_to_chunk_mapping.items():
        for part_id in partition_ids:
            _stage_to_chunk_mapping[part_id] = stage_id
    stage_to_chunk_mapping = _stage_to_chunk_mapping

    stage_rule = r"stage\d+"
    stage2node = {}
    for node in graph.graph.nodes:
        if re.match(stage_rule, node.name):
            stage2node.update({node.name: node})

    stage_deps = np.zeros((num_stage, num_stage))
    for node_name, node in stage2node.items():
        partition_id = int(node_name[5:])
        stage_id = stage_to_chunk_mapping[partition_id]
        node_user = node.users.keys()
        for u_node in node_user:
            if u_node.name in stage2node:
                u_id = int(u_node.name[5:])
                target_stage_id = stage_to_chunk_mapping[u_id]
                if stage_deps[target_stage_id][stage_id] or stage_id == num_stage - 1:
                    # no recurring edge!
                    continue
                stage_deps[stage_id][target_stage_id] = 1

    # construct p2p index mapping
    p2p_index_mapping = {}
    for node_name, node in stage2node.items():
        partition_id = int(node_name[5:])
        stage_id = stage_to_chunk_mapping[partition_id]
        args_mapping = []
        for input_id, arg_node in enumerate(node.args):
            if arg_node.name in stage2node:
                arg_partition_id = int(arg_node.name[5:])
                arg_stage_id = stage_to_chunk_mapping[arg_partition_id]
                args_mapping.append(PipelineP2PSpec(arg_stage_id, input_id))
            else:  # should from local
                args_mapping.append(PipelineP2PSpec(stage_id, input_id))
        p2p_index_mapping.update({stage_id: args_mapping})

    return stage_deps, p2p_index_mapping


def _establish_shared_module_groups(
    num_stage,
    virtual_chunks,
    module_partition_names_by_stage,
    split_graph,
    shared_module_path_groups,
    global_mesh: VeDeviceMesh,
):
    """
    Identify groups of modules to share gradients/weights, e.g. embedding layers
    upon initialization and at the end of a pipeline schedule run.
    """
    all_named_modules = [[] for _ in range(num_stage)]
    for stage_id in range(num_stage):
        for chunk_id in range(virtual_chunks):
            key_name = module_partition_names_by_stage[stage_id][chunk_id]
            module_graph = split_graph.get_submodule(key_name)
            all_named_modules[stage_id].append({name for name, _ in module_graph.named_modules()})

    shared_module_paths = [[] for _ in range(len(shared_module_path_groups))]
    for idx, shared_module_group in enumerate(shared_module_path_groups):
        for module_path in shared_module_group:
            stage_id, chunk_id = _locate_shared_module(module_path, all_named_modules, num_stage, virtual_chunks)
            shared_module_paths[idx].append((stage_id, chunk_id, module_path))
    shared_stages_groups = [
        [stage for stage, _, _ in shared_module_paths[idx]] for idx in range(len(shared_module_path_groups))
    ]

    all_tp_submeshes = global_mesh.get_global_tensor_parallel_meshes()
    # TODO: in future, keep track of multiple groups of shared modules
    all_tp_groups = []
    map_id = 0
    for dm in all_tp_submeshes:
        mesh_list = dm.mesh.tolist()
        converted_pp_ranks = [global_mesh.get_strategy_coordinate(_idx)[0] for _idx in mesh_list]
        assert all(i == converted_pp_ranks[0] for i in converted_pp_ranks)
        for pp_rank in shared_stages_groups[map_id]:
            if pp_rank == converted_pp_ranks[0]:
                all_tp_groups.append(mesh_list)
                break

    shared_tp_comm_groups = list(zip(*all_tp_groups))
    shared_module_process_groups = defaultdict(dict)
    shared_module_mapping = {}
    shared_module_mapping[map_id] = shared_stages_groups[map_id]
    for tp_idx, shared_group in enumerate(shared_tp_comm_groups):
        sync_embed_pg = dist.new_group(ranks=shared_group, backend="nccl")
        shared_module_process_groups[map_id][tp_idx] = sync_embed_pg
    return shared_module_process_groups, shared_module_mapping, shared_module_paths


def _locate_shared_module(module_path, all_named_modules, num_stage, virtual_chunks):
    for stage_id in range(num_stage):
        for chunk_id in range(virtual_chunks):
            if module_path in all_named_modules[stage_id][chunk_id]:
                return stage_id, chunk_id
    raise ValueError(f"Module to be synchronized not found: {module_path}")


def _build_model_chunks(stage_id, model_graph, mapping):
    assert stage_id in mapping
    pipeline_chunks = {}
    unique_id = 0
    for chunk_id, partition_id in enumerate(mapping[stage_id]):
        key = f"stage{partition_id}"
        virtual_pipeline_module = getattr(model_graph, key)
        # assign unique id for each low-level submodule
        for _, submodule in virtual_pipeline_module.named_modules():
            if len(list(submodule.children())) == 0:
                registered_module_id = f"module_{stage_id}_{chunk_id}_{unique_id}"
                virtual_pipeline_module.module_id = registered_module_id
                unique_id += 1
        pipeline_chunks[chunk_id] = virtual_pipeline_module
    return pipeline_chunks


def _build_module(model_graph: torch.fx.GraphModule, num_stages: int, num_model_chunks: int, stage_id: int):
    """
    Builds model chunks by stage, and assigns unique submodule id to every basic modules.

    Args:
        model_graph (torch.fx.GraphModule): the model trace graph with stage partitions.
        num_stages (int): number of pipeline stages.
        num_model_chunks (int): number of virtual pipeline chunks per stage.
        dist_api (VeDeviceMesh): an object of DeviceMesh API.

    Returns:
        Mapping of chunk id to model partitions of the current stage.

    """
    stage_to_chunk = _get_stage_to_chunk_mapping(num_model_chunks, num_stages)
    return _build_model_chunks(stage_id, model_graph, stage_to_chunk)


def _get_stage_to_chunk_mapping(num_model_chunks, num_stages):
    """
    Gets a mapping from stage id to model partition ids.

    Args:
        num_model_chunks (int): number of virtual pipeline chunks per stage.
        num_stages (int): number of pipeline stages.

    Returns:
        Mapping from stages to their model chunks.

    """
    if num_model_chunks == 1:
        stage_to_chunk = {i: [i] for i in range(num_stages)}
    else:
        length = num_stages * num_model_chunks
        stage_to_chunk = {i: [] for i in range(num_stages)}
        for i in range(length):
            stage_to_chunk[i % num_stages].append(i)
    return stage_to_chunk
