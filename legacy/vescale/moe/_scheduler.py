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

from typing import Dict, List, Optional, Union
import torch
from torch import nn
from vescale import DeviceMesh, DTensor, from_local
from vescale.moe.experts_allocator import ExpertsAllocator
from vescale.moe.token_dispatcher import TokenDispatcher
from vescale.moe._moe_param_buffer import MoEParamBuffer, MoELayerParamBuffer
from vescale.moe._utils import _MOE_DP, global_all_to_all_single
from vescale.dmodule._factory import FactoryDispatchModeOff
from dataclasses import dataclass


@dataclass
class MoETask:
    model: Optional[nn.Module] = None
    layer_id: int = -1
    device: Optional[torch.device] = None
    expert_id: Optional[Union[torch.Tensor, int]] = -1
    hidden_state: Optional[Union[torch.Tensor, DTensor]] = None
    token_id: Optional[Union[torch.Tensor, DTensor]] = None
    token_weight: Optional[Union[torch.Tensor, DTensor]] = None
    output_tensor: Optional[Union[torch.Tensor, DTensor]] = None


@dataclass
class MoELayerInfo:
    layer_id: int = 0
    num_experts: int = 0
    num_devices: int = 0
    experts_alloc: Optional[List[Optional[DeviceMesh]]] = None
    experts_dp_mesh: Optional[List[Optional[DeviceMesh]]] = None
    device_mask: Optional[List[torch.Tensor]] = None
    param_buffer: Optional[MoELayerParamBuffer] = None

    def set_new_experts_alloc(self, expert_list: List[nn.Module], experts_alloc_info: Dict) -> None:
        experts_alloc: List[Optional[DeviceMesh]] = experts_alloc_info["experts_alloc"]
        dp_size: torch.Tensor = experts_alloc_info["dp_size"]
        if self.param_buffer.is_initialized():
            self.param_buffer.refresh_buffer(expert_list, experts_alloc)
        else:
            self.param_buffer.init_buffer(expert_list, experts_alloc)
        self.experts_alloc = experts_alloc
        self.experts_dp_mesh = [alloc[_MOE_DP] if alloc is not None else None for alloc in experts_alloc]
        self._update_device_mask(dp_size)

    def _update_device_mask(self, dp_size: torch.Tensor) -> None:
        experts_alloc = self.experts_alloc
        num_experts, max_replica, num_devices = len(experts_alloc), dp_size.max().item(), self.num_devices
        device_type = experts_alloc[0].device_type
        device_mask = torch.zeros((num_experts, max_replica, num_devices), device=device_type, dtype=torch.bool)
        for i, alloc in enumerate(experts_alloc):
            mesh = alloc.mesh
            for r in range(mesh.shape[1]):
                device = mesh[:, r]
                device_mask[i, r, device] = True
        self.device_mask = device_mask


class MoEScheduler:
    def __init__(self, experts_allocator: ExpertsAllocator, token_dispatcher: TokenDispatcher, config: Dict):
        def wrap_as_list(value, length):
            if isinstance(value, int):
                return [value] * length
            else:
                assert len(value) == length
                return value

        self.num_layers: int = config["num_layers"]
        self.experts_allocator: ExpertsAllocator = experts_allocator
        self.token_dispatcher: TokenDispatcher = token_dispatcher

        num_experts_list = wrap_as_list(config["num_experts"], self.num_layers)
        num_devices_list = wrap_as_list(config["num_devices"], self.num_layers)

        self._param_buffer = MoEParamBuffer(self.num_layers, num_experts_list)

        self._layer_info: List[MoELayerInfo] = [
            MoELayerInfo(
                layer_id=i,
                num_experts=num_experts_list[i],
                num_devices=num_devices_list[i],
                param_buffer=self._param_buffer.get_layer_param_buffer(i),
            )
            for i in range(self.num_layers)
        ]
        self._task_per_expert: List[MoETask] = []
        self._current_info: Optional[MoELayerInfo] = None

    def init_param_buffer(self, moe_layer_list) -> None:
        assert len(moe_layer_list) == self.num_layers
        for expert_list, layer_info in zip(moe_layer_list, self._layer_info):
            experts_alloc_info = self.experts_allocator.allocate_experts_internal(layer_info.layer_id)
            self.token_dispatcher.set_experts_alloc(experts_alloc_info)
            layer_info.set_new_experts_alloc(expert_list, experts_alloc_info)

    def get_moe_param_buffer(self) -> MoEParamBuffer:
        return self._param_buffer

    def push_task(self, task: MoETask):
        self._task_per_expert.append(task)

    def num_tasks(self):
        return len(self._task_per_expert)

    def _set_context(self, layer_id: int):
        self._current_info = self._layer_info[layer_id]

    def _allocate_experts(self, layer_id: int, task_list: List[MoETask]):
        experts_alloc_info = self.experts_allocator.allocate_experts_internal(layer_id)
        expert_list = [task.model for task in task_list]
        layer_info = self._current_info

        if experts_alloc_info is not None:
            self.token_dispatcher.set_experts_alloc(experts_alloc_info)
            layer_info.set_new_experts_alloc(expert_list, experts_alloc_info)

        layer_info.param_buffer.assign_param(expert_list)

    def _concat_task_per_expert(self, _task_per_expert: List[MoETask]):
        device = _task_per_expert[0].device

        token_id_list, expert_id_list, hidden_state_list, token_weight_list = [], [], [], []
        for task in _task_per_expert:
            token_num = task.token_id.shape[0]
            token_id_list.append(task.token_id)
            expert_id_list.append(torch.full((token_num,), task.expert_id, device=device))
            hidden_state_list.append(task.hidden_state)
            token_weight_list.append(task.token_weight)

        task_full = MoETask(
            layer_id=_task_per_expert[0].layer_id,
            token_id=torch.cat(token_id_list),
            expert_id=torch.cat(expert_id_list),
            hidden_state=torch.cat(hidden_state_list),
            token_weight=torch.cat(token_weight_list),
            device=device,
        )

        return task_full

    def _distribute_workload(self, task_full: MoETask):
        layer_id = task_full.layer_id

        eid, rid = self.token_dispatcher.dispatch_token(layer_id)
        device_mask = self._current_info.device_mask
        token_id, device_id = torch.where(
            device_mask[eid, rid]
        )  # TODO: implement a dedicated kernel for batched slice to avoid sync

        device_id, sort_idx = torch.sort(device_id)
        token_id = token_id[sort_idx]
        device_id_start = torch.searchsorted(
            device_id, torch.arange(self._current_info.num_devices + 1, device=device_id.device)
        )
        pre_split_sizes = device_id_start.diff()

        pre_expert_id = task_full.expert_id[token_id]
        pre_hidden_state = task_full.hidden_state[token_id]
        pre_token_id = task_full.token_id[token_id]
        pre_token_weight = task_full.token_weight[token_id]

        process_split_sizes = torch.empty_like(pre_split_sizes)
        torch.distributed.all_to_all_single(process_split_sizes, pre_split_sizes)
        pre_split_sizes = pre_split_sizes.tolist()
        process_split_sizes = process_split_sizes.tolist()
        process_expert_id = global_all_to_all_single(pre_expert_id, pre_split_sizes, process_split_sizes)
        process_hidden_state = global_all_to_all_single(pre_hidden_state, pre_split_sizes, process_split_sizes)

        return (
            pre_split_sizes,
            pre_token_id,
            pre_token_weight,
            process_split_sizes,
            process_expert_id,
            process_hidden_state,
        )

    def _compute_local_experts(self, process_models, process_expert_id, process_hidden_state):
        if process_hidden_state.numel() == 0:
            return process_hidden_state
        process_expert_id, index_sort = torch.sort(process_expert_id)
        process_start = torch.searchsorted(
            process_expert_id, torch.arange(self._current_info.num_experts + 1, device=process_hidden_state.device)
        )
        process_hidden_state = process_hidden_state[index_sort]
        result_hidden_state = torch.empty_like(process_hidden_state)

        process_start = process_start.tolist()
        for expert_id, expert_model in enumerate(process_models):
            if process_start[expert_id] < process_start[expert_id + 1]:
                hidden_state = process_hidden_state[process_start[expert_id] : process_start[expert_id + 1]]
                hidden_state = expert_model._original_forward(hidden_state)
                result_hidden_state[index_sort[process_start[expert_id] : process_start[expert_id + 1]]] = hidden_state

        return result_hidden_state

    def _distribute_result(self, process_hidden_state, process_split_sizes, post_split_sizes):
        return global_all_to_all_single(process_hidden_state, process_split_sizes, post_split_sizes)

    def _triger_param_comm_hook(self, layer_id: int):
        def hook(*useless):
            self._param_buffer.process_reduce_scatter(layer_id)

        return hook

    @FactoryDispatchModeOff()
    def launch(self):
        layer_id = self._task_per_expert[0].layer_id
        self._set_context(layer_id)

        # step 1: call experts_allocator.allocate_experts() and reallocate experts
        # we place it at the beginning for overlapping this process with `gather` in ZeRO-2
        self._allocate_experts(layer_id, self._task_per_expert)

        # step 2: call token_dispatcher
        task_full = self._concat_task_per_expert(self._task_per_expert)
        self.token_dispatcher.assign_task(
            layer_id,
            token_id=task_full.token_id,
            expert_id=task_full.expert_id,
            hidden_state=task_full.hidden_state,
            token_weight=task_full.token_weight,
        )
        task_full.hidden_state.register_hook(self._triger_param_comm_hook(layer_id))

        # step 3: distribute tokens
        (
            post_split_sizes,
            post_token_id,
            post_token_weight,
            process_split_sizes,
            process_expert_id,
            process_hidden_state,
        ) = self._distribute_workload(task_full)

        # step 4: processing experts
        process_models = [task.model for task in self._task_per_expert]
        process_hidden_state = self._compute_local_experts(process_models, process_expert_id, process_hidden_state)

        # step 5: accumulate the results
        post_hidden_state = self._distribute_result(process_hidden_state, process_split_sizes, post_split_sizes)
        post_hidden_state *= post_token_weight

        output_tensor = self._task_per_expert[0].output_tensor
        device_mesh: DeviceMesh = self._task_per_expert[0].output_tensor.device_mesh
        placements = self._task_per_expert[0].output_tensor.placements
        output_local_tensor = output_tensor._local_tensor
        output_local_tensor.index_add_(0, post_token_id, post_hidden_state)

        # step 6: collect entire workload distribution and call experts_allocator.assign_workload()
        pass

        # step 7: clear the task list and return the result
        self._task_per_expert = []
        return from_local(
            output_local_tensor,
            device_mesh,
            placements,
            run_check=False,
        )
