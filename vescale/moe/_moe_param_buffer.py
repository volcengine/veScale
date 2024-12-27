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

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
from torch import nn
import torch.distributed.distributed_c10d as c10d

from vescale import DeviceMesh, DTensor, Replicate, Shard, from_local
from vescale.moe._utils import _MOE_DP
from vescale.dtensor._collective_utils import mesh_all_gather, mesh_reduce_scatter, mesh_wait


_MOE_BUFFER_TRANSPOSE_TAG = "_MOE_BUFFER_TRANSPOSE_TAG"
aten = torch.ops.aten


class MoEBufferTensor(torch.Tensor):
    def __init__(self, tensor):
        self.tensor = tensor

    def __repr__(self):
        return f"MoEBufferTensor({self.tensor})"

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        print(func, args)
        if len(args) == 1:
            return MoEBufferTensor(func(args[0].tensor))


class MoELayerParamBuffer:
    def __init__(self, layer_id: int, num_experts: int, param_buffer: "MoEParamBuffer"):
        self.layer_id = layer_id
        self.num_experts = num_experts
        self._param_buffer = param_buffer

        self._mesh_id: Dict[DeviceMesh, int] = {}
        self._local_param_buffer: List[torch.Tensor] = []
        self._local_grad_buffer: List[torch.Tensor] = []
        self._global_param_buffer: List[torch.Tensor] = []
        self._global_grad_buffer: List[torch.Tensor] = []
        self._buffer_mesh: List[DeviceMesh] = []
        self._buffer_shape: List[Tuple[int]] = []
        self._buffer_stride: List[Tuple[int]] = []

        self._mesh_id_to_expert_id: List[List[int]] = []
        self._expert_dp_mesh: List[DeviceMesh] = []
        self._expert_local_param_buffer: List[Optional[torch.Tensor]] = []
        self._expert_global_grad_buffer: List[Optional[torch.Tensor]] = []
        self._expert_stride: List[Tuple[int]] = []

        self._initialized_flag: bool = False
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._dirty_flag: bool = False

    def mark_dirty(self):
        self._dirty_flag = True

    def is_dirty(self):
        return self._dirty_flag

    def reset_dirty(self):
        self._dirty_flag = False

    def init_buffer(self, expert_list: List[nn.Module], experts_alloc: List[DeviceMesh]):
        self._initialized_flag = True

        for expert, alloc in zip(expert_list, experts_alloc):
            for m in expert.modules():
                if hasattr(m, "weight"):
                    placement = m.weight.placements[0]
                    m._moe_device_mesh = alloc
                    m._moe_placements = (placement, Shard(1 - placement.dim))
                    m._moe_buffer_global_shape = m.weight.shape
                    assert placement.is_shard()
                    if placement.is_shard() and placement.dim == 1:
                        setattr(m, _MOE_BUFFER_TRANSPOSE_TAG, True)
                    param = m.weight.cross_mesh_redistribute(
                        alloc,
                        placements=(placement, Shard(1 - placement.dim)),
                        async_op=False,
                    )
                    m.weight = nn.Parameter(param)

        tmp_buffer = []
        self._mesh_id = {}
        self._buffer_mesh = []
        self._mesh_id_to_expert_id = []
        self._expert_dp_mesh = []

        for expert_id, (expert, alloc) in enumerate(zip(expert_list, experts_alloc)):
            dp_mesh = alloc[_MOE_DP]
            self._expert_dp_mesh.append(dp_mesh)
            if dp_mesh is None:
                continue

            if dp_mesh not in self._mesh_id:
                mesh_id = len(self._mesh_id)
                self._mesh_id[dp_mesh] = mesh_id
                self._mesh_id_to_expert_id.append([])
                tmp_buffer.append(None)
                self._buffer_mesh.append(dp_mesh)
            else:
                mesh_id = self._mesh_id[dp_mesh]
            self._mesh_id_to_expert_id[mesh_id].append(expert_id)

            for m in expert.modules():
                if hasattr(m, "weight"):
                    if hasattr(m, _MOE_BUFFER_TRANSPOSE_TAG):
                        tensor = m.weight._local_tensor.T
                        local_shape = (tensor.shape[1], tensor.shape[0])
                        dp_shape = (tensor.shape[1] * dp_mesh.ndevice, tensor.shape[0])
                    else:
                        tensor = m.weight._local_tensor
                        local_shape = (tensor.shape[0], tensor.shape[1])
                        dp_shape = (tensor.shape[0], tensor.shape[1] * dp_mesh.ndevice)
                    prev_tensor = tmp_buffer[mesh_id]
                    if prev_tensor is None:
                        storage_offset = 0
                        tmp_buffer[mesh_id] = tensor
                    else:
                        storage_offset = prev_tensor.shape[0]
                        tmp_buffer[mesh_id] = torch.cat([prev_tensor, tensor])
                    m._moe_buffer_local_shape = local_shape
                    m._moe_buffer_dp_shape = dp_shape
                    m._moe_buffer_storage_offset = storage_offset

        self._local_param_buffer = []
        self._global_grad_buffer = []
        self._buffer_shape = []
        self._buffer_stride = []

        for mesh_id, tensor in enumerate(tmp_buffer):
            shape = (tensor.numel() * self._buffer_mesh[mesh_id].ndevice,)
            self._local_param_buffer.append(tensor.T.flatten())
            self._global_grad_buffer.append(torch.empty(shape, device=tensor.device, dtype=tensor.dtype))
            self._buffer_shape.append(shape)
            self._buffer_stride.append((1, tensor.shape[0]))

        self._expert_local_param_buffer = []
        self._expert_global_grad_buffer = []
        self._expert_stride = []

        for expert_id in range(self.num_experts):
            dp_mesh = self._expert_dp_mesh[expert_id]
            if dp_mesh is None:
                self._expert_local_param_buffer.append(None)
                self._expert_global_grad_buffer.append(None)
                self._expert_stride.append(None)
            else:
                mesh_id = self._mesh_id[dp_mesh]
                self._expert_local_param_buffer.append(self._local_param_buffer[mesh_id])
                self._expert_global_grad_buffer.append(self._global_grad_buffer[mesh_id])
                self._expert_stride.append(self._buffer_stride[mesh_id])

        self.run_all_gather()

    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        self._optimizer = optimizer

    def is_initialized(self):
        return self._initialized_flag

    def refresh_buffer(self, expert_list: List[nn.Module], experts_alloc: List[Optional[DeviceMesh]]) -> None:
        def collect_optimizer_states():
            tensor_state_keys = []
            scalar_state = {}
            param = self._local_param_buffer[0]
            for state_name, state in self._optimizer.state[param].items():
                if state.shape == param.shape:
                    tensor_state_keys.append(state_name)
                else:
                    scalar_state[state_name] = state
            return tensor_state_keys, scalar_state

        device = self._local_param_buffer[0].device
        tensor_state_keys, scalar_state = collect_optimizer_states()
        param_key = "params"
        refresh_keys = [param_key] + tensor_state_keys
        refresh_buffer = {}
        for key in refresh_keys:
            refresh_buffer[key] = []

        self._mesh_id = {}
        self._buffer_mesh = []
        self._mesh_id_to_expert_id = []
        self._expert_dp_mesh = []

        for expert_id, (expert, alloc) in enumerate(zip(expert_list, experts_alloc)):
            dp_mesh = alloc[_MOE_DP]
            self._expert_dp_mesh.append(dp_mesh)
            if dp_mesh is not None:
                if dp_mesh not in self._mesh_id:
                    mesh_id = len(self._mesh_id)
                    self._mesh_id[dp_mesh] = mesh_id
                    self._mesh_id_to_expert_id.append([])
                    for key in refresh_keys:
                        refresh_buffer[key].append(None)
                    self._buffer_mesh.append(dp_mesh)
                else:
                    mesh_id = self._mesh_id[dp_mesh]
                self._mesh_id_to_expert_id[mesh_id].append(expert_id)
            expert_stride = self._expert_stride[expert_id]

            for nm, m in expert.named_modules():
                if hasattr(m, "weight"):
                    device_mesh = m._moe_device_mesh
                    placements = m._moe_placements
                    global_shape = m._moe_buffer_global_shape
                    current_refresh_buffer = {}

                    param = self._expert_local_param_buffer[expert_id]
                    if param is None:
                        current_refresh_buffer[param_key] = torch.tensor([], device=device).view(0, 0)
                        for key in tensor_state_keys:
                            current_refresh_buffer[key] = torch.tensor([], device=device).view(0, 0)
                        stride = (0, 0)
                    else:
                        local_shape = m._moe_buffer_local_shape
                        storage_offset = m._moe_buffer_storage_offset
                        optimizer_state = self._optimizer.state[param]
                        if hasattr(m, _MOE_BUFFER_TRANSPOSE_TAG):
                            stride = (expert_stride[1], expert_stride[0])
                        else:
                            stride = expert_stride
                        current_refresh_buffer[param_key] = param.as_strided(local_shape, stride, storage_offset)
                        for key in tensor_state_keys:
                            current_refresh_buffer[key] = optimizer_state[key].as_strided(
                                local_shape, stride, storage_offset
                            )

                    for key in refresh_keys:
                        tensor = current_refresh_buffer[key]
                        dtensor = from_local(
                            tensor,
                            device_mesh,
                            placements,
                            run_check=False,
                            shape=global_shape,
                            stride=stride,
                        )
                        current_refresh_buffer[key] = dtensor.redistribute(
                            alloc, placements, async_op=False
                        )._local_tensor

                    m._moe_device_mesh = alloc

                    if dp_mesh is None:
                        delattr(m, "_moe_buffer_local_shape")
                        delattr(m, "_moe_buffer_dp_shape")
                        delattr(m, "_moe_buffer_storage_offset")
                        continue

                    for key in refresh_keys:
                        tmp_buffer = refresh_buffer[key]
                        if hasattr(m, _MOE_BUFFER_TRANSPOSE_TAG):
                            tensor = current_refresh_buffer[key].T
                        else:
                            tensor = current_refresh_buffer[key]

                        prev_tensor = tmp_buffer[mesh_id]
                        if prev_tensor is None:
                            storage_offset = 0
                            tmp_buffer[mesh_id] = tensor
                        else:
                            storage_offset = prev_tensor.shape[0]
                            tmp_buffer[mesh_id] = torch.cat([prev_tensor, tensor])

                    tensor = current_refresh_buffer[param_key]
                    local_shape = (tensor.shape[0], tensor.shape[1])
                    if hasattr(m, _MOE_BUFFER_TRANSPOSE_TAG):
                        dp_shape = (tensor.shape[0] * dp_mesh.ndevice, tensor.shape[1])
                    else:
                        dp_shape = (tensor.shape[0], tensor.shape[1] * dp_mesh.ndevice)

                    m._moe_buffer_local_shape = local_shape
                    m._moe_buffer_dp_shape = dp_shape
                    m._moe_buffer_storage_offset = storage_offset

        for param in self._local_param_buffer:
            del self._optimizer.state[param]

        self._local_param_buffer.clear()
        self._global_grad_buffer.clear()
        self._buffer_shape.clear()
        self._buffer_stride.clear()

        for mesh_id, tensor in enumerate(refresh_buffer[param_key]):
            shape = (tensor.numel() * self._buffer_mesh[mesh_id].ndevice,)
            self._local_param_buffer.append(tensor.T.flatten())
            self._global_grad_buffer.append(torch.empty(shape, device=tensor.device, dtype=tensor.dtype))
            self._buffer_shape.append(shape)
            self._buffer_stride.append((1, tensor.shape[0]))

        self._expert_local_param_buffer.clear()
        self._expert_global_grad_buffer.clear()
        self._expert_stride.clear()

        for expert_id in range(self.num_experts):
            dp_mesh = self._expert_dp_mesh[expert_id]
            if dp_mesh is None:
                self._expert_local_param_buffer.append(None)
                self._expert_global_grad_buffer.append(None)
                self._expert_stride.append(None)
            else:
                mesh_id = self._mesh_id[dp_mesh]
                self._expert_local_param_buffer.append(self._local_param_buffer[mesh_id])
                self._expert_global_grad_buffer.append(self._global_grad_buffer[mesh_id])
                self._expert_stride.append(self._buffer_stride[mesh_id])

        for i, param in enumerate(self._local_param_buffer):
            self._optimizer.state[param] = {}
            for state_key in tensor_state_keys:
                self._optimizer.state[param][state_key] = refresh_buffer[state_key][i].T.flatten()
            self._optimizer.state[param] |= scalar_state

        self.mark_dirty()
        self.run_all_gather()

    def assign_param(self, expert_list: List[nn.Module]) -> None:
        expert_global_param_buffer = [None] * self.num_experts

        for mesh_id in range(len(self._local_param_buffer)):
            mesh = self._buffer_mesh[mesh_id]
            if mesh.ndevice == 1:
                global_tensor = self._global_param_buffer[mesh_id]
            else:
                global_tensor = mesh_wait(self._global_param_buffer[mesh_id])
            for expert_id in self._mesh_id_to_expert_id[mesh_id]:
                expert_global_param_buffer[expert_id] = global_tensor
            self._global_grad_buffer[mesh_id].zero_()

        self._param_buffer.finish_all_gather()

        for expert_id, expert in enumerate(expert_list):
            param_buffer = expert_global_param_buffer[expert_id]
            if param_buffer is None:
                continue
            grad_buffer = self._expert_global_grad_buffer[expert_id]
            expert_stride = self._expert_stride[expert_id]
            for m in expert.modules():
                if hasattr(m, "weight"):
                    if hasattr(m, _MOE_BUFFER_TRANSPOSE_TAG):
                        stride = (expert_stride[1], expert_stride[0])
                    else:
                        stride = expert_stride
                    shape = m._moe_buffer_dp_shape
                    storage_offset = m._moe_buffer_storage_offset
                    tensor = param_buffer.as_strided(shape, stride, storage_offset)
                    param = nn.Parameter(tensor)
                    param.grad = grad_buffer.as_strided(shape, stride, storage_offset)
                    m.weight = param

    def get_local_param_buffer(self):
        return self._local_param_buffer

    def setup_grad(self) -> None:
        for mesh_id, local_grad in enumerate(self._local_grad_buffer):
            mesh = self._buffer_mesh[mesh_id]
            if mesh.ndevice == 1:
                self._local_param_buffer[mesh_id].grad = local_grad
            else:
                self._local_param_buffer[mesh_id].grad = mesh_wait(local_grad)

    def run_all_gather(self) -> None:
        self._global_param_buffer = []
        for mesh_id, local_tensor in enumerate(self._local_param_buffer):
            mesh = self._buffer_mesh[mesh_id]
            if mesh.ndevice == 1:
                self._global_param_buffer.append(local_tensor)
            else:
                self._global_param_buffer.append(
                    mesh_all_gather(local_tensor, self._buffer_shape[mesh_id], self._buffer_mesh[mesh_id], 0, 0)
                )

    def run_reduce_scatter(self) -> None:
        self._local_grad_buffer = []
        for mesh_id, global_tensor in enumerate(self._global_grad_buffer):
            mesh = self._buffer_mesh[mesh_id]
            if mesh.ndevice == 1:
                self._local_grad_buffer.append(global_tensor)
            else:
                self._local_grad_buffer.append(mesh_reduce_scatter(global_tensor, mesh, c10d.ReduceOp.SUM, 0, 0))


class MoEParamBuffer:
    def __init__(self, num_layers: int, num_experts_list: List[int]):
        self.num_layers = num_layers
        self._current_layer_id = 0
        self._buffer_list = [MoELayerParamBuffer(i, num_experts_list[i], self) for i in range(num_layers)]
        self._optimizer: Optional[torch.optim.Optimizer] = None

    def get_layer_param_buffer(self, layer_id: int) -> MoELayerParamBuffer:
        return self._buffer_list[layer_id]

    def get_param_group(self) -> Dict:
        params = []
        for buffer in self._buffer_list:
            params.extend(buffer.get_local_param_buffer())
        return params

    def setup_grad(self) -> None:
        is_dirty = False
        for buffer in self._buffer_list:
            buffer.setup_grad()
            is_dirty |= buffer.is_dirty()
        if is_dirty:
            self._optimizer.param_groups[0]["params"] = self.get_param_group()
            for buffer in self._buffer_list:
                buffer.reset_dirty()

    def set_optimizer(self, optimizer) -> None:
        self._optimizer = optimizer
        for buffer in self._buffer_list:
            buffer.set_optimizer(optimizer)

    def process_all_gather(self) -> None:
        self._current_layer_id = 0
        self._process_all_gather(self._current_layer_id)

    def _process_all_gather(self, layer_id: int) -> None:
        self._buffer_list[layer_id].run_all_gather()

    def finish_all_gather(self) -> None:
        if self._current_layer_id < self.num_layers - 1:
            self._current_layer_id += 1
            self._process_all_gather(self._current_layer_id)

    def process_reduce_scatter(self, layer_id: int) -> None:
        self._buffer_list[layer_id].run_reduce_scatter()
