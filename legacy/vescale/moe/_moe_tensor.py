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

from typing import Any, List, Deque, Dict, Mapping, Optional, Tuple, Union
from dataclasses import dataclass
from vescale import Placement, DeviceMesh
from vescale.moe._scheduler import MoEScheduler, MoETask

import torch
from torch import nn


class MoETensorPlaceholderState(int):
    def __new__(cls, state):
        obj = int.__new__(cls, state)
        return obj


_MOE_STATE_EXPERT_OUTPUT = MoETensorPlaceholderState(0)
_MOE_STATE_WEIGHTED_EXPERT_OUTPUT = MoETensorPlaceholderState(1)
_MOE_STATE_ACCUMULATING_RESULTS = MoETensorPlaceholderState(2)


@dataclass
class MoETensorPlaceholder:
    task: Optional[MoETask] = None
    scheduler: Optional[MoEScheduler] = None
    state: MoETensorPlaceholderState = _MOE_STATE_EXPERT_OUTPUT

    def __mul__(self, weight: torch.Tensor):
        assert self.state == _MOE_STATE_EXPERT_OUTPUT
        self.task.token_weight = weight
        self.state = _MOE_STATE_WEIGHTED_EXPERT_OUTPUT
        return self

    def __rmul__(self, weight: torch.Tensor):
        assert self.state == _MOE_STATE_EXPERT_OUTPUT
        self.task.token_weight = weight
        self.state = _MOE_STATE_WEIGHTED_EXPERT_OUTPUT
        return self

    def to(self, *args, **kwargs):
        return self


def moe_tensor_forward(expert: nn.Module, task: MoETask, scheduler: MoEScheduler):
    def forward(x):
        task.hidden_state = x
        task.device = x.device
        return MoETensorPlaceholder(task=task, scheduler=scheduler) if expert.training else expert._original_forward(x)

    return forward


def register_tensor_operations_with_placeholder(config):
    original_index_add_ = torch.Tensor.index_add_

    def wrappped_index_add_(self, dim: int, index: torch.Tensor, value: Union[torch.Tensor, MoETensorPlaceholder]):
        if isinstance(value, MoETensorPlaceholder):
            assert value.state == _MOE_STATE_WEIGHTED_EXPERT_OUTPUT
            if not hasattr(self, "_moe_tensor_placeholder"):
                scheduler = value.scheduler
                self._moe_tensor_placeholder = MoETensorPlaceholder(
                    task=None, state=_MOE_STATE_ACCUMULATING_RESULTS, scheduler=scheduler
                )
            else:
                assert self._moe_tensor_placeholder.state == _MOE_STATE_ACCUMULATING_RESULTS
                scheduler = self._moe_tensor_placeholder.scheduler
                assert scheduler is value.scheduler
            value.task.token_id = index
            value.task.output_tensor = self
            scheduler.push_task(value.task)
            if scheduler.num_tasks() == config["num_experts"]:
                tensor = self._moe_tensor_placeholder.scheduler.launch()
                self.copy_(tensor)
            return self
        else:
            with torch._C.DisableTorchFunctionSubclass():
                return original_index_add_(self, dim, index, value)

    torch.Tensor.index_add_ = wrappped_index_add_
