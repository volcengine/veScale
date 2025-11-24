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

from collections import OrderedDict
import re
import warnings
from typing import List, Dict, Optional, Union

import torch
from torch import nn

from vescale.ddp.distributed_data_parallel import DistributedDataParallel as DDP
from vescale.moe.experts_allocator import ExpertsAllocator, BasicExpertsAllocator
from vescale.moe.token_dispatcher import TokenDispatcher, BasicTokenDispatcher
from vescale.moe._moe_tensor import moe_tensor_forward, register_tensor_operations_with_placeholder
from vescale.moe._scheduler import MoEScheduler, MoETask
from vescale.moe._utils import _TAG_EXPERTS_PARALLIZED


__all__ = ["Experts"]


class Experts:
    @staticmethod
    def is_experts_parallized(module: nn.Module) -> bool:
        return hasattr(module, _TAG_EXPERTS_PARALLIZED)

    @staticmethod
    def set_experts_parallized(module: nn.Module) -> None:
        if hasattr(module, _TAG_EXPERTS_PARALLIZED):
            warnings.warn(f"resetting `{module.__class__}` as parallized experts!", UserWarning)
        setattr(module, _TAG_EXPERTS_PARALLIZED, True)

    @staticmethod
    @torch.no_grad()
    def init_scheduler(
        input_module: nn.Module,
        experts: Optional[Union[str, List[str]]],
        experts_allocator: Optional[ExpertsAllocator] = None,
        token_dispatcher: Optional[TokenDispatcher] = None,
        config: Optional[Dict] = None,
    ) -> None:
        if experts_allocator is None:
            experts_allocator = BasicExpertsAllocator()
        if token_dispatcher is None:
            token_dispatcher = BasicTokenDispatcher()

        experts = experts if type(experts) is List else [experts]

        if isinstance(input_module, DDP):
            core_module = input_module.module
        else:
            core_module = input_module

        moe_layer_list = []
        for experts_pattern in experts:
            for submod_fqn, submod in core_module.named_modules():
                if re.fullmatch(experts_pattern, submod_fqn):
                    if isinstance(submod, nn.ModuleList):
                        for i in range(len(submod)):
                            for nm, m in submod[i].named_modules():
                                weight = submod_fqn + f".{i}." + nm
                                if weight in core_module._param_sharding_plan:
                                    m._weight_placement = core_module._param_sharding_plan[weight]["weight"].placements[
                                        0
                                    ]
                                    m._backward_hooks = OrderedDict()
                                    m._forward_hooks = OrderedDict()
                                    m._forward_pre_hooks = OrderedDict()
                        moe_layer_list.append(submod)
                    elif isinstance(submod, nn.Parameter):
                        # TODO: override bmm
                        raise NotImplementedError
                    else:
                        raise ValueError

        scheduler = MoEScheduler(experts_allocator, token_dispatcher, config)
        for layer_id, moe_layer in enumerate(moe_layer_list):
            for i, expert_model in enumerate(moe_layer):
                task = MoETask(model=expert_model, layer_id=layer_id, expert_id=i)
                expert_model._original_forward = expert_model.forward
                expert_model.forward = moe_tensor_forward(expert_model, task, scheduler)
        scheduler.init_param_buffer(moe_layer_list)
        input_module.moe_param_buffer = scheduler.get_moe_param_buffer()

    @staticmethod
    @torch.no_grad()
    def init_forward(config) -> None:
        register_tensor_operations_with_placeholder(config)
