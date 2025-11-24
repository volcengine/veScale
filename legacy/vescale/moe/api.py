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
import warnings

from torch import nn
from vescale.moe._experts import Experts
from vescale.moe.experts_allocator import ExpertsAllocator, BasicExpertsAllocator
from vescale.debug import DebugLogger
from vescale.moe.token_dispatcher import BasicTokenDispatcher, TokenDispatcher

__all__ = ["parallelize_experts", "is_experts_parallized", "ExpertsAllocator"]


def parallelize_experts(
    module: nn.Module,
    experts_expr: Optional[Union[str, List[str]]] = None,
    experts_allocator: Optional[ExpertsAllocator] = None,
    token_dispatcher: Optional[TokenDispatcher] = None,
    config: Optional[Dict] = None,
) -> nn.Module:
    DebugLogger.update_vescale_debug_mode_from_env()

    if Experts.is_experts_parallized(module):
        warnings.warn(f"{module} has already parallelized experts. Skip `parallelize_experts`", UserWarning)
        return module

    Experts.init_scheduler(module, experts_expr, experts_allocator, token_dispatcher, config)

    Experts.init_forward(config)

    Experts.set_experts_parallized(module)

    return module


is_experts_parallized = Experts.is_experts_parallized
