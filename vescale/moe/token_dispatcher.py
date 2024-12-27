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

from typing import Dict, List, Optional, Tuple
import torch
import torch.distributed as dist
from abc import ABC, abstractmethod
from vescale import DeviceMesh


class TokenDispatcher(ABC):
    @abstractmethod
    def __init__(self, exp_config=None, env_config=None):
        pass

    @abstractmethod
    def assign_task(self, layer_id, token_id, expert_id, hidden_state, token_weight):
        pass

    @abstractmethod
    def set_experts_alloc(self, experts_alloc):
        pass

    @abstractmethod
    def dispatch_token(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class BasicTokenDispatcher(TokenDispatcher):
    def __init__(self, exp_config=None, env_config=None):
        self.experts_alloc: Optional[List[Optional[DeviceMesh]]] = None
        self.expert_id: Optional[torch.Tensor] = None
        self.num_replicate: Optional[torch.Tensor] = None

    def assign_task(self, layer_id, token_id, expert_id, hidden_state, token_weight):
        self.expert_id = expert_id

    def set_experts_alloc(self, experts_alloc_info: Dict) -> None:
        self.experts_alloc = experts_alloc_info["experts_alloc"]
        self.num_replicate = experts_alloc_info["dp_size"]

    def dispatch_token(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        num_replicate = self.num_replicate[self.expert_id]
        replicate_id = torch.randint_like(num_replicate, 65535) % num_replicate
        return self.expert_id, replicate_id
