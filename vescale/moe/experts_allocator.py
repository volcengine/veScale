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

from typing import Dict, Optional, Union, List
import torch
import torch.distributed as dist
from abc import ABC, abstractmethod
from vescale import DeviceMesh
from vescale.moe._utils import _MOE_DP, _MOE_TP


class ExpertsAllocator(ABC):
    @abstractmethod
    def __init__(self, model_config=None, env_config=None):
        pass

    @abstractmethod
    def collect_performance(self, perf, iter=-1):
        pass

    @abstractmethod
    def allocate_experts(self, layer_id, iter=-1):
        pass

    def allocate_experts_internal(self, layer_id, iter=-1) -> Optional[Dict]:
        experts_alloc = self.allocate_experts(layer_id, iter)
        if experts_alloc is None:
            return None
        new_experts_alloc = []
        dp_size, tp_size = [], []
        for alloc in experts_alloc:
            assert alloc.ndim == 2
            dp_size.append(alloc.mesh.shape[0])
            tp_size.append(alloc.mesh.shape[1])
            device_type = alloc.device_type
            mesh_dim_names = (_MOE_TP, _MOE_DP)
            new_alloc = DeviceMesh(device_type, alloc.mesh.t(), mesh_dim_names=mesh_dim_names)
            new_experts_alloc.append(new_alloc)
        dp_size = torch.tensor(dp_size, device=device_type)
        tp_size = torch.tensor(tp_size, device=device_type)
        experts_alloc_info = {
            "experts_alloc": new_experts_alloc,
            "dp_size": dp_size,
            "tp_size": tp_size,
        }
        return experts_alloc_info


class BasicExpertsAllocator(ExpertsAllocator):
    def __init__(self, exp_config=None, env_config=None):
        self.experts_num = 8
        self.visit_flag = set()

        self.experts_allocation = []  # A list of DP * TP
        world_size = dist.get_world_size()
        devices = torch.arange(world_size)
        for _ in range(self.experts_num):
            self.experts_allocation.append(DeviceMesh("cuda", devices.reshape(1, -1), mesh_dim_names=("DP", "TP")))

    def collect_performance(self, perf, iter=-1):
        pass

    def allocate_experts(self, layer_id, iter=-1) -> Union[None, List[DeviceMesh]]:
        if layer_id not in self.visit_flag:
            self.visit_flag.add(layer_id)
            return self.experts_allocation
        else:
            return None
