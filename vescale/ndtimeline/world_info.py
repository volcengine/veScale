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

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=False)
class TopoInfo:
    rank: int = 0
    dp_rank: int = 0
    ddp_rank: int = 0
    tp_rank: int = 0
    pp_rank: int = 0
    local_rank: int = 0
    ip: str = "0.0.0.0"
    dp_size: int = 1
    ddp_size: int = 1
    tp_size: int = 1
    pp_size: int = 1
    world_size: int = 1

    def __post_init__(self):
        # validation
        for field_name in self.__dict__:
            field_content = self.__dict__[field_name]
            if field_name.endswith("rank") and field_content < 0:
                raise ValueError(f"TopoInfo instance's {field_name}={field_content}, expected nonnegative number")
            if field_name.endswith("size") and field_content <= 0:
                raise ValueError(f"WorldInfo instance's {field_name}={field_content}, expected positive number")


@dataclass(frozen=False)
class TrainingInfo:
    role_id: int = 0
    trial_id: int = 0
    run_id: int = 0

    def __post_init__(self):
        # validation
        for field_name in self.__dict__:
            field_content = self.__dict__[field_name]
            if field_content < 0:
                raise ValueError(f"TrainingInfo instance's {field_name}={field_content}, expected nonnegative number")


class WorldInfo:
    def __init__(
        self,
        rank: int,
        local_rank: int,
        dp_rank: int = 0,
        ddp_rank: int = 0,
        tp_rank: int = 0,
        pp_rank: int = 0,
        dp_size: int = 1,
        ddp_size: int = 1,
        tp_size: int = 1,
        pp_size: int = 1,
        world_size: int = 1,
        ip: str = "0.0.0.0",
        role_id: int = 0,
        run_id: int = 0,
        trial_id: int = 0,
        **extra_meta: Dict[str, Any],
    ):
        self.topo_info = TopoInfo(
            rank=rank,
            local_rank=local_rank,
            dp_rank=dp_rank,
            ddp_rank=ddp_rank,
            tp_rank=tp_rank,
            pp_rank=pp_rank,
            dp_size=dp_size,
            ddp_size=ddp_size,
            tp_size=tp_size,
            pp_size=pp_size,
            world_size=world_size,
            ip=ip,
        )
        self.training_info = TrainingInfo(
            role_id=role_id,
            trial_id=trial_id,
            run_id=run_id,
        )
        self.extra_info = {}
        for k in extra_meta:
            self.extra_info[k] = extra_meta[k]

    def __repr__(self) -> str:
        return f"WorldInfo: {self.topo_info.__repr__()} {self.training_info.__repr__()} {self.extra_info.__repr__()}"

    def __getitem__(self, key: str):
        if key in self.topo_info.__dict__:
            return self.topo_info.__dict__[key]
        if key in self.training_info.__dict__:
            return self.training_info.__dict__[key]
        if key in self.extra_info:
            return self.extra_info[key]
        raise KeyError(f"{key} is not found")

    def __setitem__(self, key: str, value: Any):
        if key in self.topo_info.__dict__:
            self.topo_info.__dict__[key] = value
        if key in self.training_info.__dict__:
            self.training_info.__dict__[key] = value
        if key in self.extra_info:
            self.extra_info[key] = value
        raise KeyError(f"{key} is not found")
