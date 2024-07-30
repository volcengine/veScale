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

from enum import Enum, auto
from dataclasses import dataclass
from typing import TypeVar


__all__ = [
    "ModeType",
    "PipelineP2PSpec",
    "PipelineSplitMethodType",
    "PipelineScheduleType",
    "TracerType",
]

ArrayLike = TypeVar("ArrayLike")


class ModeType(Enum):
    """Type of parallel modes"""

    EAGER = auto()
    MANUAL_EAGER = auto()
    GRAPH_EAGER = auto()


class PipelineSplitMethodType(Enum):
    """Type of pipeline stage partitioning methods"""

    MANUAL = auto()
    UNIFORM = auto()
    PARAMETERS = auto()
    AUTO = auto()
    SIMULATOR = auto()
    FLOPS = auto()


class PipelineScheduleType(Enum):
    """Type of pipeline parallel schedules"""

    SIMPLE_1F1B = auto()
    INTERLEAVED_1F1B = auto()
    GPIPE = auto()
    ZERO_BUBBLE = auto()
    GRAPH_PIPE = auto()


class TracerType(Enum):
    VESCALE_FX = auto()
    VESCALE_EXPORT = auto()
    HF_FX = auto()
    TORCH_FX = auto()
    TORCH_DYNAMO = auto()
    TORCH_EXPORT = auto()
    AUTO = auto()


@dataclass
class PipelineP2PSpec:
    """The p2p spec for communication p2p spec in manual pipeline plan."""

    peer_stage_idx: int
    peer_output_idx: int = 0
