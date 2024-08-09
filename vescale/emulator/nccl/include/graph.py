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
# Some code comes from graph.h in NCCL
# Original license:
# Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
################################################################################

from dataclasses import dataclass

from vescale.emulator.nccl.constants import *  # noqa: F403

NCCL_TOPO_CPU_ARCH_X86 = 1
NCCL_TOPO_CPU_ARCH_POWER = 2
NCCL_TOPO_CPU_ARCH_ARM = 3
NCCL_TOPO_CPU_VENDOR_INTEL = 1
NCCL_TOPO_CPU_VENDOR_AMD = 2
NCCL_TOPO_CPU_VENDOR_ZHAOXIN = 3
NCCL_TOPO_CPU_TYPE_BDW = 1
NCCL_TOPO_CPU_TYPE_SKL = 2
NCCL_TOPO_CPU_TYPE_YONGFENG = 1

NCCL_TOPO_MAX_NODES = 256

NCCL_TOPO_PATTERN_BALANCED_TREE = 1
NCCL_TOPO_PATTERN_SPLIT_TREE = 2
NCCL_TOPO_PATTERN_TREE = 3
NCCL_TOPO_PATTERN_RING = 4
NCCL_TOPO_PATTERN_NVLS = 5


@dataclass
class NcclTopoGraph:
    # Input / output
    # id: int
    pattern: int  # used
    # crossNic: int
    # collNet: int
    # minChannels: int
    # maxChannels: int

    # Output
    nChannels: int  # used
    bwIntra: float  # used
    bwInter: float  # used
    latencyInter: float  # used
    typeIntra: int  # used
    # typeInter: int
    sameChannels: int  # used
    # nHops: int

    # Arrays
    # intra: List[int] = field(default_factory=lambda: [0] * (MAXCHANNELS * NCCL_TOPO_MAX_NODES))
    # inter: List[int] = field(default_factory=lambda: [0] * (MAXCHANNELS * 2))
