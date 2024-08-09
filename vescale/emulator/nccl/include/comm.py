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
# Some code comes from comm.h in NCCL
# Original license:
# Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
################################################################################

import collections
from dataclasses import dataclass


@dataclass
class NcclComm:
    nChannels: int = 0
    nNodes: int = 0
    nRanks: int = 0
    minCompCap: int = 0

    bandwidths = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(lambda: 0)))
    latencies = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(lambda: 0)))
    ringbdw = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
    max_threads = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
    thread_thresholds = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
    buff_sizes = collections.defaultdict(lambda: 0)
