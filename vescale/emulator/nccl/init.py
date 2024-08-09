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
# Some code comes from NCCL
# Original license:
# Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
################################################################################

from vescale.emulator.nccl.constants import *  # noqa: F403
from vescale.emulator.nccl.include.comm import NcclComm
from vescale.emulator.nccl.include.info import NcclInfo, nccl_info_set_derived

from vescale.emulator.nccl.graph.tuning import nccl_topo_get_algo_time, nccl_topo_tune_model


def compute_buff_sizes(comm: NcclComm):
    # envs = [ncclParamLlBuffSize(), ncclParamLl128BuffSize(), ncclParamBuffSize()] # load from env variables
    envs = [-2, -2, -2]
    defaults = [DEFAULT_LL_BUFFSIZE, DEFAULT_LL128_BUFFSIZE, DEFAULT_BUFFSIZE]

    for p in range(0, NCCL_NUM_PROTOCOLS, 1):
        if envs[p] != -2:
            comm.buff_sizes[p] = envs[p]
        else:
            comm.buff_sizes[p] = defaults[p]
    return comm


def init(coll, count, dtype, nChannels, nNodes, nRanks, minCompCap, maxCompCap, graphs):
    if coll == NcclFunc.ncclFuncAllReduce:
        chunkSteps = ALLREDUCE_CHUNKSTEPS
        sliceSteps = ALLREDUCE_SLICESTEPS
    elif coll == NcclFunc.ncclFuncReduceScatter:
        chunkSteps = REDUCESCATTER_CHUNKSTEPS
        sliceSteps = REDUCESCATTER_SLICESTEPS
    else:
        raise Exception("Unsupported collective operation")
    comm = NcclComm(nChannels, nNodes, nRanks, minCompCap)
    info = NcclInfo(coll, comm, chunkSteps, sliceSteps, count=count, datatype=dtype)
    info = nccl_info_set_derived(info, nRanks)
    info.comm = compute_buff_sizes(info.comm)
    info.comm = nccl_topo_tune_model(info.comm, minCompCap, maxCompCap, graphs)

    if comm.nRanks == 1:
        info.algorithm = NCCL_ALGO_RING
        info.protocol = NCCL_PROTO_SIMPLE
    else:
        info.algorithm = NCCL_ALGO_UNDEF
        info.protocol = NCCL_PROTO_UNDEF
        min_time = 3600000000.0
        backup_min_time = 3600000000.0
        backup = False
        backupAlgo = NCCL_ALGO_UNDEF
        backupProto = NCCL_PROTO_UNDEF
        info.algorithm = -1
        info.protocol = -1
        nAlgos = NCCL_NUM_ALGORITHMS
        for a in range(nAlgos):
            for p in range(NCCL_NUM_PROTOCOLS):
                time, backup = nccl_topo_get_algo_time(info, a, p, 1)
                if not backup:
                    if time >= 0 and time < min_time:
                        info.algorithm = a
                        info.protocol = p
                        min_time = time
                else:
                    if time >= 0 and time < backupMinTime:
                        backupAlgo = a
                        backupProto = p
                        backupMinTime = time
    return info
