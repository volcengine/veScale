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


from vescale.emulator.nccl.constants import *  # noqa: F403
from vescale.emulator.nccl.include.graph import NCCL_TOPO_PATTERN_TREE
from vescale.emulator.nccl.include.info import NcclInfo
from vescale.emulator.nccl.init import init
from vescale.emulator.nccl.nccl_profiler_result import parse_nccl_topo


def topo_get_algo_info(info: NcclInfo, nchannels: int, algo: int, proto: int, nranks: int):
    info_nchannels = info.nChannels
    comm_nchannels = nchannels
    nc = info_nchannels if info_nchannels > 0 else comm_nchannels
    nt = info.comm.max_threads[algo][proto]
    thread_threshold = info.comm.thread_thresholds[algo][proto]

    info_nBytes = info.nBytes

    while info_nBytes < nc * nt * thread_threshold:
        if nc >= 2:
            nc -= 1
        else:
            if (nt % 128) == 0:
                nt /= 2
            else:
                break
    if proto == NCCL_PROTO_SIMPLE:
        if algo == NCCL_ALGO_RING:
            nt += WARP_SIZE
        if algo == NCCL_ALGO_TREE:
            nt += 4 * WARP_SIZE
    nt = 3 * WARP_SIZE if nt / WARP_SIZE < 3 else nt
    return nc, nt


def get_info_nchannels_nthreads_proto(pg, coll, count, dtype, nranks, nnodes):
    graphs, nchannels, minCompCap, maxCompCap = parse_nccl_topo(pg)
    info = init(coll, count, dtype, nchannels, nnodes, nranks, minCompCap, maxCompCap, graphs)

    nchannels, nthreads = topo_get_algo_info(info, nchannels, info.algorithm, info.protocol, nranks)
    info.nChannels = nchannels
    info.nThreads = nthreads
    return info, nchannels, nthreads, info.protocol


def calcBytePerStep(id, comm):
    if id == NCCL_PROTO_SIMPLE:
        return comm.buff_sizes[NCCL_PROTO_SIMPLE] / NCCL_STEPS
    elif id == NCCL_PROTO_LL:
        return comm.buff_sizes[NCCL_PROTO_LL] / NCCL_STEPS / 2
    else:
        return (comm.buff_sizes[NCCL_PROTO_LL128] / NCCL_STEPS) * NCCL_LL128_DATAELEMS / NCCL_LL128_LINEELEMS


def calcBytePerGrain(id):
    if id == NCCL_PROTO_SIMPLE:
        return sizeof_uint64_t
    elif id == NCCL_PROTO_LL:
        return sizeof_uint64_t
    else:
        return NCCL_LL128_SHMEM_ELEMS_PER_THREAD * NCCL_LL128_DATAELEMS * sizeof_uint64_t / NCCL_LL128_LINEELEMS


def get_pattern_info(info: NcclInfo):
    if info.coll == NcclFunc.ncclFuncReduceScatter:
        info.pattern = NcclPattern.Ring
    elif info.coll == NcclFunc.ncclFuncAllReduce:
        if info.algorithm == NCCL_ALGO_TREE:
            info.pattern = NcclPattern.TreeUpDown
        else:
            info.pattern = NcclPattern.RingTwice
    else:
        raise ValueError(f"Unsupported collective: {info.coll}")
    return info


def get_loop_info(info: NcclInfo):
    if info.pattern == NcclPattern.TreeUpDown:
        info.nstepsPerLoop = 1
        info.nchunksPerLoop = 1
    elif info.pattern == NcclPattern.Ring:
        info.nstepsPerLoop = info.comm.nRanks - 1
        info.nchunksPerLoop = info.comm.nRanks
    elif info.pattern == NcclPattern.RingTwice:
        info.nstepsPerLoop = 2 * (info.comm.nRanks - 1)
        info.nchunksPerLoop = info.comm.nRanks
    else:
        raise ValueError(f"Unsupported pattern: {info.pattern}")
    return info


def compute_last_chunk_size(info: NcclInfo):
    nNodes = info.comm.nNodes
    depth = info.comm.nRanks / nNodes - 1 + log2i(nNodes)

    info = get_pattern_info(info)
    info = get_loop_info(info)

    stepSize = info.comm.buff_sizes[info.protocol] / NCCL_STEPS
    if info.protocol == NCCL_PROTO_SIMPLE and info.algorithm == NCCL_ALGO_RING:
        chunkSteps = info.chunkSteps
        sliceSteps = info.sliceSteps
    else:
        chunkSteps = 1
        sliceSteps = 1
    chunkSize = stepSize * chunkSteps

    lastChunkSize = 0

    if info.algorithm == NCCL_ALGO_TREE and info.protocol == NCCL_PROTO_SIMPLE:
        if info.pattern == NCCL_TOPO_PATTERN_TREE:
            # Optimize chunkSize / nSteps
            while info.nBytes / (info.nChannels * chunkSize) < depth * 8 and chunkSize > 131072:
                chunkSize /= 2
            while info.nBytes / (info.nChannels * chunkSize) < depth * 4 and chunkSize > 65536:
                chunkSize /= 2
            while info.nBytes / (info.nChannels * chunkSize) < depth and chunkSize > 32768:
                chunkSize /= 2
        # Use lastChunkSize as chunkSize
        lastChunkSize = chunkSize / info.datatype.itemsize
    elif info.protocol == NCCL_PROTO_LL:
        sliceSize = stepSize * sizeof_uint64_t / sizeof_union_ncclLLFifoLine
        loopSize = info.nChannels * info.nchunksPerLoop * sliceSize
        lastChunkSize = div_up(
            (info.nBytes - (info.nBytes // loopSize) * loopSize), info.nChannels * info.nchunksPerLoop
        )
        ALIGN_SIZE(lastChunkSize, info.nThreads * sizeof_uint64_t)
        lastChunkSize /= info.datatype.itemsize
    elif info.algorithm == NCCL_ALGO_TREE and info.protocol == NCCL_PROTO_LL128:
        nNodes = info.comm.nNodes
        ppn = info.comm.nRanks / nNodes
        nstepsLL128 = 1 + log2i(nNodes) + 0.1 * ppn
        while (info.nBytes / (info.nChannels * chunkSize) < nstepsLL128 * 64 / ppn) and (chunkSize > 131072):
            chunkSize /= 2
        while (info.nBytes / (info.nChannels * chunkSize) < nstepsLL128 * 16 / ppn) and (chunkSize > 32768):
            chunkSize /= 2
        lastChunkSize = chunkSize * NCCL_LL128_DATAELEMS // (NCCL_LL128_LINEELEMS * info.datatype.itemsize)
    return lastChunkSize
