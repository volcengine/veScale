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

from enum import Enum

NCCL_NUM_ALGORITHMS = 2  # Tree/Ring/CollNet*
NCCL_ALGO_UNDEF = -1
NCCL_ALGO_TREE = 0
NCCL_ALGO_RING = 1
# TODO: add more algorithms
# NCCL_ALGO_COLLNET_DIRECT = 2
# NCCL_ALGO_COLLNET_CHAIN = 3
# NCCL_ALGO_NVLS = 4
# NCCL_ALGO_NVLS_TREE = 5

NCCL_NUM_PROTOCOLS = 3  # Simple/LL/LL128
NCCL_PROTO_UNDEF = -1
NCCL_PROTO_LL = 0
NCCL_PROTO_LL128 = 1
NCCL_PROTO_SIMPLE = 2

sizeof_uint64_t = 8  # assume on 64bit platform

NCCL_STEPS = 8
ALLREDUCE_SLICESTEPS = int(NCCL_STEPS / 4)
ALLREDUCE_CHUNKSTEPS = int(NCCL_STEPS / 2)
REDUCESCATTER_SLICESTEPS = int(NCCL_STEPS / 4)
REDUCESCATTER_CHUNKSTEPS = int(NCCL_STEPS / 2)

WARP_SIZE = 32
MAXCHANNELS = 32
NCCL_MAX_NTHREADS = 640
NCCL_SIMPLE_MAX_NTHREADS = 512
NCCL_LL_MAX_NTHREADS = 512
NCCL_LL_LINES_PER_THREAD = 8

NCCL_LL128_LINESIZE = 128
NCCL_LL128_LINEELEMS = int(NCCL_LL128_LINESIZE / sizeof_uint64_t)
NCCL_LL128_DATAELEMS = NCCL_LL128_LINEELEMS - 1

NCCL_LL128_MAX_NTHREADS = 640
NCCL_LL128_ELEMS_PER_THREAD = 120

NCCL_LL128_SHMEM_ELEMS_PER_THREAD = 8

PCI_BW = 12.0

NCCL_LL_THREAD_THRESHOLD = 8
NCCL_LL128_THREAD_THRESHOLD = 8
NCCL_SIMPLE_THREAD_THRESHOLD = 64


NCCL_NUM_FUNCTIONS = 5


class NcclFunc(Enum):
    ncclFuncBroadcast = 0
    ncclFuncReduce = 1
    ncclFuncAllGather = 2
    ncclFuncReduceScatter = 3
    ncclFuncAllReduce = 4
    ncclFuncSendRecv = 5
    ncclFuncSend = 6
    ncclFuncRecv = 7
    ncclNumFuncs = 8


def log2i(n: int) -> int:
    l = 0
    n >>= 1
    while n:
        l += 1
        n >>= 1
    return l


def div_up(x, y):
    return (x + y - 1) // y


def round_up(x, y):
    return (x + y - 1) - (x + y - 1) % y


def align_up(x, a):
    return (x + a - 1) & (-a)


def ALIGN_SIZE(size, align):
    size = ((size + (align) - 1) // (align)) * (align)


# #define NCCL_MAX_WORK_ELEMENTS ((NCCL_WORK_SIZE - alignUp(sizeof(ncclWorkHeader), alignof(ncclWorkElem)))/sizeof(ncclWorkElem))
# static_assert(NCCL_MAX_WORK_ELEMENTS == 9, "Sanity check: NCCL_MAX_WORK_ELEMENTS == 9");
NCCL_MAX_WORK_ELEMENTS = 9


# Define constants to match the C code definitions
NCCL_TOPO_CPU_ARCH_X86 = 1
NCCL_TOPO_CPU_ARCH_POWER = 2
NCCL_TOPO_CPU_ARCH_ARM = 3
NCCL_TOPO_CPU_VENDOR_INTEL = 1
NCCL_TOPO_CPU_VENDOR_AMD = 2
NCCL_TOPO_CPU_VENDOR_ZHAOXIN = 3
NCCL_TOPO_CPU_TYPE_BDW = 1
NCCL_TOPO_CPU_TYPE_SKL = 2
NCCL_TOPO_CPU_TYPE_YONGFENG = 1


LINK_LOC = 0
LINK_NVL = 1
#  Skipping 2 for PATH_NVB
LINK_PCI = 3
#  Skipping 4 for PATH_PXB
#  Skipping 5 for PATH_PXN
#  Skipping 6 for PATH_PHB
LINK_SYS = 7
LINK_NET = 8

sizeof_union_ncclLLFifoLine = 16
DEFAULT_LL_BUFFSIZE = NCCL_LL_LINES_PER_THREAD * NCCL_LL_MAX_NTHREADS * NCCL_STEPS * sizeof_union_ncclLLFifoLine
DEFAULT_LL128_BUFFSIZE = NCCL_LL128_ELEMS_PER_THREAD * NCCL_LL128_MAX_NTHREADS * NCCL_STEPS * sizeof_uint64_t
DEFAULT_BUFFSIZE = 1 << 22  # 4MiB


class NcclPattern(Enum):
    Ring = 0
    RingTwice = 1
    # PipelineFrom = 2
    # PipelineTo = 3
    # TreeUp = 4
    # TreeDown = 5
    TreeUpDown = 6
    # CollnetChain = 7
    # CollnetDirect = 8
    # Nvls = 9
    # NvlsTree = 10
    # Send = 11
    # Recv = 12
