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
# Some code comes from tuning.cc in NCCL
# Original license:
# Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
################################################################################

from typing import List

from vescale.emulator.nccl.include.info import NcclInfo
from vescale.emulator.nccl.include.comm import NcclComm
from vescale.emulator.nccl.include.graph import *  # noqa: F403
from vescale.emulator.nccl.constants import *  # noqa: F403
import platform
import subprocess

VOLTA_COMPCAP_IDX = 0
AMPERE_COMPCAP_IDX = 1
HOPPER_COMPCAP_IDX = 2

llMaxBws = [[39.0, 39.0, 20.4], [87.7, 22.5, 19.0], [87.7, 22.5, 19.0]]

perChMaxRingLL128Bws = [[20.0, 20.0, 20.0], [20.0, 20.0, 20.0], [36.7, 36.7, 36.7]]

perChMaxTreeLL128Bws = [[20.0, 20.0, 20.0], [20.0, 20.0, 20.0], [36.7, 36.7, 29.0]]

perChMaxTreeBws = [[26.5, 18.5, 10.0], [24.0, 23.6, 17.8], [38.7, 41.4, 36.0]]

# Constants
NCCL_HW_NVLINK = 0
NCCL_HW_PCI = 1
NCCL_HW_NET = 2
VOLTA_COMPCAP_IDX = 0
AMPERE_COMPCAP_IDX = 1
HOPPER_COMPCAP_IDX = 2

baseLat = [[6.8, 14.0, 0.0], [6.6, 14.0, 8.4], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

hwLat = [
    # NVLINK
    [[0.6, 1.25, 28.0], [0.6, 1.9, 3.4], [0.0, 0.0, 3.7], [0.0, 0.0, 2.8], [0.0, 0.0, 23.0], [0.0, 0.0, 23.0]],
    # PCI
    [[1.0, 1.9, 28.0], [1.0, 2.5, 5.7], [0.0, 0.0, 3.7], [0.0, 0.0, 2.8], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    # NET
    [[5.0, 8.5, 28.0], [2.7, 4.0, 14.0], [0.0, 0.0, 31.0], [0.0, 0.0, 30.0], [0.0, 0.0, 18.0], [0.0, 0.0, 14.0]],
]

NCCL_NTHREADS = -2
NCCL_LL128_NTHREADS = -2


def get_nthreads(name: str, env: int, min_v: int, max_v: int, default: int):
    nt = env
    if nt > 0:
        if nt % WARP_SIZE != 0:
            nt = max_v
        elif nt > max_v:
            nt = max_v
        elif nt < min_v:
            nt = min_v
    else:
        nt = default
    return nt


# Define the mappings
cpu_arch_map = {
    "x86_64": NCCL_TOPO_CPU_ARCH_X86,
    "ppc64le": NCCL_TOPO_CPU_ARCH_POWER,
    "aarch64": NCCL_TOPO_CPU_ARCH_ARM,
}

cpu_vendor_map = {
    "GenuineIntel": NCCL_TOPO_CPU_VENDOR_INTEL,
    "AuthenticAMD": NCCL_TOPO_CPU_VENDOR_AMD,
    "Shanghai": NCCL_TOPO_CPU_VENDOR_ZHAOXIN,  # Example for Zhaoxin vendor
}

cpu_model_map = {
    "Broadwell": NCCL_TOPO_CPU_TYPE_BDW,
    "Skylake": NCCL_TOPO_CPU_TYPE_SKL,
    "Yongfeng": NCCL_TOPO_CPU_TYPE_YONGFENG,  # Example for Yongfeng model
}


def get_cpu_info():
    """Get CPU information using platform and subprocess modules."""
    cpu_arch = platform.machine()

    # Get CPU vendor and model using lscpu on Linux
    if platform.system() == "Linux":
        try:
            lscpu_output = subprocess.check_output("lscpu", shell=True).decode().split("\n")
            cpu_vendor = ""
            cpu_model = ""
            for line in lscpu_output:
                if "Vendor ID:" in line:
                    cpu_vendor = line.split(":")[1].strip()
                elif "Model name:" in line:
                    cpu_model = line.split(":")[1].strip().split()[0]  # Simplified for this example
            return cpu_arch, cpu_vendor, cpu_model
        except Exception as e:
            print(f"Error retrieving CPU info: {e}")
            return cpu_arch, "Unknown Vendor", "Unknown Model"

    # For other OSes (macOS, Windows), use platform module or other methods
    elif platform.system() == "Darwin":  # macOS
        try:
            cpu_vendor = subprocess.check_output("sysctl -n machdep.cpu.vendor", shell=True).decode().strip()
            cpu_model = (
                subprocess.check_output("sysctl -n machdep.cpu.brand_string", shell=True).decode().strip().split()[0]
            )
            return cpu_arch, cpu_vendor, cpu_model
        except Exception as e:
            print(f"Error retrieving CPU info: {e}")
            return cpu_arch, "Unknown Vendor", "Unknown Model"

    elif platform.system() == "Windows":
        try:
            cpu_vendor = platform.processor()
            cpu_model = platform.processor()  # Windows often reports the same for both
            return cpu_arch, cpu_vendor, cpu_model
        except Exception as e:
            print(f"Error retrieving CPU info: {e}")
            return cpu_arch, "Unknown Vendor", "Unknown Model"

    else:
        return cpu_arch, "Unknown Vendor", "Unknown Model"


def ncclTopoCpuType():
    cpu_arch, cpu_vendor, cpu_model = get_cpu_info()

    arch_code = cpu_arch_map.get(cpu_arch, None)
    vendor_code = cpu_vendor_map.get(cpu_vendor, None)
    model_code = cpu_model_map.get(cpu_model, None)

    return arch_code, vendor_code, model_code


def getNetOverhead(comm: NcclComm):
    cpuArch, cpuVendor, cpuModel = ncclTopoCpuType()
    if cpuArch == NCCL_TOPO_CPU_ARCH_X86 and cpuVendor == NCCL_TOPO_CPU_VENDOR_INTEL:
        return 1.0
    elif cpuArch == NCCL_TOPO_CPU_ARCH_X86 and cpuVendor == NCCL_TOPO_CPU_VENDOR_AMD:
        return 2.0
    else:
        return 1.0


def nccl_topo_tune_model(comm: NcclComm, minCompCap: int, maxCompCap: int, graphs: List[NcclTopoGraph]):
    if graphs[NCCL_ALGO_RING].bwIntra * graphs[NCCL_ALGO_RING].nChannels <= PCI_BW:
        simpleDefaultThreads = 256
    else:
        simpleDefaultThreads = NCCL_SIMPLE_MAX_NTHREADS

    comm.max_threads[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] = get_nthreads(
        "NCCL_NTHREADS", NCCL_NTHREADS, 2 * WARP_SIZE, NCCL_SIMPLE_MAX_NTHREADS, simpleDefaultThreads
    )
    comm.max_threads[NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE] = get_nthreads(
        "NCCL_NTHREADS", NCCL_NTHREADS, 2 * WARP_SIZE, NCCL_SIMPLE_MAX_NTHREADS, NCCL_SIMPLE_MAX_NTHREADS
    )
    comm.max_threads[NCCL_ALGO_RING][NCCL_PROTO_LL] = comm.max_threads[NCCL_ALGO_TREE][NCCL_PROTO_LL] = get_nthreads(
        "NCCL_NTHREADS", NCCL_NTHREADS, 2 * WARP_SIZE, NCCL_LL_MAX_NTHREADS, NCCL_LL_MAX_NTHREADS
    )
    comm.max_threads[NCCL_ALGO_RING][NCCL_PROTO_LL128] = comm.max_threads[NCCL_ALGO_TREE][NCCL_PROTO_LL128] = (
        get_nthreads(
            "NCCL_LL128_NTHREADS",
            NCCL_LL128_NTHREADS,
            NCCL_LL128_MAX_NTHREADS / 4,
            NCCL_LL128_MAX_NTHREADS,
            NCCL_LL128_MAX_NTHREADS,
        )
    )

    nNodes = comm.nNodes
    nRanks = comm.nRanks
    if nRanks <= 1:
        return

    compCapIndex = (
        HOPPER_COMPCAP_IDX if minCompCap >= 90 else AMPERE_COMPCAP_IDX if minCompCap >= 80 else VOLTA_COMPCAP_IDX
    )
    cpuArch, cpuVendor, cpuModel = ncclTopoCpuType()

    index2 = nNodes - 1 if nNodes <= 2 else 2

    index1 = compCapIndex if nNodes == 1 else 1 if cpuVendor == NCCL_TOPO_CPU_VENDOR_AMD else 0

    llMaxBw = llMaxBws[index1][index2]
    perChMaxTreeBw = perChMaxTreeBws[compCapIndex][index2]
    perChMaxRingLL128Bw = perChMaxRingLL128Bws[compCapIndex][index2]
    perChMaxTreeLL128Bw = perChMaxTreeLL128Bws[compCapIndex][index2]

    if cpuArch == NCCL_TOPO_CPU_ARCH_POWER:
        hwLat[NCCL_HW_PCI][NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE] = hwLat[NCCL_HW_PCI][NCCL_ALGO_RING][NCCL_PROTO_SIMPLE]

    ppn = float(nRanks) / nNodes

    intraHw = [NCCL_HW_NVLINK if graphs[a].typeIntra == LINK_NVL else NCCL_HW_PCI for a in range(NCCL_NUM_ALGORITHMS)]
    hw = [intraHw[a] if nNodes == 1 else NCCL_HW_NET for a in range(NCCL_NUM_ALGORITHMS)]

    for coll_i in range(NCCL_NUM_FUNCTIONS):
        coll = NcclFunc(coll_i)
        if coll == NcclFunc.ncclFuncAllReduce:
            nsteps = 2 * (nRanks - 1)
        elif coll == NcclFunc.ncclFuncReduceScatter or coll == NcclFunc.ncclFuncAllGather:
            nsteps = nRanks - 1
        else:
            nsteps = nRanks

        if coll == NcclFunc.ncclFuncAllReduce:
            if nNodes > 1:
                nInterSteps = 2 * nNodes
            else:
                nInterSteps = 0
        elif coll == NcclFunc.ncclFuncReduceScatter or coll == NcclFunc.ncclFuncAllGather:
            nInterSteps = nNodes - 1
        else:
            nInterSteps = nNodes

        for a in range(NCCL_NUM_ALGORITHMS):
            if coll == NcclFunc.ncclFuncBroadcast and a != NCCL_ALGO_RING:
                continue
            if coll == NcclFunc.ncclFuncReduce and a != NCCL_ALGO_RING:
                continue
            if coll == NcclFunc.ncclFuncReduceScatter and a != NCCL_ALGO_RING:  # and a != NCCL_ALGO_NVLS:
                continue
            if coll == NcclFunc.ncclFuncAllGather and a != NCCL_ALGO_RING:  # and a != NCCL_ALGO_NVLS:
                continue

            for p in range(NCCL_NUM_PROTOCOLS):
                # if (a == NCCL_ALGO_NVLS or a == NCCL_ALGO_NVLS_TREE) and p != NCCL_PROTO_SIMPLE:
                #     continue
                # collnet = (a == NCCL_ALGO_COLLNET_DIRECT or a == NCCL_ALGO_COLLNET_CHAIN)
                if nNodes <= 2:  # or collnet:
                    bw = graphs[a].bwIntra
                else:
                    bw = graphs[a].bwInter
                # if a == NCCL_ALGO_NVLS:
                #     bw = min(graphs[a].bwIntra, graphs[a].bwInter)
                # if a == NCCL_ALGO_NVLS_TREE:
                #     if nNodes <= 2:
                #         tmp_bwInter = graphs[a].bwInter
                #     else:
                #         tmp_bwInter = graphs[a].bwInter/2
                #     bw = min(graphs[a].bwIntra, tmp_bwInter)
                busBw = graphs[a].nChannels * bw

                # Various model refinements
                if a == NCCL_ALGO_RING and p == NCCL_PROTO_LL:
                    busBw = min(
                        llMaxBw,
                        busBw
                        * (
                            1.0 / 4.0
                            if (nNodes > 1 or coll in [NcclFunc.ncclFuncAllReduce, NcclFunc.ncclFuncReduce])
                            else 1.0 / 3.0
                        ),
                    )
                if a == NCCL_ALGO_RING and p == NCCL_PROTO_LL128:
                    busBw = min(busBw * (0.7 if ppn < 2 else 0.92), graphs[a].nChannels * perChMaxRingLL128Bw)
                if a == NCCL_ALGO_TREE:
                    busBw = min(busBw * 0.92, graphs[a].nChannels * perChMaxTreeBw)
                if a == NCCL_ALGO_TREE and p == NCCL_PROTO_LL:
                    busBw = min(busBw * 1.0 / 3.8, llMaxBw)
                if a == NCCL_ALGO_TREE and p == NCCL_PROTO_LL128:
                    busBw = min(
                        busBw * (7.0 / 9.0 if nNodes == 1 else 120.0 / 128.0), graphs[a].nChannels * perChMaxTreeLL128Bw
                    )
                if a == NCCL_ALGO_TREE and graphs[a].pattern == NCCL_TOPO_PATTERN_TREE:
                    busBw *= 0.85
                # skip collnet direct/chain for now

                # Convert bus BW to algorithm BW
                if a == NCCL_ALGO_RING:
                    ratio = (1.0 * nRanks) / nsteps
                # elif a == NCCL_ALGO_NVLS or a == NCCL_ALGO_NVLS_TREE:
                #     ratio = 5.0/6.0
                else:
                    ratio = 0.5
                comm.bandwidths[coll][a][p] = busBw * ratio
                # Ring bandwidth backup
                if a == NCCL_ALGO_RING:
                    comm.ringbdw[coll][p] = comm.bandwidths[coll][NCCL_ALGO_RING][p]
                comm.latencies[coll][a][p] = baseLat[a][p]
                intraLat = hwLat[intraHw[a]][a][p]
                interLat = hwLat[NCCL_HW_NET][a][p] + graphs[a].latencyInter
                # Also add the flush extra latency
                if p == NCCL_PROTO_SIMPLE:
                    interLat += graphs[a].latencyInter

                if a == NCCL_ALGO_RING:
                    lat = hwLat[hw[a]][a][p]
                    if coll == NcclFunc.ncclFuncReduce or coll == NcclFunc.ncclFuncBroadcast:
                        if graphs[a].sameChannels:
                            comm.latencies[coll][a][p] += lat
                        else:
                            if p == NCCL_PROTO_SIMPLE:
                                lat = hwLat[hw[a]][NCCL_ALGO_TREE][
                                    p
                                ]  # Add some chunk latency, waiting for proper chunk modeling
                            comm.latencies[coll][a][p] += nsteps * lat
                    else:
                        # Inter-node rings still have to launch nsteps * net overhead.
                        netOverhead = 0.0
                        if nNodes > 1:
                            netOverhead = getNetOverhead(comm)
                            if p == NCCL_PROTO_SIMPLE:
                                netOverhead *= 3
                        intraLat = max(intraLat, netOverhead)
                        comm.latencies[coll][a][p] += (nsteps - nInterSteps) * intraLat + nInterSteps * interLat
                elif a == NCCL_ALGO_TREE:
                    comm.latencies[coll][a][p] += 2 * ((nRanks / nNodes - 1) * intraLat + log2i(nNodes) * interLat)
                else:
                    # skip collnet direct/chain for now
                    raise NotImplementedError

    comm.thread_thresholds[NCCL_ALGO_RING][NCCL_PROTO_LL] = comm.thread_thresholds[NCCL_ALGO_TREE][NCCL_PROTO_LL] = (
        NCCL_LL_THREAD_THRESHOLD
    )
    comm.thread_thresholds[NCCL_ALGO_RING][NCCL_PROTO_LL128] = comm.thread_thresholds[NCCL_ALGO_TREE][
        NCCL_PROTO_LL128
    ] = NCCL_LL128_THREAD_THRESHOLD
    comm.thread_thresholds[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] = comm.thread_thresholds[NCCL_ALGO_TREE][
        NCCL_PROTO_SIMPLE
    ] = NCCL_SIMPLE_THREAD_THRESHOLD
    comm.thread_thresholds[NCCL_ALGO_RING][NCCL_PROTO_LL] *= nRanks
    return comm


tree_correction_factor = [
    [1.0, 1.0, 1.0, 1.0, 0.9, 0.8, 0.7, 0.7, 0.7, 0.7, 0.6, 0.5, 0.4, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.8, 0.8, 0.8, 0.7, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.8, 0.9, 0.9, 0.9, 0.9, 1.0, 1.0],
    [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.8, 0.7, 0.6, 0.6, 0.5, 0.5, 0.5, 0.5, 0.6, 0.7, 0.8, 0.7, 0.7, 0.8, 0.9, 0.9],
]


def DIVUP(x, y):
    return ((x) + (y) - 1) / (y)


def nccl_topo_get_algo_time(info: NcclInfo, algorithm: int, protocol: int, numPipeOps: int):
    bw = info.comm.bandwidths[info.coll][algorithm][protocol]
    lat = info.comm.latencies[info.coll][algorithm][protocol]
    backup = False

    if algorithm == NCCL_ALGO_RING and bw == 0.0:
        # Try backup RING algorithm
        bw = info.comm.ringbdw[info.coll][protocol]
        backup = True

    if bw == 0:
        return -1.0, backup

    logSize = int(log2i(info.nBytes >> 6))
    if algorithm == NCCL_ALGO_TREE and logSize < 23:
        bw *= tree_correction_factor[protocol][logSize]
    # if info.nChannels != 0:
    #     bw = bw / info.comm.nChannels * info.nChannels
    if (
        algorithm == NCCL_ALGO_RING
        and protocol == NCCL_PROTO_SIMPLE
        and info.comm.nNodes > 1
        and info.coll == NcclFunc.ncclFuncAllReduce
        and info.nBytes / (info.comm.nChannels * info.comm.nRanks) >= 64
    ):
        lat *= 1.9 if info.comm.minCompCap < 80 else 1.4  # Plateau effect of ring

    latCount = numPipeOps if algorithm == NCCL_ALGO_RING else DIVUP(numPipeOps, NCCL_MAX_WORK_ELEMENTS)
    time = lat * latCount + info.nBytes / (1000 * bw)
    return time, backup
