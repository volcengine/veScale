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
# Some code comes from reduce_scatter.cc in NCCL
# Original license:
# Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
################################################################################

from typing import List

import torch
from vescale.emulator.nccl.include.info import NcclInfo
from vescale.emulator.reduce_kernel import ReduceOp
from vescale.emulator.topo import Ring
from vescale.emulator.primitives import RingPrimitive
from vescale.emulator.calculate_chunk_size import calcBytePerStep, calcBytePerGrain, compute_last_chunk_size
from vescale.emulator.nccl.constants import *  # noqa: F403


def contract_tensor_list(tensor_list):
    """
    Contract a list of tensors into a list of tensors with the same size but with the first dimension
    contracted to be the product of the sizes of the original tensors.
    """
    n = len(tensor_list)
    a = len(tensor_list[0]) // n

    # Create a list to hold the contracted tensors
    contracted_list = []

    for i in range(n):
        # Extract the ith segment of size 'a' from the ith tensor
        contracted_tensor = tensor_list[i][i * a : (i + 1) * a]

        # Add the contracted tensor to the list
        contracted_list.append(contracted_tensor)

    return contracted_list


def run_ring_reduce_scatter(
    info: NcclInfo,
    nchannels: int,
    nwarps: int,
    protocol: int,
    data_list: List[torch.Tensor],
    ranks: List[int],
    device: torch.device,
    chunk_count: int,
    channel_count: int,
    grid_offset: int,
    reduce_op: ReduceOp,
) -> List[torch.Tensor]:
    """
    Run a ring reduce-scatter operation on the given data_list.

    Args:
        info: NcclInfo object containing information about the communication.
        nchannels: Number of channels in the communication.
        nwarps: Number of warps in the kernel.
        protocol: Protocol to use for communication.
        data_list: List of tensors to be reduced and scattered.
        ranks: List of ranks in the communication.
        device: Device to run the operation on.
        chunk_count: Size of chunks in the communication.
        channel_count: Total size of elements in a rank to be sent in the communication.
        grid_offset: Offset of the grid.
        reduce_op: Reduction operation to perform.

    Returns:
        List of tensors with the reduced and scattered data.
    """
    ring = Ring(ranks)
    prims = RingPrimitive(data_list, ring, reduce_op, device)
    count = len(data_list[0]) // ring.nranks

    nthreads = nwarps * WARP_SIZE

    sizeof_T = data_list[0].element_size()

    chunk_count = int(
        calcBytePerStep(protocol, info.comm)
        / sizeof_T
        * (REDUCESCATTER_CHUNKSTEPS if protocol == NCCL_PROTO_SIMPLE else 1)
    )
    min_chunk_size_LL128 = int(nthreads * (calcBytePerGrain(protocol) / sizeof_T) / 2)
    loop_size = nchannels * chunk_count
    last_chunk_size = compute_last_chunk_size(info)

    for elem_offset in range(0, channel_count, loop_size):
        if protocol == NCCL_PROTO_SIMPLE:
            real_chunk_size = min(chunk_count, div_up(count - elem_offset, nchannels))
            real_chunk_size = round_up(real_chunk_size, (nthreads - WARP_SIZE) * sizeof_uint64_t / sizeof_T)
        elif protocol == NCCL_PROTO_LL:
            real_chunk_size = last_chunk_size if count - elem_offset < loop_size else chunk_count
        elif protocol == NCCL_PROTO_LL128:
            real_chunk_size = min(
                div_up(count - elem_offset, nchannels * min_chunk_size_LL128) * min_chunk_size_LL128, chunk_count
            )
        real_chunk_size = int(real_chunk_size)

        for bid in range(nchannels):
            chunk_offset = elem_offset + bid * real_chunk_size
            nelem = min(real_chunk_size, count - chunk_offset)

            for ring_idx in range(ring.nranks):
                rank_dest = ring.mod_rank(ring_idx + ring.nranks - 1)
                offset = chunk_offset + rank_dest * count
                prims.send(ring_idx, offset, nelem)

            for j in range(2, ring.nranks, 1):
                for ring_idx in range(ring.nranks):
                    rank_dest = ring.mod_rank(ring_idx + ring.nranks - j)
                    offset = chunk_offset + rank_dest * count
                    prims.recv_reduce_send(ring_idx, offset, nelem)

            for ring_idx in range(ring.nranks):
                rank_dest = ring_idx
                offset = chunk_offset + rank_dest * count
                prims.recv_reduce_copy(ring_idx, offset, nelem)

    return contract_tensor_list(prims.convert_to_original_device_and_datatype())
