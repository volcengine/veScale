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
# Some code comes from all_reduce.cc in NCCL
# Original license:
# Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
################################################################################

from typing import List
import torch

from vescale.emulator.nccl.include.info import NcclInfo
from vescale.emulator.reduce_kernel import ReduceOp
from vescale.emulator.topo import Ring, DoubleTree
from vescale.emulator.primitives import RingPrimitive, TreePrimitive
from vescale.emulator.calculate_chunk_size import calcBytePerStep, calcBytePerGrain, compute_last_chunk_size
from vescale.emulator.nccl.constants import (
    NCCL_PROTO_LL,
    NCCL_PROTO_LL128,
    NCCL_PROTO_SIMPLE,
)
from vescale.emulator.nccl.constants import *  # noqa: F403


def run_ring_all_reduce(
    info: NcclInfo,
    nchannels: int,
    nwarps: int,
    protocol: int,
    data_list: List[torch.Tensor],
    ranks: List[int],
    device: torch.device,
    channel_count: int,
    grid_offset: int,
    reduce_op: ReduceOp,
) -> List[torch.Tensor]:
    """
    Run a ring all-reduce operation on the given data_list. This function can be regarded
    as a ring reduce_scatter followed by a ring all-gather.

    Args:
        info: NcclInfo object containing information about the communication.
        nchannels: Number of channels in the communication.
        nwarps: Number of warps in the kernel.
        protocol: Protocol to use for communication.
        data_list: List of tensors to be reduced.
        ranks: List of ranks in the communication.
        device: Device to run the operation on.
        channel_count: Size of elements each channel communicates in an iteration.
        grid_offset: Offset of the data.
        reduce_op: Reduction operation to perform.

    Returns:
        List of tensors with the reduced data.
    """
    ring = Ring(ranks)
    prims = RingPrimitive(data_list, ring, reduce_op, device)

    nthreads = nwarps * WARP_SIZE

    sizeof_T = data_list[0].element_size()

    chunk_count = int(
        calcBytePerStep(protocol, info.comm) / sizeof_T * (ALLREDUCE_CHUNKSTEPS if protocol == NCCL_PROTO_SIMPLE else 1)
    )
    loop_count = nchannels * ring.nranks * chunk_count

    min_chunk_count = 0
    if protocol == NCCL_PROTO_LL:
        min_chunk_count = nthreads * (calcBytePerGrain(protocol) / sizeof_T)
    elif protocol == NCCL_PROTO_LL128:
        min_chunk_count = nthreads * (calcBytePerGrain(protocol) / sizeof_T) / 2

    count = 0
    for elem_offset in range(0, channel_count, loop_count):
        if protocol == NCCL_PROTO_SIMPLE:
            real_chunk_count = min(chunk_count, div_up(channel_count - elem_offset, nchannels * ring.nranks))
            real_chunk_count = round_up(real_chunk_count, (nthreads - WARP_SIZE) * sizeof_uint64_t / sizeof_T)
        else:
            real_chunk_count = min(
                chunk_count,
                div_up(channel_count - elem_offset, nchannels * ring.nranks * min_chunk_count) * min_chunk_count,
            )
        real_chunk_count = int(real_chunk_count)
        for bid in range(nchannels):

            def calc_offset(chunk):
                if protocol == NCCL_PROTO_SIMPLE:
                    return elem_offset + bid * ring.nranks * real_chunk_count + chunk * real_chunk_count
                else:
                    return elem_offset + (chunk * nchannels + bid) * real_chunk_count

            # ring reduce_scatter starts
            for ring_idx in range(ring.nranks):
                chunk = ring.mod_rank(ring_idx + ring.nranks - 1)
                offset = calc_offset(chunk)
                nelem = min(real_chunk_count, channel_count - offset)
                prims.send(ring_idx, offset, nelem)

            for j in range(2, ring.nranks, 1):
                for ring_idx in range(ring.nranks):
                    chunk = ring.mod_rank(ring_idx + ring.nranks - j)
                    offset = calc_offset(chunk)
                    nelem = min(real_chunk_count, channel_count - offset)
                    prims.recv_reduce_send(ring_idx, offset, nelem)

            # ring reduce_scatter ends and ring all-gather starts
            for ring_idx in range(ring.nranks):
                chunk = ring_idx
                offset = calc_offset(chunk)
                nelem = min(real_chunk_count, channel_count - offset)
                prims.direct_recv_reduce_copy_send(ring_idx, offset, nelem)

            for j in range(1, ring.nranks - 1, 1):
                for ring_idx in range(ring.nranks):
                    chunk = ring.mod_rank(ring_idx + ring.nranks - j)
                    offset = calc_offset(chunk)
                    nelem = min(real_chunk_count, channel_count - offset)
                    prims.direct_recv_copy_send(ring_idx, offset, nelem)

            for ring_idx in range(ring.nranks):
                chunk = ring.mod_rank(ring_idx + 1)
                offset = calc_offset(chunk)
                nelem = min(real_chunk_count, channel_count - offset)
                prims.direct_recv(ring_idx, offset, nelem)
            # ring all-gather ends

    return prims.convert_to_original_device_and_datatype()


def run_tree_up_down(
    info: NcclInfo,
    nchannels: int,
    nwarps: int,
    protocol: int,
    data_list: List[torch.Tensor],
    tree_structure: List[List[int]],
    ranks: List[int],
    mapping: List[int],
    mode: int,
    device: torch.device,
    channel_count: int,
    grid_offset: int,
    reduce_op: ReduceOp,
    tree_idx: int,
) -> List[torch.Tensor]:
    """
    Run a single tree all-reduce operation on the given data_list. This function can be regarded
    as a tree reduce followed by a tree broadcast.

    Args:
        info: NcclInfo object containing information about the communication.
        nchannels: Number of channels in the communication.
        nwarps: Number of warps in the kernel.
        protocol: Protocol to use for communication.
        data_list: List of tensors to be reduced.
        tree_structure: Tree structure of the servers.
        ranks: List of ranks in the communication.
        mapping: Mapping of ranks to nodes in the tree.
        mode: Mode of the communication.
        device: Device to run the operation on.
        channel_count: Size of elements each channel communicates in an iteration.
        grid_offset: Offset of the data.
        reduce_op: Reduction operation to perform.
        tree_idx: Index of the tree.

    Returns:
        List of tensors with the reduced data.
    """
    tree = DoubleTree(tree_structure, ranks, mapping, pattern=mode)
    prims = TreePrimitive(data_list, tree, reduce_op, device)

    sizeof_T = data_list[0].element_size()
    nthreads = nwarps * WARP_SIZE
    last_chunk_count = compute_last_chunk_size(info)
    if protocol == NCCL_PROTO_SIMPLE:
        chunk_count = int(last_chunk_count)
        min_chunk_count = int((nthreads - 2 * WARP_SIZE) * 8 * (sizeof_uint64_t / sizeof_T))
    else:
        chunk_count = int(calcBytePerStep(protocol, info.comm) / sizeof_T)
        min_chunk_count = int(nthreads * (calcBytePerGrain(protocol) / sizeof_T))

    loopsize = int(nchannels * chunk_count)
    size = data_list[0].size()[0]

    if loopsize > size:
        chunk_count = div_up(int(size), int(nchannels * min_chunk_count)) * int(min_chunk_count)

    def get_root(tree_idx):
        node = 0
        while tree.tree[tree_idx][node].up != -1:
            node = tree.tree[tree_idx][node].up
        return node

    def tree_reduce_helper(rank, offset, nelem):
        for d in tree.tree[tree_idx][rank].down:
            if d != -1:
                tree_reduce_helper(d, offset, nelem)
        if tree.tree[tree_idx][rank].up == -1:
            prims.recv_reduce_copy(rank, tree_idx, offset, nelem)
        elif all(d == -1 for d in tree.tree[tree_idx][rank].down):
            prims.send(rank, tree_idx, offset, nelem)
        else:
            prims.recv_reduce_send(rank, tree_idx, offset, nelem)

    def tree_broadcast_helper(rank, offset, nelem):
        if tree.tree[tree_idx][rank].up == -1:
            prims.direct_send_from_output(rank, tree_idx, offset, nelem)
        elif all(d == -1 for d in tree.tree[tree_idx][rank].down):
            prims.direct_recv(rank, tree_idx, offset, nelem)
        else:
            prims.direct_recv_copy_send(rank, tree_idx, offset, nelem)
        for d in tree.tree[tree_idx][rank].down:
            if d != -1:
                tree_broadcast_helper(d, offset, nelem)

    # tree reduce
    root = get_root(tree_idx)
    elem_offset = 0
    while elem_offset < channel_count:
        offset = grid_offset + elem_offset
        nelem = min(chunk_count, channel_count - elem_offset)
        tree_reduce_helper(root, offset, nelem)
        elem_offset += chunk_count

    # tree broadcast
    elem_offset = 0
    while elem_offset < channel_count:
        offset = grid_offset + elem_offset
        nelem = min(chunk_count, channel_count - elem_offset)
        tree_broadcast_helper(root, offset, nelem)
        elem_offset += chunk_count

    return prims.data_list[tree_idx]


def split_tensors(tensor_list: List[torch.Tensor]):
    """
    Split a list of tensors into two lists of tensors, each containing half of the original tensors.
    """
    list1 = []
    list2 = []

    for tensor in tensor_list:
        size = tensor.size(0)
        half_size = size // 2

        tensor1 = tensor[:half_size]
        tensor2 = tensor[half_size:]

        list1.append(tensor1)
        list2.append(tensor2)

    return list1, list2


def concatenate_tensors(list1: List[torch.tensor], list2: List[torch.tensor]):
    """
    Concatenate two lists of tensors along the first dimension.
    """
    concatenated_list = []

    # Check if both lists are of the same length
    assert len(list1) == len(list2), "Both lists must have the same length"

    for tensor1, tensor2 in zip(list1, list2):
        # Concatenate the tensors along the first dimension
        concatenated_tensor = torch.cat((tensor1, tensor2), dim=0)
        concatenated_list.append(concatenated_tensor)

    return concatenated_list


def run_tree_all_reduce(
    info: NcclInfo,
    nchannels: int,
    nwarps: int,
    protocol: int,
    data_list: List[torch.Tensor],
    tree_structure: List[List[int]],
    ranks: List[int],
    mapping: List[int],
    mode: int,
    device: torch.device,
    channel_count: int,
    grid_offset: int,
    reduce_op: ReduceOp,
) -> List[torch.Tensor]:
    """
    Run a double tree all-reduce operation on the given data_list.

    Args:
        info: NcclInfo object containing information about the communication.
        nchannels: Number of channels in the communication.
        nwarps: Number of warps in the kernel.
        protocol: Protocol to use for communication.
        data_list: List of tensors to be reduced.
        tree_structure: Tree structure of the servers.
        ranks: List of ranks in the communication.
        mapping: Mapping of ranks to nodes in the tree.
        mode: Mode of the communication.
        device: Device to run the operation on.
        channel_count: Size of elements each channel communicates in an iteration.
        grid_offset: Offset of the data.
        reduce_op: Reduction operation to perform.

    Returns:
        List of reduced tensors.
    """
    tensor_list_half_list = split_tensors(data_list)
    result_list_half_list = []
    for i in range(2):
        result_list_half_list.append(
            run_tree_up_down(
                info,
                nchannels,
                nwarps,
                protocol,
                tensor_list_half_list[i],
                tree_structure,
                ranks,
                mapping,
                mode,
                device,
                tensor_list_half_list[i][0].size()[0],
                grid_offset,
                reduce_op,
                i,
            )
        )
    result_list = concatenate_tensors(result_list_half_list[0], result_list_half_list[1])
    return result_list
