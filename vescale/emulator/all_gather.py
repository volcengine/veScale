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
# Some code comes from all_gather.cc in NCCL
# Original license:
# Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
#
# See LICENSE.txt for license information
################################################################################

from typing import List
import torch

from vescale.emulator.topo import Ring
from vescale.emulator.primitives import RingPrimitive


def expand_tensor_list(tensor_list: List[torch.Tensor]) -> None:
    """
    Expand a list of tensors into a list of tensors with the same size but with the first dimension
    expanded to be the product of the sizes of the original tensors.
    """
    n = len(tensor_list)
    a = tensor_list[0].size(0)

    # Create a list to hold the expanded tensors
    expanded_list = []

    for i in range(n):
        # Create a new tensor of size (n * a) filled with zeros
        expanded_tensor = torch.zeros(n * a, dtype=tensor_list[i].dtype, device=tensor_list[i].device)

        # Copy the ith original tensor to the ith segment of the expanded tensor
        expanded_tensor[i * a : (i + 1) * a] = tensor_list[i]

        # Add the expanded tensor to the list
        expanded_list.append(expanded_tensor)

    return expanded_list


def run_ring_all_gather(
    data_list: List[torch.Tensor],
    ranks: List[int],
    device: torch.device,
    chunk_count: int,
    part_count: int,
    part_offset: int,
) -> List[torch.Tensor]:
    """
    Run a ring all-gather operation on the given data_list.

    Args:
        data_list: List of tensors to be gathered.
        ranks: List of ranks in the communication.
        device: Device to run the operation on.
        chunk_count: Size of chunks in the communication.
        part_count: Total size of elements in a rank to be gathered in the communication.
        part_offset: Offset of the data.

    Returns:
        List of tensors with the gathered data.
    """
    count = len(data_list[0])
    data_list = expand_tensor_list(data_list)

    ring = Ring(ranks)
    prims = RingPrimitive(data_list, ring, device=device)

    for elem_offset in range(0, part_count, chunk_count):
        nelem = min(chunk_count, part_count - elem_offset)
        data_offset = part_offset + elem_offset

        for ring_idx in range(ring.nranks):
            rank_dest = ring_idx
            offset = data_offset + rank_dest * count
            prims.send(ring_idx, offset, nelem)

        for j in range(1, ring.nranks - 1, 1):
            for ring_idx in range(ring.nranks):
                rank_dest = ring.mod_rank(ring_idx + ring.nranks - j)
                offset = data_offset + rank_dest * count
                prims.direct_recv_copy_send(ring_idx, offset, nelem)

        for ring_idx in range(ring.nranks):
            rank_dest = ring.mod_rank(ring_idx + 1)
            offset = data_offset + rank_dest * count
            prims.direct_recv(ring_idx, offset, nelem)

    return prims.convert_to_original_device_and_datatype()
