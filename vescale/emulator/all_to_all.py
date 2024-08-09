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

from typing import List

import torch
from vescale.emulator.primitives import Point2PointPrimitive


def run_all_to_all(
    data_list: List[torch.tensor],
    ranks: List[int],
    device: torch.device,
    datatype: torch.dtype,
    chunk_count: int,
    part_count: int,
    part_offset: int,
) -> List[torch.tensor]:
    """
    Run a all-to-all operation on the given data_list. This function calls a list of send
    and recv primitives to send and receive data between ranks.

    Args:
        data_list: List of tensors to be sent.
        ranks: List of ranks in the communication.
        device: Device to run the operation on.
        datatype: Data type of the tensor.
        chunk_count: Size of chunks in the communication.
        part_count: Total size of elements in a rank to be sent in the communication.
        part_offset: Offset of the data.

    Returns:
        List of tensors with the gathered data.

    """
    group_ranks = list(range(len(ranks)))
    prims = Point2PointPrimitive(data_list, group_ranks, device=device, datatype=datatype)
    count = len(data_list[0]) // len(group_ranks)

    for src_rank in group_ranks:
        for dst_rank in group_ranks:
            send_offset = dst_rank * count
            recv_offset = src_rank * count
            cursor = 0
            while cursor < part_count:
                n = min(chunk_count, part_count - cursor)
                src_offset = part_offset + send_offset + cursor
                dst_offset = part_offset + recv_offset + cursor
                prims.send(src_rank, src_offset, n, datatype, dst_rank, dst_offset)
                cursor += n

    for src_rank in group_ranks:
        for dst_rank in group_ranks:
            send_offset = dst_rank * count
            recv_offset = src_rank * count
            cursor = 0
            while cursor < part_count:
                n = min(chunk_count, part_count - cursor)
                src_offset = part_offset + send_offset + cursor
                dst_offset = part_offset + recv_offset + cursor
                prims.recv(src_rank, src_offset, n, datatype, dst_rank, dst_offset)
                cursor += n

    return prims.convert_to_original_device_and_datatype()
