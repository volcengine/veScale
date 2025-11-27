################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

from typing import List, Tuple
import math
import functools
import torch

from torch.distributed.checkpoint.metadata import MetadataIndex, ChunkStorageMetadata, TensorProperties
from torch.distributed.checkpoint.planner import (
    WriteItem,
    WriteItemType,
    TensorWriteData,
)
from torch._prims_common import ShapeType

from vescale.dtensor import DTensor
from vescale.dtensor._utils import compute_local_shape_and_global_offset

from .ragged_shard_utils import (
    unravel_index,
    get_ragged_shard,
    cvt_inclusive_to_exclusive,
    get_unflattened_shape_and_offset_before_ragged_shard_,
)

__all__ = [
    "_create_chunk_for_ragged_shard",
    "_create_write_items_for_ragged_shard",
    "_get_tensor_shard_for_ragged_shard",
]


def _correct_ragged_offsets(
    ragged_dims: Tuple[int, ...],
    ragged_offsets: torch.Size,
    global_offsets: torch.Size,
    global_tensor_shape: ShapeType,
):
    ragged_offsets_unflattened_dims = (range(0, ragged_dims[0]), range(ragged_dims[0] + 1, len(ragged_offsets)))
    flattened_offsets = tuple(global_offsets[d] for d in ragged_dims)
    flattened_offsets_size = tuple(global_tensor_shape[d] for d in ragged_dims)
    global_offsets_unflattened_size = sum(a * b for a, b in zip(flattened_offsets, (*flattened_offsets_size[1:], 1)))

    new_ragged_dim_size = ragged_offsets[ragged_dims[0]] - global_offsets_unflattened_size
    return (
        *(ragged_offsets[a] for a in ragged_offsets_unflattened_dims[0]),
        new_ragged_dim_size,
        *(ragged_offsets[a] for a in ragged_offsets_unflattened_dims[1]),
    )


def _correct_result_offsets(
    ragged_dims: Tuple[int, ...],
    result_offsets: torch.Size,  # result offset is unragged
    global_offsets: torch.Size,
):
    new_result_ragged_dim_size = (result_offsets[d] + global_offsets[d] for d in ragged_dims)
    return (*result_offsets[: ragged_dims[0]], *new_result_ragged_dim_size, *result_offsets[ragged_dims[-1] + 1 :])


@functools.lru_cache(maxsize=1024)
def _break_ragged_box(
    ragged_sizes: torch.Size,
    ragged_offsets: torch.Size,
    ragged_dims: Tuple[int, ...],
    unragged_tensor_shape: ShapeType,
    original_tensor_shape: ShapeType,
    global_offsets: torch.Size,
) -> Tuple[List[Tuple[int, ...]], List[Tuple[int, ...]]]:
    # nothing local does not have any part of the tensor
    if ragged_sizes == (0,) and ragged_offsets == ():
        return [], []
    assert len(ragged_sizes) - 1 + len(ragged_dims) == len(unragged_tensor_shape)
    compute_offset = _correct_ragged_offsets(ragged_dims, ragged_offsets, global_offsets, original_tensor_shape)
    flatten_dim = ragged_dims[0]
    unflattened_sizes = (
        tuple(ragged_sizes[i] for i in range(0, flatten_dim)),
        tuple(ragged_sizes[i] for i in range(flatten_dim + 1, len(ragged_sizes))),
    )
    unflattened_offsets = (
        tuple(compute_offset[i] for i in range(0, flatten_dim)),
        tuple(compute_offset[i] for i in range(flatten_dim + 1, len(ragged_sizes))),
    )
    make_output_sizes = lambda x: (*unflattened_sizes[0], *x, *unflattened_sizes[1])
    make_output_offsets = lambda x: _correct_result_offsets(
        ragged_dims,
        torch.Size(
            (
                *unflattened_offsets[0],
                *x,
                *unflattened_offsets[1],
            )
        ),
        global_offsets,
    )
    flattened_shape = tuple(unragged_tensor_shape[dim] for dim in ragged_dims)
    flatten_start = compute_offset[flatten_dim]
    flatten_end = flatten_start + ragged_sizes[flatten_dim]

    offsets_list = []
    sizes_list = []

    start_coord = unravel_index(flatten_start, flattened_shape)
    inclusive_end_coord = unravel_index(flatten_end - 1, flattened_shape)
    exclusive_end_coord = cvt_inclusive_to_exclusive(inclusive_end_coord, flattened_shape)

    # we have nothing to save, technically, this should be impossible.
    if flatten_end == flatten_start:
        return [], []

    # find the first diff coord/dim
    first_diff_coord_idx = len(start_coord) - 1
    for i, (s, e) in enumerate(zip(start_coord, exclusive_end_coord)):
        if s != e:
            first_diff_coord_idx = i
            break

    # 1d case
    if first_diff_coord_idx == len(start_coord) - 1:
        _size = [1 for _ in range(len(start_coord))]
        _size[-1] = exclusive_end_coord[-1] - start_coord[-1]
        return [make_output_sizes(_size)], [make_output_offsets(start_coord)]

    # nd case
    walker = list(start_coord)

    # for start coord, make sub boxes
    for i in range(len(walker) - 1, first_diff_coord_idx, -1):
        if walker[i] == 0:
            continue
        fill_size = flattened_shape[i] - walker[i]
        size = [1 for _ in range(i)]
        size.append(fill_size)
        size.extend(flattened_shape[i + 1 :])
        sizes_list.append(make_output_sizes(size))
        offsets_list.append(make_output_offsets(walker))
        walker[i] = 0
        walker[i - 1] += 1
    # make the coord for the middle sub boxes
    diff_amount = exclusive_end_coord[first_diff_coord_idx] - walker[first_diff_coord_idx]
    if diff_amount > 0:
        padding = [1 for _ in range(first_diff_coord_idx)]
        sizes_list.append(make_output_sizes((*padding, diff_amount, *flattened_shape[first_diff_coord_idx + 1 :])))
        offsets_list.append(make_output_offsets(walker))
    walker[first_diff_coord_idx] += diff_amount
    # reach out to the end coord
    assert walker[first_diff_coord_idx] == exclusive_end_coord[first_diff_coord_idx]
    walker2 = list(exclusive_end_coord)
    unreversed_sizes_list = []
    unreversed_offsets_list = []
    for i in range(len(walker2) - 1, first_diff_coord_idx, -1):
        if walker2[i] == 0:
            continue
        fill_size = walker2[i]
        walker2[i] = 0
        size = [1 for _ in range(i)]
        size.append(fill_size)
        size.extend(flattened_shape[i + 1 :])
        unreversed_sizes_list.append(make_output_sizes(size))
        unreversed_offsets_list.append(make_output_offsets(walker2))
    sizes_list.extend(reversed(unreversed_sizes_list))
    offsets_list.extend(reversed(unreversed_offsets_list))

    return sizes_list, offsets_list


def _create_write_items_for_ragged_shard(fqn: str, tensor: DTensor) -> List[WriteItem]:
    sizes, ragged_offsets = compute_local_shape_and_global_offset(tensor.shape, tensor.device_mesh, tensor.placements)
    sizes, ragged_offsets = torch.Size(sizes), torch.Size(ragged_offsets)
    if len(ragged_offsets) == 0:
        return []
    if len(sizes) == len(tensor.shape):
        return [
            WriteItem(
                index=MetadataIndex(fqn, ragged_offsets),
                type=WriteItemType.SHARD,
                tensor_data=TensorWriteData(
                    chunk=ChunkStorageMetadata(
                        offsets=ragged_offsets,
                        sizes=sizes,
                    ),
                    properties=TensorProperties.create_from_tensor(tensor.to_local()),
                    size=tensor.size(),
                ),
            )
        ]
    i, p = get_ragged_shard(tensor.placements)
    ragged_dims = p.dims

    # we need to break ragged box to regular boxes
    local_shape, global_offset = get_unflattened_shape_and_offset_before_ragged_shard_(
        tensor.shape, tensor.device_mesh, tensor.placements
    )
    sizes_list, offsets_list = _break_ragged_box(
        sizes, ragged_offsets, ragged_dims, local_shape, tensor.shape, global_offset
    )

    write_items = []

    for s, o in zip(sizes_list, offsets_list):
        write_items.append(
            WriteItem(
                index=MetadataIndex(fqn, o),
                type=WriteItemType.SHARD,
                tensor_data=TensorWriteData(
                    chunk=ChunkStorageMetadata(
                        offsets=torch.Size(o),
                        sizes=torch.Size(s),
                    ),
                    properties=TensorProperties.create_from_tensor(tensor.to_local()),
                    size=tensor.size(),
                ),
            )
        )
    return write_items


def _create_chunk_for_ragged_shard(tensor: DTensor) -> List[ChunkStorageMetadata]:
    sizes, offsets = compute_local_shape_and_global_offset(tensor.shape, tensor.device_mesh, tensor.placements)
    sizes, offsets = torch.Size(sizes), torch.Size(offsets)
    if len(offsets) == 0:
        return []
    if len(sizes) == len(tensor.shape):
        return [
            ChunkStorageMetadata(
                offsets=offsets,
                sizes=sizes,
            )
        ]

    i, p = get_ragged_shard(tensor.placements)
    ragged_dims = p.dims

    # we need to break ragged box to regular boxes
    local_shape, global_offset = get_unflattened_shape_and_offset_before_ragged_shard_(
        tensor.shape, tensor.device_mesh, tensor.placements
    )
    sizes_list, offsets_list = _break_ragged_box(sizes, offsets, ragged_dims, local_shape, tensor.shape, global_offset)

    chunks = []
    for s, o in zip(sizes_list, offsets_list):
        chunks.append(
            ChunkStorageMetadata(
                offsets=torch.Size(o),
                sizes=torch.Size(s),
            )
        )
    return chunks


def _get_tensor_shard_for_ragged_shard(tensor: DTensor, index: MetadataIndex) -> torch.Tensor:
    sizes, offsets = compute_local_shape_and_global_offset(tensor.shape, tensor.device_mesh, tensor.placements)
    sizes, offsets = torch.Size(sizes), torch.Size(offsets)
    local_tensor = tensor._local_tensor
    if len(sizes) == len(tensor.shape):
        return local_tensor.view(sizes)

    i, p = get_ragged_shard(tensor.placements)
    ragged_dims = p.dims
    local_shape, global_offset = get_unflattened_shape_and_offset_before_ragged_shard_(
        tensor.shape, tensor.device_mesh, tensor.placements
    )
    sizes_list, offsets_list = _break_ragged_box(sizes, offsets, ragged_dims, local_shape, tensor.shape, global_offset)

    offset = getattr(index, "offset", None)
    assert offset is not None
    idx = offsets_list.index(tuple(offset))
    target_size = sizes_list[idx]
    flatten_size = math.prod(target_size)
    local_flatten_offset = sum(math.prod(s) for s in sizes_list[:idx])
    rt = local_tensor[local_flatten_offset : local_flatten_offset + flatten_size]
    return rt.view(target_size)
