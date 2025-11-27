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

import math
from typing import List, Tuple, Sequence
import functools

import torch
from vescale.dtensor.placement_types import RaggedShard, Placement, DTensorSpec, Replicate


# TODO(jiacheng) in the future, we will create a lot of helper function for ragged shard.

# all functions in this file assume RaggedShard is used in placements

__all__ = [
    "unravel_index",
    "flatten_index",
    "cvt_inclusive_to_exclusive",
    "get_ragged_shard",
    "get_unflattened_dims",
    "get_unflattened_shape_and_offset_before_ragged_shard",
    "get_unflattened_shape_and_offset_before_ragged_shard_",
    "retrieve_flattened_index_before_ragged_shard",
    "best_effort_reshape",
    "substitute_ragged_with_replicate",
]


def unravel_index(idx: int, shape: Tuple[int, ...]) -> List[int]:
    coords = [0] * len(shape)
    for k in range(len(shape) - 1, -1, -1):
        coords[k] = idx % shape[k]
        idx //= shape[k]
    return coords


def flatten_index(index: Tuple[int, ...], shape: Tuple[int, ...]) -> int:
    """
    Convert an N-dimensional index into a single flattened index assuming the
    tensor is stored in **row-major (C-contiguous)** memory layout.

    Args:
        index: Tuple of integer indices (i0, i1, ..., i{N-1}) for each dimension.
        shape: Tuple of sizes for each dimension (s0, s1, ..., s{N-1}).

    Returns:
        The corresponding 0-based flattened index.

    Raises:
        ValueError: If `index` and `shape` lengths don't match.
        IndexError: If any index is out of bounds for its dimension.

    Notes:
        The implementation walks the dimensions from last to first, accumulating
        the stride (product of later dimension sizes) to compute:
            flat = i0 * (s1*s2*...) + i1 * (s2*...) + ... + i{N-1}.
    """
    if len(shape) != len(index):
        raise ValueError(f"Shape length {len(shape)} and index length {len(index)} must match")

    flat_index = 0
    stride = 1
    for s, i in zip(reversed(shape), reversed(index)):
        if not (0 <= i < s):
            raise IndexError(f"Index {i} out of bounds for dimension size {s}")
        flat_index += i * stride
        stride *= s

    return flat_index


def cvt_inclusive_to_exclusive(inclusive_end_coord: List[int], flattened_shape: Tuple[int, ...]) -> List[int]:
    exclusive_end_coord = [*inclusive_end_coord]
    idx = len(exclusive_end_coord) - 1
    while True:
        exclusive_end_coord[idx] += 1
        if idx == 0:
            break
        if exclusive_end_coord[idx] < flattened_shape[idx]:
            break
        elif exclusive_end_coord[idx] == flattened_shape[idx]:
            exclusive_end_coord[idx] = 0
            idx -= 1
        else:
            raise RuntimeError(f"{inclusive_end_coord=} {flattened_shape=}")

    return exclusive_end_coord


def get_ragged_shard(placements: Sequence[Placement]) -> Tuple[int, RaggedShard]:
    ragged_placement: RaggedShard | None = None
    ragged_placement_idx: int | None = None
    n_other_placements = 0
    for i, p in enumerate(placements):
        if p.is_replicate():
            continue
        if isinstance(p, RaggedShard):
            if ragged_placement is not None:
                raise RuntimeError("only 1 ragged shard is allowed for now")
            if n_other_placements != 0:
                raise RuntimeError(f"the placements sequence is not allowed {placements}")
            ragged_placement = p
            ragged_placement_idx = i
            continue
        n_other_placements += 1
    assert (ragged_placement is not None) and (ragged_placement_idx is not None)
    return ragged_placement_idx, ragged_placement


@functools.lru_cache(maxsize=2048)
def get_unflattened_dims(spec: DTensorSpec) -> Tuple[int, ...]:
    _, p = get_ragged_shard(spec.placements)
    if p.dims != tuple(range(len(p.dims))):
        raise RuntimeError(f"unsupported ragged dims {p=}")
    return tuple(range(p.dims[-1] + 1, len(spec.shape)))


def best_effort_reshape(tensor: torch.Tensor, spec: DTensorSpec):
    local_shape, _ = get_unflattened_shape_and_offset_before_ragged_shard(spec)
    unflattened_dims = get_unflattened_dims(spec)
    return tensor.view(-1, *local_shape[-len(unflattened_dims) :])


# todo merge with vescale/dtensor/_utils.py
def get_unflattened_shape_and_offset_before_ragged_shard(spec: DTensorSpec) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    return get_unflattened_shape_and_offset_before_ragged_shard_(spec.shape, spec.device_mesh, spec.placements)


def substitute_ragged_with_replicate(placements):
    i, _ = get_ragged_shard(placements)
    no_ragged_shard_placement = (*placements[:i], Replicate(), *placements[i + 1 :])
    return no_ragged_shard_placement


@functools.lru_cache(maxsize=2048)
def get_unflattened_shape_and_offset_before_ragged_shard_(
    shape, device_mesh, placements
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

    no_ragged_shard_placement = substitute_ragged_with_replicate(placements)
    local_shape, global_offset = compute_local_shape_and_global_offset(shape, device_mesh, no_ragged_shard_placement)
    return local_shape, global_offset


# TODO(jiacheng) merge this with ckpt utils
# This api return index follow python sematic inclusive start + exclusive end
@functools.lru_cache(maxsize=2048)
def retrieve_flattened_index_before_ragged_shard(spec: DTensorSpec) -> Tuple[int, int]:
    tensor_meta = spec.tensor_meta
    assert tensor_meta is not None
    # compute_local_shape_and_global_offset produces sizes and offsets in terms of ragged shape
    local_shape, global_offset = get_unflattened_shape_and_offset_before_ragged_shard(spec)
    if len(global_offset) == 0:
        assert local_shape == (0,)
        return 0, 0
    my_coordinate = spec.mesh.get_coordinate()
    assert my_coordinate is not None
    i, p = get_ragged_shard(spec.placements)
    numels = math.prod(local_shape)
    total_units = sum(p.local_units)
    assert numels % total_units == 0
    ratio = numels // total_units
    start = sum(p.local_units[: my_coordinate[i]]) * ratio
    end = sum(p.local_units[: my_coordinate[i] + 1]) * ratio
    return start, end
