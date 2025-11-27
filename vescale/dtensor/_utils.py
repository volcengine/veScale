################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################
from collections.abc import Sequence
from typing import cast

import math

import torch
from torch._prims_common import ShapeType
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._utils import (
    try_find_mesh_from_args,
    compute_local_stride,
    normalize_to_torch_size,
)

from torch.distributed.tensor._utils import (
    compute_local_shape_and_global_offset as torch_compute_local_shape_and_global_offset,
)


from vescale.dtensor.placement_types import (
    Placement,
    Shard,
    Replicate,
    Partial,
    RaggedShard,
)

__all__ = [
    "compute_local_shape_and_global_offset",
    "compute_global_tensor_info",
    "try_find_mesh_from_args",
    "compute_local_stride",
    "normalize_to_torch_size",
]

"""
In this file, compute_local_shape_and_global_offset is modified to support RaggedShard.
Other functions are not modified and do not create a new dtensor, so they can be reused.

"""


def compute_global_tensor_info(
    tensor: torch.Tensor, mesh: DeviceMesh, placements: Sequence[Placement]
) -> tuple[list[int], list[int]]:
    """
    Compute the global size and stride of a DTensor from the given local tensor.
    The local size is multiplited by `world_size` per Sharding dim.
    The local stride is multiplited by `world_size` per Sharding dim, as long as the
    dimension is outside sharding dim.

    For example, if we have a local tensor with size (4, 8, 2) and stride (16, 1, 8).
    If the DTensor placements are [Shard(2)] and world_size is 2;
    then the global size is (4, 8, 4) and stride is (16 * 2, 1, 8).

    Args:
        tensor (:class:`torch.Tensor`):
            Local tensor which DTensor will be constructed from.
        mesh (:class:`DeviceMesh`):
            Object which describes the mesh topology
            of devices for the DTensor.
        placements (Sequence[:class:`Placement`]]):
            The attribute of the DTensor that describes its layout
            on the mesh topology.

    Return:
        tensor_shape: A List of int which specifies the size of DTensor which build
            on top of the local tensor.
        tensor_stride: A List of int which specifies the stride of DTensor.
    """
    tensor_shape = list(tensor.size())
    tensor_stride = list(tensor.stride())
    for idx, placement in enumerate(placements):
        mesh_dim_size = mesh.size(idx)
        if placement.is_shard():
            shard_placement = cast(Shard, placement)
            if shard_placement.dim < 0:
                raise AssertionError(
                    f"Shard placements should have negative dims normalized in the user-facing APIs: {shard_placement}"
                )
            shard_dim = shard_placement.dim

            assert shard_dim < tensor.ndim, (
                f"Sharding dim {shard_dim} greater than tensor ndim {tensor.ndim} for placement number {idx}."
            )

            local_dim_size = tensor_shape[shard_dim]
            tensor_shape[shard_dim] = local_dim_size * mesh_dim_size

            # recover tensor stride by modifying the stride that larger than
            # the current stride on the shard_dim
            for i in range(len(tensor_stride)):
                if i != shard_dim and tensor_stride[i] >= tensor_stride[shard_dim]:
                    # rescale the stride by the shard size
                    tensor_stride[i] = tensor_stride[i] * mesh_dim_size
        elif placement.is_ragged_shard():
            ragged_placement = cast(RaggedShard, placement)
            assert len(ragged_placement.local_units) == mesh.size(idx)
            ratio = ragged_placement.local_units[mesh.get_coordinate()[idx]]
            tot_size = sum(ragged_placement.local_units)
            assert len(tensor_shape) == 1, "RaggedShard's local_tensor is always flat."
            tensor_shape[0] = tensor_shape[0] // ratio * tot_size
        elif not isinstance(placement, (Replicate, Partial)):
            raise RuntimeError(f"placement type {type(placement)} not supported!")
    return tensor_shape, tensor_stride


def compute_local_shape_and_global_offset(
    global_shape: ShapeType, mesh: DeviceMesh, placements: Sequence[Placement]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    from vescale.dtensor.vescale_utils import (
        get_ragged_shard,
        get_unflattened_shape_and_offset_before_ragged_shard_,
    )

    # ragged shard must be the first placement except replicate
    # we must always apply ragged shard after all other placement taken effect.

    my_coordinate = mesh.get_coordinate()

    if my_coordinate is None:
        # if rank not in the mesh, return empty offset
        return ((0,), ())

    if not any(isinstance(p, RaggedShard) for p in placements):
        return torch_compute_local_shape_and_global_offset(global_shape, mesh, placements)

    ragged_idx, ragged_p = get_ragged_shard(placements)

    local_shape, global_offset = get_unflattened_shape_and_offset_before_ragged_shard_(
        tuple(global_shape), mesh, placements
    )

    if ragged_p is None:
        return local_shape, global_offset

    n_shard_dims = len(ragged_p.dims)
    if n_shard_dims == 0:  # origin dim is kept
        numel = math.prod(local_shape)
        assert sum(ragged_p.local_units) == max(ragged_p.local_units)
        if ragged_p.local_units[my_coordinate[ragged_idx]] == 0:
            local_shape = (0,)
            global_offset = ()
        return tuple(local_shape), tuple(global_offset)

    # for shard dims
    numel = math.prod(local_shape[:n_shard_dims])
    total_units = sum(ragged_p.local_units)
    assert numel % total_units == 0, f"{ragged_p=} {local_shape=}"
    ratio = numel // total_units
    unit = ragged_p.local_units[my_coordinate[ragged_idx]]
    if unit == 0:  #
        local_shape = (0,)
        global_offset = ()
    else:
        flatten_offset = sum(a * b for a, b in zip(global_offset[:n_shard_dims], (*global_shape[1:n_shard_dims], 1)))
        local_shape = [unit * ratio, *local_shape[n_shard_dims:]]
        global_offset = [
            flatten_offset + sum(ragged_p.local_units[: my_coordinate[ragged_idx]]) * ratio,
            *global_offset[n_shard_dims:],
        ]
    return tuple(local_shape), tuple(global_offset)
