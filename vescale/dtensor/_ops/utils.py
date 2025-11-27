# mypy: allow-untyped-defs
################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

import torch

from vescale.dtensor.placement_types import Partial, Placement, Replicate, Shard, RaggedShard

from torch.distributed.tensor._ops.utils import (
    register_prop_rule,
    register_op_strategy,
    as_list,
    normalize_dim,
    normalize_dims,
    prod,
    is_tensor_shardable,
    is_tensor_evenly_shardable,
    is_tensor_dim_sharded,
    is_tensor_partial,
    infer_broadcast_dims_map,
    generate_redistribute_costs,
    expand_to_full_mesh_op_strategy,
)

__all__ = [
    "register_prop_rule",
    "register_op_strategy",
    "as_list",
    "normalize_dim",
    "normalize_dims",
    "prod",
    "is_tensor_shardable",
    "is_tensor_evenly_shardable",
    "is_tensor_dim_sharded",
    "is_tensor_partial",
    "infer_broadcast_dims_map",
    "generate_redistribute_costs",
    "expand_to_full_mesh_op_strategy",
]

"""
In this file, map_placements_after_broadcast is modified to add support for RaggedShard.

We can reuse register_prop_rule and register_op_strategy as we copy all registered rules and strategies from torch in ShardingPropagator
"""


def map_placements_after_broadcast(
    placements: tuple[Placement, ...],
    shape: torch.Size,
    broadcast_dims_map: list[int],
) -> tuple[Placement, ...]:
    """Map each placement based on the output shape after broadcast."""
    new_placements: list[Placement] = []
    for placement in placements:
        if isinstance(placement, (Replicate, Partial)):
            new_placements.append(placement)
        elif isinstance(placement, RaggedShard):
            has_broadcast = False
            for dim in placement.dims:
                if broadcast_dims_map[dim] == -1:
                    has_broadcast = True
            if has_broadcast:
                new_placements.append(Replicate())
            else:
                new_placements.append(placement)
        else:
            assert isinstance(placement, Shard)
            shard_dim = normalize_dim(placement.dim, len(shape))
            new_shard_dim = broadcast_dims_map[shard_dim]
            if new_shard_dim != -1:
                # there's a map from the common shape shard dim to
                # the input shape shard dim before broadcasting,
                # use that instead
                new_placements.append(Shard(new_shard_dim))
            else:
                # there's no map between common shape shard dim and
                # the input shape shard dim before broadcasting,
                # in this case it means implicit broadcasting happen
                # in this dim, so we can just mark it as replicate
                # and implict broadcast will broadcast automatically
                # to the sharded shape
                new_placements.append(Replicate())

    return tuple(new_placements)
