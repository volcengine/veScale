################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

import functools
import operator
from typing import Iterable, List, Sequence, Tuple, Union

import torch

from vescale.dtensor._collective_utils import redistribute_cost
from vescale.dtensor.dtensor import DTensor
from vescale.dtensor.op_schema import OpStrategy
from vescale.dtensor.placement_types import DTensorSpec, InterleavedShard, Partial, Placement, Replicate, Shard

# convenient wrapper to register sharding propagation rules


def register_prop_rule(op, schema_info=None):
    def wrapper(impl):
        overloads = op if isinstance(op, list) else [op]
        for overload in overloads:
            DTensor._propagator.register_sharding_prop_rule(overload, impl, schema_info)
        return impl

    return wrapper


def register_op_strategy(op, schema_info=None):
    # pyre-fixme[53]: Captured variable `func` is not annotated.
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def wrapper(impl):
        overloads = op if isinstance(op, list) else [op]
        for overload in overloads:
            DTensor._propagator.register_op_strategy(overload, impl, schema_info)
        return impl

    return wrapper


def as_list(
    x: Union[List[object], object],
    # pyre-fixme[11]: Annotation `immutable_list` is not defined as a type.
) -> Union[List[object], torch.fx.immutable_collections.immutable_list]:  # type: ignore[valid-type]
    # During tracing, `aten.sum.dim_IntList` uses `immutable_list` for its args,
    # which is an object but treated as a list by the tracer. Therefore, keep
    # `immutable_list` intact here as well.
    if type(x) is list or isinstance(x, torch.fx.immutable_collections.immutable_list):
        return x
    else:
        return [x]


def normalize_dim(dim: int, ndim: int) -> int:
    return dim if dim >= 0 else dim + ndim


def normalize_dims(dims: Union[int, Sequence[int]], ndim: int) -> Sequence[int]:
    """
    normalize a dim or a sequence of dims, so that they
    are all positive.
    """
    if isinstance(dims, int):
        dims = (normalize_dim(dims, ndim),)
    elif isinstance(dims, list):
        dims = [normalize_dim(dim, ndim) for dim in dims]
    elif isinstance(dims, tuple):
        dims = tuple([normalize_dim(dim, ndim) for dim in dims])
    return dims


def normalize_to_torch_size(size) -> torch.Size:
    """
    Unify variable types of size argument to torch.Size
    Acceptable types include:
        int, Sequence[int], Tuple[int], Tuple[Sequence[int]],
        or torch.Size
    """
    if isinstance(size, torch.Size):
        return size

    if isinstance(size, int):
        torch_size = [size]
    elif len(size) == 1 and isinstance(size[0], Sequence):
        torch_size = list(size[0])
    else:
        torch_size = list(size)
    return torch.Size(torch_size)


def prod(xs: Iterable[int]) -> int:
    return functools.reduce(operator.mul, xs, 1)


def is_tensor_shardable(shape: Sequence[int], spec: DTensorSpec) -> bool:
    """
    Check if the shape is shardable according to the spec.
    """
    # number of shards in each tensor dimension
    shards_map = [1] * len(shape)
    for i, placement in enumerate(spec.placements):
        # NOTE: placement might be InterleavedShard
        if placement.is_shard():
            shards_map[placement.dim] *= spec.mesh.size(i)

    for i, dim_size in enumerate(shape):
        # TODO: maybe we should determine is_shardable based on
        #       whether it's evenly sharded or not
        if dim_size < shards_map[i]:
            return False

    return True


def is_tensor_dim_sharded(spec: DTensorSpec, dim: int) -> bool:
    """Return True if tensor dim is sharded"""
    return any(p.is_shard(dim) for p in spec.placements)


def is_tensor_dim_interleaved_sharded(spec: DTensorSpec, dim: int) -> bool:
    """Return True if tensor dim is interleaved sharded"""
    return any(p.is_interleaved_shard(dim) for p in spec.placements)


def is_tensor_partial(spec: DTensorSpec) -> bool:
    """Return True if tensor is partial on the mesh"""
    return any(p.is_partial() for p in spec.placements)


def is_tensor_all_replicate(spec: DTensorSpec) -> bool:
    """Return True if tensor is replicate on all mesh dimensions"""
    return all(p.is_replicate() for p in spec.placements)


def map_placements_after_broadcast(
    placements: Tuple[Placement, ...],
    shape: torch.Size,
    broadcast_dims_map: List[int],
) -> Tuple[Placement, ...]:
    """
    Map each placement based on the output shape after broadcast.
    """
    new_placements: List[Placement] = []
    for placement in placements:
        if isinstance(placement, (Replicate, Partial)):
            new_placements.append(placement)
        else:
            # NOTE: placement might be InterleavedShard
            assert placement.is_shard()
            shard_dim = normalize_dim(placement.dim, len(shape))
            new_shard_dim = broadcast_dims_map[shard_dim]
            if new_shard_dim != -1:
                # there's a map from the common shape shard dim to
                # the input shape shard dim before broadcasting,
                # use that instead
                if placement.is_shard():
                    new_placements.append(Shard(new_shard_dim))
                else:
                    new_placements.append(InterleavedShard(new_shard_dim, placement.interleaved_size))
            else:
                # there's no map between common shape shard dim and
                # the input shape shard dim before broadcasting,
                # in this case it means implicit broadcasting happen
                # in this dim, so we can just mark it as replicate
                # and implict broadcast will broadcast automatically
                # to the sharded shape
                new_placements.append(Replicate())

    return tuple(new_placements)


def infer_broadcast_dims_map(common_shape: torch.Size, input_shape: torch.Size) -> List[int]:
    # infer the broadcast dims map, where it maps from the common shape dim to the input shape dim
    # this is aligned with the broadcast semantics
    common_ndim = len(common_shape)
    input_ndim = len(input_shape)
    broadcast_dims_map = [-1] * common_ndim
    for idx in range(-1, -1 - input_ndim, -1):
        if input_shape[idx] == common_shape[idx]:
            broadcast_dims_map[common_ndim + idx] = input_ndim + idx
    return broadcast_dims_map


def generate_redistribute_costs(src_strategy: OpStrategy, dst_spec: DTensorSpec) -> List[float]:
    redistribute_costs: List[float] = []
    for strat in src_strategy.strategies:
        redistribute_costs.append(redistribute_cost(strat.output_spec, dst_spec))

    return redistribute_costs
