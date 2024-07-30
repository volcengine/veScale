################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

from typing import List, Optional, Sequence, Tuple, cast
import warnings

import copy
import numpy as np
import torch

from vescale.dtensor._utils import compute_local_shape
from vescale.dtensor.op_schema import (
    OpSchema,
    OutputSharding,
    RuntimeSchemaInfo,
    PlacementStrategy,
    StrategyType,
    OpStrategy,
    TupleStrategy,
)
from vescale.dtensor.ops.common_rules import pointwise_rule
from vescale.dtensor.ops.utils import (
    is_tensor_dim_sharded,
    is_tensor_dim_interleaved_sharded,
    is_tensor_partial,
    normalize_dim,
    prod,
    register_prop_rule,
    register_op_strategy,
)
from vescale.dtensor.placement_types import (
    DTensorSpec,
    Partial,
    Placement,
    Replicate,
    InterleavedShard,
    Shard,
    TensorMeta,
)
from vescale.dtensor import DeviceMesh

aten = torch.ops.aten


@register_op_strategy(
    [
        aten._to_copy.default,
        aten.clone.default,
        aten.contiguous.default,
        aten.copy_.default,
        aten.cumsum.default,
        aten.detach.default,
        aten.equal.default,
        aten.fill_.Scalar,
        aten.is_same_size.default,
        aten.zero_.default,
    ]
)
def default_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> StrategyType:
    # Default strategy by default just propagate the first input strategy
    select_strategy = op_schema.args_schema[0]
    assert isinstance(select_strategy, OpStrategy)
    return OpStrategy(
        [
            # PlacementStrategy(arg_strategy.output_spec)
            PlacementStrategy(
                DTensorSpec(
                    arg_strategy.output_spec.mesh,
                    arg_strategy.output_spec.placements,
                    arg_strategy.output_spec.tensor_meta,
                )
            )
            for arg_strategy in select_strategy.strategies
        ]
    )


@register_op_strategy(
    [
        aten.empty_like.default,
        aten.ones_like.default,
        aten.rand_like.default,
        aten.randn_like.default,
        aten.zeros_like.default,
    ],
    schema_info=RuntimeSchemaInfo(1, ["dtype"]),
)
@register_op_strategy(
    [aten.full_like.default],
    schema_info=RuntimeSchemaInfo(2, ["dtype"]),
)
@register_op_strategy(
    [
        aten.randint_like.default,
        aten.randint_like.low_dtype,
        aten.randint_like.low_dtype_out,
    ],
    schema_info=RuntimeSchemaInfo(3, ["dtype"]),
)
def create_like_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> StrategyType:
    # create_like_strategy deals with ops that creating tensors with same
    # shape as input, but with specific content that does not depend on
    # the input, we can propagate sharding, but we have to make sure we
    # move from partial to replicated.
    select_strategy = op_schema.args_schema[0]
    create_like_strategy = OpStrategy([])
    assert isinstance(select_strategy, OpStrategy)
    for arg_strategy in select_strategy.strategies:
        arg_spec = arg_strategy.output_spec
        if is_tensor_partial(arg_spec):
            # if the arg_spec have partial, accept partial
            # in the input_specs but output replicate for
            # those corresponding mesh dims

            output_spec = DTensorSpec(
                mesh=arg_spec.mesh,
                placements=tuple(Replicate() if isinstance(p, Partial) else p for p in arg_spec.placements),
            )
            create_like_strategy.strategies.append(
                PlacementStrategy(
                    output_spec=output_spec,
                    input_specs=(arg_spec,),
                )
            )

        else:
            create_like_strategy.strategies.append(PlacementStrategy(arg_spec))

    return create_like_strategy


@register_prop_rule(
    [
        aten.new_empty.default,
        aten.new_full.default,
        aten.new_ones.default,
        # aten.new_zeros.default,
    ],
    schema_info=RuntimeSchemaInfo(1, ["dtype"]),
)
def new_factory_rule(op_schema: OpSchema) -> OutputSharding:
    # TODO: maybe we should generate all possible shardings intead of just stay
    # replicated for new factory methods
    args = op_schema.args_schema
    input_spec = args[0]
    assert isinstance(input_spec, DTensorSpec)
    mesh = input_spec.mesh
    output_shape = args[1]

    # has partial spec
    if any(p.is_partial() for p in input_spec.placements):
        raise RuntimeError("constrcuting partial tensors using new-factory methods is ambigious")

    # no shard spec
    if all(not p.is_shard() for p in input_spec.placements):
        replica_spec = DTensorSpec(mesh, tuple([Replicate()] * mesh.ndim))
        return OutputSharding(replica_spec)
    # has shard spec, we refer to view op to do the sharding prop.
    assert (
        input_spec.tensor_meta is not None
    ), "tensor meta must not be None if you are constructing a sharded tensor using `new_zeros` or something like that"
    original_numel = prod(input_spec.tensor_meta.shape)
    target_numel = prod(output_shape)

    from vescale.dtensor.ops.vescale_view_ops import vescale_view_rule_prop, ops

    if original_numel != target_numel:
        return OutputSharding(
            output_spec=DTensorSpec(
                mesh=mesh,
                placements=tuple([Replicate()] * mesh.ndim),
            )
        )

    spec = ops[torch.Tensor.view]
    output_sharding = vescale_view_rule_prop(op_schema=op_schema, spec=spec)
    return output_sharding


# (Hongyu): support partial new_zeros
@register_prop_rule(
    aten.new_zeros.default,
    schema_info=RuntimeSchemaInfo(1, ["dtype"]),
)
def new_zeros_rule(op_schema: OpSchema) -> OutputSharding:
    args = op_schema.args_schema
    input_spec, output_shape = args[0], args[1]
    assert isinstance(input_spec, DTensorSpec)
    mesh = input_spec.mesh
    output_stride = list(np.cumprod(output_shape[::-1])[:-1][::-1])
    output_stride.append(1)

    if input_spec.is_partial():
        partial_spec = DTensorSpec(
            mesh=mesh,
            placements=input_spec.placements,
            tensor_meta=TensorMeta(
                torch.Size(output_shape),
                tuple(output_stride),
                input_spec.tensor_meta.dtype,
            ),
        )
        return OutputSharding(partial_spec)

    # no shard spec
    if all(not p.is_shard() for p in input_spec.placements):
        replica_spec = DTensorSpec(mesh, tuple([Replicate()] * mesh.ndim))
        return OutputSharding(replica_spec)
    # has shard spec, we refer to view op to do the sharding prop.
    assert (
        input_spec.tensor_meta is not None
    ), "tensor meta must not be None if you are constructing a sharded tensor using `new_zeros` or something like that"
    original_numel = prod(input_spec.tensor_meta.shape)
    target_numel = prod(output_shape)
    assert original_numel == target_numel, "for now, we only support the same numel in new_factory methods"

    from vescale.dtensor.ops.vescale_view_ops import vescale_view_rule_prop, ops

    spec = ops[torch.Tensor.view]
    output_sharding = vescale_view_rule_prop(op_schema=op_schema, spec=spec)
    return output_sharding


@register_prop_rule(
    aten.new_empty_strided.default,
    schema_info=RuntimeSchemaInfo(1, ["dtype"]),
)
def new_empty_strided_rule(op_schema: OpSchema) -> OutputSharding:
    # TODO: maybe we should generate all possible shardings intead of just stay
    # replicated for new factory methods
    args = op_schema.args_schema
    input_spec = args[0]
    output_shape = args[1]
    output_stride = args[2]
    assert isinstance(input_spec, DTensorSpec)

    mesh = input_spec.mesh

    # all replicate
    if all(isinstance(p, Replicate) for p in input_spec.placements):
        return OutputSharding(output_spec=DTensorSpec(mesh=mesh, placements=tuple([Replicate()] * mesh.ndim)))
    # has partial
    if any(isinstance(p, Partial) for p in input_spec.placements):
        return OutputSharding(
            output_spec=None,
            schema_suggestions=[
                OpSchema(
                    op=op_schema.op,
                    args_schema=(
                        DTensorSpec(
                            mesh=mesh,
                            placements=tuple([Replicate()] * mesh.ndim),
                            tensor_meta=input_spec.tensor_meta,
                        )
                    )
                    + args[1:],
                    kwargs_schema=op_schema.kwargs_schema,
                )
            ],
        )

    input_shape = list(input_spec.tensor_meta.shape)

    in_out_shape_same = True
    output_continuous = True
    for in_s, out_s in zip(input_shape, output_shape):
        if in_s != out_s:
            in_out_shape_same = False
            break
    if output_stride[-1] != 1:
        output_continuous = False
    if output_continuous:
        for i in range(len(output_stride) - 2, -1, -1):
            if output_stride[i] != output_shape[i + 1] * output_stride[i + 1]:
                output_continuous = False
                break
    # redistribute input to Replicate
    if not in_out_shape_same or not output_continuous:
        return OutputSharding(
            output_spec=None,
            schema_suggestions=[
                OpSchema(
                    op=op_schema.op,
                    args_schema=(
                        DTensorSpec(
                            mesh=mesh,
                            placements=tuple([Replicate()] * mesh.ndim),
                            tensor_meta=input_spec.tensor_meta,
                        )
                    )
                    + args[1:],
                    kwargs_schema=op_schema.kwargs_schema,
                )
            ],
        )

    local_in_out_shape = compute_local_shape(input_shape, mesh, input_spec.placements)
    local_stride = [1] * len(local_in_out_shape)
    for i in range(len(local_in_out_shape) - 1, 0, -1):
        local_stride[i - 1] = local_stride[i] * local_in_out_shape[i]
    suggested_schema = OpSchema(
        op=op_schema.op,
        args_schema=args[:1] + (tuple(local_in_out_shape), tuple(local_stride)) + args[3:],
        kwargs_schema=op_schema.kwargs_schema,
    )
    return OutputSharding(
        output_spec=DTensorSpec(
            mesh=mesh,
            placements=input_spec.placements,
            tensor_meta=TensorMeta(tuple(output_shape), tuple(output_stride), input_spec.tensor_meta.dtype),
        ),
        schema_suggestions=[suggested_schema],
        needs_redistribute=True,
    )


@register_op_strategy(aten.bucketize.Tensor)
def gen_bucketize_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> StrategyType:
    """
    Just propagate input sharding, but expect replicated for boundaries input.
    """
    input_strategy = op_schema.args_schema[0]
    bucketize_strategy = OpStrategy([])
    assert isinstance(input_strategy, OpStrategy)
    for arg_strategy in input_strategy.strategies:
        arg_spec = DTensorSpec(mesh, arg_strategy.output_spec.placements)
        replica_spec = DTensorSpec(mesh, tuple([Replicate()] * mesh.ndim))
        bucketize_strategy.strategies.append(
            PlacementStrategy(output_spec=arg_spec, input_specs=(arg_spec, replica_spec))
        )

    return bucketize_strategy


# NOTE: we change to use rule-based way because we want to change non tensor args
@register_prop_rule(aten.slice.Tensor, schema_info=RuntimeSchemaInfo(1))
def prop_slice(op_schema: OpSchema) -> OutputSharding:
    """
    forwards all shardings except the slice dimension.
    """
    defaults = (None, 0, None, None, 1)
    input_spec, dim, start, end, step = op_schema.args_schema + defaults[len(op_schema.args_schema) :]
    assert isinstance(input_spec, DTensorSpec)
    input_shape = input_spec.tensor_meta.shape
    input_ndim = len(input_shape)
    assert isinstance(dim, int)
    if start is None:
        start = 0
    if end is None or end > input_shape[dim]:
        end = input_shape[dim]
    assert isinstance(start, int)
    assert isinstance(end, int)
    assert isinstance(step, int)

    mesh = input_spec.mesh

    # normalize args
    slice_dim = normalize_dim(dim, input_ndim)
    start = normalize_dim(start, input_shape[dim])
    end = normalize_dim(end, input_shape[dim])

    redundant_slice = start == 0 and end == input_shape[dim] and step == 1
    # calculate slice tensor meta
    output_shape_list = list(input_shape)
    output_shape_list[dim] = end - start

    if not is_tensor_dim_sharded(input_spec, dim=slice_dim) or redundant_slice:
        # only add the strategy if the slice dim is not sharded
        out_spec = DTensorSpec(mesh, input_spec.placements)
        return OutputSharding(output_spec=out_spec)
    if is_tensor_dim_interleaved_sharded(input_spec, dim=slice_dim):
        interleaved_size = None
        interleaved_shard_mesh_dim = None
        for i, p in enumerate(input_spec.placements):
            if p.is_interleaved_shard(dim=slice_dim):
                if interleaved_size is None:
                    interleaved_size = p.interleaved_size
                    interleaved_shard_mesh_dim = i
                else:
                    raise NotImplementedError(
                        "for now, we don't support slice tensor along dim which is interleaved sharded two or more times"
                    )
        interleaved_unit_size = input_spec.tensor_meta.shape[slice_dim] // interleaved_size
        if step != 1:
            raise NotImplementedError("for now, we only support constant 1 step in slice op")

        slice_size = end - start
        if slice_size % interleaved_unit_size != 0 or start % interleaved_unit_size != 0:
            raise NotImplementedError(
                "for now, we only support slice boundary strictly aligning with the sharding spec"
            )
        new_placements = list(copy.deepcopy(input_spec.placements))
        new_interleaved_size = slice_size // interleaved_unit_size
        if new_interleaved_size == 1:
            new_placements[interleaved_shard_mesh_dim] = Shard(slice_dim)
        else:
            new_placements[interleaved_shard_mesh_dim] = InterleavedShard(slice_dim, new_interleaved_size)
        out_spec = DTensorSpec(mesh, new_placements)
        return OutputSharding(
            output_spec=out_spec,
            schema_suggestions=[
                OpSchema(
                    op_schema.op,
                    args_schema=(
                        input_spec,
                        dim,
                        start // mesh.size(interleaved_shard_mesh_dim),
                        end // mesh.size(interleaved_shard_mesh_dim),
                        step,
                    ),
                    kwargs_schema=op_schema.kwargs_schema,
                )
            ],
            needs_redistribute=True,
        )


@register_op_strategy([aten._local_scalar_dense.default])
def replica_only_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> StrategyType:
    """Only allow replication on the input/ouput"""
    replicate_spec = DTensorSpec(mesh, tuple([Replicate()] * mesh.ndim))
    return OpStrategy([PlacementStrategy(replicate_spec)])


@register_prop_rule(aten.select.int)
def _prop_select(op_schema: OpSchema) -> OutputSharding:
    tensor, dim = op_schema.args_schema[:2]
    assert isinstance(tensor, DTensorSpec)
    assert isinstance(dim, int)
    placements: Sequence[Placement] = tensor.placements
    assert all(not p.is_shard(dim) for p in placements), "DTensor does not support select on sharded dimension."

    # select will remove one dimension, decrement dim of Shard placements by 1
    # if they are larger than dim.
    new_placements: List[Placement] = []
    for p in placements:
        # Using isinstance instead of is_shard so that mypy won't complain
        # about accessing dim attribute.
        if isinstance(p, Shard) and p.dim > dim:
            new_placements.append(Shard(p.dim - 1))
        else:
            new_placements.append(p)

    return OutputSharding(output_spec=DTensorSpec(mesh=tensor.mesh, placements=tuple(new_placements)))


@register_prop_rule(aten.gather.default, schema_info=RuntimeSchemaInfo(1))
def prop_gather(op_schema: OpSchema) -> OutputSharding:
    values_spec, dim, indices_spec = op_schema.args_schema

    assert isinstance(values_spec, DTensorSpec)
    assert isinstance(dim, int)
    assert isinstance(indices_spec, DTensorSpec)

    return OutputSharding(
        output_spec=DTensorSpec(
            mesh=values_spec.mesh,
            placements=values_spec.placements,
        ),
    )


@register_op_strategy([aten.scatter_.value, aten.scatter.value, aten.scatter_.src, aten.scatter.src])
def scatter_value(mesh: DeviceMesh, op_schema: OpSchema) -> StrategyType:
    value, _, index, src = op_schema.args_schema
    if isinstance(src, OpStrategy):
        src_target = src.strategies[0].output_spec
    else:
        src_target = src
    value_target = value.strategies[0].output_spec
    index_target = index.strategies[0].output_spec

    if isinstance(src, OpStrategy):
        output_spec = DTensorSpec(mesh=mesh, placements=src_target.placements)
    else:
        output_spec = DTensorSpec(mesh=mesh, placements=[Replicate()])
    input_specs = [value_target, index_target]
    if isinstance(src, OpStrategy):
        input_specs.append(src_target)
    return OpStrategy(
        [
            PlacementStrategy(
                output_spec=output_spec,
                input_specs=input_specs,
            )
        ]
    )


# (Hongyu): allow partial index_put here
@register_prop_rule([aten.index_put_.default, aten.index_put.default])
def index_put_rule(op_schema: OpSchema) -> OutputSharding:
    src_spec: DTensorSpec = op_schema.args_schema[0]
    value_spec: DTensorSpec = op_schema.args_schema[2]
    assert (
        src_spec.placements == value_spec.placements
    ), "Currently we only allow equal placements for src and value in index_put op"
    return OutputSharding(src_spec)


@register_op_strategy([aten.constant_pad_nd.default])
def pad_const(mesh: DeviceMesh, op_schema: OpSchema) -> StrategyType:
    input = op_schema.args_schema[0]
    input_spec: DTensorSpec = input.strategies[0].output_spec
    output_spec = DTensorSpec(mesh, tuple(input_spec.placements))
    return OpStrategy([PlacementStrategy(output_spec)])


def unshard_tensor_dim(placements: Sequence[Placement], dim: int) -> Tuple[Placement, ...]:
    """Disallow the given tensor dimension to be sharded"""
    return tuple(p if (not isinstance(p, Shard) or p.dim != dim) else Replicate() for p in placements)


def replicate_tensor_dim(placements: Sequence[Placement], dim: int) -> Tuple[Placement, ...]:
    """Force the given tensor dimension to be replicated"""
    # Not using p.is_shard() to avoid mypy complain about Placement not having
    # attribute dim.
    return tuple(Replicate() if p.is_partial() or isinstance(p, Shard) and p.dim == dim else p for p in placements)


@register_op_strategy(aten.slice_scatter.default, schema_info=RuntimeSchemaInfo(2))
def gen_slice_scatter_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> StrategyType:
    # 1. number of dimensions in input and src need to match.
    # 2. number of elements on all non-dim need to match between input and src.
    # 3. numer of elements in src in dim need to match the slice size.
    # Given the above:
    # - We suggest for src to follow the sharding of input, except on the scatter dimension,
    #   where our best bet for now is to make them replicated as a fall-back.
    #   TODO: Ideally we'd like to make sure the output is re-sharded afterwards to keep input sharding.

    input_strategy = op_schema.args_schema[0]
    assert isinstance(input_strategy, OpStrategy)
    input_ndim = input_strategy.output_ndim
    slice_dim = cast(int, op_schema.args_schema[2]) if len(op_schema.args_schema) > 2 else 0
    slice_dim = normalize_dim(slice_dim, input_ndim)

    slice_scatter_strategy = OpStrategy([])
    # by default follow the input strategy for both input and src
    for arg_strategy in input_strategy.strategies:
        arg_spec = arg_strategy.output_spec
        if not (is_tensor_dim_sharded(arg_spec, dim=slice_dim) or is_tensor_partial(arg_spec)):
            # only add the strategy if the slice_scatter dim is not sharded or partial
            slice_scatter_strategy.strategies.append(PlacementStrategy(output_spec=arg_spec))

    if not slice_scatter_strategy.strategies:
        # if all strategies are filtered out, replicating all specs on slice_scatter dim
        # of the input strategy, and use that as the op strategy
        for arg_strategy in input_strategy.strategies:
            arg_spec = arg_strategy.output_spec
            replicate_spec = DTensorSpec(mesh, replicate_tensor_dim(arg_spec.placements, dim=slice_dim))
            slice_scatter_strategy.strategies.append(PlacementStrategy(output_spec=replicate_spec))
    return slice_scatter_strategy


@register_prop_rule(aten.index_select.default, schema_info=RuntimeSchemaInfo(1))
def prop_index_select(op_schema: OpSchema) -> OutputSharding:
    values_spec, dim, indices_spec = op_schema.args_schema

    assert isinstance(values_spec, DTensorSpec)
    assert isinstance(dim, int)
    assert isinstance(indices_spec, DTensorSpec)

    all_indices_spec: List[Optional[DTensorSpec]] = [
        indices_spec if dim == i else None for i in range(values_spec.ndim)
    ]

    result = prop_index(
        OpSchema(
            op=op_schema.op,
            args_schema=(values_spec, all_indices_spec),
            kwargs_schema=op_schema.kwargs_schema,
        )
    )
    if result.schema_suggestions:
        result.schema_suggestions = [
            OpSchema(
                op=op_schema.op,
                args_schema=(s.args_schema[0], dim, s.args_schema[1][dim]),
                kwargs_schema=op_schema.kwargs_schema,
            )
            for s in result.schema_suggestions
        ]
    return result


@register_prop_rule(aten.index.Tensor, schema_info=RuntimeSchemaInfo(needs_pytree=True))
def prop_index(op_schema: OpSchema) -> OutputSharding:
    """
    Expect replicated on the first input; _mostly_ pointwise on the second input.
    TODO: exception: when the dtype of second input is "bool", then a torch.nonzero needs to be triggered first.
    """
    # Current sharding constraints:
    # For values:
    #   1. We currently require that the dimension of values_spec be replicated or partial
    #      if they are being indexed on.
    #   2. Other dimensions of values_spec can remain sharded if they are so.
    # For indices:
    #   Indices can be either sharded or replicated. All index tensors need to be sharded
    #   in a compatible way, following the pointwise rule (including resolving Partial
    #   into either sharded or replicated)

    values_spec, multi_indices_spec = op_schema.args_schema
    assert isinstance(values_spec, DTensorSpec)
    assert isinstance(multi_indices_spec, list)
    multi_indices_spec = cast(List[Optional[DTensorSpec]], multi_indices_spec)
    valid_indices_spec: List[Tuple[int, DTensorSpec]] = [
        (i, a) for i, a in enumerate(multi_indices_spec) if a is not None
    ]

    # 1. All indices have to be sharded equally. Moreover, indices can be broadcast.
    #    Here, we piggyback on the pointwise sharding rule for indices.
    indices_out = pointwise_rule(
        OpSchema(
            op=op_schema.op,
            args_schema=tuple(v[1] for v in valid_indices_spec),
            kwargs_schema={},
        )
    )
    need_reshard_on_indices = indices_out.output_spec is None

    if not need_reshard_on_indices:
        # this means that our inputs are already sharded properly and we will use that as our indices_spec
        assert isinstance(indices_out.output_spec, DTensorSpec)
        indices_spec: DTensorSpec = indices_out.output_spec
    else:
        assert indices_out.schema_suggestions is not None
        valid_indices_suggestion = indices_out.schema_suggestions[0]
        for i, v in enumerate(valid_indices_suggestion.args_spec):
            multi_indices_spec[valid_indices_spec[i][0]] = v
        # we'll need to call pointwise_rule again to see what's our ideal indices_spec and then
        # use that to compute our ideal values_spec
        indices_output_spec = pointwise_rule(valid_indices_suggestion).output_spec
        assert isinstance(indices_output_spec, DTensorSpec)
        indices_spec = indices_output_spec

    lookup_dims = {v[0] for v in valid_indices_spec}

    need_reshard_on_values = tuple(
        (isinstance(vp, Shard) and (vp.dim in lookup_dims or isinstance(ip, Shard)))
        for vp, ip in zip(values_spec.placements, indices_spec.placements)
    )

    if not need_reshard_on_indices and not any(need_reshard_on_values):
        value_placements = values_spec.placements

        all_dims_consecutive = all(b[0] - a[0] == 1 for b, a in zip(valid_indices_spec[1:], valid_indices_spec[:-1]))
        if all_dims_consecutive:
            # if all index vectors are consecutives, insert at the dimension of the first index
            insert_dim: int = valid_indices_spec[0][0]
        else:
            # else, insert on the first dimension
            insert_dim = 0

        def place(vp: Placement, ip: Placement) -> Placement:
            if isinstance(vp, Shard):
                return Shard(
                    vp.dim
                    if vp.dim < insert_dim
                    # accounts for the offset in output dimensions
                    else vp.dim + indices_spec.ndim - sum(1 if vp.dim > v[0] else 0 for v in valid_indices_spec)
                )
            if isinstance(ip, Shard):
                return Shard(ip.dim + insert_dim)
            # Partial or Replicated
            return vp

        value_placements = tuple(place(vp, ip) for vp, ip in zip(values_spec.placements, indices_spec.placements))
        result = OutputSharding(
            output_spec=DTensorSpec(
                mesh=values_spec.mesh,
                placements=value_placements,
            )
        )
        return result
    else:
        result = OutputSharding(
            output_spec=None,
            schema_suggestions=[
                OpSchema(
                    op=op_schema.op,
                    args_schema=(
                        DTensorSpec(
                            mesh=values_spec.mesh,
                            placements=tuple(
                                [
                                    Replicate() if need_reshard_on_values[i] else v
                                    for i, v in enumerate(values_spec.placements)
                                ]
                            ),
                            tensor_meta=values_spec.tensor_meta,
                        ),
                        multi_indices_spec,
                    ),
                    kwargs_schema=op_schema.kwargs_schema,
                )
            ],
            needs_redistribute=True,
        )
        return result


@register_prop_rule([aten.cat.default], schema_info=RuntimeSchemaInfo(1, needs_pytree=True))
def cat_rule(op_schema: OpSchema) -> OutputSharding:
    # torch.cat requires all tensors must either have the same shape (except
    # in the concatenating dimension) or be "empty". "Empty" here strictly means
    # tensor.shape is torch.Size([0]). When tensor.ndim > 1, it will be treated
    # as a non-empty tensor and the shape must match on non-cat dimensions.
    def is_empty(spec: DTensorSpec) -> bool:
        return list(spec.shape) == [0]

    # the first arg is a list of input tensor specs
    tensor_list_specs = cast(List[DTensorSpec], op_schema.args_schema[0])
    assert len(tensor_list_specs) > 0, "torch.cat expects a non-empty list of tensors"
    non_empty_specs = [spec for spec in tensor_list_specs if not is_empty(spec)]

    if len(non_empty_specs) == 0:
        # all tensors are empty, we can return any output sharding
        return OutputSharding(
            output_spec=DTensorSpec(
                mesh=tensor_list_specs[0].mesh,
                placements=tensor_list_specs[0].placements,
            )
        )

    assert all(
        spec.ndim == non_empty_specs[0].ndim for spec in non_empty_specs
    ), f"Expect all tensors to have same shape or empty, but got {tensor_list_specs}"
    assert all(
        spec.mesh == tensor_list_specs[0].mesh for spec in tensor_list_specs
    ), f"Expect all tensors to have same mesh, but got {tensor_list_specs}"

    # ndim will also be the result's ndim
    ndim = 1
    for spec in tensor_list_specs:
        ndim = max(ndim, spec.ndim)

    dim = 0  # default dim = 0
    if len(op_schema.args_schema) > 1:
        dim = cast(int, op_schema.args_schema[1])
    dim = normalize_dim(dim, ndim)

    # Make sure all tensors are replciated on cat dimension
    need_reshard = False
    tensor_list_specs_after: List[DTensorSpec] = []
    shard = None
    shard_idx = None
    for spec in tensor_list_specs:
        if (
            not is_empty(spec) and (is_tensor_dim_sharded(spec, dim=dim)) and shard is None
        ):  # Hongyu: allow torch.cat DTensors with Partial placements
            shard_idx = next(
                idx for idx, p in enumerate(spec.placements) if p.is_shard(dim)
            )  # find the index of target dim placement
            shard = spec.placements[shard_idx]
        tensor_list_specs_after.append(spec)

    tensor_list_specs = tensor_list_specs_after

    # align non-cat dimensions placements based on reshard cost
    non_empty_specs = [spec for spec in tensor_list_specs if not is_empty(spec)]

    if shard and any(
        spec.shape != non_empty_specs[0].shape or spec.placements[shard_idx] != shard for spec in non_empty_specs
    ):
        warnings.warn("Invalid concat Shard dim: tensors have different shapes or placements.")
        need_reshard = True
        tensor_list_specs_after = []
        for spec in tensor_list_specs:
            if not is_empty(spec) and (is_tensor_dim_sharded(spec, dim=dim)):
                tensor_list_specs_after.append(
                    DTensorSpec(
                        mesh=spec.mesh,
                        placements=replicate_tensor_dim(spec.placements, dim=dim),
                        tensor_meta=spec.tensor_meta,
                    )
                )
            else:
                tensor_list_specs_after.append(spec)
        tensor_list_specs = tensor_list_specs_after
        non_empty_specs = [spec for spec in tensor_list_specs if not is_empty(spec)]

    mesh = non_empty_specs[0].mesh
    ndim = non_empty_specs[0].ndim
    new_placements: List[Placement] = []
    for mesh_dim in range(mesh.ndim):
        # compute the minimum cost of resharding on this mesh_dim
        if any(spec.placements[mesh_dim] != non_empty_specs[0].placements[mesh_dim] for spec in non_empty_specs):
            # only reshard if there is a mismatch
            need_reshard = True
            reshard_cost = []
            for shard_dim in range(ndim):
                # compute the cost of resharding on this shard_dim
                cost: float = 0.0
                for spec in non_empty_specs:
                    global_shape = spec.shape
                    if global_shape[shard_dim] < mesh.size(mesh_dim):
                        # found one tensor where the shard_dim is smaller than
                        # mesh_dim. In this case, we cannot shard on this shard_dim,
                        # and hence set cost to infinity.
                        cost = +float("inf")
                    elif is_tensor_dim_sharded(spec, dim=shard_dim) or prod(global_shape) == 0:
                        continue
                    else:
                        local_shape = compute_local_shape(global_shape, spec.mesh, spec.placements)
                        cost += prod(local_shape) * spec.mesh.size(mesh_dim)
                reshard_cost.append(cost)
            best_dim = reshard_cost.index(min(reshard_cost))
            new_placements.append(Shard(best_dim))
        else:
            # no mismatch, keep the original placement
            new_placements.append(non_empty_specs[0].placements[mesh_dim])

    if need_reshard:
        tensor_list_specs_after = []
        for spec in tensor_list_specs:
            if is_empty(spec):
                tensor_list_specs_after.append(spec)
            else:
                tensor_list_specs_after.append(
                    DTensorSpec(
                        mesh=spec.mesh,
                        placements=tuple(new_placements),
                        tensor_meta=spec.tensor_meta,
                    )
                )

        return OutputSharding(
            output_spec=None,
            schema_suggestions=[
                OpSchema(
                    op=op_schema.op,
                    args_schema=(
                        tuple(tensor_list_specs_after),
                        *op_schema.args_schema[1:],
                    ),
                    kwargs_schema=op_schema.kwargs_schema,
                ),
            ],
        )
    else:
        if shard:
            output_placements = tuple(
                InterleavedShard(
                    p.dim,
                    p.interleaved_size * len(non_empty_specs) if p.is_interleaved_shard() else len(non_empty_specs),
                )
                if p.is_shard(dim)
                else p
                for p in non_empty_specs[0].placements
            )
            return OutputSharding(
                output_spec=DTensorSpec(
                    mesh=non_empty_specs[0].mesh,
                    placements=output_placements,
                ),
            )
        else:
            return OutputSharding(
                output_spec=DTensorSpec(
                    mesh=non_empty_specs[0].mesh,
                    placements=non_empty_specs[0].placements,
                ),
            )


def _derive_follow_placements_from_tuple_strategy(
    tuple_strategy: TupleStrategy,
) -> Sequence[Placement]:
    """
    derive the placements to follow from the tuple strategy, mainly used by
    aten.stack, aten.cat, where each operand have the same shape, and correspondingly
    expecting the same sharding
    """

    def merge_placement(cur_placement: Placement, new_placement: Placement) -> Placement:
        # semantic if we already have a follow placement, we
        # check each placement for the current arg placement
        # to see if we want to merge/adjust the placement to follow
        # the priority: Partial -> Shard -> Replicate
        if cur_placement == new_placement:
            return cur_placement

        if cur_placement.is_partial():
            if new_placement.is_shard():
                # follow new placement
                return new_placement
            elif new_placement.is_partial():
                # different partial types, we can't merge and have to replicate all here
                return Replicate()
            else:
                # follow partial
                return cur_placement
        elif cur_placement.is_shard():
            if new_placement.is_shard():
                # cur/new placement are different sharding (i.e. different shard dim)
                # currently fallback to replicate all args
                return Replicate()
            else:
                # for partial/replicate, follow the current shard placement
                return cur_placement
        else:
            # current replicate, just follow new placement
            return new_placement

    follow_placements: Optional[List[Placement]] = None
    for arg_strategy in tuple_strategy.childs:
        assert isinstance(arg_strategy, OpStrategy)
        for placement_strategy in arg_strategy.strategies:
            arg_placements = placement_strategy.output_spec.placements
            if follow_placements is None:
                follow_placements = list(arg_placements)
                continue
            mesh_ndim = len(follow_placements)
            assert follow_placements is not None
            for mesh_idx in range(mesh_ndim):
                # merge placements with the priority
                follow_placements[mesh_idx] = merge_placement(follow_placements[mesh_idx], arg_placements[mesh_idx])
    assert follow_placements is not None, "follow placements should not be None!"
    return follow_placements


def normalize_shard_for_stack(placements: Sequence[Placement], insert_dim: int = 0) -> Sequence[Placement]:
    # stack op would "insert" new dim, so all sharded dim >= the inserted dim need to
    # be normalized with the new Shard placement
    normalized_placements: List[Placement] = []
    for placement in placements:
        if isinstance(placement, Shard) and placement.dim >= insert_dim:
            normalized_placements.append(Shard(placement.dim + 1))
        else:
            normalized_placements.append(placement)
    return normalized_placements


@register_op_strategy(aten.stack.default, RuntimeSchemaInfo(1, needs_pytree=True))
def stack_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> StrategyType:
    args_schema = op_schema.args_schema
    input_tuple_strategy = args_schema[0]
    assert isinstance(input_tuple_strategy, TupleStrategy), f"{input_tuple_strategy}"
    first_input_strategy = input_tuple_strategy.childs[0]
    assert isinstance(first_input_strategy, OpStrategy), f"{first_input_strategy}"
    common_input_ndim = first_input_strategy.output_ndim
    dim = cast(int, args_schema[1]) if len(args_schema) > 1 else 0
    # normalize the dim to be within the common input ndim
    dim = normalize_dim(dim, common_input_ndim)

    follow_placements = _derive_follow_placements_from_tuple_strategy(input_tuple_strategy)
    follow_placements = normalize_shard_for_stack(follow_placements, dim)

    # create op strategy base on the follow placements
    op_strategy = OpStrategy([])

    input_specs = tuple(DTensorSpec(mesh, tuple(follow_placements)) for _ in range(len(input_tuple_strategy.childs)))
    op_strategy.strategies.append(
        PlacementStrategy(
            output_spec=DTensorSpec(mesh, tuple(follow_placements)),
            input_specs=input_specs,
        )
    )
    return op_strategy


@register_prop_rule([aten.split.Tensor, aten.split_with_sizes.default], schema_info=RuntimeSchemaInfo(1))
def split_rule(op_schema: OpSchema) -> OutputSharding:
    output_spec_list: List[DTensorSpec] = []
    input_spec = cast(DTensorSpec, op_schema.args_schema[0])
    ndim = input_spec.ndim
    split_size_or_sections = op_schema.args_schema[1]
    dim = cast(int, op_schema.args_schema[2]) if len(op_schema.args_schema) > 2 else 0
    dim = normalize_dim(dim, ndim)

    # TODO: tensor to split cannot have Partial
    # in its placements for now. Will need to
    # support in future.
    if input_spec.sums:
        raise NotImplementedError(
            f"splitting distributed tensor with " f"Partial placement is not implemented!\n" f"DTensorSpec={input_spec}"
        )

    # TODO: just like slice op, split replicates before
    # splitting on a sharded dimension
    need_reshard = False
    interleaved_shard = None
    interleaved_shard_idx = None
    if is_tensor_dim_sharded(input_spec, dim=dim):
        interleaved_shard_idx = next(
            idx for idx, p in enumerate(input_spec.placements) if p.is_shard(dim)
        )  # find the index of target dim placement
        interleaved_shard = input_spec.placements[interleaved_shard_idx]
        target_dim_len = input_spec.shape[dim]
        if (
            not interleaved_shard.is_interleaved_shard()
            or not isinstance(split_size_or_sections, int)
            or split_size_or_sections != target_dim_len // interleaved_shard.interleaved_size
        ):
            # TODO: allow split sizes which are mutiples of interleaved section
            if interleaved_shard.is_interleaved_shard():
                warnings.warn(
                    " Invalid split InterleavedShard dim: split_size_or_sections is not int or split size is not equal to target_dim_len // interleaved_size ",
                )
            need_reshard = True
            input_spec = DTensorSpec(
                mesh=input_spec.mesh,
                placements=unshard_tensor_dim(input_spec.placements, dim=dim),
                tensor_meta=input_spec.tensor_meta,
            )

    if need_reshard:
        return OutputSharding(
            None,
            schema_suggestions=[
                OpSchema(
                    op=op_schema.op,
                    args_schema=(input_spec,) + op_schema.args_schema[1:],
                    kwargs_schema=op_schema.kwargs_schema,
                ),
            ],
        )

    def size_split(N, i):
        # Last chunk will be smaller if the tensor size N
        # along the given dimension dim is not divisible by i.
        assert i > 0
        return [i] * (N // i) + ([N % i] if N % i != 0 else [])

    output_size_list = (
        size_split(input_spec.shape[dim], split_size_or_sections)
        if isinstance(split_size_or_sections, int)
        else split_size_or_sections
    )
    if interleaved_shard is None:
        output_spec_list = [
            DTensorSpec(
                mesh=input_spec.mesh,
                placements=input_spec.placements,
            )
            for _ in range(len(output_size_list))
        ]
        return OutputSharding(output_spec_list)
    else:
        mesh_dim_size = input_spec.mesh.shape[interleaved_shard_idx]
        local_split_size_or_sections = (
            split_size_or_sections // mesh_dim_size
        )  # compute the local size of split_size_or_sections
        output_spec_list = [
            DTensorSpec(
                mesh=input_spec.mesh,
                placements=tuple(Shard(p.dim) if p.is_interleaved_shard(dim) else p for p in input_spec.placements),
            )
            for _ in range(len(output_size_list))
        ]
        suggested_schema = OpSchema(
            op=op_schema.op,
            args_schema=(op_schema.args_schema[0], local_split_size_or_sections, dim),
            kwargs_schema=op_schema.kwargs_schema,
        )
        return OutputSharding(output_spec_list, schema_suggestions=[suggested_schema], needs_redistribute=True)


@register_prop_rule([aten.unbind.int], schema_info=RuntimeSchemaInfo(1))
def unbind_rule(op_schema: OpSchema) -> OutputSharding:
    output_spec_list: List[DTensorSpec] = []
    input_spec = cast(DTensorSpec, op_schema.args_schema[0])
    ndim = input_spec.ndim
    dim = cast(int, op_schema.args_schema[1] if len(op_schema.args_schema) > 1 else 0)
    dim = normalize_dim(dim, ndim)

    # TODO: tensor to unbind cannot have Partial
    # in its placements for now. Will need to
    # support in future.
    if input_spec.sums:
        raise NotImplementedError(
            f"splitting distributed tensor with " f"Partial placement is not implemented!\n" f"DTensorSpec={input_spec}"
        )

    # TODO: just like slice op, unbind replicates before
    # splitting on a sharded dimension
    need_reshard = False
    if is_tensor_dim_sharded(input_spec, dim=dim):
        need_reshard = True
        input_spec = DTensorSpec(
            mesh=input_spec.mesh,
            placements=unshard_tensor_dim(input_spec.placements, dim=dim),
            tensor_meta=input_spec.tensor_meta,
        )

    if need_reshard:
        return OutputSharding(
            None,
            schema_suggestions=[
                OpSchema(
                    op=op_schema.op,
                    args_schema=(input_spec,) + op_schema.args_schema[1:],
                    kwargs_schema=op_schema.kwargs_schema,
                ),
            ],
        )

    # we calculate output placements here.
    output_placements = []
    for p in input_spec.placements:
        if p.is_shard():
            sharded_dim = normalize_dim(p.dim, ndim)
            if sharded_dim < dim:
                output_placements.append(p)
            else:
                if isinstance(p, InterleavedShard):
                    output_placements.append(InterleavedShard(sharded_dim - 1, p.interleaved_size))
                else:
                    output_placements.append(Shard(sharded_dim - 1))
        else:
            output_placements.append(p)

    output_size_list = input_spec.shape[dim]
    output_spec_list = [
        DTensorSpec(
            mesh=input_spec.mesh,
            placements=tuple(output_placements),
        )
        for _ in range(output_size_list)
    ]
    return OutputSharding(output_spec_list)


@register_prop_rule([aten.index_add.default, aten.index_add_.default], schema_info=RuntimeSchemaInfo(1))
def index_add_rule(op_schema: OpSchema) -> OutputSharding:
    input_spec = cast(DTensorSpec, op_schema.args_schema[0])
    ndim = input_spec.ndim
    dim = cast(int, op_schema.args_schema[1])
    dim = normalize_dim(dim, ndim)
    index_spec = cast(DTensorSpec, op_schema.args_schema[2])
    src_spec = cast(DTensorSpec, op_schema.args_schema[3])

    if not index_spec.is_replicated():
        raise RuntimeError("index must be replicate for index_add op")

    if src_spec.sums != input_spec.sums:
        raise NotImplementedError("src and input should be both partial or non-partial for index_add op")

    if src_spec.ndim != input_spec.ndim:
        raise RuntimeError("invalid index_add op detected")

    assert not is_tensor_dim_sharded(input_spec, dim) and not is_tensor_dim_sharded(
        src_spec, dim
    ), "src or input can not be sharded on the index dim for adding"
    for input_p, src_p in zip(input_spec.placements, src_spec.placements):
        assert input_p == src_p, "src and input should be samley sharded on dims other than the index dim"
    return OutputSharding(input_spec)


@register_prop_rule(aten.alias.default)
def _prop_aten_alias(op_schema: OpSchema) -> OutputSharding:
    output_spec = cast(DTensorSpec, op_schema.args_schema[0])
    return OutputSharding(output_spec=output_spec)


@register_prop_rule(aten.nonzero.default)
def _nonzero_prop(op_schema: OpSchema):
    output_spec = cast(DTensorSpec, copy.deepcopy(op_schema.args_schema[0]))
    output_spec.tensor_meta = None
    return OutputSharding(output_spec=output_spec)
