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

import torch
from torch.utils._python_dispatch import _get_current_dispatch_mode

from vescale.dtensor._utils import compute_local_shape
from vescale.dtensor.op_schema import (
    OpSchema,
    OutputSharding,
    RuntimeSchemaInfo,
    PlacementStrategy,
    StrategyType,
    OpStrategy,
)
from vescale.dtensor._diff import EnablePartialMode
from vescale.dtensor.ops.common_rules import pointwise_rule
from vescale.dtensor.ops.utils import (
    generate_redistribute_costs,
    is_tensor_dim_sharded,
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
            enable_partial = False
            mode = _get_current_dispatch_mode()
            if isinstance(mode, EnablePartialMode):
                enable_partial = True

            output_spec = DTensorSpec(
                mesh=arg_spec.mesh,
                placements=tuple(
                    Replicate() if (isinstance(p, Partial) and not enable_partial) else p for p in arg_spec.placements
                ),
            )
            create_like_strategy.strategies.append(PlacementStrategy(output_spec=output_spec, input_specs=(arg_spec,)))

        else:
            create_like_strategy.strategies.append(PlacementStrategy(arg_spec))

    return create_like_strategy


@register_op_strategy(
    [
        aten.new_empty.default,
        aten.new_full.default,
        aten.new_ones.default,
        aten.new_zeros.default,
    ],
    schema_info=RuntimeSchemaInfo(1, ["dtype"]),
)
def new_factory_strategy(mesh: DeviceMesh, _) -> StrategyType:
    # TODO: maybe we should generate all possible shardings intead of just stay
    # replicated for new factory methods
    replica_spec = DTensorSpec(mesh, tuple([Replicate()] * mesh.ndim))
    return OpStrategy([PlacementStrategy(replica_spec)])


@register_prop_rule(
    aten.new_empty_strided.default,
    schema_info=RuntimeSchemaInfo(1, ["dtype"]),
)
def new_empty_strided_rule(op_schema: OpSchema) -> StrategyType:
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


@register_op_strategy(aten.slice.Tensor, schema_info=RuntimeSchemaInfo(1))
def gen_slice_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> StrategyType:
    """
    forwards all shardings except the slice dimension.
    """
    defaults = (None, 0, None, None, 1)
    input_strategy, dim, start, end, step = op_schema.args_schema + defaults[len(op_schema.args_schema) :]
    assert isinstance(input_strategy, OpStrategy)
    input_shape = input_strategy.output_shape
    input_ndim = input_strategy.output_ndim
    assert isinstance(dim, int)
    if start is None:
        start = 0
    if end is None or end > input_shape[dim]:
        end = input_shape[dim]
    assert isinstance(start, int)
    assert isinstance(end, int)
    assert isinstance(step, int)

    # normalize args
    slice_dim = normalize_dim(dim, input_ndim)
    start = normalize_dim(start, input_shape[dim])
    end = normalize_dim(end, input_shape[dim])

    redundant_slice = start == 0 and end == input_shape[dim] and step == 1
    # calculate slice tensor meta
    output_shape_list = list(input_shape)
    output_shape_list[dim] = end - start
    out_tensor_meta = TensorMeta(
        shape=torch.Size(output_shape_list),
        stride=input_strategy.output_stride,
        dtype=input_strategy.output_dtype,
    )

    slice_strategy = OpStrategy([])

    for arg_strategy in input_strategy.strategies:
        arg_spec = arg_strategy.output_spec
        if not is_tensor_dim_sharded(arg_spec, dim=slice_dim) or redundant_slice:
            # only add the strategy if the slice dim is not sharded
            out_spec = DTensorSpec(mesh, arg_spec.placements, out_tensor_meta)
            slice_strategy.strategies.append(PlacementStrategy(output_spec=out_spec))
    if not slice_strategy.strategies:
        # if all strategies are filtered out, unsharding all specs on slice dim
        # of the input strategy, and use that as the op strategy
        for arg_strategy in input_strategy.strategies:
            arg_spec = arg_strategy.output_spec
            unshard_spec = DTensorSpec(
                mesh,
                unshard_tensor_dim(arg_spec.placements, dim=slice_dim),
                out_tensor_meta,
            )
            slice_strategy.strategies.append(PlacementStrategy(output_spec=unshard_spec))
    return slice_strategy


@register_op_strategy([aten._local_scalar_dense.default])
def replica_only_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> StrategyType:
    """Only allow replication on the input/ouput"""
    replicate_spec = DTensorSpec(mesh, tuple([Replicate()] * mesh.ndim))
    return OpStrategy([PlacementStrategy(replicate_spec)])


@register_op_strategy([aten.select.int])
def index_select(mesh: DeviceMesh, op_schema: OpSchema) -> StrategyType:
    """Only allow replication on the input/ouput"""
    (
        input,
        dim,
        _,
    ) = op_schema.args_schema
    dim = normalize_dim(input.output_ndim, dim)
    input_spec: DTensorSpec = input.strategies[0].output_spec
    has_shard_on_dim = any(placement.is_shard(dim=dim) for placement in input_spec.placements)
    assert not has_shard_on_dim, "currently not support shard on select dim"
    new_placements = input_spec.placements
    for pm in new_placements:
        if isinstance(pm, Shard) and pm.dim > dim:
            pm.dim -= 1
    output_spec = DTensorSpec(mesh, tuple(new_placements))
    return OpStrategy([PlacementStrategy(output_spec)])


@register_op_strategy([aten.scatter_.value, aten.scatter.value, aten.scatter_.src, aten.scatter.src])
def scatter_value(mesh: DeviceMesh, op_schema: OpSchema) -> StrategyType:
    value, _, index, src = op_schema.args_schema
    value_target = DTensorSpec(mesh, [Replicate()], value.strategies[0].output_spec.tensor_meta)
    index_target = DTensorSpec(mesh, [Replicate()], index.strategies[0].output_spec.tensor_meta)
    src_target = (
        DTensorSpec(mesh, [Replicate()], src.strategies[0].output_spec.tensor_meta)
        if isinstance(src, OpStrategy)
        else src
    )

    redistribute_value_costs = []
    # TODO: change to vescale stype redistribution
    redistribute_value_costs.append(generate_redistribute_costs(value, value_target))
    redistribute_index_costs = []
    # TODO: change to vescale stype redistribution
    redistribute_index_costs.append(generate_redistribute_costs(index, index_target))
    redistribute_costs = [[x + y for x, y in zip(redistribute_value_costs[0], redistribute_index_costs[0])]]
    if isinstance(src, OpStrategy):
        redistribute_src_costs = []
        # TODO: change to vescale stype redistribution
        redistribute_src_costs.append(generate_redistribute_costs(src, src_target))
        redistribute_costs = [[x + y for x, y in zip(redistribute_costs[0], redistribute_src_costs[0])]]

    output_spec = DTensorSpec(mesh=mesh, placements=[Replicate()])
    input_specs = [value_target, index_target]
    if isinstance(src, OpStrategy):
        input_specs.append(src_target)
    return OpStrategy(
        [
            PlacementStrategy(
                output_spec=output_spec,
                input_specs=input_specs,
                redistribute_cost=redistribute_costs,
            )
        ]
    )


@register_op_strategy([aten.index_put_.default, aten.index_put.default])
def index_put(mesh: DeviceMesh, op_schema: OpSchema) -> StrategyType:
    """Set the Output with the index sharding"""
    (input, index_list, value) = op_schema.args_schema

    value_spec: DTensorSpec = value.strategies[0].output_spec
    output_spec = DTensorSpec(mesh, tuple(value_spec.placements))
    return OpStrategy([PlacementStrategy(output_spec)])


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


@register_prop_rule(aten.cat.default, schema_info=RuntimeSchemaInfo(1, needs_pytree=True))
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
    for spec in tensor_list_specs:
        if not is_empty(spec) and (is_tensor_dim_sharded(spec, dim=dim) or is_tensor_partial(spec)):
            need_reshard = True
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

    # align non-cat dimensions placements based on reshard cost
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
        # at this point, the cat dim is not sharded,
        return OutputSharding(
            output_spec=DTensorSpec(
                mesh=non_empty_specs[0].mesh,
                placements=non_empty_specs[0].placements,
            ),
        )


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
    output_spec_list = [
        DTensorSpec(
            mesh=input_spec.mesh,
            placements=input_spec.placements,
        )
        for _ in range(len(output_size_list))
    ]
    return OutputSharding(output_spec_list)


@register_prop_rule(aten.alias.default)
def _prop_aten_alias(op_schema: OpSchema) -> OutputSharding:
    output_spec = cast(DTensorSpec, op_schema.args_schema[0])
    return OutputSharding(output_spec=output_spec)
