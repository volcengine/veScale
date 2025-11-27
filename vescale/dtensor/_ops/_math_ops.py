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
from collections.abc import Sequence
from typing import cast

import torch
from torch.distributed.tensor._ops._math_ops import (
    Reduction,
    NormReduction,
    ReductionOpType,
    _NormPartial,
    _infer_reduction_dims,
    _infer_reduce_dims_map,
    _replicate_dims_start_at,
    _skip_dim,
    replicate_reduction_dims,
    get_placement_from_reduction_op,
    LINEAR_REDUCTION_OP_MAP,
)
from vescale.dtensor._dtensor_spec import DTensorSpec
from vescale.dtensor._op_schema import (
    OpSchema,
    OpStrategy,
    PlacementStrategy,
    RuntimeSchemaInfo,
    TupleStrategy,
)
from vescale.dtensor._ops.utils import (
    generate_redistribute_costs,
    is_tensor_evenly_shardable,
    register_op_strategy,
)
from vescale.dtensor._utils import normalize_to_torch_size
from vescale.dtensor.placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
    RaggedShard,
)

__all__ = [
    "Reduction",
    "NormReduction",
    "ReductionOpType",
    "_NormPartial",
    "_infer_reduction_dims",
    "_infer_reduce_dims_map",
    "_replicate_dims_start_at",
    "_skip_dim",
    "replicate_reduction_dims",
    "get_placement_from_reduction_op",
    "LINEAR_REDUCTION_OP_MAP",
    "map_placements_after_reduction",
    "common_reduction_strategy",
    "linear_reduction_strategy",
    "var_reduction_strategy",
    "vector_norm_strategy",
    "foreach_norm_strategy",
    "nll_loss_forward_strategy",
    "layer_norm_bwd_strategy",
]

"""
In this file, we modified map_placements_after_reduction.
The followings are kept but unmodified as they directly or indirectly used map_placements_after_reduction.
1. common_reduction_strategy
2. linear_reduction_strategy
3. var_reduction_strategy
4. vector_norm_strategy
5. foreach_norm_strategy
6. nll_loss_forward_strategy
7. layer_norm_bwd_strategy

Functions not mentioned above are removed as they are not modified
"""

aten = torch.ops.aten


def map_placements_after_reduction(
    placements: tuple[Placement, ...],
    reduction_dims: list[int],
    reduction_dims_map: list[int],
    reduction_op: ReductionOpType,
) -> tuple[Placement, ...]:
    """
    Map each placement based on the output shape after reduction.
    """
    new_placements: list[Placement] = []
    for placement in placements:
        if isinstance(placement, (Replicate, Partial)):
            new_placements.append(placement)
        elif isinstance(placement, RaggedShard):
            collapsed = False
            for dim in placement.dims:
                if dim in reduction_dims or reduction_dims_map[dim] == -1:
                    collapsed = True
                    break
            if collapsed:
                new_placements.append(get_placement_from_reduction_op(reduction_op))
            else:
                new_placements.append(placement)
        else:
            assert isinstance(placement, Shard)
            shard_dim = placement.dim
            new_shard_dim = reduction_dims_map[shard_dim]
            if new_shard_dim == -1 or shard_dim in reduction_dims:
                # if new_shard_dim collapsed or its in the reduction dims
                # (i.e. for the case where keepdims=True), we generate partial
                new_placements.append(get_placement_from_reduction_op(reduction_op))
            else:
                new_placements.append(Shard(new_shard_dim))
    return tuple(new_placements)


def common_reduction_strategy(
    input_strategy: OpStrategy,
    reduce_dims: list[int],
    keep_dim: bool = False,
    reduction_linear: bool = True,
    reduction_op: ReductionOpType = "sum",
) -> OpStrategy:
    """
    reduction_linear means that the reduction `f` follows this rule:
        f([f(a), f(b)]) = f([a, b])

    reduction linear should be super set of linearity.
    """
    # by default follow reduction input strategy
    reduction_strategy = OpStrategy([])

    for strtg in input_strategy.strategies:
        if not reduction_linear:
            # input placements for this strategy should clear out pending sum and sharding
            # on the reduction dimension
            input_placements = replicate_reduction_dims(strtg.output_spec.placements, reduce_dims)
        else:
            input_placements = strtg.output_spec.placements

        input_spec = DTensorSpec(
            mesh=input_strategy.mesh,
            placements=input_placements,
            tensor_meta=strtg.output_spec.tensor_meta,
        )

        reduce_dims_map = _infer_reduce_dims_map(reduce_dims, input_spec.ndim, keep_dim)
        out_placements = map_placements_after_reduction(
            input_spec.placements, reduce_dims, reduce_dims_map, reduction_op
        )
        redistribute_cost = [generate_redistribute_costs(input_strategy, input_spec)]
        reduction_strategy.strategies.append(
            PlacementStrategy(
                output_specs=DTensorSpec(
                    mesh=input_strategy.mesh,
                    placements=out_placements,
                ),
                input_specs=(input_spec,),
                redistribute_cost=redistribute_cost,
            )
        )

    return reduction_strategy


@register_op_strategy(list(LINEAR_REDUCTION_OP_MAP.keys()), schema_info=RuntimeSchemaInfo(1))
def linear_reduction_strategy(op_schema: OpSchema) -> OpStrategy:
    args_schema = op_schema.args_schema
    input_strategy = args_schema[0]
    assert isinstance(input_strategy, OpStrategy)

    dims = None
    if len(op_schema.args_schema) > 1:
        dims = _infer_reduction_dims(args_schema[1], input_strategy.ndim)

    reduce_dims = list(range(input_strategy.ndim)) if dims is None else dims

    keep_dim = len(op_schema.args_schema) > 2 and bool(op_schema.args_schema[2])
    reduction_op = LINEAR_REDUCTION_OP_MAP[op_schema.op]
    return common_reduction_strategy(
        input_strategy,
        reduce_dims,
        keep_dim=keep_dim,
        reduction_linear=True,
        reduction_op=reduction_op,
    )


@register_op_strategy(
    [aten.var.correction, aten.var.correction_out],
    schema_info=RuntimeSchemaInfo(1, ["keepdim"]),
)
def var_reduction_strategy(op_schema: OpSchema) -> OpStrategy:
    args_schema = op_schema.args_schema
    input_strategy = args_schema[0]
    assert isinstance(input_strategy, OpStrategy)
    dims = None
    if len(op_schema.args_schema) > 1:
        dims = _infer_reduction_dims(args_schema[1], input_strategy.ndim)

    reduce_dims = list(range(input_strategy.ndim)) if dims is None else dims

    keep_dim = cast(bool, op_schema.kwargs_schema.get("keepdim", False))
    return common_reduction_strategy(input_strategy, reduce_dims, keep_dim=keep_dim, reduction_linear=False)


@register_op_strategy([aten.linalg_vector_norm.default], schema_info=RuntimeSchemaInfo(1))
def vector_norm_strategy(op_schema: OpSchema) -> OpStrategy:
    args_schema = op_schema.args_schema
    input_strategy = args_schema[0]
    assert isinstance(input_strategy, OpStrategy)

    norm_type = args_schema[1] if len(args_schema) > 1 else 2
    assert isinstance(norm_type, (int, float, str)), f"{norm_type}"
    dim = args_schema[2] if len(args_schema) > 2 else None
    keepdim = args_schema[3] if len(args_schema) > 3 else False
    dims = _infer_reduction_dims(dim, input_strategy.ndim)
    reduce_dims = list(range(input_strategy.ndim)) if dims is None else dims
    return common_reduction_strategy(
        input_strategy,
        reduce_dims,
        keep_dim=cast(bool, keepdim),
        reduction_linear=True,
        reduction_op=NormReduction(norm_type),
    )


@register_op_strategy([aten._foreach_norm.Scalar], schema_info=RuntimeSchemaInfo(1, needs_pytree=True))
def foreach_norm_strategy(op_schema: OpSchema) -> TupleStrategy:
    args_schema = op_schema.args_schema
    input_tuple_strategy = args_schema[0]
    assert isinstance(input_tuple_strategy, TupleStrategy)
    norm_type = args_schema[1] if len(args_schema) > 1 else 2
    assert isinstance(norm_type, (int, float, str)), f"{norm_type}"
    output_tuple_strategy_childs: list[OpStrategy] = []
    for op_strategy in input_tuple_strategy.childs:
        assert isinstance(op_strategy, OpStrategy), f"{op_strategy}"
        reduce_dims = list(range(op_strategy.ndim))
        output_strategy = common_reduction_strategy(
            op_strategy,
            reduce_dims,
            reduction_linear=True,
            reduction_op=NormReduction(norm_type),
        )
        output_tuple_strategy_childs.append(output_strategy)
    return TupleStrategy(output_tuple_strategy_childs)


@register_op_strategy(
    [aten.nll_loss_forward.default, aten.nll_loss2d_forward.default],
    schema_info=RuntimeSchemaInfo(3),
)
def nll_loss_forward_strategy(op_schema: OpSchema) -> OpStrategy:
    mesh = op_schema.get_mesh_from_args()

    assert len(op_schema.args_schema) == 5

    (
        input_strategy,
        target_strategy,
        weight_strategy,
        reduction,
        _,
    ) = op_schema.args_schema
    input_strategy = cast(OpStrategy, input_strategy)
    target_strategy = cast(OpStrategy, target_strategy)
    reduction = cast(int, reduction)

    input_shape = input_strategy.shape
    channel_dim = 1 if len(input_shape) >= 2 else 0

    output_strategy = OpStrategy([])
    for idx, input_placement_strategy in enumerate(input_strategy.strategies):
        op_args_target_specs = []
        redistribute_costs = []

        # make sure input is replicated along the channel dim
        input_src_spec = input_placement_strategy.output_spec
        input_expected_spec = DTensorSpec(
            mesh=mesh,
            placements=replicate_reduction_dims(input_src_spec.placements, [channel_dim]),
            tensor_meta=input_src_spec.tensor_meta,
        )
        op_args_target_specs.append(input_expected_spec)
        redistribute_costs.append(generate_redistribute_costs(input_strategy, input_expected_spec))

        # target doesn't have channel dim, and it follows input on other dims
        target_src_spec = target_strategy.strategies[idx].output_spec
        target_expected_spec = DTensorSpec(
            mesh=mesh,
            placements=_skip_dim(input_expected_spec.placements, channel_dim),
            tensor_meta=target_src_spec.tensor_meta,
        )
        op_args_target_specs.append(target_expected_spec)
        redistribute_costs.append(generate_redistribute_costs(target_strategy, target_expected_spec))

        # weight tensor, if given, has to be a Tensor of size input_shape[channel_dim]
        # make sure it is replicated
        if weight_strategy is not None:
            assert isinstance(weight_strategy, OpStrategy)
            weight_src_spec = weight_strategy.strategies[idx].output_spec
            weight_expected_spec = DTensorSpec(
                mesh=mesh,
                placements=_replicate_dims_start_at(weight_src_spec.placements),
                tensor_meta=weight_src_spec.tensor_meta,
            )
            op_args_target_specs.append(weight_expected_spec)
            redistribute_costs.append(generate_redistribute_costs(weight_strategy, weight_expected_spec))

        if reduction == Reduction.NONE.value:
            output_expected_spec = target_expected_spec
            total_weight_expected_spec = DTensorSpec(mesh=mesh, placements=tuple([Replicate()] * mesh.ndim))
        else:
            if reduction == Reduction.MEAN.value:
                reduction_op = "avg"
                if not is_tensor_evenly_shardable(target_expected_spec.shape, target_expected_spec):
                    raise ValueError(
                        "The intermediate results of nll_loss cannot be evenly sharded, \
                        resulting in biased mean result."
                    )
            else:  # reduction == Reduction.SUM.value:
                reduction_op = "sum"
            reduce_dims = list(range(target_expected_spec.ndim))
            reduce_dims_map = _infer_reduce_dims_map(reduce_dims, target_expected_spec.ndim, keep_dim=False)
            out_placements = map_placements_after_reduction(
                target_expected_spec.placements,
                reduce_dims,
                reduce_dims_map,
                reduction_op,
            )
            output_expected_spec = DTensorSpec(
                mesh=mesh,
                placements=out_placements,
            )

            # whether reduction is sum or mean, the total weight has to be summed up if not replicated
            total_weight_placements = map_placements_after_reduction(
                target_expected_spec.placements,
                reduce_dims,
                reduce_dims_map,
                "sum",
            )
            total_weight_expected_spec = DTensorSpec(
                mesh=mesh,
                placements=total_weight_placements,
            )

        output_strategy.strategies.append(
            PlacementStrategy(
                output_specs=(output_expected_spec, total_weight_expected_spec),
                input_specs=op_args_target_specs,
                redistribute_cost=redistribute_costs,
            )
        )

    return output_strategy


@register_op_strategy(
    [aten.native_layer_norm_backward.default],
    schema_info=RuntimeSchemaInfo(2),
)
def layer_norm_bwd_strategy(op_schema: OpSchema) -> OpStrategy:
    # backward op does not need to validate the mesh since forward op has already done it
    mesh = op_schema.get_mesh_from_args(validate=False)

    # args must be: grad_out, input, normalized_shape, mean, rstd,
    # weight, bias, output_mask. For None weight and bias, their
    # corresponding objects will be None as well.

    assert len(op_schema.args_schema) == 8
    (
        grad_out_strategy,
        input_strategy,
        normalized_shape,
        mean_strategy,
        rstd_strategy,
        weight_strategy,
        bias_strategy,
        output_mask,
    ) = op_schema.args_schema

    assert isinstance(grad_out_strategy, OpStrategy)
    assert isinstance(input_strategy, OpStrategy)
    assert isinstance(mean_strategy, OpStrategy)
    assert isinstance(rstd_strategy, OpStrategy)

    assert isinstance(normalized_shape, (int, Sequence, torch.Size))
    normalized_size = normalize_to_torch_size(normalized_shape)
    input_ndim = input_strategy.ndim
    axis = input_ndim - len(normalized_size)
    outer_dims = list(range(axis))

    assert isinstance(output_mask, list) and len(output_mask) == 3

    # output triple: (d_input, d_weight, d_bias)
    out_tuple_strategy = OpStrategy([])
    for idx, input_placement_strategy in enumerate(input_strategy.strategies):
        # args for PlacementStrategy
        output_specs_list: list[DTensorSpec | None] = []
        input_specs_list: list[DTensorSpec] = []
        redistribute_costs = []

        input_src_spec = input_placement_strategy.output_spec
        # arg: grad_out
        # TODO: change the strategy to the following rule.
        # d_input is basically a product of element-wise mul of
        # grad_out, rstd, and normalized input, among which rstd
        # and normalized input (x_hat) should have the same sharding
        # placements, and grad_out's sharding is determined by the
        # pointwise result of x_hat and weight/bias.
        # TODO: now grad_out spec follows input spec. we may need
        # to change it to apply a pointwise rule over grad_out,
        # input, and weight.
        grad_out_target_spec = DTensorSpec(
            mesh=mesh,
            placements=_replicate_dims_start_at(input_src_spec.placements, axis),
            tensor_meta=input_src_spec.tensor_meta,
        )
        input_specs_list.append(grad_out_target_spec)
        redistribute_costs.append(generate_redistribute_costs(grad_out_strategy, grad_out_target_spec))
        output_specs_list.append(grad_out_target_spec if output_mask[0] else None)

        # arg: input
        input_target_spec = DTensorSpec(
            mesh=mesh,
            placements=_replicate_dims_start_at(input_src_spec.placements, axis),
            tensor_meta=input_src_spec.tensor_meta,
        )
        input_specs_list.append(input_target_spec)
        redistribute_costs.append(generate_redistribute_costs(input_strategy, input_target_spec))

        # arg: mean, rstd
        mean_src_spec = mean_strategy.strategies[idx].output_spec
        input_specs_list.append(mean_src_spec)
        redistribute_costs.append([0.0 for _ in mean_strategy.strategies])
        rstd_src_spec = rstd_strategy.strategies[idx].output_spec
        input_specs_list.append(rstd_src_spec)
        redistribute_costs.append([0.0 for _ in rstd_strategy.strategies])

        def _add_target_input_spec(strategy) -> DTensorSpec:
            # shared logic for setting the weight and bias target input specs
            assert isinstance(strategy, OpStrategy)
            src_spec = strategy.strategies[idx].output_spec
            # no need to redistribute since they should be replicated in forward pass
            input_specs_list.append(src_spec)
            redistribute_costs.append([0.0 for _ in strategy.strategies])
            return src_spec

        # arg: weight
        # d_weight = sum(grad_out * (input - mean) / rstd, outer_dim, keepdim=False)
        if weight_strategy is not None:
            weight_src_spec = _add_target_input_spec(weight_strategy)
            # TODO: now d_weight spec follows input spec w/ a reduction.
            # we may need to change to a pointwise rule over grad_out and
            # input, then apply a reduction.
            inp_placements = _replicate_dims_start_at(input_src_spec.placements, axis)
            reduce_dims_map = _infer_reduce_dims_map(outer_dims, input_src_spec.ndim, False)
            out_placements = map_placements_after_reduction(inp_placements, outer_dims, reduce_dims_map, "sum")
            weight_out_spec = DTensorSpec(
                mesh=mesh,
                placements=out_placements,
                tensor_meta=weight_src_spec.tensor_meta,
            )
            output_specs_list.append(weight_out_spec if output_mask[1] else None)
        else:
            assert output_mask[1] is False, (
                "output_mask[1] should not be `True` while weight argument is `None` in native_layer_norm_backward."
            )
            output_specs_list.append(None)

        # arg: bias
        # d_bias = sum(grad_out, outer_dim, keepdim=False)
        if bias_strategy is not None:
            bias_src_spec = _add_target_input_spec(bias_strategy)
            # d_bias spec follows a reduction over grad_out
            inp_placements = _replicate_dims_start_at(grad_out_target_spec.placements, axis)
            reduce_dims_map = _infer_reduce_dims_map(outer_dims, grad_out_target_spec.ndim, False)
            out_placements = map_placements_after_reduction(inp_placements, outer_dims, reduce_dims_map, "sum")
            bias_out_spec = DTensorSpec(
                mesh=mesh,
                placements=out_placements,
                tensor_meta=bias_src_spec.tensor_meta,
            )
            output_specs_list.append(bias_out_spec if output_mask[2] else None)
        else:
            assert output_mask[2] is False, (
                "output_mask[2] should not be `True` while bias argument is `None` in native_layer_norm_backward."
            )
            output_specs_list.append(None)

        out_tuple_strategy.strategies.append(
            PlacementStrategy(
                output_specs=tuple(output_specs_list),
                input_specs=input_specs_list,
                redistribute_cost=redistribute_costs,
            )
        )

    return out_tuple_strategy
