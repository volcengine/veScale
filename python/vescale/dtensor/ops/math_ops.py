################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

from typing import cast, List, Optional, Sequence, Tuple

import torch
import torch.distributed.distributed_c10d as c10d

from vescale.dtensor import DeviceMesh
from vescale.dtensor.op_schema import OpSchema, OutputSharding, RuntimeSchemaInfo, OpStrategy, PlacementStrategy
from vescale.dtensor.ops.common_rules import pointwise_rule
from vescale.dtensor.ops.utils import (
    as_list,
    generate_redistribute_costs,
    normalize_dims,
    normalize_dim,
    normalize_to_torch_size,
    register_op_strategy,
    register_prop_rule,
)
from vescale.dtensor.placement_types import DTensorSpec, Partial, Placement, Replicate, Shard

aten = torch.ops.aten


def _infer_reduction_dims(dims_arg: object, ndim: int) -> Optional[List[int]]:
    if dims_arg is None:
        return None
    dims = cast(List[int], as_list(dims_arg))
    dims = cast(List[int], normalize_dims(dims, ndim))
    empty_dims = [[0], [-1], []]
    if ndim == 0 and dims_arg in empty_dims:
        return None
    return dims


def _infer_reduce_dims_map(reduction_dims: List[int], input_ndim: int, keep_dim=False) -> List[int]:
    reduction_dims_map = []
    new_dim_count = 0
    for input_dim in range(input_ndim):
        if input_dim in reduction_dims and not keep_dim:
            # if input dim in reduction dims, mark it as -1
            reduction_dims_map.append(-1)
        else:
            # otherwise mark it as the new dim
            reduction_dims_map.append(new_dim_count)
            new_dim_count += 1

    return reduction_dims_map


def replicate_reduction_dims(placements: Tuple[Placement, ...], reduction_dims: List[int]) -> Tuple[Placement, ...]:
    # replicate the reduction dims if not reduction_linear
    new_placements: List[Placement] = []

    for p in placements:
        if p.is_partial():
            new_placements.append(Replicate())
        elif isinstance(p, Shard) and p.dim in reduction_dims:
            new_placements.append(Replicate())
        else:
            new_placements.append(p)

    return tuple(new_placements)


def map_placements_after_reduction(
    placements: Tuple[Placement, ...],
    reduction_dims: List[int],
    reduction_dims_map: List[int],
    reduction_op: c10d.ReduceOp.RedOpType,
) -> Tuple[Placement, ...]:
    """
    Map each placement based on the output shape after reduction.
    """
    new_placements: List[Placement] = []
    for placement in placements:
        if isinstance(placement, (Replicate, Partial)):
            new_placements.append(placement)
        else:
            assert isinstance(placement, Shard)
            shard_dim = placement.dim
            new_shard_dim = reduction_dims_map[shard_dim]
            if new_shard_dim == -1 or shard_dim in reduction_dims:
                # if new_shard_dim collapsed or its in the reduction dims
                # (i.e. for the case where keepdims=True), we generate partial
                new_placements.append(Partial(reduction_op))
            else:
                new_placements.append(Shard(reduction_dims_map[shard_dim]))
    return tuple(new_placements)


def common_reduction_strategy(
    mesh: DeviceMesh,
    input_strategy: OpStrategy,
    reduce_dims: List[int],
    keep_dim: bool = False,
    reduction_linear: bool = True,
    reduction_op: c10d.ReduceOp.RedOpType = c10d.ReduceOp.SUM,
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
            mesh=mesh,
            placements=input_placements,
            tensor_meta=strtg.output_spec.tensor_meta,
        )

        reduce_dims_map = _infer_reduce_dims_map(reduce_dims, input_spec.ndim, keep_dim)
        out_placements = map_placements_after_reduction(
            input_spec.placements, reduce_dims, reduce_dims_map, reduction_op
        )
        reduction_strategy.strategies.append(
            PlacementStrategy(
                output_spec=DTensorSpec(
                    mesh=mesh,
                    placements=out_placements,
                ),
                input_specs=(input_spec,),
            )
        )

    return reduction_strategy


LINEAR_REDUCTION_OP_MAP = {
    aten.all.default: c10d.ReduceOp.SUM,
    aten.all.dim: c10d.ReduceOp.SUM,
    aten.count_nonzero.default: c10d.ReduceOp.SUM,
    aten.linalg_vector_norm.default: c10d.ReduceOp.SUM,
    aten.sum.default: c10d.ReduceOp.SUM,
    aten.sum.dim_IntList: c10d.ReduceOp.SUM,
    aten.prod.default: c10d.ReduceOp.PRODUCT,
    aten.prod.dim_int: c10d.ReduceOp.PRODUCT,
    aten.prod.int_out: c10d.ReduceOp.PRODUCT,
    aten.mean.default: c10d.ReduceOp.AVG,
    aten.mean.dim: c10d.ReduceOp.AVG,
    aten.mean.out: c10d.ReduceOp.AVG,
    aten.max.default: c10d.ReduceOp.MAX,
    aten.max.dim: c10d.ReduceOp.MAX,
    aten.max.out: c10d.ReduceOp.MAX,
    aten.min.default: c10d.ReduceOp.MIN,
    aten.min.dim: c10d.ReduceOp.MIN,
    aten.min.out: c10d.ReduceOp.MIN,
}


@register_op_strategy(list(LINEAR_REDUCTION_OP_MAP.keys()), schema_info=RuntimeSchemaInfo(1))
def linear_reduction_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    args_schema = op_schema.args_schema
    input_strategy = args_schema[0]
    assert isinstance(input_strategy, OpStrategy)
    dims = None
    if len(op_schema.args_schema) > 1:
        dims = _infer_reduction_dims(args_schema[1], input_strategy.output_ndim)

    reduce_dims = list(range(input_strategy.output_ndim)) if dims is None else dims

    keep_dim = len(op_schema.args_schema) > 2 and bool(op_schema.args_schema[2])
    reduction_op = LINEAR_REDUCTION_OP_MAP[op_schema.op]
    return common_reduction_strategy(
        mesh,
        input_strategy,
        reduce_dims,
        keep_dim=keep_dim,
        reduction_linear=True,
        reduction_op=reduction_op,
    )


@register_op_strategy([aten.argmax.default, aten.argmin.default], schema_info=RuntimeSchemaInfo(1))
def arg_max_min(mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    args_schema = op_schema.args_schema
    input_strategy = args_schema[0]
    assert isinstance(input_strategy, OpStrategy)
    dim = None
    if len(op_schema.args_schema) > 1:
        dim = cast(int, op_schema.args_schema[1])
    reduce_dims = list(range(input_strategy.output_ndim)) if dim is None else [dim]
    keep_dim = len(op_schema.args_schema) > 2 and bool(op_schema.args_schema[2])
    return common_reduction_strategy(
        mesh,
        input_strategy,
        reduce_dims,
        keep_dim=keep_dim,
        reduction_linear=False,
    )


@register_op_strategy(
    [aten.var.correction, aten.var.correction_out],
    schema_info=RuntimeSchemaInfo(1, ["keepdim"]),
)
def var_reduction_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    args_schema = op_schema.args_schema
    input_strategy = args_schema[0]
    assert isinstance(input_strategy, OpStrategy)
    dims = None
    if len(op_schema.args_schema) > 1:
        dims = _infer_reduction_dims(args_schema[1], input_strategy.output_ndim)

    reduce_dims = list(range(input_strategy.output_ndim)) if dims is None else dims

    keep_dim = cast(bool, op_schema.kwargs_schema.get("keepdim", False))
    return common_reduction_strategy(mesh, input_strategy, reduce_dims, keep_dim=keep_dim, reduction_linear=False)


@register_op_strategy([aten.topk.default])
def topk(mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    input_strategy = op_schema.args_schema[0]
    dim = op_schema.args_schema[2] if len(op_schema.args_schema) > 2 else -1
    input_strategy = cast(OpStrategy, input_strategy)
    dim = cast(int, dim)
    dim = normalize_dim(dim, input_strategy.output_ndim)

    output_strategy = OpStrategy([])
    for input_placement_strategy in input_strategy.strategies:
        redistribute_costs = []
        input_src_spec = input_placement_strategy.output_spec

        # make sure input is replicated along the sort dim
        input_target_spec = DTensorSpec(
            mesh=mesh,
            placements=replicate_reduction_dims(input_src_spec.placements, [dim]),
            tensor_meta=input_src_spec.tensor_meta,
        )
        # TODO: change to vescale stype redistribution
        redistribute_costs.append(generate_redistribute_costs(input_strategy, input_target_spec))
        output_target_spec = DTensorSpec(
            mesh=mesh,
            placements=input_target_spec.placements,
        )
        output_strategy.strategies.append(
            PlacementStrategy(
                output_spec=output_target_spec,
                input_specs=[input_target_spec],
                redistribute_cost=redistribute_costs,
            )
        )

    return output_strategy


@register_op_strategy([aten._unique2.default])
def unique2(mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    input_strategy = op_schema.args_schema[0]

    output_strategy = OpStrategy([])
    for input_placement_strategy in input_strategy.strategies:
        redistribute_costs = []
        input_src_spec = input_placement_strategy.output_spec

        # make sure input is replicated
        input_target_spec = DTensorSpec(
            mesh=mesh,
            placements=(Replicate(),),
            tensor_meta=input_src_spec.tensor_meta,
        )
        # TODO: change to vescale stype redistribution
        redistribute_costs.append(generate_redistribute_costs(input_strategy, input_target_spec))
        output_spec = DTensorSpec(mesh=mesh, placements=[Replicate()])

        output_strategy.strategies.append(
            PlacementStrategy(
                output_spec=output_spec,
                input_specs=[input_target_spec],
                redistribute_cost=redistribute_costs,
            )
        )

    return output_strategy


@register_prop_rule([aten._log_softmax.default, aten._softmax.default], schema_info=RuntimeSchemaInfo(1))
def softmax_rule(op_schema: OpSchema) -> OutputSharding:
    input_spec, softmax_dim, _ = op_schema.args_schema
    input_spec = cast(DTensorSpec, input_spec)
    softmax_dim = cast(int, softmax_dim)
    dim_map = input_spec.dim_map
    if softmax_dim < len(dim_map) and dim_map[softmax_dim] >= 0:
        raise RuntimeError("Cannot run softmax on sharding dimension!")
    return OutputSharding(input_spec)


@register_prop_rule(
    [
        aten._log_softmax_backward_data.default,
        aten._softmax_backward_data.default,
    ],
    schema_info=RuntimeSchemaInfo(2),
)
def softmax_bwd_rule(op_schema: OpSchema) -> OutputSharding:
    grad_out_spec, out_spec, softmax_dim, _ = op_schema.args_schema
    grad_out_spec = cast(DTensorSpec, grad_out_spec)
    out_spec = cast(DTensorSpec, out_spec)
    softmax_dim = cast(int, softmax_dim)
    grad_out_dim_map = grad_out_spec.dim_map
    out_dim_map = out_spec.dim_map
    if softmax_dim < len(grad_out_dim_map) and (grad_out_dim_map[softmax_dim] >= 0 or out_dim_map[softmax_dim] >= 0):
        raise RuntimeError("Cannot run _softmax_backward_data on sharding dimension!")
    return pointwise_rule(op_schema)


@register_op_strategy(
    [aten.native_layer_norm.default],
    schema_info=RuntimeSchemaInfo(1),
)
def layer_norm_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    # args must be: input, normalized_shape, weight, bias, eps
    # for None weight and bias, their corresponding objects will
    # be None as well. layer_norm_strategy returns one OpStrategy
    # for the triple return values (out, mean, rstd).
    assert len(op_schema.args_schema) == 5
    (
        input_strategy,
        normalized_shape,
        weight_strategy,
        bias_strategy,
        _,
    ) = op_schema.args_schema

    # the current layer norm implementation requires that all
    # input DTensor's sharding must be in form of OpStrategy
    assert isinstance(input_strategy, OpStrategy)
    assert isinstance(normalized_shape, (int, Sequence, torch.Size))
    normalized_size = normalize_to_torch_size(normalized_shape)

    input_ndim = input_strategy.output_ndim
    axis = input_ndim - len(normalized_size)

    # we use OpStrategy because the output (out, mean, rstd)
    # should have the same placements
    output_strategy = OpStrategy([])
    for idx, input_placement_strategy in enumerate(input_strategy.strategies):
        op_args_target_specs = []
        input_src_spec = input_placement_strategy.output_spec

        # for the input tensor, we replicate it on the inner dims if necessary
        # TODO: we can avoid forcing the redistribution once we figure out
        # how to decompose layer norm
        input_target_spec = DTensorSpec(
            mesh=mesh,
            placements=_replicate_dims_start_at(input_src_spec.placements, axis),
            tensor_meta=input_src_spec.tensor_meta,
        )
        op_args_target_specs.append(input_target_spec)

        if weight_strategy is not None:
            assert isinstance(weight_strategy, OpStrategy)
            weight_src_spec = weight_strategy.strategies[idx].output_spec

            # for the weight tensor, we replicate it on all dims if necessary
            # TODO: we can avoid forcing the redistribution once we figure out
            # how to decompose layer norm
            weight_target_spec = DTensorSpec(
                mesh=mesh,
                placements=_replicate_dims_start_at(weight_src_spec.placements),
                tensor_meta=weight_src_spec.tensor_meta,
            )
            op_args_target_specs.append(weight_target_spec)

        if bias_strategy is not None:
            assert isinstance(bias_strategy, OpStrategy)
            bias_src_spec = bias_strategy.strategies[idx].output_spec

            # for the bias tensor, we replicate it on all dims if necessary
            # TODO: we can avoid forcing the redistribution once we figure out
            # how to decompose layer norm
            bias_target_spec = DTensorSpec(
                mesh=mesh,
                placements=_replicate_dims_start_at(bias_src_spec.placements),
                tensor_meta=bias_src_spec.tensor_meta,
            )
            op_args_target_specs.append(bias_target_spec)

        # the output spec is the same as input spec
        output_target_spec = input_target_spec
        output_strategy.strategies.append(
            PlacementStrategy(
                output_spec=output_target_spec,
                input_specs=op_args_target_specs,
            )
        )

    return output_strategy


@register_prop_rule(aten.native_layer_norm_backward.default)
def _prop_native_layer_norm_backward(op_schema: OpSchema) -> OutputSharding:
    (
        grad,
        input,
        normalized_shape,
        result1,
        result2,
        weight,
        bias,
        grad_input_mask,
    ) = op_schema.args_schema
    assert isinstance(grad, DTensorSpec)
    assert isinstance(grad_input_mask, (list, tuple))
    if weight is not None:
        assert isinstance(weight, DTensorSpec)
        assert all(isinstance(s, Replicate) for s in weight.placements)
    if bias is not None:
        assert isinstance(bias, DTensorSpec)
        assert all(isinstance(s, Replicate) for s in bias.placements)

    # NOTE: we preassume this fact: input of layernorm is either sharded
    # only along seq_len dim or replicate.

    grad_placements = grad.placements
    if all(isinstance(s, Replicate) for s in grad_placements):
        weight_grad = (
            DTensorSpec(
                mesh=weight.mesh,
                placements=tuple([Replicate()] * weight.mesh.ndim),
            )
            if weight
            else None
        )
        bias_grad = (
            DTensorSpec(
                mesh=bias.mesh,
                placements=tuple([Replicate()] * bias.mesh.ndim),
            )
            if bias
            else None
        )
        return OutputSharding(
            # NOTE: type errors below are legit. This is because DTensor currently
            # doesn't support Optional return values. Need to be fixed in DTensor repo.
            output_spec=(
                grad if grad_input_mask[0] else None,
                weight_grad if grad_input_mask[1] else None,
                bias_grad if grad_input_mask[2] else None,
            ),
        )
    sharded_input_mesh_dims = {}
    for i, s in enumerate(grad_placements):
        if s.is_replicate():
            continue
        assert not s.is_partial(), "input/output of layernorm must not be partial"
        input_dim = s.dim
        if input_dim not in sharded_input_mesh_dims:
            sharded_input_mesh_dims[input_dim] = []
        sharded_input_mesh_dims[input_dim].append(i)
    assert len(sharded_input_mesh_dims) == 1, "input of layernorm must be sharded along only one dim"
    param_grad_placements = [Replicate()] * weight.mesh.ndim
    sharded_input_dim = list(sharded_input_mesh_dims.keys())[0]
    for mesh_dim in sharded_input_mesh_dims[sharded_input_dim]:
        param_grad_placements[mesh_dim] = Partial()

    weight_grad = (
        DTensorSpec(
            mesh=weight.mesh,
            placements=tuple(param_grad_placements),
        )
        if weight
        else None
    )
    bias_grad = (
        DTensorSpec(
            mesh=bias.mesh,
            placements=tuple(param_grad_placements),
        )
        if bias
        else None
    )
    return OutputSharding(
        # NOTE: type errors below are legit. This is because DTensor currently
        # doesn't support Optional return values. Need to be fixed in DTensor repo.
        output_spec=(
            grad if grad_input_mask[0] else None,
            weight_grad if grad_input_mask[1] else None,
            bias_grad if grad_input_mask[2] else None,
        ),
    )


def _replicate_dims_start_at(placements: Sequence[Placement], start_dim: int = 0) -> Tuple[Placement, ...]:
    new_placements: List[Placement] = []
    for p in placements:
        if p.is_partial() or (isinstance(p, Shard) and p.dim >= start_dim):
            new_placements.append(Replicate())  # make it replicate
        else:
            new_placements.append(p)  # keep the placement
    return tuple(new_placements)
