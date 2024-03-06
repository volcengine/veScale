################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

from typing import cast

import torch

from vescale.dtensor.device_mesh import DeviceMesh
from vescale.dtensor.op_schema import OpSchema, OpStrategy, OutputSharding
from vescale.dtensor.ops.basic_strategy import gen_einsum_strategies
from vescale.dtensor.ops.utils import (
    generate_redistribute_costs,
    infer_broadcast_dims_map,
    is_tensor_shardable,
    map_placements_after_broadcast,
    register_op_strategy,
    register_prop_rule,
)
from vescale.dtensor.placement_types import DTensorSpec, InterleavedShard, Shard, TensorMeta

aten = torch.ops.aten


@register_prop_rule(aten.t.default)
def transpose_rule(op_schema: OpSchema) -> OutputSharding:
    # rule-based op sharding propagation will be deprecated. We only handle
    # aten.t.default here only.
    input_spec = op_schema.args_spec[0]
    out_placements = []
    for p in input_spec.placements:
        if p.is_interleaved_shard():
            p = cast(InterleavedShard, p)
            out_placements.append(InterleavedShard(1 - p.dim, p.interleaved_size))
        elif p.is_shard():
            p = cast(Shard, p)
            out_placements.append(Shard(1 - p.dim))
        else:
            out_placements.append(p)

    out_tensor_meta = None
    if input_spec.tensor_meta is not None:
        out_shape = torch.Size([input_spec.tensor_meta.shape[-1], input_spec.tensor_meta.shape[0]])
        out_stride = (input_spec.tensor_meta.stride[-1], input_spec.tensor_meta.stride[0])
        out_dtype = input_spec.tensor_meta.dtype
        out_tensor_meta = TensorMeta(out_shape, out_stride, out_dtype)

    return OutputSharding(
        output_spec=DTensorSpec(input_spec.mesh, out_placements, out_tensor_meta),
        schema_suggestions=None,
        failed_reason=None,
        needs_redistribute=False,
    )


def _mm_like_strategy(mm_equation: str, mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    # generate all possible strategies for mm
    lhs, rhs = op_schema.args_schema
    assert isinstance(lhs, OpStrategy)
    assert isinstance(rhs, OpStrategy)
    mm_strategy = gen_einsum_strategies(mm_equation, mesh, lhs, rhs)
    # filter out invalid strategies and associate costs
    # TODO(cery.zhai) add check here
    return mm_strategy


def _addmm_like_strategy(mm_equation: str, mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    self_strategy, mat1_strategy, mat2_strategy = op_schema.args_schema
    assert isinstance(self_strategy, OpStrategy)
    assert isinstance(mat1_strategy, OpStrategy)
    assert isinstance(mat2_strategy, OpStrategy)
    self_shape = self_strategy.output_shape
    mm_out_shape = torch.Size(
        [
            mat2_strategy.output_shape[-1] if i == len(mat1_strategy.output_shape) - 1 else dim_size
            for i, dim_size in enumerate(mat1_strategy.output_shape)
        ]
    )
    # generate all possible strategies for mm
    mm_strategy = gen_einsum_strategies(mm_equation, mesh, mat1_strategy, mat2_strategy)
    # filter out invalid strategies and associate costs
    strategies = mm_strategy.strategies
    strtg = strategies[0]
    # construct new strategy by consider the self arg
    assert strtg.input_specs is not None
    mat1_spec = strtg.input_specs[0]
    mat2_spec = strtg.input_specs[1]
    out_spec = strtg.output_spec

    # self arg's spec should follow the output of mm, but need
    # to consider broadcast for the self arg
    broadcast_dims_map = infer_broadcast_dims_map(mm_out_shape, self_shape)
    self_placements = map_placements_after_broadcast(out_spec.placements, mm_out_shape, broadcast_dims_map)
    self_spec = DTensorSpec(mesh=mesh, placements=self_placements)

    if is_tensor_shardable(mat1_strategy.output_shape, mat1_spec) and is_tensor_shardable(
        mat2_strategy.output_shape, mat2_spec
    ):
        # update input specs with new self spec
        strtg.input_specs = (self_spec, mat1_spec, mat2_spec)

        # associate costs
        redistribute_cost = [
            generate_redistribute_costs(self_strategy, self_spec),
            # generate_redistribute_costs(mat1_strategy, mat1_spec), # we do not support reshard by annotation
            # generate_redistribute_costs(mat2_strategy, mat2_spec),
        ]
        strtg.redistribute_cost = redistribute_cost
    mm_strategy.strategies = [strtg]
    return mm_strategy


@register_op_strategy(aten.mm.default)
def mm_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    return _mm_like_strategy("mk,kn->mn", mesh, op_schema)


@register_op_strategy(aten.addmm.default)
def addmm_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    return _addmm_like_strategy("mk,kn->mn", mesh, op_schema)


@register_op_strategy(aten.bmm.default)
def bmm_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    return _mm_like_strategy("bmk,bkn->bmn", mesh, op_schema)


@register_op_strategy(aten.baddbmm.default)
def baddmm_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    return _addmm_like_strategy("bmk,bkn->bmn", mesh, op_schema)
