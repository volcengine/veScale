################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

from typing import List

import numpy as np

import torch
from vescale.dtensor.op_schema import OpSchema, OutputSharding
from vescale.dtensor.ops.utils import register_prop_rule
from vescale.dtensor.placement_types import (
    DTensorSpec,
    TensorMeta,
)
from vescale.dtensor._utils import compute_local_shape

aten = torch.ops.aten


def slice_select_backward(op_schema, grad_output_spec, input_sizes, dim, start, end, step) -> OutputSharding:
    assert isinstance(grad_output_spec, DTensorSpec)
    assert isinstance(input_sizes, List)
    assert grad_output_spec.tensor_meta is not None
    grad_input_stride = list(np.cumprod(input_sizes[::-1])[:-1][::-1])
    grad_input_stride.append(1)
    dim_map = grad_output_spec.dim_map
    sums = grad_output_spec.sums

    grad_input_tensor_meta = TensorMeta(
        torch.Size(input_sizes),
        tuple(grad_input_stride),
        grad_output_spec.tensor_meta.dtype,
    )
    grad_input_spec = DTensorSpec.from_dim_map(
        grad_output_spec.mesh,
        dim_map,
        sums,
        tensor_meta=grad_input_tensor_meta,
    )

    new_input_sizes = compute_local_shape(input_sizes, grad_input_spec.mesh, grad_output_spec.placements)
    suggested_schema = None
    if new_input_sizes != tuple(input_sizes):
        suggested_schema = OpSchema(
            op_schema.op,
            (grad_output_spec, new_input_sizes, dim, start, end, step),
            op_schema.kwargs_schema,
            schema_info=op_schema.schema_info,
        )
        return OutputSharding(
            output_spec=grad_input_spec,
            schema_suggestions=[suggested_schema],
            needs_redistribute=True,
        )
    return OutputSharding(grad_input_spec)


@register_prop_rule(aten.slice_backward.default)
def slice_backward_rules(op_schema: OpSchema) -> OutputSharding:
    grad_output_spec, input_sizes, dim, start, end, step = op_schema.args_schema
    return slice_select_backward(op_schema, grad_output_spec, input_sizes, dim, start, end, step)


@register_prop_rule(aten.select_backward.default)
def index_select_backward(op_schema: OpSchema) -> OutputSharding:
    grad_output_spec, input_sizes, dim, start_index = op_schema.args_schema
    return slice_select_backward(op_schema, grad_output_spec, input_sizes, dim, start_index, start_index + 1, 1)


@register_prop_rule(aten.nll_loss_forward.default)
def nll_loss_forward_rules(op_schema: OpSchema) -> OutputSharding:
    input_spec = op_schema.args_schema[0]
    assert isinstance(input_spec, DTensorSpec)
    assert input_spec.tensor_meta is not None
    result_shape: List[int] = []
    result_stride: List[int] = []
    result_dim = 0
    total_weight_shape: List[int] = []
    total_weight_stride: List[int] = []
    total_weight_dim = 0

    result_tensor_meta = TensorMeta(
        torch.Size(result_shape),
        tuple(result_stride),
        input_spec.tensor_meta.dtype,
    )
    total_weight_tensor_meta = TensorMeta(
        torch.Size(total_weight_shape),
        tuple(result_stride),
        input_spec.tensor_meta.dtype,
    )
    result_spec = DTensorSpec.from_dim_map(
        input_spec.mesh,
        [-1 for _ in range(result_dim)],
        [],
        tensor_meta=result_tensor_meta,
    )
    total_weight_spec = DTensorSpec.from_dim_map(
        input_spec.mesh,
        [-1 for _ in range(total_weight_dim)],
        [],
        tensor_meta=total_weight_tensor_meta,
    )
    return OutputSharding([result_spec, total_weight_spec])


@register_prop_rule(aten.nll_loss_backward.default)
def nll_loss_backward_rules(op_schema: OpSchema) -> OutputSharding:
    input_spec = op_schema.args_schema[1]
    assert isinstance(input_spec, DTensorSpec)
    return OutputSharding(input_spec)
