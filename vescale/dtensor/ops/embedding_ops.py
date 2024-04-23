################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################
# implement matrix related ops for distributed tensor

import copy

import torch

from vescale.dtensor.op_schema import OpSchema, OutputSharding
from vescale.dtensor.ops.utils import (
    register_prop_rule,
    is_tensor_all_replicate,
    is_tensor_all_replicate_except_sharded_at_dim,
    is_tensor_partial,
)
from vescale.dtensor.placement_types import DTensorSpec, Partial, Replicate, Shard

aten = torch.ops.aten


# TODO: Enable BWD for embedding op.
@register_prop_rule(aten.embedding.default)
def embedding_rules(op_schema: OpSchema) -> OutputSharding:
    weight_spec, inp_spec = op_schema.args_spec
    if any(placement.is_shard(0) for placement in weight_spec.placements):
        raise NotImplementedError("DTensor does not support row-wise sharded embedding operation yet!")

    if all(placement.is_replicate() for placement in weight_spec.placements) and inp_spec.placements == [Shard(0)]:
        # Embedding table is replicated, input ids are sharded along batch
        # dimension. Output lookups should match input sharding spec in this case.
        return OutputSharding(output_spec=DTensorSpec(mesh=inp_spec.mesh, placements=inp_spec.placements))

    if all(placement.is_replicate() for placement in inp_spec.placements):
        weight_dim_map = weight_spec.dim_map
        output_dim_map = inp_spec.dim_map
        output_dim_map.append(weight_dim_map[1])
        return OutputSharding(output_spec=DTensorSpec.from_dim_map(inp_spec.mesh, output_dim_map, []))

    return OutputSharding(
        output_spec=None,
        schema_suggestions=[
            OpSchema(
                op=op_schema.op,
                args_schema=(
                    weight_spec,
                    DTensorSpec(
                        mesh=inp_spec.mesh,
                        placements=tuple([Replicate()] * len(inp_spec.placements)),
                        tensor_meta=inp_spec.tensor_meta,
                    ),
                ),
                kwargs_schema=op_schema.kwargs_schema,
            )
        ],
    )


@register_prop_rule(aten.embedding_renorm_.default)
def embedding_renorm_rules(op_schema: OpSchema) -> OutputSharding:
    raise NotImplementedError("DTensor does not support sharded embedding operation with max_norm yet!")


@register_prop_rule(aten.embedding_dense_backward.default)
def embedding_dense_backward_rules(op_schema: OpSchema) -> OutputSharding:
    grad_output, indices = op_schema.args_schema[:2]
    assert isinstance(grad_output, DTensorSpec)
    assert isinstance(indices, DTensorSpec)

    mesh = grad_output.mesh

    # Situation 1: All replicate
    if is_tensor_all_replicate(grad_output) and is_tensor_all_replicate(indices):
        return OutputSharding(output_spec=DTensorSpec(mesh=mesh, placements=tuple([Replicate()] * mesh.ndim)))

    # Situation 2: Colwise sharding
    if is_tensor_all_replicate_except_sharded_at_dim(
        spec=grad_output, tensor_dim=grad_output.ndim - 1
    ) and is_tensor_all_replicate(indices):
        result_placements = []
        for p in grad_output.placements:
            if p.is_shard():
                tmp_p = copy.deepcopy(p)
                tmp_p.dim = 1
                result_placements.append(tmp_p)
            else:
                result_placements.append(p)
        return OutputSharding(output_spec=DTensorSpec(mesh=mesh, placements=tuple(result_placements)))

    # Situation 3: Sharded on dims other than hidden dim
    sharded_on_no_hidden_flag = False
    sharded_on_no_hidden_mesh_dims = []
    for mesh_idx, idx_p in enumerate(indices.placements):
        grad_out_p = grad_output.placements[mesh_idx]
        if idx_p.is_partial() or grad_out_p.is_partial():
            sharded_on_no_hidden_flag = False
            break
        if idx_p.is_replicate() or grad_out_p.is_replicate():
            continue
        if idx_p != grad_out_p:
            sharded_on_no_hidden_flag = False
            break
        sharded_on_no_hidden_flag = True
        sharded_on_no_hidden_mesh_dims.append(mesh_idx)

    if sharded_on_no_hidden_flag:
        result_placements = [Replicate()] * mesh.ndim
        for mesh_idx in sharded_on_no_hidden_mesh_dims:
            result_placements[mesh_idx] = Partial()
        return OutputSharding(output_spec=DTensorSpec(mesh=mesh, placements=tuple(result_placements)))

    # Situation 4: grad_output is partial, but indices is replicate
    if (
        is_tensor_all_replicate(indices)
        and is_tensor_partial(grad_output)
        and not any(p.is_shard() for p in grad_output.placements)
    ):
        return OutputSharding(output_spec=grad_output)

    raise NotImplementedError(
        "Unsupported embedding dense backward schema:\n" f"grad_output - {grad_output}\n" f"indices - {indices}"
    )
