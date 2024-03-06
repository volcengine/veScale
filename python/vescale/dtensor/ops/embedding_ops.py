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

import torch

from vescale.dtensor.op_schema import OpSchema, OutputSharding
from vescale.dtensor.ops.utils import register_prop_rule
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
    if grad_output.placements == indices.placements:
        # The embedding table is replicated, and input/oupput activations are
        # sharded. In this case, gradients for the embedding table should be
        # Partial.
        return OutputSharding(output_spec=DTensorSpec(mesh=indices.mesh, placements=(Partial(),)))
    elif grad_output.placements == [Partial()] and indices.placements == [Replicate()]:
        # The embedding table is replicated and the indices is also replicated
        # (local is a more precise term). This is postional embedding. In this
        # case, gradients for the embmedding table should be Partial.
        return OutputSharding(output_spec=DTensorSpec(mesh=indices.mesh, placements=(Partial(),)))
    elif all(placement.is_replicate() for placement in indices.placements):
        # BWD for colwise sharding case
        return OutputSharding(output_spec=DTensorSpec(mesh=indices.mesh, placements=(Shard(1),)))
    else:
        raise NotImplementedError(
            "Unsupported embedding dense backward schema:\n" f"grad_output - {grad_output}\n" f"indices - {indices}"
        )
