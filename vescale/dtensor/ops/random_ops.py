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

from vescale.dtensor import DeviceMesh
from vescale.dtensor.op_schema import OpSchema, OpStrategy, PlacementStrategy, StrategyType
from vescale.dtensor.ops.utils import register_op_strategy, is_tensor_partial
from vescale.dtensor.placement_types import DTensorSpec, Partial, Replicate

aten = torch.ops.aten


@register_op_strategy(
    [
        aten.normal_.default,
        aten.uniform_.default,
        aten.bernoulli_.float,
    ]
)
def random_op_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> StrategyType:
    self_strategy = op_schema.args_schema[0]
    assert isinstance(self_strategy, OpStrategy)

    random_strategy = OpStrategy([])
    for arg_strategy in self_strategy.strategies:
        arg_spec = arg_strategy.output_spec
        if is_tensor_partial(arg_spec):
            # if the arg_spec have partial, accept partial
            # in the input_specs but output replicate for
            # those corresponding mesh dims

            output_spec = DTensorSpec(
                mesh=arg_spec.mesh,
                placements=tuple(Replicate() if isinstance(p, Partial) else p for p in arg_spec.placements),
            )
            random_strategy.strategies.append(
                PlacementStrategy(
                    output_spec=output_spec,
                    input_specs=(arg_spec,),
                )
            )
        else:
            random_strategy.strategies.append(PlacementStrategy(output_spec=arg_spec))

    return random_strategy


@register_op_strategy(aten.native_dropout.default)
def random_op_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> StrategyType:
    self_strategy = op_schema.args_schema[0]
    assert isinstance(self_strategy, OpStrategy)

    random_strategy = OpStrategy([])
    for arg_strategy in self_strategy.strategies:
        arg_spec = arg_strategy.output_spec
        random_strategy.strategies.append(PlacementStrategy(output_spec=arg_spec))

    return random_strategy
