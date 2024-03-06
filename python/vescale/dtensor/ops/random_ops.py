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

aten = torch.ops.aten


@register_op_strategy([aten.normal_.default, aten.uniform_.default, aten.native_dropout.default])
def random_op_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> StrategyType:
    self_strategy = op_schema.args_schema[0]
    assert isinstance(self_strategy, OpStrategy)

    random_strategy = OpStrategy([])
    for arg_strategy in self_strategy.strategies:
        arg_spec = arg_strategy.output_spec
        if is_tensor_partial(arg_spec):
            # TODO: figure out how inplace random op should behave when it's partial
            raise RuntimeError(f"{op_schema.op} with Partial is not supported yet!")
        random_strategy.strategies.append(PlacementStrategy(output_spec=arg_spec))

    return random_strategy
