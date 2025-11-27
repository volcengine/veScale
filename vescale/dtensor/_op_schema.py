################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################
from torch.distributed.tensor._op_schema import (
    OpInfo,
    _is_inplace_op,
    _is_out_variant_op,
    OpSchema,
    OpStrategy,
    OutputSharding,
    OutputSpecType,
    PlacementStrategy,
    TupleStrategy,
    PlacementList,
    RuntimeSchemaInfo,
    StrategyType,
)

__all__ = [
    "OpInfo",
    "_is_inplace_op",
    "_is_out_variant_op",
    "OpSchema",
    "OpStrategy",
    "OutputSharding",
    "OutputSpecType",
    "PlacementStrategy",
    "TupleStrategy",
    "PlacementList",
    "RuntimeSchemaInfo",
    "StrategyType",
]
