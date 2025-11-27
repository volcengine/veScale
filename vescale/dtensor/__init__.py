################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

"""
Compared to Pytorch 2.7 dtensor code, _random.py is deleted.
"""

import torch
import vescale.dtensor._ops  # force import all built-in dtensor ops
from ._api import DTensor, distribute_tensor, ones, empty, full, rand, randn, zeros
from ._dtensor_spec import DTensorSpec, TensorMeta
from .placement_types import (
    Placement,
    Shard,
    Replicate,
    Partial,
    RaggedShard,
    _StridedShard,
    _StridedRaggedShard,
    _Partial,
    is_ragged_shard,
)

__all__ = [
    "DTensor",
    "distribute_tensor",
    "ones",
    "empty",
    "full",
    "rand",
    "randn",
    "zeros",
    "DTensorSpec",
    "TensorMeta",
    "Placement",
    "Shard",
    "Replicate",
    "Partial",
    "RaggedShard",
    "_StridedRaggedShard",
    "_StridedShard",
    "_Partial",
    "is_ragged_shard",
]


torch.serialization.add_safe_globals([DTensor])
