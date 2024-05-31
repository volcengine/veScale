################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################
# meta_type.py saves all constants and data types commonly used in vescale.checkpoint

from enum import Enum
from typing import Dict, Any, TypeVar
from typing_extensions import Protocol, runtime_checkable


STATE_DICT_TYPE = Dict[str, Any]

MODEL_STR = "model"
OPTIMIZER_STR = "optimizer"
STATE_DICT_STR = "state_dict"


class SupportedStrategy(Enum):
    Megatron = 0
    FSDP = 1
    VeScale = 2


@runtime_checkable
class Stateful(Protocol):
    def state_dict(self) -> Dict[str, Any]: ...

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None: ...


T = TypeVar("T", bound=Stateful)
CheckpointState = Dict[str, T]
