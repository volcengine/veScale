################################################################################
#
# Copyright 2023 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################

import warnings
from torch import nn
from inspect import signature


def validate_single_input(module: nn.Module):
    sig = signature(module.forward)
    if len(sig.parameters) <= 1:
        return

    if len(sig.parameters) == 2:
        param_str = [str(param) for param in sig.parameters.values()]
        if param_str[0].startswith("*") and param_str[1].startswith("**"):
            return

    warnings.warn(
        "Current policies might not support multiple inputs; " "If failed, try move all-except-first args to kwargs."
    )
