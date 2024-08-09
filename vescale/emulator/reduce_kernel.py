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

import torch
from typing import List, Union, Optional


class ReduceOp:
    """
    ReduceOp is a class that contains all the reduce operations that can be used in the all-reduce operation.
    It contains the following operations:
    - SUM: Sum of all the elements in the tensor.
    - MAX: Maximum of all the elements in the tensor.
    - MIN: Minimum of all the elements in the tensor.
    - PRODUCT: Product of all the elements in the tensor.
    """

    @staticmethod
    def SUM(a: Union[torch.Tensor, List[torch.Tensor]], b: Optional[torch.Tensor] = None):
        if b is not None:
            return a + b
        else:
            return torch.sum(torch.stack(a), dim=0)

    def MAX(a: Union[torch.Tensor, List[torch.Tensor]], b: Optional[torch.Tensor] = None):
        if b is not None:
            return torch.max(torch.stack([a, b]), dim=0)[0]
        else:
            return torch.max(torch.stack(a), dim=0)[0]

    def MIN(a: Union[torch.Tensor, List[torch.Tensor]], b: Optional[torch.Tensor] = None):
        if b is not None:
            return torch.min(torch.stack([a, b]), dim=0)[0]
        else:
            return torch.min(torch.stack(a), dim=0)[0]

    def PRODUCT(a: Union[torch.Tensor, List[torch.Tensor]], b: Optional[torch.Tensor] = None):
        if b is not None:
            return a * b
        else:
            return torch.prod(torch.stack(a), dim=0)
