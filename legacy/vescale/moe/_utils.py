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
from typing import List

_TAG_EXPERTS_PARALLIZED = "_EXPERTS_PARALLIZED"
_MOE_TP = "_MOE_TP"
_MOE_DP = "_MOE_DP"


def global_all_to_all_single(
    tensor: torch.Tensor,
    input_split_sizes: List[int],
    output_split_sizes: List[int],
    async_op: bool = False,
):
    return _AllToAllSingle.apply(tensor, input_split_sizes, output_split_sizes, async_op)


class _AllToAllSingle(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input_tensor: torch.Tensor,
        input_split_sizes: List[int],
        output_split_sizes: List[int],
        async_op: bool = False,
    ):
        ctx.input_split_sizes = input_split_sizes
        ctx.output_split_sizes = output_split_sizes
        ctx.async_op = async_op
        output_tensor = input_tensor.new_empty(
            size=[sum(output_split_sizes)] + list(input_tensor.size()[1:]),
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )
        torch.distributed.all_to_all_single(
            output_tensor,
            input_tensor,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            async_op=async_op,
        )
        return output_tensor

    @staticmethod
    def backward(ctx, grad):
        return (
            _AllToAllSingle.apply(grad, ctx.output_split_sizes, ctx.input_split_sizes, ctx.async_op),
            None,
            None,
            None,
        )
