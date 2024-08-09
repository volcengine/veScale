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
from vescale.emulator.reduce_kernel import ReduceOp


def flatten_tensors(tensor_list):
    """
    Flatten a list of tensors into a single tensor.
    """
    flattened_list = []
    original_shapes = []

    for tensor in tensor_list:
        original_shapes.append(tensor.size())
        flattened_tensor = tensor.view(-1)
        flattened_list.append(flattened_tensor)

    return flattened_list, original_shapes


def restore_tensors(flattened_list, original_shapes):
    """
    Restore a list of flattened tensors to their original shapes.
    """
    restored_list = []

    for flattened_tensor, shape in zip(flattened_list, original_shapes):
        restored_tensor = flattened_tensor.view(shape)
        restored_list.append(restored_tensor)

    return restored_list


def torch_reduce_op_to_emulator(torch_reduce_op):
    """
    Convert torch reduce op to emulator reduce op.
    """
    if torch_reduce_op == torch.distributed.ReduceOp.SUM:
        return ReduceOp.SUM
    elif torch_reduce_op == torch.distributed.ReduceOp.PRODUCT:
        return ReduceOp.PRODUCT
    elif torch_reduce_op == torch.distributed.ReduceOp.MIN:
        return ReduceOp.MIN
    elif torch_reduce_op == torch.distributed.ReduceOp.MAX:
        return ReduceOp.MAX
    else:
        raise ValueError(f"Unsupported reduce op: {torch_reduce_op}")


def emulator_reduce_op_to_torch(reduce_op):
    """
    Convert emulator reduce op to torch reduce op.
    """
    if reduce_op == ReduceOp.SUM:
        return torch.distributed.ReduceOp.SUM
    elif reduce_op == ReduceOp.PRODUCT:
        return torch.distributed.ReduceOp.PRODUCT
    elif reduce_op == ReduceOp.MAX:
        return torch.distributed.ReduceOp.MAX
    elif reduce_op == ReduceOp.MIN:
        return torch.distributed.ReduceOp.MIN
    else:
        raise ValueError(f"Unsupported reduce op: {reduce_op}")
