################################################################################
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

import torch
from vescale.model.utils import divide


def split_tensor_along_last_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
):
    """Split a tensor along its last dimension.

    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                in memory.

    Returns:
        A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


def mha_split(mixed_x_layer, num_attention_heads_per_partition, hidden_size_per_attention_head):
    new_tensor_shape = mixed_x_layer.size()[:-1] + (
        num_attention_heads_per_partition,
        3 * hidden_size_per_attention_head,
    )
    mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)
    (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)
    return query_layer, key_layer, value_layer


def mqa_split(mixed_x_layer, hidden_size_per_attention_head, projection_size_per_partion, kv_dim_per_partion):
    new_tensor_shape = mixed_x_layer.size()[:-1] + (-1, hidden_size_per_attention_head)

    key_start = projection_size_per_partion
    value_start = projection_size_per_partion + kv_dim_per_partion

    key_layer = mixed_x_layer[:, :, key_start:value_start].view(*new_tensor_shape)
    value_layer = mixed_x_layer[:, :, value_start:].view(*new_tensor_shape)

    query_layer = mixed_x_layer[:, :, :key_start].view(*new_tensor_shape)
    key_layer = key_layer.repeat(1, 1, projection_size_per_partion // kv_dim_per_partion, 1)
    value_layer = value_layer.repeat(1, 1, projection_size_per_partion // kv_dim_per_partion, 1)

    return query_layer, key_layer, value_layer
