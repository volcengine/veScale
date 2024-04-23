################################################################################
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

import torch
from einops import rearrange
from torch.nn.parameter import Parameter
from torch import nn

from vescale.dtensor.dtensor import DTensor
from vescale.model.random import get_cuda_rng_tracker
from vescale.model.utils import divide

from .util import mha_split, mqa_split


class ParallelAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        kv_channels,
        num_attention_heads,
        world_size,
        n_shared_qhead,
        params_dtype,
        init_method=torch.nn.init.xavier_normal_,
        attention_dropout=0.0,
    ):
        super().__init__()
        self.n_shared_qhead = n_shared_qhead
        self.attention_dropout = attention_dropout

        projection_size = kv_channels * num_attention_heads
        assert projection_size % self.n_shared_qhead == 0
        self.kv_dim = projection_size // self.n_shared_qhead

        # Per attention head and per partition values.
        self.hidden_size_per_attention_head = divide(projection_size, num_attention_heads)
        self.projection_size_per_partion = divide(projection_size, world_size)
        self.kv_dim_per_partion = divide(self.kv_dim, world_size)
        self.num_attention_heads_per_partition = divide(num_attention_heads, world_size)

        # Strided linear layer.
        self.query_key_value = nn.Linear(hidden_size, projection_size + self.kv_dim * 2, dtype=params_dtype, bias=True)
        init_method(self.query_key_value.weight)
        with torch.no_grad():
            self.query_key_value.bias.zero_()

        # Output.
        self.dense = nn.Linear(projection_size, hidden_size, dtype=params_dtype, bias=False)
        init_method(self.dense.weight)
        self.dense.register_parameter("skipped_bias", Parameter(torch.zeros(hidden_size, dtype=params_dtype)))

    def forward(self, hidden_states, cu_seqlens, **kwargs):
        # getting qkv?
        mixed_x_layer = self.query_key_value(hidden_states)
        local_mixed_x_layer = mixed_x_layer.to_local()

        if self.n_shared_qhead == 1:
            (query_layer, key_layer, value_layer) = mha_split(
                local_mixed_x_layer, self.num_attention_heads_per_partition, self.hidden_size_per_attention_head
            )
        else:
            (query_layer, key_layer, value_layer) = mqa_split(
                local_mixed_x_layer,
                self.hidden_size_per_attention_head,
                self.projection_size_per_partion,
                self.kv_dim_per_partion,
            )

        q, k, v = (rearrange(x, "s b ... -> b s ...").contiguous() for x in (query_layer, key_layer, value_layer))

        with get_cuda_rng_tracker().fork():
            local_context_layer = nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=self.attention_dropout, is_causal=True
            )

        local_context_layer = rearrange(local_context_layer, "b s h d -> s b (h d)").contiguous()

        context_layer = DTensor.from_local(local_context_layer, mixed_x_layer.device_mesh, mixed_x_layer.placements)

        output = self.dense(context_layer)

        return output, self.dense.skipped_bias
