################################################################################
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

import torch
from torch import nn

from vescale.model.utils import bias_gelu_impl, openai_gelu


class SwitchMLP(nn.Module):
    """
    Routes input to one of N MLP "experts"
    """

    def __init__(self, hidden_size, num_experts):
        super().__init__()
        self.router = nn.Linear(hidden_size, num_experts)
        self.experts = torch.nn.ModuleList()
        for _ in range(num_experts):
            self.experts.append(ParallelMLP(hidden_size))

    def forward(self, hidden_states):
        # hidden_states: [s, b, h]
        s = hidden_states.size(0)
        b = hidden_states.size(1)
        h = hidden_states.size(2)
        route = self.router(hidden_states)
        route = torch.nn.functional.softmax(route, dim=2)
        max_prob, max_ind = torch.max(route, dim=2)
        max_prob = torch.unsqueeze(max_prob, 2)  # [s b 1]

        # TODO (rprenger) TODO this could be made easier to read
        # Converting [s, b, h] to [s*b, h].
        # Each vector could be routed differently
        # [s*b h]
        hidden_states = hidden_states.view(-1, hidden_states.size(2))
        max_prob = max_prob.view(-1, max_prob.size(2))  # [s*b 1]
        max_ind = max_ind.view(-1)  # [s*b]

        output_total = torch.empty_like(hidden_states)
        output_bias_total = torch.empty_like(hidden_states)
        # TODO (rprenger) This does each expert in serial, but it could be parallelized

        for expert_num, expert in enumerate(self.experts):
            local_indices = (max_ind == expert_num).nonzero()
            hidden = hidden_states[local_indices, :]
            output, output_bias = expert(hidden)
            output_bias = output_bias.expand_as(output)
            output_total[local_indices, :] = output
            output_bias_total[local_indices, :] = output_bias

        output_total = output_total * max_prob
        output_bias_total = output_bias_total * max_prob
        output_total = output_total.view(s, b, h)
        output_bias_total = output_bias_total.view(s, b, h)

        return output_total, output_bias_total


class ParallelMLP(nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, h, param_dtype=torch.float32, bias_gelu_fusion=None):
        super().__init__()

        # Project to 4h.
        self.dense_h_to_4h = nn.Linear(h, h * 4, bias=False, dtype=param_dtype)
        # torch.nn.init.normal_(self.dense_h_to_4h.weight, mean=0.0, std=0.02)
        torch.nn.init.xavier_normal_(self.dense_h_to_4h.weight)
        self.dense_h_to_4h_bias = nn.Parameter(torch.zeros(4 * h, dtype=param_dtype))

        self.bias_gelu_fusion = bias_gelu_fusion
        self.activation_func = openai_gelu

        # Project back to h.
        self.dense_4h_to_h = nn.Linear(4 * h, h, bias=False, dtype=param_dtype)
        torch.nn.init.xavier_uniform_(self.dense_4h_to_h.weight)
        # torch.nn.init.normal_(self.dense_4h_to_h.weight, mean=0.0, std=0.02)
        self.dense_4h_to_h_bias = nn.Parameter(torch.zeros(h, dtype=param_dtype))

    def forward(self, hidden_states):
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        bias_parallel = self.dense_h_to_4h_bias

        if self.bias_gelu_fusion:
            intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)
        else:
            intermediate_parallel = self.activation_func(intermediate_parallel + bias_parallel)

        # [s, b, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        output_bias = self.dense_4h_to_h_bias
        return output, output_bias
