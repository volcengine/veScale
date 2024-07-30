################################################################################
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
################################################################################
# Modification Copyright 2024 ByteDance Ltd. and/or its affiliates.
################################################################################

"""This code is copied fron NVIDIA apex:
   https://github.com/NVIDIA/apex
with some changes."""

import importlib
import numbers

import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter


fused_layer_norm_cuda = importlib.import_module("fused_layer_norm_cuda")

try:
    from apex.contrib.layer_norm.layer_norm import FastLayerNormFN

    HAVE_PERSIST_LAYER_NORM = True
except ImportError:
    HAVE_PERSIST_LAYER_NORM = False

try:
    from apex.normalization.fused_layer_norm import FusedLayerNormAffineFunction

    HAVE_FUSED_LAYER_NORM = True
except ImportError:
    HAVE_FUSED_LAYER_NORM = False


class MixedFusedLayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape,
        eps=1e-5,
        no_persist_layer_norm=True,
        param_dtype=torch.float32,
        sequence_parallel=False,
        apply_layernorm_1p=False,
    ):
        super().__init__()

        self.apply_layernorm_1p = apply_layernorm_1p

        # List of hiddens sizes supported in the persistent layer norm kernel
        # If the hidden size is not supported, fall back to the non-persistent
        # kernel.
        persist_ln_hidden_sizes = [
            1024,
            1536,
            2048,
            2304,
            3072,
            3840,
            4096,
            5120,
            6144,
            8192,
            10240,
            12288,
            12800,
            15360,
            16384,
            18432,
            20480,
            24576,
            25600,
            30720,
            32768,
            40960,
            49152,
            65536,
        ]
        if normalized_shape not in persist_ln_hidden_sizes or not HAVE_PERSIST_LAYER_NORM:
            no_persist_layer_norm = True

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.weight = Parameter(torch.Tensor(*normalized_shape).to(param_dtype))
        self.bias = Parameter(torch.Tensor(*normalized_shape).to(param_dtype))
        self.reset_parameters()
        self.no_persist_layer_norm = no_persist_layer_norm
        self.sequence_parallel = sequence_parallel

    def reset_parameters(self):
        if self.apply_layernorm_1p:
            init.zeros_(self.weight)
            init.zeros_(self.bias)
        else:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input):
        weight = self.weight + 1 if self.apply_layernorm_1p else self.weight

        if self.no_persist_layer_norm:
            assert (
                FusedLayerNormAffineFunction is not None
            ), "FusedLayerNormAffineFunction is not available, please install apex from https://github.com/NVIDIA/apex"
            out = FusedLayerNormAffineFunction.apply(input, weight, self.bias, self.normalized_shape, self.eps, False)
            return out
        else:
            output = FastLayerNormFN.apply(input, weight, self.bias, self.eps)

            # Apex's fast layer norm function outputs a 'view' tensor (i.e., has
            # a populated '_base' field). This will result in schedule.py's
            # deallocate_output_tensor() throwing an error, so a viewless tensor is
            # created to prevent this.
            # output = make_viewless_tensor(
            #     inp=output, requires_grad=input.requires_grad, keep_graph=True)
            return output
