################################################################################
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
################################################################################
# Modification Copyright 2024 ByteDance Ltd. and/or its affiliates.
################################################################################

import enum


class ModelType(enum.Enum):
    encoder_or_decoder = 1
    encoder_and_decoder = 2


class LayerType(enum.Enum):
    encoder = 1
    decoder = 2


class AttnType(enum.Enum):
    self_attn = 1
    cross_attn = 2


class AttnMaskType(enum.Enum):
    padding = 1
    causal = 2
