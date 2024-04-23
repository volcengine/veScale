################################################################################
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

import torch
from torch import nn
from contextlib import nullcontext
from vescale.dtensor.dtensor import DTensor
from vescale.initialize.deferred_init import deferred_init
from vescale.model.base_gpt.transformer_layer import ParallelTransformerLayer
from vescale.model.random import get_cuda_rng_tracker


class TransformerBlock(nn.Module):
    """Transformer class."""

    def __init__(
        self,
        num_layer,
        args,
        drop_path_rate=0.0,
        pre_process=True,
        deferred_init=False,
    ):
        super().__init__()

        self.config = args
        self.drop_path_rate = drop_path_rate
        self.pre_process = pre_process
        self.num_layer = num_layer
        self.deferred_init = deferred_init

        # required for pipeline parallel schedules
        self.input_tensor = None
        self._build_layers()

    def _build_layers(self):
        # Transformer layers.
        # @jcasper can we improve how we deal with layer_number?
        # currently it's only used in CoreAttention?
        # if self.apply_query_key_layer_scaling:
        #     coeff = self.layer_number
        #     self.norm_factor *= coeff
        def build_layer(layer_number):
            if self.deferred_init:
                layer_config = {
                    "init_method": self.config.init_method,
                    "output_layer_init_method": self.config.output_layer_init_method,
                    "layer_number": layer_number,
                    "args": self.config,
                    "drop_path_rate": self.drop_path_rate,
                }
                layer = deferred_init(ParallelTransformerLayer, **layer_config)
            else:
                layer = ParallelTransformerLayer(
                    self.config.init_method,
                    self.config.output_layer_init_method,
                    layer_number,
                    self.config,
                    self.drop_path_rate,
                )

            return layer

        # offset is implicit in TransformerLayer
        self.transformer_layers = torch.nn.ModuleList([build_layer(i + 1) for i in range(self.num_layer)])
        self.layers = torch.nn.Sequential()
        for i in range(len(self.transformer_layers)):
            self.layers.append(self.transformer_layers[i])

    def _get_layer(self, layer_number):
        return self.transformer_layers[layer_number]

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    def forward(
        self, hidden_states, attention_mask, encoder_output=None, enc_dec_attn_mask=None, inference_params=None
    ):
        # hidden_states (float): [s, b, h]
        # attention_mask (bool): [1, 1, s, s]

        if not self.pre_process:
            # See set_input_tensor()
            hidden_states = self.input_tensor

        # Viewless tensor.
        # - We only need to create a viewless tensor in the case of micro batch
        #   size (mbs) == 1, since in this case, 'hidden_states.transpose()'
        #   above creates a view tensor, and '.contiguous()' is a pass-through.
        #   For mbs >= 2, '.contiguous()' creates a new tensor, eliminating
        #   the need to make it viewless.
        #
        #   However, we don't explicitly check mbs == 1 here because
        #   make_viewless_tensor() has negligible overhead when its input
        #   is already viewless.
        #
        # - For the 'else' case above, calling make_viewless_tensor() here is
        #   likely redundant, since p2p_communication.py (likely originator)
        #   already creates viewless tensors. That said, make_viewless_tensor()
        #   is called here to be future-proof and corner-case-proof.
        # hidden_states = make_viewless_tensor(
        #     inp=hidden_states,
        #     requires_grad=True,
        #     keep_graph=True,
        # )

        rng_context = nullcontext()
        if isinstance(hidden_states, DTensor):
            placements = hidden_states.placements
            # check sbh, for s
            is_sp = any(placement.is_shard(dim=0) for placement in placements)
            if is_sp:
                rng_context = get_cuda_rng_tracker().fork()

        with rng_context:
            for layer in self.transformer_layers:
                hidden_states = layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    encoder_output=encoder_output,
                    enc_dec_attn_mask=enc_dec_attn_mask,
                    inference_params=inference_params,
                )

        return hidden_states
