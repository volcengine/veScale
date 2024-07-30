################################################################################
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
################################################################################
# Modification Copyright 2024 ByteDance Ltd. and/or its affiliates.
################################################################################

from contextlib import nullcontext

import torch
from torch import nn

from vescale.model.attention.dmodule_parallel_attention import ParallelAttention
from vescale.model.base_gpt.fuse_layer_norm import MixedFusedLayerNorm as LayerNorm
from vescale.model.base_gpt.jit_func import (
    bias_dropout_add_fused_inference,
    bias_dropout_add_fused_train,
    get_bias_dropout_add,
)
from vescale.model.base_gpt.mlp import ParallelMLP, SwitchMLP


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample
    (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_state):
        if self.drop_prob == 0.0 or not self.training:
            return hidden_state
        keep_prob = 1 - self.drop_prob
        # work with diff dim tensors, not just 2D ConvNets
        # hidden_state: [s, b, h]
        random_tensor = keep_prob + torch.rand_like(hidden_state)
        random_tensor.floor_()  # binarize
        output = hidden_state.div(keep_prob) * random_tensor
        return output


class ParallelTransformerLayer(nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(
        self,
        init_method,
        output_layer_init_method,
        layer_number,
        args,
        drop_path_rate=0.0,
    ):
        super().__init__()
        self.layer_number = layer_number

        self.apply_residual_connection_post_layernorm = args.apply_residual_connection_post_layernorm

        self.bf16 = args.bf16

        # Layernorm on the input data.
        self.input_layernorm = LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon,
            no_persist_layer_norm=not args.persist_layer_norm,
            param_dtype=args.param_dtype,
        )

        # Self attention.
        self.self_attention = ParallelAttention(
            args.hidden_size,
            args.kv_channels,
            args.num_attention_heads,
            args.world_size,
            1,  # n_shared_qhead
            args.param_dtype,
        )
        self.hidden_dropout = args.hidden_dropout
        self.bias_dropout_fusion = args.bias_dropout_fusion
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else None

        # Layernorm on the attention output
        self.post_attention_layernorm = LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon,
            no_persist_layer_norm=not args.persist_layer_norm,
            param_dtype=args.param_dtype,
        )

        # MLP
        if args.num_experts is not None:
            self.mlp = SwitchMLP(init_method, output_layer_init_method, args)
        else:
            self.mlp = ParallelMLP(args.hidden_size, param_dtype=args.param_dtype)

        # Set bias+dropout+add fusion grad_enable execution handler.
        TORCH_MAJOR = int(torch.__version__.split(".")[0])
        TORCH_MINOR = int(torch.__version__.split(".")[1])
        use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        self.bias_dropout_add_exec_handler = nullcontext if use_nvfuser else torch.enable_grad

    def forward(
        self, hidden_states, attention_mask, encoder_output=None, enc_dec_attn_mask=None, inference_params=None
    ):
        # hidden_states: [s, b, h]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        attention_output, attention_bias = self.self_attention(
            layernorm_output, attention_mask, inference_params=inference_params
        )

        # assert not torch.isnan(attention_output.to_local()
        #                        ).any(), attention_output
        # assert not torch.isnan(attention_bias.to_local()).any(), attention_bias

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        if self.drop_path is None:
            # jit scripting for a nn.module (with dropout) is not
            # trigerring the fusion kernel. For now, we use two
            # different nn.functional routines to account for varying
            # dropout semantics during training and inference phases.
            if self.bias_dropout_fusion:
                if self.training:
                    bias_dropout_add_func = bias_dropout_add_fused_train
                else:
                    bias_dropout_add_func = bias_dropout_add_fused_inference
            else:
                bias_dropout_add_func = get_bias_dropout_add(self.training)

            with self.bias_dropout_add_exec_handler():
                layernorm_input = bias_dropout_add_func(
                    attention_output, attention_bias.expand_as(residual), residual, self.hidden_dropout
                )
        else:
            out = attention_output + attention_bias
            out = torch.nn.functional.dropout(out, p=self.hidden_dropout, training=self.training)
            layernorm_input = residual + self.drop_path(out)

        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        # assert not torch.isnan(layernorm_output).any()

        # MLP.
        mlp_output, mlp_bias = self.mlp(layernorm_output)
        # assert not torch.isnan(mlp_output.to_local()).any(), mlp_output
        # assert not torch.isnan(mlp_bias.to_local()).any(), mlp_bias

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        if self.drop_path is None:
            with self.bias_dropout_add_exec_handler():
                output = bias_dropout_add_func(mlp_output, mlp_bias.expand_as(residual), residual, self.hidden_dropout)

            # Jit compiled function creates 'view' tensor. This tensor
            # potentially gets saved in the MPU checkpoint function context,
            # which rejects view tensors. While making a viewless tensor here
            # won't result in memory savings (like the data loader, or
            # p2p_communication), it serves to document the origin of this
            # 'view' tensor.
            # output = dtensor.utils.make_viewless_tensor(inp=output, requires_grad=output.requires_grad, keep_graph=True)
            #
        else:
            out = mlp_output + mlp_bias
            out = torch.nn.functional.dropout(out, p=self.hidden_dropout, training=self.training)
            output = residual + self.drop_path(out)

        return output

    def forward_util(self, input_tensor, data):
        ret = {
            "hidden_states": input_tensor if input_tensor is not None else data["hidden_states"],
            "attention_mask": data["attention_mask"],
        }
        return [ret["hidden_states"], ret["attention_mask"]]

    def output_utils(self, p2p_tensor):
        p2p_tensor = torch.permute(p2p_tensor, (0, 2, 1))
        return p2p_tensor
