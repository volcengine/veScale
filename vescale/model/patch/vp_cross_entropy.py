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
# Some code are adapted from megatron/core/tensor_parallel/layers.py of Megatron-LM.
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
################################################################################

from functools import partial
from types import MethodType
from typing import Any, Sequence
import warnings

import torch
import torch.nn.functional as F

from vescale.dtensor.api import from_local
from vescale.dtensor.device_mesh import DeviceMesh
from vescale.dtensor.dtensor import DTensor
from vescale.dtensor.placement_types import Partial, Replicate


def _get_vocab_range(per_partition_vocab_size: int, rank, world_size: int) -> Sequence[int]:
    index_f = rank * per_partition_vocab_size
    index_l = index_f + per_partition_vocab_size
    return index_f, index_l


class _VocabParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, d_vocab_parallel_logits: DTensor, d_target: DTensor, device_mesh: DeviceMesh, label_smoothing=0.0):
        # Maximum value along vocab dimension across all GPUs.
        logits_max = torch.max(d_vocab_parallel_logits, dim=-1)[0].redistribute(placements=[Replicate()])

        # Subtract the maximum value.
        d_vocab_parallel_logits = d_vocab_parallel_logits - logits_max.unsqueeze(dim=-1)

        l_vocab_parallel_logits = d_vocab_parallel_logits.to_local()
        l_target = d_target.to_local().reshape(-1, 1)

        # Get the partition's vocab indecies
        partition_vocab_size = l_vocab_parallel_logits.size()[-1]
        rank = device_mesh.get_rank()
        world_size = device_mesh.ndevice
        vocab_start_index, vocab_end_index = _get_vocab_range(partition_vocab_size, rank, world_size)

        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (l_target < vocab_start_index) | (l_target >= vocab_end_index)
        masked_target = l_target.clone() - vocab_start_index
        masked_target[target_mask] = 0

        # Get predicted-logits = logits[target].
        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
        logits_2d = l_vocab_parallel_logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(start=0, end=logits_2d.size()[0], device=logits_2d.device)
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()
        predicted_logits = predicted_logits_1d.view_as(l_target)
        predicted_logits[target_mask] = 0.0

        d_predicted_logits = from_local(predicted_logits, device_mesh, [Partial()])
        # All reduce is needed to get the chunks from other GPUs.
        d_predicted_logits = d_predicted_logits.redistribute(device_mesh, [Replicate()])

        # Sum of exponential of logits along vocab dimension across all GPUs.
        exp_logits = d_vocab_parallel_logits
        torch.exp(d_vocab_parallel_logits, out=exp_logits)
        sum_exp_logits = exp_logits.sum(dim=-1).redistribute(placements=[Replicate()])

        # Loss = log(sum(exp(logits))) - predicted-logit.
        d_loss = torch.log(sum_exp_logits) - d_predicted_logits.squeeze(dim=-1)

        # Normalize and optionally smooth logits
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))

        vocab_size = exp_logits.size(-1)
        if label_smoothing > 0:
            r"""
            We'd like to assign 1 / (K - 1) probability mass to every index that is not the ground truth.
            = (1 - alpha) * y_gt + alpha * mean(y_{i for i != gt})
            = (1 - alpha) * y_gt + (alpha / (K - 1)) * \sum_{i != gt} y_i
            = ((K - 1) * (1 - alpha) / (K - 1)) * y_gt + (alpha / (K - 1)) * \sum_{i != gt} y_i
            = (K * (1 - alpha) - 1) / (K - 1)) * y_gt  + (alpha / (K - 1)) * \sum_{i} y_i
            = (1 - (alpha * K) / (K - 1)) * y_gt + ( (alpha * K) / (K - 1) ) * \sum_{i} y_i / K
            From: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/losses/smoothed_cross_entropy.py
            """
            assert 1.0 > label_smoothing > 0.0
            smoothing = label_smoothing * vocab_size / (vocab_size - 1)

            # Exp logits at this point are normalized probabilities. So we can just take the log to get log-probs.
            log_probs = torch.log(exp_logits)
            mean_log_probs = log_probs.mean(dim=-1).redistribute(device_mesh, [Replicate()])
            d_loss = (1.0 - smoothing) * d_loss - smoothing * mean_log_probs

        ctx.label_smoothing, ctx.vocab_size = label_smoothing, vocab_size

        # Store softmax, target-mask and masked-target for backward pass.
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)

        return d_loss

    @staticmethod
    def backward(ctx: Any, grad_output: DTensor) -> Any:
        # Retreive tensors from the forward path.
        softmax, target_mask, masked_target_1d = ctx.saved_tensors
        label_smoothing, vocab_size = ctx.label_smoothing, ctx.vocab_size

        # All the inputs have softmax as thier gradient.
        grad_input = softmax
        # For simplicity, work with the 2D gradient.
        partition_vocab_size = softmax.size()[-1]
        grad_2d = grad_input.view(-1, partition_vocab_size).to_local()

        # Add the gradient from matching classes.
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=grad_2d.device)

        softmax_update = 1.0 - target_mask.view(-1).float()

        if label_smoothing > 0:
            smoothing = label_smoothing * vocab_size / (vocab_size - 1)
            grad_2d[arange_1d, masked_target_1d] -= (1.0 - smoothing) * softmax_update
            average_grad = 1 / vocab_size
            grad_2d[arange_1d, :] -= smoothing * average_grad
        else:
            grad_2d[arange_1d, masked_target_1d] -= softmax_update

        # Finally elementwise multiplication with the output gradients.
        grad_input.mul_(grad_output.unsqueeze(dim=-1))

        return grad_input, None, None, None


class VocabParallelCrossEntropy:
    @staticmethod
    def forward(self: torch.nn.CrossEntropyLoss, input, label, device_mesh: DeviceMesh) -> DTensor:
        def original_f():
            return F.cross_entropy(
                input,
                label,
                weight=self.weight,
                ignore_index=self.ignore_index,
                reduction=self.reduction,
                label_smoothing=self.label_smoothing,
            )

        if input.ndim != 2:
            return original_f()
        if not isinstance(input, DTensor):
            return original_f()
        if not input.placements[0].is_shard(1):
            return original_f()
        if device_mesh is None:
            device_mesh = input.device_mesh

        loss = _VocabParallelCrossEntropy.apply(input, label, device_mesh, self.label_smoothing)
        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError(f"{self.reduction} is not a valid value for reduction")

    @staticmethod
    def patch(root: torch.nn.Module) -> None:
        r"""
        Post-patch support for a vocab parallel cross entropy loss

        For a vocab parallel cross entropy, the activation dtenosr is shard on the 2nd dimension (a.k.a. the class dimension).
        This causes each rank can only calculate partial result of exp(x_(n,yn)) and sum(exp(x_n,c) for c in C).
        One solution is masking out index that are not part of the local weight, but this requires
        extra code. Therefore, this patch automaticlly add the 'extra code' to a CrossEntropy forward
        only when the input is shard(1).
        """

        for submod_path, submod in root.named_modules():
            if not isinstance(submod, torch.nn.CrossEntropyLoss):
                continue
            if submod.weight is not None:
                warnings.warn(
                    f"Unable to patch `{submod_path}` because its weight is not None. "
                    "Continue with pytorch original CrossEntropyLoss with out any vocab parallel patch, "
                    "which maybe intended and favorable behavoior.",
                    UserWarning,
                )
                continue

            if submod.label_smoothing != 0.0:
                warnings.warn(
                    f"Unable to patch `{submod_path}` because its label_smoothing is not 0.0. "
                    "Continue with pytorch original CrossEntropyLoss with out any vocab parallel patch, "
                    "which maybe intended and favorable behavoior.",
                    UserWarning,
                )
                continue

            if submod.ignore_index != -100:
                warnings.warn(
                    f"Unable to patch `{submod_path}` because its ignore_index is not -100. "
                    "Continue with pytorch original CrossEntropyLoss with out any vocab parallel patch, "
                    "which maybe intended and favorable behavoior.",
                    UserWarning,
                )
                continue

            # skip no input_placements
            input_pis = root.get_fwd_plan(submod_path + ".input")
            # None placement is ok because default plan is replicate
            if input_pis is not None:
                assert isinstance(input_pis, Sequence) and len(input_pis) != 0
                in_pi = input_pis[0]
                if in_pi.placements:
                    assert (
                        isinstance(in_pi.placements, Sequence) and len(in_pi.placements) == 1
                    ), "Only 1D sharding is considered now!"
                    if not in_pi.placements[-1].is_shard(1):
                        warnings.warn(
                            f"Unable to patch `{submod_path}` because its input is not shard(1). "
                            "Continue with pytorch original CrossEntropyLoss with out any vocab parallel patch, "
                            "which maybe intended and favorable behavoior.",
                            UserWarning,
                        )
                        continue

            # replace nn.Linear's forward with customized forward.
            submod.forward = MethodType(
                partial(VocabParallelCrossEntropy.forward, device_mesh=None),
                submod,
            )
