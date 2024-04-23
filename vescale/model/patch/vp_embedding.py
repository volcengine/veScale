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
from typing import Sequence
import warnings

import torch
import torch.nn.functional as F

from vescale.dtensor.api import from_local
from vescale.dtensor.device_mesh import DeviceMesh
from vescale.dtensor.dtensor import DTensor
from vescale.dtensor.placement_types import Partial, Replicate


class VocabParallelEmbedding:
    @staticmethod
    def forward(self: torch.nn.Embedding, input: DTensor, device_mesh: DeviceMesh) -> DTensor:
        assert input.placements[
            0
        ].is_replicate(), f"current patch only support replicate input but got {input.placements}"
        # todo fix uneven

        if device_mesh.ndevice == 1:
            return self(input)

        tp_world_size = device_mesh.ndevice

        # self.num_embeddings
        assert self.num_embeddings % tp_world_size == 0, (
            f"current patch assume num_embeddings is divisible by tp world size, "
            f"but got num_embeddings = {self.num_embeddings} and tp world size = {tp_world_size}"
        )
        per_partition_vocab_size = self.num_embeddings // tp_world_size
        vocab_start_index = per_partition_vocab_size * device_mesh.get_rank()
        vocab_end_index = vocab_start_index + per_partition_vocab_size

        local_input = input.to_local()
        local_weight = self.weight.to_local()

        input_mask = (local_input < vocab_start_index) | (local_input >= vocab_end_index)

        masked_input = local_input.clone() - vocab_start_index
        masked_input[input_mask] = 0

        local_output = F.embedding(
            masked_input,
            local_weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

        local_output[input_mask, :] = 0.0
        dtensor_output = from_local(local_output, device_mesh, [Partial()])
        return dtensor_output.redistribute(device_mesh, [Replicate()])

    @staticmethod
    def patch(root: torch.nn.Module) -> None:
        r"""
        Post-patch support for a row-wise sharded embedding operation

        For a vocab parallel embedding, the activation dtenosr is expected to be replicate.
        However, the weight is shard along the first dimision.
        This requires the local actication tensor access the weight that are not part of its shard.
        One solution is masking out index that are not part of the local weight, but this requires
        extra code. Therefore, this patch automaticlly add the 'extra code' to a embedding forward
        only when the input is replicate and the weight is shard(0)

        """

        for submod_path, submod in root.named_modules():
            if not isinstance(submod, torch.nn.Embedding):
                continue
            if not isinstance(submod.weight, DTensor):
                continue
            if not submod.weight.placements[0].is_shard(0):
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
                    if not in_pi.placements[-1].is_replicate():
                        warnings.warn(
                            f"`{submod_path}` is a row-wise sharded embedding operation, and current patch only support replicate input. But got {in_pi.placements}.",
                            UserWarning,
                        )
                        continue

            # replace nn.Linear's forward with customized forward.
            submod.forward = MethodType(
                partial(VocabParallelEmbedding.forward, device_mesh=submod.weight.device_mesh),
                submod,
            )
