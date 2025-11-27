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

import itertools

import torch
from torch.distributed.device_mesh import DeviceMesh

from vescale.dtensor import DTensor, DTensorSpec, TensorMeta
from vescale.dtensor.placement_types import Shard, _StridedRaggedShard


def make_strided_shard(tensor: torch.Tensor, device_mesh: DeviceMesh, dims, local_units):
    fsdp_rank = device_mesh["fsdp"].get_local_rank()
    ep_rank = device_mesh["ep"].get_local_rank()
    placements = (_StridedRaggedShard(dims, local_units, split_factor=2), Shard(0))
    accum_local_units = (0, *itertools.accumulate(local_units))
    factor = tensor.shape[0] // device_mesh["ep"].size()
    local_tensor = tensor[factor * ep_rank : factor * (ep_rank + 1)].flatten()
    ratio = local_tensor.numel() // sum(local_units)
    local_tensor = local_tensor[accum_local_units[fsdp_rank] * ratio : accum_local_units[fsdp_rank + 1] * ratio]

    dtensor = DTensor(
        local_tensor,
        spec=DTensorSpec(
            device_mesh,
            placements,
            tensor_meta=TensorMeta(shape=tensor.shape, stride=tensor.stride(), dtype=tensor.dtype),
        ),
        requires_grad=tensor.requires_grad,
    )  # type: ignore

    return dtensor
