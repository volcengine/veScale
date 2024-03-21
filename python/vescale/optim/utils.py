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
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
################################################################################

import warnings


def _zero_grad_group_helper(group, set_to_none: bool = True):
    """Zero out the gradient for a group of parameters.
    Note: copied from torch.optim.optimizer."""
    for param in group:
        if param.grad is not None:
            if set_to_none:
                param.grad = None
            else:
                if param.grad.grad_fn is not None:
                    param.grad.detach_()
                else:
                    param.grad.requires_grad_(False)
                param.grad.zero_()


def param_is_sharded_or_replicate_on_first_rank(param):
    if not hasattr(param, "_spec") or not param._spec:
        warnings.warn(
            "Judge sharded info for no DTensor or torch.Tensor converted from DTensor, return False by default"
        )
        return False

    if any(p.is_partial() for p in param._spec.placements):
        raise RuntimeError("Detect partial sharded parameter, please check your param_sharding_plan")
    if all(p.is_shard() for p in param._spec.placements):
        return True

    param_device_mesh = param._spec.mesh
    for mesh_idx, p in enumerate(param._spec.placements):
        if not p.is_replicate():
            continue
        cur_coordinate = param_device_mesh.get_coordinate()
        if cur_coordinate and cur_coordinate[mesh_idx] == 0:
            return True
        else:
            return False
    return False


# TODO:
def param_is_shared(param):
    return False
