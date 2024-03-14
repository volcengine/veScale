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

# version info
__version__ = "0.0.1"
__branch__ = None
__commit_id__ = None

import warnings
import functools

import torch
import torch.utils._pytree as pytree

from vescale.dmodule.api import parallelize_module, is_dmodule, PlacementsInterface
from vescale.dtensor.api import normalize_placements, distribute_tensor, from_local, redistribute_dtensor, to_local
from vescale.dtensor.device_mesh import DeviceMesh, init_device_mesh
from vescale.dtensor.dtensor import DTensor
from vescale.dtensor.placement_types import Placement, Partial, Replicate, Shard, InterleavedShard
from vescale.dtensor import zeros, ones, empty, full, randn
from vescale.dtensor._utils import equal, allclose
from vescale.dmp import auto_parallelize_module, set_plan_overriding_policy, get_plan_overriding_policy
from vescale.initialize.deferred_init import deferred_init, is_deferred, materialize_dtensor, materialize_dparameter

# All public APIs from vescale package
__all__ = [
    "parallelize_module",
    "is_dmodule",
    "PlacementsInterface",
    "set_plan_overriding_policy",
    "get_plan_overriding_policy",
    "auto_parallelize_module",
    "DTensor",
    "DeviceMesh",
    "init_device_mesh",
    "normalize_placements",
    "distribute_tensor",
    "redistribute_dtensor",
    "Placement",
    "Partial",
    "Replicate",
    "Shard",
    "InterleavedShard",
    "from_local",
    "to_local",
    "equal",
    "allclose",
    "zeros",
    "ones",
    "empty",
    "full",
    "randn",
    "deferred_init",
    "is_deferred",
    "materialize_dtensor",
    "materialize_dparameter",
    "deprecated_function",
]


def deprecated_function(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn("deprecated jit.script function", UserWarning)
        return func(*args, **kwargs)

    return wrapper


torch.jit.script = deprecated_function


# dynamo utils
def switch_dtensor_for_torch_export(ep: torch.export.ExportedProgram):
    print(ep.graph_signature.parameters)
    if not isinstance(ep, torch.export.ExportedProgram):
        return
    for name, buffer_or_param in ep.state_dict.items():
        if not isinstance(buffer_or_param, DTensor):
            continue
        if isinstance(buffer_or_param, torch.nn.Parameter):
            ep.state_dict[name] = torch.nn.Parameter(buffer_or_param._local_tensor)
        else:
            ep.state_dict[name] = buffer_or_param._local_tensor
    # switch dtensor for constant tensors
    for name, tensor in ep._tensor_constants.items():
        if not isinstance(tensor, DTensor):
            continue
        ep._tensor_constants[name] = tensor._local_tensor
    # switch dtensor for example_inputs
    flat_example_inputs, tree_spec = pytree.tree_flatten(ep._example_inputs)
    tensor_flat_example_inputs = [x._local_tensor if isinstance(x, DTensor) else x for x in flat_example_inputs]
    ep._example_inputs = pytree.tree_unflatten(tensor_flat_example_inputs, tree_spec)

    # TODO: range_constraints, equality_constraints may also needed to be mofified.
    return ep
