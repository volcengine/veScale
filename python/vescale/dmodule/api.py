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
# some code is inspired by torch/distributed/tensor/parallel/api.py
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
################################################################################

from typing import Dict, Optional, Union, Any
import warnings

from torch import nn
from vescale.dtensor.device_mesh import DeviceMesh, mesh_resources
from vescale.dmodule._dmodule import DModule
from vescale.dmodule.placements_interface import PlacementsInterface

__all__ = ["parallelize_module", "is_dmodule", "PlacementsInterface"]


def parallelize_module(
    module: nn.Module,
    device_mesh: Optional[DeviceMesh] = None,
    sharding_plan: Optional[Dict[str, Dict[str, Any]]] = None,
    *,
    is_model_sharded: bool = False,
    grad_sync: Union[bool, Dict] = True,
) -> nn.Module:
    r"""
    Parallelize this `nn.Module` instance by inplace converting its parameters/buffers/activations from Tensor to DTensor:
        1. onto target `device_mesh`
        2. with target `param_sharding_plan` and `fwd_resharding_plan`

    Args:
        device_mesh: the device mesh used in this entire DModule and its submodules

        sharding_plan:
            'parameter': the plan to specify which and how weights are sharded on device mesh during initalization.
                                Format: `{ <fully qualified name of weight/bias> : <sharding placements> }`
                                where
                                - <fully qualified name> is torch-native (see https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.get_parameter)
                                - <fully qualified name> uses regex match in form of `<regex pattern for submodule>.<weight/bias>`
                                - <sharding placements> can be either:
                                    - `None` for no op
                                    - `Sequence[Placement]` for sharding spec (see `Placement` in `python/vescale/dtensor/README.md`)
                                    - `PlacementsInterface(Sequence[Placement], <optional flags>)` for sharding spec with DTensor flags

                                Note: Non-specified parameters in module will be converted to DTensor in `Replicate`, i.e., the "default" param plan.

                                Example:
                                >>> # find submodule "fc1"'s "weight" and convert it to DTensor in `tensor.dim=1` sharded on `device_mesh.dim=0`
                                >>> # and convert the rest parameters to DTensor in `Replicate()`
                                >>> param_plan = { "fc1.weight" : [Shard(1)] }

            'forward': the plan to specify which and how submodules' input/weight/output are resharded on device mesh during forward pass.
                                Format: `{ <fully qualified name of input/weight/output> : <resharding placements> }`
                                where
                                - <fully qualified name> is same as above
                                - <fully qualified name of input/weight/output> uses regex match in form of `<regex pattern for submodule>.<input/weight/output>`
                                - <resharding placements> can be defined in two forms:

                                    1. List form: Just list all desired <sharding placements> (same as above), in a list.

                                        The order of placement should follow the order the the arguments are defined.
                                        And, for `*args`, the placements should be defined in the order of input.

                                        Example:
                                        >>> def foo(a, b, *args, c=1.0, **kwargs): pass
                                        >>> fwd_plan = {
                                                "input":[
                                                    [Replicate()], # placement for a
                                                    [Shard(0)],  # placement for b
                                                    [Shard(1)],  # placement for args[0]
                                                    None,        # no op for args[1]
                                                    None,       # no op for c
                                                    [Partial]  # placement for the first key in kwargs.
                                                ]
                                            }
                                        >>> foo(
                                                tensor0,  # will be Replicate
                                                tensor1,  # will be Shard(0)
                                                tensor2,  # will be Shard(1)
                                                tensor3,  # will be torch.Tenosr
                                                d = tensor4 # will be Partial
                                            )


                                    2. Dictionary form: Use the arg name as the key, and the <sharding placements> as the value.
                                        There is a special case where the key is `*args` and then the value is a list of placements.

                                        Example:
                                        >>> def foo(a, b, *args, c=1.0, **kwargs): pass
                                        >>> fwd_plan={
                                                "input":{
                                                    "a": [Replicate()], # placement for a
                                                    "b": [Shard(0)],  # placement for b
                                                    "args":[[Shard(1)], None],  # list of placements for args
                                                    "c": None,       # placement for c
                                                    "d": [Partial]  # placement for the first key in kwargs (called as d)
                                                }
                                            }
                                        >>> foo(
                                                tensor0,  # will be Replicate
                                                tensor1,  # will be Shard(0)
                                                tensor2,  # will be Shard(1)
                                                tensor3,  # will be torch.Tenosr
                                                d = tensor4 # will be Partial
                                            )

        is_model_sharded: is this model (parameters/buffers) already sharded?
                            - `False` (Default): each rank holds a full model
                            - `True`: each rank holds a only shard
                            which will be used for initalization internally.

        grad_sync: whether to turn on gradient synchronization (i.e., auto-allreduce `Partial` gradients) after backward pass.
                    Format: `True` or `False` or `{ submodule_cls : (param_name1, param_name2, ...) }`
                    where
                    - `True`: looking for all `submodule_cls` and all `param_names` whose `Partial` gradients will be allreduced.
                    - `False` or `{}`: disable gradient synchronization
                    - `{ submodule_cls : True }`: only looking for this `submodule_cls`'s all `param_name` for gradient synchronization.
                    - `{ submodule_cls : False or [] }`: exclude this `submodule_cls` for gradient synchronization.
                    - `{ submodule_cls : [param_name1] }`: only looking for this `submodule_cls`'s `param_name1` for gradient synchronization.

                    Note:
                    - If turned on, use `finish_grad_sync()` to wait for the gradient synchronization finish.
                    - If using veScale's optimizer, `finish_grad_sync()` is automatic and doesn't require manual call.

    Returns:
        (Optional) this parallelized model instance


    Example:: using `plans`

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(4, 8)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(8, 4)

            def forward(self, x):
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)
                return x

        with torch.device("cpu"):
            mlp = MLP()

        device_mesh = DeviceMesh("cuda", [0, 1, 2, 3])

        param_sharding_plan = {
            "fc1.weight": [Shard(0)],
            "fc1.bias": [Shard(0)],
            "fc2.weight": [Shard(1)],
            "fc2.bias": [Replicate()],
        }
        fwd_resharding_plan = {
            "fc1.input": [[Replicate()]],
            "fc2.output": [[Replicate()]],
        }

        dmlp = parallelize_module(mlp, device_mesh, param_sharding_plan, fwd_resharding_plan)
        output = dmlp(input)


    Example:: using `is_model_sharded`

        ...
        with torch.device("cpu"):
            mlp_shard = MLPShardedPerRank()
        ...
        dmlp = parallelize_module(mlp_shard, ..., is_model_sharded = True)


    Example:: using deferred initialization

        ...
        fake_model = deferred_init(MLP)
        ...
        dmlp = parallelize_module(fake_model, device_mesh, param_sharding_plan, fwd_resharding_plan)


    Example:: using gradient synchronization with customized target

        ...
        dmlp = parallelize_module(model, ..., grad_sync={ nn.LayerNorm: ["weight"] })
        dmlp.finish_grad_sync()
        optimizer.step()

    """

    if DModule.is_dmodule(module):
        warnings.warn(f"{module} is already parallelized `DModule`. Skip `parallelize_module`", UserWarning)
        return module

    # check sharding plan
    sharding_plan = DModule.check_and_sanitize_sharding_plan(sharding_plan)

    # create dmodule attributes
    DModule.initialize_attributes(module)

    # bind dmodule methods
    DModule.initialize_methods(module)

    # register mesh, plans, and more to self
    device_mesh = device_mesh or mesh_resources.get_current_mesh()
    DModule.register_sharding_plan(module, device_mesh, sharding_plan["parameter"], sharding_plan["forward"])

    # distribute params on target device mesh
    DModule.init_parameters(module, is_model_sharded)

    # install forward hooks
    DModule.init_forward(module)

    # post-patch submodules
    DModule.post_patch_submodules(module)

    # prepare gradient sync
    DModule.prepare_grad_sync(module, grad_sync)

    # tag this module as parallelized dmodule
    DModule.set_dmodule(module)

    return module


is_dmodule = DModule.is_dmodule
