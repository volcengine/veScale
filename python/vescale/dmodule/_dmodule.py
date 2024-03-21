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

import collections
import re
import difflib
import warnings
from types import MethodType
from typing import Any, List, Deque, Dict, Mapping, Optional, Sequence, Set, Tuple, Union

import torch
from torch import nn, Tensor

from vescale.dtensor.api import distribute_tensor
from vescale.dtensor.device_mesh import DeviceMesh
from vescale.dtensor.dtensor import DTensor
from vescale.dtensor.placement_types import Placement, Replicate
import vescale.dmodule._grad_sync as _grad_sync
from vescale.dmodule.placements_interface import PlacementsInterface as PI
from vescale.dmodule._hook import *  # noqa: F403

__all__ = ["DModule"]


_TAG_DMODULE = "DMODULE"
_ORGIN_LOAD_STATE_DICT = "_origin_load_state_dict"


class DModule:
    def __init__(self) -> None:
        raise RuntimeError("`DModule` inheritance is deprecated. Use `parallelize_module` instead.")

    @staticmethod
    def check_and_sanitize_sharding_plan(sharding_plan: Optional[Dict[str, Dict[str, Any]]]):
        if sharding_plan is None:
            sharding_plan = {}
        if "parameter" not in sharding_plan:
            sharding_plan["parameter"] = {}
        if "forward" not in sharding_plan:
            sharding_plan["forward"] = {}
        accepted_keys = ("parameter", "forward")
        for key in sharding_plan.keys():
            if key in accepted_keys:
                continue
            possible_key = difflib.get_close_matches(key, accepted_keys, n=1)
            if len(possible_key) > 0:
                raise KeyError(
                    f"parallelize_module does not support `{key}`as key, probably you mean {possible_key[0]}?"
                )
            else:
                raise KeyError(f"parallelize_module does not support `{key}`as key.")
        return sharding_plan

    @staticmethod
    def is_dmodule(module: nn.Module) -> bool:
        return hasattr(module, _TAG_DMODULE)

    @staticmethod
    def set_dmodule(module: nn.Module) -> None:
        if hasattr(module, _TAG_DMODULE):
            warnings.warn(f"resetting `{module.__class__}` as DModule!", UserWarning)
        setattr(module, _TAG_DMODULE, True)

    @staticmethod
    def initialize_attributes(module: nn.Module) -> None:
        """registered device mesh for this DModule"""
        module._device_mesh: Optional[DeviceMesh] = None

        """
        registered plans for this DModule

        format:
        _param_sharding_plan = { <submod_fqn> : { <param_name> : <pi> } }
        _fwd_resharding_plan = { <submod_fqn> :
                                { <input/output/weight>:
                                    [<pi>,]|{ karg/param_name : <pi>  or "arg" : [<pi>, ]}
                                }
                            }
        """
        module._param_sharding_plan: Dict[str, Dict[str, Optional[PI]]] = {}
        module._fwd_resharding_plan: Dict[
            str,
            Dict[
                str,
                Sequence[Optional[PI]] or Dict[str, Optional[PI] or Sequence[Optional[PI]]],
            ],
        ] = {}

        """
        registered others for this DModule
        """
        module._grad_sync_candidate: List[Tuple[str, DTensor]] = []  # [(param_fqn, .param)]

    @staticmethod
    def has_all_attributes(module: nn.Module) -> bool:
        return (
            hasattr(module, "_device_mesh")
            and hasattr(module, "_param_sharding_plan")
            and hasattr(module, "_fwd_resharding_plan")
            and hasattr(module, "_grad_sync_candidate")
        )

    @staticmethod
    def register_sharding_plan(
        module: nn.Module,
        device_mesh: DeviceMesh,
        param_sharding_plan: Optional[Dict[str, Any]],
        fwd_resharding_plan: Optional[Dict[str, Any]],
    ) -> None:
        """
        Validate, normalize, and register arguments into self's attributes
        """
        assert DModule.has_all_attributes(module)

        if device_mesh.device_type not in ("cpu", "cuda", "meta"):
            if device_mesh.device_type.startswith("cuda:"):
                warnings.warn("`device_mesh.device_type` include `rank` is deprecated", UserWarning)
            else:
                raise ValueError(f"Unknown `device_mesh.device_type`: {device_mesh.device_type}!")
        if module._device_mesh is not None and module._device_mesh != device_mesh:
            raise RuntimeError("Trying to register sharding plan with different device mesh")
        module._device_mesh = device_mesh

        def _normalize_one_placements(
            placements: Union[None, Sequence[Placement], PI],
        ) -> Optional[PI]:
            if not isinstance(placements, (type(None), Sequence, PI)):
                raise ValueError(
                    f"<placements> must either be `None, Sequence[Placement], PlacementsInterface,  but found: {placements}!"
                )
            if isinstance(placements, Sequence) and not all(isinstance(p, Placement) for p in placements):
                raise ValueError(f"<placements> must be `Sequence[Placement]`, but found: {placements}!")
            # wrap as PI
            pi = PI.from_placements(placements)
            if pi.is_none():  # None for None
                return None
            pi.normalize_placements(device_mesh.ndim)
            return pi  # valid PI

        if param_sharding_plan:
            for param_path, placements in param_sharding_plan.items():
                submod_pattern, _, param_name = param_path.rpartition(".")  # including empty pattern (self module)
                for submod_fqn, _ in module.named_modules():
                    if not re.fullmatch(submod_pattern, submod_fqn):
                        continue
                    if submod_fqn not in module._param_sharding_plan:
                        module._param_sharding_plan[submod_fqn] = {}
                    if param_name not in module._param_sharding_plan[submod_fqn]:
                        module._param_sharding_plan[submod_fqn][param_name] = _normalize_one_placements(placements)
                    else:
                        warnings.warn(f"Duplicated param sharding plan for {submod_fqn}.{param_name}!")
                        module._param_sharding_plan[submod_fqn][param_name] = _normalize_one_placements(placements)

        if fwd_resharding_plan:
            for tensor_path, placements in fwd_resharding_plan.items():
                submod_pattern, _, tensor_name = tensor_path.rpartition(".")  # including empty pattern (self module)
                tensor_category = tensor_name  # for separate specific weight
                for submod_fqn, _ in module.named_modules():
                    if not re.fullmatch(submod_pattern, submod_fqn):
                        continue

                    if submod_fqn not in module._fwd_resharding_plan:
                        module._fwd_resharding_plan[submod_fqn] = {
                            "input": None,
                            "output": None,
                            "weight": {},
                        }

                    if tensor_category in ("input", "output"):
                        # normalize plan
                        if isinstance(placements, Sequence) and any(isinstance(p, Dict) for p in placements):
                            raise NotImplementedError(
                                "Currently, no support for mix `args and kwargs placement` in fwd plan!"
                            )
                        if isinstance(placements, Sequence):
                            pis = [_normalize_one_placements(p) for p in placements]
                        elif isinstance(placements, Dict):
                            pis = {}
                            for k, v in placements.items():
                                if k == "args":
                                    assert isinstance(
                                        v, Sequence
                                    ), "the placements for variable position arguments have to be list"
                                    pis[k] = [_normalize_one_placements(p) for p in v]
                                else:
                                    pis[k] = _normalize_one_placements(v)
                        # register plan
                        module._fwd_resharding_plan[submod_fqn][tensor_category] = pis
                    else:  # weight/bias
                        tensor_category = "weight"
                        module._fwd_resharding_plan[submod_fqn][tensor_category][tensor_name] = (
                            _normalize_one_placements(placements)
                        )

    @staticmethod
    @torch.no_grad()
    def _distribute_parameter(
        param: Union[nn.Parameter, torch.Tensor],
        device_mesh: DeviceMesh,
        pi: PI,
        is_sharded: bool = False,
    ):
        """
        Initialize/materialize a distributed parameter or buffer.

        Behave as follow:
            in: param (type: Parameter), param.data (type: Tensor)
            out: param (type: DTensor, isinstance: Parameter & DTensor), param.data (type: DTensor)

            in: buff (type: Tensor), buff.data (type: Tensor)
            out: buff (type: DTensor), buff.data (type: DTensor)
        """

        if type(param) is nn.Parameter:
            is_param = True
            t = param.data
        else:  # buffer
            is_param = False
            t = param
        assert not isinstance(t, DTensor), "_distribute_parameter should not take `DTensor`!"

        # deferred materialization
        from vescale.initialize.deferred_init import is_deferred, materialize_dtensor, materialize_dparameter

        if is_param and is_deferred(param):  # have to use `param` to check
            if is_sharded:
                raise NotImplementedError(
                    "Deferred initialization for tensor that is already sharded!"
                    "Use global tensor without being sharded instead!"
                )
            return materialize_dparameter(param, device_mesh, pi.placements)
        elif is_param is False and is_deferred(t):
            if is_sharded:
                raise NotImplementedError(
                    "Deferred initialization for tensor that is already sharded!"
                    "Use global tensor without being sharded instead!"
                )
            return materialize_dtensor(t, device_mesh, pi.placements)

        # regular intialization
        if is_sharded:
            dt = DTensor.from_local(t, device_mesh, pi.placements, run_check=pi.run_check)
        else:
            dt = distribute_tensor(t, device_mesh, pi.placements)
        return nn.Parameter(dt, requires_grad=param.requires_grad) if is_param else dt

    @staticmethod
    def init_parameters(module: nn.Module, is_sharded: bool):
        """
        Initialize/materialize distributed parameters and buffers onto `device_mesh`.
        Appointed parameters and buffers will be initialized using `param_sharding_plan`.
        Non-appointed parameters and buffers will be `Replicate` (i.e., default plan).
        """
        assert DModule.has_all_attributes(module)
        # pre-order traverse root and submodules
        for submod_path, submod in module.named_modules():
            # get assigned plans from root
            assigned_submod_plan = module._param_sharding_plan.get(submod_path, {})

            # distribute immediate weights and bias
            for param_name, param in submod.named_parameters(recurse=False):
                param_pi = assigned_submod_plan.get(param_name, None)  # appointed plan
                if param_pi is None:
                    param_pi = PI()
                if param_pi.placements is None:  # default plan
                    param_pi.placements = [Replicate()] * module._device_mesh.ndim
                param = DModule._distribute_parameter(param, module._device_mesh, param_pi, is_sharded)
                submod.register_parameter(param_name, param)

            # distribute immediate buffers
            for buffer_name, buffer in submod.named_buffers(recurse=False):
                buffer_pi = assigned_submod_plan.get(buffer_name, None)  # appointed plan
                if buffer_pi is None:
                    buffer_pi = PI()
                if buffer_pi.placements is None:  # default plan
                    buffer_pi.placements = [Replicate()] * module._device_mesh.ndim
                buffer = DModule._distribute_parameter(buffer, module._device_mesh, buffer_pi, is_sharded)
                submod.register_buffer(buffer_name, buffer)

    @staticmethod
    def init_forward(module: nn.Module):
        """
        Install pre/post-forward hooks appointed by `fwd_resharding_plan` onto `device_mesh`.
        """
        assert DModule.has_all_attributes(module)
        # pre-order traverse root and submodules
        for submod_path, submod in module.named_modules():
            # install input hook
            input_sharding_plan = module.get_fwd_plan(submod_path + ".input")
            if input_sharding_plan:
                submod.register_forward_pre_hook(
                    PreHookInput.get_hook(module._device_mesh, input_sharding_plan),
                    with_kwargs=True,
                )

            # install output hook
            output_sharding_plan = module.get_fwd_plan(submod_path + ".output")
            if output_sharding_plan:
                submod.register_forward_hook(
                    PostHookOutput.get_hook(module._device_mesh, output_sharding_plan),
                )

            # install weight hook
            weight_sharding_plan = module.get_fwd_plan(submod_path + ".weight")
            if weight_sharding_plan:
                submod.register_forward_pre_hook(
                    PreHookWeight.get_hook(module._device_mesh, weight_sharding_plan),
                )
                # submod.register_forward_hook(PostHookWeight.get_hook(module._device_mesh, weight_sharding_plan))

            # install weight grad hook
            if weight_sharding_plan and any(bool(v.grad) for v in weight_sharding_plan.values() if v is not None):
                for param_name, param in submod.named_parameters(recurse=False):  # immediate grad only
                    if not param.requires_grad:
                        continue
                    if param_name not in weight_sharding_plan:
                        continue
                    weight_pi = weight_sharding_plan[param_name]
                    if weight_pi is None or not weight_pi.grad:
                        continue
                    param.register_hook(PostHookGrad.get_hook(module._device_mesh, weight_pi.grad))

    @staticmethod
    def post_patch_submodules(module: nn.Module) -> None:
        r"""Post patching specific submodules with implementation under `vescale.model.patch`.
        (DModule is generic module class and should NOT contain specific layers or ops.).
        """
        import vescale.model.patch as model_patch

        for specific_model_patch in model_patch.get_all_model_patch():
            specific_model_patch(module)

    @staticmethod
    def prepare_grad_sync(module: nn.Module, grad_sync: Union[bool, Dict]) -> None:
        """
        parse the given `grad_sync` and prepare a list of candidiates for gradient sync.
        """
        assert DModule.has_all_attributes(module)

        module._grad_sync_candidate = []
        if not grad_sync:  # False or {}
            return

        def is_candidate(mod: nn.Module, pname: str) -> bool:
            if grad_sync is True:
                return True

            for clss, pnames in grad_sync.items():
                if not isinstance(mod, clss):
                    continue
                if not pnames:  # False or []
                    continue
                if pnames is True or pname in pnames:
                    return True
            return False

        for submod_fqn, submod in module.named_modules():
            for param_name, param in submod.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue
                if not isinstance(param.data, DTensor):
                    continue
                if is_candidate(submod, param_name):
                    module._grad_sync_candidate.append((f"{submod_fqn}.{param_name}".lstrip("."), param))

    """ ============ Bound Methods Below ============ """

    @staticmethod
    def initialize_methods(module: nn.Module) -> None:
        # assign new method
        assert not hasattr(module, "get_fwd_plan"), f"{module.__class__} already has a method called `get_fwd_plan`!"
        module.get_fwd_plan = MethodType(DModule.get_fwd_plan, module)

        assert not hasattr(
            module, "start_grad_sync"
        ), f"{module.__class__} already has a method called `start_grad_sync`!"
        module.start_grad_sync = MethodType(DModule.start_grad_sync, module)

        assert not hasattr(
            module, "finish_grad_sync"
        ), f"{module.__class__} already has a method called `finish_grad_sync`!"
        module.finish_grad_sync = MethodType(DModule.finish_grad_sync, module)

        assert not hasattr(
            module, "list_grad_sync"
        ), f"{module.__class__} already has a method called `list_grad_sync`!"
        module.list_grad_sync = MethodType(DModule.list_grad_sync, module)

        assert not hasattr(module, "repr_params"), f"{module.__class__} already has a method called `repr_params`!"
        module.repr_params = MethodType(DModule.repr_params, module)

        assert not hasattr(
            module, "dump_mesh_plans"
        ), f"{module.__class__} already has a method called `dump_mesh_plans`!"
        module.dump_mesh_plans = MethodType(DModule.dump_mesh_plans, module)

        # replace original method
        assert not hasattr(
            module, _ORGIN_LOAD_STATE_DICT
        ), f"{module.__class__} already has a method called `{_ORGIN_LOAD_STATE_DICT}`!"
        setattr(module, _ORGIN_LOAD_STATE_DICT, module.load_state_dict)
        module.load_state_dict = MethodType(DModule.load_state_dict, module)

    def get_fwd_plan(self: nn.Module, tensor_path: str) -> Any:
        """Get registered fwd resharding plan for the given `tensor_path` fqn.
        No regex for `tensor_path`.
        """
        tensor_path = tensor_path.lstrip(".")
        submod_path, _, tensor_name = tensor_path.rpartition(".")  # including empty pattern (self module)
        tensor_category = tensor_name  # for separate specific weight

        assgined_fwd_resharding_plan = self._fwd_resharding_plan.get(submod_path, {})
        if tensor_category in ("input", "output"):
            return assgined_fwd_resharding_plan.get(tensor_category, None)
        else:  # weight/bias
            tensor_category = "weight"
            return assgined_fwd_resharding_plan.get("weight", None)

    def start_grad_sync(self: nn.Module) -> None:
        self._grad_sync_list = _grad_sync.generate_grad_sync_list(self._grad_sync_candidate)
        _grad_sync.sync_gradients(self._grad_sync_list, self._device_mesh)

    def finish_grad_sync(self: nn.Module) -> None:
        # TODO: think about overlapping with backwarding
        self.start_grad_sync()

    def list_grad_sync(self: nn.Module) -> List[Tuple[str, Union[Tensor, DTensor]]]:
        """
        list which gradients are used for gradient sync.
        """
        print("*** format: [(fqn, .main_grad or .grad on Partial)] ***")
        for fqn, grad in self._grad_sync_list:
            print(f"{fqn}:\t{grad._spec}")
        print("*******************************************************")
        return self._grad_sync_list

    def repr_params(
        self: nn.Module, show_shape=True, show_type=True, show_shard=True, show_mesh=True, show_ltensor_shape=True
    ):
        r"""
        Print all parameter stats in a tree-like structure
        """

        def _param_str(param: nn.Parameter) -> str:
            if isinstance(param.data, DTensor):
                param_str = "DTensorParam("
                if show_shape:
                    param_str += f"shape={param.data.shape}, "
                if show_type:
                    param_str += f"shape={param.data.type()}, "
                if show_shard:
                    param_str += f"placements={param.data.placements}, "
                if show_mesh:
                    param_str += f"mesh_shape={param.data._spec.mesh.mesh.shape}, "
                if show_ltensor_shape:
                    param_str += f"local_shape={param.data._local_tensor.shape}, "
                if param_str.endswith(", "):
                    param_str = param_str[:-2]
                param_str += ")"
            elif isinstance(param.data, nn.Tensor):
                param_str = "TensorParam("
                if show_shape:
                    param_str += f"shape={param.data.shape}, "
                if show_type:
                    param_str += f"shape={param.data.type()}, "
                if param_str.endswith(", "):
                    param_str = param_str[:-2]
                param_str += ")"
            else:
                param_str = "NonTensorParam"
            return param_str

        def _traverse_submod(module: nn.Module) -> str:
            # Track the visited modules in case of shared modules, which implies the
            # module graph is no longer a tree
            visited_modules: Set[nn.Module] = set()

            # Perform depth-first search from `module` to ensure that we do not
            # traverse into an incompatible API's subtree (use DFS instead of BFS to
            # match `.modules()` order)
            deque: Deque[Tuple(str, nn.Module, int)] = collections.deque([("", module, 0)])
            main_str = ""
            while deque:
                modname, submodule, indent = deque.popleft()
                visited_modules.add(submodule)
                for name, child_module in reversed(list(submodule.named_children())):
                    if child_module not in visited_modules:
                        deque.appendleft((name, child_module, indent + 2))
                main_str += " " * indent + f"({modname}): {submodule._get_name()}("
                print_param = False
                if len(list(submodule.children())) == 0:  # leaf submodule
                    for name, param in submodule.named_parameters():
                        main_str += "\n" + " " * (indent + 2) + f"{name}: {_param_str(param)}"
                        print_param = True
                    if print_param:
                        main_str += "\n"
                if print_param:
                    main_str += " " * indent
                main_str += ")\n"
            return main_str

        return _traverse_submod(self)

    def dump_mesh_plans(self: nn.Module, is_print: bool = True, msg: str = "") -> Tuple[DeviceMesh, Dict, Dict]:
        if is_print:
            print()
            print(f"************ {msg} used device_mesh  ************")
            print(self._device_mesh)
            print()
            print(f"************ {msg} used param_sharding_plan  ************")
            for k, v in self._param_sharding_plan.items():
                print(f"{k}\t:\t{v}")
            print()
            print(f"************ {msg} used fwd_sharding_plan  ************")
            for k, v in self._fwd_resharding_plan.items():
                print(f"{k}\t:\t{v}")
            print()

        return self._device_mesh, self._param_sharding_plan, self._fwd_resharding_plan

    def load_state_dict(self: nn.Module, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        r"""
        This is a wrapper around user module's original `load_state_dict`.
        This wrapper supports:
        - loading DTensor state dict
        - customized logic within user's original `load_state_dict`
        But this wrapper might not support:
        - pre-hook & post-hook on user's `load_state_dict`, due to method overloading

        -------------------------------------------------------------------------------------
        Inplace copies parameters and buffers (either torch.Tensor or DTensor)
        from :attr:`state_dict` into this module and its descendants.
        If :attr:`strict` is ``True``, then the keys of :attr:`state_dict` must
        exactly match the keys returned by this module's :meth:`~DModule.state_dict` function.

        .. warning::
            :attr:`assign` as ``True`` is not supported currently.

        Args:
            state_dict (dict): a dict containing parameters and persistent buffers.
            strict (bool, optional): whether to strictly enforce that the keys
                in :attr:`state_dict` match the keys returned by this module's
                :meth:`~DModule.state_dict` function. Default: ``True``
            assign (bool, optional): whether to assign items in the state
                dictionary to their corresponding keys in the module instead
                of copying them inplace into the module's current parameters and buffers.
                When ``False``, the properties of the tensors in the current
                module are preserved while when ``True``, the properties of the
                Tensors in the state dict are preserved.
                Default: ``False``

        Returns:
            ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
                * **missing_keys** is a list of str containing the missing keys
                * **unexpected_keys** is a list of str containing the unexpected keys

        Example::

            class MLP(nn.Module):
                ...

            dmlp1 = parallelize_module(...)
            torch.save(dmlp1.state_dict(), PATH)

            dmlp2 = parallelize_module(...)
            dmlp2.load_state_dict(torch.load(PATH))
            output = dmlp2(input)

        Note:
            If a parameter or buffer is registered as ``None`` and its corresponding key
            exists in :attr:`state_dict`, :meth:`load_state_dict` will raise a
            ``RuntimeError``.
        """
        if assign:
            raise NotImplementedError("load_dist_state_dict by `assign` is not supported yet!")

        # find all device meshs
        mesh_type: Sequence[Sequence[DeviceMesh, str]] = [
            (v._spec.mesh, v._spec.mesh.device_type) for k, v in state_dict.items() if isinstance(v, DTensor)
        ]
        # enforce cpu mesh tensor, as it should be
        for dm, _ in mesh_type:
            dm.enforce_cpu_mesh_tensor()
        # set device mesh as self device type
        self_device_type = next(self.parameters(), torch.empty(0)).device.type
        for dm, _ in mesh_type:
            dm.device_type = self_device_type

        # call regular nn.Module.load_state_dict:
        #   global tensor and local tensor: in-place copy across device, holds self tensor's attributes
        #   device mesh: compare mesh tensor, if __eq__ then reuse self device mesh, else raise error
        assert hasattr(self, _ORGIN_LOAD_STATE_DICT)
        ret = getattr(self, _ORGIN_LOAD_STATE_DICT)(state_dict, strict=strict, assign=assign)

        # restore device mesh's device type
        for dm, device_type in mesh_type:
            dm.device_type = device_type

        return ret
