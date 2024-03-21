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

import warnings
import logging
import inspect
from functools import partial
from typing import Any, Dict, Optional, Sequence, Union
from dataclasses import fields, is_dataclass

import torch
from torch import nn

from vescale.dtensor.device_mesh import DeviceMesh
from vescale.dtensor.dtensor import DTensor
from vescale.dtensor.placement_types import Placement
from vescale.dtensor._diff import DeferReshardMode
from vescale.dmodule.placements_interface import PlacementsInterface as PI

__all__ = ["PreHookInput", "PreHookWeight", "PostHookWeight", "PostHookOutput", "PostHookGrad"]

SeqFwdPIs = Sequence[Optional[PI]]
DictFwdPIs = Dict[str, Union[type(None), PI, SeqFwdPIs]]
FwdPIs = Union[SeqFwdPIs, DictFwdPIs]

WeightPIs = Dict[str, Optional[PI]]


def _convert_by_pi(
    x: Any, pi: Optional[PI], device_mesh: DeviceMesh, *, allow_defer=False, raise_err=False, ignore_none=True
):
    if pi is None or not pi.placements:
        return x
    if isinstance(x, DTensor):
        if allow_defer and pi.defer_reshard:
            DeferReshardMode._push_sharding(pi.placements)
            return x
        return x.redistribute(device_mesh, pi.placements, async_op=pi.async_op)
    if isinstance(x, torch.Tensor):
        return DTensor.from_local(x, device_mesh, pi.placements, run_check=pi.run_check, async_input=pi.async_op)
    if not raise_err:
        logging.info("binding a placement %s with a %s obj: %s. The placement is ignored.", pi.placements, type(x), x)
        return x
    if ignore_none and (x is None):
        return x
    raise RuntimeError(f"Trying to redistribute non-tensor values {type(x)}")


class PreHookInput:
    @staticmethod
    def _convert(x: Any, pi: Optional[PI], device_mesh: DeviceMesh):
        return _convert_by_pi(x, pi, device_mesh, raise_err=False)

    @staticmethod
    def _hook(module: nn.Module, args: Any, kwargs: Any, device_mesh: DeviceMesh, input_pis: FwdPIs):
        convert = lambda x, pi: PreHookInput._convert(x, pi, device_mesh)
        func_sig = inspect.signature(module.forward)
        bound_args = func_sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        parameters = func_sig.parameters

        var_position_name = None
        var_keyward_name = None
        for param in parameters.values():
            if param.kind == param.VAR_POSITIONAL and var_position_name is None:
                var_position_name = param.name
            if param.kind == param.VAR_KEYWORD and var_keyward_name is None:
                var_keyward_name = param.name

        if isinstance(input_pis, Sequence):
            n_pis = len(input_pis)
            n_args = len(bound_args.args)
            n_kwargs = len(bound_args.kwargs)
            if n_pis > (n_args + n_kwargs):
                warnings.warn(
                    f"The size of placements {n_pis} "
                    f"is bigger than the total args given {n_args + n_kwargs}, "
                    "ignoreing the remain"
                )
            input_pis = list(input_pis)
            input_pis += [None] * (n_args + n_kwargs - n_pis)
            arg_pis = input_pis[: len(bound_args.args)]
            kwarg_pis = input_pis[len(bound_args.args) :]
            new_args = tuple(convert(x, pi) for x, pi in zip(bound_args.args, arg_pis))
            new_kwargs = {kv[0]: convert(kv[1], pi) for kv, pi in zip(bound_args.kwargs.items(), kwarg_pis)}
            return new_args, new_kwargs
        if isinstance(input_pis, Dict):
            arg_set = set(bound_args.arguments.keys())
            if var_keyward_name is not None:
                arg_set = arg_set.union(set(bound_args.arguments[var_keyward_name].keys()))
                arg_set.remove(var_keyward_name)
            mismatched = input_pis.keys() - arg_set
            if len(mismatched) > 0:
                warnings.warn(f"placement mismatches with function definition. mismatched keys: {mismatched}")
            for k, v in bound_args.arguments.items():
                if k not in input_pis:
                    continue
                if k == "args":
                    arg_pis = list(input_pis["args"])
                    var_pos = bound_args.arguments["args"]
                    if len(arg_pis) > len(var_pos):
                        warnings.warn(
                            f"The size of placements {len(arg_pis)} "
                            f"is bigger than the total args given {len(var_pos)}, ignoreing the remain"
                        )
                    arg_pis += [None] * (len(var_pos) - len(arg_pis))
                    new_var_pos = [convert(v, pi) for v, pi in zip(var_pos, arg_pis)]
                    bound_args.arguments["args"] = new_var_pos
                    continue
                pi = input_pis[k]
                bound_args.arguments[k] = convert(v, pi)
            if var_keyward_name is None:
                return bound_args.args, bound_args.kwargs
            for k, v in bound_args.arguments[var_keyward_name].items():
                if k not in input_pis:
                    continue
                pi = input_pis[k]
                bound_args.arguments[var_keyward_name][k] = convert(v, pi)
            return bound_args.args, bound_args.kwargs
        raise TypeError(f"not supported forward placement type {type(input_pis)}")

    @staticmethod
    def get_hook(device_mesh: DeviceMesh, input_pis: FwdPIs):
        return partial(PreHookInput._hook, device_mesh=device_mesh, input_pis=input_pis)


class PreHookWeight:
    @staticmethod
    def _hook(module: nn.Module, input: Any, device_mesh: DeviceMesh, weight_pis: WeightPIs):
        for fqn, pi in weight_pis.items():  # in future, this should recurisvely redistribute parameter
            if pi is None or not pi.placements:
                continue
            submod_path, _, param_name = fqn.rpartition(".")
            submod = module.get_submodule(submod_path)
            param = submod.get_parameter(param_name)
            if isinstance(param.data, DTensor):
                submod.register_parameter(
                    param_name,
                    torch.nn.Parameter(
                        param.data.redistribute(device_mesh, pi.placements, async_op=pi.async_op),
                        requires_grad=param.requires_grad,
                    ),
                )
            else:
                raise RuntimeError("Trying to redistribute a non-DTensor in forward!")

    @staticmethod
    def get_hook(device_mesh: DeviceMesh, weight_pis: WeightPIs):
        return partial(PreHookWeight._hook, device_mesh=device_mesh, weight_pis=weight_pis)


class PostHookWeight:
    @staticmethod
    def _hook(module: nn.Module, input: Any, output: Any, device_mesh: DeviceMesh, weight_pis: WeightPIs):
        raise NotImplementedError("Placeholder for FSDP.")

    @staticmethod
    def get_hook(device_mesh: DeviceMesh, weight_pis: WeightPIs):
        return partial(PostHookWeight._hook, device_mesh=device_mesh, weight_pis=weight_pis)


class PostHookOutput:
    @staticmethod
    def _convert(x: Any, pi: Optional[PI], device_mesh: DeviceMesh):
        return _convert_by_pi(x, pi, device_mesh, allow_defer=True, raise_err=True, ignore_none=True)

    @staticmethod
    def _convert_dictlike(output_dict: Dict[str, Any], pi_dict: DictFwdPIs, device_mesh: DeviceMesh):
        assert isinstance(pi_dict, Dict), f"{type(output_dict)}"
        new_output = {}
        for key in output_dict:
            if key in pi_dict:
                new_output[key] = PostHookOutput._convert(output_dict[key], pi_dict[key], device_mesh)
            else:
                new_output[key] = output_dict[key]
        return type(output_dict)(**new_output)

    @staticmethod
    def _hook(
        module: nn.Module,
        input: Any,
        output: Union[Sequence, DTensor, torch.Tensor, Dict, Any],
        device_mesh: DeviceMesh,
        output_pis: FwdPIs,
    ):
        if isinstance(output, Sequence) and isinstance(output_pis, Sequence):
            assert len(output) == len(
                output_pis
            ), f"Mismatched actual output size: {output} vs. plaments size: {output_pis}!"
            return [PostHookOutput._convert(o, pi, device_mesh) for o, pi in zip(output, output_pis)]
        if isinstance(output, DTensor) and output_pis[0] is not None:
            return PostHookOutput._convert(output, output_pis[0], device_mesh)
        if isinstance(output, torch.Tensor) and output_pis[0] is not None:
            return PostHookOutput._convert(output, output_pis[0], device_mesh)
        if isinstance(output, Dict):
            return PostHookOutput._convert_dictlike(output, output_pis, device_mesh)
        if is_dataclass(output):
            value_dict = {field.name: getattr(output, field.name) for field in fields(output)}
            rt = PostHookOutput._convert_dictlike(value_dict, output_pis, device_mesh)
            return type(output)(**rt)
        return None  # TODO: to check None case

    @staticmethod
    def get_hook(device_mesh: DeviceMesh, output_pis: FwdPIs):
        return partial(PostHookOutput._hook, device_mesh=device_mesh, output_pis=output_pis)


class PostHookGrad:
    @staticmethod
    def _hook(grad: Any, device_mesh: DeviceMesh, grad_placements: Optional[Sequence[Placement]]):
        if not grad_placements:
            return grad
        if grad is None:
            return None
        if isinstance(grad, DTensor):
            grad._spec.placements = tuple(grad_placements)
            return grad
        raise ValueError("grad_hook should only be registered on DTensor parameter and DTensor grad.")

    @staticmethod
    def get_hook(device_mesh: DeviceMesh, grad_placements: Optional[Sequence[Placement]]):
        return partial(PostHookGrad._hook, device_mesh=device_mesh, grad_placements=grad_placements)
