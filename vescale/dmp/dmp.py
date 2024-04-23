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

"""DModule Parallelism (DMP): TP + SP + optional DP"""

from typing import Union, Dict, Tuple, Optional, List
from copy import deepcopy

from torch import nn

from vescale.dmodule.api import parallelize_module
from vescale.dtensor.device_mesh import DeviceMesh

__all__ = ["auto_parallelize_module", "set_plan_overriding_policy", "get_plan_overriding_policy"]


_IS_DEBUG = True

_PARAM_PLAN_OVERRIDING_POLICY = "PARAM_PLAN_OVERRIDING_POLICY"
_FWD_PLAN_OVERRIDING_POLICY = "FWD_PLAN_OVERRIDING_POLICY"


def set_plan_overriding_policy(
    module: nn.Module, param_sharding_plan: Optional[Dict] = None, fwd_resharding_plan: Optional[Dict] = None
) -> None:
    """Set root plans at this module, covering all submodules.
    Overrides auto_parallelize_module's policy."""
    assert isinstance(module, nn.Module)

    for name, submod in module.named_modules():
        if hasattr(submod, _PARAM_PLAN_OVERRIDING_POLICY) or hasattr(submod, _FWD_PLAN_OVERRIDING_POLICY):
            raise NotImplementedError(f"nested `set_plan_overriding_policy({name})` is not supported!")
            # hint: upper set_plan overrides inner set_plan?

    if param_sharding_plan is not None:
        setattr(module, _PARAM_PLAN_OVERRIDING_POLICY, param_sharding_plan)

    if fwd_resharding_plan is not None:
        setattr(module, _FWD_PLAN_OVERRIDING_POLICY, fwd_resharding_plan)


def get_plan_overriding_policy(module: nn.Module) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Get plan of this module only. plans are set by `set_plan_overriding_policy`."""
    return getattr(module, _PARAM_PLAN_OVERRIDING_POLICY, None), getattr(module, _FWD_PLAN_OVERRIDING_POLICY, None)


class PlanGenerator:
    # class attribute shared for all instances
    from .policies import REGISTRY

    _registry = REGISTRY

    def __init__(self, model: nn.Module, policy: Union[str, None, Dict[str, str]]):
        self.model = model
        if isinstance(policy, str):
            self.policy = policy.upper()
            if not self._registry.has_policy(self.policy):
                raise ValueError(
                    f"policy `{self.policy}` is not registered! "
                    f"Registered polices are: {self._registry.get_all_policies()}!"
                )
        else:
            raise NotImplementedError("Currently `policy` only supports specific `str`!")

    def generate(self) -> Tuple[Dict, Dict]:
        # 1. parse module tree
        # (reverse pre-order-traversal into pseudo post-order-traversal,
        #  so outer module's plans overrides inner module's plans, when nested module plans exists)
        fqn_mod = list(reversed(list(self.model.named_modules())))

        # 2a. get setted plans per module (by "get_plan_overriding_policy")
        # 2b. build its root plan
        setted_fqn_plans = []
        setted_fqn_for_param, setted_fqn_for_fwd = [], []  # for self module in `set_plan_overriding_policy(self)`
        for fqn, mod in fqn_mod:
            param_plan, fwd_plan = get_plan_overriding_policy(mod)
            if param_plan is None:
                param_plan = {}
            else:
                setted_fqn_for_param.append(fqn)
                if _IS_DEBUG:
                    print(f"[DEBUG] {fqn} : {mod.__class__.__name__} --set--> {param_plan}\n")  # noqa: E701
            if fwd_plan is None:
                fwd_plan = {}
            else:
                setted_fqn_for_fwd.append(fqn)
                if _IS_DEBUG:
                    print(f"[DEBUG] {fqn} : {mod.__class__.__name__} --set--> {fwd_plan}\n")  # noqa: E701
            setted_fqn_plans.append((fqn, param_plan, fwd_plan))
        setted_root_param_plan, setted_root_fwd_plan = self._build_root_plan(setted_fqn_plans)

        # 3a. get policied plans per module
        # 3b. build its root plan
        policied_fqn_plans = []
        for fqn, mod in fqn_mod:
            param_plan, fwd_plan = self._get_plans_by_policy(fqn, mod, self.policy)
            policied_fqn_plans.append((fqn, param_plan, fwd_plan))
            if _IS_DEBUG:
                print(f"[DEBUG] {fqn} : {mod.__class__.__name__} --policy--> {param_plan} & {fwd_plan}\n")  # noqa: E701
        policied_root_param_plan, policied_root_fwd_plan = self._build_root_plan(policied_fqn_plans)

        # 4. let setted root plan override policied root plan
        # (param_plans & fwd_plans are independent)
        root_param_plan = self._override_root_plan(
            setted_fqn_for_param, setted_root_param_plan, policied_root_param_plan
        )
        root_fwd_plan = self._override_root_plan(setted_fqn_for_fwd, setted_root_fwd_plan, policied_root_fwd_plan)

        # 5. TODO: patch above plans for cross-module correlation:
        #   e.g. MLP/Attention.RowLinear --Shard without redistribute--> LayerNorm

        # results
        return (
            root_param_plan,
            root_fwd_plan,
            setted_root_param_plan,
            setted_root_fwd_plan,
            policied_root_param_plan,
            policied_root_fwd_plan,
        )

    def _get_policy_provider(self, module):
        mod_cls_name = module.__class__.__name__.upper()
        # i. exact match
        policy_provider = self._registry.get_policy_provider(mod_cls_name)
        if policy_provider is not None:
            return policy_provider
        # ii. contained match
        policy_provider = self._registry.get_policy_provider_if_module_contains_registered_name(mod_cls_name)
        if policy_provider is not None:
            return policy_provider
        # iii. no match found
        return {}

    def _get_plans_by_policy(self, fqn, module, policy):
        policy_provider = self._get_policy_provider(module)
        provider = policy_provider.get(policy.upper(), None)
        if provider is None:
            # warnings.warn(f"module ({module.__class__.__name__}) has no policy ({policy}) registered. "
            #                "Use default_plan (Replicate) for this module.")
            return {}, {}
        param_plan, fwd_plan = provider(fqn, module)
        return param_plan, fwd_plan

    def _build_root_plan(self, fqn_plans: List) -> Tuple[Dict, Dict]:
        root_param_plan, root_fwd_plan = {}, {}
        for fqn, param_plan, fwd_plan in fqn_plans:
            for k, v in param_plan.items():
                root_param_plan[k if fqn == "" else fqn + "." + k] = v
            for k, v in fwd_plan.items():
                root_fwd_plan[k if fqn == "" else fqn + "." + k] = v
        return root_param_plan, root_fwd_plan

    def _inplace_remove_subplans_from_plan(self, fqns_start_to_remove: List, plan: Dict) -> None:
        for fqn in fqns_start_to_remove:
            keys_to_remove = [key for key in plan.keys() if key.startswith(fqn)]
            for k in keys_to_remove:
                plan.pop(k)

    def _override_root_plan(self, high_mod_paths: List, high_plan: Dict, low_plan: Dict) -> Dict:
        low_plan = deepcopy(low_plan)
        # extract high from low
        fqns_start_to_remove = high_mod_paths
        self._inplace_remove_subplans_from_plan(fqns_start_to_remove, low_plan)
        # add high into low
        for high_fqn, high_place in high_plan.items():
            low_plan[high_fqn] = high_place
        return low_plan


def auto_parallelize_module(
    model: nn.Module,
    device_mesh: Optional[DeviceMesh] = None,
    policy: Union[str, None, Dict[str, str]] = "MEGATRON",
    plan_only: bool = False,
):
    """
    High-level API to automatically parallelize a single-device model onto multiple devices, with policy guided sharding plan generation.

    Argument
        `model`: single-device model instance to parallelize

        `device_mesh`: the device mesh to parallelize model onto

        `policy`: the policy of sharding plans for the each submodule in this `model`.
                    - case "MEGATRON" -- use this "MEGATRON" policy for all submodules in this `model`
                    - case "None" (AUTO) -- automatically choose a policy for all submodules in this `model`
                    - case { "MLP" : "MEGATRON", "OTHERS" : "GIGATRON" } -- use "MEGATRON" policy for submodule class "MLP", and "GIGATRON" policy for all other module classes.

                  The plans generated by this policy can be overrided by manual set plans with `set_plan_overriding_policy`

        `plan_only`: flag to turn on for plan generation only but without really parallelize the `model`


    Return:
        the parallelized model under generated plans.

    NOTE: This function is experimental and is subject to change.
    """
    if _IS_DEBUG:
        print(f"[DEBUG] model={model}")  # noqa: E701

    # generate plans
    (
        root_param_plan,
        root_fwd_plan,
        setted_root_param_plan,
        setted_root_fwd_plan,
        policied_root_param_plan,
        policied_root_fwd_plan,
    ) = PlanGenerator(model, policy).generate()

    if _IS_DEBUG:
        print()
        print(f"[DEBUG] setted_root_param_plan = {setted_root_param_plan}")
        print(f"[DEBUG] policied_root_param_plan = {policied_root_param_plan}")
        print(f"[DEBUG] root_param_plan = {root_param_plan}")
        print()
        print(f"[DEBUG] setted_root_fwd_plan = {setted_root_fwd_plan}")
        print(f"[DEBUG] policied_root_fwd_plan = {policied_root_fwd_plan}")
        print(f"[DEBUG] root_fwd_plan = {root_fwd_plan}")
        print()

    # parallelize as planned
    sharding_plan = {"parameter": root_param_plan, "forward": root_fwd_plan}
    if plan_only:
        return model, device_mesh, sharding_plan
    return parallelize_module(model, device_mesh, sharding_plan)
