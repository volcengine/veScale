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

import functools
from typing import Callable, Optional, List, Union, Dict, Set


class Registry:
    def __init__(self):
        self.mpp: Dict[str, Dict[str, Callable]] = {}
        """
        Format:
            { <module class name> : { <policy name> : <plan provider funcation> } }

        Example:
            {
                "MLP" : { "MEGATRON" : lambda n,m : ({ "fc1.weight" : [Shard(0)]},
                                                   { "input" : [[Replicate()]] }),
                          "GIGATRON" : lambda n,m : ({}, {}) },
            }
        """

    def provide_register_for_policy(self, policy_name: str) -> Callable:
        """
        Return a @register decorator under this <policy>.
        Use @register decorator to register the <plan_provider_fn> for the <module_cls_name>.

        Example:

            register = registry_inst.provide_register_for_policy("MEGATRON")

            @register("MLP" | ["MLPA", "MLPB"])
            def mlp_plan_provider(fqn, mod):
                ...
                return plans

        """
        policy_name = policy_name.upper()

        def register(module_cls_name: Union[str, List[str]]) -> Callable:
            # print(f"[DEBUG] register {module_cls_name} under {policy_name}!")

            # normalize names
            if isinstance(module_cls_name, str):
                module_cls_name = [module_cls_name]
            module_cls_name = [mcn.upper() for mcn in module_cls_name]

            # record decorated func into registry dict
            def decorator_func(plan_provider_fn: Callable) -> Callable:
                for mcn in module_cls_name:
                    policy_provider = self.mpp.get(mcn, None)
                    if policy_provider is None:
                        self.mpp[mcn] = {policy_name: plan_provider_fn}
                    else:
                        if policy_name in policy_provider:
                            raise ValueError(f"Conflicting policy `{policy_name}` for module `{mcn}`!")
                        policy_provider[policy_name] = plan_provider_fn
                return plan_provider_fn

            return decorator_func

        return register

    def __repr__(self) -> str:
        ret = ["====== REGISTRY ====="]
        ret.append(r"== ModuleClsName : { PolicyName : PlanProviderFn } ==")
        for mcn, policy_provider in self.mpp.items():
            ret.append(f"{mcn} : {policy_provider}")
        ret.append("=====================")
        return "\n".join(ret)

    def has_module(self, module_cls_name: str) -> bool:
        return module_cls_name.upper() in self.mpp

    def get_all_modules(self) -> Set[str]:
        return set(self.mpp.keys())

    def has_policy(self, policy_name: str) -> bool:
        return policy_name.upper() in self.get_all_policies()

    @functools.cache
    def get_all_policies(self) -> Set[str]:
        ret = set()
        for policies in self.mpp.values():
            for p in policies.keys():
                ret.add(p)
        return ret

    def get_policy_provider(self, module_cls_name: str) -> Optional[Dict]:
        return self.mpp.get(module_cls_name.upper(), None)

    def get_policy_provider_if_module_contains_registered_name(self, module_cls_name: str) -> Optional[Dict]:
        module_cls_name = module_cls_name.upper()
        for reg_cls_name, policy_provider in self.mpp.items():
            if reg_cls_name in module_cls_name:
                return policy_provider
        return None


REGISTRY = Registry()
