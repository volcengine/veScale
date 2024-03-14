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
from typing import Tuple, Dict
from torch import nn
from vescale.dtensor.placement_types import Shard, Replicate

from .registry import REGISTRY
from .utils import validate_single_input


_IS_DEBUG = True

register = REGISTRY.provide_register_for_policy("MEGATRON")


@register(["MLP"])
def mlp_plan_provider(fqn: str, module: nn.Module, *, sync_dropout: bool = True) -> Tuple[Dict, Dict]:
    if _IS_DEBUG:
        print(f"[DEBUG] mlp_plan_provider({fqn}, {module})")  # noqa: E701
    validate_single_input(module)

    param_plan, fwd_plan = {}, {}
    #  [ ("fc1", nn.Linear()), ("fc2", nn.Linear()) ]
    linear_children = list(
        filter(lambda child_basename_module: isinstance(child_basename_module[1], nn.Linear), module.named_children())
    )
    last_even_num = (len(linear_children) // 2) * 2
    for i in range(last_even_num):
        child_basename, child_module = linear_children[i]
        if i % 2 == 0:  # ColWise
            input_placement = [[Replicate()]]
            weight_placement = [Shard(0)]
            bias_placement = [Shard(0)]
            output_placement = [[Shard(-1)]]
        else:  # RowWise
            input_placement = [[Shard(-1)]]
            weight_placement = [Shard(1)]
            bias_placement = [Replicate()]
            output_placement = [[Replicate()]]  # TODO: auto [[Shard(-2)]] if followed by SP block
        # nn.Linear always has "weight" (Tensor) and "bias" (None/Tensor)
        fwd_plan[child_basename + ".input"] = input_placement
        fwd_plan[child_basename + ".output"] = output_placement
        param_plan[child_basename + ".weight"] = weight_placement
        if child_module.bias is not None:
            param_plan[child_basename + ".bias"] = bias_placement

    # Below is for 2nd linear (Partial output) only
    # dropout_children = list(
    #     filter(lambda child_basename_module: isinstance(child_basename_module[1], nn.Dropout),
    #             module.named_children())
    # )
    # if dropout_children and sync_dropout:
    #     # after 2nd linear (Partial output), we might have a dropouput:
    #     #  - when each rank is locally seeded with the same random seed, no sync_dropout is necessary
    #     #  - when each rank is globally seeded with dtensor's random seed, sync_droupt is necessary
    #     # Future: automatally choose sync_dropout by detecting which case above
    #     for child_basename, child_module in dropout_children:
    #         fwd_plan[child_basename + ".input"] = [[Replicate()]]

    if len(linear_children) > last_even_num:
        warnings.warn(
            f"MLP of MEGATRON policy encountered odd num of Linears ({len(linear_children)}); "
            "Use default plan (Replicate) for the last Linear, "
            "while keeping front even num of Linears as Megatron-Pairs!"
        )
        fwd_plan[linear_children[-1][0] + ".input"] = [[Replicate()]]

    # fwd_plan["output"] = [[Replicate()]] # for Partial() output only

    return param_plan, fwd_plan


@register(["Attention", "Attn", "Atten"])
def attention_plan_provider(fqn: str, module: nn.Module) -> Tuple[Dict, Dict]:
    validate_single_input(module)
    if _IS_DEBUG:
        print(f"[DEBUG] attention_plan_provider({fqn}, {module})")  # noqa: E701

    param_plan, fwd_plan = {}, {}

    linear_children = list(
        filter(lambda child_basename_module: isinstance(child_basename_module[1], nn.Linear), module.named_children())
    )

    if len(linear_children) == 4:  # case: three linear(q/k/v) + linear(out)
        # confirm hidden size of linears
        unique_sizes = set()
        for _, child_module in linear_children:
            unique_sizes.add(int(child_module.in_features))
            unique_sizes.add(int(child_module.out_features))
        if len(unique_sizes) == 1:
            # check if linear are named as q/k/v
            q_linear, k_linear, v_linear = {}, {}, {}
            for child_basename, child_module in linear_children:
                if "q" in child_basename.lower():
                    q_linear[child_basename] = child_module
                if "k" in child_basename.lower():
                    k_linear[child_basename] = child_module
                if "v" in child_basename.lower():
                    v_linear[child_basename] = child_module
            unique_qkv_names = set(list(q_linear.keys()) + list(k_linear.keys()) + list(v_linear.keys()))
            # the rest are output linear
            out_linear = {}
            out_names = set([name for name, _ in linear_children]) - unique_qkv_names  # noqa: C403
            for child_basename, child_module in linear_children:
                if child_basename in out_names:
                    out_linear[child_basename] = child_module
            # if named q/k/v exist
            if len(q_linear) == 1 and len(k_linear) == 1 and len(v_linear) == 1 and len(unique_qkv_names) == 3:
                assert len(out_linear) == 1
            else:  # unnamed q/k/v
                q_linear = {linear_children[0][0]: linear_children[0][1]}
                k_linear = {linear_children[1][0]: linear_children[1][1]}
                v_linear = {linear_children[2][0]: linear_children[2][1]}
                out_linear = {linear_children[3][0]: linear_children[3][1]}
            # set param plan
            for x_linear in (q_linear, k_linear, v_linear):
                child_basename, child_module = next(iter(x_linear.items()))
                param_plan[child_basename + ".weight"] = [Shard(0)]
                if child_module.bias is not None:
                    param_plan[child_basename + ".bias"] = [Shard(0)]
            child_basename, child_module = next(iter(out_linear.items()))
            param_plan[child_basename + ".weight"] = [Shard(1)]
            if child_module.bias is not None:
                param_plan[child_basename + ".bias"] = [Replicate()]
            # set fwd plan
            fwd_plan[child_basename + ".output"] = [[Replicate()]]  # TODO: auto [[Shard(-2)]] if followed by SP block
        else:
            warnings.warn(
                "Attention of MEGATRON policy encountered inequal hidden size of linears. Failback to default plan (Replicate)."
            )
    elif len(linear_children) == 2:  # case: linear(qkv) + linear(out) --> merge qkv
        warnings.warn(
            "Attention of MEGATRON policy encountered Merged-QKV. Unsupported yet. Failback to default plan (Replicate)."
        )
    else:  # unknown case: no plans
        warnings.warn("Attention of MEGATRON policy encountered unknown case. Failback to default plan (Replicate).")

    fwd_plan["input"] = [[Replicate()]]
    # fwd_plan["output"] = [[Replicate()]]  # for Partial() output only

    return param_plan, fwd_plan


@register(["LayerNorm"])
def layernorm_plan_provider(fqn: str, module: nn.Module, *, seq_dim: int = 1) -> Tuple[Dict, Dict]:
    validate_single_input(module)
    if _IS_DEBUG:
        print(f"[DEBUG] layernorm_plan_provider({fqn}, {module})")  # noqa: E701

    param_plan, fwd_plan = {}, {}
    fwd_plan["input"] = [[Shard(seq_dim)]]

    return param_plan, fwd_plan


@register(["Embedding", "Embed"])
def embedding_plan_provider(fqn: str, module: nn.Module) -> Tuple[Dict, Dict]:
    validate_single_input(module)
    if _IS_DEBUG:
        print(f"[DEBUG] embedding_plan_provider({fqn}, {module})")  # noqa: E701

    warnings.warn(
        f"{module.__class__.__name__} will be sharded along hidden size dim; "
        "Currently, DTensor doesn't support sharding along vocab size dim."
    )

    param_plan, fwd_plan = {}, {}
    fwd_plan["input"] = [[Replicate()]]
    param_plan["weight"] = [Shard(1)]  # shard on hidden size dim
    fwd_plan["output"] = [[Replicate()]]

    return param_plan, fwd_plan


@register(["Linear"])
def lm_linear_plan_provider(fqn: str, module: nn.Module) -> Tuple[Dict, Dict]:
    mod_path, _, base_name = fqn.rpartition(".")
    if "lm" not in base_name and "logit" not in base_name:
        return {}, {}

    validate_single_input(module)
    if _IS_DEBUG:
        print(f"[DEBUG] lm_linear_plan_provider({fqn}, {module})")  # noqa: E701

    param_plan, fwd_plan = {}, {}
    fwd_plan["input"] = [[Shard(-1)]]
    param_plan["weight"] = [Shard(1)]  # shard as row linear
    if getattr(module, "bias", None) is not None:
        param_plan["bias"] = [Replicate()]
    fwd_plan["output"] = [[Replicate()]]

    return param_plan, fwd_plan


@register(["Dropout", "Drop"])
def dropout_plan_provider(fqn: str, module: nn.Module) -> Tuple[Dict, Dict]:
    return {}, {}


@register(["Conv"])
def conv_plan_provider(fqn: str, module: nn.Module) -> Tuple[Dict, Dict]:
    return {}, {}
