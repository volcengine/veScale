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

from typing import Tuple, Sequence, List, Dict
import itertools

import torch

from vescale.dtensor.device_mesh import DeviceMesh
from vescale.dtensor.op_schema import (
    OpSchema,
    OutputSharding,
    OpStrategy,
    PlacementStrategy,
)

from vescale.dtensor.ops.utils import (
    register_prop_rule,
    register_op_strategy,
)
from vescale.dtensor.placement_types import (
    DTensorSpec,
    Placement,
    Partial,
    Replicate,
    Shard,
    InterleavedShard,
)


aten = torch.ops.aten


def keeped_only_one_dim_shard(
    input_arg_specs: List[DTensorSpec],
    input_keeped_dim_idxs: List[int],
    output_keeped_dim_idxs: List[int],
) -> List[Tuple[Placement]]:
    assert len(input_arg_specs) == len(input_keeped_dim_idxs)
    device_mesh = input_arg_specs[0].mesh

    def check_input_shard(spec: DTensorSpec, dim_idx: int, mesh_sharding_info: Dict[int, Placement] = None):
        expected_sharded_mesh_idxs = set(mesh_sharding_info.keys()) if mesh_sharding_info else set()
        for i, p in enumerate(spec.placements):
            assert not p.is_partial(), "Input tensor are not allowed to be partial sharded"
            if p.is_replicate():
                continue
            sharded_dim = p.dim
            assert (
                sharded_dim == dim_idx
            ), f"input tensor is expected to be sharded at {dim_idx} dim, but got {sharded_dim}"
            if expected_sharded_mesh_idxs:
                assert (
                    i in expected_sharded_mesh_idxs
                ), f"input tensor is expected to be sharded on mesh dims {str(expected_sharded_mesh_idxs)}, but found {i}"
                if p.is_interleaved_shard() and mesh_sharding_info[i].is_interleaved_shard():
                    assert (
                        p.interleaved_size == mesh_sharding_info[i].interleaved_size
                    ), f"input tensor is expected to be interleaved sharded with interleaved_size {mesh_sharding_info[i].interleaved_size} on mesh dim {i}, but found {p.interleaved_size}"
                elif p.is_shard() and mesh_sharding_info[i].is_shard():
                    pass
                else:
                    raise AssertionError(
                        f"input tensor is expected to be sharded as {str(mesh_sharding_info[i])} on mesh dim {i}, but found {str(p)}"
                    )
                expected_sharded_mesh_idxs.remove(i)
            else:
                mesh_sharding_info[i] = p

        assert (
            len(expected_sharded_mesh_idxs) == 0
        ), f"input tensor is expected to be sharded on mesh dims {str(expected_sharded_mesh_idxs)}, but found not"
        return mesh_sharding_info

    input_mesh_sharding_info = {}
    for spec, dim_idx in zip(input_arg_specs, input_keeped_dim_idxs):
        input_mesh_sharding_info = check_input_shard(spec, dim_idx, input_mesh_sharding_info)

    output_placements = []
    for output_dim_idx in output_keeped_dim_idxs:
        if output_dim_idx is None:
            output_placements.append(None)
            continue
        placements = [Replicate()] * device_mesh.ndim
        for mesh_idx in input_mesh_sharding_info:
            if input_mesh_sharding_info[mesh_idx].is_interleaved_shard():
                placements[mesh_idx] = InterleavedShard(
                    output_dim_idx,
                    interleaved_size=input_mesh_sharding_info[mesh_idx].interleaved_size,
                )
            elif input_mesh_sharding_info[mesh_idx].is_shard():
                placements[mesh_idx] = Shard(output_dim_idx)
            else:
                raise ValueError(
                    f"Invalid placement other than (Interleaved)Shard found, it's {str(input_mesh_sharding_info[mesh_idx])}"
                )
        output_placements.append(tuple(placements))

    return output_placements


def check_spec_contains_parital(spec: DTensorSpec):
    for p in spec.placements:
        if isinstance(p, Partial):
            return True
    return False


def check_spec_all_replicate(spec: DTensorSpec):
    for p in spec.placements:
        if not isinstance(p, Replicate):
            return False
    return True


def check_spec_replicate_at_input_dims(spec: DTensorSpec, input_dims: Sequence[int]):
    for p in spec.placements:
        if p.is_shard() or p.is_interleaved_shard():
            if p.dim in input_dims:
                return False
    return True


def check_spec_sharding_info_same_at_input_dims(specs: Sequence[DTensorSpec], input_dims: Sequence[int]):
    assert len(specs) > 0
    mesh = specs[0].mesh
    sharding_info = {}
    for i, p in enumerate(specs[0].placements):
        if p.is_shard() or p.is_interleaved_shard():
            if p.dim not in input_dims:
                continue
            sinfo = (p.dim, "S", None) if p.is_shard() else (p.dim, "IS", p.interleaved_size)
            sharding_info[i] = sinfo
    for i in range(mesh.ndim):
        for j in range(1, len(specs)):
            spec = specs[j]
            p = spec.placements[i]
            if p.is_replicate() or p.is_partial():
                assert i not in sharding_info, f"sharding info at mesh dim {i} is not consistent, IS/S against R/P"
            else:
                if p.dim not in input_dims:
                    continue
                assert i in sharding_info, f"sharding info at mesh dim {i} is not consistent, None against S/IS"
                sinfo = sharding_info[i]
                assert (
                    sinfo[0] == p.dim
                ), f"sharding info at mesh dim {i} is not consistent found input_dim {sinfo[0]} against {p.dim}"
                if p.is_interleaved_shard():
                    assert (
                        sinfo[1] == "IS" and sinfo[2] == p.interleaved_size
                    ), f"sharding info at mesh dim {i} is not consistent, {sinfo[1]}({sinfo[2]}) against IS({p.interleaved_size})"
                else:
                    assert (
                        sinfo[1] == "S"
                    ), f"sharding info at mesh dim {i} is not consistent, {sinfo[1]}({sinfo[2]}) against S(1)"
    return sharding_info


@register_prop_rule(aten._scaled_dot_product_flash_attention.default)
def _prop__scaled_dot_product_flash_attention(op_schema: OpSchema) -> OutputSharding:
    (q, k, v) = op_schema.args_schema
    return_debug_mask = op_schema.kwargs_schema.get("return_debug_mask", False)
    assert isinstance(q, DTensorSpec) and isinstance(k, DTensorSpec) and isinstance(v, DTensorSpec)
    mesh = q.mesh
    base_rank = len(q.tensor_meta.shape)

    if any(check_spec_contains_parital(x) for x in [q, k, v]):
        raise AssertionError("q, k, v are not allowed to be partial sharded")
    if not all(check_spec_replicate_at_input_dims(x, range(base_rank - 2, base_rank)) for x in [q, k, v]):
        raise AssertionError("q, k, v must be replicate at last two dims")

    sharding_info = check_spec_sharding_info_same_at_input_dims([q, k, v], range(0, base_rank - 2))

    attention_spec = DTensorSpec(
        mesh=mesh,
        placements=q.placements,
    )
    logsumexp_spec_placements = [Replicate()] * mesh.ndim
    for i in sharding_info:
        if sharding_info[i][1] == "IS":
            logsumexp_spec_placements[i] = InterleavedShard(
                dim=sharding_info[i][0], interleaved_size=sharding_info[i][2]
            )
        else:
            logsumexp_spec_placements[i] = Shard(dim=sharding_info[i][0])
    logsumexp_spec = DTensorSpec(mesh=mesh, placements=tuple(logsumexp_spec_placements))
    cum_seq_q_spec = DTensorSpec(mesh=mesh, placements=tuple([Replicate()] * mesh.ndim))
    cum_seq_k_spec = DTensorSpec(mesh=mesh, placements=tuple([Replicate()] * mesh.ndim))
    max_seqlen_batch_q_spec = None
    max_seqlen_batch_k_spec = None
    philox_seed_spec = DTensorSpec(mesh=mesh, placements=tuple([Replicate()] * mesh.ndim))
    philox_offset_spec = DTensorSpec(mesh=mesh, placements=tuple([Replicate()] * mesh.ndim))

    debug_attn_mask_spec = DTensorSpec(mesh=mesh, placements=q.placements) if return_debug_mask else None
    return OutputSharding(
        output_spec=(
            attention_spec,
            logsumexp_spec,
            cum_seq_q_spec,
            cum_seq_k_spec,
            max_seqlen_batch_q_spec,
            max_seqlen_batch_k_spec,
            philox_seed_spec,
            philox_offset_spec,
            debug_attn_mask_spec,
        ),
    )


@register_prop_rule(aten._scaled_dot_product_flash_attention_backward.default)
def _prop__scaled_dot_product_flash_attention_backward(
    op_schema: OpSchema,
) -> OutputSharding:
    # raise NotImplementedError()
    grad_out = op_schema.args_schema[0]
    q = op_schema.args_schema[1]
    k = op_schema.args_schema[2]
    v = op_schema.args_schema[3]
    out = op_schema.args_schema[4]
    logsumexp = op_schema.args_schema[5]
    philox_seed = op_schema.args_schema[-2]
    philox_offset = op_schema.args_schema[-1]

    assert (
        isinstance(grad_out, DTensorSpec)
        and isinstance(q, DTensorSpec)
        and isinstance(k, DTensorSpec)
        and isinstance(v, DTensorSpec)
        and isinstance(out, DTensorSpec)
        and isinstance(logsumexp, DTensorSpec)
        and isinstance(philox_seed, DTensorSpec)
        and isinstance(philox_offset, DTensorSpec)
    )

    mesh = grad_out.mesh
    assert (
        mesh == q.mesh == k.mesh == v.mesh == out.mesh == logsumexp.mesh == philox_seed.mesh == philox_offset.mesh
    ), "grad_out, q, k, v, out, logsumexp, philox_seed, philox_offset must be on the same mesh"

    if any(
        check_spec_contains_parital(x)
        for x in [
            grad_out,
            q,
            k,
            v,
            out,
            logsumexp,
            philox_seed,
            philox_offset,
        ]
    ):
        raise AssertionError("input of scaled_dot_product_flash_attention_backward must not be partial")
    if not all(check_spec_all_replicate(x) for x in [philox_seed, philox_offset]):
        raise AssertionError("philox_seed, philox_offset must must be replicate")
    base_rank = len(grad_out.tensor_meta.shape)
    if not all(
        check_spec_replicate_at_input_dims(x, range(base_rank - 2, base_rank)) for x in [grad_out, q, k, v, out]
    ):
        raise AssertionError("grad_out, q, k, v, out must be replicate at last two dims")
    assert check_spec_replicate_at_input_dims(logsumexp, [base_rank - 2]), "logsumexp must be replicate at last dim"
    check_spec_sharding_info_same_at_input_dims([grad_out, q, k, v, out], range(base_rank - 2))
    return OutputSharding(output_spec=(q, k, v))


@register_op_strategy(aten._scaled_dot_product_efficient_attention.default)
def __mem_efficient_attention_fwd_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    attention_strategy = OpStrategy([])
    q = op_schema.args_schema[0]
    k = op_schema.args_schema[1]
    v = op_schema.args_schema[2]

    ndim = len(q.strategies[0].output_spec.tensor_meta.shape)

    allowed_placements = (Replicate(), *(Shard(i) for i in range(ndim - 2)))
    is_legal_placements = lambda ps: all(p in allowed_placements for p in ps)

    for qs, ks, vs in itertools.product(q.strategies, k.strategies, v.strategies):
        q_spec = qs.output_spec
        k_spec = ks.output_spec
        v_spec = vs.output_spec
        if is_legal_placements(q_spec.placements):
            in_placements = tuple(q_spec.placements)
            out_placements = tuple(q_spec.placements)
        elif is_legal_placements(k_spec.placements):
            in_placements = tuple(k_spec.placements)
            out_placements = tuple(k_spec.placements)
        elif is_legal_placements(v_spec.placements):
            in_placements = tuple(v_spec.placements)
            out_placements = tuple(v_spec.placements)
        else:
            in_placements = (Replicate(),)
            out_placements = (Replicate(),)

        input_specs = []
        for input_arg in op_schema.args_schema:
            if isinstance(input_arg, OpStrategy):
                input_arg_spec = input_arg.strategies[0].output_spec
                input_arg_target_spec = DTensorSpec(
                    mesh=mesh,
                    placements=in_placements,
                    tensor_meta=input_arg_spec.tensor_meta,
                )
                input_specs.append(input_arg_target_spec)

        attention_strategy.strategies.append(
            PlacementStrategy(
                output_spec=DTensorSpec(
                    mesh=mesh,
                    placements=out_placements,
                ),
                input_specs=input_specs,
                redistribute_cost=None,
            )
        )
    return attention_strategy


@register_op_strategy(aten._scaled_dot_product_efficient_attention_backward.default)
def __mem_efficient_attention_bwd_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    attention_bwd_strategy = OpStrategy([])
    grad = op_schema.args_schema[0]

    for gs in grad.strategies:
        grad_spec = gs.output_spec
        out_placements: Tuple[Placement] = tuple(p for p in grad_spec.placements)

        input_specs = []
        for input_arg in op_schema.args_schema:
            if isinstance(input_arg, OpStrategy):
                input_arg_spec = input_arg.strategies[0].output_spec
                input_arg_target_spec = DTensorSpec(
                    mesh=mesh,
                    placements=input_arg_spec.placements,
                    tensor_meta=input_arg_spec.tensor_meta,
                )
                input_specs.append(input_arg_target_spec)

        attention_bwd_strategy.strategies.append(
            PlacementStrategy(
                output_spec=DTensorSpec(
                    mesh=mesh,
                    placements=out_placements,
                ),
                input_specs=input_specs,
                redistribute_cost=None,
            )
        )
    return attention_bwd_strategy
