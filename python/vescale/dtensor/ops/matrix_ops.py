################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

from typing import cast, Tuple, Sequence, List, Dict
import itertools

import torch

from vescale.dtensor.device_mesh import DeviceMesh
from vescale.dtensor.op_schema import (
    OpSchema,
    OutputSharding,
    OpStrategy,
    PlacementStrategy,
)
from vescale.dtensor.ops.basic_strategy import gen_einsum_strategies
from vescale.dtensor.ops.utils import (
    generate_redistribute_costs,
    infer_broadcast_dims_map,
    is_tensor_shardable,
    map_placements_after_broadcast,
    register_op_strategy,
    register_prop_rule,
)
from vescale.dtensor.placement_types import (
    DTensorSpec,
    Placement,
    Partial,
    Replicate,
    Shard,
    InterleavedShard,
    TensorMeta,
)

aten = torch.ops.aten


@register_prop_rule(aten.t.default)
def transpose_rule(op_schema: OpSchema) -> OutputSharding:
    # rule-based op sharding propagation will be deprecated. We only handle
    # aten.t.default here only.
    input_spec = op_schema.args_spec[0]
    out_placements = []
    for p in input_spec.placements:
        if p.is_interleaved_shard():
            p = cast(InterleavedShard, p)
            out_placements.append(InterleavedShard(1 - p.dim, p.interleaved_size))
        elif p.is_shard():
            p = cast(Shard, p)
            out_placements.append(Shard(1 - p.dim))
        else:
            out_placements.append(p)

    out_tensor_meta = None
    if input_spec.tensor_meta is not None:
        out_shape = torch.Size([input_spec.tensor_meta.shape[-1], input_spec.tensor_meta.shape[0]])
        out_stride = (input_spec.tensor_meta.stride[-1], input_spec.tensor_meta.stride[0])
        out_dtype = input_spec.tensor_meta.dtype
        out_tensor_meta = TensorMeta(out_shape, out_stride, out_dtype)

    return OutputSharding(
        output_spec=DTensorSpec(input_spec.mesh, out_placements, out_tensor_meta),
        schema_suggestions=None,
        failed_reason=None,
        needs_redistribute=False,
    )


def _mm_like_strategy(mm_equation: str, mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    # generate all possible strategies for mm
    lhs, rhs = op_schema.args_schema
    assert isinstance(lhs, OpStrategy)
    assert isinstance(rhs, OpStrategy)
    mm_strategy = gen_einsum_strategies(mm_equation, mesh, lhs, rhs)
    # filter out invalid strategies and associate costs
    # TODO(cery.zhai) add check here
    return mm_strategy


def _addmm_like_strategy(mm_equation: str, mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    self_strategy, mat1_strategy, mat2_strategy = op_schema.args_schema
    assert isinstance(self_strategy, OpStrategy)
    assert isinstance(mat1_strategy, OpStrategy)
    assert isinstance(mat2_strategy, OpStrategy)
    self_shape = self_strategy.output_shape
    mm_out_shape = torch.Size(
        [
            mat2_strategy.output_shape[-1] if i == len(mat1_strategy.output_shape) - 1 else dim_size
            for i, dim_size in enumerate(mat1_strategy.output_shape)
        ]
    )
    # generate all possible strategies for mm
    mm_strategy = gen_einsum_strategies(mm_equation, mesh, mat1_strategy, mat2_strategy)
    # filter out invalid strategies and associate costs
    strategies = mm_strategy.strategies
    strtg = strategies[0]
    # construct new strategy by consider the self arg
    assert strtg.input_specs is not None
    mat1_spec = strtg.input_specs[0]
    mat2_spec = strtg.input_specs[1]
    out_spec = strtg.output_spec

    # self arg's spec should follow the output of mm, but need
    # to consider broadcast for the self arg
    broadcast_dims_map = infer_broadcast_dims_map(mm_out_shape, self_shape)
    self_placements = map_placements_after_broadcast(out_spec.placements, mm_out_shape, broadcast_dims_map)
    self_spec = DTensorSpec(mesh=mesh, placements=self_placements)

    if is_tensor_shardable(mat1_strategy.output_shape, mat1_spec) and is_tensor_shardable(
        mat2_strategy.output_shape, mat2_spec
    ):
        # update input specs with new self spec
        strtg.input_specs = (self_spec, mat1_spec, mat2_spec)

        # associate costs
        redistribute_cost = [
            generate_redistribute_costs(self_strategy, self_spec),
            # generate_redistribute_costs(mat1_strategy, mat1_spec), # we do not support reshard by annotation
            # generate_redistribute_costs(mat2_strategy, mat2_spec),
        ]
        strtg.redistribute_cost = redistribute_cost
    mm_strategy.strategies = [strtg]
    return mm_strategy


@register_op_strategy(aten.mm.default)
def mm_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    return _mm_like_strategy("mk,kn->mn", mesh, op_schema)


@register_op_strategy(aten.addmm.default)
def addmm_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    return _addmm_like_strategy("mk,kn->mn", mesh, op_schema)


@register_op_strategy(aten.bmm.default)
def bmm_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    return _mm_like_strategy("bmk,bkn->bmn", mesh, op_schema)


@register_op_strategy(aten.baddbmm.default)
def baddmm_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    return _addmm_like_strategy("bmk,bkn->bmn", mesh, op_schema)


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
    q = op_schema.args_schema[0]
    k = op_schema.args_schema[1]
    v = op_schema.args_schema[2]
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
