################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

from dataclasses import dataclass
from typing import List, Tuple

from vescale.dtensor import DeviceMesh
from vescale.dtensor.op_schema import OpStrategy, PlacementStrategy
from vescale.dtensor.placement_types import DTensorSpec, InterleavedShard, Partial, Replicate, Shard


@dataclass
class EinsumDims:
    contracting_dims: List[str]
    batch_dims: List[str]
    lhs_out_only_dims: List[str]
    rhs_out_only_dims: List[str]

    @classmethod
    def parse_equation(cls, equation: str) -> Tuple[List[str], str]:
        # parse einop equation and extract arg specs
        """
        Parse the einsum equation str to input dim chars and output dim char
        """
        inputs, outputs = equation.split("->")
        input_dims, output_dims = inputs.split(","), outputs.split(",")

        # NOTE: only support at most two inputs, and single output
        # extend to support more inputs if needed in future
        assert len(input_dims) <= 2, "Only support at most two inputs"
        assert len(output_dims) == 1, "Only support single output"
        output_dim = output_dims[0]
        return input_dims, output_dim

    @classmethod
    def parse_dims(cls, input_dims: List[str], output_dim: str) -> "EinsumDims":
        """
        Parse the dims and extract the contracting, batch, and free dimensions
        for the left and right hand sides.
        """
        dim_char_set = set()
        for input_dim in input_dims:
            for input_char in list(input_dim):
                dim_char_set.add(input_char)

        # get a determinisitc order of all dim chars
        all_dim_chars = sorted(dim_char_set)

        # parse input and output dimensions
        lhs_out_only_dims, rhs_out_only_dims = [], []
        batch_dims, contracting_dims = [], []

        for dim_char in all_dim_chars:
            if dim_char not in output_dim:
                contracting_dims.append(dim_char)
            else:
                is_batch_dim = True
                for input_dim in input_dims:
                    is_batch_dim = is_batch_dim and dim_char in input_dim

                if is_batch_dim:
                    batch_dims.append(dim_char)
                else:
                    assert len(input_dims) == 2, "free dimension only supported for two inputs!"
                    lhs, rhs = input_dims
                    if dim_char in lhs:
                        lhs_out_only_dims.append(dim_char)
                    elif dim_char in rhs:
                        rhs_out_only_dims.append(dim_char)
                    else:
                        raise RuntimeError("Invalid dimension character")

        return cls(
            contracting_dims=contracting_dims,
            batch_dims=batch_dims,
            lhs_out_only_dims=lhs_out_only_dims,
            rhs_out_only_dims=rhs_out_only_dims,
        )


def gen_einsum_strategies(
    equation: str,
    mesh: DeviceMesh,
    mat1: OpStrategy,
    mat2: OpStrategy,
    *,
    linearity: bool = False,
) -> OpStrategy:
    """
    Generate a strategy list for the ops that follow einsum style notation.
    """
    # parse einop equation and extract dims
    input_dims, output_dim = EinsumDims.parse_equation(equation)
    edims = EinsumDims.parse_dims(input_dims, output_dim)

    assert isinstance(mat1, OpStrategy)
    assert isinstance(mat2, OpStrategy)

    lhs_strategy: PlacementStrategy = mat1.strategies[0]  # pick first placements
    rhs_strategy: PlacementStrategy = mat2.strategies[0]
    lhs_spec = lhs_strategy.output_spec
    rhs_spec = rhs_strategy.output_spec
    lhs_placements = lhs_spec.placements
    rhs_placements = rhs_spec.placements

    inputs, output = equation.split("->")
    lhs, rhs = inputs.split(",")
    # {"b": [(0, S), (1, IS)]}
    lhs_shard_dim_infos = {}
    rhs_shard_dim_infos = {}

    for i, p in enumerate(lhs_placements):
        if not p.is_shard():
            continue
        if lhs[p.dim] not in lhs_shard_dim_infos:
            lhs_shard_dim_infos[lhs[p.dim]] = {}
        if p.is_interleaved_shard():
            lhs_shard_dim_infos[lhs[p.dim]][i] = ("IS", p.interleaved_size)
        else:
            lhs_shard_dim_infos[lhs[p.dim]][i] = ("S", None)

    for i, p in enumerate(rhs_placements):
        if not p.is_shard():
            continue
        if rhs[p.dim] not in rhs_shard_dim_infos:
            rhs_shard_dim_infos[rhs[p.dim]] = {}
        if p.is_interleaved_shard():
            rhs_shard_dim_infos[rhs[p.dim]][i] = ("IS", p.interleaved_size)
        else:
            rhs_shard_dim_infos[rhs[p.dim]][i] = ("S", None)

    if linearity:
        lhs_spec = DTensorSpec(mesh, lhs_placements)
        rhs_spec = DTensorSpec(mesh, rhs_placements)
        out_spec = DTensorSpec(mesh, lhs_placements)
        placement = PlacementStrategy(output_spec=out_spec, input_specs=[lhs_spec, rhs_spec])
        return OpStrategy([placement])

    out_shard_dim_infos = {}
    out_reduce_dim_info = {}
    """
    Validation Check And Generate OutShardDimInfo
    """
    # 1. same batch and constrating dims
    for d in edims.batch_dims + edims.contracting_dims:
        if d not in lhs_shard_dim_infos and d not in rhs_shard_dim_infos:
            continue
        if d not in lhs_shard_dim_infos and d in rhs_shard_dim_infos:
            raise ValueError(f"found rhs sharded on {d}, but lhs not")
        if d in lhs_shard_dim_infos and d not in rhs_shard_dim_infos:
            raise ValueError(f"found lhs sharded on {d}, but rhs not")
        assert len(lhs_shard_dim_infos[d]) == len(
            rhs_shard_dim_infos[d]
        ), "lhs and rhs must be sharded on the same number of mesh dims"
        for mesh_dim in lhs_shard_dim_infos[d]:
            # assert lhs_shard_dim_infos[d][mesh_dim][0] != "P", "batch or contract dims must not be partial sharded"
            assert mesh_dim in rhs_shard_dim_infos[d], f"found lhs sharded on mesh dim @{mesh_dim}, but rhs not"
            lp = lhs_shard_dim_infos[d][mesh_dim]
            rp = rhs_shard_dim_infos[d][mesh_dim]
            assert (
                lp[0] == rp[0] and lp[1] == rp[1]
            ), f"lhs and rhs must be samely sharded on mesh dim @{mesh_dim}, found {lp} and {rp}"

            if d in edims.batch_dims:
                if d not in out_shard_dim_infos:
                    out_shard_dim_infos[d] = {}
                out_shard_dim_infos[d][mesh_dim] = lp
            else:
                out_reduce_dim_info[mesh_dim] = ("P", None)

    # 2. lhs only dims
    for d in edims.lhs_out_only_dims:
        if d not in lhs_shard_dim_infos:
            continue
        out_shard_dim_infos[d] = {}
        for mesh_dim in lhs_shard_dim_infos[d]:
            out_shard_dim_infos[d][mesh_dim] = lhs_shard_dim_infos[d][mesh_dim]

    # 3. rhs only dims
    for d in edims.rhs_out_only_dims:
        if d not in rhs_shard_dim_infos:
            continue
        out_shard_dim_infos[d] = {}
        for mesh_dim in rhs_shard_dim_infos[d]:
            out_shard_dim_infos[d][mesh_dim] = rhs_shard_dim_infos[d][mesh_dim]

    # 4. no-shard dims
    lhs_partial_mesh_dims = lhs_spec.sums
    rhs_partial_mesh_dims = rhs_spec.sums
    if lhs_partial_mesh_dims and rhs_partial_mesh_dims:
        raise ValueError("rhs and lhs can not be both partial")
    for mesh_dim in lhs_partial_mesh_dims + rhs_partial_mesh_dims:
        out_reduce_dim_info[mesh_dim] = ("P", None)

    placements = [Replicate()] * mesh.ndim
    for d in out_shard_dim_infos:
        output_tensor_dim = output.index(d)
        for mesh_dim in out_shard_dim_infos[d]:
            if out_shard_dim_infos[d][mesh_dim][0] == "S":
                placements[mesh_dim] = Shard(output_tensor_dim)
            elif out_shard_dim_infos[d][mesh_dim][0] == "IS":
                placements[mesh_dim] = InterleavedShard(output_tensor_dim, out_shard_dim_infos[d][mesh_dim][1])
            else:
                pass
    for mesh_dim in out_reduce_dim_info:
        if out_reduce_dim_info[mesh_dim][0] == "P":
            placements[mesh_dim] = Partial()
        else:
            pass

    out_spec = DTensorSpec(mesh, tuple(placements))
    placement = PlacementStrategy(output_spec=out_spec, input_specs=[lhs_spec, rhs_spec])
    return OpStrategy([placement])
