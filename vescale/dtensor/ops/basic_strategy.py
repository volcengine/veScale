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
from typing import Dict, List, Tuple

from vescale.dtensor import DeviceMesh
from vescale.dtensor.op_schema import OpStrategy, PlacementStrategy
from vescale.dtensor.placement_types import DTensorSpec, InterleavedShard, Partial, Placement, Replicate, Shard


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


"""
for any einsum option there can be conclusion in below ways
a batch matmul peration
C_b_i_j = \\sum{A_b_i_k * B_b_k_j}
mesh shape is (n0,n1)

# | parallel mapping   | outspec   | inputspec       | cost
1 |   i->0, j->1       | RS^0S^1   | RS^0R,RRS^1     | 0
2 |   i->0, k->1       | RS^0R     | RS^0S^1,RS^1R   | all-reduce(M/n0, 1)
3 |   j->0, k->1       | RRS^0     | RRS^1,RS^1S^0   | all-reduce(M/n0, 1)
4 |   b->0, i->1       | S^0S^1R   | S^0S^1R,S^0RR   | 0
5 |   b->0, k->1       | S^0RR     | S^0RS^1,S^0S^1R | all-reduce(M/n0, 1)
6 |   i->{0, 1}        | RS^01R    | RS^01R,RRR      | 0
7 |   k->{0, 1}        | RRR       | RRS^01,RS^01R   | all-reduce(M, {0, 1})
"""


def deduce_out_mode(
    lhs_mode: int,
    rhs_mode: int,
    edims: EinsumDims,
    lhs_mesh_dims_map: Dict[str, int],
    rhs_mesh_dims_map: Dict[str, int],
    lhs_interleaved_shard_dims: Dict[str, int],
    rhs_interleaved_shard_dims: Dict[str, int],
):
    split_batch = lhs_mode & 4 or rhs_mode & 4
    split_concat = lhs_mode & 1
    lhs_split_spartial = lhs_mode & 2
    rhs_split_spartial = rhs_mode & 2

    out_mode = 0
    out_mesh_dim_mapping = {}
    out_interleaved_shard_dims = {}
    reshard_cost = None
    if split_batch:
        out_mode |= 4
        for batch_dim in edims.batch_dims:
            if batch_dim not in out_mesh_dim_mapping:
                # batch dim is not sharded in lhs and rhs
                if batch_dim not in lhs_mesh_dims_map and batch_dim not in rhs_mesh_dims_map:
                    continue
                # make sure sharding information of batch_dim stays consistent between lhs and rhs
                if (batch_dim not in lhs_mesh_dims_map) or (batch_dim not in rhs_mesh_dims_map):
                    raise ValueError("batch dim must be sharded in both lhs and rhs")
                assert (
                    lhs_mesh_dims_map[batch_dim] == rhs_mesh_dims_map[batch_dim]
                ), f"batch dim sharding information inconsistent, {lhs_mesh_dims_map[batch_dim]} vs {rhs_mesh_dims_map[batch_dim]}"
                if (batch_dim in lhs_interleaved_shard_dims and batch_dim not in rhs_interleaved_shard_dims) or (
                    batch_dim in rhs_interleaved_shard_dims and batch_dim not in lhs_interleaved_shard_dims
                ):
                    raise ValueError("batch dim sharding information inconsistent, found InterleavedShard and Shard")
                if batch_dim in lhs_interleaved_shard_dims:
                    assert (
                        lhs_interleaved_shard_dims[batch_dim] == rhs_interleaved_shard_dims[batch_dim]
                    ), f"batch dim sharding information inconsistent, found InterleavedShard({lhs_interleaved_shard_dims[batch_dim]}) vs Interleaved shard({rhs_interleaved_shard_dims[batch_dim]})"
                out_mesh_dim_mapping[batch_dim] = lhs_mesh_dims_map[batch_dim]
                if batch_dim in lhs_interleaved_shard_dims:
                    out_interleaved_shard_dims[batch_dim] = lhs_interleaved_shard_dims[batch_dim]

    if split_concat:
        # output will be partial
        reduce_mappings = {}
        for reduce_dim in edims.contracting_dims:
            assert len(lhs_mesh_dims_map[reduce_dim]) == len(
                rhs_mesh_dims_map[reduce_dim]
            ), "reduce dim in different mesh is not allowed"
            if reduce_dim in lhs_interleaved_shard_dims and reduce_dim in rhs_interleaved_shard_dims:
                assert (
                    lhs_interleaved_shard_dims[reduce_dim] == rhs_interleaved_shard_dims[reduce_dim]
                ), "reduce dim should be interleaved sharded of same interleaved size"
            else:
                assert (
                    reduce_dim not in lhs_interleaved_shard_dims and reduce_dim not in rhs_interleaved_shard_dims
                ), "one reduce dim is interleaved sharded, but the other not"
            reduce_mappings[reduce_dim] = lhs_mesh_dims_map[reduce_dim]
        reshard_cost = reduce_mappings

    if lhs_split_spartial:
        for d in edims.lhs_out_only_dims:
            if d in lhs_interleaved_shard_dims:
                out_interleaved_shard_dims[d] = lhs_interleaved_shard_dims[d]
        out_mode |= 2
    if rhs_split_spartial:
        for d in edims.rhs_out_only_dims:
            if d in rhs_interleaved_shard_dims:
                out_interleaved_shard_dims[d] = rhs_interleaved_shard_dims[d]
        out_mode |= 1
    return out_mode, out_interleaved_shard_dims, reshard_cost


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
    # bitset mode to represent split
    lhs_shard_dims = [lhs[shard.dim] for shard in lhs_placements if shard.is_shard() or shard.is_interleaved_shard()]
    rhs_shard_dims = [rhs[shard.dim] for shard in rhs_placements if shard.is_shard() or shard.is_interleaved_shard()]

    def generate_interleaved_shard_dims(dims, placements):
        interleaved_shard_dims = {}
        for p in placements:
            if not p.is_interleaved_shard() and not p.is_shard():
                continue

            input_dim = dims[p.dim]
            if input_dim not in interleaved_shard_dims:
                if p.is_interleaved_shard():
                    interleaved_shard_dims[input_dim] = p.interleaved_size
            else:
                raise ValueError(
                    "vescale doesn't support mulitiple shard (and one of them is InterleavedShard) of one input dim"
                )
        return interleaved_shard_dims

    lhs_interleaved_shard_dims = generate_interleaved_shard_dims(lhs, lhs_placements)
    rhs_interleaved_shard_dims = generate_interleaved_shard_dims(rhs, rhs_placements)

    if linearity:
        lhs_spec = DTensorSpec(mesh, lhs_placements)
        rhs_spec = DTensorSpec(mesh, rhs_placements)
        out_spec = DTensorSpec(mesh, lhs_placements)
        placement = PlacementStrategy(output_spec=out_spec, input_specs=[lhs_spec, rhs_spec])
        return OpStrategy([placement])

    def construct_tensor_dim_to_mesh_dim(placements, dims):
        maps = {}
        for idx, placement in enumerate(placements):
            if placement.is_shard() or placement.is_interleaved_shard():
                char = dims[placement.dim]
                if char not in maps:
                    maps[char] = set()
                maps[char].add(idx)
        return maps

    def deduce_sharding_mode(shard_dim):
        mode = 0
        for dim in shard_dim:
            if dim in edims.batch_dims:
                mode |= 1 << 2  # SRR
            if dim in edims.contracting_dims:
                mode |= 1  # RRS
            else:
                mode |= 1 << 1  # RSR
        return mode

    lhs_sharding_map = construct_tensor_dim_to_mesh_dim(lhs_placements, lhs)
    rhs_sharding_map = construct_tensor_dim_to_mesh_dim(rhs_placements, rhs)
    lhs_shard_mode = deduce_sharding_mode(lhs_shard_dims)
    rhs_shard_mode = deduce_sharding_mode(rhs_shard_dims)
    out_mode, out_interleaved_shard_dims, reshard_cost = deduce_out_mode(
        lhs_shard_mode,
        rhs_shard_mode,
        edims,
        lhs_sharding_map,
        rhs_sharding_map,
        lhs_interleaved_shard_dims,
        rhs_interleaved_shard_dims,
    )

    # not split batch
    # RS * SR , SS * SR, RS * SS
    placements = [Replicate()] * mesh.ndim
    if out_mode & 4:
        for dim in edims.batch_dims:
            if dim in lhs_sharding_map or dim in rhs_sharding_map:
                # lhs_sharding_map[dim] = rhs_sharding_map[dim] here.
                # it's guaranteed in function `deduce_out_mode`.
                for mesh_dim in lhs_sharding_map[dim]:
                    if dim not in out_interleaved_shard_dims:
                        placements[mesh_dim] = Shard(output.index(dim))
                    else:
                        placements[mesh_dim] = InterleavedShard(
                            output.index(dim), interleaved_size=out_interleaved_shard_dims[dim]
                        )

    def generate_placement(placements, dim_maps: Dict, type: Placement):
        mesh_dims = []
        for dim in dim_maps:
            mesh_dim = dim_maps[dim]
            mesh_dims.extend(list(mesh_dim))
        for dim in mesh_dims:
            placements[dim] = type
        return placements

    if reshard_cost is not None:
        placements = generate_placement(placements, reshard_cost, Partial())

    for dim in edims.lhs_out_only_dims:
        if dim in lhs_sharding_map:
            if dim in lhs_interleaved_shard_dims:
                placements = generate_placement(
                    placements,
                    {dim: lhs_sharding_map[dim]},
                    InterleavedShard(output.index(dim), interleaved_size=out_interleaved_shard_dims[dim]),
                )
            else:
                placements = generate_placement(placements, {dim: lhs_sharding_map[dim]}, Shard(output.index(dim)))

    for dim in edims.rhs_out_only_dims:
        if dim in rhs_sharding_map:
            if dim in rhs_interleaved_shard_dims:
                placements = generate_placement(
                    placements,
                    {dim: rhs_sharding_map[dim]},
                    InterleavedShard(output.index(dim), interleaved_size=out_interleaved_shard_dims[dim]),
                )
            else:
                placements = generate_placement(placements, {dim: rhs_sharding_map[dim]}, Shard(output.index(dim)))

    assert (lhs_shard_mode & 4) == (rhs_shard_mode & 4), "vescale only support both split batch dim"
    assert (lhs_shard_mode & 1) == (rhs_shard_mode & 1), "vescale only support both split concat dim"
    out_spec = DTensorSpec(mesh, tuple(placements))
    placement = PlacementStrategy(output_spec=out_spec, input_specs=[lhs_spec, rhs_spec])
    return OpStrategy([placement])
