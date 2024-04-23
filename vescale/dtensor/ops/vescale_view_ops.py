################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

"""
This shape propagation is specific for view-like ops when
InterleavedShard and Shard have inconsistent behavior.
"""

from typing import Callable, Optional, Sequence, Set, Tuple, cast

import torch
from torch import Tensor

from vescale.dtensor._utils import compute_local_shape
from vescale.dtensor.op_schema import OpSchema, OutputSharding, RuntimeSchemaInfo
from vescale.dtensor.ops.utils import prod, register_prop_rule
from vescale.dtensor.ops.view_ops import DimMap, DimSpec, Flatten, InputDim, Op, Shape, Singleton, Split, ops
from vescale.dtensor.placement_types import DTensorSpec, InterleavedShard, Placement, Replicate, Shard, TensorMeta

TORCH_VERSION_BIGGER_THAN_2_2 = torch.__version__ >= "2.2"
aten = torch.ops.aten

""" There are only three cases for aten.view and aten.reshape op:
[2, 3, 4], [2, 12] -> (
    InputDim(0),
    Flatten((InputDim(1), InputDim(2)))
)
[6, 4] -> [2, 3, 4] (
    Split(InputDim(0), (2, 3), 0),
    Split(InputDim(0), (2, 3), 1),
    InputDim(1)
)
[2, 3], [3, 2] -> (
    Split(Flatten((InputDim(0), InputDim(1))), (3, 2), 0),
    Split(Flatten((InputDim(0), InputDim(1))), (3, 2), 1),
)
"""


def propagate_shape_and_sharding(
    in_shard: Sequence[Placement],
    local_in_shape: Shape,
    rule: DimMap,
    mesh_sizes: Shape,
) -> Tuple[Shape, Optional[Sequence[Placement]], torch.Tensor]:
    assert len(in_shard) == len(mesh_sizes), "Input Shard and mesh dimensions do not match"
    shard_map_from_input_dim_to_mesh_dim = {}
    for i, placement in enumerate(in_shard):
        if isinstance(placement, Shard):
            input_dim = placement.dim
            if input_dim not in shard_map_from_input_dim_to_mesh_dim:
                shard_map_from_input_dim_to_mesh_dim[input_dim] = []
            shard_map_from_input_dim_to_mesh_dim[placement.dim].append(i)

    shardable_dims: torch.Tensor = torch.ones((len(local_in_shape), len(mesh_sizes)), dtype=torch.bool)
    needs_reshard = False

    # in case an input dimension disappears (e.g. collapsing, reduction)
    # we cannot shard in that dimension (we need a replication fall-back rule)

    seen_input_dims: Set[int] = set()

    def collect_used_inputs(cmd: DimSpec) -> None:
        if isinstance(cmd, InputDim):
            seen_input_dims.add(cmd.input_dim)
        for inp in cmd.inputs():
            collect_used_inputs(inp)

    for cmd in rule:
        collect_used_inputs(cmd)
    for dim in range(len(local_in_shape)):
        shardable_dims[dim, :] = dim in seen_input_dims

    new_placements = [Replicate()] * len(mesh_sizes)
    out_shape = []
    for out_dim, cmd in enumerate(rule):
        if isinstance(cmd, InputDim):
            out_shape.append(local_in_shape[cmd.input_dim])
            if cmd.input_dim in shard_map_from_input_dim_to_mesh_dim:
                for mesh_dim in shard_map_from_input_dim_to_mesh_dim[cmd.input_dim]:
                    new_placements[mesh_dim] = Shard(out_dim)
        elif isinstance(cmd, Flatten):
            assert all(isinstance(d, InputDim) for d in cmd.input_dims)
            out_shape.append(prod(local_in_shape[d.input_dim] for d in cmd.input_dims))
            sharded_input_dims = [
                d.input_dim for d in cmd.input_dims if d.input_dim in shard_map_from_input_dim_to_mesh_dim
            ]
            for d in sharded_input_dims:
                shardable_dims[d, :] = False

            if len(sharded_input_dims) > 1:
                needs_reshard = True

            elif len(sharded_input_dims) == 1:
                shardable_dims[sharded_input_dims[0], :] = True
                # shard on the first of collapsed input dimensions
                if sharded_input_dims[0] == cmd.input_dims[0].input_dim:
                    for mesh_dim in shard_map_from_input_dim_to_mesh_dim[sharded_input_dims[0]]:
                        new_placements[mesh_dim] = Shard(out_dim)
                # interleaved shard on not the first of collapsed input dimensions
                else:
                    assert (
                        len(shard_map_from_input_dim_to_mesh_dim[sharded_input_dims[0]]) == 1
                    ), "We now only support interleaved sharding on a single mesh dimension"
                    mesh_dim = shard_map_from_input_dim_to_mesh_dim[sharded_input_dims[0]][0]
                    interleaved_size = 1
                    for id in cmd.input_dims:
                        if id.input_dim == sharded_input_dims[0]:
                            break
                        else:
                            interleaved_size *= local_in_shape[id.input_dim]
                    new_placements[mesh_dim] = InterleavedShard(out_dim, interleaved_size)

            # none of collapsed input dims is sharded. Do nothing.
        elif isinstance(cmd, Split):
            out_dim_size = cmd.group_shape[cmd.split_id]
            out_shape.append(out_dim_size)
            if isinstance(cmd.input_dim, InputDim):
                input_dim = cmd.input_dim.input_dim
                # the corresponding input dim is sharded
                if input_dim in shard_map_from_input_dim_to_mesh_dim:
                    if cmd.split_id == 0:
                        for mesh_dim, mesh_dim_size in enumerate(mesh_sizes):
                            shardable_dims[input_dim, mesh_dim] = out_dim_size % mesh_dim_size == 0
                        submesh_size = 1
                        for size, shard in zip(mesh_sizes, in_shard):
                            if isinstance(shard, Shard) and shard.dim == input_dim:
                                submesh_size *= size
                        assert (
                            out_dim_size % submesh_size == 0
                        ), f"Resulting dimension size {out_dim_size} is not divisible by its mesh dimension {submesh_size}."

                        for mesh_dim in shard_map_from_input_dim_to_mesh_dim[input_dim]:
                            new_placements[mesh_dim] = Shard(out_dim)

            elif isinstance(cmd.input_dim, Flatten):
                flatten = cast(Flatten, cmd.input_dim)
                assert all(isinstance(d, InputDim) for d in flatten.input_dims)

                sharded_input_dims = [
                    d.input_dim for d in flatten.input_dims if d.input_dim in shard_map_from_input_dim_to_mesh_dim
                ]
                for d in sharded_input_dims:
                    shardable_dims[d, :] = False

                if len(sharded_input_dims) > 1:
                    needs_reshard = True

                elif len(sharded_input_dims) == 1:
                    only_sharded_input_dim = sharded_input_dims[0]
                    shardable_dims[only_sharded_input_dim, :] = True
                    # shard on the first input dimension
                    if only_sharded_input_dim == flatten.input_dims[0].input_dim:
                        if cmd.split_id == 0:
                            for mesh_dim, mesh_dim_size in enumerate(mesh_sizes):
                                shardable_dims[only_sharded_input_dim, mesh_dim] = out_dim_size % mesh_dim_size == 0
                            submesh_size = 1
                            for size, shard in zip(mesh_sizes, in_shard):
                                if isinstance(shard, Shard) and shard.dim == only_sharded_input_dim:
                                    submesh_size *= size
                            assert (
                                out_dim_size % submesh_size == 0
                            ), f"Resulting dimension size {out_dim_size} is not divisible by its mesh dimension {submesh_size}."

                            for mesh_dim in shard_map_from_input_dim_to_mesh_dim[only_sharded_input_dim]:
                                new_placements[mesh_dim] = Shard(out_dim)
                    # interleaved shard on not the first input dimension
                    else:
                        # get interleaved_size
                        interleaved_size = 1
                        for id in flatten.input_dims:
                            if id.input_dim == only_sharded_input_dim:
                                break
                            else:
                                interleaved_size *= local_in_shape[id.input_dim]
                        sharded_dim_size = local_in_shape[only_sharded_input_dim]
                        prev_size = 1
                        for prev_split_id in range(cmd.split_id):
                            prev_size *= cmd.group_shape[prev_split_id]

                        if interleaved_size > prev_size * out_dim_size:
                            continue
                        if interleaved_size * sharded_dim_size < prev_size:
                            continue
                        if interleaved_size * sharded_dim_size <= prev_size * out_dim_size:
                            if interleaved_size % prev_size != 0:
                                needs_reshard = True
                                continue
                            assert (
                                len(shard_map_from_input_dim_to_mesh_dim[only_sharded_input_dim]) == 1
                            ), "Interleaved sharding only supports one dimension being sharded."
                            for mesh_dim, mesh_dim_size in enumerate(mesh_sizes):
                                shardable_dims[only_sharded_input_dim, mesh_dim] = out_dim_size % mesh_dim_size == 0
                            new_placements[shard_map_from_input_dim_to_mesh_dim[only_sharded_input_dim][0]] = (
                                InterleavedShard(out_dim, interleaved_size // prev_size)
                            )
                        else:
                            needs_reshard = True
            else:
                raise RuntimeError("Unkown input dim for Split.")
        elif isinstance(cmd, Singleton):
            out_shape.append(1)
        else:
            raise RuntimeError("Unknown command in prop rule")

    for i, (original_placement, new_placement) in enumerate(zip(in_shard, new_placements)):
        if isinstance(new_placement, Replicate) and original_placement != new_placement:
            new_placements[i] = original_placement

    if needs_reshard:
        return tuple(out_shape), None, shardable_dims
    else:
        return tuple(out_shape), tuple(new_placements), shardable_dims


def remove_interleaved_shard(*args_schema, **kwargs_schema):
    def replace_interleaved_shard(spec: DTensorSpec) -> DTensorSpec:
        # new_spec = copy.deepcopy(spec)
        new_spec = DTensorSpec(spec.mesh, spec.placements, spec.tensor_meta)
        placements = spec.placements
        interleaved_shard_dims = {
            placement.dim: placement for placement in placements if isinstance(placement, InterleavedShard)
        }
        if not interleaved_shard_dims:
            return new_spec
        if spec.tensor_meta is not None:
            new_shape = []
            new_stride = []
            for d in range(spec.ndim):
                if d in interleaved_shard_dims:
                    interleaved_shard = interleaved_shard_dims[d]
                    first_dim_size = interleaved_shard.interleaved_size
                    second_dim_size = spec.tensor_meta.shape[d] // interleaved_shard.interleaved_size
                    new_shape.append(first_dim_size)
                    new_shape.append(second_dim_size)
                    new_stride.append(second_dim_size * spec.tensor_meta.stride[d])
                    new_stride.append(spec.tensor_meta.stride[d])
                else:
                    new_shape.append(spec.tensor_meta.shape[d])
                    new_stride.append(spec.tensor_meta.stride[d])
            new_spec.tensor_meta = TensorMeta(shape=new_shape, stride=new_stride, dtype=spec.tensor_meta.dtype)
        new_placements = []
        sharded_dim_offset = 1
        for i, placement in enumerate(placements):
            if isinstance(placement, InterleavedShard):
                new_placements.append(Shard(placement.dim + sharded_dim_offset))
                sharded_dim_offset += 1
            else:
                new_placements.append(placement)
        new_spec.placements = tuple(new_placements)
        return new_spec

    new_args_schema = [replace_interleaved_shard(arg) if isinstance(arg, DTensorSpec) else arg for arg in args_schema]
    new_kwargs_schema = {
        key: replace_interleaved_shard(value) if isinstance(value, DTensorSpec) else value
        for key, value in kwargs_schema.items()
    }

    return new_args_schema, new_kwargs_schema


def register_rule_for_view_and_reshape_ops(
    aten_op_overload: torch._ops.OpOverload,
    local_op_name: Callable[..., torch.Tensor],
    schema_info: Optional[RuntimeSchemaInfo] = None,
) -> None:
    spec: Op = ops[local_op_name]

    @register_prop_rule(aten_op_overload, schema_info=schema_info)
    def vescale_view_rule_prop(op_schema: OpSchema) -> OutputSharding:
        new_args_schema, new_kwargs_schema = remove_interleaved_shard(*op_schema.args_schema, **op_schema.kwargs_schema)
        rules = spec.dim_map(*new_args_schema, **new_kwargs_schema)
        input_dtensor_spec = cast(DTensorSpec, new_args_schema[0])
        mesh = input_dtensor_spec.mesh

        assert isinstance(input_dtensor_spec, DTensorSpec), "Expected first input to be a DTensorSpec"
        global_in_shape = input_dtensor_spec.shape
        assert global_in_shape is not None, "Shape required."

        if TORCH_VERSION_BIGGER_THAN_2_2:
            from torch._subclasses.fake_tensor import unset_fake_temporarily
            from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing

            with disable_proxy_modes_tracing(), unset_fake_temporarily():
                (
                    global_out_shape,
                    shard_out,
                    shardable_dims,
                ) = propagate_shape_and_sharding(
                    input_dtensor_spec.placements,
                    tuple(global_in_shape),
                    rules,
                    tuple(mesh.mesh.shape),
                )
        else:
            (
                global_out_shape,
                shard_out,
                shardable_dims,
            ) = propagate_shape_and_sharding(
                input_dtensor_spec.placements,
                tuple(global_in_shape),
                rules,
                tuple(mesh.mesh.shape),
            )

        if shard_out is not None:
            # no reshard needed
            output_dtensor_spec = DTensorSpec(mesh=mesh, placements=tuple(shard_out))

            # We only need the local shape to lower the call into the local op
            args = op_schema.args_schema
            shape_argnum = spec.shape_argnum
            if shape_argnum is not None:
                # compute the local shape from the global shape, then return
                # a resharding even if we don't really reshard, the only reason
                # for this type of resharding is to lower the global shape to
                # local shape
                local_out_shape = compute_local_shape(list(global_out_shape), mesh, shard_out)

                suggested_schema = OpSchema(
                    op=op_schema.op,
                    args_schema=args[:shape_argnum] + (tuple(local_out_shape),) + args[shape_argnum + 1 :],
                    kwargs_schema=op_schema.kwargs_schema,
                )
                return OutputSharding(
                    output_spec=output_dtensor_spec,
                    schema_suggestions=[suggested_schema],
                    needs_redistribute=True,
                )

            return OutputSharding(output_spec=output_dtensor_spec)

        else:
            # TODO: optimize this. we shouldn't simply blindly replicate
            #       unshardable dims ...
            # FIXME: this can be wrong for situations where we have
            #        [Shard(0), Shard(0)]
            # NOTE: generating suggested_placments for InterleavedShard is complex.
            # Just Replicate tensor if it's Sharded.
            suggested_placements = [
                p if not isinstance(p, Shard) else Replicate() for _, p in enumerate(input_dtensor_spec.placements)
            ]
            return OutputSharding(
                output_spec=None,
                schema_suggestions=[
                    OpSchema(
                        op=op_schema.op,
                        args_schema=(
                            DTensorSpec(
                                placements=tuple(suggested_placements),
                                mesh=input_dtensor_spec.mesh,
                                tensor_meta=input_dtensor_spec.tensor_meta,
                            ),
                        )
                        + op_schema.args_schema[1:],
                        kwargs_schema=op_schema.kwargs_schema,
                    )
                ],
                needs_redistribute=True,
            )


register_rule_for_view_and_reshape_ops(aten.view.default, Tensor.view, schema_info=RuntimeSchemaInfo(1))
register_rule_for_view_and_reshape_ops(aten.reshape.default, torch.reshape, schema_info=RuntimeSchemaInfo(1))
register_rule_for_view_and_reshape_ops(aten._unsafe_view.default, Tensor.view, schema_info=RuntimeSchemaInfo(1))
