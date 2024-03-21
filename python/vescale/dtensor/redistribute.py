################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

from typing import Dict, List, Tuple, cast

import torch
import torch.distributed.distributed_c10d as c10d
from torch.utils._python_dispatch import _get_current_dispatch_mode

import vescale.dtensor.dtensor as dtensor
from vescale.dtensor._collective_utils import (
    mesh_all_gather,
    mesh_all_reduce,
    mesh_broadcast,
    mesh_reduce_scatter,
    mesh_scatter,
    wait,
)
from vescale.dtensor._diff import EnablePartialMode, switch_partial_mode
from vescale.dtensor.device_mesh import DeviceMesh
from vescale.dtensor.op_schema import DTensorSpec
from vescale.dtensor.placement_types import InterleavedShard, Partial, Placement, Replicate, Shard

_PlacementItem = Tuple[int, Tuple[Placement, Placement]]


def _replicate_then_shard(val: _PlacementItem) -> int:
    """
    Replicate from inner to outer dimension.
    Shard from outer to inner dimension.
    """
    i, (current, target) = val
    if (target.is_replicate() or target.is_partial()) and current.is_shard():
        return -i
    elif (current.is_replicate() or current.is_partial()) and target.is_shard():
        return i
    else:
        return 0


def _decompose_reshard(val: List[_PlacementItem]) -> List[_PlacementItem]:
    """
    Decompose Si -> Sj into Si -> R -> Sj
    There's 2 ways a shardings can differ within a mesh dimension:
      1) sharding on different tensor dimensions, e.g. Shard(0) -> Shard(1)
      2) different sub-shards of a repeated shard ("mis-aligned sharding")
          (Shard(0), Shard(0)) -> (Replicate(), Shard(0))
          Here the Shard(0) -> Shard(0) for mesh dimension 2 is actually
          a reshard, because in the first case it's a sub-sharding of an already tensor dimension 0,
          and in the second case, it's the first sharding on tensor dimension 0.
    """
    # detect mis-aligned repeated shardings
    from collections import defaultdict

    repeat_dim_current: Dict[int, int] = defaultdict(int)
    repeat_dim_target: Dict[int, int] = defaultdict(int)

    output: List[_PlacementItem] = []

    for i, (current, target) in val:
        # detect mis-aligned sharding
        if current.is_shard():
            repeat_dim_current[cast(Shard, current).dim] += 1
        if target.is_shard():
            repeat_dim_target[cast(Shard, target).dim] += 1
        if (
            isinstance(current, Shard)
            and isinstance(target, Shard)
            and (current.dim != target.dim or repeat_dim_current[current.dim] != repeat_dim_target[target.dim])
        ):
            # decompose Shard(i) -> Shard(j) into Shard(i) -> Replicate() -> Shard(j)
            output.append((i, (current, Replicate())))
            output.append((i, (Replicate(), target)))
        else:
            output.append((i, (current, target)))

    return output


def _reshard_to_replicate_with_pad_one_dim(
    local_tensor: torch.Tensor, size: torch.Size, mesh: DeviceMesh, mesh_dim: int, shard_dim: int
) -> torch.Tensor:
    """
    This function all_gather all shards and return a tensor that
    is replicated on the previously sharded mesh dimension
    """
    my_coordinate = mesh.get_coordinate()
    num_chunks = mesh.size(dim=mesh_dim)

    if my_coordinate is None:
        # if rank is not part of mesh, we simply return local_tensor,
        # which should be an empty tensor
        return local_tensor

    # check if it needs to pad input tensor before all_gather
    full_chunk_size = (size[shard_dim] + num_chunks - 1) // num_chunks
    chunk_sizes = [
        max(
            min(size[shard_dim], full_chunk_size * (idx + 1)) - full_chunk_size * idx,
            0,
        )
        for idx in range(num_chunks)
    ]
    pad_sizes = [full_chunk_size - chunk_size for chunk_size in chunk_sizes]
    is_padded = size[shard_dim] % num_chunks != 0
    pad_size = pad_sizes[my_coordinate[mesh_dim]]
    if pad_size > 0:
        local_tensor = _pad_tensor_on_shard_dim(local_tensor, pad_size, shard_dim)
    local_tensor = local_tensor.contiguous()
    global_shape = list(local_tensor.shape)
    global_shape[shard_dim] *= mesh.size(mesh_dim)
    result = mesh_all_gather(
        local_tensor,
        torch.Size(global_shape),
        mesh,
        shard_dim,
        mesh_dim,
    )
    # Unpad the tensor if the input tensor was padded
    if is_padded:
        full_pad_size = sum(pad_sizes)
        result = _unpad_tensor_on_shard_dim(result, full_pad_size, shard_dim)

    return result


def _pad_tensor_on_shard_dim(
    tensor: torch.Tensor,
    pad_size: int,
    shard_dim: int,
):
    pad = [0, 0] * (tensor.ndim - shard_dim)
    pad[-1] = pad_size
    return torch.nn.functional.pad(tensor, pad)


def _unpad_tensor_on_shard_dim(tensor: torch.Tensor, pad_size: int, shard_dim: int):
    # NOTE: torch.narrow doesn't change stride meta, add contiguous to make sure
    # it doesn't fail if followed by view ops.
    return tensor.narrow(
        shard_dim,
        start=0,
        length=tensor.size(shard_dim) - pad_size,
    ).contiguous()


def _scatter_tensor_by_shard(tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int, shard_spec: Shard) -> torch.Tensor:
    """
    shard and scatter a tensor on a mesh dimension (use coordinate
    0 on the mesh dimension as source of truth)
    """
    my_coordinate = mesh.get_coordinate()
    num_chunks = mesh.size(dim=mesh_dim)

    if my_coordinate is None:
        # if rank is not part of mesh, we simply return an empty tensor
        return tensor.new_empty(0, requires_grad=tensor.requires_grad)

    scatter_list, pad_sizes = shard_spec._split_tensor(tensor, num_chunks, with_padding=True, contiguous=True)

    output = torch.empty_like(scatter_list[my_coordinate[mesh_dim]])
    mesh_scatter(output, scatter_list, mesh, mesh_dim=mesh_dim)

    # Only unpad if the local_tensor was padded on the dimension.
    pad_size = pad_sizes[my_coordinate[mesh_dim]]
    if pad_size > 0:
        output = shard_spec._unpad_tensor(output, pad_size)
    return output


def _replicate_tensor(tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int) -> torch.Tensor:
    """
    Replicate (broadcast) a torch.Tensor on a mesh dimension (use
    the first coordinate on the mesh dimension as source of truth)
    """
    my_coordinate = mesh.get_coordinate()
    if my_coordinate is None:
        # if rank is not part of mesh, we simply return an empty tensor
        return tensor.new_empty(0, requires_grad=tensor.requires_grad)

    tensor = tensor.contiguous()
    tensor = mesh_broadcast(tensor, mesh, mesh_dim=mesh_dim)
    return tensor


def _reduce_scatter_to_shard_with_pad(
    tensor: torch.Tensor,
    mesh: DeviceMesh,
    reduce_op: c10d.ReduceOp.RedOpType,
    mesh_dim: int,
    shard_spec: Shard,
):
    """
    reduce and scatter a tensor on a mesh dimension
    """
    my_coordinate = mesh.get_coordinate()
    num_chunks = mesh.size(dim=mesh_dim)

    if my_coordinate is None:
        # if rank is not part of mesh, we simply return local_tensor,
        # which should be an empty tensor
        return tensor

    is_padded = tensor.size(shard_spec.dim) % num_chunks != 0
    if is_padded:
        scattered_list, pad_sizes = shard_spec._split_tensor(tensor, num_chunks, with_padding=True, contiguous=True)
        tensor = torch.cat(scattered_list, dim=shard_spec.dim)

    output = mesh_reduce_scatter(tensor, mesh, reduce_op, shard_spec.dim, mesh_dim)

    if is_padded:
        output = _unpad_tensor_on_shard_dim(output, pad_sizes[my_coordinate[mesh_dim]], shard_spec.dim)
    return output


@switch_partial_mode
def redistribute_local_tensor(
    local_tensor: torch.Tensor,
    current_spec: DTensorSpec,
    target_spec: DTensorSpec,
    async_op: bool = True,
) -> torch.Tensor:
    """
    This redistribute the local tensor (torch.Tensor) from the current DTensorSpec to
    the target DTensorSpec, which involves the necessary collective calls to transform
    the local shard of the DTensor from its current spec to the target spec.
    """

    if current_spec.mesh != target_spec.mesh:
        # TODO: alltoall/permute reshuffling to change device_mesh if they are not the same
        raise NotImplementedError("Cross device mesh comm not supported yet!")

    new_local_tensor = None

    current_placements = current_spec.placements
    target_placements = target_spec.placements
    sorted_placements = list(enumerate(zip(current_placements, target_placements)))
    sorted_placements = _decompose_reshard(sorted_placements)
    sorted_placements.sort(key=_replicate_then_shard)

    device_mesh = current_spec.mesh

    for i, (current, target) in sorted_placements:
        my_coordinate = device_mesh.get_coordinate()
        num_chunks = device_mesh.size(dim=i)

        if my_coordinate is None:
            # if rank is not part of mesh, we simply return local_tensor,
            # which should be an empty tensor
            return local_tensor

        if current == target:
            # short cut, just use the original local tensor
            new_local_tensor = local_tensor
            continue

        if target.is_replicate():
            # Case 1: target is Replicate
            if current.is_partial():
                partial_spec = cast(Partial, current)
                new_local_tensor = mesh_all_reduce(
                    tensor=local_tensor,
                    mesh=device_mesh,
                    reduce_op=partial_spec.reduce_op,
                    mesh_dim=i,
                )
            elif current.is_interleaved_shard():
                interleaved_shard = cast(InterleavedShard, current)
                original_global_shape = current_spec.shape
                new_global_shape = torch.Size(
                    original_global_shape[: interleaved_shard.dim]
                    + (
                        interleaved_shard.interleaved_size,
                        original_global_shape[interleaved_shard.dim] // interleaved_shard.interleaved_size,
                    )
                    + original_global_shape[interleaved_shard.dim + 1 :]
                )
                original_local_shape = local_tensor.shape
                new_local_shape = torch.Size(
                    original_local_shape[: interleaved_shard.dim]
                    + (interleaved_shard.interleaved_size, -1)
                    + original_local_shape[interleaved_shard.dim + 1 :]
                )

                reshaped_local_tensor = local_tensor.reshape(new_local_shape)
                new_reshaped_local_tensor = mesh_all_gather(
                    tensor=reshaped_local_tensor,
                    global_size=new_global_shape,
                    mesh=device_mesh,
                    scatter_dim=interleaved_shard.dim + 1,
                    mesh_dim=i,
                )
                new_local_tensor = new_reshaped_local_tensor.reshape(
                    torch.Size(
                        original_local_shape[: interleaved_shard.dim]
                        + (-1,)
                        + original_local_shape[interleaved_shard.dim + 1 :]
                    )
                )
            elif current.is_shard():
                current_placement = cast(Shard, current)
                new_local_tensor = _reshard_to_replicate_with_pad_one_dim(
                    local_tensor, current_spec.shape, device_mesh, i, current_placement.dim
                )
            else:
                raise RuntimeError(f"redistribute from {current_placements} to {target_placements} not supported yet")
        elif target.is_interleaved_shard():
            target_placement = cast(InterleavedShard, target)
            if current.is_partial():
                # partial -> interleaved_shard = partial -> replicate -> interleaved_shard
                partial_spec = cast(Partial, current)
                replicate_local_tensor = mesh_all_reduce(
                    tensor=local_tensor,
                    mesh=device_mesh,
                    reduce_op=partial_spec.reduce_op,
                    mesh_dim=i,
                )
                shards = target_placement._split_tensor(
                    tensor=replicate_local_tensor, num_chunks=num_chunks, contiguous=False
                )
                new_local_tensor = shards[my_coordinate[i]].clone()
                pass
            elif current.is_replicate():
                shards = target_placement._split_tensor(
                    tensor=local_tensor,
                    num_chunks=num_chunks,
                    contiguous=False,
                )
                new_local_tensor = shards[my_coordinate[i]].clone()
            else:
                # FIXME(wujiawei.aml): for now, we don't support conversion
                # between InterleavedShard and Shard. Maybe we should provide
                # a method to transfer InterleavedShard to a contiguous Shard?
                raise NotImplementedError("Redistributiom from Shard to InterleavedShard is not supported")
        elif target.is_shard():
            # Case 2: target is Shard
            target_placement = cast(Shard, target)
            if current.is_partial():
                partial_spec = cast(Partial, current)
                new_local_tensor = _reduce_scatter_to_shard_with_pad(
                    local_tensor, device_mesh, partial_spec.reduce_op, i, target_placement
                )
            elif current.is_replicate():
                # split the tensor and return the corresponding cloned local shard
                shards, _ = target_placement._split_tensor(
                    local_tensor,
                    num_chunks,
                    with_padding=False,
                    contiguous=False,
                )
                new_local_tensor = shards[my_coordinate[i]].clone()
            elif current.is_interleaved_shard():
                raise NotImplementedError("Redistribution from InterleavedShard to Shard is not suported")
            else:
                # NOTE: this case shouldn't hit _decompose_sharding, decompose sharding should
                # decompose Shard(0) -> Shard(1) into Shard(0) -> Replicate -> Shard(1)
                assert current.is_shard(), f"Current placement should be shard but found {current}"
                shard_spec = cast(Shard, current)
                if shard_spec.dim != target_placement.dim:
                    # TODO: enable this with all_to_all
                    raise NotImplementedError("Changing sharding dim is not supported yet!")

        elif target.is_partial():
            if current.is_partial():
                mode = _get_current_dispatch_mode()
                if isinstance(mode, EnablePartialMode):
                    # P -> P
                    new_local_tensor = local_tensor
                else:
                    # P -> R
                    partial_spec = cast(Partial, current)
                    new_local_tensor = mesh_all_reduce(local_tensor, device_mesh, partial_spec.reduce_op, i)

            elif current.is_replicate():
                # For replicate -> partial, we zero out all other ranks of the current mesh dim
                # and leave only 1 rank have the data, to perform a "zero cost" reshard.
                if my_coordinate[i] != 0:
                    new_local_tensor = local_tensor.zero_()
                else:
                    new_local_tensor = local_tensor
            elif current.is_interleaved_shard():
                # For sharded tensor -> partial, we reduce the tensor,
                # then follow a same way as the second case.
                interleaved_shard = cast(InterleavedShard, current)
                original_global_shape = current_spec.shape
                new_global_shape = torch.Size(
                    original_global_shape[: interleaved_shard.dim]
                    + (
                        interleaved_shard.interleaved_size,
                        original_global_shape[interleaved_shard.dim] // interleaved_shard.interleaved_size,
                    )
                    + original_global_shape[interleaved_shard.dim + 1 :]
                )
                original_local_shape = local_tensor.shape
                new_local_shape = torch.Size(
                    original_local_shape[: interleaved_shard.dim]
                    + (interleaved_shard.interleaved_size, -1)
                    + original_local_shape[interleaved_shard.dim + 1 :]
                )

                reshaped_local_tensor = local_tensor.reshape(new_local_shape)
                new_reshaped_local_tensor = mesh_all_gather(
                    tensor=reshaped_local_tensor,
                    global_size=new_global_shape,
                    mesh=device_mesh,
                    scatter_dim=interleaved_shard.dim + 1,
                    mesh_dim=i,
                )
                new_local_tensor = new_reshaped_local_tensor.reshape(
                    torch.Size(
                        original_local_shape[: interleaved_shard.dim]
                        + (-1,)
                        + original_local_shape[interleaved_shard.dim + 1 :]
                    )
                )
                if my_coordinate[i] != 0:
                    new_local_tensor = new_local_tensor.zero_()
                pass
            elif current.is_shard():
                # For sharded tensor -> partial, we reduce the tensor,
                # then follow a same way as the second case.
                current_placement = cast(Shard, current)
                new_local_tensor = _reshard_to_replicate_with_pad_one_dim(
                    local_tensor, current_spec.shape, device_mesh, i, current_placement.dim
                )
                if my_coordinate[i] != 0:
                    new_local_tensor = new_local_tensor.zero_()
            else:
                raise RuntimeError(f"redistribute from {current_placements} to {target_placements} not supported yet")

        assert new_local_tensor is not None
        if not async_op:
            new_local_tensor = wait(new_local_tensor)
        local_tensor = new_local_tensor

    assert new_local_tensor is not None, "redistribute failed!"

    return new_local_tensor


class Redistribute(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        # pyre-fixme[2]: Parameter must be annotated.
        ctx,
        input: "dtensor.DTensor",
        device_mesh: DeviceMesh,
        placements: List[Placement],
        async_op: bool = True,
    ):
        current_spec = input._spec
        ctx.current_spec = current_spec
        ctx.async_op = async_op
        target_spec = DTensorSpec(device_mesh, tuple(placements), tensor_meta=input._spec.tensor_meta)

        local_tensor = input._local_tensor
        output = redistribute_local_tensor(local_tensor, current_spec, target_spec, async_op)
        output.requires_grad_(input.requires_grad)

        return dtensor.DTensor(
            output,
            device_mesh,
            target_spec.placements,
            shape=input.shape,
            dtype=input.dtype,
            requires_grad=input.requires_grad,
            stride=input.stride(),
        )

    @staticmethod
    # type: ignore[override]
    def backward(ctx, grad_output: "dtensor.DTensor"):
        previous_spec = ctx.current_spec
        async_op = ctx.async_op
        # When we run backward pass of redistribute (i.e. manual redistribute from
        # user code instead of torch_dispatch), we scan first and see if we need
        # to change the target placement for one special case:
        #   replicate -> partial.
        # In this case we keep the grad as replicate, this is because we don't
        # want to convert the replicated gradients back to partial, although
        # that's logically conform with the same layout, converting the gradients
        # back to partial is actually useless as you would have to do reduce later
        # which would be more expensive than keeping it replicate! For this reason,
        # we keep the replicate grad here.
        # TODO: see if this make sense for all cases.
        current_spec = grad_output._spec

        target_placements: List[Placement] = []
        for current, target in zip(current_spec.placements, previous_spec.placements):
            if not current.is_partial() and target.is_partial():
                # keep target placement to replicate instead of partial in this case
                target_placements.append(Replicate())
            else:
                target_placements.append(target)
        target_spec = DTensorSpec(previous_spec.mesh, tuple(target_placements), tensor_meta=previous_spec.tensor_meta)

        local_tensor = grad_output._local_tensor
        output = redistribute_local_tensor(local_tensor, current_spec, target_spec, async_op)
        output_dtensor = dtensor.DTensor(
            output,
            target_spec.mesh,
            target_spec.placements,
            shape=grad_output.shape,
            dtype=grad_output.dtype,
            requires_grad=grad_output.requires_grad,
            stride=grad_output.stride(),
        )

        return (output_dtensor, None, None, None)
