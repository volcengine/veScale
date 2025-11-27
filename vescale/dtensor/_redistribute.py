# mypy: allow-untyped-defs
################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################
import logging

import torch
import torch.distributed._functional_collectives as funcol
from torch.distributed.tensor.device_mesh import DeviceMesh
from torch.distributed.tensor._redistribute import redistribute_local_tensor as torch_redistribute_local_tensor

import vescale.dtensor as dtensor
from vescale.dtensor._dtensor_spec import DTensorSpec, TensorMeta
from vescale.dtensor.placement_types import (
    Placement,
    Replicate,
    RaggedShard,
)

from vescale.dtensor.vescale_utils import (
    get_ragged_shard,
    get_unflattened_shape_and_offset_before_ragged_shard,
    substitute_ragged_with_replicate,
)

"""
In this file, we modified Redistribute to make it return a vescale dtensor.
"""

__all__ = [
    "Redistribute",
    "redistribute_local_tensor",
]

logger = logging.getLogger(__name__)


def substitute_ragged_spec(spec: DTensorSpec):
    return DTensorSpec(spec.device_mesh, substitute_ragged_with_replicate(spec.placements), spec.tensor_meta)


def redistribute_local_tensor(
    local_tensor: torch.Tensor,
    current_spec: DTensorSpec,
    target_spec: DTensorSpec,
    *,
    async_op: bool = False,
    is_backward: bool = False,
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
    device_mesh = current_spec.mesh

    my_coordinate = device_mesh.get_coordinate()

    if my_coordinate is None:
        # if rank is not part of mesh, we skip redistribute and simply return local_tensor,
        # which should be an empty tensor
        return local_tensor

    has_symints = any(isinstance(s, torch.SymInt) for s in current_spec.shape) or any(
        isinstance(s, torch.SymInt) for s in target_spec.shape
    )
    current_has_ragged_shard = any(isinstance(p, RaggedShard) for p in current_spec.placements)
    target_has_ragged_shard = any(isinstance(p, RaggedShard) for p in target_spec.placements)

    if not (current_has_ragged_shard or target_has_ragged_shard):
        return torch_redistribute_local_tensor(
            local_tensor, current_spec, target_spec, async_op=async_op, is_backward=is_backward
        )

    assert not has_symints, "ragged shard does not support symint"

    if current_has_ragged_shard and target_has_ragged_shard:
        current_i, current_p = get_ragged_shard(current_spec.placements)
        target_i, target_p = get_ragged_shard(target_spec.placements)
        assert current_i == target_i, "currently we only support redistributing ragged shard in the same mesh dim"
        local_shape, _ = get_unflattened_shape_and_offset_before_ragged_shard(current_spec)
        new_local_tensor = current_p._to_new_ragged_shard(
            local_tensor, device_mesh, current_i, list(local_shape), target_p.local_units
        )
    elif current_has_ragged_shard:  # current has but target does not
        current_i, current_p = get_ragged_shard(current_spec.placements)
        local_shape, _ = get_unflattened_shape_and_offset_before_ragged_shard(current_spec)
        new_local_tensor = current_p._to_replicate_tensor(local_tensor, device_mesh, current_i, list(local_shape))
        new_local_tensor = new_local_tensor.view(local_shape)
        new_current_spec = substitute_ragged_spec(current_spec)
        if new_current_spec.placements != target_spec.placements:
            new_local_tensor = torch_redistribute_local_tensor(
                new_local_tensor,
                new_current_spec,
                target_spec,
                async_op=async_op,
                is_backward=is_backward,
            )
    else:  # target has but current does not
        target_i, target_p = get_ragged_shard(target_spec.placements)
        new_target_spec = substitute_ragged_spec(target_spec)
        new_local_tensor = local_tensor
        if current_spec.placements != new_target_spec.placements:
            new_local_tensor = torch_redistribute_local_tensor(
                local_tensor, current_spec, new_target_spec, async_op=async_op, is_backward=is_backward
            )
        chunk_lst = target_p._split_tensor(new_local_tensor, device_mesh.size(target_i))
        new_local_tensor = chunk_lst[my_coordinate[target_i]].contiguous()

    assert new_local_tensor is not None, "redistribute failed!"

    if not async_op and isinstance(new_local_tensor, funcol.AsyncCollectiveTensor):
        new_local_tensor = new_local_tensor.wait()

    return new_local_tensor


class Redistribute(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        # pyre-fixme[2]: Parameter must be annotated.
        ctx,
        input: "dtensor.DTensor",
        device_mesh: DeviceMesh,
        placements: tuple[Placement, ...],
        async_op: bool = False,
    ):
        current_spec = input._spec
        ctx.current_spec = current_spec
        ctx.async_op = async_op

        if current_spec.placements != placements:
            target_spec = DTensorSpec(device_mesh, placements, tensor_meta=input._spec.tensor_meta)

            local_tensor = input._local_tensor
            output = redistribute_local_tensor(local_tensor, current_spec, target_spec, async_op=async_op)
        else:
            # use the same local tensor if placements are the same.
            output = input._local_tensor
            target_spec = current_spec

        return dtensor.DTensor(
            output,
            target_spec,
            requires_grad=input.requires_grad,
        )

    @staticmethod
    def backward(ctx, grad_output: "dtensor.DTensor"):  # type: ignore[override]
        previous_spec = ctx.current_spec
        current_spec = grad_output._spec
        async_op = ctx.async_op

        local_tensor = grad_output._local_tensor
        output = redistribute_local_tensor(
            local_tensor,
            current_spec,
            previous_spec,
            async_op=async_op,
            is_backward=True,
        )
        # normalize the target placement to replicate if it is partial
        normalized_placements: list[Placement] = []
        for previous_placement in previous_spec.placements:
            if previous_placement.is_partial():
                # keep target placement to replicate instead of partial in this case
                normalized_placements.append(Replicate())
            else:
                normalized_placements.append(previous_placement)

        spec = DTensorSpec(
            previous_spec.device_mesh,
            tuple(normalized_placements),
            tensor_meta=TensorMeta(
                shape=grad_output.shape,
                stride=grad_output.stride(),
                dtype=grad_output.dtype,
            ),
        )
        output_dtensor = dtensor.DTensor(
            output,
            spec,
            requires_grad=grad_output.requires_grad,
        )

        return (
            output_dtensor,
            None,
            None,
            None,
        )
