################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

import os
import warnings
from typing import List, Optional, Sequence, Tuple, Union, cast

import torch
import torch.distributed._functional_collectives as funcol

import vescale.dtensor.random as random
from vescale.dtensor._collective_utils import mesh_broadcast, mesh_scatter
from vescale.dtensor._utils import compute_global_tensor_info, gather_local_tensor_shape
from vescale.dtensor.device_mesh import DeviceMesh, mesh_resources
from vescale.dtensor.dtensor import DTensor
from vescale.dtensor.ops.utils import normalize_dims
from vescale.dtensor.placement_types import DTensorSpec, InterleavedShard, Placement, Replicate, Shard
from vescale.dtensor.random import OffsetBasedRNGTracker, is_rng_supported_mesh
from vescale.dtensor.redistribute import (
    Redistribute,
    _replicate_tensor,
    _scatter_tensor_by_shard,
    redistribute_local_tensor,
)


__all__ = [
    "distribute_tensor",
    "to_local",
    "from_local",
    "redistribute_dtensor",
    "normalize_placements",
]

VESCALE_DISABLE_RUN_CHECK = os.environ.get("VESCALE_DISABLE_RUN_CHECK", "0") == "1"


def normalize_placements(
    placements: Optional[Sequence[Placement]], mesh_ndim: int, none_as_replicate: bool = False
) -> Optional[Tuple[Placement]]:
    """
    normalize a placements to be valid.
    """
    if placements is None:
        return tuple(Replicate() for _ in range(mesh_ndim)) if none_as_replicate else None

    if len(placements) > mesh_ndim:
        raise ValueError(f"`placements` (len={len(placements)}) have larger length than `mesh_ndim` ({mesh_ndim})!")

    if len(placements) < mesh_ndim:
        warnings.warn(
            "`placements` have less elements than `mesh_ndim`!. We will postpend Replicate placement to the end.",
            UserWarning,
        )
        placements = list(placements) + [Replicate()] * (mesh_ndim - len(placements))

    for p in placements:
        if not isinstance(p, Placement):
            raise ValueError(f"Unsupported placements = {placements}!")

    return tuple(placements)


# NOTE [Autograd interaction between torch.Tensor]
#
# The autograd functions defined below are being used by the public
# facing APIs (i.e. from_local, to_local) to ensure our DTensor
# works together with torch.Tensor within autograd engine. This
# allows DistributedTensor to exist on part of the module hierarchy
# and still able to calculate gradients across the torch.Tensor and
# DistributedTensor boundary.
# As an example, we have the a module that consists of submodules
# A, B, and C, the execution flow would be like:
#  input(torch.Tensor) -> Module A -> Module B -> Module C -> output (torch.Tensor)
#
# Suppose I only want to make Module B be a sharded module with
# DistributedTensor params, we would need to make the following
# flow to work:
#
#  input(torch.Tensor) -> Module A
#       -> DTensor input -> Sharded Module B -> DTensor output
#           -> output (torch.Tensor) -> Module C -> output (torch.Tensor)
#
# We need the conversion from Module A to DTensor input, which is
# `from_local`, and conversion from DTensor output to output, which
# is `to_local`, thus these two functions must be Autograd functions.
#
class _ToTorchTensor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: "DTensor", grad_placements: Optional[Sequence[Placement]], async_output: bool):
        ctx.dtensor_spec = input._spec
        ctx.grad_placements = grad_placements
        local_tensor = input._local_tensor
        if not async_output and type(local_tensor) is funcol.AsyncCollectiveTensor:
            # synchronously wait for any pending collectives to get the result tensor
            local_tensor = local_tensor.trigger_wait()
            local_tensor = local_tensor.elem  # type: ignore[attr-defined]
        # We need to return a fresh Tensor object there as autograd metadata
        # will be inplaced into it. So we don't want to pollute the Tensor
        # object stored in the _local_tensor of this DTensor.
        return local_tensor.view_as(local_tensor)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        dtensor_spec = ctx.dtensor_spec
        mesh = dtensor_spec.mesh
        grad_placements = ctx.grad_placements
        dtensor_meta = dtensor_spec.tensor_meta

        if grad_placements is not None:
            grad_spec = DTensorSpec(mesh, grad_placements)
            grad_output = redistribute_local_tensor(grad_output, grad_spec, dtensor_spec)

        _, tensor_stride = compute_global_tensor_info(grad_output, mesh, dtensor_spec.placements)

        return (
            DTensor(
                grad_output,
                mesh,
                tuple(dtensor_spec.placements),
                shape=dtensor_meta.shape,
                dtype=dtensor_meta.dtype,
                requires_grad=grad_output.requires_grad,
                stride=tuple(tensor_stride),
            ),
            None,
            None,
        )


class _FromTorchTensor(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        device_mesh: DeviceMesh,
        placements: Tuple[Placement, ...],
        run_check: bool,
        shape: Optional[torch.Size] = None,
        stride: Optional[Tuple[int, ...]] = None,
        async_input: bool = True,
    ) -> "DTensor":
        ctx.previous_placement = placements
        ctx.previous_device_mesh = device_mesh
        ctx.async_input = async_input

        if shape and stride:  # use given global shape and stride
            tensor_shape, tensor_stride = shape, stride
        elif not shape and not stride:  # use inferred global shape and stride
            if run_check:  # support uneven shard
                meshdim_localtensor_shape = gather_local_tensor_shape(input, device_mesh, placements)
                global_shape, global_stride = compute_global_tensor_info(
                    input, device_mesh, placements, meshdim_localtensor_shape
                )
            else:  # assume even shard
                global_shape, global_stride = compute_global_tensor_info(input, device_mesh, placements)
            tensor_shape, tensor_stride = torch.Size(global_shape), tuple(global_stride)
        else:
            raise ValueError(
                f"Found shape:{shape}, stride:{stride}.",
                "Please pass both shape and stride at the same time.",
            )

        if device_mesh.get_coordinate() is None:
            # if the global rank is not participating in the device mesh, we
            # simply set the local tensor to an empty tensor
            # TODO: set global shape/stride as 0 as well, and simplify code
            input = input.new_empty(0, requires_grad=input.requires_grad)
        elif run_check:
            # Assume global tensor_shape/tensor_stride are the same across ranks
            # TODO: add assertion for Inferred local shape == actual local shape
            # TODO: See if we need to make this run_check logic have a corresponding backward.
            for idx, placement in enumerate(placements):
                if placement.is_replicate():
                    # broadcast rank 0 tensor to all ranks
                    # only broadcast if run_check is True
                    input = input.contiguous()
                    input = mesh_broadcast(input, device_mesh, mesh_dim=idx)
                elif placement.is_interleaved_shard():
                    if input.shape[placement.dim] % placement.interleaved_size != 0:
                        raise ValueError(
                            f"Tensor size at dim {placement.dim} is not divisible by {placement.interleaved_size}"
                        )

        # We want a fresh Tensor object that shares memory with the input tensor
        dist_tensor = DTensor(
            input.view_as(input),
            device_mesh,
            placements,
            shape=tensor_shape,
            dtype=input.dtype,
            # requires_grad of the dist tensor depends on if input requires_grad or not
            requires_grad=input.requires_grad,
            stride=tensor_stride,
        )
        return dist_tensor

    @staticmethod
    # type: ignore[override]
    def backward(ctx, grad_output: "DTensor"):
        previous_placement = ctx.previous_placement
        previous_device_mesh = ctx.previous_device_mesh
        async_input = ctx.async_input

        # reshard to the placement when creating DistributedTensor
        # so that the gradient layout matches, and we could return
        # local gradients directly
        if grad_output.placements != previous_placement:
            grad_output = Redistribute.apply(grad_output, previous_device_mesh, previous_placement, False)

        local_tensor = grad_output._local_tensor
        if not async_input and type(local_tensor) is funcol.AsyncCollectiveTensor:
            # synchronously wait for any pending collectives to get the result tensor
            local_tensor = local_tensor.trigger_wait()
            local_tensor = local_tensor.elem  # type: ignore[attr-defined]

        # TODO: backward is also differentiable now, add a test
        # to test higher level gradients.
        return local_tensor.view_as(local_tensor), None, None, None, None, None, None


def from_local(
    local_tensor: torch.Tensor,
    device_mesh: Optional[DeviceMesh] = None,
    placements: Optional[Sequence[Placement]] = None,
    *,
    run_check: bool = True,
    shape: Optional[torch.Size] = None,
    stride: Optional[Tuple[int, ...]] = None,
    async_input: bool = True,
) -> "DTensor":
    """
    Create a :class:`DTensor` from a local torch.Tensor on each rank
    according to the `device_mesh` and `placements` specified.

    Args:
        local_tensor (torch.Tensor): local torch.Tensor on each rank.
        device_mesh (:class:`DeviceMesh`, optional): DeviceMesh to place the
            tensor, if not specified, must be called under a DeviceMesh
            context manager, default: None
        placements (List[:class:`Placement`], optional): the placements that
            describes how to place the local torch.Tensor on DeviceMesh, must
            have the same number of elements as `device_mesh.ndim`. If not
            specified, we will by default replicate the tensor across the
            `device_mesh` from the first rank of each dimension of the `device_mesh`.

    Keyword args:
        run_check (bool, optional): indicate whether to run check across ranks
            to check meta information and data.
            If True (default), ensure correctness at cost of extra communication across ranks:
            - allgather shapes for uneven sharding
            - broadcast data for `Replicate` `placements` (the data on first rank of
                the device mesh dimension will be broadcasted to other ranks.)
            If False, no correctness guarantee but communication free.
        shape (torch.Size, optional): the global shape of DTensor.
            If given, use this as overriding global shape to build DTensor; This is useful when
            local shape of `local_tensor` are different across the ranks (i.e., uneven sharding).
            If not given, `shape` will be inferred either assuming the DTensor is evenly sharded
            across ranks or gathering other ranks's shape to build global shape.
        stride (tuple[int], optional): the global stride of DTensor.
            Usage is same as `shape`.
        async_input (bool, optional): indicate whether to get async input grad when
            backwarding `from_local`.

    Returns:
        A :class:`DTensor` object

    Example:

        # manual given shape and stride (support uneven sharding)
        saved_shape, saved_stride = dinput.shape, dinput.stride()
        out = dinput.to_local()
        dout = from_local(out, mesh, placements, shape=saved_shape, stride=saved_stride)

        # auto inferred shape and stride (support uneven sharding)
        out = dinput.to_local()
        dout = from_local(out, mesh, placements)

        # manual given shape and stride (support uneven sharding),
        # without run_check's extra communication
        saved_shape, saved_stride = dinput.shape, dinput.stride()
        out = dinput.to_local()
        dout = from_local(out, mesh, placements, run_check=False, shape=saved_shape, stride=saved_stride)

        # auto inferred shape and stride (only even sharding),
        # without run_check's extra communication
        out = dinput.to_local()
        dout = from_local(out, mesh, placements, run_check=False)


    .. note:: `from_local` is differentiable, the `requires_grad` of the created
        `DTensor` object will depend on if `local_tensor` requires_grad or not.
    """
    assert type(local_tensor) is not DTensor
    assert type(getattr(local_tensor, "data", None)) is not DTensor

    if VESCALE_DISABLE_RUN_CHECK:
        run_check = False

    # if same shape/dtype, no need to run_check, if not, must allgather
    # the metadatas to check the size/dtype across ranks
    # There should be no data communication unless there's replication
    # strategy, where we broadcast the replication from the first rank
    # in the mesh dimension
    device_mesh = device_mesh or mesh_resources.get_current_mesh()
    device_type = device_mesh.device_type

    # convert the local tensor to desired device base on device mesh's device_type
    if device_type != local_tensor.device.type and not local_tensor.is_meta:
        local_tensor = local_tensor.to(device_type)

    # validate placements
    placements: Tuple[Placement] = normalize_placements(placements, device_mesh.ndim, none_as_replicate=True)

    # TODO: fix later
    # if any(p.is_partial() for p in placements if p is not None):
    #     warnings.warn(
    #         "DTensor.from_local(.., [Partial]) has no zero-out feature yet! Use Partial with caution.", UserWarning
    #     )

    # `from_local` is differentiable, and the gradient of the dist tensor this function
    # created should flow back the gradients to the local_tensor, so we call an autograd
    # function to construct the dist tensor instead.
    return _FromTorchTensor.apply(  # pyre-ignore[16]: autograd func
        local_tensor, device_mesh, placements, run_check, shape, stride, async_input
    )


def to_local(
    dtensor: "DTensor",
    *,
    grad_placements: Optional[Sequence[Placement]] = None,
    async_output: bool = True,
) -> torch.Tensor:
    """
    Get the local tensor of this DTensor on its current rank. For sharding it returns
    a local shard of the logical tensor view, for replication it returns the replica on
    its current rank.

    Keyword args:
        grad_placements (List[:class:`Placement`], optional): the placements describes
            the future layout of any gradient layout of the Tensor returned from this
            function.
            `to_local` converts DTensor to local tensor and the returned local tensor
            might not be used as the original DTensor layout later in the code. This
            argument is the hint that user can give to autograd in case the gradient
            layout of the returned tensor does not match the original DTensor layout.
            If not specified, we will assume the gradient layout remains the same
            as the original DTensor and use that for gradient computation.
        async_output (bool): whether to get async output tensor.

    Returns:
        A :class:`torch.Tensor` or `AsyncCollectiveTensor` object. it represents the
            local tensor on its current rank.

    .. note:: `to_local` is differentiable, the `requires_grad` of the local tensor returned
        will depend on if the `DTensor` requires_grad or not.
    """
    return _ToTorchTensor.apply(dtensor, grad_placements, async_output)


def distribute_tensor(
    tensor: torch.Tensor,
    device_mesh: Optional[DeviceMesh] = None,
    placements: Optional[Sequence[Placement]] = None,
) -> "DTensor":
    """
    Distribute a global `torch.Tensor` to the `device_mesh` according to the `placements`
    specified. The rank of `device_mesh` and `placements` must be the same.

    Args:
        tensor (torch.Tensor): global torch.Tensor to be distributed. Note that if you
            want to shard a tensor on a dimension that is not evenly divisible by
            the number of devices in that mesh dimension, we use `torch.chunk`
            semantic to shard the tensor and scatter the shards.
        device_mesh (:class:`DeviceMesh`, optional): DeviceMesh to distribute the
            tensor, if not specified, must be called under a DeviceMesh context
            manager, default: None
        placements (List[:class:`Placement`], optional): the placements that
            describes how to place the tensor on DeviceMesh, must have the same
            number of elements as `device_mesh.ndim`. If not specified, we will
            by default replicate the tensor across the `device_mesh` from the
            first rank of each dimension of the `device_mesh`.

    Returns:
        A :class:`DTensor` object

    Best practice to save memory:
        >>> dist_tensor = distribute_tensor(global_tensor, ...)
        >>> del global_tensor
    """

    torch._C._log_api_usage_once("torch.dtensor.distribute_tensor")

    # get default device mesh if there's nothing specified
    device_mesh = device_mesh or mesh_resources.get_current_mesh()
    device_type = device_mesh.device_type

    # instantiate a RNG tracker if haven't. By default DTensor uses an
    # OffsetBasedRNGTracker to perform random operators.
    # TODO: the value assignment to global variable is not the ideal solution
    # we can replace it in future.
    if is_rng_supported_mesh(device_mesh) and not random._rng_tracker:
        random._rng_tracker = OffsetBasedRNGTracker(device_type)

    if not tensor.is_leaf:
        raise RuntimeError("`distribute_tensor` should be used to distribute leaf tensors! but found non-leaf tensor!")

    # convert tensor to the corresponding device type if it's not in that device type
    if device_type != tensor.device.type and not tensor.is_meta:
        tensor = tensor.to(device_type)

    # validate placements
    placements: Tuple[Placement] = normalize_placements(placements, device_mesh.ndim, none_as_replicate=True)

    # validate tensor type
    if isinstance(tensor, DTensor):
        # if the tensor is already a DTensor, we just need to check if the
        # device mesh and placements are the same
        if tensor.device_mesh != device_mesh:
            raise ValueError(
                f"Cannot distribute a DTensor with device mesh {tensor.device_mesh} "
                f"to a different device mesh {device_mesh}."
            )
        if tensor.placements != placements:
            raise ValueError(
                f"Cannot distribute a DTensor with placements {tensor.placements} "
                f"to a different placements {placements}. do you want to call "
                f"`redistribute` instead?"
            )
        return tensor

    local_tensor = tensor

    # distribute the tensor according to the placements.
    for idx, placement in enumerate(placements):
        if placement.is_interleaved_shard():
            interleaved_shard = cast(InterleavedShard, placement)
            if interleaved_shard.dim < 0:
                # normalize interleaved shard placement dim
                interleaved_shard.dim += tensor.ndim
            my_coordinate = device_mesh.get_coordinate()
            # if rank is not part of mesh, we simply return an empty tensor
            output = local_tensor.new_empty(0, requires_grad=local_tensor.requires_grad)
            if my_coordinate is not None:
                scatter_tensor_list = interleaved_shard._split_tensor(
                    local_tensor, num_chunks=device_mesh.size(idx), contiguous=True
                )
                output = torch.empty_like(scatter_tensor_list[my_coordinate[idx]])
                mesh_scatter(
                    output=output, scatter_list=scatter_tensor_list, mesh=device_mesh, mesh_dim=idx, async_op=False
                )
            local_tensor = output
        elif placement.is_shard():
            shard = cast(Shard, placement)
            if shard.dim < 0:
                # normalize shard placement dim
                shard.dim += tensor.ndim
            local_tensor = _scatter_tensor_by_shard(local_tensor, device_mesh, idx, shard)
        elif placement.is_replicate():
            placement = cast(Replicate, placement)
            local_tensor = _replicate_tensor(local_tensor, device_mesh, idx)
        elif placement.is_partial():
            my_coordinate = device_mesh.get_coordinate()
            if my_coordinate is None:
                # if rank is not part of mesh, we simply return an empty tensor
                local_tensor = local_tensor.new_empty(0, requires_grad=local_tensor.requires_grad)
            # we zero out all other ranks of the current mesh dim
            # and leave only 1 rank (by default, rank 0) have the data, to perform a "zero cost" shard.
            is_req_grad = local_tensor.requires_grad
            local_tensor = local_tensor.contiguous()
            if my_coordinate and my_coordinate[idx] != 0:
                with torch.no_grad():
                    local_tensor.zero_()  # inplace memset to zero
                local_tensor = local_tensor.requires_grad_(is_req_grad)
        else:
            raise RuntimeError(
                f"Trying to distribute tensor with unsupported placements {placement} on device mesh dimension {idx}!"
            )

    assert local_tensor is not None, "distributing a tensor should not be None"
    # detach the local tensor passed to DTensor since after the construction
    # of DTensor, autograd would work on top of DTensor instead of local tensor
    return DTensor(
        local_tensor.detach().requires_grad_(tensor.requires_grad),
        device_mesh,
        placements,
        shape=tensor.size(),
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad,
        stride=tensor.stride(),
    )


def redistribute_dtensor(
    dtensor: "DTensor",
    device_mesh: Optional[DeviceMesh] = None,
    placements: Optional[Sequence[Placement]] = None,
    async_op: bool = True,
) -> "DTensor":
    """
    `redistribute_dtensor` performs necessary collective operations that redistribute the current
    DTensor from its current placements to a new placements, or from is current DeviceMesh
    to a new DeviceMesh. i.e. we can turn a Sharded DTensor to a Replicated DTensor by
    specifying a Replicate placement for each dimension of the DeviceMesh.

    Args:
        device_mesh (:class:`DeviceMesh`, optional): DeviceMesh to place the
            DTensor, if not specified, must be called under a DeviceMesh
            context manager, default: None
        placements (List[:class:`Placement`], optional): the new placements that
            describes how to place the DTensor into the DeviceMesh, must
            have the same number of elements as `device_mesh.ndim`.

    Returns:
        A :class:`DTensor` object

    .. note:: `redistribute_dtensor` is differentiable.
    """
    # NOTE: This redistribute API currently only supports out
    # of place redistribution, i.e. it always create a new
    # DTensor object and leave the original one unchanged.

    # if device_mesh is not specified, use the current device_mesh
    device_mesh = device_mesh or dtensor.device_mesh
    # raise error if new placements not specified
    if placements is None:
        raise RuntimeError("placements is needed for redistribute!")

    for placement in placements:
        if isinstance(placement, (Shard, InterleavedShard)) and placement.dim < 0:
            # normalize shard dim to be positive
            placement.dim += dtensor.ndim

    # Early return the original DTensor if the placements are the same.
    if dtensor._spec.placements == placements:
        return dtensor

    return Redistribute.apply(dtensor, device_mesh, placements, async_op)


def vescale_all_gather(
    d_tensor: "DTensor",
    mesh_dims: Optional[Union[int, List[int]]] = None,
    async_op: bool = True,
) -> "DTensor":
    """
    all gather the DTensor along specified dimensions.
    Args:
        dtensor (:class:`DTensor`): DTensor to be all gathered.
        mesh_dims (int or List[int], optional): device mesh dimensions along which to
            perform the all gather operation. If not specified, all sharded dimensions are used.
        async_op (bool, optional): whether the operation is asynchronous, default: True.

    Returns:
        A :class:`DTensor` object
    """
    assert isinstance(d_tensor, DTensor), "DTensor is required for vescale_all_gather"
    dtensor_spec = d_tensor._spec
    dst_placements = list(d_tensor.placements)
    device_mesh = d_tensor.device_mesh

    # get all sharded dims of device mesh
    all_sharded_mesh_dims = []
    for i, p in enumerate(dtensor_spec.placements):
        if p.is_shard():
            all_sharded_mesh_dims.append(i)

    if mesh_dims is None or mesh_dims == []:
        mesh_dims = all_sharded_mesh_dims

    mesh_dims = normalize_dims(mesh_dims, dtensor_spec.ndim)

    for mesh_dim in mesh_dims:
        if mesh_dim not in all_sharded_mesh_dims:
            raise ValueError(f"DeviceDim {mesh_dim} is not sharded, cannot use it for all gather")
        dst_placements[mesh_dim] = Replicate()

    return redistribute_dtensor(d_tensor, device_mesh, tuple(dst_placements), async_op)


def vescale_all_reduce(
    d_tensor: "DTensor",
    mesh_dims: Optional[Union[int, List[int]]] = None,
    async_op: bool = True,
) -> "DTensor":
    """
    all reduce dtensor along given dimensions.

    Args:
        dtensor (:class:`DTensor`): DTensor to be reduced.
        mesh_dims (int or List[int], optional): the device mesh dimensions to perform reduce operation.
        async_op (bool, optional): whether the operation is asynchronous.

    Returns:
        A :class:`DTensor` object
    """
    assert isinstance(d_tensor, DTensor), "DTensor is required for vescale_all_reduce"
    dtensor_spec = d_tensor._spec
    dst_placements = list(d_tensor.placements)
    device_mesh = d_tensor.device_mesh

    if mesh_dims is None or mesh_dims == []:
        mesh_dims = dtensor_spec.sums

    mesh_dims = normalize_dims(mesh_dims, device_mesh.ndim)

    for mesh_dim in mesh_dims:
        if mesh_dim not in dtensor_spec.sums:
            raise ValueError(f"MeshDim {mesh_dim} is not a reduction dimension, cannot use it for all reduce")
        dst_placements[mesh_dim] = Replicate()
    return redistribute_dtensor(d_tensor, device_mesh, tuple(dst_placements), async_op)


def vescale_reduce_scatter(
    d_tensor: "DTensor",
    reduce_mesh_dims: Optional[Union[int, List[int]]] = None,
    scatter_dims: Union[int, List[int]] = None,
    mesh_dims: Union[int, List[int]] = None,
    async_op: bool = True,
) -> "DTensor":
    """
    reduce scatter a DTensor on a specified device mesh dimension.

    Args:
        dtensor (:class:`DTensor`): DTensor to be reduce-scattered.
        reduce_mesh_dims (int or List[int], optional): the device mesh dimensions to perform reduce operation.
        scatter_dims (int or List[int]): the tensor dimensions to scatter the reduced tensor.
        mesh_dims (int or List[int]): the device mesh dimensions to scatter the reduced tensor,
            it should be the same size as `scatter_dims`.
        async_op (bool, optional): whether the operation is asynchronous.

    Returns:
        A :class:`DTensor` object
    """
    assert isinstance(d_tensor, DTensor), "DTensor is required for vescale_reduce_scatter"
    assert scatter_dims is not None, "Scatter dimensions must be specified"
    assert mesh_dims is not None, "Mesh dimensions must be specified"
    if isinstance(scatter_dims, int):
        scatter_dims = [scatter_dims]
    if isinstance(mesh_dims, int):
        mesh_dims = [mesh_dims]

    assert len(mesh_dims) == len(scatter_dims), "Number of scatter_dims and mesh_dims must be the same"

    dtensor_spec = d_tensor._spec
    dst_placements = list(d_tensor.placements)
    device_mesh = d_tensor.device_mesh

    if reduce_mesh_dims is None or reduce_mesh_dims == []:
        reduce_mesh_dims = dtensor_spec.sums

    scatter_dims = normalize_dims(scatter_dims, dtensor_spec.ndim)
    mesh_dims = normalize_dims(mesh_dims, device_mesh.ndim)
    reduce_mesh_dims = normalize_dims(reduce_mesh_dims, device_mesh.ndim)

    for mesh_dim in dtensor_spec.sums:
        dst_placements[mesh_dim] = Replicate()
    for scatter_dim, mesh_dim in zip(scatter_dims, mesh_dims):
        dst_placements[mesh_dim] = Shard(scatter_dim)

    return redistribute_dtensor(d_tensor, device_mesh, tuple(dst_placements), async_op)
