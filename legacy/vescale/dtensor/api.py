################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

from typing import List, Optional, Sequence, Tuple, Union, cast

import torch

import vescale.dtensor.random as random
from vescale.dtensor._collective_utils import mesh_scatter
from vescale.dtensor.device_mesh import DeviceMesh, mesh_resources
from vescale.dtensor.dtensor import DTensor, normalize_placements
from vescale.dtensor.ops.utils import normalize_dims
from vescale.dtensor.placement_types import Placement, Replicate, Shard, InterleavedShard
from vescale.dtensor.random import init_vescale_rng_tracker, is_rng_supported_mesh
from vescale.dtensor.redistribute import (
    _replicate_tensor,
    _scatter_tensor_by_shard,
)

__all__ = [
    "normalize_placements",
    "from_local",
    "to_local",
    "distribute_tensor",
    "redistribute_dtensor",
    "vescale_all_gather",
    "vescale_all_reduce",
    "vescale_reduce_scatter",
]


def from_local(
    local_tensor: torch.Tensor,
    device_mesh: Optional[DeviceMesh] = None,
    placements: Optional[Sequence[Placement]] = None,
    *,
    run_check: bool = True,
    shape: Optional[torch.Size] = None,
    stride: Optional[Tuple[int, ...]] = None,
    support_uneven: bool = True,
    async_input: bool = True,
) -> DTensor:
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
        run_check (bool, optional): indicate whether to run check across ranks to check meta information and data.
            If True (default), ensure correctness at cost of extra communication across ranks:
                - broadcast data for `Replicate` `placements` (the data on first rank of
                    the device mesh dimension will be broadcasted to other ranks.)
                - gather global shapes for other `placements` to ensure the same across ranks
            If False, no correctness guarantee but communication free.
        shape (torch.Size, optional): the global shape of DTensor.
            If given, use this as overriding global shape to build DTensor; This is useful when
            local shape of `local_tensor` are different across the ranks (i.e., uneven sharding).
            If not given, `shape` will be inferred at the cost of communication (see `support_uneven`).
        stride (tuple[int], optional): the global stride of DTensor.
            Usage is same as `shape`.
            `shape` and `stride` must be given togather or not given togather.
        support_uneven (bool, optional): indicate whether to support uneven sharding at the cost
            of extra communication across ranks.
            If True (default), use gather communication to infer global shape (that can be unevenly sharded).
            If False, use local shape to infer global shape (that must be evenly sharded).
        async_input (bool, optional): indicate whether to get asynchrounous input grad when
            backwarding `from_local`.

    Returns:
        A :class:`DTensor` object

    Example of uneven sharding:

        # manually given shape and stride (support uneven sharding)
        >>> saved_shape, saved_stride = dinput.shape, dinput.stride()
        >>> out = dinput.to_local()
        >>> dout = from_local(out, mesh, placements, shape=saved_shape, stride=saved_stride)

        # auto inferred shape and stride (support uneven sharding) with gather communication overhead
        >>> out = dinput.to_local()
        >>> dout = from_local(out, mesh, placements)

        # auto inferred shape and stride (only even sharding) without gather communication overhead
        >>> out = dinput.to_local()
        >>> dout = from_local(out, mesh, placements, support_uneven=False)


    .. note::
        - `from_local` is differentiable
        - the `requires_grad` of the created `DTensor` object will depend on if `local_tensor` requires_grad or not.
    """
    return DTensor.from_local(
        local_tensor,
        device_mesh,
        placements,
        run_check=run_check,
        shape=shape,
        stride=stride,
        support_uneven=support_uneven,
        async_input=async_input,
    )


def to_local(
    dtensor: DTensor,
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
    return dtensor.to_local(grad_placements=grad_placements, async_output=async_output)


def distribute_tensor(
    tensor: torch.Tensor,
    device_mesh: Optional[DeviceMesh] = None,
    placements: Optional[Sequence[Placement]] = None,
) -> DTensor:
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
        random._rng_tracker = init_vescale_rng_tracker(device_type)

    if not tensor.is_leaf:
        raise RuntimeError("`distribute_tensor` should be used to distribute leaf tensors! but found non-leaf tensor!")

    # convert tensor to the corresponding device type if it's not in that device type
    if device_type != tensor.device.type and not tensor.is_meta:
        tensor = tensor.to(device_type)

    # validate placements
    placements: Tuple[Placement] = normalize_placements(
        placements, device_mesh.ndim, tensor_ndim=tensor.ndim, none_as_replicate=True
    )

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

    my_coordinate = device_mesh.get_coordinate()
    # if rank is not part of mesh, we simply create an empty local tensor
    if my_coordinate is None:
        local_tensor = tensor.new_empty(0, requires_grad=tensor.requires_grad)
    else:
        local_tensor = tensor
        # distribute the tensor according to the placements.
        for idx, placement in enumerate(placements):
            if placement.is_interleaved_shard():
                interleaved_shard = cast(InterleavedShard, placement)
                assert interleaved_shard.dim >= 0
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
                assert shard.dim >= 0
                local_tensor = _scatter_tensor_by_shard(local_tensor, device_mesh, idx, shard)
            elif placement.is_replicate():
                placement = cast(Replicate, placement)
                local_tensor = _replicate_tensor(local_tensor, device_mesh, idx)
            elif placement.is_partial():
                # we zero out all other ranks of the current mesh dim
                # and leave only 1 rank (by default, rank 0) have the data, to perform a "zero cost" shard.
                local_tensor = local_tensor.contiguous()
                if my_coordinate[idx] != 0:
                    is_req_grad = local_tensor.requires_grad
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
    dtensor: DTensor,
    device_mesh: Optional[DeviceMesh] = None,
    placements: Optional[Sequence[Placement]] = None,
    async_op: bool = True,
) -> DTensor:
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
        async_op (bool, optional): whether this redistribute is asynchronous in communication (for both forward and backward).
            - True: the default asynchronous behavior for performance
            - False: mostly used for third-party plugin op that doesn't accept asynchronous collective tensor.

    Returns:
        A :class:`DTensor` object

    .. Note::
        - `redistribute_dtensor` is differentiable (i.e., redistribute happen for both forward and backward)
        - This redistribute API currently only supports out of place redistribution, i.e. it always create a new DTensor object and leave the original one unchanged.
    """
    return dtensor.redistribute(device_mesh, placements, async_op)


def vescale_all_gather(
    d_tensor: DTensor,
    mesh_dims: Optional[Union[int, List[int]]] = None,
    async_op: bool = True,
) -> DTensor:
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

    return d_tensor.redistribute(device_mesh, tuple(dst_placements), async_op)


def vescale_all_reduce(
    d_tensor: DTensor,
    mesh_dims: Optional[Union[int, List[int]]] = None,
    async_op: bool = True,
) -> DTensor:
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

    return d_tensor.redistribute(device_mesh, tuple(dst_placements), async_op)


def vescale_reduce_scatter(
    d_tensor: DTensor,
    reduce_mesh_dims: Optional[Union[int, List[int]]] = None,
    scatter_dims: Union[int, List[int]] = None,
    mesh_dims: Union[int, List[int]] = None,
    async_op: bool = True,
) -> DTensor:
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

    return d_tensor.redistribute(device_mesh, tuple(dst_placements), async_op)
