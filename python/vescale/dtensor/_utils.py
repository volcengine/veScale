################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

import warnings
from typing import List, Sequence, Tuple, Optional, Dict, Set

import torch
import torch.distributed._functional_collectives as funcol
from torch._prims_common import ShapeType

from vescale.dtensor.device_mesh import DeviceMesh
from vescale.dtensor.dtensor import DTensor
from vescale.dtensor.placement_types import InterleavedShard, Partial, Placement, Replicate, Shard
from vescale.dtensor._collective_utils import mesh_all_gather


def compute_local_shape(global_shape: ShapeType, mesh: DeviceMesh, placements: Sequence[Placement]) -> Tuple[int, ...]:
    """
    Compute the shape of a local shard of the given DTensor on its current
    coordinate of the mesh.
    """
    my_coordinate = mesh.get_coordinate()

    if my_coordinate is None:
        # if rank not in the mesh, return empty shape
        return ()
    else:
        local_shape = list(global_shape)  # start with global shape
        ndim = len(global_shape)
        for idx, placement in enumerate(placements):
            mesh_dim_size = mesh.size(idx)
            if isinstance(placement, (Shard, InterleavedShard)):
                shard_dim = placement.dim
                assert shard_dim < ndim, f"Sharding dim {shard_dim} greater than tensor ndim {ndim}"
                local_shard_size, _ = placement._local_shard_size_on_dim(
                    local_shape[shard_dim], mesh_dim_size, my_coordinate[idx]
                )
                assert isinstance(local_shard_size, int)
                local_shape[shard_dim] = local_shard_size

        return tuple(local_shape)


def compute_local_shape_and_global_offset(
    global_shape: ShapeType, mesh: DeviceMesh, placements: Sequence[Placement]
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Compute the local tensor shape and the global offsets into the original tensor
    of a DTensor on its current global rank. This is useful for checkpointing purpose.
    Example (2 host with 4GPUs each):
    # Below is a DeviceMesh with mesh_shape of (2, 4)
    mesh = DeviceMesh(device_type="cuda",
                        mesh=[
                        [0, 1, 2, 3],
                        [4, 5, 6, 7]
                        ],
    )
    Let's say we distribute a global_tensor of shape (8,4) over the above DeviceMesh
    with a placements of [Shard(0), Shard(0)].
    The local shape and global offset will be as follows:
    rank0 -- local_shape:[1, 4], global_offset:[0, 0]
    rank1 -- local_shape:[1, 4], global_offset:[1, 0]
    rank2 -- local_shape:[1, 4], global_offset:[2, 0]
    rank5 -- local_shape:[1, 4], global_offset:[5, 0]
    rank3 -- local_shape:[1, 4], global_offset:[3, 0]
    rank4 -- local_shape:[1, 4], global_offset:[4, 0]
    rank6 -- local_shape:[1, 4], global_offset:[6, 0]
    rank7 -- local_shape:[1, 4], global_offset:[7, 0]
    """
    my_coordinate = mesh.get_coordinate()

    if my_coordinate is None:
        # if rank not in the mesh, return empty offset
        return ((), ())
    else:
        local_shape = list(global_shape)
        global_offset = [0] * len(global_shape)

        for idx, placement in enumerate(placements):
            mesh_dim_size = mesh.size(idx)
            if isinstance(placement, (Shard, InterleavedShard)):
                shard_dim = placement.dim
                local_offset = [0] * len(global_shape)
                assert shard_dim < len(
                    local_shape
                ), f"Sharding dim {shard_dim} greater than tensor ndim {len(local_shape)}"
                # TODO: what if placement is InterleavedShard
                shard_size, shard_offset = placement._local_shard_size_on_dim(
                    local_shape[shard_dim],
                    mesh_dim_size,
                    my_coordinate[idx],
                    return_offset=True,
                )

                local_shape[shard_dim] = shard_size
                local_offset[shard_dim] = shard_offset

                # On a given dimension, if the local_offset[shard_dim] is smaller than global_offset[shard_dim],
                # it means that this dimension has been already sharded in previous placement.
                # Therefore, we cannot simply replace the global_offset[shard_dim] with local_offset[shard_dim].
                # Instead, for the given shard_dim, we need to add local_offset[shard_dim] to existing global_offset[shard_dim].
                if global_offset[shard_dim] <= local_offset[shard_dim]:
                    global_offset[shard_dim] = local_offset[shard_dim]
                else:
                    global_offset[shard_dim] += local_offset[shard_dim]

        return tuple(local_shape), tuple(global_offset)


def is_same_shape_across_ranks(tensor_shape: ShapeType, device_mesh: DeviceMesh, placements: Sequence[Placement]):
    # check if tensor shapes are the same across ranks
    self_shape = torch.tensor([tuple(tensor_shape)], dtype=torch.int64, device=device_mesh.device_type)
    for mesh_dim, _ in enumerate(placements):  # TODO for perf: use a process group for the entire DeviceMesh
        all_shapes = mesh_all_gather(
            self_shape,
            torch.Size([device_mesh.size(mesh_dim), self_shape.size(1)]),
            device_mesh,
            scatter_dim=0,
            mesh_dim=mesh_dim,
        )
        if not torch.all(self_shape == all_shapes):
            return False
    return True


def gather_local_tensor_shape(
    self_local_tensor: torch.Tensor, device_mesh: DeviceMesh, placements: Sequence[Placement], shard_only: bool = True
) -> Optional[Dict[int, List[List[int]]]]:
    """All gather local tensor shapes per mesh dimension.
    When `shard_only is True`, all gather only sharded mesh dim."""
    if device_mesh.get_coordinate() is None:  # if rank is not part of mesh
        return None

    self_local_shape = torch.tensor([list(self_local_tensor.shape)], dtype=torch.int64, device=device_mesh.device_type)
    meshdim_localtensor_shape = {}
    for mesh_dim, place in enumerate(placements):
        if shard_only and not isinstance(place, (Shard, InterleavedShard)):
            continue
        stacked_local_shape = mesh_all_gather(
            self_local_shape,
            torch.Size([device_mesh.size(mesh_dim), self_local_shape.size(1)]),
            device_mesh,
            scatter_dim=0,
            mesh_dim=mesh_dim,
        )
        if type(stacked_local_shape) is funcol.AsyncCollectiveTensor:
            # synchronously wait for any pending collectives to get the result tensor
            stacked_local_shape = stacked_local_shape.trigger_wait()
            stacked_local_shape = stacked_local_shape.elem  # type: ignore[attr-defined]
        meshdim_localtensor_shape[mesh_dim] = stacked_local_shape.detach().cpu().tolist()
    return meshdim_localtensor_shape


def compute_global_tensor_info(
    tensor: torch.Tensor,
    mesh: DeviceMesh,
    placements: Sequence[Placement],
    meshdim_localtensor_shape: Optional[Dict[int, List[List[int]]]] = None,
) -> Tuple[List[int], List[int]]:
    """
    Compute the global size and stride of a DTensor from the given local tensor.

    When `meshdim_localtensor_shape` is None (be default):
        The local size is multiplited by `world_size` per Sharding dim.
        The local stride is multiplited by `world_size` per Sharding dim, as long as the
        dimension is outside sharding dim.

        For example, if we have a local tensor with size (4, 8, 2) and stride (16, 1, 8).
        If the DTensor placements are [Shard(2)] and world_size is 2;
        then the global size is (4, 8, 4) and stride is (16 * 2, 1, 8).

    When `meshdim_localtensor_shape` is provided:
        All local sizes are summed togather as global Sharding dim.
        The local stride is scaled by global Sharding dim divided by local Sharding dim,
        as long as the dimension is outside sharding dim.

        For example, if we have a local tensor with size (4, 8, 2) and stride (16, 1, 8) on rank0,
        and a local tensor with size (4, 8, 1) and stride (8, 1, 8) on rank1.
        If the DTensor placements are [Shard(2)] and world_size is 2;
        then the global size is (4, 8, 3) and stride is (8 * 3, 1, 8).

    Args:
        tensor (:class:`torch.Tensor`):
            Local tensor which DTensor will be constructed from.
        mesh (:class:`DeviceMesh`):
            Object which describes the mesh topology of devices for the DTensor.
        placements (Sequence[:class:`Placement`]]):
            The attribute of the DTensor that describes its layout on the mesh topology.
        meshdim_localtensor_shape (:class:`Dict[int, List[List[int]]]`):
            Default None.
            Otherwise, a given list for local tensor shapes per device mesh dim.

    Return:
        tensor_shape: A List of int which specifies the global size of DTensor which build
            on top of the local tensor.
        tensor_stride: A List of int which specifies the global stride of DTensor.
    """
    if meshdim_localtensor_shape is None:  # assume even sharding (contiguous or non-contiguous)
        tensor_shape = list(tensor.size())
        tensor_stride = list(tensor.stride())
    else:  # support uneven sharding (contiguous only)
        if not tensor.is_contiguous():
            warnings.warn(
                "`from_local` take non-contiguous local tensor, which is not supported in uneven sharding. Treat as contiguous.",
                UserWarning,
            )
        # a meta empty tensor is created for obtaining correct local stride,
        # especially when local tensor is non-contiguous or narrowed from padding or zero dimmed.
        # TODO: rethink supporting non-contiguous local tensor which is narrowed from padding or zero dimmed.
        tensor = torch.empty(tensor.shape, dtype=tensor.dtype, device="meta")
        tensor_shape = list(tensor.size())
        tensor_stride = list(tensor.stride())

    # record occured shard dim
    shard_dim_occured: Set[int] = set()

    for idx, placement in enumerate(placements):
        mesh_dim_size: int = mesh.size(idx)

        # TODO: rethink about this InterleavedShard.
        if placement.is_shard() or placement.is_interleaved_shard():
            if placement.dim < 0:
                placement.dim += len(tensor_shape)
            shard_dim = placement.dim

            assert (
                shard_dim < tensor.ndim
            ), f"Sharding dim {shard_dim} greater than tensor ndim {tensor.ndim} for placement number {idx}."

            if shard_dim in shard_dim_occured:
                warnings.warn(
                    "Sharding the same tensor dim is not supported for uneven sharding. Treat as even sharding.",
                    UserWarning,
                )
                is_shard_same_dim = True
            else:
                shard_dim_occured.add(shard_dim)
                is_shard_same_dim = False

            # recover global shape
            local_dim_size = tensor_shape[shard_dim]
            if meshdim_localtensor_shape is None or is_shard_same_dim:
                # duplicate local shape at this sharded dim as global shape
                tensor_shape[shard_dim] = local_dim_size * mesh_dim_size
            else:
                # concat local shapes at this sharded dim as global shape
                global_dim_size = sum(shape[shard_dim] for shape in meshdim_localtensor_shape[idx])
                tensor_shape[shard_dim] = global_dim_size

            # recover tensor stride by modifying the stride that larger than
            # the current stride on the shard_dim
            for i in range(len(tensor_stride)):
                if i != shard_dim and tensor_stride[i] >= tensor_stride[shard_dim]:
                    # rescale the stride by the shard size
                    if meshdim_localtensor_shape is None or is_shard_same_dim:
                        tensor_stride[i] = tensor_stride[i] * mesh_dim_size
                    else:
                        if local_dim_size == 0:
                            tensor_stride[i] *= global_dim_size
                        else:
                            assert tensor_stride[i] % local_dim_size == 0
                            tensor_stride[i] = tensor_stride[i] // local_dim_size * global_dim_size

        elif not isinstance(placement, (Replicate, Partial)):
            raise RuntimeError(f"placement type {type(placement)} not supported!")

    return tensor_shape, tensor_stride


def is_zero_out_local_shard(mesh: DeviceMesh, placements: Sequence[Placement]) -> bool:
    """
    Compute whether we need to zero out the local shard of current rank, for Partial().

    e.g. we want a bias tensor in [Partial(), Shard(0), Partial()]
        [ [[b1, 0.]
           [b2, 0.]]

          [[0., 0.]
           [0., 0.]] ]
        on a 3D-DeviceMesh:
        [ [[0, 1]
           [2, 3]]

          [[4, 5]
           [6, 7]] ]
        The computed result should be:
        [ [[False, True]
           [False, True]]

          [[True, True]
           [True, True]] ]
    """
    my_coordinate = mesh.get_coordinate()

    if my_coordinate is None:  # if rank not in the mesh, nothing to zero out
        return False

    for idx, placement in enumerate(placements):
        if not placement.is_partial():
            continue
        # we zero out all other ranks of the current mesh dim
        # and leave only src-of-truth rank 0 have the data, to perform a "zero cost" shard.
        if my_coordinate[idx] != 0:
            return True

    return False


def _equal_meta_data(dt1: DTensor, dt2: DTensor, exact_device: bool) -> bool:
    if type(dt1) is not DTensor or type(dt2) is not DTensor:
        return False
    # check itself
    if exact_device and (dt1.device.type != dt2.device.type):
        return False
    if dt1.shape != dt2.shape:
        return False
    if dt1.dtype != dt2.dtype:
        return False
    if dt1.layout != dt2.layout:  # torch.strided (dense) or torch.sparse_*
        return False
    if dt1.stride() != dt2.stride():
        return False
    if dt1.requires_grad != dt2.requires_grad:
        return False
    # check global spec
    if exact_device:
        if dt1._spec.mesh != dt2._spec.mesh:
            return False
    else:
        if not dt1._spec.mesh.mesh.equal(dt2._spec.mesh.mesh):
            return False
    if dt1._spec.placements != dt2._spec.placements:
        return False
    if dt1._spec.tensor_meta != dt2._spec.tensor_meta:
        return False
    # check local tensor (ref: https://github.com/pytorch/pytorch/blob/63ae1051e17b1cf4fe55ac6b6f17c16672d44150/aten/src/ATen/native/cuda/Equal.cpp#L15)
    t1, t2 = dt1._local_tensor, dt2._local_tensor
    if exact_device and (t1.device.type != t2.device.type):
        return False
    if t1.shape != t2.shape:
        return False
    if t1.dtype != t2.dtype:
        return False
    if t1.layout != t2.layout:  # torch.strided (dense) or torch.sparse_*
        return False
    if t1.is_contiguous() != t2.is_contiguous():
        return False
    if t1.stride() != t2.stride():
        return False
    if t1.storage_offset() != t2.storage_offset():
        return False
    if t1.requires_grad != t2.requires_grad:
        return False
    return True


def equal(dt1: DTensor, dt2: DTensor, exact_device: bool = True) -> bool:
    """
    check if two DTensors are 'exactly' equal
    """
    if not _equal_meta_data(dt1, dt2, exact_device):
        return False
    if dt1.is_meta and dt2.is_meta:
        return True
    if exact_device:
        return torch.equal(dt1._local_tensor, dt2._local_tensor)  # check value only
    else:
        return torch.equal(dt1._local_tensor.cpu(), dt2._local_tensor.cpu())  # check value only


def allclose(
    dt1: DTensor,
    dt2: DTensor,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
    exact_device: bool = True,
) -> bool:
    """
    check if two DTensors are 'allclose'
    """
    if not _equal_meta_data(dt1, dt2, exact_device):
        return False
    if dt1.is_meta and dt2.is_meta:
        return True
    if exact_device:
        return torch.allclose(
            dt1._local_tensor, dt2._local_tensor, rtol=rtol, atol=atol, equal_nan=equal_nan
        )  # check value only
    else:
        return torch.allclose(
            dt1._local_tensor.cpu(), dt2._local_tensor.cpu(), rtol=rtol, atol=atol, equal_nan=equal_nan
        )  # check value only


def compute_local_offset(global_shape: ShapeType, mesh: DeviceMesh, placements: Sequence[Placement]) -> Tuple[int, ...]:
    """
    Compute the offsets of a local shard of the given DTensor on its current
    global rank. This is mostly used by distributed checkpointing to know the
    exact offsets of the local shard.
    """
    my_coordinate = mesh.get_coordinate()

    if my_coordinate is None:
        # if rank not in the mesh, return empty offset
        return ()
    else:
        local_offsets = [0] * len(global_shape)
        local_shape = list(global_shape)

        for idx, placement in enumerate(placements):
            mesh_dim_size = mesh.size(idx)
            if isinstance(placement, Shard):
                shard_dim = placement.dim
                assert shard_dim < len(
                    local_shape
                ), f"Sharding dim {shard_dim} greater than tensor ndim {len(local_shape)}"
                shard_size, shard_offset = placement._local_shard_size_on_dim(
                    local_shape[shard_dim],
                    mesh_dim_size,
                    my_coordinate[idx],
                    return_offset=True,
                )
                local_shape[shard_dim] = shard_size
                local_offsets[shard_dim] = shard_offset
        return tuple(local_offsets)
