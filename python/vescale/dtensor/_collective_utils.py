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
from typing import List, Optional
import math

import torch
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
from torch.distributed.distributed_c10d import (
    GroupMember,
    ProcessGroup,
    Work,
    all_to_all,
    get_global_rank,
    get_rank,
    broadcast,
    get_world_size,
    scatter,
)

from vescale.dtensor.device_mesh import DeviceMesh, mesh_resources
from vescale.dtensor.placement_types import DTensorSpec


logger = logging.getLogger(__name__)
TORCH_VERSION_BIGGER_THAN_2_2 = torch.__version__ >= "2.2"


# NOTE: upstream are working to migrate the following three collective
# apis to be functional, pay attention to it.


def mesh_scatter(
    output: torch.Tensor,
    scatter_list: List[torch.Tensor],
    mesh: DeviceMesh,
    mesh_dim: int = 0,
    async_op: bool = False,
) -> Optional[Work]:
    """
    scatter a list of tensors to a device mesh dimension. We by default
    use the first rank of the mesh dimension as the source of truth, i.e
    for a 2d mesh [[0, 1], [2, 3]], if we scatter on mesh_dim = 1, we will
    scatter the tensor list on rank 0 to rank 0/1, and tensor list on rank
    2 to rank 2/3.

    Args:
        output (torch.Tensor): the tensor to receive the scattered list.
        scatter_list (List[torch.Tensor]): the tensor list to be scattered.
        mesh_dim (int, optional): indicate which mesh dimension we want
            to scatter on, we by default choose the first rank on the
            mesh dimension as source of truth.

    Returns:
        A :class:`Work` object
    """
    # TODO: Ideally we should use the meta tensor way
    # (to register a meta kernel for the collective op)
    # so that it would avoid the communication. Need to
    # remove the check below once that is done.
    if output.is_meta:
        return None
    dim_group = mesh.get_dim_groups(mesh_dim)
    assert isinstance(dim_group, ProcessGroup)
    # src need to be global rank
    src_for_dim = 0

    if dim_group is not GroupMember.WORLD:
        src_for_dim = get_global_rank(dim_group, 0)

    if src_for_dim == get_rank():
        fut = scatter(
            output,
            scatter_list=scatter_list,
            src=src_for_dim,
            group=dim_group,
            async_op=async_op,
        )
    else:
        fut = scatter(
            output,
            scatter_list=None,
            src=src_for_dim,
            group=dim_group,
            async_op=async_op,
        )

    return fut


# TODO: test uneven split on GLOO and NCCL


def mesh_all_to_all(
    output_tensor_list: List[torch.Tensor],
    input_tensor_list: List[torch.Tensor],
    mesh: DeviceMesh,
    mesh_dim: int = 0,
    async_op: bool = False,
) -> Optional[Work]:
    dim_group = mesh.get_dim_groups(mesh_dim)
    assert isinstance(dim_group, ProcessGroup)

    work = None
    # no direct dist.all_to_all support on 'gloo' so we manually do scatters
    if mesh.device_type == "cpu":
        logger.warning("ProcessGroupGloo does not support all_to_all, falling back with scatters!")
        # TODO: pull the handle of uneven case in #492
        dim_group_size = get_world_size(dim_group)
        for i in range(dim_group_size):
            # src need to be global rank
            src_for_dim = i
            if dim_group is not GroupMember.WORLD:
                src_for_dim = get_global_rank(dim_group, i)

            work = scatter(
                output_tensor_list[i],
                input_tensor_list if mesh.get_rank() == src_for_dim else [],
                group=dim_group,
                src=src_for_dim,
                async_op=async_op,
            )
    else:
        work = all_to_all(
            output_tensor_list,
            input_tensor_list,
            dim_group,
            async_op=async_op,
        )
    return work


def mesh_broadcast(
    tensor: torch.Tensor,
    mesh: DeviceMesh,
    mesh_dim: int = 0,
    async_op=False,
) -> torch.Tensor:
    """
    broadcast the tensor to a device mesh dimension. We by default
    use the first rank of the mesh dimension as the source of truth, i.e
    for a 2d mesh [[0, 1], [2, 3]], if we broadcast on mesh_dim = 1, we will
    broadcast the tensor on rank 0 to rank 0/1, and tensor on rank 2
    to rank 2/3.

    Args:
        tensor (torch.Tensor): tensor to broadcast.
        mesh_dim (int, optional): indicate which mesh dimension we want
            to scatter on, we by default choose the first rank on the
            mesh dimension as source of truth.

    Returns:
        A :class:`Tensor` object
    """
    dim_group = mesh.get_dim_groups(mesh_dim)
    assert isinstance(dim_group, ProcessGroup)
    # src need to be global rank
    src_for_dim = 0
    if dim_group is not GroupMember.WORLD:
        src_for_dim = get_global_rank(dim_group, 0)
    if TORCH_VERSION_BIGGER_THAN_2_2:
        aysnc_tensor = funcol.broadcast(tensor, src=src_for_dim, group=dim_group)
        if not async_op:
            return funcol.wait_tensor(aysnc_tensor)
        return aysnc_tensor
    else:
        work = broadcast(tensor, src=src_for_dim, group=dim_group, async_op=async_op)
        if not async_op:
            return tensor
        from torch.distributed._functional_collectives_impl import _register_tensor_work
        from torch.distributed._functional_collectives import _maybe_wrap_tensor

        _register_tensor_work(tensor, work)
        return _maybe_wrap_tensor(tensor)


def mesh_reduce_scatter(
    tensor: torch.Tensor, mesh: DeviceMesh, reduce_op: c10d.ReduceOp.RedOpType, scatter_dim: int, mesh_dim: int
) -> torch.Tensor:
    """
    First peform all_reduce on the tensor, then split the tensor at scatter_dim
    and scatter them to a device mesh dimension.
    """
    my_coordinate = mesh.get_coordinate()
    num_chunks = mesh.size(dim=mesh_dim)

    if my_coordinate is None:
        # if rank is not part of mesh, simply return local_tensor,
        # which should be an empty tensor
        return tensor
    # for now, we only support that size at `scatter_dim`` is divisable by
    # the mesh size at `mesh_dim`
    assert (
        tensor.size(scatter_dim) % num_chunks == 0
    ), f"tensor size at {scatter_dim} is not divisable by the mesh size at {mesh_dim}"
    output = funcol.reduce_scatter_tensor(
        tensor, reduceOp=reduce_op.name, scatter_dim=scatter_dim, group=mesh._dim_group_infos[mesh_dim][1]
    )
    return output


def mesh_all_gather(
    tensor: torch.Tensor,
    global_size: torch.Size,
    mesh: DeviceMesh,
    scatter_dim: int,
    mesh_dim: int,
) -> torch.Tensor:
    """
    all_gather all shards and return a tensor that is replicated
    on the previously sharded mesh dimension
    """
    my_coordinate = mesh.get_coordinate()
    num_chunks = mesh.size(dim=mesh_dim)

    # for now, we only support that global size at `scatter_dim` is equal with
    # the multuple of mesh size at `mesh_dim` and local_tensor size at `scatter_dim`
    assert (
        tensor.size(scatter_dim) * num_chunks == global_size[scatter_dim]
    ), f"global tensor size at {scatter_dim} is not equal with the multiply of mesh size at {mesh_dim} and local_tensor size at {scatter_dim}"

    if my_coordinate is None:
        # if rank is not part of mesh, we simply return local_tensor,
        # which should be an empty tensor
        return tensor

    tensor = tensor.contiguous()
    output = funcol.all_gather_tensor(tensor, gather_dim=scatter_dim, group=mesh._dim_group_infos[mesh_dim][1])
    return output


def mesh_all_reduce(
    tensor: torch.Tensor,
    mesh: DeviceMesh,
    reduce_op: c10d.ReduceOp.RedOpType,
    mesh_dim: int,
) -> torch.Tensor:
    return funcol.all_reduce(tensor, reduceOp=reduce_op.name, group=mesh._dim_group_infos[mesh_dim][1])


def wait(tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(tensor, funcol.AsyncCollectiveTensor):
        return funcol.wait_tensor(tensor)
    return tensor


def spec_to_bytes(spec: DTensorSpec) -> int:
    assert spec.tensor_meta is not None, "spec should have tensor meta defined!"
    return spec.tensor_meta.dtype.itemsize * math.prod(spec.shape)


def get_bandwidth_factor(mesh: DeviceMesh) -> List[float]:
    # generate bandwidth factor for intra-host/inter-host communication pattern
    factors = [1.0] * mesh.ndim
    num_devices_per_host = mesh_resources.num_devices_per_host(mesh.device_type)

    num_devices = 1
    for mesh_dim in reversed(range(mesh.ndim)):
        num_devices *= mesh.size(mesh_dim)
        if num_devices <= num_devices_per_host:
            # magic number for intra-host communication bandwidth factor
            # TODO: see if we need to tweak this or offer a way for user
            # to specify the bandwidths
            factors[mesh_dim] = 0.2

    return factors


def allgather_cost(num_bytes: float, mesh: DeviceMesh, mesh_dim: int) -> float:
    num_devices_on_mesh_dim = mesh.size(mesh_dim)
    bandwidth_factor = get_bandwidth_factor(mesh)[mesh_dim]
    # constant latency factor + bandwidth cost
    return 1 + bandwidth_factor * num_bytes * (num_devices_on_mesh_dim - 1) / num_devices_on_mesh_dim


def allreduce_cost(num_bytes: float, mesh: DeviceMesh, mesh_dim: int) -> float:
    num_devices_on_mesh_dim = mesh.size(mesh_dim)
    bandwidth_factor = get_bandwidth_factor(mesh)[mesh_dim]
    # allreduce have 2x comm bytes compare to allgather/reduce_scatter
    return 1 + 2 * bandwidth_factor * num_bytes * (num_devices_on_mesh_dim - 1) / num_devices_on_mesh_dim


def reduce_scatter_cost(
    num_bytes: float,
    mesh: DeviceMesh,
    mesh_dim: int,
) -> float:
    num_devices_on_mesh_dim = mesh.size(mesh_dim)
    bandwidth_factor = get_bandwidth_factor(mesh)[mesh_dim]
    # constant latency factor + bandwidth cost
    return 1 + bandwidth_factor * num_bytes * (num_devices_on_mesh_dim - 1) / num_devices_on_mesh_dim


def redistribute_cost(
    current_spec: DTensorSpec,
    target_spec: DTensorSpec,
) -> float:
    """
    This function returns the cost of redistribute from current to target DTensorSpec.

    NOTE:
    1. Only consider communication cost here, since computation costs for redistribute
       are quite trival (i.e. we only need to narrow or simple division)
    2. Only consider redistribute cost on same mesh, cross mesh communication cost is
       not quite needed for operator strategy estimation/selection.
    """
    if current_spec.mesh != target_spec.mesh:
        # make infinite cost if meshes are not same
        # TODO: see if we want to support this once there's cross mesh communication
        return float("inf")

    if current_spec.is_replicated():
        # short-cut:
        # comm cost is 0 if current spec is already full replication
        return 0.0

    mesh = current_spec.mesh
    cost = 0.0
    comm_bytes = spec_to_bytes(current_spec) / current_spec.num_shards
    # Transformation that considered for redistribute cost:
    # 1. allgather 2. alltoall
    # 3. allreduce 4. reduce_scatter
    for i, (current, target) in enumerate(zip(current_spec.placements, target_spec.placements)):
        if current == target:
            continue
        if current.is_shard() and target.is_replicate():
            # allgather gives larger comm bytes
            comm_bytes *= mesh.size(i)
            # add up allgather comm cost
            cost += allgather_cost(comm_bytes, current_spec.mesh, i)
        elif current.is_shard() and target.is_shard():
            # should be alltoall comm, since we haven't implement it yet, add penalty
            # to favor allgather instead
            cost += allgather_cost(comm_bytes, current_spec.mesh, i) + 1.0
        elif current.is_partial() and target.is_replicate():
            # add up allreduce comm cost
            cost += allreduce_cost(comm_bytes, current_spec.mesh, i)
        elif current.is_partial() and target.is_shard():
            # add up reduce_scatter comm cost
            cost += reduce_scatter_cost(comm_bytes, current_spec.mesh, i)
            # after reduce_scatter the comm bytes for further collectives halved.
            comm_bytes /= mesh.size(i)
        elif current.is_shard() and target.is_partial():
            # ban shard/interleaved_shard -> partial as it does not make sense to perform
            # this redistribute
            return float("inf")

    return cost
