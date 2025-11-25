################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

from typing import List
import torch
from vescale.dtensor.placement_types import DTensorSpec, Partial, Placement, Replicate, Shard

from abc import abstractmethod, ABCMeta

from vescale.emulator.device_mesh import DeviceMesh
from vescale.emulator.mesh_collectives import mesh_all_gather, mesh_all_reduce, mesh_broadcast, mesh_scatter
from vescale.emulator.utils import torch_reduce_op_to_emulator


def _replicate_tensor(tensors: List[torch.Tensor], mesh: DeviceMesh, mesh_dim: int) -> torch.Tensor:
    """
    Replicate (broadcast) a list of torch.Tensor on a mesh dimension (use
    the first coordinate on the mesh dimension as source of truth)
    """
    for i, tensor in enumerate(tensors):
        tensors[i] = tensor.contiguous()
    tensors = mesh_broadcast(tensors, mesh, mesh_dim=mesh_dim)
    return tensors


def _reshard_to_replicate_with_pad_one_dim(
    local_tensors: List[torch.Tensor], size_list: List[torch.Size], mesh: DeviceMesh, mesh_dim: int, shard_dim: int
) -> List[torch.Tensor]:
    """
    This function all_gather all shards and return a list of tensors that
    is replicated on the previously sharded mesh dimension
    """
    num_chunks = mesh.size(dim=mesh_dim)

    tag_rank_list = mesh._dim_group_infos[mesh_dim]
    dim_group = mesh.get_dim_groups()[mesh_dim]

    is_padded_list = [0 for _ in range(len(local_tensors))]
    pad_sizes_list = [0 for _ in range(len(local_tensors))]

    for (tag, ranks), pg in zip(tag_rank_list, dim_group):
        for i, rank in enumerate(ranks):
            local_tensor = local_tensors[rank]
            size = size_list[rank]

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

            is_padded_list[rank] = is_padded
            pad_sizes_list[rank] = pad_sizes

            pad_size = pad_sizes[i]

            if pad_size > 0:
                local_tensor = _pad_tensor_on_shard_dim(local_tensor, pad_size, shard_dim)
            local_tensors[rank] = local_tensor.contiguous()

    results = mesh_all_gather(
        local_tensors,
        mesh,
        shard_dim,
        mesh_dim,
    )
    # Unpad the tensor if the input tensor was padded
    for (tag, ranks), pg in zip(tag_rank_list, dim_group):
        for i, rank in enumerate(ranks):
            is_padded = is_padded_list[rank]
            if is_padded:
                pad_sizes = pad_sizes_list[rank]
                result = results[rank]

                full_pad_size = sum(pad_sizes)
                result = _unpad_tensor_on_shard_dim(result, full_pad_size, shard_dim)

    return results


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


def _scatter_tensor_by_shard(
    tensors: List[torch.Tensor], mesh: DeviceMesh, mesh_dim: int, shard_spec: Shard
) -> torch.Tensor:
    """
    shard and scatter a list of tensor on a mesh dimension (use coordinate
    0 on the mesh dimension as source of truth)
    """
    scatter_list_list = [0 for _ in range(len(tensors))]
    pad_sizes_list = [0 for _ in range(len(tensors))]
    outputs = [0 for _ in range(len(tensors))]

    tag_rank_list = mesh._dim_group_infos[mesh_dim]
    dim_group = mesh.get_dim_groups()[mesh_dim]

    for (tag, ranks), pg in zip(tag_rank_list, dim_group):
        for i, rank in enumerate(ranks):
            tensor = tensors[rank]
            scatter_list, pad_sizes = shard_spec._split_tensor(
                tensor, num_chunks=mesh.size(dim=mesh_dim), with_padding=True, contiguous=True
            )
            output = torch.empty_like(scatter_list[i])
            scatter_list_list[rank] = scatter_list
            pad_sizes_list[rank] = pad_sizes
            outputs[rank] = output
    mesh_scatter(outputs, scatter_list_list, mesh, mesh_dim=mesh_dim)

    # Only unpad if the local_tensor was padded on the dimension.
    for (tag, ranks), pg in zip(tag_rank_list, dim_group):
        for i, rank in enumerate(ranks):
            pad_sizes = pad_sizes_list[rank]
            pad_size = pad_sizes[i]
            if pad_size > 0:
                output = outputs[rank]
                outputs[rank] = shard_spec._unpad_tensor(output, pad_size)
    return outputs


class BaseRedistributeFunc(metaclass=ABCMeta):
    def __init__(self, global_shape: torch.Size = None):
        self.global_shape = global_shape

    def get_or_compute_global_shape(
        self, local_tensors, shard_placement: Shard, mesh: DeviceMesh, shard_mesh_dim: int
    ) -> List[torch.Size]:
        if self.global_shape is not None:
            return [self.global_shape for _ in local_tensors]
        # get the shape from annotation is another option
        from vescale.plan.hooks.annotate_hook import ANNOT_NAME

        annot_shape_list = []
        for local_tensor in local_tensors:
            annot_spec = getattr(local_tensor, ANNOT_NAME, None)
            if (annot_spec is not None) and (annot_spec.shape is not None):
                annot_shape_list.append(annot_spec.shape)
            else:
                break
        if len(annot_shape_list) == len(local_tensors):
            return annot_shape_list

        tag_rank_list = mesh._dim_group_infos[shard_mesh_dim]
        dim_group = mesh.get_dim_groups()[shard_mesh_dim]

        global_shape_list = [0 for _ in range(len(local_tensors))]
        for (tag, ranks), pg in zip(tag_rank_list, dim_group):
            first = ranks[0]
            global_shape = list(first.shape)
            for rank in ranks[1:]:
                local_tensor = local_tensors[rank]
                shape = list(local_tensor.shape)
                global_shape[shard_placement.dim] += shape[shard_placement.dim]

            global_shape = torch.Size(global_shape)

            for rank in ranks:
                global_shape_list[rank] = global_shape
        return global_shape_list

    @abstractmethod
    def name(self): ...

    @abstractmethod
    def __call__(
        self,
        tensors: List[torch.Tensor],
        current_placement: Placement,
        target_placement: Placement,
        mesh: DeviceMesh,
        mesh_dim: int = 0,
    ) -> List[torch.Tensor]: ...

    @torch.no_grad()
    def __fake_call__(
        self,
        global_tensors: List[torch.Tensor],
        current_placement: Placement,
        target_placement: Placement,
        mesh: DeviceMesh,
        mesh_dim: int = 0,
    ):
        # by default, we return a contiguous output.
        return [global_tensor.new_empty(global_tensor.shape) for global_tensor in global_tensors]


class R2R(BaseRedistributeFunc):
    def name(self):
        return "R->R"

    @torch.no_grad()
    def __call__(
        self,
        tensors: List[torch.Tensor],
        current_placement: Replicate,
        target_placement: Replicate,
        mesh: DeviceMesh,
        mesh_dim: int = 0,
        emit_comm: bool = False,
    ):
        if not emit_comm:
            return tensors
        return _replicate_tensor(tensors=tensors, mesh=mesh, mesh_dim=mesh_dim)

    @torch.no_grad()
    def __fake_call__(
        self,
        global_tensors: List[torch.Tensor],
        current_placement: Placement,
        target_placement: Placement,
        mesh: DeviceMesh,
        mesh_dim: int = 0,
        emit_comm: bool = False,
    ):
        if not emit_comm:
            return global_tensors
        # contiguous output
        return [global_tensor.new_empty(global_tensor.shape) for global_tensor in global_tensors]


class R2S(BaseRedistributeFunc):
    def name(self):
        return "R->S"

    @torch.no_grad()
    def __call__(
        self,
        tensors: List[torch.Tensor],
        current_placement: Replicate,
        target_placement: Shard,
        mesh: DeviceMesh,
        mesh_dim: int = 0,
        emit_comm: bool = False,
    ):
        if emit_comm:
            return _scatter_tensor_by_shard(tensors=tensors, mesh=mesh, mesh_dim=mesh_dim, shard_spec=target_placement)
        return _scatter_tensor_by_shard(tensors=tensors, mesh=mesh, mesh_dim=mesh_dim, shard_spec=target_placement)

    @torch.no_grad()
    def __fake_call__(
        self,
        global_tensors: List[torch.Tensor],
        current_placement: Placement,
        target_placement: Placement,
        mesh: DeviceMesh,
        mesh_dim: int = 0,
        emit_comm: bool = False,
    ):
        if not emit_comm:
            return global_tensors
        # contiguous output
        return [global_tensor.new_empty(global_tensor.shape) for global_tensor in global_tensors]


class P2R(BaseRedistributeFunc):
    def name(self):
        return "P->R"

    @torch.no_grad()
    def __call__(
        self,
        tensors: List[torch.Tensor],
        current_placement: Partial,
        target_placement: Replicate,
        mesh: DeviceMesh,
        mesh_dim: int = 0,
    ):
        reduce_op = torch_reduce_op_to_emulator(current_placement.reduce_op)
        result = mesh_all_reduce(
            tensors=tensors,
            mesh=mesh,
            reduce_op=reduce_op,
            mesh_dim=mesh_dim,
        )
        return result


class S2R(BaseRedistributeFunc):
    def name(self):
        return "S->R"

    @torch.no_grad()
    def __call__(
        self,
        local_tensors: List[torch.Tensor],
        current_placement: Shard,
        target_placement: Replicate,
        mesh: DeviceMesh,
        mesh_dim: int = 0,
    ):
        def normalize_dim_for_shard(placement, tensor):
            if placement.dim >= 0:
                return placement
            tensor_ndim = tensor.ndim if isinstance(tensor, torch.Tensor) else tensor
            if tensor_ndim == 0:
                return placement
            new_dim = placement.dim + tensor_ndim
            return Shard(new_dim)

        current_placement = normalize_dim_for_shard(current_placement, tensor=local_tensors[0])
        return _reshard_to_replicate_with_pad_one_dim(
            local_tensors,
            self.get_or_compute_global_shape(local_tensors, current_placement, mesh, mesh_dim),
            mesh,
            mesh_dim,
            current_placement.dim,
        )


def get_redistribute_fn(
    current_spec: DTensorSpec,
    target_spec: DTensorSpec,
    current_placement: Placement,
    target_placement: Placement,
) -> BaseRedistributeFunc:
    # P ->
    if current_placement.is_partial():
        if target_placement.is_replicate():
            return P2R()
    # S ->
    elif current_placement.is_shard():
        if target_placement.is_replicate():
            return S2R(global_shape=current_spec.shape if current_spec.tensor_meta is not None else None)
    # R ->
    elif current_placement.is_replicate():
        if target_placement.is_replicate():
            return R2R()
        elif target_placement.is_shard():
            return R2S()
    raise RuntimeError(f"redistribute from {current_placement} to {target_placement} not supported yet")
