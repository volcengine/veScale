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
import math

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import Placement, Shard, Replicate, Partial, _StridedShard, _Partial

from vescale.dtensor._collective_utils import mesh_scatter_ragged
from vescale.dtensor._dtensor_spec import DTensorSpec, TensorMeta
from vescale.utils.monkey_patch import patch_method

__all__ = [
    "Placement",
    "Shard",
    "Replicate",
    "Partial",
    "RaggedShard",
    "_StridedShard",
    "_Partial",
    "is_ragged_shard",
    "DTensorSpec",
    "TensorMeta",
    "_StridedRaggedShard",
]

"""
In this file, we add RaggedShard and is_ragged_shard method.
"""


@patch_method(target=Placement, method_name="is_ragged_shard")
def is_ragged_shard(self) -> bool:
    return isinstance(self, RaggedShard)


@dataclass(frozen=True)
class RaggedShard(Placement):
    """
    The ``RaggedShard`` placement specifies DTensor sharding based on the
    flattened storage of the tensor.

    Args:
        dims (tuple[int]): The dimension(s) to shard. For example, if a tensor
            of shape (n, m, k) is ragged-sharded with dims=(0,), then each
            ``local_tensor.numel()`` is guaranteed to be a multiple of (m * k).
        local_units (list[int]): A list of integers specifying the relative
            allocation of elements across shards. The length of this list must
            equal ``mesh.size(dim)``. For example, ``[1, 2, 1, 1]`` assigns
            20% of elements to device 0, 40% to device 1, and 20% total to
            devices 2 and 3.
    """

    dims: tuple[int, ...]
    local_units: tuple[int, ...]

    def _split_tensor(
        self,
        tensor: torch.Tensor,
        num_chunks: int,
    ) -> list[torch.Tensor]:
        """
        Split *flattened* ``tensor`` into ``num_chunks`` shards whose sizes are
        dictated by ``self.local_units``.
        """
        # Preconditions
        assert tensor.is_contiguous(), "RaggedShard expects a contiguous tensor."
        assert num_chunks == len(self.local_units), "num_chunks must equal len(local_units)."

        total_numel = tensor.numel()
        ratio = total_numel // sum(self.local_units)
        assert total_numel % sum(self.local_units) == 0, "Sum of local_units must be a divisor of tensor.numel()."
        flat_tensor = tensor.view(-1)

        start_idx = 0
        shard_list: list[torch.Tensor] = []
        for shard_len in self.local_units:
            shard = flat_tensor.narrow(0, start_idx, shard_len * ratio)
            start_idx += shard_len * ratio
            shard_list.append(shard)

        return shard_list

    def _ragged_shard_tensor(
        self,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        src_data_rank: int | None = 0,
    ) -> torch.Tensor:
        """
        shard and scatter a tensor on a mesh dimension (use coordinate
        0 on the mesh dimension as source of truth)
        """
        my_coordinate = mesh.get_coordinate()
        num_chunks = mesh.size(mesh_dim=mesh_dim)
        assert len(self.local_units) == num_chunks, "len(local_units) must equal number of ranks on the mesh dimension."
        assert tensor.numel() % sum(self.local_units) == 0, (
            "Sum of local_units must be divisible by total number of elements in tensor."
        )

        if my_coordinate is None:
            # if rank is not part of mesh, we simply return an empty tensor
            return tensor.new_empty(0, requires_grad=tensor.requires_grad)

        mesh_dim_local_rank = my_coordinate[mesh_dim]

        if src_data_rank is None:
            # src_data_rank specified as None explicitly means to skip the
            # communications, simply split
            scatter_list = self._split_tensor(tensor, num_chunks)
            return scatter_list[mesh_dim_local_rank]

        scatter_list = self._split_tensor(tensor, num_chunks)
        output = torch.empty_like(scatter_list[mesh_dim_local_rank])

        mesh_scatter_ragged(output, scatter_list, mesh, mesh_dim=mesh_dim, group_src=src_data_rank)
        return output

    def _to_replicate_tensor(
        self,
        local_tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        current_logical_shape: list[int],
    ) -> torch.Tensor:
        """
        Replicate the local tensor to all ranks.
        """
        tensor_lst = []
        tot = sum(self.local_units)
        logical_numel = math.prod(current_logical_shape)
        assert logical_numel % tot == 0, "current_logical_shape must be divisible by tot"
        for i in range(mesh.size(mesh_dim)):
            length = logical_numel // tot * self.local_units[i]
            tensor_lst.append(torch.zeros(length, dtype=local_tensor.dtype, device=local_tensor.device))
        torch.distributed.all_gather(
            tensor_lst,
            local_tensor,
            group=mesh.get_group(mesh_dim),
        )
        return torch.cat(tensor_lst)

    def _to_new_ragged_shard(
        self,
        local_tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        current_logical_shape: list[int],
        new_local_units: tuple[int, ...],
    ) -> torch.Tensor:
        numel = math.prod(current_logical_shape)
        src_factor, dst_factor = numel // sum(self.local_units), numel // sum(new_local_units)
        src_slice = tuple(u * src_factor for u in self.local_units)
        dst_slice = tuple(u * dst_factor for u in new_local_units)
        coord = mesh.get_coordinate()[mesh_dim]

        input_tensor_list = []
        left = sum(src_slice[:coord])
        right = left + src_slice[coord]
        for i in range(len(new_local_units)):
            li = sum(dst_slice[:i])
            ri = li + dst_slice[i]
            if ri <= left or li >= right:
                input_tensor_list.append(torch.empty(0, dtype=local_tensor.dtype, device=local_tensor.device))
            else:
                length = min(ri, right) - max(li, left)
                input_tensor_list.append(local_tensor.narrow(0, max(li, left) - left, length))

        output_tensor_list = []
        left = sum(dst_slice[:coord])
        right = left + dst_slice[coord]
        for i in range(len(new_local_units)):
            li = sum(src_slice[:i])
            ri = li + src_slice[i]
            length = max(0, min(ri, right) - max(li, left))
            output_tensor_list.append(torch.zeros(length, dtype=local_tensor.dtype, device=local_tensor.device))

        torch.distributed.all_to_all(
            output_tensor_list,
            input_tensor_list,
            group=mesh.get_group(mesh_dim),
        )
        return torch.cat(output_tensor_list)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RaggedShard):
            return False
        return self.dims == other.dims and self.local_units == other.local_units

    def __hash__(self) -> int:
        return hash((self.dims, self.local_units))

    def __repr__(self) -> str:
        """
        machine readable representation of the Shard placement
        """
        return f"RaggedShard(dims={self.dims}, local_units={self.local_units})"

    def __str__(self) -> str:
        """human readable representation of the Shard placement"""
        return f"RaggedShard(dims={self.dims}, local_units={self.local_units})"

    def reconstruct_tensor_from_flat(
        self,
        flat_tensor: torch.Tensor,
        shape: tuple[int, ...],
    ) -> torch.Tensor:
        assert flat_tensor.ndim == 1, "flat_tensor must be 1-dimensional"

        ndim = len(self.dims)
        if self.dims != tuple(range(ndim)):
            raise ValueError(f"dims must be [0,1,…,i-1], got {self.dims}")
        assert flat_tensor.numel() % math.prod(shape[ndim:]) == 0, (
            "flat_tensor.numel() must be the same as math.prod(shape)"
        )
        return flat_tensor.view(-1, *shape[ndim:])


@dataclass(frozen=True)
class _StridedRaggedShard(RaggedShard):
    split_factor: int

    # TODO(jiacheng) this is buggy. Avoid using it in redistribute.
    # def _to_replicate_tensor(
    #     self,
    #     local_tensor: torch.Tensor,
    #     mesh: DeviceMesh,
    #     mesh_dim: int,
    #     current_logical_shape: list[int],
    # ) -> torch.Tensor:
    #     """
    #     Replicate the local tensor to all ranks, add reshuffle.
    #     """
    #     tensor_lst = []
    #     rank = mesh.get_coordinate()[mesh_dim]
    #     tot = sum(self.local_units)
    #     logical_numel = math.prod(current_logical_shape)
    #     assert logical_numel % tot == 0, "current_logical_shape must be divisible by tot"
    #     for i in range(mesh.size(mesh_dim)):
    #         length = logical_numel // tot * self.local_units[i]
    #         tensor_lst.append(torch.zeros(length, dtype=local_tensor.dtype, device=local_tensor.device))
    #     torch.distributed.all_gather(
    #         tensor_lst,
    #         local_tensor,
    #         group=mesh.get_group(mesh_dim),
    #     )
    #     output = torch.cat(tensor_lst)
    #     # reshuffle.
    #     assert logical_numel % (self.split_factor * tot) == 0, (
    #         f"logical_numel {logical_numel} must be divisible by split_factor * sum(local_units) = {self.split_factor * tot}"
    #     )
    #     tensor_lst = []
    #     for j in range(self.split_factor):
    #         offset = 0
    #         for i in range(mesh.size(mesh_dim)):
    #             ragged_len = logical_numel // tot // self.split_factor * self.local_units[i]
    #             tensor_lst.append(output[offset + j * ragged_len : offset + (j + 1) * ragged_len])
    #             offset += ragged_len * self.split_factor
    #     return torch.cat(tensor_lst)
