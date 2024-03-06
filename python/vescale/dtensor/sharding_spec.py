################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

import torch
import torch.distributed as dist
import torch.distributed._shard.sharded_tensor.metadata as sharded_tensor_meta
from dataclasses import dataclass
from typing import List, Union, TYPE_CHECKING
from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed._shard.sharded_tensor.shard import Shard
from torch.distributed._shard.sharding_spec.api import ShardingSpec
from torch.distributed._shard.sharded_tensor.utils import _parse_and_validate_remote_device


if TYPE_CHECKING:
    # Only include ShardedTensor when do type checking, exclude it
    # from run-time to resolve circular dependency.
    from torch.distributed._shard.sharded_tensor import ShardedTensor


def generate_placements(process_group=None):
    world = dist.get_world_size(process_group)
    backend = dist.get_backend(process_group)
    if backend == "nccl":
        return [
            f"rank:{i}/cuda:{dist.get_global_rank(process_group, i) % torch.cuda.device_count()}" for i in range(world)
        ]
    else:
        return [f"rank:{i}/cpu" for i in range(world)]


def build_unbalanced_spec(
    dim: int,
    shard_sizes: List[int],
    placements: List[Union[dist._remote_device, str]],
    process_group: dist.ProcessGroup = None,
):
    assert len(shard_sizes) == dist.get_world_size(
        process_group
    ), "Shard sizes must have equal length as group world size"
    return UnbalancedShardingSpec(dim=dim, placements=placements, shard_sizes=shard_sizes)


@dataclass
class UnbalancedShardingSpec(ShardingSpec):
    dim: int
    placements: List[Union[dist._remote_device, str]]
    shard_sizes: List[int]

    def __post_init__(self):
        assert len(self.placements) == len(self.shard_sizes)
        for i, remote_device in enumerate(self.placements):
            if not isinstance(remote_device, torch.distributed._remote_device):
                self.placements[i] = torch.distributed._remote_device(remote_device)

    def build_metadata(
        self, tensor_sizes: torch.Size, tensor_properties: sharded_tensor_meta.TensorProperties, reverse: bool = False
    ) -> sharded_tensor_meta.ShardedTensorMetadata:
        shards_metadata = []
        tensor_num_dim = len(tensor_sizes)
        for idx, placement in enumerate(self.placements):
            shard_size = list(tensor_sizes)
            shard_size[self.dim] = self.shard_sizes[idx]
            current_offsets = [0] * tensor_num_dim
            if reverse:
                current_offsets[self.dim] = sum(self.shard_sizes) - sum(self.shard_sizes[:idx])
            else:
                current_offsets[self.dim] = sum(self.shard_sizes[:idx])
            shard_metadata = ShardMetadata(
                shard_offsets=current_offsets,
                shard_sizes=shard_size,
                placement=placement,
            )
            shards_metadata.append(shard_metadata)

        return sharded_tensor_meta.ShardedTensorMetadata(shards_metadata, tensor_sizes, tensor_properties)

    def shard(self, tensor: torch.Tensor, src_rank: int = 0, process_group=None, reverse=False) -> "ShardedTensor":
        """
        Different from ChunkShardingSpec which uses scatter op for each rank. We hope tensor here is
        on meta device which will not cost much memory.
        """
        # relative imports to avoid circular dependency
        from torch.distributed._shard.sharded_tensor import ShardedTensor

        tensor_properties = sharded_tensor_meta.TensorProperties(
            dtype=tensor.dtype,
            layout=tensor.layout,
            requires_grad=tensor.requires_grad,
            memory_format=torch.contiguous_format,
            pin_memory=tensor.is_pinned(),
        )
        current_rank = dist.get_rank(process_group)
        if tensor.size(self.dim) != sum(self.shard_sizes):
            # consider as local tensor
            assert (
                tensor.size(self.dim) == self.shard_sizes[current_rank]
            ), f"User input a local tensor({tensor.size()}) with wrong shape({self.shard_sizes})"
            complete_size = list(tensor.size())
            complete_size[self.dim] = sum(self.shard_sizes)
            complete_size = torch.Size(complete_size)
        else:
            complete_size = tensor.size()

        tensor_meta = self.build_metadata(complete_size, tensor_properties)
        local_shards = []
        local_tensor = None
        local_metadata = None
        for shard_meta in tensor_meta.shards_metadata:
            rank, device = _parse_and_validate_remote_device(process_group, shard_meta.placement)
            if current_rank == rank:
                # only support 1-dim tensor
                local_tensor = torch.empty(
                    shard_meta.shard_sizes, dtype=tensor.dtype, layout=tensor.layout, device=device
                )
                local_metadata = shard_meta
                if device != torch.device("meta"):
                    # we copy value from tensor
                    start = sum(self.shard_sizes[:rank])
                    end = sum(self.shard_sizes[: rank + 1])
                    local_tensor[start:end] = tensor

        # each rank should have local_tensor and local_metadata initialized if we build
        # the metadata list in a correct way.
        assert local_tensor is not None
        assert local_metadata is not None

        if list(local_tensor.size()) != local_metadata.shard_sizes:
            # detach again after receiving to ensure local shards remain a leaf node
            print(local_metadata.shard_sizes)
            local_tensor = local_tensor.resize_(local_metadata.shard_sizes).detach()

        # Sync requires_grad to local_shard.
        local_tensor.requires_grad = tensor.requires_grad

        local_shards.append(Shard(tensor=local_tensor, metadata=local_metadata))
        st = ShardedTensor._init_from_local_shards_and_global_metadata(
            local_shards, tensor_meta, process_group=process_group
        )
        st._sharding_spec = self

        return st
