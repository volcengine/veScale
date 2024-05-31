################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

from typing import Any, List
import torch
from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed.checkpoint.planner import WriteItem, ReadItem, WriteItemType, LoadItemType, TensorWriteData
from torch.distributed.checkpoint.metadata import (
    STATE_DICT_TYPE,
    STORAGE_TYPES,
    MetadataIndex,
    ChunkStorageMetadata,
    BytesStorageMetadata,
    TensorStorageMetadata,
)
from torch.distributed._shard.sharded_tensor import TensorProperties
from torch.distributed.checkpoint.resharding import (
    _check_shard_metadata_pair_overlap,
    _shards_get_overlap_region_wrt_saved_tensor,
)

from vescale.dtensor import DTensor
from vescale.dtensor._utils import compute_local_shape, compute_local_offset
from vescale.optim.distributed_optimizer import OptimizerStateSpec


def _create_write_items_for_dtensor(fqn, tensor: DTensor) -> WriteItem:
    sizes = torch.Size(compute_local_shape(tensor.shape, tensor.device_mesh, tensor.placements))
    offsets = torch.Size(compute_local_offset(tensor.shape, tensor.device_mesh, tensor.placements))

    return WriteItem(
        index=MetadataIndex(fqn=fqn, offset=offsets),
        type=WriteItemType.SHARD,
        tensor_data=TensorWriteData(
            chunk=ChunkStorageMetadata(offsets=offsets, sizes=sizes),
            properties=TensorProperties.create_from_tensor(tensor._local_tensor),  # keep out of autograd
            size=tensor.size(),
        ),
    )


def _create_chunk_from_dtensor(tensor: DTensor) -> ChunkStorageMetadata:
    sizes = torch.Size(compute_local_shape(tensor.shape, tensor.device_mesh, tensor.placements))
    offsets = torch.Size(compute_local_offset(tensor.shape, tensor.device_mesh, tensor.placements))
    return ChunkStorageMetadata(offsets=offsets, sizes=sizes)


def _create_write_item_for_tensor(fqn: str, tensor: torch.Tensor) -> WriteItem:
    offsets = torch.Size([0] * len(tensor.size()))
    return WriteItem(
        index=MetadataIndex(fqn, offsets),
        type=WriteItemType.TENSOR,
        tensor_data=TensorWriteData(
            chunk=ChunkStorageMetadata(offsets=offsets, sizes=tensor.size()),
            properties=TensorProperties.create_from_tensor(tensor),
            size=tensor.size(),
        ),
    )


def _create_write_item_for_optimizer_state(fqn, object: OptimizerStateSpec) -> WriteItem:
    sizes = object.local_shape
    offsets = object.global_offset

    return WriteItem(
        index=MetadataIndex(fqn=fqn, offset=offsets),
        type=WriteItemType.SHARD,
        tensor_data=TensorWriteData(
            chunk=ChunkStorageMetadata(offsets=offsets, sizes=sizes),
            properties=TensorProperties.create_from_tensor(object.local_tensor),
            size=object.global_shape,
        ),
    )


def _create_write_item_for_bytesio(fqn: str, bytes: Any):
    return WriteItem(
        index=MetadataIndex(fqn),
        type=WriteItemType.BYTE_IO,
    )


def _create_write_items(fqn: str, object: Any) -> List[WriteItem]:
    if isinstance(object, DTensor):
        return [_create_write_items_for_dtensor(fqn, object)]
    elif isinstance(object, torch.Tensor):
        return [_create_write_item_for_tensor(fqn, object)]
    elif isinstance(object, OptimizerStateSpec):
        return [_create_write_item_for_optimizer_state(fqn, object)]
    else:
        return [_create_write_item_for_bytesio(fqn, object)]


def _create_read_item_for_tensor(dest_index, dest_offsets, storage_index, storage_offsets, lengths):
    return ReadItem(
        type=LoadItemType.TENSOR,
        dest_index=dest_index,
        dest_offsets=torch.Size(dest_offsets),
        storage_index=storage_index,
        storage_offsets=torch.Size(storage_offsets),
        lengths=torch.Size(lengths),
    )


def create_read_items_for_chunk_list(
    fqn: str,
    checkpoint_md: TensorStorageMetadata,
    local_chunks: List[ChunkStorageMetadata],
) -> List[ReadItem]:
    """
    Creates a list of ``ReadItem`` based on the checkpoint and local chunks.

    This applies the resharding algorithm and computes the reads needed
    to satisfy ``local_chunks`` with a checkpoint described by ``checkpoint_md``.

    Args:
        fqn (str) : The state_dict FQN to pass to ``ReadItem``.
        checkpoint_md (TensorStorageMetadata): metadata for a given tensor
            from a checkpoint.
        local_chunks (List[ChunkStorageMetadata]): Local chunks that needs to be
            loaded.

    Returns:
        A list of ``ReadItem`` that will satisfy all input chunks.
    """
    read_items = []
    # this is a naive quadratic algo that can be optimized later
    for idx, shard in enumerate(local_chunks):
        for storage_idx, storage_md in enumerate(checkpoint_md.chunks):
            if not _check_shard_metadata_pair_overlap(shard, storage_md):
                continue

            storage_offsets = []
            dest_offsets = []
            lengths = []
            for (
                dim,
                offset_for_saved_tensor,
                offset_for_current_tensor,
                length,
            ) in _shards_get_overlap_region_wrt_saved_tensor(saved_shard=storage_md, current_shard=shard):
                storage_offsets.append(offset_for_saved_tensor)
                dest_offsets.append(offset_for_current_tensor)
                lengths.append(length)

            read_items.append(
                _create_read_item_for_tensor(
                    dest_index=MetadataIndex(fqn, shard.offsets, idx),
                    dest_offsets=dest_offsets,
                    storage_index=MetadataIndex(fqn, storage_md.offsets, storage_idx),
                    storage_offsets=storage_offsets,
                    lengths=lengths,
                )
            )
    return read_items


def _create_chunk_from_tensor(tensor: torch.Tensor) -> ChunkStorageMetadata:
    return ChunkStorageMetadata(offsets=torch.Size([0] * len(tensor.size())), sizes=tensor.size())


def _create_read_item_for_byteio(dest_index, dest_offset, storage_index, storage_offset, length):
    return ReadItem(
        type=LoadItemType.BYTE_IO,
        dest_index=dest_index,
        dest_offsets=torch.Size((dest_offset,)),
        storage_index=storage_index,
        storage_offsets=torch.Size((storage_offset,)),
        lengths=torch.Size((length,)),
    )


def _create_chunk_from_optimizer_spec(obj: OptimizerStateSpec) -> ChunkStorageMetadata:
    return ChunkStorageMetadata(offsets=obj.global_offset, sizes=obj.local_shape)


def _create_read_items(fqn: str, md: STORAGE_TYPES, obj: Any) -> List[ReadItem]:
    if not isinstance(md, BytesStorageMetadata):
        if isinstance(obj, DTensor):
            local_chunks = [_create_chunk_from_dtensor(obj)]
        elif isinstance(obj, torch.Tensor):
            local_chunks = [_create_chunk_from_tensor(obj)]
        elif isinstance(obj, OptimizerStateSpec):
            local_chunks = [_create_chunk_from_optimizer_spec(obj)]
        else:
            raise ValueError(
                f"Invalid checkpoint metadata for {fqn}, " + f"expected BytesStorageMetadata but found {type(md)}"
            )
        return create_read_items_for_chunk_list(fqn, md, local_chunks)
    else:
        return [
            _create_read_item_for_byteio(
                dest_index=MetadataIndex(fqn),
                dest_offset=0,
                storage_index=MetadataIndex(fqn),
                storage_offset=0,
                length=0,
            )
        ]


def _chunk_for_shard(shard_md: ShardMetadata) -> ChunkStorageMetadata:
    return ChunkStorageMetadata(
        offsets=torch.Size(shard_md.shard_offsets),
        sizes=torch.Size(shard_md.shard_sizes),
    )


def find_tensor_shard(tensor: torch.Tensor, index: MetadataIndex) -> torch.Tensor:
    if isinstance(tensor, DTensor):
        return tensor.to_local()
    if index.offset is not None:
        # special case looking up a tensor by origin
        if index.offset == torch.Size([0] * len(tensor.size())):
            return tensor
        raise ValueError(f"FQN: '{index.fqn}' is not a DTensor, can't find by offset: '{index.offset}'")
    return tensor


def find_state_dict_object(state_dict: STATE_DICT_TYPE, index: MetadataIndex) -> Any:
    # Called when real writing happened
    # The filesystem writer calls resolve_data , then it will
    # call find_state_dict_object
    if index.fqn not in state_dict:
        raise ValueError(f"Could not find FQN: '{index.fqn}'")
    obj = state_dict[index.fqn]

    if isinstance(obj, torch.Tensor):
        return find_tensor_shard(obj, index)
    elif isinstance(obj, OptimizerStateSpec):
        return obj.local_tensor
    elif index.offset is not None:
        raise ValueError(
            f"FQN: '{index.fqn}' is not a DTensor, it is a {type(obj)} can't find by offset: '{index.offset}'"
        )
    return obj
