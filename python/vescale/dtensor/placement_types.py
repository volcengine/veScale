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
from typing import List, NamedTuple, Optional, Tuple, cast

import torch
import torch.distributed.distributed_c10d as c10d

from vescale.dtensor.device_mesh import DeviceMesh


class Placement:
    # base class Placement type

    # convenient utils to check for placement types
    def is_shard(self, dim: Optional[int] = None) -> bool:
        if dim is not None and isinstance(self, Shard):
            return self.dim == dim
        else:
            return isinstance(self, Shard)

    def is_interleaved_shard(self, dim: Optional[int] = None) -> bool:
        if dim is not None and isinstance(self, InterleavedShard):
            return self.dim == dim
        else:
            return isinstance(self, InterleavedShard)

    def is_replicate(self) -> bool:
        return isinstance(self, Replicate)

    def is_partial(self) -> bool:
        return isinstance(self, Partial)

    def serialize_to_tensor(self, device) -> torch.Tensor:
        if self.is_replicate():
            return torch.tensor([0, 0, 0], device=device, dtype=torch.int64)
        elif self.is_partial():
            return torch.tensor([1, 0, 0], device=device, dtype=torch.int64)
        elif self.is_shard():
            return torch.tensor([2, self.dim, 0], device=device, dtype=torch.int64)
        elif self.is_interleaved_shard():
            return torch.tensor([3, self.dim, self.interleaved_size], device=device, dtype=torch.int64)

    @staticmethod
    def serialize_from_tensor(tensor: torch.Tensor):
        if tensor[0] == 0:
            return Replicate()
        elif tensor[0] == 1:
            return Partial()
        elif tensor[0] == 2:
            return Shard(dim=tensor[1])
        elif tensor[0] == 3:
            return InterleavedShard(dim=tensor[1], interleaved_size=tensor[2])


class Shard(Placement):
    # shard placement, shard on a dim
    def __init__(self, dim: int):
        self.dim = dim

    def _split_tensor(
        self,
        tensor: torch.Tensor,
        num_chunks: int,
        *,
        with_padding: bool = True,
        contiguous: bool = True,
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """
        This function uses torch.chunk to split a tensor into num_chunks shards along
        the Shard placement dimension, and return a list of shards with their pad sizes.

        Keyword args:
            with_padding (bool, optional): when True, we pad the tensor on the last
            few ranks before calling the collectives (i.e. scatter/all_gather, etc.).
            This is because collectives usually require equal size tensor inputs

        Example:
            >>> Given a 2D global tensor with Shard(0)
            >>> Run this method:
            >>> torch.chunk(torch.tensor([[i] * 2 for i in range(13)]), num_chunks=6, dim=0)

            tensor1([[0, 0],
                     [1, 1],
                     [2, 2]])

            tensor2([[3, 3],
                     [4, 4],
                     [5, 5]])

            tensor3([[6, 6],
                     [7, 7],
                     [8, 8]])

            tensor4([[ 9,  9],
                     [10, 10],
                     [11, 11]])

            tensor5([[12, 12],
                     [<pad>, <pad>],
                     [<pad>, <pad>]])

            <empty6>([[<pad>, <pad>],
                      [<pad>, <pad>],
                      [<pad>, <pad>]])
        """
        assert self.dim <= tensor.ndim, f"Sharding dim {self.dim} greater than tensor ndim {tensor.ndim}"
        assert tensor.size(self.dim) > 0, f"Tensor size along dim{self.dim} is 0. There is nothing to be sharded."

        # chunk tensor over dimension `dim` into n slices with padding if necessary
        tensor_list = list(torch.chunk(tensor, num_chunks, dim=self.dim))
        # compute the chunk size inline with ``torch.chunk`` (round up to int)
        full_chunk_size = (tensor.size(self.dim) + num_chunks - 1) // num_chunks

        # Compute chunk size for each chunk for ``self.dim``
        chunk_sizes = [tensor_list[idx].size(self.dim) if idx < len(tensor_list) else 0 for idx in range(num_chunks)]
        # Compute pad size on each chunk
        pad_sizes = [full_chunk_size - chunk_size for chunk_size in chunk_sizes]

        # Reuse tensor to fill empty chunk with empty tensor
        num_empty_tensors = num_chunks - len(tensor_list)
        if num_empty_tensors > 0:
            tensor_size = list(tensor_list[0].size())
            tensor_size = [size if idx != self.dim else 0 for idx, size in enumerate(tensor_size)]
            tensor = tensor.new_zeros(tensor_size)  # (allocate empty chunk)
            for _ in range(num_empty_tensors):
                tensor_list.append(tensor)

        if with_padding or contiguous:
            shard_list = []
            for shard, pad_size in zip(tensor_list, pad_sizes):
                # Fill the empty tensor with zeroes with padding.
                if with_padding and pad_size > 0:
                    shard = self._pad_tensor(shard, pad_size)
                shard = shard.contiguous() if contiguous else shard
                shard_list.append(shard)
            return shard_list, pad_sizes
        else:
            return tensor_list, pad_sizes

    def _pad_tensor(
        self,
        tensor: torch.Tensor,
        pad_size: int,
    ) -> torch.Tensor:
        pad = [0, 0] * (tensor.ndim - self.dim)
        pad[-1] = pad_size
        return torch.nn.functional.pad(tensor, pad)

    def _unpad_tensor(
        self,
        tensor: torch.Tensor,
        pad_size: int,
    ) -> torch.Tensor:
        return tensor.narrow(
            self.dim,
            start=0,
            length=tensor.size(self.dim) - pad_size,
        )

    def _local_shard_size_on_dim(
        self,
        size_on_dim: int,
        num_chunks: int,
        rank: int,
        return_offset: bool = False,
    ) -> Tuple[int, int]:
        """
        returns the local shard size and offset on a given tensor dim
        """
        assert (
            size_on_dim >= num_chunks
        ), f"Size to be sharded on dim {self.dim} must be at least as large as the number of devices in that dimension {num_chunks}"

        # Compute the chunk size inline with ``torch.chunk``
        full_chunk_size = (size_on_dim + num_chunks - 1) // num_chunks

        # Compute chunk size for each chunk on the dimension.
        chunk_sizes = [
            max(
                min(size_on_dim, full_chunk_size * (idx + 1)) - full_chunk_size * idx,
                0,
            )
            for idx in range(num_chunks)
        ]
        local_shard_size = chunk_sizes[rank]

        local_offset_on_dim = -1
        if return_offset:
            # Return global tensor dim size of current dimension if for empty shard
            # to represent the end of the corresponding tensor dim.
            local_offset_on_dim = sum(chunk_sizes[:rank])

        return (local_shard_size, local_offset_on_dim)

    def __hash__(self) -> int:
        ret = self.dim + 128  # restrict sharding dim in [-128, +128]; should be sufficient
        assert ret >= 0
        return ret

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Shard):
            return False
        return self.dim == other.dim

    def __repr__(self) -> str:
        """
        machine readable representation of the Shard placement
        """
        return f"Shard(dim={self.dim})"

    def __str__(self) -> str:
        """human readable representation of the Shard placement"""
        return f"S({self.dim})"


class Replicate(Placement):
    # replicate placement
    def __hash__(self) -> int:
        # every replicate placement is the same
        return -1

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Replicate):
            return False
        return True

    def __repr__(self) -> str:
        """
        machine readable representation of the Replicate placement
        """
        return "Replicate()"

    def __str__(self) -> str:
        """
        human readable representation of the Replicate placement
        """
        return "R"


class Partial(Placement):
    # This is a default partial placement with element-wise reduce op
    # when doing reduction it follows the contract of `_to_replicate`
    # and `_to_shard` to do the reduction and convert the local tensor
    # to the corresponding state (replicate or shard)
    #
    # We can implement custom reductions as needed by subclassing this
    # class and override those contracts.

    def __init__(self, reduce_op: c10d.ReduceOp.RedOpType = c10d.ReduceOp.SUM):
        self.reduce_op: c10d.ReduceOp.RedOpType = reduce_op

    def __hash__(self) -> int:
        ret = -3 - hash(self.reduce_op)  # hash(reduce_op) gives 0~8
        assert ret <= -3
        return ret

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Partial):
            return False
        return self.reduce_op == other.reduce_op

    def __repr__(self) -> str:
        """
        machine readable representation of the Partial placement
        """
        return f"Partial(reduce_op={self.reduce_op})"

    def __str__(self) -> str:
        """
        human readable representation of the Partial placement
        """
        return "P"


class InterleavedShard(Shard):
    """
    The major difference between this placement and Shard is that the global
    tensor with a `InterleavedShard` placement is not contiguous. But you can
    always treat a InterleavedShard(dim=x, interleaved_size=y) as a
    Shard(dim=x+1)) on a tensor by reshaping the original one from
    ``[..., size(x), ...]`` to ``[..., y, size(x) // y, ...]``

    NOTE: We currently don't support padding in InterleavedShard, which means
    we cannot interleaved shard a tensor when it's size is not divisible by
    the multiply of interleaved_size and corresponding mesh size.
    """

    def __init__(self, dim: int, interleaved_size: int):
        self.dim = dim
        # TODO: make this attribute a list to support multi interleaved shard
        self.interleaved_size = interleaved_size

    def _split_tensor(
        self,
        tensor: torch.Tensor,
        num_chunks: int,
        *,
        contiguous: bool = True,
    ) -> Tuple[List[torch.Tensor]]:
        assert self.dim <= tensor.ndim, f"Interleaved Sharding dim {self.dim} greater than tensor ndim {tensor.ndim}"
        assert tensor.size(self.dim) > 0, f"Tensor size along dim {self.dim} is 0. There is nothing to be sharded."
        assert (
            tensor.size(self.dim) % self.interleaved_size == 0
        ), f"Tensor size along dim {self.dim} is not a multiple of interleaved size {self.interleaved_size}."
        assert (
            tensor.size(self.dim) // self.interleaved_size
        ) % num_chunks == 0, "InterleavedShard doesn't allow padding"

        # step 1: reshape tensor
        tensor = tensor.view(tensor.shape[: self.dim] + (self.interleaved_size, -1) + tensor.shape[self.dim + 1 :])

        # step 2: split tensor
        # chunk tensor over dimension `dim` into n slices with padding if necessary
        tensor_list = list(torch.chunk(tensor, num_chunks, dim=self.dim + 1))

        # step 3: reshape back
        result_list = []
        for t in tensor_list:
            if contiguous:
                t = t.contiguous()
            # NOTE: view op might be not okay here, because tensor returned by chunk op
            # is not contiguous.
            shard = t.reshape(tensor.shape[: self.dim] + (-1,) + tensor.shape[self.dim + 2 :])
            result_list.append(shard)

        return result_list

    def _local_shard_size_on_dim(
        self,
        size_on_dim: int,
        num_chunks: int,
        rank: int,
        return_offset: bool = False,
    ) -> Tuple[int, int]:
        """
        returns the local shard size and offset on a given tensor dim.
        NOTE: argument ``rank`` and ``return_offset`` is useless here. The reason for
        keeping them is to align this API with the one of ``Shard`` placement.
        """
        assert (
            size_on_dim >= num_chunks
        ), f"Size to be sharded on dim {self.dim} must be at least as large as the number of devices in that dimension {num_chunks}"

        # Compute the chunk size inline with ``torch.chunk``
        full_chunk_size = size_on_dim // num_chunks
        return (full_chunk_size, None)

    def __hash__(self) -> int:
        assert self.dim >= 0 and self.interleaved_size >= 0, "negatives (-1 & -2) can result in hash collison"
        return hash((self.dim, self.interleaved_size))

    def __repr__(self) -> str:
        return f"InterleavedShard(dim={self.dim}, interleaved_size={self.interleaved_size})"

    def __str__(self) -> str:
        return f"IS({self.dim}, {self.interleaved_size})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, InterleavedShard):
            return False
        return self.dim == other.dim and self.interleaved_size == other.interleaved_size


class TensorMeta(NamedTuple):
    # simple named tuple to represent tensor metadata
    # intentionally to stay simple only for sharding
    # propagation purposes.
    shape: torch.Size
    stride: Tuple[int, ...]
    dtype: torch.dtype

    def __hash__(self) -> int:
        assert isinstance(self.stride, Tuple)
        return hash((self.shape, self.stride, self.dtype))

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, TensorMeta):
            return False
        return (
            self.shape == __o.shape  # type: ignore[union-attr]
            and self.stride == __o.stride  # type: ignore[union-attr]
            and self.dtype == __o.dtype  # type: ignore[union-attr]
        )


# used internally to propagate the placements


@dataclass
class DTensorSpec:
    mesh: DeviceMesh
    placements: Tuple[Placement, ...]

    # tensor meta will only be set during sharding propagation
    tensor_meta: Optional[TensorMeta] = None

    def __hash__(self) -> int:
        # hashing and equality check for DTensorSpec are used to cache the sharding
        # propagation results. We only need to consider the mesh, placements, tensor_meta.
        assert isinstance(self.placements, Tuple) and all(isinstance(p, Placement) for p in self.placements)

        return hash(
            (
                self.mesh,
                tuple(self.placements),
                self.tensor_meta,  # None is hashable
            )
        )

    def __eq__(self, __o: object) -> bool:
        if not (
            isinstance(__o, DTensorSpec) and self.mesh == __o.mesh and tuple(self.placements) == tuple(__o.placements)
        ):
            return False
        return self.tensor_meta == __o.tensor_meta  # None included

    def __str__(self) -> str:
        """
        human readable representation of the DTensorSpec
        """
        if len(self.placements) == 1:
            placement_str = str(self.placements[0])
        else:
            placement_str = str(self.placements)

        if self.tensor_meta is not None:
            tensor_shape = str(tuple(self.tensor_meta.shape))
        else:
            tensor_shape = "unknown shape"

        return f"Spec({placement_str} on {tensor_shape})"

    @property
    def shape(self) -> torch.Size:
        if self.tensor_meta is None:
            raise ValueError("tensor_meta is not set")
        return self.tensor_meta.shape

    @property
    def ndim(self) -> int:
        if self.tensor_meta is None:
            raise ValueError("tensor_meta is not set")
        return len(self.tensor_meta.shape)

    @property
    def num_shards(self) -> int:
        num_shards = 1
        for i, placement in enumerate(self.placements):
            if placement.is_shard() or placement.is_interleaved_shard():
                num_shards *= self.mesh.size(i)
        return num_shards

    @property
    def dim_map(self) -> List[int]:
        """
        dim_map is a property we derive from `placements` of
        the distributed tensor. It simply return a list of ints
        where dim_map[i] denotes the sharding mapping to the mesh
        dimension, and len(dim_map) == dist_tensor.ndim
        dim_map[i] = -1: means tensor dim i replicate on mesh
        dim_map[i] = j: means tensor dim i shard on mesh dim j

        For example, we have a dist tensor that have the shape of
        [18, 20, 30], and device_mesh([0, 1, 2, 3]), placements:
        [Shard(1)], the dim_map of this placement would be:
        [-1, 0, -1]. This representation is pretty helpful during
        sharding propagation where we could know exactly each
        tensor dimension is sharded or not.

        Note that if placements contains `Partial`, we have to
        explicitly deal with it, so that when we create a DTensorSpec
        with dim_map, we could properly record the pending sums.
        """
        # dims mapping of dist tensor sharding
        # return size of tensor ndim, -1 represent replicate
        # and int >=0 represent shard on that device mesh dim
        r = [-1] * self.ndim
        for i, placement in enumerate(self.placements):
            if placement.is_shard() or placement.is_interleaved_shard():
                shard_dim = placement.dim
                if r[shard_dim] > -1:
                    raise ValueError(
                        f"Tensor dim {shard_dim} is already sharded on mesh dim {r[shard_dim]},"
                        " DTensor operator implementation does not support things like hybrid"
                        " sharding strategies yet (i.e. [Shard(0), Shard(0)])"
                    )
                r[shard_dim] = i
        return r

    @property
    def sums(self) -> List[int]:
        """
        sums is a property we derive from `placements` of the
        distributed tensor. It simply return a list of ints where
        sums[i] denotes the pending sum (partial) on mesh dim i
        """
        return [idx for idx, placement in enumerate(self.placements) if placement.is_partial()]

    @classmethod
    def from_dim_map(
        cls,
        mesh: DeviceMesh,
        dim_map: List[int],
        sums: List[int],
        tensor_meta: Optional[TensorMeta] = None,
    ) -> "DTensorSpec":
        """
        Construct a DTensorSpec from dim_map list and pending sum.

        Args:
            mesh (class:`DeviceMesh`): device mesh to be used in the DTensorSpec
            dim_map (List[int]): a list of integer that represents sharding on each
                tensor dimension, see `dim_map` property doc for details
            sums (List[int]): a list of integer that represents the dist tensor have
                pending sum on which device mesh dimension.
            tensor meta (TensorMeta): DTensor metadata

        Return:
            a class:`DTensorSpec` object
        """
        # by default replicate on device mesh dims
        placements: List[Placement] = [Replicate() for _ in range(mesh.ndim)]

        # find all mesh dims that need pending reductions
        for s in sums:
            placements[s] = Partial()

        for i, m in enumerate(dim_map):
            if m >= 0:
                placement = placements[m]
                if placement.is_shard():
                    placement = cast(Shard, placement)
                    raise RuntimeError(
                        f"DeviceMesh dimension cann't be mapped to two dimension of the same tensor: {i} and {placement.dim}"
                    )
                elif placement.is_partial():
                    raise RuntimeError(f"DeviceMesh dimension {m} cannot be both shard and partial!")
                placements[m] = Shard(i)

        return cls(mesh, tuple(placements), tensor_meta=tensor_meta)

    def is_replicated(self):
        """
        return True if the current DTensorSpec replicates on all mesh dims (devices)
        """
        return all(placement.is_replicate() for placement in self.placements)
