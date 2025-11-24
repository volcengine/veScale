################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################
# implement matrix related ops for distributed tensor

import copy
from dataclasses import dataclass, field
from typing import cast, Optional, Tuple

import torch
import torch.distributed.distributed_c10d as c10d
import torch.distributed._functional_collectives as funcol

from vescale.dtensor.op_schema import OpSchema, OutputSharding
from vescale.dtensor.ops.utils import (
    register_prop_rule,
    is_tensor_all_replicate,
    is_tensor_all_replicate_except_sharded_at_dim,
    is_tensor_partial,
)
from vescale.dtensor.device_mesh import DeviceMesh
from vescale.dtensor.placement_types import DTensorSpec, Placement, Partial, Replicate, Shard
from vescale.dtensor.redistribute import _reduce_scatter_to_shard_with_pad

aten = torch.ops.aten


@dataclass
class MaskBuffer:
    data: Optional[torch.Tensor] = None

    def materialize_mask(self, mask):
        if self.data is not None:
            raise RuntimeError("MaskBuffer has already been materialized")
        self.data = mask

    def release_mask(self):
        # TODO: evaluate if we need to release the mask buffer or the buffer
        # can just have the same lifetime as the _Partial placement
        if self.data is None:
            raise RuntimeError("MaskBuffer has not been materialized")
        self.data = None

    def apply_mask(self, tensor):
        if self.data is None:
            raise RuntimeError("MaskBuffer has not been materialized")

        # NOTE: _MaskPartial is being used by the embedding op and the gather op.
        # For gather, the mask has the same dimension as the output tensor, whereas
        # the output of the embedding op has an additional dimension compare to the input,
        # hence the output masking logic below having two different cases.
        if tensor.ndim == self.data.ndim:
            tensor[self.data] = 0.0
        else:
            tensor[self.data, :] = 0.0


@dataclass(frozen=True)
class _MaskPartial(Partial):
    """
    A partial mask placement devised for rowwise sharded embedding op, where we need
    to mask and adjust the indices to the local embedding shard, embedding masking
    is a special type of the Partial placement

    NOTE: the lifecycle of this MaskPartial placement follows the corresponding DTensor
    lifecycle, i.e. the indices_mask would only be alive during the lifetime of the DTensor.
    """

    logical_dim_size: int = -1
    mask_buffer: MaskBuffer = field(default_factory=MaskBuffer)
    reduce_op: c10d.ReduceOp.RedOpType = c10d.ReduceOp.SUM

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
        ), f"Size to be sharded on with dim_size {size_on_dim} must be at least as large \
        as the number of devices in that dimension {num_chunks}"

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

    def _partition_value(self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int) -> torch.Tensor:
        # override parent logic to perform partial mask for embedding
        num_chunks = mesh.size(mesh_dim)
        # get local shard size and offset on the embedding_dim
        local_shard_size, local_offset_on_dim = self._local_shard_size_on_dim(
            self.logical_dim_size,
            num_chunks,
            mesh.get_local_rank(mesh_dim),
            return_offset=True,
        )
        # Build the input mask and save it for the current partial placement
        # this is so that the output of embedding op can reuse the same partial
        # placement saved mask to perform mask + reduction
        mask = (tensor < local_offset_on_dim) | (tensor >= local_offset_on_dim + local_shard_size)
        # mask the input tensor
        masked_tensor = tensor.clone() - local_offset_on_dim
        masked_tensor[mask] = 0
        # materialize the mask buffer to be used for reduction
        self.mask_buffer.materialize_mask(mask)
        return masked_tensor

    def _reduce_value(self, tensor: torch.Tensor, mesh: DeviceMesh, mesh_dim: int) -> torch.Tensor:
        # by the time we ned reduction, we should have already saved the mask
        assert self.mask_buffer.data is not None

        # apply the mask to the tensor that pending reduction
        self.mask_buffer.apply_mask(tensor)

        # clear the mask buffer
        self.mask_buffer.release_mask()

        # perform sum reduction
        return funcol.all_reduce(tensor, reduceOp=self.reduce_op.name, group=mesh._dim_group_infos[mesh_dim][1])

    def _reduce_shard_value(
        self,
        tensor: torch.Tensor,
        mesh: DeviceMesh,
        mesh_dim: int,
        shard_spec: Placement,
    ) -> torch.Tensor:
        # by the time we ned reduction, we should have already saved the mask
        assert self.mask_buffer.data is not None

        # apply the mask to the tensor that pending reduction
        self.mask_buffer.apply_mask(tensor)

        # clear the mask buffer
        self.mask_buffer.release_mask()

        # call reduce_shard_tensor of the shard_spec.
        shard_spec = cast(Shard, shard_spec)
        return _reduce_scatter_to_shard_with_pad(tensor, mesh, self.reduce_op, mesh_dim, shard_spec)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _MaskPartial):
            return False

        # if either data is not None, we invalidate the sharding cache, as this indicates
        # the current MaskPartial placement is still in use and should not be used for cache hit.
        if self.mask_buffer.data is not None or other.mask_buffer.data is not None:
            return False

        return self.reduce_op == other.reduce_op and self.logical_dim_size == other.logical_dim_size

    def __hash__(self) -> int:
        return 1 + hash((self.logical_dim_size, id(self.mask_buffer.data), self.reduce_op))

    def __repr__(self) -> str:
        """
        machine readable representation of the MaskPartial placement
        """
        return f"_MaskPartial(logical_dim_size={self.logical_dim_size})"

    def __str__(self) -> str:
        """
        human readable representation of the MaskPartial placement
        """
        return "MaskP"


# TODO: Enable BWD for embedding op.
@register_prop_rule(aten.embedding.default)
def embedding_rules(op_schema: OpSchema) -> OutputSharding:
    weight_spec, inp_spec = op_schema.args_spec
    if any(placement.is_shard(0) for placement in weight_spec.placements):
        raise NotImplementedError("DTensor does not support row-wise sharded embedding operation yet!")

    if all(placement.is_replicate() for placement in weight_spec.placements) and inp_spec.placements == [Shard(0)]:
        # Embedding table is replicated, input ids are sharded along batch
        # dimension. Output lookups should match input sharding spec in this case.
        return OutputSharding(output_spec=DTensorSpec(mesh=inp_spec.mesh, placements=inp_spec.placements))

    if all(placement.is_replicate() for placement in inp_spec.placements):
        weight_dim_map = weight_spec.dim_map
        output_dim_map = inp_spec.dim_map
        output_dim_map.append(weight_dim_map[1])
        return OutputSharding(output_spec=DTensorSpec.from_dim_map(inp_spec.mesh, output_dim_map, []))

    return OutputSharding(
        output_spec=None,
        schema_suggestions=[
            OpSchema(
                op=op_schema.op,
                args_schema=(
                    weight_spec,
                    DTensorSpec(
                        mesh=inp_spec.mesh,
                        placements=tuple([Replicate()] * len(inp_spec.placements)),
                        tensor_meta=inp_spec.tensor_meta,
                    ),
                ),
                kwargs_schema=op_schema.kwargs_schema,
            )
        ],
    )


@register_prop_rule(aten.embedding_renorm_.default)
def embedding_renorm_rules(op_schema: OpSchema) -> OutputSharding:
    raise NotImplementedError("DTensor does not support sharded embedding operation with max_norm yet!")


@register_prop_rule(aten.embedding_dense_backward.default)
def embedding_dense_backward_rules(op_schema: OpSchema) -> OutputSharding:
    grad_output, indices = op_schema.args_schema[:2]
    assert isinstance(grad_output, DTensorSpec)
    assert isinstance(indices, DTensorSpec)

    mesh = grad_output.mesh

    # Situation 1: All replicate
    if is_tensor_all_replicate(grad_output) and is_tensor_all_replicate(indices):
        return OutputSharding(output_spec=DTensorSpec(mesh=mesh, placements=tuple([Replicate()] * mesh.ndim)))

    # Situation 2: Colwise sharding
    if is_tensor_all_replicate_except_sharded_at_dim(
        spec=grad_output, tensor_dim=grad_output.ndim - 1
    ) and is_tensor_all_replicate(indices):
        result_placements = []
        for p in grad_output.placements:
            if p.is_shard():
                tmp_p = copy.deepcopy(p)
                tmp_p.dim = 1
                result_placements.append(tmp_p)
            else:
                result_placements.append(p)
        return OutputSharding(output_spec=DTensorSpec(mesh=mesh, placements=tuple(result_placements)))

    # Situation 3: Sharded on dims other than hidden dim
    sharded_on_no_hidden_flag = False
    sharded_on_no_hidden_mesh_dims = []
    for mesh_idx, idx_p in enumerate(indices.placements):
        grad_out_p = grad_output.placements[mesh_idx]
        if idx_p.is_partial() or grad_out_p.is_partial():
            sharded_on_no_hidden_flag = False
            break
        if idx_p.is_replicate() or grad_out_p.is_replicate():
            continue
        if idx_p != grad_out_p:
            sharded_on_no_hidden_flag = False
            break
        sharded_on_no_hidden_flag = True
        sharded_on_no_hidden_mesh_dims.append(mesh_idx)

    if sharded_on_no_hidden_flag:
        result_placements = [Replicate()] * mesh.ndim
        for mesh_idx in sharded_on_no_hidden_mesh_dims:
            result_placements[mesh_idx] = Partial()
        return OutputSharding(output_spec=DTensorSpec(mesh=mesh, placements=tuple(result_placements)))

    # Situation 4: grad_output is partial, but indices is replicate
    if (
        is_tensor_all_replicate(indices)
        and is_tensor_partial(grad_output)
        and not any(p.is_shard() for p in grad_output.placements)
    ):
        return OutputSharding(output_spec=grad_output)

    raise NotImplementedError(
        "Unsupported embedding dense backward schema:\n" f"grad_output - {grad_output}\n" f"indices - {indices}"
    )
