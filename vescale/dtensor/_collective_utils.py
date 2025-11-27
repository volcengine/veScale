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

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.distributed_c10d import (
    get_rank,
    ProcessGroup,
    send,
    recv,
    Work,
    get_global_rank,
)

from torch.distributed.tensor._collective_utils import (
    _shard_dim_alltoall_meta,
    shard_dim_alltoall,
    mesh_scatter,
    mesh_broadcast,
    pad_tensor,
    unpad_tensor,
    fill_empty_tensor_to_shards,
    check_tensor_meta,
    spec_to_bytes,
    MeshTopoInfo,
    allgather_cost,
    allreduce_cost,
    reduce_scatter_cost,
    redistribute_cost,
)


__all__ = [
    "_shard_dim_alltoall_meta",
    "shard_dim_alltoall",
    "mesh_scatter",
    "mesh_broadcast",
    "pad_tensor",
    "unpad_tensor",
    "fill_empty_tensor_to_shards",
    "check_tensor_meta",
    "spec_to_bytes",
    "MeshTopoInfo",
    "allgather_cost",
    "allreduce_cost",
    "reduce_scatter_cost",
    "redistribute_cost",
    "mesh_scatter_ragged",
]
"""
In this file, we add mesh_scatter_ragged.
"""

logger = logging.getLogger(__name__)


# TODO(jiacheng) Use broadcast instead of send recv.
def mesh_scatter_ragged(
    output: torch.Tensor,
    scatter_list: list[torch.Tensor],
    mesh: DeviceMesh,
    mesh_dim: int = 0,
    *,
    group_src: int = 0,
) -> Work | None:
    """
    A copy from mesh_scatter, but use send/recv to do ragged scatter.
    """
    if output.is_meta:
        return None
    dim_group = mesh.get_group(mesh_dim)
    assert isinstance(dim_group, ProcessGroup)

    if group_src == get_rank(dim_group):
        # Zezhou: performance issue, serialized send/recv, can we do async launch?
        for rank in range(mesh.size(mesh_dim)):
            if rank == group_src:
                continue
            # send the tensor to the rank
            fut = send(
                scatter_list[rank],
                dst=get_global_rank(dim_group, rank),
            )
        output.copy_(scatter_list[group_src])
    else:
        fut = recv(
            output,
            src=get_global_rank(dim_group, group_src),
        )

    return None
