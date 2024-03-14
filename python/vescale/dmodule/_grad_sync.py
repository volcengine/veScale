################################################################################
#
# Copyright 2023 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################
# Some code comes from megatron/core/distributed/finalize_model_grads.py in Megatron-LM
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
################################################################################

"""This file handles gradient allreduce for DModule

NOTE:
- If wrapped by DDP, it is called after DDP.finish_grad_sync()
- `generate_grad_sync_list` is not recommended to be placed into a param.grad pre-hook, because:
    i) having multiple hooks on param.grad complicates the design and debugging
    ii) gradient accumlation will repeatedly fire param.grad pre-hook, degrading performance

"""

from typing import List, Tuple, Union

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch import Tensor

from vescale.dtensor.dtensor import DTensor
from vescale.dtensor.placement_types import DTensorSpec, Replicate
from vescale.dtensor.device_mesh import DeviceMesh

__all__ = ["generate_grad_sync_list", "sync_gradients"]


def generate_grad_sync_list(candidate: List[Tuple[str, DTensor]]) -> List[Tuple[str, Union[Tensor, DTensor]]]:
    """obtain Partial gradient list from the candiate list."""
    grad_sync_list = []
    for fqn, param in candidate:
        assert param.requires_grad
        assert isinstance(param.data, DTensor)
        if hasattr(param, "main_grad"):
            if param.main_grad is None:
                continue
            grad_spec = getattr(param.main_grad, "_spec", None)
            assert grad_spec is not None, "DDP's .main_grad must save DTensor .grad's _spec"
            placements = grad_spec.placements
            fqn += ".main_grad"
            grad = param.main_grad
        else:
            assert hasattr(param, "grad")
            if param.grad is None:
                continue
            placements = param.grad.placements
            fqn += ".grad"
            grad = param.grad
        if any(p.is_partial() for p in placements):
            grad_sync_list.append((fqn, grad))
    return grad_sync_list


@torch.no_grad()
def _allreduce_by_bucket(
    tensors: List[torch.Tensor],
    process_group: torch.distributed.ProcessGroup,
    bucket_size: int = 40000000,
) -> None:
    if not tensors:
        return

    dtype_to_tensors = {}
    for g in tensors:
        dtype = g.dtype
        if dtype not in dtype_to_tensors:
            dtype_to_tensors[dtype] = []
        dtype_to_tensors[dtype].append(g)

    dtype_to_buckets = {}
    for dtype in dtype_to_tensors:
        dtype_to_buckets[dtype] = []
        if bucket_size > 0:
            size = 0
            bucket = []
            for g in dtype_to_tensors[dtype]:
                size += g.numel() * g.element_size()
                bucket.append(g)
                if size > bucket_size:
                    dtype_to_buckets.append(bucket)
                    size = 0
                    bucket = []
            if bucket:
                dtype_to_buckets[dtype].append(bucket)
                bucket = []
                size = 0
        else:
            dtype_to_buckets[dtype].append(dtype_to_tensors[dtype])

    for dtype in dtype_to_buckets:
        buckets = dtype_to_buckets[dtype]
        for gs in buckets:
            coalesced = _flatten_dense_tensors(gs)
            torch.distributed.all_reduce(coalesced, group=process_group)
            for buf, synced in zip(gs, _unflatten_dense_tensors(coalesced, gs)):
                buf.copy_(synced)


def sync_gradients(grad_sync_list: List[Tuple[str, Union[Tensor, DTensor]]], device_mesh: DeviceMesh) -> None:
    r"""
    AllReduce-Sum all gradients of Partial (given by `grad_sync_list`) on device_mesh.
    """
    if not grad_sync_list:
        return

    # get local tensors to allreduce + get process group to allreduce
    local_gradients = []
    partial_mesh_idxes = set()
    for fqn, grad in grad_sync_list:
        if fqn.endswith("main_grad"):
            local_gradients.append(grad.data)
        else:
            local_gradients.append(grad._local_tensor)
        partial_mesh_idxes.update([i for i, p in enumerate(grad._spec.placements) if p.is_partial()])
    assert len(partial_mesh_idxes) == 1, "currently, we only consider a single Partial on the same mesh dim."
    partial_pg = device_mesh.get_dim_groups(partial_mesh_idxes.pop())

    # allreduce local gradients
    _allreduce_by_bucket(local_gradients, partial_pg)

    # change DTensor gradients from partial to replicate placement
    for fqn, grad in grad_sync_list:
        new_placements = [Replicate() if p.is_partial() else p for p in grad._spec.placements]
        grad._spec = DTensorSpec(grad._spec.mesh, tuple(new_placements), grad._spec.tensor_meta)
