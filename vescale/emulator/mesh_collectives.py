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

import torch
from vescale.emulator.device_mesh import DeviceMesh
from vescale.emulator.reduce_kernel import ReduceOp
from typing import List


def mesh_all_gather(
    tensors: List[torch.Tensor],
    mesh: DeviceMesh,
    scatter_dim: int,
    mesh_dim: int,
) -> List[torch.Tensor]:
    """
    all_gather all shards and return a tensor that is replicated
    on the previously sharded mesh dimension
    """
    tag_rank_list = mesh._dim_group_infos[mesh_dim]
    dim_group = mesh.get_dim_groups()[mesh_dim]

    results = [0 for _ in range(len(tensors))]
    for (tag, ranks), pg in zip(tag_rank_list, dim_group):
        inputs = []
        outputs_list = []
        for rank in ranks:
            inputs.append(tensors[rank])
            outputs_list.append([torch.empty_like(tensors[rank]) for _ in range(len(ranks))])

        pg.all_gather(outputs_list, inputs)

        for i, outputs in enumerate(outputs_list):
            outputs_list[i] = torch.cat(outputs, dim=scatter_dim)

        for i, rank in enumerate(ranks):
            results[rank] = outputs_list[i]

    return results


def mesh_all_reduce(
    tensors: List[torch.Tensor],
    mesh: DeviceMesh,
    reduce_op: ReduceOp,
    mesh_dim: int,
    tree_structure=None,
) -> List[torch.Tensor]:
    """
    all_reduce all tensors in the list and return a tensor that is replicated
    on the previously sharded mesh dimension
    """
    tag_rank_list = mesh._dim_group_infos[mesh_dim]
    dim_group = mesh.get_dim_groups()[mesh_dim]

    results = [0 for _ in range(len(tensors))]
    for (tag, ranks), pg in zip(tag_rank_list, dim_group):
        inputs = []
        for rank in ranks:
            inputs.append(tensors[rank])
        pg.all_reduce(inputs, op=reduce_op, tree_structure=tree_structure)

        for i, rank in enumerate(ranks):
            results[rank] = inputs[i]

    return results


def mesh_reduce_scatter(
    tensors: List[torch.Tensor],
    mesh: DeviceMesh,
    reduce_op: ReduceOp,
    scatter_dim: int,
    mesh_dim: int,
):
    """
    First peform all_reduce on the tensor, then split the tensor at scatter_dim
    and scatter them to a device mesh dimension.
    """
    tag_rank_list = mesh._dim_group_infos[mesh_dim]
    dim_group = mesh.get_dim_groups()[mesh_dim]

    results = [0 for _ in range(len(tensors))]
    for (tag, ranks), pg in zip(tag_rank_list, dim_group):
        inputs = []
        outputs = []
        for rank in ranks:
            split_size = tensors[rank].size()[scatter_dim] // len(ranks)
            input_list = torch.split(tensors[rank], split_size, dim=scatter_dim)
            output = torch.empty_like(input_list[0])
            outputs.append(output)
            inputs.append(input_list)
        pg.reduce_scatter(outputs, inputs, op=reduce_op)

        for i, rank in enumerate(ranks):
            results[rank] = outputs[i]

    return results


def mesh_all_to_all(
    output_tensor_list: List[List[torch.Tensor]],
    input_tensor_list: List[List[torch.Tensor]],
    mesh: DeviceMesh,
    mesh_dim: int = 0,
    async_op: bool = False,
):
    """
    Perform all_to_all on the tensor list.
    """
    tag_rank_list = mesh._dim_group_infos[mesh_dim]
    dim_group = mesh.get_dim_groups(mesh_dim)

    for (tag, ranks), pg in zip(tag_rank_list, dim_group):
        inputs = []
        outputs = []
        for rank in ranks:
            inputs.append(input_tensor_list[rank])
            outputs.append(input_tensor_list[rank])
        pg.all_to_all(outputs, inputs)

        for i, rank in enumerate(ranks):
            output_tensor_list[rank] = outputs[i]

    return output_tensor_list


def mesh_broadcast(
    tensors: List[torch.Tensor],
    mesh: DeviceMesh,
    mesh_dim: int = 0,
    async_op=False,
) -> List[torch.Tensor]:
    """
    broadcast the tensor to a device mesh dimension. We by default
    use the first rank of the mesh dimension as the source of truth, i.e
    for a 2d mesh [[0, 1], [2, 3]], if we broadcast on mesh_dim = 1, we will
    broadcast the tensor on rank 0 to rank 0/1, and tensor on rank 2
    to rank 2/3.

    Args:
        tensors (List[torch.Tensor]): tensor to broadcast.
        mesh_dim (int, optional): indicate which mesh dimension we want
            to broadcast on, we by default choose the first rank on the
            mesh dimension as source of truth.

    Returns:
        A list of :class:`Tensor` object
    """

    # NOTE: funcol impl already check and force tensor contiguous, we do nothing here.
    tag_rank_list = mesh._dim_group_infos[mesh_dim]
    dim_group = mesh.get_dim_groups()[mesh_dim]

    results = [0 for _ in range(len(tensors))]
    for (tag, ranks), pg in zip(tag_rank_list, dim_group):
        source = ranks[0]
        for rank in ranks:
            results[rank] = tensors[source]

    return results


def mesh_scatter(
    outputs: List[torch.Tensor],
    scatter_list_list: List[List[torch.Tensor]],
    mesh: DeviceMesh,
    mesh_dim: int = 0,
    async_op: bool = False,
) -> torch.Tensor:
    """
    scatter a list of tensors to a device mesh dimension. We by default
    use the first rank of the mesh dimension as the source of truth, i.e
    for a 2d mesh [[0, 1], [2, 3]], if we scatter on mesh_dim = 1, we will
    scatter the tensor list on rank 0 to rank 0/1, and tensor list on rank
    2 to rank 2/3.

    Args:
        outputs (List[torch.Tensor]): the tensor to receive the scattered list.
        scatters_list (List[List[torch.Tensor]]): the tensor list to be scattered.
        mesh (DeviceMesh): device mesh.
        mesh_dim (int, optional): indicate which mesh dimension we want
            to scatter on, we by default choose the first rank on the
            mesh dimension as source of truth.

    Returns:
        A list of :class:`torch.Tensor` object
    """
    tag_rank_list = mesh._dim_group_infos[mesh_dim]
    dim_group = mesh.get_dim_groups()[mesh_dim]

    for (tag, ranks), pg in zip(tag_rank_list, dim_group):
        source = ranks[0]
        for i, rank in enumerate(ranks):
            outputs[rank] = scatter_list_list[source][i]

    return outputs
