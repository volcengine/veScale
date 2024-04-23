################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################
import io
import dataclasses
import logging
import torch
from typing import Any, Dict, Union, List, Tuple
from torch.distributed.checkpoint.default_planner import (
    DefaultSavePlanner,
    DefaultLoadPlanner,
)
import math
import torch.distributed as dist
from torch.distributed.checkpoint.planner import (
    SavePlan,
    LoadPlan,
    ReadItem,
    WriteItem,
    WriteItemType,
)
from vescale.optim.distributed_optimizer import OptimizerStateSpec
from torch.distributed._shard.sharded_tensor.api import ShardedTensor
from torch.distributed.checkpoint.metadata import MetadataIndex, Metadata
from vescale.dtensor import DTensor
from .vescale_planner_helpers import (
    _create_write_items,
    _create_read_items,
    find_state_dict_object,
)

from vescale.devicemesh_api import veDeviceMesh

logger: logging.Logger = logging.getLogger(__file__)

__all__ = [
    "VeScaleSavePlanner",
    "VeScaleLoadPlanner",
    "create_default_local_load_plan",
    "create_default_local_save_plan",
]


def sort_rank_ranges(process_list: List[Tuple]) -> List[Tuple]:
    """
    Decide which rank is receiver and writer
    Let rank with most parameters receives and writes tensors
    for the best communication cost
    If two ranks has the same data size, choose the smaller rank
    Args:
        A process list with tuples, each tuple is (rank, data_size)
    Returns:
        A sorted list, data size are sorted in descending order,
        if two ranks has the same data size, ranks are in the asceonding order
    """
    sorted_process_list = sorted(process_list, key=lambda x: (-x[1], x[0]))
    return sorted_process_list


def custom_dedup_tensors(all_plans: List[SavePlan]) -> List[SavePlan]:
    """
    A function to remove duplicate tensors to write
    when creating global writing plan for saving checkpoint
    """
    all_plans = list(all_plans)
    key_to_plan: Dict[MetadataIndex, List[int]] = {}
    for plan_idx, plan in enumerate(all_plans):
        for write_item in plan.items:
            # NOTE: the only difference from pytorch official
            if write_item.type != WriteItemType.SHARD:
                key_to_plan.setdefault(write_item.index, []).append(plan_idx)

    replicated_items = {k: v for k, v in key_to_plan.items() if len(v) > 1}

    # Remove duplicates by always keeping the first entry.
    # Compute the per-rank remove set.
    plan_to_keys: Dict[int, List[MetadataIndex]] = {}
    for key, plans in replicated_items.items():
        for plan_idx in plans[1:]:
            plan_to_keys.setdefault(plan_idx, []).append(key)
    logger.info("Duplicate keys to remove: %s", plan_to_keys)

    for plan_idx, keys in plan_to_keys.items():
        key_set = set(keys)
        # rewrite items and remove elements
        new_items = [write_item for write_item in all_plans[plan_idx].items if write_item.index not in key_set]
        all_plans[plan_idx] = dataclasses.replace(all_plans[plan_idx], items=new_items)

    return all_plans


class VeScaleLoadPlanner(DefaultLoadPlanner):
    """
    A planner class for loading vescale checkpoint using PyTorch DCP
    """

    def __init__(self):
        super().__init__()

    def create_local_plan(self) -> LoadPlan:
        return create_default_local_load_plan(self.state_dict, self.metadata)

    def resolve_tensor(self, read_item: ReadItem):
        tensor = self.lookup_tensor(read_item.dest_index)
        return self.transform_tensor(read_item, tensor)

    def lookup_tensor(self, index: MetadataIndex) -> torch.Tensor:
        """
        This is an extension from the planner interface to make it easy to extend the default planner
        """
        return find_state_dict_object(self.state_dict, index)


def create_default_local_load_plan(state_dict: Dict[str, Any], metadata: Metadata) -> LoadPlan:
    """
    A function for creating local loading plan for loading checkpoint
    """
    requests = []
    for fqn, obj in state_dict.items():
        md = metadata.state_dict_metadata[fqn]
        if isinstance(obj, DTensor):
            if obj.device_mesh.get_coordinate() is not None:
                requests += _create_read_items(fqn, md, obj)
        elif isinstance(obj, ShardedTensor):
            # For veScale DOptimizer, it will provide empty shards
            # if current process does not own the shard of tensor
            local_shards = obj.local_shards()
            total_size = 0
            for local_shard in local_shards:
                for size in local_shard.metadata.shard_sizes:
                    size += total_size
            if size > 0:
                requests += _create_read_items(fqn, md, obj)
        elif isinstance(obj, OptimizerStateSpec):
            # If the state is distributed on multiple dp ranks
            # Read with local_shape, then in DOptimizer then
            # get flaaten to 1D and get the part belonging to current dp rank
            if obj.dp_ranks_ranges:
                obj.local_tensor = torch.zeros(
                    obj.local_shape, dtype=obj.local_tensor.dtype, device=obj.local_tensor.device
                )
                requests += _create_read_items(fqn, md, obj)
            else:
                # If the state is owned by only one dp rank
                # Read directly
                obj.local_tensor = obj.local_tensor.reshape(obj.local_shape)
                requests += _create_read_items(fqn, md, obj)
        else:
            requests += _create_read_items(fqn, md, obj)
    return LoadPlan(requests)


class VeScaleSavePlanner(DefaultSavePlanner):
    """
    A planner class for saving vescale checkpoint using PyTorch DCP
    """

    def __init__(self):
        super().__init__()

    def resolve_data(self, write_item: WriteItem) -> Union[torch.Tensor, io.BytesIO]:
        object = self.lookup_object(write_item.index)
        return self.transform_object(write_item, object)

    def create_local_plan(self) -> SavePlan:
        plan = create_default_local_save_plan(self.state_dict, self.is_coordinator)
        if self.flatten_state_dict:
            plan = dataclasses.replace(plan, planner_data=self.mappings)
        self.plan = plan
        return self.plan

    def lookup_object(self, index: MetadataIndex) -> Any:
        return find_state_dict_object(self.state_dict, index)

    def create_global_plan(self, all_plans: List[SavePlan]) -> Tuple[List[SavePlan], Metadata]:
        self.dedup_replicated_tensors = True
        # all_plans = custom_dedup_tensors(all_plans)
        rst_value = super().create_global_plan(all_plans)
        return rst_value


def create_default_local_save_plan(state_dict: Dict[str, Any], is_coordinator: bool) -> SavePlan:
    """
    A function for creating local saving plan for saving checkpoint
    """
    requests = []
    for fqn, obj in state_dict.items():
        # Since DTensor supports submesh, adding extra check to ensure _create_write_items()
        # gets called only when the current rank is part of the mesh for the corresponding DTensor.
        if isinstance(obj, DTensor):
            if obj.device_mesh.get_coordinate() is not None:
                requests += _create_write_items(fqn, obj)
        elif isinstance(obj, ShardedTensor):
            # For veScale DOptimizer, it will provide empty shards
            # if current process does not own the shard of tensor
            local_shards = obj.local_shards()
            total_size = 0
            for local_shard in local_shards:
                for size in local_shard.metadata.shard_sizes:
                    size += total_size
            if size > 0:
                requests += _create_write_items(fqn, obj)
        elif isinstance(obj, OptimizerStateSpec):
            # Create write requests if the process is the real writer
            if obj.dp_ranks_ranges:
                process_list = []
                for rank, param_range in obj.dp_ranks_ranges.items():
                    process_list.append((rank, len(param_range)))
                sorted_list = sort_rank_ranges(process_list)
                writer_rank = sorted_list[0][0]
                p2p_ops = []
                recv_tensors = {}

                # Case 1: I am writer
                # Receive tensors

                if dist.get_rank() == writer_rank:
                    for k, param_range in obj.dp_ranks_ranges.items():
                        if k != dist.get_rank():
                            recv_tensor = torch.zeros(
                                (len(param_range),), dtype=obj.local_tensor.dtype, device=obj.local_tensor.device
                            )
                            recv_op = dist.P2POp(
                                op=dist.irecv,
                                tensor=recv_tensor,
                                peer=k,
                                group=veDeviceMesh.get_data_parallel_dim_groups(),
                            )
                            recv_tensors[k] = recv_tensor
                            p2p_ops.append(recv_op)
                else:
                    # Case 2: I am not writer
                    # Send my tensor
                    send_op = dist.P2POp(
                        op=dist.isend,
                        tensor=obj.local_tensor,
                        peer=writer_rank,
                        group=veDeviceMesh.get_data_parallel_dim_groups(),
                    )
                    p2p_ops.append(send_op)

                reqs = dist.batch_isend_irecv(p2p_ops)

                for req in reqs:
                    req.wait()

                if writer_rank == dist.get_rank():
                    new_local_tensor = torch.zeros(
                        (math.prod(obj.local_shape),), dtype=obj.local_tensor.dtype, device=obj.local_tensor.device
                    )
                    new_local_tensor[obj.dp_ranks_ranges[writer_rank].start : obj.dp_ranks_ranges[writer_rank].end] = (
                        obj.local_tensor
                    )
                    for k, param_range in obj.dp_ranks_ranges.items():
                        if k != writer_rank:
                            new_local_tensor[param_range.start : param_range.end] = recv_tensors[k]
                    obj.local_tensor = new_local_tensor

                    obj.local_tensor = obj.local_tensor.reshape(obj.local_shape)
                    requests += _create_write_items(fqn, obj)
            else:
                obj.local_tensor = obj.local_tensor.reshape(obj.local_shape)
                requests += _create_write_items(fqn, obj)
        elif isinstance(obj, (torch.Tensor)) or is_coordinator:
            requests += _create_write_items(fqn, obj)

    return SavePlan(requests)
