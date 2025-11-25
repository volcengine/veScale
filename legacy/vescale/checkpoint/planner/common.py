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
import dataclasses
from typing import Any, Dict, List, Tuple, Hashable, Optional
from collections import OrderedDict
from torch.distributed.checkpoint.planner import SavePlan
from torch.distributed.checkpoint.metadata import MetadataIndex, Metadata
import collections
from ..utilities.logger import get_vescale_checkpoint_logger

logger = get_vescale_checkpoint_logger()


@dataclasses.dataclass
class P2PTensorsInfo:
    """
    Record data about tesnors which are across dp ranks
    recv_tensors: A dictionary
        Key: fqn
        Value: a dictionary
               key is the process rank,
               value is a tuple with (tensor, 1d_range)
    send_p2p_reqs: a list of p2p send requests to wait
    recv_p2p_reqs: a list p2p receive requests to wait
    """

    recv_tensors: Dict[str, Any]
    send_p2p_reqs: List[Any]
    recv_p2p_reqs: List[Any]


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


_MAX_CACHE_SIZE = 8


class PlanLRUCache:
    def __init__(self) -> None:
        self._cache: OrderedDict[Hashable, Tuple[SavePlan, Metadata]] = OrderedDict()
        self._capacity = _MAX_CACHE_SIZE

    def get(self, key: Hashable) -> Optional[Tuple[SavePlan, Metadata]]:
        if key in self._cache:
            return self._cache[key]
        else:
            return None

    def put(self, key: Hashable, plan_value: SavePlan, metadata_value: Metadata) -> None:
        if key in self._cache:
            self._cache.move_to_end(key, last=False)
        else:
            self._cache[key] = (plan_value, metadata_value)
            if len(self._cache) > self._capacity:
                self._cache.popitem()

    def clear(self) -> None:
        self._cache.clear()
        self._capacity = _MAX_CACHE_SIZE

    def __repr__(self) -> str:
        return f"PlanLURCache(capacity: {self._capacity}, keys: {tuple(self._cache.keys())})"


def custom_dedup_tensors(all_plans: List[SavePlan]) -> List[SavePlan]:
    """
    A function to remove duplicate tensors to write
    when creating global writing plan for saving checkpoint
    During the deduplication,
    we balance the workloads for duplicated tensors
    """
    key_to_plan: Dict[MetadataIndex, List[int]] = {}
    for plan_idx, plan in enumerate(all_plans):
        for write_item in plan.items:
            key_to_plan.setdefault(write_item.index, []).append(plan_idx)

    replicated_items = {k: v for k, v in key_to_plan.items() if len(v) > 1}
    # Remove duplicates by always keeping the first entry (Not balance).
    # Compute the per-rank remove set.
    plan_to_keys: Dict[int, List[MetadataIndex]] = {}
    # Record the number of non-duplicated tensors assigned to each rank
    assigned_work_load = collections.defaultdict(int)
    for plan_idx, plan in enumerate(all_plans):
        for write_item in plan.items:
            if write_item.index not in replicated_items:
                assigned_work_load[plan_idx] += 1

    for key, plans in replicated_items.items():
        # For duplicated tensors, select the rank assigned with minimum number tensors so far
        writer_id = min(plans, key=lambda k: assigned_work_load[k])
        assigned_work_load[writer_id] += 1
        for plan_idx in plans:
            # If the rank is not writer rank, remove the key in the rank's plan
            if plan_idx != writer_id:
                plan_to_keys.setdefault(plan_idx, []).append(key)
    logger.info("Duplicate keys to remove: %s", plan_to_keys)

    for plan_idx, keys in plan_to_keys.items():
        # Key Set contains keys to remove
        key_set = set(keys)
        # rewrite items and remove elements
        new_items = [write_item for write_item in all_plans[plan_idx].items if write_item.index not in key_set]
        all_plans[plan_idx] = dataclasses.replace(all_plans[plan_idx], items=new_items)

    return all_plans
