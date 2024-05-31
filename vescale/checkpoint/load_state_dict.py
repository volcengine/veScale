################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

from typing import Optional
import torch.distributed as dist
from torch.distributed.checkpoint.planner import LoadPlanner
from torch.distributed.checkpoint.utils import _DistWrapper
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from .storage.filesystem import FileSystemReader
from .api.meta_type import STATE_DICT_TYPE
import time
from .utilities.logger import get_vescale_checkpoint_logger
from vescale.checkpoint.planner.vescale.vescale_planner import VeScaleLoadPlanner

logger = get_vescale_checkpoint_logger()

META_DATA_FILE = ".metadata"


def load_state_dict(
    state_dict: STATE_DICT_TYPE,
    path: str,
    process_group: Optional[dist.ProcessGroup] = None,
    coordinator_rank: int = 0,
    no_dist: bool = False,
    planner: Optional[LoadPlanner] = None,
    broadcast_tensors=False,
) -> None:
    load_start_time = time.time()
    """
    [veScale version] Loads a distributed ``state_dict`` in SPMD style. Fix sub-group storage.
    """
    storage_reader = FileSystemReader(
        path,
        broadcast_tensors=broadcast_tensors,
        data_parallel_process_group=process_group,
    )

    # Step 0: create distributed world based on process group and coordinator rank
    distW = _DistWrapper(process_group, not no_dist, coordinator_rank)
    if process_group:
        distW.coordinator_rank = dist.get_global_rank(process_group, distW.coordinator_rank)
    if planner is None:
        planner = DefaultLoadPlanner()
    plan_start_time = time.time()

    # Step 1: all processes create local read plan,
    # then coordinator gathers all local plans and create global plan.
    def local_step():
        assert planner is not None
        meta_read_start_time = time.time()
        metadata = storage_reader.read_metadata()
        meat_read_cost_time = time.time() - meta_read_start_time
        logger.info(f"Finish read meta file. Cost time: {meat_read_cost_time}s")
        planner.set_up_planner(state_dict, metadata, distW.is_coordinator)
        storage_reader.set_up_storage_reader(metadata, distW.is_coordinator)

        local_plan = planner.create_local_plan()
        local_plan = storage_reader.prepare_local_plan(local_plan)
        return local_plan

    def global_step(all_local_plans):
        assert planner is not None
        all_local_plans = planner.create_global_plan(all_local_plans)
        all_local_plans = storage_reader.prepare_global_plan(all_local_plans)
        return all_local_plans

    if isinstance(planner, VeScaleLoadPlanner):
        central_plan = distW.reduce_scatter("plan", local_step, global_step)
    else:
        raise AssertionError("Unsupported planner for saving checkpoint")
    load_ckpt_plan_cost_time = time.time() - plan_start_time
    logger.info(f"Finish planning. Cost time: {load_ckpt_plan_cost_time}s")

    read_start_time = time.time()

    # Step 2: all processes read data from the given path
    def read_data():
        assert planner is not None
        final_local_plan = planner.finish_plan(central_plan)
        all_reads = storage_reader.read_data(final_local_plan, planner)
        all_reads.wait()
        return None

    _ = distW.all_gather("read", read_data)
    read_cost_time = time.time() - read_start_time
    logger.info(f"Finish reading. Cost time: {read_cost_time}s")

    load_ckpt_cost_time = time.time() - load_start_time
    logger.info(f"Finish loading. Cost time: {load_ckpt_cost_time}s")
