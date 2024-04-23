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
import torch
from .utilities.mem_checkpoint import TorchCheckpointRecorder
import torch.distributed as dist
from torch.distributed.checkpoint.filesystem import FileSystemWriter
from torch.distributed.checkpoint.planner import SavePlanner
from torch.distributed.checkpoint.utils import _DistWrapper
from torch.distributed.checkpoint.metadata import Metadata
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
from .api.meta_type import STATE_DICT_TYPE
from .utilities.logger import get_omnistore_logger
import time
import atexit

logger = get_omnistore_logger()
_io_workers = None


def _clean_up():
    if _io_workers:
        _io_workers.terminate()
        _io_workers.join()


atexit.register(_clean_up)


def save_state_dict(
    state_dict: STATE_DICT_TYPE,
    path: str,
    # storage_writer: StorageWriter,
    process_group: Optional[dist.ProcessGroup] = None,
    coordinator_rank: int = 0,
    no_dist: bool = False,
    planner: Optional[SavePlanner] = None,
    strategy=None,
) -> Metadata:
    """
    [veScale version] Saves a distributed model in SPMD style. Fix sub-group storage.
    Args and usage is the same as `torch.distributed.checkpoint.save_state_dict`.
    """
    save_ckpt_start_time = time.time()
    torch._C._log_api_usage_once("omnistore.checkpoint.vescale_checkpoint.save_state_dict")

    # Step 0: create distributed world based on process group and coordinator rank
    distW = _DistWrapper(process_group, not no_dist, coordinator_rank)
    if process_group:
        distW.coordinator_rank = dist.get_global_rank(process_group, distW.coordinator_rank)
    if planner is None:
        planner = DefaultSavePlanner()
    assert planner is not None

    global_metatadata = None

    storage_writer = FileSystemWriter(path)

    # Step 1: all processes create local write plan,
    # then coordinator gathers all local plans and create global plan.
    def local_step():
        assert planner is not None
        planner.set_up_planner(state_dict, distW.is_coordinator)
        storage_writer.set_up_storage_writer(distW.is_coordinator)
        local_plan = planner.create_local_plan()
        local_plan = storage_writer.prepare_local_plan(local_plan)
        return local_plan

    def global_step(all_local_plans):
        nonlocal global_metatadata

        assert planner is not None
        all_local_plans, global_metatadata = planner.create_global_plan(all_local_plans)
        all_local_plans = storage_writer.prepare_global_plan(all_local_plans)
        return all_local_plans

    plan_start_time = time.time()
    central_plan = distW.reduce_scatter("plan", local_step, global_step)
    plan_cost_time = time.time() - plan_start_time
    logger.info(f"Finish planning. Cost time: {plan_cost_time}s")

    # Step 2: all processes write data from GPUs to pinned memory pool, then dump to local path
    # then coordinator write meta-data to local path.
    def write_data():
        assert planner is not None
        final_local_plan = planner.finish_plan(central_plan)
        # Use pinned memory pool and mult_processing for dumping ckpt to local directory efficiently
        global _io_workers
        if not _io_workers:
            _io_workers = torch.multiprocessing.get_context("spawn").Pool(2)
        with TorchCheckpointRecorder(async_worker=_io_workers):
            all_writes = storage_writer.write_data(final_local_plan, planner)
        all_writes.wait()
        return all_writes.value()

    def finish_checkpoint(all_results):
        assert global_metatadata is not None
        storage_writer.finish(metadata=global_metatadata, results=all_results)
        return global_metatadata

    dump_local_start_time = time.time()
    all_reduce_results = distW.all_reduce("write", write_data, finish_checkpoint)
    dump_local_cost_time = time.time() - dump_local_start_time
    logger.info(f"Finish dumping. Cost time: {dump_local_cost_time}s")

    return all_reduce_results
