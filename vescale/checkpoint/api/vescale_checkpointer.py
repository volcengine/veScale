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

from concurrent.futures import ProcessPoolExecutor
from .base_checkpointer import BaseCheckpointer
from .meta_type import CheckpointState, MODEL_STR, OPTIMIZER_STR
from ..save_state_dict import save_state_dict
from ..load_state_dict import load_state_dict
from ..planner.vescale.vescale_planner import VeScaleSavePlanner, VeScaleLoadPlanner
from vescale.devicemesh_api import VESCALE_DEVICE_MESH
from ..utilities import bfile
import os
from vescale.optim.distributed_optimizer import initialize_optimizer_state
import torch.distributed as dist
from ..utilities.logger import get_vescale_checkpoint_logger
import atexit

logger = get_vescale_checkpoint_logger()

VESCALE_SUPPORTED_TYPES = {MODEL_STR, OPTIMIZER_STR}
NUM_IO_WORKER = 1


def deduplicate_2d_list(lst):
    seen = set()
    deduplicated_list = []
    for item in lst:
        # Convert the inner list to a tuple for hashing
        tuple_item = tuple(sorted(item))  # Sorting to treat [1, 2] and [2, 1] as the same
        if tuple_item not in seen:
            seen.add(tuple_item)
            # Convert back to list to preserve original type
            deduplicated_list.append(item)
    return deduplicated_list


def get_optim_ckpt_process_group():
    # Get the process group based on current rank
    # The processes with same pipeline stage ID
    # are in the same process group
    device_mesh = VESCALE_DEVICE_MESH.get()
    sub_mesh = device_mesh.get_submesh(mesh_dims=["TP", "DP"])
    two_dim_list = sub_mesh.mesh.tolist()
    flatten_rank_list = [item for sublist in two_dim_list for item in sublist]
    all_flatten_lists = [[] for _ in range(dist.get_world_size())]
    dist.all_gather_object(all_flatten_lists, flatten_rank_list)
    all_flatten_lists = deduplicate_2d_list(all_flatten_lists)
    my_rank = dist.get_rank()
    pg = None
    for rank_list in all_flatten_lists:
        new_pg = dist.new_group(ranks=flatten_rank_list)
        if my_rank in rank_list:
            pg = new_pg
    return pg


class VeScaleCheckpointer(BaseCheckpointer):
    """
    The Checkpointer class for VeScale, A PyTorch Native Auto Parallelism Framework
    """

    save_planner = VeScaleSavePlanner()
    load_planner = VeScaleLoadPlanner()

    optim_ckpt_proces_group = None
    for key in VESCALE_SUPPORTED_TYPES:
        BaseCheckpointer.state_io_workers[key] = ProcessPoolExecutor(max_workers=NUM_IO_WORKER)
        BaseCheckpointer.state_write_futures[key] = []

    @classmethod
    def save(
        cls,
        path: str,
        checkpoint_state: CheckpointState,
        async_checkpoint: bool = False,
    ):
        """
        async_checkpoint: A boolean value indicating if saving checkpoint asynchronously,
                                   i.e. after dumping tensors from GPU memory to Host memory,
                                   the training program can continue training immediately.
                                   Then vescale.checkpoint will serialize tensors and dumping to the persistent storage asynchronously.
        """
        # Check if we support saving the components
        for key in checkpoint_state.keys():
            if key not in VESCALE_SUPPORTED_TYPES:
                raise ValueError(f"{key} is not supported by VeScaleCheckpointer")

        # Start saving checkpoint
        for key, value in checkpoint_state.items():
            if key == MODEL_STR:
                # Get model path
                model_path = os.path.join(path, MODEL_STR)
                # Create a "model" folder on under root path
                if dist.get_rank() == 0:
                    bfile.makedirs(model_path)
                dist.barrier()
                # Save model.
                _, new_write_futures = save_state_dict(
                    state_dict=value.state_dict(),
                    path=model_path,
                    process_group=None,
                    coordinator_rank=0,
                    no_dist=False,
                    planner=cls.save_planner,
                    async_io=async_checkpoint,
                    last_write_futures=cls.state_write_futures[MODEL_STR],
                    io_workers=cls.state_io_workers[MODEL_STR],
                )
                # Record new write futures.
                cls.state_write_futures[MODEL_STR] = new_write_futures
            elif key == OPTIMIZER_STR:
                # Create a "optimizer" folder on under root path
                # to save different parts of optimizer
                optim_root_path = os.path.join(path, OPTIMIZER_STR)
                if dist.get_rank() == 0:
                    bfile.makedirs(optim_root_path)
                dist.barrier()
                # Get process group for saving optimizer,
                # All processes with the same pipeline rank are in the same pg
                if not cls.optim_ckpt_proces_group:
                    cls.optim_ckpt_proces_group = get_optim_ckpt_process_group()

                # Get optimizer path based on PP rank
                pp_rank = VESCALE_DEVICE_MESH.get_pipeline_parallel_rank()
                optimizer_path = os.path.join(optim_root_path, f"pp_{pp_rank}")
                # Create optimizer folder on under root path
                if dist.get_rank(cls.optim_ckpt_proces_group) == 0:
                    bfile.makedirs(optimizer_path)
                dist.barrier()
                # Save optimizer
                _, new_write_futures = save_state_dict(
                    state_dict=value.state_dict(),
                    path=optimizer_path,
                    process_group=cls.optim_ckpt_proces_group,
                    coordinator_rank=0,
                    no_dist=False,
                    planner=cls.save_planner,
                    async_io=async_checkpoint,
                    last_write_futures=cls.state_write_futures[OPTIMIZER_STR],
                    io_workers=cls.state_io_workers[OPTIMIZER_STR],
                )
                # Record new write futures.
                cls.state_write_futures[OPTIMIZER_STR] = new_write_futures

    @classmethod
    def load(
        cls,
        path: str,
        checkpoint_state: CheckpointState,
        broadcast_checkpoint: bool = False,
    ):
        """
        broadcast_checkpoint: A boolean value decides if load a model replica from one data parallel process group
                                 then broadcast tensors to other data parallel process group using GPUs
                                 to reduce the file system access
                                 For example, when data parellel size = 2,
                                 processes with data parallel rank = 0 load model from file system
                                 then broadcast it to processes with data parallel rank = 1
        """
        # Add warning
        if bfile.is_local_path(path):
            logger.warning(
                "The local path for checkpointing should be accessible to all ranks. It can be a NFS/FUSE path"
            )
        # Check if we support loading the component.
        for key in checkpoint_state.keys():
            if key not in VESCALE_SUPPORTED_TYPES:
                raise ValueError(f"{key} is not supported by VeScaleCheckpointer")

        # Start loading checkpoint
        for key, value in checkpoint_state.items():
            if key == MODEL_STR:
                # Get model path
                model_path = os.path.join(path, MODEL_STR)
                # Get model state dictionary
                model_state = value.state_dict()
                # Set process group
                if broadcast_checkpoint:
                    model_load_process_group = VESCALE_DEVICE_MESH.get_data_parallel_dim_groups()
                else:
                    model_load_process_group = None
                # Load model
                load_state_dict(
                    state_dict=model_state,
                    path=model_path,
                    process_group=model_load_process_group,
                    coordinator_rank=0,
                    no_dist=False,
                    planner=cls.load_planner,
                    broadcast_tensors=broadcast_checkpoint,
                )
                # Load back to model
                value.load_state_dict(model_state)
            elif key == OPTIMIZER_STR:
                # Get process group for loading optimizer,
                # All processes with the same pipeline rank are in the same pg
                if not cls.optim_ckpt_proces_group:
                    cls.optim_ckpt_proces_group = get_optim_ckpt_process_group()
                # Get optimizer path based on TP and PP ranks
                pp_rank = VESCALE_DEVICE_MESH.get_pipeline_parallel_rank()
                optimizer_path = os.path.join(path, f"{OPTIMIZER_STR}", f"pp_{pp_rank}")
                # Initialize optimizer states
                initialize_optimizer_state(value)
                # Get optimizer state
                optimizer_state = value.state_dict()
                # Load optimizer state dictionary
                load_state_dict(
                    state_dict=optimizer_state,
                    path=optimizer_path,
                    process_group=cls.optim_ckpt_proces_group,
                    coordinator_rank=0,
                    no_dist=False,
                    planner=cls.load_planner,
                    broadcast_tensors=False,
                )
                # Load back to optimizer
                value.load_state_dict(optimizer_state)
            dist.barrier()

    @classmethod
    def __cleanup(cls):
        """
        Wait for all write futures to finish before exit, then do the cleanup works.

        WARNING: this method cannot be called by the users.
        """
        cls.save_planner.clear_cache()
        BaseCheckpointer._cleanup_futures()

    @classmethod
    def _register_cleanup(cls):
        atexit.register(VeScaleCheckpointer.__cleanup)


VeScaleCheckpointer._register_cleanup()
