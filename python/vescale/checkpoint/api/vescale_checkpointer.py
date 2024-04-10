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
from .base_checkpointer import BaseCheckpointer
from .meta_type import CheckpointState, MODEL_STR, OPTIMIZER_STR
from ..save_state_dict import save_state_dict
from ..load_state_dict import load_state_dict
from ..planner.vescale.vescale_planner import VeScaleSavePlanner, VeScaleLoadPlanner


from ..utilities import bfile
import os
from vescale.optim.distributed_optimizer import initialize_optimizer_state
import torch.distributed as dist
from ..utilities.logger import get_omnistore_logger

logger = get_omnistore_logger()

VESCALE_SUPPORTED_TYPES = {MODEL_STR, OPTIMIZER_STR}


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


class VeScaleCheckpointer(BaseCheckpointer):
    """
    The Checkpointer class for VeScale, A PyTorch Native Auto Parallelism Framework
    """

    save_planner = VeScaleSavePlanner()
    load_planner = VeScaleLoadPlanner()

    @classmethod
    def save(cls, path: str, checkpoint_state: CheckpointState):
        # Check if we support saving the components
        for key in checkpoint_state.keys():
            if key not in VESCALE_SUPPORTED_TYPES:
                raise ValueError(f"{key} is not supported by VeScaleCheckpointer")
        if bfile.is_local_path(path):
            logger.warning(
                "The local path for checkpointing should be accessible to all ranks. It can be a NFS/FUSE path"
            )

        # Start saving checkpoint
        for key, value in checkpoint_state.items():
            if key == MODEL_STR:
                # Get model path
                model_path = os.path.join(path, MODEL_STR)
                # Create a "model" folder on under root path
                if dist.get_rank() == 0:
                    bfile.makedirs(model_path)
                dist.barrier()
                # Save model
                save_state_dict(
                    state_dict=value.state_dict(),
                    path=model_path,
                    process_group=None,
                    coordinator_rank=0,
                    no_dist=False,
                    planner=cls.save_planner,
                )
            elif key == OPTIMIZER_STR:
                # Create a "optimizer" folder on under root path
                # to save different parts of optimizer
                optim_root_path = os.path.join(path, OPTIMIZER_STR)
                if dist.get_rank() == 0:
                    bfile.makedirs(optim_root_path)
                dist.barrier()

                # Get optimizer path based on PP rank
                optimizer_path = os.path.join(optim_root_path, "pp_0")
                # Create optimizer folder on under root path
                if dist.get_rank() == 0:
                    bfile.makedirs(optimizer_path)
                dist.barrier()

                save_state_dict(
                    state_dict=value.state_dict(),
                    path=optimizer_path,
                    process_group=None,
                    coordinator_rank=0,
                    no_dist=False,
                    planner=cls.save_planner,
                )

    @classmethod
    def load(cls, path: str, checkpoint_state: CheckpointState):
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
                # Load model
                load_state_dict(
                    state_dict=model_state,
                    path=model_path,
                    process_group=None,
                    coordinator_rank=0,
                    no_dist=False,
                    planner=cls.load_planner,
                )
                # Load back to model
                value.load_state_dict(model_state)
            elif key == OPTIMIZER_STR:
                optimizer_path = os.path.join(path, f"{OPTIMIZER_STR}", "pp_0")
                # Initialize optimizer states
                initialize_optimizer_state(value)
                # Get optimizer state
                optimizer_state = value.state_dict()
                # Load optimizer state dictionary
                load_state_dict(
                    state_dict=optimizer_state,
                    path=optimizer_path,
                    process_group=None,
                    coordinator_rank=0,
                    no_dist=False,
                    planner=cls.load_planner,
                )
                # Load back to optimizer
                value.load_state_dict(optimizer_state)
            dist.barrier()
