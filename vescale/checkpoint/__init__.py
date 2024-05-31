################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################
# The "checkpoint" folder is ONLY USED for "open source" version veScale

from .api.vescale_checkpointer import VeScaleCheckpointer
from .api.meta_type import CheckpointState


def save(path: str, checkpoint_state: CheckpointState, async_checkpoint=False):
    """
    Save a checkpoint to a given path
    Args:
        path: Defines the storage path for checkpoint.
        checkpoint_state: A dictionary contains key-value pairs for model and optimizer.
                            - Model: Identified by 'model' key, value should be a model instance.
                            - Optimizer: Identified by 'optimizer' key, value should be an optimizer instance.
        async_checkpoint: A boolean value indicating if saving checkpoint asynchronously,
                                 i.e. after dumping tensors from GPU memory to Host memory,
                                 the training program can continue training immediately.
                                 Then vescale.checkpoint will serialize tensors and dumping to the persistent storage asynchronously.
    Example:
        >>> checkpoint_state = { "model": distributd_model, "optimizer": distributed_optimizer }
        >>> vescale.checkpoint.save("/user/vescale/gpt/", checkpoint_state)
    """
    VeScaleCheckpointer.save(path, checkpoint_state, async_checkpoint=async_checkpoint)


def load(path: str, checkpoint_state: CheckpointState, broadcast_checkpoint=False):
    """
    Load a checkpoint from a given path
    Args:
        path: Defines the storage path for checkpoint.
        checkpoint_state: A dictionary contains key-value pairs for model and optimizer.
                            - Model: Identified by 'model' key, value should be a model instance.
                            - Optimizer: Identified by 'optimizer' key, value should be an optimizer instance.
        broadcast_checkpoint: A boolean value decides if load a model replica from one data parallel process group
                                 then broadcast tensors to other data parallel process group using GPUs
                                 to reduce the file system access
                                 For example, when data parellel size = 2,
                                 processes with data parallel rank = 0 load model from file system
                                 then broadcast it to processes with data parallel rank = 1
    Example:
        >>> checkpoint_state = { "model": distributd_model, "optimizer": distributed_optimizer }
        >>> vescale.checkpoint.load("/user/vescale/gpt/", checkpoint_state)
    """
    VeScaleCheckpointer.load(path, checkpoint_state, broadcast_checkpoint=broadcast_checkpoint)
