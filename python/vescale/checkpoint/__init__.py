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
# If you use veScale in ByteDance, please use OmniStore

from .api.vescale_checkpointer import VeScaleCheckpointer
from .api.meta_type import CheckpointState


def save(path: str, checkpoint_state: CheckpointState):
    """
    Save a checkpoint to a given path
    Args:
        path: Defines the storage path for checkpoint.
        checkpoint_state: A dictionary contains key-value pairs for model and optimizer.
                            - Model: Identified by 'model' key, value should be a model instance.
                            - Optimizer: Identified by 'optimizer' key, value should be an optimizer instance.
    Example:
        >>> checkpoint_state = { "model": distributd_model, "optimizer": distributed_optimizer }
        >>> vescale.checkpoint.save("/user/vescale/gpt/", checkpoint_state)
    """
    VeScaleCheckpointer.save(path, checkpoint_state)


def load(path: str, checkpoint_state: CheckpointState):
    """
    Load a checkpoint from a given path
    Args:
        path: Defines the storage path for checkpoint.
        checkpoint_state: A dictionary contains key-value pairs for model and optimizer.
                            - Model: Identified by 'model' key, value should be a model instance.
                            - Optimizer: Identified by 'optimizer' key, value should be an optimizer instance.
    Example:
        >>> checkpoint_state = { "model": distributd_model, "optimizer": distributed_optimizer }
        >>> vescale.checkpoint.load("/user/vescale/gpt/", checkpoint_state)
    """
    VeScaleCheckpointer.load(path, checkpoint_state)
