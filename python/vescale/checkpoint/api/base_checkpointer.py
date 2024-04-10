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
from .meta_type import CheckpointState


class BaseCheckpointer:
    """
    The Checkpointer class offers APIs that enable users to save and load state dictionarie.
    It is designed for extension across various training frameworks.
    """

    @classmethod
    def save(cls, path: str, checkpoint_state: CheckpointState):
        """
        A Method for saving checkpoint
        Args:
            path: Defines the storage path for checkpoint.
            checkpoint_state: A dictionary contains key-value pairs for model, optimizer and dataloader(TODO).
                              - Model: Identified by 'model' key, value should be a model instance.
                              - Optimizer: Identified by 'optimizer' key, value should be an optimizer instance.
        """
        raise NotImplementedError()

    def load(cls, path: str, checkpoint_state: CheckpointState):
        """
        A Method for loading checkpoint
        Args:
            path: Defines the storage path for checkpoint.
            checkpoint_state: A dictionary contains key-value pairs for model, optimizer and dataloader(TODO).
                              - Model: Identified by 'model' key, value should be a model instance.
                              - Optimizer: Identified by 'optimizer' key, value should be an optimizer instance.
        """
        raise NotImplementedError()
