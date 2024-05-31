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
from typing import Dict, List
from concurrent.futures import Future, ProcessPoolExecutor
from torch.distributed.checkpoint.storage import WriteResult
from .meta_type import MODEL_STR, OPTIMIZER_STR

SUPPORTED_TYPES = {MODEL_STR, OPTIMIZER_STR}


class BaseCheckpointer:
    """
    The Checkpointer class offers APIs that enable users to save and load state dictionarie.
    It is designed for extension across various training frameworks.
    """

    # Async IO related members.
    state_io_workers: Dict[str, ProcessPoolExecutor] = {}
    state_write_futures: Dict[str, Future[List[WriteResult]]] = {}

    @classmethod
    def save(cls, path: str, checkpoint_state: CheckpointState):
        """
        A Method for saving checkpoint
        Args:
            path: Defines the storage path for checkpoint.
            checkpoint_state: A dictionary contains key-value pairs for model and optimizer.
                              - Model: Identified by 'model' key, value should be a model instance.
                              - Optimizer: Identified by 'optimizer' key, value should be an optimizer instance.

        """
        raise NotImplementedError()

    @classmethod
    def load(cls, path: str, checkpoint_state: CheckpointState):
        """
        A Method for loading checkpoint
        Args:
            path: Defines the storage path for checkpoint.
            checkpoint_state: A dictionary contains key-value pairs for model and optimizer.
                              - Model: Identified by 'model' key, value should be a model instance.
                              - Optimizer: Identified by 'optimizer' key, value should be an optimizer instance.

        """
        raise NotImplementedError()

    @classmethod
    def _cleanup_futures(cls):
        """
        Wait for all write futures to finish before exit, then do the cleanup works.

        WARNING: this method cannot be called by the users.
        """
        for key in SUPPORTED_TYPES:
            if key in cls.state_write_futures:
                futures = cls.state_write_futures[key]
                for fut in futures:
                    fut.result()
                cls.state_write_futures[key] = []
                if cls.state_io_workers[key] is not None:
                    cls.state_io_workers[key].shutdown()
                    cls.state_io_workers[key] = None
