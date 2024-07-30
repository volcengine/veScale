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

import logging
import os
from typing import List, Dict, Any
from logging import Formatter
from logging.handlers import RotatingFileHandler

from .handler_base import NDHandler
from ..world_info import WorldInfo
from ..variables import LOCAL_LOGGING_PATH


CHUNK_SZ = 1024 * 1024 * 128  # 128 MiB
BACKUP_CNT = 8


class LocalRawNDHandler(NDHandler):
    def __init__(
        self, run_id: int, log_path: str = LOCAL_LOGGING_PATH, chunk_sz: int = CHUNK_SZ, backup_cnt: int = BACKUP_CNT
    ) -> None:
        """if a trial of log exceeds `chunk_sz`, it will be dropped"""
        super().__init__()
        if not os.path.exists(log_path):
            os.makedirs(log_path, exist_ok=True)
        file_name = f"timeline_run{run_id}_raw.log"
        formatter = Formatter("%(asctime)s - %(message)s")
        handler = RotatingFileHandler(
            filename=os.path.join(log_path, file_name), maxBytes=chunk_sz, backupCount=backup_cnt
        )
        handler.setFormatter(formatter)
        self.logger = logging.getLogger("LocalRawNDHandler")
        self.logger.propagate = False
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)

    def call_impl(
        self,
        metric_name: str,
        elapsed: float,
        recent_elapsed_raw_parts: List[float],
        recent_since_start_raw_parts: List[float],
        tags: List[Dict[str, Any]],
        step_range: range,
        world_info: WorldInfo,
        extra: Dict[str, Any],
    ) -> Any:
        msg = (
            f"metric_name: {metric_name}, elapsed: {elapsed}, recent_elapsed_raw_parts: {recent_elapsed_raw_parts}, recent_since_start_raw_parts: {recent_since_start_raw_parts},"
            f" tags: {tags}, step_range: {step_range}, world_info: {world_info}"
        )
        self.logger.info(msg)
