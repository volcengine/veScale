################################################################################
#
# Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
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

from typing import List, Any, Dict
from abc import ABC, abstractmethod
from ..variables import NDTIMELINE_FLUSH_SEPCIAL
from ..world_info import WorldInfo


class NDHandler(ABC):
    def __init__(self, designated_key="", ignore_metrics=None) -> None:
        super().__init__()
        self._dispatch_key = self.__class__.__name__
        self._ignore_metrics = ignore_metrics if ignore_metrics is not None else [NDTIMELINE_FLUSH_SEPCIAL]
        if designated_key != "":
            self._dispatch_key = designated_key

    @property
    def dispatch_key(self):
        return self._dispatch_key

    @property
    def ignore_metrics(self):
        return self._ignore_metrics

    def __repr__(self) -> str:
        return f"NDHandler instance with dispatch key: {self._dispatch_key}"

    def __call__(
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
        if metric_name in self.ignore_metrics:
            return
        return self.call_impl(
            metric_name,
            elapsed,
            recent_elapsed_raw_parts,
            recent_since_start_raw_parts,
            tags,
            step_range,
            world_info,
            extra,
        )

    @abstractmethod
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
        pass
