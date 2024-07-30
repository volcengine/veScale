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

from typing import List, Dict, Any

from .handler_base import NDHandler
from ..world_info import WorldInfo
from ..logger import NDTimelineLogger


class LoggingNDHandler(NDHandler):
    def __init__(self) -> None:
        super().__init__()

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
        NDTimelineLogger().debug(
            f"#recent_elapsed_raw_parts: {len(recent_elapsed_raw_parts)}, #recent_since_start_raw_parts {len(recent_since_start_raw_parts)}"
        )
        if len(step_range) < 1:
            raise ValueError(f"step_range length is {len(step_range)}")
        NDTimelineLogger().info(
            f"[rank{world_info.topo_info.rank}, step{step_range[0]}-{step_range[-1]}]: {len(recent_since_start_raw_parts)} times {metric_name} total cost: {elapsed*1000:.2f}ms"
        )
