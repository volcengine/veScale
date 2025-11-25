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

from . import handlers  # noqa: F401
from . import exceptions  # noqa: F401
from . import logger  # noqa: F401
from . import predefined  # noqa: F401

from .binary_protocol import serialize_to_package, encode_package, loads_fn, dumps_fn
from .pool import DefaultEventPool, CudaEventPool
from .world_info import WorldInfo, TrainingInfo, TopoInfo
from .timer import (
    NDTimerManager,
    NDTimerManagerSingleton,
    DeviceTimerMeta,
    ndtimeit,
    NDMetricLevel,
    ndtimer,
    ndtimeit_p2p,
)
from .sock_streamer import NDtimelineStreamer
from .variables import (
    NDTIMELINE_INNER_GLOBAL_STEP_KEY,
    SOCK_TIMEOUT_CLIENT,
    SOCK_PARENT_DIR,
    SOCK_PATH,
    NDTIMELINE_STREAM_KEY,
)
from .stream import get_nccl_p2p_stream, get_nccl_coll_stream
from .api import flush, wait, init_ndtimers, set_global_step, inc_step

__all__ = [
    "handlers",
    "logger",
    "exceptions",
    "predefined",
    "serialize_to_package",
    "encode_package",
    "loads_fn",
    "dumps_fn",
    "DefaultEventPool",
    "CudaEventPool",
    "WorldInfo",
    "TrainingInfo",
    "TopoInfo",
    "NDTimerManager",
    "NDTimerManagerSingleton",
    "DeviceTimerMeta",
    "ndtimeit",
    "NDtimelineStreamer",
    "NDTIMELINE_INNER_GLOBAL_STEP_KEY",
    "SOCK_TIMEOUT_CLIENT",
    "SOCK_PARENT_DIR",
    "SOCK_PATH",
    "NDTIMELINE_STREAM_KEY",
    "NDMetricLevel",
    "get_nccl_p2p_stream",
    "get_nccl_coll_stream",
    "ndtimer",
    "ndtimeit_p2p",
    "flush",
    "wait",
    "init_ndtimers",
    "set_global_step",
    "inc_step",
]

try:
    import _internal

    __all__.append("_internal")
except ImportError:
    pass
