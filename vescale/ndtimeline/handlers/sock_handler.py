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

import time
import traceback
import socket
from typing import List, Dict, Any

from ..logger import NDTimelineLogger
from ..binary_protocol import serialize_to_package
from .handler_base import NDHandler
from ..world_info import WorldInfo
from ..variables import SOCK_PATH, SOCK_TIMEOUT_CLIENT


class SockNDHandler(NDHandler):
    def __init__(self, timeout: float = SOCK_TIMEOUT_CLIENT, sock_path: str = SOCK_PATH):
        super().__init__(ignore_metrics=[])
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        self.sock_path = sock_path
        self.timeout = timeout
        self.initialized = False
        self.server_exited = False
        self.try_to_connect()

    def try_to_connect(self, must=False):
        if self.initialized:
            return
        if must:
            retry = 50
        else:
            retry = 1
        backoff = 0.8  # seconds
        for _ in range(retry + 1):
            err_msg = ""
            try:
                self.sock.connect(self.sock_path)
                self.initialized = True
                break
            except OSError as e:
                if e.errno == 106 and e.strerror == "Transport endpoint is already connected":
                    # might be called in multiple threads
                    # but for one process, only one connection is required
                    self.initialized = True
                    break
                else:
                    err_msg = traceback.format_exc()
                    time.sleep(backoff)
            except Exception:
                err_msg = traceback.format_exc()
                time.sleep(backoff)

        if must and not self.initialized:
            NDTimelineLogger().error(f"initialize sock handler failed: {err_msg}")

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
        self.try_to_connect(True)
        if self.server_exited:
            return
        try:
            st = time.perf_counter()
            pkg = serialize_to_package(
                {
                    "metric_name": metric_name,
                    "elapsed": elapsed,
                    "recent_elapsed_raw_parts": recent_elapsed_raw_parts,
                    "recent_since_start_raw_parts": recent_since_start_raw_parts,
                    "tags": tags,
                    "step_range": step_range,
                    "world_info": world_info,
                    "extra": extra,
                }
            )
            self.sock.sendall(pkg)
            NDTimelineLogger().debug(f"serialize and send data: {(time.perf_counter() - st) * 1000:3.3f}ms")
        except BrokenPipeError as e:
            NDTimelineLogger().error(f"{e}, server exit")
            self.server_exited = True
        except socket.timeout:
            NDTimelineLogger().warning(f"socket timeout {traceback.format_exc()}")
        except Exception:
            NDTimelineLogger().error(traceback.format_exc())
