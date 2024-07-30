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

import socket
import socketserver
import os
import traceback
import queue
import threading
from typing import Callable, List, Optional

import torch.multiprocessing as mp
from torch.multiprocessing import ProcessContext

from .logger import NDTimelineLogger
from .binary_protocol import recv_and_validate, loads_fn
from .exceptions import ProtocolValidationError, NDHandlerError
from .variables import SOCK_PATH, SOCK_PARENT_DIR

q = None


def internal_queue_consume(handlers: Optional[List[Callable]] = None):
    if handlers is None:
        handlers = []
    global q
    while True:
        try:
            args = q.get(block=True)
            for handler in handlers:
                handler(
                    args["metric_name"],
                    args["elapsed"],
                    args["recent_elapsed_raw_parts"],
                    args["recent_since_start_raw_parts"],
                    args["tags"],
                    args["step_range"],
                    args["world_info"],
                    args["extra"],
                )
        except NDHandlerError as e:
            NDTimelineLogger().error(e)
            NDTimelineLogger().warning(traceback.format_exc())
            continue
        except queue.Empty:
            continue
        except Exception as e:
            NDTimelineLogger().error(e)
            NDTimelineLogger().error(traceback.format_exc())
            continue


class MsgHandler(socketserver.BaseRequestHandler):
    def handle(self):
        global q
        # self.request is a socket, automatically closed after `handle`
        assert q is not None
        preload_data = bytearray()
        while True:
            try:
                payload = recv_and_validate(self.request.recv, preload_data)
                args = loads_fn(payload)
                q.put(args)
            except ProtocolValidationError:
                pass
            except ValueError as e:
                NDTimelineLogger().error(e)
                NDTimelineLogger().error(traceback.format_exc())
            except socket.timeout:
                NDTimelineLogger().error("socket.timeout")
                NDTimelineLogger().error(traceback.format_exc())
            except BrokenPipeError:
                NDTimelineLogger().info("client exit")
                break
            except Exception:
                NDTimelineLogger().error(traceback.format_exc())
                break


class NDtimelineStreamer:
    p: ProcessContext
    initialized: bool = False

    @classmethod
    def init(cls, local_rank: int, handlers: Optional[List[Callable]] = None):
        if local_rank != 0:
            return
        if cls.initialized:
            NDTimelineLogger().warning("NDtimelineStreamer has already been initialized, skipped")
            return
        handlers = handlers if handlers is not None else []
        try:
            if os.path.exists(SOCK_PATH):
                os.remove(SOCK_PATH)
            if not os.path.exists(SOCK_PARENT_DIR):
                os.makedirs(SOCK_PARENT_DIR, exist_ok=True)
            cls.p = mp.spawn(
                fn=NDtimelineStreamer.run, args=(handlers,), nprocs=1, join=False, daemon=True, start_method="spawn"
            )
            NDTimelineLogger().info("ndtimeline streamer started")
            cls.initialized = True
        except Exception:
            NDTimelineLogger().error("NDtimelineStreamer init failed")
            NDTimelineLogger().error(traceback.format_exc())

    @staticmethod
    def run(process_index, handlers: List[Callable]):
        global q
        # in order to save memory of main process, `q` is initialized here
        q = queue.Queue(500000)
        mq_thread = threading.Thread(
            target=internal_queue_consume, args=(handlers,), daemon=True, name="internal_queue_consume"
        )
        mq_thread.start()

        with socketserver.ThreadingUnixStreamServer(SOCK_PATH, MsgHandler) as server:
            server.daemon_threads = True
            server.serve_forever()
