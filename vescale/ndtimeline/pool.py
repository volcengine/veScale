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

import threading
from collections import deque
from typing import Dict, Any, Optional

import torch
from torch.cuda import Event

from .variables import DEFAULT_CUDA_EVENT_POOL_SIZE


class CudaEventPool:
    def __init__(
        self, device: Optional[int] = None, init_sz: int = DEFAULT_CUDA_EVENT_POOL_SIZE, blocking: bool = False
    ) -> None:
        self._pool = deque()
        self._device = device
        self._event_attr = {"enable_timing": True, "blocking": blocking, "interprocess": False}

        self._mtx = threading.Lock()

        for _ in range(init_sz):
            event = Event(**self._event_attr)
            event.tag = {}
            self._pool.append(event)
            event.record()  # warmup

    def get(self, tag: Dict[str, Any]):
        device = torch.cuda.current_device()
        if self._device is not None:
            device = self._device
        with torch.cuda.device(device):
            try:
                with self._mtx:
                    event = self._pool.popleft()
            except IndexError:
                event = Event(**self._event_attr)
        event.tag = tag.copy()
        return event

    def release(self, event: Event):
        with self._mtx:
            self._pool.append(event)


class DefaultEventPool:
    initialized = False

    @classmethod
    def init(cls, device: Optional[int] = None):
        assert not cls.initialized
        cls._default_cuda_event_pool = CudaEventPool(device=device, blocking=True)
        cls.initialized = True

    @classmethod
    def get(cls, tag: Optional[Dict[str, Any]] = None):
        tag = tag if tag is not None else {}
        return cls._default_cuda_event_pool.get(tag)

    @classmethod
    def release(cls, event: Event):
        cls._default_cuda_event_pool.release(event)
