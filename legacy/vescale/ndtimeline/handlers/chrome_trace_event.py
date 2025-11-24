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

import random
from dataclasses import dataclass
from typing import Union, Optional, List, Dict, Tuple
from abc import ABC, abstractmethod


class TracingEvent(ABC):
    """
    chrome trace event format see doc:
      https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview#
    """

    @abstractmethod
    def to_objects(self) -> List[dict]:
        pass


@dataclass
class CompleteEvent(TracingEvent):
    name: str
    cat: str
    pid: Union[str, int]
    tid: Union[str, int]

    # 起始和持续时间长度（单位都是us）
    ts: float
    dur: float

    args: Optional[dict] = None

    def to_objects(self) -> List[dict]:
        return [
            {
                "name": self.name,
                "cat": self.cat,
                "pid": self.pid,
                "tid": self.tid,
                "args": self.args or {},
                "ts": self.ts,
                "dur": self.dur,
                "ph": "X",
            }
        ]


@dataclass
class BeginEvent(TracingEvent):
    name: str
    cat: str
    pid: Union[str, int]
    tid: Union[str, int]

    # 起始和持续时间长度（单位都是us）
    ts: float
    stack: Optional[List[int]] = None

    args: Optional[dict] = None

    def to_objects(self) -> List[dict]:
        return [
            {
                "name": self.name,
                "cat": self.cat,
                "pid": self.pid,
                "tid": self.tid,
                "args": self.args or {},
                "ts": self.ts,
                "ph": "B",
            }
        ]


@dataclass
class EndEvent(TracingEvent):
    name: str
    cat: str
    pid: Union[str, int]
    tid: Union[str, int]

    # 起始和持续时间长度（单位都是us）
    ts: float
    stack: Optional[List[int]] = None

    args: Optional[dict] = None

    def to_objects(self) -> List[dict]:
        return [
            {
                "name": self.name,
                "cat": self.cat,
                "pid": self.pid,
                "tid": self.tid,
                "args": self.args or {},
                "ts": self.ts,
                "ph": "E",
            }
        ]


flow_event_id_counter = 0


@dataclass
class FlowEvent(TracingEvent):
    # {"ph": "f", "id": 246, "pid": "172.20.133.93", "tid": 13, "ts": 1669171992173028, \
    # "cat": "async_gpu", "name": "cudaLaunchKernel", "bp": "e"}
    name: str
    cat: str

    # list of (pid, tid, ts)
    flows: List[Tuple[Union[str, int], Union[str, int], float]]

    def to_objects(self) -> List[dict]:
        global flow_event_id_counter
        flow_event_id_counter += 1
        gen_id = flow_event_id_counter  # use stable predictable id
        ret = []
        # 起始时间比结束时间更晚的话没意义，不会被渲染，所以修正一下
        for i in range(1, len(self.flows)):
            _, _, ts0 = self.flows[i - 1]
            pid, tid, ts1 = self.flows[i]
            if ts1 <= ts0:
                self.flows[i] = (pid, tid, ts0 + 1)
        for f in self.flows:
            pid, tid, ts = f
            ret.append(
                {
                    "name": self.name,
                    "cat": self.cat,
                    "pid": pid,
                    "tid": tid,
                    "ts": ts,
                    "ph": "t",
                    "bp": "e",
                    "id": gen_id,
                }
            )
        ret[0]["ph"] = "s"
        ret[-1]["ph"] = "f"
        ret[-1]["ts"] += 1
        return ret


@dataclass
class CounterEvent(TracingEvent):
    name: str
    pid: Union[str, int]

    # 起始和持续时间长度（单位都是us）
    ts: float

    # 计数的数据序列
    data: Dict[str, Union[int, float]]

    def to_objects(self) -> List[dict]:
        return [
            {
                "name": self.name,
                "pid": self.pid,
                "args": self.data,
                "ts": self.ts,
                "ph": "C",
            }
        ]


class CombinedEvents(TracingEvent):
    """
    将多个tracing event合并一起，表示成1个event，最后按顺序展开每个object
    """

    def __init__(self, events: List[TracingEvent]):
        self.events = events

    def to_objects(self) -> List[dict]:
        obj = []
        for e in self.events:
            obj.extend(e.to_objects())
        return obj


@dataclass
class ProcessMetadataEvent(TracingEvent):
    pid: Union[str, int]
    sort_index: int
    process_name: Optional[str] = None
    process_labels: List[str] = None

    def to_objects(self) -> List[dict]:
        ret = [
            {
                "name": "process_sort_index",
                "pid": self.pid,
                "ph": "M",
                "args": {
                    "sort_index": self.sort_index,
                },
            }
        ]
        if self.process_labels is not None:
            ret.append(
                {
                    "name": "process_labels",
                    "pid": self.pid,
                    "ph": "M",
                    "args": {
                        "labels": ",".join(self.process_labels),
                    },
                }
            )
        if self.process_name is not None:
            ret.append(
                {
                    "name": "process_name",
                    "pid": self.pid,
                    "ph": "M",
                    "args": {
                        "name": self.process_name,
                    },
                }
            )
        return ret


@dataclass
class ThreadMetadataEvent(TracingEvent):
    pid: Union[str, int]
    tid: Union[str, int]
    sort_index: int
    thread_name: Optional[str] = None

    def to_objects(self) -> List[dict]:
        ret = [
            {
                "name": "thread_sort_index",
                "pid": self.pid,
                "tid": self.tid,
                "ph": "M",
                "args": {
                    "sort_index": self.sort_index,
                },
            }
        ]
        if self.thread_name is not None:
            ret.append(
                {
                    "name": "thread_name",
                    "pid": self.pid,
                    "tid": self.tid,
                    "ph": "M",
                    "args": {
                        "name": self.thread_name,
                    },
                }
            )
        return ret


class DummyEvent(TracingEvent):
    def to_objects(self) -> List[dict]:
        return [
            {
                "name": "dummy",
                "cat": "dummy",
                "pid": random.randint(1, 100),
                "tid": random.randint(1, 100),
                "args": {
                    "content": "*" * random.randint(100, 1000),
                },
                "ts": random.randint(1, 9999),
                "dur": random.randint(1, 100),
                "ph": "i",
            }
        ]
