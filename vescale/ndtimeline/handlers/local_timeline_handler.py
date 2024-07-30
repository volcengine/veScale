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

import json
from typing import List, Dict, Any, Set, Deque, Optional
from collections import deque

import torch

from .chrome_trace_event import CompleteEvent, ThreadMetadataEvent, ProcessMetadataEvent
from ..world_info import WorldInfo
from ..variables import NDTIMELINE_FLUSH_SEPCIAL
from .handler_base import NDHandler
from .parser_handler import parse_record, DeviceTimerStreamRecord


# thread_index_table


def build_thread_index_table(tab, metrics, index, index_name):
    for m in metrics:
        tab[m] = (index, index_name)


major_metrics = {
    "forward-compute",
    "backward-compute",
    "embedding-grads-all-reduce",
    "optimizer",
    "optimizer-clip-main-grad",
    "optimizer-inner-step",
    "optimizer-copy-to-main-grad",
    "optimizer-copy-main-to-model-params",
}

tp_stream_metrics = {
    "tp-allreduce",
    "tp-allgather",
    "tp-reducescatter",
    "layernorm-grads-all-reduce",
}

dp_stream_metrics = {
    "grads-reduce-scatter",
    "params-all-gather",
    "separate-grads-all-reduce",
    "grads-reduce-scatter-nonoverlapping",
    "params-all-gather-nonoverlapping",
}

pp_batch_stream_metrics = {
    "backward-send-backward-recv",
    "backward-send-forward-recv",
    "forward-send-backward-recv",
    "forward-send-forward-recv",
    "forward-backward-send-forward-backward-recv",
    "cross-mesh-recv",
    "cross-mesh-send",
}

pp_forward_stream_metrics = {
    "forward-recv",
    "backward-send",
}

pp_backward_stream_metrics = {
    "forward-send",
    "backward-recv",
}


thread_sort_index = {}
build_thread_index_table(thread_sort_index, major_metrics, 0, "main")
build_thread_index_table(thread_sort_index, pp_forward_stream_metrics, 1, "pp ->")
build_thread_index_table(thread_sort_index, pp_backward_stream_metrics, 2, "pp <-")
build_thread_index_table(thread_sort_index, pp_batch_stream_metrics, 3, "pp send/recv")
build_thread_index_table(thread_sort_index, tp_stream_metrics, 4, "tp collective")
build_thread_index_table(thread_sort_index, dp_stream_metrics, 5, "dp collective")
sort_index_other = 6
index_name_other = "other"


events = []
tid_names = {}  # tid -> (pid, name)

MAX_UINT64 = 18446744073709551615
NEGTIVE_ONE = -1


class LocalTimelineNDHandler(NDHandler):
    def __init__(self, n_rank_per_host: Optional[int] = None):
        super().__init__(ignore_metrics=[])
        if n_rank_per_host is None:
            n_rank_per_host = torch.cuda.device_count()
        self.n_rank_per_host = n_rank_per_host
        self.rank2buffer: List[List[DeviceTimerStreamRecord]] = [[] for _ in range(n_rank_per_host)]
        # rank -> deque(set(steps), set(steps), empty set)
        self.rank2steps: List[Deque[Set[int]]] = [deque(set() for _ in range(1)) for _ in range(n_rank_per_host)]

    def dump_records(self):
        output_ranks = set()
        events = []
        min_step, max_step = MAX_UINT64, NEGTIVE_ONE
        buffer = [record for rank in range(self.n_rank_per_host) for record in self.rank2buffer[rank]]
        for record in buffer:
            metric, step, rank, dp_rank = record.metric, record.step, record.rank, record.dp_rank
            if step < 0:
                continue
            min_step = min(min_step, step)
            max_step = max(max_step, step)
            output_ranks.add((dp_rank, rank))
            sort_index, index_name = thread_sort_index.get(metric, (sort_index_other, index_name_other))
            tid = rank * 10 + sort_index  # 乘以10表示让出个位数给thread_sort_index编码
            tid_names[tid] = (dp_rank, f"rank[{rank}] {index_name}")
            for ts, dur in zip(record.start_ts, record.duration):
                args = {
                    "rank": rank,
                    "step": step,
                    "tp": record.tp_rank,
                    "pp": record.pp_rank,
                }
                ev = CompleteEvent(name=metric, cat=metric, pid=dp_rank, tid=tid, ts=ts * 1e6, dur=dur * 1e6, args=args)
                events.append(ev)
        for tid, (dp_rank, name) in tid_names.items():
            ev = ThreadMetadataEvent(
                pid=dp_rank,
                tid=tid,
                sort_index=tid,
                thread_name=name,
            )
            events.append(ev)
        for dp_rank in {dp_rank for dp_rank, _ in output_ranks}:
            ev = ProcessMetadataEvent(pid=dp_rank, sort_index=dp_rank, process_name=f"dp rank[{dp_rank}]")
            events.append(ev)
        spans = []
        for ev in events:
            spans.extend(ev.to_objects())
        with open(f"trace_step{min_step}_{max_step}", "w") as f:
            json.dump(spans, f)

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
        local_rank = world_info["local_rank"]
        if metric_name == NDTIMELINE_FLUSH_SEPCIAL:
            self.rank2steps[local_rank].append(set())
            if all(len(self.rank2steps[i]) >= 2 for i in range(self.n_rank_per_host)):
                # split
                new_rank2buffer: List[List[DeviceTimerStreamRecord]] = [[] for _ in range(self.n_rank_per_host)]
                for rank in range(self.n_rank_per_host):
                    # use record.copy to avoid gc failure and memory leaking
                    new_rank2buffer[rank] = [
                        record.copy()
                        for record in self.rank2buffer[rank]
                        if record.step not in self.rank2steps[rank][0]
                    ]
                    self.rank2buffer[rank] = [
                        record for record in self.rank2buffer[rank] if record.step in self.rank2steps[rank][0]
                    ]
                self.dump_records()
                # update
                self.rank2buffer = new_rank2buffer
                for rank in range(self.n_rank_per_host):
                    self.rank2steps[rank].popleft()
        else:
            # assume local_rank is in [0...n_rank_per_device-1]
            records = parse_record(
                metric_name,
                elapsed,
                recent_elapsed_raw_parts,
                recent_since_start_raw_parts,
                tags,
                step_range,
                world_info,
                extra,
            )
            self.rank2buffer[local_rank].extend(records)
            for record in records:
                self.rank2steps[local_rank][-1].add(record.step)
