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
import itertools
from dataclasses import dataclass
from typing import List, Dict, Any

from ..logger import NDTimelineLogger
from .handler_base import NDHandler
from ..exceptions import NDHandlerError
from ..world_info import WorldInfo
from ..variables import NDTIMELINE_INNER_GLOBAL_STEP_KEY


@dataclass
class DeviceTimerStreamRecord:
    ts: int  # record time for partition purpose
    rank: int
    metric: str
    iteration: int  # legacy field, no meaning
    step: int
    avg_dur: float  # time elapsed, legacy name
    start_ts: List[float]
    duration: List[float]
    model_chunk: int  # vpp model chunk id, start from 0
    pp_rank: int  # pp_rank legacy problem
    dp_rank: int  # the rank of existing dp group
    tp_rank: int  # the rank of existing tp group
    ip: str
    role_id: int  # multi-role in RL
    trial_id: int  # trial id
    run_id: int  # run_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts,
            "rank": self.rank,
            "metric": self.metric,
            "iteration": self.iteration,
            "step": self.step,
            "value": self.avg_dur,
            "start_ts": self.start_ts,
            "duration": self.duration,
            "model_chunk": self.model_chunk,
            "stage": self.pp_rank,
            "dp_rank": self.dp_rank,
            "tp_rank": self.tp_rank,
            "ip": self.ip,
            "role": self.role_id,
            "trial": str(self.trial_id),
            "run_id": self.run_id,
        }

    def copy(self):
        return DeviceTimerStreamRecord(
            self.ts,
            self.rank,
            self.metric,
            self.iteration,
            self.step,
            self.avg_dur,
            self.start_ts,
            self.duration,
            self.model_chunk,
            self.pp_rank,
            self.dp_rank,
            self.tp_rank,
            self.ip,
            self.role_id,
            self.trial_id,
            self.run_id,
        )


def parse_record(
    metric_name: str,
    elapsed: float,
    recent_elapsed_raw_parts: List[float],
    recent_since_start_raw_parts: List[float],
    tags: List[Dict[str, Any]],
    step_range: range,
    world_info: WorldInfo,
    extra: Dict[str, Any],
) -> List[DeviceTimerStreamRecord]:
    if len(recent_elapsed_raw_parts) != len(recent_since_start_raw_parts):
        raise NDHandlerError(
            f"recent_elapsed_raw_parts {len(recent_elapsed_raw_parts)} not"
            f"equal to recent_since_start_raw_parts {len(recent_since_start_raw_parts)}"
        )
    if len(recent_elapsed_raw_parts) != len(tags):
        raise NDHandlerError(f"recent_elapsed_raw_parts {len(recent_elapsed_raw_parts)} not equal to tags {len(tags)}")

    if len(recent_elapsed_raw_parts) == 0:
        return []

    specified_steps = [tag[NDTIMELINE_INNER_GLOBAL_STEP_KEY] for tag in tags if NDTIMELINE_INNER_GLOBAL_STEP_KEY in tag]

    now = int(time.time())
    records = []
    if len(specified_steps) != 0:
        # metric with `INNER_GLOBAL_STEP_KEY` does not respect `step_range`
        # but it should always be set with `INNER_GLOBAL_STEP_KEY` and monotonically increasing
        if len(specified_steps) != len(tags):
            raise NDHandlerError("timer with INNER_GLOBAL_STEP_KEY's step is not always set")

        # to understand the following codes,
        # you can `print(list(itertools.groupby([21,22,23,23,23,46,46,49,50])))`
        i = 0
        # NDTimelineLogger().debug("{}: {}".format(metric_name, len(tags)))
        for step, group_v in itertools.groupby(specified_steps):
            op_counts = sum(1 for _ in group_v)  # memory efficient version of `len(list(group_v))`
            avg_dur = sum(recent_elapsed_raw_parts[i : i + op_counts]) / op_counts if op_counts != 0 else 0.0
            record = DeviceTimerStreamRecord(
                ts=now,
                rank=world_info.topo_info.rank,
                metric=metric_name,
                iteration=0,
                step=step,
                avg_dur=avg_dur,
                start_ts=recent_since_start_raw_parts[i : i + op_counts],
                duration=recent_elapsed_raw_parts[i : i + op_counts],
                model_chunk=0,
                pp_rank=world_info.topo_info.pp_rank,
                dp_rank=world_info.topo_info.dp_rank,
                tp_rank=world_info.topo_info.tp_rank,
                ip=world_info.topo_info.ip,
                role_id=world_info["role_id"],
                trial_id=world_info["trial_id"],
                run_id=world_info["run_id"],
            )
            records.append(record)
            i += op_counts
    else:
        if len(step_range) == 0:
            raise NDHandlerError(f"step_range {step_range} length is zero")
        if len(recent_elapsed_raw_parts) % len(step_range) != 0:
            fmt_str = (
                "len(recent_elapsed_raw_parts) {} of {} "
                + "is not multiply of len(step_range) {}; "
                + "if you can't ensure op counts in every step are equal,"
                + "please explicitly use `step_getter`"
            )
            raise NDHandlerError(fmt_str.format(metric_name, len(recent_elapsed_raw_parts), len(step_range)))
        NDTimelineLogger().debug(f"{metric_name}: {len(recent_elapsed_raw_parts)} in {step_range}")
        num_step_ops = len(recent_elapsed_raw_parts) // len(step_range)
        for i, step in enumerate(step_range):
            avg_dur = sum(recent_elapsed_raw_parts[i * num_step_ops : (i + 1) * num_step_ops]) / num_step_ops
            record = DeviceTimerStreamRecord(
                ts=now,
                rank=world_info.topo_info.rank,
                metric=metric_name,
                iteration=0,
                step=step,
                avg_dur=avg_dur,
                start_ts=recent_since_start_raw_parts[i * num_step_ops : (i + 1) * num_step_ops],
                duration=recent_elapsed_raw_parts[i * num_step_ops : (i + 1) * num_step_ops],
                model_chunk=0,
                pp_rank=world_info.topo_info.pp_rank,
                dp_rank=world_info.topo_info.dp_rank,
                tp_rank=world_info.topo_info.tp_rank,
                ip=world_info.topo_info.ip,
                role_id=world_info["role_id"],
                trial_id=world_info["trial_id"],
                run_id=world_info["run_id"],
            )
            records.append(record)
    return records


class ParserNDHandler(NDHandler):
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
        return parse_record(
            metric_name,
            elapsed,
            recent_elapsed_raw_parts,
            recent_since_start_raw_parts,
            tags,
            step_range,
            world_info,
            extra,
        )
