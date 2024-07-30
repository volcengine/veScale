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

import dataclasses
import time
import traceback
import gc
import contextlib
from decimal import Decimal
from enum import Enum, unique
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from typing import List, Dict, Any, Callable, Optional, Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from functools import wraps

import torch

from .pool import DefaultEventPool, Event
from .world_info import WorldInfo
from .stream import get_nccl_p2p_stream, get_nccl_coll_stream
from .logger import NDTimelineLogger
from .variables import (
    NDTIMELINE_INNER_GLOBAL_STEP_KEY,
    NDTIMELINE_STREAM_KEY,
    NDTIMELINE_FLUSH_SEPCIAL,
)


class GlobalReferenceTime:
    local_rank: int = 0
    world_size: int = 0
    device: torch.device = None
    # global ref events
    ref_events: List[torch.cuda.Event] = []
    ref_pointer: int = 0
    clock_diff: float = 0.0  # ms
    initial_min_clock: int = 0  # ns
    last_calibrated_at: float = 0.0  # ms
    gpu_clock_residual_coef: float = 1.0
    initialized: bool = False

    @classmethod
    def init(cls, world_sz: int, device: Optional[Union[int, torch.device]] = None):
        if isinstance(device, int):
            cls.device = torch.device(f"cuda:{device}")
            cls.local_rank = device
        elif isinstance(device, torch.device):
            cls.device = device
            cls.local_rank = device.index
        elif device is None:
            cls.device = torch.device(f"cuda:{torch.cuda.current_device()}")
            cls.local_rank = torch.cuda.current_device()
        else:
            raise RuntimeError(f"device must be int or torch.device or None, but got {type(device)}")
        cls.world_size = world_sz
        assert isinstance(cls.device, torch.device)
        with torch.cuda.device(cls.device.index):
            cls.ref_events = [
                torch.cuda.Event(enable_timing=True, blocking=False, interprocess=False) for _ in range(2)
            ]
            # warmup
            for e in cls.ref_events:
                e.record(stream=torch.cuda.default_stream())
            for e in cls.ref_events:
                e.synchronize()
        cls.calibrate()
        if cls.local_rank == 0:
            NDTimelineLogger().debug(f"cls.initial_min_clock: {cls.initial_min_clock}ns")
        cls.initialized = True

    @classmethod
    def sync_events(cls):
        for i in range(len(cls.ref_events)):
            cls.ref_events[i].synchronize()

    @classmethod
    def calibrate(cls):
        # round-robin
        calibrate_st = time.perf_counter()
        next_pointer = (cls.ref_pointer + 1) % len(cls.ref_events)
        cls.ref_pointer = next_pointer
        ref = cls.ref_events[next_pointer]
        with torch.cuda.device(cls.device.index):
            if not cls.initialized:
                torch.distributed.barrier()
            torch.cuda.synchronize()
            # torch.cuda.default_stream().synchronize()
            ref.record(stream=torch.cuda.default_stream())
            ref.synchronize()
            ts_ns = int(time.time_ns())
        ts = ts_ns / 1e6

        if not cls.initialized:
            my_clock = torch.tensor([ts_ns], dtype=torch.long, device=cls.device)
            world_clocks = [torch.zeros([1], dtype=torch.long, device=cls.device) for _ in range(cls.world_size)]
            torch.distributed.all_gather(world_clocks, my_clock)
            all_clocks = [r.cpu().tolist()[0] for r in world_clocks]
            min_clock = min(all_clocks)
            cls.initial_min_clock = min_clock

        cls.clock_diff = (ts_ns - cls.initial_min_clock) / 1e6  # to unit ms

        # cpu-gpu calibrate
        cpu_time = ts - cls.last_calibrated_at  # ms
        gpu_time = 0.0
        cls.last_calibrated_at = ts  # ms
        if cls.initialized and 2 * 1e3 < cpu_time < 200000 * 1e3:
            gpu_time = abs(cls.ref_events[0].elapsed_time(cls.ref_events[1]))  # ms
            gpu_cpu_diff = Decimal((gpu_time) - (cpu_time)) / Decimal(gpu_time)
            cls.gpu_clock_residual_coef = float(1 - gpu_cpu_diff)
        if cls.local_rank == 0:
            NDTimelineLogger().info(
                f"local rank 0, calibrate sync cpu moment: {ts_ns} ns, clock diff: {cls.clock_diff} ms, "
                f"initial min: {cls.initial_min_clock} ns, "
                f"gpu clock redidual coef: {cls.gpu_clock_residual_coef}, "
                f"calibrate cpu: {cpu_time}ms, calibrate gpu: {gpu_time}ms"
            )
        NDTimelineLogger().info(
            f"rank {cls.local_rank} calibrate cost {1000 * (time.perf_counter() - calibrate_st):4.2f}ms"
        )

    @classmethod
    def elapsed_time(cls, end_event):
        # cuda event elapsed_time return in unit ms
        gpu_time = cls.ref_events[cls.ref_pointer].elapsed_time(end_event)
        return gpu_time * cls.gpu_clock_residual_coef + cls.last_calibrated_at

    @classmethod
    def since_global_start_ts(cls, unix_ts):
        # unix_ts in unit s
        return unix_ts - cls.initial_min_clock / 1e9


@unique
class NDMetricLevel(Enum):
    """
    NDMetricLevel is used to define the level of metric.
    """

    FRAMEWORK_INFO = 2
    USER_INFO = 3
    INFO = 4

    FRAMEWORK_DEBUG = 12
    USER_DEBUG = 13
    DEBUG = 14

    FRAMEWORK_TRACE = 102
    USER_TRACE = 103
    TRACE = 104

    def __lt__(self, other) -> bool:
        return self.value < other.value

    def __le__(self, other) -> bool:
        return self.value <= other.value

    def __gt__(self, other) -> bool:
        return self.value > other.value

    def __ge__(self, other) -> bool:
        return self.value >= other.value

    def __eq__(self, other) -> bool:
        return self.value == other.value

    def __neq__(self, other) -> bool:
        return self.value != other.value


@dataclass(frozen=False)
class DeviceTimerMeta:
    name: str = ""
    is_cpu_op: bool = False
    legal_tags: List[str] = dataclasses.field(default_factory=list)
    step_getter: Optional[Callable] = None
    enabled: bool = True
    level: NDMetricLevel = dataclasses.field(default_factory=lambda: NDMetricLevel.FRAMEWORK_DEBUG)
    device_id: int = -1
    dispatch_mode: Literal["selected", "all"] = "all"
    dst_names: List[str] = dataclasses.field(default_factory=list)
    specified_extra: Dict[str, Any] = dataclasses.field(default_factory=dict)
    common_extra: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if self.dispatch_mode not in ["selected", "all"]:
            raise ValueError(f"invalid dispatch_mode {self.dispatch_mode}")
        if not isinstance(self.level, NDMetricLevel):
            raise ValueError(f"invalid type of level {type(self.level)}")

    def copy(self):
        return DeviceTimerMeta(
            self.name,
            self.is_cpu_op,
            self.legal_tags.copy(),
            self.step_getter,
            self.enabled,
            self.level,
            self.device_id,
            self.dispatch_mode,
            self.dst_names.copy(),
            self.specified_extra.copy(),
            self.common_extra.copy(),
        )


class DeviceTimer:
    def __init__(
        self,
        name: str,
        is_cpu_op: bool = False,
        legal_tags: Optional[List[str]] = None,
        step_getter: Optional[Callable] = None,
        enabled: bool = True,
        level: NDMetricLevel = NDMetricLevel.FRAMEWORK_DEBUG,
        device_id: int = 0,
        dispatch_mode: Literal["selected", "all"] = "all",
        dst_names: Optional[List[str]] = None,
        specified_extra: Optional[Dict[str, Any]] = None,
        common_extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        legal_tags = legal_tags if legal_tags is not None else []
        dst_names = dst_names if dst_names is not None else []
        specified_extra = specified_extra if specified_extra is not None else {}
        common_extra = common_extra if common_extra is not None else {}

        if dispatch_mode not in ["all", "selected"]:
            raise ValueError(f"invaid dispatch_mode {dispatch_mode} {type(dispatch_mode)}")
        self.meta = DeviceTimerMeta(
            name,
            is_cpu_op,
            legal_tags,
            step_getter,
            enabled,
            level,
            device_id,
            dispatch_mode,
            dst_names,
            specified_extra,
            common_extra,
        )
        for field_name in self.meta.__dict__:
            setattr(self, field_name, getattr(self.meta, field_name))
        if NDTIMELINE_INNER_GLOBAL_STEP_KEY not in self.legal_tags and step_getter is not None:
            legal_tags.append(NDTIMELINE_INNER_GLOBAL_STEP_KEY)
        if NDTIMELINE_STREAM_KEY not in self.legal_tags:
            legal_tags.append(NDTIMELINE_STREAM_KEY)
        self._started: bool = False
        self._stream: torch.cuda.Stream = None
        # list of [start_event, stop_event]
        self._event_pairs: List[List[Event, Event]] = []
        self._pool = DefaultEventPool
        # list of [start_ts, duration, tag]
        self._extra_records: List[List[float, float, Dict[str, Any]]] = []

    def __repr__(self) -> str:
        return f"DeviceTimer with {self.meta.__repr__()}"

    def is_enabled(self) -> bool:
        return self.enabled

    def enable(self):
        self.meta.enabled = True
        self.enabled = True

    def disable(self):
        self.meta.enabled = False
        self.enabled = False

    def insert_record(
        self,
        start_ts: float,
        duration: float,
        tag: Optional[Dict[str, Any]] = None,
        level: NDMetricLevel = NDMetricLevel.FRAMEWORK_DEBUG,
    ):
        if not self.enabled or self.meta.level > level:
            return
        tag = tag if tag is not None else {}
        if self.step_getter is not None:
            tag[NDTIMELINE_INNER_GLOBAL_STEP_KEY] = self.step_getter()
        self._extra_records.append([start_ts, duration, tag])

    def start(
        self,
        stream: torch.cuda.Stream = None,
        tag: Optional[Dict[str, Any]] = None,
        level: NDMetricLevel = NDMetricLevel.FRAMEWORK_DEBUG,
    ) -> None:
        """Start the timer"""
        if not self.enabled or self.meta.level > level:
            return
        assert not self._started, "timer has already been started"
        tag = tag if tag is not None else {}
        if self.step_getter is not None:
            tag[NDTIMELINE_INNER_GLOBAL_STEP_KEY] = self.step_getter()
        if self.is_cpu_op:
            self._extra_records.append([time.time(), None, tag])
            self._started = True
            return
        start_event = self._pool.get(tag=tag)
        stream_args = {}
        if stream is not None:
            self._stream = stream
            self._stream.wait_stream(torch.cuda.default_stream())
            stream_args = {"stream": self._stream}
        start_event.record(**stream_args)
        self._event_pairs.append([start_event, None])
        self._started = True

    def stop(self, tag: Optional[Dict[str, Any]] = None, level: NDMetricLevel = NDMetricLevel.FRAMEWORK_DEBUG) -> None:
        """Stop the timer. May be called in another thread."""
        if not self.enabled or self.meta.level > level:
            return
        assert self._started, "timer is not started"
        tag = tag if tag is not None else {}
        if self.is_cpu_op:
            now = time.time()
            assert self._extra_records[-1][1] is None, "duration is already set"
            self._extra_records[-1][1] = now - self._extra_records[-1][0]
            self._extra_records[-1][2] = {**tag, **self._extra_records[-1][2]}
            self._started = False
            return
        stop_event = self._pool.get(tag=tag)
        stream_args = {}
        if self._stream is not None:
            stream_args = {"stream": self._stream}
        stop_event.record(**stream_args)
        assert self._event_pairs[-1][-1] is None, "stop_event is already set"
        self._event_pairs[-1][-1] = stop_event
        self._started = False

    def reset(self) -> None:
        self._started = False
        self._stream = None
        self._event_pairs = []
        self._extra_records = []

    def elapsed(self, reset=True) -> Tuple[float, List[float], List[float], List[Dict[str, Any]]]:
        """Calculate the elapsed time."""
        if not self.enabled:
            return 0.0, [], [], []
        recent_elapsed_raw_parts = [0.0] * len(self._event_pairs)
        recent_since_start_raw_parts = [0.0] * len(self._event_pairs)
        tags = [{}] * len(self._event_pairs)
        elapsed = 0.0
        with torch.cuda.device(self.device_id):
            for i, (start_event, stop_event) in enumerate(self._event_pairs):
                stop_event.synchronize()
                start_event.synchronize()
                single_elapsed = start_event.elapsed_time(stop_event) / 1e3
                single_since = GlobalReferenceTime.elapsed_time(start_event) / 1e3
                elapsed += single_elapsed
                recent_elapsed_raw_parts[i] = single_elapsed
                recent_since_start_raw_parts[i] = single_since
                tags[i] = {**start_event.tag, **stop_event.tag}
                tags[i] = {k: tags[i][k] for k in tags[i] if k in self.legal_tags}
                self._pool.release(start_event)
                self._pool.release(stop_event)

        if len(self._extra_records) > 0:
            try:
                elapsed += sum([record[1] for record in self._extra_records])
            except TypeError as e:
                NDTimelineLogger().error(
                    f"exception {e} detected in `elapsed` of {self.name}, possible unmatched start stop"
                )
                return 0.0, [], [], []
            self._extra_records.sort(key=lambda x: x[0])
            if len(recent_since_start_raw_parts) == 0:
                recent_since_start_raw_parts = [record[0] for record in self._extra_records]
                recent_elapsed_raw_parts = [record[1] for record in self._extra_records]
                tags = [record[2] for record in self._extra_records]
            else:
                i = 0
                for record in self._extra_records:
                    while i < len(recent_since_start_raw_parts) and recent_since_start_raw_parts[i] < record[0]:
                        i += 1
                    # a.insert(len(a), x) is equivalent to a.append(x).
                    recent_since_start_raw_parts.insert(i, record[0])
                    recent_elapsed_raw_parts.insert(i, record[1])
                    tags.insert(i, record[2])
                    i += 1
        if reset:
            self.reset()
        return elapsed, recent_elapsed_raw_parts, recent_since_start_raw_parts, tags


class NDTimerManager:
    def __init__(
        self,
        world_info: WorldInfo,
        handlers: Optional[List[Callable]] = None,
        max_workers: int = 3,
        device_id: Optional[int] = None,
        init_cuda_dist: bool = True,
        metric_level: NDMetricLevel = NDMetricLevel.TRACE,
        is_nature_step: bool = True,
    ) -> None:
        self._name2timer = {}
        self._name2active_tmp: Dict[str, bool] = {}
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures = []
        self._is_initailized = False
        self.world_info = world_info
        self.handlers = handlers if handlers is not None else []
        self._device_id = device_id
        self.metric_level = metric_level
        self.is_nature_step = is_nature_step
        self._unregistered_timer_start = []
        self._unregistered_timer_stop = []
        self._unregistered_timer_records_insert = []
        self._cur_global_step = 0

        if init_cuda_dist:
            self.init_cuda_dist_associated(device_id=device_id)

    @property
    def global_step(self):
        return self._cur_global_step

    @global_step.setter
    def global_step(self, step: int):
        if not isinstance(step, int):
            raise ValueError(f"step {step} is not int")
        self._cur_global_step = step

    def init_cuda_dist_associated(self, device_id: Optional[int] = None):
        self._device_id = device_id
        if self._device_id is not None:
            DefaultEventPool.init(device=self._device_id)
            GlobalReferenceTime.init(device=self._device_id, world_sz=self.world_info["world_size"])
        else:
            DefaultEventPool.init()
            GlobalReferenceTime.init(world_sz=self.world_info["world_size"])

    def register_timers(self, timer_metas: List[DeviceTimerMeta]) -> None:
        for meta in timer_metas:
            if meta.device_id == -1:
                if not meta.is_cpu_op:
                    meta.device_id = torch.cuda.current_device()
                else:
                    meta.device_id = 0
            if meta.step_getter is None and self.is_nature_step:

                def getter():
                    return self._cur_global_step

                meta.step_getter = getter
        assert not self._is_initailized, "DeviceTimerManager should only be initialized once"
        NDTimerManager._register_timers(timer_metas, self._name2timer)
        self._is_initailized = True

    @staticmethod
    def _register_timers(timer_metas: List[DeviceTimerMeta], d: Dict[str, DeviceTimer]):
        for meta in timer_metas:
            d[meta.name] = DeviceTimer(**meta.__dict__)

    @staticmethod
    def _flush_timers(
        handlers: List[Callable],
        name2timer: Dict[str, DeviceTimer],
        step_range: range,
        world_info: WorldInfo,
        require_calibrate: bool = False,
    ) -> None:
        if require_calibrate:
            GlobalReferenceTime.calibrate()
        for name in name2timer:
            timer = name2timer[name]
            elapsed_result = timer.elapsed()
            for handler in handlers:
                if timer.dispatch_mode == "selected" and handler.dispatch_key not in timer.dst_names:
                    continue
                extra = timer.common_extra
                if handler.dispatch_key in timer.specified_extra:
                    specified_extra = timer.specified_extra[handler.dispatch_key]
                    extra = {**extra, **specified_extra}
                try:
                    handler(name, *elapsed_result, step_range, world_info, extra)
                except Exception as e:
                    NDTimelineLogger().error(f"handler {handler} failed: {e}")
                    NDTimelineLogger().error(traceback.format_exc())
            timer.meta = None  # in case of CudaTimer obj gc failure due to meta obj

        for handler in handlers:
            handler(NDTIMELINE_FLUSH_SEPCIAL, 0.0, [], [], [], range(0, 1), world_info, extra)

    def start_timer(self, name: str, tag: Optional[Dict[str, Any]] = None) -> None:
        assert isinstance(self, NDTimerManager) or issubclass(type(self), NDTimerManager)
        tag = tag if tag is not None else {}
        try:
            if name not in self._unregistered_timer_start:
                stream = None
                if NDTIMELINE_STREAM_KEY in tag:
                    stream = tag[NDTIMELINE_STREAM_KEY]
                    del tag[NDTIMELINE_STREAM_KEY]
                self._name2timer[name].start(stream=stream, tag=tag, level=self.metric_level)
        except KeyError:
            self._unregistered_timer_start.append(name)
            NDTimelineLogger().warning(f"metric {name} is not registered when `start_timer`, skipped")
        except Exception:
            NDTimelineLogger().error(f"trigger exception when `start_timer` metric {name}")
            NDTimelineLogger().error(traceback.format_exc())

    def stop_timer(self, name, tag: Optional[Dict[str, Any]] = None) -> None:
        assert isinstance(self, NDTimerManager) or issubclass(type(self), NDTimerManager)
        tag = tag if tag is not None else {}
        try:
            if name not in self._unregistered_timer_stop:
                if NDTIMELINE_STREAM_KEY in tag:
                    del tag[NDTIMELINE_STREAM_KEY]
                self._name2timer[name].stop(tag=tag, level=self.metric_level)
        except KeyError:
            self._unregistered_timer_stop.append(name)
            NDTimelineLogger().warning(f"metric {name} is not registered when `stop_timer`, skipped")
        except Exception:
            NDTimelineLogger().error(f"trigger exception when `start_timer` metric {name}")
            NDTimelineLogger().error(traceback.format_exc())

    def insert_record(self, name, start_ts: float, duration: float, tag: Optional[Dict[str, Any]] = None):
        assert isinstance(self, NDTimerManager) or issubclass(type(self), NDTimerManager)
        tag = tag if tag is not None else {}
        try:
            if name not in self._unregistered_timer_records_insert:
                self._name2timer[name].insert_record(start_ts, duration, tag, self.metric_level)
        except KeyError:
            self._unregistered_timer_records_insert.append(name)
            NDTimelineLogger().warning(f"metric {name} is not registered when `insert_record`, skipped")
        except Exception:
            NDTimelineLogger().error(f"trigger exception when `insert_record` metric {name}")
            NDTimelineLogger().error(traceback.format_exc())

    def clear(self):
        self.async_flush(
            step_range=range(0, 10),
            next_iter_enabled=False,
            collect_future=False,
            submit2handler=False,
            keep_timer_state=True,
        )

    def disable_and_save(self):
        is_autogc = gc.isenabled()
        if is_autogc:
            gc.disable()
        for k in self._name2timer:
            self._name2active_tmp[k] = self._name2timer[k].is_enabled()
            self._name2timer[k].disable()
        if is_autogc:
            gc.enable()

    def recover_from_history(self):
        is_autogc = gc.isenabled()
        if is_autogc:
            gc.disable()
        for k in self._name2timer:
            if k in self._name2active_tmp:
                if self._name2active_tmp[k]:
                    self._name2timer[k].enable()
                else:
                    self._name2timer[k].disable()
                del self._name2active_tmp[k]
        if is_autogc:
            gc.enable()

    def async_flush(
        self,
        step_range: range,
        next_iter_enabled: bool = True,
        world_info: Optional[WorldInfo] = None,
        handlers: Optional[List[Callable[..., None]]] = None,
        collect_future: bool = True,
        submit2handler: bool = True,
        force_calibrate: bool = False,
        dynamic_calibrate: bool = False,
        keep_timer_state: bool = False,
        sequential_calibrate: bool = True,
    ):
        st = time.perf_counter()
        handlers = handlers if handlers is not None else []
        enabled_timer_names = [name for name in self._name2timer if self._name2timer[name].meta.enabled]
        NDTimelineLogger().debug(f"async flush triggered, {enabled_timer_names}")

        unregistered = self._unregistered_timer_start.copy()
        unregistered.extend(self._unregistered_timer_stop)
        unregistered.extend(self._unregistered_timer_records_insert)
        unregistered = list(set(unregistered))
        if len(unregistered) > 0:
            NDTimelineLogger().warning(f"unregistered timers: {unregistered}")

        past_name2timer = self._name2timer
        fresh_name2timer = {}
        timer_metas = [past_name2timer[name].meta.copy() for name in past_name2timer]

        if not keep_timer_state:
            for meta in timer_metas:
                meta.enabled = next_iter_enabled

        # filter enabled timer
        past_name2timer = {
            name: past_name2timer[name]
            for name in past_name2timer
            if past_name2timer[name].meta.enabled and past_name2timer[name].meta.level <= self.metric_level
        }

        NDTimerManager._register_timers(timer_metas, fresh_name2timer)

        is_autogc = gc.isenabled()
        if is_autogc:
            gc.disable()
        self._name2timer = fresh_name2timer
        if is_autogc:
            gc.enable()

        if collect_future:
            i = 0
            while i < len(self._futures):
                if self._futures[i].done():
                    e = self._futures[i].exception()
                    if e is not None:
                        NDTimelineLogger().error("".join(traceback.format_exception(type(e), e, e.__traceback__)))
                    self._futures.pop(i)
                else:
                    i += 1

        if len(handlers) == 0:
            handlers = self.handlers

        require_calibrate = force_calibrate or (
            dynamic_calibrate and GlobalReferenceTime.last_calibrated_at < (time.time() - 30 * 60) * 1e3
        )
        if require_calibrate and sequential_calibrate:
            GlobalReferenceTime.calibrate()
            require_calibrate = False

        if submit2handler and len(past_name2timer) > 0:
            world_info = self.world_info if world_info is None else self.world_info
            future = self._executor.submit(
                NDTimerManager._flush_timers, handlers, past_name2timer, step_range, world_info, require_calibrate
            )
            self._futures.append(future)

        NDTimelineLogger().debug(f"async flush cost {1000 * (time.perf_counter() - st):4.2f}ms")

    def wait(self) -> None:
        if len(self._futures) == 0:
            return
        torch.distributed.barrier()
        # wait at most 10 seconds
        wait(self._futures, timeout=10, return_when=ALL_COMPLETED)
        for f in self._futures:
            e = f.exception(timeout=0.001)
            if e is not None:
                NDTimelineLogger().error("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        self._futures = []
        # streamer can not respond to training process now
        # assume msg will be handled in 3 seconds
        time.sleep(3)


class Singleton(type):
    _instances = {}

    def __call__(self, *args, **kwargs):
        if self not in self._instances:
            self._instances[self] = super().__call__(*args, **kwargs)
            self._singleton_inited = True
        return self._instances[self]


class NDTimerManagerSingleton(NDTimerManager, metaclass=Singleton):
    @classmethod
    def is_initialized(cls) -> bool:
        return hasattr(cls, "_singleton_inited") and cls._singleton_inited


@contextlib.contextmanager
def ndtimeit(name: str, tag: Optional[Dict[str, Any]] = None):
    """reentrant timeit context manager"""
    if not NDTimerManagerSingleton.is_initialized():
        yield
        return
    tag = tag if tag is not None else {}
    NDTimerManagerSingleton().start_timer(name, tag)
    try:
        yield
    finally:
        NDTimerManagerSingleton().stop_timer(name)


@contextlib.contextmanager
def ndtimeit_p2p(name: str, nccl_pg, peer: int, is_batched: bool = True, tag: Optional[Dict[str, Any]] = None):
    if not NDTimerManagerSingleton.is_initialized():
        yield
        return
    p2p_stream = get_nccl_p2p_stream(name=name, nccl_pg=nccl_pg, peer=peer, is_batched=is_batched)
    if tag is not None:
        tag[NDTIMELINE_STREAM_KEY] = p2p_stream
    else:
        tag = {NDTIMELINE_STREAM_KEY: p2p_stream}
    NDTimerManagerSingleton().start_timer(name, tag)
    try:
        yield
    finally:
        NDTimerManagerSingleton().stop_timer(name)


@contextlib.contextmanager
def ndtimeit_coll(name: str, pg, tensor: torch.Tensor, tag: Optional[Dict[str, Any]] = None):
    if not NDTimerManagerSingleton.is_initialized():
        yield
        return
    coll_stream = get_nccl_coll_stream(name, pg, tensor)
    if tag is not None:
        tag[NDTIMELINE_STREAM_KEY] = coll_stream
    else:
        tag = {NDTIMELINE_STREAM_KEY: coll_stream}
    NDTimerManagerSingleton().start_timer(name, tag)
    try:
        yield
    finally:
        NDTimerManagerSingleton().stop_timer(name)


def ndtimer(metric: str, tags: Optional[Dict[str, Any]] = None):
    def _ndtimeit_decorator(func):
        @wraps(func)
        def with_ndtimeit(*args, **kwargs):
            with ndtimeit(metric, tags):
                return func(*args, **kwargs)

        return with_ndtimeit

    return _ndtimeit_decorator
