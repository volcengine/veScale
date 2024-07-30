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

from typing import Optional, List, Callable, Tuple
import math
from copy import deepcopy

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
import torch.distributed as dist

from .is_internal import is_internal_vescale

if is_internal_vescale():
    from vescale.fsdp import FullyShardedDataParallel, ShardingStrategy
    from vescale.fsdp._init_utils import HYBRID_SHARDING_STRATEGIES
    from ._internal import _get_ip_by_env, _get_role_id, _get_run_id, _get_trial_id
else:
    # make python happy
    class FullyShardedDataParallel:
        pass

    class ShardingStrategy:
        pass

    HYBRID_SHARDING_STRATEGIES = ""


from vescale.dtensor.device_mesh import DeviceMesh
from vescale.devicemesh_api import VESCALE_DEVICE_MESH
from .timer import NDTimerManagerSingleton, DeviceTimerMeta, NDMetricLevel
from .handlers import NDHandler, SockNDHandler, LocalTimelineNDHandler
from .world_info import WorldInfo
from .sock_streamer import NDtimelineStreamer
from .logger import NDTimelineLogger
from .predefined import (
    FORWARD_COMPUTE,
    BACKWARD_COMPUTE,
    CROSS_MESH_RECV,
    CROSS_MESH_SEND,
    RECV_FORWARD,
    RECV_BACKWARD,
    SEND_FORWARD,
    SEND_BACKWARD,
    SEND_FORWARD_RECV_BACKWARD,
    SEND_BACKWARD_RECV_FORWARD,
    UNSHARD_AG,
    GRAD_RS,
    GRAD_AR,
)
from .fsdp_patch import patch_fsdp


def init_ndtimers(
    rank: Optional[int] = None,
    mode: Literal["fsdp", "hybrid"] = "hybrid",
    wrapped_fsdp_module: Optional[FullyShardedDataParallel] = None,
    device_mesh: Optional[DeviceMesh] = None,
    mesh_shape: Optional[Tuple[int, ...]] = None,
    local_rank: Optional[int] = None,
    step_getter: Optional[Callable[[], int]] = None,
    enable_streamer: bool = True,
    n_rank_per_host: Optional[int] = None,
    pre_handlers: Optional[List[NDHandler]] = None,
    post_handlers: Optional[List[NDHandler]] = None,
    user_spcified_timers: Optional[List[DeviceTimerMeta]] = None,
    level: NDMetricLevel = NDMetricLevel.DEBUG,
    ip: str = "0.0.0.0",
    **kwargs,
):
    """
    High level api to enable timers.
    It MUST be called after both torch.cuda.set_device and default process group are initialized.

    Args:
        rank (int): rank id. If rank is None, it will be determined by torch.distributed.get_rank.

        mode (str): `fsdp` or `hybrid` mode, `fsdp` currently is only supported in internal version.

        wrapped_fsdp_module (FullyShardedDataParallel): `FullyShardedDataParallel` wrapped torch.nn.module,
        only used in fsdp mode and only valid in internal version.

        device_mesh (DeviceMesh): only used in fsdp mode and only valid in internal version.

        mesh_shape (Tuple): only used in fsdp mode and only valid in internal version.

        local_rank (int): local rank id. If local_rank is None, it will be determined by VESCALE_DEVICE_MESH.

        step_getter (Callable[[], int]): func to get current global step. If it is None, steps will be always set as 0.
        Another choice is to use `set_global_step` and `inc_step` to maintain step.

        enable_streamer (bool): If set, a streamer process will be forked and then post_handlers can be enabled.

        n_rank_per_host (int): number of devices on one machine. If it is None, it will be determined by torch.cuda.device_count.

        pre_handlers (List[NDHandler]): List of NDHandlers triggered immediately after `flush` on each training process.
        `SockNDHandler` will be automatically injected in pre_handlers when streamer enabled and no pre_handlers are given.

        post_handlers (List[NDHandler]): List of NDHandlers triggered in streamer process.
        `LocalTimelineNDHandler` will be automatically injected when streamer enabled and no post_handlers are given.

        user_spcified_timers (List[DeviceTimerMeta]): List of DeviceTimerMeta registered by user.

        level (NDMetricLevel): metrics of which the level is lower than this will be ignored.

        ip (str): pod/host ip.

    Returns:
        Nothing
    """

    post_handlers = [] if post_handlers is None else post_handlers
    pre_handlers = [] if pre_handlers is None else pre_handlers
    user_spcified_timers = [] if user_spcified_timers is None else user_spcified_timers

    if mode not in ["hybrid", "fsdp"]:
        raise NotImplementedError(f"mode {mode} not implemented")

    if mode == "fsdp" and not is_internal_vescale():
        raise NotImplementedError("fsdp is not currently supported for opensource version")

    if mode != "fsdp" and wrapped_fsdp_module is not None:
        raise ValueError("wrapped_fsdp_module and mode should be set accordingly")

    if NDTimerManagerSingleton.is_initialized():
        NDTimelineLogger().warning("timers initialized, no need for initialization")
        return

    local_rank = VESCALE_DEVICE_MESH.get_local_rank() if local_rank is None else local_rank
    rank = torch.distributed.get_rank() if rank is None else rank
    n_rank_per_host = torch.cuda.device_count() if n_rank_per_host is None else n_rank_per_host

    world_size = dist.get_world_size()
    ddp_rank, ddp_size = 0, 1
    if mode == "hybrid":
        tp_size = VESCALE_DEVICE_MESH.get_strategy_size("TP")
        dp_size = VESCALE_DEVICE_MESH.get_strategy_size("DP")
        pp_size = VESCALE_DEVICE_MESH.get_strategy_size("PP")

        tp_rank = VESCALE_DEVICE_MESH.get_tensor_parallel_rank()
        pp_rank = VESCALE_DEVICE_MESH.get_pipeline_parallel_rank()
        dp_rank = VESCALE_DEVICE_MESH.get_data_parallel_rank()

        assert (
            tp_size * dp_size * pp_size == world_size
        ), f"tp_size: {tp_size}, dp_size: {dp_size}, pp_size: {pp_size}, world_size: {world_size}"
    elif mode == "fsdp":
        tp_size, pp_size = 1, 1
        tp_rank, pp_rank = 0, 0

        patch_fsdp()
        if wrapped_fsdp_module is not None:
            intra_node_group = wrapped_fsdp_module.process_group
            inter_node_group = getattr(wrapped_fsdp_module, "_inter_node_pg", None)
            dp_rank, dp_size, ddp_rank, ddp_size = _calculate_topo(
                intra_node_group,
                inter_node_group,
                wrapped_fsdp_module.sharding_strategy,
                world_size,
            )
        elif device_mesh is not None:
            dp_rank, dp_size, ddp_rank, ddp_size = _calculate_topo_by_shape(tuple(device_mesh.mesh.shape), rank)
        elif mesh_shape is not None:
            dp_rank, dp_size, ddp_rank, ddp_size = _calculate_topo_by_shape(mesh_shape, rank)
        else:
            raise ValueError("for fsdp, device_mesh or wrapped_fsdp_module or mesh_shape must be given at least 1")

    if enable_streamer:
        if local_rank == 0:
            if len(post_handlers) > 0:
                NDtimelineStreamer.init(local_rank, post_handlers)
            else:
                NDtimelineStreamer.init(
                    local_rank,
                    [
                        LocalTimelineNDHandler(n_rank_per_host),
                    ],
                )
        if len(pre_handlers) == 0 or all(not isinstance(handler, SockNDHandler) for handler in pre_handlers):
            pre_handlers.append(SockNDHandler())

    trial_id, run_id, role_id = 0, 0, 0

    if is_internal_vescale():
        if ip == "0.0.0.0":
            ip = _get_ip_by_env()
        trial_id = _get_trial_id()
        run_id = _get_run_id()
        role_id = _get_role_id()

    NDTimerManagerSingleton(
        WorldInfo(
            rank=rank,
            local_rank=local_rank,
            tp_rank=tp_rank,
            pp_rank=pp_rank,
            dp_rank=dp_rank,
            ddp_rank=ddp_rank,
            tp_size=tp_size,
            pp_size=pp_size,
            dp_size=dp_size,
            ddp_size=ddp_size,
            world_size=world_size,
            ip=ip,
            trial_id=trial_id,
            run_id=run_id,
            role_id=role_id,
        ),
        init_cuda_dist=True,
        handlers=pre_handlers,
        metric_level=level,
    )

    extra = {}
    mq_sinks = []
    if is_internal_vescale():
        from ._internal import MQNDHandler

        for handler in post_handlers:
            if isinstance(handler, MQNDHandler):
                mq_sinks.extend(handler.mq_sinks)
        if len(mq_sinks) != 0:
            extra = {"sinks": mq_sinks}

    if mode == "hybrid":
        predefined_timers = [
            DeviceTimerMeta(SEND_BACKWARD, is_cpu_op=False, step_getter=step_getter),
            DeviceTimerMeta(SEND_FORWARD, is_cpu_op=False, step_getter=step_getter),
            DeviceTimerMeta(RECV_FORWARD, is_cpu_op=False, step_getter=step_getter),
            DeviceTimerMeta(RECV_BACKWARD, is_cpu_op=False, step_getter=step_getter),
            DeviceTimerMeta(SEND_FORWARD_RECV_BACKWARD, is_cpu_op=False, step_getter=step_getter),
            DeviceTimerMeta(SEND_BACKWARD_RECV_FORWARD, is_cpu_op=False, step_getter=step_getter),
            DeviceTimerMeta(CROSS_MESH_RECV, is_cpu_op=False, step_getter=step_getter),
            DeviceTimerMeta(CROSS_MESH_SEND, is_cpu_op=False, step_getter=step_getter),
            DeviceTimerMeta(FORWARD_COMPUTE, is_cpu_op=False, step_getter=step_getter),
            DeviceTimerMeta(BACKWARD_COMPUTE, is_cpu_op=False, step_getter=step_getter),
        ]
    else:
        predefined_timers = [
            DeviceTimerMeta(
                UNSHARD_AG,
                is_cpu_op=False,
                step_getter=step_getter,
                common_extra=deepcopy(extra),
            ),
            DeviceTimerMeta(
                GRAD_RS,
                is_cpu_op=False,
                step_getter=step_getter,
                common_extra=deepcopy(extra),
            ),
            DeviceTimerMeta(
                GRAD_AR,
                is_cpu_op=False,
                step_getter=step_getter,
                common_extra=deepcopy(extra),
            ),
            DeviceTimerMeta(
                FORWARD_COMPUTE,
                is_cpu_op=False,
                step_getter=step_getter,
                common_extra=deepcopy(extra),
            ),
            DeviceTimerMeta(
                BACKWARD_COMPUTE,
                is_cpu_op=False,
                step_getter=step_getter,
                common_extra=deepcopy(extra),
            ),
        ]
    predefined_timers.extend(user_spcified_timers)
    NDTimerManagerSingleton().register_timers(predefined_timers)


def wait():
    """
    High level api for timers to exit gracefully
    """
    if NDTimerManagerSingleton.is_initialized():
        NDTimerManagerSingleton().wait()


def set_global_step(global_step: int = 0):
    """
    Another choice to set global step when `global_step_getter` is None
    """
    if NDTimerManagerSingleton.is_initialized():
        NDTimerManagerSingleton().global_step = global_step


def inc_step(step: int = 1):
    """
    Another choice beside `global_step_getter` to increase global step when `global_step_getter` is None
    """
    if NDTimerManagerSingleton.is_initialized():
        step_increased = NDTimerManagerSingleton().global_step + step
        NDTimerManagerSingleton().global_step = step_increased


def flush(
    step_range: Optional[range] = None,
    next_iter_enabled: bool = True,
    submit2handler: bool = True,
    dynamic_calibrate: bool = False,
    keep_timer_state: bool = False,
    sequential_calibrate: bool = True,
):
    """
    High level api for timers to flush metrics to handlers.

    Args:
        step_range (range): global step range. Theorically, NO step_getter is acceptable if user use lower level api.
        Therefore, step_range is used to allocating steps to metrics. If step_getter is given, step_range can be ignored.

        next_iter_enabled (bool): whether timers continue to be enabled after flushed

        submit2handler (bool): whether metrics should be dropped. False means dropping metrics.

        dynamic_calibrate (bool): whether calibrate clocks at least every 20 minutes.

        keep_timer_state (bool): keep timers being enable or disabled state after flushed, if True; next_iter_enabled ignored if True

        sequential_calibrate (bool): calibrate clocks in main thread or other threads

    Returns:
        Nothing

    """
    if NDTimerManagerSingleton.is_initialized():
        step_range = range(0, 1) if step_range is None else step_range
        NDTimerManagerSingleton().async_flush(
            step_range,
            next_iter_enabled=next_iter_enabled,
            submit2handler=submit2handler,
            dynamic_calibrate=dynamic_calibrate,
            keep_timer_state=keep_timer_state,
            sequential_calibrate=sequential_calibrate,
        )


def _calculate_topo(
    intra_node_group: dist.ProcessGroup,
    inter_node_group: dist.ProcessGroup,
    sharding_strategy: ShardingStrategy,
    world_size: int,
) -> Tuple[int, int, int, int]:
    if sharding_strategy in HYBRID_SHARDING_STRATEGIES:
        ddp_size = inter_node_group.size()
        ddp_rank = inter_node_group.rank()
        dp_size = intra_node_group.size()
        dp_rank = intra_node_group.rank()
        assert (
            world_size == intra_node_group.size() * inter_node_group.size()
        ), f"world_size: {world_size} intra_node_group: {dp_size} inter_node_group: {ddp_size}"
        return dp_rank, dp_size, ddp_rank, ddp_size
    elif sharding_strategy == ShardingStrategy.FULL_SHARD:
        dp_size = intra_node_group.size()
        dp_rank = intra_node_group.rank()
        assert world_size == intra_node_group.size(), f"world_size: {world_size}"
        return dp_rank, dp_size, 0, 1
    else:
        raise NotImplementedError("not implemented for ddp")


def _calculate_topo_by_shape(mesh_shape: Tuple[int, ...], rank: int) -> Tuple[int, int, int, int]:
    for m in mesh_shape:
        assert m > 0 and isinstance(m, int)
    if len(mesh_shape) == 2:
        dim0, dim1 = mesh_shape[0], mesh_shape[1]
        ddp_size, dp_size = dim0, dim1
        mesh = torch.arange(math.prod(mesh_shape)).view(mesh_shape)
        ddp_rank, dp_rank = torch.where(mesh == rank)
        ddp_rank, dp_rank = int(ddp_rank), int(dp_rank)
        return dp_rank, dp_size, ddp_rank, ddp_size
    elif len(mesh_shape) == 1:
        return rank, math.prod(mesh_shape), 0, 1
    else:
        raise ValueError(f"invalid mesh_shape {mesh_shape}")
