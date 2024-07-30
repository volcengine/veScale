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

from typing import List, Sequence, Optional, Dict
from collections import deque, defaultdict
from dataclasses import dataclass
from inspect import signature
import contextlib

import torch

from vescale.pipe._schedules.instruction_base import (
    InstructionGenerator,
    StageDeps,
    CommPacket,
    register_instruction,
    Shape,
    registed_functions,
    VESCALE_INTRUCTION_BUILDER as builder,
)
from vescale.pipe.p2p_communication import (
    recv_backward,
    recv_forward,
    send_backward,
    send_forward,
)
from vescale.dtensor._diff import manage_dump_file
from vescale.dtensor.device_mesh import DeviceMesh
from vescale.dtensor.dtensor import DTensor, make_dtensor
from vescale.ndtimeline import ndtimeit_p2p
from vescale.ndtimeline.predefined import CROSS_MESH_RECV, CROSS_MESH_SEND
from torch.distributed._functional_collectives import send, recv
from vescale.dtensor.placement_types import Placement
from vescale.dtensor._utils import compute_global_tensor_info
from torch.distributed.distributed_c10d import _get_default_group
from vescale.model.base_gpt.utils import switch_dtensor

import logging

logger = logging.getLogger(__file__)


def maybe_tensor(tensor):
    if isinstance(tensor, DTensor):
        return tensor._local_tensor
    elif isinstance(tensor, torch.Tensor):
        return tensor
    else:
        raise RuntimeError(f"Error parsing tensor {tensor}")


def cross_mesh_recv(comm, p2p_tensor):
    mapping_group = comm.cur_mesh.get_mapping_rank(comm.peer_mesh)
    if isinstance(mapping_group, int):  # equal size
        default_pg = _get_default_group()
        with ndtimeit_p2p(CROSS_MESH_RECV, default_pg, mapping_group, is_batched=False):
            tensor = torch.empty((3, 3), device=p2p_tensor.device, dtype=torch.int64)
            recv(tensor, mapping_group, default_pg)
            p_size = sum(tensor[:, 0] >= 0)
            tensor = tensor[:p_size]
            sharding_type = [Placement.serialize_from_tensor(p) for p in tensor]
            sharding = sharding_type
            if len(sharding_type) > 0:
                global_shape, global_stride = compute_global_tensor_info(p2p_tensor, comm.cur_mesh, sharding)
                p2p_tensor = make_dtensor(
                    p2p_tensor,
                    comm.cur_mesh,
                    sharding,
                    shape=torch.Size(global_shape),
                    dtype=p2p_tensor.dtype,
                    requires_grad=p2p_tensor.requires_grad,
                    stride=tuple(global_stride),
                )
                return p2p_tensor
            else:
                return p2p_tensor
    else:
        raise NotImplementedError("currently not support change mesh size")


def cross_mesh_send(comm, dt):
    mapping_group = comm.cur_mesh.get_mapping_rank(comm.peer_mesh)
    if isinstance(mapping_group, int):  # equal size
        default_pg = _get_default_group()
        with ndtimeit_p2p(CROSS_MESH_SEND, default_pg, mapping_group, is_batched=False):
            if isinstance(dt, DTensor):
                send_sharding = torch.stack(
                    [p.serialize_to_tensor(dt.device) for p in dt._spec.placements]
                    + [
                        torch.full((3,), -1, device=dt.device, dtype=torch.int64)
                        for _ in range(3 - len(dt._spec.placements))
                    ]
                )
                send(send_sharding, mapping_group, default_pg)
            else:  # tensor
                send(torch.full((3, 3), -1, device=dt.device, dtype=torch.int64), mapping_group, default_pg)
    else:
        raise NotImplementedError("currently not support change mesh size")


def cross_mesh_double(comm, fwd_tensor, p2p_tensor):
    if isinstance(fwd_tensor, DTensor):
        placements = fwd_tensor._spec.placements
        global_shape, global_stride = compute_global_tensor_info(p2p_tensor, comm.cur_mesh, placements)
        p2p_tensor = make_dtensor(
            p2p_tensor,
            comm.cur_mesh,
            placements,
            shape=torch.Size(global_shape),
            dtype=p2p_tensor.dtype,
            requires_grad=p2p_tensor.requires_grad,
            stride=tuple(global_stride),
        )
    return p2p_tensor


@dataclass(eq=True, frozen=True)
class ScheduledNode:
    type: str
    chunk: int
    stage: int
    minibatch: int
    start_time: int
    completion_time: int
    rollback: bool = False

    def get_send_comms(self, total_stages, deps):
        if self.chunk == 0:
            return (
                [
                    CommPacket(
                        cur_mesh=deps.get_current_mesh(self.stage),
                        peer_mesh=deps.get_current_mesh(self.stage + 1),
                        input_id=0,
                        peer_stage=self.stage + 1,
                    )
                ]
                if self.stage != total_stages
                else []
            )
        else:
            return (
                [
                    CommPacket(
                        cur_mesh=deps.get_current_mesh(self.stage),
                        peer_mesh=deps.get_current_mesh(self.stage - 1),
                        input_id=0,
                        peer_stage=self.stage - 1,
                    )
                ]
                if self.stage != 0
                else []
            )

    def get_recv_comms(self, total_stages, deps):
        if self.chunk == 0:
            return (
                [
                    CommPacket(
                        cur_mesh=deps.get_current_mesh(self.stage),
                        peer_mesh=deps.get_current_mesh(self.stage - 1),
                        input_id=0,
                        peer_stage=self.stage - 1,
                    )
                ]
                if self.stage != 0
                else []
            )
        else:
            return (
                [
                    CommPacket(
                        cur_mesh=deps.get_current_mesh(self.stage),
                        peer_mesh=deps.get_current_mesh(self.stage + 1),
                        input_id=0,
                        peer_stage=self.stage + 1,
                    )
                ]
                if self.stage != total_stages
                else []
            )


class CostGraph:
    def __init__(self, n_stage, n_micro, f_cost, b_cost, w_cost, c_cost, f_mem, b_mem, w_mem, max_mem=None):
        self.n_node = 6 * n_stage * n_micro
        self.n_stage = n_stage
        self.n_micro = n_micro
        self.f_cost = f_cost
        self.b_cost = b_cost
        self.w_cost = w_cost
        self.c_cost = c_cost
        self.f_mem = f_mem
        self.b_mem = b_mem
        self.w_mem = w_mem
        self.fbw_cost = [f_cost, b_cost, w_cost]
        self.fbw_mem = [f_mem, b_mem, w_mem]
        self.max_mem = max_mem or f_mem * self.n_stage * 2

    def get_id(self, cat, chunk, stage, micro):
        return (
            cat * 2 * self.n_stage * self.n_micro + chunk * self.n_stage * self.n_micro + stage * self.n_micro + micro
        )

    def try_v_schedule(self, fill_f=True, fill_b=True, approved_bubble=None):
        count = []
        for i in range(self.n_stage):
            count.append([0] * 6)

        end_time = [-1] * self.n_node
        cur_time = [0] * self.n_stage
        mem = [0] * self.n_stage
        stage_bubble = [0] * self.n_stage
        pending_w = [deque() for _ in range(self.n_stage)]
        schedule = [[] for _ in range(self.n_stage)]
        stage_str = ["    " * i for i in range(self.n_stage)]

        if approved_bubble is None:
            approved_bubble = [-1] * self.n_stage
        max_approved_bubble = max(approved_bubble)

        def get_max_stage_bubble(stage=-1):
            max_stage_bubble = 0
            for bb in stage_bubble:
                max_stage_bubble = max(max_stage_bubble, bb)
            if stage >= 0:
                max_stage_bubble = max(max_stage_bubble, max_approved_bubble - approved_bubble[stage])
            return max_stage_bubble

        def put_w(stage):
            assert len(pending_w[stage]) > 0
            _, chunk_, _ = pending_w[stage].popleft()
            put(2, chunk_, stage)

        def put(cat, chunk, stage, assert_cnt=True):
            _tmp = _no_bubble = cur_time[stage] + self.fbw_cost[cat]
            _cnt = count[stage][cat * 2 + chunk]
            if _cnt >= self.n_micro:
                if not assert_cnt:
                    stage_str[stage] += "    "
                    cur_time[stage] = _tmp  # TODO
                    return
                raise AssertionError()
            assert mem[stage] + self.fbw_mem[cat] <= self.max_mem
            stage_str[stage] += "FfBbWw"[cat * 2 + chunk] + str(_cnt + 1) + " " * (3 - len(str(_cnt + 1)))
            if cat > 0 or chunk > 0:
                last_id = cat * 2 + chunk - 1
                if cat < 2:
                    assert end_time[self.get_id(last_id // 2, last_id % 2, stage, _cnt)] >= 0
                else:
                    assert end_time[self.get_id(1, chunk, stage, _cnt)] >= 0
            if chunk == 1 and cat < 2:
                if stage < self.n_stage - 1:
                    _fa_id = self.get_id(cat, chunk, stage + 1, _cnt)
                    assert end_time[_fa_id] >= 0
                    _tmp = max(_tmp, end_time[_fa_id] + self.c_cost + self.fbw_cost[cat])
            if chunk == 0 and cat < 2:
                if stage > 0:
                    _fa_id = self.get_id(cat, chunk, stage - 1, _cnt)
                    assert end_time[_fa_id] >= 0, f"{cat}, {chunk}, {stage}, {_cnt}"
                    _tmp = max(_tmp, end_time[_fa_id] + self.c_cost + self.fbw_cost[cat])
            _id = self.get_id(cat, chunk, stage, _cnt)
            if count[stage][0] > 0:
                stage_bubble[stage] += _tmp - _no_bubble
            end_time[_id] = _tmp
            cur_time[stage] = _tmp
            mem[stage] += self.fbw_mem[cat]
            # noinspection PyTypeChecker
            schedule[stage].append((cat, chunk, _cnt))
            if cat == 1:
                pending_w[stage].append((2, chunk, _cnt))
            count[stage][cat * 2 + chunk] += 1

        for i in range(self.n_stage):
            put(0, 0, i)
        for i in range(self.n_stage - 1, -1, -1):
            if i == self.n_stage - 1:
                put(0, 1, i)
                continue
            tmp = end_time[self.get_id(0, 1, i + 1, 0)] + self.c_cost
            while (
                mem[i] + self.fbw_mem[0] * (2 + i * 2) <= self.max_mem
                and cur_time[i] + self.fbw_cost[0] <= tmp
                and count[i][0] < self.n_micro
            ):
                for j in range(i + 1):
                    put(0, 0, j)
            put(0, 1, i)
        iter_chunk_ = 0
        end_tmp = 0
        for i in range(self.n_stage):
            if i == 0:
                end_tmp = cur_time[0] + self.fbw_cost[1]
                continue
            tmp = end_tmp + self.c_cost
            while (
                count[i][0] + count[i][1] < count[i - 1][0] + count[i - 1][1]
                or count[i][1] <= count[i - 1][1] < self.n_micro
            ):
                for j in range(self.n_stage - 1, i - 1, -1):
                    if count[j][iter_chunk_] < self.n_micro:
                        put(0, iter_chunk_, j)
                iter_chunk_ = 1 - iter_chunk_

        for _ in range(2 * self.n_micro):
            # check mem before putting b
            for i in range(self.n_stage):
                while mem[i] + self.fbw_mem[1] > self.max_mem:
                    assert len(pending_w[i]) > 0
                    put_w(i)
            b0_ranks, b1_ranks = [], []
            for i in range(self.n_stage):
                if count[i][3] >= count[i][2]:
                    b0_ranks.append(i)
                elif i == self.n_stage - 1:
                    b1_ranks.append(i)
                else:
                    fa_id = self.get_id(1, 1, i + 1, count[i][3])
                    if end_time[fa_id] >= 0 or count[i][2] >= self.n_micro:
                        b1_ranks.append(i)
                    else:
                        b0_ranks.append(i)
            b_ranks = []
            # put b1
            for i in reversed(b1_ranks):
                b_ranks.append((i, 1))
            # put b0
            for i in b0_ranks:
                b_ranks.append((i, 0))
            for i, _chunk_ in b_ranks:
                fa_id = -1
                if _chunk_ == 1 and i < self.n_stage - 1:
                    fa_id = self.get_id(1, 1, i + 1, count[i][3])
                if _chunk_ == 0 and i > 0:
                    fa_id = self.get_id(1, 0, i - 1, count[i][2])
                while (
                    len(pending_w[i]) > 0
                    and fa_id >= 0
                    and end_time[fa_id] + self.c_cost >= cur_time[i] + self.fbw_cost[2]
                ):
                    # fill the bubble
                    put_w(i)
                if (
                    len(pending_w[i]) > 0
                    and end_time[fa_id] + self.c_cost - cur_time[i] > get_max_stage_bubble(i) - stage_bubble[i]
                ):
                    if _chunk_ == 1:
                        put_w(i)
                    elif fill_b:
                        put_w(i)
                put(1, _chunk_, i)

            # put f
            for i in range(self.n_stage):
                if count[i][1] >= self.n_micro:
                    continue
                put_item = None
                if count[i][1] >= count[i][0]:
                    put_item = 0
                elif i == self.n_stage - 1:
                    put_item = 1
                else:
                    if end_time[self.get_id(0, 1, i + 1, count[i][1])] >= 0:
                        put_item = 1
                    elif count[i][0] < self.n_micro:
                        if i == 0:
                            put_item = 0
                        elif end_time[self.get_id(0, 0, i - 1, count[i][0])] >= 0:
                            put_item = 0
                if put_item is None:
                    continue
                # check mem before putting f
                while mem[i] + self.fbw_mem[0] > self.max_mem:
                    assert len(pending_w[i]) > 0
                    put_w(i)
                fa_id = -1
                if put_item == 0 and i > 0:
                    fa_id = self.get_id(0, 0, i - 1, count[i][0])
                if put_item == 1 and i < self.n_stage - 1:
                    fa_id = self.get_id(0, 1, i + 1, count[i][1])
                while (
                    len(pending_w[i]) > 0
                    and fa_id >= 0
                    and end_time[fa_id] + self.c_cost >= cur_time[i] + self.fbw_cost[2]
                ):
                    # fill the bubble
                    put_w(i)
                if (
                    len(pending_w[i]) > 0
                    and end_time[fa_id] + self.c_cost - cur_time[i] > get_max_stage_bubble(i) - stage_bubble[i]
                ):
                    if fill_f:
                        put_w(i)
                put(0, put_item, i)

        for i in range(self.n_stage):
            while len(pending_w[i]) > 0:
                put_w(i)

        max_bubble = get_max_stage_bubble()
        expected_time = sum(self.fbw_cost) * self.n_micro * 2
        bubble_rate = max_bubble / expected_time
        if max_approved_bubble < 0 or max_bubble < max_approved_bubble:
            _schedule, _end_time, _max_bubble = self.try_v_schedule(
                fill_f=fill_f,
                fill_b=fill_b,
                approved_bubble=stage_bubble,
            )
            if _max_bubble < max_bubble:
                return _schedule, _end_time, _max_bubble
        return schedule, end_time, max_bubble

    def print_details(self, end_time, print_scaling=1):
        for stage in range(self.n_stage):
            stage_str = ["."] * int(max(end_time) / print_scaling)
            for _cat in range(3):
                for _chunk in range(2):
                    for _micro in range(self.n_micro):
                        _id = self.get_id(_cat, _chunk, stage, _micro)
                        if end_time[_id] < 0:
                            continue
                        end = int(end_time[_id] / print_scaling)
                        start = int((end_time[_id] - self.fbw_cost[_cat]) / print_scaling)
                        for j in range(start, end):
                            if j == start or j == end - 1:
                                stage_str[j] = "FfBbWw"[_cat * 2 + _chunk]
                            elif j == start + 1:
                                if _micro >= 10:
                                    stage_str[j] = str(_micro // 10)
                                else:
                                    stage_str[j] = str(_micro)
                            elif j == start + 2 and _micro >= 10:
                                stage_str[j] = str(_micro % 10)
                            else:
                                stage_str[j] = "-"
            _str = ""
            for _c in stage_str:
                _str += _c
            print(_str)

    def get_v_schedule(self, only_run_time=False):
        schedule, end_time, max_bubble = None, None, None
        expected_time = sum(self.fbw_cost) * self.n_micro * 2
        for fill_b in [True, False]:
            for fill_f in [True, False]:
                _schedule, _end_time, _max_bubble = self.try_v_schedule(fill_b=fill_b, fill_f=fill_f)
                if max_bubble is None or _max_bubble < max_bubble:
                    max_bubble = _max_bubble
                    schedule = _schedule
                    end_time = _end_time
        if only_run_time:
            return max_bubble + expected_time
        bubble_rate = max_bubble / (expected_time + max_bubble)
        msg = "%2d %3d, [%5d %5d %5d %5d], %6d -> %6.4f" % (
            self.n_stage,
            self.n_micro,
            *self.fbw_cost,
            self.c_cost,
            self.max_mem // self.f_mem,
            bubble_rate,
        )

        logger.info(msg)
        local_order = [[] for _ in range(self.n_stage)]
        comm_id = {}
        comm_id_counter = 0
        post_validation_time = 0
        for i in range(self.n_stage - 1, -1, -1):
            pv_id = min(2 * (self.n_stage - 1 - i), self.n_micro - 1)
            post_validation_time = max(
                post_validation_time, end_time[self.get_id(0, 0, i, pv_id)] - self.fbw_cost[0] - self.c_cost
            )
            for it in ["RECV_", "SEND_", ""]:
                if i == 0 and it == "SEND_":
                    continue
                if i == self.n_stage - 1 and it == "RECV_":
                    continue
                stage_ = i
                local_order[stage_].append(
                    ScheduledNode(
                        type=it + "POST_VALIDATION",
                        chunk=0,
                        stage=stage_,
                        minibatch=0,
                        start_time=post_validation_time,
                        completion_time=post_validation_time,
                    )
                )
                comm_id[local_order[stage_][-1]] = comm_id_counter
                comm_id_counter += 1
        for i in range(self.n_stage):
            for _cat_, _chunk_, _micro_ in schedule[i]:
                complete_time = end_time[self.get_id(_cat_, _chunk_, i, _micro_)]
                local_order[i].append(
                    ScheduledNode(
                        type="FBW"[_cat_],
                        chunk=_chunk_ if _cat_ == 0 else 1 - _chunk_,
                        stage=i,
                        minibatch=_micro_,
                        start_time=complete_time - self.fbw_cost[_cat_],
                        completion_time=complete_time,
                    )
                )
                if _cat_ == 2:  # no communication for W
                    continue
                cat_str = "FORWARD" if _cat_ == 0 else "BACKWARD"

                def communicate(send_recv, stage_):
                    # noinspection PyTypeChecker
                    local_order[stage_].append(
                        ScheduledNode(
                            type=send_recv + cat_str,
                            chunk=_chunk_ if _cat_ == 0 else 1 - _chunk_,
                            stage=stage_,
                            minibatch=_micro_,
                            start_time=complete_time,
                            completion_time=complete_time,
                        )
                    )
                    comm_id[local_order[stage_][-1]] = comm_id_counter

                if _chunk_ == 1 and i > 0:
                    communicate("SEND_", i)
                    communicate("RECV_", i - 1)
                if _chunk_ == 0 and i < self.n_stage - 1:
                    communicate("SEND_", i)
                    communicate("RECV_", i + 1)
                comm_id_counter += 1
        for rank in range(self.n_stage):
            # For nodes with the same timestamp on the same stage, communication will be prioritized.
            def even_breaker(x: ScheduledNode):
                # Compute nodes are always delayed.
                if x.type in ["F", "B", "W"]:
                    return comm_id_counter
                # For comm nodes, order by their unique comm id
                return comm_id[x]

            local_order[rank] = sorted(local_order[rank], key=lambda x: (x.start_time, even_breaker(x)))
            # If a recv with intersects with previous computation, reorder them so that recv
            # is executed before computation and hence can be overlapped.
            for i in range(len(local_order[rank])):
                if (
                    i > 0
                    and local_order[rank][i - 1].type in {"F", "B", "W"}
                    and local_order[rank][i].type.startswith("RECV")
                    and "POST_VALIDATION" not in local_order[rank][i].type
                    and local_order[rank][i].start_time <= local_order[rank][i - 1].completion_time
                ):
                    local_order[rank][i], local_order[rank][i - 1] = local_order[rank][i - 1], local_order[rank][i]

        local_order_with_rollback = [[] for _ in range(self.n_stage)]
        for rank in range(self.n_stage):
            rollback_comm = set()
            if rank > 0:
                for node in local_order[rank - 1]:
                    if node.type == "POST_VALIDATION":
                        break
                    if node.type == "SEND_FORWARD":
                        assert node.chunk == 0
                        rollback_comm.add(node.minibatch)
            for node in local_order[rank]:
                if node.type == "RECV_FORWARD" and node.chunk == 0 and node.minibatch in rollback_comm:
                    rollback = True
                    rollback_comm.remove(node.minibatch)
                else:
                    rollback = False
                local_order_with_rollback[rank].append(
                    ScheduledNode(
                        type=node.type,
                        chunk=node.chunk,
                        stage=node.stage,
                        minibatch=node.minibatch,
                        start_time=node.start_time,
                        completion_time=node.completion_time,
                        rollback=rollback,
                    )
                )
            assert len(rollback_comm) == 0
            msg = ""
            for node in local_order_with_rollback[rank]:
                msg += f"{node.type}-{node.minibatch}-{int(node.rollback)},"
            msg = msg[:-1] + "\n"
            logger.info(msg)

        return local_order_with_rollback


class ZeroBubbleVInstrcutionGenerator(InstructionGenerator):
    def __init__(
        self,
        deps: StageDeps,
        meshes: List[DeviceMesh],
        batches: int,
        f_cost: int,
        b_cost: int,
        w_cost: int,
        c_cost: int,
        f_mem: int,
        b_mem: int,
        w_mem: int,
        max_mem=None,
        default_shape: Optional[Shape] = None,
        default_dtype: Optional[torch.dtype] = None,
    ):
        self.num_chunk = 2  # for ZBV, manually set num chunks be 2 for each worker
        self.deps = deps
        n_stage = deps.num_stage
        n_micro = batches
        self.cost_graph = CostGraph(n_stage, n_micro, f_cost, b_cost, w_cost, c_cost, f_mem, b_mem, w_mem, max_mem=None)
        self.num_stage = len(meshes)
        self.schema = self.cost_graph.get_v_schedule()
        self.default_shape = default_shape
        self.default_dtype = default_dtype

    def gen_instruction(self):
        self.instruction_list = [[] for _ in range(self.num_stage)]
        self.instruction_list_str = ["" for _ in range(self.num_stage)]

        for stage in range(self.num_stage):
            stage_str = ""
            for node in self.schema[stage]:
                self._set_inst(node, stage)
                stage_str += node.type + ","
            stage_str = stage_str[:-1]

        self.gen_instruction_str_list()

    def gen_instruction_str_list(self):
        instruction_lists = self.instruction_list
        stage_strs = defaultdict(str)
        for stage_id, instruction_list in enumerate(instruction_lists):
            cur_stage_str = stage_strs[stage_id]
            for inst in instruction_list:
                cur_stage_str += f"{VESCALE_INSTRUCTION_MAPPING_ZBV[inst.type]},"
            cur_stage_str = cur_stage_str[:-1]
            stage_strs[stage_id] = cur_stage_str
        builder.build_from_dict(stage_strs)

    @manage_dump_file
    def execute(
        self,
        stage_id,
        autocast_dtype=torch.float32,
        enable_autocast=False,
        grad_scaler=None,
        deallocate_pipeline_outputs=False,
    ):
        # init constant data
        builder.constant_data["autocast_dtype"] = autocast_dtype
        builder.constant_data["enable_autocast"] = enable_autocast
        builder.constant_data["grad_scaler"] = grad_scaler
        builder.constant_data["deallocate_pipeline_outputs"] = deallocate_pipeline_outputs
        builder.constant_data["total_stages"] = self.num_stage
        builder.constant_data["stagedeps"] = self.deps
        builder.constant_data["default_shape"] = self.default_shape
        builder.constant_data["default_dtype"] = self.default_dtype

        # Model chunk IDs with synchronized grads
        builder.user_data["synchronized_model_chunks"] = set()
        builder.user_data["input_tensors"] = [[] for _ in range(self.num_chunk)]
        builder.user_data["output_tensors"] = [[] for _ in range(self.num_chunk)]
        builder.user_data["output_tensor_grads"] = [[] for _ in range(self.num_chunk)]
        builder.user_data["fwd_wait_handles"] = None
        builder.user_data["bwd_wait_handles"] = None
        builder.user_data["output_tensor"] = None
        builder.user_data["input_tensor"] = (None, None)
        builder.user_data["output_tensor_grad"] = None
        builder.user_data["forward_data_store"] = []
        model = self.deps.get_current_model(stage_id)

        builder.model = model
        instruction_list = self.get_instruction_list(stage_id)
        builder.stage_id = stage_id
        builder_instruction_list = builder.global_instructions_funcs[stage_id]

        assert len(instruction_list) == len(builder_instruction_list)
        # print(f"cur stage {stage_id} debug inst list: {instruction_list} len inst {len(instruction_list)}")

        for inst, fn in zip(instruction_list, builder_instruction_list):
            builder.user_data["inst"] = inst
            fn()

        return builder.user_data["forward_data_store"]


# communication


@register_instruction(name="vescale_zbv_send_forward")
def vescale_zbv_send_forward():
    inst = builder.user_data["inst"]
    output_tensors = builder.user_data["output_tensor"]

    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]

    def f(info):
        output_tensor, comm, shape = info
        send_forward(
            output_tensor=maybe_tensor(output_tensor),
            current_device_mesh=comm.cur_mesh,
            peer_device_mesh=comm.peer_mesh,
            tensor_shape=shape,
        )
        cross_mesh_send(comm, output_tensor)

    comm_packages = inst.get_send_comms(builder.constant_data["total_stages"], builder.constant_data["stagedeps"])

    shapes = [builder.constant_data["default_shape"] for _ in comm_packages]
    infos = zip(output_tensors, comm_packages, shapes)
    return list(map(f, infos))


@register_instruction(name="vescale_zbv_recv_forward")
def vescale_zbv_recv_forward():
    inst = builder.user_data["inst"]
    chunk_id = inst.chunk
    mbx = inst.minibatch

    def f(info):
        comm, shape, dtype = info
        p2p_tensor = recv_forward(
            tensor_shape=shape,
            recv_dtype=dtype,
            current_device_mesh=comm.cur_mesh,
            peer_device_mesh=comm.peer_mesh,
        )
        p2p_tensor = cross_mesh_recv(comm, p2p_tensor)
        return p2p_tensor

    comm_packages = inst.get_recv_comms(builder.constant_data["total_stages"], builder.constant_data["stagedeps"])
    shapes = [builder.constant_data["default_shape"] for _ in comm_packages]
    dtypes = [builder.constant_data["default_dtype"] for _ in comm_packages]
    infos = zip(comm_packages, shapes, dtypes)
    out = list(map(f, infos))
    input_tensor = out if len(out) > 0 else None
    builder.user_data["input_tensor"] = (input_tensor, mbx)
    builder.user_data["input_tensors"][chunk_id].append((input_tensor, mbx))
    return input_tensor


@register_instruction(name="vescale_zbv_send_backward")
def vescale_zbv_send_backward():
    inst = builder.user_data["inst"]
    input_tensor_grad = builder.user_data["input_tensor_grad"]
    if not isinstance(input_tensor_grad, list):
        input_tensor_grad = [input_tensor_grad]

    def f(info):
        grad, comm, shape = info
        send_backward(
            input_tensor_grad=maybe_tensor(grad),
            current_device_mesh=comm.cur_mesh,
            peer_device_mesh=comm.peer_mesh,
            tensor_shape=shape,
        )
        cross_mesh_send(comm, grad)

    recv_comms = inst.get_recv_comms(builder.constant_data["total_stages"], builder.constant_data["stagedeps"])
    shapes = [builder.constant_data["default_shape"] for _ in recv_comms]
    infos = zip(input_tensor_grad, recv_comms, shapes)
    return list(map(f, infos))


@register_instruction(name="vescale_zbv_recv_backward")
def vescale_zbv_recv_backward():
    inst = builder.user_data["inst"]
    chunk_id = inst.chunk

    def f(info):
        comm, shape, dtype = info
        p2p_tensor = recv_backward(
            tensor_shape=shape,
            recv_dtype=dtype,
            current_device_mesh=comm.cur_mesh,
            peer_device_mesh=comm.peer_mesh,
        )
        p2p_tensor = cross_mesh_recv(comm, p2p_tensor)
        return p2p_tensor

    comm_packages = inst.get_send_comms(builder.constant_data["total_stages"], builder.constant_data["stagedeps"])
    shapes = [builder.constant_data["default_shape"] for _ in comm_packages]
    dtypes = [builder.constant_data["default_dtype"] for _ in comm_packages]
    infos = zip(comm_packages, shapes, dtypes)
    out = list(map(f, infos))
    output_tensor_grad = out if len(out) > 0 else None

    builder.user_data["output_tensor_grad"] = output_tensor_grad
    builder.user_data["output_tensor_grads"][chunk_id].append(output_tensor_grad)
    return output_tensor_grad


# forward


@register_instruction(name="vescale_zbv_forward")
def vescale_zbv_forward():
    inst = builder.user_data["inst"]
    chunk_id = inst.chunk
    stage_id = inst.stage
    mbx = inst.minibatch
    cur_model = builder.model[chunk_id]

    user_data = builder.user_data
    forward_data_store = user_data["forward_data_store"]
    input_tensors = user_data["input_tensors"]
    output_tensors = user_data["output_tensors"]

    constant_data = builder.constant_data
    autocast_dtype = constant_data["autocast_dtype"]
    enable_autocast = constant_data["enable_autocast"]

    is_pp_first_stage = stage_id == 0 and chunk_id == 0
    is_pp_last_stage = stage_id == 0 and chunk_id == 1

    # forward step
    if is_pp_first_stage:
        if len(input_tensors[chunk_id]) == len(output_tensors[chunk_id]):
            input_tensors[chunk_id].append(None)

    # find corresponding input tensor
    input_tensor = None
    for cur_item in input_tensors[chunk_id]:
        if cur_item is not None and cur_item[1] == mbx:
            input_tensor = cur_item[0]

    if not is_pp_first_stage:
        assert input_tensor is not None

    if enable_autocast:
        context_manager = torch.autocast("cuda", dtype=autocast_dtype)
    else:
        context_manager = contextlib.nullcontext()

    with context_manager:

        def prepare_data():
            model_chunk_id = builder.user_data["model_chunk_id"]
            ground_truth = []
            if builder.user_data["is_pp_first_stage"]:
                true_input_tensor = next(builder.dataloader[model_chunk_id])
                # keep the input tensor in builder
                if len(input_tensors[chunk_id]) == len(output_tensors[chunk_id]) + 1:
                    true_input_tensor.requires_grad_()
                    builder.user_data["input_tensors"][chunk_id].pop()
                    builder.user_data["input_tensors"][chunk_id].append((true_input_tensor, mbx))
            else:
                local_tensors = next(builder.dataloader[model_chunk_id])
                if isinstance(local_tensors, Sequence) and len(local_tensors) > 1:
                    ground_truth.append(local_tensors[-1])
                elif isinstance(local_tensors, Dict) and "labels" in local_tensors:
                    ground_truth.append(local_tensors["labels"])
                true_input_tensor = builder.user_data["p2p_tensors"]
                if isinstance(true_input_tensor, Sequence):
                    true_input_tensor = true_input_tensor[0]

            return (true_input_tensor,), {}, ground_truth

        builder.user_data["model_chunk_id"] = chunk_id
        builder.user_data["p2p_tensors"] = input_tensor
        builder.user_data["is_pp_first_stage"] = is_pp_first_stage
        builder.user_data["is_pp_last_stage"] = is_pp_last_stage
        builder.user_data["prepare_data_fn"] = prepare_data
        args, kwargs, ground_truth = builder.user_data["prepare_data_fn"]()
        builder.user_data["ground_truth"] = ground_truth
        output_tensor = cur_model(*args, **kwargs)

    if is_pp_last_stage:
        output_tensor, loss_tensor = registed_functions["vescale_zbv_loss_fn"](output_tensor)
        forward_data_store.append((output_tensor, loss_tensor))
        output_tensor = output_tensor if builder.loss_fn is None else loss_tensor

    if stage_id + 1 == builder.constant_data["total_stages"] and chunk_id == 0:
        # turn around the forward direction
        builder.user_data["input_tensor"] = (output_tensor, mbx)
        builder.user_data["input_tensors"][chunk_id + 1].append((output_tensor, mbx))

    builder.user_data["output_tensors"][chunk_id].append(output_tensor)
    user_data["output_tensor"] = output_tensor


# backward


@register_instruction(name="vescale_zbv_backward_b")
def vescale_zbv_backward_b():
    inst = builder.user_data["inst"]
    chunk_id = inst.chunk
    stage_id = inst.stage
    grad_scaler = builder.constant_data["grad_scaler"]
    deallocate_pipeline_outputs = builder.constant_data["deallocate_pipeline_outputs"]

    input_tensors = builder.user_data["input_tensors"]
    output_tensors = builder.user_data["output_tensors"]
    output_tensor_grads = builder.user_data["output_tensor_grads"]

    is_pp_last_stage = stage_id == 0 and chunk_id == 1

    if is_pp_last_stage:
        if len(output_tensor_grads[chunk_id]) == 0:
            output_tensor_grads[chunk_id].append(None)
    input_tensor = input_tensors[chunk_id].pop(0)[0]
    output_tensor = output_tensors[chunk_id][0]
    output_tensor_grad = output_tensor_grads[chunk_id][0]

    # Retain the grad on the input_tensor.
    unwrap_input_tensor_grad = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_input_tensor_grad = True
    for x in input_tensor:
        if x is not None:
            x.retain_grad()

    if not isinstance(output_tensor, list):
        output_tensor = [output_tensor]
    if not isinstance(output_tensor_grad, list):
        output_tensor_grad = [output_tensor_grad]

    # extract loss value from output tensors
    if isinstance(output_tensor[0], Sequence):
        for j in range(len(output_tensor[0])):
            if output_tensor[0][j].ndim == 0 and output_tensor[0][j].numel() == 1:
                loss_value = output_tensor[0][j]
                break
        else:
            loss_value = output_tensor[0][-1]
    else:
        loss_value = output_tensor[0]

    # Backward pass.
    if output_tensor_grad[0] is None and grad_scaler is not None:
        loss_value = grad_scaler(loss_value)
    # FIXME: For virtual pipeline, there may exist frozen layer without grad;
    # Need to verify if this solution is correct
    if not loss_value.requires_grad:
        return None

    if deallocate_pipeline_outputs:
        assert 0
        # custom_backward(output_tensor[0], output_tensor_grad[0])
    else:
        input_tensor_grad = switch_dtensor(torch.autograd.grad)(
            loss_value,
            input_tensor,
            grad_outputs=output_tensor_grad[0],
            retain_graph=True,
            allow_unused=True,
            materialize_grads=True,
        )[0]

    if unwrap_input_tensor_grad:
        input_tensor_grad = input_tensor_grad[0]

    def f(input_tensor):
        if input_tensor is not None:
            assert isinstance(input_tensor, (torch.Tensor, DTensor)), input_tensor
            input_tensor.grad = None

        nonlocal output_tensor

        if not isinstance(output_tensor, Sequence):
            output_tensor = [output_tensor]

        if (output_tensor is None) or (not deallocate_pipeline_outputs):
            return
        assert isinstance(
            output_tensor, [torch.Tensor, DTensor]
        ), f"expected Tensor, found {type(output_tensor).__name__}."
        assert output_tensor._base is None, "counter-productive to free a view of another tensor."
        if isinstance(output_tensor, [torch.Tensor, DTensor]):
            output_tensor._local_tensor.data = torch.empty(
                (1,),
                device=output_tensor.device,
                dtype=output_tensor.dtype,
            )
        else:
            output_tensor.data = torch.empty(
                (1,),
                device=output_tensor.device,
                dtype=output_tensor.dtype,
            )
        return

    if not isinstance(input_tensor, Sequence):
        map(f, [input_tensor])
    else:
        map(f, input_tensor)

    if stage_id + 1 == builder.constant_data["total_stages"] and chunk_id == 1:
        # turn around the forward direction
        builder.user_data["output_tensor_grad"] = input_tensor_grad
        builder.user_data["output_tensor_grads"][chunk_id - 1].append(output_tensor_grad)

    builder.user_data["input_tensor_grad"] = input_tensor_grad


@register_instruction(name="vescale_zbv_backward_w")
def vescale_zbv_backward_w():
    inst = builder.user_data["inst"]
    chunk_id = inst.chunk
    stage_id = inst.stage
    cur_model = builder.model[chunk_id]
    grad_scaler = builder.constant_data["grad_scaler"]
    deallocate_pipeline_outputs = builder.constant_data["deallocate_pipeline_outputs"]

    output_tensors = builder.user_data["output_tensors"]
    output_tensor_grads = builder.user_data["output_tensor_grads"]

    is_pp_last_stage = stage_id == 0 and chunk_id == 1

    if is_pp_last_stage:
        if len(output_tensor_grads[chunk_id]) == 0:
            output_tensor_grads[chunk_id].append(None)
    output_tensor = output_tensors[chunk_id].pop(0)
    output_tensor_grad = output_tensor_grads[chunk_id].pop(0)

    if not isinstance(output_tensor, list):
        output_tensor = [output_tensor]
    if not isinstance(output_tensor_grad, list):
        output_tensor_grad = [output_tensor_grad]

    # Backward pass.
    if output_tensor_grad[0] is None and grad_scaler is not None:
        output_tensor = grad_scaler(output_tensor[0])
    # FIXME: For virtual pipeline, there may exist frozen layer without grad;
    # Need to verify if this solution is correct
    if not output_tensor[0].requires_grad:
        return None

    # Gather params
    nps = {}
    for key, value in cur_model.named_parameters():
        nps[key] = value

    if deallocate_pipeline_outputs:
        assert 0
    else:
        params_grad = switch_dtensor(torch.autograd.grad)(
            output_tensor[0],
            nps.values(),
            grad_outputs=output_tensor_grad[0],
            retain_graph=True,
            allow_unused=True,
            materialize_grads=True,
        )

    # Manually set each params grad
    for param, grad in zip(nps.values(), params_grad):
        param.grad = grad


# validation


@register_instruction(name="vescale_zbv_post_validation")
def vescale_zbv_post_validation():
    pass


@register_instruction(name="vescale_zbv_recv_post_validation")
def vescale_zbv_recv_post_validation():
    pass


@register_instruction(name="vescale_zbv_send_post_validation")
def vescale_zbv_send_post_validation():
    pass


# loss


@register_instruction(name="vescale_zbv_loss_fn")
def vescale_zbv_loss_fn(output_tensor):
    loss_func = builder.loss_fn
    if loss_func is None:
        return output_tensor, None
    temp_tensor = output_tensor
    args_spec = signature(loss_func)
    args_len = len(args_spec.parameters.keys())
    if args_len == 1:
        output_tensor = loss_func(output_tensor)
    else:
        ground_truth = builder.user_data["ground_truth"]
        loss_fn_inputs = [output_tensor] + ground_truth
        output_tensor = loss_func(*loss_fn_inputs)
        assert args_len == len(loss_fn_inputs), "Mismatch of loss function #args and #actual inputs!"
    builder.user_data["output_tensor"] = output_tensor
    return temp_tensor, output_tensor


VESCALE_INSTRUCTION_MAPPING_ZBV = {
    "RECV_FORWARD": "vescale_zbv_recv_forward",
    "SEND_FORWARD": "vescale_zbv_send_forward",
    "F": "vescale_zbv_forward",
    "B": "vescale_zbv_backward_b",
    "W": "vescale_zbv_backward_w",
    "RECV_BACKWARD": "vescale_zbv_recv_backward",
    "SEND_BACKWARD": "vescale_zbv_send_backward",
    "RECV_POST_VALIDATION": "vescale_zbv_recv_post_validation",
    "SEND_POST_VALIDATION": "vescale_zbv_send_post_validation",
    "POST_VALIDATION": "vescale_zbv_post_validation",
}

if __name__ == "__main__":
    settings = [
        # p,   n,     f,     b,     w,   c,    h,  a,  l
        # (8, 24, 18522, 18086, 9337, 601, 2304, 24, 24),
        # (8, 32, 18513, 18086, 9331, 626, 2304, 24, 24),
        # (8, 64, 18546, 18097, 9321, 762, 2304, 24, 24),
        # (8, 24, 29718, 29444, 19927, 527, 4096, 32, 32),
        # (8, 32, 29802, 29428, 19530, 577, 4096, 32, 32),
        # (8, 64, 29935, 29621, 19388, 535, 4096, 32, 32),
        # (16, 48, 11347, 11248, 8132, 377, 5120, 40, 48),
        # (16, 64, 11307, 11254, 8101, 379, 5120, 40, 48),
        # (16, 128, 11325, 11308, 8109, 378, 5120, 40, 48),
        # (32, 96, 10419, 10207, 7715, 408, 6144, 48, 64),
        # (32, 128, 10408, 10204, 7703, 408, 6144, 48, 64),
        # (32, 256, 10402, 10248, 7698, 460, 6144, 48, 64),
        (4, 8, 6, 4, 4, 1, 4096, 32, 32),
        # (8, 24, 29444, 29718, 19927, 527, 4096, 32, 32),
        # ( 8, 32, 16099, 16504,  7589,  540, 2304, 24, 16),
        # (16, 48, 14407, 14380, 9676, 1610, 4096, 32, 32),
        # (16, 64, 14412, 14393, 9688, 1621, 4096, 32, 32),
        # (16, 128, 14316, 14306, 9639, 1619, 4096, 32, 32),
        # (24, 72, 6763, 6969, 5251, 755, 5120, 40, 48),
        # (24, 96, 6783, 6984, 5259, 758, 5120, 40, 48),
        # (24, 192, 6785, 6990, 5260, 770, 5120, 40, 48),
        # (32, 96, 9458, 9748, 7288, 879, 6144, 48, 64),
        # (32, 128, 9469, 9744, 7306, 892, 6144, 48, 64),
        # (32, 256, 9447, 9644, 7193, 887, 6144, 48, 64),
    ]
    s = 1024

    # h, a, s = 4096, 32, 1024
    # cost_f, cost_b, cost_w, cost_c = 29718, 29444, 19927, 527
    for p, n, f, b, w, c, h, a, _ in settings:
        mem_f = 34 * h + 5 * a * s
        mem_w = -32 * h
        mem_b = -mem_w - mem_f
        for m_offset in range(p + 1):
            graph = CostGraph(
                n_stage=p,
                n_micro=n,
                f_cost=f,
                b_cost=b,
                w_cost=w,
                c_cost=c,
                f_mem=mem_f,
                b_mem=mem_b,
                w_mem=mem_w,
                max_mem=mem_f * (p * 2 + m_offset),
            )
            graph.get_v_schedule()
            break
