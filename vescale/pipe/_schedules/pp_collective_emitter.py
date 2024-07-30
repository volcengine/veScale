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

from typing import List, Union, Dict
import logging

import torch
from torch.export.graph_signature import TensorArgument

from vescale.pipe.pipe_emmiter import ScheduleEngine, OneFOneBInstrcutionGenerator
from vescale.plan.spec import ScheduleType
from vescale.pipe._schedules.instruction_base import (
    BaseInstruction,
    CompilePPCollectiveKind,
    CompilePPCollectiveOperator,
)

logger = logging.getLogger(__name__)


def read_fg(fg):
    num_inputs = 0
    num_outputs = None
    for node in fg.graph.nodes:
        if node.op == "placeholder":
            num_inputs += 1
        if node.op == "output":
            num_outputs = len(node.args[0])
    return num_inputs, num_outputs


class PPCollectiveOpEmitter:
    def __init__(self, curr_rank: int = None) -> None:
        self.num_params_and_buffers = self.num_real_inputs = self.num_real_outputs = None

        self.curr_rank = curr_rank

        self.fwd_send_dsts = []
        self.bwd_send_dsts = []
        self.fwd_recv_srcs = []
        self.bwd_recv_srcs = []

    def gen_pp_collective_topo_from_schedule_engine(self, pipe_engine: ScheduleEngine):
        fwd_recv_srcs, fwd_send_dsts, bwd_send_dsts, bwd_recv_srcs = set(), set(), set(), set()
        assert (
            pipe_engine.schedule == ScheduleType.SIMPLE_1F1B
        ), "For inserting send/recv operators, we only need the topology information, please consider use this simplier PipeSchedule"
        assert isinstance(
            pipe_engine.p_emmiter.instruction_generator, OneFOneBInstrcutionGenerator
        ), "For inserting send/recv operators, we only need the topology information, please consider use this simplier PipeSchedule"
        insts: List[BaseInstruction] = pipe_engine.get_instruction_list(pipe_engine.stage_id)
        compiled_insts: List[List[CompilePPCollectiveOperator]] = [
            inst.compile() for inst in insts if hasattr(inst, "compile")
        ]
        flat_compile_insts = []
        for list_insts in compiled_insts:
            flat_compile_insts.extend(list_insts)
        for inst in flat_compile_insts:
            if inst.kind is CompilePPCollectiveKind.BORADCAST:
                raise NotImplementedError("broadcast is not supported now")
            elif inst.kind is CompilePPCollectiveKind.SEND:
                if inst.is_backward:
                    bwd_send_dsts.add(inst.dst)
                else:
                    fwd_send_dsts.add(inst.dst)
            elif inst.kind is CompilePPCollectiveKind.RECV:
                if inst.is_backward:
                    bwd_recv_srcs.add(inst.src)
                else:
                    fwd_recv_srcs.add(inst.src)
            else:
                raise NotImplementedError("Unknown collective operators")
        self.gen_pp_collective_topo_from_given(
            list(fwd_send_dsts), list(fwd_recv_srcs), list(bwd_send_dsts), list(bwd_recv_srcs)
        )

    def gen_pp_collective_topo_from_given(
        self,
        fwd_send_dsts: List[int] = None,
        fwd_recv_srcs: List[int] = None,
        bwd_send_dsts: List[int] = None,
        bwd_recv_srcs: List[int] = None,
    ):
        self.fwd_send_dsts = fwd_send_dsts
        self.fwd_recv_srcs = fwd_recv_srcs
        self.bwd_send_dsts = bwd_send_dsts
        self.bwd_recv_srcs = bwd_recv_srcs

    # this function should return a dict to indicate a output_spec change in ExportedProgram
    def insert_send_fwd(self, fg: torch.fx.GraphModule) -> Dict[str, str]:
        if not self.fwd_send_dsts:
            return {}
        assert len(self.fwd_send_dsts) == self.num_real_outputs
        replaced_outputs = {}
        for node in fg.graph.nodes:
            if node.op != "output":
                continue
            with fg.graph.inserting_before(node):
                node_args = node.args[0]
                for i in range(self.num_real_outputs):
                    arg = node_args[i]
                    new_node = fg.graph.create_node(
                        op="call_function",
                        target=torch.ops.c10d_functional.send.default,
                        args=(
                            arg,
                            self.fwd_send_dsts[i],
                            f"{self.curr_rank}{self.fwd_send_dsts[i]}",
                            [self.curr_rank, self.fwd_send_dsts[i]],
                            2,
                        ),
                        kwargs={},
                        name="pp_send_fwd",
                    )
                    new_node.meta["stack_trace"] = "inserted by pp_collective_emitter"
                    new_node.meta["val"] = arg.meta.get("val", None)
                    new_node.meta["tensor_meta"] = arg.meta.get("tensor_meta", None)
                    replaced_outputs[arg.name] = new_node.name
                    node.replace_input_with(arg, new_node)
        fg.recompile()
        return replaced_outputs

    def insert_recv_fwd(self, fg: torch.fx.GraphModule):
        if not self.fwd_recv_srcs:
            return
        assert len(self.fwd_recv_srcs) == self.num_real_inputs
        seen_placeholders = 0
        for node in fg.graph.nodes:
            if node.op != "placeholder":
                continue
            seen_placeholders += 1
            if seen_placeholders <= self.num_params_and_buffers:
                continue
            real_input_idx = seen_placeholders - self.num_params_and_buffers - 1
            with fg.graph.inserting_after(node):
                src = self.fwd_recv_srcs[real_input_idx]
                new_node = fg.graph.create_node(
                    op="call_function",
                    target=torch.ops.c10d_functional.recv.default,
                    args=(
                        node,
                        src,
                        f"{src}{self.curr_rank}",
                        [src, self.curr_rank],
                        2,
                    ),
                    kwargs={},
                    name="pp_recv_fwd",
                )
                new_node.meta["stack_trace"] = "inserted by pp_collective_emitter"
                new_node.meta["val"] = node.meta.get("val", None)
                new_node.meta["tensor_meta"] = node.meta.get("tensor_meta", None)
                for user in list(node.users):
                    if user == new_node:
                        continue
                    user.replace_input_with(node, new_node)

        fg.recompile()

    def insert_send_bwd(self, fg: torch.fx.GraphModule):
        if not self.bwd_send_dsts:
            return
        assert len(self.bwd_send_dsts) == self.num_real_inputs
        for node in fg.graph.nodes:
            if node.op != "output":
                continue
            with fg.graph.inserting_before(node):
                args = node.args[0]
                for i in range(self.num_real_inputs):
                    dst = self.bwd_send_dsts[i]
                    arg = args[i + self.num_params_and_buffers]
                    new_node = fg.graph.create_node(
                        op="call_function",
                        target=torch.ops.c10d_functional.send.default,
                        args=(
                            arg,
                            dst,
                            f"{self.curr_rank}{dst}",
                            [self.curr_rank, dst],
                            2,
                        ),
                        kwargs={},
                        name="pp_send_bwd",
                    )
                    new_node.meta["stack_trace"] = "inserted by pp_collective_emitter"
                    new_node.meta["val"] = arg.meta.get("val", None)
                    new_node.meta["tensor_meta"] = arg.meta.get("tensor_meta", None)
                    node.replace_input_with(arg, new_node)
        fg.recompile()

    def insert_recv_bwd(self, fg: torch.fx.GraphModule):
        if not self.bwd_recv_srcs:
            return
        assert len(self.bwd_recv_srcs) == self.num_real_outputs
        seen_placeholders = 0
        for node in fg.graph.nodes:
            if node.op != "placeholder":
                continue
            seen_placeholders += 1
            if seen_placeholders <= self.num_params_and_buffers:
                continue
            with fg.graph.inserting_after(node):
                src = self.bwd_recv_srcs[seen_placeholders - self.num_params_and_buffers - 1]
                new_node = fg.graph.create_node(
                    op="call_function",
                    target=torch.ops.c10d_functional.recv.default,
                    args=(
                        node,
                        src,
                        f"{src}{self.curr_rank}",
                        [src, self.curr_rank],
                        2,
                    ),
                    kwargs={},
                    name="pp_recv_bwd",
                )
                new_node.meta["stack_trace"] = "inserted by pp_collective_emitter"
                new_node.meta["val"] = node.meta.get("val", None)
                new_node.meta["tensor_meta"] = node.meta.get("tensor_meta", None)
                for user in list(node.users):
                    if user == new_node:
                        continue
                    user.replace_input_with(node, new_node)

        fg.recompile()

    def load_original_graph_module(self, original_gm):
        named_parameters = dict(original_gm.named_parameters(remove_duplicate=False))
        named_buffers = dict(original_gm.named_buffers(remove_duplicate=False))
        self.num_params_and_buffers = len(named_buffers) + len(named_parameters)
        self.num_real_inputs, self.num_real_outputs = read_fg(original_gm)

    def run(self, fg: Union[torch.fx.GraphModule, torch.export.ExportedProgram] = None, is_backward: bool = None):
        if isinstance(fg, torch.fx.GraphModule):
            logging.info(
                "You are inserting PP collective operators to a torch.compiled graph, make sure call PPCollectiveOpEmitter.load_original_graph_module first"
            )
            assert (
                self.num_real_outputs is not None
                and self.num_params_and_buffers is not None
                and self.num_real_inputs is not None
            ), "Please call PPCollectiveOpEmitter.load_original_graph_module first"

            assert is_backward is not None, "Please provide is_backward argument"
            if not is_backward:
                num_total_inputs, _ = read_fg(fg)
            else:
                _, num_total_inputs = read_fg(fg)
            assert num_total_inputs == self.num_real_inputs + self.num_params_and_buffers
            if not is_backward:
                self.insert_send_fwd(fg)
                self.insert_recv_fwd(fg)
            else:
                self.insert_send_bwd(fg)
                self.insert_recv_bwd(fg)
            return fg

        elif isinstance(fg, torch.export.ExportedProgram):
            logging.info("You are inserting PP collective operators to a torch.exported graph")
            ep = fg
            self.num_params_and_buffers = len(ep.state_dict)
            fg = ep.graph_module
            self.num_real_inputs, self.num_real_outputs = read_fg(fg)
            self.num_real_inputs -= self.num_params_and_buffers
            replaced_outputs = self.insert_send_fwd(fg)
            self.insert_recv_fwd(fg)

            # output_spec changes
            for o_spec in ep._graph_signature.output_specs:
                if isinstance(o_spec.arg, TensorArgument) and o_spec.arg.name in replaced_outputs:
                    o_spec.arg = TensorArgument(replaced_outputs[o_spec.arg.name])
            return ep

        else:
            raise NotImplementedError("Unknown model type")
