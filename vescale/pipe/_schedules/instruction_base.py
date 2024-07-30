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

import enum
from dataclasses import dataclass
from collections import defaultdict
from abc import ABCMeta, abstractmethod
from typing import Sequence, Callable
import torch
from torch.distributed.distributed_c10d import get_rank
from vescale.dtensor.device_mesh import DeviceMesh
from vescale.dtensor.placement_types import Placement
from vescale.pipe.pipe_stage import PipeModule
from typing import List, Tuple, Union, Optional, Dict, Any
import logging
import numpy as np
from vescale.plan.spec import PipelineP2PSpec

Shape = Union[List[int], torch.Size]

logger = logging.getLogger(__name__)
registed_functions = {}


def register_instruction(name):
    assert name is not None, "The Instruction must have name"
    if name in registed_functions:
        msg = f"{name} allready in registed instruction"
        logger.warning(msg)

    def _register_instruction(func):
        def wrap(*args, **kwargs):
            return func(*args, **kwargs)

        registed_functions.update({name: func})
        return wrap

    return _register_instruction


@dataclass
class CommPacket:
    cur_mesh: DeviceMesh
    peer_mesh: DeviceMesh
    input_id: int
    peer_stage: int
    peer_sharding: List[Placement] = None
    cur_sharding: List[Placement] = None
    is_kwargs: bool = False


class StageDeps:
    def __init__(
        self,
        dep: np.ndarray,
        meshes: List[DeviceMesh],
        vpp_module_list: Union[List, PipeModule],
        p2p_index_mapping: Optional[Dict[int, List[PipelineP2PSpec]]] = None,
    ):
        self.D = dep
        self.M = vpp_module_list
        self.meshes = meshes
        self.is_vpp = self.get_num_chunks() > 1
        self.mapping: Dict = {}
        if p2p_index_mapping is None:
            self.mapping = defaultdict(list)
            self.generate_one_forward_mapping()
        else:
            self.mapping = p2p_index_mapping
            self.parsing_forward_mapping()

        self.recv_tables: Dict[int, List[CommPacket]] = defaultdict(list)
        self.send_tables: Dict[int, List[CommPacket]] = defaultdict(list)
        self.local_dataloader_list: Dict[Any, List[CommPacket]] = defaultdict(list)
        self.construct_communication_graph()

    def construct_communication_graph(self):
        for i in range(self.num_stage):
            cur_mesh = self.get_current_mesh(i)
            cur_mapping = self.mapping[i]  # get the index mapping i
            prior_list = []
            local_data_list = []
            # stage_id: [input_idx, ...]
            for p2p_spec in cur_mapping:
                prev_stage_id = p2p_spec.peer_stage_idx
                input_id = p2p_spec.peer_output_idx
                if prev_stage_id != i:  # not from self
                    prior_list.append((self.get_current_mesh(prev_stage_id), prev_stage_id, input_id))
                else:  # from self stage
                    local_data_list.append(input_id)

            prior_list = sorted(prior_list, key=lambda item: (item[1], item[2]))
            for device, pre, input_id in prior_list:
                sr = CommPacket(
                    cur_mesh=cur_mesh, peer_mesh=device, input_id=input_id, peer_stage=pre
                )  # input is single
                self.recv_tables[i].append(sr)
            for input_id in local_data_list:
                sr = CommPacket(
                    cur_mesh=cur_mesh,
                    peer_mesh=None,
                    input_id=input_id,
                    peer_stage=None,
                )
                self.local_dataloader_list[i].append(sr)

        # construct out degree
        for i in range(self.num_stage):
            prior_list = []
            for j in range(self.num_stage):
                if i == j:  # don't check self , no cycle
                    continue
                j_recvs = self.recv_tables[j]
                for recv in j_recvs:
                    if recv.peer_stage == i:  # is i send to j
                        send = CommPacket(
                            cur_mesh=recv.peer_mesh,
                            peer_mesh=recv.cur_mesh,
                            input_id=recv.input_id,
                            peer_stage=j,
                        )
                        prior_list.append(send)
            # sort by input_id stage id is unneeded
            sorted(prior_list, key=lambda item: item.input_id)
            self.send_tables[i] = prior_list

    def generate_one_forward_mapping(self):
        for i in range(self.num_stage):
            cur_mapping = self.mapping[i]
            pre_stages = self.get_pre_stage(i, ignore_virtual=False)
            assert len(pre_stages) <= 1, "multi branch stage need parse p2p_index_mapping"
            for pre in pre_stages:
                cur_mapping.append(PipelineP2PSpec(pre, 0))

            if self.is_pipeline_first_stage(i):
                cur_mapping.append(PipelineP2PSpec(i, 0))

    def parsing_forward_mapping(self):
        # 1: [(0,0), (1,0), (0,2)]
        for i in range(self.num_stage):
            if i not in self.mapping:
                cur_indexing = []
                pre_stages = self.get_pre_stage(i, ignore_virtual=False)
                assert len(pre_stages) <= 1, "multi branch stage need parse p2p_index_mapping"
                for pre in pre_stages:
                    cur_indexing.append(PipelineP2PSpec(pre, 0))
                if self.is_pipeline_first_stage(i):
                    cur_indexing.append(PipelineP2PSpec(i, 0))
                self.mapping.update({i: cur_indexing})

    def get_send_comms(self, i):
        return self.send_tables[i]

    def get_recv_comms(self, i):
        return self.recv_tables[i]

    def get_local_comms(self, i):
        return self.local_dataloader_list[i]

    @property
    def num_stage(self):
        return len(self.D)

    def is_first(self, s_id):
        pre = self.D[:, s_id]
        non_zero = np.count_nonzero(pre)
        if non_zero == 0:
            return True
        return False

    def is_last(self, s_id):
        post = self.D[s_id]
        non_zero = np.count_nonzero(post)
        if non_zero == 0:
            return True
        return False

    def get_pre_stage(self, i, ignore_virtual=True):
        pre = self.D[:, i]
        stage_ids = np.where(pre == 1)[0].tolist()
        if self.is_first(i) and self.is_vpp and not ignore_virtual:
            last_stages = list(filter(self.is_last, range(self.num_stage)))
            return last_stages
        else:
            return stage_ids

    def get_post_stage(self, i, ignore_virtual=True):
        post = self.D[i]
        stage_ids = np.where(post == 1)[0].tolist()

        if self.is_last(i) and self.is_vpp and not ignore_virtual:
            first_stages = list(filter(self.is_first, range(self.num_stage)))
            return first_stages
        else:
            return stage_ids

    def get_first_stage(self):
        stages = []
        for i in range(self.num_stage):
            pre_stages = self.get_pre_stage(i)
            if len(pre_stages) == 0:  # in-degree is 0
                stages.append(i)
        return stages

    def get_last_stage(self):
        stages = []
        for i in range(self.num_stage):
            post_stages = self.get_post_stage(i)
            if len(post_stages) == 0:  # out-degree is 0
                stages.append(i)
        return stages

    def get_current_model(self, i):
        return self.M

    def is_pipeline_first_stage(self, i):
        pre = self.get_pre_stage(i)
        return len(pre) == 0  # first stage has no input

    def is_pipeline_last_stage(self, i):
        post = self.get_post_stage(i)
        return len(post) == 0  # last stage has no output

    def is_vpp_first_stage(self, i, chunk_id):
        return self.is_pipeline_first_stage(i) and chunk_id == 0

    def is_vpp_last_stage(self, i, chunk_id):
        return self.is_pipeline_last_stage(i) and (chunk_id == (self.get_num_chunks() - 1))

    def get_num_chunks(self):
        if isinstance(self.M, list):
            return len(self.M)
        else:
            return self.M.virtual_chunks

    def get_current_mesh(self, i):
        return self.meshes[i]

    def __str__(self):
        tmp = "\n\n"
        tmp += f"stages: {self.num_stage}, deps:{self.D}\n"
        for i in range(self.num_stage):
            tmp += f"\n===================stage:{i} start=======================\n"
            tmp += "recv : \n"
            for comm in self.recv_tables[i]:
                tmp += f"\t\t recv from {comm.peer_stage} with input:{comm.input_id} comm:{comm}\n"
            tmp += "send : \n"
            for comm in self.send_tables[i]:
                tmp += f"\t\t send to {comm.peer_stage} with  input:{comm.input_id} comm:{comm}\n"
            tmp += "local_dataloader_list : \n"
            for comm in self.local_dataloader_list[i]:
                tmp += f"\t\t local_dataloader with  input:{comm.input_id} comm:{comm}\n"

            tmp += f"===================stage:{i} end=======================\n\n"
        return tmp


def get_linear_pp_module_dep2(module_list: List, device_mesh_list: List[DeviceMesh]):
    stage_len = len(device_mesh_list)  # for forward
    dep = np.zeros((stage_len, stage_len), dtype=np.int64)
    for i in range(stage_len - 1):
        dep[i][i + 1] = 1  # direct graph
    return StageDeps(dep, device_mesh_list, module_list)


@dataclass
class Status:
    batch_idx: int = 0
    stage_id: int = 0
    chunk_id: int = 0
    f_b: "str" = ""  # forward or backward
    stg: "str" = ""  # stage for 1f1b
    k: int = 0

    def __str__(self):
        return f"b:{self.batch_idx}, c:{self.chunk_id}, {self.stg + '-' + self.f_b}"


class PipelineSchema(metaclass=ABCMeta):
    """
    we define this class to abstract the pipeline execute
    Args:
        dep: the dependency for adjacency martrix
        meshes: the list for stage of

    """

    def __init__(self, num_stage: int, meshes: Union[List[DeviceMesh], int], batches: int = 1):
        self.num_stage = num_stage
        self.meshes = meshes
        self.batches = batches
        self._schedules: List[List[Tuple]] = self._gen_schedule()

    @property
    @abstractmethod
    def name(self):
        """print schedule name"""
        raise NotImplementedError()

    @abstractmethod
    def _gen_schedule(self):
        """generator the pipelinne schedule for engine"""
        raise NotImplementedError("not impl")

    def __str__(self):
        """print the pipeline clock work"""
        stream = "\n"
        d = " ".join([f"d{d:<24}" for d in range(self.num_mesh)])
        stream += f"T k :{d:<24} \n"
        for time, scheds in enumerate(self.schedules):
            sched_str = " ".join([f"{str(sched):<24}" for sched in scheds])
            stream += f"T {time:<2}: {sched_str} \n"
        return stream

    @property
    def schedules(self):
        """return schedules"""
        return self._schedules

    @property
    def num_mesh(self):
        """return the num mesh of tp group"""
        if isinstance(self.meshes, Sequence):
            return len(self.meshes)
        elif isinstance(self.meshes, int):
            return self.meshes
        else:
            raise NotImplementedError("unsupport device mesh list")

    @property
    def num_clock(self):
        """return num schedule for the num clock"""

        return len(self._schedules)


@dataclass
class BaseInstruction(metaclass=ABCMeta):
    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError("unsupport run command")

    @property
    def name(self):
        return "base_instruction"

    def dump(self):
        return f"{get_rank()}: {self}"


class InstructionGenerator(metaclass=ABCMeta):
    def __init__(
        self,
        deps: StageDeps,
        meshes: int,
        batches: int,
        default_shape: Optional[Shape] = None,
        default_dtype: Optional[torch.dtype] = None,
        batch_shape_lists: Optional[List[Any]] = None,
        batch_dtype_lists: Optional[List[Any]] = None,
        forward_only=False,
        num_chunk=1,
    ):
        self.deps = deps
        self.meshes = meshes
        self.num_chunk = num_chunk
        self.batches = batches
        self.default_shape = default_shape
        self.default_dtype = default_dtype
        self.batch_shape_lists = batch_shape_lists
        self.batch_dtype_lists = batch_dtype_lists
        self.forward_only = forward_only
        self.instruction_list: List = []

    """
    generate instruction
    """

    @abstractmethod
    def gen_instruction(self):
        raise NotImplementedError("not implement")

    """
    get current stage instruction
    """

    def get_instruction_list(self, stage: int):
        return self.instruction_list[stage]

    """
        update with batch idx, stage idx
    """

    def _set_inst(self, inst: BaseInstruction, s: int):
        self.instruction_list[s].append(inst)

    """
        set instruction type
    """

    def execute(self, *args, **kwargs):
        raise NotImplementedError("not implement")


class InstructionBuilder:
    global_instructions_funcs = defaultdict(list)
    global_instructions_str = defaultdict(list)

    constant_data = defaultdict()
    user_data = defaultdict()
    loss_fn: Callable = torch.sum
    dataloader: Any
    topo: StageDeps
    model: Callable
    stage_id: int
    _pos = 0
    _stack = None

    def build_from_dict(self, instructions: Dict):
        assert isinstance(instructions, dict), "instructions should be dict"
        for stage_id, instruction_list in instructions.items():
            cur_stage_ins_list = instruction_list
            if isinstance(cur_stage_ins_list, str):
                instructions_funcs = cur_stage_ins_list.split(",")
            else:
                instructions_funcs = cur_stage_ins_list

            mapped_functions = [registed_functions[x] for x in instructions_funcs]

            self.global_instructions_funcs[stage_id] = mapped_functions
            self.global_instructions_str[stage_id] = instructions_funcs

    def draw_instructions(self):
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots()
        # draw rectangle
        stage_nums = len(self.global_instructions_str.keys())
        for stage_id, instuctions_strs in self.global_instructions_str.items():
            for id, stage_str in enumerate(instuctions_strs):
                ax.add_patch(plt.Rectangle((id, -1 * stage_id), 1, 1, fill=False, edgecolor="black", lw=2))
                ax.text(id + 0.5, -1 * stage_id + 0.5, stage_str, ha="center", va="center")

        for stage_id in range(stage_nums):
            ax.text(-0.5, -1 * stage_id + 0.5, stage_id, ha="center", va="center")
        # set max xlim and ylim
        max_stages = max(len(x) for x in self.global_instructions_str.values())
        ax.set_xlim(0, max_stages)
        ax.set_ylim(-1 * stage_nums + 1, 1)
        ax.axis("off")
        plt.savefig("instructions.png")

    @property
    def pos(self):
        return self._pos

    @property
    def last(self):
        return self._stack

    def run(self, stage_id: int):
        output = []
        for pos, fn in enumerate(self.global_instructions_funcs[stage_id]):
            self._pos = pos
            out = fn()
            self._stack = out
            output.append(out)
        return output

    def export(self, stage_id, *args, **kwargs):
        func_lists = self.global_instructions_funcs[stage_id]

        class Model(torch.nn.Module):
            def __init__(self, func_lists, model):
                super().__init__()
                self.func_lists = func_lists
                self.model = model

            def forward(self, *args, **kwargs):
                for f in self.func_lists:
                    # TODO: handle this to make forward inst work.
                    if f.__name__ == "forward":
                        activation = self.model(*args, **kwargs)
                        args = (activation,)
                    else:
                        args, kwargs = f(*args, **kwargs)
                return args, kwargs

        model = Model(func_lists, self.model)
        graph = torch.export.export(model, args)
        return graph


class CompilePPCollectiveKind(enum.Enum):
    SEND = 1
    RECV = 2
    BORADCAST = 3  # for cross mesh collective
    UNKNOWN = 4


class CompilePPCollectiveOperator:
    def __init__(
        self,
        kind: CompilePPCollectiveKind,
        src: int = None,
        dst: List[int] = None,
        is_backward: bool = False,
    ) -> None:
        assert kind in (
            CompilePPCollectiveKind.BORADCAST,
            CompilePPCollectiveKind.SEND,
            CompilePPCollectiveKind.RECV,
        )
        self.kind = kind
        self.is_backward = is_backward

        if self.kind is CompilePPCollectiveKind.SEND:
            assert dst is not None and isinstance(dst, int)
        elif self.kind is CompilePPCollectiveKind.RECV:
            assert src is not None and isinstance(src, int)
        else:
            assert src is not None and isinstance(src, int)
            assert dst is not None and isinstance(dst, List[int])
            assert src in dst

        self.src = src
        self.dst = dst
        pass

    def __hash__(self) -> int:
        if isinstance(self.dst, List[int]):
            dst = tuple(self.dst)
        else:
            dst = self.dst
        return hash((self.kind, self.src, dst, self.is_backward))


VESCALE_INTRUCTION_BUILDER = InstructionBuilder()
