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

from vescale.pipe._schedules.instruction_base import (
    PipelineSchema,
    Status,
    InstructionGenerator,
    Shape,
    BaseInstruction,
    StageDeps,
    CommPacket,
    CompilePPCollectiveKind,
    CompilePPCollectiveOperator,
    VESCALE_INTRUCTION_BUILDER as builder,
    register_instruction,
    registed_functions,
)
from functools import partial
from dataclasses import dataclass
from dataclasses import field
from collections import defaultdict
from vescale.dtensor.dtensor import DTensor, make_dtensor
import contextlib
import torch
import torch.distributed as dist
from inspect import signature
from vescale.dtensor.device_mesh import DeviceMesh
from typing import Sequence, Optional, List, Union, Dict, Callable, Tuple
import numpy as np
from vescale.pipe.p2p_communication import (
    recv_backward,
    recv_forward,
    send_backward,
    send_forward,
    send_forward_recv_backward,
    send_backward_recv_forward,
)
from vescale.ndtimeline import ndtimer, ndtimeit_p2p
from vescale.ndtimeline.predefined import FORWARD_COMPUTE, BACKWARD_COMPUTE, CROSS_MESH_RECV, CROSS_MESH_SEND
from vescale.pipe.pipe_stage import PipeModule
from vescale.dtensor._diff import dummy_p2p, manage_dump_file
from torch.distributed._functional_collectives import send, recv
from vescale.dtensor.placement_types import Placement
from vescale.dtensor._utils import compute_global_tensor_info
from torch.distributed.distributed_c10d import _get_default_group


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


@dataclass
class RECV_FORWARD(BaseInstruction):
    comm_packages: List[CommPacket] = field(default_factory=list)
    tensor_shapes: Union[List[Shape], Shape] = field(default_factory=list)
    tensor_dtypes: Union[List[torch.dtype], torch.dtype] = field(default_factory=list)
    batch_id: Optional[int] = None
    debug: str = ""

    @property
    def name(self):
        return "recv_forward"

    def run(self) -> List:
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

        infos = zip(self.comm_packages, self.tensor_shapes, self.tensor_dtypes)
        out = list(map(f, infos))
        return out if len(out) > 0 else None

    def compile(self) -> List[CompilePPCollectiveOperator]:
        out: List[CompilePPCollectiveOperator] = []
        for comm in self.comm_packages:
            cur_mesh, peer_mesh = comm.cur_mesh, comm.peer_mesh
            coordinate = (cur_mesh.mesh == dist.get_rank()).nonzero(as_tuple=True)
            src = peer_mesh.mesh[coordinate].item()

            out.append(CompilePPCollectiveOperator(kind=CompilePPCollectiveKind.RECV, src=src))
        return out


@dataclass
class SEND_FORWARD(BaseInstruction):
    comm_packages: List[CommPacket] = field(default_factory=list)
    tensor_shapes: List[Shape] = field(default_factory=list)
    batch_id: int = 0

    @property
    def name(self):
        return "send_forward"

    @dummy_p2p
    def run(self, output_tensors: List[torch.Tensor]):
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

        infos = zip(output_tensors, self.comm_packages, self.tensor_shapes)
        return list(map(f, infos))

    def compile(self) -> List[CompilePPCollectiveOperator]:
        out: List[CompilePPCollectiveOperator] = []
        for comm in self.comm_packages:
            cur_mesh, peer_mesh = comm.cur_mesh, comm.peer_mesh
            coordinate = (cur_mesh.mesh == dist.get_rank()).nonzero(as_tuple=True)
            dst = peer_mesh.mesh[coordinate].item()

            out.append(CompilePPCollectiveOperator(kind=CompilePPCollectiveKind.SEND, dst=dst))
        return out


@dataclass
class RECV_BACKWARD(BaseInstruction):
    comm_packages: List[CommPacket] = field(default_factory=list)
    tensor_shapes: Union[List[Shape], Shape] = field(default_factory=list)
    tensor_dtypes: List[torch.dtype] = field(default_factory=list)

    @property
    def name(self):
        return "recv_backward"

    @dummy_p2p
    def run(self):
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

        infos = zip(self.comm_packages, self.tensor_shapes, self.tensor_dtypes)
        out = list(map(f, infos))
        return out if len(out) > 0 else None

    def compile(self) -> List[CompilePPCollectiveOperator]:
        out: List[CompilePPCollectiveOperator] = []
        for comm in self.comm_packages:
            cur_mesh, peer_mesh = comm.cur_mesh, comm.peer_mesh
            coordinate = (cur_mesh.mesh == dist.get_rank()).nonzero(as_tuple=True)
            src = peer_mesh.mesh[coordinate].item()

            out.append(CompilePPCollectiveOperator(kind=CompilePPCollectiveKind.RECV, src=src, is_backward=True))
        return out


@dataclass
class SEND_BACKWARD(BaseInstruction):
    recv_comms: List[CommPacket] = field(default_factory=list)
    tensor_shapes: Union[List[Shape], Shape] = field(default_factory=list)

    @property
    def name(self):
        return "send_backward"

    @dummy_p2p
    def run(self, input_tensor_grad):
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

        infos = zip(input_tensor_grad, self.recv_comms, self.tensor_shapes)
        return list(map(f, infos))

    def compile(self) -> List[CompilePPCollectiveOperator]:
        out: List[CompilePPCollectiveOperator] = []
        for comm in self.recv_comms:
            cur_mesh, peer_mesh = comm.cur_mesh, comm.peer_mesh
            coordinate = (cur_mesh.mesh == dist.get_rank()).nonzero(as_tuple=True)
            dst = peer_mesh.mesh[coordinate].item()

            out.append(CompilePPCollectiveOperator(kind=CompilePPCollectiveKind.SEND, dst=dst, is_backward=True))
        return out


@dataclass
class SEND_FORWARD_RECV_BACKWARD(BaseInstruction):
    comm_packages: List[CommPacket] = field(default_factory=list)
    tensor_shapes: Union[List[Shape], Shape] = field(default_factory=list)
    tensor_dtypes: Union[List[torch.dtype], torch.dtype] = field(default_factory=list)
    send_batch_id: int = 0
    recv_batch_id: int = 0

    @property
    def name(self):
        return "send_forward_recv_backward"

    @dummy_p2p
    def run(self, output_tensors):
        if not isinstance(output_tensors, list):
            output_tensors = [output_tensors]

        def f(info):
            output_tensor, comm, shape, dtype = info
            p2p_tensor = send_forward_recv_backward(
                output_tensor=maybe_tensor(output_tensor),
                current_device_mesh=comm.cur_mesh,
                peer_device_mesh=comm.peer_mesh,
                tensor_shape=shape,
                recv_dtype=dtype,
            )
            p2p_tensor = cross_mesh_double(comm, output_tensor, p2p_tensor)
            return p2p_tensor

        infos = zip(output_tensors, self.comm_packages, self.tensor_shapes, self.tensor_dtypes)
        out = list(map(f, infos))
        return out if len(out) > 0 else None

    def compile(self) -> List[CompilePPCollectiveOperator]:
        out: List[CompilePPCollectiveOperator] = []
        for comm in self.comm_packages:
            cur_mesh, peer_mesh = comm.cur_mesh, comm.peer_mesh
            coordinate = (cur_mesh.mesh == dist.get_rank()).nonzero(as_tuple=True)
            peer_rank = peer_mesh.mesh[coordinate].item()

            out.append(CompilePPCollectiveOperator(kind=CompilePPCollectiveKind.SEND, dst=peer_rank))
            out.append(CompilePPCollectiveOperator(kind=CompilePPCollectiveKind.RECV, src=peer_rank, is_backward=True))
        return out


@dataclass
class SEND_BACKWARD_RECV_FORWARD(BaseInstruction):
    recv_comms: List[CommPacket]
    tensor_shapes: Union[List[Shape], Shape] = field(default_factory=list)
    tensor_dtypes: Union[List[torch.dtype], torch.dtype] = field(default_factory=list)

    @property
    def name(self):
        return "send_backward_recv_forward"

    @dummy_p2p
    def run(self, input_tensor_grad):
        if not isinstance(input_tensor_grad, list):
            input_tensor_grad = [input_tensor_grad]

        def f(info):
            grad, comm, shape, dtype = info
            p2p_tenosr = send_backward_recv_forward(
                input_tensor_grad=maybe_tensor(grad),
                current_device_mesh=comm.cur_mesh,
                peer_device_mesh=comm.peer_mesh,
                tensor_shape=shape,
                recv_dtype=dtype,
            )
            p2p_tenosr = cross_mesh_double(comm, grad, p2p_tenosr)
            return p2p_tenosr

        infos = zip(input_tensor_grad, self.recv_comms, self.tensor_shapes, self.tensor_dtypes)

        out = list(map(f, infos))
        return out if len(out) > 0 else None

    def compile(self) -> List[CompilePPCollectiveOperator]:
        out: List[CompilePPCollectiveOperator] = []
        for comm in self.recv_comms:
            cur_mesh, peer_mesh = comm.cur_mesh, comm.peer_mesh
            coordinate = (cur_mesh.mesh == dist.get_rank()).nonzero(as_tuple=True)
            peer_rank = peer_mesh.mesh[coordinate].item()

            out.append(CompilePPCollectiveOperator(kind=CompilePPCollectiveKind.SEND, dst=peer_rank, is_backward=True))
            out.append(CompilePPCollectiveOperator(kind=CompilePPCollectiveKind.RECV, src=peer_rank))
        return out


@dataclass
class FORWARD_STEP(BaseInstruction):
    model: Optional[Union[torch.nn.Module, PipeModule]] = None
    is_pp_first_stage: bool = False
    is_pp_last_stage: bool = False
    local_comm: List[CommPacket] = field(default_factory=list)
    p2p_comm: List[CommPacket] = field(default_factory=list)
    p2p_index_mapping: List[Tuple[int, int]] = field(default_factory=list)
    stage_id: int = 0
    batch_id: int = 0
    forward_only: bool = False

    @property
    def name(self):
        return "forward_step"

    def construct_input_args(self, p2p_tensors, local_inputs):
        """
        stage 0: a , c
        stage 1: b
        stage 2: dataloader

        stage 2: forward(c,b,dataloader,a)

        p2p_order: [(0, 2), (1, 0), (2, 0), (0, 0)]
        send_order: [(0, 0), (0, 2), (1, 0)]
        we assume that the p2p send is follow interge order

        we assume that the p2p will allways be args

        """
        if not isinstance(local_inputs, (Sequence, Dict)):
            local_inputs = [local_inputs]
        if not isinstance(p2p_tensors, list):
            p2p_tensors = [p2p_tensors]
        p2p_index_without_local = list(
            filter(lambda item: item.peer_stage_idx != self.stage_id, self.p2p_index_mapping)
        )
        p2p_send_order = sorted(p2p_index_without_local)
        local_input_mapping = list(filter(lambda item: item.peer_stage_idx == self.stage_id, self.p2p_index_mapping))

        args = []
        kwargs = {}
        ground_truth = []
        for item in self.p2p_index_mapping:
            if item.peer_stage_idx == self.stage_id:
                index = local_input_mapping.index(item)
                args.append(local_inputs[index])
            else:
                index = p2p_send_order.index(item)
                args.append(p2p_tensors[index])
        if isinstance(local_inputs, Sequence) and len(local_inputs) > 1:
            ground_truth.append(local_inputs[-1])
        elif isinstance(local_inputs, Dict) and "labels" in local_inputs:
            ground_truth.append(local_inputs["labels"])
        return args, kwargs, ground_truth

    @dummy_p2p
    def run(self, input_tensor, kwargs):
        """Forward step for passed-in model.

        If first stage, input tensor is obtained from data_iterator, otherwise
        passed-in input_tensor is used.

        Returns output tensor."""

        data_iterator, forward_data_store, autocast_dtype, enable_autocast = (
            kwargs["data_iterator"],
            kwargs["forward_data_store"],
            kwargs["autocast_dtype"],
            kwargs["enable_autocast"],
        )
        if enable_autocast:
            context_manager = torch.autocast("cuda", dtype=autocast_dtype)
        else:
            context_manager = contextlib.nullcontext()
        with context_manager:

            def prepare_data():
                local_tensors = []
                ground_truth = []
                if data_iterator is not None:
                    if isinstance(data_iterator, list):
                        if len(data_iterator) > self.batch_id:
                            local_tensors = data_iterator[self.batch_id]
                    else:
                        local_tensors = next(data_iterator)
                if isinstance(local_tensors, Sequence) and len(local_tensors) > 1:
                    ground_truth.append(local_tensors[-1])
                elif isinstance(local_tensors, Dict) and "labels" in local_tensors:
                    ground_truth.append(local_tensors["labels"])
                return input_tensor, local_tensors, ground_truth

            builder.user_data["prepare_data_fn"] = prepare_data
            builder.user_data["batch_id"] = self.batch_id
            builder.user_data["p2p_tensors"] = input_tensor
            p2p_tensor, local_tensors, ground_truth = registed_functions["vescale_1f1b_pre_forward_data"]()
            builder.user_data["ground_truth"] = ground_truth
            output_tensor = registed_functions["vescale_1f1b_forward"](p2p_tensor, local_tensors)
            builder.user_data["output_tensor"] = output_tensor

        if self.is_pp_last_stage:
            # update status machine
            output_tensor, loss_tensor = registed_functions["vescale_1f1b_loss_fn"]()
            forward_data_store.append((output_tensor, loss_tensor))
            if builder.loss_fn is None:
                return output_tensor
            else:
                return loss_tensor

        return output_tensor


@dataclass
class BACKWARD_STEP(BaseInstruction):
    @property
    def name(self):
        return "backward step"

    @dummy_p2p
    def run(self, input_tensor, output_tensor, output_tensor_grad, kwargs):
        """Backward step through passed-in output tensor.

        If last stage, output_tensor_grad is None, otherwise gradient of loss
        with respect to stage's output tensor.

        Returns gradient of loss with respect to input tensor (None if first
        stage)."""

        grad_scaler = kwargs["grad_scaler"]
        deallocate_pipeline_outputs = kwargs["deallocate_pipeline_outputs"]
        # NOTE: This code currently can handle at most one skip connection. It
        # needs to be modified slightly to support arbitrary numbers of skip
        # connections.

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
        if len(output_tensor_grad) == 0 and grad_scaler is not None:
            output_tensor = grad_scaler(loss_value)

        if deallocate_pipeline_outputs:
            assert 0
        else:
            torch.autograd.backward(loss_value, grad_tensors=output_tensor_grad[0])

        # Collect the grad of the input_tensor.
        input_tensor_grad = [None]
        if input_tensor is not None:
            input_tensor_grad = []
            for x in input_tensor:
                if x is None:
                    input_tensor_grad.append(None)
                else:
                    input_tensor_grad.append(x.grad)

        if unwrap_input_tensor_grad:
            input_tensor_grad = input_tensor_grad[0]

        return input_tensor_grad


@dataclass
class DEALLOCATE_OUTPUT_TENSOR(BaseInstruction):
    deallocate_out: bool = True

    @property
    def name(self):
        return "deallocate output tensor "

    @dummy_p2p
    def run(self, out, deallocate_pipeline_outputs=False):
        """Pseudo-deallocate (i.e., set to scalar) the output tensor's '.data' field.

        This method should be called right after the output tensor has been
        sent to the next pipeline stage. At this point, the output tensor is
        only useful for its '.grad_fn' field, and not its '.data'.
        """
        # TODO: support DTensor
        if (out is None) or (not deallocate_pipeline_outputs):
            return

        def f(out):
            assert isinstance(out, [torch.Tensor, DTensor]), f"expected Tensor, found {type(out).__name__}."
            assert out._base is None, "counter-productive to free a view of another tensor."
            if isinstance(out, [torch.Tensor, DTensor]):
                out._local_tensor.data = torch.empty(
                    (1,),
                    device=out.device,
                    dtype=out.dtype,
                )
            else:
                out.data = torch.empty(
                    (1,),
                    device=out.device,
                    dtype=out.dtype,
                )

        if not isinstance(out, list):
            for o in out:
                f(o)
        else:
            f(out)


@dataclass
class APPEND_INPUTS(BaseInstruction):
    @property
    def name(self):
        return "append_inputs"

    @dummy_p2p
    def run(self, input_tensors, input_tensor):
        input_tensors.append(input_tensor)


@dataclass
class APPEND_OUTPUTS(BaseInstruction):
    @property
    def name(self):
        return "append_outputs"

    @dummy_p2p
    def run(self, output_tensors, output_tensor):
        output_tensors.append(output_tensor)


@dataclass
class POP_INPUT(BaseInstruction):
    @property
    def name(self):
        return "pop input"

    @dummy_p2p
    def run(self, input_tensors):
        input_tensor = input_tensors.pop(0)
        return input_tensor


@dataclass
class POP_OUTPUT(BaseInstruction):
    @property
    def name(self):
        return "pop output"

    @dummy_p2p
    def run(self, output_tensors):
        output_tensor = output_tensors.pop(0)
        return output_tensor


class PipeDream(PipelineSchema):
    """
    generate pipedream schedule (a.k.a 1f1b)
    memory-efficient than gpipe
    """

    @property
    def name(self):
        return "1f1b"

    def _gen_schedule(self):
        """
        run forward then run backward
        the sequence timeline as show before
        d: device
        m: batches
        T: timeline

        T (m,d)   (m,d)   (m,d)
        - ------  ------  -------

        0 (0,0,F)
        1 (1,0,F) (0,1,F)
        2 (2,0,F) (1,1,F) (0,2,F)
        3                 (0,2,B)
        4         (0,1,B) (1,2,F)
        5 (0,0,B) (2,1,F) (1,2,B)
        6 (3,0,F) (1,1,B) (2,2,F)
        ...
        """
        m = self.batches
        d = self.num_mesh

        num_clock = (m + d - 1) * 2  # time todo flush
        schedules = [[None] * d for c in range(num_clock)]
        warmup_batches = [min(d - i - 1, m) for i in range(d)]
        remain_batches = [m - i for i in warmup_batches]
        next_fwd_batch_idx = [0 for _ in range(d)]
        next_bwd_batch_idx = [0 for _ in range(d)]

        self.warmup_batches = warmup_batches
        self.remain_batches = remain_batches

        new_timeline = list(range(d))
        """
        t_i|m
             0   1   2
        0    0   0   0
        1    0   0   0
        2    0   0   0
        3    0   0   1
        4    0   1   1
        5    1   1   1
        1f1b
        """
        bwd_done_idx = np.zeros(shape=[num_clock, d], dtype=np.int32)
        # warm-up steps
        for i in range(d):
            for k in range(warmup_batches[i]):
                t_i = new_timeline[i]
                schedules[t_i][i] = Status(batch_idx=next_fwd_batch_idx[i], stage_id=i, f_b="F", stg="WUp", k=k)
                new_timeline[i] += 1  # self add for new timeline
                next_fwd_batch_idx[i] += 1  # do next micro batch

        # run 1f1b steps
        for i in reversed(range(d)):
            for idx in range(remain_batches[i]):
                # do forward
                t_i = new_timeline[i]
                schedules[t_i][i] = Status(batch_idx=next_fwd_batch_idx[i], stage_id=i, f_b="F", stg="1f1b", k=idx)
                next_fwd_batch_idx[i] += 1
                bwd_done_idx[t_i][i] = next_bwd_batch_idx[i]
                t_i += 1

                # do backward
                if i + 1 < d:
                    while bwd_done_idx[t_i][i + 1] < next_bwd_batch_idx[i]:
                        # if the stage 2 is done, the stage i must be equal 0
                        assert bwd_done_idx[t_i - 1][i] == next_bwd_batch_idx[i]
                        bwd_done_idx[t_i][i] = bwd_done_idx[t_i - 1][i]
                        t_i = t_i + 1

                if idx == remain_batches[i] - 1:  # last iterator
                    schedules[t_i][i] = Status(
                        batch_idx=next_bwd_batch_idx[i], stage_id=i, f_b="B", stg="1f1b-l", k=idx
                    )
                else:
                    schedules[t_i][i] = Status(batch_idx=next_bwd_batch_idx[i], stage_id=i, f_b="B", stg="1f1b", k=idx)
                bwd_done_idx[t_i][i] = next_bwd_batch_idx[i]
                next_bwd_batch_idx[i] += 1
                new_timeline[i] = t_i + 1

        # run cool duwn
        for i in reversed(range(d)):
            for k in range(warmup_batches[i]):
                assert i + 1 < d
                t_i = new_timeline[i]
                while bwd_done_idx[t_i][i + 1] <= next_bwd_batch_idx[i]:
                    bwd_done_idx[t_i][i] = next_bwd_batch_idx[i]
                    t_i = t_i + 1
                schedules[t_i][i] = Status(batch_idx=next_bwd_batch_idx[i], stage_id=i, f_b="B", stg="CD", k=k)
                bwd_done_idx[t_i][i] = next_bwd_batch_idx[i]
                next_bwd_batch_idx[i] += 1
                new_timeline[i] = t_i + 1
            if i > 0:
                bwd_done_idx[new_timeline[i] : num_clock, i] = m
        return schedules


class OneFOneBInstrcutionGenerator(InstructionGenerator):
    def __init__(
        self,
        deps: StageDeps,
        meshes: List[DeviceMesh],
        batches: int,
        default_shape: Optional[Shape] = None,
        default_dtype: Optional[torch.dtype] = None,
        batch_shape_lists: Optional[List[Dict[int, Shape]]] = None,
        batch_dtype_lists: Optional[List[Dict[int, torch.dtype]]] = None,
        forward_only: bool = False,
    ):
        forward_only = True if not torch.is_grad_enabled() else forward_only
        super().__init__(
            deps=deps,
            meshes=meshes,
            batches=batches,
            default_shape=default_shape,
            default_dtype=default_dtype,
            batch_shape_lists=batch_shape_lists,
            batch_dtype_lists=batch_dtype_lists,
            forward_only=forward_only,
        )
        self.num_stage = len(meshes)
        self.schema = PipeDream(num_stage=self.num_stage, meshes=meshes, batches=self.batches)
        self.forward_only = forward_only

    def get_tensor_shape(self, microbatch_id, input_id):
        if self.batch_shape_lists:
            if input_id in self.batch_shape_lists[microbatch_id].keys():
                return self.batch_shape_lists[microbatch_id][input_id]
        return self.default_shape

    def get_tensor_dtype(self, microbatch_id, input_id):
        if self.batch_dtype_lists:
            if input_id in self.batch_dtype_lists[microbatch_id].keys():
                return self.batch_dtype_lists[microbatch_id][input_id]
        return self.default_dtype

    def get_tensor_shapes_and_dtypes(self, comm_packages: List[CommPacket], microbatch_id: int):
        def get_shape_or_dtype(f: Callable, package: CommPacket):
            return f(microbatch_id, package.input_id)

        shapes = map(partial(get_shape_or_dtype, self.get_tensor_shape), comm_packages)
        dtypes = map(partial(get_shape_or_dtype, self.get_tensor_dtype), comm_packages)
        return list(shapes), list(dtypes)

    # call by pipe emitter
    def gen_instruction(self):
        # If the context is torch.no_grad(), only execute forward
        _forward_only = self.forward_only
        if not torch.is_grad_enabled():
            self.forward_only = True

        schedules = self.schema.schedules
        self.instruction_list = [[] for _ in range(self.num_stage)]
        stack = defaultdict(list)  # for 1f1b
        first_time_1f1b = [True] * self.num_stage
        for clk, stages_schemas in enumerate(schedules):
            for s, schema in enumerate(stages_schemas):
                send_comms = self.deps.get_send_comms(s)
                recv_comms = self.deps.get_recv_comms(s)
                p2p_index_mapping = self.deps.mapping[s]
                cur_model = self.deps.get_current_model(s)
                local_comm = self.deps.get_local_comms(s)
                is_pp_first_stage = self.deps.is_pipeline_first_stage(s)
                is_pp_last_stage = self.deps.is_pipeline_last_stage(s)
                if isinstance(cur_model, Sequence):
                    assert self.num_chunk == 1, "1f1b support model chunk is 1."
                    cur_model = cur_model[0]
                # batch size, stage idx, forward backward,
                if schema:
                    b_idx = schema.batch_idx
                    stg = schema.stg
                    if "WUp" in stg:  # warmup stage
                        # recv forward
                        recv_shapes, recv_dtypes = self.get_tensor_shapes_and_dtypes(recv_comms, b_idx)
                        send_shapes, _ = self.get_tensor_shapes_and_dtypes(send_comms, b_idx)
                        self._set_inst(
                            RECV_FORWARD(
                                comm_packages=recv_comms,
                                tensor_shapes=recv_shapes,
                                tensor_dtypes=recv_dtypes,
                                batch_id=b_idx,
                                debug="warm-up",
                            ),
                            s,
                        )
                        self._set_inst(
                            FORWARD_STEP(
                                model=cur_model,
                                is_pp_first_stage=is_pp_first_stage,
                                is_pp_last_stage=is_pp_last_stage,
                                local_comm=local_comm,
                                p2p_comm=recv_comms,
                                p2p_index_mapping=p2p_index_mapping,
                                stage_id=s,
                                batch_id=b_idx,
                                forward_only=self.forward_only,
                            ),
                            s,
                        )
                        self._set_inst(
                            SEND_FORWARD(
                                comm_packages=send_comms,
                                tensor_shapes=send_shapes,
                                batch_id=b_idx,
                            ),
                            s,
                        )

                        if not self.forward_only:
                            self._set_inst(APPEND_INPUTS(), s)
                            self._set_inst(APPEND_OUTPUTS(), s)
                            self._set_inst(DEALLOCATE_OUTPUT_TENSOR(), s)
                    elif "1f1b" in stg:  # 1f1b stage
                        cur_st = stack[s]
                        if len(cur_st) < 2:
                            cur_st.append(schema)  # lazy do
                        else:
                            raise RuntimeError("unknown schedule")

                        if len(cur_st) == 2:
                            if first_time_1f1b[s]:
                                recv_shapes, recv_dtypes = self.get_tensor_shapes_and_dtypes(recv_comms, b_idx)
                                # before run 1f1b
                                self._set_inst(
                                    RECV_FORWARD(
                                        comm_packages=recv_comms,
                                        tensor_shapes=recv_shapes,
                                        tensor_dtypes=recv_dtypes,
                                        batch_id=b_idx,
                                        debug="first 1f1b",
                                    ),
                                    s,
                                )
                                first_time_1f1b[s] = False
                            fwd = cur_st[0]
                            bwd = cur_st[1]
                            fw_b_idx = fwd.batch_idx
                            bw_b_idx = bwd.batch_idx
                            self._set_inst(
                                FORWARD_STEP(
                                    model=cur_model,
                                    is_pp_first_stage=is_pp_first_stage,
                                    is_pp_last_stage=is_pp_last_stage,
                                    local_comm=local_comm,
                                    p2p_comm=recv_comms,
                                    p2p_index_mapping=p2p_index_mapping,
                                    stage_id=s,
                                    batch_id=fw_b_idx,
                                    forward_only=self.forward_only,
                                ),
                                s,
                            )

                            if self.forward_only:
                                send_shapes, _ = self.get_tensor_shapes_and_dtypes(send_comms, fw_b_idx)
                                self._set_inst(
                                    SEND_FORWARD(
                                        comm_packages=send_comms, tensor_shapes=send_shapes, batch_id=fw_b_idx
                                    ),
                                    s,
                                )
                                last_iteration = fwd.k == (self.schema.remain_batches[s] - 1)
                                if not last_iteration:
                                    recv_shapes, recv_dtypes = self.get_tensor_shapes_and_dtypes(recv_comms, fw_b_idx)
                                    self._set_inst(
                                        RECV_FORWARD(
                                            comm_packages=recv_comms,
                                            tensor_shapes=recv_shapes,
                                            tensor_dtypes=recv_dtypes,
                                            batch_id=fw_b_idx,
                                            debug="last_1f1b",
                                        ),
                                        s,
                                    )
                                stack[s].clear()
                            else:
                                send_shapes, send_dtypes = self.get_tensor_shapes_and_dtypes(send_comms, bw_b_idx)
                                self._set_inst(
                                    SEND_FORWARD_RECV_BACKWARD(
                                        comm_packages=send_comms,
                                        tensor_shapes=send_shapes,
                                        tensor_dtypes=send_dtypes,
                                        send_batch_id=fw_b_idx,
                                        recv_batch_id=bw_b_idx,
                                    ),
                                    s,
                                )
                                self._set_inst(APPEND_INPUTS(), s)
                                self._set_inst(APPEND_OUTPUTS(), s)
                                self._set_inst(DEALLOCATE_OUTPUT_TENSOR(), s)
                                self._set_inst(POP_INPUT(), s)
                                self._set_inst(POP_OUTPUT(), s)
                                self._set_inst(BACKWARD_STEP(), s)
                                self._set_inst(DEALLOCATE_OUTPUT_TENSOR(deallocate_out=False), s)

                                if stg == "1f1b-l":
                                    recv_shapes, recv_dtypes = self.get_tensor_shapes_and_dtypes(recv_comms, bw_b_idx)
                                    self._set_inst(SEND_BACKWARD(recv_comms=recv_comms, tensor_shapes=recv_shapes), s)
                                else:
                                    recv_shapes, recv_dtypes = self.get_tensor_shapes_and_dtypes(recv_comms, fw_b_idx)
                                    self._set_inst(
                                        SEND_BACKWARD_RECV_FORWARD(
                                            recv_comms=recv_comms, tensor_shapes=recv_shapes, tensor_dtypes=recv_dtypes
                                        ),
                                        s,
                                    )
                                stack[s].clear()  # save for next
                        else:  # 1f1b do f
                            continue
                    elif stg == "CD":  # cool down stage
                        if not self.forward_only:
                            self._set_inst(POP_INPUT(), s)
                            self._set_inst(POP_OUTPUT(), s)
                            # recv backward

                            send_shapes, send_dtypes = self.get_tensor_shapes_and_dtypes(send_comms, b_idx)
                            self._set_inst(
                                RECV_BACKWARD(
                                    comm_packages=send_comms, tensor_shapes=send_shapes, tensor_dtypes=send_dtypes
                                ),
                                s,
                            )
                            # backward step
                            self._set_inst(BACKWARD_STEP(), s)
                            # deallocate input, output
                            self._set_inst(DEALLOCATE_OUTPUT_TENSOR(), s)
                            self._set_inst(DEALLOCATE_OUTPUT_TENSOR(deallocate_out=False), s)
                            # send backward
                            recv_shapes, recv_dtypes = self.get_tensor_shapes_and_dtypes(recv_comms, b_idx)
                            self._set_inst(SEND_BACKWARD(recv_comms=recv_comms, tensor_shapes=recv_shapes), s)
                else:  # bubble
                    # TODO
                    # do any other
                    continue
        self.gen_instruction_str_list()

        # restore original self.forward_only if the current context manager is torch.no_grad()
        if not torch.is_grad_enabled():
            self.forward_only = _forward_only

        return self.instruction_list

    def gen_instruction_str_list(self):
        instruction_lists = self.instruction_list
        stage_strs = defaultdict(str)
        for stage_id, instruction_list in enumerate(instruction_lists):
            cur_stage_str = stage_strs[stage_id]
            for inst in instruction_list:
                cur_stage_str += f"{VESACLE_INSTRUCTION_MAPPING[type(inst)]},"
            cur_stage_str = cur_stage_str[:-1]
            stage_strs[stage_id] = cur_stage_str
        builder.build_from_dict(stage_strs)

    @manage_dump_file
    def execute(
        self,
        stage_id,
        autocast_dtype=torch.float,
        enable_autocast=False,
        grad_scaler=None,
        deallocate_pipeline_outputs=False,
    ):
        builder.constant_data["autocast_dtype"] = autocast_dtype
        builder.constant_data["enable_autocast"] = enable_autocast
        builder.constant_data["grad_scaler"] = grad_scaler
        builder.constant_data["deallocate_pipeline_outputs"] = deallocate_pipeline_outputs

        user_data = builder.user_data
        user_data["input_tensors"] = []
        user_data["output_tensors"] = []
        user_data["input_tensor"] = None  # engine need to maintain the dataflow
        user_data["output_tensor"] = None  # engine need to maintian the output flow
        user_data["output_tensor_grad"] = None
        user_data["input_tensor_grad"] = None
        user_data["forward_data_store"] = []

        instruction_list = self.get_instruction_list(stage_id)
        builder.stage_id = stage_id
        builder_instruction_list = builder.global_instructions_funcs[stage_id]

        _forward_only = self.forward_only
        if not torch.is_grad_enabled():
            self.forward_only = True

        for inst, fn in zip(instruction_list, builder_instruction_list):
            user_data["inst"] = inst
            fn()

        # restore original self.forward_only if the current context manager is torch.no_grad()
        if not torch.is_grad_enabled():
            self.forward_only = _forward_only

        return user_data["forward_data_store"]


@register_instruction(name="vescale_1f1b_recv_forward")
def vescale_recv_forward():
    user_data = builder.user_data
    inst = user_data["inst"]
    input_tensor = inst.run()
    builder.user_data["input_tensor"] = input_tensor
    return input_tensor


@register_instruction(name="vescale_1f1b_recv_backward")
def vescale_recv_backward():
    user_data = builder.user_data
    inst = user_data["inst"]
    output_tensor_grad = inst.run()
    builder.user_data["output_tensor_grad"] = output_tensor_grad
    return output_tensor_grad


@register_instruction(name="vescale_1f1b_send_forward")
def vescale_send_forward():
    user_data = builder.user_data
    inst = user_data["inst"]
    output_tensor = user_data["output_tensor"]
    inst.run(output_tensors=output_tensor)


@register_instruction(name="vescale_1f1b_send_backward")
def vescale_send_backward():
    user_data = builder.user_data
    inst = user_data["inst"]
    input_tensor_grad = user_data["input_tensor_grad"]
    inst.run(input_tensor_grad=input_tensor_grad)


@register_instruction(name="vescale_1f1b_send_forward_recv_backward")
def vescale_send_forward_recv_backward():
    user_data = builder.user_data
    inst = user_data["inst"]
    output_tensor = user_data["output_tensor"]
    output_tensor_grad = inst.run(output_tensors=output_tensor)
    builder.user_data["output_tensor_grad"] = output_tensor_grad


@register_instruction(name="vescale_1f1b_send_backward_recv_forward")
def vescale_send_backward_recv_forward():
    user_data = builder.user_data
    inst = user_data["inst"]
    input_tensor_grad = user_data["input_tensor_grad"]
    with torch.no_grad():
        input_tensor = inst.run(input_tensor_grad=input_tensor_grad)
    builder.user_data["input_tensor"] = input_tensor


@register_instruction(name="vescale_1f1b_forward_step")
@ndtimer(FORWARD_COMPUTE)
def vescale_forward_step():
    user_data = builder.user_data
    constant_data = builder.constant_data
    inst = user_data["inst"]
    input_tensor = user_data["input_tensor"]
    forward_data_store = user_data["forward_data_store"]
    autocast_dtype = constant_data["autocast_dtype"]
    builder.model = inst.model
    if not autocast_dtype:
        autocast_dtype = torch.float32
    enable_autocast = constant_data["enable_autocast"]
    if not enable_autocast:
        enable_autocast = False
    if forward_data_store is None:
        forward_data_store = []
    forward_args = {
        "data_iterator": builder.dataloader,
        "forward_data_store": forward_data_store,
        "autocast_dtype": autocast_dtype,
        "enable_autocast": enable_autocast,
    }
    output_tensor = inst.run(input_tensor=input_tensor, kwargs=forward_args)
    builder.user_data["output_tensor"] = output_tensor
    builder.user_data["forward_data_store"] = forward_data_store


@register_instruction(name="vescale_1f1b_loss_fn")
def loss_fn():
    user_data = builder.user_data
    output_tensor = user_data["output_tensor"]
    loss_func = builder.loss_fn
    if loss_func is None or output_tensor is None:
        return output_tensor, None
    temp_tensor = output_tensor
    ground_truth = user_data["ground_truth"]
    # signature provides a more uniform way to parse callable arguments, including lambda functions
    args_spec = signature(loss_func)
    args_len = len(args_spec.parameters.keys())
    if args_len == 1:
        output_tensor = loss_func(output_tensor)
    else:
        ground_truth = builder.user_data["ground_truth"]
        loss_fn_inputs = [output_tensor] + ground_truth
        output_tensor = loss_func(*loss_fn_inputs)
        assert args_len == len(loss_fn_inputs), "Mismatch of loss function #args and #actual inputs!"
    return temp_tensor, output_tensor


@register_instruction(name="vescale_1f1b_pre_forward_data")
def prepare_data():
    user_data = builder.user_data
    return user_data["prepare_data_fn"]()


@register_instruction(name="vescale_1f1b_forward")
def forward_fn(p2p_input, local_input):
    if isinstance(builder.model, PipeModule):
        return builder.model(p2p_input, local_input, chunk_id=0)
    else:

        def _feed_input(model, data):
            if isinstance(data, Sequence):
                return model(*data)
            elif isinstance(data, Dict):
                return model(**data)
            else:
                return model(data)

        if p2p_input is not None:
            return _feed_input(builder.model, p2p_input)
        else:
            return _feed_input(builder.model, local_input)


@register_instruction(name="vescale_1f1b_backward_step")
@ndtimer(BACKWARD_COMPUTE)
def vescale_backward_step():
    constant_data = builder.constant_data
    grad_scaler = constant_data["grad_scaler"]
    deallocate_pipeline_outputs = constant_data["deallocate_pipeline_outputs"]
    backward_args = {
        "grad_scaler": grad_scaler,
        "deallocate_pipeline_outputs": deallocate_pipeline_outputs,
    }

    user_data = builder.user_data
    input_tensor = user_data["input_tensor"]
    output_tensor = user_data["output_tensor"]
    output_tensor_grad = user_data["output_tensor_grad"]
    inst = user_data["inst"]

    input_tensor_grad = inst.run(
        input_tensor=input_tensor,
        output_tensor=output_tensor,
        output_tensor_grad=output_tensor_grad,
        kwargs=backward_args,
    )
    builder.user_data["input_tensor_grad"] = input_tensor_grad


@register_instruction(name="vescale_1f1b_pop_input")
def vescale_1f1b_pop_input():
    user_data = builder.user_data
    inst = user_data["inst"]
    input_tensors = user_data["input_tensors"]
    input_tensor = inst.run(input_tensors=input_tensors)
    builder.user_data["input_tensor"] = input_tensor


@register_instruction(name="vescale_1f1b_pop_output")
def vescale_1f1b_pop_output():
    user_data = builder.user_data
    inst = user_data["inst"]
    output_tensors = user_data["output_tensors"]
    output_tensor = inst.run(output_tensors=output_tensors)
    builder.user_data["output_tensor"] = output_tensor


@register_instruction(name="vescale_1f1b_append_inputs")
def vescale_1f1b_append_inputs():
    user_data = builder.user_data
    inst = user_data["inst"]
    input_tensors = user_data["input_tensors"]
    input_tensor = user_data["input_tensor"]
    if input_tensors is None:
        input_tensors = []
    inst.run(input_tensors=input_tensors, input_tensor=input_tensor)
    user_data["input_tensors"] = input_tensors


@register_instruction(name="vescale_1f1b_append_outputs")
def vescale_1f1b_append_outputs():
    user_data = builder.user_data
    inst = user_data["inst"]
    output_tensors = user_data["output_tensors"]
    output_tensor = user_data["output_tensor"]
    if output_tensors is None:
        output_tensors = []
    inst.run(output_tensors=output_tensors, output_tensor=output_tensor)
    user_data["output_tensors"] = output_tensors


@register_instruction(name="vescale_1f1b_deallocate_output_tensor")
def vescale_1f1b_deallocate_output_tensor():
    user_data = builder.user_data
    inst = user_data["inst"]
    const_data = builder.constant_data
    deallocate_pipeline_outputs = const_data["deallocate_pipeline_outputs"]
    if inst.deallocate_out:
        output_tensor = user_data["output_tensor"]
        inst.run(output_tensor, deallocate_pipeline_outputs=deallocate_pipeline_outputs)
    else:
        input_tensor = user_data["input_tensor"]
        if input_tensor and input_tensor[0] is not None:
            input_tensor[0].grad = None
        inst.run(input_tensor, deallocate_pipeline_outputs=deallocate_pipeline_outputs)


VESACLE_INSTRUCTION_MAPPING = {
    RECV_FORWARD: "vescale_1f1b_recv_forward",
    RECV_BACKWARD: "vescale_1f1b_recv_backward",
    SEND_FORWARD: "vescale_1f1b_send_forward",
    SEND_BACKWARD: "vescale_1f1b_send_backward",
    SEND_FORWARD_RECV_BACKWARD: "vescale_1f1b_send_forward_recv_backward",
    SEND_BACKWARD_RECV_FORWARD: "vescale_1f1b_send_backward_recv_forward",
    FORWARD_STEP: "vescale_1f1b_forward_step",
    BACKWARD_STEP: "vescale_1f1b_backward_step",
    POP_INPUT: "vescale_1f1b_pop_input",
    POP_OUTPUT: "vescale_1f1b_pop_output",
    APPEND_INPUTS: "vescale_1f1b_append_inputs",
    APPEND_OUTPUTS: "vescale_1f1b_append_outputs",
    DEALLOCATE_OUTPUT_TENSOR: "vescale_1f1b_deallocate_output_tensor",
}
