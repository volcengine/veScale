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

from vescale.pipe._schedules.instruction_base import (
    PipelineSchema,
    Status,
    Shape,
    InstructionGenerator,
    StageDeps,
    BaseInstruction,
    CommPacket,
    VESCALE_INTRUCTION_BUILDER as builder,
    register_instruction,
    registed_functions,
)
import contextlib
from dataclasses import dataclass, field
from vescale.dtensor.dtensor import DTensor
import torch
from collections import defaultdict
from inspect import signature
import numpy as np
from vescale.dtensor.device_mesh import DeviceMesh
from typing import List, Sequence, Optional, Dict, Union, Callable
from functools import partial
from vescale.dtensor._diff import dummy_p2p, manage_dump_file
from vescale.pipe.p2p_communication import (
    recv_forward,
    drain_send_reqs,
    drain_recv_reqs,
    send_forward_backward_recv_forward_backward,
    send_forward_recv_forward,
    send_backward_recv_backward,
)
from vescale.model.base_gpt.utils import switch_dtensor


@dataclass
class RECV_FORWARD(BaseInstruction):
    comm_packages: List[CommPacket] = field(default_factory=list)
    tensor_shapes: Union[List[Shape], Shape] = field(default_factory=list)
    tensor_dtypes: Union[List[torch.dtype], torch.dtype] = field(default_factory=list)
    batch_p2p_comm: bool = True
    batch_id: Optional[int] = None
    is_pp_first_stage: bool = False
    debug: str = ""

    @property
    def name(self):
        return "recv_forward"

    @dummy_p2p
    def run(self) -> List:
        if self.is_pp_first_stage:
            return None

        def f(info):
            comm, shape, dtype = info
            return recv_forward(
                tensor_shape=shape,
                recv_dtype=dtype,
                current_device_mesh=comm.cur_mesh,
                peer_device_mesh=comm.peer_mesh,
                batch_p2p_comm=self.batch_p2p_comm,
            )

        infos = zip(self.comm_packages, self.tensor_shapes, self.tensor_dtypes)
        out = list(map(f, infos))
        return out if len(out) > 0 else None


@dataclass
class WAIT_FWD(BaseInstruction):
    @property
    def name(self):
        return "wait_forward"

    @dummy_p2p
    def run(self, fwd_wait_handles: Optional[Sequence]):
        if fwd_wait_handles is not None:
            for req in fwd_wait_handles:
                req.wait()


@dataclass
class DRAIN_SEND_REQS(BaseInstruction):
    @property
    def name(self):
        return "drain_send_reqs"

    @dummy_p2p
    def run(self):
        drain_send_reqs()


@dataclass
class DRAIN_RECV_REQS(BaseInstruction):
    drain_type: str = "all"
    check_bwd_wait: bool = False

    @property
    def name(self):
        return "drain_recv_reqs"

    @dummy_p2p
    def run(self, bwd_wait_handles: Optional[Sequence]):
        if self.check_bwd_wait:
            if bwd_wait_handles is not None:
                drain_recv_reqs(self.drain_type)
        else:
            drain_recv_reqs(self.drain_type)


@dataclass
class DEALLOCATE_OUTPUT_TENSOR(BaseInstruction):
    @property
    def name(self):
        return "deallocate tensor"

    @dummy_p2p
    def run(self, output_tensor, deallocate_pipeline_outputs):
        def deallocate(output_tensor):
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

        if not isinstance(output_tensor, Sequence):
            output_tensor = [output_tensor]
        map(deallocate, output_tensor)


@dataclass
class APPEND_INPUTS(BaseInstruction):
    chunk: int = 0

    @property
    def name(self):
        return "append inputs"

    @dummy_p2p
    def run(self, input_tensor, input_tensors):
        input_tensors[self.chunk].append(input_tensor)


@dataclass
class APPEND_GRADS(BaseInstruction):
    chunk: int = 0

    @property
    def name(self):
        return "append grads"

    @dummy_p2p
    def run(self, output_tensor_grad, output_tensor_grads):
        output_tensor_grads[self.chunk].append(output_tensor_grad)


@dataclass
class SEND_FORWARD_BACKWARD_RECV_FORWARD_BACKWARD(BaseInstruction):
    recv_prev: bool = False
    recv_next: bool = False
    send_comms: List[CommPacket] = field(default_factory=list)
    recv_comms: List[CommPacket] = field(default_factory=list)
    recv_shapes: List[Shape] = field(default_factory=list)
    recv_dtypes: List[torch.dtype] = field(default_factory=list)
    batch_p2p_comm: bool = True
    debug: str = ""

    @property
    def name(self):
        return "send forward backward recv forward backward"

    @dummy_p2p
    def run(self, output_tensor, input_tensor_grad):
        if not isinstance(output_tensor, Sequence):
            output_tensor = [output_tensor]
        if not isinstance(input_tensor_grad, Sequence):
            input_tensor_grad = [input_tensor_grad]

        def f(info):
            output_tensor, input_tensor_grad, recv_comm, send_comm, tensor_shape, dtype = info
            if isinstance(output_tensor, DTensor):
                output_tensor = output_tensor._local_tensor
            if isinstance(input_tensor_grad, DTensor):
                input_tensor_grad = input_tensor_grad._local_tensor

            input_tensor, output_tensor_grad = send_forward_backward_recv_forward_backward(
                output_tensor=output_tensor,
                input_tensor_grad=input_tensor_grad,
                recv_prev=self.recv_prev,
                recv_next=self.recv_next,
                current_device_mesh=send_comm.cur_mesh,
                prev_device_mesh=recv_comm.peer_mesh,
                next_device_mesh=send_comm.peer_mesh,
                tensor_shape=tensor_shape,
                recv_dtype=dtype,
                batch_p2p_comm=self.batch_p2p_comm,
            )
            return input_tensor, output_tensor_grad

        zipped_data = list(
            zip(
                output_tensor,
                input_tensor_grad,
                self.recv_comms,
                self.send_comms,
                self.recv_shapes,
                self.recv_dtypes,
            )
        )

        outputs = list(map(f, zipped_data))

        if len(outputs) > 1:
            if self.overlap_p2p_comm:
                out = [x[0] for x in outputs]
                handle = [x[1] for x in outputs]
                return out, handle
            else:
                return outputs
        else:
            return outputs[0]


@dataclass
class SEND_FORWARD_RECV_FORWARD(BaseInstruction):
    recv_prev: bool = False
    send_shapes: List[Shape] = field(default_factory=list)
    send_tensor_shapes_unpad: List[Shape] = field(default_factory=list)
    send_dtypes: List[torch.dtype] = field(default_factory=list)
    batch_p2p_comm: bool = True
    overlap_p2p_comm: bool = False
    send_comms: List[CommPacket] = field(default_factory=list)
    recv_comms: List[CommPacket] = field(default_factory=list)
    microbatch_id: int = 0
    debug: str = ""

    @property
    def name(self):
        return "send forward recv forward"

    @dummy_p2p
    def run(self, output_tensor):
        if not isinstance(output_tensor, Sequence):
            output_tensor = [output_tensor]

        def f(info):
            output_tensor, recv_comm, send_comm, tensor_shape, tensor_shape_unpad, dtype = info
            if isinstance(output_tensor, DTensor):
                output_tensor = output_tensor._local_tensor
            output = send_forward_recv_forward(
                output_tensor,
                recv_prev=self.recv_prev,
                tensor_shape=tensor_shape,
                send_tensor_shape_unpad=tensor_shape_unpad,
                overlap_p2p_comm=self.overlap_p2p_comm,
                batch_p2p_comm=self.batch_p2p_comm,
                recv_dtype=dtype,
                current_device_mesh=send_comm.cur_mesh,
                prev_device_mesh=recv_comm.peer_mesh,
                next_device_mesh=send_comm.peer_mesh,
            )
            return output

        zipped_data = list(
            zip(
                output_tensor,
                self.recv_comms,
                self.send_comms,
                self.send_shapes,
                self.send_tensor_shapes_unpad,
                self.send_dtypes,
            )
        )

        outputs = list(map(f, zipped_data))

        if len(outputs) > 1:
            if self.overlap_p2p_comm:
                out = [x[0] for x in outputs]
                handle = [x[1] for x in outputs]
                return out, handle
            else:
                return outputs
        else:
            return outputs[0]


@dataclass
class SEND_BACKWARD_RECV_BACKWARD(BaseInstruction):
    recv_next: bool = False
    send_shapes: List[Shape] = field(default_factory=list)
    send_tensor_shapes_unpad: List[Shape] = field(default_factory=list)
    send_dtypes: List[torch.dtype] = field(default_factory=list)
    batch_p2p_comm: bool = True
    overlap_p2p_comm: bool = False
    send_comms: List[CommPacket] = field(default_factory=list)
    recv_comms: List[CommPacket] = field(default_factory=list)
    debug: str = ""

    @property
    def name(self):
        return "send backward recv backward"

    @dummy_p2p
    def run(self, input_tensor_grad):
        if not isinstance(input_tensor_grad, Sequence):
            input_tensor_grad = [input_tensor_grad]

        def f(info):
            input_tensor_grad, recv_comm, send_comm, tensor_shape, tensor_shape_unpad, dtype = info
            if isinstance(input_tensor_grad, DTensor):
                input_tensor_grad = input_tensor_grad._local_tensor
            output = send_backward_recv_backward(
                input_tensor_grad,
                recv_next=self.recv_next,
                tensor_shape=tensor_shape,
                send_tensor_shape_unpad=tensor_shape_unpad,
                overlap_p2p_comm=self.overlap_p2p_comm,
                batch_p2p_comm=self.batch_p2p_comm,
                recv_dtype=dtype,
                current_device_mesh=send_comm.cur_mesh,
                prev_device_mesh=recv_comm.peer_mesh,
                next_device_mesh=send_comm.peer_mesh,
            )
            return output

        zipped_data = list(
            zip(
                input_tensor_grad,
                self.recv_comms,
                self.send_comms,
                self.send_shapes,
                self.send_tensor_shapes_unpad,
                self.send_dtypes,
            )
        )

        output = list(map(f, zipped_data))

        if len(output) > 1:
            if self.overlap_p2p_comm:
                return [x[0] for x in output], [x[1] for x in output]
            else:
                return output
        else:
            return output[0]


@dataclass
class SET_INPUTGRAD_TO_NONE(BaseInstruction):
    @property
    def name(self):
        return "set inputgrad to none"

    @dummy_p2p
    def run(self):
        return None


@dataclass
class SET_OUTPUT_TO_NONE(BaseInstruction):
    @property
    def name(self):
        return "set output to none"

    @dummy_p2p
    def run(self):
        return None


@dataclass
class BWD(BaseInstruction):
    is_vpp_last_stage: bool = False
    last_microbatch_for_model_chunk: bool = False
    grad_sync_chunk_id: int = 0
    grad_sync_microbatch_id: int = 0
    model_chunk_id: int = 0
    microbatch_id: int = 0
    debug: str = ""

    @property
    def name(self):
        return "backward"

    def backward_step(
        self,
        input_tensor,
        output_tensor,
        output_tensor_grad,
        grad_scaler=None,
        deallocate_pipeline_outputs=False,
    ):
        """Backward step through passed-in output tensor.

        If last stage, output_tensor_grad is None, otherwise gradient of loss
        with respect to stage's output tensor.

        Returns gradient of loss with respect to input tensor (None if first
        stage)."""

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
        if output_tensor_grad[0] is None and grad_scaler is not None:
            loss_value = grad_scaler(loss_value)
        # FIXME: For virtual pipeline, there may exist frozen layer without grad;
        # Need to verify if this solution is correct
        if not loss_value.requires_grad:
            return None

        model_chunk_id = builder.user_data["model_chunk_id"]
        model = builder.model[model_chunk_id]
        if deallocate_pipeline_outputs:
            assert 0
        else:
            switch_dtensor(torch.autograd.backward)(loss_value, grad_tensors=output_tensor_grad[0])

        model_chunk_id = builder.user_data["model_chunk_id"]
        model = builder.model[model_chunk_id]

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

    @dummy_p2p
    def run(
        self,
        input_tensors,
        output_tensors,
        output_tensor_grads,
        grad_sync_func,
        synchronized_model_chunks,
        kwargs: dict,
    ):
        grad_scaler, model, deallocate_pipeline_outputs = (
            kwargs["grad_scaler"],
            kwargs["model"],
            kwargs["deallocate_pipeline_outputs"],
        )
        if self.is_vpp_last_stage:
            if len(output_tensor_grads[self.model_chunk_id]) == 0:
                output_tensor_grads[self.model_chunk_id].append(None)
        input_tensor = input_tensors[self.model_chunk_id].pop(0)
        output_tensor = output_tensors[self.model_chunk_id].pop(0)
        output_tensor_grad = output_tensor_grads[self.model_chunk_id].pop(0)
        input_tensor_grad = self.backward_step(
            input_tensor, output_tensor, output_tensor_grad, grad_scaler, deallocate_pipeline_outputs
        )

        def f(input_tensor):
            if input_tensor is not None:
                assert isinstance(input_tensor, (torch.Tensor, DTensor)), input_tensor
                input_tensor.grad = None
            DEALLOCATE_OUTPUT_TENSOR().run(input_tensor, deallocate_pipeline_outputs)

        if not isinstance(input_tensor, Sequence):
            map(f, [input_tensor])
        else:
            map(f, input_tensor)

        # launch grad synchronization (custom grad sync)
        # Note: Asynchronous communication tends to slow down compute.
        # To reduce idling from mismatched microbatch times, we launch
        # asynchronous communication at the same time across the
        # pipeline-parallel group.
        if grad_sync_func is not None:
            if self.grad_sync_microbatch_id >= 0 and self.last_microbatch_for_model_chunk:
                grad_sync_func(model[self.grad_sync_chunk_id])
                synchronized_model_chunks.add(self.grad_sync_chunk_id)
        return input_tensor_grad


@dataclass
class FWD(BaseInstruction):
    microbatch_id: int = 0
    model_chunk_id: int = 0
    param_sync_chunk_id: int = 0
    is_vpp_first_stage: bool = False
    is_vpp_last_stage: bool = False
    forward_only: bool = False
    num_model_chunks: int = 1
    num_microbatches: int = 1
    param_sync_microbatch_id: int = 0
    first_microbatch_for_model_chunk: bool = True
    optimizer_step_successful: bool = True
    overlap_p2p_comm: bool = False
    param_sync_overlap: bool = False
    debug: str = ""

    @property
    def name(self):
        return "forward"

    def forward_step(
        self,
        data_iterator,
        input_tensor,
        model,
        forward_data_store,
        is_pp_first_stage: bool,
        is_pp_last_stage: bool,
        autocast_dtype=torch.float,
        enable_autocast=False,
        model_chunk_id=0,
    ):
        """Forward step for passed-in model.

        If first stage, input tensor is obtained from data_iterator, otherwise
        passed-in input_tensor is used.

        Returns output tensor."""
        if enable_autocast:
            context_manager = torch.autocast("cuda", dtype=autocast_dtype)
        else:
            context_manager = contextlib.nullcontext()
        with context_manager:

            def prepare_data():
                model_chunk_id = builder.user_data["model_chunk_id"]
                ground_truth = []
                if builder.user_data["is_pp_first_stage"]:
                    local_tensors = next(builder.dataloader[model_chunk_id])
                    true_input_tensor = None
                else:
                    local_tensors = next(builder.dataloader[model_chunk_id])
                    if isinstance(local_tensors, Sequence) and len(local_tensors) > 1:
                        ground_truth.append(local_tensors[-1])
                    elif isinstance(local_tensors, Dict) and "labels" in local_tensors:
                        ground_truth.append(local_tensors["labels"])
                    true_input_tensor = builder.user_data["p2p_tensors"]
                    if isinstance(true_input_tensor, Sequence) and len(true_input_tensor) == 1:
                        true_input_tensor = true_input_tensor[0]

                return true_input_tensor, local_tensors, ground_truth

            builder.user_data["model_chunk_id"] = model_chunk_id
            builder.user_data["p2p_tensors"] = input_tensor
            builder.user_data["is_pp_first_stage"] = is_pp_first_stage
            builder.user_data["is_pp_last_stage"] = is_pp_last_stage
            builder.user_data["prepare_data_fn"] = prepare_data
            p2p_input, local_input, ground_truth = registed_functions["vescale_interleaved_1f1b_pre_forward_data"]()
            builder.user_data["ground_truth"] = ground_truth
            output_tensor = registed_functions["vescale_interleaved_1f1b_forward"](p2p_input, local_input)
            builder.user_data["output_tensor"] = output_tensor

        if is_pp_last_stage:
            output_tensor, loss_tensor = registed_functions["vescale_interleaved_1f1b_loss_fn"]()
            forward_data_store.append((output_tensor, loss_tensor))
            if builder.loss_fn is None:
                return output_tensor
            else:
                return loss_tensor

        return output_tensor

    @dummy_p2p
    def run(self, input_tensors, output_tensors, param_sync_func, kwargs):
        # dump arguments for underlying fwd/bwd helpers
        data_iterator, model, forward_data_store, dtype, enable_autocast = (
            kwargs["data_iterator"],
            kwargs["model"],
            kwargs["forward_data_store"],
            kwargs["dtype"],
            kwargs["enable_autocast"],
        )

        assert param_sync_func is None
        # TODO: implment logic for param_sync_func with PipeModule's utils
        if param_sync_func is not None:
            if self.param_sync_microbatch_id < self.num_microbatches and self.first_microbatch_for_model_chunk:
                if 1 < self.param_sync_chunk_id < self.num_model_chunks:
                    param_sync_func(model[self.param_sync_chunk_id].parameters())

        if self.overlap_p2p_comm and self.param_sync_overlap:
            drain_recv_reqs("forward")

        # forward step
        if self.is_vpp_first_stage:
            if len(input_tensors[self.model_chunk_id]) == len(output_tensors[self.model_chunk_id]):
                input_tensors[self.model_chunk_id].append(None)

        input_tensor = input_tensors[self.model_chunk_id][-1]
        output_tensor = self.forward_step(
            data_iterator=data_iterator,
            input_tensor=input_tensor,
            model=model,
            forward_data_store=forward_data_store,
            is_pp_first_stage=self.is_vpp_first_stage,
            is_pp_last_stage=self.is_vpp_last_stage,
            autocast_dtype=dtype,
            enable_autocast=enable_autocast,
            model_chunk_id=self.model_chunk_id,
        )
        output_tensors[self.model_chunk_id].append(output_tensor)

        # if forward-only, no need to save tensors for a backward pass
        if self.forward_only:
            input_tensors[self.model_chunk_id].pop()
            output_tensors[self.model_chunk_id].pop()

        return output_tensor


@dataclass
class BUBBLE(BaseInstruction):
    @property
    def name(self):
        return "bubble"

    def run(self):
        return


@dataclass
class LAUNCH_SHARED_UNITS_SYNC(BaseInstruction):
    num_chunks: int = 1

    @property
    def name(self):
        return "launch remain grad sync"

    @dummy_p2p
    def run(self, model):
        for model_chunk_id in range(self.num_chunks):
            # if isinstance(model, PipeModule):
            #     model.sync_shared_params(share_params=False, model_chunk_id=model_chunk_id)
            ...


class InterleavedPipeDreramFlush(PipelineSchema):
    def __init__(
        self,
        num_chunks: int,
        meshes: Sequence[DeviceMesh],
        default_shape: Shape,
        default_dtype: torch.dtype = torch.float32,
        batches: int = 1,
        input_shapes: Optional[List] = None,
        input_shapes_unpad: Optional[List] = None,
        **kwargs,
    ):
        assert batches % len(meshes) == 0, "Interleaved 1f1b only support mircobatch size mode device size"
        assert batches // len(meshes) > 1, "Interleaved 1f1b only support mircobatch size = Interger * device size"
        self.num_chunks = num_chunks
        self.total_num_microbatches = num_chunks * batches
        self.input_shapes = input_shapes
        self.input_shapes_unpad = input_shapes_unpad
        self.default_tensor_shape = default_shape
        self.default_dtype = default_dtype
        super().__init__(len(meshes), meshes, batches)

    @property
    def name(self):
        return "Interleaved 1f1b"

    def get_variable_tensor_shape(self, microbatch_id: int):
        if self.input_shapes is None or len(self.input_shapes) == 0 or microbatch_id >= self.total_num_microbatches:
            return self.default_tensor_shape

        microbatch_group_size = self.num_mesh * self.num_chunks
        microbatch_group_id = microbatch_id // microbatch_group_size
        microbatch_id_in_group = microbatch_id % microbatch_group_size
        microbatch_id_curr_model_chunk = microbatch_group_id * self.num_mesh + microbatch_id_in_group % self.num_mesh
        tensor_shape = self.input_shapes[microbatch_id_curr_model_chunk]

        return tensor_shape

    def get_variable_tensor_shape_unpad(self, microbatch_id: int):
        if (
            self.input_shapes_unpad is None
            or len(self.input_shapes_unpad) == 0
            or microbatch_id >= self.total_num_microbatches
        ):
            return None

        microbatch_group_size = self.num_mesh * self.num_chunks
        microbatch_group_id = microbatch_id // microbatch_group_size
        microbatch_id_in_group = microbatch_id % microbatch_group_size
        microbatch_id_curr_model_chunk = microbatch_group_id * self.num_mesh + microbatch_id_in_group % self.num_mesh
        return self.input_shapes_unpad[microbatch_id_curr_model_chunk]

    def get_model_chunk_id(self, microbatch_id: int, forward: bool):
        """Helper method to get the model chunk ID given the iteration number."""
        microbatch_id_in_group = microbatch_id % (self.num_mesh * self.num_chunks)
        model_chunk_id = microbatch_id_in_group // self.num_mesh
        if not forward:
            model_chunk_id = self.num_chunks - model_chunk_id - 1
        return model_chunk_id

    def is_first_microbatch_for_model_chunk_eager(self, microbatch_id: int) -> bool:
        """Check if an iteration is the first for a model chunk eagerly"""
        if microbatch_id % self.num_mesh != 0:
            # Not the first time to run this model chunk
            # For pipeline stage 0, chunk 0 is used by mb(0)
            # mb(p), mb(2p), ...
            return False
        # grouping microbatches by pp_size, the groups will run different model chunk iteratively
        microbatch_group_id = microbatch_id // self.num_mesh
        if microbatch_group_id < self.num_chunks:
            return True
        return False

    def is_first_microbatch_for_model_chunk(self, microbatch_id: int) -> bool:
        """Check if an iteration is the first for a model chunk."""
        # pp(0): mb(3+1)
        # pp(1): mb(2+1)
        # pp(2): mb(1+1)
        # pp(3): mb(0+1)
        microbatch_group_size = self.num_mesh * self.num_chunks
        num_microbatch_groups = self.total_num_microbatches // microbatch_group_size
        microbatch_group_id = microbatch_id // microbatch_group_size
        microbatch_id_in_group = microbatch_id % microbatch_group_size
        if microbatch_group_id == 0:
            return microbatch_id_in_group % self.num_mesh == 0
        else:
            return False

    def is_last_microbatch_for_model_chunk(self, microbatch_id: int) -> bool:
        """Check if an iteration is the last for a model chunk."""
        microbatch_group_size = self.num_mesh * self.num_chunks
        num_microbatch_groups = self.total_num_microbatches // microbatch_group_size
        microbatch_group_id = microbatch_id // microbatch_group_size
        microbatch_id_in_group = microbatch_id % microbatch_group_size
        if microbatch_group_id == num_microbatch_groups - 1:
            return microbatch_id_in_group % self.num_mesh == self.num_mesh - 1
        else:
            return False

    def _gen_schedule(self):
        b = self.batches
        d = self.num_mesh
        s = self.num_chunks

        warmup_batches = [min((d - i - 1) * 2 + (s - 1) * d, b * s) for i in range(d)]
        self.warmup_batches = warmup_batches
        remaining = [(b * s - w) for w in warmup_batches]
        self.remaining = remaining
        num_clock = (b * s + d - 1) * 2  # time todo flush
        schedules = [[None] * d for c in range(num_clock)]
        new_timeline = list(range(d))
        bwd_done_idx = np.zeros(shape=[num_clock, d, s], dtype=np.int32)
        next_fwd_batch_idx = np.zeros(shape=[d, s], dtype=np.int32)
        next_bwd_batch_idx = np.zeros(shape=[d, s], dtype=np.int32)
        # warm-up steps
        for i in range(d):
            for k in range(warmup_batches[i]):
                t_i = new_timeline[i]
                chunk_id = self.get_model_chunk_id(k, forward=True)
                schedules[t_i][i] = Status(next_fwd_batch_idx[i][chunk_id], i, chunk_id, "F", "WUp", k)
                new_timeline[i] += 1  # self add for new timeline
                next_fwd_batch_idx[i][chunk_id] += 1  # do next micro batch

        for i in reversed(range(d)):
            for k in range(remaining[i]):
                t_i = new_timeline[i]
                f_k = k + warmup_batches[i]
                chunk_id = self.get_model_chunk_id(f_k, forward=True)
                schedules[t_i][i] = Status(next_fwd_batch_idx[i][chunk_id], i, chunk_id, "F", "1f1b", k)
                next_fwd_batch_idx[i][chunk_id] += 1  # do next micro batch
                bwd_k = k
                chunk_id = self.get_model_chunk_id(bwd_k, forward=False)
                bwd_done_idx[t_i][i] = bwd_done_idx[t_i - 1][i]
                bwd_done_idx[t_i][i][chunk_id] = next_bwd_batch_idx[i][chunk_id]
                t_i += 1

                # do backward
                if i + 1 < d:
                    while bwd_done_idx[t_i][i + 1][chunk_id] < next_bwd_batch_idx[i][chunk_id]:
                        assert bwd_done_idx[t_i - 1][i][chunk_id] == next_bwd_batch_idx[i][chunk_id]
                        bwd_done_idx[t_i][i][chunk_id] = bwd_done_idx[t_i - 1][i][chunk_id]
                        t_i = t_i + 1

                if k == remaining[i] - 1:  # last iterator
                    schedules[t_i][i] = Status(next_bwd_batch_idx[i][chunk_id], i, chunk_id, "B", "1f1b-l", k)
                else:
                    schedules[t_i][i] = Status(next_bwd_batch_idx[i][chunk_id], i, chunk_id, "B", "1f1b", k)

                bwd_done_idx[t_i][i] = bwd_done_idx[t_i - 1][i]
                bwd_done_idx[t_i][i][chunk_id] = next_bwd_batch_idx[i][chunk_id]
                next_bwd_batch_idx[i][chunk_id] += 1
                new_timeline[i] = t_i + 1

        # run cooldown passes
        for i in reversed(range(d)):
            for k in range(remaining[i], self.total_num_microbatches):
                t_i = new_timeline[i]
                bwd_k = k
                chunk_id = self.get_model_chunk_id(bwd_k, forward=False)
                if i + 1 < d:
                    while bwd_done_idx[t_i][i + 1][chunk_id] <= next_bwd_batch_idx[i][chunk_id]:
                        bwd_done_idx[t_i][i] = bwd_done_idx[t_i - 1][i]
                        bwd_done_idx[t_i][i][chunk_id] = next_bwd_batch_idx[i][chunk_id]
                        t_i = t_i + 1
                schedules[t_i][i] = Status(next_bwd_batch_idx[i][chunk_id], i, chunk_id, "B", "CD", k)
                bwd_done_idx[t_i][i] = bwd_done_idx[t_i - 1][i]
                bwd_done_idx[t_i][i] = next_bwd_batch_idx[i]
                next_bwd_batch_idx[i][chunk_id] += 1
                new_timeline[i] = t_i + 1
            bwd_done_idx[new_timeline[i] : num_clock, i, :] = b

        return schedules


class InterleavedOneFOneBInstructionGenerator(InstructionGenerator):
    def __init__(
        self,
        deps: StageDeps,
        meshes: List[DeviceMesh],
        batches: int,
        default_shape: Optional[Shape] = None,
        default_dtype: Optional[torch.dtype] = None,
        batch_shape_lists: Optional[List[Dict[int, Shape]]] = None,
        batch_dtype_lists: Optional[List[Dict[int, torch.dtype]]] = None,
        input_shapes: List[Dict[int, Shape]] = None,
        input_shapes_unpad: List[Dict[int, Shape]] = None,
        num_chunks: int = 1,
        batch_p2p_comm: bool = True,
        param_sync_overlap: bool = False,
        overlap_p2p_comm: bool = False,
        grad_sync_overlap: bool = False,
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
            num_chunk=num_chunks,
            forward_only=forward_only,
        )
        self.batch_p2p_comm = batch_p2p_comm
        self.overlap_p2p_comm = overlap_p2p_comm
        self.param_sync_overlap = param_sync_overlap
        self.grad_sync_overlap = grad_sync_overlap
        self.num_stage = len(meshes)
        self.num_chunks = num_chunks
        self.num_meshes = self.num_stage
        self.schema = InterleavedPipeDreramFlush(
            num_chunks=self.num_chunks,
            meshes=self.meshes,
            batches=self.batches,
            default_shape=default_shape,
            default_dtype=default_dtype,
            input_shapes=input_shapes,
            input_shapes_unpad=input_shapes_unpad,
        )
        self.forward_only = forward_only

    def get_tensor_shape(self, microbatch_id: int, input_id: int = 0):
        if (
            self.schema.input_shapes is None
            or len(self.schema.input_shapes) == 0
            or microbatch_id >= self.schema.total_num_microbatches
        ):
            return self.schema.default_tensor_shape
        microbatch_group_size = self.num_mesh * self.num_chunks
        microbatch_group_id = microbatch_id // microbatch_group_size
        microbatch_id_in_group = microbatch_id % microbatch_group_size
        microbatch_id_curr_model_chunk = microbatch_group_id * self.num_mesh + microbatch_id_in_group % self.num_mesh
        tensor_shape = self.schema.input_shapes[microbatch_id_curr_model_chunk]
        if isinstance(tensor_shape, Dict):
            tensor_shape = tensor_shape[input_id]
        return tensor_shape

    def get_variable_tensor_shape_unpad(self, microbatch_id: int, input_id: int = 0):
        if (
            self.schema.input_shapes is None
            or len(self.schema.input_shapes) == 0
            or microbatch_id >= self.schema.total_num_microbatches
        ):
            return self.schema.default_tensor_shape
        microbatch_group_size = self.num_mesh * self.num_chunks
        microbatch_group_id = microbatch_id // microbatch_group_size
        microbatch_id_in_group = microbatch_id % microbatch_group_size
        microbatch_id_curr_model_chunk = microbatch_group_id * self.num_mesh + microbatch_id_in_group % self.num_mesh
        tensor_shape = self.schema.input_shapes_unpad[microbatch_id_curr_model_chunk]
        if isinstance(tensor_shape, Dict):
            tensor_shape = tensor_shape[input_id]
        return tensor_shape

    def get_tensor_dtype(self, microbatch_id: int, input_id: int = 0):
        if (
            self.batch_dtype_lists is None
            or len(self.batch_dtype_lists) == 0
            or microbatch_id >= self.schema.total_num_microbatches
        ):
            return self.default_dtype
        microbatch_group_size = self.num_mesh * self.num_chunks
        microbatch_group_id = microbatch_id // microbatch_group_size
        microbatch_id_in_group = microbatch_id % microbatch_group_size
        microbatch_id_curr_model_chunk = microbatch_group_id * self.num_mesh + microbatch_id_in_group % self.num_mesh
        tensor_dtype = self.batch_dtype_lists[microbatch_id_curr_model_chunk]
        if isinstance(tensor_dtype, Dict):
            tensor_dtype = tensor_dtype[input_id]
        return tensor_dtype

    def get_shape_or_dtype(self, ff: Callable, comm_packages: List[CommPacket], microbatch_id):
        def _get_shape_or_dtype(f: Callable, package: CommPacket):
            return f(microbatch_id, package.input_id)

        return list(map(partial(_get_shape_or_dtype, ff), comm_packages))

    # call by pipe emitter
    def gen_instruction(self):
        schedules: List = self.schema.schedules
        self.instruction_list = [[] for _ in range(self.num_stage)]
        first_time_1f1b = [True] * self.num_stage
        first_time_cool_down = [True] * self.num_stage
        _forward_only = self.forward_only
        if not torch.is_grad_enabled():
            self.forward_only = True

        # before warmup
        for s in range(self.num_meshes):
            recv_comms = self.deps.get_recv_comms(s)
            tensor_shapes = self.get_shape_or_dtype(self.get_tensor_shape, recv_comms, 0)
            tensor_dtypes = self.get_shape_or_dtype(self.get_tensor_dtype, recv_comms, 0)
            self._set_inst(
                RECV_FORWARD(
                    comm_packages=recv_comms,
                    tensor_shapes=tensor_shapes,
                    tensor_dtypes=tensor_dtypes,
                    batch_p2p_comm=self.batch_p2p_comm,
                    batch_id=0,
                    is_pp_first_stage=self.deps.is_pipeline_first_stage(s),
                    debug="before warm-up",
                ),
                s,
            )

        one_f_one_b_set = [set() for _ in range(self.num_meshes)]

        for clk, stages_schemas in enumerate(schedules):
            for s, schema in enumerate(stages_schemas):
                is_pp_first_stage = self.deps.is_pipeline_first_stage(s)
                is_pp_last_stage = self.deps.is_pipeline_last_stage(s)
                send_comms = self.deps.get_send_comms(s)
                recv_comms = self.deps.get_recv_comms(s)
                if schema:
                    stg = schema.stg
                    k = schema.k
                    send_shapes = self.get_shape_or_dtype(self.get_tensor_shape, send_comms, k)
                    send_dtypes = self.get_shape_or_dtype(self.get_tensor_dtype, send_comms, k)
                    send_shapes_unpad = self.get_shape_or_dtype(self.get_variable_tensor_shape_unpad, send_comms, k)
                    recv_shapes = self.get_shape_or_dtype(self.get_tensor_shape, recv_comms, k)
                    recv_dtypes = self.get_shape_or_dtype(self.get_tensor_dtype, recv_comms, k)
                    if "WUp" in stg:
                        if not self.overlap_p2p_comm:
                            self._set_inst(WAIT_FWD(), s)
                        elif not self.param_sync_overlap:
                            self._set_inst(DRAIN_RECV_REQS(drain_type="forward"), s)
                        # TODO: all warmup batch check

                        model_chunk_id = self.schema.get_model_chunk_id(k, forward=True)
                        is_vpp_first_stage = self.deps.is_vpp_first_stage(s, model_chunk_id)
                        is_vpp_last_stage = self.deps.is_vpp_last_stage(s, model_chunk_id)
                        param_sync_microbatch_id = k + self.schema.num_mesh
                        param_sync_chunk_id = self.schema.get_model_chunk_id(param_sync_microbatch_id, forward=True) + 1
                        first_microbatch_for_model_chunk = self.schema.is_first_microbatch_for_model_chunk(k)
                        self._set_inst(
                            FWD(
                                microbatch_id=k,
                                model_chunk_id=model_chunk_id,
                                param_sync_chunk_id=param_sync_chunk_id,
                                is_vpp_first_stage=is_vpp_first_stage,
                                is_vpp_last_stage=is_vpp_last_stage,
                                forward_only=self.forward_only,
                                num_model_chunks=self.num_chunk,
                                num_microbatches=self.batches * self.num_chunk,
                                param_sync_microbatch_id=param_sync_microbatch_id,
                                first_microbatch_for_model_chunk=first_microbatch_for_model_chunk,
                                overlap_p2p_comm=self.overlap_p2p_comm,
                                param_sync_overlap=self.param_sync_overlap,
                            ),
                            s,
                        )
                        # Determine if tensor should be received from previous stage.
                        next_forward_model_chunk_id = self.schema.get_model_chunk_id(k + 1, forward=True)
                        recv_prev = True
                        if is_pp_first_stage:
                            if next_forward_model_chunk_id == 0:
                                recv_prev = False
                        if k == (self.schema.total_num_microbatches - 1):
                            recv_prev = False

                        if is_vpp_last_stage:
                            self._set_inst(SET_OUTPUT_TO_NONE(), s)

                        if not self.overlap_p2p_comm:
                            if k == (self.schema.warmup_batches[s] - 1) and not self.forward_only:
                                self._set_inst(SET_INPUTGRAD_TO_NONE(), s)
                                recv_next = True
                                if is_pp_last_stage:
                                    recv_next = False
                                self._set_inst(
                                    SEND_FORWARD_BACKWARD_RECV_FORWARD_BACKWARD(
                                        recv_prev=recv_prev,
                                        recv_next=recv_next,
                                        recv_comms=recv_comms,
                                        send_comms=send_comms,
                                        recv_shapes=recv_shapes,
                                        recv_dtypes=recv_dtypes,
                                        batch_p2p_comm=self.batch_p2p_comm,
                                        debug="none p2p overlap, last batch warm-up",
                                    ),
                                    s,
                                )

                                self._set_inst(APPEND_GRADS(chunk=self.num_chunk - 1), s)
                            else:
                                self._set_inst(
                                    SEND_FORWARD_RECV_FORWARD(
                                        recv_prev=recv_prev,
                                        send_shapes=send_shapes,
                                        send_tensor_shapes_unpad=send_shapes_unpad,
                                        send_dtypes=send_dtypes,
                                        batch_p2p_comm=self.batch_p2p_comm,
                                        overlap_p2p_comm=self.overlap_p2p_comm,
                                        microbatch_id=k,
                                        send_comms=send_comms,
                                        recv_comms=recv_comms,
                                        debug="none p2p overlap, warm-up",
                                    ),
                                    s,
                                )

                            self._set_inst(APPEND_INPUTS(chunk=next_forward_model_chunk_id), s)
                        else:
                            tensor_shapes = self.get_shape_or_dtype(self.get_tensor_shape, send_comms, k + 1)
                            tensor_dtypes = self.get_shape_or_dtype(self.get_tensor_dtype, send_comms, k + 1)

                            self._set_inst(
                                SEND_FORWARD_RECV_FORWARD(
                                    recv_prev=recv_prev,
                                    send_shapes=tensor_shapes,
                                    send_tensor_shapes_unpad=send_shapes_unpad,
                                    send_dtypes=tensor_dtypes,
                                    batch_p2p_comm=self.batch_p2p_comm,
                                    overlap_p2p_comm=self.overlap_p2p_comm,
                                    send_comms=send_comms,
                                    recv_comms=recv_comms,
                                    debug="p2p overlap, warm up",
                                ),
                                s,
                            )
                            if k == (self.schema.warmup_batches[s] - 1) and not self.forward_only:
                                self._set_inst(SET_INPUTGRAD_TO_NONE(), s)
                                recv_next = True
                                if is_pp_last_stage:
                                    recv_next = False
                                self._set_inst(
                                    SEND_BACKWARD_RECV_BACKWARD(
                                        recv_next=recv_next,
                                        send_shapes=send_shapes,
                                        send_tensor_shapes_unpad=send_shapes_unpad,
                                        send_dtypes=send_dtypes,
                                        batch_p2p_comm=self.batch_p2p_comm,
                                        overlap_p2p_comm=self.overlap_p2p_comm,
                                        send_comms=send_comms,
                                        recv_comms=recv_comms,
                                        debug="warm-up",
                                    ),
                                    s,
                                )
                                self._set_inst(APPEND_GRADS(self.num_chunk - 1), s)
                            self._set_inst(APPEND_INPUTS(chunk=next_forward_model_chunk_id), s)
                        self._set_inst(DEALLOCATE_OUTPUT_TENSOR(), s)
                    elif "1f1b" in stg:  # 1f1b stage
                        forward_k = k + self.schema.warmup_batches[s]
                        if first_time_1f1b[s]:
                            if self.overlap_p2p_comm:
                                self._set_inst(DRAIN_SEND_REQS(), s)
                            first_time_1f1b[s] = False
                        if k in one_f_one_b_set[s]:
                            continue
                        else:
                            one_f_one_b_set[s].add(k)
                        if self.overlap_p2p_comm:
                            if not self.param_sync_overlap:
                                self._set_inst(DRAIN_RECV_REQS(drain_type="forward"), s)
                            self._set_inst(DEALLOCATE_OUTPUT_TENSOR(), s)

                            model_chunk_id = self.schema.get_model_chunk_id(forward_k, forward=True)
                            is_vpp_first_stage = self.deps.is_vpp_first_stage(s, model_chunk_id)
                            is_vpp_last_stage = self.deps.is_vpp_last_stage(s, model_chunk_id)
                            param_sync_microbatch_id = forward_k + self.schema.num_mesh
                            param_sync_chunk_id = (
                                self.schema.get_model_chunk_id(param_sync_microbatch_id, forward=True) + 1
                            )
                            first_microbatch_for_model_chunk = self.schema.is_first_microbatch_for_model_chunk(
                                forward_k
                            )
                            self._set_inst(
                                FWD(
                                    microbatch_id=forward_k,
                                    model_chunk_id=model_chunk_id,
                                    param_sync_chunk_id=param_sync_chunk_id,
                                    param_sync_microbatch_id=param_sync_microbatch_id,
                                    param_sync_overlap=self.param_sync_overlap,
                                    first_microbatch_for_model_chunk=first_microbatch_for_model_chunk,
                                    is_vpp_first_stage=is_vpp_first_stage,
                                    is_vpp_last_stage=is_vpp_last_stage,
                                    forward_only=self.forward_only,
                                    num_model_chunks=self.num_chunk,
                                    num_microbatches=self.batches * self.num_chunk,
                                    debug="1f1b",
                                ),
                                s,
                            )
                            # Determine if current stage has anything to send in either direction,
                            # otherwise set tensor to None.
                            # Last virtual stage no activation tensor to send
                            if is_vpp_last_stage:
                                self._set_inst(SET_OUTPUT_TO_NONE(), s)
                            # Determine if peers are sending, and where in data structure to put
                            # received tensors.
                            recv_prev = True
                            if is_pp_first_stage:
                                # First stage is ahead of last stage by (pipeline_parallel_size - 1).
                                next_forward_model_chunk_id = self.schema.get_model_chunk_id(
                                    forward_k - (self.schema.num_mesh - 1), forward=True
                                )
                                if next_forward_model_chunk_id == (self.schema.num_chunks - 1):
                                    recv_prev = False
                                next_forward_model_chunk_id += 1
                            else:
                                next_forward_model_chunk_id = self.schema.get_model_chunk_id(
                                    forward_k + 1, forward=True
                                )

                            # If last iteration, don't receive; we already received one extra
                            # before the start of the for loop.
                            if k == (self.schema.remaining[s] - 1):
                                recv_prev = False

                            # Send activation tensor to the next stage and receive activation tensor from the
                            # previous stage
                            tensor_shape = self.schema.get_variable_tensor_shape(forward_k + 1)
                            send_tensor_shape_unpad = self.schema.get_variable_tensor_shape_unpad(forward_k)
                            self._set_inst(
                                SEND_FORWARD_RECV_FORWARD(
                                    recv_prev=recv_prev,
                                    send_shapes=send_shapes,
                                    send_tensor_shapes_unpad=send_shapes_unpad,
                                    send_dtypes=send_dtypes,
                                    batch_p2p_comm=self.batch_p2p_comm,
                                    overlap_p2p_comm=self.overlap_p2p_comm,
                                    send_comms=send_comms,
                                    recv_comms=recv_comms,
                                    microbatch_id=forward_k,
                                    debug="1f1b",
                                ),
                                s,
                            )
                            self._set_inst(DRAIN_RECV_REQS(drain_type="backward"), s)

                            # Backward pass.
                            backward_k = k
                            grad_sync_microbatch_id = backward_k - s
                            grad_sync_chunk_id = self.schema.get_model_chunk_id(grad_sync_microbatch_id, forward=False)
                            last_microbatch_for_model_chunk = self.schema.is_last_microbatch_for_model_chunk(
                                grad_sync_microbatch_id
                            )
                            backward_model_chunk_id = self.schema.get_model_chunk_id(backward_k, forward=False)
                            is_vpp_first_stage = self.deps.is_vpp_first_stage(s, backward_model_chunk_id)
                            is_vpp_last_stage = self.deps.is_vpp_last_stage(s, backward_model_chunk_id)
                            self._set_inst(
                                BWD(
                                    is_vpp_last_stage=is_vpp_last_stage,
                                    last_microbatch_for_model_chunk=last_microbatch_for_model_chunk,
                                    grad_sync_chunk_id=grad_sync_chunk_id,
                                    grad_sync_microbatch_id=grad_sync_microbatch_id,
                                    model_chunk_id=backward_model_chunk_id,
                                    microbatch_id=backward_k,
                                    debug="1f1b",
                                ),
                                s,
                            )

                            # First virtual stage no activation gradient tensor to send
                            if is_vpp_first_stage:
                                self._set_inst(SET_INPUTGRAD_TO_NONE(), s)

                            # Determine if the current virtual stage has an activation gradient tensor to receive
                            recv_next = True
                            if is_pp_last_stage:
                                # Last stage is ahead of first stage by (pipeline_parallel_size - 1).
                                next_backward_model_chunk_id = self.schema.get_model_chunk_id(
                                    backward_k - (self.schema.num_mesh - 1), forward=False
                                )
                                if next_backward_model_chunk_id == 0:
                                    recv_next = False
                                next_backward_model_chunk_id -= 1
                            else:
                                next_backward_model_chunk_id = self.schema.get_model_chunk_id(
                                    backward_k + 1, forward=False
                                )

                            tensor_shape = self.schema.get_variable_tensor_shape(backward_k + 1)
                            send_tensor_shape_unpad = self.schema.get_variable_tensor_shape_unpad(backward_k)
                            self._set_inst(
                                SEND_BACKWARD_RECV_BACKWARD(
                                    recv_next=recv_next,
                                    send_shapes=send_shapes,
                                    send_tensor_shapes_unpad=send_shapes_unpad,
                                    send_dtypes=send_dtypes,
                                    batch_p2p_comm=self.batch_p2p_comm,
                                    overlap_p2p_comm=self.overlap_p2p_comm,
                                    send_comms=send_comms,
                                    recv_comms=recv_comms,
                                    debug="1f1b",
                                ),
                                s,
                            )
                        else:
                            model_chunk_id = self.schema.get_model_chunk_id(forward_k, forward=True)
                            is_vpp_first_stage = self.deps.is_vpp_first_stage(s, model_chunk_id)
                            is_vpp_last_stage = self.deps.is_vpp_last_stage(s, model_chunk_id)
                            param_sync_microbatch_id = forward_k + self.schema.num_mesh
                            param_sync_chunk_id = (
                                self.schema.get_model_chunk_id(param_sync_microbatch_id, forward=True) + 1
                            )
                            first_microbatch_for_model_chunk = self.schema.is_first_microbatch_for_model_chunk(
                                forward_k
                            )
                            self._set_inst(
                                FWD(
                                    microbatch_id=forward_k,
                                    model_chunk_id=model_chunk_id,
                                    is_vpp_first_stage=is_vpp_first_stage,
                                    is_vpp_last_stage=is_vpp_last_stage,
                                    param_sync_chunk_id=param_sync_chunk_id,
                                    param_sync_microbatch_id=param_sync_microbatch_id,
                                    first_microbatch_for_model_chunk=first_microbatch_for_model_chunk,
                                    forward_only=self.forward_only,
                                ),
                                s,
                            )

                            # Backward pass.
                            backward_k = k
                            grad_sync_microbatch_id = backward_k - s
                            grad_sync_chunk_id = self.schema.get_model_chunk_id(grad_sync_microbatch_id, forward=False)
                            last_microbatch_for_model_chunk = self.schema.is_last_microbatch_for_model_chunk(
                                grad_sync_microbatch_id
                            )
                            backward_model_chunk_id = self.schema.get_model_chunk_id(backward_k, forward=False)
                            is_vpp_first_stage = self.deps.is_vpp_first_stage(s, backward_model_chunk_id)
                            is_vpp_last_stage = self.deps.is_vpp_last_stage(s, backward_model_chunk_id)
                            self._set_inst(
                                BWD(
                                    microbatch_id=backward_k,
                                    model_chunk_id=backward_model_chunk_id,
                                    is_vpp_last_stage=is_vpp_last_stage,
                                    last_microbatch_for_model_chunk=last_microbatch_for_model_chunk,
                                    grad_sync_microbatch_id=grad_sync_microbatch_id,
                                    grad_sync_chunk_id=grad_sync_chunk_id,
                                    debug="1f1b",
                                ),
                                s,
                            )

                            # Send output_tensor and input_tensor_grad, receive input_tensor
                            # and output_tensor_grad.

                            # Determine if current stage has anything to send in either direction,
                            # otherwise set tensor to None.
                            forward_model_chunk_id = self.schema.get_model_chunk_id(forward_k, forward=True)
                            is_vpp_last_stage = self.deps.is_vpp_last_stage(s, forward_model_chunk_id)
                            if is_vpp_last_stage:
                                self._set_inst(SET_OUTPUT_TO_NONE(), s)
                            backward_model_chunk_id = self.schema.get_model_chunk_id(backward_k, forward=False)
                            is_vpp_first_stage = self.deps.is_vpp_first_stage(s, backward_model_chunk_id)
                            if is_vpp_first_stage:
                                self._set_inst(SET_INPUTGRAD_TO_NONE(), s)

                            # Determine if peers are sending, and where in data structure to put
                            # received tensors.
                            recv_prev = True
                            if is_pp_first_stage:
                                # First stage is ahead of last stage by (pipeline_parallel_size - 1).
                                next_forward_model_chunk_id = self.schema.get_model_chunk_id(
                                    forward_k - (self.num_meshes - 1), forward=True
                                )
                                if next_forward_model_chunk_id == (self.num_chunks - 1):
                                    recv_prev = False
                                next_forward_model_chunk_id += 1
                            else:
                                next_forward_model_chunk_id = self.schema.get_model_chunk_id(
                                    forward_k + 1, forward=True
                                )

                            recv_next = True
                            if is_pp_last_stage:
                                # Last stage is ahead of first stage by (pipeline_parallel_size - 1).
                                next_backward_model_chunk_id = self.schema.get_model_chunk_id(
                                    backward_k - (self.num_meshes - 1), forward=False
                                )
                                if next_backward_model_chunk_id == 0:
                                    recv_next = False
                                next_backward_model_chunk_id -= 1
                            else:
                                next_backward_model_chunk_id = self.schema.get_model_chunk_id(
                                    backward_k + 1, forward=False
                                )

                            # If last iteration, don't receive; we already received one extra
                            # before the start of the for loop.
                            if k == (self.schema.remaining[s] - 1):
                                recv_prev = False

                            self._set_inst(
                                SEND_FORWARD_BACKWARD_RECV_FORWARD_BACKWARD(
                                    recv_prev=recv_prev,
                                    recv_next=recv_next,
                                    send_comms=send_comms,
                                    recv_comms=recv_comms,
                                    recv_shapes=recv_shapes,
                                    recv_dtypes=recv_dtypes,
                                    batch_p2p_comm=self.batch_p2p_comm,
                                    debug="1f1b",
                                ),
                                s,
                            )

                            self._set_inst(DEALLOCATE_OUTPUT_TENSOR(), s)

                        # Put input_tensor and output_tensor_grad in data structures in the
                        # right location.
                        if recv_prev:
                            self._set_inst(APPEND_INPUTS(chunk=next_forward_model_chunk_id), s)
                        if recv_next:
                            self._set_inst(APPEND_GRADS(chunk=next_backward_model_chunk_id), s)

                        # launch grad_sync_func here to overlap with p2p communication
                        if self.grad_sync_overlap:
                            raise NotImplementedError("grad sync is not implement yet")
                    elif stg == "CD":  # cool down stage
                        if first_time_cool_down[s]:
                            self._set_inst(DEALLOCATE_OUTPUT_TENSOR(), s)
                            if self.overlap_p2p_comm:
                                self._set_inst(DRAIN_SEND_REQS(), s)
                            if not self.forward_only:
                                if self.overlap_p2p_comm:
                                    self._set_inst(DRAIN_RECV_REQS(drain_type="all", check_bwd_wait=True), s)
                            first_time_cool_down[s] = False
                        if self.forward_only:
                            continue  # forward have no backward phase

                        grad_sync_microbatch_id = k - s
                        grad_sync_chunk_id = self.schema.get_model_chunk_id(grad_sync_microbatch_id, forward=False)
                        last_microbatch_for_model_chunk = self.schema.is_last_microbatch_for_model_chunk(
                            grad_sync_microbatch_id
                        )
                        model_chunk_id = self.schema.get_model_chunk_id(k, forward=False)
                        is_vpp_last_stage = self.deps.is_vpp_first_stage(s, model_chunk_id)
                        self._set_inst(
                            BWD(
                                microbatch_id=k,
                                is_vpp_last_stage=is_vpp_last_stage,
                                model_chunk_id=model_chunk_id,
                                last_microbatch_for_model_chunk=last_microbatch_for_model_chunk,
                                grad_sync_chunk_id=grad_sync_chunk_id,
                                grad_sync_microbatch_id=grad_sync_microbatch_id,
                                debug="cooldown",
                            ),
                            s,
                        )
                        next_backward_model_chunk_id = self.schema.get_model_chunk_id(k + 1, forward=False)
                        recv_next = True
                        if is_pp_last_stage:
                            if next_backward_model_chunk_id == (self.schema.num_chunks - 1):
                                recv_next = False
                        if k == (self.schema.total_num_microbatches - 1):
                            recv_next = False

                        tensor_shape = self.schema.get_variable_tensor_shape(k + 1)
                        send_tensor_shape_unpad = self.schema.get_variable_tensor_shape_unpad(k)
                        self._set_inst(
                            SEND_BACKWARD_RECV_BACKWARD(
                                recv_next=recv_next,
                                send_shapes=send_shapes,
                                send_tensor_shapes_unpad=send_shapes_unpad,
                                send_dtypes=send_dtypes,
                                batch_p2p_comm=self.batch_p2p_comm,
                                overlap_p2p_comm=self.overlap_p2p_comm,
                                send_comms=send_comms,
                                recv_comms=recv_comms,
                                debug="cooldown",
                            ),
                            s,
                        )
                        self._set_inst(APPEND_GRADS(chunk=next_backward_model_chunk_id), s)

                        if self.grad_sync_overlap:
                            raise NotImplementedError("grad sync is not support yet")

                        if self.overlap_p2p_comm:
                            self._set_inst(DRAIN_RECV_REQS(drain_type="all"), s)
                    else:  # bubble
                        # do any other
                        self._set_inst(BUBBLE(), s)
            # Launch any remaining grad reductions
            # if grad_sync_func is not None:
            #     for model_chunk_id in range(num_model_chunks):
            #         if model_chunk_id not in synchronized_model_chunks:
            #             grad_sync_func(model[model_chunk_id], model_chunk_id)
            #             synchronized_model_chunks.add(model_chunk_id)

        # add cool down things
        for s in range(self.num_meshes):
            if not self.forward_only:
                # Launch any remaining grad reductions
                self._set_inst(LAUNCH_SHARED_UNITS_SYNC(num_chunks=self.deps.get_num_chunks()), s)

            if self.overlap_p2p_comm:
                self._set_inst(DRAIN_SEND_REQS(), s)

        # restore original self.forward_only if the current context manager is torch.no_grad()
        if not torch.is_grad_enabled():
            self.forward_only = _forward_only

        self.gen_instruction_str_list()
        return self.instruction_list

    def gen_instruction_str_list(self):
        instruction_lists = self.instruction_list
        stage_strs = defaultdict(str)
        for stage_id, instruction_list in enumerate(instruction_lists):
            cur_stage_str = stage_strs[stage_id]
            for inst in instruction_list:
                cur_stage_str += f"{VESACLE_INSTRUCTION_MAPPING_V[type(inst)]},"
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
        param_sync_func=None,
        grad_sync_func=None,
    ):
        # if the current context manager is torch.no_grad(), do not compute backward
        temp_forward_only = self.forward_only
        if not torch.is_grad_enabled():
            self.forward_only = False

        # init constant data
        builder.constant_data["autocast_dtype"] = autocast_dtype
        builder.constant_data["enable_autocast"] = enable_autocast
        builder.constant_data["grad_scaler"] = grad_scaler
        builder.constant_data["deallocate_pipeline_outputs"] = deallocate_pipeline_outputs
        builder.constant_data["param_sync_func"] = param_sync_func
        builder.constant_data["grad_sync_func"] = grad_sync_func

        # Model chunk IDs with synchronized grads
        builder.user_data["synchronized_model_chunks"] = set()
        builder.user_data["input_tensors"] = [[] for _ in range(self.num_chunk)]
        builder.user_data["output_tensors"] = [[] for _ in range(self.num_chunk)]
        builder.user_data["output_tensor_grads"] = [[] for _ in range(self.num_chunk)]
        builder.user_data["fwd_wait_handles"] = None
        builder.user_data["bwd_wait_handles"] = None
        builder.user_data["output_tensor"] = None
        builder.user_data["input_tensor"] = None
        builder.user_data["output_tensor_grad"] = None
        builder.user_data["forward_data_store"] = []
        model = self.deps.get_current_model(stage_id)

        builder.model = model
        instruction_list = self.get_instruction_list(stage_id)
        builder.stage_id = stage_id
        builder_instruction_list = builder.global_instructions_funcs[stage_id]

        for inst, fn in zip(instruction_list, builder_instruction_list):
            builder.user_data["inst"] = inst
            fn()

        # restore original self.forward_only if the current context manager is torch.no_grad()
        if not torch.is_grad_enabled():
            self.forward_only = temp_forward_only

        return builder.user_data["forward_data_store"]


@register_instruction(name="vescale_interleavd_1f1b_recv_forward")
def vpp_recv_forward():
    inst = builder.user_data["inst"]
    tmp = inst.run()
    input_tensors = builder.user_data["input_tensors"]
    input_tensors[0].append(tmp)


@register_instruction(name="vescale_interleavd_1f1b_forward")
def vpp_forward():
    inst = builder.user_data["inst"]
    user_data = builder.user_data
    forward_data_store = user_data["forward_data_store"]
    input_tensors = user_data["input_tensors"]
    output_tensors = user_data["output_tensors"]

    constant_data = builder.constant_data
    autocast_dtype = constant_data["autocast_dtype"]
    enable_autocast = constant_data["enable_autocast"]
    param_sync_func = constant_data["param_sync_func"]

    forward_args = {
        "data_iterator": builder.dataloader,
        "model": builder.model,
        "forward_data_store": forward_data_store,
        "dtype": autocast_dtype,
        "enable_autocast": enable_autocast,
    }
    output_tensor = inst.run(
        input_tensors=input_tensors,
        output_tensors=output_tensors,
        param_sync_func=param_sync_func,
        kwargs=forward_args,
    )
    user_data["output_tensor"] = output_tensor


@register_instruction(name="vescale_interleaved_1f1b_backward")
def vpp_backward():
    inst = builder.user_data["inst"]
    model = builder.model
    grad_scaler = builder.constant_data["grad_scaler"]
    deallocate_pipeline_outputs = builder.constant_data["deallocate_pipeline_outputs"]
    backward_args = {
        "grad_scaler": grad_scaler,
        "model": model,
        "deallocate_pipeline_outputs": deallocate_pipeline_outputs,
    }
    grad_sync_func = builder.constant_data["grad_sync_func"]
    input_tensors = builder.user_data["input_tensors"]
    output_tensors = builder.user_data["output_tensors"]
    output_tensor_grads = builder.user_data["output_tensor_grads"]
    synchronized_model_chunks = builder.user_data["synchronized_model_chunks"]

    input_tensor_grad = inst.run(
        input_tensors=input_tensors,
        output_tensors=output_tensors,
        output_tensor_grads=output_tensor_grads,
        grad_sync_func=grad_sync_func,
        synchronized_model_chunks=synchronized_model_chunks,
        kwargs=backward_args,
    )
    builder.user_data["input_tensor_grad"] = input_tensor_grad


@register_instruction(name="vescale_interleavd_1f1b_set_output_to_none")
def vpp_set_output_to_none():
    inst = builder.user_data["inst"]
    output_tensor = inst.run()
    builder.user_data["output_tensor"] = None


@register_instruction(name="vescale_interleavd_1f1b_set_input_grad_to_none")
def vpp_set_input_grad_to_none():
    inst = builder.user_data["inst"]
    input_tensor_grad = inst.run()
    builder.user_data["input_tensor_grad"] = input_tensor_grad


@register_instruction(name="vescale_interleaved_1f1b_send_forward_recv_forward")
def vpp_send_forward_recv_forward():
    inst = builder.user_data["inst"]
    output_tensor = builder.user_data["output_tensor"]
    input_tensor = inst.run(output_tensor=output_tensor)
    if inst.overlap_p2p_comm:
        input_tensor, fwd_wait_handles = input_tensor
        builder.user_data["fwd_wait_handles"] = fwd_wait_handles
    builder.user_data["input_tensor"] = input_tensor


@register_instruction(name="vescale_interleavd_1f1b_send_backward_recv_backward")
def vpp_send_backward_recv_backward():
    inst = builder.user_data["inst"]
    input_tensor_grad = builder.user_data["input_tensor_grad"]
    output_tensor_grad = inst.run(input_tensor_grad=input_tensor_grad)
    if inst.overlap_p2p_comm:
        output_tensor_grad, bwd_wait_handles = output_tensor_grad
        builder.user_data["bwd_wait_handles"] = bwd_wait_handles
    builder.user_data["output_tensor_grad"] = output_tensor_grad


@register_instruction(name="vescale_interleaved_1f1b_send_forward_backward_recv_forward_backward")
def vpp_send_forward_backward_recv_forward_backward():
    inst = builder.user_data["inst"]
    output_tensor = builder.user_data["output_tensor"]
    input_tensor_grad = builder.user_data["input_tensor_grad"]
    input_tensor, output_tensor_grad = inst.run(output_tensor=output_tensor, input_tensor_grad=input_tensor_grad)
    builder.user_data["input_tensor"] = input_tensor
    builder.user_data["output_tensor_grad"] = output_tensor_grad


@register_instruction(name="vescale_interleavd_1f1b_append_grads")
def vpp_append_grads():
    inst = builder.user_data["inst"]
    output_tensor_grads = builder.user_data["output_tensor_grads"]
    output_tensor_grad = builder.user_data["output_tensor_grad"]
    inst.run(output_tensor_grad, output_tensor_grads)


@register_instruction(name="vescale_interleavd_1f1b_append_inputs")
def vpp_append_inputs():
    inst = builder.user_data["inst"]
    input_tensor = builder.user_data["input_tensor"]
    input_tensors = builder.user_data["input_tensors"]
    inst.run(input_tensor, input_tensors)


@register_instruction(name="vescale_interleavd_1f1b_deallocate_output_tensor")
def vpp_deallocate_tensors():
    inst = builder.user_data["inst"]
    deallocate_pipeline_outputs = builder.constant_data["deallocate_pipeline_outputs"]
    output_tensor = builder.user_data["output_tensor"]
    inst.run(output_tensor=output_tensor, deallocate_pipeline_outputs=deallocate_pipeline_outputs)


@register_instruction(name="vescale_interleaved_1f1b_drain_send_reqs")
def vpp_drain_send_reqs():
    inst = builder.user_data["inst"]
    inst.run()


@register_instruction(name="vescale_interleaved_1f1b_drain_recv_reqs")
def vpp_drain_recv_reqs():
    inst = builder.user_data["inst"]
    bwd_wait_handles = builder.user_data["bwd_wait_handles"]
    inst.run(bwd_wait_handles=bwd_wait_handles)


@register_instruction(name="vescale_interleaved_1f1b_wait_fwd")
def vpp_wait_fwd():
    inst = builder.user_data["inst"]
    fwd_wait_handles = builder.user_data["fwd_wait_handles"]
    inst.run(fwd_wait_handles=fwd_wait_handles)


@register_instruction(name="vescale_interleavd_1f1b_launch_shared_units_sync")
def vpp_launch_shared_units_sync():
    model = builder.model
    inst = builder.user_data["inst"]
    inst.run(model=model)


@register_instruction(name="vescale_interleaved_1f1b_pre_forward_data")
def vpp_prepare_forward_args():
    fn = builder.user_data["prepare_data_fn"]
    return fn()


@register_instruction(name="vescale_interleaved_1f1b_forward")
def forward_fn(p2p_input, local_input):
    model_chunk_id = builder.user_data["model_chunk_id"]
    if isinstance(builder.model, Sequence):

        def _feed_input(data):
            if isinstance(data, Sequence):
                return model(*data)
            elif isinstance(data, Dict):
                return model(**data)
            else:
                return model(data)

        model = builder.model[model_chunk_id]
        if p2p_input is not None:
            return _feed_input(p2p_input)
        else:
            return _feed_input(local_input)
    else:
        return builder.model(p2p_input, local_input, model_chunk_id)


@register_instruction(name="vescale_interleaved_1f1b_loss_fn")
def loss_fn():
    loss_func = builder.loss_fn
    output_tensor = builder.user_data["output_tensor"]
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


VESACLE_INSTRUCTION_MAPPING_V = {
    RECV_FORWARD: "vescale_interleavd_1f1b_recv_forward",
    FWD: "vescale_interleavd_1f1b_forward",
    BWD: "vescale_interleaved_1f1b_backward",
    SET_OUTPUT_TO_NONE: "vescale_interleavd_1f1b_set_output_to_none",
    SET_INPUTGRAD_TO_NONE: "vescale_interleavd_1f1b_set_input_grad_to_none",
    SEND_FORWARD_RECV_FORWARD: "vescale_interleaved_1f1b_send_forward_recv_forward",
    SEND_BACKWARD_RECV_BACKWARD: "vescale_interleavd_1f1b_send_backward_recv_backward",
    SEND_FORWARD_BACKWARD_RECV_FORWARD_BACKWARD: "vescale_interleaved_1f1b_send_forward_backward_recv_forward_backward",
    APPEND_GRADS: "vescale_interleavd_1f1b_append_grads",
    APPEND_INPUTS: "vescale_interleavd_1f1b_append_inputs",
    DEALLOCATE_OUTPUT_TENSOR: "vescale_interleavd_1f1b_deallocate_output_tensor",
    DRAIN_SEND_REQS: "vescale_interleaved_1f1b_drain_send_reqs",
    DRAIN_RECV_REQS: "vescale_interleaved_1f1b_drain_recv_reqs",
    WAIT_FWD: "vescale_interleaved_1f1b_wait_fwd",
    LAUNCH_SHARED_UNITS_SYNC: "vescale_interleavd_1f1b_launch_shared_units_sync",
}
