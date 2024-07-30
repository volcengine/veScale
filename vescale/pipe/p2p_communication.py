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
# Some code are adapted p2p_communication.py in Megatron-LM.
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
################################################################################

from enum import Enum
import os
import torch
import torch.distributed as dist
from vescale.dtensor.dtensor import DTensor
from vescale.dtensor.device_mesh import DeviceMesh
from vescale.ndtimeline import ndtimeit_p2p
from vescale.ndtimeline.predefined import (
    RECV_FORWARD,
    RECV_BACKWARD,
    SEND_FORWARD,
    SEND_BACKWARD,
    SEND_FORWARD_RECV_BACKWARD,
    SEND_BACKWARD_RECV_FORWARD,
)
from typing import Optional, List, Union, Tuple
from torch.distributed.distributed_c10d import ProcessGroup

try:
    from torch.distributed.distributed_c10d import _coalescing_manager
except ImportError:
    print("Warning: cannot import coalescing_manager. It may impact PP performance")

# Types
Shape = Union[List[int], torch.Size]
# For P2P overlap, currently we do not differ fwd/bwd reqs;
# Hence, drain func will sync both fwd and bwd p2p ops.
GLOBAL_COUNTER = 0
INTERMEDIATE_SHAPES = []
MINIBATCH_STEPS = 0


def reset_global_counter():
    global GLOBAL_COUNTER
    global MINIBATCH_STEPS
    GLOBAL_COUNTER = 0
    MINIBATCH_STEPS += 1


class OpType(Enum):
    SEND, RECV_FWD, RECV_BWD = 0, 1, 2


p2p_overlap = False
send_reqs = []
recv_fwd_reqs = []
recv_bwd_reqs = []


# Sync P2P-send OP
def drain_send_reqs():
    global send_reqs
    if len(send_reqs) == 0:
        return
    for req in send_reqs:
        req.wait()
    send_reqs.clear()


# Sync P2P-recv OP: we differ forward recv reqs from backward recv reqs
# to enable 1F1B P2P communication overlap
def drain_recv_reqs(drain_type="all"):
    global recv_fwd_reqs, recv_bwd_reqs
    if drain_type == "all" or drain_type == "forward":
        if len(recv_fwd_reqs) > 0:
            for req in recv_fwd_reqs:
                req.wait()
            recv_fwd_reqs.clear()
    if drain_type == "all" or drain_type == "backward":
        if len(recv_bwd_reqs) > 0:
            for req in recv_bwd_reqs:
                req.wait()
            recv_bwd_reqs.clear()


def _mapping_local_rank_to_target_rank_by_device_mesh(
    *, current_device_mesh: DeviceMesh, target_device_mesh: DeviceMesh, local_rank: int
):
    """Mapping local rank in current device mesh to find target rank in target device mesh

    Takes the following arguments:
        current_device_mesh: current device mesh for locate rank position
        target_device_mesh: target device mesh for mapping to target rank
    Returns:
        target_rank
    """
    if target_device_mesh is None:
        return None
    current_device_mesh_list = current_device_mesh.mesh.view(-1).tolist()
    assert local_rank in current_device_mesh_list
    current_rank_pos = current_device_mesh_list.index(local_rank)
    target_rank = target_device_mesh.mesh.view(-1).tolist()[current_rank_pos]
    return target_rank


def _get_p2p_send_recv_process_group(
    *, current_device_mesh: DeviceMesh, target_device_mesh: DeviceMesh, local_rank: int
):
    target_rank = _mapping_local_rank_to_target_rank_by_device_mesh(
        current_device_mesh=current_device_mesh, target_device_mesh=target_device_mesh
    )
    return list(local_rank, target_rank)


def _communicate_shapes(
    *,
    tensor_send_next: torch.tensor,
    tensor_send_prev: torch.tensor,
    prev_rank: int,
    next_rank: int,
    recv_prev: bool,
    recv_next: bool,
    local_rank: int,
    shape_dim: int = 3,
):
    """Communicate tensor shapes between stages. Used to communicate
    tensor shapes before the actual tensor communication happens.
    This is required when the sequence lengths across micro batches
    are not uniform.

    Takes the following arguments:
        tensor_send_next: DTensor or torch.tensor to send to next rank (no tensor sent if
                          set to None).
        tensor_send_prev: DTensor or torch.tensor to send to prev rank (no tensor sent if
                          set to None).
        prev_rank: prev rank for send/recv rank
        next_rank: next rank for send/recv rank
        recv_prev: boolean for whether tensor should be received from
                   previous rank.
        recv_next: boolean for whether tensor should be received from
                   next rank.
        shape_dim: default to 3, which is set in megatron, in this refactor func, you can set shape dim
    Returns:
        (recv_prev_shape, recv_next_shape)
    """

    recv_prev_shape_tensor = None
    recv_next_shape_tensor = None
    send_prev_shape_tensor = None
    send_next_shape_tensor = None

    if recv_prev:
        recv_prev_shape_tensor = torch.empty((shape_dim), device=torch.cuda.current_device(), dtype=torch.int64)
    if recv_next:
        recv_next_shape_tensor = torch.empty((shape_dim), device=torch.cuda.current_device(), dtype=torch.int64)
    if tensor_send_prev is not None:
        if isinstance(tensor_send_prev, DTensor):
            send_prev_shape_tensor = torch.tensor(
                tensor_send_prev._local_tensor.size(), device=torch.cuda.current_device(), dtype=torch.int64
            )
        else:
            send_prev_shape_tensor = torch.tensor(
                tensor_send_prev.size(), device=torch.cuda.current_device(), dtype=torch.int64
            )
    if tensor_send_next is not None:
        if isinstance(tensor_send_next, DTensor):
            send_next_shape_tensor = torch.tensor(
                tensor_send_next._local_tensor.size(), device=torch.cuda.current_device(), dtype=torch.int64
            )
        else:
            send_next_shape_tensor = torch.tensor(
                tensor_send_next.size(), device=torch.cuda.current_device(), dtype=torch.int64
            )
    ops = []
    if send_prev_shape_tensor is not None:
        send_prev_op = torch.distributed.P2POp(torch.distributed.isend, send_prev_shape_tensor, prev_rank)
        ops.append(send_prev_op)
    if recv_next_shape_tensor is not None:
        recv_next_op = torch.distributed.P2POp(torch.distributed.irecv, recv_next_shape_tensor, next_rank)
        ops.append(recv_next_op)
    if send_next_shape_tensor is not None:
        send_next_op = torch.distributed.P2POp(torch.distributed.isend, send_next_shape_tensor, next_rank)
        ops.append(send_next_op)
    if recv_prev_shape_tensor is not None:
        recv_prev_op = torch.distributed.P2POp(torch.distributed.irecv, recv_prev_shape_tensor, prev_rank)
        ops.append(recv_prev_op)

    if len(ops) > 0:
        reqs = torch.distributed.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

    # To protect against race condition when using batch_isend_irecv().
    # should take this out once the bug with batch_isend_irecv is resolved.
    if not _coalescing_manager:
        torch.cuda.synchronize()

    recv_prev_shape = [0, 0, 0]
    if recv_prev_shape_tensor is not None:
        recv_prev_shape = recv_prev_shape_tensor.tolist()

    recv_next_shape = [0, 0, 0]
    if recv_next_shape_tensor is not None:
        recv_next_shape = recv_next_shape_tensor.tolist()

    return recv_prev_shape, recv_next_shape


def _batched_p2p_ops(
    *,
    tensor_send_prev: Optional[torch.Tensor],
    tensor_recv_prev: Optional[torch.Tensor],
    tensor_send_next: Optional[torch.Tensor],
    tensor_recv_next: Optional[torch.Tensor],
    prev_rank: int,
    next_rank: int,
    group: torch.distributed.ProcessGroup,
    local_rank: int,
    send_tensor_shape_unpad: Shape = None,
    p2p_overlap=False,
):
    ops = []
    if tensor_send_prev is not None:
        send_prev_op = torch.distributed.P2POp(torch.distributed.isend, tensor_send_prev, prev_rank)
        ops.append(send_prev_op)
    if tensor_recv_prev is not None:
        recv_prev_op = torch.distributed.P2POp(torch.distributed.irecv, tensor_recv_prev, prev_rank)
        ops.append(recv_prev_op)
    if tensor_send_next is not None:
        send_next_op = torch.distributed.P2POp(torch.distributed.isend, tensor_send_next, next_rank)
        ops.append(send_next_op)
    if tensor_recv_next is not None:
        recv_next_op = torch.distributed.P2POp(torch.distributed.irecv, tensor_recv_next, next_rank)
        ops.append(recv_next_op)
    if len(ops) > 0:
        reqs = torch.distributed.batch_isend_irecv(ops)
    else:
        reqs = []
    return reqs


def check_nan(tensor_list, check=False):
    if check:
        for t in tensor_list:
            assert not torch.isnan(t).any(), (
                "tensor shape: "
                + str(t.shape)
                + ", dtype: "
                + str(t.dtype)
                + ", device: "
                + str(t.device)
                + ", # of NaN elements: "
                + str(torch.sum(torch.isnan(t)).item())
                + ", NaN element indexes: "
                + str(torch.isnan(t).nonzero())
            )


def _p2p_ops(
    *,
    tensor_send_prev: Optional[torch.Tensor],
    tensor_recv_prev: Optional[torch.Tensor],
    tensor_send_next: Optional[torch.Tensor],
    tensor_recv_next: Optional[torch.Tensor],
    prev_rank: int,
    next_rank: int,
    group: torch.distributed.ProcessGroup,
    local_rank: int,
    p2p_overlap=False,
    send_tensor_shape_unpad: Shape = None,
    # file=None,
):
    reqs = []

    """
        by now the megatron pingpong
        send recv is not supported because the global
        devicemeshmanager is not impled. we will use
        the ucx and mpi two-end no-blocking api to do
        the send recv
    """
    stage_id = int(os.environ.get("STAGE_ID", "0"))
    op_type = []
    if stage_id % 2:
        if tensor_send_next is not None:
            if send_tensor_shape_unpad is not None:
                assert (
                    send_tensor_shape_unpad[0] <= tensor_send_next.shape[0]
                ), f"{send_tensor_shape_unpad} vs {tensor_send_next.shape}"
                check_nan([tensor_send_next[: send_tensor_shape_unpad[0]]])
            else:
                check_nan([tensor_send_next])
            send_next_req = torch.distributed.isend(
                tensor=tensor_send_next,
                dst=next_rank,
                group=group,
            )
            reqs.append(send_next_req)
            op_type.append(OpType.SEND)

        if tensor_recv_prev is not None:
            recv_prev_req = torch.distributed.irecv(
                tensor=tensor_recv_prev,
                src=prev_rank,
                group=group,
            )
            reqs.append(recv_prev_req)
            op_type.append(OpType.RECV_FWD)

        if tensor_send_prev is not None:
            if send_tensor_shape_unpad is not None:
                assert (
                    send_tensor_shape_unpad[0] <= tensor_send_prev.shape[0]
                ), f"{send_tensor_shape_unpad} vs {tensor_send_prev.shape}"
                check_nan([tensor_send_prev[: send_tensor_shape_unpad[0]]])
            else:
                check_nan([tensor_send_prev])

            send_prev_req = torch.distributed.isend(
                tensor=tensor_send_prev,
                dst=prev_rank,
                group=group,
            )
            reqs.append(send_prev_req)
            op_type.append(OpType.SEND)

        if tensor_recv_next is not None:
            recv_next_req = torch.distributed.irecv(
                tensor=tensor_recv_next,
                src=next_rank,
                group=group,
            )
            reqs.append(recv_next_req)
            op_type.append(OpType.RECV_BWD)

    else:
        if tensor_recv_prev is not None:
            recv_prev_req = torch.distributed.irecv(
                tensor=tensor_recv_prev,
                src=prev_rank,
                group=group,
            )
            reqs.append(recv_prev_req)
            op_type.append(OpType.RECV_FWD)
        if tensor_send_next is not None:
            if send_tensor_shape_unpad is not None:
                assert (
                    send_tensor_shape_unpad[0] <= tensor_send_next.shape[0]
                ), f"{send_tensor_shape_unpad} vs {tensor_send_next.shape}"
                check_nan([tensor_send_next[: send_tensor_shape_unpad[0]]])
            else:
                check_nan([tensor_send_next])
            send_next_req = torch.distributed.isend(
                tensor=tensor_send_next,
                dst=next_rank,
                group=group,
            )
            reqs.append(send_next_req)
            op_type.append(OpType.SEND)

        if tensor_recv_next is not None:
            recv_next_req = torch.distributed.irecv(
                tensor=tensor_recv_next,
                src=next_rank,
                group=group,
            )
            reqs.append(recv_next_req)
            op_type.append(OpType.RECV_BWD)

        if tensor_send_prev is not None:
            if send_tensor_shape_unpad is not None:
                assert (
                    send_tensor_shape_unpad[0] <= tensor_send_prev.shape[0]
                ), f"{send_tensor_shape_unpad} vs {tensor_send_prev.shape}"
                check_nan([tensor_send_prev[: send_tensor_shape_unpad[0]]])
            else:
                check_nan([tensor_send_prev])

            send_prev_req = torch.distributed.isend(
                tensor=tensor_send_prev,
                dst=prev_rank,
                group=group,
            )
            reqs.append(send_prev_req)
            op_type.append(OpType.SEND)

    if p2p_overlap:
        # For P2P-comm overlap
        global send_reqs, recv_fwd_reqs, recv_bwd_reqs
        for i in range(len(op_type)):
            if op_type[i] == OpType.SEND:
                send_reqs.append(reqs[i])
            elif op_type[i] == OpType.RECV_FWD:
                recv_fwd_reqs.append(reqs[i])
            elif op_type[i] == OpType.RECV_BWD:
                recv_bwd_reqs.append(reqs[i])

    return reqs


def _communicate(
    *,
    tensor_send_next: Optional[torch.Tensor],
    tensor_send_prev: Optional[torch.Tensor],
    recv_prev: bool,
    recv_next: bool,
    current_device_mesh: DeviceMesh,
    prev_device_mesh: DeviceMesh = None,
    next_device_mesh: DeviceMesh = None,
    tensor_shape: Shape = None,
    send_tensor_shape_unpad: Shape = None,
    batch_p2p_comm: bool = True,
    wait_on_reqs: bool = True,
    dtype: Optional[torch.dtype],
    group: ProcessGroup = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Communicate tensors between stages. Used as helper method in other
    communication methods that are used in vescale/schedules.py.

    Arguments:
        tensor_send_next (torch.Tensor, optional):
            Tensor to send to next rank (no tensor sent if None)

        tensor_send_prev (torch.Tensor, optional):
            Tensor to send to prev rank (no tensor sent if None)

        current_device_mesh (DeviceMesh, required):
            Current device mesh for locate rank position

        prev_device_mesh (DeviceMesh, required):
            Target device mesh for mapping to pre rank

        next_device_mesh (DeviceMesh, required):
            Target device mesh for mapping to next rank

        recv_prev (boolean, required):
            whether tensor should be received from previous rank.

        recv_next (boolean, required):
            whether tensor should be received from next rank.

        tensor_shape (List[int] or torch.Size, required):
            shape of tensor to receive (this method assumes that all
            tensors sent and received in a single function call are
            the same shape). If none, using dynamic shape

        batch_p2p_comm (boolean, required):
            If true use batch_isend_irecv, otherwise use individual
            isend and irecv calls.

        wait_on_reqs (boolean, optional, default=False):
            For non-batched p2p communication, wait on each request
            before returning.

        dtype (torch.dtype, required if either recv_{prev,next} is True):
            this must be the type of the tensors that will be
            received, will typically be params_dtype, but in the case
            of fp32 residual connections might be torch.float.

        variable_seq_lengths (bool, optional, default=False):
            Support for variable sequence lengths across
            microbatches. Setting this communicates the size of
            tensors during pipeline parallelism communication, because
            of this extra overhead it should only be set if the
            sequence length is not constant during training.

    Returns:
        tuple containing

        - tensor_recv_prev: torch.Tensor if recv_prev is True, None otherwise.
        - tensor_recv_next: torch.Tensor if recv_next is True, None otherwise.

    """
    # Init p2p_overlap: Use a global var to enable p2p comm overlap,
    # so as not to change the original APIs

    global p2p_overlap
    if not wait_on_reqs and not p2p_overlap:
        p2p_overlap = True

    # Create placeholder tensors for receive in forward and backward directions
    # if needed.
    tensor_recv_prev = None
    tensor_recv_next = None

    # This will come from config in the next version, for now hard
    # code it here to match existing functionality.
    batch_p2p_sync = True
    local_rank = current_device_mesh.get_rank()
    # parse current device mesh and target device mesh
    prev_rank = _mapping_local_rank_to_target_rank_by_device_mesh(
        local_rank=local_rank, current_device_mesh=current_device_mesh, target_device_mesh=prev_device_mesh
    )
    next_rank = _mapping_local_rank_to_target_rank_by_device_mesh(
        local_rank=local_rank, current_device_mesh=current_device_mesh, target_device_mesh=next_device_mesh
    )
    # flag to reuse intermediate tensor shapes of recorded tensors in first minibatch
    reuse_intermediate_shapes = os.environ.get("REUSE_COMM_SHAPE", "0") == "1"

    if tensor_shape is not None:
        recv_prev_shape = tensor_shape
        recv_next_shape = tensor_shape
    else:
        global GLOBAL_COUNTER
        global INTERMEDIATE_SHAPES
        global MINIBATCH_STEPS
        if reuse_intermediate_shapes and MINIBATCH_STEPS > 1:
            recv_prev_shape, recv_next_shape = INTERMEDIATE_SHAPES[GLOBAL_COUNTER]
        else:
            recv_prev_shape, recv_next_shape = _communicate_shapes(
                tensor_send_next=tensor_send_next,
                tensor_send_prev=tensor_send_prev,
                recv_prev=recv_prev,
                recv_next=recv_next,
                prev_rank=prev_rank,
                next_rank=next_rank,
                local_rank=local_rank,
            )
            if reuse_intermediate_shapes:
                INTERMEDIATE_SHAPES.append((recv_prev_shape, recv_next_shape))
        GLOBAL_COUNTER += 1

    if recv_prev:
        if dtype is None:
            raise RuntimeError("dtype must be provided if recv_prev is True")
        if recv_prev_shape is None:
            raise RuntimeError(
                "tensor_shape must be specified if recv_prev is True. "
                "Common tensor_shape is (seq_length, micro_batch_size, hidden_size)"
            )
        tensor_recv_prev = torch.empty(
            recv_prev_shape, requires_grad=True, device=torch.cuda.current_device(), dtype=dtype
        )
    if recv_next:
        if dtype is None:
            raise RuntimeError("dtype must be provided if recv_next is True")
        if recv_next_shape is None:
            raise RuntimeError(
                "tensor_shape must be specified if recv_next is True. "
                "Common tensor_shape is (seq_length, micro_batch_size, hidden_size)"
            )
        tensor_recv_next = torch.empty(
            recv_next_shape, requires_grad=True, device=torch.cuda.current_device(), dtype=dtype
        )

    # Send tensors in both the forward and backward directions as appropriate.
    if batch_p2p_comm:
        assert wait_on_reqs
        p2p_func = _batched_p2p_ops
    else:
        p2p_func = _p2p_ops

    # if file:
    #     file.write(
    #         f"\np2p tensor_send_prev:{tensor_send_prev}, tensor_recv_prev:{tensor_recv_prev} {id(tensor_recv_prev)}, tensor_send_next:{tensor_send_next} {id(tensor_send_next)}, tensor_recv_next:{tensor_recv_next}, prev_rank: {prev_rank}, next_rank: {next_rank}, local_rank: {local_rank}\n"
    #     )
    #     file.flush()
    reqs = p2p_func(
        tensor_send_prev=tensor_send_prev,
        tensor_recv_prev=tensor_recv_prev,
        tensor_send_next=tensor_send_next,
        tensor_recv_next=tensor_recv_next,
        prev_rank=prev_rank,
        next_rank=next_rank,
        group=group,
        local_rank=local_rank,
        send_tensor_shape_unpad=send_tensor_shape_unpad,
        p2p_overlap=p2p_overlap,
    )

    if wait_on_reqs and len(reqs) > 0:
        for req in reqs:
            req.wait()
        reqs = None

    if batch_p2p_comm and batch_p2p_sync:
        # To protect against race condition when using batch_isend_irecv().
        if not _coalescing_manager:
            torch.cuda.synchronize()

    return tensor_recv_prev, tensor_recv_next, reqs


def recv_forward(
    tensor_shape: Shape,
    recv_dtype: torch.dtype,
    current_device_mesh: DeviceMesh,
    peer_device_mesh: Optional[DeviceMesh] = None,
    batch_p2p_comm: bool = True,
) -> torch.Tensor:
    """Receive tensor from previous rank in pipeline (forward receive).

    See _communicate for argument details.

    Args:
        tensor_shape (Shape): shape of imminenently arrived tensors
        recv_dtype (torch.dtype): data types of received tensors
        current_device_mesh (DeviceMesh): sub-DeviceMesh of current stage
        peer_device_mesh (Optional[DeviceMesh]): sub-DeviceMesh of sender/recipient stage
        batch_p2p_comm (bool): switch to execute batched p2p transfer when turned on

    Returns:
        Received forward tensor

    """
    if peer_device_mesh is None:
        intput_tensor = None
        return intput_tensor
    prev_rank = _mapping_local_rank_to_target_rank_by_device_mesh(
        local_rank=current_device_mesh.get_rank(),
        current_device_mesh=current_device_mesh,
        target_device_mesh=peer_device_mesh,
    )
    with ndtimeit_p2p(RECV_FORWARD, dist.group.WORLD, prev_rank, batch_p2p_comm):
        input_tensor, _, _ = _communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            current_device_mesh=current_device_mesh,
            prev_device_mesh=peer_device_mesh,
            recv_prev=True,
            recv_next=False,
            tensor_shape=tensor_shape,
            batch_p2p_comm=batch_p2p_comm,
            dtype=recv_dtype,
        )
    return input_tensor


def recv_backward(
    tensor_shape: Shape,
    recv_dtype: torch.dtype,
    current_device_mesh: DeviceMesh,
    peer_device_mesh: Optional[DeviceMesh] = None,
    batch_p2p_comm: bool = True,
) -> torch.Tensor:
    """Receive tensor from next rank in pipeline (backward receive).

    See _communicate for argument details.

    Args:
        tensor_shape (Shape): shape of imminenently arrived tensors
        recv_dtype (torch.dtype): data types of received tensors
        current_device_mesh (DeviceMesh): sub-DeviceMesh of current stage
        peer_device_mesh (Optional[DeviceMesh]): sub-DeviceMesh of sender/recipient stage
        batch_p2p_comm (bool): switch to execute batched p2p transfer when turned on

    Returns:
        Received output tensor gradient.

    """
    if peer_device_mesh is None:
        output_tensor_grad = None
        return output_tensor_grad
    next_rank = _mapping_local_rank_to_target_rank_by_device_mesh(
        local_rank=current_device_mesh.get_rank(),
        current_device_mesh=current_device_mesh,
        target_device_mesh=peer_device_mesh,
    )
    with ndtimeit_p2p(RECV_BACKWARD, dist.group.WORLD, next_rank, batch_p2p_comm):
        _, output_tensor_grad, _ = _communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            current_device_mesh=current_device_mesh,
            next_device_mesh=peer_device_mesh,
            recv_prev=False,
            recv_next=True,
            tensor_shape=tensor_shape,
            dtype=recv_dtype,
            batch_p2p_comm=batch_p2p_comm,
        )
    return output_tensor_grad


def send_forward(
    output_tensor: torch.Tensor,
    current_device_mesh: DeviceMesh,
    peer_device_mesh: Optional[DeviceMesh] = None,
    tensor_shape: Optional[Shape] = None,
    batch_p2p_comm: bool = True,
) -> None:
    """Send tensor to next rank in pipeline (forward send).

    See _communicate for argument details.

    Args:
        output_tensor (torch.Tensor): backward input received from previous stage
        current_device_mesh (DeviceMesh): sub-DeviceMesh of current stage
        peer_device_mesh (Optional[DeviceMesh]): sub-DeviceMesh of sender/recipient stage
        tensor_shape (Shape): shape of imminenently arrived tensors
        batch_p2p_comm (bool): switch to execute batched p2p transfer when turned on

    """

    if peer_device_mesh is None:
        return
    next_rank = _mapping_local_rank_to_target_rank_by_device_mesh(
        local_rank=current_device_mesh.get_rank(),
        current_device_mesh=current_device_mesh,
        target_device_mesh=peer_device_mesh,
    )
    with ndtimeit_p2p(SEND_FORWARD, dist.group.WORLD, next_rank, batch_p2p_comm):
        _communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            current_device_mesh=current_device_mesh,
            next_device_mesh=peer_device_mesh,
            recv_prev=False,
            recv_next=False,
            tensor_shape=tensor_shape,
            batch_p2p_comm=batch_p2p_comm,
            dtype=None,
        )


def send_backward(
    input_tensor_grad: torch.Tensor,
    current_device_mesh: DeviceMesh,
    peer_device_mesh: Optional[DeviceMesh] = None,
    tensor_shape: Optional[Shape] = None,
    batch_p2p_comm: bool = True,
) -> None:
    """Send tensor to previous rank in pipeline (backward send).

    See _communicate for argument details.

    Args:
        input_tensor_grad (torch.Tensor): input tensor gradients
        current_device_mesh (DeviceMesh): sub-DeviceMesh of current stage
        peer_device_mesh (Optional[DeviceMesh]): sub-DeviceMesh of sender/recipient stage
        tensor_shape (Shape): shape of imminenently arrived tensors
        batch_p2p_comm (bool): switch to execute batched p2p transfer when turned on

    """

    if peer_device_mesh is None:
        return
    prev_rank = _mapping_local_rank_to_target_rank_by_device_mesh(
        local_rank=current_device_mesh.get_rank(),
        current_device_mesh=current_device_mesh,
        target_device_mesh=peer_device_mesh,
    )
    with ndtimeit_p2p(SEND_BACKWARD, dist.group.WORLD, prev_rank, batch_p2p_comm):
        _communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad,
            current_device_mesh=current_device_mesh,
            prev_device_mesh=peer_device_mesh,
            recv_prev=False,
            recv_next=False,
            tensor_shape=tensor_shape,
            batch_p2p_comm=batch_p2p_comm,
            dtype=None,
        )


def send_forward_recv_backward(
    output_tensor: torch.Tensor,
    tensor_shape: Shape,
    recv_dtype: torch.dtype,
    current_device_mesh: DeviceMesh,
    peer_device_mesh: Optional[DeviceMesh] = None,
    batch_p2p_comm: bool = True,
) -> torch.Tensor:
    """Batched send and recv with next rank in pipeline.

    See _communicate for argument details.

    Args:
        output_tensor (torch.Tensor): backward input received from previous stage
        tensor_shape (Shape): shape of imminenently arrived tensors
        recv_dtype (torch.dtype): data types of received tensors
        current_device_mesh (DeviceMesh): sub-DeviceMesh of current stage
        peer_device_mesh (Optional[DeviceMesh]): sub-DeviceMesh of sender/recipient stage
        batch_p2p_comm (bool): switch to execute batched p2p transfer when turned on

    Returns:
        Received output tensor gradients.

    """

    if peer_device_mesh is None:
        output_tensor_grad = None
        return output_tensor_grad
    next_rank = _mapping_local_rank_to_target_rank_by_device_mesh(
        local_rank=current_device_mesh.get_rank(),
        current_device_mesh=current_device_mesh,
        target_device_mesh=peer_device_mesh,
    )
    with ndtimeit_p2p(SEND_FORWARD_RECV_BACKWARD, dist.group.WORLD, next_rank, batch_p2p_comm):
        _, output_tensor_grad, _ = _communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            current_device_mesh=current_device_mesh,
            next_device_mesh=peer_device_mesh,
            recv_prev=False,
            recv_next=True,
            tensor_shape=tensor_shape,
            dtype=recv_dtype,
            batch_p2p_comm=batch_p2p_comm,
        )
    return output_tensor_grad


def send_backward_recv_forward(
    input_tensor_grad: torch.Tensor,
    tensor_shape: Shape,
    recv_dtype: torch.dtype,
    current_device_mesh: DeviceMesh,
    peer_device_mesh: Optional[DeviceMesh] = None,
    batch_p2p_comm: bool = True,
) -> torch.Tensor:
    """
    Batched send and recv with previous rank in pipeline.

    See _communicate for argument details.

    Args:
        input_tensor_grad (torch.Tensor): input tensor gradients
        tensor_shape (Shape): shape of imminenently arrived tensors
        recv_dtype (torch.dtype): data types of received tensors
        current_device_mesh (DeviceMesh): sub-DeviceMesh of current stage
        peer_device_mesh (Optional[DeviceMesh]): sub-DeviceMesh of sender/recipient stage
        batch_p2p_comm (bool): switch to execute batched p2p transfer when turned on

    Returns:
        Received tensor.

    """
    if peer_device_mesh is None:
        input_tensor = None
        return input_tensor
    prev_rank = _mapping_local_rank_to_target_rank_by_device_mesh(
        local_rank=current_device_mesh.get_rank(),
        current_device_mesh=current_device_mesh,
        target_device_mesh=peer_device_mesh,
    )
    with ndtimeit_p2p(SEND_BACKWARD_RECV_FORWARD, dist.group.WORLD, prev_rank, batch_p2p_comm):
        input_tensor, _, _ = _communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad,
            current_device_mesh=current_device_mesh,
            prev_device_mesh=peer_device_mesh,
            recv_prev=True,
            recv_next=False,
            tensor_shape=tensor_shape,
            dtype=recv_dtype,
            batch_p2p_comm=batch_p2p_comm,
        )
    return input_tensor


def send_forward_recv_forward(
    output_tensor: torch.Tensor,
    recv_prev: bool,
    tensor_shape: Shape,
    current_device_mesh: DeviceMesh,
    prev_device_mesh: DeviceMesh,
    next_device_mesh: DeviceMesh,
    send_tensor_shape_unpad: Shape = None,
    overlap_p2p_comm: bool = False,
    recv_dtype: Optional[torch.dtype] = None,
    batch_p2p_comm: bool = True,
    group: ProcessGroup = None,
) -> torch.Tensor:
    """Batched recv from previous rank and send to next rank in pipeline.

    See _communicate for argument details.
    """
    # auto state change
    if prev_device_mesh is None:
        recv_prev = False
    if next_device_mesh is None:
        input_tensor, _, wait_handles = _communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            current_device_mesh=current_device_mesh,
            prev_device_mesh=prev_device_mesh,
            next_device_mesh=next_device_mesh,
            recv_prev=recv_prev,
            recv_next=False,
            tensor_shape=tensor_shape,
            send_tensor_shape_unpad=send_tensor_shape_unpad,
            batch_p2p_comm=batch_p2p_comm,
            wait_on_reqs=(not overlap_p2p_comm),
            dtype=recv_dtype,
            group=group,
        )
    else:
        input_tensor, _, wait_handles = _communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            current_device_mesh=current_device_mesh,
            prev_device_mesh=prev_device_mesh,
            next_device_mesh=next_device_mesh,
            recv_prev=recv_prev,
            recv_next=False,
            tensor_shape=tensor_shape,
            send_tensor_shape_unpad=send_tensor_shape_unpad,
            batch_p2p_comm=batch_p2p_comm,
            wait_on_reqs=(not overlap_p2p_comm),
            dtype=recv_dtype,
            group=group,
        )
    if overlap_p2p_comm:
        return input_tensor, wait_handles
    return input_tensor


def send_backward_recv_backward(
    input_tensor_grad: torch.Tensor,
    recv_next: bool,
    tensor_shape: Shape,
    current_device_mesh: DeviceMesh,
    prev_device_mesh: DeviceMesh,
    next_device_mesh: DeviceMesh,
    send_tensor_shape_unpad: Shape = None,
    overlap_p2p_comm: bool = False,
    recv_dtype: Optional[torch.dtype] = None,
    batch_p2p_comm: bool = True,
    group: ProcessGroup = None,
) -> torch.Tensor:
    """Batched recv from next rank and send to previous rank in pipeline.

    See _communicate for argument details.
    """
    # auto state change
    if next_device_mesh is None:
        recv_next = False
    if prev_device_mesh is None:
        _, output_tensor_grad, wait_handles = _communicate(
            tensor_send_next=None,
            tensor_send_prev=None,
            current_device_mesh=current_device_mesh,
            prev_device_mesh=prev_device_mesh,
            next_device_mesh=next_device_mesh,
            recv_prev=False,
            recv_next=recv_next,
            tensor_shape=tensor_shape,
            send_tensor_shape_unpad=send_tensor_shape_unpad,
            batch_p2p_comm=batch_p2p_comm,
            wait_on_reqs=(not overlap_p2p_comm),
            dtype=recv_dtype,
            group=group,
            # file=file,
        )
    else:
        _, output_tensor_grad, wait_handles = _communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad,
            current_device_mesh=current_device_mesh,
            prev_device_mesh=prev_device_mesh,
            next_device_mesh=next_device_mesh,
            recv_prev=False,
            recv_next=recv_next,
            tensor_shape=tensor_shape,
            send_tensor_shape_unpad=send_tensor_shape_unpad,
            batch_p2p_comm=batch_p2p_comm,
            wait_on_reqs=(not overlap_p2p_comm),
            dtype=recv_dtype,
            group=group,
        )
    if overlap_p2p_comm:
        return output_tensor_grad, wait_handles
    return output_tensor_grad


def send_forward_backward_recv_forward_backward(
    output_tensor: torch.Tensor,
    input_tensor_grad: torch.Tensor,
    recv_prev: bool,
    recv_next: bool,
    tensor_shape: Shape,
    current_device_mesh: DeviceMesh,
    prev_device_mesh: DeviceMesh,
    next_device_mesh: DeviceMesh,
    recv_dtype: Optional[torch.dtype] = None,
    batch_p2p_comm: bool = True,
) -> torch.Tensor:
    """Batched send and recv with previous and next ranks in pipeline.

    See _communicate for argument details.
    """
    input_tensor, output_tensor_grad, _ = _communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=input_tensor_grad,
        current_device_mesh=current_device_mesh,
        prev_device_mesh=prev_device_mesh,
        next_device_mesh=next_device_mesh,
        recv_prev=recv_prev,
        recv_next=recv_next,
        tensor_shape=tensor_shape,
        dtype=recv_dtype,
        batch_p2p_comm=batch_p2p_comm,
    )
    return input_tensor, output_tensor_grad
