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

import os
import torch
import torch.distributed as dist
from torch.testing._internal.common_utils import run_tests
from vescale import DeviceMesh, distribute_tensor
from vescale.dtensor.placement_types import Replicate
from vescale.pipe.p2p_communication import (
    _communicate,
    _communicate_shapes,
    _mapping_local_rank_to_target_rank_by_device_mesh,
    recv_forward,
    recv_backward,
    send_forward,
    send_backward,
    send_forward_recv_backward,
    send_backward_recv_forward,
    send_forward_recv_forward,
    send_backward_recv_backward,
    send_forward_backward_recv_forward_backward,
    drain_recv_reqs,
)
from common_dtensor import (
    DTensorTestBase,
    with_comms,
)


class PipeP2PTest(DTensorTestBase):
    @staticmethod
    def set_up_device_mesh_stages(world_size, device, n):
        assert world_size % n == 0, "world size must be divisible by the number of stages"
        n_device = world_size // n
        return (DeviceMesh(device, list(range(n_device * i, n_device * (i + 1)))) for i in range(n))

    @staticmethod
    def apply_xavier_normal_with_seed(tensor, seed=99999):
        torch.manual_seed(seed)
        torch.nn.init.xavier_normal_(tensor)

    @property
    def world_size(self):
        return 8

    @property
    def sequence_len(self):
        return 8

    @property
    def batch_size(self):
        return 4

    @property
    def input_size(self):
        return 2

    @property
    def stages(self):
        return 4

    def _generate_device_meshes(self):
        device = f"cuda:{self.rank}"
        # stage1
        device_mesh_stage1 = DeviceMesh(device, list(range(self.world_size // 2)))
        # stage2
        device_mesh_stage2 = DeviceMesh(device, list(range(self.world_size // 2, self.world_size)))
        return device_mesh_stage1, device_mesh_stage2

    def _generate_three_device_meshes(self):
        device = f"cuda:{self.rank}"
        # stage1
        device_mesh_stage1 = DeviceMesh(device, list(range(self.world_size // 4)))
        # stage2
        device_mesh_stage2 = DeviceMesh(device, list(range(self.world_size // 4, self.world_size // 2)))
        # stage3
        device_mesh_stage3 = DeviceMesh(device, list(range(self.world_size // 2, self.world_size // 4 * 3)))
        return device_mesh_stage1, device_mesh_stage2, device_mesh_stage3

    @with_comms
    def test_communicate_shapes(self):
        """
        Test correctness function of _communicate_shapes().
        """
        device = f"cuda:{self.rank}"
        # must do this: https://pytorch.org/docs/stable/distributed.html
        torch.cuda.set_device(device)
        device_mesh_stage1, device_mesh_stage2 = self._generate_device_meshes()

        # stage 1 tensor
        tensor_stage1 = torch.empty(self.sequence_len, self.batch_size, self.input_size, device=device)
        torch.nn.init.xavier_normal_(tensor_stage1)
        dist.all_reduce(tensor_stage1, async_op=False)
        dtensor_stage1 = distribute_tensor(tensor_stage1, device_mesh_stage1, placements=[Replicate()])
        if self.rank in device_mesh_stage1.mesh.tolist():
            target_rank = _mapping_local_rank_to_target_rank_by_device_mesh(
                local_rank=self.rank, current_device_mesh=device_mesh_stage1, target_device_mesh=device_mesh_stage2
            )
            _communicate_shapes(
                local_rank=self.rank,
                tensor_send_next=dtensor_stage1,
                tensor_send_prev=None,
                next_rank=target_rank,
                prev_rank=None,
                recv_prev=False,
                recv_next=False,
            )
        if self.rank in device_mesh_stage2.mesh.tolist():
            target_rank = _mapping_local_rank_to_target_rank_by_device_mesh(
                local_rank=self.rank, current_device_mesh=device_mesh_stage2, target_device_mesh=device_mesh_stage1
            )
            recv_prev_shape, _ = _communicate_shapes(
                local_rank=self.rank,
                tensor_send_next=None,
                tensor_send_prev=None,
                prev_rank=target_rank,
                next_rank=None,
                recv_prev=True,
                recv_next=False,
            )
            self.assertTrue(recv_prev_shape == [self.sequence_len, self.batch_size, self.input_size])

    @with_comms
    def test_communicate_no_batch_p2p_comm(self):
        """
        Test correctness of p2p communication ops.
        """
        os.environ["LOCAL_RANK"] = str(self.rank)
        device = f"cuda:{self.rank}"
        # must do this: https://pytorch.org/docs/stable/distributed.html
        torch.cuda.set_device(device)
        device_mesh_stage1, device_mesh_stage2 = self._generate_device_meshes()
        # stage 1 tensor
        tensor_stage1 = torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device)
        dtensor_stage1 = distribute_tensor(tensor_stage1, device_mesh_stage1, placements=[Replicate()])
        if self.rank in device_mesh_stage1.mesh.tolist():
            _communicate(
                tensor_send_next=dtensor_stage1._local_tensor,
                tensor_send_prev=None,
                current_device_mesh=device_mesh_stage1,
                next_device_mesh=device_mesh_stage2,
                recv_prev=False,
                recv_next=False,
                tensor_shape=None,
                batch_p2p_comm=False,
                wait_on_reqs=True,
                dtype=None,
            )

        if self.rank in device_mesh_stage2.mesh.tolist():
            recv_prev_tensor, _, _ = _communicate(
                tensor_send_next=None,
                tensor_send_prev=None,
                current_device_mesh=device_mesh_stage2,
                prev_device_mesh=device_mesh_stage1,
                recv_prev=True,
                recv_next=False,
                tensor_shape=None,
                batch_p2p_comm=False,
                wait_on_reqs=True,
                dtype=torch.float32,
            )
            self.assertTrue(
                torch.equal(
                    recv_prev_tensor, torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device)
                )
            )

    @with_comms
    def test_communicate_batch_p2p_comm(self):
        """
        Test correctness of batch communication ops.
        """
        device = f"cuda:{self.rank}"
        # must do this: https://pytorch.org/docs/stable/distributed.html
        torch.cuda.set_device(device)
        device_mesh_stage1, device_mesh_stage2 = self._generate_device_meshes()
        # stage 1 tensor
        tensor_stage1 = torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device)
        dtensor_stage1 = distribute_tensor(tensor_stage1, device_mesh_stage1, placements=[Replicate()])
        if self.rank in device_mesh_stage1.mesh.tolist():
            _communicate(
                tensor_send_next=dtensor_stage1._local_tensor,
                tensor_send_prev=None,
                current_device_mesh=device_mesh_stage1,
                next_device_mesh=device_mesh_stage2,
                recv_prev=False,
                recv_next=False,
                tensor_shape=None,
                batch_p2p_comm=True,
                wait_on_reqs=True,
                dtype=None,
            )

        if self.rank in device_mesh_stage2.mesh.tolist():
            recv_prev_tensor, _, _ = _communicate(
                tensor_send_next=None,
                tensor_send_prev=None,
                current_device_mesh=device_mesh_stage2,
                prev_device_mesh=device_mesh_stage1,
                recv_prev=True,
                recv_next=False,
                tensor_shape=None,
                batch_p2p_comm=True,
                wait_on_reqs=True,
                dtype=torch.float32,
            )
            self.assertTrue(
                torch.equal(
                    recv_prev_tensor, torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device)
                )
            )

    @with_comms
    def test_send_forward_and_recv_forward(self):
        """
        Test correctness of send_forward() and recv_forward().
        """
        device = f"cuda:{self.rank}"
        # must do this: https://pytorch.org/docs/stable/distributed.html
        torch.cuda.set_device(device)
        stage_list = list(self.set_up_device_mesh_stages(self.world_size, device, self.stages))
        seed_list = list(range(99990, 99990 + self.stages))
        stage_n_dict = {(self.rank in stage.mesh.tolist()): i for i, stage in enumerate(stage_list)}
        stage_n = stage_n_dict[True]
        send_seed = seed_list[stage_n]
        recv_seed = seed_list[stage_n - 1]
        prev_stage = stage_list[stage_n - 1]
        curr_stage = stage_list[stage_n]
        next_stage = stage_list[(stage_n + 1) % len(stage_list)]
        send_t = torch.empty(self.sequence_len, self.batch_size, self.input_size, device=device)
        expt_t = torch.empty(self.sequence_len, self.batch_size, self.input_size, device=device)
        self.apply_xavier_normal_with_seed(send_t, seed=send_seed)
        self.apply_xavier_normal_with_seed(expt_t, seed=recv_seed)

        if stage_n % 2 == 0:
            send_forward(
                output_tensor=send_t,
                current_device_mesh=curr_stage,
                peer_device_mesh=next_stage,
                tensor_shape=send_t.shape,
            )
        else:
            recv_prev_tensor = recv_forward(
                tensor_shape=expt_t.shape,
                recv_dtype=expt_t.dtype,
                current_device_mesh=curr_stage,
                peer_device_mesh=prev_stage,
            )
            self.assertTrue(torch.equal(recv_prev_tensor, expt_t))

    @with_comms
    def test_send_backward_and_recv_backward(self):
        """
        Test correctness of send_backward() and recv_backward().
        """
        device = f"cuda:{self.rank}"
        # must do this: https://pytorch.org/docs/stable/distributed.html
        torch.cuda.set_device(device)
        stage_list = list(self.set_up_device_mesh_stages(self.world_size, device, self.stages))
        seed_list = list(range(99990, 99990 + self.stages))
        stage_n_dict = {(self.rank in stage.mesh.tolist()): i for i, stage in enumerate(stage_list)}
        stage_n = stage_n_dict[True]
        send_seed = seed_list[stage_n]
        recv_seed = seed_list[(stage_n + 1) % len(seed_list)]
        prev_stage = stage_list[stage_n - 1]
        curr_stage = stage_list[stage_n]
        next_stage = stage_list[(stage_n + 1) % len(stage_list)]
        send_t = torch.empty(self.sequence_len, self.batch_size, self.input_size, device=device)
        expt_t = torch.empty(self.sequence_len, self.batch_size, self.input_size, device=device)
        self.apply_xavier_normal_with_seed(send_t, seed=send_seed)
        self.apply_xavier_normal_with_seed(expt_t, seed=recv_seed)

        if stage_n % 2 == 0:
            send_backward(
                input_tensor_grad=send_t,
                current_device_mesh=curr_stage,
                peer_device_mesh=prev_stage,
                tensor_shape=send_t.shape,
            )
        else:
            recv_prev_tensor = recv_backward(
                tensor_shape=expt_t.shape,
                recv_dtype=expt_t.dtype,
                current_device_mesh=curr_stage,
                peer_device_mesh=next_stage,
            )
            self.assertTrue(torch.equal(recv_prev_tensor, expt_t))

    @with_comms
    def test_send_forward_recv_backward_and_send_backward_recv_forward(self):
        """
        Test correctness of send_backward() and recv_backward().
        """
        os.environ["LOCAL_RANK"] = str(self.rank)
        device = f"cuda:{self.rank}"
        # must do this: https://pytorch.org/docs/stable/distributed.html
        torch.cuda.set_device(device)
        stage_list = list(self.set_up_device_mesh_stages(self.world_size, device, self.stages))
        fwd_seed_list = list(range(99990, 99990 + self.stages))
        bwd_seed_list = list(range(77770, 77770 + self.stages))
        stage_n_dict = {(self.rank in stage.mesh.tolist()): i for i, stage in enumerate(stage_list)}
        stage_n = stage_n_dict[True]
        fwd_send_seed = fwd_seed_list[stage_n]
        fwd_recv_seed = fwd_seed_list[stage_n - 1]
        bwd_send_seed = bwd_seed_list[stage_n]
        bwd_recv_seed = bwd_seed_list[(stage_n + 1) % len(bwd_seed_list)]
        prev_stage = stage_list[stage_n - 1]
        curr_stage = stage_list[stage_n]
        next_stage = stage_list[(stage_n + 1) % len(stage_list)]
        fwd_send_t = torch.empty(self.sequence_len, self.batch_size, self.input_size, device=device)
        fwd_expt_t = torch.empty(self.sequence_len, self.batch_size, self.input_size, device=device)
        bwd_send_t = torch.empty(self.sequence_len, self.batch_size, self.input_size, device=device)
        bwd_expt_t = torch.empty(self.sequence_len, self.batch_size, self.input_size, device=device)
        self.apply_xavier_normal_with_seed(fwd_send_t, seed=fwd_send_seed)
        self.apply_xavier_normal_with_seed(fwd_expt_t, seed=fwd_recv_seed)
        self.apply_xavier_normal_with_seed(bwd_send_t, seed=bwd_send_seed)
        self.apply_xavier_normal_with_seed(bwd_expt_t, seed=bwd_recv_seed)
        if stage_n % 2 == 0:
            recv_bwd_tensor = send_forward_recv_backward(
                output_tensor=fwd_send_t,
                tensor_shape=bwd_expt_t.shape,
                recv_dtype=bwd_expt_t.dtype,
                current_device_mesh=curr_stage,
                peer_device_mesh=next_stage,
            )
            self.assertTrue(torch.equal(recv_bwd_tensor, bwd_expt_t))
        else:
            recv_fwd_tensor = send_backward_recv_forward(
                input_tensor_grad=bwd_send_t,
                tensor_shape=fwd_expt_t.shape,
                recv_dtype=fwd_expt_t.dtype,
                current_device_mesh=curr_stage,
                peer_device_mesh=prev_stage,
            )
            self.assertTrue(torch.equal(recv_fwd_tensor, fwd_expt_t))

    @with_comms
    def test_send_forward_recv_forward_no_shape(self):
        """
        Test correctness of send_forward_recv_forward without sharing tensor shape in advance.
        """
        os.environ["LOCAL_RANK"] = str(self.rank)
        device = f"cuda:{self.rank}"
        # must do this: https://pytorch.org/docs/stable/distributed.html
        torch.cuda.set_device(device)
        device_mesh_stage1, device_mesh_stage2, device_mesh_stage3 = self._generate_three_device_meshes()
        # stage 1 tensor
        tensor_stage1 = torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device)
        dtensor_stage1 = distribute_tensor(tensor_stage1, device_mesh_stage1, placements=[Replicate()])
        tensor_stage2 = torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device) + 1
        dtensor_stage2 = distribute_tensor(tensor_stage2, device_mesh_stage2, placements=[Replicate()])
        # send to stage 2
        if self.rank in device_mesh_stage1.mesh.tolist():
            send_forward(
                output_tensor=dtensor_stage1.to_local(),
                current_device_mesh=device_mesh_stage1,
                peer_device_mesh=device_mesh_stage2,
                tensor_shape=None,
            )

        if self.rank in device_mesh_stage2.mesh.tolist():
            stage2_recv_tensor = send_forward_recv_forward(
                output_tensor=dtensor_stage2._local_tensor,
                recv_prev=True,
                tensor_shape=None,
                current_device_mesh=device_mesh_stage2,
                prev_device_mesh=device_mesh_stage1,
                next_device_mesh=device_mesh_stage3,
                recv_dtype=torch.float32,
            )
            self.assertTrue(
                torch.equal(
                    stage2_recv_tensor, torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device)
                )
            )
        if self.rank in device_mesh_stage3.mesh.tolist():
            stage3_recv_tensor = recv_forward(
                tensor_shape=None,
                recv_dtype=torch.float32,
                current_device_mesh=device_mesh_stage3,
                peer_device_mesh=device_mesh_stage2,
            )
            self.assertTrue(
                torch.equal(
                    stage3_recv_tensor,
                    torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device) + 1,
                )
            )

    @with_comms
    def test_send_forward_recv_forward_with_shape(self):
        """
        Test correctness of send_forward_recv_forward with known tensor shape.
        """
        os.environ["LOCAL_RANK"] = str(self.rank)
        device = f"cuda:{self.rank}"
        # must do this: https://pytorch.org/docs/stable/distributed.html
        torch.cuda.set_device(device)
        device_mesh_stage1, device_mesh_stage2, device_mesh_stage3 = self._generate_three_device_meshes()
        # stage 1 tensor
        shape = (self.sequence_len, self.batch_size, self.input_size)
        tensor_stage1 = torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device)
        dtensor_stage1 = distribute_tensor(tensor_stage1, device_mesh_stage1, placements=[Replicate()])
        tensor_stage2 = torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device) + 1
        dtensor_stage2 = distribute_tensor(tensor_stage2, device_mesh_stage2, placements=[Replicate()])
        # send to stage 2
        if self.rank in device_mesh_stage1.mesh.tolist():
            send_forward(
                output_tensor=dtensor_stage1.to_local(),
                current_device_mesh=device_mesh_stage1,
                peer_device_mesh=device_mesh_stage2,
                tensor_shape=shape,
            )
        if self.rank in device_mesh_stage2.mesh.tolist():
            stage2_recv_tensor = send_forward_recv_forward(
                output_tensor=dtensor_stage2._local_tensor,
                recv_prev=True,
                tensor_shape=shape,
                current_device_mesh=device_mesh_stage2,
                prev_device_mesh=device_mesh_stage1,
                next_device_mesh=device_mesh_stage3,
                recv_dtype=torch.float32,
            )
            self.assertTrue(
                torch.equal(
                    stage2_recv_tensor, torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device)
                )
            )
        if self.rank in device_mesh_stage3.mesh.tolist():
            stage3_recv_tensor = recv_forward(
                tensor_shape=shape,
                recv_dtype=torch.float32,
                current_device_mesh=device_mesh_stage3,
                peer_device_mesh=device_mesh_stage2,
            )
            self.assertTrue(
                torch.equal(
                    stage3_recv_tensor,
                    torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device) + 1,
                )
            )

    @with_comms
    def test_send_backward_recv_backward_no_shape(self):
        """
        Test correctness of send_backward_recv_backward().
        """
        os.environ["LOCAL_RANK"] = str(self.rank)
        device = f"cuda:{self.rank}"
        # must do this: https://pytorch.org/docs/stable/distributed.html
        torch.cuda.set_device(device)
        device_mesh_stage1, device_mesh_stage2, device_mesh_stage3 = self._generate_three_device_meshes()
        # stage 1 tensor
        shape = None
        tensor_stage2 = torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device) + 1
        dtensor_stage2 = distribute_tensor(tensor_stage2, device_mesh_stage2, placements=[Replicate()])
        tensor_stage3 = torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device) + 2
        dtensor_stage3 = distribute_tensor(tensor_stage3, device_mesh_stage3, placements=[Replicate()])
        # send to stage 2
        if self.rank in device_mesh_stage1.mesh.tolist():
            stage1_recv_tensor = recv_backward(
                tensor_shape=shape,
                recv_dtype=torch.float32,
                current_device_mesh=device_mesh_stage1,
                peer_device_mesh=device_mesh_stage2,
            )
            self.assertTrue(
                torch.equal(
                    stage1_recv_tensor,
                    torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device) + 1,
                )
            )
        if self.rank in device_mesh_stage2.mesh.tolist():
            stage2_recv_tensor = send_backward_recv_backward(
                input_tensor_grad=dtensor_stage2._local_tensor,
                recv_next=True,
                tensor_shape=shape,
                current_device_mesh=device_mesh_stage2,
                prev_device_mesh=device_mesh_stage1,
                next_device_mesh=device_mesh_stage3,
                recv_dtype=torch.float32,
            )
            self.assertTrue(
                torch.equal(
                    stage2_recv_tensor,
                    torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device) + 2,
                )
            )
        if self.rank in device_mesh_stage3.mesh.tolist():
            send_backward(
                input_tensor_grad=dtensor_stage3.to_local(),
                current_device_mesh=device_mesh_stage3,
                peer_device_mesh=device_mesh_stage2,
                tensor_shape=shape,
            )

    @with_comms
    def test_send_backward_recv_backward_with_shape(self):
        """
        Test correctness of send_backward_recv_backward().
        """
        os.environ["LOCAL_RANK"] = str(self.rank)
        device = f"cuda:{self.rank}"
        # must do this: https://pytorch.org/docs/stable/distributed.html
        torch.cuda.set_device(device)
        device_mesh_stage1, device_mesh_stage2, device_mesh_stage3 = self._generate_three_device_meshes()
        # stage 1 tensor
        shape = (self.sequence_len, self.batch_size, self.input_size)
        tensor_stage2 = torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device) + 1
        dtensor_stage2 = distribute_tensor(tensor_stage2, device_mesh_stage2, placements=[Replicate()])
        tensor_stage3 = torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device) + 2
        dtensor_stage3 = distribute_tensor(tensor_stage3, device_mesh_stage3, placements=[Replicate()])
        # send to stage 2
        if self.rank in device_mesh_stage1.mesh.tolist():
            stage1_recv_tensor = recv_backward(
                tensor_shape=shape,
                recv_dtype=torch.float32,
                current_device_mesh=device_mesh_stage1,
                peer_device_mesh=device_mesh_stage2,
            )
            self.assertTrue(
                torch.equal(
                    stage1_recv_tensor,
                    torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device) + 1,
                )
            )
        if self.rank in device_mesh_stage2.mesh.tolist():
            stage2_recv_tensor = send_backward_recv_backward(
                input_tensor_grad=dtensor_stage2._local_tensor,
                recv_next=True,
                tensor_shape=shape,
                current_device_mesh=device_mesh_stage2,
                prev_device_mesh=device_mesh_stage1,
                next_device_mesh=device_mesh_stage3,
                recv_dtype=torch.float32,
            )
            self.assertTrue(
                torch.equal(
                    stage2_recv_tensor,
                    torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device) + 2,
                )
            )
        if self.rank in device_mesh_stage3.mesh.tolist():
            send_backward(
                input_tensor_grad=dtensor_stage3.to_local(),
                current_device_mesh=device_mesh_stage3,
                peer_device_mesh=device_mesh_stage2,
                tensor_shape=shape,
            )

    @with_comms
    def test_send_forward_backward_recv_forward_backward_with_shape(self):
        """
        Test correctness of send_forward_backward_recv_forward_backward().
        """
        os.environ["LOCAL_RANK"] = str(self.rank)
        device = f"cuda:{self.rank}"
        # must do this: https://pytorch.org/docs/stable/distributed.html
        torch.cuda.set_device(device)
        device_mesh_stage1, device_mesh_stage2, device_mesh_stage3 = self._generate_three_device_meshes()
        # stage 1 tensor
        shape = (self.sequence_len, self.batch_size, self.input_size)
        tensor_stage1 = torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device)
        dtensor_stage1 = distribute_tensor(tensor_stage1, device_mesh_stage1, placements=[Replicate()])
        tensor_stage2 = torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device) + 1
        dtensor_stage2 = distribute_tensor(tensor_stage2, device_mesh_stage2, placements=[Replicate()])
        tensor_stage3 = torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device) + 2
        dtensor_stage3 = distribute_tensor(tensor_stage3, device_mesh_stage3, placements=[Replicate()])
        # send to stage 2
        if self.rank in device_mesh_stage1.mesh.tolist():
            recv_bwd_tensor = send_forward_recv_backward(
                output_tensor=dtensor_stage1._local_tensor,
                tensor_shape=shape,
                recv_dtype=torch.float32,
                current_device_mesh=device_mesh_stage1,
                peer_device_mesh=device_mesh_stage2,
            )
            self.assertTrue(
                torch.equal(
                    recv_bwd_tensor, torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device) + 1
                )
            )
        if self.rank in device_mesh_stage2.mesh.tolist():
            input_tensor, output_tensor_grad = send_forward_backward_recv_forward_backward(
                output_tensor=dtensor_stage2._local_tensor,
                input_tensor_grad=dtensor_stage2._local_tensor,
                recv_prev=True,
                recv_next=True,
                tensor_shape=shape,
                current_device_mesh=device_mesh_stage2,
                prev_device_mesh=device_mesh_stage1,
                next_device_mesh=device_mesh_stage3,
                recv_dtype=torch.float32,
            )
            self.assertTrue(
                torch.equal(
                    input_tensor, torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device)
                )
            )
            self.assertTrue(
                torch.equal(
                    output_tensor_grad,
                    torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device) + 2,
                )
            )
        if self.rank in device_mesh_stage3.mesh.tolist():
            recv_fwd_tensor = send_backward_recv_forward(
                input_tensor_grad=dtensor_stage3.to_local(),
                tensor_shape=shape,
                recv_dtype=torch.float32,
                current_device_mesh=device_mesh_stage3,
                peer_device_mesh=device_mesh_stage2,
            )
            self.assertTrue(
                torch.equal(
                    recv_fwd_tensor, torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device) + 1
                )
            )

    @with_comms
    def test_send_forward_backward_recv_forward_backward_no_shape(self):
        """
        Test correctness of send_forward_backward_recv_forward_backward()
        without sharing tensor shapes in advance.
        """
        os.environ["LOCAL_RANK"] = str(self.rank)
        device = f"cuda:{self.rank}"
        # must do this: https://pytorch.org/docs/stable/distributed.html
        torch.cuda.set_device(device)
        device_mesh_stage1, device_mesh_stage2, device_mesh_stage3 = self._generate_three_device_meshes()
        # stage 1 tensor
        tensor_stage1 = torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device)
        dtensor_stage1 = distribute_tensor(tensor_stage1, device_mesh_stage1, placements=[Replicate()])
        tensor_stage2 = torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device) + 1
        dtensor_stage2 = distribute_tensor(tensor_stage2, device_mesh_stage2, placements=[Replicate()])
        tensor_stage3 = torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device) + 2
        dtensor_stage3 = distribute_tensor(tensor_stage3, device_mesh_stage3, placements=[Replicate()])
        # send to stage 2
        if self.rank in device_mesh_stage1.mesh.tolist():
            recv_bwd_tensor = send_forward_recv_backward(
                output_tensor=dtensor_stage1._local_tensor,
                tensor_shape=None,
                recv_dtype=torch.float32,
                current_device_mesh=device_mesh_stage1,
                peer_device_mesh=device_mesh_stage2,
            )
            self.assertTrue(
                torch.equal(
                    recv_bwd_tensor, torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device) + 1
                )
            )
        if self.rank in device_mesh_stage2.mesh.tolist():
            input_tensor, output_tensor_grad = send_forward_backward_recv_forward_backward(
                output_tensor=dtensor_stage2._local_tensor,
                input_tensor_grad=dtensor_stage2._local_tensor,
                recv_prev=True,
                recv_next=True,
                tensor_shape=None,
                current_device_mesh=device_mesh_stage2,
                prev_device_mesh=device_mesh_stage1,
                next_device_mesh=device_mesh_stage3,
                recv_dtype=torch.float32,
            )
            self.assertTrue(
                torch.equal(
                    input_tensor, torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device)
                )
            )
            self.assertTrue(
                torch.equal(
                    output_tensor_grad,
                    torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device) + 2,
                )
            )
        if self.rank in device_mesh_stage3.mesh.tolist():
            recv_fwd_tensor = send_backward_recv_forward(
                input_tensor_grad=dtensor_stage3.to_local(),
                tensor_shape=None,
                recv_dtype=torch.float32,
                current_device_mesh=device_mesh_stage3,
                peer_device_mesh=device_mesh_stage2,
            )
            self.assertTrue(
                torch.equal(
                    recv_fwd_tensor, torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device) + 1
                )
            )

    @with_comms
    def test_send_forward_recv_forward_with_shape_next_device_mesh_none(self):
        """
        Test correctness of send_forward_recv_forward() with tensor shapes known.
        """
        os.environ["LOCAL_RANK"] = str(self.rank)
        device = f"cuda:{self.rank}"
        # must do this: https://pytorch.org/docs/stable/distributed.html
        torch.cuda.set_device(device)
        device_mesh_stage1, device_mesh_stage2, _ = self._generate_three_device_meshes()
        # stage 1 tensor
        shape = (self.sequence_len, self.batch_size, self.input_size)
        tensor_stage1 = torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device)
        dtensor_stage1 = distribute_tensor(tensor_stage1, device_mesh_stage1, placements=[Replicate()])
        tensor_stage2 = torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device) + 1
        dtensor_stage2 = distribute_tensor(tensor_stage2, device_mesh_stage2, placements=[Replicate()])
        # send to stage 2
        if self.rank in device_mesh_stage1.mesh.tolist():
            send_forward(
                output_tensor=dtensor_stage1.to_local(),
                current_device_mesh=device_mesh_stage1,
                peer_device_mesh=device_mesh_stage2,
                tensor_shape=shape,
            )
        if self.rank in device_mesh_stage2.mesh.tolist():
            stage2_recv_tensor = send_forward_recv_forward(
                output_tensor=dtensor_stage2._local_tensor,
                recv_prev=True,
                tensor_shape=shape,
                current_device_mesh=device_mesh_stage2,
                prev_device_mesh=device_mesh_stage1,
                next_device_mesh=None,
                recv_dtype=torch.float32,
            )
            self.assertTrue(
                torch.equal(
                    stage2_recv_tensor, torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device)
                )
            )

    @with_comms
    def test_send_backward_recv_backward_with_shape_device_mesh_none(self):
        """
        Test correctness of send_backward_recv_backward() with tensor shapes known.
        """
        os.environ["LOCAL_RANK"] = str(self.rank)
        device = f"cuda:{self.rank}"
        # must do this: https://pytorch.org/docs/stable/distributed.html
        torch.cuda.set_device(device)
        device_mesh_stage1, device_mesh_stage2, device_mesh_stage3 = self._generate_three_device_meshes()
        # stage 1 tensor
        shape = (self.sequence_len, self.batch_size, self.input_size)
        tensor_stage1 = torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device)
        tensor_stage2 = torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device) + 1
        dtensor_stage2 = distribute_tensor(tensor_stage2, device_mesh_stage2, placements=[Replicate()])
        tensor_stage3 = torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device) + 2
        dtensor_stage3 = distribute_tensor(tensor_stage3, device_mesh_stage3, placements=[Replicate()])
        # send to stage 2
        if self.rank in device_mesh_stage2.mesh.tolist():
            stage2_recv_tensor = send_backward_recv_backward(
                input_tensor_grad=dtensor_stage2._local_tensor,
                recv_next=True,
                tensor_shape=shape,
                current_device_mesh=device_mesh_stage2,
                prev_device_mesh=None,
                next_device_mesh=device_mesh_stage3,
                recv_dtype=torch.float32,
            )
            self.assertTrue(
                torch.equal(
                    stage2_recv_tensor,
                    torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device) + 2,
                )
            )
        if self.rank in device_mesh_stage3.mesh.tolist():
            send_backward(
                input_tensor_grad=dtensor_stage3.to_local(),
                current_device_mesh=device_mesh_stage3,
                peer_device_mesh=device_mesh_stage2,
                tensor_shape=shape,
            )

    @with_comms
    def test_send_backward_recv_backward_with_shape_p2p_overlap(self):
        """
        Test correctness of send_backward_recv_backward() with overlapped p2p on.
        """
        os.environ["LOCAL_RANK"] = str(self.rank)
        device = f"cuda:{self.rank}"
        # must do this: https://pytorch.org/docs/stable/distributed.html
        torch.cuda.set_device(device)
        device_mesh_stage1, device_mesh_stage2, device_mesh_stage3 = self._generate_three_device_meshes()
        # stage 1 tensor
        shape = (self.sequence_len, self.batch_size, self.input_size)
        tensor_stage1 = torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device)
        tensor_stage2 = torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device) + 1
        dtensor_stage2 = distribute_tensor(tensor_stage2, device_mesh_stage2, placements=[Replicate()])
        tensor_stage3 = torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device) + 2
        dtensor_stage3 = distribute_tensor(tensor_stage3, device_mesh_stage3, placements=[Replicate()])
        # send to stage 2
        if self.rank in device_mesh_stage2.mesh.tolist():
            stage2_recv_tensor, bwd_wait_handles = send_backward_recv_backward(
                input_tensor_grad=dtensor_stage2._local_tensor,
                recv_next=True,
                tensor_shape=shape,
                current_device_mesh=device_mesh_stage2,
                prev_device_mesh=None,
                next_device_mesh=device_mesh_stage3,
                recv_dtype=torch.float32,
                overlap_p2p_comm=True,
                batch_p2p_comm=False,
            )

        if self.rank in device_mesh_stage3.mesh.tolist():
            stage3_recv_tensor, bwd_wait_handles = send_backward_recv_backward(
                input_tensor_grad=dtensor_stage3._local_tensor,
                recv_next=False,
                tensor_shape=shape,
                current_device_mesh=device_mesh_stage3,
                prev_device_mesh=device_mesh_stage2,
                next_device_mesh=None,
                recv_dtype=torch.float32,
                overlap_p2p_comm=True,
                batch_p2p_comm=False,
            )
        drain_recv_reqs("backward")
        if self.rank in device_mesh_stage2.mesh.tolist():
            self.assertTrue(
                torch.equal(
                    stage2_recv_tensor,
                    torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device) + 2,
                )
            )

    @with_comms
    def test_send_forward_recv_forward_with_shape_p2p_overlap(self):
        """
        Test correctness of send_forward_recv_forward() with overlapped p2p on.
        """
        os.environ["LOCAL_RANK"] = str(self.rank)
        device = f"cuda:{self.rank}"
        # must do this: https://pytorch.org/docs/stable/distributed.html
        torch.cuda.set_device(device)
        device_mesh_stage1, device_mesh_stage2, device_mesh_stage3 = self._generate_three_device_meshes()
        # stage 1 tensor
        shape = (self.sequence_len, self.batch_size, self.input_size)
        tensor_stage1 = torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device)
        dtensor_stage1 = distribute_tensor(tensor_stage1, device_mesh_stage1, placements=[Replicate()])
        tensor_stage2 = torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device) + 1
        dtensor_stage2 = distribute_tensor(tensor_stage2, device_mesh_stage2, placements=[Replicate()])
        # send to stage 2
        if self.rank in device_mesh_stage1.mesh.tolist():
            stage1_recv_tensor, fwd_wait_handles = send_forward_recv_forward(
                output_tensor=dtensor_stage1._local_tensor,
                recv_prev=False,
                tensor_shape=shape,
                current_device_mesh=device_mesh_stage1,
                prev_device_mesh=None,
                next_device_mesh=device_mesh_stage2,
                recv_dtype=torch.float32,
                overlap_p2p_comm=True,
                batch_p2p_comm=False,
            )
        if self.rank in device_mesh_stage2.mesh.tolist():
            stage2_recv_tensor, fwd_wait_handles = send_forward_recv_forward(
                output_tensor=dtensor_stage2._local_tensor,
                recv_prev=True,
                tensor_shape=shape,
                current_device_mesh=device_mesh_stage2,
                prev_device_mesh=device_mesh_stage1,
                next_device_mesh=None,
                recv_dtype=torch.float32,
                overlap_p2p_comm=True,
                batch_p2p_comm=False,
            )
        drain_recv_reqs("forward")
        if self.rank in device_mesh_stage2.mesh.tolist():
            self.assertTrue(
                torch.equal(
                    stage2_recv_tensor, torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device)
                )
            )

    @with_comms
    def test_send_backward_recv_backward_with_shape_p2p_overlap_auto_modify(self):
        """
        Test correctness of send_backward_recv_backward() with overlapped p2p on.
        """
        os.environ["LOCAL_RANK"] = str(self.rank)
        device = f"cuda:{self.rank}"
        # must do this: https://pytorch.org/docs/stable/distributed.html
        torch.cuda.set_device(device)
        device_mesh_stage1, device_mesh_stage2, device_mesh_stage3 = self._generate_three_device_meshes()
        # stage 1 tensor
        shape = (self.sequence_len, self.batch_size, self.input_size)
        tensor_stage1 = torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device)
        dtensor_stage1 = distribute_tensor(tensor_stage1, device_mesh_stage1, placements=[Replicate()])
        tensor_stage2 = torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device) + 1
        dtensor_stage2 = distribute_tensor(tensor_stage2, device_mesh_stage2, placements=[Replicate()])
        tensor_stage3 = torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device) + 2
        dtensor_stage3 = distribute_tensor(tensor_stage3, device_mesh_stage3, placements=[Replicate()])
        # send to stage 2
        if self.rank in device_mesh_stage2.mesh.tolist():
            stage2_recv_tensor, bwd_wait_handles = send_backward_recv_backward(
                input_tensor_grad=dtensor_stage2._local_tensor,
                recv_next=True,
                tensor_shape=shape,
                current_device_mesh=device_mesh_stage2,
                prev_device_mesh=None,
                next_device_mesh=device_mesh_stage3,
                recv_dtype=torch.float32,
                overlap_p2p_comm=True,
                batch_p2p_comm=False,
            )

        if self.rank in device_mesh_stage3.mesh.tolist():
            stage3_recv_tensor, bwd_wait_handles = send_backward_recv_backward(
                input_tensor_grad=dtensor_stage3._local_tensor,
                recv_next=True,
                tensor_shape=shape,
                current_device_mesh=device_mesh_stage3,
                prev_device_mesh=device_mesh_stage2,
                next_device_mesh=None,
                recv_dtype=torch.float32,
                overlap_p2p_comm=True,
                batch_p2p_comm=False,
            )

        drain_recv_reqs("backward")
        if self.rank in device_mesh_stage2.mesh.tolist():
            self.assertTrue(
                torch.equal(
                    stage2_recv_tensor,
                    torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device) + 2,
                )
            )

    @with_comms
    def test_send_forward_recv_forward_with_shape_p2p_overlap_auto_modify(self):
        """
        Test correctness of send_forward_recv_forward() with overlapped p2p on.
        """
        os.environ["LOCAL_RANK"] = str(self.rank)
        device = f"cuda:{self.rank}"
        # must do this: https://pytorch.org/docs/stable/distributed.html
        torch.cuda.set_device(device)
        device_mesh_stage1, device_mesh_stage2, device_mesh_stage3 = self._generate_three_device_meshes()
        # stage 1 tensor
        shape = (self.sequence_len, self.batch_size, self.input_size)
        tensor_stage1 = torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device)
        dtensor_stage1 = distribute_tensor(tensor_stage1, device_mesh_stage1, placements=[Replicate()])
        tensor_stage2 = torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device) + 1
        dtensor_stage2 = distribute_tensor(tensor_stage2, device_mesh_stage2, placements=[Replicate()])
        # send to stage 2
        if self.rank in device_mesh_stage1.mesh.tolist():
            stage1_recv_tensor, fwd_wait_handles = send_forward_recv_forward(
                output_tensor=dtensor_stage1._local_tensor,
                recv_prev=True,
                tensor_shape=shape,
                current_device_mesh=device_mesh_stage1,
                prev_device_mesh=None,
                next_device_mesh=device_mesh_stage2,
                recv_dtype=torch.float32,
                overlap_p2p_comm=True,
                batch_p2p_comm=False,
            )
        if self.rank in device_mesh_stage2.mesh.tolist():
            stage2_recv_tensor, fwd_wait_handles = send_forward_recv_forward(
                output_tensor=dtensor_stage2._local_tensor,
                recv_prev=True,
                tensor_shape=shape,
                current_device_mesh=device_mesh_stage2,
                prev_device_mesh=device_mesh_stage1,
                next_device_mesh=None,
                recv_dtype=torch.float32,
                overlap_p2p_comm=True,
                batch_p2p_comm=False,
            )

        drain_recv_reqs("forward")
        if self.rank in device_mesh_stage2.mesh.tolist():
            self.assertTrue(
                torch.equal(
                    stage2_recv_tensor, torch.ones(self.sequence_len, self.batch_size, self.input_size, device=device)
                )
            )


if __name__ == "__main__":
    run_tests()
