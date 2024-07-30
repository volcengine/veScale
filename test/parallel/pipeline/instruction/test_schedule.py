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
from common_dtensor import DTensorTestBase, with_comms
from vescale.pipe._schedules.instruction_base import get_linear_pp_module_dep2
from vescale.pipe._schedules.pipedream_flush import PipeDream
from vescale.pipe._schedules.looping_bfs import InterleavedPipeDreramFlush
from vescale.devicemesh_api import VESCALE_DEVICE_MESH
import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests
from vescale.dtensor.api import distribute_tensor
from vescale.dtensor.device_mesh import DeviceMesh
from vescale.dmodule.api import parallelize_module
from vescale.dtensor.placement_types import Replicate
from vescale.plan.spec import PipelineScheduleType
from vescale.pipe.pipe_emmiter import ScheduleEngine


class MLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.fc1 = nn.Linear(n_features, n_features * 2, bias=False)
        torch.nn.init.uniform_(self.fc1.weight, 0, 1)
        self.fc2 = nn.Linear(n_features * 2, n_features)
        torch.nn.init.uniform_(self.fc2.weight, 0, 1)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))

    def forward_utils(p2p, dataloader):
        if p2p is not None:
            return p2p
        else:
            return dataloader


class FourMLP(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.mlp1 = MLP(hidden)
        self.mlp2 = MLP(hidden)
        self.mlp3 = MLP(hidden)
        self.mlp4 = MLP(hidden)

    def forward(self, x):
        return self.mlp4(self.mlp3(self.mlp2(self.mlp1(x))))


class EightMLP(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.mlps = [MLP(hidden) for _ in range(8)]

    def forward(self, x):
        all_input_x = []
        for idx, mlp in enumerate(self.mlps):
            x = mlp(x)
            x.retain_grad()
            all_input_x.append(x)
            print(f"mlp: {idx} output : {x}")
        return x, all_input_x


class PipelineScheduleTest(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @staticmethod
    def loss_fn(x):
        return x.sum()

    @with_comms
    def test_1f1b_schedules(self):
        """
        Test generation of simple 1f1b schedule.
        """
        device = f"cuda:{self.rank}"
        # must do this: https://pytorch.org/docs/stable/distributed.html
        torch.cuda.set_device(device)
        device_mesh_stage1 = DeviceMesh(device, [0])
        device_mesh_stage2 = DeviceMesh(device, [1])
        device_mesh_stage3 = DeviceMesh(device, [2])
        device_mesh_stage4 = DeviceMesh(device, [3])
        meshes = (device_mesh_stage1, device_mesh_stage2, device_mesh_stage3, device_mesh_stage4)
        microbatch = 8
        batch = 8
        stage = 4
        schedule = PipeDream(stage, meshes, batch)
        if torch.distributed.distributed_c10d.get_rank() == 0:
            print(schedule)

    @with_comms
    def test_interleaved_1f1b_schedules(self):
        """
        Test generation of interleaved 1f1b schedule.
        """
        device = f"cuda:{self.rank}"
        # must do this: https://pytorch.org/docs/stable/distributed.html
        torch.cuda.set_device(device)
        device_mesh_stage1 = DeviceMesh(device, [0])
        device_mesh_stage2 = DeviceMesh(device, [1])
        device_mesh_stage3 = DeviceMesh(device, [2])
        device_mesh_stage4 = DeviceMesh(device, [3])
        meshes = (device_mesh_stage1, device_mesh_stage2, device_mesh_stage3, device_mesh_stage4)
        batches = 8
        num_chunks = 2
        schedule = InterleavedPipeDreramFlush(
            num_chunks, meshes, default_shape=[1, 1, 3], default_dtype=torch.float32, batches=batches
        )
        if self.rank == 0:
            print(schedule)

    @with_comms
    def test_runtime_engine_with_profiling(self):
        """
        Tests runtime engine with distributed nD timeline profiling.
        """
        # initialize global device mesh
        VESCALE_DEVICE_MESH.init_device_mesh(
            device_type="cuda",
            mesh_shape=(4, 1, 1),
            mesh_dim_names=("PP", "DP", "TP"),
        )
        global local_rank
        local_rank = self.rank
        device = f"cuda:{local_rank}"
        # must do this: https://pytorch.org/docs/stable/distributed.html
        torch.cuda.set_device(device)
        os.environ["LOCAL_RANK"] = str(local_rank)
        from vescale.ndtimeline import init_ndtimers, flush, wait

        init_ndtimers(rank=int(local_rank), local=int(local_rank), enable_streamer=True)
        n_hidden = 3
        batches = 8
        model = FourMLP(n_hidden)
        all_batches_out = []
        if self.rank == 3:
            for i in range(batches):
                print(f" ===========batch: {i}================= ")
                data = torch.zeros(1, 1, n_hidden) + i
                data = data.float().cuda(3)
                model.cuda(3)
                out = model(data)
                loss = out.sum()
                all_batches_out.append(loss)
                loss.backward(create_graph=True)
                print(loss)
                print(" ====================================== ")
        fwd_plan = {
            ".input": [[Replicate()]],
            ".output": [[Replicate()]],
        }
        model_list = []

        tp_mesh = VESCALE_DEVICE_MESH.get_tensor_parallel_mesh()
        if local_rank == 0:
            model.mlp1 = parallelize_module(model.mlp1, tp_mesh, {"parameter": None, "forward": fwd_plan})
            model_list = [model.mlp1]
        elif self.rank == 1:
            model.mlp2 = parallelize_module(model.mlp2, tp_mesh, {"parameter": None, "forward": fwd_plan})
            model_list = [model.mlp2]
        elif self.rank == 2:
            model.mlp3 = parallelize_module(model.mlp3, tp_mesh, {"parameter": None, "forward": fwd_plan})
            model_list = [model.mlp3]
        elif self.rank == 3:
            model.mlp4 = parallelize_module(model.mlp4, tp_mesh, {"parameter": None, "forward": fwd_plan})
            model_list = [model.mlp4]
        deps = get_linear_pp_module_dep2(model_list, VESCALE_DEVICE_MESH.get_global_tensor_parallel_meshes())
        data_iterator = []
        for i in range(batches):
            data = torch.zeros(1, 1, n_hidden) + i
            data_iterator.append(distribute_tensor(data.float(), tp_mesh, placements=[Replicate()]))
        pipe_engine = ScheduleEngine(
            deps=deps,
            meshes=VESCALE_DEVICE_MESH.get_global_tensor_parallel_meshes(),
            schedule=PipelineScheduleType.SIMPLE_1F1B,
            batches=batches,
            data_iterator=data_iterator,
            stage_id=local_rank,
            shape=(1, 1, 3),
            dtype=torch.float32,
        )
        _, all_forward = ScheduleEngine.execute(pipe_engine)
        if self.rank == 3:
            loss_per_microbatch = [item[1] for item in all_forward]
            for t1, t2 in zip(loss_per_microbatch, all_batches_out):
                self.assertEqual(t1._local_tensor, t2)
        flush()
        wait()

    @with_comms
    def test_interleaved_1f1b_emmiter(self):
        """
        Test schedule instructions generated by ScheduleEngine's pipeline emitter.
        """
        device = f"cuda:{self.rank}"
        # must do this: https://pytorch.org/docs/stable/distributed.html
        torch.cuda.set_device(device)
        os.environ["LOCAL_RANK"] = str(self.rank)
        n_hidden = 3
        batches = 8
        num_chunks = 2
        meshes = [DeviceMesh(device, [i]) for i in range(self.world_size)]
        model = EightMLP(n_hidden)
        fwd_plan = {
            ".input": [[Replicate()]],
            ".output": [[Replicate()]],
        }
        vpp_module_chunk_list = []
        if self.rank == 0:
            model.mlps[0] = parallelize_module(model.mlps[0], meshes[0], {"parameter": None, "forward": fwd_plan})
            model.mlps[4] = parallelize_module(model.mlps[4], meshes[0], {"parameter": None, "forward": fwd_plan})
            vpp_module_chunk_list = [model.mlps[0], model.mlps[4]]
        elif self.rank == 1:
            model.mlps[1] = parallelize_module(model.mlps[1], meshes[1], {"parameter": None, "forward": fwd_plan})
            model.mlps[5] = parallelize_module(model.mlps[5], meshes[1], {"parameter": None, "forward": fwd_plan})
            vpp_module_chunk_list = [model.mlps[1], model.mlps[5]]
        elif self.rank == 2:
            model.mlps[2] = parallelize_module(model.mlps[2], meshes[2], {"parameter": None, "forward": fwd_plan})
            model.mlps[6] = parallelize_module(model.mlps[6], meshes[2], {"parameter": None, "forward": fwd_plan})
            vpp_module_chunk_list = [model.mlps[2], model.mlps[6]]
        elif self.rank == 3:
            model.mlps[3] = parallelize_module(model.mlps[3], meshes[3], {"parameter": None, "forward": fwd_plan})
            model.mlps[7] = parallelize_module(model.mlps[7], meshes[3], {"parameter": None, "forward": fwd_plan})
            vpp_module_chunk_list = [model.mlps[3], model.mlps[7]]

        deps = get_linear_pp_module_dep2(vpp_module_chunk_list, meshes)
        data_iterator = []
        for i in range(batches):
            data = torch.zeros(1, 1, n_hidden) + i
            data_iterator.append(
                distribute_tensor(
                    data.float(), DeviceMesh(device, [self.rank], _validate_mesh=False), placements=[Replicate()]
                )
            )
        pipe_engine = ScheduleEngine(
            deps,
            meshes,
            PipelineScheduleType.INTERLEAVED_1F1B,
            batches,
            iter(data_iterator),
            self.rank,
            (1, 1, 3),
            dtype=torch.float32,
            num_chunks=num_chunks,
        )

    @with_comms
    def test_runtime_interleaved_1f1b_engine_batch(self):
        """
        Test parallelized DModules to perform interleaved 1f1b training.
        """
        device = f"cuda:{self.rank}"
        # must do this: https://pytorch.org/docs/stable/distributed.html
        torch.cuda.set_device(device)
        os.environ["LOCAL_RANK"] = str(self.rank)
        n_hidden = 3
        batches = 8
        num_chunks = 2
        meshes = [DeviceMesh(device, [i]) for i in range(self.world_size)]
        model = EightMLP(n_hidden)
        all_batches_out = []
        if self.rank == 3:
            true_model = model
            for i in range(8):
                true_model.mlps[i] = true_model.mlps[i].cuda(3)
            true_model.train()
            for i in range(batches):
                print(f" ===========batch: {i}================= ")
                data = torch.zeros(1, 1, n_hidden) + i
                data = data.float().cuda(3)
                out, all_output_x = true_model(data)
                loss = out.sum()
                all_batches_out.append(loss)
                loss.backward(create_graph=True)
                for idx, output in enumerate(all_output_x):
                    print(f"mlp{idx}.grad is {output.grad}")
                print(" ====================================== ")
        fwd_plan = {
            ".input": [[Replicate()]],
            ".output": [[Replicate()]],
        }
        vpp_module_chunk_list = []
        if self.rank == 0:
            model.mlps[0] = parallelize_module(model.mlps[0], meshes[0], {"parameter": None, "forward": fwd_plan})
            model.mlps[4] = parallelize_module(model.mlps[4], meshes[0], {"parameter": None, "forward": fwd_plan})
            vpp_module_chunk_list = [model.mlps[0], model.mlps[4]]
        elif self.rank == 1:
            model.mlps[1] = parallelize_module(model.mlps[1], meshes[1], {"parameter": None, "forward": fwd_plan})
            model.mlps[5] = parallelize_module(model.mlps[5], meshes[1], {"parameter": None, "forward": fwd_plan})
            vpp_module_chunk_list = [model.mlps[1], model.mlps[5]]
        elif self.rank == 2:
            model.mlps[2] = parallelize_module(model.mlps[2], meshes[2], {"parameter": None, "forward": fwd_plan})
            model.mlps[6] = parallelize_module(model.mlps[6], meshes[2], {"parameter": None, "forward": fwd_plan})
            vpp_module_chunk_list = [model.mlps[2], model.mlps[6]]
        elif self.rank == 3:
            model.mlps[3] = parallelize_module(model.mlps[3], meshes[3], {"parameter": None, "forward": fwd_plan})
            model.mlps[7] = parallelize_module(model.mlps[7], meshes[3], {"parameter": None, "forward": fwd_plan})
            vpp_module_chunk_list = [model.mlps[3], model.mlps[7]]
        deps = get_linear_pp_module_dep2(vpp_module_chunk_list, meshes)
        data_iterator = []
        for i in range(batches):
            data = torch.zeros(1, 1, n_hidden) + i
            data_iterator.append(
                distribute_tensor(
                    data.float(), DeviceMesh(device, [self.rank], _validate_mesh=False), placements=[Replicate()]
                )
            )
        pipe_engine = ScheduleEngine(
            deps,
            meshes,
            PipelineScheduleType.INTERLEAVED_1F1B,
            batches,
            [iter(data_iterator) for _ in range(num_chunks)],
            self.rank,
            (1, 1, 3),
            dtype=torch.float32,
            num_chunks=num_chunks,
            loss_fn=self.loss_fn,
        )
        if self.rank == 0:
            print("schedule", pipe_engine.p_emmiter.instruction_generator.schema)
        _, forward_datas = ScheduleEngine.execute(pipe_engine)
        if self.rank == 3:
            loss_per_microbatch = [item[1] for item in forward_datas]
            for t1, t2 in zip(loss_per_microbatch, all_batches_out):
                self.assertEqual(t1._local_tensor, t2)

    @with_comms
    def test_runtime_interleaved_1f1b_engine_p2p(self):
        """
        Test step-by-step initialization of pipeline engine, generation
        of simple 1f1b schedule and execution of pipeline engine with
        p2p overlapped communication.
        """
        device = f"cuda:{self.rank}"
        # must do this: https://pytorch.org/docs/stable/distributed.html
        torch.cuda.set_device(device)
        os.environ["LOCAL_RANK"] = str(self.rank)
        n_hidden = 3
        batches = 8
        num_chunks = 2
        meshes = [DeviceMesh(device, [i]) for i in range(self.world_size)]
        model = EightMLP(n_hidden)
        all_batches_out = []
        if self.rank == 3:
            true_model = model
            for i in range(8):
                true_model.mlps[i] = true_model.mlps[i].cuda(3)
            true_model.train()
            for i in range(batches):
                print(f" ===========batch: {i}================= ")
                data = torch.zeros(1, 1, n_hidden) + i
                data = data.float().cuda(3)
                out, all_output_x = true_model(data)
                loss = out.sum()
                all_batches_out.append(loss)
                loss.backward(create_graph=True)
                for idx, output in enumerate(all_output_x):
                    print(f"mlp{idx}.grad is {output.grad}")
                print(" ====================================== ")
        fwd_plan = {
            ".input": [[Replicate()]],
            ".output": [[Replicate()]],
        }
        vpp_module_chunk_list = []
        if self.rank == 0:
            model.mlps[0] = parallelize_module(model.mlps[0], meshes[0], {"parameter": None, "forward": fwd_plan})
            model.mlps[4] = parallelize_module(model.mlps[4], meshes[0], {"parameter": None, "forward": fwd_plan})
            vpp_module_chunk_list = [model.mlps[0], model.mlps[4]]
        elif self.rank == 1:
            model.mlps[1] = parallelize_module(model.mlps[1], meshes[1], {"parameter": None, "forward": fwd_plan})
            model.mlps[5] = parallelize_module(model.mlps[5], meshes[1], {"parameter": None, "forward": fwd_plan})
            vpp_module_chunk_list = [model.mlps[1], model.mlps[5]]
        elif self.rank == 2:
            model.mlps[2] = parallelize_module(model.mlps[2], meshes[2], {"parameter": None, "forward": fwd_plan})
            model.mlps[6] = parallelize_module(model.mlps[6], meshes[2], {"parameter": None, "forward": fwd_plan})
            vpp_module_chunk_list = [model.mlps[2], model.mlps[6]]
        elif self.rank == 3:
            model.mlps[3] = parallelize_module(model.mlps[3], meshes[3], {"parameter": None, "forward": fwd_plan})
            model.mlps[7] = parallelize_module(model.mlps[7], meshes[3], {"parameter": None, "forward": fwd_plan})
            vpp_module_chunk_list = [model.mlps[3], model.mlps[7]]
        deps = get_linear_pp_module_dep2(vpp_module_chunk_list, meshes)
        data_iterator = []
        for i in range(batches):
            data = torch.zeros(1, 1, n_hidden) + i
            data_iterator.append(
                distribute_tensor(data.float(), DeviceMesh(device, [0], _validate_mesh=False), placements=[Replicate()])
            )
        pipe_engine = ScheduleEngine(
            deps,
            meshes,
            PipelineScheduleType.INTERLEAVED_1F1B,
            batches,
            [iter(data_iterator) for _ in range(num_chunks)],
            self.rank,
            (1, 1, 3),
            dtype=torch.float32,
            num_chunks=num_chunks,
            overlap_p2p_comm=True,
            batch_p2p_comm=False,
            loss_fn=self.loss_fn,
        )
        if self.rank == 0:
            print("schedule", pipe_engine.p_emmiter.instruction_generator.schema)
        _, forward_datas = ScheduleEngine.execute(pipe_engine)
        if self.rank == 3:
            loss_per_microbatch = [item[1] for item in forward_datas]
            print(loss_per_microbatch, all_batches_out)
            for t1, t2 in zip(loss_per_microbatch, all_batches_out):
                self.assertEqual(t1._local_tensor, t2)

    @with_comms
    def test_zerobubble_engine(self):
        """
        Tests zero-bubble pipeline schedule with profiling.
        """
        # initialize global device mesh
        VESCALE_DEVICE_MESH.init_device_mesh(
            device_type="cuda",
            mesh_shape=(4, 1, 1),
            mesh_dim_names=("PP", "DP", "TP"),
        )
        global local_rank
        local_rank = self.rank
        device = f"cuda:{local_rank}"
        # must do this: https://pytorch.org/docs/stable/distributed.html
        torch.cuda.set_device(device)
        os.environ["LOCAL_RANK"] = str(local_rank)
        from vescale.ndtimeline import init_ndtimers, flush, wait

        init_ndtimers(rank=int(local_rank), local_rank=int(local_rank), enable_streamer=True)
        num_chunks = 2
        n_hidden = 3
        batches = 8
        model = EightMLP(n_hidden)
        for i in range(8):
            model.mlps[i] = model.mlps[i].cuda()
        all_batches_out = []
        if self.rank == 0:
            true_model = model
            for i in range(8):
                true_model.mlps[i] = true_model.mlps[i].cuda(0)
            true_model.train()
            for i in range(batches):
                print(f" ===========batch: {i}================= ")
                data = torch.zeros(1, 1, n_hidden) + i
                data = data.float().cuda(0)
                out, all_output_x = true_model(data)
                loss = out.sum()
                all_batches_out.append(loss)
                loss.backward(create_graph=True)
                for idx, output in enumerate(all_output_x):
                    print(f"mlp{idx}.grad is {output.grad}")
                print(" ====================================== ")
        fwd_plan = {
            ".input": [[Replicate()]],
            ".output": [[Replicate()]],
        }
        model_list = []

        if self.rank == 0:
            model_list = [model.mlps[0], model.mlps[7]]
        elif self.rank == 1:
            model_list = [model.mlps[1], model.mlps[6]]
        elif self.rank == 2:
            model_list = [model.mlps[2], model.mlps[5]]
        elif self.rank == 3:
            model_list = [model.mlps[3], model.mlps[4]]
        deps = get_linear_pp_module_dep2(model_list, VESCALE_DEVICE_MESH.get_global_tensor_parallel_meshes())
        data_iterator = []
        for i in range(batches):
            data = torch.zeros(1, 1, n_hidden) + i
            data_iterator.append(data.float().cuda())

        w = n_hidden * 2 * 4
        a = n_hidden * 4
        mem_f = 2 * w + 2 * a  # forward weight size
        mem_w = -2 * a
        mem_b = -mem_w - mem_f
        pipe_engine = ScheduleEngine(
            deps=deps,
            meshes=VESCALE_DEVICE_MESH.get_global_tensor_parallel_meshes(),
            schedule=PipelineScheduleType.ZERO_BUBBLE,
            batches=batches,
            data_iterator=[iter(data_iterator) for _ in range(num_chunks)],
            stage_id=local_rank,
            shape=(1, 1, 3),
            dtype=torch.float32,
            f_cost=6,
            b_cost=4,
            w_cost=4,
            c_cost=1,
            f_mem=mem_f,
            b_mem=mem_b,
            w_mem=mem_w,
            max_mem=mem_f * 4 * 2,
        )
        _, all_forward = ScheduleEngine.execute(pipe_engine)
        if self.rank == 0:
            loss_per_microbatch = [item[1] for item in all_forward]
            print(loss_per_microbatch, all_batches_out)
            for t1, t2 in zip(loss_per_microbatch, all_batches_out):
                self.assertEqual(t1, t2)

        flush()
        wait()


if __name__ == "__main__":
    run_tests()
