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

from common_dtensor import DTensorTestBase, with_comms
from torch.testing._internal.common_utils import run_tests
from vescale.pipe._schedules.instruction_base import (
    register_instruction,
    VESCALE_INTRUCTION_BUILDER as builder,
    StageDeps,
)
from vescale.initialize.deferred_init import deferred_init
from vescale.pipe import PipeParser
from vescale.pipe.pipe_stage import _generate_stage_dependencies
from vescale.dmodule.api import parallelize_module
from vescale.dtensor.device_mesh import DeviceMesh
import torch
from four_mlp import FourMLP, sharding_plan
from vescale.pipe._schedules.pipedream_flush import maybe_tensor, cross_mesh_send, cross_mesh_recv

from torch.distributed._functional_collectives import send, recv

from vescale.plan.pipeline_parallel import PipelineParallelPlan
from vescale.plan.spec import PipelineSplitMethodType


class PowerUserScheduleTest(DTensorTestBase):
    @with_comms
    def test_user_define_schedule(self):
        """
        Tests user-defined pipeline schedule.
        """
        global_mesh = DeviceMesh("cuda", [[0, 1], [2, 3]])
        torch.cuda.set_device(self.rank)

        @register_instruction(name="send")
        def send_forward():
            topo = builder.topo
            send_data = builder.last
            send_comms = topo.send_tables[builder.stage_id]
            send_comm = send_comms[0]
            mapping_group = send_comm.cur_mesh.get_mapping_rank(send_comm.peer_mesh)
            send(maybe_tensor(send_data), mapping_group, torch.distributed.distributed_c10d._get_default_group())
            cross_mesh_send(send_comm, send_data)

        @register_instruction(name="recv")
        def recv_forward():
            topo = builder.topo
            recv_comms = topo.recv_tables[builder.stage_id]
            recv_comm = recv_comms[0]
            recv_tensor = torch.empty((1, 1, 8), requires_grad=True, dtype=torch.float32).cuda()
            mapping_group = recv_comm.cur_mesh.get_mapping_rank(recv_comm.peer_mesh)
            recv_tensor = recv(recv_tensor, mapping_group, torch.distributed.distributed_c10d._get_default_group())
            recv_dtensor = cross_mesh_recv(recv_comm, recv_tensor)
            return recv_dtensor

        @register_instruction(name="forward")
        def forward():
            model = builder.model
            last_data = builder.last
            activation = model(last_data)
            return activation

        @register_instruction(name="load_data")
        def load_data():
            dataloader = builder.dataloader
            pos = builder.pos
            data_id = pos // 3
            return dataloader[data_id]

        instruction_list = {
            0: "load_data,forward,send,load_data,forward,send,load_data,forward,send",
            1: "recv,forward,recv,forward,recv,forward",
        }
        builder.build_from_dict(instruction_list)
        builder.draw_instructions()

        deferred_model = deferred_init(FourMLP, hidden=8)
        parser = PipeParser()
        pipe_config = PipelineParallelPlan(
            num_stages=2,
            split_method=PipelineSplitMethodType.MANUAL,
            smallest_unsplittable_units=[f"mlp{i + 1}" for i in range(4)],
            split_points=["mlp2", "mlp4"],
        )
        parser_args = {"shard_plan": sharding_plan}
        graph = parser.parse(deferred_model, pipe_config, **parser_args)
        root_graph = parser.partition_stage(deferred_model, graph, pipe_config)

        if self.rank in [0, 1]:
            pipeline_stage_id = 0
        elif self.rank in [2, 3]:
            pipeline_stage_id = 1

        stage_model_pp = root_graph.get_submodule(f"stage{pipeline_stage_id}")

        stage_model_pp_tp = parallelize_module(
            stage_model_pp,
            global_mesh.get_submesh([1]),
            sharding_plan,
            factory=False,
        )

        global_tp_meshes = [
            DeviceMesh("cuda", [0, 1], _validate_mesh=False),
            DeviceMesh("cuda", [2, 3], _validate_mesh=False),
        ]
        np_deps, p2p_index_mapping = _generate_stage_dependencies(root_graph, 2, 1)

        deps = StageDeps(np_deps, global_tp_meshes, [stage_model_pp_tp], p2p_index_mapping)
        builder.topo = deps
        builder.model = stage_model_pp_tp
        builder.stage_id = pipeline_stage_id

        data_iterator = []
        if self.rank in [0, 1]:
            for i in range(3):
                data = torch.zeros((1, 1, 8), dtype=torch.float32) + i
                data_iterator.append(data)
        builder.dataloader = data_iterator
        outputs = builder.run(pipeline_stage_id)
        if self.rank in [2, 3]:
            print(outputs)

    def _define_instructions(self):
        @register_instruction(name="send")
        def send_forward(*args, **kwargs):
            send_data = args[0]
            dst = builder.send_dist
            send(maybe_tensor(send_data), dst, torch.distributed.distributed_c10d._get_default_group())
            return (send_data,), {}

        @register_instruction(name="recv")
        def recv_forward(*args, **kwargs):
            dst = builder.recv_dist
            recv_tensor = torch.empty_like(args[0])
            recv_tensor = recv(recv_tensor, dst, torch.distributed.distributed_c10d._get_default_group())
            return (recv_tensor,), {}

        # instruction should be stateless.
        @register_instruction(name="forward")
        def forward(model, *args, **kwargs):
            activation = model(*args, **kwargs)
            return (activation,), {}

        instruction_list = {
            0: "forward,send",
            1: "recv,forward",
        }

        builder.build_from_dict(instruction_list)
        builder.draw_instructions()

    def _parallelize_model(self, global_mesh):
        deferred_model = deferred_init(FourMLP, hidden=8)
        parser = PipeParser()
        pipe_config = PipelineParallelPlan(
            num_stages=2,
            split_method=PipelineSplitMethodType.MANUAL,
            smallest_unsplittable_units=[f"mlp{i + 1}" for i in range(4)],
            split_points=["mlp2", "mlp4"],
        )
        parser_args = {"shard_plan": sharding_plan}
        graph = parser.parse(deferred_model, **parser_args)
        root_graph = parser.partition_stage(deferred_model, graph, pipe_config)

        if self.rank in [0, 1]:
            pipeline_stage_id = 0
        elif self.rank in [2, 3]:
            pipeline_stage_id = 1

        stage_model_pp = root_graph.get_submodule(f"stage{pipeline_stage_id}")

        tp_submesh = global_mesh.get_submesh([1])
        stage_model_pp_tp = parallelize_module(
            stage_model_pp,
            tp_submesh,
            sharding_plan,
            factory=False,
        )

        return stage_model_pp_tp, root_graph


if __name__ == "__main__":
    run_tests()
