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

from common_dtensor import DTensorTestBase, with_comms
from vescale.pipe._schedules.instruction_base import StageDeps
from vescale.devicemesh_api import VESCALE_DEVICE_MESH
import numpy as np
import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests
from vescale.plan.spec import PipelineP2PSpec


class MLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.fc1 = nn.Linear(n_features, n_features * 2, bias=False)
        torch.nn.init.uniform_(self.fc1.weight, 0, 1)
        self.fc2 = nn.Linear(n_features * 2, n_features)
        torch.nn.init.uniform_(self.fc2.weight, 0, 1)
        self.gelu = nn.GELU()

    def forward(self, x, y=None):
        out = self.fc2(self.gelu(self.fc1(x)))
        if y is not None:
            out = out + y
        return out


class FourMLP(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.mlp1 = MLP(hidden)
        self.mlp2 = MLP(hidden)
        self.mlp3 = MLP(hidden)
        self.mlp4 = MLP(hidden)

    def forward(self, x):
        stage1 = self.mlp1(x)
        stage2 = self.mlp2(stage1)
        stage3 = self.mlp3(stage2, x)
        stage4 = self.mlp4(stage3)
        return stage4


class MultiStageCommTest(DTensorTestBase):
    def test_send_order(self):
        """
        Tests send order.

        stage 0: a , c
        stage 1: b
        stage 2: dataloader

        stage 2: forward(c,b,dataloader,a)

        """
        a = torch.tensor(0)
        b = torch.tensor(1)
        c = torch.tensor(2)
        d = torch.tensor(3)
        p2p_tensors = [a, c, b]
        p2p_index = [PipelineP2PSpec(0, 2), PipelineP2PSpec(1, 0), PipelineP2PSpec(2, 0), PipelineP2PSpec(0, 0)]
        local_inputs = [d]

        p2p_index_without_local = list(filter(lambda item: item.peer_stage_idx != 2, p2p_index))
        p2p_send_order = sorted(p2p_index_without_local, key=lambda x: (x.peer_stage_idx, x.peer_output_idx))
        p2p_tensor_order = [p2p_send_order.index(item) for item in p2p_index_without_local]
        ordered_p2p_tensors = [p2p_tensors[x] for x in p2p_tensor_order]

        assert ordered_p2p_tensors == [c, b, a]

        args = []
        local_input_mapping = list(filter(lambda item: item.peer_stage_idx == 2, p2p_index))
        for item in p2p_index:
            if item.peer_stage_idx == 2:
                index = local_input_mapping.index(item)
                args.append(local_inputs[index])
            else:
                index = p2p_send_order.index(item)
                args.append(p2p_tensors[index])
        assert args == [c, b, d, a]

    @with_comms
    def test_stage_deps(self):
        """
        Tests abstraction of inter-stage communication dependency.
        """
        # initialize global device mesh
        VESCALE_DEVICE_MESH.init_device_mesh(
            device_type="cuda",
            mesh_shape=(4, 1, 1),
            mesh_dim_names=("PP", "DP", "TP"),
        )
        print(VESCALE_DEVICE_MESH.get())

        # case 1 - sequential input is one
        single_deps = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
        stage = StageDeps(
            single_deps,
            VESCALE_DEVICE_MESH.get_global_pipeline_parallel_meshes(),
            [],
        )
        if torch.distributed.distributed_c10d.get_rank() == 0:
            print(stage)

        # case 2 - sequential multi input
        single_deps = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
        p2p_index_mapping = {1: [PipelineP2PSpec(0, 0), PipelineP2PSpec(0, 1)]}
        stage = StageDeps(
            single_deps,
            VESCALE_DEVICE_MESH.get_global_pipeline_parallel_meshes(),
            [],
            p2p_index_mapping=p2p_index_mapping,
        )
        if torch.distributed.distributed_c10d.get_rank() == 0:
            print(stage)

        # case 3 - sequential multi input with local_dataloader
        """
        The adjacency matrix for 4 stages is formulated as a 4x4 matrix. The meaning can be interpreted as followed:
            Row (Stage) 0: [0, 1, 0, 0]. stage 0 sends output to stage 1 (index position 1).
            Row (Stage) 1: [0, 0, 1, 0]: stage 1 sends output to stage 2 (index position 2).
            Row (Stage) 2: [0, 0, 0, 1]: stage 2 sends output to stage 3 (index position 3).
            Row (Stage) 3: [0, 0, 0, 0]: stage 3 sends no output to any other stage.
        """
        single_deps = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
        p2p_index_mapping = {1: [PipelineP2PSpec(0, 2), PipelineP2PSpec(1, 0), PipelineP2PSpec(0, 0)]}
        stage = StageDeps(
            single_deps,
            VESCALE_DEVICE_MESH.get_global_pipeline_parallel_meshes(),
            [],
            p2p_index_mapping=p2p_index_mapping,
        )
        if torch.distributed.distributed_c10d.get_rank() == 0:
            print(stage)

        # case 4 - multi branch input with single data
        """
        The adjacency matrix for 4 stages is formulated as a 4x4 matrix. The meaning can be interpreted as followed:
            Row (Stage) 0: [0, 1, 0, 0]. stage 0 sends output to stage 1 (index position 1).
            Row (Stage) 1: [0, 0, 1, 0]: stage 1 sends output to stage 2 (index position 2).
            Row (Stage) 2: [0, 0, 0, 1]: stage 2 sends output to stage 3 (index position 3).
            Row (Stage) 3: [0, 0, 0, 0]: stage 3 sends no output to any other stage.
        """
        single_deps = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
        p2p_index_mapping = {2: [PipelineP2PSpec(0, 0), PipelineP2PSpec(1, 0)]}
        stage = StageDeps(
            single_deps,
            VESCALE_DEVICE_MESH.get_global_pipeline_parallel_meshes(),
            [],
            p2p_index_mapping=p2p_index_mapping,
        )
        if torch.distributed.distributed_c10d.get_rank() == 0:
            print(stage)

        # case 5 - vpp test
        """
        The adjacency matrix for 4 stages is formulated as a 4x4 matrix. The meaning can be interpreted as followed:
            Row (Stage) 0: [0, 1, 0, 0]. stage 0 sends output to stage 1 (index position 1).
            Row (Stage) 1: [0, 0, 1, 0]: stage 1 sends output to stage 2 (index position 2).
            Row (Stage) 2: [0, 0, 0, 1]: stage 2 sends output to stage 3 (index position 3).
            Row (Stage) 3: [0, 0, 0, 0]: stage 3 sends no output to any other stage.
        """
        single_deps = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
        stage = StageDeps(
            single_deps,
            VESCALE_DEVICE_MESH.get_global_pipeline_parallel_meshes(),
            [0, 1],
        )
        if torch.distributed.distributed_c10d.get_rank() == 0:
            print(stage)


if __name__ == "__main__":
    run_tests()
