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
import torch
from torch.testing._internal.common_utils import run_tests
from torch.distributed import get_rank
from torch.distributed.distributed_c10d import get_process_group_ranks
from vescale.devicemesh_api import veDeviceMesh
from vescale.dtensor.device_mesh import DeviceMesh
from common_dtensor import DTensorTestBase, with_comms


class TestBasicAPI(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @with_comms
    def test_initialize(self):
        """
        Test utilities to initialize global DeviceMesh.
        """
        # the initialized global device mesh is an outcome of initializing veDeviceMesh API
        global_device_mesh = veDeviceMesh.init_device_mesh(
            device_type="cuda",
            mesh_shape=(2, 2),
            mesh_dim_names=("DP", "TP"),
        )
        device_mesh = DeviceMesh(self.device_type, torch.tensor([[0, 1], [2, 3]]))
        self.assertEqual(global_device_mesh.mesh, device_mesh.mesh)
        self.assertEqual(global_device_mesh, veDeviceMesh.get())
        initial_config = {
            "device_type": "cuda",
            "mesh_shape": (2, 2),
            "mesh_dim_names": ("dp", "tp"),
        }
        # Taking as input parameters of veDeviceMesh.init_device_mesh, get() can initialize global DeviceMesh
        second_global_device_mesh = veDeviceMesh.get(**initial_config)
        self.assertEqual(veDeviceMesh.get().mesh, second_global_device_mesh.mesh)

    @with_comms
    def test_basic_properties(self):
        """
        Test utilities to perform basic properties inherited from upstream DeviceMesh.
        """
        # veDeviceMesh returns the global device mesh upon which is is initialized
        _ = veDeviceMesh.init_device_mesh(
            device_type="cuda",
            mesh_shape=(2, 2),
            mesh_dim_names=("DP", "TP"),
        )
        self.assertEqual(veDeviceMesh.shape, tuple([2, 2]))
        self.assertEqual(veDeviceMesh.ndim, 2)
        self.assertEqual(veDeviceMesh.size(), 4)
        self.assertEqual(veDeviceMesh.size(0), 2)
        self.assertEqual(veDeviceMesh.size(1), 2)
        self.assertFalse("PP" in veDeviceMesh._MESH_DIM_NAMES_LOOKUP)
        dp_mesh = veDeviceMesh["DP"]
        dp_submesh_mesh = dp_mesh.mesh.tolist()
        tp_mesh = veDeviceMesh["TP"]
        tp_submesh_mesh = tp_mesh.mesh.tolist()
        # upstream DeviceMesh's get_coordinate utility
        strategy_coordinate = veDeviceMesh.get_coordinate()
        if get_rank() == 0:
            self.assertEqual(dp_submesh_mesh, [0, 2])
            self.assertEqual(tp_submesh_mesh, [0, 1])
            self.assertEqual(strategy_coordinate, [0, 0])
        if get_rank() == 2:
            self.assertEqual(dp_submesh_mesh, [0, 2])
            self.assertEqual(tp_submesh_mesh, [2, 3])
            self.assertEqual(strategy_coordinate, [1, 0])

    @with_comms
    def test_basic_utils(self):
        """
        Test utilities to perform basic utilities with regards to local ranks and strategies.
        """
        # veDeviceMesh returns the global device mesh upon which is is initialized
        _ = veDeviceMesh.init_device_mesh(
            device_type="cuda",
            mesh_shape=(2, 2),
            mesh_dim_names=("DP", "TP"),
        )
        self.assertEqual(veDeviceMesh.get_local_rank(), get_rank())
        self.assertEqual(veDeviceMesh.get_strategy_size(0), veDeviceMesh.get_strategy_size("DP"))
        self.assertEqual(veDeviceMesh.get_strategy_size("TP"), 2)
        self.assertEqual(veDeviceMesh.lookup_rank("TP"), veDeviceMesh.get_strategy_coordinate()[1])
        self.assertEqual(veDeviceMesh.lookup_rank("DP"), veDeviceMesh.get_strategy_coordinate()[0])
        self.assertEqual(veDeviceMesh.get_strategy_coordinate(local_rank=0), [0, 0])
        self.assertEqual(veDeviceMesh.get_strategy_coordinate(local_rank=3), [1, 1])


class TestStrategyUtil(DTensorTestBase):
    @property
    def world_size(self):
        return 8

    @with_comms
    def test_strategy_rank(self):
        """
        Test utilities to get id of a global rank along dimensions.
        """
        # the initialized global device mesh is an outcome of initializing veDeviceMesh API
        device_mesh_one = veDeviceMesh.init_device_mesh(
            device_type="cuda",
            mesh_shape=(2, 2, 2),
            mesh_dim_names=("PP", "DP", "TP"),
        )
        pp_rank = veDeviceMesh.get_pipeline_parallel_rank()
        dp_rank = veDeviceMesh.get_data_parallel_rank()
        tp_rank = veDeviceMesh.get_tensor_parallel_rank()
        if get_rank() == 7:
            self.assertEqual((pp_rank, dp_rank, tp_rank), (1, 1, 1))
        # now update a new global device mesh
        device_mesh_two = veDeviceMesh.init_device_mesh(
            device_type="cuda",
            mesh_shape=(4, 1, 2),
            mesh_dim_names=("PP", "DP", "TP"),
        )
        pp_rank_two = veDeviceMesh.get_pipeline_parallel_rank()
        dp_rank_two = veDeviceMesh.get_data_parallel_rank()
        tp_rank_two = veDeviceMesh.get_tensor_parallel_rank()
        if get_rank() == 0:
            self.assertEqual((pp_rank_two, dp_rank_two, tp_rank_two), (0, 0, 0))
        if get_rank() == 7:
            self.assertEqual((pp_rank_two, dp_rank_two, tp_rank_two), (3, 0, 1))

    @with_comms
    def test_strategy_mesh(self):
        """
        Test veDeviceMesh utilities to generate sub-DeviceMesh along a parallel dimension.
        """
        # veDeviceMesh returns the global device mesh upon which is is initialized
        _ = veDeviceMesh.init_device_mesh(
            device_type="cuda",
            mesh_shape=(2, 2, 2),
            mesh_dim_names=("PP", "DP", "TP"),
        )
        # sub-DeviceMesh for TP view
        tp_mesh = veDeviceMesh.get_tensor_parallel_mesh()
        # sub-DeviceMesh for DP view
        dp_mesh = veDeviceMesh.get_data_parallel_mesh()
        # sub-DeviceMesh for PP view (2 stages)
        pp_mesh = veDeviceMesh.get_pipeline_parallel_mesh()
        if get_rank() == 6:
            self.assertEqual(tp_mesh.mesh.tolist(), [6, 7])
            self.assertEqual(dp_mesh.mesh.tolist(), [4, 6])
            self.assertEqual(pp_mesh.mesh.tolist(), [6, 7])

    @with_comms
    def test_process_groups(self):
        """
        Test veDeviceMesh utilities to query process groups in Omnistore
        and distributed data parallel APIs.
        """
        # the initialized global device mesh is an outcome of initializing veDeviceMesh API
        device_mesh_one = veDeviceMesh.init_device_mesh(
            device_type="cuda",
            mesh_shape=(2, 1, 4),
            mesh_dim_names=("PP", "DP", "TP"),
        )
        tp_process_group = veDeviceMesh.get_tensor_parallel_dim_groups()
        dp_process_group = veDeviceMesh.get_data_parallel_dim_groups()
        tp_member_ranks = get_process_group_ranks(tp_process_group)
        dp_member_ranks = get_process_group_ranks(dp_process_group)
        if get_rank() == 4:
            self.assertEqual(tp_member_ranks, [0, 4])
            self.assertEqual(dp_member_ranks, [4])
        if get_rank() == 5:
            self.assertEqual(tp_member_ranks, [1, 5])
            self.assertEqual(dp_member_ranks, [5])
        # now update a new global device mesh
        device_mesh_two = veDeviceMesh.init_device_mesh(
            device_type="cuda",
            mesh_shape=(4, 2),
            mesh_dim_names=("DP", "TP"),
        )
        tp_process_group = veDeviceMesh.get_tensor_parallel_dim_groups()
        dp_process_group = veDeviceMesh.get_data_parallel_dim_groups()
        tp_member_ranks = get_process_group_ranks(tp_process_group)
        dp_member_ranks = get_process_group_ranks(dp_process_group)
        if get_rank() == 4:
            self.assertEqual(tp_member_ranks, [0, 2, 4, 6])
            self.assertEqual(dp_member_ranks, [0, 2, 4, 6])
        if get_rank() == 5:
            self.assertEqual(tp_member_ranks, [1, 3, 5, 7])
            self.assertEqual(dp_member_ranks, [1, 3, 5, 7])

    @with_comms
    def test_global_meshes(self):
        """
        Test veDeviceMesh utilities to retrieve a list of tensor parallel,
        and pipeline parallel submeshes.
        """
        # veDeviceMesh returns the global device mesh upon which is is initialized
        device_mesh = veDeviceMesh.init_device_mesh(
            device_type="cuda",
            mesh_shape=(4, 1, 2),
            mesh_dim_names=("PP", "DP", "TP"),
        )
        tensor_parallel_meshes = veDeviceMesh.get_global_tensor_parallel_meshes()
        tensor_meshes = [item.mesh.tolist() for item in tensor_parallel_meshes]
        self.assertEqual(tensor_meshes, [[0, 1], [2, 3], [4, 5], [6, 7]])
        pipeline_parallel_meshes = veDeviceMesh.get_global_pipeline_parallel_meshes()
        pipeline_meshes = [item.mesh.tolist() for item in pipeline_parallel_meshes]
        self.assertEqual(pipeline_meshes, [[[0, 1]], [[2, 3]], [[4, 5]], [[6, 7]]])

    @with_comms
    def test_stage_query(self):
        """
        Test veDeviceMesh utilities to query whether current pipeline stage
        is the first and last stage.
        """
        # veDeviceMesh returns the global device mesh upon which is is initialized
        device_mesh = veDeviceMesh.init_device_mesh(
            device_type="cuda",
            mesh_shape=(4, 1, 2),
            mesh_dim_names=("PP", "DP", "TP"),
        )
        self.assertEqual(veDeviceMesh.is_first_stage(), veDeviceMesh.get_pipeline_parallel_rank() == 0)
        self.assertEqual(
            veDeviceMesh.is_last_stage(),
            veDeviceMesh.get_pipeline_parallel_rank() == veDeviceMesh.get_strategy_size("PP") - 1,
        )


if __name__ == "__main__":
    run_tests()
