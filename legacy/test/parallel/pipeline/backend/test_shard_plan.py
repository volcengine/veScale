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
from common_dtensor import DTensorTestBase, with_comms
from vescale.pipe.pipe_parser import PipeParser
from vescale.initialize.deferred_init import deferred_init
from eight_mlp import EightMLP, sharding_plan
from vescale.dmodule.api import parallelize_module
from vescale.dtensor.api import distribute_tensor
from vescale.devicemesh_api.api import VESCALE_DEVICE_MESH
from vescale.dtensor.placement_types import Replicate
from vescale.plan import PipelineParallelPlan, PipelineSplitMethodType


class ShardPlanRegistrationTest(DTensorTestBase):
    @with_comms
    def test_manual_split_register_hook(self):
        """
        Tests manual stage split and registers hooks.
        """
        VESCALE_DEVICE_MESH.init_device_mesh("cuda", (2, 1, 2), mesh_dim_names=("PP", "DP", "TP"))
        deferred_mlp = deferred_init(EightMLP, hidden=8)
        partition_units = ["mlp1", "mlp8"]
        pipe_config = PipelineParallelPlan(
            num_stages=2,
            split_method=PipelineSplitMethodType.UNIFORM,
            smallest_unsplittable_units=partition_units,
        )
        pipe_parser = PipeParser()
        input = torch.randn((3, 8))
        model_graph = pipe_parser.parse(
            deferred_mlp,
            pipe_config,
            **{"shard_plan": sharding_plan},
        )
        pipe_spec = pipe_parser.partition_stage(deferred_mlp, model_graph, pipe_config)
        model_chunks = []
        model_partition = pipe_spec.stage0
        model = parallelize_module(
            model_partition, VESCALE_DEVICE_MESH.get_tensor_parallel_mesh(), sharding_plan, factory=False
        )

        # hooks are successfully registered on target modules, as they now have been hierarchically flattened!
        def hook(sel, args):
            print("hook registered. Successful registration will trigger this printout!")
            return args

        model.get_submodule("mlp1").register_forward_pre_hook(hook)
        d_input = distribute_tensor(input, VESCALE_DEVICE_MESH.get_tensor_parallel_mesh(), [Replicate()])
        d_out = model(d_input)
        model_chunks.append(model)
        assert model_chunks[0].mlp1.fc1.weight._spec.placements[0].is_shard()


if __name__ == "__main__":
    run_tests()
