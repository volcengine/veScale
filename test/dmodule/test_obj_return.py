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
from torch import nn

from torch.testing._internal.common_utils import run_tests, instantiate_parametrized_tests, parametrize, subtest

from vescale.dtensor.device_mesh import DeviceMesh
from vescale.dtensor.api import distribute_tensor, DTensor
from vescale.dtensor.placement_types import Replicate, Shard
from vescale.dmodule.api import parallelize_module

from dataclasses import dataclass

from common_dtensor import DTensorTestBase, with_comms


test_type_candidates = []

try:
    from transformers.utils.generic import ModelOutput
    from transformers.modeling_outputs import BaseModelOutput

    test_type_candidates.append(subtest(ModelOutput, name="dict_like"))
    test_type_candidates.append(subtest(BaseModelOutput, name="mixed_data_class_dict"))
except ImportError as _:
    pass


@dataclass
class DataclassReturnType:
    last_hidden_state: torch.Tensor = None
    hidden_states: torch.Tensor = None


test_type_candidates.append(subtest(DataclassReturnType, name="data_class"))


class TestModel(nn.Module):
    def __init__(self, rt_type, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.rt_type = rt_type

    def forward(self, x):
        return self.rt_type(
            last_hidden_state=x,
            hidden_states=x + 1,
        )


class ObjReturnTest(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @with_comms
    @parametrize("rt_type", test_type_candidates)
    def test_dataclass_return(self, rt_type):
        device_mesh = DeviceMesh("cuda", range(self.world_size))
        param_sharding_plan = {}
        fwd_resharding_plan = {".output": {"last_hidden_state": [Replicate()], "hidden_states": [Shard(0)]}}

        sharding_plan = {"parameter": param_sharding_plan, "forward": fwd_resharding_plan}

        dmodule = parallelize_module(TestModel(rt_type), device_mesh, sharding_plan)

        input = torch.rand(4, 4, 4).cuda()
        d_input = distribute_tensor(input.detach(), device_mesh, [Shard(1)])

        output = dmodule(d_input)

        self.assertIsInstance(output, rt_type)
        self.assertIsInstance(output.last_hidden_state, DTensor)
        self.assertIsInstance(output.hidden_states, DTensor)
        self.assertTrue(output.last_hidden_state.placements[0].is_replicate())
        self.assertTrue(output.hidden_states.placements[0].is_shard(0))


instantiate_parametrized_tests(ObjReturnTest)

if __name__ == "__main__":
    run_tests()
