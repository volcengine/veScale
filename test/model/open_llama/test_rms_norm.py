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

import os
import torch
from torch.testing._internal.common_utils import run_tests

from vescale.dtensor.device_mesh import DeviceMesh
from vescale.dtensor.api import distribute_tensor
from vescale.dtensor.placement_types import Replicate, Shard
from vescale.dmodule.api import parallelize_module

from common_dtensor import DTensorTestBase, with_comms

from transformers import AutoConfig, LlamaModel

dir_path = os.path.dirname(os.path.realpath(__file__))


def get_model():
    config = AutoConfig.from_pretrained(os.path.join(dir_path, "config.json"))
    config.num_hidden_layers = 1
    model = LlamaModel(config)
    rms_norm = model.layers[0].input_layernorm
    return rms_norm


class RMSNormTest(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @with_comms
    def test_rms_norm(self):
        bsz = 6
        s = 18
        hidden_size = 4096
        # -----------golden-----------

        input = torch.rand(bsz, s, hidden_size).cuda()
        input.requires_grad_()
        input.retain_grad()
        non_parallel_norm = get_model().cuda()
        golden_output = non_parallel_norm(input)
        golden_loss = golden_output.mean()
        golden_loss.backward()

        # -----------vescale----------
        device_mesh = DeviceMesh(self.device_type, range(self.world_size))
        vescale_norm = get_model().cuda()
        fwd_resharding_plan = {
            ".input": [[Shard(1)]],
        }
        param_sharding_plan = {}

        vescale_norm = parallelize_module(
            vescale_norm,
            device_mesh,
            {"parameter": param_sharding_plan, "forward": fwd_resharding_plan},
        )
        d_input = distribute_tensor(input.detach(), device_mesh, [Shard(1)])
        d_input.requires_grad_()
        d_input.retain_grad()

        vescale_output = vescale_norm(d_input)
        vescale_output = vescale_output.redistribute(placements=[Replicate()] * device_mesh.ndim)
        vescale_loss = vescale_output.mean()

        vescale_loss.backward()
        vescale_norm.finish_grad_sync()
        d_gard = d_input.grad.redistribute(placements=[Replicate()] * device_mesh.ndim)
        torch.testing.assert_close(vescale_output._local_tensor, golden_output)
        torch.testing.assert_close(vescale_loss._local_tensor, golden_loss)
        torch.testing.assert_close(d_gard._local_tensor, input.grad)
        torch.testing.assert_close(vescale_norm.weight.grad._local_tensor, non_parallel_norm.weight.grad)


if __name__ == "__main__":
    run_tests()
