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
    attn = model.layers[0].self_attn
    return attn, config


class AttentionTest(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @with_comms
    def test_attention(self):
        bsz = 6
        s = 18
        hidden_size = 4096
        # -----------golden-----------

        input = torch.rand(bsz, s, hidden_size).cuda()
        input.requires_grad_()
        input.retain_grad()
        non_parallel_attention, _ = get_model()
        non_parallel_attention = non_parallel_attention.cuda()
        golden_outputs = non_parallel_attention(input)
        golden_loss = golden_outputs[0].mean()
        golden_loss.backward()

        # -----------vescale----------
        device_mesh = DeviceMesh(self.device_type, range(self.world_size))
        vescale_attention, config = get_model()
        fwd_resharding_plan = {
            # atten
            ".input": {"hidden_states": [Replicate()]},
            "o_proj.output": [[Shard(1)]],
            ".output": [[Shard(1)], None, None],
        }
        param_sharding_plan = {
            # atten weight, no bias
            "q_proj.weight": [Shard(0)],
            "k_proj.weight": [Shard(0)],
            "v_proj.weight": [Shard(0)],
            "o_proj.weight": [Shard(1)],
        }

        vescale_attention = parallelize_module(
            vescale_attention, device_mesh, {"parameter": param_sharding_plan, "forward": fwd_resharding_plan}
        )

        d_input = distribute_tensor(input.detach(), device_mesh, [Shard(1)])
        d_input.requires_grad_()
        d_input.retain_grad()

        vescale_outputs = vescale_attention(d_input)
        vescale_outputs[0] = vescale_outputs[0].redistribute(placements=[Replicate()] * device_mesh.ndim)
        vescale_loss = vescale_outputs[0].mean()

        vescale_loss.backward()


if __name__ == "__main__":
    run_tests()
