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

import copy
import torch
from torch.testing._internal.common_utils import run_tests

from vescale.dtensor.device_mesh import DeviceMesh
from vescale.dtensor.placement_types import Replicate, Shard
from vescale.dmodule.api import parallelize_module

from transformers.models.mixtral.configuration_mixtral import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralAttention

from common_dtensor import DTensorTestBase, with_comms, skip_unless_torch_gpu

torch.manual_seed(9999)


class MixtralAttentionBlockTest(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @skip_unless_torch_gpu
    @with_comms
    def test_tp_mixtral_attn(
        self,
    ):
        bsz = 6
        seqlen = 18
        config = MixtralConfig()
        hidden_size = config.hidden_size

        device_mesh = DeviceMesh(self.device_type, range(self.world_size))
        base_attn = MixtralAttention(config, 0).cuda()
        attn = copy.deepcopy(base_attn)

        base_input = torch.rand(bsz, seqlen, hidden_size).cuda()
        input = copy.deepcopy(base_input)

        # =---------------- baseline ----------------= #
        base_output, _, _ = base_attn(base_input)
        base_loss = base_output.mean()
        base_loss.backward()

        # =---------------- vescale ----------------= #
        param_sharding_plan = {
            r"q_proj.weight": [Shard(0)],
            r"k_proj.weight": [Shard(0)],
            r"v_proj.weight": [Shard(0)],
            # TODO: buggy, cos_cached or sin_cached can be updated or recreated if seqlen exceeds the max seqlen.
            r"rotary_emb.cos_cached": [Replicate()],
            r"rotary_emb.sin_cached": [Replicate()],
            r"o_proj.weight": [Shard(1)],
        }
        fwd_resharding_plan = {
            r"input": [[Replicate()]],
            r"output": {"attn_output": [Replicate()], "attn_weights": None, "past_key_value": None},
            r"o_proj.output": [[Replicate()]],
        }

        attn = parallelize_module(
            attn,
            device_mesh=device_mesh,
            sharding_plan={"parameter": param_sharding_plan, "forward": fwd_resharding_plan},
            factory=True,
        )
        output, _, _ = attn(input)
        loss = output.mean()
        loss.backward()

        torch.testing.assert_close(base_output, output._local_tensor, atol=1e2, rtol=1e2)
        torch.testing.assert_close(base_loss, loss._local_tensor, atol=1e2, rtol=1e2)
        for fc_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            base_param_grad = base_attn.get_parameter(f"{fc_name}.weight").grad
            param_grad = (
                attn.get_parameter(f"{fc_name}.weight")
                .grad.redistribute(device_mesh, [Replicate()], async_op=False)
                ._local_tensor
            )
            torch.testing.assert_close(base_param_grad, param_grad, atol=1e2, rtol=1e2)


if __name__ == "__main__":
    run_tests()
