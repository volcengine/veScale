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
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

from common_dtensor import DTensorTestBase, with_comms, skip_unless_torch_gpu

torch.manual_seed(9999)


class MixtralSparseMoeBlockTest(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @skip_unless_torch_gpu
    @with_comms
    def test_tp_moe(
        self,
    ):
        bsz = 6
        seqlen = 18
        config = MixtralConfig()
        hidden_size = config.hidden_size

        device_mesh = DeviceMesh(self.device_type, range(self.world_size))
        base_moe = MixtralSparseMoeBlock(config).cuda()
        moe = copy.deepcopy(base_moe)

        base_input = torch.rand(bsz, seqlen, hidden_size).cuda()
        input = copy.deepcopy(base_input)

        # =---------------- baseline ----------------= #
        base_output, _ = base_moe(base_input)
        base_loss = base_output.mean()
        base_loss.backward()

        # =---------------- vescale ----------------= #
        param_sharding_plan = {
            r"gate.weight": [Replicate()],
            r"experts.\d+.w1.weight": [Shard(0)],
            r"experts.\d+.w3.weight": [Shard(0)],
            r"experts.\d+.w2.weight": [Shard(1)],
        }
        fwd_resharding_plan = {
            r"input": [[Replicate()]],
            r"gate.output": [[Replicate()]],
            r"output": {"final_hidden_states": [Replicate()], "router_logits": [Replicate()]},
            r"experts.\d+.w1.input": [[Replicate()]],
            r"experts.\d+.w3.input": [[Replicate()]],
            r"experts.\d+.w2.output": [[Replicate()]],
        }

        moe = parallelize_module(
            moe,
            device_mesh=device_mesh,
            sharding_plan={"parameter": param_sharding_plan, "forward": fwd_resharding_plan},
            factory=True,
        )
        output, _ = moe(input)
        loss = output.mean()
        loss.backward()

        torch.testing.assert_close(base_output, output._local_tensor, atol=1e2, rtol=1e2)
        torch.testing.assert_close(base_loss, loss._local_tensor, atol=1e2, rtol=1e2)
        for i in range(config.num_local_experts):
            for fc_name in ["w1", "w2", "w3"]:
                base_param = base_moe.get_parameter(f"experts.{i}.{fc_name}.weight")
                param = moe.get_parameter(f"experts.{i}.{fc_name}.weight")
                if param.grad is None or base_param.grad is None:
                    continue
                base_param_grad = base_param.grad
                param_grad = param.grad.redistribute(device_mesh, [Replicate()], async_op=False)._local_tensor
                torch.testing.assert_close(base_param_grad, param_grad, atol=1e2, rtol=1e2)
        base_gate_grad = base_moe.get_parameter("gate.weight").grad
        gate_grad = moe.get_parameter("gate.weight").grad._local_tensor
        torch.testing.assert_close(base_gate_grad, gate_grad, atol=1e2, rtol=1e2)


if __name__ == "__main__":
    run_tests()
