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
from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer

from common_dtensor import DTensorTestBase, with_comms, skip_unless_torch_gpu

torch.manual_seed(9999)


class MixtralDecoderLayerTest(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @skip_unless_torch_gpu
    @with_comms
    def test_tp_mixtral_decoder(
        self,
    ):
        bsz = 6
        seqlen = 18
        config = MixtralConfig()
        hidden_size = config.hidden_size

        device_mesh = DeviceMesh(self.device_type, range(self.world_size))
        base_decoder = MixtralDecoderLayer(config, 0).cuda()
        decoder = copy.deepcopy(base_decoder)

        base_input = torch.rand(bsz, seqlen, hidden_size).cuda()
        input = copy.deepcopy(base_input)

        # =---------------- baseline ----------------= #
        base_output = base_decoder(base_input)[0]
        base_loss = base_output.mean()
        base_loss.backward()

        # =---------------- vescale ----------------= #
        param_sharding_plan = {
            r"input_layernorm.weight": [Replicate()],  # MixtralRMSNorm
            r"self_attn.q_proj.weight": [Shard(0)],
            r"self_attn.k_proj.weight": [Shard(0)],
            r"self_attn.v_proj.weight": [Shard(0)],
            # TODO: buggy, cos_cached or sin_cached can be updated or recreated if seqlen exceeds the max seqlen.
            r"self_attn.rotary_emb.cos_cached": [Replicate()],
            r"self_attn.rotary_emb.sin_cached": [Replicate()],
            r"self_attn.o_proj.weight": [Shard(1)],
            r"post_attention_layernorm.weight": [Replicate()],
            r"block_sparse_moe.gate.weight": [Replicate()],
            r"block_sparse_moe.experts.\d+.w1.weight": [Shard(0)],
            r"block_sparse_moe.experts.\d+.w3.weight": [Shard(0)],
            r"block_sparse_moe.experts.\d+.w2.weight": [Shard(1)],
        }

        fwd_resharding_plan = {
            r"input": [[Replicate()]],
            # No SP
            # r"input_layernorm.input": [[Replicate()]],
            # r"input_layernorm.output": [[Replicate()]],
            # SP
            r"input_layernorm.input": [[Shard(1)]],
            r"input_layernorm.output": [[Shard(1)]],
            # TODO: buggy: attn mask is torch.Tensor, in training, it's a None
            r"self_attn.input": [[Replicate()]],
            r"self_attn.output": {
                "attn_output": [Replicate()],
                "attn_weights": None,
                "past_key_value": None,
            },
            r"self_attn.o_proj.output": [[Replicate()]],
            # No SP
            # r"post_attention_layernorm.input": [[Replicate()]],
            # r"post_attention_layernorm.output": [[Replicate()]],
            # SP
            r"post_attention_layernorm.input": [[Shard(1)]],
            r"post_attention_layernorm.output": [[Shard(1)]],
            r"block_sparse_moe.input": [[Replicate()]],
            r"block_sparse_moe.gate.output": [[Replicate()]],
            r"block_sparse_moe.output": {
                "final_hidden_states": [Replicate()],
                "router_logits": [Replicate()],
            },
            r"block_sparse_moe.experts.\d+.w1.input": [[Replicate()]],
            r"block_sparse_moe.experts.\d+.w3.input": [[Replicate()]],
            r"block_sparse_moe.experts.\d+.w2.output": [[Replicate()]],
            r"output": [[Replicate()]],
        }

        decoder = parallelize_module(
            decoder,
            device_mesh=device_mesh,
            sharding_plan={"parameter": param_sharding_plan, "forward": fwd_resharding_plan},
            factory=True,
        )
        output = decoder(input)[0]
        loss = output.mean()
        loss.backward()

        torch.testing.assert_close(base_output, output._local_tensor)
        torch.testing.assert_close(base_loss, loss._local_tensor)
        for name, base_param in base_decoder.named_parameters():
            param = decoder.get_parameter(name)
            if base_param.grad is None or param.grad is None:
                continue
            base_param_grad = base_param.grad
            param_grad = param.grad.redistribute(device_mesh, [Replicate()], async_op=False)._local_tensor
            torch.testing.assert_close(base_param_grad, param_grad)


if __name__ == "__main__":
    run_tests()
