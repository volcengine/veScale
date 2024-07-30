################################################################################
#
# Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
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

from vescale.dtensor.device_mesh import DeviceMesh, init_device_mesh
from vescale.dtensor.api import distribute_tensor
from vescale.dtensor.placement_types import Replicate, Shard
from vescale.ddp.distributed_data_parallel import DistributedDataParallel as DDP
from vescale.optim.distributed_optimizer import DistributedOptimizer
from vescale.dmodule.api import parallelize_module

from common_dtensor import DTensorTestBase, with_comms

from transformers import AutoConfig, LlamaModel


dir_path = os.path.dirname(os.path.realpath(__file__))


def get_model(layer_number=None):
    config = AutoConfig.from_pretrained(os.path.join(dir_path, "config.json"))
    if layer_number is None:
        config.num_hidden_layers = 1
    else:
        config.num_hidden_layers = layer_number
    model = LlamaModel(config)
    model = model
    return model, config


decoder_fwd_resharding_plan = {
    "input": {
        "hidden_states": [Shard(1)],
        # "attention_mask": [Replicate()],
        "position_ids": [Replicate()],
    },
    # atten
    "self_attn.input": {
        "hidden_states": [Replicate()],
    },
    "self_attn.o_proj.output": [[Shard(1)]],
    "self_attn.output": [[Shard(1)], None, None],
    # feedforward(mlp) no bias
    "mlp.input": [[Replicate()]],
    "mlp.output": [[Shard(1)]],
    "output": [[Shard(1)], None],
}
decoder_param_sharding_plan = {
    # atten weight, no bias
    "self_attn.q_proj.weight": [Shard(0)],
    "self_attn.k_proj.weight": [Shard(0)],
    "self_attn.v_proj.weight": [Shard(0)],
    "self_attn.o_proj.weight": [Shard(1)],
    # feedforward(mlp)
    "mlp.up_proj.weight": [Shard(0)],
    "mlp.gate_proj.weight": [Shard(0)],
    "mlp.down_proj.weight": [Shard(1)],
}

model_fwd_resharding_plan = {
    ".input": [[Replicate()]],
    "norm.input": [[Shard(1)]],
    ".output": {
        "last_hidden_state": [Replicate()],
    },
    **{rf"layers.\d+.{k}": v for k, v in decoder_fwd_resharding_plan.items()},
}
model_param_sharding_plan = {
    "embed_tokens.weight": [Shard(1)],
    **{rf"layers.\d+.{k}": v for k, v in decoder_param_sharding_plan.items()},
}


class llama2Test(DTensorTestBase):
    @property
    def world_size(self):
        return 8

    @with_comms
    def test_llama2_layer4(self):
        bsz = 6
        s = 18
        # -----------golden-----------

        non_parallel_llama2, config = get_model(layer_number=4)
        input = torch.randint(low=0, high=config.vocab_size, size=(bsz, s)).cuda()
        non_parallel_llama2 = non_parallel_llama2.cuda()
        golden_output = non_parallel_llama2(input).last_hidden_state
        golden_loss = golden_output.mean()
        golden_loss.backward()

        # -----------vescale----------
        device_mesh = DeviceMesh(self.device_type, range(self.world_size))
        vescale_model, config = get_model(layer_number=4)

        vescale_model = parallelize_module(
            vescale_model,
            device_mesh,
            {"parameter": model_param_sharding_plan, "forward": model_fwd_resharding_plan},
        )

        d_input = distribute_tensor(input.detach(), device_mesh, [Shard(1)])

        vescale_output = vescale_model(d_input).last_hidden_state
        vescale_output = vescale_output.redistribute(placements=[Replicate()] * device_mesh.ndim)
        vescale_loss = vescale_output.mean()

        vescale_loss.backward()
        vescale_model.finish_grad_sync()

    @with_comms
    def test_llama2_layer32_with_ddp(self):
        bsz = 6
        s = 18
        device_mesh = init_device_mesh(self.device_type, (2, 4), mesh_dim_names=("DP", "TP"))
        vescale_model, config = get_model()
        input = torch.randint(low=0, high=config.vocab_size, size=(bsz, s)).cuda()

        vescale_model = parallelize_module(
            vescale_model,
            device_mesh["TP"],
            {"parameter": model_param_sharding_plan, "forward": model_fwd_resharding_plan},
        )

        ddp_model = DDP(
            vescale_model,
            data_pg_or_device_mesh=device_mesh["DP"],
            use_distributed_optimizer=True,
        )
        orig_optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.01)

        ve_optimizer = DistributedOptimizer(
            orig_optimizer,
            clip_grad=0.0,
            overlap_param_gather=True,
            models=[ddp_model],
        )

        ve_optimizer.zero_grad()
        vescale_output = ddp_model(input.detach()).last_hidden_state
        vescale_loss = vescale_output.mean()
        vescale_loss.backward()
        ve_optimizer.step()


if __name__ == "__main__":
    run_tests()
