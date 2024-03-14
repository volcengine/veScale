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

from common_dtensor import DTensorTestBase, with_comms, skip_unless_torch_gpu
from torch.testing._internal.common_utils import run_tests

import torch
from torch import nn

from vescale import DeviceMesh, auto_parallelize_module, distribute_tensor
from vescale.dtensor.placement_types import Replicate

from .nano_gpt import GPTConfig, Block, GPT


class TestNanoGPT(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @skip_unless_torch_gpu
    @with_comms
    def test_block(self):
        B, S = 4, 8

        config = GPTConfig()
        config.block_size = -1
        config.vocab_size = -1
        config.n_layer = -1
        config.n_head = 12
        config.n_embd = 768
        config.dropout = 0.1
        config.bias = True

        class RootDModule(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.m = Block(config)

            def forward(self, x):
                return self.m(x)

        torch.cuda.manual_seed(self.rank)
        with DeviceMesh(self.device_type, torch.arange(self.world_size)):
            model = RootDModule(config)
            dmodel = auto_parallelize_module(model, policy="MEGATRON")
            data = distribute_tensor(torch.empty(B, S, config.n_embd), placements=[Replicate()])
        loss = dmodel(data).to_local().sum()
        loss.backward()

    @skip_unless_torch_gpu
    @with_comms
    def test_gpt(self):
        # -----------------------------------------------------------------------------
        num_iters = 1
        # data
        batch_size = 4
        block_size = 8
        vocab_size = 32
        # model
        n_layer = 2
        n_head = 4
        n_embd = 16
        dropout = 0.1  # for pretraining 0 is good, for finetuning try 0.1+
        bias = True  # do we use bias inside LayerNorm and Linear layers?
        # system
        torch.manual_seed(1337)
        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        # torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
        # -----------------------------------------------------------------------------
        # fake data loader
        data_set = []
        for _ in range(num_iters):
            idx = torch.randint(0, vocab_size, (batch_size, block_size), dtype=torch.int64).cuda()
            target = torch.randint(0, vocab_size, (batch_size, block_size), dtype=torch.int64).cuda()
            data_set.append((idx, target))

        # model config
        model_args = dict(
            block_size=block_size,
            vocab_size=vocab_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dropout=dropout,
            bias=bias,
        )
        gptconf = GPTConfig(**model_args)

        # initialize model
        with DeviceMesh(self.device_type, torch.arange(self.world_size)):  # + + +
            model = GPT(gptconf)
            dmodel = auto_parallelize_module(model, policy="MEGATRON")  # + + +

        # training loop
        for X, Y in data_set:
            logits, loss = dmodel(X, Y)
            loss.backward()

        # debug
        if self.rank == 0:
            device_mesh, used_param_plan, used_fwd_plan = dmodel.dump_mesh_plans()  # + + +
            # print()
            # print(f"************ used_param_plan ************")
            # for k, v in reversed(used_param_plan.items()):
            #     print(f"{k}\t:\t{v}")
            # print()
            # print(f"************ used_fwd_plan ************")
            # for k, v in reversed(used_fwd_plan.items()):
            #     print(f"{k}\t:\t{v}")
            # print()


if __name__ == "__main__":
    run_tests()
