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
from torch.testing._internal.common_utils import (
    run_tests,
)
import torch.distributed as dist
from common_dtensor import DTensorTestBase, skip_unless_torch_gpu, with_comms

from vescale.dtensor.dtensor import DTensor
from vescale.dtensor.device_mesh import DeviceMesh
from vescale.dtensor.placement_types import Replicate
from vescale.dmodule.api import parallelize_module
from vescale.ddp.distributed_data_parallel import DistributedDataParallel as DDP
from vescale.optim.base_optimizer import BasicOptimizer
from vescale.optim.distributed_optimizer import DistributedOptimizer
from vescale.initialize.deferred_init import deferred_init

from transformers.models.mixtral.configuration_mixtral import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralModel
from sharding_plan import mixtral_plan


torch.manual_seed(9999)

vocab_size = 30  # default 32000
hidden_size = 64  # default 4096
# TODO: if changed to use default intermediate_size, accuracy error: 0.016
intermediate_size = 128  # default 14336
num_hidden_layers = 2  # default 32
num_attention_heads = 16  # default 32
num_key_value_heads = 8  # default 8
attn_implementation = "eager"  # options are ["eager", "sdpa", "flash_attention_2"]
bsz = 7
seqlen = 9

mixtral_config = MixtralConfig(
    vocab_size=vocab_size,
    hidden_size=hidden_size,
    intermediate_size=intermediate_size,
    num_hidden_layers=num_hidden_layers,
    num_attention_heads=num_attention_heads,
    num_key_value_heads=num_key_value_heads,
)


class MixtralTPSPTest(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    def gen_golden(self, mixtral_model, x):
        outs = mixtral_model(x)
        hidden_states = outs.last_hidden_state
        hidden_states.sum().backward()

    def compare_model_weights_and_grads(self, base_model, model):
        for name, base_param in base_model.named_parameters():
            param = model.get_parameter(name)
            base_grad = base_param.grad.data
            grad = param.grad
            if isinstance(param, DTensor):
                param = param.redistribute(param.device_mesh, [Replicate()], async_op=False)._local_tensor

            torch.testing.assert_close(param, base_param)
            if isinstance(grad.data, DTensor):
                grad = grad.data.redistribute(grad.data.device_mesh, [Replicate()], async_op=False)._local_tensor
            torch.testing.assert_close(base_grad, grad, atol=1e-4, rtol=1e-4)

    @skip_unless_torch_gpu
    @with_comms
    def test_tp_sp(self):
        device_mesh = DeviceMesh("cuda", range(self.world_size))
        mixtral_model = MixtralModel(mixtral_config).cuda()
        base_mixtral_model = copy.deepcopy(mixtral_model)

        mixtral_model = parallelize_module(
            mixtral_model,
            device_mesh,
            mixtral_plan,
            factory=True,
        )

        token_ids = torch.randint(0, vocab_size, (bsz, seqlen)).cuda()
        dist.all_reduce(token_ids, op=dist.ReduceOp.MAX)
        base_token_ids = copy.deepcopy(token_ids)
        outs = mixtral_model(token_ids)

        hidden_states = outs.last_hidden_state
        hidden_states.to_local().sum().backward()

        mixtral_model.finish_grad_sync()

        self.gen_golden(base_mixtral_model, base_token_ids)
        self.compare_model_weights_and_grads(base_mixtral_model, mixtral_model)

    @skip_unless_torch_gpu
    @with_comms
    def test_tp_sp_deferred(self):
        device_mesh = DeviceMesh("cuda", range(self.world_size))
        mixtral_model = deferred_init(MixtralModel, mixtral_config)

        mixtral_model = parallelize_module(
            mixtral_model,
            device_mesh,
            mixtral_plan,
            factory=True,
        )

        token_ids = torch.randint(0, vocab_size, (bsz, seqlen)).cuda()
        dist.all_reduce(token_ids, op=dist.ReduceOp.MAX)
        outs = mixtral_model(token_ids)

        hidden_states = outs.last_hidden_state
        hidden_states.to_local().sum().backward()

        mixtral_model.finish_grad_sync()


class Mixtral4DTest(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    def gen_golden(
        self,
        mixtral_model,
        token_ids_batch_1_epoch_1,
        token_ids_batch_2_epoch_1,
        token_ids_batch_1_epoch_2,
        token_ids_batch_2_epoch_2,
    ):
        optim = torch.optim.Adam(mixtral_model.parameters(), lr=0.01)

        # epoch 1
        optim.zero_grad()
        outs = mixtral_model(token_ids_batch_1_epoch_1)
        outs.last_hidden_state.sum().backward()
        outs = mixtral_model(token_ids_batch_2_epoch_1)
        outs.last_hidden_state.sum().backward()

        # manually reduce mean the grad
        for param in mixtral_model.parameters():
            if param.grad is not None:
                param.grad /= 2
        optim.step()

        # epoch 2
        optim.zero_grad()
        outs = mixtral_model(token_ids_batch_1_epoch_2)
        outs.last_hidden_state.sum().backward()
        outs = mixtral_model(token_ids_batch_2_epoch_2)
        outs.last_hidden_state.sum().backward()

        # manually reduce mean the grad
        for param in mixtral_model.parameters():
            if param.grad is not None:
                param.grad /= 2
        optim.step()

    def compare_model_weights(self, base_model, model):
        for name, base_param in base_model.named_parameters():
            param = model.get_parameter(name)
            if base_param.grad is None:
                continue
            if isinstance(param, DTensor):
                param = param.redistribute(param.device_mesh, [Replicate()], async_op=False)._local_tensor
            torch.testing.assert_close(param, base_param, atol=2e-4, rtol=2e-4)

    @skip_unless_torch_gpu
    @with_comms
    def test_tp_sp_ddp(self):
        device_mesh = DeviceMesh("cuda", [[0, 1], [2, 3]], mesh_dim_names=("DP", "TP"))

        mixtral_model = MixtralModel(mixtral_config).cuda()
        base_mixtral_model = copy.deepcopy(mixtral_model)

        mixtral_model = parallelize_module(
            mixtral_model,
            device_mesh["TP"],
            mixtral_plan,
            factory=True,
        )

        ddp_mixtral_model = DDP(
            mixtral_model,
            device_mesh["DP"],
            accumulate_allreduce_grads_in_fp32=True,
            overlap_grad_reduce=True,
            use_distributed_optimizer=False,
        )

        optim = BasicOptimizer(torch.optim.Adam(mixtral_model.parameters(), lr=0.01), models=[ddp_mixtral_model])

        token_ids_batch_1_epoch_1 = torch.randint(0, vocab_size, (bsz, seqlen)).cuda()
        token_ids_batch_2_epoch_1 = torch.randint(0, vocab_size, (bsz, seqlen)).cuda()
        token_ids_batch_1_epoch_2 = torch.randint(0, vocab_size, (bsz, seqlen)).cuda()
        token_ids_batch_2_epoch_2 = torch.randint(0, vocab_size, (bsz, seqlen)).cuda()
        dist.all_reduce(token_ids_batch_1_epoch_1, op=dist.ReduceOp.MAX)
        dist.all_reduce(token_ids_batch_2_epoch_1, op=dist.ReduceOp.MAX)
        dist.all_reduce(token_ids_batch_1_epoch_2, op=dist.ReduceOp.MAX)
        dist.all_reduce(token_ids_batch_2_epoch_2, op=dist.ReduceOp.MAX)
        base_token_ids_batch_1_epoch_1 = copy.deepcopy(token_ids_batch_1_epoch_1)
        base_token_ids_batch_2_epoch_1 = copy.deepcopy(token_ids_batch_2_epoch_1)
        base_token_ids_batch_1_epoch_2 = copy.deepcopy(token_ids_batch_1_epoch_2)
        base_token_ids_batch_2_epoch_2 = copy.deepcopy(token_ids_batch_2_epoch_2)

        # epoch 1
        optim.zero_grad()
        if self.rank in [0, 1]:
            x = token_ids_batch_1_epoch_1
        else:
            x = token_ids_batch_2_epoch_1
        ddp_mixtral_model(x).last_hidden_state.to_local().sum().backward()
        ddp_mixtral_model.finish_grad_sync()
        optim.step()

        # epoch 2
        optim.zero_grad()
        if self.rank in [0, 1]:
            x = token_ids_batch_1_epoch_2
        else:
            x = token_ids_batch_2_epoch_2
        ddp_mixtral_model(x).last_hidden_state.to_local().sum().backward()
        ddp_mixtral_model.finish_grad_sync()
        optim.step()

        self.gen_golden(
            base_mixtral_model,
            base_token_ids_batch_1_epoch_1,
            base_token_ids_batch_2_epoch_1,
            base_token_ids_batch_1_epoch_2,
            base_token_ids_batch_2_epoch_2,
        )
        self.compare_model_weights(base_mixtral_model, mixtral_model)

    @skip_unless_torch_gpu
    @with_comms
    def test_tp_sp_ddp_doptim(self):
        device_mesh = DeviceMesh("cuda", [[0, 1], [2, 3]], mesh_dim_names=("DP", "TP"))

        mixtral_model = MixtralModel(mixtral_config).cuda()
        base_mixtral_model = copy.deepcopy(mixtral_model)

        mixtral_model = parallelize_module(
            mixtral_model,
            device_mesh["TP"],
            mixtral_plan,
            factory=True,
        )

        ddp_mixtral_model = DDP(
            mixtral_model,
            device_mesh["DP"],
            accumulate_allreduce_grads_in_fp32=True,
            overlap_grad_reduce=True,
            use_distributed_optimizer=True,
        )

        doptim = DistributedOptimizer(
            torch.optim.Adam(mixtral_model.parameters(), lr=0.01),
            models=[ddp_mixtral_model],
            overlap_param_gather=False,
        )

        token_ids_batch_1_epoch_1 = torch.randint(0, vocab_size, (bsz, seqlen)).cuda()
        token_ids_batch_2_epoch_1 = torch.randint(0, vocab_size, (bsz, seqlen)).cuda()
        token_ids_batch_1_epoch_2 = torch.randint(0, vocab_size, (bsz, seqlen)).cuda()
        token_ids_batch_2_epoch_2 = torch.randint(0, vocab_size, (bsz, seqlen)).cuda()
        dist.all_reduce(token_ids_batch_1_epoch_1, op=dist.ReduceOp.MAX)
        dist.all_reduce(token_ids_batch_2_epoch_1, op=dist.ReduceOp.MAX)
        dist.all_reduce(token_ids_batch_1_epoch_2, op=dist.ReduceOp.MAX)
        dist.all_reduce(token_ids_batch_2_epoch_2, op=dist.ReduceOp.MAX)
        base_token_ids_batch_1_epoch_1 = copy.deepcopy(token_ids_batch_1_epoch_1)
        base_token_ids_batch_2_epoch_1 = copy.deepcopy(token_ids_batch_2_epoch_1)
        base_token_ids_batch_1_epoch_2 = copy.deepcopy(token_ids_batch_1_epoch_2)
        base_token_ids_batch_2_epoch_2 = copy.deepcopy(token_ids_batch_2_epoch_2)

        # epoch 1
        doptim.zero_grad()
        if self.rank in [0, 1]:
            x = token_ids_batch_1_epoch_1
        else:
            x = token_ids_batch_2_epoch_1
        ddp_mixtral_model(x).last_hidden_state.to_local().sum().backward()
        ddp_mixtral_model.finish_grad_sync()
        doptim.step()

        # epoch 2
        doptim.zero_grad()
        if self.rank in [0, 1]:
            x = token_ids_batch_1_epoch_2
        else:
            x = token_ids_batch_2_epoch_2
        ddp_mixtral_model(x).last_hidden_state.to_local().sum().backward()
        ddp_mixtral_model.finish_grad_sync()
        doptim.step()

        self.gen_golden(
            base_mixtral_model,
            base_token_ids_batch_1_epoch_1,
            base_token_ids_batch_2_epoch_1,
            base_token_ids_batch_1_epoch_2,
            base_token_ids_batch_2_epoch_2,
        )
        self.compare_model_weights(base_mixtral_model, mixtral_model)


if __name__ == "__main__":
    run_tests()
