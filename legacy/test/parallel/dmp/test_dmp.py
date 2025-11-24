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

from typing import Dict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


from copy import deepcopy
from vescale import (
    DeviceMesh,
    set_plan_overriding_policy,
    auto_parallelize_module,
    distribute_tensor,
)
from vescale.dtensor.placement_types import Shard, Replicate
from vescale.dmodule.api import parallelize_module


class CustomMlp(nn.Module):  # for stress test
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.dropout = nn.Dropout()
        self.fc3 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x, x2=None):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# class CustomMlpMIMO(CustomMlp): # TODO: to support
#     def forward(self, in1, in2=None, has_out2=False):
#         if in2 is None:
#             out1 = super().forward(in1)
#         else:
#             out1 = super().forward(in1) + in2
#         if has_out2 is False:
#             return out1
#         else:
#             return out1, 1.0


class CustomAttention(nn.Module):  # nanoGPT + separate Q,K,V
    def __init__(self, n_embd=32, n_head=8, dropout=0.1, block_size=8, bias=True):
        super().__init__()
        assert n_embd % n_head == 0
        # + + + key, query, value projections in separation below + + +
        # self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.q_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.k_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=bias)
        # + + + key, query, value projections in separation above + + +
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
            )

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # + + + calculate query, key, values in separation below + + +
        # q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # + + + calculate query, key, values in separation above + + +
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class CustomLayerNorm(nn.Module):
    def __init__(self, ndim, bias=True):  # ndim = (HD)
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):  # (Batch, Seq, HD) -> (Batch, Seq, normalized HD)
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CustomBlock(nn.Module):
    def __init__(self, B, S, HD):
        super().__init__()
        self.ln_1 = CustomLayerNorm(HD, bias=True)
        self.attn = CustomAttention(HD)
        self.ln_2 = CustomLayerNorm(HD, bias=True)
        self.mlp = CustomMlp(HD, 4 * HD, HD)

    def forward(self, x):  # (B, S, HD) --> (B, S, HD)
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TestCustomModel(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @skip_unless_torch_gpu
    @with_comms
    def test_mlp(self):
        class RootDModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = CustomMlp(8, 32, 8)

            def forward(self, x):
                return self.m(x)

        torch.cuda.manual_seed(self.rank)
        with DeviceMesh(self.device_type, torch.arange(self.world_size)):
            model = RootDModule()
            dmodel = auto_parallelize_module(model, policy="MEGATRON")
            data = distribute_tensor(torch.empty(8, 8, 8), placements=[Shard(1)])
        loss = dmodel(data).to_local().sum()
        loss.backward()

    @skip_unless_torch_gpu
    @with_comms
    def test_attention(self):
        class RootDModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = CustomAttention(32)

            def forward(self, x):
                return self.m(x)

        torch.cuda.manual_seed(self.rank)
        with DeviceMesh(self.device_type, torch.arange(self.world_size)):
            model = RootDModule()
            dmodel = auto_parallelize_module(model, policy="MEGATRON")

            #     param_sharding_plan={
            #         "m.q_proj.weight": [Shard(0)],
            #         "m.q_proj.bias": [Shard(0)],
            #         "m.k_proj.weight": [Shard(0)],
            #         "m.k_proj.bias": [Shard(0)],
            #         "m.v_proj.weight": [Shard(0)],
            #         "m.v_proj.bias": [Shard(0)],
            #         "m.c_proj.weight": [Shard(1)],
            #         "m.c_proj.bias": [Replicate()],
            #     },
            #     fwd_resharding_plan={
            #         "m.input": [[Replicate()]],
            #         "m.c_proj.output": [[Replicate()]],
            #     },

            data = distribute_tensor(torch.empty(8, 8, 32), placements=[Shard(1)])

        loss = dmodel(data).to_local().sum()
        loss.backward()

    @skip_unless_torch_gpu
    @with_comms
    def test_layernorm(self):
        class RootDModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = CustomLayerNorm(32)

            def forward(self, x):
                return self.m(x)

        torch.cuda.manual_seed(self.rank)
        with DeviceMesh(self.device_type, torch.arange(self.world_size)):
            model = RootDModule()
            dmodel = auto_parallelize_module(model, policy="MEGATRON")
            data = distribute_tensor(torch.empty(8, 16, 32), placements=[Shard(1)])
        loss = dmodel(data).to_local().sum()
        loss.backward()

    @skip_unless_torch_gpu
    @with_comms
    def test_embedding(self):
        num_embeddings, embedding_dim = 32, 8
        input_shape = (6, 8, 6)

        class RootDModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = nn.Embedding(num_embeddings, embedding_dim)

            def forward(self, x):
                return self.m(x)

        torch.cuda.manual_seed(self.rank)
        with DeviceMesh(self.device_type, torch.arange(self.world_size)):
            model = RootDModule()
            dmodel = auto_parallelize_module(model, policy="MEGATRON")
            data = distribute_tensor(torch.randint(0, num_embeddings, input_shape), placements=[Shard(1)])
        loss = dmodel(data).to_local().sum()
        loss.backward()

    @skip_unless_torch_gpu
    @with_comms
    def test_lm_linear(self):
        class RootDModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.lm_head = nn.Linear(32, 8, bias=False)

            def forward(self, x):
                return self.lm_head(x)

        torch.cuda.manual_seed(self.rank)
        with DeviceMesh(self.device_type, torch.arange(self.world_size)):
            model = RootDModule()
            dmodel = auto_parallelize_module(model, policy="MEGATRON")
            data = distribute_tensor(torch.empty(8, 16, 32), placements=[Replicate()])
        loss = dmodel(data).to_local().sum()
        loss.backward()

        num_embeddings, embedding_dim = 32, 8
        input_shape = (6, 8, 6)

        class RootDModule2(nn.Module):
            def __init__(self):
                super().__init__()
                self.wte = nn.Embedding(num_embeddings, embedding_dim)
                self.lm_head = nn.Linear(embedding_dim, num_embeddings, bias=False)
                self.wte.weight = self.lm_head.weight  # tie weight # TODO: support auto-tie DTensorized weight

            def forward(self, x):
                x = self.wte(x)
                return self.lm_head(x)

        torch.cuda.manual_seed(self.rank)
        with DeviceMesh(self.device_type, torch.arange(self.world_size)):
            model = RootDModule2()
            dmodel = auto_parallelize_module(model, policy="MEGATRON")
            data = distribute_tensor(torch.randint(0, num_embeddings, input_shape), placements=[Shard(1)])
        loss = dmodel(data).to_local().sum()
        loss.backward()

    @skip_unless_torch_gpu
    @with_comms
    def test_block(self):
        B, S, HD = 4, 8, 32

        class RootDModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.m = CustomBlock(B, S, HD)

            def forward(self, x):
                return self.m(x)

        torch.cuda.manual_seed(self.rank)
        with DeviceMesh(self.device_type, torch.arange(self.world_size)):
            model = RootDModule()
            dmodel = auto_parallelize_module(model, policy="MEGATRON")
            data = distribute_tensor(torch.empty(B, S, HD), placements=[Replicate()])
        loss = dmodel(data).to_local().sum()
        loss.backward()


class TestDebug(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @skip_unless_torch_gpu
    @with_comms
    def test_debug(self):
        class SimpleMlp(nn.Module):
            def __init__(self, size=4):
                super().__init__()
                self.fc1 = nn.Linear(size, size)
                self.gelu = nn.GELU()
                self.fc2 = nn.Linear(size, size)

        model = SimpleMlp(4)
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
        model, device_mesh, sharding_plan = auto_parallelize_module(model, mesh, policy="MEGATRON", plan_only=True)

        if self.rank == 0:
            print(f"\n root_param_plan: {sharding_plan['parameter']}")
            print(f"\n root_fwd_plan: {sharding_plan['forward']}")

        parallelize_module(model, device_mesh, sharding_plan)

        if self.rank == 0:
            device_mesh, used_param_plan, used_fwd_plan = model.dump_mesh_plans()


class TestSetPlan(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    def test_set_plan_same_level(self):
        class BaselineMLP(nn.Module):
            def __init__(self, size=4):
                super().__init__()
                self.fc1 = nn.Linear(size, size)
                self.fc2 = nn.Linear(size, size)

        _, _, base_sharding_plan = auto_parallelize_module(BaselineMLP(), policy="MEGATRON", plan_only=True)

        class SetNoneNoneMLP(nn.Module):
            def __init__(self, size=4):
                super().__init__()
                self.fc1 = nn.Linear(size, size)
                self.fc2 = nn.Linear(size, size)
                set_plan_overriding_policy(self)
                # --> same as baseline

        class SetParamMLP(nn.Module):
            def __init__(self, size=4):
                super().__init__()
                self.fc1 = nn.Linear(size, size)
                self.fc2 = nn.Linear(size, size)
                set_plan_overriding_policy(self, param_sharding_plan={"fc2.weight": "FAKED"})
                # --> param_sharding_plan becomes this, fwd plan as baseline

        class SetFwdMLP(nn.Module):
            def __init__(self, size=4):
                super().__init__()
                self.fc1 = nn.Linear(size, size)
                self.fc2 = nn.Linear(size, size)
                set_plan_overriding_policy(self, fwd_resharding_plan={"output": "FAKED"})
                # --> param_sharding_plan as baseline, fwd plan becomes this

        class SetBothMLP(nn.Module):
            def __init__(self, size=4):
                super().__init__()
                self.fc1 = nn.Linear(size, size)
                self.fc2 = nn.Linear(size, size)
                set_plan_overriding_policy(self, {"fc2.weight": "FAKED"}, {"output": "FAKED"})
                # --> both become these

        _, _, dut_sharding_plan = auto_parallelize_module(SetNoneNoneMLP(), policy="MEGATRON", plan_only=True)
        self.assertTrue(dut_sharding_plan["parameter"] == base_sharding_plan["parameter"])
        self.assertTrue(dut_sharding_plan["forward"] == base_sharding_plan["forward"])

        _, _, dut_sharding_plan = auto_parallelize_module(SetParamMLP(), policy="MEGATRON", plan_only=True)
        self.assertTrue(dut_sharding_plan["parameter"] == {"fc2.weight": "FAKED"})
        self.assertTrue(dut_sharding_plan["forward"] == base_sharding_plan["forward"])

        _, _, dut_sharding_plan = auto_parallelize_module(SetFwdMLP(), policy="MEGATRON", plan_only=True)
        self.assertTrue(dut_sharding_plan["parameter"] == base_sharding_plan["parameter"])
        self.assertTrue(dut_sharding_plan["forward"] == {"output": "FAKED"})

        _, _, dut_sharding_plan = auto_parallelize_module(SetBothMLP(), policy="MEGATRON", plan_only=True)
        self.assertTrue(dut_sharding_plan["parameter"] == {"fc2.weight": "FAKED"})
        self.assertTrue(dut_sharding_plan["forward"] == {"output": "FAKED"})

    def test_set_plan_sub_level(self):
        def update_subplan(root_plan: Dict, sub_prefix: str, sub_plan: Dict) -> Dict:
            root_plan = deepcopy(root_plan)
            keys_to_remove = [fqn for fqn in root_plan if fqn.startswith(sub_prefix)]
            for k in keys_to_remove:
                root_plan.pop(k)

            for name, place in sub_plan.items():
                fqn = sub_prefix + "." + name
                root_plan[fqn] = place
            return root_plan

        class BaselineMLP(nn.Module):
            def __init__(self, size=4):
                super().__init__()
                self.fc1 = nn.Linear(size, size)
                self.fc2 = nn.Linear(size, size)

        _, _, base_sharding_plan = auto_parallelize_module(BaselineMLP(), policy="MEGATRON", plan_only=True)

        class SetSubNoneNoneMLP(nn.Module):
            def __init__(self, size=4):
                super().__init__()
                self.fc1 = nn.Linear(size, size)
                self.fc2 = nn.Linear(size, size)
                set_plan_overriding_policy(self.fc2)
                # --> same as MLP

        class SetSubParamMLP(nn.Module):
            def __init__(self, size=4):
                super().__init__()
                self.fc1 = nn.Linear(size, size)
                self.fc2 = nn.Linear(size, size)
                set_plan_overriding_policy(self.fc2, param_sharding_plan={"weight": "FAKED"})
                # --> all as MLP, except this sub param plan

        class SetSubFwdMLP(nn.Module):
            def __init__(self, size=4):
                super().__init__()
                self.fc1 = nn.Linear(size, size)
                self.fc2 = nn.Linear(size, size)
                set_plan_overriding_policy(self.fc2, fwd_resharding_plan={"output": "FAKED"})
                # --> all as MLP, except this sub fwd plan

        class SetSubBothMLP(nn.Module):
            def __init__(self, size=4):
                super().__init__()
                self.fc1 = nn.Linear(size, size)
                self.fc2 = nn.Linear(size, size)
                set_plan_overriding_policy(self.fc2, {"weight": "FAKED"}, {"output": "FAKED"})
                # --> only fc2's two plan become these

        _, _, dut_sharding_plan = auto_parallelize_module(SetSubNoneNoneMLP(), policy="MEGATRON", plan_only=True)
        self.assertTrue(dut_sharding_plan["parameter"] == base_sharding_plan["parameter"])
        self.assertTrue(dut_sharding_plan["forward"] == base_sharding_plan["forward"])

        _, _, dut_sharding_plan = auto_parallelize_module(SetSubParamMLP(), policy="MEGATRON", plan_only=True)
        gold_param_plan = update_subplan(base_sharding_plan["parameter"], "fc2", {"weight": "FAKED"})
        self.assertTrue(dut_sharding_plan["parameter"] == gold_param_plan)
        self.assertTrue(dut_sharding_plan["forward"] == base_sharding_plan["forward"])

        _, _, dut_sharding_plan = auto_parallelize_module(SetSubFwdMLP(), policy="MEGATRON", plan_only=True)
        gold_fwd_plan = update_subplan(base_sharding_plan["forward"], "fc2", {"output": "FAKED"})
        self.assertTrue(dut_sharding_plan["parameter"] == base_sharding_plan["parameter"])
        self.assertTrue(dut_sharding_plan["forward"] == gold_fwd_plan)

        _, _, dut_sharding_plan = auto_parallelize_module(SetSubBothMLP(), policy="MEGATRON", plan_only=True)
        self.assertTrue(dut_sharding_plan["parameter"] == gold_param_plan)
        self.assertTrue(dut_sharding_plan["forward"] == gold_fwd_plan)

    def test_set_plan_upper_level(self):
        class MLP(nn.Module):
            def __init__(self, size=4):
                super().__init__()
                self.fc1 = nn.Linear(size, size)
                self.fc2 = nn.Linear(size, size)

        class SetUpperBaseline(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = MLP()

        _, _, base_sharding_plan = auto_parallelize_module(SetUpperBaseline(), policy="MEGATRON", plan_only=True)

        class SetUpperNoneNone(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = MLP()
                set_plan_overriding_policy(self)
                # --> same as baseline

        class SetUpperParam(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = MLP()
                set_plan_overriding_policy(self, param_sharding_plan={"mlp.fc1.weight": "FAKED"})
                # --> root param plan becomes this, root fwd as baseline

        class SetUpperFwd(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = MLP()
                set_plan_overriding_policy(self, fwd_resharding_plan={"mlp.output": "FAKED"})
                # --> root fwd becomes this, root param plan as baseline

        class SetUpperBoth(nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = MLP()
                set_plan_overriding_policy(self, {"mlp.fc1.weight": "FAKED"}, {"mlp.output": "FAKED"})
                # --> both root becomes these

        _, _, dut_sharding_plan = auto_parallelize_module(SetUpperNoneNone(), policy="MEGATRON", plan_only=True)
        self.assertTrue(dut_sharding_plan["parameter"] == base_sharding_plan["parameter"])
        self.assertTrue(dut_sharding_plan["forward"] == base_sharding_plan["forward"])

        _, _, dut_sharding_plan = auto_parallelize_module(SetUpperParam(), policy="MEGATRON", plan_only=True)
        self.assertTrue(dut_sharding_plan["parameter"] == {"mlp.fc1.weight": "FAKED"})
        self.assertTrue(dut_sharding_plan["forward"] == base_sharding_plan["forward"])

        _, _, dut_sharding_plan = auto_parallelize_module(SetUpperFwd(), policy="MEGATRON", plan_only=True)
        self.assertTrue(dut_sharding_plan["parameter"] == base_sharding_plan["parameter"])
        self.assertTrue(dut_sharding_plan["forward"] == {"mlp.output": "FAKED"})

        _, _, dut_sharding_plan = auto_parallelize_module(SetUpperBoth(), policy="MEGATRON", plan_only=True)
        self.assertTrue(dut_sharding_plan["parameter"] == {"mlp.fc1.weight": "FAKED"})
        self.assertTrue(dut_sharding_plan["forward"] == {"mlp.output": "FAKED"})


if __name__ == "__main__":
    run_tests()
