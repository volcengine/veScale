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


import torch.nn as nn
from torch.testing._internal.common_utils import run_tests
from vescale.pipe.tracer import ModelTracer, HFModelTracer, register_partition_module, hf_symbolic_trace
from common_dtensor import DTensorTestBase, with_comms
from transformers import LlamaModel, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm


class MLP(nn.Module):
    def __init__(self, features_in, features_out, value):
        super().__init__()
        self.value = value
        self.fc1 = nn.Linear(features_in, 2 * features_in, bias=False)
        self.fc1.weight.data.fill_(value)
        self.fc2 = nn.Linear(2 * features_in, features_out, bias=False)
        self.fc2.weight.data.fill_(value * 2)
        self.gelu = nn.GELU()

    def forward(self, x):
        t = self.fc1(x)
        t = self.gelu(t)
        t = self.fc2(t)
        return t


class EightMLP(nn.Module):
    def __init__(self, hidden=1024):
        super().__init__()
        self.mlp1 = MLP(hidden, hidden, 0)
        self.mlp2 = MLP(hidden, hidden, 1)
        self.mlp3 = MLP(hidden, hidden, 2)
        self.mlp4 = MLP(hidden, hidden, 3)
        self.mlp5 = MLP(hidden, hidden, 4)
        self.mlp6 = MLP(hidden, hidden, 5)
        self.mlp7 = MLP(hidden, hidden, 6)
        self.mlp8 = MLP(hidden, hidden, 7)
        self.sequence = nn.Sequential(
            self.mlp1,
            self.mlp2,
            self.mlp3,
            self.mlp4,
            self.mlp5,
            self.mlp6,
            self.mlp7,
            self.mlp8,
        )

    def forward(self, x):
        return self.sequence(x)


class TracerTest(DTensorTestBase):
    @property
    def world_size(self):
        return 1

    @with_comms
    def test_simple_model_tracer(self):
        """
        Test fx tracer to capture native symbolic trace
        of simple model.
        """
        model = EightMLP(16)
        tracer = ModelTracer()
        traced_graph = tracer.trace(model)
        print("Simple Model Graph Trace:")
        print(traced_graph)

    @with_comms
    def test_simple_model_tracer_with_partition_units(self):
        """
        Test fx tracer to capture symbolic trace with granularity of
        MLP level (do not dive into operators of MLP) of simple model.
        """
        model = EightMLP(16)
        register_partition_module(model.mlp1)
        register_partition_module(model.mlp2)
        register_partition_module(model.mlp3)
        register_partition_module(model.mlp4)
        register_partition_module(model.mlp5)
        register_partition_module(model.mlp6)
        register_partition_module(model.mlp7)
        register_partition_module(model.mlp8)
        tracer = ModelTracer()
        traced_graph = tracer.trace(model)
        print(traced_graph)

    @with_comms
    def test_huggingface_model_tracer_with_partition_units(self):
        """
        Test huggingface tracer to capture symbolic trace with granularity
        of LlamaDecoderLayer and LlamaRMSNorm.
        """
        configuration = LlamaConfig()
        configuration.hidden_size = 1024
        configuration.intermediate_size = 5504
        configuration.num_attention_heads = 1
        configuration.num_hidden_layers = 2

        model = LlamaModel(configuration)
        submodule_qualified_names = ["layers.0", "layers.1", "norm"]
        # submodules indicated by submodule_qualified_names are modules that have the classes below
        partition_unit_modules = [LlamaDecoderLayer, LlamaRMSNorm] + submodule_qualified_names
        traced_graph = hf_symbolic_trace(
            model,
            input_names=["input_ids", "attention_mask"],
            tracer_cls=HFModelTracer,
            partition_modules=partition_unit_modules,
        )
        print("HF Model Graph Trace:")
        print(traced_graph)


if __name__ == "__main__":
    run_tests()
