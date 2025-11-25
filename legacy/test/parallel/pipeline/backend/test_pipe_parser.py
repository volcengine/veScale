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

from torch.testing._internal.common_utils import run_tests
from common_dtensor import DTensorTestBase, with_comms
from vescale.pipe.pipe_parser import PipeParser
from vescale.initialize.deferred_init import deferred_init
from vescale.plan import PipelineParallelPlan, PipelineScheduleType, ModeType, PipelineSplitMethodType
from eight_mlp import EightMLP, EightMLPWithOps, EightMLPWithOpsTail


class TestPipeParser(DTensorTestBase):
    @with_comms
    def test_parse_naive_model(self):
        """
        Tests trace capture with torch.fx symbolic tracer under user-defined granularity.
        """
        deferred_mlp = deferred_init(EightMLP, hidden=8)
        partition_units = ["mlp4", "mlp8"]
        pipe_parser = PipeParser()
        model_graph = pipe_parser.parse(deferred_mlp)
        print(model_graph)
        assert not all(node.target in partition_units for node in model_graph.graph.nodes)

        pipe_config = PipelineParallelPlan(
            mode=ModeType.GRAPH_EAGER,
            smallest_unsplittable_units=partition_units,
        )
        model_graph_partition_units = pipe_parser.parse(deferred_mlp, pipe_config)
        print(model_graph_partition_units)
        assert any(node.target in partition_units for node in model_graph_partition_units.graph.nodes)

    @with_comms
    def test_parse_huggingface_model(self):
        """
        Tests trace capture with huggingface symbolic tracer under user-defined granularity.
        """
        from transformers import LlamaModel, LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm

        configuration = LlamaConfig()
        configuration.hidden_size = 256
        configuration.intermediate_size = 1376
        configuration.num_attention_heads = 1
        configuration.num_hidden_layers = 2
        model = LlamaModel(configuration)

        # below two lists of partition units refer to the same submodules we never wish to partition
        partition_units = ["layers.0", "layers.1", "norm"]
        partition_units_equivalent = [LlamaDecoderLayer, LlamaRMSNorm]
        pipe_config = PipelineParallelPlan(smallest_unsplittable_units=partition_units)
        pipe_config_equivalent = PipelineParallelPlan(smallest_unsplittable_units=partition_units_equivalent)

        pipe_parser = PipeParser()
        model_graph = pipe_parser.parse(model)
        print(model_graph)
        assert not all(node.target in partition_units for node in model_graph.graph.nodes)

        model_graph_partition_units = pipe_parser.parse(model, pipe_config)
        print(model_graph_partition_units)
        result = [node.target in partition_units for node in model_graph_partition_units.graph.nodes]
        assert any(result)

        # the resulting graph should be identical to the one parsed by model_graph_partition_units
        model_graph_partition_units_equivalent = pipe_parser.parse(model, pipe_config_equivalent)
        print(model_graph_partition_units_equivalent)
        result_two = [node.target in partition_units for node in model_graph_partition_units_equivalent.graph.nodes]
        assert any(result_two)
        self.assertEqual(result, result_two)

    @with_comms
    def test_uniform_split(self):
        """
        Tests uniform stage split.
        """
        deferred_mlp = deferred_init(EightMLP, hidden=8)
        layers = 8
        pipe_parser = PipeParser()
        pipe_config = PipelineParallelPlan(
            mode=ModeType.GRAPH_EAGER,
            split_method=PipelineSplitMethodType.UNIFORM,
            num_stages=2,
            virtual_chunks=1,
            smallest_unsplittable_units=[f"mlp{i + 1}" for i in range(layers)],
            batch_p2p_comm=False,
            overlap_p2p_comm=True,
            schedule_type=PipelineScheduleType.SIMPLE_1F1B,
        )
        model_graph_partition_units = pipe_parser.parse(deferred_mlp, pipe_config)
        print(model_graph_partition_units)
        splited_graph = pipe_parser.partition_stage(deferred_mlp, model_graph_partition_units, pipe_config)
        self.assertEqual(
            [node.name for node in splited_graph.stage0.graph.nodes][1:-1], ["mlp1", "mlp2", "mlp3", "mlp4"]
        )
        self.assertEqual(
            [node.name for node in splited_graph.stage1.graph.nodes][1:-1], ["mlp5", "mlp6", "mlp7", "mlp8"]
        )

    @with_comms
    def test_uniform_split_model_with_ops(self):
        """
        Tests uniform stage split with torch operators as graph components.
        """
        deferred_mlp = deferred_init(EightMLPWithOpsTail, hidden=8)
        layers = 8
        pipe_config = PipelineParallelPlan(
            mode=ModeType.GRAPH_EAGER,
            split_method=PipelineSplitMethodType.UNIFORM,
            num_stages=2,
            virtual_chunks=1,
            smallest_unsplittable_units=[f"mlp{i + 1}" for i in range(layers)],
            batch_p2p_comm=False,
            overlap_p2p_comm=True,
            schedule_type=PipelineScheduleType.SIMPLE_1F1B,
        )
        pipe_parser = PipeParser()
        model_graph_partition_units = pipe_parser.parse(deferred_mlp, pipe_config)
        print(model_graph_partition_units)
        splited_graph = pipe_parser.partition_stage(deferred_mlp, model_graph_partition_units, pipe_config)
        self.assertEqual(
            [node.name for node in splited_graph.stage0.graph.nodes][1:-1],
            ["add", "mlp1", "mul", "mlp2", "mul_1", "mlp3", "mlp4"],
        )
        self.assertEqual(
            [node.name for node in splited_graph.stage1.graph.nodes][1:-1],
            ["mlp5", "mlp6", "mlp7", "mlp8", "mul_2", "mul_3", "add_1"],
        )

    @with_comms
    def test_uniform_split_on_modules(self):
        """
        Tests uniform stage split on modules with modules and torch operators.
        """
        deferred_mlp = deferred_init(EightMLPWithOps, hidden=8)
        layers = 8
        pipe_parser = PipeParser()
        pipe_config = PipelineParallelPlan(
            mode=ModeType.GRAPH_EAGER,
            split_method=PipelineSplitMethodType.UNIFORM,
            num_stages=2,
            virtual_chunks=1,
            smallest_unsplittable_units=[f"mlp{i + 1}" for i in range(layers)],
            batch_p2p_comm=False,
            overlap_p2p_comm=True,
            schedule_type=PipelineScheduleType.SIMPLE_1F1B,
            uniform_split_ops=True,
        )
        model_graph_partition_units = pipe_parser.parse(deferred_mlp, pipe_config)
        print(model_graph_partition_units)
        splited_graph = pipe_parser.partition_stage(deferred_mlp, model_graph_partition_units, pipe_config)
        stage_one_modules = ["add", "mlp1", "mul", "mlp2", "mul_1", "mul_2", "mul_3", "mlp3", "mlp4"]
        stage_two_modules = ["mlp5", "mlp6", "mlp7", "mlp8"]
        self.assertEqual([node.name for node in splited_graph.stage0.graph.nodes][1:-1], stage_one_modules)
        self.assertEqual([node.name for node in splited_graph.stage1.graph.nodes][1:-1], stage_two_modules)


if __name__ == "__main__":
    run_tests()
