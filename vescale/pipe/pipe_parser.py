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


from typing import Sequence, Dict, List, Union, Any, Optional
import torch
import re
import torch.nn as nn
import logging
from inspect import signature
from vescale.pipe.tracer import ModelTracer, HFModelTracer, hf_symbolic_trace
from torch.fx.passes.split_utils import split_by_tags
from vescale.plan.pipeline_parallel import PipelineParallelPlan
from vescale.plan.spec import PipelineSplitMethodType, TracerType

NUM_DEFAULT_ARGS = 3

try:
    # New import path
    from torch.export._trace import _export_to_torch_ir  # noqa: PGH004
except ImportError:
    try:
        # Old import path
        from torch._export import _export_to_torch_ir  # noqa: F401
    except ImportError:
        print("Could not import _export_to_torch_ir. Please make sure your PyTorch " "version is newer than 2.2.0.")


logger = logging.Logger(__file__)


class PipeParser:
    def __init__(self):
        self.orig_to_split_fqn_mapping = {}

    def parse(
        self, module: nn.Module, plan: Optional[PipelineParallelPlan] = None, **kwargs: Any
    ) -> torch.fx.GraphModule:
        """
        Applies cascade trace capture using upstream torch.fx symbolic tracer, huggingface
        tracer and dynamo export tracer respectively. To trigger cascade parser, select
        TracerType.AUTO in PipelineParallelPlan's tracer_type field

        Args:
            module (nn.Module): the model from which we trace its forward execution graph.

        Returns:
            Model trace graph.

        """
        parser_args = {}
        if plan and plan.smallest_unsplittable_units:
            parser_args["partition_units"] = plan.smallest_unsplittable_units
        if kwargs:
            parser_args.update(kwargs)
        try:
            msg = "Applying Default torch.fx symbolic tracing..."
            logger.info(msg)
            traced = self.parse_torch_fx(module, **parser_args)
        except Exception as e:
            try:
                msg = f"Default torch.fx symbolic tracing failed: {e}\nApplying HuggingFace Tracer..."
                logger.warning(msg)
                traced = self.parse_huggingface_fx(module, **parser_args)
            except Exception as e2:
                try:
                    msg = f"HuggingFace tracing failed: {e2}\nApplying Dynamo Export Tracer..."
                    logger.warning(msg)
                    traced = self.parse_dynamo_export(module, **parser_args)
                except Exception as e3:
                    msg = f"Dynamo export tracing failed: {e3}"
                    logger.warning(msg)
                    raise e3
        print(f"Below is visualization of the traced model graph:\n{traced}")
        return traced

    def partition_stage(
        self, module: nn.Module, model_graph: torch.fx.GraphModule, plan: PipelineParallelPlan
    ) -> List[str]:
        """
        Partitions models by split criterion. The function first annotates graph nodes and ops by stage
        boundary, and then split stages into model partition modules (torch.fx.GraphModule).

        Args:
            module (nn.Module): the model.
            model_graph (torch.fx.GraphModule): the trace graph of the model.
            plan (PipelineParallelPlan): configuration of pipeline paralellism API.

        Returns:
            The executable trace graph partitioned by stage boundary,
            and mappings of submodules before and after partition.

        """
        split_points = self.split(model_graph, plan)
        plan.split_points = split_points
        splited_graph = self.split_stage(model_graph, module, plan)
        return splited_graph

    def split(self, graph: torch.fx.GraphModule, plan: PipelineParallelPlan):
        """
        Generates or verifies pipeline split points, and writes updates to PipelineParallelPlan.

        Args:
            graph (torch.fx.GraphModule): symbolic trace graph of the entire model
            plan (PipelineParallelPlan): configuration of attributes for pipeline parallleism API

        Returns:
            A list of fully qualified names of stage split points.

        """
        criterion = plan.split_method
        boundaries = plan.split_points
        nodes = list(graph.graph.nodes)
        trimmed_nodes = nodes[1:-1]  # remove input and output nodes in graph
        node_names = [nd.name for nd in nodes]
        trimmed_node_names = []
        for nd in nodes[1:-1]:
            if nd.op == "call_module":
                trimmed_node_names.append(nd.target)
            else:
                trimmed_node_names.append(nd.name)
        num_stages = plan.num_stages
        num_chunk_per_stage = plan.virtual_chunks
        num_model_partitions = num_stages * num_chunk_per_stage
        nodes_size = len(trimmed_nodes)
        trimmed_module_indices = [idx for idx in range(nodes_size) if trimmed_nodes[idx].op == "call_module"]
        modules_only_size = len(trimmed_module_indices)
        assert criterion in [
            PipelineSplitMethodType.UNIFORM,
            PipelineSplitMethodType.MANUAL,
            PipelineSplitMethodType.AUTO,
            PipelineSplitMethodType.PARAMETERS,
            PipelineSplitMethodType.SIMULATOR,
            PipelineSplitMethodType.FLOPS,
        ]
        if criterion == PipelineSplitMethodType.UNIFORM:
            if plan.uniform_split_ops:
                module_indices = self._partition_uniform(modules_only_size, num_model_partitions)
                indices = [trimmed_module_indices[module_indices[idx]] for idx in range(len(module_indices))]
            else:
                indices = self._partition_uniform(nodes_size, num_model_partitions)
            final_boundaries = []
            for idx in indices:
                if nodes[idx].op == "call_module" and trimmed_nodes[idx].name != trimmed_nodes[idx].target:
                    final_boundaries.append(trimmed_nodes[idx].name.replace("_", "."))
                else:
                    final_boundaries.append(trimmed_nodes[idx].name)
            plan.split_points = final_boundaries
        elif criterion == PipelineSplitMethodType.MANUAL:
            assert boundaries, "Must provide stage boundaries for MANUAL mode during stage partition!"
            if boundaries and all(isinstance(x, str) for x in boundaries):
                for fqn in boundaries:
                    assert (
                        fqn in node_names
                        or fqn.replace(".", "_") in node_names
                        or any(name.startswith(fqn) for name in node_names)
                    )
            elif boundaries and all(isinstance(x, int) for x in boundaries):
                # Under indexing-based partition, model graph's execution order is visualized as followed
                boundaries.sort()
                assert 0 <= boundaries[0] <= boundaries[-1] < len(nodes)
                # convert submodule indices into fully qualified names
                new_boundaries = []
                for idx in boundaries:
                    if nodes[idx].op == "call_module":
                        new_boundaries.append(nodes[idx].name.replace("_", "."))
                    else:
                        new_boundaries.append(nodes[idx].name)
                boundaries = new_boundaries
            else:
                raise ValueError("Input must be either a list of path strings or partition indices!")
            if boundaries[-1] != node_names[-2]:
                boundaries.append(node_names[-2])

            final_boundaries = self._handle_virtual_stage_boundaries(
                boundaries,
                trimmed_node_names,
                num_chunk_per_stage,
                plan.enable_vpp_split_points,
            )
            # assert no stage boundary is a prefix of other boundaries
            _boundaries = set(final_boundaries)
            for this_bd in _boundaries:
                for bd in _boundaries:
                    if this_bd != bd:
                        assert not this_bd.startswith(bd)
            assert len(final_boundaries) == num_model_partitions
        else:
            raise NotImplementedError
        return final_boundaries

    def _partition_uniform(self, num_items, num_parts):
        assert num_items % num_parts == 0, "#graph nodes must be partitioned by #stages!"
        assert num_items >= num_parts, "#model partitions must not be less than #graph nodes!"
        parts = [0] * (num_parts + 1)
        # First check for the trivial edge case
        if num_items <= num_parts:
            for p in range(num_parts + 1):
                parts[p] = min(p, num_items)
        else:
            chunksize = num_items // num_parts
            residual = num_items - (chunksize * num_parts)
            parts = torch.arange(0, (num_parts + 1) * chunksize, chunksize)
            for i in range(residual):
                parts[i + 1 :] += 1
            parts = parts.tolist()
        if parts[0] == 0:
            parts = parts[1:]
        parts = [x - 1 for x in parts]
        return parts

    def _handle_virtual_stage_boundaries(
        self,
        boundaries: List[Union[str, int]],
        node_names: List[str],
        num_chunk_per_stage: int,
        use_manual_vpp_boundary: bool,
    ):
        if isinstance(boundaries[0], int):
            boundaries = [node_names[idx] for idx in boundaries]
        if num_chunk_per_stage > 1 and not use_manual_vpp_boundary:
            new_indices = []
            indices = list(range(len(node_names)))
            raw_stage_indices = []
            for fqn in boundaries:
                if fqn not in node_names:
                    fqn = fqn.replace(".", "_")
                raw_stage_indices.append(node_names.index(fqn))
            if raw_stage_indices[-1] < len(node_names) - 1:
                raw_stage_indices[-1].append(len(node_names) - 1)
            for i in range(len(raw_stage_indices)):
                if i == 0:
                    sublist = torch.tensor(indices[: raw_stage_indices[i] + 1])
                else:
                    sublist = torch.tensor(indices[raw_stage_indices[i - 1] + 1 : raw_stage_indices[i] + 1])
                assert (
                    len(sublist) >= num_chunk_per_stage
                ), "#operators and modules in a stage must be no smaller than #virtual pipeline chunks!"
                sublist_list = sublist.tensor_split(num_chunk_per_stage)
                new_indices += [int(sub[-1]) for sub in sublist_list]
            boundaries = [node_names[idx] for idx in new_indices]
        return boundaries

    def annotate_pipeline_stage(
        self, graph: torch.fx.GraphModule, root_module: nn.Module, boundaries: List, partition_units: List
    ):
        """
        Annotates stage split boundaries of each stage on the model graph.

        Args:
            graph (torch.fx.GraphModule): model trace graph
            root_module (nn.Module): raw model
            boundaries (List): a list of pipeline stage split points in the form of fully qualified names
            partition_units (List): smallest unsplittable unit in a model trace graph

        Returns:
            Model graph with stage split points annotated.

        """

        def identify_base_units(submodule, partition_units, submodule_name):
            return (
                len(list(submodule.children())) == 0
                or submodule_name in partition_units
                or type(submodule) in partition_units
            )

        splited_module_names = boundaries
        assert len(splited_module_names) > 0, "need to have bigger than 1 nodes"
        max_dfn_for_modules = [0 for _ in range(len(splited_module_names))]
        node_lists = list(graph.graph.nodes)
        node_lists_names = [node.name for node in node_lists]
        node_lists_target_names = [node.target for node in node_lists]
        submodule_paths = {name for name, _ in root_module.named_modules()}
        for stage_id, submodule_name in enumerate(splited_module_names):
            stage_tag = stage_id
            sub_module_unions = []
            if submodule_name in node_lists_names:
                boundary_node = node_lists[node_lists_names.index(submodule_name)]
            else:
                boundary_node = node_lists[node_lists_target_names.index(submodule_name)]
            if submodule_name in submodule_paths:
                submodule = root_module.get_submodule(submodule_name)
                if identify_base_units(submodule, partition_units, submodule_name):  # for leaf module
                    sub_module_unions.append(submodule_name)
                else:
                    for name, _ in submodule.named_children():
                        sub_module_unions.append(submodule_name + "." + name)
                sub_module_unions = [re.sub(r"\.", "_", name) for name in sub_module_unions]
            else:
                if boundary_node.op == "call_method" or boundary_node.op == "call_function":
                    sub_module_unions.append(boundary_node.name)
                else:
                    raise ValueError(
                        "Stage boundary can only be of ``call_module``, ``call_method`` and ``call_function``!"
                    )
            stage_max_dfn = 0
            # set tag with the node Sequence, to O(N)
            for dfn in range(len(node_lists)):
                node = node_lists[dfn]
                if node.name in sub_module_unions:
                    # TODO: tag should be partition_chunk{id} instead of stage, as it may lead to confusion in interleaved 1F1B schedules
                    node.tag = f"stage{str(stage_tag)}"
                    stage_max_dfn = max(dfn, stage_max_dfn)
            max_dfn_for_modules[stage_id] = stage_max_dfn

        # annotate the first stage
        for dfn in range(len(node_lists)):
            if dfn <= max_dfn_for_modules[0]:
                node_lists[dfn].tag = "stage0"
            else:
                break

        slow = 0
        cur_dfn_num = 0
        fast = max_dfn_for_modules[cur_dfn_num]
        # using fast slow ptr to annotate graph

        while fast < len(node_lists) and slow < len(node_lists):
            while slow <= fast:
                node_lists[slow].tag = node_lists[fast].tag
                slow += 1
            cur_dfn_num += 1
            if cur_dfn_num < len(max_dfn_for_modules):
                fast = max_dfn_for_modules[cur_dfn_num]
            else:
                while slow < len(node_lists):
                    node_lists[slow].tag = node_lists[fast].tag
                    slow += 1
        return graph

    def split_stage(
        self, graph: torch.fx.GraphModule, root_module: nn.Module, plan: PipelineParallelPlan
    ) -> torch.fx.GraphModule:
        """
        Split a model graph into multiple pipeline stage subgraphs.

        Args:
            graph (torch.fx.GraphModule): model graph
            root_module (nn.Module): raw model
            plan (PipelineParallelPlan): configuration of attributes for pipeline parallleism API

        Returns:
            Edited model graph that contains subgraph of each virtual module chunk of a pipeline stage.
            For example,
            ```
            Before:
            original_graph:
                module1: xxx
                module2: xxx
                module3: xxx
                module4: xxx

            After:
            split_graph:
                stage0:
                    module1: xxx
                    module2: xxx
                stage1:
                    module3: xxx
                    module4: xxx
            ```

        """
        if graph is None:
            return None

        boundaries = plan.split_points
        partition_units = plan.smallest_unsplittable_units
        graph = self.annotate_pipeline_stage(graph, root_module, boundaries, partition_units)
        tags = [f"stage{str(num)}" for num in range(len(boundaries))]
        # split by PyTorch upstream's split_by_tags
        split_graph, orig_to_split_fqn_mapping = split_by_tags(graph, tags, return_fqn_mapping=True)
        for i in range(1, len(tags)):
            # input placeholder node of each stage-specific graph
            placeholder_node = list(getattr(split_graph, tags[i]).graph.nodes)[0]
            if placeholder_node.op == "placeholder" and placeholder_node.name != "x":
                placeholder_node.name = "x"

        return split_graph

    def parse_torch_fx(
        self, model: nn.Module, partition_units: List[str] = None, shard_plan: Dict = None
    ) -> torch.fx.GraphModule:
        """
        Applies torch.fx symbolic trace to capture model graph.

        Args:
            model (nn.Module): raw model
            partition_units (List[str]): a list of smallest unsplittable modules such that the parser will
                not flatten their underlying components during parsing
            shard_plan (Dict): dictionary of sharding plan, if users would like to wrap up tensor parallelized
                modules as unsplittable units

        Returns:
            Captured torch.fx.GraphModule

        """
        if partition_units is None:
            partition_units = []
        input_names = list(signature(model.forward).parameters.keys())
        if "input_ids" in input_names and "inputs_embeds" in input_names:
            input_names.remove("inputs_embeds")
        if shard_plan:
            hierarchy_substructure_qualified_names = self._hierarchy_structure_names(model, shard_plan)
            partition_units += hierarchy_substructure_qualified_names
        traced: torch.fx.GraphModule = hf_symbolic_trace(
            model,
            input_names=input_names,
            disable_check=True,
            tracer_cls=ModelTracer,
            partition_modules=partition_units,
        )
        return traced

    def parse_dynamo_export(self, model: nn.Module, *args: Sequence, **kwargs: Dict):
        """
        Applies capture model graph with torch dynamo.export.

        Args:
            model (nn.Module): raw model

        Returns:
            Captured torch.fx.GraphModule

        """
        traced: torch.fx.GraphModule = _export_to_torch_ir(model, args=args, kwargs=kwargs)
        return traced

    def parse_huggingface_fx(
        self, model, partition_units: List[str] = None, shard_plan: Dict = None, default_settings: bool = True
    ):
        """
        Applies symbolic trace with huggingface-like fx.

        Args:
            model (nn.Module): raw model
            partition_units (List[str]): a list of smallest unsplittable modules such that the parser will
                not flatten their underlying components during parsing
            shard_plan (Dict): dictionary of sharding plan, if users would like to wrap up tensor parallelized
                modules as unsplittable units

        Returns:
            Captured torch.fx.GraphModule

        """
        if partition_units is None:
            partition_units = []
        input_arguments = signature(model.forward).parameters.keys()
        # parser flattens module hierachy during parse. Maintain hierachy so that it can still be accessed by a sharding plan
        if shard_plan:
            hierarchy_substructure_qualified_names = self._hierarchy_structure_names(model, shard_plan)
            partition_units += hierarchy_substructure_qualified_names
        input_names = list(input_arguments)
        if default_settings:
            default_input_names, default_unit_modules = self._default_parse_info(model, input_names)
            if default_input_names:
                input_names = default_input_names
            if default_unit_modules:
                partition_units = default_unit_modules
        if "input_ids" in input_names and "inputs_embeds" in input_names:
            # two arguments cannot occur simultanenously
            input_names.remove("inputs_embeds")
        input_names = input_names[:NUM_DEFAULT_ARGS]
        traced: torch.fx.GraphModule = hf_symbolic_trace(
            model,
            input_names=input_names,
            disable_check=True,
            tracer_cls=HFModelTracer,
            partition_modules=partition_units,
        )
        return traced

    def _hierarchy_structure_names(self, model, shard_plan):
        modules_to_maintain_hierarchy = set()
        self._collect_hierachical_modules_paths(model, shard_plan["forward"], modules_to_maintain_hierarchy)
        self._collect_hierachical_modules_paths(model, shard_plan["parameter"], modules_to_maintain_hierarchy)
        return modules_to_maintain_hierarchy

    def _collect_hierachical_modules_paths(self, model, plan_dict, module_paths):
        for path_to_submodule, _ in model.named_modules():
            for plan_fqn in plan_dict:
                pattern = plan_fqn.rsplit(".", 1)[0]
                if (
                    re.match(pattern, path_to_submodule)
                    and len(list(model.get_submodule(path_to_submodule).children())) != 0
                ):
                    module_paths.add(path_to_submodule)

    def _locate_module_classes(self, model, paths_to_submodules):
        if paths_to_submodules is None:
            return paths_to_submodules
        visited = set(paths_to_submodules)
        submodule_classes = set()
        for name, submodule in model.named_modules():
            if name in visited:
                submodule_classes.add(type(submodule))
        return list(submodule_classes)

    def _default_parse_info(self, model, input_names, num_default_args=3):
        from transformers.models.whisper.modeling_whisper import WhisperModel
        from transformers.models.mixtral.modeling_mixtral import (
            MixtralModel,
            MixtralRMSNorm,
            MixtralSparseMoeBlock,
            MixtralAttention,
        )
        from transformers.models.biogpt.modeling_biogpt import BioGptModel, BioGptAttention
        from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Model, DisentangledSelfAttention
        from transformers.models.marian.modeling_marian import MarianModel, MarianAttention, MarianEncoderLayer
        from transformers.models.blenderbot.modeling_blenderbot import (
            BlenderbotModel,
            BlenderbotAttention,
            BlenderbotEncoderLayer,
        )
        from transformers.models.layoutlmv3.modeling_layoutlmv3 import LayoutLMv3Model, LayoutLMv3SelfAttention
        from transformers.models.phi.modeling_phi import PhiModel, PhiAttention
        from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXModel, GPTNeoXAttention
        from transformers.models.falcon.modeling_falcon import FalconModel, FalconAttention
        from transformers.models.gpt_bigcode.modeling_gpt_bigcode import GPTBigCodeModel, GPTBigCodeAttention
        from transformers.models.vit.modeling_vit import ViTModel, ViTEmbeddings, ViTSelfAttention
        from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2Attention
        from transformers.models.speecht5.modeling_speecht5 import SpeechT5Model, SpeechT5Attention
        from transformers.models.bloom.modeling_bloom import BloomModel, BloomAttention

        model_type = type(model)
        input_names = partition_unit_classes = None
        if model_type == MixtralModel:
            partition_unit_classes = [MixtralRMSNorm, MixtralSparseMoeBlock, MixtralAttention]
        elif model_type == BioGptModel:
            partition_unit_classes = [BioGptAttention]
        elif model_type == DebertaV2Model:
            partition_unit_classes = [DisentangledSelfAttention]
        elif model_type == MarianModel:
            partition_unit_classes = [MarianAttention, MarianEncoderLayer]
        elif model_type == BlenderbotModel:
            partition_unit_classes = [BlenderbotAttention, BlenderbotEncoderLayer]
        elif model_type == LayoutLMv3Model:
            partition_unit_classes = [LayoutLMv3SelfAttention]
        elif model_type == PhiModel:
            partition_unit_classes = [PhiAttention]
        elif model_type == GPTNeoXModel:
            partition_unit_classes = [GPTNeoXAttention]
        elif model_type == FalconModel:
            partition_unit_classes = [FalconAttention]
        elif model_type == GPTBigCodeModel:
            partition_unit_classes = [GPTBigCodeAttention]
        elif model_type == ViTModel:
            partition_unit_classes = [ViTEmbeddings, ViTSelfAttention]
        elif model_type == Wav2Vec2Model:
            partition_unit_classes = [Wav2Vec2Attention]
        elif model_type == SpeechT5Model:
            partition_unit_classes = [SpeechT5Attention]
        elif model_type == BloomModel:
            input_names = ["attention_mask", "head_mask", "inputs_embeds"]
            partition_unit_classes = [BloomAttention]
        elif model_type == WhisperModel:
            input_names = ["input_features", "decoder_input_ids"]

        if input_names:
            input_names = input_names[:num_default_args]
        return input_names, partition_unit_classes


def parse_model_graph(parser: PipeParser, model: nn.Module, plan: PipelineParallelPlan) -> torch.fx.GraphModule:
    """
    Pipeline Parallelism API that performs parsing given tracer types.

    Args:
        parser (PipeParser): model parser
        model (nn.Module): raw model
        plan (PipelineParallelPlan): configuration of pipeline paralellism API.

    Returns:
        Captured torch.fx.GraphModule

    """
    tracer_type = plan.tracer_type
    tracer_kwargs = plan.tracer_kwargs
    if tracer_kwargs is None:
        tracer_kwargs = {}
    if tracer_type == TracerType.AUTO:
        model_graph = parser.parse(model, plan)
    else:
        if "partition_units" not in tracer_kwargs and tracer_type in [TracerType.TORCH_FX, TracerType.HF_FX]:
            tracer_kwargs["partition_units"] = plan.smallest_unsplittable_units
        if tracer_type == TracerType.TORCH_FX:
            model_graph = parser.parse_torch_fx(model, **tracer_kwargs)
        elif tracer_type == TracerType.HF_FX:
            model_graph = parser.parse_huggingface_fx(model, **tracer_kwargs)
        elif tracer_type == TracerType.TORCH_DYNAMO:
            model_graph = parser.parse_dynamo_export(model, **tracer_kwargs)
        else:
            raise NotImplementedError(f"Logic of tracer {tracer_type} has not been implemented yet.")
    return model_graph


def split_pipeline_point(model: nn.Module, plan: PipelineParallelPlan):
    """
    Pipeline Parallelism API that updates pipeline stage split points.

    Args:
        model (nn.Module): raw model
        plan (PipelineParallelPlan): configuration of pipeline paralellism API.

    Returns:
        Captured torch.fx.GraphModule.

    """
    # obtain the traced graph of entire model if pipeline parallelism is on
    parser = PipeParser()
    model_graph = parse_model_graph(parser, model, plan)
    split_points = parser.split(model_graph, plan)
    plan.split_points = split_points
    return split_points, model_graph, parser


def construct_pipeline_split_graph(model: nn.Module, plan: PipelineParallelPlan, update_split_points: bool = False):
    """
    Pipeline Parallelism API that performs pipeline stage split.

    Args:
        model (nn.Module): raw model
        plan (PipelineParallelPlan): configuration of pipeline paralellism API.
        update_split_points (bool): set this switch on to update pipeline split points in-place.

    Returns:
        Captured torch.fx.GraphModule.

    """
    parser = PipeParser()
    model_graph = parse_model_graph(parser, model, plan)
    if update_split_points:
        split_points = parser.split(model_graph, plan)
        plan.split_points = split_points
    # partition model graph into virtual pipeline chunks per stage
    split_graph = parser.split_stage(model_graph, model, plan)
    return split_graph
