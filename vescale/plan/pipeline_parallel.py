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


from dataclasses import dataclass, field
from typing import List, Dict
from .spec import *  # noqa: F403
import torch

__all__ = ["PipelineParallelPlan"]


@dataclass
class PipelineParallelPlan:
    # PP mode:
    mode: ModeType = ModeType.GRAPH_EAGER

    ########## model graph and partition ##########

    # type of tracer to obtain the model execution graph
    # fit modes: [GRAPH_EAGER]
    # format: Enum
    # consumer: PipeParser
    tracer_type: TracerType = TracerType.AUTO

    # kwargs to be fed to different parser, e.g. torch.fx, dynamo, export, etc
    # fit modes: [GRAPH_EAGER]
    # format: Enum
    # consumer: PipeParser
    tracer_kwargs: Dict = None

    # method of stage partitioning for all modes
    # fit modes: [EAGER, GRAPH_EAGER]
    # format: Enum
    # consumer: PipeParser and ManualPipeParser
    split_method: PipelineSplitMethodType = PipelineSplitMethodType.MANUAL

    # number of pipeline stages
    # fit modes: [EAGER, GRAPH_EAGER]
    # format: int
    # consumer: PipeParser
    num_stages: int = 2

    # number of virtual module chunks per pipeline stage
    # fit modes: [EAGER, GRAPH_EAGER]
    # format: int
    # consumer: ScheduleEngine, PipeModule
    virtual_chunks: int = 1

    # list of minimum un-partitionable units in model forward graph. Internal hierarchy
    #     of a partition unit is maintained during stage splitting
    # fit modes: [EAGER, GRAPH_EAGER]
    # format: list of fqns to particular modules/callable or module classes
    # consumer: ScheduleEngine, PipeModule
    smallest_unsplittable_units: List = field(default_factory=list)

    # stage boundaries
    # fit modes: [EAGER, GRAPH_EAGER]
    # format: a list of fqns or index integers of particular modules / callables
    # consumer: PipeParser and ManualParser
    split_points: List = field(default_factory=list)

    # enables to manually define boundaries of virtual stage chunks in interleaved 1F1B schedule
    # fit modes: [EAGER, GRAPH_EAGER]
    # format: boolean
    # consumer: PipeParser and ManualParser
    enable_vpp_split_points: bool = False

    # enables to uniformly split stages by modules and operators when split_method==PipelineSplitMethodType.UNIFORM
    # fit modes: [EAGER, GRAPH_EAGER]
    # format: boolean
    # consumer: PipeParser and ManualParser
    uniform_split_ops: bool = False

    ########## end of model graph generation, partition ##########

    ########## pipeline runtime ##########

    # executes batched p2p communication for simple 1f1b and interleaved 1f1b,
    #     mutually exclusive to overlap_p2p_comm
    # fit modes: [EAGER, GRAPH_EAGER]
    # format: bool
    # consumer: ScheduleEngine
    batch_p2p_comm: bool = False

    # executes overlapped p2p communication for simple 1f1b and interleaved 1f1b,
    #     mutually exclusive to batch_p2p_comm
    # fit modes: [EAGER, GRAPH_EAGER]
    # format: bool
    # consumer: ScheduleEngine
    overlap_p2p_comm: bool = True

    # sets to True in inference, so that pipeline schedule only executes forward propagation
    # fit modes: [EAGER, GRAPH_EAGER]
    # format: bool
    # consumer: ScheduleEngine
    forward_only: bool = False

    # pipeline schedule type
    # fit modes: [EAGER, GRAPH_EAGER]
    # format: Enum
    # consumer: ScheduleEngine
    schedule_type: PipelineScheduleType = PipelineScheduleType.SIMPLE_1F1B

    # reuses data tensor shapes in some use cases instead of communicating
    #     shapes before tensors. Use with caution!
    # fit modes: [EAGER, GRAPH_EAGER]
    # format: bool
    # consumer: ScheduleEngine
    reuse_p2p_tensor_shape: bool = False

    # precision types of communicated tensors during pipeline execution
    # fit modes: [EAGER, GRAPH_EAGER]
    # format: torch.dtype
    # consumer: ScheduleEngine
    p2p_tensor_dtype: torch.dtype = torch.float32

    ########## end of pipeline schedule ##########

    ########## other information ##########

    # list of groups of fqns whose parameters or gradients will be synchronized per step, e.g. embedding modules
    # fit modes: [EAGER, GRAPH_EAGER]
    # format: [ [word_embeddingA, word_embeddingB], [vision_embeddingA, vision_embeddingB] ]
    # consumer: build utilities in vescale/api.py
    shared_modules: List[List[str]] = field(default_factory=list)

    ########## end of other information ##########
