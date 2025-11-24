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

# mypy: ignore-errors
from vescale.dtensor.device_mesh import DeviceMesh
from vescale.plan.pipeline_parallel import PipelineParallelPlan
from vescale.plan.spec import PipelineScheduleType
from vescale.pipe._schedules import (
    OneFOneBInstrcutionGenerator,
    InterleavedOneFOneBInstructionGenerator,
    ZeroBubbleVInstrcutionGenerator,
    StageDeps,
    Shape,
)
from vescale.pipe._schedules.instruction_base import VESCALE_INTRUCTION_BUILDER as builder
from vescale.pipe.p2p_communication import reset_global_counter
from vescale.devicemesh_api.api import VeDeviceMesh
from collections import OrderedDict
from typing import Callable, Iterator, List, Sequence, Union
import torch
import torch.distributed as dist
import logging
import os


logger = logging.Logger(__file__)


class PipelineEmitter:
    """Pipeline Emitter."""

    def __init__(
        self,
        deps: StageDeps,
        meshes: Sequence[DeviceMesh],
        schedule: str,
        batches: int,
        tensor_shape: Shape,
        dtype: torch.dtype,
        num_chunks: int = 1,
        input_shapes: List[Shape] = None,
        input_shapes_unpad: List[Shape] = None,
        forward_only=False,
        overlap_p2p_comm=False,
        batch_p2p_comm: bool = True,
        param_sync_overlap=False,
        grad_sync_overlap=False,
        **kwargs,
    ):
        self.deps = deps
        self.num_stage = deps.num_stage
        self.meshes = meshes
        self.batches = batches
        self.num_chunks = num_chunks
        self.overlap_p2p_comm = overlap_p2p_comm
        self.batch_p2p_comm = batch_p2p_comm
        self.param_sync_overlap = param_sync_overlap
        self.forward_only = forward_only
        self.grad_sync_overlap = grad_sync_overlap
        if schedule == PipelineScheduleType.SIMPLE_1F1B:
            self.num_meshes = meshes
            self.instruction_generator = OneFOneBInstrcutionGenerator(
                deps=deps,
                meshes=self.meshes,
                batches=batches,
                default_shape=tensor_shape,
                default_dtype=dtype,
                forward_only=self.forward_only,
            )

        elif schedule == PipelineScheduleType.INTERLEAVED_1F1B:
            self.instruction_generator = InterleavedOneFOneBInstructionGenerator(
                deps=deps,
                meshes=self.meshes,
                batches=batches,
                default_shape=tensor_shape,
                default_dtype=dtype,
                input_shapes=input_shapes,
                input_shapes_unpad=input_shapes_unpad,
                num_chunks=self.num_chunks,
                batch_p2p_comm=batch_p2p_comm,
                overlap_p2p_comm=overlap_p2p_comm,
                param_sync_overlap=param_sync_overlap,
                grad_sync_overlap=grad_sync_overlap,
                forward_only=forward_only,
            )

        elif schedule == PipelineScheduleType.ZERO_BUBBLE:
            self.instruction_generator = ZeroBubbleVInstrcutionGenerator(
                deps=deps,
                meshes=self.meshes,
                batches=batches,
                default_shape=tensor_shape,
                default_dtype=dtype,
                **kwargs,
            )
        else:
            raise NotImplementedError("unsupport schedule type")
        self.instruction_list: List[List] = self.gen_instruction()

    def gen_instruction(self):
        """
        Generates instruction steps of a pipeline schedule.
        """
        return self.instruction_generator.gen_instruction()

    def get_instruction_list(self, stage: int):
        """
        Generates instruction steps of a pipeline schedule for a particular pipeline stage.

        Args:
            stage (int): pipeline stage id

        """
        return self.instruction_generator.get_instruction_list(stage)


class ScheduleEngine:
    def __init__(
        self,
        deps: StageDeps,
        meshes: int,
        schedule: PipelineScheduleType,
        batches: int,
        data_iterator: Union[Iterator, List[Iterator]],
        stage_id: int,
        shape: Union[Shape, Sequence[Shape]],
        dtype: Union[torch.dtype, Sequence[torch.dtype]] = torch.float32,
        num_chunks=1,
        input_shapes: List[Shape] = None,
        input_shapes_unpad: List[Shape] = None,
        forward_only=False,
        overlap_p2p_comm=False,
        batch_p2p_comm: bool = True,
        param_sync_overlap=False,
        grad_sync_overlap=False,
        send_dtypes_map: OrderedDict = None,
        loss_fn: Callable = lambda x: torch.sum(x),
        global_mesh: VeDeviceMesh = None,
        **kwargs,
    ):
        os.environ["STAGE_ID"] = str(stage_id)
        self.p_emmiter = PipelineEmitter(
            deps,
            meshes,
            schedule,
            batches,
            shape,
            dtype,
            num_chunks=num_chunks,
            input_shapes=input_shapes,
            input_shapes_unpad=input_shapes_unpad,
            forward_only=forward_only,
            overlap_p2p_comm=overlap_p2p_comm,
            batch_p2p_comm=batch_p2p_comm,
            param_sync_overlap=param_sync_overlap,
            grad_sync_overlap=grad_sync_overlap,
            **kwargs,
        )
        self.schedule = schedule
        self.deps = deps
        self.instruction_list = self.get_instruction_list(stage_id)
        self.stage_id = stage_id
        self.shape = shape
        self.dtype = dtype
        self.chunk = num_chunks
        self.send_dtypes_map = send_dtypes_map
        builder.topo = deps
        builder.dataloader = data_iterator
        builder.loss_fn = loss_fn
        self.src_loss_rank = -1
        self.global_mesh = global_mesh
        if self.global_mesh:
            all_ranks = list(range(dist.get_world_size()))
            dp_rank = self.global_mesh.get_data_parallel_rank()
            tp_rank = self.global_mesh.get_tensor_parallel_rank()
            same_pipeline_group = [
                rank for rank in all_ranks if self.global_mesh.get_strategy_coordinate(rank)[1:] == [dp_rank, tp_rank]
            ]
            for rank in same_pipeline_group:
                if self.global_mesh.get_strategy_coordinate(rank)[0] == self.global_mesh.size(0) - 1:
                    self.src_loss_rank = rank
                    break
            # the group for all ranks in the same pipeline to share final loss outputs
            self.sync_loss_group = dist.new_group(ranks=same_pipeline_group, backend="nccl")

    def set_data_iterator(self, data_iterator: List, data_shape=None):
        """
        Assigns minibatch data to instruction builder.

        Args:
            data_iterator (List): a minibatch list of microbatch data

        """
        assert builder.dataloader
        builder.dataloader = data_iterator
        if data_shape:
            self.shape = data_shape
            builder.constant_data["shape"] = data_shape

    def get_instruction_list(self, stage_id):
        return self.p_emmiter.get_instruction_list(stage_id)

    def sync_output_loss_per_pipeline(self, loss: torch.Tensor):
        """
        A debug mode function that synchronizes minibatch loss
        with all stages of a pipeline.

        Args:
            data_iterator (List): a minibatch list of microbatch data

        """
        assert self.global_mesh, "Must initialize per-pipeline dist group before synchronizing loss!"
        if loss is None:
            loss = torch.tensor(0.0, dtype=torch.float).cuda(dist.get_rank())
        dist.broadcast(loss, src=self.src_loss_rank, group=self.sync_loss_group)

        # monkey patch torch.tensor loss backward as empty tensor to make it a dummy function
        def _empty_backward():
            return None

        loss.backward = _empty_backward
        return loss

    def _collect_microbatch_losses(self, outputs):
        # monkey patch torch.tensor loss backward as empty tensor to make it a dummy function
        def _empty_backward():
            return None

        output_losses = []
        for microbatch_output, microbatch_loss in outputs:
            if microbatch_loss is None:
                if isinstance(microbatch_output, Sequence):
                    for j in range(len(microbatch_output)):
                        if microbatch_output[j].ndim == 0 and microbatch_output[j].numel() == 1:
                            loss_value = microbatch_output[j]
                            break
                    else:
                        raise ValueError("Loss values not found.")
                else:
                    loss_value = microbatch_output
            else:
                # monkey patch microbatch loss backward as empty tensor to make it a dummy function
                loss_value = microbatch_loss
            output_losses.append(loss_value)
        if not output_losses:
            return None
        tensor_device = output_losses[0].device
        minibatch_loss = torch.tensor(sum(output_losses), device=tensor_device)
        minibatch_loss.backward = _empty_backward
        return minibatch_loss

    @staticmethod
    def execute(
        instance,
        *,
        deallocate_pipeline_outputs: bool = False,
        autocast_dtype: torch.dtype = torch.float,
        enable_autocast: bool = False,
        grad_scaler=None,
        param_sync_func=None,
        grad_sync_func=None,
        debug_mode=False,
    ):
        """
        Main entry point of executing forward and backward
        computation of a minibatch.

        Args:
            instance (ScheduleEngine): a minibatch list of microbatch data
            deallocate_pipeline_outputs (bool): deallocate tensors
            autocast_dtype (torch.dtype): autocast data types
            enable_autocast (bool): turn on to enable tensor autocast
            grad_scaler (Callable): gradient scaler
            param_sync_func (Callable): gradient synchronization function
            debug_mode (bool): turn on to generate debugging outputs

        Returns:
            A tuple of two elements:
                1). loss of this minibatch of data,
                2). a list of tuple of outputs per microbatch, where for each tuple:
                    - 2.1). the first element is output of the original model
                    - 2.2). the second element is the loss of this microbatch.
                        If loss_fn is not provided at initialization, it means loss
                        is computed in 2.1) and here will return None

        """
        reset_global_counter()
        if instance.schedule == PipelineScheduleType.SIMPLE_1F1B:
            minibatch_outputs = instance.p_emmiter.instruction_generator.execute(
                stage_id=instance.stage_id,
                enable_autocast=enable_autocast,
                autocast_dtype=autocast_dtype,
                grad_scaler=grad_scaler,
                deallocate_pipeline_outputs=deallocate_pipeline_outputs,
            )
            minibatch_loss = instance._collect_microbatch_losses(minibatch_outputs)
            if debug_mode:
                minibatch_loss = instance.sync_output_loss_per_pipeline(minibatch_loss)
            return minibatch_loss, minibatch_outputs
        elif instance.schedule == PipelineScheduleType.INTERLEAVED_1F1B:
            minibatch_outputs = instance.p_emmiter.instruction_generator.execute(
                stage_id=instance.stage_id,
                enable_autocast=enable_autocast,
                autocast_dtype=autocast_dtype,
                grad_scaler=grad_scaler,
                deallocate_pipeline_outputs=deallocate_pipeline_outputs,
                param_sync_func=param_sync_func,
                grad_sync_func=grad_sync_func,
            )
            minibatch_loss = instance._collect_microbatch_losses(minibatch_outputs)
            if debug_mode:
                minibatch_loss = instance.sync_output_loss_per_pipeline(minibatch_loss)
            return minibatch_loss, minibatch_outputs
        elif instance.schedule == PipelineScheduleType.ZERO_BUBBLE:
            minibatch_outputs = instance.p_emmiter.instruction_generator.execute(
                stage_id=instance.stage_id,
                enable_autocast=enable_autocast,
                autocast_dtype=autocast_dtype,
                grad_scaler=grad_scaler,
                deallocate_pipeline_outputs=deallocate_pipeline_outputs,
            )
            minibatch_loss = instance._collect_microbatch_losses(minibatch_outputs)
            if debug_mode:
                minibatch_loss = instance.sync_output_loss_per_pipeline(minibatch_loss)
            return minibatch_loss, minibatch_outputs
        else:
            raise NotImplementedError("Unsupported Schedule!")


def validate_pipeline_schedule(plan: PipelineParallelPlan):
    """
    Validates pipeline schedule settings in Pipeline ParallelPlan.

    Args:
        plan (PipelineParallelPlan): configuration of pipeline parallel API attributes

    """
    if plan.schedule_type == PipelineScheduleType.INTERLEAVED_1F1B:
        assert plan.virtual_chunks > 1
    elif plan.schedule_type == PipelineScheduleType.SIMPLE_1F1B:
        assert plan.virtual_chunks == 1
