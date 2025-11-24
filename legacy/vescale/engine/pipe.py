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

from collections import defaultdict
from typing import Any, List, Callable
from vescale.pipe.pipe_stage import PipeModule
from vescale.plan.pipeline_parallel import PipelineParallelPlan
from vescale.pipe.pipe_emmiter import ScheduleEngine, StageDeps
from vescale.devicemesh_api import VeDeviceMesh
from vescale.plan.spec import PipelineScheduleType
from vescale.ddp.distributed_data_parallel import DistributedDataParallel as DDP
from copy import deepcopy
import torch
import torch.nn as nn
import torch.distributed as dist
import os


class PipeEngine:
    def __init__(
        self,
        module: PipeModule,
        global_mesh: VeDeviceMesh,
        loss_fn: Callable,
        plan: PipelineParallelPlan,
    ):
        """
        Training engine for pipeline parallelism and multi-dimensional
        parallelism that underlies pipeline parallelism (distributed optimizer, data parallel,
        tensor model parallel, and sequence parallel, etc).
        The training engine is responsible for materializes stage partitioning, module registration,
        training, and optimizer synchronization.
        """
        self.module = module
        self.virtual_chunks_per_stage = plan.virtual_chunks
        self.engine_plan = plan
        self.optimizer = self.module.get_optimizer
        self.lr_scheduler = self.module.get_lr_scheduler
        self.global_mesh = global_mesh
        if isinstance(loss_fn, nn.Module):
            self.loss_fn = loss_fn
        else:
            try:
                self.loss_fn = deepcopy(loss_fn.__func__)
            except:  # noqa: E722
                self.loss_fn = loss_fn
        self.schedule_engine = None
        self.reuse_comm_shape = self.engine_plan.reuse_p2p_tensor_shape
        if self.reuse_comm_shape:
            os.environ["REUSE_COMM_SHAPE"] = "1"
        if (
            self.engine_plan.schedule_type == PipelineScheduleType.INTERLEAVED_1F1B
            and self.virtual_chunks_per_stage == 1
        ):
            print("[warning]: #virtual pipeline chunks is 1. Falling back to simple 1F1B schedule.")
            self.engine_plan.schedule_type = PipelineScheduleType.SIMPLE_1F1B
        self.schedule_type = self.engine_plan.schedule_type

    def build_schedule(self, minibatches, data_shape=None):
        """
        Build pipeline parallel training schedules.
        """
        meshes = self.global_mesh.get_global_tensor_parallel_meshes()
        dp_rank, tp_rank = self.global_mesh.get_data_parallel_rank(), self.global_mesh.get_tensor_parallel_rank()
        tp_meshes_dict = defaultdict(list)

        def _locate_tp_mesh(_rank):
            for tp_mesh in meshes:
                if _rank in tp_mesh.mesh.tolist():
                    return tp_mesh
            else:
                raise ValueError("TP submesh not found.")

        for _rank in range(torch.distributed.get_world_size()):
            _coordinate = self.global_mesh.get_strategy_coordinate(_rank)
            tp_mesh = _locate_tp_mesh(_rank)
            _dp_rank, _tp_rank = _coordinate[1], _coordinate[2]
            tp_meshes_dict[(_dp_rank, _tp_rank)].append(tp_mesh)

        new_meshes = tp_meshes_dict[(dp_rank, tp_rank)]
        meshes = new_meshes
        first_stage_rank = self.global_mesh.get_strategy_coordinate(local_rank=0)[0]
        # FIXME: the input can either be PipeModule, or a sequence of DDP modules? In the latter case, how to get stage dependency
        pipe_module = self.module
        stage_dep_matrix, p2p_index_mapping = pipe_module.stage_deps, pipe_module.p2p_index_mapping
        stage_dependency = StageDeps(
            dep=stage_dep_matrix,
            meshes=meshes,
            vpp_module_list=pipe_module,
            p2p_index_mapping=p2p_index_mapping,
        )
        num_minibatches = self._align_num_batches(first_stage_rank, len(minibatches))
        # TODO: insert shape inference
        batch_p2p_comm = self.engine_plan.batch_p2p_comm
        # if on interleaved 1f1b schedule, set batch_p2p_comm to False to execute p2p communication
        schedule_type = self.schedule_type
        if schedule_type in [PipelineScheduleType.INTERLEAVED_1F1B, PipelineScheduleType.ZERO_BUBBLE]:
            data_iterator = [iter(minibatches) for _ in range(self.virtual_chunks_per_stage)]
            batch_p2p_comm = False
        elif schedule_type == PipelineScheduleType.SIMPLE_1F1B:
            data_iterator = minibatches
        else:
            raise NotImplementedError(f"Schedule {schedule_type} not implemented yet.")
        return ScheduleEngine(
            stage_dependency,
            meshes,
            schedule_type,
            num_minibatches,
            data_iterator=data_iterator,
            stage_id=self.global_mesh.get_pipeline_parallel_rank(),
            shape=data_shape,
            dtype=self.engine_plan.p2p_tensor_dtype,
            num_chunks=self.virtual_chunks_per_stage,
            input_shapes=None,
            input_shapes_unpad=None,
            # send_dtypes_map=self.module.recv_dtypes_dict,
            overlap_p2p_comm=self.engine_plan.overlap_p2p_comm,
            batch_p2p_comm=batch_p2p_comm,
            loss_fn=self.loss_fn,
            global_mesh=self.global_mesh,
            forward_only=self.engine_plan.forward_only,
        )

    def forward_backward(
        self,
        minibatch,
        reuse_schedule=False,
        data_shape=None,
        debug_mode: bool = False,
    ):
        """
        Execute the pipeline schedule to complete forward,
        backward, and gradient step of one minibatch.

        Invoke Scheduler's execute_pipeline() to run a minibatch.
        """
        assert isinstance(minibatch, List), "Input must be a list of microbatches"
        if reuse_schedule:
            if self.schedule_engine is None:
                schedule_engine = self.build_schedule(minibatch, data_shape=data_shape)
            else:
                schedule_engine = self.schedule_engine
                schedule_engine.set_data_iterator(minibatch, data_shape=data_shape)
        else:
            schedule_engine = self.build_schedule(minibatch, data_shape=data_shape)
        # returns model output tensors and losses per microbatch
        return ScheduleEngine.execute(schedule_engine, debug_mode=debug_mode)

    def forward(self, *args: Any, **kwargs: Any):
        raise ValueError("Forward is done in PipeEngine.forward_backward()!")

    def __call__(self, *args: Any, **kwargs: Any):
        return self.forward_backward(*args, **kwargs)

    def backward(self, *args: Any, **kwargs: Any):
        raise ValueError("Backward is done in PipeEngine.forward_backward()!")

    @property
    def get_optimizer(self):
        """
        Return this stage's optimizer.
        """
        return self.optimizer

    @property
    def get_lr_scheduler(self):
        return self.lr_scheduler

    def zero_grad_buffer(self, zero_buffer: bool = True):
        for vpp_module in self.module.stage_modules.values():
            if isinstance(vpp_module, DDP):
                vpp_module.zero_grad_buffer(zero_buffer)

    def finish_grad_sync(self):
        for vpp_module in self.module.stage_modules.values():
            if isinstance(vpp_module, DDP):
                vpp_module.finish_grad_sync()

    def train(self, mode: bool = True):
        for vpp_module in self.module.stage_modules.values():
            vpp_module.train(mode)

    def eval(self):
        for vpp_module in self.module.stage_modules.values():
            vpp_module.eval()

    def parameters(self, including_frozen=False):
        """
        Return meta information of the entire model's
        parameters.
        """
        if including_frozen:
            return self.module.parameters()
        else:
            return filter(lambda p: p.requires_grad, self.module.parameters())

    def sync_shared_params(self, group_id: int = 0, share_params=True) -> None:
        """
        Synchronize gradients and weights among groups of specified units, dictated by
        "partition_units" in PipelineParallelPlan. Typically, this function is used for
        synchronizing gradients and weights of embeddings layers in Transformer-based
        architecture.
        Args:
            group_id (int): specify groups of modules across stages to synchronize. Default by 0.
            share_params (bool): if True, sync weight parameters; otherwise, share gradients.
        """
        local_rank = dist.distributed_c10d.get_rank()
        tp_coordinate = self.module.device_mesh_management.get_tensor_parallel_rank()
        if self.module.shared_module_mapping and local_rank in dist.distributed_c10d.get_process_group_ranks(
            self.module.shared_module_process_groups[group_id][tp_coordinate]
        ):
            self.module.sync_shared_params(self.global_mesh, group_id=group_id, share_params=share_params)

    def _align_num_batches(self, first_stage_rank, batches):
        """
        Aligns all ranks must have the same number of mini-batches as rank 0.
        """
        num_batches = torch.tensor([batches], dtype=torch.int64).cuda(dist.get_rank())
        dist.broadcast(num_batches, src=first_stage_rank)
        is_consistent = num_batches.item() == batches
        if not is_consistent:
            batches = num_batches.item()
        return batches
