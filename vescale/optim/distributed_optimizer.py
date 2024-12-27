################################################################################
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

"""Megatron distributed optimizer."""

import math
from dataclasses import dataclass
from typing import Any, Type, Union, Dict, Sequence, Tuple, Optional, List

import torch
import torch.distributed as dist

from vescale.dtensor.dtensor import DTensor
from vescale.ddp.distributed_data_parallel import DistributedDataParallel as DDP
from vescale.optim.base_optimizer import OptimizerBase
from vescale.dtensor._utils import compute_local_shape_and_global_offset
from vescale.ddp.grad_buffer import Bucket

from vescale.optim.utils import param_is_shared, param_is_sharded_or_replicate_on_first_rank, zero_grad_group_helper
from vescale.optim.clip_grads import clip_grad_norm_fp32


class Range:
    """
    A range represents a start and end points for indexing a shard
    from a full tensor.
    """

    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.size = end - start

    def normalize(self, start=0):
        return Range(start, start + self.size)

    def __str__(self):
        return "%d,%d [%d]" % (self.start, self.end, self.size)

    def __len__(self):
        return self.end - self.start

    def __repr__(self) -> str:
        return "Range(%d,%d [%d])" % (self.start, self.end, self.size)

    def __eq__(self, other):
        if not isinstance(other, Range):
            return False
        return self.start == other.start and self.end == other.end


@dataclass
class OptimizerStateSpec:
    """This class represents mapping between local flattened 1D tensor
    and global original DTensor in DOptimzier, it is used for
    loading or saving optimizer states using vescale.checkpoint (PyTorch DCP)
    and load-time checkpoint resharding when changing tp size or dp size.

    For example, a linear layer in Vescale is DTensor(size=[1024, 1024])
    It first divides into two parts along dim=0 with tensor parallel size = 2

    tensor_part_0 = DTensor(size=[512, 1024])
    tensor_part_1 = DTensor(size=[512, 1024])

    Then each part's optimizer states are initalized in DOptimizer sepearately

    Assume dp=2
    For process with dp=0 tp=0, the flatten tensor is torch.Tensor(size=[262144])
    global_shape=(1024, 1024), local_shape=(256, 1024), global_offset=(0, 0) local=torch.Tensor(size=[262144]).view(local_shape)

    For process with dp=1 tp=0, the flatten tensor is torch.Tensor(size=[262144])
    global_shape=(1024, 1024), local_shape=(256, 1024), global_offset=(256, 0) local=torch.Tensor(size=[262144]).view(local_shape)

    For process with dp=0 tp=1, the flatten tensor is torch.Tensor(size=[262144])
    mapping to [512:768, 0:1024] in original DTensor
    global_shape=(1024, 1024), local_shape=(256, 1024), global_offset=(512, 0) local=torch.Tensor(size=[262144]).view(local_shape)

    For process with dp=1 tp=1, the flatten tensor is torch.Tensor(size=[262144])
    global_shape=(1024, 1024), local_shape=(256, 1024), global_offset=(768, 0) local=torch.Tensor(size=[262144]).view(local_shape)
    """

    # The original DTensor shape
    global_shape: Tuple[int]
    # The local tensor shape ***before flattened into 1D tensor***
    local_shape: Tuple[int]
    # The local tensor's offset with respect to origianl DTensor
    global_offset: Tuple[int]
    # The unflattened local tensor after create view using local_shape on the flattened 1D Tensor in DOptimizer
    # NOTE: In order to support TP resharding and state cross dp ranks, we defer the reshaping from 1D to local_shape
    # to generate saving plan using vescale.checkpoint (PyTorch DCP)
    local_tensor: torch.Tensor
    # If the current optimizer state is sharded by multiple dp ranks,
    # we should record all ranks and their ranges
    dp_ranks_ranges: Optional[Dict[int, Range]]


@dataclass
class ZeroParamSpec:
    """Records of parameter distribution and GradBuffer's mapping in Zero optimization"""

    # group index: (global group index, local index in group)
    # the ddp bucket that this parameter belongs to
    ddp_bucket: Bucket
    # the shard parameter for this dp rank
    shard_param: torch.Tensor
    # the shard main parameter (fp32) for this dp rank.
    # this field will be set to None if not using mixed precision or
    # the parameter is already in fp32
    shard_main_param: Optional[torch.Tensor]
    # the slicing index of this shard parameter in the global gradient buffer
    global_gbuf_slice: Optional[slice]
    # the slicing index of this shard parameter in the (tp) parameter
    param_slice: Optional[slice]
    # shard (main) parameter buffer (linked to DDP's grad buffer).
    # this is used for updated (main) param with later all-gather
    shard_param_buffer: torch.Tensor
    # parameter buffer (shared storage with DDP's grad buffer).
    # this is for all-gathered param to copy back to model param
    param_buffer: torch.Tensor


def _convert_dict_with_sharded(
    param_state: dict,
    global_shape: Tuple[int],
    local_shape: Tuple[int],
    global_offset: Tuple[int],
    dp_ranks_ranges: Optional[Dict[int, Range]],
):
    new_param_state = {}
    for k, v in param_state.items():
        if isinstance(v, DTensor):
            v = v._local_tensor
        if isinstance(v, torch.Tensor) and v.dim() >= 1:
            # Don't unflatten tensor here, see the comments above
            if not dp_ranks_ranges:
                if math.prod(local_shape) != math.prod(v.shape):
                    print(f"rank={dist.get_rank()} name={k} global shape={global_shape}\
                    local_shape={local_shape} global_offset={global_offset} real shape={v.shape}")
                    raise AssertionError()
            new_param_state[k] = OptimizerStateSpec(
                global_shape, local_shape, global_offset, v, dp_ranks_ranges
            )  # , process_group)
        else:
            new_param_state[k] = v
    return new_param_state


def _convert_dict_sharded_to_tensor(param_state: dict, range_1d: Optional[Range]):
    for k, v in param_state.items():
        if isinstance(v, OptimizerStateSpec):
            # If the state is distributed on multiple dp ranks
            # Get my parts
            if range_1d:
                param_state[k] = v.local_tensor.flatten()[range_1d.start : range_1d.end]
            else:
                param_state[k] = v.local_tensor.flatten()
    return param_state


class DistributedOptimizer(OptimizerBase):
    """Distributed optimizer, for all data types (bf16, and fp32).

    The mixed precision happens with following steps:
    1. [init] The model parameters are saved with fp32 copies.
    2. [forward] forward with bf16 data / parameters.
    3. [backward (in DDP)] gradients are accumulated in fp32 (at .main_grad region).
    4. [backward (in DDP)] reduce-scatter on the fp32 gradients.
    5. [step] update sharded fp32 parameters with fp32 gradients with Zero optimization
    6. [step] cast updated and sharded master fp32 parameters into sharded bf16 parameters
    7. [step] all-gather bf16 parameters

    Args:
        optimizer: core optimizer instance or class, such as Adam or SGD
        models: list of DDP models (i.e., the virtual pipelining models).
            This is used by the distributed optimizer for mapping parameters.
        plan: the plan guide the initialization of distribued optimizer,
            for example, whether clip gradeints with this global L2 norm, or
            whether verlaping parameter all gathering with
            forward and so on.

    Example:
        ```python
        # The following program will create a DistributedOptimizer with basic
        # ZeRO features.

        from vescale.ddp import DistributedDataParallel as DDP
        from vescale.optim import DistributedOptimizer
        from vescale.dmodule.api import parallelize_module

        mlp = parallelize_module(MLP(), mesh, ..., ...)
        ddp_module = DDP(
            module=mlp,
            data_pg_or_device_mesh=mesh
        )
        # if you pass a optimizer instance
        optim = torch.optim.Adam(mlp.parameters())
        doptim = DistributedOptimizer(optim, [ddp_model])

        # if you pass a optimizer class
        doptim = DistributedOptimizer(
            torch.optim.Adam,
            [ddp_model],
            optimizer_kwargs={}
        )
        # do the forward and backward
        ddp_model(torch.rand(xxx)).sum().backward()
        # run optimizer `step`
        doptim.step()
        ```
    """

    def __init__(
        self,
        optimizer: Union[torch.optim.Optimizer, Type],
        models: Sequence[DDP],
        clip_grad: float = 0.0,
        overlap_param_gather: bool = False,
        grad_to_fp32: bool = True,
        optimizer_kwargs: Dict[str, Any] = None,
        **kwargs,
    ):
        """
        The steps in this method create the core mapping between DDP grad
        buffers, parameters, and parameter shard ranges, that is needed for
        converting between model param indexes and main parameter shard
        indexes. This method also updates the optimizer parameter groups
        with the newly created shards.
        """

        # make sure that all models are wrapped by DDP
        assert len(models) >= 1, "At least one model must be provided"
        assert all(isinstance(model, DDP) for model in models), "All models must be wrapped by DDP"

        if isinstance(optimizer, type):
            assert (
                optimizer_kwargs is not None
            ), "to build core optimizer in DistributedOptimizer, please provide optimizer_kwargs"
            params = []
            for model in models:
                params.extend(model.module.parameters())
            optim = optimizer(params, **optimizer_kwargs)
            super().__init__(optimizer=optim)
        else:
            super().__init__(optimizer=optimizer)

        # retrive data parallel group info from DDP models.
        self.data_parallel_group = None
        for m in models:
            if self.data_parallel_group is None:
                self.data_parallel_group = m.data_parallel_group
            elif self.data_parallel_group != m.data_parallel_group:
                raise RuntimeError("Detect model chunks of warious data-parallel process groups")

        self.data_parallel_ranks = list(dist.distributed_c10d._pg_group_ranks[self.data_parallel_group].keys())
        self.current_local_rank = dist.get_rank(self.data_parallel_group)
        self.current_global_rank = dist.get_global_rank(self.data_parallel_group, self.current_local_rank)
        self.data_parallel_world_size = dist.get_world_size(self.data_parallel_group)
        self.models = models
        self.clip_grad = clip_grad

        self.overlap_param_gather = overlap_param_gather
        self.grad_to_fp32 = grad_to_fp32

        # model_parameter -> ZeroParamSpec
        self.param_zero_spec: Dict[torch.Tensor, ZeroParamSpec] = {}
        # [model_parameter][global_rank] -> Range
        self.param_across_dp_ranks_info: Dict[torch.Tensor, Dict[int, Range]] = {}
        # [model_index][torch.dtype] -> [param_buffers of a bucket]
        self.param_buffers: List[Dict[torch.dtype, List[torch.Tensor]]] = []

        # build the above three fields
        self._build_param_spec_for_zero()

        # record param metadata for checkpoint
        self.record_param_global_shape()

        # re-build optimizer states
        # [param] -> (global index of param group, local index with param group in this rank)
        self.model_param_group_index_map: Dict[torch.Tensor, Tuple[int, int]] = {}
        self._build_shard_optim_groups()

        # Now construct data structures to manage all-gather handles.
        self.all_gather_handles = []
        self.all_gather_handle_index_to_bucket_index_map = []
        self.model_index_to_all_gather_handle_index_map = {}
        self.param_to_all_gather_handle_index_map = {}
        self.param_buffer_copied = []
        self.removable_pre_hook_handles = []
        self.removable_post_hook_handles = []

        self.pbuf_view_items = self.get_model_param_buffer_dp_views()
        for model_index, dtype, bucket_index, _, _ in self.pbuf_view_items:
            self.all_gather_handle_index_to_bucket_index_map.append((model_index, dtype, bucket_index))
            all_gather_handle_index = len(self.all_gather_handle_index_to_bucket_index_map) - 1

            # Store all all_gather_handle_indices relevant to a particular model chunk.
            if model_index not in self.model_index_to_all_gather_handle_index_map:
                self.model_index_to_all_gather_handle_index_map[model_index] = []
            self.model_index_to_all_gather_handle_index_map[model_index].append(all_gather_handle_index)

            for param in self.models[model_index].grad_buffers[dtype].buckets[bucket_index].params_list:
                self.param_to_all_gather_handle_index_map[param] = all_gather_handle_index
            self.param_buffer_copied.append(False)
        self.num_all_gather_handles = len(self.all_gather_handle_index_to_bucket_index_map)

        if self.overlap_param_gather:
            self._enable_pre_hook()

        # A flag indicates whether the `step` has been called. It will be reset after invoking `zero_grad`.
        self.step_issued = False

    def _build_param_spec_for_zero(self):
        """
        Construct ZeroParamSpec for each parameter in Zero optimization
        """

        # get main param dtype
        parameters = {}
        for m in self.models:
            for param in m.parameters():
                parameters.setdefault(param.dtype, []).append(param)
        if any(dtype not in {torch.bfloat16, torch.float} for dtype in parameters):
            raise NotImplementedError(
                f"we only torch.bfloat16 and torch.float, but got dtypes of {set(parameters.keys())}"
            )
        main_param_dtype = max(parameters, key=lambda dtype: len(parameters[dtype]))

        # record group index of each param in optimizer groups
        global_group_idx: Dict[torch.Tensor, (int, int)] = {}
        for gidx, group in enumerate(self.optimizer.param_groups):
            for local_idx, param in enumerate(group["params"]):
                # print(f'debugging: recording parameters: {self.param_to_name[param]}')
                global_group_idx[param] = (gidx, local_idx)

        for model in self.models:
            for dtype, gbuff in model.grad_buffers.items():
                # .storage() ignores views / slices, so param_buffer now points to the start
                # of the grad_buffer instead of to the start of each bucket. As a result,
                # add bucket.offset to make sure param_buffers point to the right region of
                # memory.
                # Since we want the start of each bucket's param_buffer to coincide with the
                # start of the same bucket's grad_buffer (this ensures that zeroing the grad
                # buffer does not zero out params in the param_buffer before they are copied
                # into the model_params), multiply the offset by the size ratio of grads and
                # params.
                size_ratio = torch.finfo(dtype).bits // torch.finfo(main_param_dtype).bits
                storage = gbuff.data.untyped_storage()
                pbuff = torch.tensor([], dtype=main_param_dtype, device=storage.device).set_(storage)

                # collect parameter buffers for all-gather
                curr_param_buffers = {}
                for bucket in gbuff.buckets:
                    offset = bucket.offset * size_ratio
                    # note the parameter buffer may be shorter than the gradient buffer.
                    # this typically happens in mixed precision training where the
                    # gradient buffer is of torch.float32 but the parameter buffer
                    # only needs for torch.bfloat16
                    param_buffer = pbuff[offset : offset + bucket.data.numel()]
                    assert param_buffer.data_ptr() == bucket.data.data_ptr()
                    assert param_buffer.numel() == bucket.data.numel()
                    curr_param_buffers.setdefault(dtype, []).append(param_buffer)
                self.param_buffers.append(curr_param_buffers)

                # build ZeroParamSpec for each param

                # {param: (data_start_index, data_end_index, bucket_id)}
                for param, indices in gbuff.param_index_map.items():
                    # the offset is the global grad buffer offset
                    bucket = gbuff.param_to_bucket[param]
                    bucket_start = bucket.offset * size_ratio
                    param_start, param_end, _ = indices
                    # param_buff = bucket.data
                    param_start = bucket_start + (param_start - bucket.offset)
                    param_end = bucket_start + (param_end - bucket.offset)
                    param_buff = pbuff[param_start:param_end]
                    numel = bucket.data.numel()
                    if numel % self.data_parallel_world_size != 0:
                        raise RuntimeError(
                            f"Each bucket's buffer size should be divisible by {self.data_parallel_world_size}"
                        )
                    chunk_size = numel // self.data_parallel_world_size

                    # build global parameter split info of all dp ranks
                    curr_rank_gbuf_slice, curr_param_offset = None, None
                    for rank in range(self.data_parallel_world_size):
                        # the local buffer owned by the `rank`
                        buffer_start = bucket_start + rank * chunk_size
                        buffer_end = buffer_start + chunk_size
                        if buffer_end <= param_start or buffer_start >= param_end:
                            continue
                        # the index on global gradient buffer
                        rank_slice = slice(max(param_start, buffer_start), min(param_end, buffer_end))
                        assert (
                            rank_slice.start < rank_slice.stop
                        ), f"{rank_slice} is not a valid slice for param size: {param.size()}"
                        # the index on the parameter
                        param_offset_start = max(0, buffer_start - param_start)
                        param_offset_stop = param_offset_start + rank_slice.stop - rank_slice.start
                        param_offset = slice(param_offset_start, param_offset_stop)
                        # record the slice only when the parameter is (partly) owned by this local dp rank
                        if rank == self.current_local_rank:
                            curr_rank_gbuf_slice = rank_slice
                            curr_param_offset = param_offset
                        # the case that the parameter is split across
                        # multipe dp ranks
                        overlap = max(param_start, buffer_start) < min(param_end, buffer_end)
                        if overlap and (param_start < buffer_start or param_end > buffer_end):
                            global_rank = dist.get_global_rank(self.data_parallel_group, rank)
                            self.param_across_dp_ranks_info.setdefault(param, dict())[global_rank] = Range(
                                param_offset.start, param_offset.stop
                            )

                    # the parameter is (partly) owned by this local dp rank
                    shard_param, shard_main_param, shard_param_buff = None, None, None
                    if curr_rank_gbuf_slice is not None:
                        spec = getattr(param, "_spec", None)
                        local_param = param._local_tensor if isinstance(param, DTensor) else param
                        shard_param = local_param.detach().view(-1)[curr_param_offset]
                        shard_param._spec = spec

                        # note we only need to copy an fp32 master weight
                        # from bf16 in mixed-precision training. If the
                        # param is already in fp32, we don't need to copy
                        if self.grad_to_fp32 and param.dtype == torch.bfloat16:
                            shard_main_param = shard_param.clone().float()
                            shard_main_param._spec = spec

                        # this is used for calculating gnorm in clip_grads.
                        # the copy of shared parameters will be skipped in clip_grads.
                        if hasattr(param, "shared"):
                            shard_param.shared = param.shared
                            if shard_main_param is not None:
                                shard_main_param.shared = param.shared

                        shard_param_buff = pbuff[curr_rank_gbuf_slice]

                    self.param_zero_spec[param] = ZeroParamSpec(
                        ddp_bucket=bucket,
                        # if `shard_param` is None, indicating this rank doesn't
                        # own this parameter.
                        shard_param=shard_param,
                        shard_main_param=shard_main_param,
                        global_gbuf_slice=curr_rank_gbuf_slice,
                        param_slice=curr_param_offset,
                        shard_param_buffer=shard_param_buff,
                        param_buffer=param_buff,
                    )

    def _build_shard_optim_groups(self):
        """Re-build the sharding parameter groups for the optimizer"""

        ngroups = len(self.optimizer.param_groups)
        local_param_fp32_groups = [[] for _ in range(ngroups)]
        local_param_bf16_groups = [[] for _ in range(ngroups)]

        # group index, local index within the group
        param_group_map: Dict[torch.Tensor, Tuple[int, int]] = {}
        for gidx, group in enumerate(self.optimizer.param_groups):
            # filter out the parameters that are owned by other dp ranks
            own_group = [p for p in group["params"] if self.param_zero_spec[p].shard_param is not None]
            # count fp32 param num
            fp32_param_num = len([p for p in own_group if p.dtype == torch.float32])
            fp32_local_group_idx = 0
            bf16_local_group_idx = fp32_param_num  # bf16 params appear after fp32 params
            for param in own_group:
                spec = self.param_zero_spec[param]
                if param.dtype == torch.float32:
                    local_param_fp32_groups[gidx].append(spec.shard_param)
                    # print(f'rank{dist.get_rank()}: updating param {self.param_to_name[param]} with {(gidx, fp32_local_group_idx)}')
                    param_group_map[param] = (gidx, fp32_local_group_idx)
                    fp32_local_group_idx += 1
                elif param.dtype == torch.bfloat16:
                    opt_param = spec.shard_main_param if spec.shard_main_param is not None else spec.shard_param
                    local_param_bf16_groups[gidx].append(opt_param)
                    param_group_map[param] = (gidx, bf16_local_group_idx)
                    # print(f'rank{dist.get_rank()}: updating param {self.param_to_name[param]} with {(gidx, bf16_local_group_idx)}')
                    bf16_local_group_idx += 1
                else:
                    raise NotImplementedError(f"param dtype {param.dtype} not supported")

        # merge the fp32 and bf16 groups by putting fp32 params before bf16 params
        local_param_groups = [
            {"params": fp32s + bf16s} for fp32s, bf16s in zip(local_param_fp32_groups, local_param_bf16_groups)
        ]
        # update optimizer
        for idx, group in enumerate(self.optimizer.param_groups):
            group["params"] = local_param_groups[idx]["params"]

        # leverage state_dict() and load_state_dict() to
        # recast preexisting per-param state tensors.
        self.optimizer.load_state_dict(self.optimizer.state_dict())
        self.model_param_group_index_map = param_group_map

    def build_param_sharding_info_for_checkpoint(self, model: DDP, dtype, gbuf_world_all_ranges):
        param_world_index_map = model.grad_buffer_param_index_map[dtype]
        for param, param_world_indexes in param_world_index_map.items():
            if param not in self.param_shard_info:
                self.param_shard_info[param] = []
            for gbuf_world_range in gbuf_world_all_ranges:
                param_world_start, param_world_end, _ = param_world_indexes
                param_local_start = max(0, param_world_start - gbuf_world_range.start)
                param_local_end = min(gbuf_world_range.size, param_world_end - gbuf_world_range.start)

                # Add param, if within local gbuf range.
                if param_local_end > param_local_start:
                    self.param_shard_info[param].append(param_local_end - param_local_start)
                else:
                    self.param_shard_info[param].append(0)

    def state_dict(self):
        """
        The state dict contains all non-DP-rank-dependent (i.e., non-parameter-
        related) optimizer variables. The returned state dict can be stored in
        the standard model/RNG checkpoint file. The parameter and dependent
        optimizer state (e.g., exp_avg, exp_avg_sq) are stored in a separate
        checkpoint file by calling 'save_parameter_state()'.
        """
        # all gather ddp module
        if self.overlap_param_gather:
            for m in self.models:
                self._param_all_gather(m)
            # we disable all pre_forward hook needed for param sync, and reenable them
            # at the end of subsequent forward.
            self._disable_pre_hook()

        optimizer_state = self.optimizer.state_dict()

        distributed_state = {
            "param_group_meta": optimizer_state["param_groups"],
        }
        self.prefix_sum_param_groups = []
        param_groups = self.optimizer.state_dict()["param_groups"]

        for i, _ in enumerate(param_groups):
            if i == 0:
                self.prefix_sum_param_groups.append(0)
            else:
                self.prefix_sum_param_groups.append(
                    len(param_groups[i - 1]["params"]) + self.prefix_sum_param_groups[i - 1]
                )

        for i, model in enumerate(self.models):
            for param in model.parameters():
                # skip params that is totally owned by other ranks
                spec = self.param_zero_spec.get(param, None)
                if spec is None or spec.shard_param is None:
                    continue
                bucket: Bucket = self.param_zero_spec[param].ddp_bucket
                # will be torch.float32 if in mixed-precision training
                dtype = bucket.data.dtype
                if self.grad_to_fp32:
                    assert dtype == torch.float32
                name = self.param_to_name[param]
                group_id, local_id_in_group = self.model_param_group_index_map[param]
                distributed_state.setdefault(dtype, {})[name] = _convert_dict_with_sharded(
                    optimizer_state["state"][self.prefix_sum_param_groups[group_id] + local_id_in_group],
                    self.param_global_shape_info[param],
                    self.param_local_shape_info[param],
                    self.param_global_offset_info[param],
                    self.param_across_dp_ranks_info.get(param),
                )
        # If it is mix percision training, we should save master fp32 weights
        if self.grad_to_fp32:
            for param, spec in self.param_zero_spec.items():
                if spec.shard_main_param is not None:
                    name = self.param_to_name[param]
                    group_id, local_id_in_group = self.model_param_group_index_map[param]
                    distributed_state[torch.float32][name]["shard_casted_float16_groups"] = OptimizerStateSpec(
                        self.param_global_shape_info[param],
                        self.param_local_shape_info[param],
                        self.param_global_offset_info[param],
                        spec.shard_main_param,
                        self.param_across_dp_ranks_info.get(param),
                    )

        return distributed_state

    def load_state_dict(self, state_dict):
        """
        Load the state dict from a full state dict
        """
        optimizer_state = {"param_groups": state_dict["param_group_meta"]}
        original_optimizer_state = self.optimizer.state_dict()
        # update params
        for i, param_group in enumerate(optimizer_state["param_groups"]):
            # Just assign param indices, assign param directly leading to deepcopy error
            if len(param_group["params"]) != len(original_optimizer_state["param_groups"][i]["params"]):
                param_group["params"] = original_optimizer_state["param_groups"][i]["params"]
        # resume optimizer state:
        optimizer_state["state"] = {}

        for model in self.models:
            for param in model.parameters():
                # skip params that is totally owned by other ranks
                spec = self.param_zero_spec.get(param, None)
                if spec is None or spec.shard_param is None:
                    continue
                local_slice = self.param_across_dp_ranks_info.get(param, None)
                # convert to range
                if local_slice is not None:
                    local_slice = local_slice[self.current_global_rank]
                group_id, local_id_in_group = self.model_param_group_index_map[param]
                dtype = spec.ddp_bucket.data.dtype
                name = self.param_to_name[param]
                optimizer_state["state"][self.prefix_sum_param_groups[group_id] + local_id_in_group] = (
                    _convert_dict_sharded_to_tensor(state_dict[dtype][name], local_slice)
                )

        self.optimizer.load_state_dict(optimizer_state)

        # load fp32 master parameters
        if self.grad_to_fp32:
            for param, spec in self.param_zero_spec.items():
                if spec.shard_main_param is not None:
                    name = self.param_to_name[param]
                    group_id, local_id_in_group = self.model_param_group_index_map[param]
                    spec.shard_main_param.copy_(state_dict[torch.float32][name]["shard_casted_float16_groups"])

    def zero_grad(self, set_to_none=True):
        """
        Zero grads.

        We only need to zero the model related parameters, i.e.,
        model_float16_groups & model_fp32_groups. We additionally zero
        the remaining groups as a memory optimization to reduce
        fragmentation; in the case of set_to_none==True, the space
        used by this field can be safely deallocated at this point.
        """

        params = []
        for param, spec in self.param_zero_spec.items():
            params.append(param)
            if spec.shard_param is not None:
                params.append(spec.shard_param)
            if spec.shard_main_param is not None:
                params.append(spec.shard_main_param)
        zero_grad_group_helper(params, set_to_none)

        for m in self.models:
            # We can not zero grad buffer when overlap_param_buffer is True.
            # Because param gathering may be not finished and
            # updated param buffer is not copied to model param.
            m.zero_grad_buffer(zero_buffer=(not self.overlap_param_gather))

        # If overlapping param all-gather with forward compute, launch all-gather
        # for first accessed bucket here before forward compute is initiated.
        # The all-gather for the next bucket will be launched in the forward
        # pre-hook when this all-gather finishes (to ensure that the communication
        # kernels don't head-of-line block the compute kernels since we run with
        # CUDA_DEVICE_MAX_CONNECTIONS=1 to support sequence parallelism).
        # NOTE: we shouldn't issue param all-gather if runned before any `optim.step`.
        if self.overlap_param_gather and self.step_issued:
            self._dispatch_gather_model_params(all_gather_handle_index=0)

        self.step_issued = False

    def get_model_param_buffer_dp_views(self):
        """
        Get shard views of each of the param buffers.

        In this nested list, the top level is grouped by the virtual model
        index and the buffer's data type. The sub-level is a list of
        shards of that buffer, where each shard in the list represents
        a contiguous view of the buffer, that is owned by a data-parallel
        rank. The shard boundary does not respect parameter boundaries, and
        so the elements of some parameters are split across data parallel
        ranks.

        Additionally, return references to the entire buffers, for use
        in _all_gather_base.
        """

        # Buffer views.
        # Add in reverse order in each model chunk since buckets start from the end of the model but we want
        # all-gathers to run first for the start of the model (same order as forward pass).
        # We keep the view_items in model chunk order since we want to still first run all_gather and
        # all_gather_handle.wait() for the first model chunk.
        # In all cases, we want all_gather and all_gather_handle.wait() to be called in the same order,
        # and all_gather_handle.wait() needs to be called just before the corresponding forward pass.
        view_items = []
        for model_index, buffers in enumerate(self.param_buffers):
            view_items_per_model_chunk = []
            for dtype, buf_for_all_buckets in buffers.items():
                for bucket_index, buf in enumerate(buf_for_all_buckets):
                    # shard_buffer_among_dp_world
                    assert buf.numel() % self.data_parallel_world_size == 0
                    shard_size = buf.numel() // self.data_parallel_world_size
                    buf_views = [
                        buf[(r * shard_size) : ((r + 1) * shard_size)] for r in range(self.data_parallel_world_size)
                    ]
                    view_items_per_model_chunk.insert(0, (model_index, dtype, bucket_index, buf, buf_views))
            view_items.extend(view_items_per_model_chunk)

        return view_items

    def _dispatch_gather_model_params(self, all_gather_handle_index):
        """
        All-gather updated model params.

        The DDP's param buffer is used for the all-gather, and thus no
        tensors are dynamically allocated. After the all-gather, the params
        can be copied from the param buffer to the param.
        """
        data_parallel_rank = self.current_local_rank
        data_parallel_group = self.data_parallel_group

        # All-gather updated main params.
        # All param_buf views are guaranteed to have the same number of elements
        # across all data-parallel ranks, due to padding (done in grad_buffer.py),
        # and extended to the param_bufs. Thus, all sub-views will have consistent
        # start / end indexes across data-parallel ranks.
        (model_index, dtype, bucket_index, pbuf, pbuf_views) = self.pbuf_view_items[all_gather_handle_index]
        assert all_gather_handle_index == len(self.all_gather_handles)
        all_gather_handle = torch.distributed.all_gather_into_tensor(
            pbuf,
            pbuf_views[data_parallel_rank],
            group=data_parallel_group,
            async_op=True,
        )

        self.all_gather_handles.append(all_gather_handle)
        assert self.all_gather_handle_index_to_bucket_index_map[all_gather_handle_index] == (
            model_index,
            dtype,
            bucket_index,
        )
        self.param_buffer_copied.append(False)

        if not self.overlap_param_gather:
            all_gather_handle.wait()
            self._copy_params_from_param_buffer(all_gather_handle_index)

    def _enable_pre_hook(self):
        for m in self.models:
            m = m.module
            for subm in m.modules():
                self.removable_pre_hook_handles.append(subm.register_forward_pre_hook(self._make_forward_pre_hook()))

    def _disable_pre_hook(self):
        for handle in self.removable_pre_hook_handles:
            handle.remove()
        self.removable_pre_hook_handles = []
        for m in self.models:
            self.removable_post_hook_handles.append(m.register_forward_hook(self._restart_pre_hook()))

    def _restart_pre_hook(self):
        def hook(*args, **kwargs):
            self._enable_pre_hook()
            for handle in self.removable_post_hook_handles:
                handle.remove()
            self.removable_post_hook_handles = []

        return hook

    def _param_all_gather(self, module):
        # Make sure all parameters in this module have been all-gathered as necessary.
        for param in module.parameters(recurse=False):
            # Skip parameters that don't require grad.
            if not param.requires_grad:
                continue
            try:  # TODO: avoid try-except
                all_gather_handle_index = self.param_to_all_gather_handle_index_map[param]
                self._finish_param_sync_helper(all_gather_handle_index)
            except KeyError:
                continue

    def _make_forward_pre_hook(self):
        """
        Create a forward pre-hook to wait on all-gather handles when necessary (i.e.,
        when a module uses a parameter in a bucket with a still incomplete all-gather)
        and then copy the results from the param_buffer into model_params.
        """

        def hook(module, *unused):
            assert self.overlap_param_gather, "Should use pre-hook only when overlap_param_gather is True"
            self._param_all_gather(module)

        return hook

    def finish_param_sync(self, model_index, *unused):
        """
        Finishes all necessary param syncs for the model_index'th model chunk.
        """
        all_gather_handle_indices = self.model_index_to_all_gather_handle_index_map[model_index]
        for all_gather_handle_index in all_gather_handle_indices:
            self._finish_param_sync_helper(all_gather_handle_index)

    def _finish_param_sync_helper(self, all_gather_handle_index):
        """
        Waits on all_gather_handle if necessary, then copies params from param_buffer
        into model_params if necessary.
        """

        # First check if there is an outstanding all-gather handle for this param.
        # If so, wait on the handle to ensure the communication is finished.
        if all_gather_handle_index >= len(self.all_gather_handles):
            return

        all_gather_handle = self.all_gather_handles[all_gather_handle_index]
        if all_gather_handle is not None:
            all_gather_handle.wait()
            self.all_gather_handles[all_gather_handle_index] = None

            # Launch the all-gather for the next bucket now.
            # We can't pre-launch all-gathers for all buckets at once since we don't
            # want to head-of-line block the compute kernels with communication kernels
            # (since we run with CUDA_DEVICE_MAX_CONNECTIONS=1 to support sequence
            # parallelism).
            next_all_gather_handle_index = all_gather_handle_index + 1
            if next_all_gather_handle_index < self.num_all_gather_handles:
                self._dispatch_gather_model_params(next_all_gather_handle_index)

        # Also check if we have already copied from the param buffer for this
        # handle; if not, complete the copy and mark as such.
        if not self.param_buffer_copied[all_gather_handle_index]:
            self._copy_params_from_param_buffer(all_gather_handle_index)
            self.param_buffer_copied[all_gather_handle_index] = True

    def record_param_global_shape(self):
        # Model parameter sharding info for vescale.checkpoint
        self.param_to_name = {}
        self.param_shard_info = {}
        self.param_global_shape_info = {}
        self.param_local_shape_info = {}
        self.param_global_offset_info = {}
        for i, model in enumerate(self.models):
            module = model.module
            if isinstance(module, torch.fx.GraphModule) and getattr(module, "id_mapping", None):
                # reassign global id mapping to params in case they are erased by DTensor init
                graph_id_mapping = module.id_mapping
                for sub_node in module.graph.nodes:
                    if sub_node.op == "call_module":
                        stage_submodule = module.get_submodule(sub_node.target)
                        for name, param in stage_submodule.named_parameters():
                            _param = param._local_tensor if isinstance(param, DTensor) else param
                            _param.global_id = graph_id_mapping.get(sub_node.name + "." + name, -1)

            for name, param in module.named_parameters():
                _param = param._local_tensor if isinstance(param, DTensor) else param
                if hasattr(_param, "global_id"):
                    self.param_to_name[param] = (_param.global_id, name)
                else:
                    self.param_to_name[param] = (i, name)

                # this indicates no tensor parallelism is applied to the module,
                # then the parameter is not partitioned across devices
                if not hasattr(param, "_spec"):
                    global_shape = param.size()
                    local_shape = tuple(global_shape)
                    global_offset = (0,) * len(local_shape)
                else:
                    global_shape = param._spec.tensor_meta.shape
                    local_shape, global_offset = compute_local_shape_and_global_offset(
                        global_shape, param._spec.mesh, param._spec.placements
                    )
                self.param_global_shape_info[param] = global_shape
                self.param_local_shape_info[param] = local_shape
                self.param_global_offset_info[param] = global_offset

    def _copy_params_from_param_buffer(self, all_gather_handle_index):
        """
        Copy params from param_buffer to model_params.
        This happens after all-gather
        """

        (
            model_index,
            dtype,
            bucket_index,
        ) = self.all_gather_handle_index_to_bucket_index_map[all_gather_handle_index]
        model = self.models[model_index]

        # Copy from param buffer to each param.
        param_map = model.grad_buffer_param_index_map[dtype]
        for param, (
            buf_start,
            buf_end,
            bucket_index_in_param_map,
        ) in param_map.items():
            if bucket_index == bucket_index_in_param_map:
                buffer = self.param_zero_spec[param].param_buffer
                local_param = param._local_tensor if isinstance(param, DTensor) else param
                local_param.view(-1).detach().copy_(buffer)

        # Zero out the grad buffer in preparation for next set of fwd / bwd passes after copy
        # completes (since param_buffer and grad_buffer are shared for each bucket).
        param_buf = self.param_buffers[model_index][dtype][bucket_index]
        grad_buf = model.grad_buffers[dtype].buckets[bucket_index].data
        assert param_buf.data_ptr() == grad_buf.data_ptr()
        grad_buf.zero_()

    def _collect_main_grad_data_for_unscaling(self):
        """
        Note: this should be equivalent to the float-16 optimizer's method,
        but writtent differently, so the two should be combined.
        """
        return [
            param.grad.data if not isinstance(param.grad.data, DTensor) else param.grad.data._local_tensor
            for group in self.optimizer.param_groups
            for param in group["params"]
        ]

    def _copy_model_grads_to_main_grads(self):
        """
        Copy model grads to main grads.

        This happens before the self.optimizer.step()

        Since this step follows a reduce-scatter through the DDP's grad
        buffer, this method is responsible for copying the updated grads
        from the grad buffer to the main shard's grad field.
        """
        for param, spec in self.param_zero_spec.items():
            if spec.shard_param is not None:
                param_grad = param.main_grad
                shard_grad = param_grad.view(-1)[spec.param_slice]
                # for mixed-precision bf16 weights
                if spec.shard_main_param is not None:
                    spec.shard_main_param.grad = shard_grad
                # for fp32 weights or non-mixed precision mode
                else:
                    spec.shard_param.grad = shard_grad

    def _copy_main_params_to_model_params(self):
        """
        Copy main params to model params.

        This happens after optimizer.step and before the all-gather

        Since this step is followed by an all-gather through the DDP's grad
        buffer, this method is responsible for copying the updated params
        from the main shards into the correct position in the grad buffer.
        """
        for spec in self.param_zero_spec.values():
            if spec.shard_param is not None:
                updated_param = spec.shard_param
                if spec.shard_main_param is not None:
                    updated_param = spec.shard_main_param
                spec.shard_param_buffer.data.copy_(updated_param)

    def clip_grad_norm(self, clip_grad):
        params = self.get_parameters()
        grads_for_norm = self.get_main_grads_for_grad_norm()
        return clip_grad_norm_fp32(
            params,
            grads_for_norm,
            clip_grad,
            # for accumulate grad norm, use the whole world.
            model_parallel_group=None,
        )

    @torch.no_grad()
    def step(self):
        # Wait for reduce-scatter on main gradient to finish
        for model in self.models:
            model.finish_grad_sync()

        # Copy gradients from model params to main params.
        self._copy_model_grads_to_main_grads()

        # Clip the main gradients.
        grad_norm = None
        if self.clip_grad > 0.0:
            grad_norm = self.clip_grad_norm(self.clip_grad)

        # Step the optimizer.
        self.optimizer.step()

        # Update params from main params.
        self._copy_main_params_to_model_params()

        # Reset metadata needed to track results of all-gathers.
        self.all_gather_handles = []
        self.param_buffer_copied = []

        # If not overlapping all-gather for parameters, launch synchronous all-gather
        # communication calls here.
        if not self.overlap_param_gather:
            for all_gather_handle_index in range(self.num_all_gather_handles):
                self._dispatch_gather_model_params(all_gather_handle_index)

        self.step_issued = True
        return grad_norm

    def get_parameters(self):
        params = []
        for param_group in self.optimizer.param_groups:
            for param in param_group["params"]:
                params.append(param)
        return params

    def get_main_grads_for_grad_norm(self):
        # Filter parameters based on:
        #   - grad should not be none
        #   - parameter should not be shared
        #   - should not be a replica due to tensor model parallelism
        params = self.get_parameters()
        grads_for_norm = []
        for param in params:
            grad = param.grad
            grad_not_none = grad is not None
            is_not_shared = not param_is_shared(param)
            is_not_tp_duplicate = True
            # skip checking duplication for pure DP. FIXME: handle PP case
            if self.data_parallel_world_size < dist.get_world_size():
                is_not_tp_duplicate = param_is_sharded_or_replicate_on_first_rank(param)
            if grad_not_none and is_not_shared and is_not_tp_duplicate:
                grads_for_norm.append(grad._local_tensor if isinstance(grad, DTensor) else grad)

        return grads_for_norm
