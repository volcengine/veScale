################################################################################
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

"""Megatron distributed optimizer."""

import math
import inspect
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple, Optional, Any

import torch
import torch.distributed as dist

from vescale.dtensor.dtensor import DTensor
from vescale.ddp.distributed_data_parallel import DistributedDataParallel as DDP
from vescale.optim.base_optimizer import OptimizerBase
from vescale.dtensor._utils import compute_local_shape_and_global_offset

from vescale.optim.utils import param_is_shared, param_is_sharded_or_replicate_on_first_rank, _zero_grad_group_helper
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


def convert_dict_with_sharded(
    param_state: dict,
    global_shape: Tuple[int],
    local_shape: Tuple[int],
    global_offset: Tuple[int],
    dp_ranks_ranges: Optional[Dict[int, Range]],
):
    new_param_state = {}
    for k, v in param_state.items():
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


def convert_dict_sharded_to_tensor(param_state: dict, range_1d: Optional[Range]):
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

    Args:
        optimizer: base optimizer instance or class, such as Adam or SGD
        models: list of DDP models (i.e., the virtual pipelining models).
            This is used by the distributed optimizer for mapping parameters.
        clip_grad: clip gradeints with this global L2 norm. Note
            that clipping is ignored if clip_grad == 0.
        overlap_param_gather: whether overlaping parameter all gathering with
            forward. By default, False.
        grad_to_fp32: whether casting both the gradients and the optimizer
            states to fp32. By default, True.
        optimizer_kwargs: used to initialize base optimizer instance when class
            is provided for `optimizer` argument. By default, None.

    Example:
        ```python
        # The following program will create a DistributedOptimizer with basic
        # ZeRO features.

        from vescale.ddp.distributed_data_parallel import DistributedDataParallel as DDP
        from vescale.optim.distributed_optimizer import DistributedOptimizer
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
        optimizer,
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
            ), "to build basic optimizer in DistributedOptimizer, please provide optimizer_kwargs"
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
        if not all(x.use_distributed_optimizer for x in models):
            raise ValueError("Please open `use_distributed_optimizer` in DDP initialization for better performance.")

        param_dtype_cnt = {}
        main_param_dtype_cnt = 0
        for m in models:
            for name, param in m.named_parameters():
                dtype = param.dtype
                param_dtype_cnt[dtype] = param_dtype_cnt.get(dtype, 0) + 1
                if param_dtype_cnt[dtype] > main_param_dtype_cnt:
                    main_param_dtype = dtype
        assert main_param_dtype in [
            torch.bfloat16,
            torch.float,
        ], "currently, we only support bf16 of float32 parameters"

        self.main_param_dtype = main_param_dtype

        self.data_parallel_ranks = list(dist.distributed_c10d._pg_group_ranks[self.data_parallel_group].keys())
        self.current_local_rank = dist.get_rank(self.data_parallel_group)
        self.current_global_rank = dist.get_global_rank(self.data_parallel_group, self.current_local_rank)
        self.data_parallel_world_size = dist.get_world_size(self.data_parallel_group)
        self.models = models
        self.clip_grad = clip_grad

        self.overlap_param_gather = overlap_param_gather
        self.grad_to_fp32 = grad_to_fp32

        # Model parameter sharding info for vescale.checkpoint
        self.param_to_name = {}
        self.param_shard_info = {}
        self.param_global_shape_info = {}
        self.param_local_shape_info = {}
        self.param_global_offset_info = {}
        for i, model in enumerate(self.models):
            for name, param in model.module.named_parameters():
                # Each parameter should have unique id for distributed checkpointing and pp re-sharding
                if hasattr(model.module, "global_id"):
                    self.param_to_name[param] = (model.module.global_id, name)
                else:
                    self.param_to_name[param] = (i, name)
                # Save all necessary tensor related meta-data for distributed checkpointing
                global_shape = param.shape
                local_shape, global_offset = compute_local_shape_and_global_offset(
                    global_shape, param.device_mesh, param.placements
                )
                self.param_global_shape_info[param] = global_shape
                self.param_local_shape_info[param] = local_shape
                self.param_global_offset_info[param] = global_offset

        # Model grad buffer ranges.
        self.model_gbuf_ranges = []
        self.per_bucket_numel = []
        self.param_across_dp_ranks_info = {}
        # Mapping fp32 master weights and original fp16 weights
        # for mix percision training
        self.param_to_origin_param_for_shard_casted_float16_groups = {}

        for model_chunk in self.models:
            self.per_bucket_numel.append(
                {
                    dtype: [bucket.data.numel() for bucket in model_chunk.grad_buffers[dtype].buckets]
                    for dtype in model_chunk.grad_buffers
                }
            )
            self.model_gbuf_ranges.append(self.build_model_gbuf_range_map(model_chunk))
        self.model_param_gbuf_map = self.build_model_param_gbuf_map(self.model_gbuf_ranges)

        # Optimizer ranges.
        (
            self.model_param_group_index_map,
            self.opt_group_ranges,
        ) = self.build_optimizer_group_ranges(self.optimizer.param_groups, self.model_gbuf_ranges)

        # Allocate main param shards.
        (
            self.model_float16_groups,
            self.model_fp32_groups,
            self.shard_float16_groups,
            self.shard_fp32_groups,
            self.shard_casted_float16_groups,
        ) = self.build_model_and_main_param_groups(
            self.model_gbuf_ranges, self.model_param_gbuf_map, self.opt_group_ranges
        )

        # Initialize param buffers.
        # - These are views on the DDP model's grad buffers, that share
        #   storage & have their own dtype. This is safe because the param
        #   dtype size is always <= grad dtype size.
        self.param_buffers = []
        for model_index, model in enumerate(self.models):
            current_param_buffers = {}
            for dtype, grad_buffer in model.grad_buffers.items():
                size_ratio = torch.finfo(dtype).bits // torch.finfo(self.main_param_dtype).bits
                current_param_buffers[dtype] = []
                for bucket in grad_buffer.buckets:
                    # Handle older/newer method for getting untyped storage.
                    try:
                        storage = bucket.data.untyped_storage()
                    except Exception:
                        try:
                            storage = bucket.data.storage()._untyped()
                        except Exception:
                            storage = bucket.data.storage().untyped()

                    # use tensor._set(storage) to avoid unnecessary memory allocation.
                    param_buffer = torch.tensor([], dtype=self.main_param_dtype, device=storage.device).set_(storage)

                    # .storage() ignores views / slices, so param_buffer now points to the start
                    # of the grad_buffer instead of to the start of each bucket. As a result,
                    # add bucket.offset to make sure param_buffers point to the right region of
                    # memory.
                    # Since we want the start of each bucket's param_buffer to coincide with the
                    # start of the same bucket's grad_buffer (this ensures that zeroing the grad
                    # buffer does not zero out params in the param_buffer before they are copied
                    # into the model_params), multiply the offset by the size ratio of grads and
                    # params.
                    offset = bucket.offset * size_ratio
                    param_buffer = param_buffer[offset : offset + bucket.data.numel()]
                    assert (
                        param_buffer.data_ptr() == bucket.data.data_ptr()
                    ), "param_buffer and grad_buffer for same bucket should start at the same byte address"
                    assert (
                        param_buffer.numel() == bucket.data.numel()
                    ), "param_buffer and grad_buffer for same bucket should have the same number of elements"
                    current_param_buffers[dtype].append(param_buffer)
            self.param_buffers.append(current_param_buffers)

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

        # Update optimizer groups.
        # - Also, leverage state_dict() and load_state_dict() to
        #   recast preexisting per-param state tensors.
        self.optimizer.param_groups = [g["orig_group"] for g in self.opt_group_ranges]
        self.optimizer.load_state_dict(self.optimizer.state_dict())

        # A flag indicates whether the `step` has been called. It will be reset after invoking `zero_grad`.
        self.step_issued = False

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

    def build_model_gbuf_param_range_map(self, model: DDP, dtype, gbuf_world_range, bucket_offset):
        """
        Build mapping from param reference to grad buffer shard ranges.

        This method builds a mapping from parameter references to grad
        buffer shard ranges, specific to each data-parallel (DP) rank's
        set of 'owned' parameters. Each grad buffer (padded to be an even
        multiple of DP-world-size) is conceptually divided into DP-world-size
        contiguous regions, where each DP rank 'owns' a contiguous regions.
        Ownership in this sense means DP rank is responsible for reducing
        the relevant subset of grads, and updating the relevant subset of
        params.

        This conceptual partitioning of the grad buffer does NOT respect
        parameter boundaries, and as such it is assumed that each created
        range references a shard (or subset) of the full parameter. It is
        easiest to think of each DP rank as operating (i.e., reducing,
        gathering) purely on views into the grad buffer, for all model-to-
        main & main-to-model operations.

        This method creates four ranges:
        - The param's range within the entire grad buffer (i.e., world index).
        - The param's range within the relevant grad bucket's buffer.
        - The param's range within the DP rank's local view of the grad buffer.
        - The param's range within itself (i.e., its shard).
        """

        # Param range map.
        param_world_index_map = model.grad_buffer_param_index_map[dtype]
        param_range_map = {}
        for param, param_world_indexes in param_world_index_map.items():
            # Param range.
            param_world_start, param_world_end, _ = param_world_indexes
            param_local_start = max(0, param_world_start - gbuf_world_range.start)
            param_local_end = min(gbuf_world_range.size, param_world_end - gbuf_world_range.start)

            # Add param, if within local gbuf range.
            if param_local_end > param_local_start:
                param_local_range = Range(param_local_start, param_local_end)
                param_world_range = param_local_range.normalize(param_local_start + gbuf_world_range.start)
                param_world_range_in_bucket = Range(
                    param_world_range.start - bucket_offset,
                    param_world_range.end - bucket_offset,
                )
                sub_param_start = max(0, gbuf_world_range.start - param_world_start)
                sub_param_range = param_local_range.normalize(sub_param_start)
                param_range_map[param] = {
                    "gbuf_world": param_world_range,
                    "gbuf_world_in_bucket": param_world_range_in_bucket,
                    "gbuf_local": param_local_range,
                    "param": sub_param_range,
                }

        return param_range_map

    def build_model_gbuf_range(self, model, dtype, bucket_index):
        """
        Build mapping between params and their grad buffers.

        This method does the initial setup for the method above. This setup
        includes determining the shard ranges into the DDP's grad buffer for
        each data-parallel (DP) rank. Each DP rank keeps range info for
        all other DP ranks, for the purpose of creating args for
        reduce-scatter and all-gather.
        """

        data_parallel_rank = self.current_local_rank
        data_parallel_world_size = self.data_parallel_world_size

        bucket = model.grad_buffers[dtype].buckets[bucket_index]
        bucket_buffer = bucket.data
        gbuf_size = bucket_buffer.numel()
        assert (
            gbuf_size % data_parallel_world_size == 0
        ), f"Each bucket's buffer size should be divisible by {data_parallel_world_size}"
        max_gbuf_range_size = gbuf_size // data_parallel_world_size

        # All world ranges (i.e., across all data parallel ranks).
        gbuf_world_all_ranges = []
        for r in range(data_parallel_world_size):
            # Compute start of chunk in this bucket.
            gbuf_world_start = r * max_gbuf_range_size
            gbuf_world_end = min(gbuf_size, gbuf_world_start + max_gbuf_range_size)
            # Add bucket's offset in grad buffer.
            gbuf_world_range = Range(gbuf_world_start + bucket.offset, gbuf_world_end + bucket.offset)
            gbuf_world_all_ranges.append(gbuf_world_range)

        # Local DP's ranges.
        gbuf_world_range = gbuf_world_all_ranges[data_parallel_rank]

        # Get parameter sharding info of all ranks, for checkpointing.
        self.build_param_sharding_info_for_checkpoint(model, dtype, gbuf_world_all_ranges)

        # Get each param's ranges.
        param_range_map = self.build_model_gbuf_param_range_map(model, dtype, gbuf_world_range, bucket.offset)

        # For state which shared across dp ranks
        # Compute all ranks own those tensors and get there corresponding 1D ranges for TP resharding
        for r in range(data_parallel_world_size):
            # Get ranges for each dp rank
            temp_range_map = self.build_model_gbuf_param_range_map(
                model, dtype, gbuf_world_all_ranges[r], bucket.offset
            )
            # Get global rank for each dp rank
            rank = dist.get_global_rank(self.data_parallel_group, r)
            for param, ranges in temp_range_map.items():
                range_1d = ranges["param"]
                local_shape = self.param_local_shape_info[param]
                if len(range_1d) != math.prod(local_shape):
                    if param not in self.param_across_dp_ranks_info:
                        self.param_across_dp_ranks_info[param] = {}
                    self.param_across_dp_ranks_info[param][rank] = range_1d
        # Group into dict.
        data = {
            "param_map": param_range_map,
        }

        return data

    def build_model_gbuf_range_map(self, model):
        """
        Create param-to-grad-buffer mappings, for grad buffer data types
        within a specific virtual model.
        """
        # Iterate through all buckets to construct param ranges that this rank "owns"
        # (the dp_rank'th shard of each bucket, where each shard is 1/dp_world_size
        # of the bucket).
        return {
            dtype: [
                self.build_model_gbuf_range(model, dtype, bucket_index)
                for bucket_index in range(len(model.grad_buffers[dtype].buckets))
            ]
            for dtype in model.grad_buffers
        }

    def build_model_param_gbuf_map(self, model_gbuf_ranges):
        """
        Create a reverse of the model_gbuf_ranges, for referencing in
        opposite direction.
        """
        param_gbuf_map = {}
        for model_index, model_gbuf_range_map in enumerate(model_gbuf_ranges):
            for dtype, gbuf_range_map_for_all_buckets in model_gbuf_range_map.items():
                for bucket_index, gbuf_range_map in enumerate(gbuf_range_map_for_all_buckets):
                    for param in gbuf_range_map["param_map"].keys():
                        assert (
                            param not in param_gbuf_map
                        ), "Param should not be in param_gbuf_map; each param only belongs to a single bucket"
                        param_gbuf_map[param] = (model_index, dtype, bucket_index)
        return param_gbuf_map

    def build_optimizer_group_ranges(self, param_groups, model_gbuf_ranges):
        """
        Create optimizer groups.

        Given the set of parameter shard ranges that are owned by the current
        data-parallel (DP) rank, gather the set of parameters that will be
        used (in the method below) to create the current DP's optimizer
        groups.
        """

        num_groups = len(param_groups)

        # Param group map.
        # World param group map.
        # - Store a mapping of <model_parameter:group_index> for all parameters
        #   across all DP ranks. This is necessary because it is our first
        #   cross reference between the DDP mappings and the optimizer group
        #   parameters. This mapping only for use in the next step of building
        #   the local mapping over this DP rank's parameters.
        world_param_group_map = {}
        for group_index, group in enumerate(param_groups):
            for param in group["params"]:
                assert param.requires_grad
                world_param_group_map[param] = group_index

        # Optimizer group ranges & param-group mapping.
        # - Build a mapping from groups to their contained parameters, and also
        #   from parameters to their containing group index and order within
        #   the group. The group index and order are particularly important for
        #   saving and loading checkpoints.
        local_param_group_map = {}
        group_ranges = [{"params": []} for _ in param_groups]
        for model_gbuf_range_map in model_gbuf_ranges:
            for gbuf_range_map_for_all_buckets in model_gbuf_range_map.values():
                for gbuf_range_map in gbuf_range_map_for_all_buckets:
                    for param in gbuf_range_map["param_map"]:
                        group_index = world_param_group_map[param]
                        group_range = group_ranges[group_index]
                        group_range["params"].append(param)
                        local_param_group_map[param] = (
                            group_index,
                            len(group_range["params"]) - 1,
                        )

        # Squeeze zero-size group ranges.
        for group_index, group_range in enumerate(group_ranges):
            group_range["orig_group"] = param_groups[group_index]
            group_range["orig_group_idx"] = param_groups[group_index]

        return local_param_group_map, group_ranges

    def build_model_and_main_param_groups(self, model_gbuf_ranges, param_gbuf_map, opt_group_ranges):
        """
        Create main parameter groups needed for the optimizer step.

        These groups encompass both: 1) groups used by this class, for
        reducing/gather, and 2) groups used by the inner optimizer for the
        parameter update. Given that the conceptual grad buffer partitioning
        (created in earlier method) doesn't respect parameter boundaries,
        the optimizer operates on shards of the model parameters, rather than
        the full parameters.
        """

        # Parameter groups:
        #   model_float16_groups: original float16 parameters
        #   model_fp32_groups: original fp32 parameters
        #   shard_float16_groups: shards of original float16 parameters
        #   shard_fp32_groups: shards of original fp32 parameters
        #   shard_casted_float16_groups: fp32 copy of float16 parameters if cast_shard_param_to_fp32, otherwise, they are just copies
        model_float16_groups = []
        model_fp32_groups = []
        shard_float16_groups = []
        shard_fp32_groups = []
        shard_casted_float16_groups = []

        # Allocate (or slice) each group's param shard.
        for group_index, group_range in enumerate(opt_group_ranges):
            # Params of this group.
            model_float16_params_this_group = []
            model_fp32_params_this_group = []
            shard_float16_params_this_group = []
            shard_fp32_params_this_group = []
            shard_casted_float16_params_this_group = []
            model_float16_groups.append(model_float16_params_this_group)
            model_fp32_groups.append(model_fp32_params_this_group)
            shard_float16_groups.append(shard_float16_params_this_group)
            shard_fp32_groups.append(shard_fp32_params_this_group)
            shard_casted_float16_groups.append(shard_casted_float16_params_this_group)

            for model_param in group_range["params"]:
                assert model_param.requires_grad

                model_index, dtype, bucket_index = param_gbuf_map[model_param]
                gbuf_range = model_gbuf_ranges[model_index][dtype][bucket_index]
                param_range = gbuf_range["param_map"][model_param]["param"]

                # fp16, bf16 params.
                if model_param.type() in [
                    "torch.cuda.HalfTensor",
                    "torch.cuda.BFloat16Tensor",
                ]:
                    # Clone model -> main.
                    model_param_tensor = (
                        model_param if not isinstance(model_param, DTensor) else model_param._local_tensor
                    )
                    shard_model_param = model_param_tensor.detach().view(-1)[param_range.start : param_range.end]
                    if self.grad_to_fp32:
                        shard_main_param = shard_model_param.clone().float()
                    else:
                        shard_main_param = shard_model_param.clone()
                    # copy sharded info from DTensor
                    shard_model_param._spec = None if not isinstance(model_param, DTensor) else model_param._spec
                    if hasattr(model_param, "shared"):
                        shard_model_param.shared = model_param.shared
                        shard_main_param.shared = model_param.shared

                    # copy sharded info from DTensor
                    shard_model_param._spec = None if not isinstance(model_param, DTensor) else model_param._spec
                    shard_main_param._spec = None if not isinstance(model_param, DTensor) else model_param._spec

                    # Add to group.
                    model_float16_params_this_group.append(model_param)
                    shard_float16_params_this_group.append(shard_model_param)
                    shard_casted_float16_params_this_group.append(shard_main_param)
                    self.param_to_origin_param_for_shard_casted_float16_groups[shard_main_param] = model_param
                # fp32 params.
                elif model_param.type() == "torch.cuda.FloatTensor":
                    model_param_tensor = (
                        model_param if not isinstance(model_param, DTensor) else model_param._local_tensor
                    )
                    shard_model_param = model_param_tensor.view(-1)[param_range.start : param_range.end]
                    model_fp32_params_this_group.append(model_param)
                    shard_fp32_params_this_group.append(shard_model_param)
                    if hasattr(model_param, "shared"):
                        shard_model_param.shared = model_param.shared

                    # copy sharded info from DTensor
                    shard_model_param._spec = None if not isinstance(model_param, DTensor) else model_param._spec

                else:
                    raise TypeError(
                        "Wrapped parameters must be one of "
                        "torch.cuda.FloatTensor,  "
                        "torch.cuda.HalfTensor, or "
                        "torch.cuda.BFloat16Tensor. "
                        f"Received {model_param.type()}"
                    )

            # Update optimizer's params.
            # NOTE: group_range["orig_group"] is mapped to param_groups[group_idx]
            # changing group_range will implicitly change self.optimzer.param_groups.
            group_range["orig_group"]["params"] = [
                *shard_fp32_params_this_group,
                *shard_casted_float16_params_this_group,
            ]
            # update the param group map because group_range changes
            fp32_param_num = len(model_fp32_params_this_group)
            float16_param_idx = fp32_param_num  # float16 index starts after fp32 params
            fp32_param_idx = 0
            for model_param in group_range["params"]:
                old_group_idx, old_param_idx = self.model_param_group_index_map[model_param]
                assert old_group_idx == group_index
                if model_param.type() in [
                    "torch.cuda.HalfTensor",
                    "torch.cuda.BFloat16Tensor",
                ]:
                    self.model_param_group_index_map[model_param] = (group_index, float16_param_idx)
                    float16_param_idx += 1
                elif model_param.type() == "torch.cuda.FloatTensor":
                    self.model_param_group_index_map[model_param] = (group_index, fp32_param_idx)
                    fp32_param_idx += 1
                else:
                    raise TypeError(
                        "Wrapped parameters must be one of "
                        "torch.cuda.FloatTensor,  "
                        "torch.cuda.HalfTensor, or "
                        "torch.cuda.BFloat16Tensor. "
                        f"Received {model_param.type()}"
                    )

        return (
            model_float16_groups,
            model_fp32_groups,
            shard_float16_groups,
            shard_fp32_groups,
            shard_casted_float16_groups,
        )

    def get_model_param_range_map(self, param):
        """
        Given a model param, get the index sub-range of the param that this
        data-parallel rank owns.
        """
        model_index, dtype, bucket_index = self.model_param_gbuf_map[param]
        gbuf_range_map = self.model_gbuf_ranges[model_index][dtype][bucket_index]
        param_range_map = gbuf_range_map["param_map"][param]
        return param_range_map

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
            # key is name,
            # value is dtype
            param_dtype = {}
            for dtype, param_maps in self.model_gbuf_ranges[i].items():
                if dtype not in distributed_state:
                    distributed_state[dtype] = {}
                for param_map in param_maps:
                    for param in param_map["param_map"].keys():
                        param_dtype[param] = dtype

            for param in model.parameters():
                if param in param_dtype.keys():
                    dtype = param_dtype[param]
                    param_key = self.param_to_name[param]
                    group_id, local_id_in_group = self.model_param_group_index_map[param]
                    distributed_state[dtype][param_key] = convert_dict_with_sharded(
                        optimizer_state["state"][self.prefix_sum_param_groups[group_id] + local_id_in_group],
                        self.param_global_shape_info[param],
                        self.param_local_shape_info[param],
                        self.param_global_offset_info[param],
                        self.param_across_dp_ranks_info.get(param),
                    )
        # If it is mix percision training, we should save master fp32 weights
        if not all(not group for group in self.shard_casted_float16_groups):
            for group in self.shard_casted_float16_groups:
                for param in group:
                    original_param = self.param_to_origin_param_for_shard_casted_float16_groups[param]
                    name = self.param_to_name[original_param]
                    dtype = torch.float32 if self.grad_to_fp32 else param.dtype
                    distributed_state[dtype][name]["shard_casted_float16_groups"] = OptimizerStateSpec(
                        self.param_global_shape_info[original_param],
                        self.param_local_shape_info[original_param],
                        self.param_global_offset_info[original_param],
                        param,
                        self.param_across_dp_ranks_info.get(original_param),
                    )

        return distributed_state

    def load_state_dict(self, state_dict):
        """
        Load the state dict.
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
        param_index = 0

        for i, model in enumerate(self.models):
            param_dtype = {}
            for dtype, param_maps in self.model_gbuf_ranges[i].items():
                for param_map in param_maps:
                    for param in param_map["param_map"].keys():
                        param_dtype[param] = dtype
            param_list = []
            for param in model.parameters():
                if param in param_dtype.keys():
                    dtype = param_dtype[param]
                    ranges = self.param_across_dp_ranks_info.get(param)
                    name = self.param_to_name[param]
                    param_list.append((param, dtype, ranges, name))

            for param_info in param_list:
                if param_info[2]:
                    my_range = param_info[2][self.current_global_rank]
                else:
                    my_range = None
                group_id, local_id_in_group = self.model_param_group_index_map[param_info[0]]
                optimizer_state["state"][self.prefix_sum_param_groups[group_id] + local_id_in_group] = (
                    convert_dict_sharded_to_tensor(state_dict[param_info[1]][param_info[3]], my_range)
                )
                param_index += 1

        self.optimizer.load_state_dict(optimizer_state)

        if not all(not group for group in self.shard_casted_float16_groups):
            for group in self.shard_casted_float16_groups:
                for param in group:
                    original_param = self.param_to_origin_param_for_shard_casted_float16_groups[param]
                    name = self.param_to_name[original_param]
                    # The weights have been flatten into 1D and get range based on current rank (if necessary)
                    # in the "resume optimizer state loop
                    dtype = torch.float32 if self.grad_to_fp32 else param.dtype
                    param.copy_(state_dict[dtype][name]["shard_casted_float16_groups"])
        # state_dict['shard_fp32_from_float16_groups']
        # optimizer_state['shard_fp32_from_float16_groups']
        # TODO: Copy data for the main params.
        # for current_group, saved_group in zip(
        #         self.shard_fp32_from_float16_groups,
        #         state_dict["shard_fp32_from_float16_groups"]):
        #     for current_param, saved_param in zip(current_group, saved_group):
        #         if isinstance(current_param.data, DTensor):
        #             current_param.data._local_tensor.copy_(saved_param.data)
        #         else:
        #             current_param.data.copy_(saved_param.data)

    def zero_grad(self, set_to_none=True):
        """
        Zero grads.

        We only need to zero the model related parameters, i.e.,
        model_float16_groups & model_fp32_groups. We additionally zero
        the remaining groups as a memory optimization to reduce
        fragmentation; in the case of set_to_none==True, the space
        used by this field can be safely deallocated at this point.
        """
        for groups in (
            self.model_float16_groups,
            self.model_fp32_groups,
            self.shard_float16_groups,  # grad empty/unused here?
            self.shard_fp32_groups,  # throws grad-access warning
            self.shard_casted_float16_groups,
        ):
            for group in groups:
                _zero_grad_group_helper(group, set_to_none)

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

    def shard_buffer_among_dp_world(self, buffer: torch.Tensor):
        assert buffer.numel() % self.data_parallel_world_size == 0
        shard_size = buffer.numel() // self.data_parallel_world_size
        return [buffer[(r * shard_size) : ((r + 1) * shard_size)] for r in range(self.data_parallel_world_size)]

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
                    buf_views = self.shard_buffer_among_dp_world(buf)
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
        all_gather_handle = torch.distributed._all_gather_base(
            pbuf,
            pbuf_views[data_parallel_rank],
            group=data_parallel_group,
            async_op=False,
        )

        self.all_gather_handles.append(all_gather_handle)
        assert self.all_gather_handle_index_to_bucket_index_map[all_gather_handle_index] == (
            model_index,
            dtype,
            bucket_index,
        )
        self.param_buffer_copied.append(False)

        if not self.overlap_param_gather:
            self._copy_params_from_param_buffer(all_gather_handle_index)

    def _enable_pre_hook(self):
        for m in self.models:
            m = m.module
            self.removable_pre_hook_handles.append(m.register_forward_pre_hook(self._make_forward_pre_hook()))

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
        for param in module.parameters(recurse=True):
            # Skip parameters that don't require grad.
            if not param.requires_grad:
                continue
            assert param in self.param_to_all_gather_handle_index_map
            all_gather_handle_index = self.param_to_all_gather_handle_index_map[param]
            self._finish_param_sync_helper(all_gather_handle_index)

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

    def _copy_params_from_param_buffer(self, all_gather_handle_index):
        """
        Copy params from param_buffer to model_params.
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
                bucket_offset = model.grad_buffers[dtype].buckets[bucket_index].offset
                param_buf = self.param_buffers[model_index][dtype][bucket_index]
                # buf_start and buf_end store position of this parameter in the full grad_buffer,
                # so need to adjust these indices (by subtracting out bucket_offset) since we
                # have independent param_bufs for each bucket.
                param_buf_shard = param_buf[buf_start - bucket_offset : buf_end - bucket_offset]
                assert (
                    param.data.nelement() == param_buf_shard.nelement()
                    or param.data._local_tensor.nelement() == param_buf_shard.nelement()
                )
                if isinstance(param.data, DTensor):
                    param._local_tensor.view(-1).detach().copy_(param_buf_shard)
                else:
                    param.view(-1).detach().copy_(param_buf_shard)

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

    def _get_model_and_main_params_data_float16(self):
        """
        Get aligned list of model and main params.
        """
        model_data = []
        main_data = []
        for model_group, main_group in zip(self.shard_float16_groups, self.shard_casted_float16_groups):
            for model_param, main_param in zip(model_group, main_group):
                model_data.append(model_param.data)
                main_data.append(main_param.data)
        return model_data, main_data

    def _copy_model_grads_to_main_grads(self):
        """
        Copy model grads to main grads.

        Since this step follows a reduce-scatter through the DDP's grad
        buffer, this method is responsible for copying the updated grads
        from the grad buffer to the main shard's grad field.
        """

        # Utility method for copying group grads.
        def copy_group_grads(model_groups, shard_main_groups):
            for model_group, shard_main_group in zip(model_groups, shard_main_groups):
                for model_param, shard_main_param in zip(model_group, shard_main_group):
                    param_range_map = self.get_model_param_range_map(model_param)
                    param_range = param_range_map["param"]
                    assert param_range.size == shard_main_param.nelement()

                    model_grad = model_param.main_grad
                    shard_model_grad = model_grad.view(-1)[param_range.start : param_range.end]
                    if self.grad_to_fp32:
                        shard_main_param.grad = shard_model_grad.float()
                    else:
                        shard_main_param.grad = shard_model_grad

        # Copy model groups to shard groups.
        copy_group_grads(self.model_float16_groups, self.shard_casted_float16_groups)
        copy_group_grads(self.model_fp32_groups, self.shard_fp32_groups)

    def _copy_main_params_to_model_params(self):
        """
        Copy main params to model params.

        Since this step is followed by an all-gather through the DDP's grad
        buffer, this method is responsible for copying the updated params
        from the main shards into the correct position in the grad buffer.
        """

        # Utility method for copying group params.
        def copy_group_params(shard_main_groups, model_groups):
            for shard_main_group, model_group in zip(shard_main_groups, model_groups):
                for shard_main_param, model_param in zip(shard_main_group, model_group):
                    param_range_map = self.get_model_param_range_map(model_param)
                    world_range = param_range_map["gbuf_world_in_bucket"]

                    assert world_range.size == shard_main_param.nelement()

                    model_id, dtype, bucket_id = self.model_param_gbuf_map[model_param]
                    model_param_buffer = self.param_buffers[model_id][dtype][bucket_id]

                    shard_model_param = model_param_buffer.view(-1)[world_range.start : world_range.end]

                    shard_model_param.data.copy_(shard_main_param)

        # Copy shard groups to model groups.
        copy_group_params(self.shard_casted_float16_groups, self.model_float16_groups)
        copy_group_params(self.shard_fp32_groups, self.model_fp32_groups)

    def _copy_model_params_to_main_params(self):
        """
        Copy model params to main params.

        During finetuning, this method is used to reload the main params from
        the model params. This copy does not make use of the grad buffer as
        an intermediary.
        """

        # Utility method for copying group params.
        def copy_group_params(model_groups, shard_main_groups):
            for model_group, shard_main_group in zip(model_groups, shard_main_groups):
                for model_param, shard_main_param in zip(model_group, shard_main_group):
                    param_range_map = self.get_model_param_range_map(model_param)
                    param_range = param_range_map["param"]
                    assert param_range.size == shard_main_param.nelement()

                    shard_model_param = model_param.view(-1)[param_range.start : param_range.end]
                    shard_main_param.data.copy_(shard_model_param)

        # Copy model groups to shard groups.
        copy_group_params(self.model_float16_groups, self.shard_casted_float16_groups)
        copy_group_params(self.model_fp32_groups, self.shard_fp32_groups)

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
            # TODO: Add utility to judge whether a param is shared.
            is_not_shared = not param_is_shared(param)
            is_not_tp_duplicate = param_is_sharded_or_replicate_on_first_rank(param)
            if grad_not_none and is_not_shared and is_not_tp_duplicate:
                grads_for_norm.append(grad._local_tensor if isinstance(grad, DTensor) else grad)

        return grads_for_norm


def initialize_optimizer_state(optimizer: DistributedOptimizer):
    optimizer._copy_model_grads_to_main_grads()
    orig_optimizer = optimizer.optimizer
    for group in orig_optimizer.param_groups:
        param_list = inspect.signature(orig_optimizer._init_group).parameters
        num_params = len(param_list)
        args = [group] + [[] for i in range(num_params - 1)]
        orig_optimizer._init_group(*args)
