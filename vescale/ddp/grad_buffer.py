################################################################################
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

import math
import warnings
from typing import Dict, List, Union, Sequence

import torch
import torch.distributed as dist

from vescale.dtensor.dtensor import DTensor
from vescale.dtensor.device_mesh import DeviceMesh
from vescale.dtensor.placement_types import Placement


def get_param_nelements(param: Union[torch.nn.Parameter, torch.Tensor]) -> int:
    if isinstance(param, torch.nn.Parameter):
        param = param.data
    if isinstance(param, DTensor):
        return param._local_tensor.nelement()
    return param.nelement()


class Bucket:
    """
    Bucket to keep track of a subset of the model's gradients. Provides functionality to register
    when params in the bucket have grads ready to be synced; an asynchronous communication call
    is automatically launched when _all_ params in the bucket have grads ready.

    Args:
        params: List of parameters whose gradients are collated in this bucket.
        data: View in larger GradBuffer that this bucket is responsible for.
        offset: Offset of this bucket's view in the larger GradBuffer.
        data_parallel_group: Data-parallel process group.
        overlap_grad_reduce: If true, overlap communication with backprop computation by
            breaking up grads into buckets. If false, single synchronous communication call
            is used instead.
        use_distributed_optimizer: If true, issue reduce-scatter communication calls as part
            of distributed optimizer. If false, issue all-reduce communication calls.
    """

    def __init__(
        self,
        params: List[torch.nn.Parameter],
        data: torch.Tensor,
        offset: int,
        data_parallel_group: dist.ProcessGroup,
        overlap_grad_reduce: bool,
        use_distributed_optimizer: bool,
    ):
        # State for bookkeeping: params is the set of parameters this bucket is
        # responsible for, params_with_grad is the set of parameters with grads
        # available. When overlap_grad_reduce is True, communication (all-reduce
        # or reduce-scatter) is issued when params_with_grad equals params.
        self.params_list = params
        self.params = set(params)
        self.params_with_grad = set()
        self.whitelist_params = set()
        self.data = data
        # The distributed optimizer needs to keep track of this bucket's offset
        # within the full grad_buffer.
        self.offset = offset
        self.data_parallel_group = data_parallel_group
        self.overlap_grad_reduce = overlap_grad_reduce
        self.use_distributed_optimizer = use_distributed_optimizer

        self.data_parallel_world_size = dist.get_world_size(group=data_parallel_group)
        self.data_parallel_rank = dist.get_rank(group=data_parallel_group)

        self.reset()

    def reset(self):
        """
        Reset metadata in bucket in preparation for the next iteration of training.
        """
        self.params_with_grad = set()
        self.whitelist_params = set()
        self.communication_handle = None
        self.partial_grad_communication_handle = None
        self.communication_issued = False
        self.partial_grad_communication_issued = False

    def shard_buffer(self, buffer: torch.Tensor):
        """
        Shard buffer into data_parallel_world_size chunks of equal size.
        """
        assert buffer.numel() % self.data_parallel_world_size == 0
        shard_size = buffer.numel() // self.data_parallel_world_size
        sharded_buffer = [
            buffer[(r * shard_size) : ((r + 1) * shard_size)] for r in range(self.data_parallel_world_size)
        ]
        return sharded_buffer

    def all_reduce_partial_grad(
        self, partial_main_grad, model_parallel_device_mesh: DeviceMesh, placements: Sequence[Placement]
    ):
        # wait for the last partial grad all-reduce finish
        if self.partial_grad_communication_handle is not None and self.partial_grad_communication_issued:
            self.partial_grad_communication_handle.wait()

        # TODO: there may be other invalid cases, we should add more checks here.
        partial_mesh_idxes = [i for i, p in enumerate(placements) if p.is_partial()]
        assert len(partial_mesh_idxes) == 1, "currently, we only consider a single Partial on the same mesh dim."
        model_parallel_pg = model_parallel_device_mesh.get_dim_groups(partial_mesh_idxes[0])

        self.partial_grad_communication_handle = dist.all_reduce(
            partial_main_grad, group=model_parallel_pg, async_op=True
        )
        self.partial_grad_communication_issued = True

    def start_grad_sync(self):
        """
        Initiates grad sync (all-reduce or reduce-scatter) communication operation
        for this bucket.

        When overlap_grad_reduce is set to True, dispatches an asynchronous
        communication call. When overlap_grad_reduce is set to False, makes
        synchronous call.
        """

        # We must wait until all partial grad in this bucket is all-reduced.
        if self.partial_grad_communication_handle is not None and self.partial_grad_communication_issued:
            self.partial_grad_communication_handle.wait()

        assert (
            self.communication_handle is None and not self.communication_issued
        ), "Should not have multiple communication calls in flight at once"

        self.data /= self.data_parallel_world_size
        # Use async_op only when overlap_grad_reduce is True.
        if self.use_distributed_optimizer:
            local_data_view = self.shard_buffer(self.data)[self.data_parallel_rank]
            self.communication_handle = dist._reduce_scatter_base(
                local_data_view,
                self.data,
                group=self.data_parallel_group,
                async_op=self.overlap_grad_reduce,
            )
        else:
            self.communication_handle = dist.all_reduce(
                self.data,
                group=self.data_parallel_group,
                async_op=self.overlap_grad_reduce,
            )
        self.communication_issued = True

    def finish_grad_sync(self):
        """
        Finishes grad sync (all-reduce or reduce-scatter) communication operation
        for this bucket.

        When overlap_grad_reduce is set to True, waits for asynchronous communication
        call to complete. When overlap_grad_reduce is set to False, makes synchronous call.
        """
        # If overlap_grad_reduce is False, start (and finish) synchronous communication call here.
        if not self.overlap_grad_reduce:
            self.start_grad_sync()
            return

        # in case there are parameters not used in forward call.
        # we shell issues the communication for the avaliable parameters
        if self.communication_handle is None or (not self.communication_issued):
            warnings.warn(
                f"DDP Bucket expects {len(self.params)} params all having .grad"
                f"but gets {len(self.params_with_grad)} grad available, "
                f"and gets {len(self.whitelist_params - self.params_with_grad)} params marked as absent in backward."
                f"This may be due to unused and unmarked model parameters. "
                "We issue blocking communication for this bucket after other overlapped communications."
            )
            self.start_grad_sync()

        assert self.communication_handle is not None and self.communication_issued, (
            f"Communication call has not been issued for this bucket "
            f"({len(self.params_with_grad)}/{len(self.params)} params have grad available)"
            f"({len(self.whitelist_params - self.params_with_grad)}/{len(self.params)} params have grad marked as absent in backward)"
        )
        self.communication_handle.wait()

    def register_grad_ready(self, param: torch.nn.Parameter):
        """
        Registers grads for the passed-in param to be "ready" for grad sync.

        When the number of microbatches is greater than 1, we only want to register
        grads as ready when processing the last microbatch and overlap_grad_reduce is True.
        """
        assert param in self.params, "Param is not in the bucket"
        assert param not in self.params_with_grad, "Cannot set grad twice"
        assert self.overlap_grad_reduce, "register_grad_ready() should be called only when overlapping grad reduce"
        self.params_with_grad.add(param)
        # If all params in bucket have grads available or marked as absent in backward, issue communication call.
        if len(self.params_with_grad.union(self.whitelist_params)) == len(self.params):
            self.start_grad_sync()

    def register_grad_maybe_absent(self, param: torch.nn.Parameter):
        """
        Registers grads for the passed-in param to be "ready" for grad sync.

        NOTE: This API should only be called when there is a sparse model structure, like MOE.
        """
        assert param in self.params, "Param is not in the bucket"
        assert self.overlap_grad_reduce, "register_grad_ready() should be called only when overlapping grad reduce"
        if param in self.params_with_grad:
            return
        self.whitelist_params.add(param)
        # If all params in bucket have grads available or marked as absent in backward, issue communication call.
        if len(self.params_with_grad.union(self.whitelist_params)) == len(self.params):
            self.start_grad_sync()

    def register_partial_grad_ready(
        self,
        param: torch.nn.Parameter,
        model_parallel_device_mesh: DeviceMesh,
        placements: Sequence[Placement],
    ):
        """
        Immediately trigger partial gradient all-reduce in an async way.
        """
        assert param in self.params, "Param is not in the bucket"
        assert any(p.is_partial() for p in placements), "Param's grad should be partial sharded"
        self.all_reduce_partial_grad(param.main_grad, model_parallel_device_mesh, placements)


class GradBuffer:
    """
    Groups gradients into a contiguous buffer, and then breaks the buffer into buckets with
    roughly `bucket_size` parameters each.

    Args:
        dtype: Type of underlying tensor.
        params: List of parameters whose gradients are collated in the underlying tensor.
        data_parallel_group: Data-parallel process group.
        bucket_size: The rough size of each bucket in terms of number of parameters.
        param_to_name: Mapping from `torch.nn.Parameter` to name (for logging purposes).
        overlap_grad_reduce: If true, overlap communication with backprop computation by
            breaking up grads into buckets. If false, single synchronous communication call
            is used instead.
        use_distributed_optimizer: If true, issue reduce-scatter communication calls as part
            of distributed optimizer. If false, issue all-reduce communication calls.
    """

    def __init__(
        self,
        dtype: torch.dtype,
        params: List[torch.nn.Parameter],
        data_parallel_group: dist.ProcessGroup,
        bucket_size: int,
        param_to_name: Dict[torch.nn.Parameter, str],  # TODO: rethink the usage of param_to_name
        overlap_grad_reduce: bool,
        use_distributed_optimizer: bool,
    ):
        # Check that params are unique.
        unique_params = set()
        for param in params:
            assert param not in unique_params
            unique_params.add(param)
        del unique_params

        # Store attributes that will be needed later.
        self.dtype = dtype
        self.data_parallel_group = data_parallel_group
        self.data_parallel_world_size = torch.distributed.get_world_size(group=self.data_parallel_group)
        self.overlap_grad_reduce = overlap_grad_reduce
        self.use_distributed_optimizer = use_distributed_optimizer
        self.is_last_microbatch = True

        # Data structures to store underlying buckets and relevant indexing data.
        self.buckets = []
        self.param_to_bucket = {}  # Param -> bucket mapping.
        self.param_index_map = {}  # Param -> location in buffer mapping (used in dist. optimizer).

        def _pad_if_needed(data_index: int):
            """Pads data indices if using distributed optimizer (to ensure uniform sharding)."""
            if use_distributed_optimizer:
                return int(math.ceil(data_index / self.data_parallel_world_size)) * self.data_parallel_world_size
            return data_index

        # First, figure out how many elements should be in the underlying buffer storage.
        # Note that if we need to split the buffer into smaller buckets, each of these
        # might need to be padded as well (if using the distributed optimizer).
        data_start_index = 0
        bucket_data_start_index = data_start_index
        bucket_params = set()

        self.bucket_indices = []
        bucket_id = 0
        for param in params[::-1]:
            # Iterate through parameters in reverse order to roughly follow backprop order,
            # and skip parameters that don't require gradients.
            if not param.requires_grad:
                continue
            this_numel = get_param_nelements(param)
            data_end_index = data_start_index + this_numel
            self.param_index_map[param] = (
                data_start_index,
                data_end_index,
                bucket_id,
            )
            bucket_params.add(param)

            # If we have enough elements already, form a new bucket.
            # If bucket_size is None, accumulate everything into a single bucket.

            # TODO: Remove len(bucket_params) > 1 when the final head that transforms token
            # representations from hidden space to vocabulary space is in a PyTorch module
            # whose forward method is called. If it is not and a bucket contains only this
            # one parameter, we get incorrect behavior (i.e., higher losses) since we do not
            # call the wait function on the bucket's all_gather_handle (we use forward pre-
            # hooks on PyTorch modules to do this when --overlap-param-gather is used).
            # As a temporary workaround, we make sure that no bucket has only one parameter.
            if bucket_size is not None:
                if (data_end_index - bucket_data_start_index) >= bucket_size and len(bucket_params) > 1:
                    data_end_index = _pad_if_needed(data_end_index)
                    self.bucket_indices.append((bucket_data_start_index, data_end_index))
                    bucket_data_start_index = data_end_index
                    bucket_params = set()
                    bucket_id += 1
            data_start_index = data_end_index

        # Add remaining params to a new bucket.
        if len(bucket_params) > 0:
            data_end_index = _pad_if_needed(data_end_index)
            self.bucket_indices.append((bucket_data_start_index, data_end_index))

        # Next, create underlying storage for buffer (with numel elements that includes
        # padding as necessary).
        self.numel = data_end_index
        if use_distributed_optimizer:
            assert self.numel % self.data_parallel_world_size == 0
        self.data = torch.zeros(
            self.numel,
            dtype=self.dtype,
            device=torch.cuda.current_device(),
            requires_grad=False,
        )

        # Finally, map main_grad fields for each parameter with a .grad field.
        bucket_params = set()
        bucket_data_start_index = 0
        cur_bucket_id = 0
        for param in params[::-1]:
            if not param.requires_grad:
                continue
            data_start_index, data_end_index, bucket_id = self.param_index_map[param]
            param_shape = param.data.shape if not isinstance(param.data, DTensor) else param.data._local_tensor.shape
            param.main_grad = self._get(param_shape, data_start_index)
            if bucket_id != cur_bucket_id:
                bucket_data_end_index = _pad_if_needed(data_start_index)
                self._set_bucket(
                    bucket_params,
                    bucket_data_start_index,
                    bucket_data_end_index,
                    cur_bucket_id,
                )
                bucket_data_start_index = bucket_data_end_index
                bucket_params = set()
                assert cur_bucket_id + 1 == len(self.buckets)
                assert bucket_id == cur_bucket_id + 1
                cur_bucket_id = bucket_id
            bucket_params.add(param)

        # Add remaining params to a new bucket.
        if len(bucket_params) > 0:
            bucket_data_end_index = _pad_if_needed(data_end_index)
            self._set_bucket(
                bucket_params,
                bucket_data_start_index,
                bucket_data_end_index,
                cur_bucket_id,
            )

        if not overlap_grad_reduce:
            assert len(bucket_params) == len(
                params
            ), "All params should be in one bucket when overlap_grad_reduce is False"

    def _set_bucket(
        self,
        bucket_params: List[torch.nn.Parameter],
        start_index: int,
        end_index: int,
        bucket_id: int,
    ):
        """
        Helper function to create new bucket, add it to list of buckets, and
        also update param->bucket mapping.
        """

        # Assert that indices are correctly padded (if needed), and that bucket
        # position is same as originally computed.
        if self.use_distributed_optimizer:
            assert start_index % self.data_parallel_world_size == 0
            assert end_index % self.data_parallel_world_size == 0
        assert (start_index, end_index) == self.bucket_indices[bucket_id]

        # Get appropriate view into global GradBuffer.
        bucket_data = self._get(torch.Size([end_index - start_index]), start_index)
        bucket = Bucket(
            params=bucket_params,
            data=bucket_data,
            offset=start_index,
            data_parallel_group=self.data_parallel_group,
            overlap_grad_reduce=self.overlap_grad_reduce,
            use_distributed_optimizer=self.use_distributed_optimizer,
        )
        self.buckets.append(bucket)
        for bucket_param in bucket_params:
            assert bucket_param not in self.param_to_bucket
            self.param_to_bucket[bucket_param] = bucket

    def _get(self, shape: torch.Size, start_index: int) -> torch.Tensor:
        """
        Return a tensor with the input `shape` as a view into the 1-D data starting at
        `start_index`.
        """
        end_index = start_index + shape.numel()
        assert end_index <= self.numel, "Requested tensor is out of buffer range"
        buffer_tensor = self.data[start_index:end_index]
        buffer_tensor = buffer_tensor.view(shape)
        return buffer_tensor

    def reset(self, zero_buffer: bool):
        """
        Zero out the underlying buffer and reset all buckets in preparation for the next
        iteration of training.

        When zero_buffer is set to True, the underlying buffer is zeroed out.
        """
        if zero_buffer:
            self.data.zero_()
        for bucket in self.buckets:
            bucket.reset()
        self.is_last_microbatch = True

    def start_grad_sync(self):
        """
        Initiates grad sync (all-reduce or reduce-scatter) communication operations
        for all buckets in the grad buffer.

        When overlap_grad_reduce is set to True, dispatches asynchronous communication
        calls. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        for bucket in self.buckets:
            bucket.start_grad_sync()

    def finish_grad_sync(self):
        """
        Finishes grad sync (all-reduce or reduce-scatter) communication operations
        for all buckets in the grad buffer.

        When overlap_grad_reduce is set to True, waits for asynchronous communication
        calls to complete. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        for bucket in self.buckets:
            bucket.finish_grad_sync()

    def register_grad_ready(self, param: torch.nn.Parameter):
        """
        Registers grads for the passed-in param to be "ready" for grad sync.

        When the number of microbatches is greater than 1, we only want to register
        grads as ready when processing the last microbatch and overlap_grad_reduce is True.
        """
        assert self.overlap_grad_reduce, "register_grad_ready() should only be called when overlap_grad_reduce is True"
        if self.is_last_microbatch:
            bucket = self.param_to_bucket[param]
            bucket.register_grad_ready(param)

    def register_partial_grad_ready(
        self,
        param: torch.nn.Parameter,
        model_parallel_device_mesh: DeviceMesh,
        placements: Sequence[Placement],
    ):
        """
        Immediately trigger partial gradient all-reduce in an async way.
        """
        bucket = self.param_to_bucket[param]
        bucket.register_partial_grad_ready(param, model_parallel_device_mesh, placements)

    def register_grad_maybe_absent(self, param: torch.nn.Parameter):
        """
        Registers grads for the passed-in param to be "ready" for grad sync.

        NOTE: This API should only be called when there is a sparse model structure, like MOE.
        """
        assert self.overlap_grad_reduce, "register_grad_ready() should only be called when overlap_grad_reduce is True"
        if self.is_last_microbatch:
            bucket = self.param_to_bucket[param]
            bucket.register_grad_maybe_absent(param)
