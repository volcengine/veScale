################################################################################
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

from typing import Dict, Union, List, Type

import torch
import torch.distributed.distributed_c10d as c10d

from vescale.dtensor.dtensor import DTensor
from vescale.dtensor.device_mesh import DeviceMesh
from vescale.ddp.grad_buffer import GradBuffer


_DDP_IGNORE_TAG = "DDP_IGNORE"


class DistributedDataParallel(torch.nn.Module):
    """
    DDP wrapper which stores grads in contiguous buffers. Also has option of overlapping
    communication with backprop computation by breaking up full model's gradients into smaller
    buckets and running all-reduce / reduce-scatter on each bucket asynchronously. This class
    also provides the option to do the gradient accumulation in a type other than the param type
    (e.g., fp32 for a bf16 model).

    Args:
        module (nn.Module): Underlying torch.nn.Module.
        data_pg_or_device_mesh (Union[ProcessGroup, DeviceMesh]): Data-parallel process group or DeviceMesh instance.
            If DeviceMesh is provided, we by defualt treat the first dimension as the data-parallel dimension.
        accumulate_allreduce_grads_in_fp32: If true, do the gradient accumulation and
            communication in fp32.
        overlap_grad_reduce (bool): If true, overlap communication with backprop computation by
            breaking up grads into buckets. If false, single synchronous communication call
            is used instead.
        use_distributed_optimizer (bool): If true, issue reduce-scatter communication calls as part
            of distributed optimizer. If false, issue all-reduce communication calls.
        disable_bucketing (bool): If true, force assign all parameters to a single bucket. If false,
            use standard bucketing policy: assign parameters to smaller buckets and all-reduce
            per bucket _if_ overlap_grad_reduce is True and pp_rank is 0.
        bucket_size (int): the size of single bucket, only useful when bucketing is enabled. By default,
            40000000.
        module_to_enforce (List[Type]): Types of sparse submodules. By default, None.
        param_to_ignore (List[str]): A list of fully qualified names of parameters to be ignored duing gradient
            syncronization. By default, None.

    Returns:
        A :class:`DistributedDataParallel` object.

    Example:
        ```python
        # The following program will create a DDP object which contains a mlp as the core module. It runs
        # in distributed environment with 4 devices. We treat the first dimension as data-parallel, i,e,.
        # modules on rank 0 and rank 2 share the same weights, as do the modules on rank 1 and rank 3.

        from vescale.ddp.distributed_data_parallel import DistributedDataParallel as DDP
        from vescale.dmodule.api import parallelize_module

        mesh = DeviceMesh("cuda", mesh=[[0, 1], [2, 3]])
        mlp = parallelize_module(MLP(), mesh, ..., ...)
        ddp_module = DDP(
            module=mlp,
            data_pg_or_device_mesh=mesh,
            module_to_enforce=[MoEBlock]
        )
        # run the forward.
        ddp_module(torch.rand(xxx))
        ```
    """

    def __init__(
        self,
        module: torch.nn.Module,
        data_pg_or_device_mesh: Union[c10d.ProcessGroup, DeviceMesh],
        accumulate_allreduce_grads_in_fp32: bool = True,
        overlap_grad_reduce: bool = True,
        use_distributed_optimizer: bool = False,
        disable_bucketing: bool = False,
        bucket_size: int = 40000000,  # Unit: number of the elements
        module_to_enforce: List[Type] = None,
        param_to_ignore: List[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.module = module

        if isinstance(data_pg_or_device_mesh, DeviceMesh):
            # By default, we treat the first dim of devicemesh as data parallel dim.
            self.data_parallel_group = data_pg_or_device_mesh.get_dim_groups(0)
        elif isinstance(data_pg_or_device_mesh, c10d.ProcessGroup):
            self.data_parallel_group = data_pg_or_device_mesh
        else:
            raise ValueError("Please provide a DeviceMesh or ProcessGroup object")

        self.data_parallel_ranks = list(c10d._pg_group_ranks[self.data_parallel_group].keys())  # global ranks
        self.data_parallel_world_size = torch.distributed.get_world_size(self.data_parallel_group)

        # Set bucket_size to infinity if overlap_grad_reduce is False.
        self.overlap_grad_reduce = overlap_grad_reduce
        self.use_distributed_optimizer = use_distributed_optimizer

        # Turn off bucketing if overlap_grad_reduce is False, if we are on a pipeline stage
        # that is not the first (since data-parallel communication on these stages is not on
        # the critical path), or if disable_bucketing is True (e.g., we might not want to
        # break up model parameters into buckets for model chunks after the first
        # in the interleaved schedule).
        if not self.overlap_grad_reduce:
            bucket_size = None
        # TODO: refine about this
        # if parallel_state.get_pipeline_model_parallel_rank() > 0:
        #     bucket_size = None
        if disable_bucketing:
            bucket_size = None
        self.bucket_size = bucket_size

        param_to_ignore = set() if param_to_ignore is None else set(param_to_ignore)

        self.module = module
        self.grad_buffers = {}
        self.grad_buffer_param_index_map = {}
        self.param_to_grad_buffer = {}
        self.ignored_param = []

        # Group parameters by their gradient type.
        grad_dtype_to_params = {}
        param_to_name = {}

        for name, param in self.module.named_parameters():
            if param.requires_grad:
                dtype = torch.float if accumulate_allreduce_grads_in_fp32 else param.dtype

                if name in param_to_ignore:
                    if not hasattr(param, _DDP_IGNORE_TAG):
                        setattr(param, _DDP_IGNORE_TAG, True)
                        param.main_grad = None
                        self.ignored_param.append(param)
                else:
                    assert not hasattr(param, _DDP_IGNORE_TAG), "registering a parameter that has been ignored by DDP"
                    param_to_name[param] = name
                    params = grad_dtype_to_params.get(dtype, [])
                    params.append(param)
                    grad_dtype_to_params[dtype] = params

        # Allocate the grad buffers and map the grads.
        # The grad buffer under the hood creates buckets as appropriate based on bucket_size.
        for dtype, params in grad_dtype_to_params.items():
            self.grad_buffers[dtype] = GradBuffer(
                dtype,
                params,
                self.data_parallel_group,
                bucket_size,
                param_to_name,
                self.overlap_grad_reduce,
                self.use_distributed_optimizer,
            )

            self.grad_buffer_param_index_map[dtype] = self.grad_buffers[dtype].param_index_map
            for param in params:
                self.param_to_grad_buffer[param] = self.grad_buffers[dtype]

        # Register backward hook.
        # Accumulation function for the gradients need to be stored so they
        # don't go out of scope.
        self.grad_accs = []
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                # Expand so we get access to grad_fn.
                param_tmp = param.expand_as(param)
                # Get the gradient accumulator function.
                grad_acc = param_tmp.grad_fn.next_functions[0][0]
                grad_acc.register_hook(self._make_param_hook(name, param, self.param_to_grad_buffer))
                self.grad_accs.append(grad_acc)

        # Register backward hook for submodules of sparse structure.
        if module_to_enforce is not None and self.overlap_grad_reduce:
            for submod in self.module.modules():
                is_sparse = False
                for t in module_to_enforce:
                    if isinstance(submod, t):
                        is_sparse = True
                        break
                if not is_sparse:
                    continue
                submod.register_forward_pre_hook(self._make_sparse_module_pre_hook(), prepend=True)

        self.fx_grad_sharding = {}
        self.model_parallel_device_mesh = None

    def forward(self, *inputs, **kwargs):
        """
        Calls the wrapped module's forward() method.
        """
        return self.module(*inputs, **kwargs)

    def _make_param_hook(
        self,
        fqn: str,
        param: torch.nn.Parameter,
        param_to_grad_buffer: Dict[torch.nn.Parameter, GradBuffer],
    ):
        """
        Creates the all-reduce / reduce-scatter hook for back propagation.
        """

        def param_hook(*unused):
            if param.requires_grad:
                if hasattr(param, _DDP_IGNORE_TAG):
                    if isinstance(param.data, DTensor):
                        grad = param.grad._local_tensor.data
                    else:
                        grad = param.grad.data
                    param.grad = None
                    if param.main_grad is None:
                        param.main_grad = grad
                    else:
                        param.main_grad.add_(grad)
                    return

                if self.overlap_grad_reduce:
                    assert param.grad is not None, "param.grad being None is not safe when overlap_grad_reduce is True"
                model_parallel_device_mesh, placements = None, None
                if param.grad is not None:
                    if isinstance(param.data, DTensor):
                        param.main_grad.add_(param.grad._local_tensor.data)  # add DTensor's data
                        model_parallel_device_mesh = param.grad._spec.mesh
                        placements = param.grad._spec.placements
                        self.model_parallel_device_mesh = model_parallel_device_mesh
                    else:
                        param.main_grad.add_(param.grad.data)
                        model_parallel_device_mesh = self.model_parallel_device_mesh
                        placements = self.fx_grad_sharding[fqn].grad_sharding if fqn in self.fx_grad_sharding else None
                param.grad = None

                if (
                    model_parallel_device_mesh is not None
                    and placements is not None
                    and any(p.is_partial() for p in placements)
                ):
                    param_to_grad_buffer[param].register_partial_grad_ready(
                        param, model_parallel_device_mesh, placements
                    )
                if self.overlap_grad_reduce:
                    param_to_grad_buffer[param].register_grad_ready(param)

        return param_hook

    def _make_sparse_module_backward_hook(self, sparse_module, param_to_grad_buffer):
        """
        Creates the all-reduce / reduce-scatter hook for back propagation of sparse Modules, like MOE.
        """

        def backward_hook(*unused):
            # we do nothing if not overlap_grad_reduce.
            if not self.overlap_grad_reduce:
                return
            # force to mark all parameters in the sparse_module as ready for allreduce
            # once we found the back propagation of the module is finished.
            for param in sparse_module.parameters():
                param_to_grad_buffer[param].register_grad_maybe_absent(param)

        return backward_hook

    def _make_sparse_module_pre_hook(self):
        def sparse_module_pre_hook(module, args):
            for x in args:
                if isinstance(x, torch.Tensor) and x.requires_grad:
                    x.register_hook(self._make_sparse_module_backward_hook(module, self.param_to_grad_buffer))
                    break

        return sparse_module_pre_hook

    def load_fx_grad_sharding(self, grad_sharding, mesh):
        self.fx_grad_sharding = grad_sharding
        self.model_parallel_device_mesh = mesh

    def start_grad_sync(self, *unused):
        """
        Initiates grad sync (all-reduce or reduce-scatter) communication operations
        for all model gradients.

        When overlap_grad_reduce is set to True, dispatches asynchronous communication
        calls. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        for grad_buffer in self.grad_buffers.values():
            grad_buffer.start_grad_sync()

    def finish_grad_sync(self):
        """
        Finishes grad sync (all-reduce or reduce-scatter) communication operations
        for all model gradients.

        When overlap_grad_reduce is set to True, waits for asynchronous communication
        calls to complete. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        for grad_buffer in self.grad_buffers.values():
            grad_buffer.finish_grad_sync()

    def zero_grad_buffer(self, zero_buffer: bool = True):
        """
        Zeros out all grad buffers. Needs to be called at the beginning of each
        training iteration.

        When zero_buffer is set to True, the underlying grad buffer is zeroed out.
        """
        for grad_buffer in self.grad_buffers.values():
            grad_buffer.reset(zero_buffer)
        for param in self.ignored_param:
            param.main_grad = None

    def state_dict(self, prefix="", keep_vars=False):
        """
        Returns a dictionary containing references to the whole state of the
        wrapped module.

        Both parameters and persistent buffers (e.g. running averages) are included.
        Keys are corresponding parameter and buffer names. Parameters and buffers
        set to None are not included.
        """
        return self.module.state_dict(prefix=prefix, keep_vars=keep_vars)

    def state_dict_for_save_checkpoint(self, prefix="", keep_vars=False):
        """
        Returns wrapped module's state_dict for checkpoint saving.
        """
        return self.module.state_dict_for_save_checkpoint(prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        """
        Copies parameters and buffers from state_dict into the wrapped module and its
        descendants. If strict is True, then the keys of state_dict must exactly match
        the keys returned by this module’s state_dict() function.
        """
        self.module.load_state_dict(state_dict, strict=strict)
