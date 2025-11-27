################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################
import logging
from typing import cast, Tuple, Dict

import torch
from torch.distributed.tensor._dispatch import OpDispatcher as TorchOpDispatcher
from torch.distributed.tensor import DTensor as TorchDTensor
from vescale.dtensor._dtensor_spec import DTensorSpec, TensorMeta
from vescale.dtensor._op_schema import (
    OpInfo,
    OpSchema,
    OutputSpecType,
)

import vescale.dtensor as dtensor
from vescale.dtensor._redistribute import redistribute_local_tensor
from vescale.dtensor._sharding_prop import ShardingPropagator
from vescale.dtensor.placement_types import (
    Partial,
    Placement,
    Replicate,
)

from vescale.dtensor.vescale_utils import (
    retrieve_flattened_index_before_ragged_shard,
    get_unflattened_shape_and_offset_before_ragged_shard,
    best_effort_reshape,
)

try:
    from torch.utils import _cxx_pytree as pytree
except ImportError:
    from torch.utils import _pytree as pytree  # type: ignore[no-redef]

"""
In this file, we did the following modifications:
1. found_inf_reduce_handler is kept but unmodified to use our dtensor.
2. ragged_norm_op_handler is add to support ragged shard.
3. OpDispatcher inherits from pytorch OpDispatcher
4. OpDispatcher.__init__ is modified. super.__init__ is called and more custom handler is used.
5. OpDispatcher._cvt_dtensor is a new method.
6. OpDispatcher.dispatch is modified to call OpDispatcher._cvt_dtensor first and then call super.dispatch
7. OpDispatcher.wrap is kept but unmodified to use our dtensor.
8. OpDispatcher.redistribute_local_args is kept but unmodified to use our redistribute_local_tensor.
"""


aten = torch.ops.aten
logger = logging.getLogger(__name__)


def found_inf_reduce_handler(
    op_call: torch._ops.OpOverload,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> None:
    op_info = dtensor.DTensor._op_dispatcher.unwrap_to_op_info(op_call, args, kwargs)
    local_tensor_args = pytree.tree_unflatten(
        cast(list[object], op_info.local_args),
        op_info.args_tree_spec,  # type: ignore[arg-type]
    )
    local_tensor_args = cast(tuple[object, ...], local_tensor_args)
    op_call(*local_tensor_args, **op_info.local_kwargs)

    grad_dtensor = cast(list[dtensor.DTensor], args[0])[0]
    grad_placements = grad_dtensor.placements
    mesh = grad_dtensor.device_mesh

    found_inf_placements: list[Placement] = []
    for placement in grad_placements:
        if isinstance(placement, Replicate):
            found_inf_placements.append(placement)
        else:
            found_inf_placements.append(Partial("max"))

    target_tensor = cast(torch.Tensor, args[1])
    spec = DTensorSpec(
        mesh=mesh,
        placements=tuple(found_inf_placements),
        tensor_meta=TensorMeta(
            shape=target_tensor.size(),
            stride=target_tensor.stride(),
            dtype=target_tensor.dtype,
        ),
    )
    found_inf_dtensor = dtensor.DTensor(local_tensor=target_tensor, spec=spec, requires_grad=False)
    found_inf = found_inf_dtensor.full_tensor()
    target_tensor.copy_(found_inf)


def _check_ragged_shard(placements):
    rt = None
    for p in placements:
        if p.is_ragged_shard():
            assert rt is None
            rt = p
    return rt


def is_contiguous(shape, stride):
    expected = 1
    for i in reversed(range(len(shape))):
        if shape[i] != 1:
            if stride[i] != expected:
                return False
            expected *= shape[i]
    return True


def fused_adamw_sgd_op_handler(
    op_call: torch._ops.OpOverload,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    def _unwrap_adamw_dtensors(args):
        rt = []
        for tensor_list in args:
            new_list = [t._local_tensor if hasattr(t, "_local_tensor") else t for t in tensor_list]
            rt.append(new_list)
        return rt

    local_tensor_args = _unwrap_adamw_dtensors(args)
    local_tensor_args = cast(tuple[object, ...], local_tensor_args)
    return op_call(*local_tensor_args, **kwargs)


_zero_tensor_by_dtype_and_device: Dict[Tuple[torch.dtype, torch.device], torch.Tensor] = {}


def _get_default_scalar_tensor(dtype: torch.dtype, device: torch.device):
    key = (dtype, device)
    if key not in _zero_tensor_by_dtype_and_device:
        _zero_tensor_by_dtype_and_device[key] = torch.zeros(torch.Size([]), dtype=dtype, device=device)
    local_results = _zero_tensor_by_dtype_and_device[key]
    return local_results


@torch.compile
def ragged_norm_kernel(global_shape, local_tensor: torch.Tensor, global_start_idx, global_end_idx, norm_ord, dim):
    t0 = torch.zeros(*global_shape, dtype=local_tensor.dtype, device=local_tensor.device)
    t1 = t0.view(-1)
    t1[global_start_idx:global_end_idx].copy_(local_tensor)
    return torch.linalg.vector_norm(t0, ord=norm_ord, dim=dim)


def ragged_norm_op_handler(
    op_call: torch._ops.OpOverload,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> Tuple[bool, object]:
    from vescale.dtensor.placement_types import RaggedShard

    global _zero_tensor_by_dtype_and_device
    # Unwrap inputs to local tensors and execute op locally
    dispatcher = dtensor.DTensor._op_dispatcher
    op_info = dispatcher.unwrap_to_op_info(op_call, args, kwargs)
    dispatcher.sharding_propagator.propagate(op_info)
    assert op_info.output_sharding is not None
    output_spec = op_info.output_sharding.output_spec
    assert isinstance(output_spec, DTensorSpec)
    local_tensor_args = (
        pytree.tree_unflatten(
            cast(list[object], op_info.local_args),
            op_info.args_tree_spec,  # type: ignore[arg-type]
        )
        if op_info.args_tree_spec
        else op_info.local_args
    )
    local_tensor_args = cast(tuple[object, ...], local_tensor_args)
    tensor_spec: DTensorSpec = cast(DTensorSpec, op_info.flat_args_schema[0])
    tensor_meta = tensor_spec.tensor_meta
    assert tensor_meta is not None
    ragged_shard_p = _check_ragged_shard(tensor_spec.placements)

    if ragged_shard_p is None:
        return False, ()

    ragged_shard_p = cast(RaggedShard, ragged_shard_p)
    assert ragged_shard_p.dims == tuple(range(len(ragged_shard_p.dims))), f"{ragged_shard_p} {ragged_shard_p.dims}"

    n_args = len(local_tensor_args)
    if not ((0 < n_args <= 3) and len(kwargs) == 0):
        raise RuntimeError(
            f"RaggedShard only supports {op_call} without keepdim, dtype=None, and out but got {local_tensor_args=} {kwargs=}"
        )

    input_t = local_tensor_args[0]
    assert isinstance(input_t, torch.Tensor)
    norm_ord = 2
    dim: None | int = None
    if n_args >= 2:
        norm_ord = cast(int, local_tensor_args[1])
    if n_args >= 3:
        t = cast(list, local_tensor_args[2])
        assert len(t) == 1, "currently we only support norm with 1 dim"
        dim = cast(int, t[0])
        if dim < 0:
            dim = len(tensor_meta.shape) + dim

    if dim is None:
        # short pass for whole tensor norm
        if input_t.numel() == 0:
            local_result = _get_default_scalar_tensor(input_t.dtype, input_t.device)
        else:
            local_result = op_call(*local_tensor_args, **op_info.local_kwargs)
        return True, dtensor.DTensor(
            local_result,
            output_spec,
            requires_grad=False,
        )  # type: ignore

    if dim not in ragged_shard_p.dims:
        if input_t.numel() == 0:
            local_result = torch.empty(0, dtype=input_t.dtype, device=input_t.device)
        else:
            if len(ragged_shard_p.dims) > 0:
                dim = dim - (len(ragged_shard_p.dims) - 1)
            input_view = best_effort_reshape(input_t, tensor_spec)
            local_result = op_call(input_view, norm_ord, dim)
        return True, dtensor.DTensor(
            local_result.view(-1),
            output_spec,
            requires_grad=False,
        )  # type: ignore

    unflattened_shapes, _ = get_unflattened_shape_and_offset_before_ragged_shard(tensor_spec)
    flat_start_idx, flat_end_idx = retrieve_flattened_index_before_ragged_shard(tensor_spec)

    # convert to exclusive end
    local_result = ragged_norm_kernel(unflattened_shapes, input_t, flat_start_idx, flat_end_idx, norm_ord, dim)

    return True, dtensor.DTensor(
        local_result,
        output_spec,
        requires_grad=False,
    )  # type: ignore


class OpDispatcher(TorchOpDispatcher):
    """
    Op dispatching class instance to handle args/kwargs pre-processing (un-wrapping), sharding
    propagation, redistribute local args, local compute, and post-processing (re-wrapping). It
    also handles any op specific logic if necessary.

    NOTE: Given the runtime overhead of Tensor subclass (__torch_dispatch__), the OpDispatcher
    is designed to minimize the CPU overhead by using the tricks of proper unflattening, faster
    pytree if needed, and leveraging various caching mechanisms implemented in the sharding
    propagation and redistribute modules. The CPU overhead is critical to eager mode performance,
    one need to carefully measure the CPU overhead when making significant changes to the
    OpDispatcher and ShardingPropagator.
    """

    def __init__(self) -> None:
        self._torch_dispatcher = TorchDTensor._op_dispatcher  # this must go before super().__init__()
        super().__init__()
        self.sharding_propagator = ShardingPropagator()
        self._cond_op_handlers = {
            aten.linalg_vector_norm.default: ragged_norm_op_handler,
            # aten._foreach_norm.Scalar: ragged_norm_op_handler,
        }
        self._custom_op_handlers[aten._fused_adamw_.default] = fused_adamw_sgd_op_handler
        self._custom_op_handlers[aten._fused_sgd_.default] = fused_adamw_sgd_op_handler
        self._custom_op_handlers[aten._amp_foreach_non_finite_check_and_unscale_.default] = found_inf_reduce_handler

    @property
    def _allow_implicit_replication(self):
        return self._torch_dispatcher._allow_implicit_replication

    @_allow_implicit_replication.setter
    def _allow_implicit_replication(self, value):
        self._torch_dispatcher._allow_implicit_replication = value

    def _cvt_dtensor(
        self,
        op_call: torch._ops.OpOverload,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> Tuple[torch._ops.OpOverload, tuple[object, ...], dict[str, object]]:
        # Zezhou: enable DTensor ops with Tensor, simply transform Tensors into Replicate().

        if op_call not in (aten._foreach_mul_.Tensor, aten._foreach_norm.Scalar, aten.mul_.Tensor):
            return op_call, args, kwargs

        if all((not isinstance(x, torch.Tensor)) or isinstance(x, dtensor.DTensor) for x in pytree.tree_leaves(args)):
            return op_call, args, kwargs

        ref_mesh = None
        for x in pytree.tree_leaves(args):
            if isinstance(x, dtensor.DTensor):
                ref_mesh = x.device_mesh
                break
        if ref_mesh is not None:
            args = pytree.tree_map(
                lambda v: v
                if isinstance(v, dtensor.DTensor) or not isinstance(v, torch.Tensor)
                else dtensor.DTensor(
                    v,
                    DTensorSpec(
                        ref_mesh,
                        [Replicate()] * ref_mesh.ndim,
                        tensor_meta=TensorMeta(v.size(), v.stride(), v.dtype),
                    ),
                    requires_grad=False,
                ),
                args,
            )
        return op_call, args, kwargs

    def dispatch(
        self,
        op_call: torch._ops.OpOverload,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> object:
        """
        Main dispatching logic
        """
        op_call, args, kwargs = self._cvt_dtensor(op_call, args, kwargs)
        if op_call in self._cond_op_handlers:
            cond, ret = self._cond_op_handlers[op_call](op_call, args, kwargs)
            if cond:
                return ret
        return super().dispatch(op_call, args, kwargs)

    @staticmethod
    def redistribute_local_args(
        op_info: OpInfo,
        suggested_input_schema: OpSchema,
    ) -> None:
        # NOTE: it's very rare that we need to reshard kwargs so we intentionally skip it
        if op_info.args_tree_spec is not None:
            flatten_args_schema_to_reshard = tuple(pytree.tree_leaves(suggested_input_schema.args_schema))
        else:
            flatten_args_schema_to_reshard = suggested_input_schema.args_schema

        new_local_args: list[object] = []
        for i, arg_spec in enumerate(op_info.flat_args_schema):
            reshard_arg_spec = flatten_args_schema_to_reshard[i]
            if isinstance(arg_spec, DTensorSpec):
                local_tensor = cast(torch.Tensor, op_info.local_args[i])
                if arg_spec != reshard_arg_spec:
                    resharded_local_tensor = redistribute_local_tensor(local_tensor, arg_spec, reshard_arg_spec)
                    new_local_args.append(resharded_local_tensor)
                else:
                    new_local_args.append(local_tensor)
            else:
                new_local_args.append(reshard_arg_spec)

        op_info.local_args = tuple(new_local_args)

    @staticmethod
    def wrap(res: object, spec: OutputSpecType) -> object:
        if isinstance(res, torch.Tensor):
            if spec is not None:
                assert isinstance(spec, DTensorSpec), (
                    f"output spec does not match with output! Expected DTensorSpec, got {spec}."
                )
                return dtensor.DTensor(res, spec, requires_grad=res.requires_grad)
            else:
                # if output does not have a DTensorSpec due to specific ops, it must be a scalar tensor
                assert res.ndim == 0, "output tensor should be scalar!"
                return res
        elif isinstance(res, (list, tuple)):
            assert spec is not None and isinstance(spec, (list, tuple)), (
                f"output spec does not match with output! Expected list/tuple, got {spec}."
            )
            res_list = []
            for e, s in zip(res, spec):
                res_list.append(OpDispatcher.wrap(e, s))

            return tuple(res_list) if isinstance(res, tuple) else res_list
        else:
            # if the res contains only non tensor values (i.e. int/float/none), we simply return it
            # without rewrapping to DTensor.
            return res
