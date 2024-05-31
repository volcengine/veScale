################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

import functools
import operator

from typing import Dict, Tuple, cast

import torch
import torch.distributed as dist

import vescale

__all__ = ["_bypass_for_dispatch"]

aten = torch.ops.aten


class BypassOpDispatch:
    """
    Register custom op handler to bypass dispatching here
    """

    def __init__(self):
        self.op_handlers = {
            # origin bypass op dispatch func
            aten.linear.default: BypassOpDispatch.decompose_handler,
            aten.is_same_size.default: BypassOpDispatch.is_same_size_handler,
            # from bypass op sharding prop
            aten.nonzero.default: BypassOpDispatch.nonzero_handler,
            aten._to_copy.default: BypassOpDispatch.copy_handler,
            aten._local_scalar_dense.default: BypassOpDispatch.scalar_handler,
            aten.equal.default: BypassOpDispatch.equal_handler,
            # other ?
        }

    def apply(
        self,
        op_call: torch._ops.OpOverload,
        args: Tuple[object, ...],
        kwargs: Dict[str, object],
    ) -> Tuple[bool, object]:
        bypass_call = self.op_handlers.get(op_call, None)
        if bypass_call is not None:
            return True, bypass_call(op_call, args, kwargs)  # type: ignore[operator]
        else:
            return False, None

    @staticmethod
    def decompose_handler(
        op_call: torch._ops.OpOverload,
        args: Tuple[object, ...],
        kwargs: Dict[str, object],
    ) -> object:
        """
        Decomposes a op to core ATen op, this handler is mostly here
        for inference mode usage where the ops are not core aten ops.
        """
        r = op_call.decompose(*args, **kwargs)
        if r is not NotImplemented:
            return r
        else:
            raise RuntimeError("Decomposition failed")

    @staticmethod
    def is_same_size_handler(
        op_call: torch._ops.OpOverload,
        args: Tuple[object, ...],
        kwargs: Dict[str, object],
    ) -> bool:
        lhs = cast(torch.Tensor, args[0])
        rhs = cast(torch.Tensor, args[1])
        return lhs.shape == rhs.shape

    @staticmethod
    def nonzero_handler(
        op_call: torch._ops.OpOverload,
        args: Tuple[object, ...],
        kwargs: Dict[str, object],
    ) -> object:
        input_ = kwargs.get("input", args[0])
        assert isinstance(input_, vescale.dtensor.DTensor)
        input_spec = input_._spec
        all_replicate = all(p.is_replicate() for p in input_spec.placements)
        assert all_replicate, "input placement has to be replicate"
        input_local = input_._local_tensor
        output_local = op_call(input_local)
        return vescale.dtensor.DTensor(
            local_tensor=output_local,
            device_mesh=input_spec.mesh,
            placements=input_spec.placements,
            shape=output_local.shape,
            dtype=output_local.dtype,
            requires_grad=output_local.requires_grad,
            stride=output_local.stride(),
        )

    @staticmethod
    def copy_handler(
        op_call: torch._ops.OpOverload,
        args: Tuple[object, ...],
        kwargs: Dict[str, object],
    ) -> object:
        input_dtensor = args[0]
        input_local = input_dtensor._local_tensor
        output_local = op_call(*(input_local, *(args[1:])), **kwargs)
        return vescale.dtensor.DTensor(
            local_tensor=output_local,
            device_mesh=input_dtensor.device_mesh,
            placements=input_dtensor.placements,
            shape=input_dtensor.shape,
            dtype=output_local.dtype,
            requires_grad=output_local.requires_grad,
            stride=input_dtensor.stride(),
        )

    @staticmethod
    def scalar_handler(
        op_call: torch._ops.OpOverload,
        args: Tuple[object, ...],
        kwargs: Dict[str, object],
    ) -> object:
        input_dtensor = args[0]
        input_local = input_dtensor._local_tensor
        return op_call(*(input_local, *(args[1:])), **kwargs)

    @staticmethod
    def equal_handler(
        op_call: torch._ops.OpOverload,
        args: Tuple[object, ...],
        kwargs: Dict[str, object],
    ) -> object:
        dtensor0 = args[0]
        dtensor1 = args[1]
        local0 = dtensor0._local_tensor
        local1 = dtensor1._local_tensor
        local_results = op_call(local0, local1, *(args[2:]), **kwargs)
        if dtensor0._spec.is_replicated() and dtensor1._spec.is_replicated():
            return local_results

        obj_list = [None] * dist.get_world_size()
        dist.all_gather_object(obj_list, local_results)  # type: ignore[possibly-undefined]
        obj_list = [e for e in obj_list if e is not None]
        # perform reduce on the collection with AND op
        # :NOTE: here is an implicit communication
        local_results = functools.reduce(operator.and_, obj_list, True)

        return local_results


_bypass_op_dispatch = BypassOpDispatch()


def _bypass_for_dispatch(
    op_call: torch._ops.OpOverload,
    args: Tuple[object, ...],
    kwargs: Dict[str, object],
) -> Tuple[bool, object]:
    """
    Put bypass logic here before entering dtensor dispatching logic
    """
    return _bypass_op_dispatch.apply(op_call, args, kwargs)
