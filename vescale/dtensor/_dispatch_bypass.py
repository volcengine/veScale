################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

from typing import Dict, Tuple, cast

import torch

from vescale.dtensor.op_schema import (
    DTensorSpec,
    OpInfo,
    OutputSharding,
)
from vescale.dtensor.placement_types import TensorMeta

__all__ = ["_bypass_for_dispatch", "_bypass_for_sharding_prop"]

aten = torch.ops.aten


class BypassOpDispatch:
    """
    Register custom op handler to bypass dispatching here
    """

    def __init__(self):
        self.op_handlers = {
            aten.linear.default: BypassOpDispatch.decompose_handler,
            aten.is_same_size.default: BypassOpDispatch.is_same_size_handler,
            aten.nonzero.default: BypassOpDispatch.nonzero_handler,
        }

    def apply(
        self,
        op_call: torch._ops.OpOverload,
        args: Tuple[object, ...],
        kwargs: Dict[str, object],
    ) -> Tuple[bool, object]:
        is_bypass = op_call in self.op_handlers
        if is_bypass:
            return True, self.op_handlers[op_call](op_call, args, kwargs)  # type: ignore[operator]
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
        from vescale.dtensor import DTensor

        input_ = kwargs.get("input", args[0])
        assert isinstance(input_, DTensor)
        input_spec = input_._spec
        all_replicate = all(p.is_replicate() for p in input_spec.placements)
        assert all_replicate, "input placement has to be replicate"
        input_local = input_._local_tensor
        output_local = op_call(input_local)
        return DTensor(
            local_tensor=output_local,
            device_mesh=input_spec.mesh,
            placements=input_spec.placements,
            shape=output_local.shape,
            dtype=output_local.dtype,
            requires_grad=output_local.requires_grad,
            stride=output_local.stride(),
        )


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


class BypassOpShardingProp:
    """
    Register custom op handler to bypass sharding propagation here
    """

    def __init__(self):
        self.op_handlers = {
            aten._to_copy.default: BypassOpShardingProp.copy_handler,
            aten._local_scalar_dense.default: BypassOpShardingProp.scalar_handler,
            aten.equal.default: BypassOpShardingProp.scalar_handler,
        }

    def apply(self, op_info: OpInfo) -> bool:
        is_bypass = op_info.schema.op in self.op_handlers
        if is_bypass:
            op_info.output_sharding = self.op_handlers[op_info.schema.op](op_info)
            return True
        else:
            return False

    @staticmethod
    def copy_handler(op_info: OpInfo) -> OutputSharding:
        op_schema = op_info.schema
        kwargs = op_schema.gen_fake_kwargs()
        dtype = kwargs["dtype"]
        args_spec0 = op_schema.args_spec[0]
        out_tensor_meta = TensorMeta(
            shape=args_spec0.tensor_meta.shape,
            stride=args_spec0.tensor_meta.stride,
            dtype=dtype,
        )
        return OutputSharding(
            output_spec=DTensorSpec(
                mesh=args_spec0.mesh,
                placements=args_spec0.placements,
                tensor_meta=out_tensor_meta,
            )
        )

    @staticmethod
    def scalar_handler(op_info: OpInfo) -> OutputSharding:
        return OutputSharding(None, [op_info.schema])


_bypass_op_sharding_prop = BypassOpShardingProp()


def _bypass_for_sharding_prop(op_info: OpInfo) -> bool:
    """
    Put bypass logic here before entering dtensor sharding propagation logic
    """
    return _bypass_op_sharding_prop.apply(op_info)
