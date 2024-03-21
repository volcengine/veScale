################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

import warnings
from typing import Optional, Sequence, Tuple

import torch

import vescale.dtensor.dispatch as op_dispatch
from vescale.dtensor.device_mesh import DeviceMesh
from vescale.dtensor.placement_types import DTensorSpec, Placement, Replicate, TensorMeta
from vescale.dtensor.sharding_prop import ShardingPropagator


__all__ = ["DTensor"]
aten = torch.ops.aten

_OK_TO_USE_DATA_PTR = True


def _dispatch_torch_make_wrapper_subclass(*args, data_ptr, **kwargs):
    global _OK_TO_USE_DATA_PTR

    if _OK_TO_USE_DATA_PTR:
        try:
            return torch.Tensor._make_wrapper_subclass(*args, data_ptr=data_ptr, **kwargs)
        except TypeError:
            warnings.warn(
                "The current torch version does not have the _make_wrapper_subclass "
                "that is compatible with the usage of local data_ptr."
                "This may disable supports for plug-in ops(a.k.a. non native torch ops) on VeScale Dtenors."
                "To re-enable plug-in ops, try to install the modified PyTorch."
            )
            _OK_TO_USE_DATA_PTR = False
    return torch.Tensor._make_wrapper_subclass(*args, **kwargs)


class DTensor(torch.Tensor):
    _local_tensor: torch.Tensor
    _spec: DTensorSpec
    __slots__ = ["_local_tensor", "_spec"]

    # class attribute that handles operator placements propagation
    # rules, keyed by aten op name, value is propagation func
    _propagator: ShardingPropagator = ShardingPropagator()

    @staticmethod
    def __new__(
        cls,
        local_tensor: torch.Tensor,
        device_mesh: DeviceMesh,
        placements: Tuple[Placement, ...],
        *,
        shape: torch.Size,
        dtype: torch.dtype,
        requires_grad: bool,
        stride: Tuple[int, ...],
    ) -> "DTensor":
        """
        Construct a DTensor from a local tensor, device mesh, and placement and
        other tensor properties (i.e. shape, requires_grad, strides, etc).
        Note: This is not a public API and it's only supposed to be used by the
            operator implementations and internals. If you want to construct a
            DTensor from a local tensor, consider using `DTensor.from_local`, if
            you want to construct a DTensor from a "global" tensor (where you
            already have tensor initialized and want to shard this tensor),
            consider using `distribute_tensor`.
        """
        if requires_grad != local_tensor.requires_grad:
            warnings.warn(
                "To construct DTensor from torch.Tensor, it's recommended to "
                "use local_tensor.detach() and make requires_grad consistent."
            )

        # new method instruct wrapper tensor from local_tensor and add
        # placement spec, it does not do actual distribution
        r = _dispatch_torch_make_wrapper_subclass(
            cls,
            shape,
            strides=stride,
            dtype=dtype,
            device=local_tensor.device,
            layout=local_tensor.layout,
            requires_grad=requires_grad,
            data_ptr=local_tensor.data_ptr(),
        )

        tensor_meta = TensorMeta(shape, stride, dtype)
        # deepcopy and set spec
        r._spec = DTensorSpec(device_mesh, tuple(placements), tensor_meta=tensor_meta)
        r._local_tensor = local_tensor
        return r

    def __repr__(self):
        # TODO: consider all_gather the local tensors for better debugging
        return f"DTensor(local_tensor={self._local_tensor}, device_mesh={self._spec.mesh}, placements={self._spec.placements})"

    def __tensor_flatten__(self):
        """
        protocol to inform how to flatten a DTensor to local tensor
        for PT2 tracing
        """
        return ["_local_tensor"], (self._spec, self.requires_grad)

    @staticmethod
    def __tensor_unflatten__(inner_tensors, flatten_spec, outer_size, outer_stride):
        assert flatten_spec is not None, "Expecting spec to be not None from `__tensor_flatten__` return value!"
        local_tensor = inner_tensors["_local_tensor"]
        spec, requires_grad = flatten_spec
        return DTensor(
            local_tensor,
            spec.mesh,
            spec.placements,
            shape=outer_size,
            dtype=spec.tensor_meta.dtype,
            requires_grad=requires_grad,
            stride=outer_stride,
        )

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        return op_dispatch.operator_dispatch(
            func,
            args,
            kwargs or {},
            DTensor._propagator,
        )

    def full_tensor(self, *, grad_placements: Optional[Sequence[Placement]] = None) -> torch.Tensor:
        """
        Return the full tensor of this DTensor. It will perform necessary collectives
        to gather the local tensors from other ranks in its DeviceMesh and concatenate
        them together. It's a syntatic sugar of the following code:
        `dtensor.redistribute(placements=[Replicate()] * mesh.ndim).to_local()`
        Keyword args:
            grad_placements (List[:class:`Placement`], optional): the placements describes
                the future layout of any gradient layout of the full Tensor returned from this
                function.
                `full_tensor` converts DTensor to a full torch.Tensor and the returned torch.tensor
                might not be used as the original replicated DTensor layout later in the code. This
                argument is the hint that user can give to autograd in case the gradient
                layout of the returned tensor does not match the original replicated DTensor layout.
                If not specified, we will assume the gradient layout of the full tensor be replicated.
        Returns:
            A :class:`torch.Tensor` object that represents the full tensor of this DTensor.
        .. note:: `full_tensor` is differentiable.
        """
        from vescale.dtensor.api import to_local

        redist_res = self.redistribute(placements=[Replicate()] * self.device_mesh.ndim)
        return to_local(redist_res, grad_placements=grad_placements, async_output=False)

    @property
    def device_mesh(self) -> DeviceMesh:
        """
        The :class:`DeviceMesh` attribute that associates with this DTensor object.

        .. note:: device_mesh is a read-only property, it can not be set.
        """
        return self._spec.mesh

    @property
    def placements(self) -> Sequence[Placement]:
        """
        The placements attribute of this DTensor that describes the layout of this
        DTensor on the its DeviceMesh.

        .. note:: placements is a read-only property, it can not be set.
        """
        return self._spec.placements

    @staticmethod
    def from_local(
        local_tensor: torch.Tensor,
        device_mesh: Optional[DeviceMesh] = None,
        placements: Optional[Sequence[Placement]] = None,
        *,
        run_check: bool = True,
        shape: Optional[torch.Size] = None,
        stride: Optional[Tuple[int, ...]] = None,
        async_input: bool = True,
    ) -> "DTensor":
        # we have to do this to avoid circle import.
        from vescale.dtensor.api import from_local

        return from_local(
            local_tensor,
            device_mesh,
            placements,
            run_check=run_check,
            shape=shape,
            stride=stride,
            async_input=async_input,
        )

    def to_local(
        self,
        *,
        grad_placements: Optional[Sequence[Placement]] = None,
        async_output: bool = True,
    ) -> torch.Tensor:
        from vescale.dtensor.api import to_local

        return to_local(self, grad_placements=grad_placements, async_output=async_output)

    def redistribute(
        self,
        device_mesh: Optional[DeviceMesh] = None,
        placements: Optional[Sequence[Placement]] = None,
        async_op: bool = True,
    ) -> "DTensor":
        from vescale.dtensor.api import redistribute_dtensor

        return redistribute_dtensor(self, device_mesh=device_mesh, placements=placements, async_op=async_op)

    def requires_grad_(self, mode=True):
        self._local_tensor.requires_grad_(mode)
        return super().requires_grad_(mode)

    def retain_grad(self) -> None:
        self._local_tensor.retain_grad()
        return super().retain_grad()
