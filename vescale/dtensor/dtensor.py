################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

import os
import warnings
from typing import Optional, Sequence, Tuple, List, Union
from numbers import Number

import torch
import torch.distributed._functional_collectives as funcol

import vescale.dtensor.dispatch as op_dispatch
from vescale.dtensor.device_mesh import DeviceMesh, mesh_resources
from vescale.dtensor.placement_types import (
    DTensorSpec,
    TensorMeta,
    Placement,
    Replicate,
    Shard,
    InterleavedShard,
    Partial,
)
from vescale.dtensor.sharding_prop import ShardingPropagator
from vescale.dtensor.redistribute import (
    Redistribute,
    redistribute_local_tensor,
)
from vescale.dtensor._utils import compute_global_tensor_info, gather_local_tensor_shape
from vescale.dtensor._collective_utils import mesh_broadcast


__all__ = ["DTensor"]

aten = torch.ops.aten

VESCALE_DISABLE_RUN_CHECK = os.environ.get("VESCALE_DISABLE_RUN_CHECK", "0") == "1"
_OK_TO_USE_DATA_PTR = True


def normalize_placements(
    placements: Optional[Sequence[Placement]], mesh_ndim: int, *, tensor_ndim: int = 0, none_as_replicate: bool = False
) -> Optional[Tuple[Placement]]:
    """
    normalize a placements to be valid.
    """
    if placements is None:
        return tuple(Replicate() for _ in range(mesh_ndim)) if none_as_replicate else None

    if len(placements) > mesh_ndim:
        raise ValueError(f"`placements` (len={len(placements)}) have larger length than `mesh_ndim` ({mesh_ndim})!")

    if len(placements) < mesh_ndim:
        warnings.warn(
            "`placements` have less elements than `mesh_ndim`!. We will postpend Replicate placement to the end.",
            UserWarning,
        )
        placements = list(placements) + [Replicate()] * (mesh_ndim - len(placements))

    for p in placements:
        if not isinstance(p, Placement):
            raise ValueError(f"Unsupported placements = {placements}!")
        if isinstance(p, (Shard, InterleavedShard)) and p.dim < 0:
            # normalize shard dim to be positive
            p.dim += tensor_ndim

    return tuple(placements)


# NOTE [Autograd interaction between torch.Tensor]
#
# The autograd functions defined below are being used by the public
# facing APIs (i.e. from_local, to_local) to ensure our DTensor
# works together with torch.Tensor within autograd engine. This
# allows DistributedTensor to exist on part of the module hierarchy
# and still able to calculate gradients across the torch.Tensor and
# DistributedTensor boundary.
# As an example, we have the a module that consists of submodules
# A, B, and C, the execution flow would be like:
#  input(torch.Tensor) -> Module A -> Module B -> Module C -> output (torch.Tensor)
#
# Suppose I only want to make Module B be a sharded module with
# DistributedTensor params, we would need to make the following
# flow to work:
#
#  input(torch.Tensor) -> Module A
#       -> DTensor input -> Sharded Module B -> DTensor output
#           -> output (torch.Tensor) -> Module C -> output (torch.Tensor)
#
# We need the conversion from Module A to DTensor input, which is
# `from_local`, and conversion from DTensor output to output, which
# is `to_local`, thus these two functions must be Autograd functions.


class _FromTorchTensor(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        device_mesh: DeviceMesh,
        placements: Tuple[Placement, ...],
        run_check: bool,
        shape: Optional[torch.Size] = None,
        stride: Optional[Tuple[int, ...]] = None,
        support_uneven: bool = True,
        async_input: bool = True,
    ) -> "DTensor":
        ctx.previous_placement = placements
        ctx.previous_device_mesh = device_mesh
        ctx.async_input = async_input

        # infer global shape and stride
        if (shape is None) != (stride is None):
            raise ValueError(
                f"Found shape:{shape}, stride:{stride}.",
                "Please pass both shape and stride at the same time!",
            )
        elif shape and stride:  # use given global shape and stride
            tensor_shape, tensor_stride = torch.Size(shape), tuple(stride)
        elif all(
            p.is_replicate() or p.is_partial() for p in placements
        ):  # for all replicate/partial tensor, infer from local tensor
            tensor_shape, tensor_stride = input.shape, input.stride()
        else:  # infer sharded global shape and stride
            if support_uneven:  # support uneven shard
                meshdim_localtensor_shape = gather_local_tensor_shape(input, device_mesh, placements, shard_only=True)
                assert meshdim_localtensor_shape is not None, "Out-of-mesh is impossible to support uneven sharding!"
                global_shape, global_stride = compute_global_tensor_info(
                    input, device_mesh, placements, meshdim_localtensor_shape
                )
            else:  # assume even shard
                global_shape, global_stride = compute_global_tensor_info(input, device_mesh, placements)
            tensor_shape, tensor_stride = torch.Size(global_shape), tuple(global_stride)

        # if global rank is not participating in the device mesh, we simply:
        # - set the local tensor to an empty tensor
        # - set global shape/stride as the global tensor
        if device_mesh.get_coordinate() is None:
            input = input.new_empty(0, requires_grad=input.requires_grad)
        # runtime checking for in-mesh ranks
        elif run_check:
            # per placement check
            for idx, placement in enumerate(placements):
                if placement.is_replicate():
                    # broadcast rank 0 tensor to all ranks
                    input = mesh_broadcast(input.contiguous(), device_mesh, mesh_dim=idx)
                elif placement.is_interleaved_shard():
                    if input.shape[placement.dim] % placement.interleaved_size != 0:
                        raise ValueError(
                            f"Tensor size at dim {placement.dim} is not divisible by {placement.interleaved_size}"
                        )
            # [conservative] global tensor_shape/tensor_stride should be the same across ranks
            # meshdim_localtensor_shape = gather_local_tensor_shape(
            #     tensor_shape, device_mesh, placements, shard_only=False
            # )
            # for stacked_local_shape in meshdim_localtensor_shape.values():
            #     assert stacked_local_shape.count(stacked_local_shape[0]) == len(
            #         stacked_local_shape
            #     ), "The global tensor shape must be the same across ranks!"

        # We want a fresh Tensor object that shares memory with the input tensor
        return DTensor(
            input.view_as(input),
            device_mesh,
            placements,
            shape=tensor_shape,
            dtype=input.dtype,
            requires_grad=input.requires_grad,
            stride=tensor_stride,
        )

    @staticmethod
    # type: ignore[override]
    def backward(ctx, grad_output: "DTensor"):
        previous_placement = ctx.previous_placement
        previous_device_mesh = ctx.previous_device_mesh
        async_input = ctx.async_input

        # reshard to the placement when creating DistributedTensor
        # so that the gradient layout matches, and we could return
        # local gradients directly
        if grad_output.placements != previous_placement:
            grad_output = Redistribute.apply(grad_output, previous_device_mesh, previous_placement, False)

        local_tensor = grad_output._local_tensor
        if not async_input and type(local_tensor) is funcol.AsyncCollectiveTensor:
            # synchronously wait for any pending collectives to get the result tensor
            local_tensor = local_tensor.trigger_wait()
            if hasattr(local_tensor, "elem"):
                local_tensor = local_tensor.elem  # type: ignore[attr-defined]

        # TODO: backward is also differentiable now, add a test
        # to test higher level gradients.
        return local_tensor.view_as(local_tensor), None, None, None, None, None, None, None


class _ToTorchTensor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: "DTensor", grad_placements: Optional[Sequence[Placement]], async_output: bool):
        ctx.dtensor_spec = input._spec
        ctx.grad_placements = grad_placements
        local_tensor = input._local_tensor
        if not async_output and type(local_tensor) is funcol.AsyncCollectiveTensor:
            # synchronously wait for any pending collectives to get the result tensor
            local_tensor = local_tensor.trigger_wait()
            if hasattr(local_tensor, "elem"):
                local_tensor = local_tensor.elem  # type: ignore[attr-defined]
        # We need to return a fresh Tensor object there as autograd metadata
        # will be inplaced into it. So we don't want to pollute the Tensor
        # object stored in the _local_tensor of this DTensor.
        return local_tensor.view_as(local_tensor)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        dtensor_spec = ctx.dtensor_spec
        mesh = dtensor_spec.mesh
        grad_placements = ctx.grad_placements
        dtensor_meta = dtensor_spec.tensor_meta

        if grad_placements is not None:
            grad_spec = DTensorSpec(mesh, grad_placements)
            grad_output = redistribute_local_tensor(grad_output, grad_spec, dtensor_spec)

        _, tensor_stride = compute_global_tensor_info(grad_output, mesh, dtensor_spec.placements)

        return (
            DTensor(
                grad_output,
                mesh,
                tuple(dtensor_spec.placements),
                shape=dtensor_meta.shape,
                dtype=dtensor_meta.dtype,
                requires_grad=grad_output.requires_grad,
                stride=tuple(tensor_stride),
            ),
            None,
            None,
        )


################ DTensor below ################


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

        # separately handle fake/functional local_tensor which errors on data_ptr access.
        try:
            local_tensor_data_ptr = local_tensor.data_ptr()
        except Exception:
            local_tensor_data_ptr = None

        r = _dispatch_torch_make_wrapper_subclass(
            cls,
            shape,
            strides=stride,
            dtype=dtype,
            device=local_tensor.device,
            layout=local_tensor.layout,
            requires_grad=requires_grad,
            data_ptr=local_tensor_data_ptr,
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

    # NOTE: these methods come from PR: https://github.com/pytorch/pytorch/pull/118670
    def __coerce_tangent_metadata__(self):
        if not any(isinstance(p, Partial) for p in self.placements):
            return self
        placements = [Replicate() if isinstance(p, Partial) else p for p in self.placements]
        return self.redistribute(device_mesh=self.device_mesh, placements=placements)

    def __coerce_same_metadata_as_tangent__(self, metadata_tensor):
        return self.redistribute(
            device_mesh=self.device_mesh,
            placements=metadata_tensor.placements,
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
        support_uneven: bool = True,
        async_input: bool = True,
    ) -> "DTensor":
        # TODO: moving impl code here for performance, as here is on the critial path but api function is less used

        assert type(local_tensor) is not DTensor
        assert type(getattr(local_tensor, "data", None)) is not DTensor

        # if same shape/dtype, no need to run_check, if not, must allgather
        # the metadatas to check the size/dtype across ranks
        # There should be no data communication unless there's replication
        # strategy, where we broadcast the replication from the first rank
        # in the mesh dimension
        device_mesh = device_mesh or mesh_resources.get_current_mesh()
        device_type = device_mesh.device_type

        # convert the local tensor to desired device base on device mesh's device_type
        if device_type != local_tensor.device.type and not local_tensor.is_meta:
            local_tensor = local_tensor.to(device_type)

        # validate placements
        placements: Tuple[Placement] = normalize_placements(
            placements, device_mesh.ndim, tensor_ndim=local_tensor.ndim, none_as_replicate=True
        )

        # TODO: fix later
        # if any(p.is_partial() for p in placements if p is not None):
        #     warnings.warn(
        #         "DTensor.from_local(.., [Partial]) has no zero-out feature yet! Use Partial with caution.", UserWarning
        #     )

        # `from_local` is differentiable, and the gradient of the dist tensor this function
        # created should flow back the gradients to the local_tensor, so we call an autograd
        # function to construct the dist tensor instead.

        if VESCALE_DISABLE_RUN_CHECK:
            run_check = False

        if device_mesh.get_coordinate() is None and support_uneven:
            warnings.warn(
                "Out-of-mesh rank uses `DTensor.from_local` under uneven sharding support, which is impossible!"
                " We set `support_uneven` as `False`!"
                " If uneven sharding does happen, out-of-mesh rank can only assume even sharding, which disgrees with in-mesh ranks!",
                UserWarning,
            )
            support_uneven = False

        return _FromTorchTensor.apply(  # pyre-ignore[16]: autograd func
            local_tensor,
            device_mesh,
            placements,
            run_check,
            shape,
            stride,
            support_uneven,
            async_input,
        )

    def to_local(
        self,
        *,
        grad_placements: Optional[Sequence[Placement]] = None,
        async_output: bool = True,
    ) -> torch.Tensor:

        # NOTE: moving impl code here for performance, as here is on the critial path but api function is NEVER used

        if grad_placements is not None:
            grad_placements: Tuple[Placement] = normalize_placements(
                grad_placements, self._spec.mesh.ndim, tensor_ndim=self.ndim
            )
        
        return _ToTorchTensor.apply(self, grad_placements, async_output)
    def redistribute(
        self,
        device_mesh: Optional[DeviceMesh] = None,
        placements: Optional[Sequence[Placement]] = None,
        async_op: bool = True,
    ) -> "DTensor":
        # NOTE: moving impl code here for performance, as here is on the critial path but api function is rarely used

        # if device_mesh is not specified, use the current device_mesh
        device_mesh = device_mesh or self._spec.mesh

        # check new placements for not specified
        if placements is None:
            raise RuntimeError("placements is needed for redistribute!")
        placements: Tuple[Placement] = normalize_placements(placements, device_mesh.ndim, tensor_ndim=self.ndim)

        return Redistribute.apply(self, device_mesh, placements, async_op)

    def requires_grad_(self, mode=True):
        self._local_tensor.requires_grad_(mode)
        return super().requires_grad_(mode)

    def retain_grad(self) -> None:
        self._local_tensor.retain_grad()
        return super().retain_grad()

    def tolist(self) -> Union[List, Number]:
        """
        Returns the dtensor as a (nested) list.
        For scalars, a standard Python number is returned, just like with item().
        Tensors are automatically moved to the CPU first if necessary.

        Note:
        - This operation is not differentiable.
        - This operation is not dispatched but a torch function.
        """
        return self._local_tensor.tolist()
