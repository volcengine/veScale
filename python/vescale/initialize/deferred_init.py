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
from typing import Dict, Optional, Sequence, Tuple, Union, Any, Callable

import torch
from torch import nn

from torchdistx.deferred_init import deferred_init as _deferred_init
from torchdistx.deferred_init import is_deferred as _is_deferred
from torchdistx.deferred_init import _C

from vescale.dtensor.device_mesh import DeviceMesh
import vescale.dtensor.random as random
from vescale.dtensor._utils import compute_local_shape
from vescale.dtensor.device_mesh import mesh_resources
from vescale.dtensor.dtensor import DTensor
from vescale.dtensor.placement_types import DTensorSpec, Placement, TensorMeta
from vescale.dtensor.api import distribute_tensor, normalize_placements

__all__ = ["deferred_init", "is_deferred", "materialize_dtensor", "materialize_dparameter"]


def deferred_init(module_fn: Callable, *args, **kwargs) -> Union[torch.Tensor, nn.Module]:
    """Defers the initialization of a ``torch.Tensor`` or ``nn.Module``.

    This function forces all tensors constructed within ``module_fn`` to be
    fake while also recording all operations performed on them. The modules
    and tensors returned from ``module_fn`` can later be instantiated using
    the :func:`materialize_dtensor` and :func:`parallelize_module` functions.

    Args:
        module_fn:
            A factory function or module function that takes arbitrary number of arguments
            and returns a ``torch.Tensor`` or ``nn.Module`` instance.
        args, kwargs:
            The positional and keyword arguments to be passed to ``module_fn``.

    .. Warning::
        The operations performed will only be recorded while inside ``deferred_init()``.
        Avoid making changes to a torch.Tensor after its returned from ``deferred_init()``;
        otherwise it cannot be correctly materialized.
    """
    return _deferred_init(module_fn, *args, **kwargs)


def is_deferred(obj: Union[torch.Tensor, nn.Parameter, nn.Module]) -> bool:
    """Indicates whether the provided object has been constructed in a deferred-init context,
        until being materialized.

    Args:
        obj:
            A ``torch.Tensor`` or ``nn.Parameter`` or ``nn.Module`` instance.
    """
    if isinstance(obj, DTensor):
        warnings.warn(
            "`is_deferred` takes a `DTensor`! deferring a `DTensor` itself might be not supported.", UserWarning
        )
    from vescale.dmodule.api import is_dmodule

    if is_dmodule(obj):
        warnings.warn(
            "`is_deferred` takes a `DModule`! deferring a `DModule` itself might be not supported.", UserWarning
        )
    return _is_deferred(obj)


def materialize_dtensor(
    tensor: torch.Tensor,
    device_mesh: Optional[DeviceMesh] = None,
    placements: Optional[Sequence[Placement]] = None,
) -> DTensor:
    """Materializes ``DTensor`` from global fake `torch.Tensor`.

    This function is logically equal to :func:`distribute_tensor`, but with physical differences below:
    - `distribute_tensor` takes a global REAL `torch.Tensor` and shard it into `DTensor`.
    - `materialize_dtensor` takes a global FAKE `torch.Tensor` and materialize the REAL local shard on device.
    so `materialize_dtensor` only allocates device memory of local shard size, instead of global size.

    Args:
        tensor:
            The tensor instance to materialize.

    .. Warning::
        Once materialized a fake tensor will hold a reference to its
        materialized version. In order to avoid memory leaks make sure to
        dispose it when it is no longer required.
    """
    assert not isinstance(tensor, nn.Parameter), "`materialize_dtensor` does not take `Parameter`!"

    if isinstance(tensor, DTensor):
        warnings.warn(
            "`materialize_dtensor` takes a `DTensor`! deferring a `DTensor` itself might be not supported.", UserWarning
        )
        return tensor

    if not _is_deferred(tensor):
        return distribute_tensor(tensor, device_mesh, placements)

    # parse args
    global_shape = tensor.shape
    assert tensor.layout == torch.strided, f"layout={tensor.layout} is not supported!"
    torch_stride = torch._prims_common.make_contiguous_strides_for(global_shape)
    # get device_mesh
    device_mesh = device_mesh or mesh_resources.get_current_mesh()
    device = device_mesh.device_type
    # get placements
    placements: Tuple[Placement] = normalize_placements(placements, device_mesh.ndim, none_as_replicate=True)
    # get local tensor shape
    local_shape = compute_local_shape(global_shape, device_mesh, placements)
    torch_device = torch.device(device)
    # materialize local tensor
    if _C.is_gen_by_random_op(tensor):
        tensor_meta = TensorMeta(global_shape, (0,), tensor.dtype)
        spec = DTensorSpec(device_mesh, placements, tensor_meta=tensor_meta)

        assert random.is_rng_supported_mesh(
            device_mesh
        ), "currently, random DTensor only support cuda/cuda=like device!"
        if not random._rng_tracker:
            random._rng_tracker = random.OffsetBasedRNGTracker()
        assert random._rng_tracker is not None
        with random._rng_tracker._distribute_region(spec):
            tensor = _C.materialize_tensor_with_local_shape(tensor, local_shape, torch_device)
    else:
        tensor = _C.materialize_tensor_with_local_shape(tensor, local_shape, torch_device)
    # wrap as dtensor
    return DTensor(
        local_tensor=tensor,
        device_mesh=device_mesh,
        placements=placements,
        shape=global_shape,
        dtype=tensor.dtype,
        requires_grad=tensor.requires_grad,
        stride=torch_stride,
    )


def materialize_dparameter(
    param: nn.Parameter,
    device_mesh: Optional[DeviceMesh] = None,
    placements: Optional[Sequence[Placement]] = None,
) -> nn.Parameter:
    """Same as `materialize_dtensor`"""  # TODO: unify both function.
    assert isinstance(param, nn.Parameter), "`materialize_dparameter` only takes `Parameter`!"

    if isinstance(param, DTensor) or isinstance(param.data, DTensor):
        warnings.warn(
            "`materialize_dparameter` takes a `DTensor`! deferring a `DTensor` itself might be not supported.",
            UserWarning,
        )
        return param

    if not _is_deferred(param):  # `_is_deferred(param.data)` is always False
        return nn.Parameter(distribute_tensor(param.data, device_mesh, placements), param.requires_grad)

    # parse args
    requires_grad = param.requires_grad
    global_shape = param.data.shape
    assert param.data.layout == torch.strided, f"layout={param.data.layout} is not supported!"
    torch_stride = torch._prims_common.make_contiguous_strides_for(global_shape)
    # get device_mesh
    device_mesh = device_mesh or mesh_resources.get_current_mesh()
    device = device_mesh.device_type
    # get placements
    placements: Tuple[Placement] = normalize_placements(placements, device_mesh.ndim, none_as_replicate=True)
    # get local tensor shape
    local_shape = compute_local_shape(global_shape, device_mesh, placements)
    torch_device = torch.device(device)
    # materialize local tensor
    if _C.is_gen_by_random_op(param):
        tensor_meta = TensorMeta(global_shape, (0,), param.data.dtype)
        spec = DTensorSpec(device_mesh, placements, tensor_meta=tensor_meta)

        assert random.is_rng_supported_mesh(
            device_mesh
        ), "currently, random DTensor only support cuda/cuda=like device!"
        if not random._rng_tracker:
            random._rng_tracker = random.OffsetBasedRNGTracker()
        assert random._rng_tracker is not None
        with random._rng_tracker._distribute_region(spec):
            param = _C.materialize_tensor_with_local_shape(param, local_shape, torch_device)
    else:
        param = _C.materialize_tensor_with_local_shape(param, local_shape, torch_device)
    # wrap parameter's data as dtensor
    dt = DTensor(
        local_tensor=param.data,
        device_mesh=device_mesh,
        placements=placements,
        shape=global_shape,
        dtype=param.data.dtype,
        requires_grad=param.data.requires_grad,
        stride=torch_stride,
    )
    # wrap dparameter
    return nn.Parameter(dt, requires_grad)


def _materialize_dmodule(
    module: nn.Module,
    device_mesh: Optional[DeviceMesh] = None,
    param_sharding_plan: Optional[Dict[str, Any]] = None,
    fwd_resharding_plan: Optional[Dict[str, Any]] = None,
    is_model_sharded: bool = False,
    grad_sync: Union[bool, Dict] = False,
    # TODO: enable selective materialize in future
    buffers_only: bool = False,
    check_fn: Optional[Callable[[nn.Module], bool]] = None,
):
    assert not buffers_only
    assert check_fn is None

    from vescale.dmodule.api import parallelize_module

    return parallelize_module(
        module,
        device_mesh,
        param_sharding_plan,
        fwd_resharding_plan,
        is_model_sharded,
        grad_sync,
    )
