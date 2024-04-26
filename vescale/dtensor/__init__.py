################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

from numbers import Number
import math
from typing import Any, Optional, Sequence, Tuple, Union

# Import all builtin dist tensor ops
import torch

import vescale.dtensor.random as random
from vescale.dtensor._utils import (
    compute_local_shape,
    compute_local_shape_and_global_offset,
    is_zero_out_local_shard,
    equal,
    allclose,
)
from vescale.dtensor.device_mesh import DeviceMesh, mesh_resources
from vescale.dtensor.api import normalize_placements
from vescale.dtensor.dtensor import DTensor
from vescale.dtensor.ops.utils import normalize_to_torch_size
from vescale.dtensor.placement_types import DTensorSpec, Placement, Replicate, TensorMeta

__all__ = ["zeros", "ones", "empty", "full", "randn", "arange", "equal", "allclose"]


def _dtensor_init_helper(
    init_op,
    global_shape: Union[Sequence[int], Tuple[Sequence[int]], torch.Size, Tuple[torch.Size]],
    *,
    dtype: Optional[torch.dtype],
    layout: torch.layout,
    requires_grad: bool,
    device_mesh: Optional[DeviceMesh],
    placements: Optional[Sequence[Placement]],
    fill_value: Optional[Number] = None,
    arange_start: Optional[Number] = None,
    arange_end: Optional[Number] = None,
    arange_step: Optional[Number] = None,
) -> DTensor:
    # parse args
    global_shape = normalize_to_torch_size(global_shape)
    assert layout == torch.strided, f"layout={layout} is not supported!"
    torch_stride = torch._prims_common.make_contiguous_strides_for(global_shape)
    # get device_mesh
    device_mesh = device_mesh or mesh_resources.get_current_mesh()
    device = device_mesh.device_type
    # get placements
    placements: Tuple[Placement] = normalize_placements(
        placements, device_mesh.ndim, tensor_ndim=len(global_shape), none_as_replicate=True
    )
    # get local tensor shape
    local_shape = compute_local_shape(global_shape, device_mesh, placements)
    # initialize the local tensor
    if len(local_shape) == 0:
        local_tensor = torch.empty(0, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)
    elif init_op in (torch.zeros, torch.ones, torch.empty):
        local_tensor = init_op(local_shape, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)
    elif init_op == torch.full:
        local_tensor = torch.full(
            local_shape, fill_value, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad
        )
    elif init_op == torch.rand or init_op == torch.randn:
        # this tensor meta is not used except `shape`
        dtype = torch.get_default_dtype() if dtype is None else dtype

        tensor_meta = TensorMeta(global_shape, torch_stride, dtype)
        spec = DTensorSpec(device_mesh, placements, tensor_meta=tensor_meta)

        assert random.is_rng_supported_mesh(
            device_mesh
        ), "currently, random DTensor only support cuda/cuda=like device!"
        if not random._rng_tracker:
            random._rng_tracker = random.init_vescale_rng_tracker()
        assert random._rng_tracker is not None
        with random._rng_tracker._distribute_region(spec):
            local_tensor = init_op(local_shape, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)
    elif init_op == torch.arange:
        # get local tensor shape and its offset in the global tensor
        local_shape, global_offset = compute_local_shape_and_global_offset(global_shape, device_mesh, placements)
        # initialize the local tensor
        local_start = arange_start + global_offset[0] * arange_step
        local_end = local_start + local_shape[0] * arange_step
        local_tensor = torch.arange(
            local_start, local_end, arange_step, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad
        )
    else:
        raise NotImplementedError

    # handle Partial case
    if is_zero_out_local_shard(device_mesh, placements):
        with torch.no_grad():
            local_tensor.zero_()
        assert local_tensor.requires_grad == requires_grad

    return DTensor(
        local_tensor=local_tensor,
        device_mesh=device_mesh,
        placements=placements,
        shape=global_shape,
        dtype=local_tensor.dtype,
        requires_grad=requires_grad,
        stride=torch_stride,
    )


def zeros(
    *size: Union[int, Sequence[int], torch.Size],
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    requires_grad: bool = False,
    device_mesh: Optional[DeviceMesh] = None,
    placements: Optional[Sequence[Placement]] = None,
) -> DTensor:
    """
    Returns a :class:`DTensor` filled with the scalar value 0, with global shape ``size``.
    It will be on device type of device mesh; presetting default cuda rank is a must.

    Args:
        size (int or Sequence[int] or torch.Size): a sequence of integers defining the global shape of the output :class:`DTensor`.
            Can be a variable number of arguments or a collection like a list or tuple or a torch.Size.
            E.g.: zeros(1,2,3..) or zeros([1,2,3..]) or zeros((1,2,3..)) or zeros(torch.Size([1, 2]))

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned :class:`DTensor`.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
        layout (:class:`torch.layout`, optional): the desired layout of returned :class:`DTensor`.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned :class:`DTensor`. Default: ``False``.
        device_mesh: :class:`DeviceMesh` type, contains the mesh info of ranks
        placements: a sequence of :class:`Placement` type: ``Shard``, ``Replicate``, ``Partial``

    Returns:
        A :class:`DTensor` object on each rank
    """
    return _dtensor_init_helper(
        torch.zeros,
        size,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
        device_mesh=device_mesh,
        placements=placements,
    )


def ones(
    *size: Union[int, Sequence[int], torch.Size],
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    requires_grad: bool = False,
    device_mesh: Optional[DeviceMesh] = None,
    placements: Optional[Sequence[Placement]] = None,
) -> DTensor:
    """
    Returns a :class:`DTensor` filled with the scalar value 1, with global shape ``size``.
    It will be on device type of device mesh; presetting default cuda rank is a must.

    Args:
        size (int or Sequence[int] or torch.Size): a sequence of integers defining the global shape of the output :class:`DTensor`.
            Can be a variable number of arguments or a collection like a list or tuple or a torch.Size.
            E.g.: ones(1,2,3..) or ones([1,2,3..]) or ones((1,2,3..)) or ones(torch.Size([1, 2]))

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned :class:`DTensor`.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
        layout (:class:`torch.layout`, optional): the desired layout of returned DTensor.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned :class:`DTensor`. Default: ``False``.
        device_mesh: :class:`DeviceMesh` type, contains the mesh info of ranks
        placements: a sequence of :class:`Placement` type: ``Shard``, ``Replicate``, ``Partial``

    Returns:
        A :class:`DTensor` object on each rank
    """
    return _dtensor_init_helper(
        torch.ones,
        size,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
        device_mesh=device_mesh,
        placements=placements,
    )


def empty(
    *size: Union[int, Sequence[int], torch.Size],
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    requires_grad: bool = False,
    device_mesh: Optional[DeviceMesh] = None,
    placements: Optional[Sequence[Placement]] = None,
) -> DTensor:
    """
    Returns a :class:`DTensor` filled with uninitialized data, with global shape ``size``.
    It will be on device type of device mesh; presetting default cuda rank is a must.

    Args:
        size (int or Sequence[int] or torch.Size): a sequence of integers defining the global shape of the output :class:`DTensor`.
            Can be a variable number of arguments or a collection like a list or tuple or a torch.Size.
            E.g.: empty(1,2,3..) or empty([1,2,3..]) or empty((1,2,3..)) or empty(torch.Size([1, 2]))

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned :class:`DTensor`.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).\
        layout (:class:`torch.layout`, optional): the desired layout of returned :class:`DTensor`.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned :class:`DTensor`. Default: ``False``.
        device_mesh: :class:`DeviceMesh` type, contains the mesh info of ranks
        placements: a sequence of :class:`Placement` type: ``Shard``, ``Replicate``, ``Partial``

    Returns:
        A :class:`DTensor` object on each rank
    """
    return _dtensor_init_helper(
        torch.empty,
        size,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
        device_mesh=device_mesh,
        placements=placements,
    )


def full(
    size: Union[Sequence[int], torch.Size],
    fill_value: Number,
    *,
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    requires_grad: bool = False,
    device_mesh: Optional[DeviceMesh] = None,
    placements: Optional[Sequence[Placement]] = None,
) -> DTensor:
    """
    Returns a :class:`DTensor` of global shape ``size`` filled with ``fill_value``. The scalar value type should match
        ``device_mesh.device_type``.
    It will be on device type of device mesh; presetting default cuda rank is a must.

    Args:
        size (Sequence[int] or torch.Size): a sequence of integers defining the global shape of the output :class:`DTensor`.
            Can be a collection like a list or tuple or torch.Size.
            E.g.: full([1,2,3..]) or full((1,2,3..)) or full(torch.Size([1, 2]))
        fill_value (Number): the value to fill the output tensor with.

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned :class:`DTensor`.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
        layout (:class:`torch.layout`, optional): the desired layout of returned DTensor.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned :class:`DTensor`. Default: ``False``.
        device_mesh: :class:`DeviceMesh` type, contains the mesh info of ranks.
        placements: a sequence of :class:`Placement` type: ``Shard``, ``Replicate``, ``Partial``

    Returns:
        A :class:`DTensor` object on each rank
    """
    return _dtensor_init_helper(
        torch.full,
        size,
        fill_value=fill_value,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
        device_mesh=device_mesh,
        placements=placements,
    )


def randn(
    *size: Union[int, Sequence[int], torch.Size],
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    requires_grad: bool = False,
    device_mesh: Optional[DeviceMesh] = None,
    placements: Optional[Sequence[Placement]] = None,
) -> DTensor:
    """
    Returns a :class:`DTensor` filled with random numbers from a normal distribution
        with mean 0 and variance 1. The global shape of the tensor is defined by the variable
        argument ``size``.
    It will be on device type of device mesh; presetting default cuda rank is a must.

    Args:
        size (int or Sequence[int] or torch.Size): a sequence of integers defining the global shape of the output :class:`DTensor`.
            Can be a variable number of arguments or a collection like a list or tuple or a torch.Size.
            E.g.: randn(1,2,3..) or randn([1,2,3..]) or randn((1,2,3..)) or randn(torch.Size([1, 2]))

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned :class:`DTensor`.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
        layout (:class:`torch.layout`, optional): the desired layout of returned DTensor.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned :class:`DTensor`. Default: ``False``.
        device_mesh: :class:`DeviceMesh` type, contains the mesh info of ranks.
        placements: a sequence of :class:`Placement` type: ``Shard``, ``Replicate``, ``Partial``

    Returns:
        A :class:`DTensor` object on each rank
    """
    return _dtensor_init_helper(
        torch.randn,
        size,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
        device_mesh=device_mesh,
        placements=placements,
    )


def rand(
    *size: Union[int, Sequence[int], torch.Size],
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    requires_grad: bool = False,
    device_mesh: Optional[DeviceMesh] = None,
    placements: Optional[Sequence[Placement]] = None,
) -> DTensor:
    """
    Returns a :class:`DTensor` filled with random numbers from an uniform distribution
        on the interval [0, 1]. The global shape of the tensor is defined by the variable
        argument ``size``.
    It will be on device type of device mesh; presetting default cuda rank is a must.

    Args:
        size (int or Sequence[int] or torch.Size): a sequence of integers defining the global shape of the output :class:`DTensor`.
            Can be a variable number of arguments or a collection like a list or tuple or a torch.Size.
            E.g.: randn(1,2,3..) or randn([1,2,3..]) or randn((1,2,3..)) or randn(torch.Size([1, 2]))

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned :class:`DTensor`.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).
        layout (:class:`torch.layout`, optional): the desired layout of returned DTensor.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned :class:`DTensor`. Default: ``False``.
        device_mesh: :class:`DeviceMesh` type, contains the mesh info of ranks.
        placements: a sequence of :class:`Placement` type: ``Shard``, ``Replicate``, ``Partial``

    Returns:
        A :class:`DTensor` object on each rank
    """
    return _dtensor_init_helper(
        torch.rand,
        size,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
        device_mesh=device_mesh,
        placements=placements,
    )


def arange(
    *start_end_step: Number,
    dtype: Optional[torch.dtype] = None,
    layout: torch.layout = torch.strided,
    requires_grad: bool = False,
    device_mesh: Optional[DeviceMesh] = None,
    placements: Optional[Sequence[Placement]] = None,
) -> DTensor:
    # NOTE: always out=None
    """
    Returns a 1-D tensor of size `ceil((end - start) / step)` with values from the interval [start, end)
    taken with common difference step beginning from start.

    Note that non-integer step is subject to floating point rounding errors when comparing against end;
    to avoid inconsistency, we advise subtracting a small epsilon from end in such cases.

    Args:
        start (Number) - the starting value for the set of points. Default: 0.
        end (Number) - the ending value for the set of points
        step (Number) - the gap between each pair of adjacent points. Default: 1.

    Keyword args:
        dtype (torch.dtype, optional) - the desired data type of returned tensor. Default: if None, uses a global default (see torch.set_default_dtype()). If dtype is not given, infer the data type from the other input arguments. If any of start, end, or stop are floating-point, the dtype is inferred to be the default dtype, see get_default_dtype(). Otherwise, the dtype is inferred to be torch.int64.
        layout (torch.layout) - the desired layout of returned Tensor. Default: torch.strided.
        requires_grad (bool) - If autograd should record operations on the returned tensor. Default: False.
        device_mesh: :class:`DeviceMesh` type, contains the mesh info of ranks.
        placements: a sequence of :class:`Placement` type: ``Shard``, ``Replicate``, ``Partial``

    Returns:
        A :class:`DTensor` object on each rank
    """

    if len(start_end_step) == 0:
        raise ValueError("`arange` should take at least one positional arg!")
    elif len(start_end_step) == 1:
        start, end, step = 0, start_end_step[0], 1
    elif len(start_end_step) == 2:
        start, end, step = start_end_step[0], start_end_step[1], 1
    elif len(start_end_step) == 3:
        start, end, step = start_end_step
    else:
        raise ValueError("`arange` should take at most three positional args!")

    size = math.ceil((end - start) / step)
    return _dtensor_init_helper(
        torch.arange,
        size,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
        device_mesh=device_mesh,
        placements=placements,
        arange_start=start,
        arange_end=end,
        arange_step=step,
    )


if not torch._running_with_deploy():
    import vescale.dtensor._dynamo_utils
