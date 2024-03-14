################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

from typing import Any, Optional, Sequence, Tuple, Union

# Import all builtin dist tensor ops
import torch

import vescale.dtensor.random as random
from vescale.dtensor._utils import compute_local_shape, is_zero_out_local_shard
from vescale.dtensor.device_mesh import DeviceMesh, mesh_resources
from vescale.dtensor.api import normalize_placements
from vescale.dtensor.dtensor import DTensor
from vescale.dtensor.ops.utils import normalize_to_torch_size
from vescale.dtensor.placement_types import DTensorSpec, Placement, Replicate, TensorMeta


def _dtensor_init_helper(
    init_op,
    global_shape: Union[Tuple[int], Tuple[Sequence[int]]],
    *,
    dtype: Optional[torch.dtype],
    layout: torch.layout,
    requires_grad: bool,
    device_mesh: Optional[DeviceMesh],
    placements: Optional[Sequence[Placement]],
    fill_value: Optional[Any] = None,
) -> DTensor:
    # parse args
    global_shape = normalize_to_torch_size(global_shape)
    assert layout == torch.strided, f"layout={layout} is not supported!"
    torch_stride = torch._prims_common.make_contiguous_strides_for(global_shape)
    # get device_mesh
    device_mesh = device_mesh or mesh_resources.get_current_mesh()
    device = device_mesh.device_type
    # get placements
    placements: Tuple[Placement] = normalize_placements(placements, device_mesh.ndim, none_as_replicate=True)
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

        tensor_meta = TensorMeta(global_shape, (0,), dtype)
        spec = DTensorSpec(device_mesh, placements, tensor_meta=tensor_meta)

        assert random.is_rng_supported_mesh(
            device_mesh
        ), "currently, random DTensor only support cuda/cuda=like device!"
        if not random._rng_tracker:
            random._rng_tracker = random.OffsetBasedRNGTracker()
        assert random._rng_tracker is not None
        with random._rng_tracker._distribute_region(spec):
            local_tensor = init_op(local_shape, dtype=dtype, layout=layout, device=device, requires_grad=requires_grad)
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
    *size,
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
        size (int...): a sequence of integers defining the global shape of the output :class:`DTensor`.
            Can be a variable number of arguments or a collection like a list or tuple.
            E.g.: zeros(1,2,3..) or zeros([1,2,3..]) or zeros((1,2,3..))
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
    *size,
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
        size (int...): a sequence of integers defining the global shape of the output :class:`DTensor`.
            Can be a variable number of arguments or a collection like a list or tuple.
            E.g.: ones(1,2,3..) or ones([1,2,3..]) or ones((1,2,3..))

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
    *size,
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
        size (int...): a sequence of integers defining the global shape of the output :class:`DTensor`.
            Can be a variable number of arguments or a collection like a list or tuple.
            E.g.: empty(1,2,3..) or empty([1,2,3..]) or empty((1,2,3..))

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
    size,
    fill_value,
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
        size (int...): a sequence of integers defining the global shape of the output :class:`DTensor`.
            Can be a variable number of arguments or a collection like a list or tuple.
            E.g.: full(1,2,3..) or full([1,2,3..]) or full((1,2,3..))
        fill_value (Scalar): the value to fill the output tensor with.

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
    *size,
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
        size (int...): a sequence of integers defining the global shape of the output :class:`DTensor`.
            Can be a variable number of arguments or a collection like a list or tuple.
            E.g.: randn(1,2,3..) or randn([1,2,3..]) or randn((1,2,3..))
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


if not torch._running_with_deploy():
    import vescale.dtensor._dynamo_utils
