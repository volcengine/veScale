# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Dict, Optional, TypeVar, Union

from torch import Tensor
from torch.nn import Module

# We import `fake` to monkey-patch `repr()` of `Tensor`.
from . import fake  # noqa: F401
from . import _C

T = TypeVar("T", bound=Module)


def deferred_init(module_fn: Callable[..., T], *args, **kwargs) -> T:
    """Defers the initialization of a ``Module``.

    This function forces all tensors constructed within ``module_fn`` to be
    fake while also recording all operations performed on them. The modules
    and tensors returned from ``module_fn`` can later be instantiated using
    the :func:`materialize_tensor` and :func:`materialize_module` functions.

    Args:
        module_fn:
            A callable that takes arbitrary number of arguments and returns a
            ``Module`` instance.
        args, kwargs:
            The positional and keyword arguments to be passed to ``module_fn``.

    .. Warning::
        The operations performed on the parameters and buffers of a module will
        only be recorded while inside ``deferred_init()``. Avoid making changes
        to a module after its returned from ``deferred_init()``; otherwise it
        cannot be correctly materialized.
    """
    _C.enter_deferred_init()
    try:
        return module_fn(*args, **kwargs)
    finally:
        _C.leave_deferred_init()


def is_deferred(obj: Union[Tensor, Module]) -> bool:
    """Indicates whether the provided tensor or module has been constructed in
    a deferred-init context.

    Args:
        obj:
            A ``Tensor`` or ``Module`` instance.
    """
    if isinstance(obj, Tensor):
        return _C.can_materialize(obj)

    if isinstance(obj, Module):
        for prm in obj.parameters():
            if _C.can_materialize(prm):
                return True

        for buf in obj.buffers():
            if _C.can_materialize(buf):
                return True

        return False

    raise ValueError("`obj` must be of type `Tensor` or `Module`.")


def materialize_tensor(tensor: Tensor) -> Tensor:
    """Materializes ``tensor``.

    Args:
        tensor:
            The tensor instance to materialize.

    .. Warning::
        Once materialized a fake tensor will hold a reference to its
        materialized version. In order to avoid memory leaks make sure to
        dispose it when it is no longer required.
    """
    return _C.materialize_tensor(tensor)


def materialize_module(
    module: Module,
    buffers_only: bool = False,
    check_fn: Optional[Callable[[Module], bool]] = None,
) -> None:
    """Materializes ``module`` and its descendant modules.

    Args:
        module:
            The module instance to materialize.
        buffers_only:
            A boolean value indicating whether to materialize the buffer tensors
            only.
        check_fn:
            An optional callable which takes a ``Module`` instance and returns a
            boolean value indicating whether to materialize it.
    """

    def materialize_tensors(tensors: Dict[str, Optional[Tensor]]) -> None:
        for key, tensor in tensors.items():
            if tensor is None:
                continue

            try:
                tensors[key] = _C.materialize_tensor(tensor)
            except ValueError:
                raise ValueError(f"'{key}' has already been materialized.") from None

    # Materialize the child modules recursively.
    for m in module.children():
        materialize_module(m, buffers_only, check_fn)

    # Materialize this module, possibly based on a check.
    if check_fn is None or check_fn(module):
        if not buffers_only:
            materialize_tensors(module._parameters)  # type: ignore[arg-type]

        materialize_tensors(module._buffers)
