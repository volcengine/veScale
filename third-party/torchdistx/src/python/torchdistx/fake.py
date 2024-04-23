# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import contextmanager
from typing import Callable, Generator

import torch

from . import _C


# Since the `repr()` method of `Tensor` is not extensible we monkey-patch it
# to support fake tensors.
def _patch_tensor_repr() -> Callable[[torch.Tensor], str]:
    tensor_repr = torch.Tensor.__repr__

    def patched_repr(tensor: torch.Tensor) -> str:
        if _C.is_fake(tensor):
            s = f"tensor(..., size={tuple(tensor.shape)}"

            if tensor.dtype != torch.get_default_dtype():
                s += f", dtype={tensor.dtype}"

            if tensor.device.type != "cpu":
                s += f", device={tensor.device}"

            if tensor.requires_grad:
                s += ", requires_grad=True"

            return s + ", fake=True)"
        else:
            return tensor_repr(tensor)

    return patched_repr


torch.Tensor.__repr__ = _patch_tensor_repr()  # type: ignore[assignment]


@contextmanager
def fake_mode(*, fake_cuda: bool = False) -> Generator:
    """Instantiates all tensors within its context as fake.

    Args:
        fake_cuda:
            If ``True``, allows constructing fake CUDA tensors even if CUDA is
            not available. Ignored if CUDA is already available.
    """
    _C.enter_fake_mode(fake_cuda)
    try:
        yield
    finally:
        _C.leave_fake_mode()


def is_fake(tensor: torch.Tensor) -> bool:
    """Indicates whether ``tensor`` is fake.

    Args:
        tensor:
            The tensor to check.
    """
    return _C.is_fake(tensor)


def meta_like(fake: torch.Tensor) -> torch.Tensor:
    """Returns a meta tensor with the same properties as ``fake``.

    This function has the same Autograd behavior as ``detach()`` meaning the
    returned tensor won't be part of the Autograd graph.

    Args:
        fake:
            The fake tensor to copy from.
    """
    try:
        return _C.meta_like(fake)
    except ValueError:
        raise ValueError("`fake` was expected to be a fake tensor.")
