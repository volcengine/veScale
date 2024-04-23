# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchdistx.fake import fake_mode, is_fake, meta_like


def test_fake_mode_returns_cuda_tensor_if_fake_cuda_is_true() -> None:
    if torch.cuda.is_available():
        pytest.skip("Can only be tested if CUDA is not available.")

    with fake_mode(fake_cuda=True):
        a = torch.ones([10], device="cuda")

    assert a.device.type == "cuda"


def test_fake_mode_raises_error_if_fake_cuda_is_false() -> None:
    if torch.cuda.is_available():
        pytest.skip("Can only be tested if CUDA is not available.")

    with pytest.raises((AssertionError, RuntimeError)):
        with fake_mode():
            torch.ones([10], device="cuda")


def test_cuda_tensor_raises_error_after_fake_mode() -> None:
    if torch.cuda.is_available():
        pytest.skip("Can only be tested if CUDA is not available.")

    with fake_mode(fake_cuda=True):
        torch.ones([10], device="cuda")

    with pytest.raises((AssertionError, RuntimeError)):
        torch.ones([10], device="cuda")


def test_meta_like_returns_meta_tensor() -> None:
    with fake_mode():
        a = torch.ones([10])

    b = meta_like(a)

    assert not is_fake(b)
    assert b.device.type == "meta"
    assert b.dtype == a.dtype
    assert b.size() == a.size()
    assert b.stride() == a.stride()


def test_meta_like_raises_error_if_tensor_is_not_fake() -> None:
    a = torch.ones([10])

    with pytest.raises(ValueError):
        meta_like(a)
