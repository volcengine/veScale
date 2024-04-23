# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast

import torch
from torch import Tensor
from torch.nn import Module, Parameter

from torchdistx.deferred_init import (
    deferred_init,
    is_deferred,
    materialize_module,
    materialize_tensor,
)


def test_materialize_tensor_is_noop_for_real_tensors() -> None:
    a = torch.ones([10])

    e = materialize_tensor(a)

    assert a is e


def test_materialize_tensor_returns_same_tensor() -> None:
    class FooModule(Module):
        def __init__(self):
            super().__init__()

            self.param1 = Parameter(torch.ones([5]))
            self.param2 = self.param1

    module = deferred_init(FooModule)

    a = materialize_tensor(cast(Tensor, module.param1))
    b = materialize_tensor(cast(Tensor, module.param1))
    c = materialize_tensor(cast(Tensor, module.param2))

    assert a is b
    assert a is c


def test_is_deferred_returns_right_value() -> None:
    class FooModule(Module):
        def __init__(self):
            super().__init__()

            self.param1 = Parameter(torch.ones([5]))
            self.param2 = Parameter(torch.ones([5]))

    module = FooModule()

    assert not is_deferred(module)

    module = deferred_init(FooModule)

    assert is_deferred(module)

    materialize_module(module)

    assert not is_deferred(module)

    module = deferred_init(FooModule)

    module.param1 = materialize_tensor(module.param1)

    assert is_deferred(module)

    module.param2 = materialize_tensor(module.param2)

    assert not is_deferred(module)
