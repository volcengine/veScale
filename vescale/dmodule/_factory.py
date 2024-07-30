################################################################################
#
# Copyright 2023 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################

from typing import Callable, Optional, Tuple, Dict
import warnings
from inspect import signature
import functools
import contextlib

import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._python_dispatch import (
    _pop_mode,
    _push_mode,
    _len_torch_dispatch_stack,
    _get_current_dispatch_mode_stack,
)

from vescale import dtensor
from vescale.dtensor.placement_types import Replicate
from vescale.dtensor.device_mesh import DeviceMesh
from vescale.dmodule.placements_interface import PlacementsInterface as PI

__all__ = ["wrap_factory_mode"]

_IS_DEBUG = True

aten = torch.ops.aten

# NOTE: when this grows too complex, make this like DTensor op registration `aten_handler: Dict`
FACTORY_ATEN_DFACTORY = {
    torch.zeros: (aten.zeros.default, dtensor.zeros),
    torch.ones: (aten.ones.default, dtensor.ones),
    torch.empty: (aten.empty.memory_format, dtensor.empty),
    torch.full: (aten.full.default, dtensor.full),
    torch.randn: (aten.randn.default, dtensor.randn),
    torch.rand: (aten.rand.default, dtensor.rand),
    torch.arange: ((aten.arange.default, aten.arange.start, aten.arange.start_step), dtensor.arange),
}


class FactoryDispatchModeOn(TorchDispatchMode):
    def __init__(self, device_mesh: DeviceMesh, aten_dfactory_pi: Dict[Callable, Tuple[Callable, PI]]):
        super().__init__()
        self.device_mesh = device_mesh
        self.aten_dfactory_pi = aten_dfactory_pi

    def __torch_dispatch__(self, func: Callable, _, args: Tuple, kwargs: Optional[Dict] = None):
        if func not in self.aten_dfactory_pi:
            return func(*args, **(kwargs or {}))

        if _IS_DEBUG:
            print(f"[DEBUG] {func}: \t{args}\t\t{kwargs}")
        ## validate kwargs
        assert kwargs.get("out", None) is None, "DTensor factory does not support `out` kwarg yet!"
        assert kwargs.get("names", None) is None, "DTensor factory does not support `names` kwarg yet!"
        assert (
            not kwargs.get("pin_memory", None) or self.device_mesh.device_type != "cpu"
        ), "`pin_memory=True` is not supported for CPU device mesh yet!"
        # assert kwargs.get("memory_format", torch.contiguous_format) is torch.contiguous_format # don't care
        assert kwargs.get("generator", None) is None, "DTensor factory does not support `generator` kwarg yet!"
        ## get kwargs
        dtype = kwargs.get("dtype", None)
        layout = kwargs.get("layout", torch.strided)
        requires_grad = kwargs.get("requires_grad", False)

        ## replace aten func with dfactory
        dfactory, pi = self.aten_dfactory_pi[func]
        if dfactory == dtensor.arange:
            assert len(args) in (1, 2, 3)
            return dfactory(
                *args,
                dtype=dtype,
                layout=layout,
                requires_grad=requires_grad,
                device_mesh=self.device_mesh,
                placements=pi.placements,
            )
        elif dfactory == dtensor.full:
            assert len(args) == 2
            size, fill_value = args
            return dfactory(
                size,
                fill_value,
                dtype=dtype,
                layout=layout,
                requires_grad=requires_grad,
                device_mesh=self.device_mesh,
                placements=pi.placements,
            )
        else:
            assert len(args) == 1
            size = args[0]
            return dfactory(
                size,
                dtype=dtype,
                layout=layout,
                requires_grad=requires_grad,
                device_mesh=self.device_mesh,
                placements=pi.placements,
            )


def _provide_args(device_mesh: DeviceMesh, factory_pi: Dict[Callable, PI]) -> Dict[Callable, Tuple[Callable, PI]]:
    if not factory_pi:
        _replicate = PI.from_placements([Replicate()] * device_mesh.ndim)
        factory_pi = {f: _replicate for f in FACTORY_ATEN_DFACTORY.keys()}

    aten_dfactory_pi = {}
    for factory, pi in factory_pi.items():
        if factory not in FACTORY_ATEN_DFACTORY:
            warnings.warn(f"`{factory}` is not yet supported in dtensoring factory!", UserWarning)
            continue
        _aten, dfactory = FACTORY_ATEN_DFACTORY[factory]
        if isinstance(_aten, Tuple):
            for a in _aten:
                aten_dfactory_pi[a] = (dfactory, pi)
        else:
            aten_dfactory_pi[_aten] = (dfactory, pi)
    return aten_dfactory_pi


@contextlib.contextmanager
def FactoryDispatchModeOff():  # ref: `torch.utils._python_dispatch._disable_current_modes()`
    # --- enter ---
    saved_idx, saved_mode, saved_len = None, None, None
    # turn off the lastest mode from the top of stack
    if any(isinstance(_mode, FactoryDispatchModeOn) for _mode in _get_current_dispatch_mode_stack()):
        saved_len = _len_torch_dispatch_stack()
        _tmp_stack = []
        _idx = saved_len - 1
        while _idx >= 0:
            _mode: TorchDispatchMode = _pop_mode()
            if isinstance(_mode, FactoryDispatchModeOn):
                saved_idx, saved_mode = _idx, _mode
                break
            else:
                _tmp_stack.append(_mode)
                _idx -= 1
        while _tmp_stack:
            _push_mode(_tmp_stack.pop())
        assert _len_torch_dispatch_stack() == saved_len - 1, "Only one mode should be poped!"
    # ------
    try:
        yield
    # --- exit ---
    finally:
        # restore the saved mode at the original idx in the stack
        if saved_idx is not None:
            _tmp_stack = []
            _idx = saved_len - 1
            while _idx >= 0:
                if _idx == saved_idx:
                    _push_mode(saved_mode)
                    break
                else:
                    _tmp_stack.append(_pop_mode())
                    _idx -= 1
            while _tmp_stack:
                _push_mode(_tmp_stack.pop())
            assert _len_torch_dispatch_stack() == saved_len, "Stack should be restored!"
    # ------


def _provide_wrapped_forward_on(
    origin_forward: Callable,
    device_mesh: DeviceMesh,
    aten_dfactory_pi: Dict[Callable, Tuple[Callable, PI]],
):
    # new forward
    @functools.wraps(origin_forward)  # copy signatures
    def forward(*args, **kwargs):
        if _IS_DEBUG:
            print(f"[DEBUG] forward({args}, {kwargs})")
            print(f"[DEBUG] signature(forward): {signature(forward)}")
            signature(forward).bind(*args, **kwargs)  # assert that arguments are correct
            print(f"[DEBUG] origin_forward({args}, {kwargs})")
        # call original forward
        with FactoryDispatchModeOff():  # isolate this factory mode on, in case of nest modes
            with FactoryDispatchModeOn(device_mesh, aten_dfactory_pi):
                return origin_forward(*args, **kwargs)

    if _IS_DEBUG:
        print(f"[DEBUG] signature(origin_forward): {signature(origin_forward)}")
        print(f"[DEBUG] signature(forward): {signature(forward)}")

    return forward


def _provide_wrapped_forward_off(origin_forward: Callable):
    # new forward
    @functools.wraps(origin_forward)  # copy signatures
    def forward(*args, **kwargs):
        if _IS_DEBUG:
            print(f"[DEBUG] forward({args}, {kwargs})")
            print(f"[DEBUG] signature(forward): {signature(forward)}")
            signature(forward).bind(*args, **kwargs)  # assert that arguments are correct
            print(f"[DEBUG] origin_forward({args}, {kwargs})")
        # call original forward
        with FactoryDispatchModeOff():
            return origin_forward(*args, **kwargs)

    if _IS_DEBUG:
        print(f"[DEBUG] signature(origin_forward): {signature(origin_forward)}")
        print(f"[DEBUG] signature(forward): {signature(forward)}")

    return forward


def wrap_factory_mode(
    on: bool, mod: nn.Module, device_mesh: Optional[DeviceMesh] = None, factory_pi: Optional[Dict[Callable, PI]] = None
) -> None:  # noqa: B006
    if on:  # turn on factory mode
        assert device_mesh is not None
        assert factory_pi is not None

        # prepare args to factory mode (put here to avoid runtime overhead)
        aten_dfactory_pi = _provide_args(device_mesh, factory_pi)

        # wrap forward with factory mode
        # NOTE: bound method with `MethodType` will disable signature (either set by `__signature__` or `@functools.wraps``),
        # which disables forward hooks appointed by forward plan (as `(x,) != (*args, **kwargs)` )
        # so we use unbound method here to keep the same signature
        mod.forward = _provide_wrapped_forward_on(mod.forward, device_mesh, aten_dfactory_pi)
    else:  # turn off factory mode
        mod.forward = _provide_wrapped_forward_off(mod.forward)

    if _IS_DEBUG:
        print(f"[DEBUG] signature(mod.forward): {signature(mod.forward)}")
