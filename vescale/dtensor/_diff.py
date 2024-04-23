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

import functools
import os
from typing import Callable

from torch.utils._python_dispatch import TorchDispatchMode, _pop_mode, _push_mode


VESCALE_PARTIAL_MODE = os.environ.get("VESCALE_PARTIAL_MODE", "0") == "1"
VESCALE_DISABLE_REDISTRIBUTE = os.environ.get("VESCALE_DISABLE_REDISTRIBUTE", "1") == "1"

global VESCALE_SHARDING_SUGGETSION
VESCALE_SHARDING_SUGGETSION = []


def switch_partial_mode(func: Callable):
    @functools.wraps(func)
    def wrap(*args, **kwargs):
        global VESCALE_PARTIAL_MODE
        if VESCALE_PARTIAL_MODE:
            with EnablePartialMode():
                out = func(*args, **kwargs)
        else:
            out = func(*args, **kwargs)
        return out

    return wrap


class EnablePartialMode(TorchDispatchMode):
    """
    To enable the DTensor to be PartialSum for performance
    By sometimes, we find there have some optimization chance
    for partial state, so we enable to get a partial DTensor
    by torch ops

    chance one: adjust the reshard AllReduceReassociate
    The AllReduceReassociate can be simplify
    allreduce(x) + allreduce(y) to allreduce(x + y),
    there will be some alllreduce save for partial activation

    Note:
        EnablePartialMode only influence the xxx_like op porpagation
        rules. if you want this mode affect some other function, maybe
        refer to ```switch_partial_mode``` , by wrapper any function
        with ```@switch_partial_mode``` there will be also tracked by
        EnablePartialMode

    Usage:
        ```
        from vescale.dtensor.dispatch import EnablePartialMode
        with EnablePartialMode():
            partial_tensor = torch.ones_like(other)
        ```
    """

    @staticmethod
    def _enable():
        global VecaleParitalMode
        VecaleParitalMode = True

    @staticmethod
    def _disable():
        global VecaleParitalMode
        VecaleParitalMode = False

    def __enter__(self):
        EnablePartialMode._enable()
        _push_mode(self, self.__dict__.get("_dispatch_key", None))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        EnablePartialMode._disable()
        _pop_mode(self.__dict__.get("_dispatch_key", None))

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        return func(*args, **kwargs)


class DeferReshardMode:
    """
    To enable the DTensor to be PartialSum for performance
    By sometimes, we find there have some optimization chance
    for partial state, so we enable to get a partial DTensor
    by torch ops

    chance one: adjust the reshard AllReduceReassociate
    The AllReduceReassociate can be simplify
    allreduce(x) + allreduce(y) to allreduce(x + y),
    partial add option can be done use this mode
    For Example:

        Partial(x) + Partial(y) -> Shard(out)

        one common operation is that:

        Partial(x) -> Replicate(x)
        Partial(y) -> Replicate(y)
        Replicate(x) + Replicate(y) -> Replicate(out)

    there have 2 allreduce operation while with DeferReshardMode operation can be
        Partial(x) + Partial(y) -> Partial(out)
        Partial(out) -> Replicate(out)

    Usage:

        fwd_shard_plan : {
            "partial_op.output.lazy": [[Replicate()]]
        }

    """

    @staticmethod
    def _push_sharding(placement):
        global VESCALE_SHARDING_SUGGETSION
        VESCALE_SHARDING_SUGGETSION.append(placement)

    @staticmethod
    def _remove_sharding():
        global VESCALE_SHARDING_SUGGETSION
        if len(VESCALE_SHARDING_SUGGETSION) > 0:
            VESCALE_SHARDING_SUGGETSION.pop(0)

    @staticmethod
    def _enable_autoresharding():
        global VESCALE_SHARDING_SUGGETSION
        return len(VESCALE_SHARDING_SUGGETSION) > 0

    @staticmethod
    def _query_sharding():
        global VESCALE_SHARDING_SUGGETSION
        return VESCALE_SHARDING_SUGGETSION[0]
