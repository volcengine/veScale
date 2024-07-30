################################################################################
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
################################################################################
# Modification Copyright 2024 ByteDance Ltd. and/or its affiliates.
################################################################################

import functools
from typing import Callable

from optree import tree_map
from vescale.dtensor.dtensor import DTensor


def switch_dtensor(func: Callable):
    @functools.wraps(func)
    def wrap(*args, **kwargs):
        def to_tensor(x):
            if isinstance(x, DTensor):
                return x.to_local()
            return x

        new_args = tree_map(to_tensor, args)
        new_kwargs = tree_map(to_tensor, kwargs)
        out = func(*new_args, **new_kwargs)
        return out

    return wrap
