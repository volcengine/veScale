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

from functools import wraps


def decorate_function(func, indices):
    @wraps(func)
    def wrapper(*args, **kwargs):
        outputs = []
        for rank in range(len(args[indices[0]])):
            new_args = list(args)
            for i in indices:
                new_args[i] = args[i][rank]
            o = func(*new_args, **kwargs)
            outputs.append(o)
        return outputs

    if not callable(func):
        return func

    return wrapper


def instrument(obj, func_name_str, indices):
    func_name_list = func_name_str.split(".")
    func_name = func_name_list[-1]

    module_obj = obj
    if len(func_name_list) > 1:
        for module_name in func_name_list[:-1]:
            module_obj = getattr(module_obj, module_name)
    orig_func = getattr(module_obj, func_name)

    wrapped_func = decorate_function(orig_func, indices)
    setattr(module_obj, func_name, wrapped_func)
    return orig_func


def revert_instrument(obj, func_name_str, orig_func):
    func_name_list = func_name_str.split(".")
    func_name = func_name_list[-1]

    module_obj = obj
    if len(func_name_list) > 1:
        for module_name in func_name_list[:-1]:
            module_obj = getattr(module_obj, module_name)

    setattr(module_obj, func_name, orig_func)
    return orig_func


def instrument_all(obj, func_name_str_list, indices_list):
    orig_func_dict = {}
    for func_name_str, indices in zip(func_name_str_list, indices_list):
        orig_func = instrument(obj, func_name_str, indices)
        orig_func_dict[func_name_str] = orig_func
    return orig_func_dict


def revert_instrument_all(obj, func_name_str_list, orig_func_dict):
    for func_name_str in func_name_str_list:
        orig_func = orig_func_dict[func_name_str]
        revert_instrument(obj, func_name_str, orig_func)


class EmulatorInstrumentation:
    """
    A context manager to instrument emulator functions. It replaces the original function with
    a wrapper function that iterates on a list of inputs and returns a list of outputs.

    Args:
        obj (object): The object to instrument. E.g., torch.
        func_name_str_list (List[str]): A list of function names to instrument. E.g., ["mm"]
        indices_list (List[List[int]]): A list of indices to instrument. E.g., [[0, 1]]

    Example:
        >>> import torch
        >>> import vescale.emulator as emu
        >>> from vescale.emulator.emulator_instrumentation import EmulatorInstrumentation
        >>> t1 = [torch.randn(2, 3), torch.randn(2, 3)]
        >>> t2 = [torch.randn(3, 4), torch.randn(3, 4)]
        >>> with EmulatorInstrumentation(torch, ["mm"], [[0, 1]]):
        >>>     torch.mm(t1, t2)
    """

    def __init__(self, obj, func_name_str_list, indices_list) -> None:
        self.obj = obj
        self.func_name_str_list = func_name_str_list
        self.orig_func_dict = instrument_all(obj, func_name_str_list, indices_list)

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        revert_instrument_all(self.obj, self.func_name_str_list, self.orig_func_dict)
