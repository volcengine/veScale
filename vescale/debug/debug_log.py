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
import inspect
import os
from typing import Optional
import sys
from types import FrameType
from typing import Dict, Tuple, Union, Sequence

import torch
from torch import distributed as dist
from typing import TYPE_CHECKING
from logging import Logger

if TYPE_CHECKING:
    from vescale.dtensor.op_schema import OpSchema, OpInfo


__all__ = [
    "DebugLogger",
]


class DebugLogger:
    """
    Provides a centralized logging utility designed to support debugging vescale model in distributed computing environments.
    It allows for selective logging based on rank and supports dynamic adjustment of debugging verbosity through environment variables.

    Attributes:
        IS_DEBUG_MODE (Optional[bool]): Flag indicating if debug mode is enabled. Defaults to False.
        _device_mesh: Placeholder for a device mesh API. Currently None and marked for future replacement.
        _already_init (bool): Indicates whether initial setup has been completed to avoid redundant operations.
        rank (Optional[int]): The rank of the current process within the distributed setup.
        local_rank (Optional[int]): The local rank of the current process. Not currently used but reserved for future.
        world_size (Optional[int]): The total number of processes in the distributed environment. To be set externally.
        _rank_to_print (Tuple[int,...]): Specifies the ranks for which logging is enabled. Defaults to (-1, ), indicating none.
        _loggeer (Optional[Logger]): The logger object used for debug output. If None, falls back to printing.

    Static Methods:
        log(*args, **kwargs): Logs a message either to the console or through a specified logger if debug mode is on.
        update_vescale_debug_mode_from_env(): Updates the IS_DEBUG_MODE flag based on the VESCALE_DEBUG_MODE environment variable.
        set_vescale_debug_mode(on=True, *, rank_to_print=None, logger=None): Configures the debug mode, including which ranks should log messages.
        log_communication(func, *args, **kwargs): Logs communication operations.
        log_op(op_info: 'OpInfo'): Logs operations execution.
        _init_values_(): Initializes necessary values for the logger, such as ranks and world size, if not already done.

    Usage:
        Option 1: Define VESCALE_DEBUG_MODE as an environment variable at the beginning of the program.
        `         For performance reasons, VESCALE_DEBUG_MODE should be at least set before calling vescale.parallelize_module.

        Option 2 (perferred way): Using set_vescale_debug_mode at any point of your program.
                set_vescale_debug_mode also allows you to pass in a Python logging.Logger for each rank to distinguish logs from different ranks.
    """

    IS_DEBUG_MODE: Optional[bool] = False
    # TODO replace by new devicemesh api
    _device_mesh = None
    _already_init: bool = False
    rank: Optional[int] = None
    local_rank: Optional[int] = None
    world_size: Optional[int] = None  # dist.get_world_size()
    _rank_to_print: Tuple[int, ...] = (-1,)
    _loggeer: Optional[Logger] = None

    @staticmethod
    def log(*arg, **kwargs):
        if DebugLogger._loggeer is None:
            print(*arg, **kwargs)
        else:
            DebugLogger._loggeer.debug(*arg, **kwargs)

    @staticmethod
    def update_vescale_debug_mode_from_env():
        DebugLogger.IS_DEBUG_MODE = os.getenv("VESCALE_DEBUG_MODE", "") == "1"

    @staticmethod
    def set_vescale_debug_mode(
        on: bool = True, *, rank_to_print: Optional[Union[int, Sequence[int]]] = None, logger=None
    ):
        if DebugLogger.IS_DEBUG_MODE != on:
            os.environ["VESCALE_DEBUG_MODE"] = str(int(on))
            DebugLogger.IS_DEBUG_MODE = on
            DebugLogger._rank_to_print = None
            DebugLogger._loggeer = None
        if not DebugLogger.IS_DEBUG_MODE:
            DebugLogger.log("vescale debug mode is off")
            return
        DebugLogger.log("vescale debug mode is on")
        if rank_to_print is None:
            DebugLogger.log("rank_to_print is not set, using rank 0")
            DebugLogger._rank_to_print = (0,)
            return
        elif isinstance(rank_to_print, int):
            DebugLogger._rank_to_print = (rank_to_print,)
        elif isinstance(rank_to_print, Sequence) and all(isinstance(i, int) for i in rank_to_print):
            DebugLogger._rank_to_print = rank_to_print
        else:
            raise TypeError(
                "expect rank_to_print to be either int or tuple/list of int" f"but get {type(rank_to_print)}"
            )
        DebugLogger._loggeer = logger

    @staticmethod
    def log_communication(func, *args, **kwargs) -> None:
        DebugLogger._init_values_()
        _CommunicationLogger.log_communication(func, *args, **kwargs)

    @staticmethod
    def log_op(op_info: "OpInfo") -> None:
        DebugLogger._init_values_()
        _OperatorLogger.print_ops_execution(op_info)

    @staticmethod
    def _init_values_():
        if DebugLogger._already_init:
            return
        DebugLogger._already_init = True
        # TODO replace by new devicemesh api
        DebugLogger.rank = dist.get_rank()
        DebugLogger.world_size = dist.get_world_size()
        if DebugLogger._rank_to_print == (-1,):
            DebugLogger._rank_to_print = tuple(range(DebugLogger.world_size))


class _CommunicationLogger:
    _file_to_recoder = {
        "/dtensor/_utils.py",
        "/dtensor/redistribute.py",
        "/dtensor/api.py",
        "/dtensor/dtensor.py",
        "/dmodule/_hook.py",
    }
    _func_to_exclude = {"<lambda>", "<genexpr>"}

    @staticmethod
    def _trace_to_coll_inject_point():
        result = []
        for frame_record in inspect.stack():
            frame = frame_record.frame
            code = frame.f_code
            co_name = code.co_name
            if co_name in _CommunicationLogger._func_to_exclude:
                continue

            co_filename = code.co_filename
            for f in _CommunicationLogger._file_to_recoder:
                if co_filename.endswith(f):
                    result.append(f"{f}::{co_name}")
                    break
        return ", ".join(result)

    @staticmethod
    def log_communication(func, *args, **kwargs):
        DebugLogger._init_values_()
        rank = DebugLogger.rank
        if rank in DebugLogger._rank_to_print:
            inject_point = _CommunicationLogger._trace_to_coll_inject_point()
            sig = ""
            bound_arguments = inspect.signature(func).bind(*args, **kwargs)
            bound_arguments.apply_defaults()
            for param_name, value in bound_arguments.arguments.items():
                if isinstance(value, torch.Tensor):
                    sig += f"\t{param_name}: {value.shape}\n"
                elif "scatter_list" in param_name:
                    sig += f"\t{param_name}: ["
                    for i, item in enumerate(value):
                        if isinstance(item, torch.Tensor):
                            sig += f"{item.shape}, "
                    sig += "]\n"
                else:
                    sig += f"\t{param_name}: {value}\n"
            DebugLogger.log(f"[rank{rank}] {func.__name__} with stack: {inject_point}")
            DebugLogger.log(f"\t{sig[1:]}")

    @staticmethod
    def log_communication_decorator():
        """

        print_coll_comm_signature prints out the collective communication, including:
        collective_commucation_type, function signature, and the po

        Args:
            rank_to_print: the rank that is going to DebugLogger.log out the debug info. -1 means prints on all ranks

        Example::

            usage: used as decorator:
                @print_coll_comm_signature(0)
                def mesh_all_gather()

            output:
                [rank0] mesh_all_gather at _reshard_to_replicate_with_pad_one_dim
                        tensor: torch.Size([40, 11])
                        global_size: torch.Size([40, 88])
                        mesh: DeviceMesh:([0, 1, 2, 3, 4, 5, 6, 7])
                        scatter_dim: 1
                        mesh_dim: 0

        """

        def decorator(func):
            if not DebugLogger.IS_DEBUG_MODE:  # NOTE: put here for performance if no debug mode
                return func

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                _CommunicationLogger.log_communication(func, *args, **kwargs)
                return func(*args, **kwargs)

            return wrapper

        return decorator


class _OperatorLogger:
    op_map: Dict = dict()

    @staticmethod
    def trace_to_forward() -> FrameType:
        current_frame = sys._getframe()
        while current_frame:
            if current_frame.f_code.co_name == "forward":
                break
            current_frame = current_frame.f_back
        return current_frame

    @staticmethod
    def get_module_from_frame(frame) -> any:
        return frame.f_locals.get("self", None)

    @staticmethod
    def ops_info_printer(rank: int, frame: FrameType, module: object, op_info: "OpInfo"):
        DebugLogger.log(
            f"[rank{rank}] {module.__class__.__name__} forward() at {frame.f_code.co_filename}:{frame.f_lineno}"
        )

        # input
        _OperatorLogger._print_input(op_info.schema.args_schema)

        # op
        _OperatorLogger._print_op(op_info.schema)

        # output
        _OperatorLogger._print_output(op_info.output_sharding)

        DebugLogger.log("\n")

    @staticmethod
    def _print_op(op_schema: "OpSchema"):
        DebugLogger.log(f"\t{op_schema}")

    @staticmethod
    def _print_input(args_schema):
        from vescale.dtensor.placement_types import DTensorSpec

        DebugLogger.log("\tinput: [")
        for item in args_schema:
            if isinstance(item, DTensorSpec):
                _OperatorLogger.dt_spec_debug_formatter(item)
        DebugLogger.log("\t]")

    @staticmethod
    def _print_output(output_sharding):
        from vescale.dtensor.placement_types import DTensorSpec

        output_spec = output_sharding.output_spec
        DebugLogger.log("\toutput: [")
        if isinstance(output_spec, DTensorSpec):
            _OperatorLogger.dt_spec_debug_formatter(output_spec)
        elif isinstance(output_spec, (list, tuple)):
            for item in output_spec:
                if isinstance(item, DTensorSpec):
                    _OperatorLogger.dt_spec_debug_formatter(item)
        else:
            DebugLogger.log(output_spec)
        DebugLogger.log("\t]")

    @staticmethod
    def dt_spec_debug_formatter(dt_spec):
        """
        dt_spec_debug_formatter() pretty DebugLogger.log dtensor with TensorMeta, Placements, and DeviceMesh

        Args:
            DTensorSpec

        Example::

            DTensor(
                TensorMeta(shape=torch.Size([104]), stride=(1,), dtype=torch.float32)
                Placements:(Partial(reduce_op=RedOpType.SUM),)
                DeviceMesh:([0, 1, 2, 3, 4, 5, 6, 7])
            )
        """
        DebugLogger.log("\t\tDTensor(")
        DebugLogger.log(f"\t\t\t{dt_spec.tensor_meta}")
        DebugLogger.log(f"\t\t\tPlacements:{dt_spec.placements}")
        DebugLogger.log(f"\t\t\t{dt_spec.mesh}")
        DebugLogger.log("\t\t)")

    @staticmethod
    def print_ops_execution(op_info: "OpInfo") -> None:
        """
        print_ops_execution() prints out the executed ops during __torch_dispatch__, it prints out the metadata including:
        DModule name, propagation stage, line# in source code, input/output, operators.

        Args:
            OpInfo, the operator that is going to dispatch

        Example::

            [rank0] VeConv1D forward() at /vescale/model/audio/gpt2_audio.py:54
                    input: [
                        DTensor(
                            TensorMeta(shape=torch.Size([104]), stride=(1,), dtype=torch.float32)
                            Placements:(Partial(reduce_op=RedOpType.SUM),)
                            DeviceMesh:([0, 1, 2, 3, 4, 5, 6, 7])
                        )
                        DTensor(
                            TensorMeta(shape=torch.Size([40, 88]), stride=(88, 1), dtype=torch.float32)
                            Placements:(Shard(dim=1),)
                            DeviceMesh:([0, 1, 2, 3, 4, 5, 6, 7])
                        )
                        DTensor(
                            TensorMeta(shape=torch.Size([88, 104]), stride=(104, 1), dtype=torch.float32)
                            Placements:(Shard(dim=0),)
                            DeviceMesh:([0, 1, 2, 3, 4, 5, 6, 7])
                        )
                    ]
                    Op(op=aten.addmm.default, args_sharding=Spec(P on (104,)), Spec(S(1) on (40, 88)), Spec(S(0) on (88, 104))@ mesh: (8,))
                    output: [
                        DTensor(
                            TensorMeta(shape=torch.Size([40, 104]), stride=(104, 1), dtype=torch.float32)
                            Placements:(Partial(reduce_op=RedOpType.SUM),)
                            DeviceMesh:([0, 1, 2, 3, 4, 5, 6, 7])
                        )
                    ]
        """
        frame = _OperatorLogger.trace_to_forward()
        if not frame:
            return
        module = _OperatorLogger.get_module_from_frame(frame)
        rank = DebugLogger.rank
        # -1 means DebugLogger.log on all ranks
        if rank in DebugLogger._rank_to_print:
            _OperatorLogger.ops_info_printer(rank, frame, module, op_info)
