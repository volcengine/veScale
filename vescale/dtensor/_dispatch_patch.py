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

from typing import Tuple, Dict, Any

import torch

import vescale.dtensor.dtensor as dtensor
from vescale.dtensor._diff import DeferReshardMode
from vescale.dtensor.placement_types import Replicate


__all__ = ["_pre_patch_for_dispatch", "_post_patch_for_dispatch"]

aten = torch.ops.aten

_linear_pointwise_ops = {
    aten.add.Tensor,
    aten.add_.Tensor,
}


class DispatchPrePatch:
    def __init__(self) -> None:
        self.op_hackers = {
            aten.index_put.default: DispatchPrePatch.index_put_handler,
            aten.scatter_.value: DispatchPrePatch.scatter_handler,
            aten.scatter.value: DispatchPrePatch.scatter_handler,
            aten.scatter_.src: DispatchPrePatch.scatter_handler,
            aten.scatter.src: DispatchPrePatch.scatter_handler,
            aten.eq.Tensor: DispatchPrePatch.eq_tensor_handler,
            aten.index.Tensor: DispatchPrePatch.index_tensor_handler,
        }

    def apply(
        self,
        op_call: torch._ops.OpOverload,
        args: Tuple[object, ...],
        kwargs: Dict[str, object],
    ) -> Tuple[Tuple[object, ...], Dict[str, object]]:
        hack_call = self.op_hackers.get(op_call, None)
        if hack_call is not None:
            return hack_call(op_call, args, kwargs)
        else:
            return args, kwargs

    @staticmethod
    def index_put_handler(
        op_call: torch._ops.OpOverload,
        args: Tuple[object, ...],
        kwargs: Dict[str, object],
    ):
        if (
            (not isinstance(args[2], dtensor.DTensor))
            and isinstance(args[2], torch.Tensor)
            and isinstance(args[0], dtensor.DTensor)
        ):
            device_mesh = args[0]._spec.mesh
            sharding = args[0]._spec.placements
            new_args_2 = dtensor.DTensor.from_local(args[2], device_mesh, sharding, run_check=False)
            return (*args[:2], new_args_2, *args[3:]), kwargs
        return args, kwargs

    @staticmethod
    def scatter_handler(
        op_call: torch._ops.OpOverload,
        args: Tuple[object, ...],
        kwargs: Dict[str, object],
    ):
        if (
            (not isinstance(args[0], dtensor.DTensor))
            and isinstance(args[0], torch.Tensor)
            and isinstance(args[2], dtensor.DTensor)
        ):
            device_mesh = args[2]._spec.mesh
            new_args_0 = dtensor.DTensor.from_local(args[0], device_mesh, [Replicate()], run_check=False)
            return (new_args_0, *args[1:]), kwargs
        return args, kwargs

    @staticmethod
    def eq_tensor_handler(
        op_call: torch._ops.OpOverload,
        args: Tuple[object, ...],
        kwargs: Dict[str, object],
    ):
        if (
            (not isinstance(args[1], dtensor.DTensor))
            and isinstance(args[0], dtensor.DTensor)
            and isinstance(args[1], torch.Tensor)
        ):
            device_mesh = args[0]._spec.mesh
            new_args_1 = dtensor.DTensor.from_local(args[1], device_mesh, [Replicate()], run_check=False)
            return (args[0], new_args_1, *args[2:]), kwargs
        return args, kwargs

    @staticmethod
    def index_tensor_handler(
        op_call: torch._ops.OpOverload,
        args: Tuple[object, ...],
        kwargs: Dict[str, object],
    ):
        if isinstance(args[0], dtensor.DTensor):
            device_mesh = args[0]._spec.mesh
            new_args = (
                args[0],
                [
                    dtensor.DTensor.from_local(x, device_mesh, [Replicate()], run_check=False)
                    if isinstance(x, torch.Tensor) and not isinstance(x, dtensor.DTensor)
                    else x
                    for x in args[1]
                ],
            )
            return new_args, kwargs
        return args, kwargs


_dispatch_pre_patch = DispatchPrePatch()


def defer_resharding(op_call: torch._ops.OpOverload, dt_wrap: Any):
    if DeferReshardMode._enable_autoresharding() and op_call in _linear_pointwise_ops:
        suggest_sharding = DeferReshardMode._query_sharding()
        # remove lhs
        DeferReshardMode._remove_sharding()
        # remove rhs
        DeferReshardMode._remove_sharding()
        dt_wrap = dt_wrap.redistribute(dt_wrap._spec.mesh, suggest_sharding)
    return dt_wrap


def failed_on_mqa(
    op_call: torch._ops.OpOverload,
    *args: Tuple[object, ...],
    **kwargs: Dict[str, object],
):
    """
    In normal cases, the matmul style of attention with TP is:
    [bsz * num_q_heads // TP_size, seq_len, hidden_dim] * [bsz * num_q_heads // TP_size, hidden_dim, seq_len]
    But in MQA, it will become:
    [bsz * num_q_heads // TP_size, seq_len, hidden_dim] * [bsz, hidden_dim // TP_size, seq_len].
    We fail on the MQA pattern.
    """

    if op_call != aten.bmm.default:
        return

    args = args[0]
    if len(args) != 2:
        return
    lhs, rhs = args[0], args[1]

    if not isinstance(lhs, dtensor.DTensor) or not isinstance(rhs, dtensor.DTensor):
        return
    lhs_placements = lhs.placements
    rhs_placements = rhs.placements
    mesh = lhs.device_mesh
    assert mesh == rhs.device_mesh, "cross mesh op detected"

    lhs_rank = len(lhs.shape)
    rhs_rank = len(rhs.shape)
    for mesh_dim in range(mesh.ndim):
        if not lhs_placements[mesh_dim].is_shard() or not rhs_placements[mesh_dim].is_shard():
            continue

        lhs_sharded_dim = lhs_placements[mesh_dim].dim
        if lhs_sharded_dim < 0:
            lhs_sharded_dim += lhs_rank
        rhs_sharded_dim = rhs_placements[mesh_dim].dim
        if rhs_sharded_dim < 0:
            rhs_sharded_dim += rhs_rank

        if lhs_sharded_dim != 0 or rhs_sharded_dim != 1:
            continue

        raise RuntimeError("You are probably use a MQA which is not supported now")
    return


def _pre_patch_for_dispatch(*args, **kwargs):
    """
    Put patch logic here before entering dtensor dispatching logic
    """
    failed_on_mqa(*args, **kwargs)
    return _dispatch_pre_patch.apply(*args, **kwargs)


def _post_patch_for_dispatch(*args, **kwargs):
    """
    Put patch logic here after existing dtensor dispatching logic
    """
    return defer_resharding(*args, **kwargs)


def _pre_patch_for_sharding_prop(*args, **kwargs):
    """
    Put patch logic here before entering dtensor sharding propagation
    """
    raise NotImplementedError


def _post_patch_for_sharding_prop(*args, **kwargs):
    """
    Put patch logic here after existing dtensor sharding propagation
    """
    raise NotImplementedError
