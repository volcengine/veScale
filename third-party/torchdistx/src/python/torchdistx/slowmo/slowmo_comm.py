# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed as dist
from torch.distributed.algorithms._comm_hooks import default


class SlowMoState(default.DefaultState):
    r"""
    State for the `Slow Momentum <https://arxiv.org/abs/1910.00643>`_ .

    Args:
        subgroup (ProcessGroup): stores subgroups, where communication will happen,
            by default a subgroup is initialized to workers,
            belonging to the same node.
        sync_grads (bool): if `True`, gradients will be communicated
            between members of the same subgroup (default: True).
    """

    def __init__(self, subgroup, sync_grads=True):
        self.subgroup = subgroup if subgroup is not None else dist.new_subgroups()[0]
        super().__init__(self.subgroup)
        self.sync_grads = sync_grads


def slowmo_hook(state: SlowMoState, grad: torch.Tensor):
    r"""
    If ``sync_grads`` is enabled in the ``state``,
    reduces gradients between workers under the same node.

    Args:
        state (SlowMoState): State information, configures
            if gradients are going to be communicated or not,
            and subgoups for gradient communication
        grad (torch.Tensor): A gradient for the local batch
            that needs to be communicated across ranks.
    """
    if state.sync_grads:
        default.allreduce_hook(state, grad)
