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

import logging
from typing import Optional, Iterable, Dict, Any, Union

import torch
import torch.distributed.distributed_c10d as c10d

from vescale.dtensor.dtensor import DTensor
from vescale.dtensor._collective_utils import mesh_reduce_scatter, mesh_wait
from vescale.optim.base_optimizer import OptimizerBase
from vescale.moe._moe_param_buffer import MoEParamBuffer

try:
    import amp_C
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    USE_APEX_MULTI_APPLIER = False
else:
    USE_APEX_MULTI_APPLIER = True

logger = logging.getLogger(__name__)


class MoEOptimizer(OptimizerBase):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        param_buffer: MoEParamBuffer,
        clip_grad: float = 0.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(optimizer=optimizer)
        self.clip_grad = clip_grad
        self.param_buffer = param_buffer
        self.optimizer = optimizer(param_buffer.get_param_group(), *args, **kwargs)
        param_buffer.set_optimizer(self.optimizer)

    @torch.no_grad()
    def step(self) -> Optional[float]:
        self.param_buffer.setup_grad()
        self.optimizer.step()
        self.param_buffer.process_all_gather()
        return 0

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def get_loss_scale(self):
        return 1.0

    def clip_grad_norm(self, clip_grad):
        grad_list = []
        grad_norm = 0
        for pg in self.param_groups:
            for p in pg["params"]:
                if p.grad is not None:
                    grad_list.append(mesh_wait(p.grad._local_tensor))

        if not USE_APEX_MULTI_APPLIER:
            total_norm = 0
            for grad in grad_list:
                total_norm += (grad**2).sum()
        else:
            dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device="cuda")
            grad_norm, _ = multi_tensor_applier(
                amp_C.multi_tensor_l2norm,
                dummy_overflow_buf,
                [grad_list],
                False,
            )
            total_norm = grad_norm**2

        torch.distributed.all_reduce(total_norm, op=torch.distributed.ReduceOp.SUM)
        total_norm = total_norm.sqrt().item()

        clip_coeff = clip_grad / (total_norm + 1.0e-6)
        if clip_coeff < 1.0:
            dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device="cuda")
            if not USE_APEX_MULTI_APPLIER:
                for g in grad_list:
                    g.data.mul_(clip_coeff)
            else:
                multi_tensor_applier(amp_C.multi_tensor_scale, dummy_overflow_buf, [grad_list, grad_list], clip_coeff)
        return total_norm
