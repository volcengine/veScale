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
from abc import ABC, abstractmethod, abstractstaticmethod
from typing import Optional, Callable, List, Union

import torch

from vescale.dmodule._dmodule import DModule
from vescale.dtensor.dtensor import DTensor

logger = logging.getLogger(__name__)


class OptimizerBase(ABC):
    """
    Abstract base class for vescale optimizer wrapper.

    Args:
        optimizer: Underlying optimizer instance
    """

    def __init__(self, optimizer):
        self.optimizer = optimizer

    @abstractmethod
    def zero_grad(self, set_to_none=True):
        pass

    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self, state_dict):
        pass

    @abstractmethod
    def step(self):
        pass

    # Promote param_groups so it can be retrieved or set via
    # "optimizer_instance.param_groups"
    # (for example, to adjust the learning rate)
    def _get_param_groups(self):
        return self.optimizer.param_groups

    def _set_param_groups(self, value):
        self.optimizer.param_groups = value

    param_groups = property(_get_param_groups, _set_param_groups)

    def get_loss_scale(self):
        """The output should be a cuda tensor of size 1."""
        return 1.0


class GradOptimizerHookBase(ABC):
    """
    Abstract base class for hooks, that needed to be run before
    and after `optim.step`.
    """

    @abstractstaticmethod
    def step_pre_hook(optim, *args, **kwargs):
        return NotImplementedError("not impl")

    @abstractstaticmethod
    def step_post_hook(optim, *args, **kwargs):
        raise NotImplementedError("not impl")


class BasicOptimizer(OptimizerBase):
    """
    A simple wrapper around a concrete optimizer instance. It provides basic
    functionality for optimizing models in distributed training, such as main_grad
    conversion, which is needed to adapt DDP to normal optimizer,
    and triggering grad allreduce for weights like layernorm in Sequence Parallel.

    Args:
        optimizer: Underlying optimizer instance.
        models (Union[nn.Module, List[nn.modules]]): nn.Modules of which parameters
            need to be optimized.
        grad_hook (Optional[GradOptimizerHookBase]): A GradOptimizerHookBase instance,
            which contains hooks triggered before and after `step`.

    Returns:
        A :class:`BasicOptimizer` object.

    Examples:
        ```python
        # The following program creates a parallelized MLP module which is wrapped by DDP.
        # One only need to wrap a Adam optimizer by BasicOptimizer, then everything,
        # like flattened main_grad in DDP world will be hidden.

        from vescale.optim.base_optimizer import BasicOptimizer, BaseOptimizerHook
        from vescale.ddp.distributed_data_parallel import DistributedDataParallel as DDP
        from vescale.dmodule.api import parallelize_module

        mlp = parallelize_module(MLP(), mesh, ..., ...)
        ddp_model = DDP(mlp, ...)
        optim = torch.optim.Adam(model.parameters())
        optim_wrapper = BasicOptimizer(optim, mlp, grad_hook=BaseOptimizerHook())

        # do the forward and backward
        ddp_model(torch.rand(xxx)).sum().backward()
        # run optimizer `step`
        optimizer_wrapper.step()
        ```
    """

    def __init__(
        self,
        optimizer,
        models: Union[torch.nn.Module, List[torch.nn.Module]],
        grad_hook: Optional[GradOptimizerHookBase] = None,
    ) -> None:
        super().__init__(optimizer=optimizer)
        self.models = models
        if not isinstance(self.models, List):
            self.models = [self.models]

        if grad_hook is not None:
            self.register_optimizer_hook(grad_hook)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        from vescale.ddp.distributed_data_parallel import DistributedDataParallel as DDP

        for m in self.models:
            if not DModule.is_dmodule(m):
                logging.warning("module has no `finish_grad_sync` method defined, skip allreducing grads")
                continue
            # if module is wrapped by DDP, we needn't handle partial grad sync. DDP will do it.
            if isinstance(m, DDP):
                continue
            m.finish_grad_sync()
        return self.optimizer.step(closure)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def get_loss_scale(self):
        return 1.0

    def register_optimizer_hook(self, grad_hook: GradOptimizerHookBase):
        self.optimizer.register_step_pre_hook(grad_hook.step_pre_hook)
        self.optimizer.register_step_post_hook(grad_hook.step_post_hook)


class BaseOptimizerHook(GradOptimizerHookBase):
    """
    A GradOptimizerHookBase subclass, that is responsible to 'fill' flattened
    main_grad in DDP to PyTorch '.grad' fields.

    Example: see example codes for `BasicOptimizer`
    """

    def step_pre_hook(optim, *args, **kwargs):
        for param_group in optim.param_groups:
            for p in param_group["params"]:
                if p.grad is not None:
                    continue
                if p.main_grad is None:
                    continue
                if isinstance(p.data, DTensor):
                    dtensor_placements = p.data.placements
                    dtensor_device_mesh = p.data.device_mesh
                    p.grad = DTensor.from_local(
                        p.main_grad, device_mesh=dtensor_device_mesh, placements=dtensor_placements
                    )
                else:
                    p.grad = p.main_grad

    def step_post_hook(optim, *args, **kwargs):
        return None
