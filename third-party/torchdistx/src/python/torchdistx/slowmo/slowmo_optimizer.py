# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed.algorithms.model_averaging.averagers as averagers


class SlowMomentumOptimizer(torch.optim.Optimizer):
    r"""
    Wraps an arbitrary :class:`torch.optim.Optimizer` and runs
    FSDP distributed training with
    `Slow Momentum <https://arxiv.org/abs/1910.00643>`_.
    Currently, only available for FSDP modules defined
    with a `NO_SHARD` strategy.

    Args:
        base_optim (torch.optim.Optimizer):
            The base optimizer, which updates local instance of a model
        slowmo_freq (int): Specifies how often (number of iterations) slow momentum
            is to be performed (default: 48)
        slowmo_factor (float): This specifies the value of slowmo momentum
            to be used (default: 0.5)
        slowmo_lr (float): This specifies the value of slowmo learning rate
            to be used (default: 1.0)

    Example::

        >>>  import torch
        >>>  import torch.distributed as dist
        >>>  from torch.distributed.fsdp import(
        >>>    FullyShardedDataParallel as FSDP
        >>>  )
        >>>  from torchdistx.slowmo import(
        >>>     slowmo_comm,
        >>>     slowmo_optimizer
        >>>  )
        >>>
        >>>  net = torch.nn.Linear(4, 10)
        >>>  fsdp_net = FSDP(net)
        >>>  # This implementation communicates gradients between
        >>>  # workers of the same node
        >>>  # before averaging the model's parameters between nodes.
        >>>  # The following creates intra-node subgroups
        >>>  # and SlowMoState will take care of storing all required
        >>>  # parameters for intra-node communication,
        >>>  # i.e. pre- and post-division factors, and subgroups.
        >>>  # To disable any communication between workers,
        >>>  # set `sync_grads` to `False`
        >>>  cur_subgroup, _ = dist.new_subgroups()
        >>>  slowmo_state = slowmo_comm.SlowMoState(
        >>>     cur_subgroup,
        >>>     sync_grads=True
        >>>  )
        >>>
        >>>  # Register SlowMo hook, which only communicates gradients
        >>>  # in a intra-node fashion.
        >>>  fsdp_net.register_comm_hook(
        >>>     slowmo_state,
        >>>     slowmo_comm.slowmo_hook
        >>>  )
        >>>
        >>>  base_optimizer = torch.optim.SGD(
        >>>     fsdp_net_slowmo.parameters(),
        >>>     lr=1e-2
        >>>  )
        >>>  # Create a SlowMo optimizer that wraps a local optimizer.
        >>>  slowmo_optim = slowmo_optimizer.SlowMomentumOptimizer(
        >>>     base_optim=base_optimizer,
        >>>     slowmo_freq=6,
        >>>     slowmo_factor=0.5,
        >>>     slowmo_lr=0.1
        >>>  )
        >>>
        >>>  # SlowMo runs intra-node gradient averaging at every step,
        >>>  # every 6th step it will run model averaging and
        >>>  # a slow momentum update.
        >>>  for step in range(200):
        >>>     slowmo_optim.zero_grad()
        >>>     loss = loss_fn(output, labels)
        >>>     loss.backward()
        >>>     slowmo_optim.step()
    """

    def __init__(
        self,
        base_optim: torch.optim.Optimizer,
        slowmo_freq: int = 48,
        slowmo_factor: float = 0.5,
        slowmo_lr: float = 1.0,
    ):
        if base_optim is None:
            raise ValueError("Base optimizer is a required parameter.")
        self._base_optim = base_optim

        # check that base optimizer's `param_groups` are present
        if not (self._base_optim.param_groups):
            raise ValueError(
                "Provided base optimizer does not have parameters specified."
            )
        for group in self._base_optim.param_groups:
            if "lr" not in group:
                raise ValueError(
                    "All parameter groups should have learning rate specified."
                )

        self.param_groups = self._base_optim.param_groups

        if slowmo_freq < 1:
            raise ValueError(
                "Invalid ``slowmo_freq`` parameter, must be a positive value."
            )
        self.slowmo_freq = slowmo_freq

        if slowmo_factor < 0.0:
            raise ValueError(
                "Invalid ``slowmo_factor`` parameter, must be non-negative."
            )
        self.slowmo_factor = slowmo_factor

        if slowmo_lr < 0.0:
            raise ValueError("Invalid ``slowmo_lr`` parameter, must be non-negative.")
        self.slowmo_lr = slowmo_lr

        self.averager = averagers.PeriodicModelAverager(
            period=slowmo_freq, warmup_steps=0
        )
        self.buffers_initialized = False

        # Memorize initial parameters before the first `step()`.
        # Can't put them in `self.state`, because some of optimizers rely
        # `self.state` being empty during the `step()`
        # to initialize optimizer states.
        # `self._prev_parameters` must be in sync with
        # the flattened version of `self.param_groups`,
        # since this implementation relies on `self._prev_parameters`
        # having the same order of parameters as in `self.param_groups`
        # to perform a slow momentum update.
        self._prev_parameters = []
        for group in self.param_groups:
            for param in group["params"]:
                self._prev_parameters.append(param.detach().clone())

    @property
    def state(self):
        r"""
        Forwards to base optimizer's ``state``.
        """
        return self._base_optim.state

    def __repr__(self):
        return self._base_optim.__repr__()

    def state_dict(self):
        r"""
        This is the same as :class:`torch.optim.Optimizer`
        :meth:`state_dict`, but adds an extra entries to record
        Slow Momentum's specific parameters: ``slowmo_freq``,
        ``slowmo_factor``, ``slowmo_lr``, and ``step`` for the model's averager.
        """
        optim_state_dict = self._base_optim.state_dict()
        optim_state_dict["slowmo_freq"] = self.slowmo_freq
        optim_state_dict["slowmo_factor"] = self.slowmo_factor
        optim_state_dict["slowmo_lr"] = self.slowmo_lr
        optim_state_dict["step"] = self.averager.step

        return optim_state_dict

    def load_state_dict(self, state_dict):
        r"""
        This is the same as :class:`torch.optim.Optimizer`
        :meth:`load_state_dict`, but also restores Slow Momentum's
        specific parameters, saved in the provided ``state_dict``.
        """
        self.slowmo_freq = state_dict["slowmo_freq"]
        self.averager.period = state_dict.pop("slowmo_freq")
        self.slowmo_factor = state_dict.pop("slowmo_factor")
        self.slowmo_lr = state_dict.pop("slowmo_lr")
        self.averager.step = state_dict.pop("step")
        self._base_optim.load_state_dict(state_dict)
        if not self.param_groups:
            raise ValueError("Base optimizer does not have parameter groups specified.")
        for group in self._base_optim.param_groups:
            if "lr" not in group:
                raise ValueError(
                    "All parameter groups should have learning rate specified."
                )

    @torch.no_grad()
    def step(self):
        r"""
        Performs a single optimization step (parameter update)
        and a slow momentum update. Slow momentum update involves
        model's exact averaging of parameters and a momentum update,
        which happens every `slowmo_freq` step.
        """
        self._base_optim.step()
        # Averager averages parameters between workers every `slowmo_freq` step.
        # At other times it just increases step counter.
        self.averager.average_parameters(params=self.param_groups)
        # Since at this point averager has increased its step,
        # we need to check (self.averager.step - 1).
        # No need to do momentum step at step 0.
        if (self.averager.step - 1) % self.slowmo_freq == 0 and self.averager.step != 1:
            prev_param_idx = 0
            for group in self.param_groups:
                for param in group["params"]:
                    # Initialize momentums if they were not initialized
                    if "slow_momentum" not in self.state[param]:
                        self.state[param]["slow_momentum"] = torch.zeros(
                            param.shape, device=torch.cuda.current_device()
                        )

                    # Update the slow momentum
                    p_state = self.state[param]
                    factor = 1 / group["lr"]
                    p_state["slow_momentum"].mul_(self.slowmo_factor).sub_(
                        param, alpha=factor
                    ).add_(self._prev_parameters[prev_param_idx], alpha=factor)
                    # Update parameters
                    self._prev_parameters[prev_param_idx].add_(
                        p_state["slow_momentum"], alpha=-self.slowmo_lr * group["lr"]
                    )
                    param.copy_(self._prev_parameters[prev_param_idx])
                    prev_param_idx += 1

    def zero_grad(self, set_to_none: bool = False):  # type: ignore[override]
        self._base_optim.zero_grad(set_to_none=set_to_none)

    def add_param_group(self, param_group):
        self._base_optim.add_param_group(param_group)
        for param in param_group["params"]:
            self._prev_parameters.append(param.detach().clone())
