Slow Momentum for ``FullyShardedDataParallel`` training with ``NO_SHARD`` strategy
===================================================================================
Slow Momentum is a general framework to improve the accuracy of
communication-efficient distributed training methods. The Slow Momentum algorithm
requires exact-averaging of parameters before a momentum update, which is not feasible
with sharded model parameters. As a result, the current implementation is
available only for the FSDP ``NO_SHARD`` strategy.

API
---

The API consists of ``SlowMoState``, ``slowmo_hook``, and ``SlowMomentumOptimizer``.

.. autoclass:: torchdistx.slowmo.slowmo_comm.SlowMoState

.. autofunction:: torchdistx.slowmo.slowmo_comm.slowmo_hook

.. autoclass:: torchdistx.slowmo.slowmo_optimizer.SlowMomentumOptimizer
