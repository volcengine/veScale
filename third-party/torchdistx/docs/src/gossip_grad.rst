GossipGraD communication strategy for ``FullyShardedDataParallel`` training with ``NO_SHARD`` strategy
=======================================================================================================
`GossipGraD <https://arxiv.org/abs/1803.05880>`_ is a gossip communication protocol
for a large-scale training, which can provide communication efficiency over global `all_reduce`
strategy.

API
---

.. autoclass:: torchdistx.gossip_grad.Topology

.. autofunction:: torchdistx.gossip_grad.GossipGraDState

.. autoclass:: torchdistx.gossip_grad.gossip_grad_hook