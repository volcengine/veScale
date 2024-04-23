# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import random
from enum import Enum, auto
from itertools import cycle

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import ProcessGroup
from torch.distributed.algorithms._comm_hooks import default
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

# Setting a constant for situations, when communication peer
# is not present in a current environment. This may happen in CUBE topology,
# when a number of nodes is not equal to a power of 2. In this case, both
# send and receive peers are equal to INVALID_PEER and no communication is
# performed.
INVALID_PEER = -1


class Topology(Enum):
    r"""
    Specifies which topology will be used as a base for gradient communication.
    For more information, please refer to the original
    `paper <https://arxiv.org/abs/1803.05880>`_

    CUBE:
          A hypercube topology - a hierarchical virtual organization of compute nodes.
          For this topology gossiping is happening with a neighboring vertex.

    >>>      *----*
    >>>     /|   /|
    >>>    *----* |
    >>>    | * -|-*
    >>>    |/   |/
    >>>    *----*

    DISSEMINATION:
                   A dissemination topology has similar property
                   as hypercube virtual topology.
                   For this topology gossiping is happening with the neighboring node,
                   then every 2nd node, every 4th, etc.

    >>>        .  *  .
    >>>      *          *
    >>>    .              .
    >>>    *              *
    >>>    .              .
    >>>      *          *
    >>>        .  *  .

    .. note::
        Current implementation does not support uneven number of nodes for a CUBE
        topology.

    """
    CUBE = auto()
    DISSEMINATION = auto()


class GossipGraDState(default.DefaultState):
    r"""
    Stores state needed to perform GossipGraD algorithm within a communication hook.

    .. note:: Note that this hook should be used with the NCCL PG backend and users
        must set the current GPU device with `torch.cuda.set_device` prior to
        ``GossipGraDState`` initialization, otherwise it will lead to
        unexpected hang issues during the gossiping stage.

    Args:
        num_modules (int): Number of FSDP modules to identify how many communication
            calls will be performed during a backpropagation pass.
        topology (Topology): A virtual topology to be used for gradient communication.
            (default: DISSEMINATION)
        local_process_group (ProcessGroup): Stores local subgroup,
            where intra-node communication will happen,
            by default a subgroup is initialized to workers, belonging to the same node.
            Should be provided together with `num_nodes`. When every local process group
            contains only one worker, then this worker is considered to be a separate
            node and local ``all_reduce`` and ``broadcast`` are not performed.
            (default: None)
        num_nodes (int): Number of nodes in a compute environment.
            Should be provided together with `local_process_group`.
            By default is initialized to the number of generated local subgroups.
            (default: None)
        master_process_group (ProcessGroup): Stores main workers,
            which are involved in inter-node communication. By default, will be
            composed from the workers with rank 0 in the local process group.
            (default: None)
        proc_per_node (int): Number of workers in each node. By default is initialized
            to the size of a local subgroup.
            (default: None)
        random_seed (int): A random seed, so that randomly generated topologies
            were the same on every worker.
            (default: 2403)

    """

    def __init__(
        self,
        num_modules,
        topology=None,
        local_process_group=None,
        num_nodes=None,
        master_process_group=None,
        proc_per_node=None,
        random_seed=2403,
    ):
        if num_modules is None or num_modules < 1:
            raise ValueError("`num_nodes` should bea positive integer.")
        self.num_modules = num_modules
        self.topology = topology or Topology.DISSEMINATION
        if local_process_group is None and num_nodes is None:
            self.local_process_group, subgroups = dist.new_subgroups()
            self.num_nodes = len(subgroups)
        else:
            if (
                local_process_group is not None
                and num_nodes is None
                or local_process_group is None
                and num_nodes is not None
            ):
                raise ValueError(
                    "`local_process_group` and `num_nodes` should be provided together."
                )
            self.local_process_group = local_process_group
            if num_nodes < 1:
                raise ValueError("`num_nodes` should be equal to 1 or more.")
            self.num_nodes = num_nodes

        if self.num_nodes % 2 != 0 and self.topology == Topology.CUBE:
            raise ValueError(
                "Current implementation doesn't support uneven number"
                " of nodes for CUBE topology."
            )

        super().__init__(self.local_process_group)
        self.proc_per_node = (
            proc_per_node
            if proc_per_node is not None
            else self.local_process_group.size()
        )
        if self.proc_per_node < 1:
            raise ValueError("`proc_per_node` should be equal to 1 or more.")

        self.master_process_group = (
            master_process_group
            if master_process_group is not None
            else self._create_master_group()
        )

        self.random_seed = random_seed
        self.topologies = self._generate_topologies(self.random_seed)
        self.cur_topology = next(self.topologies)

        # For `num_nodes` != power of 2 `gossip_period` should still be an int.
        # If we only have 1 node, `gossip_period` should be equal to 1.
        self.gossip_period = max(1, math.ceil(math.log(self.num_nodes, 2)))
        self.iter = 0

        # Get rank for current device
        self.rank = dist.get_rank()

        # Master worker for a current local `process_group`
        self.master_worker = dist.distributed_c10d._get_global_rank(
            self.local_process_group, 0
        )

    def _create_master_group(self):
        r"""
        Creates master process group, i.e. a group of workers,
        which communicate gradients between different nodes.
        """
        # Every 0th worker on every node will be assigned to a master group,
        # i.e. if number of rocesses per node is 8, master group contains
        # 0th, 8th, 16th, 24th, 32nd, ... ranks
        ranks = [i * self.proc_per_node for i in range(self.num_nodes)]
        return dist.new_group(ranks)

    def _generate_topologies(self, random_seed):
        r"""
        Creates `num_nodes` random topology shuffles and returns an infinite iterator.
        Original topology is of the form:
            [0*K, 1*K, ... , N*K],
        where N is the number of nodes and K - the number of workers on each node.
        For example, with N=4 and K=8, original topology is
            [0, 8, 16, 24]

        Workers' rank values are used instead of node values for easier peer assignment
        in a collective communication stage.

        Returns:
            An infinite iterator over created topologies
        """
        random.seed(random_seed)
        topologies_set = []
        original_list = [i * self.proc_per_node for i in range(self.num_nodes)]
        for _ in range(self.num_nodes):
            random.shuffle(original_list)
            topologies_set.append(original_list.copy())

        return cycle(topologies_set)


def _get_send_recv_peers(state):
    r"""
    Computes peers for the collective communication stage.
    For a ``CUBE`` topology a node sends grads to and receives from
    the same neighboring vertex. A pick for a neighboring vertex
    depends on the step number and current virtual topology in use.

    For a ``DISSEMINATION`` topology a node typically sends grads
    to and receives from different neighbors, but there may be a step
    where send and receive peers are the same node. A pick for send and receive peers
    depends on the step number and current virtual topology in use.

    For more information, please refer to the original
    `paper <https://arxiv.org/abs/1803.05880>`_

    Args:
        state (GossipGradState): State for GossipGraD communication hook.

    Returns:
        Peers' global ranks to whom a current node sends gradients
        and from whom it is received.
    """
    assert state.gossip_period > 0, "`gossip_period` should be greater than 0."
    power = (state.iter // state.num_modules) % state.gossip_period
    # Our new node_rank is a position of a global rank in
    # a virtual topology
    node_rank = state.cur_topology.index(state.rank)

    if state.topology == Topology.CUBE:
        peer_idx = node_rank ^ 2**power
        if peer_idx >= len(state.cur_topology):
            return INVALID_PEER, INVALID_PEER
        return state.cur_topology[peer_idx], state.cur_topology[peer_idx]

    elif state.topology == Topology.DISSEMINATION:
        send_peer_idx = (node_rank + 2**power) % state.num_nodes
        recv_peer_idx = (node_rank - 2**power + state.num_nodes) % state.num_nodes
        return state.cur_topology[send_peer_idx], state.cur_topology[recv_peer_idx]


def _gossip(state, grad, scaling_factor=0.5):
    r"""
    Gossiping stage.

    At this step, it obtains communication peers,
    stacks ``torch.distributed.irecv`` and ``torch.distributed.isend`` operations,
    and performs communication with ``torch.distributed.batch_isend_irecv``.
    Finally, received and current gradients are added together
    and scaled appropriately, i.e. since communication happens
    only between 2 peers at a time, summed gradients are divided
    by 2 (or multiplied by 0.5)

    For more information, please refer to the original
    `paper <https://arxiv.org/abs/1803.05880>`_

    Args:
        state (GossipGradState): State for GossipGraD communication hook.
        grad (torch.Tensor): A gradient for the local batch
            that needs to be communicated across ranks.
        scaling_facto (float): Scaling factor to apply after
            received and current gradients are combined.

    """
    send_peer, recv_peer = _get_send_recv_peers(state)

    if send_peer == INVALID_PEER or recv_peer == INVALID_PEER:
        return

    assert send_peer is not None and recv_peer is not None, (
        "Failed to calculate send and receive peers: "
        f"(`send_peer` is {send_peer} and `recv_peer` is {recv_peer})"
    )
    # Need to check that send and receive peers are not equal to a current rank
    assert send_peer != state.rank and recv_peer != state.rank, (
        "Expected send and receive peers to differ from a current rank: "
        f"(current rank is {state.rank}, `send_peer` is {send_peer}\
        and `recv_peer` is {recv_peer})"
    )
    assert (
        send_peer != -1 and recv_peer != -1
    ), "Communication peers are not present in a current topology"
    recv_grad = torch.empty_like(grad)
    ops = []

    # For ranks not in the `master_process_group`,
    # `master_process_group` is an `object` instance
    assert isinstance(
        state.master_process_group, ProcessGroup
    ), "`master_process_group` is not an instance of `ProcessGroup`"

    ops.append(
        dist.P2POp(
            op=dist.isend, tensor=grad, peer=send_peer, group=state.master_process_group
        )
    )
    ops.append(
        dist.P2POp(
            op=dist.irecv,
            tensor=recv_grad,
            peer=recv_peer,
            group=state.master_process_group,
        )
    )
    reqs = dist.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()
    grad.add_(recv_grad).mul_(scaling_factor)


def get_num_modules(module: torch.nn.Module):
    r"""
    Returns number of FSDP modules in a provided FSDP instance.

    Args:
        module (torch.nn.Module): FSDP instance

    Returns:
        int: number of FSDP modules that are nested in the input ``module``,
            including self.

    """
    return len(FSDP.fsdp_modules(module))


def gossip_grad_hook(state: GossipGraDState, grad: torch.Tensor):
    r"""
    Communication hook, that follows
    `GossipGraD <https://arxiv.org/abs/1803.05880>`_ strategy.

    Every ``state.gossip_period`` step a virtual topology is changed.
    Before an inter-node communication happens, gradients are reduced locally,
    i.e. in an intra-node fashion.

    Only workers from a master process group are participating in a gossiping stage.
    Finally, every main worker broadcasts final gradient to its local subgroup

    Args:
        state (GossipGradState): State for GossipGraD communication hook.
        grad (torch.Tensor): A gradient for the local batch
            that needs to be communicated across ranks.

    Here is an example for how to initialize a default ``GossipGraD state``
    and register an fsdp model with a communication hook.
    ::

        >>>  import torch
        >>>  import torch.distributed as dist
        >>>  from torch.distributed.fsdp import(
        >>>    FullyShardedDataParallel as FSDP
        >>>  )
        >>>  from torchdistx.gossip_grad import(
        >>>     GossipGraDState,
        >>>     Topology,
        >>>     get_num_modules,
        >>>     gossip_grad_hook
        >>>  )
        >>>
        >>>  net = torch.nn.Linear(4, 10)
        >>>  fsdp_net = FSDP(net)
        >>>  state = GossipGraDState(num_modules=get_num_modules(fsdp_net))
        >>>  fsdp_net.register_comm_hook(state, gossip_grad_hook)

    """
    # Virtual topology changes every `state.gossip_period` step.
    # FSDP net can consist of multiple FSDP modules and every module will
    # increase `state.iter` during the backward pass. As a result, we need
    # to adjust for this behavior and make sure that virtual topology doesn't
    # change in the middle of the backward pass.
    if (state.iter // state.num_modules) % state.gossip_period == 0:
        state.cur_topology = next(state.topologies)

    # Reduce local gradients
    default.allreduce_hook(state, grad)
    # Perform gossiping step between master nodes (via master workers)
    if not dist._rank_not_in_group(state.master_process_group):
        _gossip(state, grad)
    # Broadcast received gradients in the local process group
    dist.broadcast(grad, src=state.master_worker, group=state.local_process_group)

    state.iter += 1
