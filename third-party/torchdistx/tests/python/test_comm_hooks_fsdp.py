# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import sys
import tempfile
from typing import Optional

import torch
import torch.nn as nn
from torch import distributed as dist
from torch.distributed.algorithms.model_averaging import averagers
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import ShardingStrategy
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

from torchdistx.gossip_grad import (
    GossipGraDState,
    Topology,
    get_num_modules,
    gossip_grad_hook,
)
from torchdistx.slowmo import slowmo_comm, slowmo_optimizer

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)


class Net(nn.Module):
    def __init__(self, has_wrapping, sharding_strategy):
        # to ensure determinism
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        super().__init__()

        self.linear1 = nn.Linear(8, 16)
        self.linear2 = nn.Linear(16, 8)
        self.out = nn.Linear(8, 4)

        fsdp_kwargs = {
            "device_id": torch.cuda.current_device(),
            "sharding_strategy": sharding_strategy,
        }

        self.net = self._maybe_wrap_fsdp(
            nn.Sequential(
                self._maybe_wrap_fsdp(
                    self.linear1, has_wrapping=has_wrapping, **fsdp_kwargs
                ),
                nn.ReLU(),
                self.linear2,
            ),
            has_wrapping=has_wrapping,
            **fsdp_kwargs,
        )

    def forward(self, x):
        return self.out(nn.functional.relu(self.net(x)))

    def _maybe_wrap_fsdp(self, module, has_wrapping, **kwargs):
        return module if not has_wrapping else FSDP(module, **kwargs)


class TestCommunicationHooks(FSDPTest):
    def _init_fsdp(self, sharding_strategy, net=None):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        net = net if net is not None else torch.nn.Linear(1, 5, bias=False)
        return FSDP(
            net,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy,
        )

    def _train_step(self, inpt, net, optim):
        optim.zero_grad()
        loss = net(inpt).sum()
        loss.backward()
        optim.step()

    def _init_averager(self, period):
        return averagers.PeriodicModelAverager(
            period=period, process_group=dist.distributed_c10d._get_default_group()
        )

    def _init_slowmo_optimizer(self, base_optim, slowmo_freq):
        return slowmo_optimizer.SlowMomentumOptimizer(
            base_optim=base_optim,
            slowmo_freq=slowmo_freq,
            slowmo_factor=0.5,
            slowmo_lr=0.1,
        )

    def _check_grads_eq_rank(self, net, inpt):
        net.zero_grad()
        loss = net(inpt).sum()
        loss.backward()
        self.assertEqual(net.params[0].grad[0], self.rank)

    @skip_if_lt_x_gpu(2)
    @parametrize("sharding_strategy", [ShardingStrategy.NO_SHARD])
    def test_slowmo_hook_with_grad_sync(
        self, sharding_strategy: Optional[ShardingStrategy]
    ):

        fsdp_net = self._init_fsdp(sharding_strategy)
        inpt = torch.tensor(
            [self.rank], dtype=torch.float, device=self.rank  # type: ignore[arg-type]
        )

        slowmo_state = slowmo_comm.SlowMoState(subgroup=None, sync_grads=True)
        # check that a default subgroup was created,
        # for small scale experiments equal to `world_size`
        self.assertEqual(slowmo_state.subgroup.size(), dist.get_world_size())

        cur_subgroup = dist.new_group(ranks=[self.rank])
        self.assertEqual(cur_subgroup.size(), 1)
        slowmo_state = slowmo_comm.SlowMoState(cur_subgroup, sync_grads=True)
        # check that state has subgroup registered
        self.assertEqual(slowmo_state.subgroup.size(), cur_subgroup.size())
        self.assertEqual(slowmo_state.subgroup.rank(), 0)

        fsdp_net.register_comm_hook(slowmo_state, slowmo_comm.slowmo_hook)

        # Make sure grads were not reduced,
        # since each subgroup is only one worker.
        # Gradient in this case is equal to rank
        self._check_grads_eq_rank(fsdp_net, inpt)

    @skip_if_lt_x_gpu(2)
    @parametrize("sharding_strategy", [ShardingStrategy.NO_SHARD])
    def test_slowmo_hook_no_grad_sync(
        self, sharding_strategy: Optional[ShardingStrategy]
    ):

        fsdp_net = self._init_fsdp(sharding_strategy)
        inpt = torch.tensor(
            [self.rank], dtype=torch.float, device=self.rank  # type: ignore[arg-type]
        )

        # create a subgroup equal to the whole WORLD
        cur_subgroup = dist.distributed_c10d._get_default_group()
        self.assertEqual(cur_subgroup.size(), dist.get_world_size())
        slowmo_state = slowmo_comm.SlowMoState(cur_subgroup, sync_grads=False)
        # check that state has subgroup registered
        self.assertEqual(slowmo_state.subgroup.size(), cur_subgroup.size())

        fsdp_net.register_comm_hook(slowmo_state, slowmo_comm.slowmo_hook)

        # Make sure grads were not reduced, since `sync_grads` is set to False
        # Gradient in this case is equal to rank
        self._check_grads_eq_rank(fsdp_net, inpt)

    @skip_if_lt_x_gpu(2)
    @parametrize("sharding_strategy", [ShardingStrategy.NO_SHARD])
    def test_slowmo_optimizer_averager(
        self, sharding_strategy: Optional[ShardingStrategy]
    ):
        fsdp_net = self._init_fsdp(
            sharding_strategy,
            net=Net(has_wrapping=True, sharding_strategy=sharding_strategy),
        )
        fsdp_net_slowmo = self._init_fsdp(
            sharding_strategy,
            net=Net(has_wrapping=True, sharding_strategy=sharding_strategy),
        )

        cur_subgroup = dist.new_group(ranks=[self.rank])
        slowmo_state = slowmo_comm.SlowMoState(cur_subgroup, sync_grads=False)
        fsdp_net.register_comm_hook(slowmo_state, slowmo_comm.slowmo_hook)
        fsdp_net_slowmo.register_comm_hook(slowmo_state, slowmo_comm.slowmo_hook)
        inpt = torch.randn(  # type: ignore[call-overload]
            (7, 8), dtype=torch.float, device=self.rank
        )

        slowmo_optim = self._init_slowmo_optimizer(
            base_optim=torch.optim.Adam(fsdp_net_slowmo.parameters(), lr=1e-2),
            slowmo_freq=6,
        )

        # Manually changing slow momentum optimizer's averager's period
        # to differ from `slowmo_freq` to check it independently from
        # the momentum's update. Basically, parameter averaging now will happen
        # every 3rd step and momentum step every 6th.
        slowmo_optim.averager.period = 3

        averager2 = self._init_averager(period=3)
        base_optimizer = torch.optim.Adam(fsdp_net.parameters(), lr=1e-2)

        for _ in range(4):
            self._train_step(inpt, fsdp_net, base_optimizer)
            self._train_step(inpt, fsdp_net_slowmo, slowmo_optim)
            averager2.average_parameters(fsdp_net.parameters())

        for slowmo_params, net_params in zip(
            fsdp_net_slowmo.parameters(), fsdp_net.parameters()
        ):
            self.assertEqual(slowmo_params, net_params)

    @skip_if_lt_x_gpu(2)
    @parametrize("sharding_strategy", [ShardingStrategy.NO_SHARD])
    def test_slowmo_optimizer_momentum_step(
        self, sharding_strategy: Optional[ShardingStrategy]
    ):
        # Test assumes `fsdp_net` has a single top-level FSDP wrap,
        # i.e. no nested FSDP modules
        fsdp_net = self._init_fsdp(sharding_strategy)
        fsdp_net_slowmo = self._init_fsdp(sharding_strategy)
        learning_rate = 1e-2

        cur_subgroup = dist.new_group(ranks=[self.rank])
        slowmo_state = slowmo_comm.SlowMoState(cur_subgroup, sync_grads=False)
        fsdp_net.register_comm_hook(slowmo_state, slowmo_comm.slowmo_hook)
        fsdp_net_slowmo.register_comm_hook(slowmo_state, slowmo_comm.slowmo_hook)
        inpt = torch.tensor(
            [(self.rank + 1)],
            dtype=torch.float,
            device=self.rank,  # type: ignore[arg-type]
        )

        slowmo_optim = self._init_slowmo_optimizer(
            base_optim=torch.optim.SGD(fsdp_net_slowmo.parameters(), lr=learning_rate),
            slowmo_freq=2,
        )
        averager2 = self._init_averager(period=2)
        base_optimizer = torch.optim.SGD(fsdp_net.parameters(), lr=learning_rate)

        for param in fsdp_net_slowmo.params:
            initial_prev_params = param.detach().clone()
            initial_slow_momentum_buffer = torch.zeros_like(initial_prev_params)

        for _ in range(3):
            self._train_step(inpt, fsdp_net, base_optimizer)
            self._train_step(inpt, fsdp_net_slowmo, slowmo_optim)
            averager2.average_parameters(fsdp_net.parameters())

        # parameters before slow momentum update and after averaging
        # are in `fsdp_net.params[0]`
        # can use them to calculate momentum update
        # momentum_(t+1) = slowmo_factor * momentum_t +
        #   (prev_param - cur_param)/base_lr
        momentum = (
            slowmo_optim.slowmo_factor * initial_slow_momentum_buffer
            + (initial_prev_params - fsdp_net.params[0]) / learning_rate
        )

        # parameter_(t+1) = prev_param - slowmo_lr * base_lr * momentum_(t+1)
        calculated_params = initial_prev_params - 0.1 * learning_rate * momentum

        self.assertEqual(fsdp_net_slowmo.params[0], calculated_params)

    @skip_if_lt_x_gpu(2)
    @parametrize("sharding_strategy", [ShardingStrategy.NO_SHARD])
    def test_slowmo_optimizer_state_dict(
        self, sharding_strategy: Optional[ShardingStrategy]
    ):
        chkpt = tempfile.gettempdir() + "/checkpoint.pt"
        fsdp_net_slowmo = FSDP(
            Net(has_wrapping=False, sharding_strategy=sharding_strategy),
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy,
        ).to(self.rank)
        n_steps = 10

        cur_subgroup = dist.new_group(ranks=[self.rank])
        slowmo_state = slowmo_comm.SlowMoState(cur_subgroup)
        fsdp_net_slowmo.register_comm_hook(slowmo_state, slowmo_comm.slowmo_hook)
        inpt = torch.randn(  # type: ignore[call-overload]
            (7, 8), dtype=torch.float, device=self.rank
        )

        slowmo_optim = self._init_slowmo_optimizer(
            base_optim=torch.optim.SGD(fsdp_net_slowmo.parameters(), lr=1e-2),
            slowmo_freq=4,
        )

        for _ in range(n_steps):
            self._train_step(inpt, fsdp_net_slowmo, slowmo_optim)

        state = {"optim_state_dict": slowmo_optim.state_dict()}

        if self.rank == 0:
            torch.save(state, chkpt)

        dist.barrier()

        map_location = {"cuda:%d" % 0: "cuda:%d" % self.rank}
        checkpoint = torch.load(chkpt, map_location=map_location)

        # initialize dummy optimizer with different parameters
        slowmo_optim_dummy = slowmo_optimizer.SlowMomentumOptimizer(
            base_optim=torch.optim.SGD(fsdp_net_slowmo.parameters(), lr=1e-2),
            slowmo_freq=2,
            slowmo_factor=3,
            slowmo_lr=4,
        )
        slowmo_optim_dummy.load_state_dict(checkpoint["optim_state_dict"])

        # make sure averager's period and step were updated
        self.assertEqual(
            slowmo_optim_dummy.averager.period, slowmo_optim.averager.period
        )
        self.assertEqual(slowmo_optim_dummy.averager.step, slowmo_optim.averager.step)

        # make sure slowmo parameters were updated
        self.assertEqual(slowmo_optim_dummy.slowmo_freq, slowmo_optim.slowmo_freq)
        self.assertEqual(slowmo_optim_dummy.slowmo_factor, slowmo_optim.slowmo_factor)
        self.assertEqual(slowmo_optim_dummy.slowmo_lr, slowmo_optim.slowmo_lr)

        for _ in range(n_steps):
            self._train_step(inpt, fsdp_net_slowmo, slowmo_optim_dummy)

        self.assertEqual(slowmo_optim_dummy.averager.step, 2 * n_steps)

        # Check abscent learning rate in a checkpoint
        checkpoint = torch.load(chkpt, map_location=map_location)
        del checkpoint["optim_state_dict"]["param_groups"][0]["lr"]
        with self.assertRaisesRegex(
            ValueError, "All parameter groups should have learning rate specified."
        ):
            slowmo_optim_dummy.load_state_dict(checkpoint["optim_state_dict"])

    @skip_if_lt_x_gpu(2)
    def test_slowmo_optimizer_errors(self):
        net = torch.nn.Linear(1, 3, bias=False)
        with self.assertRaisesRegex(
            ValueError, "Base optimizer is a required" " parameter."
        ):
            _ = slowmo_optimizer.SlowMomentumOptimizer(
                base_optim=None, slowmo_freq=4, slowmo_factor=0.5, slowmo_lr=0.1
            )

        with self.assertRaisesRegex(
            ValueError, "Invalid ``slowmo_freq`` parameter, must be a positive value."
        ):
            _ = slowmo_optimizer.SlowMomentumOptimizer(
                base_optim=torch.optim.SGD(net.parameters(), lr=1e-2),
                slowmo_freq=-3,
                slowmo_factor=0.5,
                slowmo_lr=0.1,
            )

        with self.assertRaisesRegex(
            ValueError, "Invalid ``slowmo_factor`` parameter, must be non-negative."
        ):
            _ = slowmo_optimizer.SlowMomentumOptimizer(
                base_optim=torch.optim.SGD(net.parameters(), lr=1e-2),
                slowmo_freq=4,
                slowmo_factor=-0.5,
                slowmo_lr=0.1,
            )

        with self.assertRaisesRegex(
            ValueError, "Invalid ``slowmo_lr`` parameter, must be non-negative."
        ):
            _ = slowmo_optimizer.SlowMomentumOptimizer(
                base_optim=torch.optim.SGD(net.parameters(), lr=1e-2),
                slowmo_freq=4,
                slowmo_factor=0.5,
                slowmo_lr=-0.1,
            )

    @skip_if_lt_x_gpu(2)
    @parametrize("sharding_strategy", [ShardingStrategy.NO_SHARD])
    def test_slowmo_optimizer_buffer(self, sharding_strategy):

        # default simple net has size=(1, 5)
        fsdp_net_slowmo = self._init_fsdp(sharding_strategy)
        inpt = torch.tensor(
            [self.rank], dtype=torch.float, device=self.rank  # type: ignore[arg-type]
        )
        slowmo_optim = self._init_slowmo_optimizer(
            base_optim=torch.optim.SGD(fsdp_net_slowmo.parameters(), lr=1e-2),
            slowmo_freq=2,
        )
        self.assertEqual(
            slowmo_optim._prev_parameters[0], torch.flatten(fsdp_net_slowmo.weight)
        )

        for _ in range(3):
            self._train_step(inpt, fsdp_net_slowmo, slowmo_optim)

        slowmo_statedict = slowmo_optim.state_dict()
        for entry in slowmo_statedict["state"].values():
            self.assertIn("slow_momentum", entry)
        self.assertEqual(len(slowmo_optim._prev_parameters), 1)
        w2 = torch.ones(3, 3).to(self.rank)
        w2.requires_grad = True
        slowmo_optim.add_param_group({"params": w2})
        self.assertEqual(len(slowmo_optim._prev_parameters), 2)
        # At this point we have 2 parameter groups and should be able to
        # run with both of them, `slow_momentum` should appear in optimizer's state
        # for the second group.
        for _ in range(3):
            self._train_step(inpt, fsdp_net_slowmo, slowmo_optim)
        for entry in slowmo_statedict["state"].values():
            self.assertIn("slow_momentum", entry)

    @skip_if_lt_x_gpu(2)
    @parametrize("sharding_strategy", [ShardingStrategy.NO_SHARD])
    def test_gossip_grad_state_init(self, sharding_strategy):
        num_devices = torch.cuda.device_count()
        with self.assertRaisesRegex(
            ValueError,
            "`num_nodes` should bea positive integer.",
        ):
            state = GossipGraDState(num_modules=None, num_nodes=2)
        with self.assertRaisesRegex(
            ValueError,
            "`local_process_group` and `num_nodes` should be provided together.",
        ):
            state = GossipGraDState(num_modules=1, num_nodes=2)
        with self.assertRaisesRegex(
            ValueError,
            "`local_process_group` and `num_nodes` should be provided together.",
        ):
            state = GossipGraDState(
                num_modules=1, local_process_group=dist.new_group(ranks=[self.rank])
            )
        with self.assertRaisesRegex(
            ValueError,
            "Current implementation doesn't support uneven number"
            " of nodes for CUBE topology.",
        ):
            state = GossipGraDState(
                num_modules=1,
                topology=Topology.CUBE,
                local_process_group=dist.new_group(ranks=[self.rank]),
                num_nodes=5,
            )

        state = GossipGraDState(num_modules=1)
        self.assertIsNotNone(state.topology)
        self.assertEqual(state.topology, Topology.DISSEMINATION)
        self.assertIsNotNone(state.num_nodes)
        self.assertEqual(state.num_nodes, 1)
        self.assertIsNotNone(state.local_process_group)
        self.assertEqual(state.local_process_group.size(), num_devices)
        self.assertEqual(state.proc_per_node, state.local_process_group.size())
        if self.rank == 0:
            self.assertEqual(state.master_process_group.size(), 1)
        self.assertEqual(len(state.cur_topology), 1)
        self.assertEqual(state.gossip_period, 1)
        self.assertEqual(state.rank, self.rank)
        self.assertEqual(state.master_worker, 0)

        state = GossipGraDState(
            num_modules=1,
            topology=Topology.CUBE,
            local_process_group=dist.new_group(ranks=[self.rank]),
            num_nodes=num_devices,
        )
        self.assertEqual(state.topology, Topology.CUBE)

    @skip_if_lt_x_gpu(6)
    @parametrize("sharding_strategy", [ShardingStrategy.NO_SHARD])
    def test_gossip_grad_communication_dissemination(self, sharding_strategy):
        # default simple net has size=(1, 5)
        fsdp_net = self._init_fsdp(sharding_strategy)
        inpt = torch.tensor(
            [self.rank], dtype=torch.float, device=self.rank  # type: ignore[arg-type]
        )
        # The following setting is created:
        # existing workers are assigned in groups of 2, each group is
        # considered as a node.
        local_process_group, _ = dist.new_subgroups(group_size=2)
        master_ranks = list(range(0, torch.cuda.device_count(), 2))
        master_process_group = dist.new_group(ranks=master_ranks)
        num_nodes = torch.cuda.device_count() // 2
        state = GossipGraDState(
            num_modules=get_num_modules(fsdp_net),
            topology=Topology.DISSEMINATION,
            local_process_group=local_process_group,
            num_nodes=num_nodes,
            master_process_group=master_process_group,
            proc_per_node=2,
        )
        fsdp_net.register_comm_hook(state, gossip_grad_hook)

        # For this test there will be only one default (i.e. [0, 2, 4, ...])
        # topology for ease of computation, thus mahually hardcode 1 topology
        state.topologies = itertools.cycle([master_ranks])
        state.cur_topology = next(state.topologies)
        # There will be only `state.gossip_period` different communication peer changes,
        # new iteration -> new peer.
        # Thus, only checking `state.gossip_period` possible steps.
        for _ in range(1):  # state.gossip_period*5):
            loss = fsdp_net(inpt).sum()
            loss.backward()
            dist.barrier()
            # For ease of computation, manually set virtual topology
            power = (state.iter - 1) % state.gossip_period

            # Next, I precompute estimated grads for rank 0
            recvs_from = state.cur_topology[
                (0 - 2**power + len(state.cur_topology)) % len(state.cur_topology)
            ]

            # Receiving grads are equal to 0.5 + the global rank of the neighboring node
            # Gradients on the global node 0 (current node) are (0 + 1)/2 = 0.5
            # Scaling factor is 0.5 since we have 2 nodes
            estimated_grad = ((0.5 + recvs_from) + 0.5) * 0.5

            if self.rank == 0 or self.rank == 1:
                # Rank 1 should have same gradients as rank 0,
                # because 0 broadcasts grads.
                self.assertEqual(fsdp_net.params[0].grad[0], estimated_grad)

            # Make sure that node 0 and last node have different gradients
            # The only case, when these 2 will have the same gradients, is when
            # we only have 2 nodes in total. This test skips those,
            # since minimum requirement is 6 gpus = 3 nodes.
            if self.rank == master_ranks[-1]:
                # The last master node should have different grads
                self.assertNotEqual(fsdp_net.params[0].grad[0], estimated_grad)

            fsdp_net.zero_grad()

    @skip_if_lt_x_gpu(6)
    @parametrize("sharding_strategy", [ShardingStrategy.NO_SHARD])
    def test_gossip_grad_communication_cube(self, sharding_strategy):
        # default simple net has size=(1, 5)
        fsdp_net = self._init_fsdp(sharding_strategy)
        inpt = torch.tensor(
            [self.rank], dtype=torch.float, device=self.rank  # type: ignore[arg-type]
        )

        # The following setting is created:
        # existing workers are assigned in groups of 2, each group is
        # considered as a node.
        local_process_group, _ = dist.new_subgroups(group_size=1)
        num_nodes = torch.cuda.device_count()
        master_ranks = list(range(num_nodes))
        master_process_group = dist.new_group(ranks=master_ranks)
        state = GossipGraDState(
            num_modules=get_num_modules(fsdp_net),
            topology=Topology.CUBE,
            local_process_group=local_process_group,
            num_nodes=num_nodes,
            master_process_group=master_process_group,
            proc_per_node=1,
        )
        fsdp_net.register_comm_hook(state, gossip_grad_hook)

        # For this test there will be only one default (i.e. [0, 1, 2, ...])
        # topology for ease of computation, thus mahually hardcode 1 topology
        state.topologies = itertools.cycle([master_ranks])
        state.cur_topology = next(state.topologies)

        # There will be only `state.gossip_period` different communication peer changes,
        # new iteration -> new peer.
        # Thus, only checking `state.gossip_period` possible steps.
        for _ in range(state.gossip_period):
            loss = fsdp_net(inpt).sum()
            loss.backward()
            dist.barrier()
            # For ease of computation, manually set virtual topology
            power = (state.iter - 1) % state.gossip_period

            # Next, I precompute estimated grads for rank 0
            recvs_from = state.cur_topology[(0 ^ 2**power) % len(state.cur_topology)]

            # Receiving grads are equal to the global rank of the communication peer.
            # This is because there is no intra-node reduction.
            # Gradients on the global node 0 (current node) are (0 + 1)/2 = 0.5
            # Scaling factor is 0.5 since we have 2 nodes
            estimated_grad = (0 + recvs_from) * 0.5

            if self.rank == 0 and self.rank == recvs_from:
                # Rank 1 should have same gradients as rank 0,
                # because 0 broadcasts grads.
                self.assertEqual(fsdp_net.params[0].grad[0], estimated_grad)

            # Make sure that node 0 and last node have different gradients
            # Node 0 only communicates with nodes 1, 2, 4.
            if self.rank == master_ranks[-1]:
                # The last master node should have different grads
                self.assertNotEqual(fsdp_net.params[0].grad[0], estimated_grad)

            fsdp_net.zero_grad()

    @skip_if_lt_x_gpu(2)
    @parametrize("sharding_strategy", [ShardingStrategy.NO_SHARD])
    def test_gossip_grad_get_num_modules(self, sharding_strategy):
        # default simple net has size=(1, 5)
        fsdp_net = self._init_fsdp(
            sharding_strategy,
            net=Net(has_wrapping=True, sharding_strategy=sharding_strategy),
        )
        expected_num_modules = 3
        self.assertEqual(expected_num_modules, get_num_modules(fsdp_net))

    @skip_if_lt_x_gpu(2)
    @parametrize("sharding_strategy", [ShardingStrategy.NO_SHARD])
    def test_gossip_grad_iteration_correctness(self, sharding_strategy):
        # default simple net has size=(1, 5)
        fsdp_net = self._init_fsdp(
            sharding_strategy,
            net=Net(has_wrapping=True, sharding_strategy=sharding_strategy),
        )
        inpt = torch.randn(  # type: ignore[call-overload]
            (7, 8), dtype=torch.float, device=self.rank
        )

        # The following setting is created:
        # existing workers are assigned in groups of 2, each group is
        # considered as a node.
        local_process_group, _ = dist.new_subgroups(group_size=1)
        num_nodes = torch.cuda.device_count()
        master_ranks = list(range(num_nodes))
        master_process_group = dist.new_group(ranks=master_ranks)
        state = GossipGraDState(
            num_modules=get_num_modules(fsdp_net),
            topology=Topology.CUBE,
            local_process_group=local_process_group,
            num_nodes=num_nodes,
            master_process_group=master_process_group,
            proc_per_node=1,
        )
        fsdp_net.register_comm_hook(state, gossip_grad_hook)

        # For this test there will be only one default (i.e. [0, 1, 2, ...])
        # topology for ease of computation, thus mahually hardcode 1 topology
        state.topologies = itertools.cycle([master_ranks])
        state.cur_topology = next(state.topologies)
        num_epochs = 5

        # There will be only `state.gossip_period` different communication peer changes,
        # new iteration -> new peer.
        # Thus, only checking `state.gossip_period` possible steps.
        for _ in range(num_epochs):
            loss = fsdp_net(inpt).sum()
            loss.backward()
            fsdp_net.zero_grad()

        # At this point state.iter should be equal to 15, because
        # we have 3 FSDP modules in fsdp_net, thus in 5 iterations
        # `state.iter` increases 3*num_epochstimes
        expected_iteration = 3 * num_epochs
        self.assertEqual(expected_iteration, state.iter)
        self.assertEqual(num_epochs, state.iter // get_num_modules(fsdp_net))


instantiate_parametrized_tests(TestCommunicationHooks)

if __name__ == "__main__":
    run_tests()
