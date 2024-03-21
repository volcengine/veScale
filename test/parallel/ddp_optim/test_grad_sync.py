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

from copy import deepcopy
from common_dtensor import DTensorTestBase, with_comms

import torch
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize, run_tests
from vescale.dtensor.api import distribute_tensor
from vescale.dmodule.api import parallelize_module
from vescale.dtensor.device_mesh import DeviceMesh
from vescale.dtensor.placement_types import Replicate, Shard
from vescale.ddp.distributed_data_parallel import DistributedDataParallel as DDP
from vescale.optim.distributed_optimizer import DistributedOptimizer
from vescale.optim.base_optimizer import BasicOptimizer, BaseOptimizerHook

HIDDEN_DIM = 16
BSZ = 2
SEQ_LEN = 4


class LN(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.ln = torch.nn.LayerNorm(hidden_dim)

    def forward(self, x):
        return self.ln(x)


class LNGradSyncTest(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @with_comms
    def test_counterexample(self):
        m = LN(HIDDEN_DIM)

        device_mesh = DeviceMesh(self.device_type, [0, 1, 2, 3])
        param_sharding_plan = {
            "ln.weight": [Replicate()],
            "ln.bias": [Replicate()],
        }
        inout_sharding = [Replicate()]
        fwd_sharding_plan = {"ln.input": [inout_sharding], "ln.output": [inout_sharding]}
        m = parallelize_module(
            m,
            device_mesh,
            {"parameter": param_sharding_plan, "forward": fwd_sharding_plan},
            grad_sync={torch.nn.LayerNorm: ["weight", "bias"]},
        )

        dx = distribute_tensor(torch.rand(BSZ, SEQ_LEN, HIDDEN_DIM), device_mesh, inout_sharding)
        dout = m(dx)
        torch.autograd.backward(dout, torch.ones_like(dout))
        self.assertTrue(m.ln.weight.grad.placements[0].is_replicate())
        self.assertTrue(m.ln.bias.grad.placements[0].is_replicate())
        m.finish_grad_sync()
        self.assertTrue(len(m.list_grad_sync()) == 0)

    @parametrize(
        "grad_sync",
        [
            False,
            True,
            {},
            {torch.nn.LayerNorm: []},
            {torch.nn.LayerNorm: True},
            {torch.nn.LayerNorm: ["weight", "bias"]},
            {torch.nn.LayerNorm: ["weight"]},
        ],
    )
    @with_comms
    def test_basic(self, grad_sync):
        m = LN(HIDDEN_DIM)

        device_mesh = DeviceMesh(self.device_type, [0, 1, 2, 3])
        param_sharding_plan = {
            "ln.weight": [Replicate()],
            "ln.bias": [Replicate()],
        }
        inout_sharding = [Shard(1)]
        fwd_sharding_plan = {"ln.input": [inout_sharding], "ln.output": [inout_sharding]}
        m = parallelize_module(
            m,
            device_mesh,
            {"parameter": param_sharding_plan, "forward": fwd_sharding_plan},
            grad_sync=grad_sync,
        )
        optimizer = torch.optim.Adam(m.parameters(), lr=1e-3)
        optimizer = BasicOptimizer(optimizer, models=m, grad_hook=BaseOptimizerHook)

        dx = distribute_tensor(torch.rand(BSZ, SEQ_LEN, HIDDEN_DIM), device_mesh, inout_sharding)
        dout = m(dx)
        torch.autograd.backward(dout, torch.ones_like(dout))
        self.assertTrue(m.ln.weight.grad.placements[0].is_partial())
        self.assertTrue(m.ln.bias.grad.placements[0].is_partial())
        # NOTE: now, we don't need to manually call ``m.finish_grad_sync()``, BasicOptimizer will
        # implicitly do that.
        optimizer.step()
        grad_sync_list = m.list_grad_sync()
        fqn_sync_list = set([fqn for fqn, _ in grad_sync_list])  # noqa: C403
        if grad_sync in (False, {}, {torch.nn.LayerNorm: []}):
            self.assertTrue(len(grad_sync_list) == 0)
            self.assertTrue(m.ln.weight.grad.placements[0].is_partial())
            self.assertTrue(m.ln.bias.grad.placements[0].is_partial())
        elif grad_sync in (True, {torch.nn.LayerNorm: True}, {torch.nn.LayerNorm: ["weight", "bias"]}):
            self.assertTrue(len(grad_sync_list) == 2)
            self.assertTrue("ln.weight.grad" in fqn_sync_list)
            self.assertTrue("ln.bias.grad" in fqn_sync_list)
            self.assertTrue(m.ln.weight.grad.placements[0].is_replicate())
            self.assertTrue(m.ln.bias.grad.placements[0].is_replicate())
        elif grad_sync in ({torch.nn.LayerNorm: ["weight"]},):
            self.assertTrue(len(grad_sync_list) == 1)
            self.assertTrue("ln.weight.grad" in fqn_sync_list)
            self.assertTrue("ln.bias.grad" not in fqn_sync_list)
            self.assertTrue(m.ln.weight.grad.placements[0].is_replicate())
            self.assertTrue(m.ln.bias.grad.placements[0].is_partial())
        else:
            raise ValueError(f"Unknown grad_sync: {grad_sync}")

    @parametrize("overlap_grad_reduce", [True, False])
    @parametrize("use_distributed_optimizer", [True, False])
    @with_comms
    def test_ddp(self, overlap_grad_reduce: bool, use_distributed_optimizer: bool):
        base_ln_weight_param = torch.rand(HIDDEN_DIM, device=self.device_type)
        base_ln_bias_param = torch.rand(HIDDEN_DIM, device=self.device_type)
        base_batch_1 = torch.rand(BSZ, SEQ_LEN, HIDDEN_DIM, device=self.device_type)
        base_batch_2 = torch.rand(BSZ, SEQ_LEN, HIDDEN_DIM, device=self.device_type)

        # allreduce parameter and batches to make sure they are same at all ranks
        torch.distributed.all_reduce(base_ln_weight_param)
        torch.distributed.all_reduce(base_ln_bias_param)
        torch.distributed.all_reduce(base_batch_1)
        torch.distributed.all_reduce(base_batch_2)

        ln_weight_param = deepcopy(base_ln_weight_param)
        ln_bias_param = deepcopy(base_ln_bias_param)
        batch_1 = deepcopy(base_batch_1)
        batch_2 = deepcopy(base_batch_2)

        # ------------- baseline start ------------- #
        base_ln_model = torch.nn.LayerNorm(HIDDEN_DIM)
        base_ln_model.weight = torch.nn.Parameter(base_ln_weight_param)
        base_ln_model.bias = torch.nn.Parameter(base_ln_bias_param)
        base_optimizer = torch.optim.Adam(base_ln_model.parameters(), lr=1e-3)

        base_batch = torch.cat([base_batch_1, base_batch_2], dim=0)

        base_optimizer.zero_grad()
        base_out = base_ln_model(base_batch)
        torch.autograd.backward(base_out, torch.ones_like(base_out))

        # manually reduce-mean the gradient.
        base_ln_weight_grad = base_ln_model.weight.grad / 2
        base_ln_bias_grad = base_ln_model.bias.grad / 2
        base_optimizer.step()
        base_ln_weight = base_ln_model.weight
        base_ln_bias = base_ln_model.bias

        # ------------- baseline end ------------- #

        m = LN(HIDDEN_DIM)
        m.ln.weight = torch.nn.Parameter(ln_weight_param)
        m.ln.bias = torch.nn.Parameter(ln_bias_param)

        device_mesh = DeviceMesh(self.device_type, [[0, 1], [2, 3]])
        tp_submesh = device_mesh.get_submesh([1])
        dp_pg = device_mesh.get_dim_groups(0)
        param_sharding_plan = {
            "ln.weight": [Replicate()],
            "ln.bias": [Replicate()],
        }
        inout_sharding = [Shard(1)]
        fwd_sharding_plan = {"ln.input": [inout_sharding], "ln.output": [inout_sharding]}
        m = parallelize_module(
            m,
            tp_submesh,
            {"parameter": param_sharding_plan, "forward": fwd_sharding_plan},
            grad_sync={torch.nn.LayerNorm: ["weight", "bias"]},
        )

        ddp_m = DDP(
            m,
            data_pg_or_device_mesh=dp_pg,
            accumulate_allreduce_grads_in_fp32=True,
            overlap_grad_reduce=overlap_grad_reduce,
            use_distributed_optimizer=use_distributed_optimizer,
        )
        optimizer = torch.optim.Adam(ddp_m.parameters(), lr=1e-3)
        if use_distributed_optimizer:
            optimizer = DistributedOptimizer(
                optimizer,
                clip_grad=0.0,
                overlap_param_gather=False,
                models=[ddp_m],
            )
        else:
            optimizer = BasicOptimizer(optimizer, models=ddp_m, grad_hook=BaseOptimizerHook)
        optimizer.zero_grad()
        if torch.distributed.get_rank() in (0, 1):
            dx = distribute_tensor(batch_1, tp_submesh, inout_sharding)
        else:
            dx = distribute_tensor(batch_2, tp_submesh, inout_sharding)

        dout = ddp_m(dx)
        torch.autograd.backward(dout, torch.ones_like(dout))
        ddp_m.finish_grad_sync()
        # NOTE: we must copy main_grad out here, because after `optimizer.step()`
        # grad buffer will be overwrited with updated param to perform param allgather.
        ddp_ln_weight_grad = deepcopy(ddp_m.module.ln.weight.main_grad)
        ddp_ln_bias_grad = deepcopy(ddp_m.module.ln.bias.main_grad)
        optimizer.step()
        ddp_ln_weight = ddp_m.module.ln.weight
        ddp_ln_bias = ddp_m.module.ln.bias

        # -------- check results -------- #

        grad_sync_list = ddp_m.module.list_grad_sync()
        fqn_sync_list = set([fqn for fqn, _ in grad_sync_list])  # noqa: C403
        self.assertTrue(len(grad_sync_list) == 2)
        self.assertTrue("ln.weight.main_grad" in fqn_sync_list)
        self.assertTrue("ln.bias.main_grad" in fqn_sync_list)

        self.assertTrue(ddp_ln_weight_grad._spec.placements[0].is_replicate())
        self.assertTrue(ddp_ln_bias_grad._spec.placements[0].is_replicate())

        # NOTE: we can do the following check just because of such a conincidence:
        # the bias and weight parameter of LayerNorm occupy the same size of memory,
        # after reduce_scatter grad along DP  dimension, DP 0 will hold the whole
        # gradients of bias, and DP 1 holds that of weight. Pay attention to that
        # params are traversed in a reverse order when constructing the grad buffer.
        if torch.distributed.get_rank() == 0 or torch.distributed.get_rank() == 1:
            torch.testing.assert_close(base_ln_bias_grad, ddp_ln_bias_grad)
        elif torch.distributed.get_rank() == 2 or torch.distributed.get_rank() == 3:
            torch.testing.assert_close(base_ln_weight_grad, ddp_ln_weight_grad)
        torch.testing.assert_close(base_ln_bias, ddp_ln_bias._local_tensor)
        torch.testing.assert_close(base_ln_weight, ddp_ln_weight._local_tensor)


instantiate_parametrized_tests(LNGradSyncTest)

if __name__ == "__main__":
    run_tests()
