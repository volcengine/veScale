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

import copy

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.testing._internal.common_utils import (
    run_tests,
)

from vescale.dtensor.placement_types import Shard, Replicate
from vescale.dtensor.device_mesh import init_device_mesh
from vescale.dmodule.api import parallelize_module
from vescale.dtensor.api import distribute_tensor
from vescale.ddp.distributed_data_parallel import DistributedDataParallel as DDP
from vescale.optim.distributed_optimizer import DistributedOptimizer

from common_dtensor import DTensorTestBase, with_comms


HIDDEN_DIM = 4
BSZ = 3
SEQLEN = 5
VOCAB_SIZE = 7


def get_unfied_param_and_data(bsz, seqlen, vocab_size, hidden_dim):
    weight = torch.rand(vocab_size, hidden_dim).cuda()

    batch1_epoch1 = torch.randint(low=0, high=vocab_size, size=(bsz, seqlen)).cuda()
    batch2_epoch1 = torch.randint(low=0, high=vocab_size, size=(bsz, seqlen)).cuda()
    batch1_epoch2 = torch.randint(low=0, high=vocab_size, size=(bsz, seqlen)).cuda()
    batch2_epoch2 = torch.randint(low=0, high=vocab_size, size=(bsz, seqlen)).cuda()

    # allreduce parameter and batches to make sure they are same at all ranks
    torch.distributed.all_reduce(weight)
    # use max reduce op to avoid exceeding the vocab_size
    torch.distributed.all_reduce(batch1_epoch1, op=dist.ReduceOp.MAX)
    torch.distributed.all_reduce(batch2_epoch1, op=dist.ReduceOp.MAX)
    torch.distributed.all_reduce(batch1_epoch2, op=dist.ReduceOp.MAX)
    torch.distributed.all_reduce(batch2_epoch2, op=dist.ReduceOp.MAX)

    params_and_inputs = {
        "weight": weight,
        "batch1_epoch1": batch1_epoch1,
        "batch2_epoch1": batch2_epoch1,
        "batch1_epoch2": batch1_epoch2,
        "batch2_epoch2": batch2_epoch2,
    }

    return params_and_inputs


class SharedEmbedding(torch.nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.weight = torch.nn.Parameter(torch.rand(vocab_size, hidden_dim).cuda())

    def forward(self, token_ids):
        hidden_states = F.embedding(token_ids, self.weight)
        return torch.matmul(hidden_states, self.weight.T)


class DTensorSharedTest(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    def gen_golden(self, params_and_inputs, max_norm):
        m = SharedEmbedding(VOCAB_SIZE, HIDDEN_DIM).cuda()
        m.weight = torch.nn.Parameter(params_and_inputs["weight"])

        optimizer = torch.optim.Adam(m.parameters(), lr=0.01)

        optimizer.zero_grad()
        output = m(params_and_inputs["batch1_epoch1"])
        output.sum().backward()
        output = m(params_and_inputs["batch2_epoch1"])
        output.sum().backward()

        # manually reduce-mean the grad
        for p in m.parameters():
            p.grad /= 2

        torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm, norm_type=2.0)

        return m

    @with_comms
    def test_dtensor_shared_weight_without_pp(self):
        max_norm = 2.0
        tp_parallel_size = 2
        dp_size = self.world_size // tp_parallel_size
        device_mesh = init_device_mesh(self.device_type, (dp_size, tp_parallel_size), mesh_dim_names=("DP", "TP"))

        params_and_inputs = get_unfied_param_and_data(BSZ, SEQLEN, VOCAB_SIZE, HIDDEN_DIM)
        new_params_and_inputs = copy.deepcopy(params_and_inputs)
        tp_sub_mesh = device_mesh["TP"]
        dp_pg = device_mesh.get_dim_groups(0)
        tp_pg = device_mesh.get_dim_groups(1)

        ve_model = SharedEmbedding(VOCAB_SIZE, HIDDEN_DIM).cuda(self.rank)
        ve_model.weight = torch.nn.Parameter(params_and_inputs["weight"])

        param_sharding_plan = {"weight": [Shard(1)]}
        fwd_resharding_plan = {".input": [[Replicate()]], ".output": [[Replicate()]]}
        ve_model = parallelize_module(
            ve_model, tp_sub_mesh, {"parameter": param_sharding_plan, "forward": fwd_resharding_plan}
        )

        ve_model = DDP(
            ve_model,
            data_pg_or_device_mesh=dp_pg,
            accumulate_allreduce_grads_in_fp32=True,
            overlap_grad_reduce=False,
            use_distributed_optimizer=True,
        )

        orig_optimizer = torch.optim.Adam(ve_model.parameters(), lr=0.01)
        ve_optimizer = DistributedOptimizer(
            orig_optimizer,
            models=[ve_model],
            overlap_param_gather=False,
            clip_grad=max_norm,
        )

        optimizer_params = ve_optimizer.get_parameters()

        ve_optimizer.zero_grad()
        x = params_and_inputs["batch1_epoch1"]
        if self.rank in [2, 3]:
            x = params_and_inputs["batch2_epoch1"]
        ve_model(x).to_local().sum().backward()
        ve_model.finish_grad_sync()

        # copy main grad in grad buffer to sharded parameters' grad field.
        ve_optimizer._copy_model_grads_to_main_grads()
        # do the grad norm clipping
        ve_optimizer.clip_grad_norm(ve_optimizer.clip_grad)

        golden_model = self.gen_golden(new_params_and_inputs, max_norm)
        golden_weight_grad = distribute_tensor(golden_model.weight.grad.data, tp_sub_mesh, [Shard(1)])._local_tensor
        if self.rank in [0, 1]:
            ve_weight_grad = optimizer_params[0].grad
            torch.testing.assert_close(
                golden_weight_grad.flatten()[:7],
                ve_weight_grad,
            )
        if self.rank in [2, 3]:
            ve_weight_grad = optimizer_params[0].grad
            torch.testing.assert_close(
                golden_weight_grad.flatten()[7:],
                ve_weight_grad,
            )


if __name__ == "__main__":
    run_tests()
