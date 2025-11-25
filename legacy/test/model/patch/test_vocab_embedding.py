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

import os
from unittest.mock import patch

import torch
from torch import nn
from torch.testing._internal.common_utils import run_tests
from common_dtensor import DTensorTestBase, with_comms

from vescale import Shard, Replicate, distribute_tensor, DeviceMesh
from vescale.dmodule.api import parallelize_module

CUR_DIR = os.path.dirname(os.path.realpath(__file__))

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)

NUM_EMBEDDINGS = 1024
INPUT_SIZE = 256


class VocabEmbedding(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.emb = nn.Embedding(NUM_EMBEDDINGS, 2048)

    def forward(self, x):
        return self.emb(x)


class VocabEmbeddingTest(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @with_comms
    @patch("vescale.model.patch.vp_embedding.VocabParallelEmbedding.forward")
    def test_non_parallel_embedding(self, patched_forward):
        device_mesh = DeviceMesh("cuda", list(range(self.world_size)))
        torch.manual_seed(9999)
        input_tensor = torch.randint(low=0, high=NUM_EMBEDDINGS, size=(INPUT_SIZE,)).cuda()
        dmodel = VocabEmbedding()
        param_sharding_plan = {}
        fwd_resharding_plan = {".input": [[Replicate()]]}
        parallelize_module(dmodel, device_mesh, {"parameter": param_sharding_plan, "forward": fwd_resharding_plan})
        _ = dmodel(input_tensor)
        patched_forward.assert_not_called()

    @with_comms
    @patch("vescale.model.patch.vp_embedding.VocabParallelEmbedding.forward")
    def test_embedding_parallel_embedding(self, patched_forward):
        device_mesh = DeviceMesh("cuda", list(range(self.world_size)))
        torch.manual_seed(9999)
        # input_tensor = torch.ones(128, dtype=torch.int64).cuda()
        input_tensor = torch.randint(low=0, high=NUM_EMBEDDINGS, size=(INPUT_SIZE,)).cuda()
        dmodel = VocabEmbedding()
        param_sharding_plan = {"emb.weight": [Shard(1)]}
        fwd_resharding_plan = {".input": [[Replicate()]]}
        parallelize_module(dmodel, device_mesh, {"parameter": param_sharding_plan, "forward": fwd_resharding_plan})
        _ = dmodel(input_tensor)
        patched_forward.assert_not_called()

    @with_comms
    def test_parallel_embedding(self):
        torch.manual_seed(9999)
        input_tensor = torch.randint(low=0, high=NUM_EMBEDDINGS, size=(INPUT_SIZE,)).cuda()
        golden_model = VocabEmbedding().cuda()
        golden_out = golden_model(input_tensor)
        loss = golden_out.mean()
        loss.backward()
        golden_weight_grad = golden_model.emb.weight.grad

        device_mesh = DeviceMesh("cuda", list(range(self.world_size)))
        torch.manual_seed(9999)
        input_tensor = torch.randint(low=0, high=NUM_EMBEDDINGS, size=(INPUT_SIZE,)).cuda()
        dmodel = VocabEmbedding()
        param_sharding_plan = {"emb.weight": [Shard(0)]}
        fwd_resharding_plan = {".input": [[Replicate()]]}
        parallelize_module(dmodel, device_mesh, {"parameter": param_sharding_plan, "forward": fwd_resharding_plan})
        dt_out = dmodel(input_tensor)
        local_out = dt_out.redistribute(placements=[Replicate()] * device_mesh.ndim).to_local()
        loss = local_out.mean()
        loss.backward()
        self.assertEqual(len(dmodel.emb.weight.grad.placements), 1)
        self.assertTrue(dmodel.emb.weight.grad.placements[0].is_shard(0))
        dmodel_weight_grad = dmodel.emb.weight.grad.redistribute(placements=[Replicate()] * device_mesh.ndim).to_local()

        torch.testing.assert_close(golden_out, local_out)

        torch.testing.assert_close(golden_weight_grad, dmodel_weight_grad)

    @with_comms
    def test_incorrectly_parallelled_embedding1(self):
        device_mesh = DeviceMesh("cuda", list(range(self.world_size)))
        torch.manual_seed(9999)
        input_tensor = torch.randint(low=0, high=NUM_EMBEDDINGS, size=(INPUT_SIZE,)).cuda()
        dmodel = VocabEmbedding()
        param_sharding_plan = {"emb.weight": [Shard(0)]}
        fwd_resharding_plan = {"emb.input": [[Shard(0)]]}
        parallelize_module(dmodel, device_mesh, {"parameter": param_sharding_plan, "forward": fwd_resharding_plan})
        with self.assertRaises(NotImplementedError):
            _ = dmodel(input_tensor)

    @with_comms
    def test_incorrectly_parallelled_embedding2(self):
        device_mesh = DeviceMesh("cuda", list(range(self.world_size)))
        torch.manual_seed(9999)
        input_tensor = torch.randint(low=0, high=NUM_EMBEDDINGS, size=(INPUT_SIZE,))
        input_tensor = distribute_tensor(input_tensor, device_mesh, placements=[Replicate()])
        input_tensor = input_tensor.redistribute(placements=[Shard(0)])
        dmodel = VocabEmbedding()
        param_sharding_plan = {"emb.weight": [Shard(0)]}
        fwd_resharding_plan = {}
        parallelize_module(dmodel, device_mesh, {"parameter": param_sharding_plan, "forward": fwd_resharding_plan})
        with self.assertRaises(AssertionError):
            _ = dmodel(input_tensor)


if __name__ == "__main__":
    run_tests()
