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
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    run_tests,
)

from common_dtensor import DTensorTestBase, with_comms

from vescale import Shard, Replicate, distribute_tensor, DeviceMesh
from vescale.dmodule.api import parallelize_module

CUR_DIR = os.path.dirname(os.path.realpath(__file__))

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)

VOCAB = 1024


class VocabCrossEntropy(nn.Module):
    def __init__(
        self,
        **kwarg,
    ):
        super().__init__()
        self.ls_fn = nn.CrossEntropyLoss(reduction="none", **kwarg)

    def forward(self, x, label):
        return self.ls_fn(x, label)


class VocabCrossEntropyTest(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @with_comms
    @patch("vescale.model.patch.vp_cross_entropy._VocabParallelCrossEntropy.forward")
    def test_non_parallel_cross_entropy(self, patched_forward):
        device_mesh = DeviceMesh("cuda", list(range(self.world_size)))
        torch.manual_seed(9999)
        input_tensor = torch.rand(size=(197, 1, VOCAB)).cuda().view(-1, VOCAB)
        input_label = torch.randint(low=0, high=1024, size=(197,)).cuda()
        dmodel = VocabCrossEntropy()
        param_sharding_plan = {}
        fwd_resharding_plan = {".input": [[Replicate()], [Replicate()]]}
        parallelize_module(dmodel, device_mesh, {"parameter": param_sharding_plan, "forward": fwd_resharding_plan})
        _ = dmodel(input_tensor, input_label)
        patched_forward.assert_not_called()

    @with_comms
    def test_parallel_cross_entropy(self):
        device_mesh = DeviceMesh("cuda", list(range(self.world_size)))
        model = VocabCrossEntropy()
        torch.manual_seed(9999)
        input_tensor = torch.rand(size=(197, 1, VOCAB)).cuda()
        input_tensor.requires_grad_()
        input_label = torch.randint(low=0, high=1024, size=(197,)).cuda()
        golden_out = model(input_tensor.view(-1, VOCAB), input_label)
        loss = golden_out.mean()
        loss.backward()
        golden_grad = input_tensor.grad

        # vescale
        input_tensor = input_tensor.detach().clone()
        d_input_tensor = distribute_tensor(input_tensor, device_mesh, placements=[Replicate()])
        d_input_tensor = d_input_tensor.redistribute(placements=[Shard(2)])
        d_input_tensor.requires_grad_()
        d_input_tensor.retain_grad()
        d_input_label = distribute_tensor(input_label, device_mesh, placements=[Replicate()])
        dmodel = VocabCrossEntropy()
        param_sharding_plan = {}
        fwd_resharding_plan = {".input": [[Shard(1)], [Replicate()]]}
        parallelize_module(dmodel, device_mesh, {"parameter": param_sharding_plan, "forward": fwd_resharding_plan})
        vescale_out = dmodel(d_input_tensor.view(-1, VOCAB), d_input_label)
        vescale_out = vescale_out.redistribute(placements=[Replicate()]).to_local()
        loss = vescale_out.mean()
        loss.backward()
        vescale_grad = d_input_tensor.grad.redistribute(placements=[Replicate()]).to_local()

        torch.testing.assert_close(golden_out, vescale_out)
        torch.testing.assert_close(golden_grad, vescale_grad)


instantiate_parametrized_tests(VocabCrossEntropyTest)

if __name__ == "__main__":
    run_tests()
