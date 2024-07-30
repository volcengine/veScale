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

import unittest
from four_mlp import FourMLP
import torch
import torch.optim as optim


class ZeroBubbleTest(unittest.TestCase):
    def test_split_backward(self):
        """
        Tests how to separately compute activation gradient and parameter gradient
        in zero bubble pipeline schedule.
        """
        model = FourMLP(hidden=8)

        stage0 = model.mlp1
        stage1 = model.mlp2

        input = torch.randn(8, 8, requires_grad=True)

        stage0_out = stage0(input)
        stage1_out = stage1(stage0_out)
        loss = stage1_out.sum()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # calc zbv grad
        optimizer.zero_grad()

        # calc activation grad (B)
        activation_grad_output = torch.autograd.grad(loss, stage1_out, retain_graph=True)
        activation_grad_stage1 = torch.autograd.grad(
            stage1_out,
            stage0_out,
            grad_outputs=activation_grad_output,
            retain_graph=True,
            allow_unused=True,
            materialize_grads=True,
        )
        activation_grad_stage0 = torch.autograd.grad(
            stage0_out,
            input,
            grad_outputs=activation_grad_stage1,
            retain_graph=True,
            allow_unused=True,
            materialize_grads=True,
        )

        # calc params grad (W)
        nps1 = {}
        for key, value in stage1.named_parameters():
            nps1[key] = value

        nps0 = {}
        for key, value in stage0.named_parameters():
            nps0[key] = value

        parameters_grad_stage1 = torch.autograd.grad(
            stage1_out,
            nps1.values(),
            grad_outputs=activation_grad_output,
            retain_graph=True,
            allow_unused=True,
            materialize_grads=True,
        )
        parameters_grad_stage0 = torch.autograd.grad(
            stage0_out,
            nps0.values(),
            grad_outputs=activation_grad_stage1,
            retain_graph=True,
            allow_unused=True,
            materialize_grads=True,
        )

        # calc normal grad
        optimizer.zero_grad()
        loss.backward()

        # validate grads are same
        print("fc1.weight.grad", stage1.fc1.weight.grad)
        print("fc2.weight.grad", stage1.fc2.weight.grad)

        torch.testing.assert_close(stage1.fc1.weight.grad, parameters_grad_stage1[0])
        torch.testing.assert_close(stage1.fc2.weight.grad, parameters_grad_stage1[1])
        torch.testing.assert_close(stage0.fc1.weight.grad, parameters_grad_stage0[0])
        torch.testing.assert_close(stage0.fc2.weight.grad, parameters_grad_stage0[1])


if __name__ == "__main__":
    unittest.main()
