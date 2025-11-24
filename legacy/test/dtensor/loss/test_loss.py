################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

import itertools
from common_dtensor import (
    DTensorTestBase,
    with_comms,
)

import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import run_tests
from vescale import distribute_tensor
from vescale.dtensor.placement_types import Shard
from vescale.dtensor.loss import loss_parallel


class DistLossParallelTest(DTensorTestBase):
    @with_comms
    def test_loss_parallel(self):
        device_mesh = self.build_device_mesh()

        channel_size, channel_dim = 16, 1
        test_setup = [
            (2, (8, channel_size), (8,)),  # calling aten.nll_loss_forward
            (3, (8, channel_size, 12), (8, 12)),  # calling aten.nll_loss2d_forward
        ]
        weight = torch.rand(channel_size, device=self.device_type)
        for input_ndim, input_size, target_size in test_setup:
            x = torch.rand(*input_size, device=self.device_type, requires_grad=True)
            target = torch.randint(channel_size, target_size, device=self.device_type)

            shard_dims = list(range(input_ndim))
            reductions = ["none", "mean", "sum"]
            for shard_dim, reduction in itertools.product(shard_dims, reductions):
                dist_x = distribute_tensor(x, device_mesh, [Shard(shard_dim)])
                y = F.cross_entropy(x, target, weight, reduction=reduction)
                with loss_parallel():
                    if shard_dim == channel_dim:
                        dist_y = F.cross_entropy(dist_x, target, weight, reduction=reduction)

                        self.assertTrue(dist_y.placements[0].is_replicate())
                        self.assertEqual(dist_y.to_local(), y)

                        if reduction == "none":
                            y.sum().backward()
                            dist_y.sum().backward()
                        else:
                            y.backward()
                            dist_y.backward()
                        self.assertTrue(dist_x.grad.placements[0].is_shard(shard_dim))
                        self.assertEqual(dist_x.grad.full_tensor(), x.grad)
                        x.grad.zero_()
                    else:
                        with self.assertRaisesRegex(
                            ValueError,
                            "loss_parallel",
                        ):
                            dist_y = F.cross_entropy(dist_x, target, reduction=reduction)


if __name__ == "__main__":
    run_tests()
