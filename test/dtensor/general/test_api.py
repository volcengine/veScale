################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

from common_dtensor import DTensorTestBase, with_comms

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import run_tests

from vescale import DeviceMesh, Replicate, Shard, distribute_tensor


class MyModel(nn.Module):
    def __init__(self, n_features, n_layers, device):
        super().__init__()
        self.seq = nn.Sequential(*[nn.Linear(n_features, n_features, device=device) for _ in range(n_layers)])

    def forward(self, x):
        return self.seq(x)

    def reset_parameters(self):
        for m in self.seq:
            m.reset_parameters()


@torch.jit.script
def my_jit_add(x):
    return torch.tan(x + 1)


class DTensorAPITest(DTensorTestBase):
    @property
    def world_size(self):
        if torch.cuda.is_available():
            return torch.cuda.device_count()
        else:
            return 8

    @with_comms
    def test_distribute_tensor(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        shard_spec = [Shard(0)]

        for requires_grad in [True, False]:
            tensor_to_shard = torch.randn(3 * self.world_size, 3, requires_grad=requires_grad)
            dist_tensor = distribute_tensor(tensor_to_shard, device_mesh, shard_spec)
            self.assertEqual(dist_tensor.size(), torch.Size([3 * self.world_size, 3]))
            local_tensor = dist_tensor.to_local()
            self.assertEqual(local_tensor.size(), torch.Size([3, 3]))
            if requires_grad:
                self.assertTrue(dist_tensor.requires_grad)
                self.assertTrue(dist_tensor.is_leaf)

    @with_comms
    def test_distribute_tensor_errors(self):
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size).reshape(self.world_size // 2, 2))
        tensor_shape = [3 * self.world_size, 3 * self.world_size]
        tensor_to_distribute = torch.randn(*tensor_shape)

        with self.assertRaisesRegex(RuntimeError, "distribute leaf tensor"):
            shard_spec = [Shard(0)]
            global_tensor = torch.randn(*tensor_shape, requires_grad=True)
            global_tensor_to_distribute = global_tensor + 2
            distribute_tensor(global_tensor_to_distribute, device_mesh, shard_spec)

        spec = [Shard(0), Shard(1)]
        dtensor = distribute_tensor(tensor_to_distribute, device_mesh, spec)

        with self.assertRaisesRegex(ValueError, "to a different device mesh"):
            new_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size))
            distribute_tensor(dtensor, new_mesh, [Shard(0)])

        with self.assertRaisesRegex(ValueError, "to a different placements"):
            new_spec = [Shard(0), Replicate()]
            distribute_tensor(dtensor, device_mesh, new_spec)

    @with_comms
    def test_distribute_tensor_uneven_sharding(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        input_sizes_and_shard_dims = [
            ((self.world_size * 3 + 1, 3, 3), 0),
            ((self.world_size * 3 + 2, 3, 3), 0),
            ((3, self.world_size * 3 + 1, 3), 1),
            ((3, self.world_size * 3 + 2, 3), 1),
            ((3, 3, self.world_size * 3 + 1), 2),
            ((3, 3, self.world_size * 3 + 2), 2),
        ]
        for input_size, shard_dim in input_sizes_and_shard_dims:
            shard_spec = [Shard(shard_dim)]
            tensor_to_shard = torch.randn(input_size)
            splitted_tensor_list = list(torch.chunk(tensor_to_shard, self.world_size, dim=shard_dim))
            dist_tensor = distribute_tensor(tensor_to_shard, device_mesh, shard_spec)
            self.assertEqual(dist_tensor.size(), torch.Size(input_size))
            local_tensor = dist_tensor.to_local()
            # chunk 8 got 7 pieces
            if self.rank < len(splitted_tensor_list):
                self.assertEqual(local_tensor, splitted_tensor_list[self.rank])

    @with_comms
    def test_jit_script_func(self):
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size).reshape(self.world_size // 2, 2))
        tensor_shape = [3 * self.world_size, 3 * self.world_size]
        tensor_to_distribute = torch.randn(*tensor_shape)
        spec = [Replicate()]
        dtensor = distribute_tensor(tensor_to_distribute, device_mesh, spec)
        out = my_jit_add(dtensor)
        self.assertEqual(out.to_local(), torch.tan(tensor_to_distribute + 1))


if __name__ == "__main__":
    run_tests()
