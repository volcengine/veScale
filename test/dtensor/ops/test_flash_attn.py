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

from common_dtensor import DTensorTestBase, with_comms
from flash_attn import flash_attn_func
import torch
from torch.testing._internal.common_utils import (
    run_tests,
)
from vescale.dtensor.placement_types import Shard
from vescale.dtensor.device_mesh import DeviceMesh
from vescale.dtensor.api import distribute_tensor

HIDDEN_DIM = 4
BSZ = 3


class RepeatTest(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    @with_comms
    def test_fa_v2(self):
        device_mesh = DeviceMesh(self.device_type, [0, 1])
        bsz = 3
        num_head = 32
        seqlen = 256
        head_dim = 256
        # q = torch.rand(bsz, num_head, seqlen, head_dim, dtype=torch.float16)
        # k = torch.rand(bsz, num_head, seqlen, head_dim, dtype=torch.float16)
        # v = torch.rand(bsz, num_head, seqlen, head_dim, dtype=torch.float16)
        q = torch.tensor(float("nan"), dtype=torch.float16).broadcast_to((bsz, num_head, seqlen, head_dim))
        k = torch.tensor(float("nan"), dtype=torch.float16).broadcast_to((bsz, num_head, seqlen, head_dim))
        v = torch.tensor(float("nan"), dtype=torch.float16).broadcast_to((bsz, num_head, seqlen, head_dim))
        dq = distribute_tensor(q, device_mesh, [Shard(1)])
        dv = distribute_tensor(v, device_mesh, [Shard(1)])
        dk = distribute_tensor(k, device_mesh, [Shard(1)])
        print(dq.stride())
        out = flash_attn_func(dq, dk, dv)
        print(out)
        # flash_attn_func(dq.to_local(), dk.to_local(), dv.to_local())
        # dq = distribute_tensor(q, device_mesh, [Replicate()])
        # dv = distribute_tensor(v, device_mesh, [Replicate()])
        # dk = distribute_tensor(k, device_mesh, [Replicate()])
        # print(dk.stride(1))
        # flash_attn_func(dq, dk, dv)


if __name__ == "__main__":
    run_tests()
