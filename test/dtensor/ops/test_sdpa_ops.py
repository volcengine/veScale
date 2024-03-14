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
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from vescale.dtensor.api import distribute_tensor
from vescale.dtensor.device_mesh import DeviceMesh
from vescale.dtensor.placement_types import Replicate, Shard, Partial


class SDPATest(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @with_comms
    @parametrize("enabled_kernel", ["flash", "memory_effecient"])
    def test_basic_sdpa(self, enabled_kernel: str):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        bsz = 32
        n_heads = 8
        seq_len = 384
        kv_seq_len = 768
        hidden_dim = 64

        q = torch.rand(bsz, n_heads, seq_len, hidden_dim).cuda().bfloat16()
        k = torch.rand(bsz, n_heads, kv_seq_len, hidden_dim).cuda().bfloat16()
        v = torch.rand(bsz, n_heads, kv_seq_len, hidden_dim).cuda().bfloat16()

        base_q = deepcopy(q)
        base_k = deepcopy(k)
        base_v = deepcopy(v)

        base_q.requires_grad_(True)
        base_k.requires_grad_(True)
        base_v.requires_grad_(True)
        with torch.backends.cuda.sdp_kernel(
            enable_flash=(enabled_kernel == "flash"),
            enable_math=False,
            enable_mem_efficient=(enabled_kernel == "memory_effecient"),
        ):
            base_out = torch.nn.functional.scaled_dot_product_attention(base_q, base_k, base_v)
            loss = base_out.mean()
            loss.backward()

        # distribute the q, k, v in Shard way along num_head dim.
        dq = distribute_tensor(q, device_mesh, [Replicate()])
        dk = distribute_tensor(k, device_mesh, [Replicate()])
        dv = distribute_tensor(v, device_mesh, [Replicate()])

        dq.requires_grad_(True)
        dk.requires_grad_(True)
        dv.requires_grad_(True)
        with torch.backends.cuda.sdp_kernel(
            enable_flash=(enabled_kernel == "flash"),
            enable_math=False,
            enable_mem_efficient=(enabled_kernel == "memory_effecient"),
        ):
            flash_out = torch.nn.functional.scaled_dot_product_attention(dq, dk, dv)
            torch.testing.assert_close(base_out, flash_out._local_tensor)
            loss = flash_out.mean()
            loss.backward()
            torch.testing.assert_close(
                base_q.grad,
                dq.grad._local_tensor,
            )
            torch.testing.assert_close(
                base_k.grad,
                dk.grad._local_tensor,
            )
            torch.testing.assert_close(
                base_v.grad,
                dv.grad._local_tensor,
            )

    @with_comms
    @parametrize("enabled_kernel", ["flash", "memory_effecient"])
    def test_shard_sdpa(self, enabled_kernel: str):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        bsz = 32
        n_heads = 8
        seq_len = 384
        kv_seq_len = 768
        hidden_dim = 64
        shard_tensor_dim = 1

        q = torch.rand(bsz, n_heads, seq_len, hidden_dim).cuda().bfloat16()
        k = torch.rand(bsz, n_heads, kv_seq_len, hidden_dim).cuda().bfloat16()
        v = torch.rand(bsz, n_heads, kv_seq_len, hidden_dim).cuda().bfloat16()

        base_q = deepcopy(q)
        base_k = deepcopy(k)
        base_v = deepcopy(v)

        base_q.requires_grad_(True)
        base_k.requires_grad_(True)
        base_v.requires_grad_(True)
        with torch.backends.cuda.sdp_kernel(
            enable_flash=(enabled_kernel == "flash"),
            enable_math=False,
            enable_mem_efficient=(enabled_kernel == "memory_effecient"),
        ):
            base_out = torch.nn.functional.scaled_dot_product_attention(base_q, base_k, base_v)
            loss = base_out.mean()
            loss.backward()

        # distribute the q, k, v in Shard way along num_head dim.
        dq = distribute_tensor(q, device_mesh, [Shard(shard_tensor_dim)])
        dk = distribute_tensor(k, device_mesh, [Shard(shard_tensor_dim)])
        dv = distribute_tensor(v, device_mesh, [Shard(shard_tensor_dim)])

        dq.requires_grad_(True)
        dk.requires_grad_(True)
        dv.requires_grad_(True)
        with torch.backends.cuda.sdp_kernel(
            enable_flash=(enabled_kernel == "flash"),
            enable_math=False,
            enable_mem_efficient=(enabled_kernel == "memory_effecient"),
        ):
            flash_dout = torch.nn.functional.scaled_dot_product_attention(dq, dk, dv)
            flash_out = flash_dout.redistribute(device_mesh, [Replicate()])
            torch.testing.assert_close(base_out, flash_out._local_tensor)
            loss = flash_out.mean()
            loss.backward()
            torch.testing.assert_close(
                base_q.grad,
                dq.grad.redistribute(device_mesh, [Replicate()], async_op=False)._local_tensor,
            )
            torch.testing.assert_close(
                base_k.grad,
                dk.grad.redistribute(device_mesh, [Replicate()], async_op=False)._local_tensor,
            )
            torch.testing.assert_close(
                base_v.grad,
                dv.grad.redistribute(device_mesh, [Replicate()], async_op=False)._local_tensor,
            )

    @with_comms
    def test_error_sdpa(self):
        device_mesh = DeviceMesh(self.device_type, torch.arange(self.world_size).reshape(-1, 2))

        bsz = 32
        n_heads = 6
        seq_len = 384
        kv_seq_len = 768
        hidden_dim = 64
        q = torch.rand(bsz, n_heads, seq_len, hidden_dim).cuda().bfloat16()
        k = torch.rand(bsz, n_heads, kv_seq_len, hidden_dim).cuda().bfloat16()
        v = torch.rand(bsz, n_heads, kv_seq_len, hidden_dim).cuda().bfloat16()

        dq = distribute_tensor(q, device_mesh, [Partial(), Replicate()])
        dk = distribute_tensor(k, device_mesh, [Replicate(), Replicate()])
        dv = distribute_tensor(v, device_mesh, [Replicate(), Replicate()])

        with self.assertRaisesRegex(RuntimeError, "q, k, v are not allowed to be partial sharded"):
            torch.ops.aten._scaled_dot_product_flash_attention(dq, dk, dv)[0]

        with self.assertRaisesRegex(
            RuntimeError,
            "veScale not support implicit resharding DTensor.",
        ):
            torch.ops.aten._scaled_dot_product_efficient_attention(dq, dk, dv, None, True)[0]

        dq = distribute_tensor(q, device_mesh, [Shard(0), Replicate()])
        dk = distribute_tensor(k, device_mesh, [Replicate(), Replicate()])
        dv = distribute_tensor(v, device_mesh, [Replicate(), Replicate()])

        with self.assertRaisesRegex(
            RuntimeError,
            r"sharding info at mesh dim \d+ is not consistent, IS/S against R/P",
        ):
            torch.ops.aten._scaled_dot_product_flash_attention(dq, dk, dv)[0]

        with self.assertRaisesRegex(
            RuntimeError,
            "veScale not support implicit resharding DTensor.",
        ):
            torch.ops.aten._scaled_dot_product_efficient_attention(dq, dk, dv, None, True)[0]

        dq = distribute_tensor(q, device_mesh, [Shard(0), Replicate()])
        dk = distribute_tensor(k, device_mesh, [Replicate(), Shard(0)])
        dv = distribute_tensor(v, device_mesh, [Shard(0), Replicate()])

        with self.assertRaisesRegex(
            RuntimeError,
            r"sharding info at mesh dim \d+ is not consistent, IS/S against R/P",
        ):
            torch.ops.aten._scaled_dot_product_flash_attention(dq, dk, dv)[0]

        with self.assertRaisesRegex(
            RuntimeError,
            "veScale not support implicit resharding DTensor.",
        ):
            torch.ops.aten._scaled_dot_product_efficient_attention(dq, dk, dv, None, True)[0]

        dq = distribute_tensor(q, device_mesh, [Shard(2), Replicate()])
        dk = distribute_tensor(k, device_mesh, [Shard(2), Replicate()])
        dv = distribute_tensor(v, device_mesh, [Shard(2), Replicate()])

        with self.assertRaisesRegex(RuntimeError, "q, k, v must be replicate at last two dims"):
            torch.ops.aten._scaled_dot_product_flash_attention(dq, dk, dv)[0]

        with self.assertRaisesRegex(
            RuntimeError,
            "veScale not support implicit resharding DTensor.",
        ):
            torch.ops.aten._scaled_dot_product_efficient_attention(dq, dk, dv, None, True)[0]


instantiate_parametrized_tests(SDPATest)

if __name__ == "__main__":
    run_tests()
