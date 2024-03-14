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

from common_dtensor import DTensorTestBase, with_comms, skip_unless_torch_gpu

import torch
from torch.cuda import empty_cache, memory_reserved, memory_stats, memory_summary, reset_peak_memory_stats, synchronize
from torch.testing._internal.common_utils import run_tests
from vescale import DeviceMesh, DTensor, distribute_tensor
from vescale.dtensor.placement_types import Partial, Replicate, Shard

_512 = 512
_1K = 1 * 1024
_10K = 10 * 1024
_100K = 100 * 1024
_1M = 1 * 1024 * 1024
_10M = 10 * 1024 * 1024
_100M = 100 * 1024 * 1024

_EXTRA_PEAK = 2048


class DTensorTestCuda(DTensorTestBase):
    @property
    def _device(self):
        return f"cuda:{self.rank}"

    @with_comms
    @skip_unless_torch_gpu
    def test_assert_using_cuda(self):
        self.assertEqual(self.device_type, "cuda")
        self.assertEqual(torch.cuda.get_allocator_backend(), "native")
        self.assertTrue(self.world_size > 1)

    @with_comms
    @skip_unless_torch_gpu
    def test_internal_constructor(self):
        local_shape = (_1K, _1K)
        for placements, shape_scale in zip(
            [[Shard(0)], [Shard(1)], [Replicate()], [Partial()]],
            [(self.world_size, 1), (1, self.world_size), (1, 1), (1, 1)],
        ):
            empty_cache()
            reset_peak_memory_stats(self._device)
            synchronize(self._device)
            self.assertEqual(memory_reserved(self._device), 0)

            local_tensor = torch.randn(local_shape, dtype=torch.float32, device=self._device, requires_grad=True)
            device_mesh = DeviceMesh("cuda", list(range(self.world_size)))

            synchronize(self._device)
            m1 = memory_stats(self._device)

            dist_tensor_shape = torch.Size([shape * scale for shape, scale in zip(local_shape, shape_scale)])
            dist_tensor = DTensor(
                local_tensor,
                device_mesh,
                placements,
                shape=dist_tensor_shape,
                dtype=local_tensor.dtype,
                requires_grad=True,
                stride=local_tensor.stride(),
            )

            synchronize(self._device)
            m2 = memory_stats(self._device)

            self.assertEqual(m1, m2)

            del local_tensor, device_mesh, dist_tensor_shape, dist_tensor

    @with_comms
    @skip_unless_torch_gpu
    def test_from_local__forward(self):
        local_shape = (_1K, _1K)
        for run_check in (False, True):
            for placements in [[Shard(0)], [Shard(1)], [Partial()], [Replicate()]]:
                empty_cache()
                reset_peak_memory_stats(self._device)
                synchronize(self._device)
                self.assertEqual(memory_reserved(self._device), 0)

                local_tensor = torch.randn(local_shape, dtype=torch.float32, device=self._device, requires_grad=True)
                self.assertTrue(local_tensor.is_contiguous())
                device_mesh = DeviceMesh("cuda", list(range(self.world_size)))

                synchronize(self._device)
                m1 = memory_stats(self._device)

                dist_tensor = DTensor.from_local(local_tensor, device_mesh, placements, run_check=run_check)

                synchronize(self._device)
                m2 = memory_stats(self._device)

                for (k1, v1), (k2, v2) in zip(m1.items(), m2.items()):
                    if "large_pool" in k1:
                        self.assertEqual(k1, k2)
                        self.assertEqual(v1, v2, msg=f"{k1}: {v1} vs {v2}")

                del local_tensor, device_mesh, dist_tensor

    @with_comms
    @skip_unless_torch_gpu
    def test_from_local__backward(self):
        local_shape = (_1K, _1K)
        for run_check in (False, True):
            for placements in [[Shard(0)], [Shard(1)], [Replicate()], [Partial()]]:
                empty_cache()
                reset_peak_memory_stats(self._device)
                synchronize(self._device)
                self.assertEqual(memory_reserved(self._device), 0)

                device_mesh = DeviceMesh("cuda", list(range(self.world_size)))

                # leaf local tensor
                local_tensor = torch.randn(local_shape, dtype=torch.float32, device=self._device, requires_grad=True)
                self.assertTrue(local_tensor.is_contiguous())

                # forward op
                non_leaf_dt = DTensor.from_local(local_tensor, device_mesh, placements, run_check=run_check)

                # create dummy gradient
                local_grad = torch.ones(local_shape, dtype=torch.float32, device=self._device, requires_grad=False)
                grad_dt = DTensor.from_local(local_grad, device_mesh, placements, run_check=False)

                # backward op
                synchronize(self._device)
                m3 = memory_stats(self._device)

                non_leaf_dt.backward(grad_dt)

                synchronize(self._device)
                m4 = memory_stats(self._device)

                # check
                self.assertTrue(local_tensor.is_leaf)
                self.assertIsNotNone(local_tensor.grad)
                expected_grad = torch.ones(local_shape, dtype=torch.float32, device=self._device, requires_grad=False)
                self.assertEqual(local_tensor.grad, expected_grad)
                self.assertEqual(m3, m4)

                del device_mesh, local_tensor, non_leaf_dt, local_grad, grad_dt, expected_grad

    def _main_distribute_tensor(self, global_shape, placements):
        empty_cache()
        reset_peak_memory_stats(self._device)
        synchronize(self._device)
        self.assertEqual(memory_reserved(self._device), 0)

        global_tensor = torch.randn(global_shape, dtype=torch.float32, device=self._device, requires_grad=True)
        self.assertTrue(global_tensor.is_contiguous())
        device_mesh = DeviceMesh("cuda", list(range(self.world_size)))

        synchronize(self._device)
        m1 = memory_stats(self._device)
        if self.rank == 0:
            print(f"rank{self.rank} @m1: {memory_summary(self._device)}")

        dist_tensor = distribute_tensor(global_tensor, device_mesh, placements)

        synchronize(self._device)
        m2 = memory_stats(self._device)
        if self.rank == 0:
            print(f"rank{self.rank} @m2: {memory_summary(self._device)}")

        return m1, m2, global_tensor.numel() * 4, dist_tensor._local_tensor.numel() * 4

    @with_comms
    @skip_unless_torch_gpu
    def test_distribute_tensor_rowwise(self):
        global_shape = (self.world_size * _1K, self.world_size * _1K)
        placements = [Shard(0)]

        m1, m2, _, local_tensor_bytes = self._main_distribute_tensor(global_shape, placements)

        self.assertEqual(
            m2["allocated_bytes.large_pool.allocated"] - m1["allocated_bytes.large_pool.allocated"],
            local_tensor_bytes,
            msg="only increase should be local tensor",
        )
        self.assertEqual(
            m2["allocation.large_pool.allocated"] - m1["allocation.large_pool.allocated"],
            1,
            msg="only increase should be local tensor",
        )

    @with_comms
    @skip_unless_torch_gpu
    def test_distribute_tensor_colwise(self):
        global_shape = (self.world_size * _1K, self.world_size * _1K)
        placements = [Shard(1)]

        m1, m2, global_tensor_bytes, local_tensor_bytes = self._main_distribute_tensor(global_shape, placements)

        self.assertEqual(
            m2["allocated_bytes.large_pool.allocated"] - m1["allocated_bytes.large_pool.allocated"],
            global_tensor_bytes + local_tensor_bytes,
            msg="only increase should be all contiguous tensor + one local tensor",
        )
        self.assertEqual(
            m2["allocation.large_pool.allocated"] - m1["allocation.large_pool.allocated"],
            self.world_size + 1,
            msg="only increase should be all contiguous tensor + one local tensor",
        )
        self.assertEqual(
            m2["allocated_bytes.large_pool.freed"] - m1["allocated_bytes.large_pool.freed"],
            global_tensor_bytes,
            msg="only decrease should be all contiguous tensors",
        )
        self.assertEqual(
            m2["allocation.large_pool.freed"] - m1["allocation.large_pool.freed"],
            self.world_size,
            msg="only decrease should be all contiguous tensors",
        )

    @with_comms
    @skip_unless_torch_gpu
    def test_distribute_tensor_replicate(self):
        global_shape = (self.world_size * _1K, self.world_size * _1K)
        placements = [Replicate()]

        m1, m2, _, _ = self._main_distribute_tensor(global_shape, placements)

        for (k1, v1), (k2, v2) in zip(m1.items(), m2.items()):
            if "large_pool" in k1:
                self.assertEqual(k1, k2)
                self.assertEqual(v1, v2, msg=f"{k1}: {v1} vs {v2}")

    @with_comms
    @skip_unless_torch_gpu
    def test_distribute_tensor_partial(self):
        global_shape = (self.world_size * _1K, self.world_size * _1K)
        placements = [Partial()]

        m1, m2, _, _ = self._main_distribute_tensor(global_shape, placements)

        for (k1, v1), (k2, v2) in zip(m1.items(), m2.items()):
            if "large_pool" in k1:
                self.assertEqual(k1, k2)
                self.assertEqual(v1, v2, msg=f"{k1}: {v1} vs {v2}")

    @with_comms
    @skip_unless_torch_gpu
    def test_to_local__forward(self):
        local_shape = (_1K, _1K)
        for placements, shape_scale in zip(
            [[Shard(0)], [Shard(1)], [Replicate()], [Partial()]],
            [(self.world_size, 1), (1, self.world_size), (1, 1), (1, 1)],
        ):
            empty_cache()
            reset_peak_memory_stats(self._device)
            synchronize(self._device)
            self.assertEqual(memory_reserved(self._device), 0)

            local_tensor = torch.randn(local_shape, dtype=torch.float32, device=self._device, requires_grad=True)
            device_mesh = DeviceMesh("cuda", list(range(self.world_size)))

            dist_tensor_shape = torch.Size([shape * scale for shape, scale in zip(local_shape, shape_scale)])
            dist_tensor = DTensor(
                local_tensor,
                device_mesh,
                placements,
                shape=dist_tensor_shape,
                dtype=local_tensor.dtype,
                requires_grad=True,
                stride=local_tensor.stride(),
            )

            synchronize(self._device)
            m1 = memory_stats(self._device)

            dist_tensor_local = dist_tensor.to_local()

            synchronize(self._device)
            m2 = memory_stats(self._device)

            self.assertEqual(m1, m2)

            del local_tensor, device_mesh, dist_tensor_shape, dist_tensor, dist_tensor_local

    @with_comms
    @skip_unless_torch_gpu
    def test_to_local__backward(self):
        local_shape = (_1K, _1K)
        for placements, shape_scale in zip(
            [[Shard(0)], [Shard(1)], [Replicate()]], [(self.world_size, 1), (1, self.world_size), (1, 1)]
        ):
            empty_cache()
            reset_peak_memory_stats(self._device)
            synchronize(self._device)
            self.assertEqual(memory_reserved(self._device), 0)

            device_mesh = DeviceMesh("cuda", list(range(self.world_size)))

            # leaf dist tensor
            local_tensor = torch.randn(local_shape, dtype=torch.float32, device=self._device, requires_grad=True)
            dist_tensor_shape = torch.Size([shape * scale for shape, scale in zip(local_shape, shape_scale)])
            dist_tensor = DTensor(
                local_tensor,
                device_mesh,
                placements,
                shape=dist_tensor_shape,
                dtype=local_tensor.dtype,
                requires_grad=True,
                stride=local_tensor.stride(),
            )

            # to_local's backward only works for non-leaf DTensor
            non_leaf_dt = dist_tensor * 3
            non_leaf_local = non_leaf_dt.to_local()  # forward op

            local_tensor2 = torch.randn(local_shape, dtype=torch.float32, device=self._device, requires_grad=False)
            res = non_leaf_local + local_tensor2
            loss = res.sum()

            # test to_local's backward
            synchronize(self._device)
            m3 = memory_stats(self._device)

            loss.backward()

            synchronize(self._device)
            m4 = memory_stats(self._device)

            self.assertTrue(dist_tensor.is_leaf)
            self.assertIsNotNone(dist_tensor.grad)
            local_shard_bytes = local_tensor.numel() * 4
            self.assertEqual(dist_tensor.grad._local_tensor.numel() * 4, local_shard_bytes)
            self.assertEqual(
                (m4["allocated_bytes.large_pool.allocated"] - m3["allocated_bytes.large_pool.allocated"])
                / local_shard_bytes,
                1.0,
                msg="only increase leaf's .grad",
            )
            self.assertEqual(
                m4["allocation.large_pool.allocated"] - m3["allocation.large_pool.allocated"],
                1,
                msg="only increase leaf's .grad",
            )

            del (
                local_tensor,
                device_mesh,
                dist_tensor_shape,
                dist_tensor,
                non_leaf_dt,
                non_leaf_local,
                local_tensor2,
                res,
                loss,
            )

    @with_comms
    @skip_unless_torch_gpu
    def test_parameterization(self):
        shape = (_1K, _1K)
        for create_fn in (distribute_tensor, DTensor.from_local):
            for placements in [[Shard(0)], [Shard(1)], [Replicate()], [Partial()]]:
                empty_cache()
                reset_peak_memory_stats(self._device)
                synchronize(self._device)
                self.assertEqual(memory_reserved(self._device), 0)

                tensor = torch.randn(shape, dtype=torch.float32, device=self._device, requires_grad=True)
                device_mesh = DeviceMesh("cuda", list(range(self.world_size)))
                dist_tensor = create_fn(tensor, device_mesh, placements)

                m1 = memory_stats(self._device)
                dt = dist_tensor.detach().requires_grad_(True)
                self.assertTrue(type(dt) is type(dist_tensor))
                dt._is_param = True
                self.assertTrue(dt._is_param is True)
                m2 = memory_stats(self._device)
                self.assertEqual(m1, m2)

                m3 = memory_stats(self._device)
                dt_param = torch.nn.Parameter(dist_tensor)
                m4 = memory_stats(self._device)
                self.assertEqual(m3, m4)

                del tensor, device_mesh, dist_tensor, dt, dt_param

    # @with_comms
    # @skip_unless_torch_gpu
    # def test_parameter_replacement(self):
    #     def replace(fc, create_fn, device_mesh, placements):
    #         for name, param in fc.named_parameters():
    #             dist_param = torch.nn.Parameter(create_fn(param, device_mesh, placements))
    #             fc.register_parameter(name, dist_param)
    #         # exit this function scope will automatically "gc" old param

    #     for create_fn in (distribute_tensor, DTensor.from_local):
    #         for placements in [[Shard(0)], [Shard(1)], [Replicate()], [Partial()]]:
    #             empty_cache()
    #             reset_peak_memory_stats(self._device)
    #             synchronize(self._device)
    #             self.assertEqual(memory_reserved(self._device), 0)

    #             fc = torch.nn.Linear(_10K, _10K, bias=False, device=self._device, dtype=torch.float32)
    #             device_mesh = DeviceMesh("cuda", list(range(self.world_size)))

    #             m1 = memory_stats(self._device)
    #             replace(fc, create_fn, device_mesh, placements)
    #             m2 = memory_stats(self._device)

    #             if (create_fn is DTensor.from_local) or \
    #                 (create_fn is distribute_tensor and not placements[0].is_shard()):
    #                 self.assertEqual(m1, m2)
    #             elif create_fn is distribute_tensor and placements[0].dim == 0:
    #                 self.assertEqual(
    #                     m2["allocated_bytes.large_pool.allocated"] - m1["allocated_bytes.large_pool.allocated"],
    #                     _10K * _10K * 1,
    #                     msg="only increase should be new parameter")
    #                 self.assertEqual(
    #                     m2["allocation.large_pool.allocated"] - m1["allocation.large_pool.allocated"],
    #                     1,
    #                     msg="only increase should be new parameter")
    #                 self.assertEqual(
    #                     m2["allocated_bytes.large_pool.freed"] - m1["allocated_bytes.large_pool.freed"],
    #                     _10K * _10K * 4,
    #                     msg="only decrease should be old parameter")
    #                 self.assertEqual(
    #                     m2["allocation.large_pool.freed"] - m1["allocation.large_pool.freed"],
    #                     1,
    #                     msg="only decrease should be old parameter")
    #                 self.assertEqual(
    #                     m2["allocated_bytes.large_pool.current"],
    #                     _10K * _10K * 1)
    #                 self.assertEqual(
    #                     m2["allocation.large_pool.current"],
    #                     1)
    #             elif create_fn is distribute_tensor and placements[0].dim == 1:
    #                 self.assertEqual(
    #                     m2["allocated_bytes.large_pool.allocated"] - m1["allocated_bytes.large_pool.allocated"],
    #                     _10K * _10K * (4 + 1))
    #                 self.assertEqual(
    #                     m2["allocation.large_pool.allocated"] - m1["allocation.large_pool.allocated"],
    #                     4 + 1)
    #                 self.assertEqual(
    #                     m2["allocated_bytes.large_pool.freed"] - m1["allocated_bytes.large_pool.freed"],
    #                     _10K * _10K * (4 + 4))
    #                 self.assertEqual(
    #                     m2["allocation.large_pool.freed"] - m1["allocation.large_pool.freed"],
    #                     1 + 4)
    #                 self.assertEqual(
    #                     m2["allocated_bytes.large_pool.current"],
    #                     _10K * _10K * 1)
    #                 self.assertEqual(
    #                     m2["allocation.large_pool.current"],
    #                     1)
    #             else:
    #                 raise AssertionError("should not be here")

    #             del fc, device_mesh

    @with_comms
    @skip_unless_torch_gpu
    def test_create_from_gpu(self):
        local_shape = (_1K, _1K)
        for placements in [[Shard(0)], [Shard(1)], [Partial()], [Replicate()]]:
            empty_cache()
            reset_peak_memory_stats(self._device)
            synchronize(self._device)
            self.assertEqual(memory_reserved(self._device), 0)

            # create GPU dtensor
            local_tensor = torch.randn(local_shape, dtype=torch.float32, device=self._device, requires_grad=True)
            device_mesh = DeviceMesh("cuda", list(range(self.world_size)))
            gpu_dtensor = DTensor.from_local(local_tensor, device_mesh, placements)

            # create CPU DTensor
            synchronize(self._device)
            m1 = memory_stats(self._device)

            gpu_local = gpu_dtensor.to_local()
            cpu_mesh = DeviceMesh("cpu", list(range(self.world_size)), _validate_mesh=False, _init_process_groups=False)
            cpu_dtensor = DTensor.from_local(
                gpu_local, cpu_mesh, placements, run_check=False, shape=gpu_dtensor.shape, stride=gpu_dtensor.stride()
            )

            synchronize(self._device)
            m2 = memory_stats(self._device)

            # check correctness
            self.assertTrue(cpu_dtensor._local_tensor.device.type == "cpu")
            self.assertTrue(cpu_dtensor._spec.mesh.device_type == "cpu")
            self.assertTrue(cpu_dtensor.device.type == "cpu")

            # check memory
            self.assertEqual(m1, m2)

            del local_tensor, device_mesh, gpu_dtensor, gpu_local, cpu_mesh, cpu_dtensor

    @with_comms
    @skip_unless_torch_gpu
    def test_create_from_cpu(self):
        local_shape = (_1K, _1K)
        for placements in [[Shard(0)], [Shard(1)], [Partial()], [Replicate()]]:
            empty_cache()
            reset_peak_memory_stats(self._device)
            synchronize(self._device)
            self.assertEqual(memory_reserved(self._device), 0)

            # create CPU local Tensor
            cpu_local = torch.randn(local_shape, dtype=torch.float32, device="cpu", requires_grad=True)

            # create GPU device mesh
            device_mesh = DeviceMesh("cuda", list(range(self.world_size)))

            # create GPU DTensor
            synchronize(self._device)
            m1 = memory_stats(self._device)

            gpu_dtensor = DTensor.from_local(cpu_local, device_mesh, placements)

            synchronize(self._device)
            m2 = memory_stats(self._device)

            # check correctness
            self.assertTrue(gpu_dtensor._local_tensor.device.type == "cuda")
            self.assertTrue(gpu_dtensor._spec.mesh.device_type == "cuda")
            self.assertTrue(gpu_dtensor.device.type == "cuda")

            # check memory
            self.assertEqual(
                m2["allocated_bytes.large_pool.allocated"] - m1["allocated_bytes.large_pool.allocated"],
                _1K * _1K * 4,
                msg="only increase should be gpu shard",
            )
            self.assertEqual(
                m2["allocation.large_pool.allocated"] - m1["allocation.large_pool.allocated"],
                1,
                msg="only increase should be gpu shard",
            )

            del cpu_local, device_mesh, gpu_dtensor

    @with_comms
    @skip_unless_torch_gpu
    def test_to_cpu__forward(self):
        local_shape = (_1K, _1K)
        for placements in [[Shard(0)], [Shard(1)], [Partial()], [Replicate()]]:
            empty_cache()
            reset_peak_memory_stats(self._device)
            synchronize(self._device)
            self.assertEqual(memory_reserved(self._device), 0)

            # create GPU dtensor
            local_tensor = torch.randn(local_shape, dtype=torch.float32, device=self._device, requires_grad=True)
            device_mesh = DeviceMesh("cuda", list(range(self.world_size)))
            gpu_dtensor = DTensor.from_local(local_tensor, device_mesh, placements)

            # to CPU DTensor
            synchronize(self._device)
            m1 = memory_stats(self._device)

            cpu_dtensor = gpu_dtensor.to("cpu")

            synchronize(self._device)
            m2 = memory_stats(self._device)

            # check correctness
            self.assertTrue(cpu_dtensor._local_tensor.device.type == "cpu")
            self.assertTrue(cpu_dtensor.device.type == "cpu")

            # check memory
            self.assertEqual(m1, m2)

            del local_tensor, device_mesh, gpu_dtensor, cpu_dtensor

    @with_comms
    @skip_unless_torch_gpu
    def test_to_gpu__forward(self):
        local_shape = (_1K, _1K)
        for placements, shape_scale in zip(
            [[Shard(0)], [Shard(1)], [Replicate()], [Partial()]],
            [(self.world_size, 1), (1, self.world_size), (1, 1), (1, 1)],
        ):
            empty_cache()
            reset_peak_memory_stats(self._device)
            synchronize(self._device)
            self.assertEqual(memory_reserved(self._device), 0)

            # create CPU DTensor
            cpu_local = torch.randn(local_shape, dtype=torch.float32, device="cpu", requires_grad=True)
            cpu_mesh = DeviceMesh("cpu", list(range(self.world_size)), _validate_mesh=False, _init_process_groups=False)
            global_shape = torch.Size([shape * scale for shape, scale in zip(local_shape, shape_scale)])
            global_stride = torch.empty(global_shape, device="meta").stride()
            cpu_dtensor = DTensor.from_local(
                cpu_local, cpu_mesh, placements, run_check=False, shape=global_shape, stride=global_stride
            )

            # to GPU DTensor
            synchronize(self._device)
            m1 = memory_stats(self._device)

            gpu_dtensor = cpu_dtensor.to("cuda")

            synchronize(self._device)
            m2 = memory_stats(self._device)

            # check correctness
            self.assertTrue(gpu_dtensor._local_tensor.device.type == "cuda")
            self.assertTrue(gpu_dtensor.device.type == "cuda")

            # check memory
            self.assertEqual(
                m2["allocated_bytes.large_pool.allocated"] - m1["allocated_bytes.large_pool.allocated"],
                _1K * _1K * 4,
                msg="only increase should be gpu shard",
            )
            self.assertEqual(
                m2["allocation.large_pool.allocated"] - m1["allocation.large_pool.allocated"],
                1,
                msg="only increase should be gpu shard",
            )

            del cpu_local, cpu_mesh, cpu_dtensor, gpu_dtensor


if __name__ == "__main__":
    run_tests()
