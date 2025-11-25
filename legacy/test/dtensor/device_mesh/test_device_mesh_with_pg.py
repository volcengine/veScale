################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

from typing import Optional
from common_dtensor import DTensorTestBase, with_comms

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch.distributed.distributed_c10d import ProcessGroup, new_group
from torch.testing._internal.common_utils import run_tests
from vescale.dtensor._collective_utils import mesh_all_to_all, mesh_broadcast, mesh_scatter
from vescale.dtensor.device_mesh import DeviceMesh
from vescale.dtensor.placement_types import Shard


class DeviceMeshTest(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @with_comms
    def test_construction_behaviors(self):
        # None, None
        with self.assertRaisesRegex(ValueError, "must be provided!"):
            mesh = DeviceMesh(self.device_type)

        # mesh only
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        self.assertTrue(mesh.get_dim_groups())

        # pg only, as a new world
        pgroup: Optional[ProcessGroup] = new_group(ranks=list(range(self.world_size)))
        dist.barrier()
        mesh = DeviceMesh(self.device_type, pg=pgroup)
        self.assertTrue(mesh.get_dim_groups())
        dist.destroy_process_group(pgroup)

        # pg only, as a sub world
        pgroup_ranks = [1, 3]
        pgroup: Optional[ProcessGroup] = new_group(ranks=pgroup_ranks)
        dist.barrier()
        if self.rank in pgroup_ranks:
            mesh = DeviceMesh(self.device_type, pg=pgroup)
            self.assertTrue(mesh.get_dim_groups())
            dist.destroy_process_group(pgroup)

        # both mesh and pg, inequal
        pgroup: Optional[ProcessGroup] = new_group(ranks=list(range(self.world_size)))
        dist.barrier()
        with self.assertRaisesRegex(ValueError, "must have the same content"):
            mesh = DeviceMesh(self.device_type, list(range(self.world_size // 2)), pg=pgroup)
        dist.destroy_process_group(pgroup)

        # both mesh and pg, equal
        pgroup: Optional[ProcessGroup] = new_group(ranks=list(range(self.world_size)))
        dist.barrier()
        mesh = DeviceMesh(self.device_type, list(range(self.world_size)), pg=pgroup)
        self.assertTrue(mesh.get_dim_groups())
        dist.destroy_process_group(pgroup)


class DeviceMeshCollectiveTest(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @with_comms
    def test_broadcast_1d(self):
        mesh = DeviceMesh(self.device_type, pg=new_group(ranks=list(range(self.world_size))))
        local_tensor = torch.ones(3, 3, device=self.device_type) * self.rank
        local_tensor = mesh_broadcast(local_tensor, mesh, mesh_dim=0)
        self.assertEqual(local_tensor, torch.zeros(3, 3))

    @with_comms
    def test_scatter_1d(self):
        mesh = DeviceMesh(self.device_type, pg=new_group(ranks=list(range(self.world_size))))
        scatter_tensor_shape = [3, 3, 3]
        for scatter_dim in range(len(scatter_tensor_shape)):
            shard_placement = Shard(scatter_dim)
            scatter_tensor_shape[scatter_dim] *= self.world_size
            # make the random seed same across rank
            torch.manual_seed(0)
            global_tensor = torch.randn(scatter_tensor_shape, device=self.device_type)
            splitted_list, _ = shard_placement._split_tensor(
                global_tensor, mesh.size(), with_padding=True, contiguous=True
            )
            recv_tensor = torch.empty_like(splitted_list[mesh.get_rank()])
            # scatter on dim > 0 would generate non-contiguous tensor, verify that works
            mesh_scatter(recv_tensor, splitted_list, mesh, mesh_dim=0)
            self.assertEqual(recv_tensor, splitted_list[mesh.get_rank()])

    @with_comms
    def test_scatter_uneven(self):
        device_mesh = DeviceMesh(self.device_type, pg=new_group(ranks=list(range(self.world_size))))
        my_rank = device_mesh.get_rank()
        tensor_to_split = torch.randn(device_mesh.size() + 3, device_mesh.size() + 1, device=self.device_type)

        for shard_dim in range(tensor_to_split.ndim):
            shard_placement = Shard(shard_dim)

            tensor_to_scatter = tensor_to_split.clone()
            tensor_splitted_list = list(torch.chunk(tensor_to_split, self.world_size, dim=shard_dim))
            for _ in range(self.world_size - len(tensor_splitted_list)):
                tensor_splitted_list.append(torch.tensor([], device=self.device_type))

            padded_tensor_list, pad_sizes = shard_placement._split_tensor(
                tensor_to_scatter,
                device_mesh.size(),
                with_padding=True,
                contiguous=True,
            )

            scattered_tensor = torch.empty_like(padded_tensor_list[my_rank])
            mesh_scatter(scattered_tensor, padded_tensor_list, device_mesh, mesh_dim=0)

            if pad_sizes[my_rank] != 0:
                scattered_tensor = shard_placement._unpad_tensor(scattered_tensor, pad_sizes[my_rank])

            if scattered_tensor.numel() == 0:
                # We need to check numel() instead of size if a tensor is ([]) after unpadding,
                # since the size could be ([0, 8]) after unpadding.
                self.assertEqual(scattered_tensor.numel(), tensor_splitted_list[my_rank].numel())
            else:
                self.assertEqual(scattered_tensor.size(), tensor_splitted_list[my_rank].size())
                self.assertEqual(scattered_tensor, tensor_splitted_list[my_rank])

    @with_comms
    def test_all_gather_uneven(self):
        device_mesh = DeviceMesh(self.device_type, pg=new_group(ranks=list(range(self.world_size))))
        my_rank = device_mesh.get_rank()
        tensor_to_split = torch.ones(
            device_mesh.size() + 3,
            device_mesh.size() + 1,
            device=self.device_type,
        )

        for shard_dim in range(tensor_to_split.ndim):
            shard_placement = Shard(shard_dim)
            tensor_padded_list, pad_sizes = shard_placement._split_tensor(
                tensor_to_split,
                device_mesh.size(),
                with_padding=True,
                contiguous=True,
            )
            local_tensor = tensor_padded_list[my_rank]
            big_tensor = funcol.all_gather_tensor(
                local_tensor, gather_dim=shard_dim, group=device_mesh._dim_group_infos[0][1]
            )
            big_tensor_chunks = list(torch.chunk(big_tensor, device_mesh.size(), dim=shard_dim))
            unpadded_list = [
                shard_placement._unpad_tensor(big_tensor_chunks[i], pad_sizes[i])
                if pad_sizes[i] > 0
                else big_tensor_chunks[i]
                for i, big_tensor in enumerate(big_tensor_chunks)
            ]
            all_gathered_tensor = torch.cat(unpadded_list, dim=shard_dim)

            self.assertEqual(all_gathered_tensor.size(), tensor_to_split.size())
            self.assertEqual(all_gathered_tensor, tensor_to_split)

    @with_comms
    def test_reduce_scatter_uneven(self):
        device_mesh = DeviceMesh(self.device_type, pg=new_group(ranks=list(range(self.world_size))))
        my_rank = device_mesh.get_rank()
        tensor_to_split = (
            torch.ones(
                device_mesh.size() + 3,
                device_mesh.size() + 1,
                device=self.device_type,
            )
            * self.rank
        )

        for shard_dim in range(tensor_to_split.ndim):
            shard_placement = Shard(shard_dim)
            tensor_to_scatter = tensor_to_split.clone()

            tensor_splitted_list = list(torch.chunk(tensor_to_split, self.world_size, dim=shard_dim))
            for _ in range(self.world_size - len(tensor_splitted_list)):
                tensor_splitted_list.append(torch.tensor([], device=self.device_type))

            padded_tensor_list, pad_sizes = shard_placement._split_tensor(
                tensor_to_scatter,
                device_mesh.size(),
                with_padding=True,
                contiguous=True,
            )

            tensor_to_reduce = torch.cat(padded_tensor_list, shard_dim)

            res_num = ((0 + self.world_size - 1) * self.world_size) / 2
            scattered_tensor = funcol.reduce_scatter_tensor(
                tensor_to_reduce, reduceOp="sum", scatter_dim=shard_dim, group=device_mesh._dim_group_infos[0][1]
            )

            # unpad scattered_tensor
            if pad_sizes[my_rank] > 0:
                scattered_tensor = shard_placement._unpad_tensor(scattered_tensor, pad_sizes[my_rank])

            if scattered_tensor.numel() == 0:
                # We need to check numel() instead of size if a tensor is ([]) after unpadding,
                # since the size could be ([0, 8]) after unpadding.
                self.assertEqual(scattered_tensor.numel(), tensor_splitted_list[my_rank].numel())
            else:
                self.assertEqual(scattered_tensor.size(), tensor_splitted_list[my_rank].size())
                self.assertEqual(
                    scattered_tensor,
                    torch.ones_like(tensor_splitted_list[my_rank]) * res_num,
                )

    @with_comms
    def test_all_to_all_1d(self):
        # transpose on a 2D tensor distributed over N nodes:
        mesh = DeviceMesh(self.device_type, pg=new_group(ranks=list(range(self.world_size))))
        tensor_shape = [3, 3]
        input_tensor_list = [
            torch.ones(*tensor_shape, device=self.device_type) * (rank + self.rank * self.world_size)
            for rank in range(self.world_size)
        ]
        expected_tensor_list = [
            torch.ones(tensor_shape, device=self.device_type) * (self.rank + rank * self.world_size)  # i.e. transpose
            for rank in range(self.world_size)
        ]
        for scatter_dim in range(len(tensor_shape)):
            output_tensor_list = [torch.empty_like(input_tensor_list[idx]) for idx in range(len(input_tensor_list))]
            # scatter on dim > 0 would generate non-contiguous tensor, verify that works
            mesh_all_to_all(output_tensor_list, input_tensor_list, mesh, mesh_dim=0)
            output_tensor = torch.cat(output_tensor_list, dim=scatter_dim)
            expected_tensor = torch.cat(expected_tensor_list, dim=scatter_dim)

            self.assertEqual(output_tensor, expected_tensor)


if __name__ == "__main__":
    run_tests()
