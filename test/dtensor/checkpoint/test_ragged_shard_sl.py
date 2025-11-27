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

import math
import shutil
from collections import namedtuple
import torch
from torch import nn
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests

from vescale.dtensor._api import distribute_tensor
from vescale.dtensor.placement_types import RaggedShard, Shard, Replicate

from common_dtensor import DTensorTestBase, with_comms

Config = namedtuple("Config", ["shape", "s_dims", "s_local_units", "l_dims", "l_local_units"])


from typing import Set, FrozenSet


def k_sum_distinct_sets(
    n: int,
    k: int,
) -> Set[FrozenSet[int]]:
    if k < 0:
        return set()
    max_val = n

    res: Set[FrozenSet[int]] = set()

    def backtrack(start: int, remain: int, picks_left: int, chosen: list[int]) -> None:
        if picks_left == 0:
            if remain == 0:
                res.add(frozenset(chosen))
            return
        if remain < 0:
            return

        min_possible = sum(range(start, start + picks_left))
        if min_possible > remain:
            return
        hi_start = max(start, max_val - picks_left + 1)
        max_possible = sum(range(hi_start, hi_start + picks_left))
        if max_possible < remain:
            return

        for x in range(start, max_val + 1):
            if x > remain:
                break
            chosen.append(x)
            backtrack(x + 1, remain - x, picks_left - 1, chosen)
            chosen.pop()

    backtrack(start=0, remain=n, picks_left=k, chosen=[])
    return res


# brute force test
def make_brute_force_config(shape, ragged_dims, world_size):
    assert ragged_dims[-1] == len(ragged_dims) - 1
    ragged_length = math.prod(shape[d] for d in ragged_dims)
    combinations = sorted(k_sum_distinct_sets(ragged_length, world_size))
    configs = []
    for i in range(len(combinations) - 1):
        configs.append(
            Config(
                shape=shape,
                s_dims=ragged_dims,
                s_local_units=tuple(combinations[i]),
                l_dims=ragged_dims,
                l_local_units=tuple(combinations[i + 1]),
            )
        )
    return configs


configs = [
    Config(shape=(2, 64), s_dims=tuple(), s_local_units=(0, 1, 0, 0), l_dims=None, l_local_units=None),
    Config(shape=(2, 64), s_dims=tuple(), s_local_units=(0, 1, 0, 0), l_dims=(0, 1), l_local_units=(21, 21, 22, 0)),
    Config(shape=(2, 64), s_dims=(0, 1), s_local_units=(1, 1, 1, 1), l_dims=tuple(), l_local_units=(0, 0, 0, 1)),
    Config(shape=(2, 64), s_dims=(0, 1), s_local_units=(1, 1, 1, 1), l_dims=(0, 1), l_local_units=(21, 21, 22, 0)),
    Config(shape=(2, 64), s_dims=(0,), s_local_units=(1, 1, 0, 0), l_dims=(0, 1), l_local_units=(21, 21, 22, 0)),
    Config(
        shape=(5, 3, 6, 7),
        s_dims=(0, 1),
        s_local_units=(3, 7, 4, 1),
        l_dims=(0, 1, 2),
        l_local_units=(43, 23, 7, 17),
    ),
    Config(
        shape=(5, 3, 6, 7),
        s_dims=(0, 1),
        s_local_units=(3, 7, 4, 1),
        l_dims=None,
        l_local_units=None,
    ),
    Config(
        shape=(5, 3, 6, 7),
        s_dims=None,
        s_local_units=None,
        l_dims=(0, 1, 2, 3),
        l_local_units=(17, 13, 31, 2),
    ),
    Config(
        shape=(5, 3, 6, 7),
        s_dims=(0, 1),
        s_local_units=(5, 3, 7, 0),
        l_dims=(0, 1, 2, 3),
        l_local_units=(17, 13, 31, 2),
    ),
    *make_brute_force_config(shape=(2, 3, 2, 7), ragged_dims=(0, 1, 2), world_size=4),
]


class SimpleModel(nn.Module):
    def __init__(self, shape) -> None:
        super().__init__()
        self.param = nn.Parameter(
            torch.arange(
                0,
                math.prod(shape),
                device="cuda",
                dtype=torch.float32,
            ).view(shape)
        )


class _TestRaggedShardSaveLoadBase(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    def rm_folder(self, path):
        dist.barrier()
        if self.rank == 0:
            try:
                shutil.rmtree(path)
                print(f"Folder '{path}' deleted successfully.")
            except FileNotFoundError:
                print(f"Folder '{path}' not found.")
            except OSError as e:
                print(f"Error deleting folder '{path}': {e}")
        dist.barrier()


class TestRaggedShardSaveLoad(_TestRaggedShardSaveLoadBase):
    @with_comms
    @parametrize(
        "config",
        configs.copy(),
    )
    def test_save_load_ragged_shard(self, config):
        print(f"{config=}")
        path = f"checkpoint_dcp/{self.__class__.__name__}/{self._testMethodName}"
        self.rm_folder(path)
        mesh = init_device_mesh("cuda", (self.world_size,), mesh_dim_names=("default",))
        shape, s_dims, s_local_units, l_dims, l_local_units = config
        simple_model = SimpleModel(shape)
        if s_dims is not None:
            rs_dtensor = distribute_tensor(simple_model.param, mesh, (RaggedShard(s_dims, s_local_units),))
            simple_model.param = nn.Parameter(rs_dtensor)
        state = {
            "model": simple_model.state_dict(),
        }

        dcp.state_dict_saver.save(state, checkpoint_id=path)
        torch.distributed.barrier()
        print("finish save")

        meta_simple_model = SimpleModel(shape)
        golden = meta_simple_model.param.clone()

        if l_dims is not None:
            rs_dtensor = distribute_tensor(meta_simple_model.param, mesh, (RaggedShard(l_dims, l_local_units),))
            meta_simple_model.param = nn.Parameter(rs_dtensor)
            meta_simple_model.param._local_tensor.data.fill_(0)
        else:
            meta_simple_model.param.data.fill_(0)
        ckpt = {
            "model": meta_simple_model.state_dict(),
        }
        dcp.state_dict_loader.load(ckpt, checkpoint_id=path)

        if l_dims is not None:
            result = meta_simple_model.param.full_tensor()
        else:
            result = meta_simple_model.param.data
        self.assertEqual(result, golden, f"\n {result.tolist()=} \n {golden.tolist()=}")
        self.rm_folder(path)


class TestRaggedShardShard0SaveLoad(_TestRaggedShardSaveLoadBase):
    @property
    def world_size(self):
        return 8

    @with_comms
    @parametrize(
        "config",
        configs.copy(),
    )
    def test_save_load_ragged_shard(self, config):
        print(f"{config=}")
        path = f"checkpoint_dcp/{self.__class__.__name__}/{self._testMethodName}"
        self.rm_folder(path)
        mesh = init_device_mesh("cuda", (4, 2), mesh_dim_names=("fsdp", "tp"))
        shape, s_dims, s_local_units, l_dims, l_local_units = config
        new_shape = (2 * shape[0], *shape[1:])
        simple_model = SimpleModel(new_shape)
        if s_dims is not None:
            rs_dtensor = distribute_tensor(simple_model.param, mesh, (Replicate(), Shard(0)))
            rs_dtensor = rs_dtensor.redistribute(mesh, (RaggedShard(s_dims, s_local_units), Shard(0)))
            simple_model.param = nn.Parameter(rs_dtensor)
        state = {
            "model": simple_model.state_dict(),
        }

        dcp.state_dict_saver.save(state, checkpoint_id=path)
        torch.distributed.barrier()
        print("finish save")

        meta_simple_model = SimpleModel(new_shape)
        golden = meta_simple_model.param.clone()

        if l_dims is not None:
            rs_dtensor = distribute_tensor(meta_simple_model.param, mesh, (Replicate(), Shard(0)))
            rs_dtensor = rs_dtensor.redistribute(mesh, (RaggedShard(l_dims, l_local_units), Shard(0)))
            meta_simple_model.param = nn.Parameter(rs_dtensor)
            meta_simple_model.param._local_tensor.data.fill_(0)
        else:
            meta_simple_model.param.data.fill_(0)
        ckpt = {
            "model": meta_simple_model.state_dict(),
        }
        dcp.state_dict_loader.load(ckpt, checkpoint_id=path)

        if l_dims is not None:
            result = meta_simple_model.param.full_tensor()
        else:
            result = meta_simple_model.param.data
        self.assertEqual(result, golden, f"\n {result.tolist()=} \n {golden.tolist()=}")
        self.rm_folder(path)


class TestRaggedShardShard1SaveLoad(_TestRaggedShardSaveLoadBase):
    @property
    def world_size(self):
        return 8

    @with_comms
    @parametrize(
        "config",
        configs.copy(),
    )
    def test_save_load_ragged_shard(self, config):
        print(f"{config=}")
        path = f"checkpoint_dcp/{self.__class__.__name__}/{self._testMethodName}"
        self.rm_folder(path)
        mesh = init_device_mesh("cuda", (4, 2), mesh_dim_names=("fsdp", "tp"))
        shape, s_dims, s_local_units, l_dims, l_local_units = config
        new_shape = (shape[0], 2 * shape[1], *shape[2:])
        simple_model = SimpleModel(new_shape)
        if s_dims is not None:
            rs_dtensor = distribute_tensor(simple_model.param, mesh, (Replicate(), Shard(1)))
            rs_dtensor = rs_dtensor.redistribute(mesh, (RaggedShard(s_dims, s_local_units), Shard(1)))
            simple_model.param = nn.Parameter(rs_dtensor)
        state = {
            "model": simple_model.state_dict(),
        }

        dcp.state_dict_saver.save(state, checkpoint_id=path)
        torch.distributed.barrier()
        print("finish save")

        meta_simple_model = SimpleModel(new_shape)
        golden = meta_simple_model.param.clone()

        if l_dims is not None:
            rs_dtensor = distribute_tensor(meta_simple_model.param, mesh, (Replicate(), Shard(1)))
            rs_dtensor = rs_dtensor.redistribute(mesh, (RaggedShard(l_dims, l_local_units), Shard(1)))
            meta_simple_model.param = nn.Parameter(rs_dtensor)
            meta_simple_model.param._local_tensor.data.fill_(0)
        else:
            meta_simple_model.param.data.fill_(0)
        ckpt = {
            "model": meta_simple_model.state_dict(),
        }
        dcp.state_dict_loader.load(ckpt, checkpoint_id=path)

        if l_dims is not None:
            result = meta_simple_model.param.full_tensor()
        else:
            result = meta_simple_model.param.data
        self.assertEqual(result, golden, f"\n {result.tolist()=} \n {golden.tolist()=}")
        self.rm_folder(path)


instantiate_parametrized_tests(TestRaggedShardSaveLoad)
instantiate_parametrized_tests(TestRaggedShardShard0SaveLoad)
instantiate_parametrized_tests(TestRaggedShardShard1SaveLoad)

if __name__ == "__main__":
    run_tests()
