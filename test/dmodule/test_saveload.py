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
from typing import Dict
import tempfile

import torch
import torch.distributed as dist
from torch.testing._internal.common_utils import run_tests

from common_dtensor import DTensorTestBase, with_comms_device

import vescale
from vescale.dtensor.api import DeviceMesh
from vescale.dmodule.api import parallelize_module
from vescale.initialize.deferred_init import deferred_init

from .test_initialize import DMLP, param_sharding_plan, fwd_resharding_plan, config

THIS_DIR = tempfile.gettempdir()
SL_DNAME = "tmp"
SL_FNAME = "rank?_!.pt"
local_dir = os.path.join(THIS_DIR, SL_DNAME)
if not os.path.exists(local_dir):
    os.makedirs(local_dir, exist_ok=True)


def _new_file(device_type: str, rank: int):
    assert device_type in ("cpu", "cuda")
    local_dir = os.path.join(THIS_DIR, SL_DNAME)
    assert os.path.exists(local_dir)
    path_file = os.path.join(local_dir, SL_FNAME.replace("?", str(rank)).replace("!", device_type))
    if os.path.exists(path_file):
        os.remove(path_file)
    return path_file


def _get_file(device_type: str, rank: int):
    assert device_type in ("cpu", "cuda")
    local_dir = os.path.join(THIS_DIR, SL_DNAME)
    assert os.path.exists(local_dir)
    path_file = os.path.join(local_dir, SL_FNAME.replace("?", str(rank)).replace("!", device_type))
    assert os.path.exists(path_file)
    return path_file


def _has_file(device_type: str, rank: int):
    assert device_type in ("cpu", "cuda")
    local_dir = os.path.join(THIS_DIR, SL_DNAME)
    path_file = os.path.join(local_dir, SL_FNAME.replace("?", str(rank)).replace("!", device_type))
    return os.path.exists(path_file)


class DModuleTestSL(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    def _assert_equal(self, dist_sd1: Dict, dist_sd2: Dict, exact_device: bool):
        self.assertEqual(len(dist_sd1), len(dist_sd2))
        for (k1, v1), (k2, v2) in zip(dist_sd1.items(), dist_sd2.items()):
            self.assertEqual(k1, k2)
            self.assertTrue(vescale.equal(v1, v2, exact_device))

    def _run_save(self, device_type: str):
        device_mesh = DeviceMesh(device_type, list(range(self.world_size)))
        dmlp = deferred_init(DMLP, config)
        parallelize_module(dmlp, device_mesh, {"parameter": param_sharding_plan, "forward": fwd_resharding_plan})
        dmlp.reset_parameters()
        torch.save(dmlp.state_dict(), _new_file(device_type, self.rank))

    def _run_load_model(self, saved_device_type, model_device_type):
        # load model
        device_mesh = DeviceMesh(model_device_type, list(range(self.world_size)))
        dmlp = deferred_init(DMLP, config)
        parallelize_module(dmlp, device_mesh, {"parameter": param_sharding_plan, "forward": fwd_resharding_plan})
        dmlp.load_state_dict(torch.load(_get_file(saved_device_type, self.rank)))

        # prepare golden
        dmlp_golden = deferred_init(DMLP, config)
        parallelize_module(dmlp_golden, device_mesh, {"parameter": param_sharding_plan, "forward": fwd_resharding_plan})
        dmlp_golden.reset_parameters()

        # match state dict
        self._assert_equal(dmlp.state_dict(), dmlp_golden.state_dict(), exact_device=True)

        # create data
        input_golden = torch.randn(
            config["batch_size"] * config["seq_length"],
            config["hidden_size"],
            device=model_device_type,
            requires_grad=False,
        )
        dist.all_reduce(input_golden, async_op=False)
        input_tensor = input_golden.detach().clone()

        # match forward
        self.assertTrue(vescale.allclose(dmlp(input_tensor), dmlp_golden(input_golden)))

    @with_comms_device(device_type="cpu")
    def test_cpu(self):
        self._run_save("cpu")
        self._run_load_model("cpu", "cpu")
        if _has_file("cuda", self.rank):
            self._run_load_model("cuda", "cpu")

    @with_comms_device(device_type="cuda")
    def test_cuda(self):
        self._run_save("cuda")
        self._run_load_model("cuda", "cuda")
        if _has_file("cpu", self.rank):
            self._run_load_model("cpu", "cuda")


if __name__ == "__main__":
    run_tests()
