################################################################################
#
# Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
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
import warnings
import torch
from torch.distributed import init_process_group, new_group
from common_dtensor import DTensorTestBase
from torch.testing._internal.common_utils import run_tests
from vescale.dtensor.device_mesh import DeviceMesh


class DeviceMeshInitializeTest(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    def _manual_setup(self):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12345"
        self.device_type = "cuda"
        init_process_group(
            backend="nccl",
            rank=self.rank,
            world_size=self.world_size,
        )  # In DTensorTestBase, but do not use @with_comm

    def test_init_process_group(self):
        """
        Test DeviceMesh's initialization reaction to map rank to cuda device
        when users fail to do so. We simulate the situation by setting up distributed
        environment partially. DeviceMesh initialization takes as input a process group.
        """
        self._manual_setup()
        input_pg = new_group(ranks=list(range(self.world_size)))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            device_mesh = DeviceMesh("cuda", torch.arange(self.world_size), pg=input_pg, _validate_mesh=False)
            if self.rank != 0:
                # validate on rank > 0 since torch.cuda.current_device() returns 0 when the device hasn't been set.
                all_warnings = [str(item.message) for item in w]
                self.assertEqual(len(all_warnings), 2)
                self.assertTrue(
                    any(
                        "Construction from given ProcessGroup is only supported for 1D mesh currently." in warn
                        for warn in all_warnings
                    )
                )
                self.assertTrue(any(warn.startswith("Remember to set cuda device id") for warn in all_warnings))

    def test_init_no_process_group(self):
        """
        Test DeviceMesh's initialization reaction to map rank to cuda device
        when users fail to do so. We simulate the situation by setting up distributed
        environment partially. DeviceMesh initialization takes no process group.
        """
        self._manual_setup()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")  # Catch all warnings
            device_mesh = DeviceMesh("cuda", torch.arange(self.world_size), pg=None, _validate_mesh=False)
            if self.rank != 0:
                # validate on rank > 0 since torch.cuda.current_device() returns 0 when the device hasn't been set.
                all_warnings = [str(item.message) for item in w]
                self.assertEqual(len(all_warnings), 1)
                self.assertTrue(any(warn.startswith("Remember to set cuda device id") for warn in all_warnings))


if __name__ == "__main__":
    run_tests()
