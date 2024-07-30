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

import torch
from torch.testing._internal.common_utils import run_tests

import logging
import io
import os
from vescale.dmodule.api import parallelize_module
from vescale.dtensor.device_mesh import DeviceMesh
from vescale.debug import DebugLogger
from model import MLP, sharding_plan
from contextlib import redirect_stdout
from common_dtensor import DTensorTestBase, with_comms


class DModuleTestDebugLog(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @with_comms
    def test_simple_std_out(self):
        DebugLogger.set_vescale_debug_mode(rank_to_print=(0, 1, 2, 3))
        device_mesh = DeviceMesh("cuda", list(range(self.world_size)))

        with io.StringIO() as buf, redirect_stdout(buf):
            model = MLP()
            dmodule = parallelize_module(model, device_mesh, sharding_plan)
            input = torch.ones((4, 4, 4))
            output = dmodule(input).to_local()
            output.sum().backward()
            out = buf.getvalue()
        self.assertGreater(len("".join(out.split())), 100)

    @with_comms
    def test_simple_std_out_without_set0(self):
        os.environ["VESCALE_DEBUG_MODE"] = "1"
        device_mesh = DeviceMesh("cuda", list(range(self.world_size)))

        with io.StringIO() as buf, redirect_stdout(buf):
            model = MLP()
            dmodule = parallelize_module(model, device_mesh, sharding_plan)
            input = torch.ones((4, 4, 4))
            output = dmodule(input).to_local()
            output.sum().backward()
            out = buf.getvalue()
        self.assertGreater(len("".join(out.split())), 100)

    @with_comms
    def test_simple_std_out_without_set1(self):
        device_mesh = DeviceMesh("cuda", list(range(self.world_size)))

        with io.StringIO() as buf, redirect_stdout(buf):
            model = MLP()
            os.environ["VESCALE_DEBUG_MODE"] = "1"
            dmodule = parallelize_module(model, device_mesh, sharding_plan)
            input = torch.ones((4, 4, 4))
            output = dmodule(input).to_local()
            output.sum().backward()
            out = buf.getvalue()
        self.assertGreater(len("".join(out.split())), 100)

    @with_comms
    def test_simple_only_rank1(self):
        DebugLogger.set_vescale_debug_mode(rank_to_print=(1))
        device_mesh = DeviceMesh("cuda", list(range(self.world_size)))

        with io.StringIO() as buf, redirect_stdout(buf):
            model = MLP()
            dmodule = parallelize_module(model, device_mesh, sharding_plan)
            input = torch.ones((4, 4, 4))
            output = dmodule(input).to_local()
            output.sum().backward()
            out = buf.getvalue()
        if self.rank == 1:
            self.assertGreater(len("".join(out.split())), 100)
        else:
            self.assertEqual("".join(out.split()), "")

    @with_comms
    def test_simple_logging(self):
        logger = logging.getLogger("test_simple_logging")
        logger.setLevel(logging.DEBUG)

        log_filename = f"logging_sample_rank{self.rank}.log"

        fh = logging.FileHandler(log_filename, mode="w")
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)

        DebugLogger.set_vescale_debug_mode(rank_to_print=(0, 1, 2, 3), logger=logger)
        device_mesh = DeviceMesh("cuda", list(range(self.world_size)))

        model = MLP()
        dmodule = parallelize_module(model, device_mesh, sharding_plan)
        input = torch.ones((4, 4, 4))
        output = dmodule(input).to_local()
        output.sum().backward()

        with open(log_filename) as file:
            out = file.read()
        out = "".join(out.split())
        self.assertGreater(len(out), 100)


if __name__ == "__main__":
    run_tests()
