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

import torch
import io
import os
import sys
from vescale.dmodule.api import parallelize_module
from vescale.dtensor.device_mesh import DeviceMesh
from vescale.debug import DebugLogger
from contextlib import redirect_stdout

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model import MLP, sharding_plan

world_size = int(os.environ["WORLD_SIZE"])
rank = int(os.environ["RANK"])

DebugLogger.set_vescale_debug_mode(rank_to_print=(0, 1, 2, 3))
device_mesh = DeviceMesh("cuda", list(range(world_size)))


with io.StringIO() as buf, redirect_stdout(buf):
    model = MLP()
    dmodule = parallelize_module(model, device_mesh, sharding_plan)
    input = torch.ones((4, 4, 4))
    output = dmodule(input).to_local()
    output.sum().backward()
    out = buf.getvalue()

out = "".join(out.split())

assert len(out) > 100
