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

# version info
__version__ = "0.3.4a0"
__branch__ = None
__commit_id__ = None

print(f"vescale version is {__version__}")

from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from vescale.dtensor import DTensor
from vescale.dtensor.placement_types import (
    Placement,
    Partial,
    Replicate,
    Shard,
)

# All public APIs from vescale package
__all__ = [
    "DTensor",
    "DeviceMesh",
    "init_device_mesh",
    "distribute_tensor",
    "redistribute_dtensor",
    "Placement",
    "Partial",
    "Replicate",
    "Shard",
]
