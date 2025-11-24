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
from typing import Optional

import numpy as np
import torch

from vescale.dtensor.device_mesh import DeviceMesh
from vescale import distribute_tensor
from vescale.dtensor.placement_types import Replicate
from vescale.dtensor import empty as d_empty


class DataLoader:
    def __init__(self, dataset: str, seqlen: int, mesh: Optional[DeviceMesh] = None, dp_rank: int = 0):
        self.data_dir = os.path.join("data", dataset)
        self.seqlen = seqlen
        self.mesh = mesh
        self.dp_rank = dp_rank
        if mesh is not None:
            self.device_type = mesh.device_type
        else:
            self.device_type = "cuda"

    def get_batch(self, split, bsz, lbsz):
        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        if split == "train":
            data = np.memmap(os.path.join(self.data_dir, "train.bin"), dtype=np.uint16, mode="r")
        else:
            data = np.memmap(os.path.join(self.data_dir, "val.bin"), dtype=np.uint16, mode="r")
        if self.mesh is not None:
            ix = d_empty((bsz,), device_mesh=self.mesh, placements=[Replicate()])
        else:
            ix = torch.empty((bsz,), device="cuda")
        ix = torch.randint_like(ix, len(data) - self.seqlen, dtype=torch.int64)
        if self.mesh is not None:
            ix = ix.to_local()
        if self.mesh is None or self.mesh.get_rank() == 0:
            print(f"sum(ix) {sum(ix)}")
        ix = torch.split(ix, lbsz)[self.dp_rank]
        x = torch.stack([torch.from_numpy((data[i : i + self.seqlen]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1 : i + 1 + self.seqlen]).astype(np.int64)) for i in ix])
        x, y = x.to(self.device_type), y.to(self.device_type)
        if self.mesh is not None:
            x = distribute_tensor(x, self.mesh["TP"], [Replicate()])
            y = distribute_tensor(y, self.mesh["TP"], [Replicate()])
        return x, y
