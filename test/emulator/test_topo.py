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
import numpy as np
from torch.testing._internal.common_utils import run_tests

from vescale.emulator.topo import DoubleTree
from common_dtensor import DTensorTestBase


class TestTopo(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 1

    def test_get_tree(self):
        torch.manual_seed(0)

        tree_structure = np.arange(32).reshape(4, 8)
        ranks = [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27]
        mapping = {}
        for i in range(len(ranks)):
            mapping[ranks[i]] = i

        tree = DoubleTree(tree_structure, ranks, mapping)
        tree_string = [
            [
                "[Rank 0] up: -1, down: [1, -1, -1].\n",
                "[Rank 1] up: 0, down: [2, -1, 8].\n",
                "[Rank 2] up: 1, down: [3, -1, -1].\n",
                "[Rank 3] up: 2, down: [-1, -1, -1].\n",
                "[Rank 4] up: 9, down: [5, -1, -1].\n",
                "[Rank 5] up: 4, down: [6, -1, -1].\n",
                "[Rank 6] up: 5, down: [7, -1, -1].\n",
                "[Rank 7] up: 6, down: [-1, -1, -1].\n",
                "[Rank 8] up: 1, down: [9, -1, -1].\n",
                "[Rank 9] up: 8, down: [10, 4, 12].\n",
                "[Rank 10] up: 9, down: [11, -1, -1].\n",
                "[Rank 11] up: 10, down: [-1, -1, -1].\n",
                "[Rank 12] up: 9, down: [13, -1, -1].\n",
                "[Rank 13] up: 12, down: [14, -1, -1].\n",
                "[Rank 14] up: 13, down: [15, -1, -1].\n",
                "[Rank 15] up: 14, down: [-1, -1, -1].\n",
            ],
            [
                "[Rank 0] up: 5, down: [1, -1, -1].\n",
                "[Rank 1] up: 0, down: [2, -1, -1].\n",
                "[Rank 2] up: 1, down: [3, -1, -1].\n",
                "[Rank 3] up: 2, down: [-1, -1, -1].\n",
                "[Rank 4] up: 13, down: [5, -1, -1].\n",
                "[Rank 5] up: 4, down: [6, 8, 0].\n",
                "[Rank 6] up: 5, down: [7, -1, -1].\n",
                "[Rank 7] up: 6, down: [-1, -1, -1].\n",
                "[Rank 8] up: 5, down: [9, -1, -1].\n",
                "[Rank 9] up: 8, down: [10, -1, -1].\n",
                "[Rank 10] up: 9, down: [11, -1, -1].\n",
                "[Rank 11] up: 10, down: [-1, -1, -1].\n",
                "[Rank 12] up: -1, down: [13, -1, -1].\n",
                "[Rank 13] up: 12, down: [14, -1, 4].\n",
                "[Rank 14] up: 13, down: [15, -1, -1].\n",
                "[Rank 15] up: 14, down: [-1, -1, -1].\n",
            ],
        ]
        for idx in [0, 1]:
            for i in range(len(tree.tree[idx])):
                self.assertEqual(str(tree.tree[idx][i]), tree_string[idx][i])


if __name__ == "__main__":
    run_tests()
