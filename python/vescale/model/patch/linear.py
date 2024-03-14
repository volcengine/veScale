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

from types import MethodType
from typing import Sequence, cast
import warnings

import torch

from vescale.dtensor.device_mesh import DeviceMesh
from vescale.dtensor.dtensor import DTensor
from vescale.dtensor.placement_types import Shard
from vescale.dmodule.placements_interface import PlacementsInterface


def make_new_row_parallel_linear_forward(device_mesh: DeviceMesh, out_pi: PlacementsInterface):
    r"""
    The basic idea is to move communication from post-bias to pre-bias by changing pattern:
    `linear(addmm)[Partial output] + AR/RS` to `mm[Partial output] + AR/RS + bias add`.

    This is valid all cases of placement:

    | matmul  | redist | given placement | bias | output  | redist | given placement | Adam
    | P       | AR     | R               | R    | R       | --     | R               | Yes
    | P       | RS     | S(!=-1)         | R    | S(!=-1) | --     | S(!=-1)         | Yes
    | P       | RS     | S(-1)           | S(0) | S(-1)   | --     | S(-1)           | ?
    | P       | --     | P               | P    | P       | --     | P               | No
    | P       | --     | --              | P    | P       | --     | --              | No

    """

    def forward(self: torch.nn.Module, input: DTensor):
        mm_out = torch.matmul(input, self.weight.T)
        mm_out = mm_out.redistribute(device_mesh, out_pi.placements, async_op=out_pi.async_op)
        return mm_out + self.bias

    return forward


class RowParallelLinear:
    @staticmethod
    def patch(root: torch.nn.Module):
        r"""
        Post-patch a RowParallelLinear to support DTensor.

        NOTE:

        1) This function should be used in `parallelize_module()` and after DTensorizing torch.nn.Linear.
        2) To support Adam Optimizer:
            * bias should be `Replicate()`
            * output placements should be given as `Replicate()` or `Shard(dim != -1)`.

        """

        for submod_path, submod in root.named_modules():
            if not isinstance(submod, torch.nn.Linear):
                continue
            if not isinstance(submod.weight, DTensor):
                continue

            # skip no bias
            if submod.bias is None:
                continue

            # skip no row parallel
            is_row_linear = None
            for p in submod.weight.placements:
                if p.is_replicate():
                    continue
                assert not p.is_partial() and not p.is_interleaved_shard()
                shard = cast(Shard, p)
                if shard.dim == 0:
                    is_row_linear = False
                    break
                elif shard.dim == 1:
                    if is_row_linear is not None:
                        raise ValueError(
                            f"RowParallelLinear only supports one sharding dim, " f"but got {submod.weight.placements}"
                        )
                    is_row_linear = True
            if not is_row_linear:
                continue

            # skip no output_placements
            output_pis = root.get_fwd_plan(submod_path + ".output")
            if output_pis is None:
                warnings.warn(
                    f"`{submod_path}` is a Row Parallel Linear without specifying output placements, which can cause undefined result in Adam Optimizer.",
                    UserWarning,
                )
                continue

            # reaching here means: nn.Linear + DTensor + bias + row-parallel + output_placement
            assert isinstance(submod.bias, DTensor)
            assert isinstance(output_pis, Sequence) and len(output_pis) == 1, "Linear has only a single output!"
            out_pi = output_pis[0]
            assert (
                out_pi.placements and isinstance(out_pi.placements, Sequence) and len(out_pi.placements) == 1
            ), "Only 1D sharding is considered now!"
            if any(p.is_partial() for p in submod.bias.placements) or any(p.is_partial() for p in out_pi.placements):
                warnings.warn(
                    f"`{submod_path}` is a Row Parallel Linear with `Partial` bias/output, which can cause undefined result in Adam Optimizer.",
                    UserWarning,
                )

            # replace nn.Linear's forward with customized forward.
            # NOTE: dyanmo doesn't support functools now, use function closure instead.
            # TODO: collaborate with upstream to support functools
            submod.forward = MethodType(
                make_new_row_parallel_linear_forward(device_mesh=submod.weight.device_mesh, out_pi=out_pi),
                submod,
            )
