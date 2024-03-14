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

from typing import Any, Optional, Sequence

from vescale.dtensor.api import normalize_placements
from vescale.dtensor.placement_types import Placement
from dataclasses import dataclass, fields


__all__ = ["PlacementsInterface"]


@dataclass
class PlacementsInterface:
    """The wrapper on one DTensor's placements,
    providing extra interfaces for redistribute, from_local, and more"""

    placements: Optional[Sequence[Placement]] = None  # the placements for this DTensor
    async_op: bool = True  # flag for DTensor.redistribute/from_local
    defer_reshard: bool = False  # flag for deferred resharding mode
    run_check: bool = True  # flag for DTensor.from_local
    skippable_op: bool = True  # flag for DTensor.redistribute # TODO: to enable
    grad: Optional[Sequence[Placement]] = None  # the placement to enforce on this tensor.grad

    @classmethod
    def from_placements(cls, placements: Any) -> Any:
        if isinstance(placements, cls):
            return placements
        return cls(placements)

    def normalize_placements(self, mesh_ndim: int) -> None:
        self.placements = normalize_placements(self.placements, mesh_ndim)
        self.grad = normalize_placements(self.grad, mesh_ndim)

    def is_none(self) -> bool:
        """Is it equivalent to `None` placements;
        True only if all attributes are as default value."""
        return type(self)() == self

    def is_placements(self) -> bool:
        """Is it having only placements without extra inferfaces."""
        return type(self)(self.placements) == self

    def __repr__(self) -> str:
        """Unwrapped representation; return only non-default value"""
        if self.is_none():
            return None.__repr__()
        if self.is_placements():
            return self.placements.__repr__()
        default_inst = type(self)()
        name_value = [
            (sfield.name, getattr(self, sfield.name))
            for dfield, sfield in zip(fields(default_inst), fields(self))
            if dfield.name != "placements" and getattr(default_inst, dfield.name) != getattr(self, sfield.name)
        ]
        return f"PI({self.placements},{[f'{n}={v}' for n, v in name_value].join(',')})"
