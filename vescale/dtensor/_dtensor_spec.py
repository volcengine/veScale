################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

from typing import Tuple

from torch.distributed.tensor._dtensor_spec import (
    TensorMeta,
    DTensorSpec,
)

from torch.distributed.device_mesh import _mesh_resources

from vescale.dtensor.placement_types import is_ragged_shard as _is_ragged_shard
from vescale.utils.monkey_patch import patch_method


__all__ = ["is_ragged_shard", "TensorMeta", "DTensorSpec", "get_sub_spec"]


"""
In this file, we add is_ragged_shard method.
"""


@patch_method(DTensorSpec, "is_ragged_shard")
def is_ragged_shard(self) -> bool:
    """
    return True if the current DTensorSpec is ragged on any mesh dims (devices)
    """
    return any(_is_ragged_shard(placement) for placement in self.placements)


def get_sub_spec(spec: DTensorSpec, include_dim_names: Tuple[str, ...] = (), exclude_dim_names: Tuple[str, ...] = ()):
    if len(include_dim_names) == len(exclude_dim_names) == 0:
        return spec
    spec_mesh = spec.device_mesh
    assert spec_mesh.mesh_dim_names is not None, "Cannot slice a DeviceMesh without mesh_dim_names!"
    assert len(include_dim_names) == 0 or len(exclude_dim_names) == 0, "cannot pass both"

    if len(include_dim_names) == 0:
        exclude_names_set = set(exclude_dim_names)
        include_dim_names = tuple(name for name in spec_mesh.mesh_dim_names if name not in exclude_names_set)

    root_mesh = _mesh_resources.get_root_mesh(spec_mesh)
    sub_mesh = root_mesh[include_dim_names]
    sub_placement = []
    for name in include_dim_names:
        idx = spec_mesh.mesh_dim_names.index(name)
        sub_placement.append(spec.placements[idx])

    return DTensorSpec(mesh=sub_mesh, placements=tuple(sub_placement), tensor_meta=spec.tensor_meta)
