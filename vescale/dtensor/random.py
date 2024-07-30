################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

import contextlib
import warnings
from typing import Dict, List, Optional, Tuple
from math import prod

import os
import torch
import torch.distributed as dist
from torch import Tensor

from vescale.dtensor.device_mesh import _get_device_handle, DeviceMesh
from vescale.dtensor.placement_types import DTensorSpec, Shard
from vescale.dtensor._utils import compute_local_shape_and_global_offset

_rng_tracker: Optional["RNGStateTracker"] = None

USE_THREAD_RNG_TRACKER = os.environ.get("VESCALE_SINGLE_DEVICE_RAND", "1") == "1"


def init_vescale_rng_tracker(device_type: str = "cuda"):
    if USE_THREAD_RNG_TRACKER:
        return ThreadBasedRNGTracker(device_type)
    else:
        return OffsetBasedRNGTracker(device_type)


def is_rng_supported_mesh(device_mesh: DeviceMesh) -> bool:
    """Checks if the current device of `device_mesh` supports DTensor's random APIs.
    Currently DTensor Random APIs only supports cuda/cuda-like devices. We suggest
    users call this API to test the availability before using our random APIs.

    Args:
        device_mesh (:class:`DeviceMesh`): The device mesh on which we check if the
            random ops APIs are supported.

    Returns:
        A bool value. True if `device_mesh` supports DTensor Random APIs; False otherwise.

    .. warning::
        Currently we only support correct RNG on cuda/cuda-like devices.
    """
    device_handle = _get_device_handle(device_mesh.device_type)
    if device_handle and hasattr(device_handle, "set_rng_state"):
        return True
    else:
        warnings.warn(
            f"DTensor random operators may not have complete support on {device_mesh.device_type} device mesh"
        )
        return False


def manual_seed(seed: int, device_mesh: DeviceMesh, tp_dim: int = 0) -> None:
    """Sets the seed for generating random numbers for the calling rank.

    Args:
        seed (int): The desired seed.
        device_mesh (:class:`DeviceMesh`): The device mesh to set the seed.
        tp_dim (int, optional): The mesh dimension where to apply Tensor Parallel
            Default: 0

    Returns:
        None

    .. warning::
        When calling this function, :func:`manual_seed` must be called from all ranks of the
        default `ProcessGroup` even if some ranks may not be a part of the `device_mesh`,
        with the same `seed` value.
        If ``device_mesh`` is a sub-mesh and the calling rank is not a part of it,
        `manual_seed` will not set its GPU device's generator seed.
        Current implementation only supports a GPU device mesh.
    """
    device_handle = _get_device_handle(device_mesh.device_type)
    if not device_handle:
        raise NotImplementedError(
            f"DTensor randomness only supports cuda/cuda-like device type, but got {device_mesh.device_type}"
        )

    # allgather the seed over the default PG
    object_list = [seed] * dist.get_world_size()
    dist.all_gather_object(object_list, seed)
    for rank, object in enumerate(object_list):
        if seed != int(object):
            raise RuntimeError(
                f"calling manual_seed function over {device_mesh} but received different seed values on ranks:",
                f"seed on rank {dist.get_rank()} is {seed}, and seed on rank {rank} is {object}!",
            )
    # instantiate a RNG tracker if haven't. By default DTensor uses an
    # VeScaleRNGTrackerType to perform random operators.
    global _rng_tracker
    if not _rng_tracker:
        _rng_tracker = init_vescale_rng_tracker(device_mesh.device_type)

    # the current rank is in mesh
    if device_mesh.get_coordinate() is not None:
        if isinstance(_rng_tracker, TensorParallelRNGTracker):
            _rng_tracker._manual_seed(device_mesh, seed, tp_dim)
        elif isinstance(_rng_tracker, OffsetBasedRNGTracker):
            _rng_tracker._manual_seed(seed)
        elif isinstance(_rng_tracker, ThreadBasedRNGTracker):
            _rng_tracker._manual_seed(seed)
        else:
            raise RuntimeError(f"Unknown type of cuda RNG state tracker: _rng_tracker = {_rng_tracker}")


class RNGStateTracker:
    """
    RNGStateTracker stores Random Number Generator (RNG) state (a ByteTensor object)
    in a dict, mapping from a corresponding tag to each state tensor. It also provides
    a set of convenient utility methods to help access/modify the state tensors. The most
    important interface is _distribute_region which will be used when DTensor executes
    a random op (an operator that calls RNG).
    """

    def __init__(self, device_type: str = "cuda"):
        self._device_type = device_type
        self._device_handle = _get_device_handle(device_type)
        if not (self._device_handle and self._device_handle.is_available()):
            raise RuntimeError(
                f"{self.__class__.__name__} instantiation requires the presence of CUDA/CUDA-like device"
            )

        self._states: Dict[str, Tensor] = {}
        self._devices = [self._device_handle.current_device()]
        self._use_distribute_region = True

    @property
    def rng_states(self) -> Dict[str, Tensor]:
        return self._states

    @property
    def distribute_region_enabled(self) -> bool:
        return self._use_distribute_region

    @distribute_region_enabled.setter
    def distribute_region_enabled(self, value) -> None:
        self._use_distribute_region = value

    def rng_state_is_sync(self, name) -> bool:
        return name in self.rng_states

    def get_seed(self, name: str) -> int:
        if name not in self.rng_states:
            raise RuntimeError(f"{self.__class__.__name__} does not have random state for {name}")

        seed_tensor = (self.rng_states[name])[0:8].view(dtype=torch.int64)
        return int(seed_tensor.item())

    def set_seed(self, name: str, seed: int) -> None:
        seed_tensor = torch.tensor([seed]).view(torch.uint8)
        offset_tensor = torch.tensor([0]).view(torch.uint8)
        self.rng_states[name] = torch.cat([seed_tensor, offset_tensor])

    def _distribute_region(self, spec: DTensorSpec):
        pass


class OffsetBasedRNGTracker(RNGStateTracker):
    """
    This subclass of `RNGStateTracker` defines the default policy of how RNG states
    should be shared and synchronized among all ranks to respect the semantics of DTensor
    random operators.
    """

    def __init__(self, device_type: str = "cuda"):
        super().__init__(device_type)
        # synchronize RNG state using rank 0's current one
        rng_state = self._device_handle.get_rng_state().to(device_type)
        dist.broadcast(rng_state, 0)
        self.rng_states["parallel-rng"] = rng_state.to("cpu")

    def _manual_seed(self, parallel_seed: int) -> None:
        self.set_seed("parallel-rng", parallel_seed)

    @contextlib.contextmanager
    def _distribute_region(self, spec: DTensorSpec):
        # check if the parallel rng state has been synchronized or not
        if not self.rng_state_is_sync("parallel-rng"):
            raise RuntimeError(
                "OffsetBasedRNGTracker requires the random state to be synchronized "
                "before entering into a distribute region!"
            )
        if self.distribute_region_enabled:
            old_offset = self.get_offset("parallel-rng")
            self._set_pre_op_offset(spec)
            with torch.random.fork_rng(self._devices, device_type=self._device_type):
                self._device_handle.set_rng_state(self.rng_states["parallel-rng"])
                try:
                    yield  # execute the region code
                finally:
                    # update offset to synchronize among ranks
                    self._set_post_op_offset(spec, old_offset)
        else:
            yield

    def get_offset(self, name: str) -> int:
        if name not in self.rng_states:
            raise RuntimeError(f"{self.__class__.__name__} does not have random state for {name}")

        offset_tensor = (self.rng_states[name])[8:].view(dtype=torch.int64)
        return int(offset_tensor.item())

    def set_offset(self, name: str, offset: int) -> None:
        if name not in self.rng_states:
            raise RuntimeError(f"{self.__class__.__name__} does not have random state for {name}")

        seed_tensor = (self.rng_states[name])[0:8]
        offset_tensor = torch.tensor([offset]).view(torch.uint8)
        self.rng_states[name] = torch.cat([seed_tensor, offset_tensor])

    def _set_pre_op_offset(self, spec: DTensorSpec) -> None:
        """Set the starting RNG offset for current device's local shard before actual
        op execution. The pre_op_offset value should start from the current RNG offset
        and increment by the size of local shard until it reaches the size of the whole
        DTensor. For different ranks that hold the same DTensor shard, their pre_op_offset
        will be the same.

        Args:
            spec (:class:`DTensorSpec`): the spec of the DTensor object on which
                we prepare the offset for running random ops.

        Returns:
            None

        .. warning::
            Note that, current implementation does not consider DTensor's continguity.

        Example:
            take a DTensor of shape [8, 16] as an example. Assume that the DTensor
            is placed on a device mesh with placements ([Shard(1), Replicate(), Shard(0)]),
            and the mesh is:
                [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
            ``spec.mesh.get_coordinate()`` provides the coordinate of the current rank
            in the mesh. For example, the coordinate of rank 5 is (1, 0, 1).

            Another concept to introduce besides rank coordinate is shard coordinate.
            Each rank holds a local shard of the DTensor. In the example, the DTensor
            is partitioned into 4 [4, 8] shards. The first shard has 2 replicas and
            rank 0 (coord (0, 0, 0)) and rank 2 (coord (0, 1, 0)) have 1 replica each.
            That being said, the local shard on rank 0 and rank 2 correspond to the same
            shard of the DTensor. To denote each DTensor shard, we use a shard coordinate
            (in the example, it will be a tuple (i, j) where shard (i, j) has the slice
            DTensor[4 * i : 4 * (i + 1), 8 * j : 8 * (j + 1)], 0 <= i < 2, 0 <= j < 2).

            Once we have rank coordinate and shard coordinate, we can calculate on each rank
            what shard of the DTensor the rank holds, with the help of dim_map. The dim_map
            of the above DTensor is [2, 0] so the shard coordinate of a rank with rank coord
            (x, y, z) is simply (z, x) by taking(rank_coord[dim_map[0]],rank_coord[dim_map[1]]).
            Following this calculation,
            rank 0 and rank 2 holds the shard of coord (0, 0);
            rank 1 and rank 3 holds the shard of coord (0, 1);
            rank 4 and rank 6 holds the shard of coord (1, 0);
            rank 5 and rank 7 holds the shard of coord (1, 1);

            The last value to calculate before obtaining the starting offset is the shard linear index.
            The starting offset for each rank will be its shard_linear_index * local_tensor_numel.
        """
        dtensor_shape = spec.shape
        mesh = spec.mesh
        dim_map = spec.dim_map

        # Compute shard coordinate:
        # The coordinate on each tensor dim is a tuple (idx, range)
        # If a DTensor is partitioned on its dim i into n shards, and the current rank
        # holds the j-th, then its shard coordinate will be (idx=j, range=n) on dim i
        coordinate = mesh.get_coordinate()
        assert coordinate is not None
        shard_coord = [coordinate[mesh_dim] if mesh_dim >= 0 else 0 for mesh_dim in dim_map]
        shard_size = [mesh.size(mesh_dim) if mesh_dim >= 0 else 1 for mesh_dim in dim_map]

        # compute shard linear index
        shard_linear_idx = self._calc_shard_linear_idx(shard_coord, shard_size)

        # compute starting offset using the first shard's size
        local_size_on_rank_0 = list(dtensor_shape)
        for idx, placement in enumerate(spec.placements):
            if isinstance(placement, Shard):
                mesh_dim_size = mesh.size(idx)
                shard_dim = placement.dim
                local_size_on_rank_0[shard_dim] = placement._local_shard_size_on_dim(
                    dtensor_shape[shard_dim],
                    mesh_dim_size,
                    0,
                    return_offset=False,
                )[0]

        from vescale.dtensor.ops.utils import prod

        local_size = prod(local_size_on_rank_0)

        # get current RNG offset
        current_offset = self.get_offset("parallel-rng")

        # pytorch: offset must be multiple of 4
        # source: aten/src/ATen/cuda/CUDAGeneratorImpl.cpp
        offset_incr = (shard_linear_idx * local_size + 3) // 4 * 4
        self.set_offset("parallel-rng", current_offset + offset_incr)

    def _set_post_op_offset(self, spec: DTensorSpec, old_offset: int) -> None:
        """Sets the RNG to a synchronized state after running the local random op. Every
        rank should set its RNG offset to `old_offset + DTensor.numel()` where old_offset is
        the offset before calling `set_pre_op_offset` i.e. the offset before running DTensor
        random ops.

        Args:
            spec (:class:`DTensorSpec`): the spec of the DTensor object on which
                we post-process the offset for running random ops.

        Returns:
            None
        """
        dtensor_shape = spec.shape

        numel = prod(dtensor_shape)
        # pytorch: offset must be multiple of 4
        # source: aten/src/ATen/cuda/CUDAGeneratorImpl.cpp
        numel = (numel + 3) // 4 * 4
        self.set_offset("parallel-rng", old_offset + numel)

    def _calc_shard_linear_idx(self, shard_coord: List[int], shard_size: List[int]) -> int:
        # compute shard linear index
        shard_linear_idx = 0
        shard_coord_stride = 1
        for idx, size in zip(reversed(shard_coord), reversed(shard_size)):
            shard_linear_idx += idx * shard_coord_stride
            shard_coord_stride *= size

        return shard_linear_idx


class ThreadBasedRNGTracker(OffsetBasedRNGTracker):
    """
    This subclass of `RNGStateTracker` defines how RNG states should be distributed and
    synchronized among ranks while emulating the outcome of single GPUs. In particular,
    whenever invoking a randomized operation on a DTensor, its sharding spec is passed to
    the C++/Cuda side of pytorch through the RNG state. This resolves the issue that
    OffsetBasedRNGTracker does not produce the output identical to single GPU executions.

    For example, consider generating x = torch.rand(4) given the current random seed and
    a global offset. In Cuda's RNG implementation, random numbers are accessed via a triple
    (seed, thread id, offset).

    On a single GPU, 4 GPU threads is created and the i-th thread fills the entry x[i]
    with rand(seed, i, offset). That is, we have
        | Thread 0        | Thread 1        | Thread 2        | Thread 3        |
    x = | rand(0, offset) | rand(1, offset) | rand(2, offset) | rand(3, offset) |
    After the execution of torch.rand(4), the global offset increments by 4, which is the
    granularity of cuda's RNG offsets.

    The global offset increments by the size of the randomness used in each thread, rounded
    up to the nearest multiple of 4. For instance, if 1000 GPU threads is used to generate
    7000 random numbers, each thread takes 7 random numbers from Cuda RNG and the global offset
    increases by 8 afterward.

    However, using OffsetBasedRNGTracker along with an un-patched pytorch, it outputs a
    different tensor given 2 GPUs.
        | GPU 0                                 | GPU 1                                     |
        | Thread 0 of GPU 0 | Thread 1 of GPU 0 | Thread 0 of GPU 1   | Thread 1 of GPU 1   |
    x = | rand(0, offset)   | rand(1, offset)   | rand(0, offset + 4) | rand(1, offset + 4) |
    Furthermore, after the execution, the global offset increments by 8 instead of 4.

    To resolve the issue, each physical thread of each GPU should fill the entry using the
    thread id as if there is only one GPU. In the previous example, the output should be
        | GPU 0                                         | GPU 1                                         |
        | Thread 0 of GPU 0     | Thread 1 of GPU 0     | Thread 0 of GPU 1     | Thread 1 of GPU 1     |
    x = | rand(seed, 0, offset) | rand(seed, 1, offset) | rand(seed, 2, offset) | rand(seed, 3, offset) |
    And after the execution, the global offset should increment by 4.
    This can be done if we pass the sharding info into Cuda functions that generate these
    outputs.

    To use the feature, set the environment variable VESCALE_SINGLE_DEVICE_RAND=1 before
    running your veScale code.

    .. warning::
        This feature suffers an overhead on the Cuda side as each GPU thread calls one
        `curand_init` and `curand` per entry. In contrast, without the sharding info, each
        thread calls one `curand_init` per tensor and one `curand` every 4 entries.

        This feature requires a patched pytorch. The patch is in ......
    """

    def __init__(self, device_type: str = "cuda"):
        super().__init__(device_type)
        # source: aten/src/ATen/native/cuda/DistributionTemplates.h
        self.block_size = 256
        self.unroll = 4
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        # For example, in an A100: props.max_threads_per_multi_processor = 2048, props.multi_processor_count = 108
        self.max_threads_per_multi_processor = props.max_threads_per_multi_processor
        self.blocks_per_sm = self.max_threads_per_multi_processor // self.block_size
        self.max_grid = props.multi_processor_count * self.blocks_per_sm

    def get_offset(self, name: str) -> int:
        if name not in self.rng_states:
            raise RuntimeError(f"{self.__class__.__name__} does not have random state for {name}")

        offset_tensor = (self.rng_states[name])[8:16].view(dtype=torch.int64)
        return int(offset_tensor.item())

    def set_offset(self, name: str, offset: int) -> None:
        if name not in self.rng_states:
            raise RuntimeError(f"{self.__class__.__name__} does not have random state for {name}")

        seed_tensor = (self.rng_states[name])[0:8]
        offset_tensor = torch.tensor([offset]).view(torch.uint8)
        sharding_spec_tensor = (self.rng_states[name])[16:]
        self.rng_states[name] = torch.cat([seed_tensor, offset_tensor, sharding_spec_tensor])

    def get_sharding_spec(self, name: str) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
        if name not in self.rng_states:
            raise RuntimeError(f"{self.__class__.__name__} does not have random state for {name}")

        sharding_spec_tensor = (self.rng_states[name])[16:].view(dtype=torch.int64)
        local_shape, global_offset, global_shape, global_strides = torch.split(
            sharding_spec_tensor, sharding_spec_tensor.size(0) // 4
        )
        return (
            tuple(local_shape.tolist()),
            tuple(global_offset.tolist()),
            tuple(global_shape.tolist()),
            tuple(global_strides.tolist()),
        )

    def set_sharding_spec(
        self,
        name: str,
        sharding_spec: Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]],
        offset: int,
    ) -> None:
        if name not in self.rng_states:
            raise RuntimeError(f"{self.__class__.__name__} does not have random state for {name}")

        seed_tensor = (self.rng_states[name])[0:8]
        spec_tensor = torch.tensor(sum(sharding_spec, start=(offset,))).view(torch.uint8)
        self.rng_states[name] = torch.cat([seed_tensor, spec_tensor])

    @contextlib.contextmanager
    def _distribute_region(self, spec: DTensorSpec):
        # check if the parallel rng state has been synchronized or not
        if not self.rng_state_is_sync("parallel-rng"):
            raise RuntimeError(
                "ThreadBasedRNGTracker requires the random state to be synchronized "
                "before entering into a distribute region!"
            )
        if self.distribute_region_enabled:
            old_offset = self.get_offset("parallel-rng")
            self._set_pre_op_sharding_spec(spec, old_offset)
            with torch.random.fork_rng(self._devices, device_type=self._device_type):
                self._device_handle.set_rng_state(self.rng_states["parallel-rng"])
                try:
                    yield  # execute the region code
                finally:
                    # update offset to synchronize among ranks
                    self._set_post_op_offset(spec, old_offset)
        else:
            yield

    def _set_pre_op_sharding_spec(self, spec: DTensorSpec, old_offset: int) -> None:
        """Passing the DTensor sharding info via Cuda RNG State. Later on,
        each GPU thread can use the info to deduce the correct thread id and
        offset when generating an entry of a DTensor.

        Args:
            spec (:class:`DTensorSpec`): the spec of the DTensor object on which
                we prepare the offset for running random ops.

        Returns:
            None

        .. warning::
            Note that, current implementation does not consider DTensor's continguity.
        """
        if spec.num_shards > 0:
            global_shape = spec.shape
            mesh = spec.mesh

            local_shape, global_offset = compute_local_shape_and_global_offset(global_shape, mesh, spec.placements)
            global_strides = spec.tensor_meta.stride

            if (local_shape, global_offset) == ((), ()):  # a out-of-mesh rank
                local_shape = tuple([0] * len(global_shape))
                global_offset = tuple([0] * len(global_shape))

            self.set_sharding_spec(
                "parallel-rng",
                (local_shape, global_offset, global_shape, global_strides),
                old_offset,
            )

    def _set_post_op_offset(self, spec: DTensorSpec, old_offset: int) -> None:
        """Set the RNG state as the DTensor operation is executed on a single GPU. This
        includes (1) removing the sharding info and (2) incrementing the global offset by
        the number of randomness used in each thread as if there is only one GPU.

        Args:
            spec (:class:`DTensorSpec`): the spec of the DTensor object on which
                we post-process the offset for running random ops.

        Returns:
            None
        """
        dtensor_shape = spec.shape

        numel = prod(dtensor_shape)

        grid_x = min(self.max_grid, (numel + self.block_size - 1) // self.block_size)
        offset_incr = ((numel - 1) // (self.block_size * grid_x * self.unroll) + 1) * self.unroll
        new_offset = old_offset + offset_incr
        self.set_sharding_spec("parallel-rng", ((), (), (), ()), new_offset)


class TensorParallelRNGTracker(RNGStateTracker):
    def __init__(self, device_type: str = "cuda"):
        super().__init__(device_type)
        # copy the default RNG state
        self.rng_states["tensor-parallel-rng"] = self._device_handle.get_rng_state()

    def _manual_seed(
        self,
        device_mesh: DeviceMesh,
        base_seed: int = 1234,
        tp_dim: int = 0,
    ):
        coordinate = device_mesh.get_coordinate()
        assert coordinate is not None
        tensor_parallel_rank = coordinate[tp_dim]
        # this magic number 2718 comes from Megatron's code
        # (https://github.com/NVIDIA/Megatron-LM/blob/060415572f4365a2e895f8036c4e37dad0efbdf5/megatron/core/tensor_parallel/random.py#L162-L163)
        MegatronMagicNum = 2718
        tensor_parallel_seed = base_seed + MegatronMagicNum + tensor_parallel_rank
        self.set_seed("tensor-parallel-rng", tensor_parallel_seed)

    @contextlib.contextmanager
    def _distribute_region(self, spec: DTensorSpec):
        # check if the tensor parallel rng state has been synchronized or not
        if not self.rng_state_is_sync("tensor-parallel-rng"):
            raise RuntimeError(
                "TensorParallelRNGTracker requires the random state to be synchronized "
                "before entering into a distribute region!"
            )

        if self.distribute_region_enabled:
            with torch.random.fork_rng(self._devices, device_type=self._device_type):
                self._device_handle.set_rng_state(self.rng_states["tensor-parallel-rng"])
                try:
                    yield
                finally:
                    self.rng_states["tensor-parallel-rng"] = self._device_handle.get_rng_state()
        else:
            yield
