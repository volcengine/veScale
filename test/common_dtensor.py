################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

import datetime
import itertools
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, Generator, Iterator, List, Sequence, Tuple, TypeVar, cast

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import (
    TEST_SKIPS,
    MultiProcessTestCase,
    MultiThreadedTestCase,
    skip_if_lt_x_gpu,
    TestSkip,
)
import torch.testing._internal.distributed.fake_pg as fake_pg
from torch.utils._pytree import TreeSpec, tree_flatten, tree_unflatten

import vescale
from vescale import DeviceMesh, Shard, Replicate, distribute_tensor, DTensor
from vescale.dtensor.placement_types import Placement, DTensorSpec

# add new skipped test exit code
TEST_SKIPS["torch-version-2.2"] = TestSkip(90, "Need torch version bigger than 2.2")

VALID_DEVICE_TYPE = ("cuda", "cpu", "meta")
VALID_PG_BACKEND = ("nccl", "gloo", "mpi", "cpu:gloo,cuda:nccl", "meta")

DEVICE_TYPE = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cpu"
PG_BACKEND = "nccl" if DEVICE_TYPE == "cuda" else "gloo"

NUM_DEVICES = 4
# We use this as a proxy for "multiple GPUs exist"
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    # when we actually have multiple GPUs, relax the requirement to smaller counts.
    NUM_DEVICES = min(NUM_DEVICES, torch.cuda.device_count())

T = TypeVar("T")


def skip_unless_torch_gpu(method: T) -> T:
    """
    Test decorator which skips the test unless there's a GPU available to torch.

    >>> # xdoctest: +SKIP
    >>> @skip_unless_torch_gpu
    >>> def test_some_method(self) -> None:
    >>>   ...
    """
    # The builtin @skip_if_no_gpu relies on os.environ['WORLD_SIZE'] being set.
    return cast(T, skip_if_lt_x_gpu(NUM_DEVICES)(method))


def skip_unless_torch_version_bigger_than(torch_version: str):
    """
    Test decorator which skips the test unless current torch version is
    bigger than the given number.

    >>> # xdoctest: +SKIP
    >>> @skip_unless_torch_version_bigger_than(torch_version="2.2")
    >>> def test_some_method(self) -> None:
    >>>   ...
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_torch_version = torch.__version__
            if current_torch_version >= torch_version:
                return func(*args, **kwargs)
            sys.exit(TEST_SKIPS[f"torch-version-{torch_version}"].exit_code)

        return wrapper

    return decorator


@dataclass
class RedistributeProfile:
    num_calls: int


@contextmanager
def redistribute_profiler() -> Generator[RedistributeProfile, None, None]:
    orig_redistribute_local_tensor = vescale.dtensor.redistribute.redistribute_local_tensor
    profile: RedistributeProfile = RedistributeProfile(num_calls=0)

    # pyre-ignore[53]
    def patched_redistribute_local_tensor(
        local_tensor: torch.Tensor, current_spec: DTensorSpec, target_spec: DTensorSpec, async_op: bool = False
    ) -> DTensor:
        result = orig_redistribute_local_tensor(local_tensor, current_spec, target_spec, async_op)
        profile.num_calls += 1
        return result

    try:
        # pyre-ignore[9]
        vescale.dtensor.redistribute.redistribute_local_tensor = patched_redistribute_local_tensor
        yield profile
    finally:
        vescale.dtensor.redistribute.redistribute_local_tensor = orig_redistribute_local_tensor


class DTensorTestBase(MultiProcessTestCase):
    @property
    def world_size(self) -> int:
        return NUM_DEVICES

    @property
    def device_type(self) -> str:
        return getattr(self, "_device_type", DEVICE_TYPE)

    @device_type.setter
    def device_type(self, value: str):
        assert value in VALID_DEVICE_TYPE
        self._device_type = value

    @property
    def backend(self) -> str:
        return getattr(self, "_backend", PG_BACKEND)

    @backend.setter
    def backend(self, value: str):
        assert value in VALID_PG_BACKEND
        self._backend = value

    def build_device_mesh(self) -> DeviceMesh:
        return DeviceMesh(self.device_type, list(range(self.world_size)))

    def init_pg(self) -> None:
        if "nccl" in self.backend and torch.cuda.device_count() < self.world_size:
            sys.exit(TEST_SKIPS[f"multi-gpu-{self.world_size}"].exit_code)

        if self.backend not in VALID_PG_BACKEND:
            raise RuntimeError(f"Backend {self.backend} not supported!")

        if self.backend == "meta":  # meta backend to fake pg # NOTE: upstream does not work
            store = fake_pg.FakeStore()
            dist.init_process_group(
                backend="fake",
                rank=self.rank,
                world_size=self.world_size,
                store=store,
            )
        else:
            dist.init_process_group(
                backend=self.backend,
                world_size=self.world_size,
                rank=self.rank,  # pyre-ignore[16]
                init_method=f"file://{self.file_name}",  # pyre-ignore[16]
                timeout=datetime.timedelta(seconds=1200),
            )

        # set device for nccl pg for collectives
        if "nccl" in self.backend:
            torch.cuda.set_device(self.rank)

    def destroy_pg(self) -> None:
        # Wait for all ranks to reach here before starting shutdown.
        # FIXME dist.barrier deadlocks with multiple threads and NCCL: https://github.com/pytorch/pytorch/issues/95895
        # dist.all_reduce(torch.zeros((1,), device="cuda" if torch.cuda.is_available() else "cpu"))
        # FIXME can't use the above all_reduce as it causes hangs on bionic and focal. It hangs:
        #  test_dtensor.py  -- DTensorMeshTest.test_dtensor_device_mesh_device_conversion
        dist.barrier()
        dist.destroy_process_group()

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    # pyre-ignore[2]:
    def _test_op(self, mesh: DeviceMesh, op_call, *args, **kwargs) -> None:
        with redistribute_profiler() as profile:
            out = op_call(*args, **kwargs)
            dtc = DTensorConverter(mesh, args, kwargs)
            for d_args, d_kwargs in dtc:
                # pyre can't find assertTrue anymore?
                self.assertEqual(dtc.successful(), True)
                d_out = op_call(*d_args, **d_kwargs)
                self.assertEqual(
                    d_out.redistribute(mesh, [Replicate()] * mesh.ndim).to_local(),
                    out,
                )


TestFunc = Callable[[object], object]


# wrapper to initialize comms (process group)
def with_comms(func: TestFunc) -> TestFunc:
    assert func is not None

    @wraps(func)  # pyre-ignore[6]
    def wrapper(self, *args: Tuple[object], **kwargs: Dict[str, Any]) -> None:  # type: ignore[misc]
        # save original device & backend
        origin_device_type = self.device_type
        origin_backend = self.backend

        # auto select device & backend
        if torch.cuda.is_available() and torch.cuda.device_count() >= self.world_size:
            self.device_type = "cuda"
            self.backend = "nccl"
        else:
            self.device_type = "cpu"
            self.backend = "gloo"

        # launch
        self.init_pg()
        func(self, *args, **kwargs)  # type: ignore[misc]
        self.destroy_pg()

        # restore context
        self.device_type = origin_device_type
        self.backend = origin_backend

    return wrapper


# wrapper to initialize comms (process group) within simulator
def with_comms_simulator(func: TestFunc) -> TestFunc:
    assert func is not None

    @wraps(func)  # pyre-ignore[6]
    def wrapper(self, *args: Tuple[object], **kwargs: Dict[str, Any]) -> None:  # type: ignore[misc]
        # save original device & backend
        origin_device_type = self.device_type

        # change to given device & backend
        self.device_type = "meta"

        # launch
        self.init_pg()
        func(self, *args, **kwargs)  # type: ignore[misc]
        self.destroy_pg()

        # restore original device & backend
        self.device_type = origin_device_type

    return wrapper


# wrapper to initialize comms (process group) for specific device & backend
def with_comms_device(device_type: str) -> Callable:
    """
    >>> # xdoctest: +SKIP
    >>> @with_comms_device(device_type="cpu")
    >>> def test_run_cpu(self):
    >>>   ...
    """
    assert device_type in VALID_DEVICE_TYPE

    def decorator(func: TestFunc) -> TestFunc:
        assert func is not None

        @wraps(func)  # pyre-ignore[6]
        def wrapper(self, *args: Tuple[object], **kwargs: Dict[str, Any]) -> None:  # type: ignore[misc]
            # save original device & backend
            origin_device_type = self.device_type
            origin_backend = self.backend

            # change to given device & backend
            self.device_type = device_type
            if self.device_type == "cuda":
                self.backend = "nccl"
            elif self.device_type == "cpu":
                self.backend = "gloo"
            elif self.device_type == "meta":
                self.backend = PG_BACKEND
            else:
                raise ValueError(f"Device type {self.device_type} not supported!")

            # launch
            self.init_pg()
            func(self, *args, **kwargs)  # type: ignore[misc]
            self.destroy_pg()

            # restore original device & backend
            self.device_type = origin_device_type
            self.backend = origin_backend

        return wrapper

    return decorator


class DTensorOpTestBase(MultiThreadedTestCase):
    @property
    def world_size(self) -> int:
        return NUM_DEVICES

    @property
    def device_type(self) -> str:
        return DEVICE_TYPE

    def build_device_mesh(self):
        return DeviceMesh(self.device_type, list(range(self.world_size)))

    def setUp(self) -> None:
        super().setUp()
        self._spawn_threads()


# This is a class for converting args/kwargs of an op into distributed args/kwargs
class DTensorConverter:
    def __init__(
        self,
        mesh: DeviceMesh,
        args: Tuple[object, ...],
        kwargs: Dict[str, object],
    ) -> None:
        self.hit = 0
        self.miss = 0
        self.mesh = mesh
        self.args = args
        self.kwargs = kwargs
        flatten_args, flatten_args_spec = tree_flatten(args)
        flatten_kwargs, flatten_kwargs_spec = tree_flatten(kwargs)

        self.flatten_args: List[object] = flatten_args
        self.flatten_args_spec: TreeSpec = flatten_args_spec
        self.flatten_kwargs: List[object] = flatten_kwargs
        self.flatten_kwargs_spec: TreeSpec = flatten_kwargs_spec

        choices_for_args = []
        for arg in self.flatten_args:
            if isinstance(arg, torch.Tensor):
                choices_for_args.append(self.gen_sharding_choices_for_arg(arg))

        for arg in self.flatten_kwargs:
            if isinstance(arg, torch.Tensor):
                choices_for_args.append(self.gen_sharding_choices_for_arg(arg))

        self.sharding_combs: Iterator[Sequence[Placement]] = iter(itertools.product(*choices_for_args))

    def successful(self) -> bool:
        return self.hit > 0 and self.miss == 0

    def is_supported_tensor(self, t: torch.Tensor) -> bool:
        # TODO: dist tensor need to support quantized and sparse
        # tensors, quantized tensor might be relatively easy, but
        # sparse tensor have special layouts that we need to possibly
        # deal with, until we are clear about them, we don't officially
        # support them.
        return not any(
            [
                t.is_sparse_csr,
                t.is_sparse,
                t.is_mkldnn,
                t.is_quantized,
                t.is_nested,
                torch._is_functional_tensor(t),
                t.is_neg(),
                t.is_conj(),
                t.device.type in ("lazy", "meta"),
                # We need a way to test if a tensor is batched but there
                # is no official APi to do it
                # torch._C._is_batched(t),
            ]
        )

    def gen_sharding_choices_for_arg(self, arg: torch.Tensor) -> Sequence[Placement]:
        mesh_size = self.mesh.size()
        sharding_choices: List[Placement] = [Replicate()]
        # c10d collective does not support bool tensor
        # for bool tensor we treat it as replicated
        if arg.dtype != torch.bool:
            # only generating choices with: replicate, or sharding
            # evenly on a dimension that could be sharded
            sharding_choices = sharding_choices + [
                Shard(i) for i, s in enumerate(arg.shape) if s > 1 and s % mesh_size == 0
            ]
        # TODO: add multi mesh choices
        # all_choices = itertools.product(
        #     *(self.mesh.ndim * [sharding_choices])
        # )
        return sharding_choices

    def __iter__(self) -> "DTensorConverter":
        return self

    def __next__(self) -> Tuple[Tuple[object, ...], Dict[str, object]]:
        try:
            next_sharding_choices = next(self.sharding_combs)
            idx = 0

            new_args: List[object] = []
            for arg in self.flatten_args:
                if isinstance(arg, torch.Tensor):
                    new_args.append(self.to_dist_tensor(arg, self.mesh, [next_sharding_choices[idx]]))
                    idx += 1
                else:
                    new_args.append(arg)

            new_kwargs: List[object] = []
            for arg in self.flatten_kwargs:
                if isinstance(arg, torch.Tensor):
                    new_kwargs.append(self.to_dist_tensor(arg, self.mesh, [next_sharding_choices[idx]]))
                    idx += 1
                else:
                    new_kwargs.append(arg)

            return (
                tree_unflatten(new_args, self.flatten_args_spec),
                tree_unflatten(new_kwargs, self.flatten_kwargs_spec),
            )
        except StopIteration as e:
            raise StopIteration from e

    def to_dist_tensor(self, t: torch.Tensor, mesh: DeviceMesh, placements: List[Placement]) -> torch.Tensor:
        if type(t) is torch.Tensor or type(t) is torch.nn.Parameter:
            if self.is_supported_tensor(t):
                self.hit += 1
                if t.ndim == 0:
                    # scalar tensor by default will be replicated
                    r = distribute_tensor(t, mesh, [Replicate()] * mesh.ndim)
                else:
                    # distribute non-scalar tensors
                    r = distribute_tensor(t, mesh, placements)
                if type(t) is torch.nn.Parameter:
                    r = torch.nn.Parameter(r, requires_grad=r.requires_grad)  # type: ignore[assignment]
                return r
            else:
                self.miss += 1
                return t
        elif torch.overrides.is_tensor_like(t):
            # Blindly converting tensor subclasses to dist tensor can cause
            # unpredictable problems, we explicitly disable this conversion
            # for now (i.e. we don't support DTensor holding tensor subclass
            # until there's a strong reason later).
            self.miss += 1
            return t
        else:
            raise RuntimeError(f"Trying to convert to DTensor, but got {type(t)}")
