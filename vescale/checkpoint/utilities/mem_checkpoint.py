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
import io
import dataclasses
import os
import torch
from torch import multiprocessing
import threading
from typing import Callable, Dict, Any, DefaultDict, List, Optional
import pickle

from . import bfile
from .logger import get_vescale_checkpoint_logger

logger = get_vescale_checkpoint_logger()

if hasattr(torch.storage, "TypedStorage"):
    TypedStorage = torch.storage.TypedStorage
elif hasattr(torch.storage, "_TypedStorage"):
    TypedStorage = torch.storage._TypedStorage

# TypedStorage changes in pytorch 2.
if torch.__version__ >= "2":

    def untyped_storage(o):
        return o.untyped_storage()

    def location_caster(o):
        return o
elif torch.__version__ >= "1.11":

    def untyped_storage(o):
        return o.storage()._storage

    def location_caster(o):
        return o._storage if isinstance(o, TypedStorage) else o


try:
    lib = torch.cuda.cudart()
except:
    lib = None


def _bytes_to_tensor(b: bytes):
    # Copied from `_object_to_tensor` in
    # https://pytorch.org/docs/2.0/_modules/torch/distributed/distributed_c10d.html
    byte_storage = torch.ByteStorage.from_buffer(b)
    return torch.ByteTensor(byte_storage)


class PinnedStoragePool:
    def __init__(self):
        self._l = threading.Lock()
        self._m = DefaultDict(set)

    def allocate(self, nbytes: int):
        with self._l:
            # We don't really need storage to have the exact size. So in theory we can find a
            # bigger storage that may suit here. But so far we keep everything simple here.
            s = self._m[nbytes]
            if not s:
                t = torch.empty([nbytes], dtype=torch.uint8)
                t = t.share_memory_()
                if lib is not None and nbytes != 0:
                    err = lib.cudaHostRegister(t.data_ptr(), t.numel() * t.element_size(), 0)
                    assert err == 0, err
                storage = untyped_storage(t)
                s.add(storage)
            return s.pop()

    def deallocate(self, s):
        # WARNING: Call deallocate when the reference to CPU tensor goes to zero
        # so the memory pool will reuse the memory if possbile
        # Othterwise, the memory pool will allocate memory on the used memory range,
        # leading to cuda error 712 cudaErrorHostMemoryAlreadyRegistered
        with self._l:
            self._m[s.nbytes()].add(s)


GLOBAL_POOL = PinnedStoragePool()

TID = threading.get_ident()


def copy_gpu_tensor_to_cpu_pinned_mem_pool(tensor: torch.Tensor, non_blocking=False) -> torch.Tensor:
    """
    Copy a tensor on GPU to pinned memory pool (host CPU memory).
    The input tensor will not be modified
    Args:
        tensor: a tensor on cuda device
    Return:
        a tensor on cpu, whose data is the same as input tensor
    """
    m = {}
    _old_warning = getattr(torch.storage, "_warn_typed_storage_removal", None)
    torch.storage._warn_typed_storage_removal = lambda *args, **kwags: None

    def persistent_id(o):
        if torch.is_storage(o) or isinstance(o, TypedStorage):
            storage = o
            if storage._cdata in m:
                return storage._cdata
            if storage.device.type != "cpu":
                copied = GLOBAL_POOL.allocate(storage.nbytes())
                copied.copy_(storage, non_blocking=non_blocking)
                if isinstance(storage, TypedStorage):
                    copied = storage._new_wrapped_storage(copied)
            else:
                copied = storage.clone()
            m[storage._cdata] = copied
            return storage._cdata
        return

    b = io.BytesIO()
    p = pickle.Pickler(b)
    p.persistent_id = persistent_id
    p.dump(tensor)
    b.seek(0)
    up = pickle.Unpickler(b)
    up.persistent_load = lambda i: m[i]
    cpu_tensor = up.load()
    """
    assert type(tensor) == torch.Tensor
    storage_obj = tensor.storage()
    cpu_storage = GLOBAL_POOL.allocate(storage_obj.nbytes())

    cpu_storage.copy_(storage_obj, non_blocking=non_blocking)
    cpu_tensor = torch.tensor(cpu_storage)
    """
    torch.storage._warn_typed_storage_removal = _old_warning
    return cpu_tensor


def deallocate_cpu_tensor_in_pinned_mem_pool(tensor: torch.Tensor):
    "Deallocate CPU tensor in the global pinned memory pool"
    GLOBAL_POOL.deallocate(tensor.untyped_storage())


class _CalledOnce:
    def __init__(self, func):
        self._l = threading.Lock()
        self._func = func
        self._res = None
        self._called = False

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        with self._l:
            if self._called:
                return self._res
            self._called = True
            self._res = self._func(*args, **kwargs)
        return self._res


_LOCATION_TAG_LOCK = threading.Lock()


@dataclasses.dataclass
class _SaveArgs:
    obj: object
    storage_tags: list
    pickle_module: __module__
    args: list
    kwargs: dict


def _serialize_obj_with_map(a: _SaveArgs, as_shared_tensor=False):
    """Called to serialize an object to a byte stream or a shared tensor.

    Args:
        a (_SaveArgs): The save args consist of the original tensor to serialize,
            the location tags, the pickle module and other args
        as_shared_tensor (bool): Whether to serialize to a shared tensor or a byte stream.
            Set False if no inter process communication will happen subsequently

    Returns:
        byte stream or shared tensor: The serialized object

    """
    lm = {}
    for storage, tag in a.storage_tags:
        lm[storage._cdata] = tag

    def location_tag(storage):
        loc = lm.get(storage._cdata, None)
        if loc is None:
            if storage.nbytes() == 0:
                # if return None, save will succeed, but load will fail afterwards
                return "cpu"
            raise ValueError("Unknown storage")
        return loc

    with _LOCATION_TAG_LOCK:
        old_location_tag = torch.serialization.location_tag
        torch.serialization.location_tag = location_tag

        bio = io.BytesIO()
        pickle_module = a.pickle_module or pickle
        torch.save(a.obj, bio, pickle_module=pickle_module, *a.args, **a.kwargs)

        torch.serialization.location_tag = old_location_tag
    b = bio.getvalue()
    if not as_shared_tensor:
        return b
    else:
        return _bytes_to_tensor(b).share_memory_()


def _write(f, sa):
    # Serialize tensor obj directly to a byte stream, no need to convert it
    # back to a shared tensor because the whole procedure happens in the same
    # process
    b = _serialize_obj_with_map(sa)
    bfile.safe_atomic_write(f, b)


@dataclasses.dataclass
class _PoolArgs:
    pinned_pool: PinnedStoragePool
    pooled_storages: list


class _WriteFunc:
    def __init__(self, sa: _SaveArgs, pa: _PoolArgs, async_worker):
        self._sa = sa
        if self._sa.pickle_module == pickle:
            # This makes wa serializable.
            self._sa.pickle_module = None
        self._pa = pa
        self._async_worker = async_worker

        self._enable_mp = async_worker is not None and sa.pickle_module is None
        self._des = _CalledOnce(self._des_do_not_call_directly)
        self._l = threading.RLock()
        self._serialized = None
        self._bytes = None

    def _des_do_not_call_directly(self):
        for s in self._pa.pooled_storages:
            self._pa.pinned_pool.deallocate(s)

    def __del__(self):
        self._des()

    @property
    def serialized(self):
        with self._l:
            if self._serialized is None:
                if self._enable_mp:
                    self._serialized = self._async_worker.apply(_serialize_obj_with_map, (self._sa, True))
                else:
                    self._serialized = _serialize_obj_with_map(self._sa)
                self._des()
        return self._serialized

    @property
    def bytes(self):
        if self._bytes is None:
            with self._l:
                if self._enable_mp:
                    self._bytes = self.serialized.numpy().tobytes()
                else:
                    self._bytes = self.serialized
        return self._bytes

    def __call__(self, file: str = None):
        if file is None:
            return self.bytes

        if self._async_worker:
            self._async_worker.apply(_write, (file, self._sa))
        else:
            _write(file, self._sa)
        self._des()


class TorchCheckpointRecorder:
    def __init__(
        self,
        fast_mode=None,
        async_worker: multiprocessing.Pool = None,
        pinned_pool=GLOBAL_POOL,
    ):
        self._thread_id = threading.get_ident()
        self._m = {}

        # After 1.11, typed storage is publicly accessible.
        condition = torch.__version__ >= "1.11"
        self._fast_mode = fast_mode if fast_mode is not None else condition
        # Safety check.
        assert not self._fast_mode or condition

        self._async_worker = async_worker
        self._pinned_pool = pinned_pool

    def __enter__(self):
        self._old_save = torch.save
        torch.save = self._save_wrapper
        if self._fast_mode:
            self._old_warning = getattr(torch.storage, "_warn_typed_storage_removal", None)
            torch.storage._warn_typed_storage_removal = lambda *args, **kwags: None
        return self

    def __exit__(self, *args):
        torch.save = self._old_save
        if self._fast_mode:
            if self._old_warning:
                torch.storage._warn_typed_storage_removal = self._old_warning

    def _save_wrapper(self, obj, f, pickle_module=pickle, *args, **kwargs):
        if threading.get_ident() != self._thread_id or not isinstance(f, (str, os.PathLike)):
            return self._old_save(obj, f, pickle_module, *args, **kwargs)

        if self._fast_mode:
            func = self._copy_to_buffer(obj, pickle_module, *args, **kwargs)
        else:
            func = self._save_to_buffer(obj, pickle_module, *args, **kwargs)

        self._m[str(f)] = func

    def _save_to_buffer(self, obj, *args, **kwags):
        b = io.BytesIO()
        self._old_save(obj, b, *args, **kwags)

        def gen_func(b):
            def func(f: str = None):
                if f:
                    return bfile.safe_atomic_write(f, b.getvalue())
                return b.getvalue()

            return func

        return gen_func(b)

    def _copy_to_buffer(self, obj, pickle_module, *args, **kwargs):
        m = {}
        storage_tags = []
        pooled_storages = []

        def persistent_id(o):
            if torch.is_storage(o) or isinstance(o, TypedStorage):
                storage = o
                if storage._cdata in m:
                    return storage._cdata
                if storage.device.type != "cpu":
                    copied = self._pinned_pool.allocate(storage.nbytes())
                    pooled_storages.append(copied)
                    copied.copy_(storage, non_blocking=False)
                    if isinstance(storage, TypedStorage):
                        copied = storage._new_wrapped_storage(copied)
                else:
                    copied = storage.clone()
                m[storage._cdata] = copied
                tag = torch.serialization.location_tag(location_caster(storage))
                storage_tags.append((copied, tag))
                return storage._cdata
            return

        b = io.BytesIO()
        p = pickle_module.Pickler(b)
        p.persistent_id = persistent_id
        p.dump(obj)
        b.seek(0)
        up = pickle_module.Unpickler(b)
        up.persistent_load = lambda i: m[i]
        nobj = up.load()

        sa = _SaveArgs(
            obj=nobj,
            storage_tags=storage_tags,
            pickle_module=pickle_module,
            args=args,
            kwargs=kwargs,
        )
        pa = _PoolArgs(pinned_pool=self._pinned_pool, pooled_storages=pooled_storages)

        return _WriteFunc(sa, pa, self._async_worker)

    @property
    def files(self) -> Dict[str, Callable[[Optional[List[str]]], Optional[bytes]]]:
        return self._m
