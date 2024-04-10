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
from typing import Callable, Dict, Any, DefaultDict, Tuple, List, Optional
import pickle

from .server import server_lib
from . import bfile
from .logger import get_omnistore_logger

logger = get_omnistore_logger()

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
        with self._l:
            self._m[s.nbytes()].add(s)


GLOBAL_POOL = PinnedStoragePool()


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


@dataclasses.dataclass
class Item:
    file: str
    src: int
    dsts: List[int]


@dataclasses.dataclass
class Strategy:
    eof: bool = False
    bc_ranks_and_files: List[Item] = dataclasses.field(default_factory=list)
    sr_ranks_and_files: List[Item] = dataclasses.field(default_factory=list)


def _choose_rank(ranks: list, existences: List[bool]):
    for rank in ranks:
        if existences[rank]:
            return rank
    for rank, exist in enumerate(existences):
        if exist:
            return rank

    return ranks[0]


def make_strategy(files: List[str], file_existences: Dict[str, List[bool]], bc_threshold):
    file_to_ranks = DefaultDict(list)

    for rank, file in enumerate(files):
        if not file:
            continue
        file_to_ranks[file].append(rank)

    s = Strategy()

    DOWNLOAD = 0
    BC = 1
    SR = 2

    for file, ranks in file_to_ranks.items():
        mode = DOWNLOAD

        if len(ranks) >= bc_threshold:
            mode = BC
        else:
            for rank in ranks:
                if not file_existences[file][rank]:
                    mode = SR
        item = Item(file=file, src=0, dsts=list(ranks))
        if mode == BC:
            ranks_and_files = s.bc_ranks_and_files
        elif mode == SR:
            ranks_and_files = s.sr_ranks_and_files
        else:
            ranks_and_files = None

        if ranks_and_files is not None:
            item.src = _choose_rank(ranks, file_existences[file])
            ranks_and_files.append(item)

    return s


class DistributedTorchLoader:
    """Use torch distributed communication library to distribute dump.
    The key idea here is to use `broadcast` to distribute common files.
    """

    def __init__(
        self,
        stub,
        rank: int,
        path_prefix_pair: Tuple[str, str] = ("", ""),
        bc_thres=6,
        timeout=30 * 60,
        custom_strategy=make_strategy,
    ):
        self._stub = stub
        self._rank = rank
        self._ppp = path_prefix_pair
        self._bc_thres = bc_thres
        self._custom_strategy = custom_strategy
        self._thread_id = threading.get_ident()
        self._old_load = None
        self._timeout = timeout

    def __enter__(self):
        self._old_load = torch.load
        torch.load = self._load_warpper
        return self

    def __exit__(self, *args):
        if any(i is not None for i in args):
            raise RuntimeError(
                f"[rank {self._rank}]: DistributedTorchLoader exits with Exception. The exit args are {args}"
            )
        self._end_loop()
        torch.load = self._old_load

    def _load_warpper(self, f, *args, **kwargs):
        if threading.get_ident() != self._thread_id or not isinstance(f, (str, os.PathLike)):
            return self._old_load(f, *args, **kwargs)

        f = str(f)
        if f.startswith(self._ppp[0]):
            f = f[len(self._ppp[0]) :]
            f = self._ppp[1] + f

        b, _ = self._coordinate(f)
        logger.info(f"_coordinate in _load_warpper of DistributedTorchLoader is called for file {f}")

        return self._old_load(io.BytesIO(b), *args, **kwargs)

    def _coordinate(self, input_f: str):
        s = self._make_strategy(input_f)
        file_to_bytes = self._download_files(s, input_f)
        file_to_size = self._broadcast_file_sizes(file_to_bytes)
        ret = file_to_bytes.get(input_f, None)

        for item in s.bc_ranks_and_files:
            b = self._bytes_broadcast(file_to_bytes.get(item.file, None), file_to_size[item.file], item.src)
            if item.file == input_f:
                ret = b

        for item in s.sr_ranks_and_files:
            if self._rank == item.src or self._rank in item.dsts:
                b = self._bytes_sr(
                    file_to_bytes.get(item.file, None),
                    file_to_size[item.file],
                    item.src,
                    item.dsts,
                )
                if item.file == input_f:
                    ret = b

        server_lib.barrier(self._stub, self._rank, timeout=self._timeout)
        return ret, s.eof

    def _bytes_broadcast(self, b: bytes, size: int, src: int):
        # We can't serialize `b` directly since python3.7's pickler has limit of 4GB
        t = self._bytes_to_dist_tensor(b, size)
        torch.distributed.broadcast(t, src)
        b = t.cpu().numpy().tobytes()
        del t
        return b

    def _bytes_sr(self, b: bytes, size: int, src: int, dsts: List[int]):
        t = self._bytes_to_dist_tensor(b, size)
        if src == self._rank:
            results = []
            for dst in dsts:
                if src != dst:
                    results.append(torch.distributed.isend(t, dst))
            for res in results:
                res.wait()
        else:
            torch.distributed.irecv(t).wait()
        b = t.cpu().numpy().tobytes()
        del t
        return b

    def _bytes_to_dist_tensor(self, b, size):
        pg = torch.distributed.GroupMember.WORLD
        if pg.name() == torch.distributed.Backend.NCCL:
            device = torch.device("cuda", torch.cuda.current_device())
        else:
            device = torch.device("cpu")
        if b is not None:
            return _bytes_to_tensor(b).to(device)
        else:
            return torch.empty(size, dtype=torch.uint8, device=device)

    def _end_loop(self):
        while True:
            _, eof = self._coordinate("")
            if eof:
                break

    def _gather(self, obj):
        return server_lib.gather(self._stub, 0, self._rank, obj, timeout=self._timeout)

    def _broadcast(self, obj):
        return server_lib.broadcast(self._stub, 0, self._rank, obj, timeout=self._timeout)

    def _download(self, f):
        if not bfile.exists(f):
            error_msg = f"Unable to get {f} in {self._rank}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        with bfile.BFile(f, "rb") as f_obj:
            return f_obj.read()

    def _download_files(self, s: Strategy, input_f: str):
        from_remote = False
        file_to_bytes = {}
        for item in (*s.bc_ranks_and_files, *s.sr_ranks_and_files):
            if item.src == self._rank:
                file_to_bytes[item.file] = self._download(item.file)
            if item.file == input_f:
                from_remote = True
        if input_f and not from_remote and input_f not in file_to_bytes:
            file_to_bytes[input_f] = self._download(input_f)
        return file_to_bytes

    def _broadcast_file_sizes(self, file_to_bytes):
        file_to_size = {f: len(b) for f, b in file_to_bytes.items()}
        file_to_size_list = self._gather(file_to_size)
        if self._rank == 0:
            agg_file_to_size = {}
            for ele in file_to_size_list:
                agg_file_to_size.update(ele)
            file_to_size = agg_file_to_size
        return self._broadcast(file_to_size)

    def _make_strategy(self, input_f: str) -> Strategy:
        accessible_file, inaccessible_file = None, None
        if bfile.is_local_path(input_f) and not bfile.exists(input_f):
            inaccessible_file = input_f
        else:
            accessible_file = input_f

        file_tuples = self._gather((accessible_file, inaccessible_file))

        inac_files = None
        if self._rank == 0:
            files = [t[0] or t[1] for t in file_tuples]
            inac_files = list({t[1] for t in file_tuples if t[1]})
        inac_files = self._broadcast(inac_files)
        existences = []
        for f in inac_files:
            existences.append(bfile.exists(f))
        agg_existences = self._gather(existences)
        s = None
        if self._rank == 0:
            ac_file_existence = (True,) * len(files)
            file_existences = DefaultDict(lambda: ac_file_existence)
            for file in inac_files:
                file_existences[file] = list()
            for existences in agg_existences:
                for file, exist in zip(inac_files, existences):
                    file_existences[file].append(exist)
            logger.info(f"File existences info: {file_existences}")

            eof = True
            for f in files:
                if f:
                    eof = False
            if eof:
                s = Strategy(eof=True)
            else:
                s = self._custom_strategy(files, file_existences, self._bc_thres)
        return self._broadcast(s)


class RemappingTorchLoader:
    _LOAD_LOCK = threading.Lock()

    def __init__(self, path_prefix_pair: Tuple[str, str] = ("", "")):
        self._ppp = path_prefix_pair
        self._old_load = None

    def __enter__(self):
        RemappingTorchLoader._LOAD_LOCK.acquire()
        self._old_load = torch.load
        torch.load = self._loader_wrapper
        return self

    def __exit__(self, *args):
        torch.load = self._old_load
        RemappingTorchLoader._LOAD_LOCK.release()

    def _loader_wrapper(self, f, *args, **kwargs):
        if not isinstance(f, (str, os.PathLike)):
            return self._old_load(f, *args, **kwargs)
        f = str(f)
        if f.startswith(self._ppp[0]):
            f = f[len(self._ppp[0]) :]
            f = self._ppp[1] + f
        with bfile.BFile(f, "rb") as fi:
            return self._old_load(fi, *args, **kwargs)
