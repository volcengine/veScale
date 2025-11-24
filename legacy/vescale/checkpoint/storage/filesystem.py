################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from ..utilities.mem_checkpoint import copy_gpu_tensor_to_cpu_pinned_mem_pool, deallocate_cpu_tensor_in_pinned_mem_pool
from abc import ABC, abstractmethod
import collections
from dataclasses import dataclass
import os
import dataclasses
import io
import torch.distributed as dist
import pickle
from typing import List, Tuple, Union, Dict, cast, Any
from ..utilities.logger import get_vescale_checkpoint_logger
import time
import torch
from torch import Tensor
from torch.futures import Future
from pathlib import Path

from torch.distributed.checkpoint.metadata import (
    Metadata,
    MetadataIndex,
)
from torch.distributed.checkpoint.storage import (
    StorageReader,
    StorageWriter,
    WriteResult,
)

from torch.distributed.checkpoint.planner import (
    LoadItemType,
    LoadPlanner,
    LoadPlan,
    SavePlan,
    SavePlanner,
    WriteItem,
    ReadItem,
    WriteItemType,
)

from torch.distributed.checkpoint.utils import _create_file_view

from torch.distributed._shard._utils import narrow_tensor_by_index
from torch._utils import _get_device_module

logger = get_vescale_checkpoint_logger()
from vescale.checkpoint.planner.common import P2PTensorsInfo

__all__ = [
    "FileSystemWriter",
    "FileSystemReader",
]


@dataclass
class _StorageInfo:
    """
    This is the per entry storage info
    """

    relative_path: str
    offset: int
    length: int


@dataclass
class _StoragePrefix:
    prefix: str


DEFAULT_SUFFIX = ".distcp"


def _trim(tensor: torch.Tensor) -> torch.Tensor:
    tensor = copy_gpu_tensor_to_cpu_pinned_mem_pool(tensor.detach())
    # Comment the original DCP code
    # When dumping to pinned memory,
    # the memory layout for tensor has been contiguous
    # if tensor._typed_storage()._size() != tensor.numel():
    #    tensor = tensor.clone()
    return tensor


def _result_from_write_item(item: WriteItem, size_in_bytes, storage_data) -> WriteResult:
    return WriteResult(index=item.index, size_in_bytes=size_in_bytes, storage_data=storage_data)


class _TensorLoader(ABC):
    @abstractmethod
    def add(self, fqn, size, obj):
        pass

    @abstractmethod
    def start_loading(self):
        pass

    @abstractmethod
    def values(self):
        pass


def collect_optim_state_across_dp_ranks(
    tensor: torch.Tensor, rank_ranges: Dict[int, Any], p2p_reqs: Dict[int, Any]
) -> torch.Tensor:
    orignal_shape = tensor.shape
    tensor = tensor.flatten()
    logger.debug("DEBUG: Start receiving p2p tensor")
    recv_start = time.time()
    for req in p2p_reqs:
        req.wait()
    recv_end = time.time() - recv_start
    logger.debug(f"DEBUG: Finish receiving p2p tensor. Time cost: {recv_end}s")
    for v in rank_ranges.values():
        received_tensor, param_range = v
        tensor[param_range.start : param_range.end] = received_tensor
    tensor = tensor.reshape(orignal_shape)
    return tensor


class _SerialCpuLoader(_TensorLoader):
    def __init__(self, resolve_fun, p2p_tensors_info: P2PTensorsInfo = None):
        self.resolve_fun = resolve_fun
        self.items = []
        self.p2p_tensors_info = p2p_tensors_info

    def add(self, fqn, size, obj):
        self.items.append((fqn, size, obj))

    def start_loading(self):
        pass

    def values(self):
        for fqn, _, obj in self.items:
            tensor = self.resolve_fun(obj).detach()
            if self.p2p_tensors_info and (obj.index.fqn, obj.index.offset) in self.p2p_tensors_info.recv_tensors:
                tensor = collect_optim_state_across_dp_ranks(
                    tensor=tensor,
                    rank_ranges=self.p2p_tensors_info.recv_tensors[(obj.index.fqn, obj.index.offset)],
                    p2p_reqs=self.p2p_tensors_info.recv_p2p_reqs[(obj.index.fqn, obj.index.offset)],
                )
            elif self.p2p_tensors_info and fqn in self.p2p_tensors_info.recv_tensors:
                tensor = collect_optim_state_across_dp_ranks(
                    tensor=tensor, rank_ranges=self.p2p_tensors_info.recv_tensors[fqn], p2p_reqs=self.recv_p2p_reqs[fqn]
                )
            tensor = copy_gpu_tensor_to_cpu_pinned_mem_pool(tensor)
            # Comment the original DCP code
            # When dumping to pinned memory,
            # the memory layout for tensor has been contiguous
            #   if tensor.storage().size() != tensor.numel():
            #       tensor = tensor.clone()
            yield (
                tensor,
                obj,
            )


class _OverlappingCpuLoader(_TensorLoader):
    def __init__(
        self,
        resolve_fun,
        p2p_tensors_info: P2PTensorsInfo = None,
        stream=None,
        inflight_threshhold=1_000_000,
    ):
        self.resolve_fun = resolve_fun
        self.items = []
        self.inflight_threshhold = inflight_threshhold
        self.in_flight_data = 0
        self.current_items: collections.deque = collections.deque()
        self.idx = 0
        self.started = False
        self.device_type = stream.device_type if stream else torch.device("cuda").type
        self.device_module = _get_device_module(self.device_type)
        self.p2p_tensors_info = p2p_tensors_info
        self.stream = stream or self.device_module.current_stream()
        if self.stream != self.device_module.current_stream():
            self.stream.wait_stream(self.device_module.current_stream())

    @property
    def _done(self):
        return self.idx >= len(self.items)

    def _drain(self):
        drained = []
        if self.in_flight_data >= self.inflight_threshhold:
            self.stream.synchronize()
        while self.in_flight_data >= self.inflight_threshhold:
            val = self.current_items.popleft()
            self.in_flight_data -= val[0].numel() * val[0].element_size()
            drained.append(val)
        return drained

    def _refill(self):
        with self.device_module.stream(self.stream):
            while not self._done and self.in_flight_data < self.inflight_threshhold:
                fqn, _, obj = self.items[self.idx]
                self.idx += 1
                tensor = self.resolve_fun(obj).detach()
                if self.p2p_tensors_info and (obj.index.fqn, obj.index.offset) in self.p2p_tensors_info.recv_tensors:
                    tensor = collect_optim_state_across_dp_ranks(
                        tensor=tensor,
                        rank_ranges=self.p2p_tensors_info.recv_tensors[(obj.index.fqn, obj.index.offset)],
                        p2p_reqs=self.p2p_tensors_info.recv_p2p_reqs[(obj.index.fqn, obj.index.offset)],
                    )
                elif self.p2p_tensors_info and fqn in self.p2p_tensors_info.recv_tensors:
                    tensor = collect_optim_state_across_dp_ranks(
                        tensor=tensor,
                        rank_ranges=self.p2p_tensors_info.recv_tensors[fqn],
                        p2p_reqs=self.p2p_tensors_info.recv_p2p_reqs[fqn],
                    )
                if tensor.device.type == self.device_type:
                    tensor = copy_gpu_tensor_to_cpu_pinned_mem_pool(tensor, non_blocking=True)
                # Comment the original DCP code
                # When dumping to pinned memory, the memory layout for tensor has been contiguous
                #     elif tensor.device == torch.device("cpu"):
                #         if tensor.storage().size() != tensor.numel():
                #             # this forces the tensor to be both contiguous and with minimal storage
                #             tensor = tensor.clone()

                self.current_items.append(
                    (
                        tensor,
                        obj,
                    )
                )
                self.in_flight_data += tensor.numel() * tensor.element_size()

    def _finish(self):
        assert self._done
        if len(self.current_items) > 0:
            self.stream.synchronize()
        return self.current_items

    def add(self, fqn, size, obj):
        if self.started:
            raise RuntimeError("cannot add items after loading started")
        self.items.append((fqn, size, obj))

    def start_loading(self):
        if self.started:
            return
        self.started = True
        self.items.sort(key=lambda x: x[1])
        self._refill()

    def values(self):
        self.start_loading()
        while not self._done:
            drained = self._drain()
            self._refill()
            yield from drained

        yield from self._finish()


def _item_fqn(item: WriteItem) -> str:
    return item.index.fqn


def _item_size(item: WriteItem) -> int:
    size = 1
    assert item.tensor_data is not None
    # can't use math.prod as PT needs to support older python
    for s in item.tensor_data.size:
        size *= s

    dtype = item.tensor_data.properties.dtype
    return size * torch._utils._element_size(dtype)


def _split_by_size_and_type(bins, items: List[WriteItem]) -> List[List[WriteItem]]:
    if bins == 1:
        return [items]

    bytes_w = [wi for wi in items if wi.type == WriteItemType.BYTE_IO]
    tensor_w = [wi for wi in items if wi.type != WriteItemType.BYTE_IO]

    buckets: List[List[WriteItem]] = [[] for _ in range(bins)]
    bucket_sizes = [0 for _ in range(bins)]

    tensor_w.sort(key=_item_size, reverse=True)

    for i, wi in enumerate(bytes_w):
        buckets[i % bins].append(wi)

    for wi in tensor_w:
        idx = min(enumerate(bucket_sizes), key=lambda x: x[1])[0]
        buckets[idx].append(wi)
        bucket_sizes[idx] += _item_size(wi)

    return buckets


def _write_item(stream, data, write_item, storage_key):
    offset = stream.tell()

    if write_item.type == WriteItemType.BYTE_IO:
        assert isinstance(data, io.BytesIO)
        stream.write(data.getbuffer())
    else:
        assert isinstance(data, torch.Tensor)
        assert data.device == torch.device("cpu")
        torch.save(data, stream)
    length = stream.tell() - offset

    return _result_from_write_item(write_item, length, _StorageInfo(storage_key, offset, length))


def _write_files_from_queue(
    file_name,
    storage_key,
    write_items,
    planner: SavePlanner,
    inflight_threshhold: int,
    use_fsync: bool,
    p2p_tensors_info: P2PTensorsInfo = None,
):
    loader: _TensorLoader

    if torch.cuda.is_available() and inflight_threshhold > 0:
        loader = _OverlappingCpuLoader(
            lambda x: planner.resolve_data(x),
            inflight_threshhold=inflight_threshhold,
            p2p_tensors_info=p2p_tensors_info,
        )
    else:
        loader = _SerialCpuLoader(lambda x: planner.resolve_data(x), p2p_tensors_info=p2p_tensors_info)

    tensor_w = [wi for wi in write_items if wi.type != WriteItemType.BYTE_IO]
    for write_item in tensor_w:
        loader.add(_item_fqn(write_item), _item_size(write_item), write_item)
    loader.start_loading()

    bytes_w = [wi for wi in write_items if wi.type == WriteItemType.BYTE_IO]
    write_results = []

    stream = open(file_name, "wb")
    logger.debug("Start writing byte io data.")
    byte_io_write_start = time.time()
    for write_item in bytes_w:
        data = planner.resolve_data(write_item)
        write_results.append(_write_item(stream, data, write_item, storage_key))
    byte_io_write_time = time.time() - byte_io_write_start
    logger.debug(f"Finish writing byte io data. Time cost: {byte_io_write_time}s")

    logger.debug("Start writing tensor data.")
    tensor_write_start = time.time()
    for tensor, write_item in loader.values():
        assert tensor.is_cpu
        write_results.append(_write_item(stream, tensor, write_item, storage_key))
        # WARNING: Call deallocate_cpu_tensor_in_pinned_mem_pooltensor
        # when the reference to CPU tensor goes to zero
        # so the memory pool will reuse the memory if possbile
        # Othterwise, the memory pool will allocate memory on the used memory range,
        # leading to cuda error 712 cudaErrorHostMemoryAlreadyRegistered
        deallocate_cpu_tensor_in_pinned_mem_pool(tensor)
    tensor_write_time = time.time() - tensor_write_start
    logger.debug(f"Finish writing tensor data. Time cost: {tensor_write_time}s")

    if use_fsync:
        os.fsync(stream.fileno())

    file_stream_close_start = time.time()
    stream.close()
    file_stream_close_time = time.time() - file_stream_close_start
    logger.debug(f"Finish closing file stream. Time cost: {file_stream_close_time}s")
    return write_results


def _write_files_per_proc(
    file_path: Path,
    storage_key: str,
    byte_data_item: List[Tuple[io.BytesIO, WriteItem]],
    tensor_data_item: List[Tuple[torch.Tensor, WriteItem]],
    use_fsync: bool,
) -> List[WriteResult]:
    write_results = []
    stream = open(file_path, "wb")
    # First write byte data.
    for write_data, write_item in byte_data_item:
        write_results.append(_write_item(stream, write_data, write_item, storage_key))
    # Then write tensor data.
    # NOTE: the pinned memory occupied by each tensor have been reallocated.
    for write_data, write_item in tensor_data_item:
        write_results.append(_write_item(stream, write_data, write_item, storage_key))

    if use_fsync:
        os.fsync(stream.fileno())

    return write_results


def _serialize_tensor(tensor: torch.Tensor) -> bytes:
    bio = io.BytesIO()
    # NOTE: currently use torch.save() to do the serialization.
    torch.save(tensor, bio)
    return bio.getbuffer()


def _write_to_file(stream, content: bytes, write_item: WriteItem, storage_key: str) -> WriteResult:
    offset = stream.tell()
    stream.write(content)
    length = stream.tell() - offset
    return _result_from_write_item(write_item, length, _StorageInfo(storage_key, offset, length))


def _write_files_per_proc_pipe(
    file_path: Path,
    storage_key: str,
    byte_data_item: List[Tuple[io.BytesIO, WriteItem]],
    tensor_data_item: List[Tuple[torch.Tensor, WriteItem]],
    use_fsync: bool,
) -> List[WriteResult]:
    write_futures = []
    write_results = []
    stream = open(file_path, "wb")
    executor = ThreadPoolExecutor(max_workers=1)
    # For byte data, directly write byte data.
    for write_data, write_item in byte_data_item:
        content = write_data.getbuffer()
        write_futures.append(
            executor.submit(
                _write_to_file,
                stream,
                content,
                write_item,
                storage_key,
            )
        )
        # write_results.append(_write_to_file(stream, content, write_item, storage_key))
    # For tensor data, perform serialization in process then do saving in threadpool.
    for write_data, write_item in tensor_data_item:
        content = _serialize_tensor(write_data)
        write_futures.append(
            executor.submit(
                _write_to_file,
                stream,
                content,
                write_item,
                storage_key,
            )
        )
        # write_results.append(_write_to_file(stream, content, write_item, storage_key))

    for fut in write_futures:
        write_results.append(fut.result())
    if use_fsync:
        os.fsync(stream.fileno())
    executor.shutdown(wait=False)
    return write_results


def stat_analysis(tasks, planner, p2p_tensors_info, use_fsync=True) -> List[WriteResult]:
    """
    Analyzing the overhead of D2H transfer, serialization, and save operations. Assume that
    all items are written into one file.
    """
    # Step1, aysnc D2H, dumping objects to pinned share memory.
    assert len(tasks) == 1, "please generate one write task for analysis"
    loader = _SerialCpuLoader(lambda x: planner.resolve_data(x), p2p_tensors_info=p2p_tensors_info)
    # Add Bytes.
    byte_item_to_write = []
    for task in tasks:
        _, _, write_items = task
        byte_w = [wi for wi in write_items if wi.type == WriteItemType.BYTE_IO]
        byte_item_to_write.extend(byte_w)
    # Add tenosrs.
    tensor_item_to_write = []
    for task in tasks:
        _, _, write_items = task
        tensor_w = [wi for wi in write_items if wi.type != WriteItemType.BYTE_IO]
        tensor_item_to_write.extend(tensor_w)
        for write_item in tensor_w:
            loader.add(_item_fqn(write_item), _item_size(write_item), write_item)
    loader.start_loading()
    # Step1: dump to pinned memory pool.
    d2h_dump_wait_start = time.time()
    tensor_to_serialize: List[torch.Tensor] = []
    for tensor, write_item in loader.values():
        assert tensor.is_cpu
        tensor_to_serialize.append(tensor)
        deallocate_cpu_tensor_in_pinned_mem_pool(tensor)
    d2h_dump_wait_time = torch.tensor(time.time() - d2h_dump_wait_start).cuda()
    dist.all_reduce(d2h_dump_wait_time)
    d2h_dump_wait_time = d2h_dump_wait_time.item() / dist.get_world_size()
    if dist.get_rank() == 0:
        logger.critical(f"End waiting for D2H tensors dumping Time: {d2h_dump_wait_time:.4f}s")
    # Step2: call serialization workers to serialize objects.
    serialize_wait_start = time.time()
    tensor_data_to_write = []
    bio = io.BytesIO()
    for tensor in tensor_to_serialize:
        bio.seek(0)
        bio.truncate(0)
        torch.save(tensor, bio)
        dump_b = bio.getvalue()
        assert isinstance(dump_b, bytes)
        tensor_data_to_write.append(dump_b)
    serialize_wait_time = torch.tensor(time.time() - serialize_wait_start).cuda()
    dist.all_reduce(serialize_wait_time)
    serialize_wait_time = serialize_wait_time.item() / dist.get_world_size()
    if dist.get_rank() == 0:
        logger.critical(f"End waiting for serialization Time: {serialize_wait_time:.4f}s")
    # Step3: save/upload the objects from memory to disk.
    file_path = tasks[0][0]
    storage_key = tasks[0][1]
    write_results = []
    assert isinstance(file_path, Path)
    save_upload_wait_start = time.time()
    with open(file_path, "wb") as stream:
        for write_item in byte_item_to_write:
            offset = stream.tell()
            data = planner.resolve_data(write_item)
            stream.write(data.getbuffer())
            length = stream.tell() - offset
            write_results.append(_result_from_write_item(write_item, length, _StorageInfo(storage_key, offset, length)))
        for tensor_data, write_item in zip(tensor_data_to_write, tensor_item_to_write):
            offset = stream.tell()
            stream.write(tensor_data)
            length = stream.tell() - offset
            write_results.append(_result_from_write_item(write_item, length, _StorageInfo(storage_key, offset, length)))
        if use_fsync:
            os.fsync(stream.fileno())
    save_upload_wait_time = torch.tensor(time.time() - save_upload_wait_start).cuda()
    dist.all_reduce(save_upload_wait_time)
    save_upload_wait_time = save_upload_wait_time.item() / dist.get_world_size()
    if dist.get_rank() == 0:
        logger.critical(f"End waiting for tensors saving/uploading Time: {save_upload_wait_time:.4f}s")
    return write_results


class FileSystemWriter(StorageWriter):
    """
    Basic implementation of StorageWriter using file IO.

    This implementation makes the following assumptions and simplifications:

    * The checkpoint path is an empty or non-existing directory.
    * File creation is atomic

    The checkpoint consist of one file per write request plus
    a `.metadata` file with the serialized metadata.

    """

    def __init__(
        self,
        path: Union[str, os.PathLike],
        single_file_per_rank: bool = True,
        sync_files: bool = True,
        worker_count: int = 1,
        per_process_copy_ahead: int = 10_000_000,
    ) -> None:
        """
        Initialize the writer pointing to `path`

        Args:
            path: directory where the checkpoint will be written to.
            single_file_per_rank: Produce one file per rank instead of one file per tensor/blob. Default to True.
            sync_files : force files to be synced to permanent storage. Default to True.
            worker_count: Number of IO workers (processes) to use to write. Default to 1.
            per_process_copy_ahead: How many bytes to copy from the GPU ahead of saving then. Default 10Mb.

        N. B. If sync_files is disabled, there's no guarantee that the checkpoint will be consistent in the case of a failure.
        """
        super().__init__()
        self.path = Path(path)
        self.single_file_per_rank = single_file_per_rank
        self.sync_files = sync_files
        self.worker_count = worker_count
        self.per_process_copy_ahead = per_process_copy_ahead

    def set_up_storage_writer(self, is_coordinator: bool) -> None:
        pass

    def prepare_local_plan(self, plan: SavePlan, p2p_tensors_info: P2PTensorsInfo = None) -> SavePlan:
        self.path.mkdir(parents=True, exist_ok=True)
        self.p2p_tensors_info = p2p_tensors_info
        return plan

    def prepare_global_plan(self, global_plan: List[SavePlan]) -> List[SavePlan]:
        new_plans = [
            dataclasses.replace(plan, storage_data=_StoragePrefix(f"__{i}_")) for i, plan in enumerate(global_plan)
        ]
        return new_plans

    def prepare_write_data(self, tasks: List[Tuple[Path, str, List[WriteItem]]], planner: SavePlanner):
        """
        First stage of saving, Perform Copy data to CPU (D2H).

        Args:
            tasks: partitoned tasks for workers to conduct serialization and the actual saving.
            planner: save planner used to resolve the bytes and tensor data.
            async_io: whether do asynchrous D2H.

        NOTE: Currently we do D2H synchronously.
        """

        byte_data_item_writes: List[List[Tuple[io.BytesIO, WriteItem]]] = []
        tensor_data_item_writes: List[List[Tuple[torch.Tensor, WriteItem]]] = []
        file_path_names: List[Tuple[Path, str]] = []

        # Perform D2H in copy stream.
        d2h_dump_start = time.time()
        for task in tasks:
            file_path, file_name, write_items = task
            byte_w = [wi for wi in write_items if wi.type == WriteItemType.BYTE_IO]
            tensor_w = [wi for wi in write_items if wi.type != WriteItemType.BYTE_IO]
            byte_data_item = [(planner.resolve_data(wi), wi) for wi in byte_w]
            tensor_data_item = []
            # Async copy to pinned CPU memory pool.
            for item in tensor_w:
                tensor = planner.resolve_data(item).detach()
                fqn = _item_fqn(item)

                if self.p2p_tensors_info and fqn in self.p2p_tensors_info.recv_tensors:
                    tensor = collect_optim_state_across_dp_ranks(
                        tensor=tensor,
                        rank_ranges=self.p2p_tensors_info.recv_tensors[fqn],
                        p2p_reqs=self.p2p_tensors_info.recv_p2p_reqs[fqn],
                    )
                tensor = copy_gpu_tensor_to_cpu_pinned_mem_pool(tensor, non_blocking=True)
                tensor_data_item.append((tensor, item))
            byte_data_item_writes.append(byte_data_item)
            tensor_data_item_writes.append(tensor_data_item)
            file_path_names.append((file_path, file_name))

        d2h_dump_time = time.time() - d2h_dump_start
        logger.debug(f"End waiting for D2H copy. Time cost: {d2h_dump_time}s")

        # Deallocate pinned memory.
        # NOTE: when prepare_write_data() is called next time, make sure the previous save event is completed.
        # Otherwise, tensors in pinned memory pool may be overwritten.
        for tensor_data_item in tensor_data_item_writes:
            for tensor, _ in tensor_data_item:
                assert tensor.is_cpu
                deallocate_cpu_tensor_in_pinned_mem_pool(tensor)

        return byte_data_item_writes, tensor_data_item_writes, file_path_names

    def write_data(
        self, plan: SavePlan, planner: SavePlanner, async_io: bool = False, io_workers=False
    ) -> Future[List[WriteResult]]:
        storage_plan: _StoragePrefix = plan.storage_data
        file_count = 0

        def gen_file():
            nonlocal file_count
            file_name = f"{storage_plan.prefix}{file_count}{DEFAULT_SUFFIX}"
            file_count += 1
            return file_name

        tasks: List[Tuple[Path, str, List[WriteItem]]] = []
        # Generate K tasks where K is the number of worker_count.
        if self.single_file_per_rank:
            for bucket in _split_by_size_and_type(self.worker_count, plan.items):
                file_name = gen_file()
                tasks.append((self.path / file_name, file_name, bucket))
        # Generate K tasks where K is the number of write items.
        else:
            for item in plan.items:
                file_name = gen_file()
                tasks.append((self.path / file_name, file_name, [item]))
        logger.debug(f"Rank {dist.get_rank()} writes its checkpoint into {len(tasks)} files")
        # Make sure the optimizer states across dp ranks
        # has been sending to other ranks
        # So the receiver can get it when writing tensors to local path

        if self.p2p_tensors_info:
            logger.debug("Start waiting for sending p2p tensors futures")
            p2p_tensor_send_wait_start = time.time()
            for req in self.p2p_tensors_info.send_p2p_reqs:
                req.wait()
            p2p_tensor_send_wait_time = time.time() - p2p_tensor_send_wait_start
            logger.debug(f"End waiting for sending p2p tensors futures Time: {p2p_tensor_send_wait_time}s")

        futures = []
        if not io_workers:
            executor = ProcessPoolExecutor(max_workers=self.worker_count)
            # executor = torch.multiprocessing.get_context("spawn").Pool(self.worker_count)
        else:
            executor = io_workers

        # ProcessPool VERSION.
        if isinstance(executor, ProcessPoolExecutor):
            byte_data_item_writes, tensor_data_item_writes, file_path_names = self.prepare_write_data(tasks, planner)
            for byte_data_item, tensor_data_item, file_path_name in zip(
                byte_data_item_writes, tensor_data_item_writes, file_path_names
            ):
                file_path, storage_key = file_path_name
                worker_args = (file_path, storage_key, byte_data_item, tensor_data_item, self.sync_files)
                futures.append(executor.submit(_write_files_per_proc_pipe, *worker_args))
                # futures.append(self._serialize_workers.apply_async(_write_files_per_proc, worker_args))
            if async_io:
                return futures
            else:
                logger.debug("Start waiting for writing futures (serilization + save)")
                future_wait_start = time.time()
                for fut in futures:
                    fut.result()
                    # fut.wait()
                future_wait_time = time.time() - future_wait_start
                logger.debug(f"End waiting for writing futures. Time cost: {future_wait_time}s")
                return futures
        else:
            # ThreadPool VERSION.
            for task in tasks:
                futures.append(
                    executor.submit(
                        _write_files_from_queue,
                        *task,
                        planner,
                        self.per_process_copy_ahead,
                        self.sync_files,
                        self.p2p_tensors_info,
                    )
                )
            if async_io:
                return futures
            else:
                logger.debug("Start waiting for writing futures")
                future_wait_start = time.time()
                for fut in futures:
                    fut.result()
                future_wait_time = time.time() - future_wait_start
                logger.debug(f"End waiting for writing futures. Time cost: {future_wait_time}s")
                return futures

    def finish(self, metadata: Metadata, results: List[List[WriteResult]]) -> None:
        storage_md = dict()
        for wr_list in results:
            storage_md.update({wr.index: wr.storage_data for wr in wr_list})
        metadata.storage_data = storage_md
        with (self.path / ".metadata.tmp").open("wb") as metadata_file:
            pickle.dump(metadata, metadata_file)
            os.fsync(metadata_file.fileno())

        (self.path / ".metadata.tmp").rename(self.path / ".metadata")


class FileSystemReader(StorageReader):
    def __init__(
        self,
        path: Union[str, os.PathLike],
        broadcast_tensors=False,
        data_parallel_process_group=None,
    ) -> None:
        super().__init__()
        self.path = path
        self.storage_data: Dict[MetadataIndex, _StorageInfo] = dict()
        self.broadcast_tensors = broadcast_tensors
        self.data_parallel_process_group = data_parallel_process_group

        # If broadcast_tensors is enabled, the data_parallel_process_group is not none
        if self.broadcast_tensors:
            assert self.data_parallel_process_group

    def _slice_file(self, file, sinfo: _StorageInfo):
        return _create_file_view(file, sinfo.offset, sinfo.length)

    def _get_file_path(self, relative_path):
        file_path = os.path.join(self.path, relative_path)
        return file_path

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        # group requests by file
        per_file: Dict[str, List[ReadItem]] = dict()
        for read_item in plan.items:
            item_md = self.storage_data[read_item.storage_index]
            path = item_md.relative_path
            per_file.setdefault(path, []).append(read_item)

        # If broadcasting model tensors is enabled,
        # let processes with dp_rank=0 load models and broadcast them to other processes
        if self.broadcast_tensors:
            self.read_data_with_broadcast(per_file=per_file, planner=planner)
        else:
            # Otherwise, let all ranks load tensors from files
            self.read_from_files(per_file=per_file, planner=planner)

        fut: Future = Future()
        fut.set_result(None)

        return fut

    def read_from_files(self, per_file: Dict[str, List[ReadItem]], planner: LoadPlanner):
        for relative_path, reqs in per_file.items():
            file_path = self._get_file_path(relative_path)
            with open(file_path, "rb") as file:
                reqs = sorted(reqs, key=lambda req: self.storage_data[req.storage_index].offset)
                for req in reqs:
                    item_md = self.storage_data[req.storage_index]
                    file_slice = self._slice_file(file, item_md)
                    if req.type == LoadItemType.BYTE_IO:
                        bytes = io.BytesIO(file_slice.read(item_md.length))
                        bytes.seek(0)
                        planner.load_bytes(req, bytes)
                    else:
                        tensor = cast(Tensor, torch.load(file_slice, map_location="cpu"))
                        tensor = narrow_tensor_by_index(tensor, req.storage_offsets, req.lengths)
                        target_tensor = planner.resolve_tensor(req).detach()

                        assert (
                            target_tensor.size() == tensor.size()
                        ), f"req {req.storage_index} mismatch sizes {target_tensor.size()} vs {tensor.size()}"
                        target_tensor.copy_(tensor)
                        planner.commit_tensor(req, target_tensor)

    def read_data_with_broadcast(self, per_file: Dict[str, List[ReadItem]], planner: LoadPlanner):
        for relative_path, reqs in per_file.items():
            if dist.get_rank(self.data_parallel_process_group) == 0:
                file_path = self._get_file_path(relative_path)
                file = open(file_path, "rb")
            dist.barrier(self.data_parallel_process_group)
            reqs = sorted(reqs, key=lambda req: self.storage_data[req.storage_index].offset)
            for req in reqs:
                if dist.get_rank(self.data_parallel_process_group) == 0:
                    item_md = self.storage_data[req.storage_index]
                    file_slice = self._slice_file(file, item_md)

                if req.type == LoadItemType.BYTE_IO:
                    if dist.get_rank(self.data_parallel_process_group) == 0:
                        object_list = [io.BytesIO(file_slice.read(item_md.length))]
                    else:
                        object_list = [None]

                    dist.broadcast_object_list(
                        object_list,
                        src=dist.get_global_rank(self.data_parallel_process_group, 0),
                        group=self.data_parallel_process_group,
                        device=f"cuda:{torch.cuda.current_device()}",
                    )
                    bytes = object_list[0]
                    bytes.seek(0)
                    planner.load_bytes(req, bytes)
                else:
                    if dist.get_rank(self.data_parallel_process_group) == 0:
                        object_list = [cast(Tensor, torch.load(file_slice, map_location="cuda"))]
                    else:
                        object_list = [None]
                    dist.broadcast_object_list(
                        object_list,
                        src=dist.get_global_rank(self.data_parallel_process_group, 0),
                        group=self.data_parallel_process_group,
                        device=f"cuda:{torch.cuda.current_device()}",
                    )
                    tensor = object_list[0].cpu()
                    tensor = narrow_tensor_by_index(tensor, req.storage_offsets, req.lengths)
                    target_tensor = planner.resolve_tensor(req).detach()

                    assert (
                        target_tensor.size() == tensor.size()
                    ), f"req {req.storage_index} mismatch sizes {target_tensor.size()} vs {tensor.size()}"
                    target_tensor.copy_(tensor)
                    planner.commit_tensor(req, target_tensor)

    # Implementing the abstract function in StorageReader
    def read_metadata(self) -> Metadata:
        metadata_path = self._get_file_path(".metadata")
        with open(metadata_path, "rb") as metadata_file:
            metadata = pickle.load(metadata_file)
        return metadata

    def set_up_storage_reader(self, metadata: Metadata, is_coordinator: bool) -> None:
        self.storage_data = metadata.storage_data
        assert self.storage_data is not None

    def prepare_local_plan(self, plan: LoadPlan) -> LoadPlan:
        return plan

    def prepare_global_plan(self, global_plan: List[LoadPlan]) -> List[LoadPlan]:
        return global_plan
