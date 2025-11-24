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
import dataclasses
import io
import grpc
from typing import Tuple
import os
import threading
import contextlib
import pathlib
import subprocess
import time
import queue
from concurrent import futures

from . import mem_file_service_pb2
from . import mem_file_service_pb2_grpc


class _Directory(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lock = threading.RLock()


@dataclasses.dataclass
class _File:
    content: bytes = b""


_CHUNK_SIZE = 2 * 1024 * 1024


def get_mem_server_sock_file(name: str):
    return f"/var/tmp/mem_server_{name}.sock"


class MemFileServicer(mem_file_service_pb2_grpc.VeScaleCheckpointMemFileServiceServicer):
    def __init__(self):
        self._d = _Directory()

    def Write(self, request_iterator, ctx: grpc.ServicerContext):
        b = io.BytesIO()
        name = None
        for req in request_iterator:
            if name is None:
                if not req.name:
                    ctx.abort(grpc.StatusCode.INVALID_ARGUMENT, "Name must be specified.")
                name = req.name
                d, bn = self._iterate_dir(name, ctx, create=True)
            b.write(req.content)
        if name:
            with d.lock:
                d[bn] = _File(content=b.getvalue())
        return mem_file_service_pb2.VeScaleCheckpointWriteResponse()

    def Read(self, req, ctx: grpc.ServicerContext):
        d, bn = self._iterate_dir(req.name, ctx)
        with d.lock:
            if bn not in d or not isinstance(d[bn], _File):
                ctx.abort(grpc.StatusCode.NOT_FOUND, f"{req.name} not found.")
            f: _File = d[bn]
        cur = 0
        while cur < len(f.content):
            yield mem_file_service_pb2.VeScaleCheckpointReadResponse(content=f.content[cur : cur + _CHUNK_SIZE])
            cur += _CHUNK_SIZE

    def Rename(self, req, ctx: grpc.ServicerContext):
        src_dir, src_bn = self._iterate_dir(req.src, ctx)
        dst_dir, dst_bn = self._iterate_dir(req.dst, ctx)
        if src_dir != dst_dir:
            ctx.abort(grpc.StatusCode.UNIMPLEMENTED, "Rename across dir is not supported.")
        d = src_dir
        with d.lock:
            if src_bn not in src_bn:
                ctx.abort(grpc.StatusCode.NOT_FOUND, f"{req.src} is not found.")
            if not req.overwrite and dst_bn in d:
                ctx.abort(grpc.StatusCode.ALREADY_EXISTS, f"{req.dst} already exists.")
            d[dst_bn] = d[src_bn]
            del d[src_bn]
        return mem_file_service_pb2.VeScaleCheckpointRenameResponse()

    def Remove(self, req, ctx: grpc.ServicerContext):
        d, bn = self._iterate_dir(req.name, ctx)
        if bn not in d:
            ctx.abort(grpc.StatusCode.NOT_FOUND, f"{req.name} not found.")
        with d.lock:
            del d[bn]
        return mem_file_service_pb2.VeScaleCheckpointRemoveResponse()

    def Listdir(self, req, ctx: grpc.ServicerContext):
        d, _ = self._iterate_dir(os.path.join(req.name, "*"))
        if d is None:
            return mem_file_service_pb2.VeScaleCheckpointListdirResponse()

        resp = mem_file_service_pb2.VeScaleCheckpointListdirResponse()
        with d.lock:
            for name in d:
                resp.names.append(name)
        return resp

    def Exists(self, req, ctx: grpc.ServicerContext):
        d, bn = self._iterate_dir(req.name)
        if d is None:
            return mem_file_service_pb2.VeScaleCheckpointExistsResponse(exists=False)
        with d.lock:
            return mem_file_service_pb2.VeScaleCheckpointExistsResponse(exists=bn in d)

    def _iterate_dir(self, name: str, ctx: grpc.ServicerContext = None, create=False) -> Tuple[_Directory, str]:
        if ctx is None:

            class FakeCtx:
                def abort(*args, **kwargs):
                    return None, None

            ctx = FakeCtx()
        name = str(pathlib.Path(name).absolute())[1:]
        parts = name.split("/")
        cur = self._d
        for part in parts[:-1]:
            with cur.lock:
                if part not in cur:
                    if not create:
                        return ctx.abort(grpc.StatusCode.NOT_FOUND, f"{part} doesn't exist.")
                    else:
                        cur[part] = _Directory()
                cur = cur[part]
                if not isinstance(cur, _Directory):
                    return ctx.abort(
                        grpc.StatusCode.ALREADY_EXISTS,
                        f"{part} already exist as a file.",
                    )
        return cur, parts[-1]


def start_server(name: str, force=False):
    sock = get_mem_server_sock_file(name)
    if os.path.exists(sock) and not force:
        raise OSError("Mem server is already running.")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    mem_file_service_pb2_grpc.add_VeScaleCheckpointMemFileServiceServicer_to_server(MemFileServicer(), server)
    server.add_insecure_port(f"unix:{sock}")
    server.start()
    return server


# --- Below is general file interface ---

_STUB_CACHE = {}
_STUB_CACHE_LOCK = threading.Lock()
SCHEMA = "/local_mem/"


def get_prefix(name: str):
    return SCHEMA + name


def _get_mem_name_and_name(path: str):
    path = path[len(SCHEMA) :]
    pos = path.find("/")
    if pos == -1:
        return path, "/"
    else:
        return path[:pos], path[pos:]


def _get_stub_and_name(
    path: str,
) -> Tuple[mem_file_service_pb2_grpc.VeScaleCheckpointMemFileServiceStub, str]:
    mem_name, name = _get_mem_name_and_name(path)
    if mem_name not in _STUB_CACHE:
        c = grpc.insecure_channel(f"unix:{get_mem_server_sock_file(mem_name)}")
        with _STUB_CACHE_LOCK:
            _STUB_CACHE[mem_name] = mem_file_service_pb2_grpc.VeScaleCheckpointMemFileServiceStub(c)
    return _STUB_CACHE[mem_name], name


class _FileLike:
    def __init__(self, name: str, mode: str):
        if mode not in ["rb", "wb"]:
            raise NotImplementedError(f"{mode} is not implemented.")
        self._stub, self._name = _get_stub_and_name(name)
        self._mode = mode
        self._is_write = "w" in mode
        if self._is_write:
            self._write_async()
        self._read_buf = None

    @property
    def read_buf(self):
        if self._read_buf is None:
            self._read_buf = io.BytesIO()
            for resp in self._stub.Read(mem_file_service_pb2.VeScaleCheckpointReadRequest(name=self._name)):
                self._read_buf.write(resp.content)
            self._read_buf.seek(0)
        return self._read_buf

    def __getattr__(self, name):
        if not self._is_write:
            return getattr(self.read_buf, name)

    def _write_async(self):
        self._q = queue.Queue()

        def streaming():
            while True:
                content, eof = self._q.get()
                if eof:
                    break
                cur = 0
                while cur < len(content):
                    req = mem_file_service_pb2.VeScaleCheckpointWriteRequest(content=content[cur : cur + _CHUNK_SIZE])
                    if cur == 0:
                        req.name = self._name
                    yield req
                    cur += _CHUNK_SIZE

        self._write_future = self._stub.Write.future(streaming())

    def write(self, content):
        self._q.put((content, False))

    def close(self):
        if self._is_write:
            self._q.put((None, True))
            self._write_future.result()


@contextlib.contextmanager
def open(name, mode) -> io.FileIO:
    f = _FileLike(name, mode)
    try:
        yield f
    finally:
        f.close()


def rename(src, dst, overwrite=False):
    stub, src_name = _get_stub_and_name(src)
    dst_stub, dst_name = _get_stub_and_name(dst)
    if stub != dst_stub:
        raise ValueError(f"Rename across mem file system is not supported. {src} {dst}")
    stub.Rename(mem_file_service_pb2.VeScaleCheckpointRenameRequest(src=src_name, dst=dst_name, overwrite=overwrite))


def remove(name):
    stub, subname = _get_stub_and_name(name)
    stub.Remove(mem_file_service_pb2.VeScaleCheckpointRemoveRequest(name=subname))


def listdir(name):
    try:
        stub, subname = _get_stub_and_name(name)
        resp = stub.Listdir(mem_file_service_pb2.VeScaleCheckpointListdirRequest(name=subname))
        return list(resp.names)
    except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.UNAVAILABLE:
            return []
        raise


def exists(name):
    try:
        stub, subname = _get_stub_and_name(name)
        resp = stub.Exists(mem_file_service_pb2.VeScaleCheckpointExistsRequest(name=subname))
        return resp.exists
    except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.UNAVAILABLE:
            return False
        raise


# --- interface done ---


def start_server_in_new_process(name: str):
    filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detached_mem_server.py")
    return subprocess.Popen(["python3", filename, f"--name={name}"])


def wait_until_fs_ready(name: str, timeout=120):
    stub, _ = _get_stub_and_name(os.path.join(SCHEMA, name))
    t0 = time.time()
    while time.time() < t0 + timeout:
        try:
            stub.Listdir(mem_file_service_pb2.VeScaleCheckpointListdirRequest(name="/"))
            return True
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                time.sleep(0.1)
                continue
            raise
    return False
