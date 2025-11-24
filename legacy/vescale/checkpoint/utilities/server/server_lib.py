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
import grpc
import asyncio
import threading
import dataclasses
import socket
import ipaddress
from typing import DefaultDict, Dict
import pickle
import multiprocessing
import time
import zlib

from . import report_service_pb2
from . import report_service_pb2_grpc


@dataclasses.dataclass
class Item:
    cv: asyncio.Condition = dataclasses.field(default_factory=asyncio.Condition)
    contents: dict = dataclasses.field(default_factory=dict)
    ranks: set = dataclasses.field(default_factory=set)


_GRPC_OPTIONS = [
    ("grpc.max_send_message_length", 1024 * 1024 * 1024),
    ("grpc.max_receive_message_length", 1024 * 1024 * 1024),
    ("grpc.enable_http_proxy", 0),
]


class ReportServicer(report_service_pb2_grpc.VeScaleCheckpointReportServiceServicer):
    """A servicer that simulate `gather` in sync training.
    Using asyncio since we will block all incoming requests
    until we gather all.
    Usage:
    GatherRank: servicer.wait_for(tag)
    OtherRanks: stub.ReportAndWait(Request(tag=tag))
    """

    def __init__(self, world_size: int):
        self._l = asyncio.Lock()
        self._world_size = world_size

        self._gather_dict = DefaultDict(Item)
        self._bc_dict = DefaultDict(Item)

    async def Gather(self, req: report_service_pb2.VeScaleCheckpointGatherRequest, ctx: grpc.aio.ServicerContext):
        i = await self._record(self._gather_dict, req, ctx)
        resp = report_service_pb2.VeScaleCheckpointGatherResponse()
        if req.with_result:
            resp.contents.extend([v for k, v in sorted(i.contents.items(), key=lambda x: x[0])])

        return resp

    async def Broadcast(self, req: report_service_pb2.VeScaleCheckpointBroadcastRequest, ctx: grpc.aio.ServicerContext):
        i = await self._record(self._bc_dict, req, ctx)
        return report_service_pb2.VeScaleCheckpointBroadcastResponse(content=i.contents[req.src_rank])

    async def _record(self, d: Dict[str, Item], req, ctx: grpc.aio.ServicerContext):
        async with self._l:
            i = d[req.tag]
        async with i.cv:
            if req.rank in i.ranks:
                ctx.abort(
                    grpc.StatusCode.INTERNAL,
                    f"Using the same tag in multiple threads/processes. tag: {req.tag}",
                )
            i.ranks.add(req.rank)
            if req.content:
                i.contents[req.rank] = req.content
            if len(i.ranks) == self._world_size:
                async with self._l:
                    del d[req.tag]
                i.cv.notify_all()
            await i.cv.wait_for(lambda: len(i.ranks) == self._world_size)
        return i

    async def GetStatus(self, req: report_service_pb2.VeScaleCheckpointGetStatusRequest, ctx: grpc.aio.ServicerContext):
        async with self._l:
            b = pickle.dumps(
                {
                    "world_size": self._world_size,
                    "gather_dict": self._gather_dict,
                    "bc_dict": self._bc_dict,
                }
            )
        return report_service_pb2.VeScaleCheckpointGetStatusResponse(status=b)


def _is_ipv6_address(ip: str):
    try:
        ip_obj = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return ip_obj.version == 6


def _concat_ip_and_port(ip: str, port: int):
    if not _is_ipv6_address(ip):
        return f"{ip}:{port}"
    else:
        return f"[{ip}]:{port}"


def _get_local_ip():
    try:
        return socket.getaddrinfo(socket.gethostname(), None)[0][4][0]
    except socket.gaierror:
        return socket.getaddrinfo(socket.gethostname(), None, family=socket.AF_INET6)[0][4][0]


@dataclasses.dataclass
class _AsyncObj:
    e: threading.Event = dataclasses.field(default_factory=threading.Event)
    obj: object = None


async def async_serve(servicer, async_addr: _AsyncObj):
    server: grpc.Server = grpc.aio.server(options=_GRPC_OPTIONS)
    report_service_pb2_grpc.add_VeScaleCheckpointReportServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("[::]:0")
    await server.start()
    async_addr.obj = _concat_ip_and_port(_get_local_ip(), port)
    async_addr.e.set()
    await server.wait_for_termination()


def serve(servicer) -> str:
    async_addr = _AsyncObj()
    th = threading.Thread(
        target=lambda servicer=servicer, async_addr=async_addr: asyncio.run(async_serve(servicer, async_addr)),
        daemon=True,
    )
    th.start()
    async_addr.e.wait()
    return async_addr.obj


def _serve_in_loop(world_size, conn):
    servicer = ReportServicer(world_size)
    addr = serve(servicer)
    conn.send(addr)
    conn.close()
    while True:
        time.sleep(1)


def start_server_in_new_process(world_size: int):
    parent_conn, child_conn = multiprocessing.Pipe()
    p = multiprocessing.get_context("spawn").Process(target=_serve_in_loop, args=(world_size, child_conn), daemon=True)
    p.start()
    return parent_conn.recv()


def get_stub(addr: str):
    channel = grpc.insecure_channel(addr, options=_GRPC_OPTIONS)
    return report_service_pb2_grpc.VeScaleCheckpointReportServiceStub(channel)


def _get_tag():
    return "_default_tag"


def gather(
    stub: report_service_pb2_grpc.VeScaleCheckpointReportServiceStub,
    gather_rank: int,
    rank: int,
    obj,
    tag: str = None,
    timeout=None,
):
    tag = tag or _get_tag()
    req = report_service_pb2.VeScaleCheckpointGatherRequest(
        tag=tag, rank=rank, content=pickle.dumps(obj), with_result=(gather_rank == rank)
    )
    resp = stub.Gather(req, timeout=timeout)
    if gather_rank != rank:
        return
    return [pickle.loads(content) for content in resp.contents]


def broadcast(
    stub: report_service_pb2_grpc.VeScaleCheckpointReportServiceStub,
    src_rank: int,
    rank: int,
    obj=None,
    tag: str = None,
    timeout=None,
):
    tag = tag or _get_tag()
    content = b"" if rank != src_rank else pickle.dumps(obj)
    # Since we will transfer this to all machines, compression here is important.
    c_content = zlib.compress(content)
    resp = stub.Broadcast(
        report_service_pb2.VeScaleCheckpointBroadcastRequest(tag=tag, rank=rank, content=c_content, src_rank=src_rank),
        timeout=timeout,
    )
    content = zlib.decompress(resp.content)
    return pickle.loads(content)


def barrier(
    stub: report_service_pb2_grpc.VeScaleCheckpointReportServiceStub,
    rank: int,
    tag: str = None,
    timeout=None,
):
    gather(stub, 0, rank, tag=tag, obj=None, timeout=timeout)


def get_server_status(stub: report_service_pb2_grpc.VeScaleCheckpointReportServiceStub):
    resp = stub.GetStatus(report_service_pb2.VeScaleCheckpointGetStatusRequest())
    return pickle.loads(resp.status)
