################################################################################
#
# Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
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

import torch
from .logger import NDTimelineLogger

NCCL_STREAMS = {}
DEVICE = None


def get_nccl_p2p_stream(name: str, nccl_pg: "torch.distributed.ProcessGroup", peer, is_batched):
    global NCCL_STREAMS, DEVICE
    if DEVICE is None:
        DEVICE = torch.device("cuda", index=torch.cuda.current_device())
    if name in NCCL_STREAMS and NCCL_STREAMS[name] is not None:
        return NCCL_STREAMS[name]
    if hasattr(nccl_pg, "_get_backend"):
        nccl_backend = nccl_pg._get_backend(DEVICE)
    else:
        # before torch 2.x torch._C._distributed_c10d.ProcessGroupNCCL is a subclass of
        # torch.distributed.ProcessGroup
        nccl_backend = nccl_pg
    if hasattr(nccl_backend, "get_p2p_cuda_stream_id"):
        stream_id = nccl_backend.get_p2p_cuda_stream_id(DEVICE.index, peer, is_batched)
        NDTimelineLogger().debug(f"[{DEVICE.index}]{name} [{peer}] stream_id={stream_id}")
        if stream_id < 0:
            rank = nccl_pg.rank()
            NDTimelineLogger().info(f"[{rank}]{name} is_batched={is_batched} p2p stream is not available, skipped")
            return None
        _CUDA_DEVICE = 1
        nccl_stream = torch.cuda.Stream(stream_id=stream_id, device_index=DEVICE.index, device_type=_CUDA_DEVICE)
        rank = nccl_pg.rank()
        msg = f"[{rank}]{name} nccl p2p stream id={stream_id} device={DEVICE} stream={nccl_stream}"
        NDTimelineLogger().debug(msg)
        NCCL_STREAMS[name] = nccl_stream
        return nccl_stream
    return None


def get_nccl_coll_stream(name: str, nccl_pg: "torch.distributed.ProcessGroup", nccl_tensor: torch.Tensor):
    global NCCL_STREAMS
    if name in NCCL_STREAMS and NCCL_STREAMS[name] is not None:
        return NCCL_STREAMS[name]
    device = nccl_tensor.device
    if hasattr(nccl_pg, "_get_backend"):
        nccl_backend = nccl_pg._get_backend(device)
    else:
        # before torch 2.x torch._C._distributed_c10d.ProcessGroupNCCL is a subclass of
        # torch.distributed.ProcessGroup
        nccl_backend = nccl_pg
    if hasattr(nccl_backend, "get_coll_cuda_stream_id"):
        NDTimelineLogger().info(nccl_backend)
        stream_id = nccl_backend.get_coll_cuda_stream_id([nccl_tensor])
        if stream_id < 0:
            rank = nccl_pg.rank()
            NDTimelineLogger().info(f"[{rank}]{name} coll stream is not available, skipped")
            return None
        _CUDA_DEVICE = 1
        nccl_stream = torch.cuda.Stream(stream_id=stream_id, device_index=device.index, device_type=_CUDA_DEVICE)
        rank = nccl_pg.rank()
        msg = f"[{rank}]{name} nccl coll stream id={stream_id} device={device} stream={nccl_stream}"
        NDTimelineLogger().debug(msg)
        NCCL_STREAMS[name] = nccl_stream
        return nccl_stream
    return None
