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

import torch
from vescale.emulator.reduce_kernel import ReduceOp


class RingPrimitive:
    def __init__(self, data_list, ring, reduce_op=ReduceOp.SUM, device=None, datatype=None):
        self.ring = ring
        self.reduce_op = reduce_op

        self.original_device = []
        self.original_datatype = []
        for data in data_list:
            self.original_device.append(data.device)
            self.original_datatype.append(data.dtype)

        if device is None:
            device = self.original_device
        elif isinstance(device, torch.device):
            device = [device] * len(data_list)
        else:
            assert isinstance(device, list)
            assert len(device) == len(data_list)
        self.device = device

        if datatype is None:
            datatype = self.original_datatype
        elif isinstance(datatype, torch.dtype):
            datatype = [datatype] * len(data_list)
        else:
            assert isinstance(datatype, list)
            assert len(datatype) == len(data_list)
        self.datatype = datatype

        self.device = device
        self.data_list = self._copy_tensor_list(data_list)
        self.buffer = self._init_buffer(data_list)

    def _init_buffer(self, data_list):
        buffer = []
        for i, data in enumerate(data_list):
            buffer.append(torch.zeros_like(data).to(self.device[i]).to(self.datatype[i]))
        return buffer

    def _copy_tensor_list(self, data_list):
        copy_list = []
        for i, data in enumerate(data_list):
            copy_list.append(data.detach().clone().to(self.device[i]).to(self.datatype[i]))
        return copy_list

    def send(self, ring_idx, offset, nelem):
        next_ring_idx = self.ring.next(ring_idx)
        self.buffer[next_ring_idx][offset : offset + nelem] = self.data_list[ring_idx][offset : offset + nelem]

    def recv_reduce_send(self, ring_idx, offset, nelem):
        temp = self.reduce_op(
            self.data_list[ring_idx][offset : offset + nelem], self.buffer[ring_idx][offset : offset + nelem]
        )
        next_ring_idx = self.ring.next(ring_idx)
        self.buffer[next_ring_idx][offset : offset + nelem] = temp

    def recv_reduce_copy(self, ring_idx, offset, nelem):
        temp = self.reduce_op(
            self.data_list[ring_idx][offset : offset + nelem], self.buffer[ring_idx][offset : offset + nelem]
        )
        self.data_list[ring_idx][offset : offset + nelem] = temp

    def direct_recv_reduce_copy_send(self, ring_idx, offset, nelem):
        temp = self.reduce_op(
            self.data_list[ring_idx][offset : offset + nelem], self.buffer[ring_idx][offset : offset + nelem]
        )
        next_ring_idx = self.ring.next(ring_idx)
        self.buffer[next_ring_idx][offset : offset + nelem] = temp
        self.data_list[ring_idx][offset : offset + nelem] = temp

    def direct_recv_copy_send(self, ring_idx, offset, nelem):
        temp = self.buffer[ring_idx][offset : offset + nelem]
        next_ring_idx = self.ring.next(ring_idx)
        self.buffer[next_ring_idx][offset : offset + nelem] = temp
        self.data_list[ring_idx][offset : offset + nelem] = temp

    def direct_recv(self, ring_idx, offset, nelem):
        temp = self.buffer[ring_idx][offset : offset + nelem]
        self.data_list[ring_idx][offset : offset + nelem] = temp

    def convert_to_original_device_and_datatype(self):
        results = []
        for i, data in enumerate(self.data_list):
            results.append(data.to(self.original_device[i]).to(self.original_datatype[i]))
        return results


class TreePrimitive:
    def __init__(self, data_list, tree, reduce_op, device):
        self.tree = tree
        self.reduce_op = reduce_op
        self.device = device

        self.data_list = [self._copy_tensor_list(data_list), self._copy_tensor_list(data_list)]
        self.buffer = [self._init_buffer(data_list), self._init_buffer(data_list)]

    def _init_buffer(self, data_list):
        buffer = []
        for data in data_list:
            buffer.append(torch.zeros_like(data).to(self.device))
        return buffer

    def _copy_tensor_list(self, data_list):
        copy_list = []
        for data in data_list:
            copy_list.append(data.detach().clone().to(self.device))
        return copy_list

    def send(self, rank, tree_idx, offset, nelem):
        self.buffer[tree_idx][rank][offset : offset + nelem] = self.data_list[tree_idx][rank][offset : offset + nelem]

    def recv_reduce_send(self, rank, tree_idx, offset, nelem):
        temp = []
        temp.append(self.data_list[tree_idx][rank][offset : offset + nelem])
        for d in self.tree.tree[tree_idx][rank].down:
            if d != -1:
                temp.append(self.buffer[tree_idx][d][offset : offset + nelem])
        self.buffer[tree_idx][rank][offset : offset + nelem] = self.reduce_op(temp)

    def recv_reduce_copy(self, rank, tree_idx, offset, nelem):
        temp = []
        temp.append(self.data_list[tree_idx][rank][offset : offset + nelem])
        for d in self.tree.tree[tree_idx][rank].down:
            if d != -1:
                temp.append(self.buffer[tree_idx][d][offset : offset + nelem])
        self.data_list[tree_idx][rank][offset : offset + nelem] = self.reduce_op(temp)

    def direct_send_from_output(self, rank, tree_idx, offset, nelem):
        self.buffer[tree_idx][rank][offset : offset + nelem] = self.data_list[tree_idx][rank][offset : offset + nelem]

    def direct_recv(self, rank, tree_idx, offset, nelem):
        u = self.tree.tree[tree_idx][rank].up
        temp = self.buffer[tree_idx][u][offset : offset + nelem]
        self.data_list[tree_idx][rank][offset : offset + nelem] = temp

    def direct_recv_copy_send(self, rank, tree_idx, offset, nelem):
        u = self.tree.tree[tree_idx][rank].up
        temp = self.buffer[tree_idx][u][offset : offset + nelem]
        self.data_list[tree_idx][rank][offset : offset + nelem] = temp
        self.buffer[tree_idx][rank][offset : offset + nelem] = temp


class Point2PointPrimitive:
    def __init__(self, data_list, ranks, device=None, datatype=None):
        self.ranks = ranks

        self.original_device = []
        self.original_datatype = []
        for data in data_list:
            self.original_device.append(data.device)
            self.original_datatype.append(data.dtype)

        if device is None:
            device = self.original_device
        elif isinstance(device, torch.device):
            device = [device] * len(data_list)
        else:
            assert isinstance(device, list)
            assert len(device) == len(data_list)
        self.device = device

        if datatype is None:
            datatype = self.original_datatype
        elif isinstance(datatype, torch.dtype):
            datatype = [datatype] * len(data_list)
        else:
            assert isinstance(datatype, list)
            assert len(datatype) == len(data_list)
        self.datatype = datatype

        self.device = device
        self.data_list = self._copy_tensor_list(data_list)
        self.buffer = self._init_buffer(data_list)

    def _init_buffer(self, data_list):
        buffer = []
        for i, data in enumerate(data_list):
            buffer.append(torch.zeros_like(data).to(self.device[i]).to(self.datatype[i]))
        return buffer

    def _copy_tensor_list(self, data_list):
        copy_list = []
        for i, data in enumerate(data_list):
            copy_list.append(data.detach().clone().to(self.device[i]).to(self.datatype[i]))
        return copy_list

    def send(self, send_rank, send_offset, nelem, dtype, peer_rank, peer_offset=None):
        if peer_offset is None:
            peer_offset = send_offset
        temp = self.data_list[send_rank][send_offset : send_offset + nelem]
        self.buffer[peer_rank][peer_offset : peer_offset + nelem] = temp.to(dtype)

    def recv(self, recv_rank, recv_offset, nelem, dtype, peer_rank, peer_offset=None):
        if peer_offset is None:
            peer_offset = recv_offset
        temp = self.buffer[recv_rank][recv_offset : recv_offset + nelem].to(dtype)
        self.data_list[recv_rank][recv_offset : recv_offset + nelem] = temp

    def convert_to_original_device_and_datatype(self):
        results = []
        for i, data in enumerate(self.data_list):
            results.append(data.to(self.original_device[i]).to(self.original_datatype[i]))
        return results
