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

import itertools
from vescale.emulator.nccl.constants import *  # noqa: F403
from vescale.emulator.nccl.include.graph import (
    NCCL_TOPO_PATTERN_SPLIT_TREE,
    NCCL_TOPO_PATTERN_TREE,
)


class Ring:
    def __init__(self, ranks):
        self.ranks = ranks
        self.nranks = len(self.ranks)

    def prev(self, ring_idx):
        return self.mod_rank(ring_idx + self.nranks - 1)

    def next(self, ring_idx):
        return self.mod_rank(ring_idx + 1)

    def mod_rank(self, r):
        if r >= self.nranks:
            return r - self.nranks
        else:
            return r


def global_rank_to_group_rank(global_ranks, mapping):
    result = []
    if isinstance(global_ranks, list):
        for rank in global_ranks:
            result.append(mapping[rank])
    else:
        result = mapping[global_ranks]
    return result


def filter_tree_structure(tree_structure, selected_ranks, mapping):
    result = []
    for server in tree_structure:
        filtered_server = [global_rank_to_group_rank(gpu, mapping) for gpu in server if gpu in selected_ranks]
        if filtered_server:
            result.append(filtered_server)
    return result


class TreeNode:
    def __init__(self, rank, up=-1, down0=-1, down1=-1, down2=-1):
        self.rank = rank
        self.up = up
        self.down = [down0, down1, down2]

    def update(self, up=None, down0=None, down1=None, down2=None):
        if up is not None:
            self.up = up
        if down0 is not None:
            self.down[0] = down0
        if down1 is not None:
            self.down[1] = down1
        if down2 is not None:
            self.down[2] = down2

    def __repr__(self):
        return f"[Rank {self.rank}] up: {self.up}, down: {self.down}.\n"


class DoubleTree:
    def __init__(self, tree_structure, ranks, mapping, pattern=NCCL_TOPO_PATTERN_SPLIT_TREE, ntrees=2):
        self.device_topo = filter_tree_structure(tree_structure, ranks, mapping)
        self.nranks = len(ranks)
        self.pattern = pattern

        # initialize all nodes
        self.tree = []
        for tree_idx in range(ntrees):
            self.tree.append([])
            for i in itertools.chain.from_iterable(self.device_topo):
                self.tree[tree_idx].append(TreeNode(rank=i))

        # create intra node chains
        for tree_idx in range(ntrees):
            self.get_intra_node_chains(self.device_topo, tree_idx)

        # create inter node trees
        self.get_double_tree(self.device_topo, 0, 1)

    def get_intra_node_chains(self, device_topo, tree_idx):
        for node in range(len(device_topo)):
            for i, local_rank in enumerate(device_topo[node]):
                up = None
                down0 = None
                if i == 0:
                    up = None
                else:
                    up = device_topo[node][i - 1]
                if i == len(device_topo[node]) - 1:
                    down0 = None
                else:
                    down0 = device_topo[node][i + 1]
                self.tree[tree_idx][local_rank].update(up=up, down0=down0)

    def get_binary_tree(self, device_topo, node_mask_func=None, node_mask_reverse_func=None, tree_idx=0):
        # check if device_topo is a 2D list
        nnodes = len(device_topo)
        nodes_list = list(range(nnodes))

        def get_send_rank(node):
            if node < 0:
                return node
            if self.pattern == NCCL_TOPO_PATTERN_SPLIT_TREE or self.pattern == NCCL_TOPO_PATTERN_TREE:
                return device_topo[node][0]

        def get_recv_rank(node, parentChildType=0):
            if node < 0:
                return node
            if self.pattern == NCCL_TOPO_PATTERN_SPLIT_TREE:
                assert (
                    len(device_topo[node]) > 1
                ), "NCCL_TOPO_PATTERN_SPLIT_TREE requires each node has at least two local ranks"
                return device_topo[node][1]
            if self.pattern == NCCL_TOPO_PATTERN_TREE:
                return device_topo[node][0]
            # if self.pattern == NCCL_TOPO_PATTERN_BALANCED_TREE:
            #     assert len(device_topo[node]) > 1, "NCCL_TOPO_PATTERN_BALANCED_TREE requires each node has at least two local ranks"
            #     return device_topo[node][parentChildType]

        for node in nodes_list:
            if node_mask_func is not None:
                node = node_mask_func(node)
            bit = 1
            while bit < nnodes:
                if bit & node:
                    break
                bit <<= 1
            if node == 0:
                u = -1
                d0 = -1
                d1 = bit >> 1 if nnodes > 1 else -1
                if d1 != -1:
                    if node_mask_reverse_func is not None:
                        node = node_mask_reverse_func(node)
                        d1 = node_mask_reverse_func(d1)
                    self.tree[tree_idx][get_recv_rank(node)].update(down2=get_send_rank(d1))
                continue

            up = (node ^ bit) | (bit << 1)
            if up >= nnodes:
                up = node ^ bit
            parentChildType = 0 if node < up else 1
            u = up

            lowbit = bit >> 1
            down0 = -1 if lowbit == 0 else node - lowbit
            down1 = -1 if lowbit == 0 else node + lowbit
            while down1 >= nnodes:
                down1 = -1 if lowbit == 0 else node + lowbit
                lowbit >>= 1

            if node_mask_reverse_func is not None:
                node = node_mask_reverse_func(node)
                u = node_mask_reverse_func(u)
                down0 = node_mask_reverse_func(down0)
                down1 = node_mask_reverse_func(down1)
            self.tree[tree_idx][get_send_rank(node)].update(up=get_recv_rank(u))
            self.tree[tree_idx][get_recv_rank(node)].update(down1=get_send_rank(down0), down2=get_send_rank(down1))

    def get_double_tree(self, device_topo, tree_idx_0, tree_idx_1):
        nnodes = len(device_topo)
        self.get_binary_tree(device_topo=device_topo, tree_idx=tree_idx_0)

        if nnodes % 2 == 1:
            # shift
            def node_mask_func(node):
                return (node - 1 + nnodes) % nnodes

            def node_mask_reverse_func(node):
                if node == -1:
                    return -1
                else:
                    return (node + 1) % nnodes

        else:
            # mirror
            def node_mask_func(node):
                return nnodes - 1 - node

            def node_mask_reverse_func(node):
                if node == -1:
                    return -1
                else:
                    return nnodes - 1 - node

        self.get_binary_tree(
            device_topo=device_topo,
            node_mask_func=node_mask_func,
            node_mask_reverse_func=node_mask_reverse_func,
            tree_idx=tree_idx_1,
        )
