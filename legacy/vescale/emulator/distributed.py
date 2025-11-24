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
# Some code comes from distributed_c10d.py in PyTorch
# Original license:
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################

import hashlib
import os
from typing import Dict, List, Optional
import torch
import torch.distributed
from torch.distributed.distributed_c10d import ProcessGroup as torchProcessGroup
from vescale.emulator.nccl.constants import NCCL_ALGO_RING, NCCL_ALGO_TREE, WARP_SIZE, NcclFunc
from vescale.emulator.reduce_kernel import ReduceOp
from vescale.emulator.all_reduce import run_ring_all_reduce, run_tree_all_reduce
from vescale.emulator.all_gather import run_ring_all_gather
from vescale.emulator.reduce_scatter import run_ring_reduce_scatter
from vescale.emulator.all_to_all import run_all_to_all

from vescale.emulator.utils import flatten_tensors, restore_tensors
from vescale.emulator.nccl.include.graph import NCCL_TOPO_PATTERN_TREE
from vescale.emulator.calculate_chunk_size import get_info_nchannels_nthreads_proto


RANK = -1


def set_rank(rank):
    global RANK
    RANK = rank


class ProcessGroup:
    """
    A class for the emulator ProcessGroup.
    """

    def __init__(self, store, rank, size, backend="nccl"):
        self._size = size
        if backend == "nccl":
            self._backend = backend
        else:
            raise NotImplementedError(f"backend {backend} is not supported")

    def rank(self):
        global RANK
        return RANK

    def size(self):
        return self._size

    def _set_group_name(self, group_name):
        self.group_name = group_name

    def get_nccl_graph_xml(self):
        return get_nccl_graph_xml(self)

    def get_from_group(self):
        ranks = _world.pg_group_ranks[self].keys()
        if self._backend == "nccl":
            device = torch.device("cuda")
        else:
            device = torch.device("cuda")
        nnodes = 1
        return ranks, device, nnodes

    def all_reduce(self, tensors, op=ReduceOp.SUM, tree_structure=None):
        """
        Reduces the tensor data across all tensors in the list in a way that all get the final result.

        After the call each tensor in the tensor list ``tensors`` is going to be bitwise identical.

        Args:
            tensors (List[Tensor]): Input and output of the collective. The function
                operates in-place.
            op (optional): One of the values from
                ``vescale.emulator.reduce_kernel.ReduceOp``.
                Specifies an operation used for element-wise reductions.
            tree_structure (optional): A list of lists of ranks.
                The first list represents the nodes in the cluster, and the second list is the devices of the nodes.

        Returns:
            None.
        """
        ranks, device, nnodes = self.get_from_group()
        flattenend_tensors, original_shapes = flatten_tensors(tensors)
        info, nchannels, nthreads, protocol = get_info_nchannels_nthreads_proto(
            self,
            NcclFunc.ncclFuncAllReduce,
            len(flattenend_tensors[0]),
            flattenend_tensors[0].dtype,
            len(ranks),
            nnodes,
        )
        nwarps = int(nthreads / WARP_SIZE)
        algo = info.algorithm
        if algo == NCCL_ALGO_RING:
            results = run_ring_all_reduce(
                info,
                nchannels,
                nwarps,
                protocol,
                flattenend_tensors,
                ranks,
                device,
                flattenend_tensors[0].size()[0],
                0,
                op,
            )
        elif algo == NCCL_ALGO_TREE:
            mode = NCCL_TOPO_PATTERN_TREE
            results = run_tree_all_reduce(
                info,
                nchannels,
                nwarps,
                protocol,
                flattenend_tensors,
                tree_structure,
                ranks,
                _world.pg_group_ranks[self],
                mode,
                device,
                flattenend_tensors[0].size()[0],
                0,
                op,
            )
        results = restore_tensors(results, original_shapes)
        for i in range(len(tensors)):
            tensors[i] = results[i]

    def all_gather(self, tensors_list, tensors, async_op=False):
        """
        Gathers tensors in the list and return a list of list of tensors,
        which represents the gathered result on each rank.

        Args:
            tensors_list (List[List[Tensor]]): Output list. The length is equal
                to group size and each element in the first list is the gathered
                result (List[Tensor]) of the corresponding rank.
            tensors (List[Tensor]): Tensor to be broadcast on each rank.
            async_op (bool, optional): Whether this op should be an async op

        Returns:
            None.
        """
        ranks, device, nnodes = self.get_from_group()
        tensors, original_shapes = flatten_tensors(tensors)
        results = run_ring_all_gather(
            tensors, ranks, device, max(tensors[0].size()[0] // len(ranks), 1), tensors[0].size()[0], 0
        )
        new_shape_list = []
        for shape in original_shapes:
            new_shape = []
            new_shape.append(len(ranks))
            for s in shape:
                new_shape.append(s)
            new_shape_list.append(new_shape)
        results = restore_tensors(results, new_shape_list)
        for i in range(len(tensors_list)):
            tensors_list[i] = [t.squeeze(0) for t in torch.split(results[i], 1)]

    def reduce_scatter(self, outputs, tensors_list, op=ReduceOp.SUM):
        """
        Reduces, then scatters a list of list of tensors to all ranks in a group.

        Args:
            outputs (List[Tensor]): Output list. The length is equal
                to group size and each element in the first list is the scattered tensor
                result (Tensor) of the corresponding rank.
            tensors_list (List[List[Tensor]]): List of list of tensors. The first list
                represents the data on each rank. The second list is the list of tensors
                to reduce and scatter.
            op (optional): One of the values from
                ``vescale.emulator.reduce_kernel.ReduceOp``.
                Specifies an operation used for element-wise reductions.

        Returns:
            None.

        """
        ranks, device, nnodes = self.get_from_group()
        tensors = [torch.stack(tensor_list) for tensor_list in tensors_list]
        tensor_list, original_shapes = flatten_tensors(tensors)
        info, nchannels, nthreads, protocol = get_info_nchannels_nthreads_proto(
            self, NcclFunc.ncclFuncAllReduce, len(tensor_list[0]), tensor_list[0].dtype, len(ranks), nnodes
        )
        nwarps = int(nthreads / WARP_SIZE)
        results = run_ring_reduce_scatter(
            info,
            nchannels,
            nwarps,
            protocol,
            tensor_list,
            ranks,
            device,
            tensor_list[0].size()[0] // len(ranks),
            tensor_list[0].size()[0] // len(ranks),
            0,
            op,
        )
        new_shape_list = []
        for shape in original_shapes:
            new_shape = []
            for i, s in enumerate(shape):
                if i == 0:
                    pass
                else:
                    new_shape.append(s)
            new_shape_list.append(new_shape)
        results = restore_tensors(results, new_shape_list)
        for i in range(len(outputs)):
            outputs[i] = results[i]

    def all_to_all(self, outputs_list, tensors_list, datatype=torch.float32, async_op=False):
        """
        Scatters list of input tensors to all processes in a group and return gathered list of tensors in output list.

        Args:
            outputs_list (List[List[Tensor]]): List of list of tensors to be gathered. One list
                per rank.
            tensors_list (List[List[Tensor]]): List of list of tensors to scatter. The first list
                represents the data on each rank. The second list is the list of tensors to scatter one per rank.
            datatype (torch.dtype): Data type of the input and output tensors.
            async_op (bool, optional): Whether this op should be an async op.
        """
        ranks, device, nnodes = self.get_from_group()
        tensors_list = [torch.stack(tensor_list) for tensor_list in tensors_list]

        tensor_list, original_shapes = flatten_tensors(tensors_list)
        results = run_all_to_all(
            tensor_list,
            ranks,
            device,
            datatype,
            tensor_list[0].size()[0] // len(ranks),
            tensor_list[0].size()[0] // len(ranks),
            0,
        )
        results = restore_tensors(results, original_shapes)
        for i in range(len(outputs_list)):
            for j in range(len(outputs_list[i])):
                outputs_list[i][j] = results[i][j]


def get_world_size(group: Optional[ProcessGroup] = None) -> int:
    """
    Return the number of processes in the current process group.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.

    Returns:
        The world size of the process group\
    """
    return _get_group_size(group)


# DO NOT USE THESE FIELDS DIRECTLY.
# Use them through the _world object to make sure the _world override mechanism
_pg_names: Dict[ProcessGroup, str] = {}
_pg_group_ranks: Dict[ProcessGroup, Dict[int, int]] = {}  # key: global ranks, value: group ranks
_group_count = 0
_tags_to_pg: Dict[str, List[ProcessGroup]] = {}
_pg_to_tag: Dict[ProcessGroup, str] = {}
_pg_to_xml: Dict[ProcessGroup, str] = {}  # key: ProcessGroup, value: xml file path


class _World:
    """
    Container class for emulator process group state.

    This is used during registration and lookup of PG state.
    """

    def __init__(self):
        self._default_pg = None

    @property
    def default_pg(self):
        """
        Process group that includes all ranks of the cluster.

        This default ProcessGroup is used by emulator APIs when a ProcessGroup is needed
        but None is provided.
        """
        return self._default_pg

    @default_pg.setter
    def default_pg(self, value):
        self._default_pg = value

    @property
    def tags_to_pg(self) -> Dict[str, List[ProcessGroup]]:
        global _tags_to_pg
        return _tags_to_pg

    @property
    def pg_to_tag(self) -> Dict[ProcessGroup, str]:
        global _pg_to_tag
        return _pg_to_tag

    @property
    def pg_group_ranks(self) -> Dict[ProcessGroup, Dict[int, int]]:
        """
        Process group's global rank to local rank mapping.

        TODO don't expose the map, expose fine grained ops
        """
        global _pg_group_ranks
        return _pg_group_ranks

    @property
    def pg_names(self) -> Dict[ProcessGroup, str]:
        """
        Process group's names, map from ProcessGroup to str.

        TODO don't expose the map, expose fine grained ops
        """
        global _pg_names
        return _pg_names

    @property
    def group_count(self) -> int:
        """
        Process group count for default naming.

        TODO don't expose group_count, use something else instead
        """
        global _group_count
        return _group_count

    @group_count.setter
    def group_count(self, value):
        """Use to compute the name of ProcessGroups when using global synchronization."""
        global _group_count
        _group_count = value

    @property
    def pg_to_xml(self) -> Dict[ProcessGroup, str]:
        """
        Process group's nccl graph xml file path, map from ProcessGroup to str.

        TODO don't expose the pg_to_xml, use something else instead
        """
        global _pg_to_xml
        return _pg_to_xml


_world = _World()
"""Holds the singleton instance of ``_World`` used by emulator. Experimental extension point to override it"""


class _WorldMeta(type):
    """
    Meta class of ``group`` and ``GroupMember``.

    Allows them to have the class property ``WORLD``.
    """

    @property
    def WORLD(cls) -> Optional[ProcessGroup]:
        return _world.default_pg

    @WORLD.setter
    def WORLD(cls, pg: Optional[ProcessGroup]):
        _world.default_pg = pg


class GroupMember(metaclass=_WorldMeta):
    """Group member class."""

    NON_GROUP_MEMBER = -100


def _find_pg_by_ranks_and_tag(tag: str, ranks: List[int]) -> ProcessGroup:
    if len(tag) > 0 and not tag.startswith("ptd:") and not tag.startswith("user:"):
        tag = f"user:{tag}"

    for group in _world.tags_to_pg.get(tag, []):
        if group.size() != len(ranks):
            continue

        group_ranks = get_process_group_ranks(group)
        good = all(r in group_ranks for r in ranks)
        if good:
            return group
    return None


def is_initialized() -> bool:
    """Check if the default process group has been initialized."""
    return GroupMember.WORLD is not None


def _get_default_group():
    """Get the default process group created by init_process_group."""
    if not is_initialized():
        raise ValueError(
            "Default process group has not been initialized, " "please make sure to call init_process_group."
        )
    return GroupMember.WORLD


def _get_group_size(group):
    """Get a given group's world size."""
    if group is GroupMember.WORLD or group is None:
        default_pg = _get_default_group()
        return default_pg.size()
    return group.size()


def _get_group_tag(pg: ProcessGroup) -> str:
    """Return the tag associated with ``pg``."""
    tag = _world.pg_to_tag[pg]
    if tag.startswith("user:"):
        tag = tag[5:]
    return tag


def get_process_group_ranks(group: ProcessGroup):
    """
    Get all ranks associated with ``group``.

    Args:
        group (ProcessGroup): ProcessGroup to get all ranks from.

    Returns:
        List of global ranks ordered by group rank.
    """
    return list(_world.pg_group_ranks[group].keys())


def _rank_not_in_group(group: ProcessGroup):
    """Check if the current process's rank is not in a given group."""
    if group is None:
        return False
    return group == GroupMember.NON_GROUP_MEMBER


def get_rank(group: Optional[ProcessGroup] = None) -> int:
    """
    Return the rank of the current process in the provided ``group``, default otherwise.

    Rank is a unique identifier assigned to each process within a distributed
    process group. They are always consecutive integers ranging from 0 to
    ``world_size``.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.

    Returns:
        The rank of the process group
        -1, if not part of the group

    """
    if _rank_not_in_group(group):
        return -1

    default_pg = _get_default_group()
    if group is None or group is GroupMember.WORLD:
        return default_pg.rank()

    return get_group_rank(group, default_pg.rank())


def get_group_rank(group: ProcessGroup, global_rank: int) -> int:
    """
    Translate a global rank into a group rank.

    ``global_rank`` must be part of ``group`` otherwise this raises RuntimeError.

    Args:
        group (ProcessGroup): ProcessGroup to find the relative rank.
        global_rank (int): Global rank to query.

    Returns:
        Group rank of ``global_rank`` relative to ``group``

    N.B. calling this function on the default process group returns identity
    """
    if group is GroupMember.WORLD:
        return global_rank
    if group not in _world.pg_group_ranks:
        raise ValueError(f"Group {group} is not registered, please create group with torch.distributed.new_group API")
    group_ranks = _world.pg_group_ranks[group]
    if global_rank not in group_ranks:
        raise ValueError(f"Global rank {global_rank} is not part of group {group}")

    return group_ranks[global_rank]


def new_group(ranks=None, timeout=None, backend=None, pg_options=None, use_local_synchronization=False):
    """
    Create a new emulator process group.

    Args:
        ranks (list[int]): List of ranks of group members. If ``None``, will be
            set to all ranks. Default is ``None``.
        timeout (timedelta, optional): see `init_process_group` for details and default value.
        backend (str or Backend, optional): The backend to use. Should be set to None.
        pg_options (ProcessGroupOptions, optional): process group options
            specifying what additional options need to be passed in during
            the construction of specific process groups. Should be set to None.
        use_local_synchronization (bool, optional): perform a group-local
            barrier at the end of the process group creation. Should be set to False

    Returns:
        A handle of emulator process group that can be given to collective calls.
    """
    return _new_group_with_tag(
        ranks, timeout, backend, pg_options, None, use_local_synchronization=use_local_synchronization
    )


def _new_group_with_tag(
    ranks=None, timeout=None, backend=None, pg_options=None, pg_tag=None, use_local_synchronization=False
):
    """
    Variant of ``new_group`` that exposes tag creation.
    """
    global _world

    default_pg = _get_default_group()
    global_world_size = default_pg.size()

    # checks the input ranks
    if ranks is not None:
        ranks = sorted(ranks)
        group_world_size = len(ranks)
        if group_world_size > global_world_size:
            raise ValueError(
                "the new group's world size should be less or " "equal to the world size set by " "init_process_group"
            )
        # check ranks' sanity
        for rank in ranks:
            if rank < 0 or rank >= global_world_size:
                raise ValueError("The new group's rank should be within " "the world_size set by init_process_group")
    else:
        ranks = list(range(global_world_size))
        group_world_size = global_world_size

    group_name = _process_group_name(ranks, use_hashed_name=use_local_synchronization)

    pg, _ = _new_process_group_helper(
        group_world_size, None, ranks, backend, None, group_name, pg_options=pg_options, timeout=timeout, pg_tag=pg_tag
    )

    # Create the global rank to group rank mapping
    _world.pg_group_ranks[pg] = {global_rank: group_rank for group_rank, global_rank in enumerate(ranks)}

    return pg


def _new_process_group_helper(
    group_size,
    group_rank,
    global_ranks_in_group,
    backend,
    store,
    group_name,
    pg_options=None,
    timeout=None,
    pg_tag=None,
):
    """
    Create a new emulator process group.

    This function is called with ``global_ranks_in_group == []`` for the default group.
    """
    global _world

    if group_name in _world.pg_names.values():
        raise ValueError("The specified group name has already been " "created, please use a different group name")

    if pg_tag not in [None, ""]:
        # creating with the same tag and rank set results in the same underlying PG
        existing_group = _find_pg_by_ranks_and_tag(pg_tag, global_ranks_in_group)
        if existing_group:
            return existing_group, None

    pg: ProcessGroup = ProcessGroup(None, group_rank, group_size, backend)

    # update global state
    assert group_name is not None
    _world.pg_names[pg] = group_name
    pg._set_group_name(group_name)

    # "" is the default tag for user PGs
    if pg_tag in [None, ""]:
        pg_tag = f"ptd:{group_name}"
        _world.tags_to_pg.setdefault("", []).append(pg)
    else:
        pg_tag = f"user:{pg_tag}"

    _world.tags_to_pg.setdefault(pg_tag, []).append(pg)
    _world.pg_to_tag[pg] = pg_tag
    return pg, None


# helper function for deterministically hashing a list of ranks
def _hash_ranks(ranks: List[int]):
    return hashlib.sha1(bytes("_".join(map(str, ranks)), "utf-8")).hexdigest()


def _process_group_name(ranks, use_hashed_name):
    global _world
    if use_hashed_name:
        pg_name = _hash_ranks(ranks)
        while pg_name in _world.pg_names.values():
            pg_name = hashlib.sha1(bytes(pg_name + "_", "utf-8")).hexdigest()
    else:
        pg_name = str(_world.group_count)
        _world.group_count += 1
    return pg_name


def _update_default_pg(pg):
    _world.default_pg = pg


def init_process_group(
    backend=None,
    init_method=None,
    timeout=None,
    world_size: int = -1,
    rank: int = -1,
    store=None,
    group_name: str = "",
    pg_options=None,
):
    """
    Initialize the default emulator process group.

    Args:
        backend (str or Backend, optional): The backend to use. Should be set to None
        init_method (str, optional): URL specifying how to initialize the
                                     process group. Should be set to None.
        world_size (int, optional): Number of processes participating in
                                    the job. Required if ``store`` is specified.
        rank (int, optional): Rank of the current process (it should be a
                              number between 0 and ``world_size``-1).
                              Required if ``store`` is specified.
        store(Store, optional): Key/value store accessible to all workers, used
                                to exchange connection/address information.
                                Should be set to None.
        timeout (timedelta, optional): Timeout for operations executed against
            the process group.

        group_name (str, optional, deprecated): Group name. This argument is ignored
        pg_options (ProcessGroupOptions, optional): process group options
            specifying what additional options need to be passed in during
            the construction of specific process groups. This argument is ignored
    """
    global _world

    if GroupMember.WORLD is not None:
        raise ValueError("trying to initialize the default process group twice!")

    assert (store is None) or (init_method is None), "Cannot specify both init_method and store."

    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"

    """
    Group name is not visible to users unless they access
    internals of c10d. This means we can ignore the value
    they provide as it not exposed in a public way.
    """
    group_name = _process_group_name([], use_hashed_name=False)

    default_pg, _ = _new_process_group_helper(
        world_size, rank, [], backend, store, group_name, pg_options=pg_options, timeout=timeout
    )
    _update_default_pg(default_pg)

    _world.pg_group_ranks[GroupMember.WORLD] = {i: i for i in range(GroupMember.WORLD.size())}  # type: ignore[attr-defined, index]


def destroy_process_group(group: Optional[ProcessGroup] = None):
    """
    Destroy a given process group, and deinitialize the distributed package.

    Args:
        group (ProcessGroup, optional): The process group to be destroyed, if
                                        group.WORLD is given, all process
                                        groups including the default one will
                                        be destroyed.
    """
    global _world

    if group is None:
        pg = GroupMember.WORLD
    else:
        pg = group

    if group is None or group == GroupMember.WORLD:
        _update_default_pg(None)
        _world.pg_names.clear()
        _world.pg_group_ranks.clear()
        _world.pg_to_tag.clear()
        _world.tags_to_pg.clear()
        _world.group_count = 0
    else:
        del _world.pg_names[pg]
        del _world.pg_group_ranks[pg]
        tag = _world.pg_to_tag.get(pg)
        del _world.pg_to_tag[pg]
        if tag is not None:
            try:
                _world.tags_to_pg[tag].remove(pg)
                if tag.startswith("ptd:"):
                    _world.tags_to_pg[""].remove(pg)
            except Exception:
                pass


def dump_nccl_graph(xmlfile="./ncclgraph.xml", pg=None, rank=0):
    """
    Dump NCCL graph by runing torch.distributed.all_reduce.

    Args:
        xmlfile (str, optional): The path to the xml file to dump the NCCL graph.
        pg (ProcessGroup, optional): The process group to dump the NCCL graph.
        rank (int, optional): The rank of the process to dump the NCCL graph.
    """
    original_NCCL_GRAPH_DUMP_FILE = os.environ.get("NCCL_GRAPH_DUMP_FILE", None)
    os.environ["NCCL_GRAPH_DUMP_FILE"] = xmlfile
    tensor = torch.rand(1024).cuda(rank)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, group=pg)
    if original_NCCL_GRAPH_DUMP_FILE is not None:
        os.environ["NCCL_GRAPH_DUMP_FILE"] = original_NCCL_GRAPH_DUMP_FILE
    else:
        del os.environ["NCCL_GRAPH_DUMP_FILE"]


def dump_nccl_graph_for_pg(emulator_pg: ProcessGroup, torch_pg: torchProcessGroup, rank):
    """
    Dump NCCL graph for a pair of given emulator process group and pytorch process group.

    Args:
        emulator_pg (ProcessGroup): The emulator process group to dump the NCCL graph.
        torch_pg (torch.distributed.ProcessGroup): The torch process group to dump the NCCL graph.
        rank (int): The rank of the process to dump the NCCL graph.
    """
    global _world
    ranks_mapping = _world.pg_group_ranks[emulator_pg]
    global_ranks = list(ranks_mapping.keys())
    global_ranks = sorted(global_ranks)
    xmlfile = f"ncclgraph_{'_'.join(map(str, global_ranks))}.xml"
    dump_nccl_graph(xmlfile, torch_pg, rank)


def delete_nccl_graph_for_pg(emulator_pg: ProcessGroup):
    """
    Delete NCCL graph for a given emulator process group.

    Args:
        emulator_pg (ProcessGroup): The emulator process group to delete the NCCL graph.
    """
    global _world
    ranks_mapping = _world.pg_group_ranks[emulator_pg]
    global_ranks = list(ranks_mapping.keys())
    global_ranks = sorted(global_ranks)
    xmlfile = f"ncclgraph_{'_'.join(map(str, global_ranks))}.xml"
    if os.path.exists(xmlfile):
        os.remove(xmlfile)


def get_nccl_graph_xml(pg=None):
    """
    Get the path to the NCCL graph xml file for a given process group.

    Args:
        pg (ProcessGroup, optional): The process group to get the NCCL graph xml file.

    Returns:
        str: The path to the NCCL graph xml file.
    """
    global _world
    if pg is None:
        pg = GroupMember.WORLD
    ranks_mapping = _world.pg_group_ranks[pg]
    global_ranks = list(ranks_mapping.keys())
    global_ranks = sorted(global_ranks)
    return f"ncclgraph_{'_'.join(map(str, global_ranks))}.xml"
