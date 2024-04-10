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

from abc import abstractmethod
from tqdm import tqdm
from typing import Dict
import os
import re
import torch
import torch.distributed as dist  # current we need to use mpi launch
from vescale import DeviceMesh, DTensor
from .checkpoint_format import LLMHandWriteFormat
from typing import Optional, List, Any
from torch.distributed.distributed_c10d import (
    ProcessGroup,
    get_rank,
    get_world_size,
)
from torch.distributed.checkpoint._nested_dict import flatten_state_dict, unflatten_state_dict
from torch.distributed.checkpoint.metadata import (
    STATE_DICT_TYPE,
)

from ..utilities.bfile import listdir, BFile


def _construct_megatron_downloading_map(filenames: List[str]):
    weight_dic_pattern = r"mp_rank_\d\d_\d\d\d$"
    filtered_files = [file for file in filenames if re.match(weight_dic_pattern, file)]

    download_map = {}
    for file in filtered_files:
        parts = file.split("_")
        tp_rank = int(parts[2])
        pp_rank = int(parts[3])
        if pp_rank not in download_map:
            download_map[pp_rank] = {}
        download_map[pp_rank][tp_rank] = file

    return download_map


def _construct_reverse_pp_tp_map(vescale_path: str):
    if not os.path.exists(vescale_path):
        raise RuntimeError(f"vescale_path not exists. path: {vescale_path}")
    files = os.listdir(vescale_path)
    match = r"rank\d+.pt"

    filtered_files = [file for file in files if re.match(match, file)]
    rank_map = {}
    for file in filtered_files:
        rank = re.search(r"\d+", file).group(0)
        rank_map[rank] = os.path.join(vescale_path, file)
    return rank_map


def _construct_pp_tp_map(megatron_path: str):
    """
    construct tp pp index mapping dict
    {
        # for pp 0
        0: {
            # for tp 0
            0 : "xxx.pt",
            1 : "xxx.pt"
        }
    }
    """
    dics = listdir(megatron_path)
    if len(dics) == 0:
        raise RuntimeError(f"megatron_path not exists or is empty. path: {megatron_path}")

    weight_map = dict()
    optim_map = dict()

    def update_dict(dic_, pp_r, tp_r, file_path):
        if pp_r in dic_:
            pp_dic = dic_[pp_r]
            pp_dic.update({tp_r: file_path})
        else:
            new_dic = {tp_r: file_path}
            dic_[pp_r] = new_dic

    weight_dict = r"mp_rank_\d\d_\d\d\d$"
    optim_dict = r"mp_rank_\d\d_\d\d\d_\d\d\d$"
    filtered_weights_dics = [dic for dic in dics if re.match(weight_dict, dic)]
    filtered_optim_dics = [dic for dic in dics if re.match(optim_dict, dic)]

    # construct weight 2-dims maps
    for dic in filtered_weights_dics:
        split_ul = re.split("_", dic)
        tp_rank = int(split_ul[2])
        pp_rank = int(split_ul[3])
        weight_file = os.path.join(megatron_path, dic, "model_rng.pt")
        update_dict(weight_map, pp_rank, tp_rank, weight_file)

    # construct optimize 2-dims maps
    for dic in filtered_optim_dics:
        split_ul = re.split("_", dic)
        tp_rank = int(split_ul[2])
        pp_rank = int(split_ul[3])
        optim_file = os.path.join(megatron_path, dic, "optim.pt")
        update_dict(optim_map, pp_rank, tp_rank, optim_file)
    return weight_map, optim_map


def _get_megatron_tp_group(world_size, pp_size, tp_size, dp_size, cur_rank) -> tuple[ProcessGroup, ProcessGroup]:
    """make sub pg group"""
    return dist.new_subgroups(group_size=tp_size * dp_size)


def _deduce_parallel_plan_by_device_mesh(mesh: DeviceMesh):
    """make rank to megatron tp_rank, pp_rank map"""
    # FIXME(cery.69) : current only support data parallel is 1
    # allways parallel in last dim
    tp_size = mesh.size()
    # for rank = pp_rank * tp_size + tp_rank
    # (rank - tp_rank) / tp_size  = pp_rank
    tp_rank = get_rank() % tp_size
    assert (get_rank() - tp_rank) % tp_size == 0, "megatron not support pp size undivided by tp size"
    pp_rank = (get_rank() - tp_rank) // tp_size
    return tp_rank, pp_rank


def _filter_unused_tensors_and_renaming(old_state_dict: Dict[str, Any], param_resharding_plan: Dict[str, Any]):
    new_state_dict = {}

    flatten_old_st, _ = flatten_state_dict(old_state_dict)

    for key, value in flatten_old_st.items():
        for pattern in param_resharding_plan.keys():
            start_index = key.find(pattern)
            if start_index == -1:
                continue
            else:
                new_state_dict[pattern] = value
    print(new_state_dict.keys())
    return new_state_dict


##################################################################
#####################     for visitor        #####################
##################################################################


class StateDictVisitor:
    def set_device_mesh(self, mesh: DeviceMesh):
        self.device_mesh = mesh

    @abstractmethod
    def parsing_state_dict(self, st: dict, *args, **kwargs):
        """
        flattened parsing module dict, using process function to handle each Tensor
        """
        f_st, mapping = flatten_state_dict(st)
        # flattened_key , value
        for key, value in tqdm(f_st.items()):
            if isinstance(value, (torch.Tensor, DTensor)):
                self.tensor_process_func(f_st, key, value, *args, **kwargs)
        new_st = unflatten_state_dict(f_st, mapping)
        st.update(new_st)

    @abstractmethod
    def tensor_process_func(self, parent: dict, key: str, value: Any, *args, **kwargs):
        raise NotImplementedError("method abstruct method is call")

    @abstractmethod
    def apply(self, state_dict: dict, *args, **kwargs):
        self.parsing_state_dict(state_dict, *args, **kwargs)


class DefaultM2VDFSVisitor(StateDictVisitor):
    def __init__(self, format: LLMHandWriteFormat):
        self.format = format
        super().__init__()

    def tensor_process_func(self, parent: dict, key: str, value: Any, *args, **kwargs):
        assert self.format is not None, "format is not set"
        tensor_placement = self.format.get_tensor_sharding_plan_by_name(key)
        assert isinstance(value, torch.Tensor)

        is_requires_grad = value.requires_grad
        with torch.no_grad():  # keep out of autograd
            dtensor = DTensor.from_local(value, self.device_mesh, tensor_placement)
        dtensor.requires_grad_(is_requires_grad)

        parent[key] = dtensor

    def apply(self, state_dict: dict, *args, **kwargs):
        self.parsing_state_dict(state_dict, *args, **kwargs)


class DefaultV2MDFSVisitor(StateDictVisitor):
    def __init__(self):
        super().__init__()

    def tensor_process_func(self, parent: dict, key: str, value: DTensor, *args, **kwargs):
        parent[key] = value._local_tensor  # keep out of autograd

    def apply(self, state_dict: dict, *args, **kwargs):
        self.parsing_state_dict(state_dict, *args, **kwargs)


##################################################################
#####################     for api func       #####################
##################################################################


def convert_vescale_checkpoint_to_megatron(
    vescale_path: str, megatron_path: str, visitor: StateDictVisitor, device=torch.device("cpu")
) -> STATE_DICT_TYPE:
    rank_map = _construct_reverse_pp_tp_map(vescale_path)
    world_size = len(rank_map)
    assert world_size == get_world_size(), f"world size mismatch {world_size} vs {get_world_size()}"
    rank = get_rank()
    rank_file_name = rank_map[str(rank)]
    rank_file_path = os.path.join(vescale_path, rank_file_name)
    if os.path.exists(rank_file_path):
        st = torch.load(rank_file_path, map_location=device)

        def find_device_mesh(st):
            for key in st:
                value = st[key]
                if isinstance(value, DTensor):
                    mesh = value.device_mesh
                    return mesh
                elif isinstance(value, dict):
                    mesh = find_device_mesh(value)
                    if mesh:
                        return mesh
            return None

        device_mesh = find_device_mesh(st)
        assert device_mesh is not None, "not find devicemesh in vescale format please check"
        tp_rank, pp_rank = _deduce_parallel_plan_by_device_mesh(device_mesh)
        visitor.apply(st)
        megatron_dict = f"mp_rank_{str(tp_rank).zfill(2)}_{str(pp_rank).zfill(3)}"
        tmp_path = megatron_path
        megatron_save_path = os.path.join(tmp_path, megatron_dict)
        os.makedirs(megatron_save_path, exist_ok=True)
        megatron_save_file = os.path.join(megatron_save_path, "model_rng.pt")
        if "optim" in st:
            optim = st["optim"]
            megatron_optim_dict = f"mp_rank_{str(tp_rank).zfill(2)}_{str(pp_rank).zfill(3)}_000"
            megatron_optim_dict_path = os.path.join(tmp_path, megatron_optim_dict)
            os.makedirs(megatron_optim_dict_path, exist_ok=True)
            torch.save(optim, os.path.join(megatron_optim_dict_path, "optim.pt"))
            del st["optim"]
        torch.save(st, megatron_save_file)
        # FIXME(cery.69): support dp not 1
        return st


def convert_megatron_checkpoint_to_vescale(
    megatron_path: str, visitor: DefaultM2VDFSVisitor, device=torch.device("cpu"), vescale_path: Optional[str] = None
) -> STATE_DICT_TYPE:
    weight_map, optim_map = _construct_pp_tp_map(megatron_path)
    tp_equal = [(len(weight_map[pp]) == len(weight_map[0])) for pp in weight_map]
    assert all(tp_equal), "megatron not support unmodified devided split plan"
    tp_size = len(weight_map[0])
    pp_size = len(weight_map)

    rank = get_rank()

    for pp_rank in range(0, pp_size):
        for tp_rank in range(0, tp_size):
            megatron_rank = pp_rank * tp_size + tp_rank
            if megatron_rank != rank:
                continue
            megatron_weight_pt = weight_map[pp_rank][tp_rank]
            # phase 1. parse weight
            with BFile(megatron_weight_pt, "rb") as f:
                m_st = torch.load(f, map_location=device)
                args = m_st["args"]
                megatron_cur_rank = args.rank
                megatron_world_size = args.world_size
                megatron_tp_size = args.tensor_model_parallel_size
                megatron_pp_size = args.pipeline_model_parallel_size
                megatron_dp_size = args.data_parallel_size

            local_pg, _ = _get_megatron_tp_group(
                megatron_world_size, megatron_pp_size, megatron_tp_size, megatron_dp_size, megatron_cur_rank
            )
            device_mesh = DeviceMesh(device.__str__(), None, pg=local_pg)
            visitor.set_device_mesh(device_mesh)
            visitor.apply(m_st["model"], "model")

            new_st = {}
            new_st["models"] = _filter_unused_tensors_and_renaming(
                m_st["model"], visitor.format.default_params_sharding_plan
            )
            if len(optim_map) > 0:
                megatron_optim_pt_path = optim_map[pp_rank][tp_rank]
                # phase 2. parse optimizer
                with BFile(megatron_optim_pt_path, "rb") as f:
                    optim = torch.load(f, map_location=device)
                    visitor.apply(optim, "")
                    new_st["optim"] = optim
            if vescale_path:
                save_file = f"rank{rank}.pt"
                with BFile(os.path.join(vescale_path, save_file), "wb") as f:
                    torch.save(new_st, f)
            return new_st
