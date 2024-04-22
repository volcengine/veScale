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
import warnings
from torch.distributed import get_world_size, get_rank
from vescale.dtensor.device_mesh import init_device_mesh, DeviceMesh
from typing import Optional, List, Tuple, Union, Dict
from torch.distributed.distributed_c10d import ProcessGroup

__all__ = ["veDeviceMesh"]


class VeDeviceMesh:
    _MESH_DIM_NAMES_MAPPING: Dict[int, str] = {}
    _TENSOR_PARALLEL_SIZE: int = None
    _DATA_PARALLEL_SIZE: int = None
    _PIPELINE_PARALLEL_SIZE: int = None
    _DATA_PARALLEL_GROUP: ProcessGroup = None
    _TENSOR_PARALLEL_GROUP: ProcessGroup = None
    _GLOBAL_MESH: DeviceMesh = None
    _MESH_GRIDS: torch.Tensor = None
    _DATA_PARALLEL_MESH: DeviceMesh = None
    _TENSOR_PARALLEL_MESH: DeviceMesh = None
    _GLOBAL_PIPELINE_MODEL_PARALLEL_MESHES: List[DeviceMesh] = None
    _GLOBAL_TENSOR_PARALLEL_MESHES: List[DeviceMesh] = None
    _RANK_COORDINATE: List[int] = None
    DEFAULT_DEVICE_COUNT: int = (
        torch.cuda.device_count() if torch.cuda.is_available() else 8
    )  # enables 8 ranks for CPU multi-processing
    PP_DIM: int = 0

    def init_model_parallel(self, dp_size: int, tp_size: int):
        torch.distributed.init_process_group(backend="nccl", world_size=get_world_size(), rank=get_rank())

        num_tensor_parallel_groups = dp_size
        assert self._TENSOR_PARALLEL_GROUP is None, "tensor model parallel group is already initialized"
        for i in range(num_tensor_parallel_groups):
            ranks = range(i * tp_size, (i + 1) * tp_size)
            group = torch.distributed.new_group(ranks)
            if get_rank() in ranks:
                self._TENSOR_PARALLEL_GROUP = group

        num_data_parallel_groups = get_world_size() // dp_size
        assert self._DATA_PARALLEL_GROUP is None, "data parallel group is already initialized"
        for i in range(num_data_parallel_groups):
            ranks = range(i, get_world_size(), tp_size)
            group = torch.distributed.new_group(ranks)
            if get_rank() in ranks:
                self._DATA_PARALLEL_GROUP = group

    def init_device_mesh(
        self,
        device_type: str,
        mesh_shape: Tuple[int, ...],
        *,
        mesh_dim_names: Optional[Tuple[str, ...]] = None,
        check_uniqueness: bool = True,
    ) -> DeviceMesh:
        """Initializes a `DeviceMesh` based on `device_type`, `mesh_shape`, and `mesh_dim_names` parameters.
        This creates a DeviceMesh with a mesh layout of n-d dimensional array, n being the len(mesh_shape)
        and ith dimension being in size mesh_shape[i]. If mesh_dim_names is provided, each dimension is
        labeled as mesh_dim_names[i]. Inherit this utility from upstream DeviceMesh.

        Syntax of (global) DeviceMesh created by our API:
        Dimensions follow a left-to-right, inter-instance to intra-instance fashion: i.e.
        1. Dimensions of 3-dimensional global DeviceMesh: [PIPELINE_PARALLEL_DIM, DATA_PARALLEL_DIM, TENSOR_PARALLEL_DIM]
            - When PIPELINE_PARALLEL_DIM > 1, 1). DATA_PARALLEL_DIM=1, or 2). TENSOR_PARALLEL_DIM=1, or
            3). DATA_PARALLEL_DIM=1, or 2). TENSOR_PARALLEL_DIM=1, DeviceMesh is written in 3-dimensional
        2. Dimensions of 2-dimensional global DeviceMesh: [DATA_PARALLEL_DIM, TENSOR_PARALLEL_DIM]
        3. Dimensions of 1-dimensional global DeviceMesh: [DATA_PARALLEL_DIM or TENSOR_PARALLEL_DIM]
            - 1-dimensional DeviceMesh can be used to specify process groups of data parallel and tensor model parallel dimensions

        Args:
            device_type (str): device type of the mesh. Currently supports: cpu, cuda/cuda-like.
            mesh_shape: Tuple[int]: A tuple describes the dimension of the multi-dimesnion array
            that describes the layout of devices.
        Kwargs:
            mesh_dim_names: Optional[Tuple[str]]: A tuple of mesh dim names to be assigned to each dimension
            of the multi-dimensional array that describes the layout of devices. Its length must match the length
            of `mesh_shape`. Each string in mesh_dim_names must be unique. Note that if mesh_dim_names is None,
            the function will provide a default mesh identifiers.

            check_uniqueness (bool): Set False to allow veDeviceMesh API to initialize a "global device mesh" but once.
            Otherwise, set the DeviceMesh first created by init_device_mesh as the global DeviceMesh. Default by True.

        Returns:
            A :class:`DeviceMesh` object

        .. note: If no process group is found, init_device_mesh will initialize distributed process group/groups
        behind the scene, which are required for distributed communications.

        Example:
            >>> # xdoctest: +SKIP
            >>> from vescale.devicemesh_api.device_mesh_api import veDeviceMesh
            >>>
            >>> # Example 1: create a one-dimensional DeviceMesh
            >>> mesh_1d = veDeviceMesh.init_device_mesh("cuda", mesh_shape=(8,), check_uniqueness=False)
            >>>
            >>> # Example 2: create a two-dimensional DeviceMesh
            >>> mesh_2d = veDeviceMesh.init_device_mesh("cuda", mesh_shape=(2, 8), mesh_dim_names=("dp", "tp"), check_uniqueness=False)

        Limitation: we currently only support fixed sized DeviceMesh with 1 to 3 dimensions. We will loosen this constraint in future.
        """
        if device_type.startswith("cuda") and device_type != "cuda":
            warnings.warn("'cuda:<rank>' is invalid ! Convert to pure 'cuda'!")
            device_type = "cuda"
        assert device_type in ("cuda", "cpu", "meta"), "Supports only three device types: cuda, cpu, meta!"
        if self._GLOBAL_MESH is None or not check_uniqueness:
            if mesh_dim_names is None:
                # Support two default sets of default mesh dimensions: 2-dim [dp, tp], and 3-dim [pp, dp, tp]
                mesh_dim_names = ["PP", "DP", "TP"][-len(mesh_shape) :]
            if device_type is None:
                device_type = "cuda"
            self._GLOBAL_MESH = init_device_mesh(device_type, mesh_shape, mesh_dim_names=mesh_dim_names)
            self._MESH_GRIDS = self._GLOBAL_MESH.mesh.clone().detach().cpu()
            if len(mesh_shape) == 3:
                self._PIPELINE_PARALLEL_SIZE, self._DATA_PARALLEL_SIZE, self._TENSOR_PARALLEL_SIZE = mesh_shape
            elif len(mesh_shape) == 2:
                self._DATA_PARALLEL_SIZE, self._TENSOR_PARALLEL_SIZE = mesh_shape
            else:
                self._DATA_PARALLEL_SIZE = self._TENSOR_PARALLEL_SIZE = mesh_shape[0]
            for idx, name in enumerate(mesh_dim_names[::-1]):
                self._MESH_DIM_NAMES_MAPPING[idx] = name
        elif check_uniqueness:
            raise ValueError(
                "Already initialized the global DeviceMesh! Turn 'check_uniqueness' off to remove the contraint."
            )
        return self._GLOBAL_MESH

    def get(
        self,
        **kwargs,
    ) -> Optional[DeviceMesh]:
        """
        Retrieves the global device mesh. If it has not been initialized, pass in
        arguments to initialize one.

        Args:
            **kwargs (dict): arguments to initialize the global device mesh.

        Returns:
            A :class:`DeviceMesh` object
        """
        if self._GLOBAL_MESH is None and kwargs:
            self.init_device_mesh(**kwargs)
        return self._GLOBAL_MESH

    def get_tensor_parallel_mesh(self) -> DeviceMesh:
        if self._TENSOR_PARALLEL_MESH is None:
            assert self._TENSOR_PARALLEL_GROUP is not None, "tensor model parallel group is not initialized"
            assert self._MESH_DIM_NAMES_MAPPING
            tensor_dim_name = self._MESH_DIM_NAMES_MAPPING[0]
            TP_mesh = self.get()[tensor_dim_name]
            self._TENSOR_PARALLEL_MESH = DeviceMesh(
                device_type=TP_mesh.device_type,
                mesh=TP_mesh.mesh,
                pg=self._TENSOR_PARALLEL_GROUP,
                _validate_mesh=False,
            )
        return self._TENSOR_PARALLEL_MESH

    def _get_data_parallel_mesh(self) -> DeviceMesh:
        if self._DATA_PARALLEL_MESH is None:
            assert self._DATA_PARALLEL_GROUP is not None, "data parallel group is not initialized"
            assert len(self._MESH_DIM_NAMES_MAPPING) >= 2
            data_dim_name = self._MESH_DIM_NAMES_MAPPING[1]
            DP_mesh = self.get()[data_dim_name]
            self._DATA_PARALLEL_MESH = DeviceMesh(
                device_type=DP_mesh.device_type, mesh=DP_mesh.mesh, pg=self._DATA_PARALLEL_GROUP, _validate_mesh=False
            )
        return self._DATA_PARALLEL_MESH

    def get_tensor_parallel_process_group(self) -> ProcessGroup:
        assert self._TENSOR_PARALLEL_GROUP is not None, "tensor model parallel group is not initialized"
        return self._TENSOR_PARALLEL_GROUP

    def get_data_parallel_process_group(self) -> ProcessGroup:
        assert self._DATA_PARALLEL_GROUP is not None, "data parallel group is not initialized"
        return self._DATA_PARALLEL_GROUP

    def get_strategy_coordinate(self, local_rank=None) -> List[int]:
        """
        Translate current local rank to a strategy coordinate of initialized strategy dimensions.

        Args:
            local_rank (int): rank id. If local_rank is None, return the coordinate of the local rank.

        Returns:
            Coordinate of local rank mapped to the global DeviceMesh's parallel dimensions.

        Example:
            >>> from vescale.devicemesh_api.device_mesh_api import veDeviceMesh
            >>> dp_size, tp_size = 2, 2
            >>> # Initialize global device mesh of (dp_size=2, tp_size=2)
            >>> _ = veDeviceMesh.init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("DP", "TP"))
            >>> local_rank = torch.distributed.get_rank() # local_rank is 0
                0
            >>> veDeviceMesh.get_strategy_coordinate(local_rank)
                [0, 0]
            >>> veDeviceMesh.get_strategy_coordinate(3)
                [1, 1]
        """
        if self._GLOBAL_MESH is None:
            self.get()
        if local_rank is None:
            if self._RANK_COORDINATE is None:
                self._RANK_COORDINATE = self.get_strategy_coordinate(self.get_local_rank())
            return self._RANK_COORDINATE
        rank_coordinate = [int(item) for item in (self._MESH_GRIDS == local_rank).nonzero(as_tuple=True)]
        return rank_coordinate

    def lookup_rank(self, dim: Union[int, str]) -> int:
        """
        Look up the specified 'id' from a particular dimension of the strategy coordinate.

        Args:
            dim (Union[int, str]): Dimension indicator.

        Returns:
            Specified parallel strategy 'rank' of a global rank.

        Example:
            >>> from vescale.devicemesh_api.device_mesh_api import veDeviceMesh
            >>> dp_size, tp_size = 2, 2
            >>> # Initialize global device mesh of (dp_size=2, tp_size=2)
            >>> _ = veDeviceMesh.init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("DP", "TP"))
            >>> local_rank = torch.distributed.get_rank() # local_rank = 0
                0
            >>> veDeviceMesh.get_strategy_coordinate(local_rank)
                [0, 0]
            >>> index = 1
            >>> veDeviceMesh.lookup_rank(index) # local_rank is 0
                0
            >>> dim_name = "DP"
            >>> veDeviceMesh.lookup_rank(dim_name) # local_rank is 0
                0
        """
        if isinstance(dim, int):
            assert 0 <= dim < len(self._MESH_DIM_NAMES_MAPPING)
        else:
            assert dim in self._MESH_DIM_NAMES_MAPPING.values()
        if self._RANK_COORDINATE is None:
            self.get_strategy_coordinate()
        if isinstance(dim, str):
            names = list(self._MESH_DIM_NAMES_MAPPING.values())[::-1]
            index = names.index(dim)
            return self._RANK_COORDINATE[index]
        else:
            return self._RANK_COORDINATE[dim]

    def get_strategy_size(self, dim: Union[int, str]) -> List[int]:
        """
        Return the size of a parallel strategy dimension of the global DeviceMesh.

        Args:
            dim (Union[int, str]): Dimension indicator.

        Returns:
            Size of a strategt dimension.
        """
        if isinstance(dim, int):
            assert 0 <= dim < len(self._MESH_DIM_NAMES_MAPPING)
        else:
            assert dim in self._MESH_DIM_NAMES_MAPPING.values()
        strategy_sizes = self.get().size()
        if isinstance(dim, str):
            index = ["PP", "DP", "TP"].index(dim.lower())
            return strategy_sizes[index]
        else:
            return strategy_sizes[dim]

    def get_local_rank(self) -> int:
        """
        Get rank ID based on this machine.
        """
        self.get()
        local_device_count = torch.cuda.device_count() if torch.cuda.is_available() else self.DEFAULT_DEVICE_COUNT
        return get_rank() % local_device_count

    def get_pipeline_parallel_rank(self) -> int:
        """
        Get pipeline parallel rank (stage id) of local rank id.
        """
        num_dims = len(self._MESH_DIM_NAMES_MAPPING)
        assert num_dims <= 3
        if len(self._MESH_DIM_NAMES_MAPPING) == 3:
            pipe_dim_name = self._MESH_DIM_NAMES_MAPPING[2]
            return self.lookup_rank(pipe_dim_name)
        else:
            return 0

    def get_data_parallel_rank(self) -> int:
        """
        Get data parallel rank (stage id) of local rank id.
        """
        assert len(self._MESH_DIM_NAMES_MAPPING) >= 2
        data_dim_name = self._MESH_DIM_NAMES_MAPPING[1]
        return self.lookup_rank(data_dim_name)

    def get_tensor_parallel_rank(self) -> int:
        """
        Get tensor parallel rank (stage id) of local rank id.
        """
        assert self._MESH_DIM_NAMES_MAPPING
        tensor_dim_name = self._MESH_DIM_NAMES_MAPPING[0]
        return self.lookup_rank(tensor_dim_name)

    def get_pipeline_parallel_mesh(self) -> DeviceMesh:
        """
        Return the pipeline parallel view of the global DeviceMesh.
        """
        assert len(self._MESH_DIM_NAMES_MAPPING) == 3
        pipe_dim_name = self._MESH_DIM_NAMES_MAPPING[0]
        return self.get()[pipe_dim_name]

    def get_global_pipeline_parallel_meshes(self, device_type="cuda") -> list:
        if self._GLOBAL_PIPELINE_MODEL_PARALLEL_MESHES is None:
            meshes = []
            device_mesh = self.get()
            for inner_group in device_mesh.mesh.tolist():
                meshes.append(DeviceMesh(device_type, inner_group, _validate_mesh=False))
            _GLOBAL_PIPELINE_MODEL_PARALLEL_MESHES = meshes
        return _GLOBAL_PIPELINE_MODEL_PARALLEL_MESHES

    def get_data_parallel_mesh(self) -> DeviceMesh:  # noqa: F811
        """
        Return the data parallel view of the global DeviceMesh.
        """
        assert self._MESH_DIM_NAMES_MAPPING
        dp_name = self._MESH_DIM_NAMES_MAPPING[1]
        return self.get()[dp_name]

    def get_tensor_parallel_mesh(self) -> DeviceMesh:
        """
        Return the tensor parallel view of the global DeviceMesh.
        """
        assert self._MESH_DIM_NAMES_MAPPING
        tp_name = self._MESH_DIM_NAMES_MAPPING[0]
        return self.get()[tp_name]

    def get_global_tensor_parallel_meshes(self) -> list:
        if self._GLOBAL_TENSOR_PARALLEL_MESHES is None:
            tp_meshes = []
            global_dm = self.get()
            device_type = self.get_tensor_parallel_mesh().device_type
            all_tp_list = global_dm.mesh.view(-1, global_dm.mesh.size(2))
            for tp_group in all_tp_list:
                tp_mesh = DeviceMesh(
                    device_type,
                    tp_group,
                    _validate_mesh=False,
                    _init_process_groups=False,
                )
                tp_meshes.append(tp_mesh)
            self._GLOBAL_TENSOR_PARALLEL_MESHES = tp_meshes
        return self._GLOBAL_TENSOR_PARALLEL_MESHES

    def is_first_stage(self) -> bool:
        """
        Return if the current stage is the first stage, if using pipeline parallelism.
        """
        pp_rank = self.get_pipeline_parallel_rank()
        return pp_rank == 0

    def is_last_stage(self) -> bool:
        """
        Return if the current stage is the last stage, if using pipeline parallelism.
        """
        assert len(self._MESH_DIM_NAMES_MAPPING) == 3
        device_mesh = self.get()
        num_stages = device_mesh.size(self.PP_DIM)
        pp_rank = self.get_pipeline_parallel_rank()
        return pp_rank == num_stages - 1

    def __getitem__(self, mesh_dim_name: str) -> DeviceMesh:
        """
        Slice the current DeviceMesh based on the mesh_dim_name given to create a child
        DeviceMesh. Inherit this utility from upstream DeviceMesh.

        Args:
            mesh_dim_name (str): mesh dimension name.

        Returns:
            a dimension "view" of the global DeviceMesh.
        """
        device_mesh = self.get()
        return device_mesh[mesh_dim_name]

    def get_data_parallel_dim_groups(self) -> ProcessGroup:
        """
        Match process groups of data parallel dimension given
        sizes of DeviceMesh.
        """
        device_mesh = self.get()
        dim_size = len(device_mesh.mesh.shape)
        assert 1 <= dim_size <= 3
        if dim_size <= 2:
            return device_mesh.get_dim_groups(0)
        return device_mesh.get_dim_groups(1)

    def get_tensor_parallel_dim_groups(self) -> ProcessGroup:
        """
        Return process group of the lowest dimension as
        the dimension of tensor model parallelism.
        """
        device_mesh = self.get()
        assert 1 <= len(device_mesh.mesh.shape) <= 3
        return device_mesh.get_dim_groups(0)

    def get_coordinate(self) -> Optional[List[int]]:
        """
        Return the relative indices of this rank relative to all
        dimensions of the mesh. If this rank is not part of the mesh, return None.
        Inherit this utility from upstream DeviceMesh
        """
        device_mesh = self.get()
        return device_mesh._coordinate_on_dim if device_mesh._coordinate_on_dim else None

    def size(self, dim: Optional[int] = None) -> int:
        device_mesh = self.get()
        return device_mesh.mesh.numel() if dim is None else device_mesh.mesh.size(dim)

    @property
    def ndim(self) -> int:
        device_mesh = self.get()
        return device_mesh.mesh.ndim

    @property
    def shape(self) -> Tuple[int, ...]:
        device_mesh = self.get()
        return tuple(device_mesh.mesh.shape)


veDeviceMesh = VeDeviceMesh()