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
from torch.distributed import get_rank
from vescale.dtensor.device_mesh import init_device_mesh, DeviceMesh
from typing import Optional, List, Tuple, Union, Dict
from torch.distributed.distributed_c10d import ProcessGroup

__all__ = ["VESCALE_DEVICE_MESH"]


class VeDeviceMesh:
    _MESH_DIM_NAMES_MAPPING: Dict[int, str] = {}
    _MESH_DIM_NAMES_LOOKUP: List[str] = None
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

    def init_device_mesh(
        self,
        device_type: str,
        mesh_shape: Tuple[int, ...],
        *,
        mesh_dim_names: Optional[Tuple[str, ...]] = None,
        check_uniqueness: bool = False,
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

            check_uniqueness (bool): This advanced argument is used to prevent users from spoiling global
            DeviceMesh API by creating multiple copies in a large code repository.
            Set to True to allow VESCALE_DEVICE_MESH API to check the "global device mesh" is only initialized once.
            Otherwise, users can create as many DeviceMeshes as they want just like with upstream Devicemesh.

        Returns:
            A :class:`DeviceMesh` object

        .. note: If no process group is found, init_device_mesh will initialize distributed process group/groups
        behind the scene, which are required for distributed communications.

        Example:
            >>> # xdoctest: +SKIP
            >>> from vescale.devicemesh_api import VESCALE_DEVICE_MESH
            >>>
            >>> # Example 1: initialize the global DeviceMesh as a one-dimensional DeviceMesh
            >>> VESCALE_DEVICE_MESH.init_device_mesh("cuda", mesh_shape=(8,))
            >>>
            >>> # Example 2: re-initialize the global DeviceMesh as a two-dimensional DeviceMesh
            >>> VESCALE_DEVICE_MESH.init_device_mesh("cuda", mesh_shape=(2, 8), mesh_dim_names=("dp", "tp"))

        Limitation: we currently only support fixed sized DeviceMesh with 1 to 3 dimensions. We will loosen this constraint in future.
        """
        if device_type.startswith("cuda") and device_type != "cuda":
            warnings.warn("'cuda:<rank>' is invalid ! Convert to pure 'cuda'!")
            device_type = "cuda"
        assert device_type in ("cuda", "cpu", "meta"), "Supports only three device types: cuda, cpu, meta!"
        if self._GLOBAL_MESH is None or not check_uniqueness:
            self._TENSOR_PARALLEL_SIZE = self._DATA_PARALLEL_SIZE = self._PIPELINE_PARALLEL_SIZE = None
            self._MESH_DIM_NAMES_MAPPING = {}
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
            self._MESH_DIM_NAMES_LOOKUP = list(self._MESH_DIM_NAMES_MAPPING.values())[::-1]
            self._RANK_COORDINATE = None
            self._GLOBAL_PIPELINE_MODEL_PARALLEL_MESHES = None
            self._GLOBAL_TENSOR_PARALLEL_MESHES = None
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

    def _get_tensor_parallel_mesh(self) -> DeviceMesh:
        """
        This function works the same as get_tensor_parallel_mesh(), but
        specifies _validate_mesh=False.
        """
        assert self._GLOBAL_MESH, "Must initialize global DeviceMesh first!"
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
        """
        This function works the same as get_data_parallel_mesh(), but
        specifies _validate_mesh=False.
        """
        assert self._GLOBAL_MESH, "Must initialize global DeviceMesh first!"
        if self._DATA_PARALLEL_MESH is None:
            assert self._DATA_PARALLEL_GROUP is not None, "data parallel group is not initialized"
            assert len(self._MESH_DIM_NAMES_MAPPING) >= 2
            data_dim_name = self._MESH_DIM_NAMES_MAPPING[1]
            DP_mesh = self.get()[data_dim_name]
            self._DATA_PARALLEL_MESH = DeviceMesh(
                device_type=DP_mesh.device_type, mesh=DP_mesh.mesh, pg=self._DATA_PARALLEL_GROUP, _validate_mesh=False
            )
        return self._DATA_PARALLEL_MESH

    def get_strategy_coordinate(self, local_rank=None) -> List[int]:
        """
        Translate current local rank to a strategy coordinate of initialized strategy dimensions.
        If local_rank is not provided, return coordinate of current rank.
        The only difference of this function w.r.t. upstream DeviceMesh's get_coordinate() is that
        it enables users query strategy coordinate of arbitrary ranks.

        Args:
            local_rank (int): rank id. If local_rank is None, return the coordinate of the local rank.

        Returns:
            Coordinate of local rank mapped to the global DeviceMesh's parallel dimensions.

        Example:
            >>> from vescale.devicemesh_api import VESCALE_DEVICE_MESH
            >>> dp_size, tp_size = 2, 2
            >>> # Initialize global device mesh of (dp_size=2, tp_size=2)
            >>> VESCALE_DEVICE_MESH.init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("DP", "TP"))
            >>> local_rank = torch.distributed.get_rank() # local_rank is 0
                0
            >>> VESCALE_DEVICE_MESH.get_strategy_coordinate(local_rank)
                [0, 0]
            >>> VESCALE_DEVICE_MESH.get_strategy_coordinate(3)
                [1, 1]
        """
        assert self._GLOBAL_MESH, "Must initialize global DeviceMesh first!"
        if local_rank is None:
            if self._RANK_COORDINATE is None:
                self._RANK_COORDINATE = self.get_strategy_coordinate(self.get_local_rank())
            return self._RANK_COORDINATE
        rank_coordinate = [int(item) for item in (self._MESH_GRIDS == local_rank).nonzero(as_tuple=True)]
        return rank_coordinate

    def lookup_rank(self, dim: Union[int, str]) -> int:
        """
        Look up the specified 'id' from a particular dimension of the
        current rank's strategy coordinate.

        Args:
            dim (Union[int, str]): Dimension indicator.

        Returns:
            Specified parallel strategy 'rank' of a global rank.

        Example:
            >>> from vescale.devicemesh_api import VESCALE_DEVICE_MESH
            >>> dp_size, tp_size = 2, 2
            >>> # Initialize global device mesh of (dp_size=2, tp_size=2)
            >>> VESCALE_DEVICE_MESH.init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("DP", "TP"))
            >>> local_rank = torch.distributed.get_rank() # local_rank = 0
                0
            >>> VESCALE_DEVICE_MESH.get_strategy_coordinate(local_rank)
                [0, 0]
            >>> index = 1
            >>> VESCALE_DEVICE_MESH.lookup_rank(index) # local_rank is 0
                0
            >>> dim_name = "DP"
            >>> VESCALE_DEVICE_MESH.lookup_rank(dim_name) # local_rank is 0
                0
        """
        assert self._GLOBAL_MESH, "Must initialize global DeviceMesh first!"
        if isinstance(dim, int):
            assert 0 <= dim < len(self._MESH_DIM_NAMES_MAPPING)
        else:
            assert dim in self._MESH_DIM_NAMES_MAPPING.values()
        if self._RANK_COORDINATE is None:
            self.get_strategy_coordinate()
        if isinstance(dim, str):
            index = self._MESH_DIM_NAMES_LOOKUP.index(dim)
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
        assert self._GLOBAL_MESH, "Must initialize global DeviceMesh first!"
        if isinstance(dim, int):
            assert 0 <= dim < len(self._MESH_DIM_NAMES_MAPPING)
        else:
            assert dim in self._MESH_DIM_NAMES_MAPPING.values()
        if isinstance(dim, str):
            index = self._MESH_DIM_NAMES_LOOKUP.index(dim)
            return self.size(index)
        else:
            return self.size(dim)

    def get_local_rank(self) -> int:
        """
        Get rank ID based on this machine.
        """
        assert self._GLOBAL_MESH, "Must initialize global DeviceMesh first!"
        local_device_count = torch.cuda.device_count() if torch.cuda.is_available() else self.DEFAULT_DEVICE_COUNT
        return get_rank() % local_device_count

    def get_pipeline_parallel_rank(self) -> int:
        """
        Get pipeline parallel rank (stage id) of local rank id.
        """
        assert self._GLOBAL_MESH, "Must initialize global DeviceMesh first!"
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
        assert self._GLOBAL_MESH, "Must initialize global DeviceMesh first!"
        assert len(self._MESH_DIM_NAMES_MAPPING) >= 2
        if len(self._MESH_DIM_NAMES_MAPPING) > 1:
            data_dim_name = self._MESH_DIM_NAMES_MAPPING[1]
        else:
            data_dim_name = self._MESH_DIM_NAMES_MAPPING[0]
        return self.lookup_rank(data_dim_name)

    def get_tensor_parallel_rank(self) -> int:
        """
        Get tensor parallel rank (stage id) of local rank id.
        """
        assert self._GLOBAL_MESH, "Must initialize global DeviceMesh first!"
        assert self._MESH_DIM_NAMES_MAPPING
        tensor_dim_name = self._MESH_DIM_NAMES_MAPPING[0]
        return self.lookup_rank(tensor_dim_name)

    def get_pipeline_parallel_mesh(self) -> DeviceMesh:
        """
        Return the pipeline parallel view of the global DeviceMesh.
        """
        assert self._GLOBAL_MESH, "Must initialize global DeviceMesh first!"
        assert len(self._MESH_DIM_NAMES_MAPPING) == 3
        pipe_dim_name = self._MESH_DIM_NAMES_MAPPING[0]
        return self.get()[pipe_dim_name]

    def get_global_pipeline_parallel_meshes(self, device_type="cuda") -> list:
        assert self._GLOBAL_MESH, "Must initialize global DeviceMesh first!"
        if self._GLOBAL_PIPELINE_MODEL_PARALLEL_MESHES is None:
            meshes = []
            device_mesh = self.get()
            for inner_group in device_mesh.mesh.tolist():
                meshes.append(DeviceMesh(device_type, inner_group, _validate_mesh=False))
            self._GLOBAL_PIPELINE_MODEL_PARALLEL_MESHES = meshes
        return self._GLOBAL_PIPELINE_MODEL_PARALLEL_MESHES

    def get_data_parallel_mesh(self) -> DeviceMesh:  # noqa: F811
        """
        Return the data parallel view of the global DeviceMesh.
        """
        assert self._GLOBAL_MESH, "Must initialize global DeviceMesh first!"
        assert self._MESH_DIM_NAMES_MAPPING
        dp_name = self._MESH_DIM_NAMES_MAPPING[1] if self.ndim > 1 else self._MESH_DIM_NAMES_MAPPING[0]
        return self.get()[dp_name]

    def get_tensor_parallel_mesh(self) -> DeviceMesh:
        """
        Return the tensor parallel view of the global DeviceMesh.
        """
        assert self._GLOBAL_MESH, "Must initialize global DeviceMesh first!"
        assert self._MESH_DIM_NAMES_MAPPING
        tp_name = self._MESH_DIM_NAMES_MAPPING[0]
        return self.get()[tp_name]

    def get_global_tensor_parallel_meshes(self) -> list:
        assert self._GLOBAL_MESH, "Must initialize global DeviceMesh first!"
        if self._GLOBAL_TENSOR_PARALLEL_MESHES is None:
            assert len(self._MESH_DIM_NAMES_LOOKUP) == 3
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
        assert self._GLOBAL_MESH, "Must initialize global DeviceMesh first!"
        pp_rank = self.get_pipeline_parallel_rank()
        return pp_rank == 0

    def is_last_stage(self) -> bool:
        """
        Return if the current stage is the last stage, if using pipeline parallelism.
        """
        assert self._GLOBAL_MESH, "Must initialize global DeviceMesh first!"
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
        assert self._GLOBAL_MESH, "Must initialize global DeviceMesh first!"
        device_mesh = self.get()
        return device_mesh[mesh_dim_name]

    def get_data_parallel_dim_groups(self) -> ProcessGroup:
        """
        Match process groups of data parallel dimension given
        sizes of DeviceMesh.
        """
        assert self._GLOBAL_MESH, "Must initialize global DeviceMesh first!"
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
        assert self._GLOBAL_MESH, "Must initialize global DeviceMesh first!"
        device_mesh = self.get()
        assert 1 <= len(device_mesh.mesh.shape) <= 3
        return device_mesh.get_dim_groups(0)

    def get_coordinate(self) -> Optional[List[int]]:
        """
        Return the relative indices of this rank relative to all
        dimensions of the mesh. If this rank is not part of the mesh, return None.
        Inherit this utility from upstream DeviceMesh.
        """
        assert self._GLOBAL_MESH, "Must initialize global DeviceMesh first!"
        device_mesh = self.get()
        return device_mesh.get_coordinate()

    def size(self, dim: Optional[int] = None) -> int:
        """
        Returns dimension size of DeviceMesh along 'dim' dimension. If dim is None,
        return the total number of ranks in this DeviceMesh.

        Args:
            dim (int): dimension index

        Returns:
            Dimension size, or total number of ranks if None.
        """
        assert self._GLOBAL_MESH, "Must initialize global DeviceMesh first!"
        device_mesh = self.get()
        return device_mesh.mesh.numel() if dim is None else device_mesh.mesh.size(dim)

    @property
    def ndim(self) -> int:
        assert self._GLOBAL_MESH, "Must initialize global DeviceMesh first!"
        device_mesh = self.get()
        return device_mesh.mesh.ndim

    @property
    def shape(self) -> Tuple[int, ...]:
        assert self._GLOBAL_MESH, "Must initialize global DeviceMesh first!"
        device_mesh = self.get()
        return tuple(device_mesh.mesh.shape)


VESCALE_DEVICE_MESH = VeDeviceMesh()
