################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

import logging
import math
import warnings
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed._functional_collectives as funcol
from torch.distributed.distributed_c10d import (
    ProcessGroup,
    _find_pg_by_ranks_and_tag,
    _get_default_group,
    _get_group_size,
    _get_group_tag,
    get_process_group_ranks,
    get_rank,
    get_world_size,
    init_process_group,
    is_initialized,
    new_group,
)

logger = logging.getLogger(__name__)

# only import numpy typing when type checking
if TYPE_CHECKING:
    try:
        from numpy.typing import ArrayLike
    except ImportError:
        logger.warning("DeviceMesh requires numpy >= 1.21 to be installed for type checking")


class _MeshEnv:
    def __init__(self) -> None:
        self.mesh_stack: List[DeviceMesh] = []
        self.child_to_parent_mapping: Dict[DeviceMesh, DeviceMesh] = {}

    def get_current_mesh(self) -> "DeviceMesh":
        if len(self.mesh_stack) == 0:
            raise RuntimeError("No device mesh is currently active!")
        return self.mesh_stack[-1]

    def create_child_mesh(self, device_mesh: "DeviceMesh", mesh_dim: int, mesh_dim_name: str) -> "DeviceMesh":
        # swap the current dim to the last dim then reshape to flatten out other
        # dims, so we can just extract the list of ranks which contains cur_rank.
        cur_rank = device_mesh.get_rank()
        pg_ranks_by_dim = device_mesh.mesh.swapdims(-1, mesh_dim).reshape(-1, device_mesh.mesh.size(mesh_dim))

        for mesh_1d in pg_ranks_by_dim:
            sub_mesh = DeviceMesh(
                device_mesh.device_type,
                mesh_1d,
                mesh_dim_names=(mesh_dim_name,),
                _init_process_groups=False,
            )
            if cur_rank in mesh_1d:
                res_sub_mesh = sub_mesh

        res_sub_mesh._dim_group_infos = [device_mesh._dim_group_infos[mesh_dim]]
        # Assign the current DeviceMesh as the parent of the child DeviceMesh.
        self.child_to_parent_mapping[res_sub_mesh] = device_mesh
        return res_sub_mesh

    def create_submesh_along_multi_dims(
        self, device_mesh: "DeviceMesh", mesh_dims: List[int], cur_rank: int = None
    ) -> "DeviceMesh":
        # swap the current dim to the last dim then reshape to flatten out other
        # dims, so we can just extract the list of ranks which contains cur_rank.
        # check dims
        dim_size = [-1]
        for dim in mesh_dims:
            if dim >= device_mesh.ndim:
                raise RuntimeError("Mesh dim in sub groups out of range!")
            dim_size.append(device_mesh.mesh.size(dim))
        mesh_tensor = device_mesh.mesh
        for dim in mesh_dims:
            mesh_tensor = mesh_tensor.swapdims(-1, dim)
        if cur_rank is None:
            cur_rank = device_mesh.get_rank()
        pg_ranks_by_dims = mesh_tensor.reshape(dim_size)
        for mesh_nd in pg_ranks_by_dims:
            sub_mesh = DeviceMesh(
                device_mesh.device_type,
                mesh_nd,
                _init_process_groups=False,
            )
            if cur_rank in mesh_nd:
                res_sub_mesh = sub_mesh
        res_sub_mesh._dim_group_infos = [device_mesh._dim_group_infos[dim] for dim in mesh_dims]
        self.child_to_parent_mapping[res_sub_mesh] = device_mesh
        return res_sub_mesh

    def create_submesh_group(self, device_mesh: "DeviceMesh", mesh_dim: int) -> "DeviceMesh":
        # swap the current dim to the last dim then reshape to flatten out other
        # dims, so we can just extract the list of ranks which contains cur_rank.
        # check dims
        pg_ranks_by_dim = device_mesh.mesh.swapdims(-1, mesh_dim).reshape(-1, device_mesh.mesh.size(mesh_dim))
        res = []
        for mesh_1d in pg_ranks_by_dim:
            sub_mesh = DeviceMesh(
                device_mesh.device_type,
                mesh_1d,
                _init_process_groups=False,
            )
            sub_mesh._dim_group_infos = [device_mesh._dim_group_infos[mesh_dim]]
            # Assign the current DeviceMesh as the parent of the child DeviceMesh.
            self.child_to_parent_mapping[sub_mesh] = device_mesh
            res.append(sub_mesh)
        return res

    def get_parent_mesh(self, device_mesh: "DeviceMesh") -> Optional["DeviceMesh"]:
        return self.child_to_parent_mapping.get(device_mesh, None)

    def get_parent_mesh_dim(self, device_mesh: "DeviceMesh") -> Optional[int]:
        """
        Return the index of the mesh dim in the parent mesh.
        The device_mesh passed in needs to be sliced out from a parent mesh.
        """
        parent_mesh = self.get_parent_mesh(device_mesh)
        child_mesh_dim_names = device_mesh.mesh_dim_names
        if parent_mesh and child_mesh_dim_names:
            assert len(child_mesh_dim_names) == 1, "The child mesh can only be a 1D mesh."
            child_mesh_dim_name = child_mesh_dim_names[0]
            if parent_mesh.mesh_dim_names:
                return parent_mesh.mesh_dim_names.index(child_mesh_dim_name)
        return None

    @staticmethod
    def num_devices_per_host(device_type: str) -> int:
        return _get_device_handle(device_type).device_count()

    @staticmethod
    def num_hosts(device_type: str) -> int:
        # ProcessGroup can't tell us this info so we have to infer it, assume
        # homogeneous hardware for now
        return get_world_size() // _MeshEnv.num_devices_per_host(device_type)


mesh_resources: _MeshEnv = _MeshEnv()


def _get_device_handle(device_type: str = "cuda"):
    """
    Get the module corresponding to the device_type which is cuda or cuda-like device.
    For example, when the device_type is cuda, the module `torch.cuda` is returned.
    Return None when there is no corresponding module for device_type, otherwise
    return the corresponding module.
    """
    return getattr(torch, device_type, None)


class DeviceMesh:
    """
    DeviceMesh represents a mesh of devices (given by `device_type`), where layout
    of devices could be represented as a n-d dimension array `mesh`, and each value
    of the `mesh` is the global rank in the default process group.

    DeviceMesh could be used to describe the layout of devices across the cluster
    via `mesh_dim_names`, and serves as a proxy for communication among the device lists
    within the cluster.

    By default (`pg` is `None`), we use the default ProcessGroup in this DeviceMesh class
    to implement proper communications. Note that we also add collective wrappers in this
    class. This is used to decouple detailed communication backend with the underlying
    DTensor implementation.

    By giving an existing ProcessGroup `pg`, we construct a device mesh from this `pg`,
    instead of the default ProcessGroup.

    Here are the expected behaviors:
    | `mesh` | `pg`  | result                               | catch
    ---------------------------------------------------------------------------------------------
    |  None  | None  | raise error!                         |
    |  EXIST | None  | use `mesh` + default ProcessGroup    |
    |  None  | EXIST | use `pg`'s ranks + `pg` ProcessGroup | 1D mesh only
    |  EXIST | EXIST | use `pg`'s ranks + `pg` ProcessGroup | `mesh` must equal to `pg`'s ranks

    Args:
        device_type (str): device type of the mesh. Currently supports: cpu, cuda/cuda-like, meta.
        mesh (ndarray): could be a multi-dimension array or an integer tensor that
            describes the layout of devices, the ids are global ids of the default process group.
        mesh_dim_names (Optional[Tuple[str]]): A tuple of mesh dim names to be assigned to each
            dimension of the multi-dimensional array that describes the layout of devices. Its
            length must match the length of `mesh_shape`. Each string in mesh_dim_names must be unique.
        pg (Optional[ProcessGroup]): the given ProcessGroup. See above for expected behaviors.

    Returns:
        A :class:`DeviceMesh` object

    Example (2 host with 4 GPUs each):
        ```
        # The following program runs on each process/rank in SPMD manner.
        # initialize device mesh as (2, 4) to represent the topology
        # of cross-host(dim 0), and within-host (dim 1)
        mesh = DeviceMesh(device_type="cuda",
                          mesh=[
                            [0, 1, 2, 3],
                            [4, 5, 6, 7]
                          ])
        ```
        A reduction over the first dimension of mesh will reduce across
        columns (0, 4), .. and (3, 7), a reduction over the second dimension
        of mesh reduces across rows (0, 1, 2, 3) and (4, 5, 6, 7)

    Note:
        DeviceMesh can be used as a context manager.
    """

    device_type: str
    mesh: Optional[Union[torch.Tensor, "ArrayLike"]]
    mesh_dim_names: Optional[Tuple[str, ...]]

    def __init__(
        self,
        device_type: str,
        mesh: Optional[Union[torch.Tensor, "ArrayLike"]] = None,
        *,
        mesh_dim_names: Optional[Tuple[str, ...]] = None,
        pg: Optional[ProcessGroup] = None,
        _validate_mesh: bool = True,
        _init_process_groups: bool = True,
    ) -> None:
        # check args
        if mesh is None and pg is None:
            raise ValueError("Either `mesh` or `pg` must be provided!")
        if mesh is not None and pg is not None:
            pg_mesh_tensor = torch.tensor(get_process_group_ranks(pg), dtype=torch.int, device="cpu")
            mesh_tensor = (
                mesh.detach().cpu()
                if isinstance(mesh, torch.Tensor)
                else torch.tensor(mesh, dtype=torch.int, device="cpu")
            )
            if not torch.equal(mesh_tensor, pg_mesh_tensor):
                raise ValueError(f"mesh({mesh_tensor}) and pg({pg_mesh_tensor}) must have the same content!")
        if pg is not None:
            self.mesh = torch.tensor(get_process_group_ranks(pg), dtype=torch.int, device="cpu")
            warnings.warn("Construction from given ProcessGroup is only supported for 1D mesh currently.")
            # TO FIX: use `mesh` to reshape `pg_mesh_tensor` for nD mesh tensor
        if mesh is not None:
            self.mesh = (
                mesh.detach().cpu()
                if isinstance(mesh, torch.Tensor)
                else torch.tensor(mesh, dtype=torch.int, device="cpu")
            )

        self.device_type = device_type
        self.mesh_dim_names = mesh_dim_names

        # private field to pre-generate DeviceMesh's hash
        self._flatten_mesh_list = tuple(self.mesh.flatten().tolist())
        self._hash = hash((self._flatten_mesh_list, self.mesh.shape))

        # step 1: try to create default world pg.
        if pg is None:
            pg = self._get_or_create_default_group()

        # step 2: validate the mesh before following usage.
        if _validate_mesh:
            self._validate_mesh(pg)

        # step 3: get coordinate of current global rank on the mesh.
        # The world pg is used for device mesh identity (rank) on each
        # process (we need to know if the current global rank is in the mesh or not)
        rank_coords = (self.mesh == get_rank()).nonzero()
        assert rank_coords.size(0) in (0, 1)
        self._coordinate_on_dim: Optional[List[int]] = rank_coords[0].tolist() if rank_coords.size(0) > 0 else None

        # step 4: init multi subprocess group for the mesh object.
        if _init_process_groups:
            self._init_process_groups(pg)

    def _get_or_create_default_group(self):
        default_initialized = is_initialized()
        if not default_initialized:
            init_process_group()

        world_size = get_world_size()
        if self.mesh.numel() > world_size:
            raise RuntimeError(
                f"Mesh should not be bigger than default world size, but found {self.mesh.numel()} ranks!"
            )

        device_handle = _get_device_handle(self.device_type)
        # TODO: if user want to pass pg_options, offer a way to do it
        if not default_initialized and device_handle:
            # automatically set the current cuda/cuda-like device base on num of gpu devices available in each host
            # NOTE: This device selection would only work for homogeneous hardware.
            num_devices_per_host = device_handle.device_count()
            if world_size > num_devices_per_host and world_size % num_devices_per_host != 0:
                raise RuntimeError(
                    f"DeviceMesh only support homogeneous hardware, but found "
                    f"{world_size} ranks and {num_devices_per_host} {self.device_type} devices!"
                )
            device_handle.set_device(get_rank() % num_devices_per_host)

        return _get_default_group()

    def _validate_mesh(self, pg: ProcessGroup):
        # validate rank uniqueness in mesh tensor
        unique_mesh_values = self.mesh.unique(sorted=True)
        if unique_mesh_values.numel() != self.mesh.numel():
            raise RuntimeError(f"DeviceMesh cannot have duplicate values, but found {self.mesh.tolist()}")
        # validate size
        if self.mesh.numel() > _get_group_size(pg):
            raise RuntimeError(
                f"DeviceMesh should not be bigger than world (group) size, but found {self.mesh.numel()} and {_get_group_size(pg)}"
            )
        # validate that all calling ranks pass in the same `mesh` argument.
        self_mesh = self.mesh.to(self.device_type).contiguous()
        mesh_tensor = funcol.all_gather_tensor(self_mesh, gather_dim=0, group=pg)
        mesh_tensor_chunked = torch.chunk(mesh_tensor, _get_group_size(pg))
        # aten.equal not supported for meta device
        if self.device_type == "meta":
            return
        for other_rank, other_mesh in enumerate(mesh_tensor_chunked):
            if not torch.equal(self_mesh, other_mesh):
                raise RuntimeError(
                    f"DeviceMesh initialization does not allow different mesh argument:"
                    f"rank {get_rank()} has mesh {self_mesh} while rank {get_process_group_ranks(pg)[other_rank]}"
                    f"has mesh {other_mesh}!"
                )

    def _init_process_groups(self, pg: ProcessGroup):
        # group tag/ranks associated with each mesh dimension, each mesh dimension should
        # have one sub-group per rank
        dim_group_infos: List[Tuple[str, List[int]]] = []

        if self.mesh.ndim == 1 and self.mesh.numel() == _get_group_size(pg):
            # if the mesh is the same as the given group, we just append the given
            # pg to the first dim groups.
            dim_group_infos.append((_get_group_tag(pg), get_process_group_ranks(pg)))
        else:
            # create sub pgs base on the mesh argument specified
            for dim in range(self.mesh.ndim):
                # swap the current dim to the last dim
                # then reshape to flatten out other dims
                pg_ranks_by_dim = self.mesh.swapdims(-1, dim).reshape(-1, self.mesh.size(dim))
                # multi-dim mesh, create subgroups by looping over the pg_ranks
                # for each dim and append the groups
                for dim_mesh in pg_ranks_by_dim:
                    subgroup_ranks = dim_mesh.tolist()
                    # call new_group regardless of the current rank in the
                    # pg or not, it's required that all ranks participate
                    # in subgroup construction
                    dim_group = new_group(ranks=subgroup_ranks)
                    # only add to dim_groups if the current rank in the subgroup
                    if self.get_rank() in subgroup_ranks:
                        if len(dim_group_infos) > dim:
                            raise RuntimeError(
                                f"Each device mesh dimension should get only one process group, but got {self.get_rank} "
                                f"in {subgroup_ranks}!"
                            )
                        dim_group_infos.append((_get_group_tag(dim_group), subgroup_ranks))
        self._dim_group_infos = dim_group_infos

    def __enter__(self) -> "DeviceMesh":
        # set this mesh as the current mesh in mesh env
        mesh_resources.mesh_stack.append(self)
        return self

    # pyre-fixme[2]: Parameter must be annotated.
    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        # pop this mesh from mesh env
        mesh_resources.mesh_stack.pop()

    def __repr__(self) -> str:
        return f"DeviceMesh:({self.mesh.tolist()})"

    def __hash__(self):
        # ideally, we should use object id as hash, because different device mesh objects
        # give different subprocess group, so different device meshes.
        # in practice of sharding propagation,
        # we only care about different mesh tensor (value, shape).
        return self._hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DeviceMesh):
            return False
        if id(self.mesh) == id(other.mesh):  # short-cut eq
            return True
        if self.device_type != other.device_type:
            return False
        return self.mesh.shape == other.mesh.shape and self._flatten_mesh_list == other._flatten_mesh_list

    def __getitem__(self, mesh_dim_name: str) -> "DeviceMesh":
        """
        Slice the current DeviceMesh based on the mesh_dim_name given to create a child
        DeviceMesh.

        Args:
            mesh_dim_name (str): the name of the mesh dimension of the parent DeviceMesh
            to create a child DeviceMesh for.
        Returns:
            A :class:`DeviceMesh` object

        Example (2 host with 4 GPUs each):
        ```
        # Below is a DeviceMesh with mesh_shape of (2, 4) and mesh_dim_name of ("dp", "tp")
        mesh = DeviceMesh(device_type="cuda",
                          mesh=[
                            [0, 1, 2, 3],
                            [4, 5, 6, 7]
                          ],
                          mesh_dim_names=["dp", "tp"])
                          )
        ```
        Calling mesh["tp"] on rank 0, 1, 2, 3 would return a 1D child DeviceMesh:([0, 1, 2, 3]).
        Calling mesh["tp"] on rank 4, 5, 6, 7 would return a 1D child DeviceMesh:([4, 5, 6, 7]).
        Calling mesh["dp"] on rank 0, 4 would return a 1D child DeviceMesh:([0, 4]).
        Calling mesh["dp"] on rank 1, 5 would return a 1D child DeviceMesh:([1, 5]).
        Calling mesh["dp"] on rank 2, 6 would return a 1D child DeviceMesh:([2, 6]).
        Calling mesh["dp"] on rank 3, 7 would return a 1D child DeviceMesh:([3, 7]).
        """
        if self.mesh.ndim <= 1:
            raise RuntimeError(f"Cannot slice a DeviceMesh with {self.mesh.ndim} dimension.")
        if self.mesh_dim_names is None:
            raise KeyError(
                "No `mesh_dim_names` found.",
                "To slice the device mesh, please call `init_device_mesh` with `mesh_dim_names`.",
            )
        if mesh_dim_name not in self.mesh_dim_names:
            raise KeyError(
                f"Mesh dimension '{mesh_dim_name}' does not exist.",
                f"Available mesh dimensions are: {self.mesh_dim_names}",
            )
        mesh_dim = self.mesh_dim_names.index(mesh_dim_name)
        submesh = mesh_resources.create_child_mesh(self, mesh_dim, mesh_dim_name)

        return submesh

    def get_dim_groups(self, mesh_dim: Optional[int] = None) -> Union[ProcessGroup, List[ProcessGroup]]:
        if not hasattr(self, "_dim_group_infos"):
            raise RuntimeError("DeviceMesh process groups not initialized!")
        if mesh_dim is not None:
            return _find_pg_by_ranks_and_tag(*self._dim_group_infos[mesh_dim])
        else:
            dim_groups = []
            for mesh_dim in range(self.mesh.ndim):
                dim_groups.append(_find_pg_by_ranks_and_tag(*self._dim_group_infos[mesh_dim]))
            return dim_groups

    def size(self, dim: Optional[int] = None) -> int:
        return self.mesh.numel() if dim is None else self.mesh.size(dim)

    @property
    def ndim(self) -> int:
        return self.mesh.ndim

    @property
    def ndevice(self) -> int:
        return torch.numel(self.mesh)

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.mesh.shape)

    def get_rank(self) -> int:
        return get_rank()

    def get_coordinate(self) -> Optional[List[int]]:
        """
        Return the relative indices of this rank relative to all
        dimensions of the mesh. If this rank is not part of the mesh, return None.
        """
        return self._coordinate_on_dim if self._coordinate_on_dim else None

    def enforce_cpu_mesh_tensor(self) -> None:
        """
        move `mesh` tensor to cpu for deterministic device;
        necessary for comparison and checkpoint loading.
        """
        with torch.no_grad():
            self.mesh = self.mesh.cpu()

    def get_submesh(self, mesh_dims: Union[List[int], List[str]]) -> "DeviceMesh":
        dims = []
        for dim in mesh_dims:
            if isinstance(dim, int):
                dims.append(dim)
            elif isinstance(dim, str):
                assert dim in self.mesh_dim_names, f"Mesh dimension '{dim}' does not exist."
                dims.append(self.mesh_dim_names.index(dim))
        return mesh_resources.create_submesh_along_multi_dims(self, dims)

    def get_all_submesh(self, dim: int or str) -> List["DeviceMesh"]:
        if isinstance(dim, str):
            assert dim in self.mesh_dim_names, f"Mesh dimension '{dim}' does not exist."
            mesh_dim = self.mesh_dim_names.index(dim)
        else:
            mesh_dim = dim
        return mesh_resources.create_submesh_group(self, mesh_dim)

    def get_mapping_rank(self, other: "DeviceMesh"):
        """
        for cross mesh resharding
        we assume that the mesh is 1,2,4,8
        the size will have gcd value
        """
        mesh_list = self.mesh.view(-1).tolist()
        index = mesh_list.index(self.get_rank())
        other_mesh_list = other.mesh.view(-1).tolist()
        gcd_value = math.gcd(len(mesh_list), len(other_mesh_list))
        if gcd_value == 1 and len(mesh_list) != 1 and len(other_mesh_list) != 1:
            raise RuntimeError(f"mesh resharding the wrong shape of device mesh {mesh_list} vs {other_mesh_list}")

        a = len(mesh_list)
        b = len(other_mesh_list)
        factor = max(a, b) // min(a, b)

        if a > b:  # group down
            data = {}
            for i in range((index // factor) * factor, factor):
                data.update({mesh_list[index]: other_mesh_list[index // factor]})
            return data
        elif a < b:  # group up
            return [other_mesh_list[i] for i in range(index * factor, (index + 1) * factor)]
        else:
            return other_mesh_list[index]


def init_device_mesh(
    device_type: str,
    mesh_shape: Tuple[int, ...],
    *,
    mesh_dim_names: Optional[Tuple[str, ...]] = None,
) -> DeviceMesh:
    """
    Initializes a `DeviceMesh` based on `device_type`, `mesh_shape`, and `mesh_dim_names` parameters.
    This creates a DeviceMesh with a mesh layout of n-d dimensional array, n being the len(mesh_shape)
    and ith dimension being in size mesh_shape[i]. If mesh_dim_names is provided, each dimension is
    labeled as mesh_dim_names[i].


    Args:
        device_type (str): device type of the mesh. Currently supports: cpu, cuda/cuda-like.
        mesh_shape: Tuple[int]: A tuple describes the dimension of the multi-dimesnion array
        that describes the layout of devices.
    Kwargs:
        mesh_dim_names: Optional[Tuple[str]]: A tuple of mesh dim names to be assigned to each dimension
        of the multi-dimensional array that describes the layout of devices. Its length must match the length
        of `mesh_shape`. Each string in mesh_dim_names must be unique.

    Returns:
        A :class:`DeviceMesh` object

    .. note: If no process group is found, init_device_mesh will initialize distributed process group/groups
    behind the scene, which are required for distributed communications.

    Example:
        >>> # xdoctest: +SKIP
        >>> from torch.distributed._tensor.device_mesh import init_device_mesh
        >>>
        >>> mesh_1d = init_device_mesh("cuda", mesh_shape=(8,))
        >>> mesh_2d = init_device_mesh("cuda", mesh_shape=(2, 8), mesh_dim_names=("dp", "tp"))
    """
    if mesh_dim_names is not None:
        if len(set(mesh_dim_names)) != len(mesh_dim_names):
            raise RuntimeError(
                "Each mesh_dim_name must be uqique.",
                f"Found repeated mesh_dim_name in mesh_dim_names {mesh_dim_names}",
            )

        if len(mesh_shape) != len(mesh_dim_names):
            raise RuntimeError(
                "mesh_shape and mesh_dim_names should have same length!",
                f"Found len(mesh_dim_names): {len(mesh_dim_names)} and len(mesh_shape):{len(mesh_shape)}.",
            )

    mesh = torch.arange(math.prod(mesh_shape)).view(mesh_shape)
    device_mesh = DeviceMesh(
        device_type=device_type,
        mesh=mesh,
        mesh_dim_names=mesh_dim_names,
    )

    return device_mesh
