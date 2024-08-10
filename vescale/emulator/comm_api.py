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

from typing import List, Optional, Sequence, Tuple
import torch
from torch.autograd.function import _SingleLevelFunction
from torch._subclasses.fake_tensor import FakeTensorMode

import vescale.dtensor.dtensor as dtensor
from vescale.dtensor.placement_types import DTensorSpec, Placement, Replicate, TensorMeta
from vescale.dtensor.api import normalize_placements
from vescale.dtensor.dtensor import DTensor

from vescale.emulator.device_mesh import DeviceMesh, mesh_resources
from vescale.emulator.comm_primitive import (
    get_redistribute_fn,
    R2R,
    R2S,
)


def redistribute_local_tensor(
    local_tensors: List[torch.Tensor],
    current_spec: DTensorSpec,
    target_spec: DTensorSpec,
    async_op: bool = True,
    reverse: bool = False,
    emit_comm: bool = False,
    fake: bool = False,
) -> List[torch.Tensor]:
    """
    This redistribute the list of local tensors (List[torch.Tensor]) from the current DTensorSpec to
    the target DTensorSpec, which involves the necessary emulator collective calls to transform
    the local shard of the DTensor from its current spec to the target spec.

    Args:
        current_spec (DTensorSpec): sharding info of the local_tensor.
        target_spce (DTensorSpec): sharding info of the desired tensor after communicating.
        async_op (bool, optional): whether this redistribute is asynchronous in communication (for both forward and backward).
            This argument is ignored.
        reverse (bool, optional): the order in which communication happens on mesh dims.
            - False: the default value, communicate tensor from outter mesh dims to inner mesh dims,
                i.e, preferentially communicating meshes at lower ranks
            - True: communicate tensor from inner mesh dims first.
            Note, in most cases, reverse=True or False has no impact on the final output tensor.
            But be careful when there are multi sharding on one tensor dim.
        emit_comm (bool, optional): whether to emit collective when converting tensor from Replicate spec to other spec.
            If you are not sure that source tensors are the same for all ranks, set emit_comm = True.
            For example, if you are converting tensor from Replicate to Shard, when you provide emit_comm = True,
            it will emit mesh_scatter collective, otherwise it will simply split the original tensor and take a part as the result.
        fake (bool, optional): whether to run in FakeTensorMode. Default, False.

    .. Note::
        - You shouldn't assume tensormeta exists.
        - Not differentiable
    """

    if current_spec.mesh != target_spec.mesh:
        current_spec = current_spec.cast_to(target_spec.mesh)
        if current_spec is None:
            # TODO: alltoall/permute reshuffling to change device_mesh if they are not the same
            raise NotImplementedError(
                "Cross device mesh communication not supported when DTensorSpec"
                "cannot be casted from src device mesh to target one"
            )

    if fake:
        async_op = True  # avoid invoking mesh_wait in FakeTensorMode.
    # shortcut: return local tensor if no placement changes.
    if current_spec.placements == target_spec.placements:
        return local_tensors

    device_mesh = current_spec.mesh

    new_local_tensors = None

    current_placements = current_spec.placements
    target_placements = target_spec.placements
    # sorted_placements = list(enumerate(zip(current_placements, target_placements)))
    # TODO: the order of commnunication matters, we need find a correct order when more
    # than one communication happens between a single DTenosrSpec changes.

    looped_mesh_dims = range(device_mesh.ndim)
    if reverse:
        looped_mesh_dims = reversed(looped_mesh_dims)
    for i in looped_mesh_dims:
        current = current_placements[i]
        target = target_placements[i]
        if current == target:
            # short cut: just use the original local tensor
            new_local_tensors = local_tensors
            continue

        redistribute_fn = get_redistribute_fn(current_spec, target_spec, current, target)
        if not fake:
            if isinstance(redistribute_fn, (R2R, R2S)):
                new_local_tensors = redistribute_fn(local_tensors, current, target, device_mesh, i, emit_comm=emit_comm)
            else:
                new_local_tensors = redistribute_fn(local_tensors, current, target, device_mesh, i)
        else:
            if isinstance(redistribute_fn, (R2R, R2S)):
                new_local_tensors = redistribute_fn.__fake_call__(
                    local_tensors, current, target, device_mesh, i, emit_comm=emit_comm
                )
            else:
                new_local_tensors = redistribute_fn.__fake_call__(local_tensors, current, target, device_mesh, i)

        assert new_local_tensors is not None
        local_tensors = new_local_tensors
    assert new_local_tensors is not None, "redistribute failed!"
    return new_local_tensors


######################## DTensor collective #########################


class DTensorRedistribute(torch.autograd.Function):
    @classmethod
    def apply(cls, *args, **kwargs):
        # rewrite torch.autograd.Function.apply to skip functorch check, which is unnecessary for this autograder
        return super(_SingleLevelFunction, cls).apply(*args, **kwargs)  # type: ignore[misc]

    @staticmethod
    def forward(
        ctx,
        inputs: List["dtensor.DTensor"],
        device_mesh: DeviceMesh,
        placements: Tuple[Placement],
        async_op: bool = True,
    ):
        # FIXME: We use early return (it is now moved to `redistribute_local_tensor()`) to
        # avoid view(). There are several hidden dangers here:
        # - The change of the tensor wrapper may cause the failure of the tensor's hooks.
        # - Modifying the tensor may change the result of is_param of parameters.
        # - Dynamically modifying the computation graph may cause problems with autograd.

        previous_spec = inputs[0]._spec
        target_spec = DTensorSpec(device_mesh, placements, tensor_meta=previous_spec.tensor_meta)

        ctx.previous_spec = previous_spec
        ctx.async_op = async_op

        local_tensors = [input._local_tensor for input in inputs]
        outputs = redistribute_local_tensor(local_tensors, previous_spec, target_spec, async_op, reverse=True)
        for i, input in enumerate(inputs):
            outputs[i].requires_grad_(input.requires_grad)

        # TODO: unify these
        # from vescale.plan.hooks.factory_hooks import FactoryDispatchModeOff as NewFactoryDispatchModeOff
        from vescale.dmodule._factory import FactoryDispatchModeOff as OldFactoryDispatchModeOff

        with (
            FakeTensorMode(allow_non_fake_inputs=True)
            and torch.no_grad()
            # unset factory mode.
            # and NewFactoryDispatchModeOff()
            and OldFactoryDispatchModeOff()
        ):
            fake_inputs = [torch.empty_strided(input.shape, input.stride(), dtype=input.dtype) for input in inputs]
            fake_outs = redistribute_local_tensor(
                fake_inputs, previous_spec, target_spec, async_op, reverse=True, fake=True
            )
        target_spec.tensor_meta = TensorMeta(
            shape=target_spec.tensor_meta.shape,
            stride=fake_outs[0].stride(),
            dtype=target_spec.tensor_meta.dtype,
        )
        result_list = []
        for input, output in zip(inputs, outputs):
            result_list.append(
                dtensor.DTensor(
                    output,
                    target_spec.mesh,
                    target_spec.placements,
                    shape=target_spec.tensor_meta.shape,
                    dtype=target_spec.tensor_meta.dtype,
                    requires_grad=input.requires_grad,
                    stride=target_spec.tensor_meta.stride,
                )
            )

        return result_list


def distribute_tensor(
    tensors: List[torch.Tensor],
    device_mesh: Optional[DeviceMesh] = None,
    placements: Optional[Sequence[Placement]] = None,
) -> List["dtensor.DTensor"]:
    """
    Distribute a list of global `torch.Tensor` to the `device_mesh` according to the `placements`
    specified. The rank of `device_mesh` and `placements` must be the same.

    Args:
        tensors (List[torch.Tensor]): global torch.Tensor to be distributed. Note that if you
            want to shard a tensor on a dimension that is not evenly divisible by
            the number of devices in that mesh dimension, we use `torch.chunk`
            semantic to shard the tensor and scatter the shards.
        device_mesh (:class:`DeviceMesh`, optional): emulator DeviceMesh to distribute the
            tensor, if not specified, must be called under a DeviceMesh context
            manager, default: None
        placements (List[:class:`Placement`], optional): the placements that
            describes how to place the tensor on DeviceMesh, must have the same
            number of elements as `device_mesh.ndim`. If not specified, we will
            by default replicate the tensor across the `device_mesh` from the
            first rank of each dimension of the `device_mesh`.

    Returns:
        A list of :class:`DTensor` object

    Best practice to save memory:
        >>> dist_tensor = distribute_tensor(global_tensor, ...)
        >>> del global_tensor
    """

    torch._C._log_api_usage_once("torch.dtensor.distribute_tensor")

    # get default device mesh if there's nothing specified
    device_mesh = device_mesh or mesh_resources.get_current_mesh()
    device_type = device_mesh.device_type

    for i, tensor in enumerate(tensors):
        if not tensor.is_leaf:
            raise RuntimeError(
                "`distribute_tensor` should be used to distribute leaf tensors! but found non-leaf tensor!"
            )

        # convert tensor to the corresponding device type if it's not in that device type
        if device_type != tensor.device.type and not tensor.is_meta:
            tensors[i] = tensor.to(device_type)

    # validate placements
    placements: Tuple[Placement] = normalize_placements(
        placements, device_mesh.ndim, tensor_ndim=tensor.ndim, none_as_replicate=True
    )

    # validate tensor type
    results = []
    for tensor in tensors:
        if isinstance(tensor, dtensor.DTensor):
            # if the tensor is already a DTensor, we just need to check if the
            # device mesh and placements are the same
            if tensor.device_mesh != device_mesh:
                raise ValueError(
                    f"Cannot distribute a DTensor with device mesh {tensor.device_mesh} "
                    f"to a different device mesh {device_mesh}."
                )
            if tensor.placements != placements:
                raise ValueError(
                    f"Cannot distribute a DTensor with placements {tensor.placements} "
                    f"to a different placements {placements}. do you want to call "
                    f"`redistribute` instead?"
                )
            results.append(tensor)
    if len(results) == len(tensors):
        return results

    target_spec = DTensorSpec(mesh=device_mesh, placements=placements, tensor_meta=None)
    
    placements: Tuple[Placement] = tuple([Replicate()] * device_mesh.ndim)
    tensor_meta = TensorMeta(shape=tensors[0].shape, stride=tensors[0].stride(), dtype=tensors[0].dtype)
    current_spec = DTensorSpec(mesh=device_mesh, placements=placements, tensor_meta=tensor_meta)
    local_tensors = redistribute_local_tensor(
        local_tensors=tensors,
        current_spec=current_spec,
        target_spec=target_spec,
        async_op=True,
        emit_comm=True,
    )

    result_list = []
    for tensor, local_tensor in zip(tensors, local_tensors):
        tensor_meta = TensorMeta(shape=tensor.size(), dtype=tensor.dtype, stride=tensor.stride())
        target_spec.tensor_meta = tensor_meta

        assert local_tensor is not None, "distributing a tensor should not be None"
        # detach the local tensor passed to DTensor since after the construction
        # of DTensor, autograd would work on top of DTensor instead of local tensor
        result_list.append(
            DTensor(
                local_tensor.detach().requires_grad_(tensor.requires_grad),
                target_spec.mesh,
                target_spec.placements,
                shape=target_spec.tensor_meta.shape,
                dtype=target_spec.tensor_meta.dtype,
                requires_grad=tensor.requires_grad,
                stride=target_spec.tensor_meta.stride,
            )
        )
    return result_list


def redistribute_dtensor(
    dtensors: List["dtensor.DTensor"],
    device_mesh: Optional[DeviceMesh] = None,
    placements: Optional[Sequence[Placement]] = None,
    async_op: bool = True,
) -> List["dtensor.DTensor"]:
    """
    `redistribute_dtensor` performs necessary emulator collective operations that redistribute the current
    DTensor from its current placements to a new placements, or from is current DeviceMesh
    to a new DeviceMesh. i.e. we can turn a Sharded DTensor to a Replicated DTensor by
    specifying a Replicate placement for each dimension of the DeviceMesh.

    Args:
        device_mesh (:class:`DeviceMesh`, optional): emulator DeviceMesh to place the
            DTensor, if not specified, must be called under a DeviceMesh
            context manager, default: None
        placements (List[:class:`Placement`], optional): the new placements that
            describes how to place the DTensor into the DeviceMesh, must
            have the same number of elements as `device_mesh.ndim`.
        async_op (bool, optional): whether this redistribute is asynchronous in communication (for both forward and backward).
            - True: the default asynchronous behavior for performance
            - False: mostly used for third-party plugin op that doesn't accept asynchronous collective tensor.

    Returns:
        A list of :class:`DTensor` object

    .. Note::
        - `redistribute_dtensor` is differentiable (i.e., redistribute happen for both forward and backward) TODO: backward compatibility
        - This redistribute API currently only supports out of place redistribution, i.e. it always create a new DTensor object and leave the original one unchanged.
    """
    return DTensorRedistribute.apply(
        dtensors,
        device_mesh,
        normalize_placements(placements, mesh_ndim=device_mesh.ndim, tensor_ndim=dtensors[0].ndim),
        async_op,
    )
