################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch._ops import OpOverload
from optree import PyTreeSpec, tree_map

from vescale.dtensor.device_mesh import DeviceMesh
from vescale.dtensor.placement_types import DTensorSpec

# Common type aliases
ArgsType = Tuple[object, ...]
KwargsType = Dict[str, object]
# ATen op schemas could have Tensor, Tuple[Tensor] and List[Tensor], so output type sould
# be the same set of possibilities.
OutputSpecType = Optional[Union[DTensorSpec, Sequence[Optional[DTensorSpec]]]]


def _rebuild_tensor_from_dtensor_meta(arg) -> object:
    """ "
    This is used to propagate tensor metadata, must be under fake mode
    """
    assert arg.tensor_meta is not None, "DTensorSpec does not contain tensor_meta."
    return torch.empty_strided(arg.tensor_meta.shape, arg.tensor_meta.stride, dtype=arg.tensor_meta.dtype)


def _is_inplace_op(op: OpOverload):
    # simple analysis of function schema to determine
    # if this is an inplace variant, it might not
    # be entirely correct, but it's good enough for now.
    return op._schema.name[-1] == "_"


def _is_out_variant_op(op: OpOverload):
    # simple analysis of function schema to determine
    # if this is an out variant, it might not
    # be entirely correct, but it's good enough for now.
    return "out" in op._schema.overload_name


@dataclass
class RuntimeSchemaInfo:
    """
    RuntimeSchemaInfo stores the operator schema related information for runtime (eager)
    execution. This is mainly used for two ways: 1. to generate hash for args to determine
    whether to re-run sharding prop or not 2. to determine if we need pytree
    """

    # This static_argnum records static arg "starting index" for ops that have non-tensor
    # args/kwargs which would affect sharding propagation results. All args after this
    # index would be hashed to our sharding cache.
    # Note that only a few ops need this information, e.g. view, transpose, var.dim, etc.
    static_argnum: int = 100
    # This static_kwargkey records static kwarg names which would affect sharding prop
    static_kwargkey: Optional[List[str]] = None
    # each op can decide if it wants to use pytree flatten/unflatten during operator
    # eager execution, by default we don't need to do flatten/unflatten, only if the
    # op indicate it needs to, this is to accelate eager performance.
    needs_pytree: bool = False


@dataclass
class OpSchema:
    """
    OpSchema is a data class that describes an operator input schemas, it
    includes DTensor DTensorSpecs and non-tensor args/kwargs (positional order
    preserved). It is mainly used by the dispatching logic below to run things like
    sharding propagation.

    NOTE: this should be used as a read only data class
    TODO: make this a frozen dataclass
          --> can speedup hashing, but need to frozen used class -- DTensorSpec

    Args:
        op: the operator overload we are intercepting
        args_schema: contains args except that the DTensor args have been replaced
            with its DTensorSpec
        kwargs_schema: contains kwargs except that the DTensor kwargs have been replaced
            with its DTensorSpec
    """

    op: OpOverload
    args_schema: ArgsType
    kwargs_schema: KwargsType

    schema_info: Optional[RuntimeSchemaInfo] = None

    @property
    def args_spec(self) -> Tuple[DTensorSpec, ...]:
        """
        args_spec: Tuple[DTensorSpec, ...]: contains a clean list of args spec list
            with NO non-DTensor positional arguments (i.e. int/float/tuple, etc)
            mainly used by sharding propagation to propagate the output spec
        """
        # filter out non-relevant values from args schema to get a clean spec list
        # this would mainly be used by sharding propagation rules
        return tuple(item for item in self.args_schema if isinstance(item, DTensorSpec))

    def __repr__(self) -> str:
        return f"OpSchema(op={self.op}," f" args_schema={self.args_schema}," f" kwargs_schema={self.kwargs_schema})"

    def __str__(self) -> str:
        args_sharding: List[str] = []
        mesh_shape = None
        for arg in self.args_schema:
            if isinstance(arg, DTensorSpec):
                args_sharding.append(str(arg))
                mesh_shape = arg.mesh.shape
            elif isinstance(arg, OpStrategy):
                assert len(arg.strategies) == 1
                arg_spec = arg.strategies[0].output_spec
                args_sharding.append(str(arg_spec))
                mesh_shape = arg_spec.mesh.shape
            elif isinstance(arg, TupleStrategy):
                first_op_strtgy = arg.childs[0]
                assert isinstance(first_op_strtgy, OpStrategy)
                mesh_shape = first_op_strtgy.strategies[0].output_spec.mesh.shape
                args_sharding.append(str(arg))
            else:
                args_sharding.append(str(arg))
        return f"Op(op={self.op}, args_sharding={', '.join(args_sharding)}@ mesh: {mesh_shape})"

    def __post_init__(self) -> None:
        has_symints = False
        for a in self.args_schema:
            if isinstance(a, DTensorSpec) and a.tensor_meta is not None:
                if any(isinstance(s, torch.SymInt) for s in a.tensor_meta.shape):
                    has_symints = True
                    break
        self.has_symints = has_symints

    def arg_type_tensor_or_tensor_list_like(self, arg_idx: int) -> bool:
        arg = self.args_schema[arg_idx]
        is_tensor = isinstance(arg, DTensorSpec)
        if is_tensor:
            return True

        if not isinstance(arg, list):
            return False

        return all(isinstance(e, DTensorSpec) or e is None for e in arg)

    def return_type_tuple_tensors(self) -> bool:
        return_types = self.op._schema.returns
        # all dispatch ops only return Tensor or Tuple[Tensor], so this check if enough
        return len(return_types) > 1 and isinstance(return_types[0].type, torch.TensorType)

    def __hash__(self) -> int:
        # Only hash args and kwargs that op indicates to hash
        if not self.schema_info:
            static_argnum = len(self.args_schema)
            static_kwargkey = None
        else:
            static_argnum = self.schema_info.static_argnum
            static_kwargkey = self.schema_info.static_kwargkey

        args_to_hash = tuple(
            tuple(e) if isinstance(e, list) else e
            for i, e in enumerate(self.args_schema)
            # NOTE: commented this line because we need to distinguish non-tensor args as well (e.g., slice op)
            # if self.arg_type_tensor_or_tensor_list_like(i) or i >= static_argnum
        )
        if static_kwargkey is not None:
            kwargs_to_hash = tuple(self.kwargs_schema.get(k, None) for k in static_kwargkey)
            to_hash = (self.op, args_to_hash, kwargs_to_hash)
        else:
            to_hash = (self.op, args_to_hash)
        try:
            ret = hash(to_hash)
        except TypeError:
            raise RuntimeError(
                "OpSchema.__hash__ failed. Check assumption of hashability:"
                "    1. args_schema has at most 1 nested list"
                "    2. kwargs_schema has no nesting at all"
            )
        return ret

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OpSchema):
            return False
        if self.op != other.op:
            return False

        if len(self.args_schema) != len(other.args_schema):
            return False

        # compare each element and early return if any of them is different
        if not self.schema_info:
            static_argnum = len(self.args_schema)
            static_kwargkey = None
        else:
            static_argnum = self.schema_info.static_argnum
            static_kwargkey = self.schema_info.static_kwargkey

        for i, (self_arg, other_arg) in enumerate(zip(self.args_schema, other.args_schema)):
            if isinstance(self_arg, DTensorSpec) and self_arg != other_arg:
                return False
            elif isinstance(self_arg, (list)):
                if not isinstance(other_arg, (list)):
                    return False
                if len(self_arg) != len(other_arg):
                    return False
                for i, (self_e, other_e) in enumerate(zip(self_arg, other_arg)):
                    if isinstance(self_e, DTensorSpec) and self_e != other_e:
                        return False
            elif i >= static_argnum and self_arg != other_arg:
                return False

        # check kwarg equality when there's a static kwarg key
        if static_kwargkey:
            for key in static_kwargkey:
                if self.kwargs_schema.get(key, None) != other.kwargs_schema.get(key, None):
                    return False

        return True

    def gen_fake_args(self) -> ArgsType:
        """
        gen_fake_args: generate fake args for the operator, this is mainly used
            by sharding propagation rules to generate fake args for the operator
            to run the local tensor operator and get the output spec.
        """
        return tree_map(
            lambda a: _rebuild_tensor_from_dtensor_meta(a) if isinstance(a, DTensorSpec) else a,
            self.args_schema,
            none_is_leaf=True,
        )

    def gen_fake_kwargs(self) -> KwargsType:
        """
        gen_fake_kwargs: generate fake kwargs for the operator, this is mainly used
            by sharding propagation rules to generate fake kwargs for the operator
            to run the local tensor operator and get the output spec.
        """
        return tree_map(
            lambda a: _rebuild_tensor_from_dtensor_meta(a) if isinstance(a, DTensorSpec) else a,
            self.kwargs_schema,
            none_is_leaf=True,
        )

    # this function is for compatibility with upstream, it has no actuall effect
    # for our dispatch pipeline.
    def _inplace_rewrap_schema_suggestion(self, origin_schema: "OpSchema") -> None:
        suggestion_args_spec = self.args_spec
        new_arg_schema: List[object] = []
        idx_of_args_spec = 0
        for arg in origin_schema.args_schema:
            if isinstance(arg, DTensorSpec):
                new_arg_schema.append(suggestion_args_spec[idx_of_args_spec])
                idx_of_args_spec += 1
            else:
                new_arg_schema.append(arg)
        self.args_schema = tuple(new_arg_schema)
        self.kwargs_schema = origin_schema.kwargs_schema


@dataclass
class OutputSharding:
    """
    OutputSharding is a data class that is used by the sharding propagation
    rules, it could set the output_spec upon successful propagation, and if
    it failed, output_spec would become None and sharding propagation rules
    could give a list of suggestions for inputs to reshard.

    NOTE: the schema_suggestion generated by sharding propagation should be
    exactly the same as the operator OpSchema, except the DTensor DTensorSpecs
    """

    output_spec: OutputSpecType
    schema_suggestions: Optional[List[OpSchema]] = None
    failed_reason: Optional[str] = None
    needs_redistribute: bool = False


@dataclass
class OpInfo:
    """
    All Runtime Op execution info are packed here
    """

    mesh: DeviceMesh
    schema: OpSchema
    flat_args_schema: List[object]
    local_args: Sequence[object]
    local_kwargs: Dict[str, object]
    args_tree_spec: Optional[PyTreeSpec] = None

    # the output sharding info
    output_sharding: Optional[OutputSharding] = None


@dataclass
class PlacementStrategy:
    """
    A placement strategy describes an acceptable sharding placements of the output
    and the tensor arguments of an operation.
    """

    output_spec: DTensorSpec
    input_specs: Optional[Sequence[DTensorSpec]] = None

    # redistribute costs for this op placement strategy
    # we need a nested list to record the cost for each
    # operand of this operator, and for each operand of
    # this operator it might have multiple placement strategies
    redistribute_cost: Optional[List[List[float]]] = None

    def pretty_print_placements(self, placements):
        return "".join([str(p) for p in placements])

    def __str__(self) -> str:
        if self.input_specs is None:
            input_specs_str = ""
        else:
            input_specs_str = (
                "(" + ", ".join([self.pretty_print_placements(spec.placements) for spec in self.input_specs]) + ") -> "
            )
        output_spec_str = self.pretty_print_placements(self.output_spec.placements)
        return f"{input_specs_str}{output_spec_str}"


class StrategyType:
    """
    Base class type for op strategy, We have two StrategyType:
        OpStrategy and TupleStrategy
    """

    pass


class OpStrategy(StrategyType):
    """
    OpStrategy that consists of a list of placement strategies associated with the op
    """

    def __init__(self, strategies: List[PlacementStrategy]) -> None:
        super().__init__()
        self.strategies: List[PlacementStrategy] = strategies

    def __str__(self) -> str:
        strategy_list_str = ", ".join([str(strategy) for strategy in self.strategies])
        mesh_shape = self.strategies[0].output_spec.mesh.shape
        return f"OpStrategy:[{strategy_list_str}] @mesh: {mesh_shape}"

    def max_num_shards(self) -> int:
        """
        Returns the max number of shards across all placement strategies
        """
        return max([strategy.output_spec.num_shards for strategy in self.strategies])

    @property
    def output_shape(self):
        return self.strategies[0].output_spec.shape

    @property
    def output_ndim(self):
        return self.strategies[0].output_spec.ndim

    @property
    def output_stride(self):
        if self.strategies[0].output_spec.tensor_meta is not None:
            return self.strategies[0].output_spec.tensor_meta.stride
        else:
            return None

    @property
    def output_dtype(self):
        if self.strategies[0].output_spec.tensor_meta is not None:
            return self.strategies[0].output_spec.tensor_meta.dtype
        else:
            return None


class TupleStrategy(StrategyType):
    """
    TupleStrategy represents the output strategy of this op is a tuple
    of strategy, i.e. If the output of this op is a tuple of tensors or list of tensors
    with possibly different placement strategies, we should return a TupleStrategy that
    contains a tuple of OpStrategy.

    NOTE: if the output of the op is a List[Tensor] and they share the same placement
    strategy, then we should return a single OpStrategy instead of a TupleStrategy
    """

    def __init__(self, childs: Sequence[StrategyType]) -> None:
        super().__init__()
        self.childs: Sequence[StrategyType] = childs

    def __str__(self) -> str:
        child_strategies_str = ", ".join([f"{str(strat)}" for idx, strat in enumerate(self.childs)])
        return f"TupleStrategy({child_strategies_str})"
