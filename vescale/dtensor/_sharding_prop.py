# mypy: allow-untyped-defs
################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

import torch
from torch.distributed.tensor._sharding_prop import ShardingPropagator as TorchShardingPropagator
from torch.distributed.tensor import DTensor as TorchDTensor

from vescale.dtensor._op_schema import (
    OpSchema,
    OpStrategy,
    OutputSharding,
    OutputSpecType,
    PlacementStrategy,
    TupleStrategy,
)
from vescale.dtensor._utils import compute_local_shape_and_global_offset, compute_local_stride
from vescale.dtensor._dtensor_spec import is_ragged_shard, DTensorSpec, TensorMeta

aten = torch.ops.aten

"""
In this file, our ShardingPropagator inherits from torch ShardingPropagator with two methods.
1. propagate_op_sharding_non_cached is modified to support RaggedShard.
2. _adjust_shape_and_stride_args is kept but unmodified in order to use our compute_local_shape_and_global_offset
"""


class ShardingPropagator(TorchShardingPropagator):
    def __init__(self) -> None:
        super().__init__()
        torch_propagator = TorchDTensor._op_dispatcher.sharding_propagator
        self.op_to_schema_info = {**torch_propagator.op_to_schema_info, **self.op_to_schema_info}
        self.op_to_rules = {**torch_propagator.op_to_rules, **self.op_to_rules}
        self.op_strategy_funcs = {**torch_propagator.op_strategy_funcs, **self.op_strategy_funcs}

    def propagate_op_sharding_non_cached(self, op_schema: OpSchema) -> OutputSharding:
        """
        Propagate the sharding for an operator given the op_schema.
        """
        # special case op, we don't need to propagate for local
        # scalar. TODO: figure out a better way to handle this
        if op_schema.op is aten._local_scalar_dense.default:
            return OutputSharding(None, op_schema)

        out_tensor_meta = self._propagate_tensor_meta_non_cached(op_schema)

        if op_schema.op in self.op_strategy_funcs:
            # wrap the op_schema with op strategy for sharding strategy propagation
            strategy_schema = self._wrap_with_op_strategy(op_schema)

            # run sharding strategy propagation/generation
            op_strategy = self.op_strategy_funcs[op_schema.op](strategy_schema)

            if isinstance(op_strategy, OpStrategy):
                # single Op strategy
                output_strategy = self._select_strategy(op_strategy)

                # check if we need to redistribute the input
                needs_redistribute = False
                expected_input_specs: list[DTensorSpec] = []

                # in case where the op does not specify input_specs and output_specs
                # is a DTensorSpec, we use output_specs as the spec for each DTensor
                # input arg.
                if output_strategy.input_specs is None:
                    assert isinstance(output_strategy.output_specs, DTensorSpec)

                for idx, input_spec in enumerate(op_schema.args_spec):
                    desired_spec = (
                        output_strategy.output_spec
                        if output_strategy.input_specs is None
                        else output_strategy.input_specs[idx]
                    )
                    expected_input_specs.append(desired_spec.shallow_copy_with_tensor_meta(input_spec.tensor_meta))
                    if input_spec.placements != desired_spec.placements:
                        needs_redistribute = True

                suggestion_schema = None
                if needs_redistribute:
                    suggestion_schema = OpSchema(op_schema.op, tuple(expected_input_specs), {})
                    suggestion_schema._inplace_rewrap_schema_suggestion(op_schema)

                # shape and stride args need to be modified for
                # view ops and new factory ops, potentially
                if op_schema.op in self.op_to_shape_and_stride_idx:
                    assert isinstance(output_strategy.output_spec, DTensorSpec)
                    # It happens when the output has the same shape as the input
                    # and the input placements are not all Replicate().
                    if output_strategy.output_spec.is_sharded() or is_ragged_shard(output_strategy.output_spec):
                        schema = suggestion_schema or op_schema
                        assert isinstance(out_tensor_meta, TensorMeta)
                        suggestion_schema = self._adjust_shape_and_stride_args(
                            out_tensor_meta, schema, output_strategy.output_spec
                        )
                        needs_redistribute = True

                # construct output spec for the op
                if op_schema.return_type_tuple_tensor_like():
                    # for ops that return multiple tensors and the output_specs is not
                    # a tuple, we use a tuple of that single output spec as the new
                    # output_specs
                    output_specs: OutputSpecType = output_strategy.output_specs
                    if isinstance(output_specs, DTensorSpec):
                        output_specs = tuple(
                            [
                                # create a new DTensorSpec with the same placement as the
                                # output_specs in output_strategy
                                DTensorSpec(
                                    mesh=output_specs.mesh,
                                    placements=output_specs.placements,
                                    tensor_meta=output_specs.tensor_meta,
                                )
                                for _ in range(len(op_schema.op._schema.returns))
                            ]
                        )
                elif op_schema.return_type_tensor():
                    output_specs = output_strategy.output_specs
                else:
                    output_specs = None

                output_sharding = OutputSharding(
                    output_specs,
                    suggestion_schema,
                    needs_redistribute=needs_redistribute,
                )
            elif isinstance(op_strategy, TupleStrategy):
                # tuple strategy output sharding processing
                # runtime selected placement strategy for each TupleStrategy input arg
                selected_strategies: list[PlacementStrategy] = []
                out_spec_list: list[DTensorSpec] = []
                for strategy in op_strategy.childs:
                    assert isinstance(strategy, OpStrategy)
                    selected_strategy = self._select_strategy(strategy)
                    selected_strategies.append(selected_strategy)
                    out_spec_list.append(selected_strategy.output_spec)

                needs_redistribute = False
                suggestion_args: list[object] = []
                tensor_or_list_tensor_arg_idx = 0

                for arg in op_schema.args_schema:
                    if arg and isinstance(arg, (list, tuple)) and isinstance(arg[0], DTensorSpec):
                        expected_input_spec_list: list[DTensorSpec] = []
                        for idx, arg_spec in enumerate(arg):
                            expected_input_spec = selected_strategies[idx].input_spec(tensor_or_list_tensor_arg_idx)
                            expected_input_spec = expected_input_spec.shallow_copy_with_tensor_meta(
                                arg_spec.tensor_meta
                            )
                            if arg_spec.placements != expected_input_spec.placements:
                                needs_redistribute = True
                            expected_input_spec_list.append(expected_input_spec)
                        suggestion_args.append(
                            tuple(expected_input_spec_list) if isinstance(arg, tuple) else expected_input_spec_list
                        )
                        tensor_or_list_tensor_arg_idx += 1

                    elif isinstance(arg, DTensorSpec):
                        expected_input_spec = selected_strategies[0].input_spec(tensor_or_list_tensor_arg_idx)
                        expected_input_spec = expected_input_spec.shallow_copy_with_tensor_meta(arg.tensor_meta)
                        if arg.placements != expected_input_spec.placements:
                            needs_redistribute = True
                        suggestion_args.append(expected_input_spec)
                        tensor_or_list_tensor_arg_idx += 1
                    else:
                        suggestion_args.append(arg)

                suggestion_schema = None
                if needs_redistribute:
                    suggestion_schema = OpSchema(op_schema.op, tuple(suggestion_args), op_schema.kwargs_schema)

                output_sharding = OutputSharding(
                    tuple(out_spec_list) if out_tensor_meta is not None else None,
                    suggestion_schema,
                    needs_redistribute=needs_redistribute,
                )
            else:
                raise ValueError("Unsupported op strategy type")

            # associate the output sharding with the output tensor metadata
            self._wrap_output_spec_tensor_meta(op_schema.op, output_sharding.output_spec, out_tensor_meta)
            return output_sharding
        elif op_schema.op in self.op_to_rules:
            # propagate the sharding with rule
            sharding_prop_func = self.op_to_rules[op_schema.op]

            # step 1. there's sharding propagation rule, run
            # sharding propagation to get the output sharding
            try:
                output_sharding = sharding_prop_func(op_schema)
            except NotImplementedError as e:
                raise e
            except Exception as e:
                raise RuntimeError(f"Sharding propagation failed on op {op_schema}.\nError: {e}") from e

            # step 2. if can't get output_spec from sharding
            # propagation (i.e. no rules apply for input
            # placements), we return the output sharding
            # with schema suggestions, which can be used to
            # decide how to do redistribute on inputs
            if output_sharding.output_spec is None:
                if output_sharding.redistribute_schema is None:
                    raise RuntimeError(f"Sharding propagation failed on op {op_schema}!")
                else:
                    # we do auto redistribute on inputs if necessary
                    # run sharding propagation again with suggested schema
                    propagation_res = sharding_prop_func(output_sharding.redistribute_schema)
                    # we set the output sharding with the new propagation result
                    # so that dispatching know both output_spec and redistribute_schema
                    # exist, which indicates a reshard is needed
                    output_sharding.output_spec = propagation_res.output_spec
                    output_sharding.needs_redistribute = True

            # associate the output sharding with the output tensor metadata
            self._wrap_output_spec_tensor_meta(op_schema.op, output_sharding.output_spec, out_tensor_meta)

            return output_sharding
        else:
            raise NotImplementedError(f"Operator {op_schema.op} does not have a sharding strategy registered.")

    def _adjust_shape_and_stride_args(
        self,
        out_tensor_meta: TensorMeta,
        schema: OpSchema,
        spec: DTensorSpec,
    ) -> OpSchema:
        shape_stride_idx = self.op_to_shape_and_stride_idx[schema.op]
        if isinstance(shape_stride_idx, tuple):
            shape_idx, stride_idx = shape_stride_idx
        else:
            shape_idx = shape_stride_idx
            stride_idx = None

        expected_input_schema = list(schema.args_schema)
        # adjust shape to be the same as that of the _local_tensor
        # of the DTensor input arg at index 0, which is inferred
        expected_input_schema[shape_idx], _ = compute_local_shape_and_global_offset(
            out_tensor_meta.shape, spec.mesh, spec.placements
        )

        # adjust the stride arg for aten.new_empty_strided.default
        if stride_idx:
            expected_input_schema[stride_idx] = compute_local_stride(out_tensor_meta.stride, spec.mesh, spec.placements)
        return OpSchema(schema.op, tuple(expected_input_schema), schema.kwargs_schema)
