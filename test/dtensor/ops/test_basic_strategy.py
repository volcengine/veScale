################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

from common_dtensor import DTensorTestBase, with_comms

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

from vescale import DeviceMesh
from vescale.dtensor.op_schema import DTensorSpec, OpStrategy, PlacementStrategy
from vescale.dtensor.ops.basic_strategy import EinsumDims, gen_einsum_strategies
from vescale.dtensor.placement_types import Partial, Replicate, Shard

aten = torch.ops.aten


class CommonRulesTest(TestCase):
    @property
    def world_size(self) -> int:
        # hard code world size to 4 as we need to test
        # at least with 2d mesh
        return 4

    def test_batch_dims(self):
        equation = "abc,abc->abc"
        input_dims, output_dim = EinsumDims.parse_equation(equation)
        edims = EinsumDims.parse_dims(input_dims, output_dim)

        self.assertEqual(edims.batch_dims, ["a", "b", "c"])
        self.assertEqual(edims.contracting_dims, [])
        self.assertEqual(edims.lhs_out_only_dims, [])
        self.assertEqual(edims.rhs_out_only_dims, [])

    def test_mm_dims(self):
        equation = "mk,kn->mn"
        input_dims, output_dim = EinsumDims.parse_equation(equation)
        edims = EinsumDims.parse_dims(input_dims, output_dim)

        self.assertEqual(edims.batch_dims, [])
        self.assertEqual(edims.contracting_dims, ["k"])
        self.assertEqual(edims.lhs_out_only_dims, ["m"])
        self.assertEqual(edims.rhs_out_only_dims, ["n"])

    def test_bmm_dims(self):
        equation = "bmk,bkn->bmn"
        input_dims, output_dim = EinsumDims.parse_equation(equation)
        edims = EinsumDims.parse_dims(input_dims, output_dim)

        self.assertEqual(edims.batch_dims, ["b"])
        self.assertEqual(edims.contracting_dims, ["k"])
        self.assertEqual(edims.lhs_out_only_dims, ["m"])
        self.assertEqual(edims.rhs_out_only_dims, ["n"])

        equation = "bcmk,bckn->bcmn"
        input_dims, output_dim = EinsumDims.parse_equation(equation)
        edims = EinsumDims.parse_dims(input_dims, output_dim)

        self.assertEqual(edims.batch_dims, ["b", "c"])
        self.assertEqual(edims.contracting_dims, ["k"])
        self.assertEqual(edims.lhs_out_only_dims, ["m"])
        self.assertEqual(edims.rhs_out_only_dims, ["n"])

    def test_free_dims(self):
        equation = "abc,ab->abc"
        input_dims, output_dim = EinsumDims.parse_equation(equation)
        edims = EinsumDims.parse_dims(input_dims, output_dim)

        self.assertEqual(edims.batch_dims, ["a", "b"])
        self.assertEqual(edims.contracting_dims, [])
        self.assertEqual(edims.lhs_out_only_dims, ["c"])
        self.assertEqual(edims.rhs_out_only_dims, [])

        equation = "abd,bf->abfd"
        input_dims, output_dim = EinsumDims.parse_equation(equation)
        edims = EinsumDims.parse_dims(input_dims, output_dim)

        self.assertEqual(edims.batch_dims, ["b"])
        self.assertEqual(edims.contracting_dims, [])
        self.assertEqual(edims.lhs_out_only_dims, ["a", "d"])
        self.assertEqual(edims.rhs_out_only_dims, ["f"])


class TestEinsumStrategies(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    def test_mm_1d_mesh(self):
        mesh = self.build_device_mesh()

        strategy_list = (
            ([Shard(1)], [Shard(0)], [Partial()]),
            ([Shard(0)], [Replicate()], [Shard(0)]),
            ([Replicate()], [Shard(1)], [Shard(1)]),
            ([Replicate()], [Replicate()], [Replicate()]),
        )

        for strategy in strategy_list:
            lhs, rhs, out = strategy
            l_p = PlacementStrategy(output_spec=DTensorSpec(mesh, tuple(lhs)))
            r_p = PlacementStrategy(output_spec=DTensorSpec(mesh, tuple(rhs)))
            o_p = PlacementStrategy(
                output_spec=DTensorSpec(mesh, tuple(out)),
                input_specs=[DTensorSpec(mesh, tuple(lhs)), DTensorSpec(mesh, tuple(rhs))],
            )
            mat1 = OpStrategy([l_p])
            mat2 = OpStrategy([r_p])
            out = OpStrategy([o_p])

            _out = gen_einsum_strategies("mk,kn->mn", mesh, mat1=mat1, mat2=mat2)

            self.assertEqual(_out.strategies, out.strategies)

    @with_comms
    def test_mm_2d_mesh(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size).reshape(2, 2))

        strategy_list = (
            ([Shard(0), Replicate()], [Replicate(), Replicate()], [Shard(0), Replicate()]),
            ([Replicate(), Replicate()], [Shard(1), Replicate()], [Shard(1), Replicate()]),
            ([Replicate(), Replicate()], [Replicate(), Replicate()], [Replicate(), Replicate()]),
            # TODO(cery.di) : support 2d/3d mesh strategy mapping
            ([Replicate(), Shard(1)], [Replicate(), Shard(0)], [Replicate(), Partial()]),
        )

        for strategy in strategy_list:
            lhs, rhs, out = strategy
            l_p = PlacementStrategy(output_spec=DTensorSpec(mesh, tuple(lhs)))
            r_p = PlacementStrategy(output_spec=DTensorSpec(mesh, tuple(rhs)))
            o_p = PlacementStrategy(
                output_spec=DTensorSpec(mesh, tuple(out)),
                input_specs=[DTensorSpec(mesh, tuple(lhs)), DTensorSpec(mesh, tuple(rhs))],
            )
            mat1 = OpStrategy([l_p])
            mat2 = OpStrategy([r_p])
            out = OpStrategy([o_p])

            _out = gen_einsum_strategies("mk,kn->mn", mesh, mat1=mat1, mat2=mat2)

            self.assertEqual(_out.strategies, out.strategies)

    @with_comms
    def test_bmm_1d_mesh(self):
        mesh = self.build_device_mesh()

        strategy_list = (
            ([Shard(0)], [Shard(0)], [Shard(0)]),
            ([Shard(2)], [Shard(1)], [Partial()]),
        )

        for strategy in strategy_list:
            lhs, rhs, out = strategy
            l_p = PlacementStrategy(output_spec=DTensorSpec(mesh, tuple(lhs)))
            r_p = PlacementStrategy(output_spec=DTensorSpec(mesh, tuple(rhs)))
            o_p = PlacementStrategy(
                output_spec=DTensorSpec(mesh, tuple(out)),
                input_specs=[DTensorSpec(mesh, tuple(lhs)), DTensorSpec(mesh, tuple(rhs))],
            )
            mat1 = OpStrategy([l_p])
            mat2 = OpStrategy([r_p])
            out = OpStrategy([o_p])

            _out = gen_einsum_strategies("bmk,bkn->bmn", mesh, mat1=mat1, mat2=mat2)

            self.assertEqual(_out.strategies, out.strategies)

    @with_comms
    def test_bmm_2d_mesh(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size).reshape(2, 2))

        strategy_list = (
            ([Shard(0), Shard(2)], [Shard(0), Shard(1)], [Shard(0), Partial()]),
            ([Shard(0), Shard(1)], [Shard(0)], [Shard(0), Shard(1)]),
            ([Replicate(), Shard(2)], [Replicate(), Shard(1)], [Replicate(), Partial()]),
        )

        for strategy in strategy_list:
            lhs, rhs, out = strategy
            l_p = PlacementStrategy(output_spec=DTensorSpec(mesh, tuple(lhs)))
            r_p = PlacementStrategy(output_spec=DTensorSpec(mesh, tuple(rhs)))
            o_p = PlacementStrategy(
                output_spec=DTensorSpec(mesh, tuple(out)),
                input_specs=[DTensorSpec(mesh, tuple(lhs)), DTensorSpec(mesh, tuple(rhs))],
            )
            mat1 = OpStrategy([l_p])
            mat2 = OpStrategy([r_p])
            out = OpStrategy([o_p])

            _out = gen_einsum_strategies("bmk,bkn->bmn", mesh, mat1=mat1, mat2=mat2)

            self.assertEqual(_out.strategies, out.strategies)

    @with_comms
    def test_pointwise_1d_mesh(self):
        mesh = self.build_device_mesh()
        strategy_list = (
            ([Shard(0)], [Shard(0)], [Shard(0)]),
            ([Shard(1)], [Shard(1)], [Shard(1)]),
            ([Replicate()], [Replicate()], [Replicate()]),
        )

        for strategy in strategy_list:
            lhs, rhs, out = strategy
            l_p = PlacementStrategy(output_spec=DTensorSpec(mesh, tuple(lhs)))
            r_p = PlacementStrategy(output_spec=DTensorSpec(mesh, tuple(rhs)))
            o_p = PlacementStrategy(
                output_spec=DTensorSpec(mesh, tuple(out)),
                input_specs=[DTensorSpec(mesh, tuple(lhs)), DTensorSpec(mesh, tuple(rhs))],
            )
            mat1 = OpStrategy([l_p])
            mat2 = OpStrategy([r_p])
            out = OpStrategy([o_p])

            _out = gen_einsum_strategies("abcd,abcd->abcd", mesh, mat1=mat1, mat2=mat2)
            self.assertEqual(_out.strategies, out.strategies)

        strategy_list = (
            ([Shard(0)], [Shard(1)], [Shard(1)]),
            ([Replicate()], [Shard(0)], [Shard(0)]),
        )

        for strategy in strategy_list:
            lhs, rhs, out = strategy
            l_p = PlacementStrategy(output_spec=DTensorSpec(mesh, tuple(lhs)))
            r_p = PlacementStrategy(output_spec=DTensorSpec(mesh, tuple(rhs)))
            o_p = PlacementStrategy(
                output_spec=DTensorSpec(mesh, tuple(out)),
                input_specs=[DTensorSpec(mesh, tuple(lhs)), DTensorSpec(mesh, tuple(rhs))],
            )
            mat1 = OpStrategy([l_p])
            mat2 = OpStrategy([r_p])
            out = OpStrategy([o_p])
            _out = gen_einsum_strategies("bcd,abcd->abcd", mesh, mat1=mat1, mat2=mat2)
            self.assertEqual(_out.strategies, out.strategies)

    @with_comms
    def test_linearity_1d_mesh(self):
        mesh = self.build_device_mesh()
        strategy_list = (([Partial()], [Partial()], [Partial()]),)

        for strategy in strategy_list:
            lhs, rhs, out = strategy
            l_p = PlacementStrategy(output_spec=DTensorSpec(mesh, tuple(lhs)))
            r_p = PlacementStrategy(output_spec=DTensorSpec(mesh, tuple(rhs)))
            o_p = PlacementStrategy(
                output_spec=DTensorSpec(mesh, tuple(out)),
                input_specs=[DTensorSpec(mesh, tuple(lhs)), DTensorSpec(mesh, tuple(rhs))],
            )
            mat1 = OpStrategy([l_p])
            mat2 = OpStrategy([r_p])
            out = OpStrategy([o_p])

            _out = gen_einsum_strategies("abcd,abcd->abcd", mesh, mat1=mat1, mat2=mat2, linearity=True)
            self.assertEqual(_out.strategies, out.strategies)


if __name__ == "__main__":
    run_tests()
