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

from common_dtensor import DTensorTestBase, with_comms

import torch
import torch.distributed.distributed_c10d as c10d
from torch.testing._internal.common_utils import run_tests

from vescale.dtensor.device_mesh import DeviceMesh
from vescale.dtensor.op_schema import OpSchema, RuntimeSchemaInfo
from vescale.dtensor.placement_types import DTensorSpec, Partial, Replicate, Shard, TensorMeta, InterleavedShard

""" Python's Agreement: Equal objects must have equal hash! """


class TestDeviceMesh(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @with_comms
    def test_device_mesh(self):
        mesh = DeviceMesh(self.device_type, torch.arange(self.world_size).reshape(2, -1))
        self.assertEqual(hash(mesh), hash(mesh))
        self.assertTrue(mesh == mesh)

        non_mesh = torch.arange(self.world_size)
        self.assertNotEqual(hash(mesh), hash(non_mesh))
        self.assertFalse(mesh == non_mesh)

        mesh2 = DeviceMesh(self.device_type, torch.arange(self.world_size).reshape(2, -1))
        self.assertEqual(hash(mesh), hash(mesh2))
        self.assertTrue(mesh == mesh2)

        mesh3 = DeviceMesh(self.device_type, torch.arange(self.world_size))
        self.assertNotEqual(hash(mesh), hash(mesh3))
        self.assertFalse(mesh == mesh3)


class TestPlacements(DTensorTestBase):
    """
    Test Placement Goal:
    - hash(<placement>) must be different
    - hash(tuple(<placement>,)) must be different
    - hash(tuple(<placement>, <placement>, ...)) must be different
    """

    def test_placements(self):
        # Shard(>=0)
        shard0, shard00, shard1 = Shard(0), Shard(0), Shard(1)

        self.assertEqual(hash(shard0), hash(shard0))
        self.assertTrue(shard0 == shard0)

        self.assertEqual(hash(shard0), hash(shard00))
        self.assertTrue(shard0 == shard00)

        self.assertNotEqual(hash(shard0), hash(shard1))
        self.assertFalse(shard0 == shard1)

        # Shard(<0)
        shard_1, shard_2, shard_3 = (
            Shard(-1),
            Shard(-2),
            Shard(-3),
        )
        self.assertNotEqual(hash(shard_1), hash(shard_2))
        self.assertFalse(shard_1 == shard_2)
        self.assertNotEqual(hash(shard_2), hash(shard_3))
        self.assertFalse(shard_2 == shard_3)
        self.assertNotEqual(hash(shard_3), hash(shard_1))
        self.assertFalse(shard_3 == shard_1)

        # Replicate
        replicate, replicate2 = Replicate(), Replicate()

        self.assertEqual(hash(replicate), hash(replicate))
        self.assertTrue(replicate == replicate)

        self.assertEqual(hash(replicate), hash(replicate2))
        self.assertTrue(replicate == replicate2)

        # Partial
        partial_sum, partial_sum2, partial_avg = Partial(), Partial(), Partial(reduce_op=c10d.ReduceOp.AVG)

        self.assertEqual(hash(partial_sum), hash(partial_sum))
        self.assertTrue(partial_sum == partial_sum)

        self.assertEqual(hash(partial_sum), hash(partial_sum2))
        self.assertTrue(partial_sum == partial_sum2)

        self.assertNotEqual(hash(partial_sum), hash(partial_avg))
        self.assertFalse(partial_sum == partial_avg)

        # InterleavedShard
        interleaved_shard, interleaved_shard2 = InterleavedShard(0, 2), InterleavedShard(0, 2)
        interleaved_shard3, interleaved_shard4 = InterleavedShard(0, 4), InterleavedShard(1, 2)

        self.assertEqual(hash(interleaved_shard), hash(interleaved_shard))
        self.assertTrue(interleaved_shard == interleaved_shard)

        self.assertEqual(hash(interleaved_shard), hash(interleaved_shard2))
        self.assertTrue(interleaved_shard == interleaved_shard2)

        self.assertNotEqual(hash(interleaved_shard), hash(interleaved_shard3))
        self.assertFalse(interleaved_shard == interleaved_shard3)

        self.assertNotEqual(hash(interleaved_shard), hash(interleaved_shard4))
        self.assertFalse(interleaved_shard == interleaved_shard4)

    def test_placements_cross_comparison(self):
        # hash should have no collision
        all_hash_values = []
        for dim in range(-11, 11):
            all_hash_values.append(hash(Shard(dim)))
        all_hash_values.append(hash(Replicate()))
        for op in (
            c10d.ReduceOp.SUM,
            c10d.ReduceOp.AVG,
            c10d.ReduceOp.PRODUCT,
            c10d.ReduceOp.MIN,
            c10d.ReduceOp.MAX,
            c10d.ReduceOp.BAND,
            c10d.ReduceOp.BOR,
            c10d.ReduceOp.BXOR,
            c10d.ReduceOp.PREMUL_SUM,
        ):
            all_hash_values.append(hash(Partial(op)))
        for dim in range(11):
            for interleaved_size in range(2, 5):
                all_hash_values.append(hash(InterleavedShard(dim, interleaved_size)))
        self.assertEqual(len(all_hash_values), len(set(all_hash_values)), msg="hash values must differ")

        # equal must be False
        shard = Shard(0)
        replicate = Replicate()
        partial = Partial()
        interleaved_shard = InterleavedShard(0, 2)
        self.assertFalse(shard == replicate)
        self.assertFalse(shard == partial)
        self.assertFalse(shard == interleaved_shard)
        self.assertFalse(replicate == partial)
        self.assertFalse(replicate == interleaved_shard)
        self.assertFalse(partial == interleaved_shard)

    def test_placements_1d(self):
        # Shard(>=0)
        shard0, shard00, shard1 = (Shard(0),), (Shard(0),), (Shard(1),)

        self.assertEqual(hash(shard0), hash(shard0))
        self.assertTrue(shard0 == shard0)

        self.assertEqual(hash(shard0), hash(shard00))
        self.assertTrue(shard0 == shard00)

        self.assertNotEqual(hash(shard0), hash(shard1))
        self.assertFalse(shard0 == shard1)

        # Shard(<0)
        shard_1, shard_2, shard_3 = (Shard(-1),), (Shard(-2),), (Shard(-3),)
        self.assertNotEqual(hash(shard_1), hash(shard_2))
        self.assertFalse(shard_1 == shard_2)
        self.assertNotEqual(hash(shard_2), hash(shard_3))
        self.assertFalse(shard_2 == shard_3)
        self.assertNotEqual(hash(shard_3), hash(shard_1))
        self.assertFalse(shard_3 == shard_1)

        # Replicate
        replicate, replicate2 = (Replicate(),), (Replicate(),)

        self.assertEqual(hash(replicate), hash(replicate))
        self.assertTrue(replicate == replicate)

        self.assertEqual(hash(replicate), hash(replicate2))
        self.assertTrue(replicate == replicate2)

        # Partial
        partial_sum, partial_sum2, partial_avg = (Partial(),), (Partial(),), (Partial(reduce_op=c10d.ReduceOp.AVG),)

        self.assertEqual(hash(partial_sum), hash(partial_sum))
        self.assertTrue(partial_sum == partial_sum)

        self.assertEqual(hash(partial_sum), hash(partial_sum2))
        self.assertTrue(partial_sum == partial_sum2)

        self.assertNotEqual(hash(partial_sum), hash(partial_avg))
        self.assertFalse(partial_sum == partial_avg)

        # InterleavedShard
        interleaved_shard, interleaved_shard2 = (InterleavedShard(0, 2),), (InterleavedShard(0, 2),)
        interleaved_shard3, interleaved_shard4 = (InterleavedShard(0, 4)), (InterleavedShard(1, 2),)

        self.assertEqual(hash(interleaved_shard), hash(interleaved_shard))
        self.assertTrue(interleaved_shard == interleaved_shard)

        self.assertEqual(hash(interleaved_shard), hash(interleaved_shard2))
        self.assertTrue(interleaved_shard == interleaved_shard2)

        self.assertNotEqual(hash(interleaved_shard), hash(interleaved_shard3))
        self.assertFalse(interleaved_shard == interleaved_shard3)

        self.assertNotEqual(hash(interleaved_shard), hash(interleaved_shard4))
        self.assertFalse(interleaved_shard == interleaved_shard4)

    def test_placements_1d_cross_comparison(self):
        # hash should have no collision
        all_hash_values = []
        for dim in range(-11, 11):
            all_hash_values.append(hash((Shard(dim),)))
        all_hash_values.append(hash((Replicate(),)))
        for op in (
            c10d.ReduceOp.SUM,
            c10d.ReduceOp.AVG,
            c10d.ReduceOp.PRODUCT,
            c10d.ReduceOp.MIN,
            c10d.ReduceOp.MAX,
            c10d.ReduceOp.BAND,
            c10d.ReduceOp.BOR,
            c10d.ReduceOp.BXOR,
            c10d.ReduceOp.PREMUL_SUM,
        ):
            all_hash_values.append(hash((Partial(op),)))
        for dim in range(11):
            for interleaved_size in range(2, 5):
                all_hash_values.append(hash((InterleavedShard(dim, interleaved_size),)))
        self.assertEqual(len(all_hash_values), len(set(all_hash_values)), msg="hash values must differ")

        # equal must be False
        shard = (Shard(0),)
        replicate = (Replicate(),)
        partial = (Partial(),)
        interleaved_shard = (InterleavedShard(0, 2),)
        self.assertFalse(shard == replicate)
        self.assertFalse(shard == partial)
        self.assertFalse(shard == interleaved_shard)
        self.assertFalse(replicate == partial)
        self.assertFalse(replicate == interleaved_shard)
        self.assertFalse(partial == interleaved_shard)

    def test_placements_4d(self):
        # with Shard(>=0)
        placements = (Shard(0), Shard(1), Replicate(), Partial())
        self.assertEqual(hash(placements), hash(placements))
        self.assertTrue(placements == placements)

        placements2 = (Shard(0), Shard(1), Replicate(), Partial())
        self.assertEqual(hash(placements), hash(placements2))
        self.assertTrue(placements == placements2)

        placements3 = (Shard(2), Shard(1), Replicate(), Partial())
        self.assertNotEqual(hash(placements), hash(placements3))
        self.assertFalse(placements == placements3)

        placements4 = (Replicate(), Shard(1), Replicate(), Partial())
        self.assertNotEqual(hash(placements), hash(placements4))
        self.assertFalse(placements == placements4)

        placements5 = (Shard(0), Shard(1), Replicate())
        self.assertNotEqual(hash(placements), hash(placements5))
        self.assertFalse(placements == placements5)

        # with Shard(<0)
        placements_1 = (Shard(-1), Shard(1), Replicate(), Partial())
        placements_2 = (Shard(-2), Shard(1), Replicate(), Partial())
        placements_3 = (Shard(-3), Shard(1), Replicate(), Partial())
        self.assertNotEqual(hash(placements_1), hash(placements_2))
        self.assertFalse(placements_1 == placements_2)
        self.assertNotEqual(hash(placements_2), hash(placements_3))
        self.assertFalse(placements_2 == placements_3)
        self.assertNotEqual(hash(placements_3), hash(placements_1))
        self.assertFalse(placements_3 == placements_1)

        # with InterleavedShard
        placements6 = (InterleavedShard(0, 2), Shard(1), Replicate(), Partial())
        self.assertEqual(hash(placements6), hash(placements6))
        self.assertTrue(placements6 == placements6)

        placements7 = (InterleavedShard(0, 2), Shard(1), Replicate(), Partial())
        self.assertEqual(hash(placements6), hash(placements7))
        self.assertTrue(placements6 == placements7)

        placements8 = (InterleavedShard(0, 4), Shard(1), Replicate(), Partial())
        self.assertNotEqual(hash(placements6), hash(placements8))
        self.assertFalse(placements6 == placements8)

        placements9 = (InterleavedShard(1, 2), Shard(1), Replicate(), Partial())
        self.assertNotEqual(hash(placements6), hash(placements9))
        self.assertFalse(placements6 == placements9)

    def test_placements_4d_cross_comparison(self):
        # hash should have no collision
        all_hash_values = []
        for dim in range(-11, 11):
            all_hash_values.append(hash((Shard(dim), Shard(11), Replicate(), Partial())))
        for dim in range(11):
            for interleaved_size in range(2, 5):
                all_hash_values.append(
                    hash((InterleavedShard(dim, interleaved_size), Shard(11), Replicate(), Partial()))
                )
        self.assertEqual(len(all_hash_values), len(set(all_hash_values)), msg="hash values must differ")


class TestTensorMeta(DTensorTestBase):
    def test_tensormeta(self):
        tm = TensorMeta(torch.Size([2, 2]), (2, 1), torch.float32)
        self.assertEqual(hash(tm), hash(tm))
        self.assertTrue(tm == tm)

        tm_fake = (torch.Size([2, 2]), (2, 1), torch.float32)
        self.assertEqual(hash(tm), hash(tm_fake))
        self.assertFalse(tm == tm_fake)

        tm2 = TensorMeta(torch.Size([2, 2]), (2, 1), torch.float32)
        self.assertEqual(hash(tm), hash(tm2))
        self.assertTrue(tm == tm2)

        tm3 = TensorMeta(torch.Size([2, 4]), (2, 1), torch.float32)
        self.assertNotEqual(hash(tm), hash(tm3))
        self.assertFalse(tm == tm3)

        tm4 = TensorMeta(torch.Size([2, 2]), (4, 1), torch.float32)
        self.assertNotEqual(hash(tm), hash(tm4))
        self.assertFalse(tm == tm4)

        tm5 = TensorMeta(torch.Size([2, 2]), (2, 1), torch.float64)
        self.assertNotEqual(hash(tm), hash(tm5))
        self.assertFalse(tm == tm5)


class TestDTensorSpec(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @with_comms
    def test_dtensorspec(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        placements = (Shard(0),)
        tensor_meta = TensorMeta(torch.Size([2, 2]), (2, 1), torch.float32)
        dts = DTensorSpec(device_mesh, placements, tensor_meta=tensor_meta)
        self.assertEqual(hash(dts), hash(dts))
        self.assertTrue(dts == dts)

        dts_fake = (device_mesh, placements, tensor_meta)
        self.assertFalse(dts == dts_fake)

        device_mesh2 = DeviceMesh(self.device_type, list(range(self.world_size)))
        placements2 = (Shard(0),)
        tensor_meta2 = TensorMeta(torch.Size([2, 2]), (2, 1), torch.float32)
        dts2 = DTensorSpec(device_mesh2, placements2, tensor_meta=tensor_meta2)
        self.assertEqual(hash(dts), hash(dts2))
        self.assertTrue(dts == dts2)

        dts3 = DTensorSpec(
            DeviceMesh(self.device_type, list(range(self.world_size // 2))), placements, tensor_meta=tensor_meta
        )
        self.assertNotEqual(hash(dts), hash(dts3))
        self.assertFalse(dts == dts3)

        dts4 = DTensorSpec(device_mesh, (Shard(1),), tensor_meta=tensor_meta)
        self.assertNotEqual(hash(dts), hash(dts4))
        self.assertFalse(dts == dts4)

        dts5 = DTensorSpec(device_mesh, placements, tensor_meta=TensorMeta(torch.Size([4, 2]), (2, 1), torch.float32))
        self.assertNotEqual(hash(dts), hash(dts5))
        self.assertFalse(dts == dts5)

        dts6 = DTensorSpec(device_mesh, placements, tensor_meta=None)
        self.assertNotEqual(hash(dts), hash(dts6))
        self.assertFalse(dts == dts6)

        dts7 = DTensorSpec(device_mesh, placements, tensor_meta=None)
        self.assertEqual(hash(dts6), hash(dts7))
        self.assertTrue(dts6 == dts7)

        # with Shard(<0)
        dts_1 = DTensorSpec(device_mesh, (Shard(-1),), tensor_meta=tensor_meta)
        dts_2 = DTensorSpec(device_mesh, (Shard(-2),), tensor_meta=tensor_meta)
        dts_3 = DTensorSpec(device_mesh, (Shard(-3),), tensor_meta=tensor_meta)
        self.assertNotEqual(hash(dts_1), hash(dts_2))
        self.assertFalse(dts_1 == dts_2)
        self.assertNotEqual(hash(dts_2), hash(dts_3))
        self.assertFalse(dts_2 == dts_3)
        self.assertNotEqual(hash(dts_3), hash(dts_1))
        self.assertFalse(dts_3 == dts_1)

        # with InterleavedShard
        dts8 = DTensorSpec(device_mesh, (InterleavedShard(0, 2),), tensor_meta=tensor_meta)
        dts9 = DTensorSpec(device_mesh, (InterleavedShard(0, 2),), tensor_meta=tensor_meta)
        dts10 = DTensorSpec(device_mesh, (InterleavedShard(0, 4),), tensor_meta=tensor_meta)
        dts11 = DTensorSpec(device_mesh, (InterleavedShard(1, 2),), tensor_meta=tensor_meta)
        self.assertEqual(hash(dts8), hash(dts8))
        self.assertTrue(dts8 == dts8)
        self.assertEqual(hash(dts8), hash(dts9))
        self.assertTrue(dts8 == dts9)
        self.assertNotEqual(hash(dts8), hash(dts10))
        self.assertFalse(dts8 == dts10)
        self.assertNotEqual(hash(dts8), hash(dts11))
        self.assertFalse(dts8 == dts11)


class TestOpSchema(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @with_comms
    def test_opschema(self):
        """
        example-1:
        OpSchema( op=aten.mm.default,
                  args_schema=(DTensorSpec(mesh=DeviceMesh:([0, 1, 2, 3]), placements=(Shard(dim=0),), tensor_meta=TensorMeta(shape=torch.Size([64, 4]), stride=(1, 64), dtype=torch.float32)),
                               DTensorSpec(mesh=DeviceMesh:([0, 1, 2, 3]), placements=(Replicate(),), tensor_meta=TensorMeta(shape=torch.Size([4, 36]), stride=(36, 1), dtype=torch.float32))),
                  kwargs_schema={})

        example-2:
        OpSchema( op=aten.t.default,
                  args_schema=(DTensorSpec(mesh=DeviceMesh:([0, 1, 2, 3]), placements=(Shard(dim=0),), tensor_meta=TensorMeta(shape=torch.Size([64, 36]), stride=(36, 1), dtype=torch.float32)),),
                  kwargs_schema={})
        """
        op = torch.ops.aten.mm.default
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        dts_a = DTensorSpec(
            device_mesh, (Shard(0),), tensor_meta=TensorMeta(torch.Size([64, 4]), (1, 64), torch.float32)
        )
        dts_b = DTensorSpec(
            device_mesh, (Replicate(),), tensor_meta=TensorMeta(torch.Size([4, 36]), (36, 1), torch.float32)
        )
        dts_c = DTensorSpec(
            device_mesh, (Partial(),), tensor_meta=TensorMeta(torch.Size([64, 18]), (18, 1), torch.float32)
        )
        # NOTE: args_schema in args_schema should be a list, not a tuple
        args_schema = (
            [
                dts_a,
            ],
            dts_b,
            1.0,
            True,
            None,
        )
        kwargs_schema = {"dts": dts_c, "scalar": 0.0, "bool": False, "other": None}
        rt_schema = RuntimeSchemaInfo(5, ["dts", "scalar", "bool", "other"])
        op_schema = OpSchema(op, args_schema, kwargs_schema, rt_schema)
        self.assertEqual(hash(op_schema), hash(op_schema))
        self.assertTrue(op_schema == op_schema)

        self.assertFalse(op_schema == (op, args_schema, kwargs_schema, rt_schema))

        op2 = torch.ops.aten.mm.default
        device_mesh2 = DeviceMesh(self.device_type, list(range(self.world_size)))
        dts_a2 = DTensorSpec(
            device_mesh2, (Shard(0),), tensor_meta=TensorMeta(torch.Size([64, 4]), (1, 64), torch.float32)
        )
        dts_b2 = DTensorSpec(
            device_mesh2, (Replicate(),), tensor_meta=TensorMeta(torch.Size([4, 36]), (36, 1), torch.float32)
        )
        dts_c2 = DTensorSpec(
            device_mesh2, (Partial(),), tensor_meta=TensorMeta(torch.Size([64, 18]), (18, 1), torch.float32)
        )
        args_schema2 = (
            [
                dts_a2,
            ],
            dts_b2,
            1.0,
            True,
            None,
        )
        kwargs_schema2 = {"dts": dts_c2, "scalar": 0.0, "bool": False, "other": None}
        rt_schema2 = RuntimeSchemaInfo(5, ["dts", "scalar", "bool", "other"])
        op_schema2 = OpSchema(op2, args_schema2, kwargs_schema2, rt_schema2)
        self.assertEqual(hash(op_schema), hash(op_schema2))
        self.assertTrue(op_schema == op_schema2)

        op_schema3 = OpSchema(torch.ops.aten.t.default, args_schema, kwargs_schema, rt_schema2)
        self.assertNotEqual(hash(op_schema), hash(op_schema3))
        self.assertFalse(op_schema == op_schema3)

        dts_a4 = DTensorSpec(
            device_mesh, (Shard(0),), tensor_meta=TensorMeta(torch.Size([64, 4]), (1, 64), torch.bfloat16)
        )
        op_schema4 = OpSchema(
            op,
            (
                [
                    dts_a4,
                ],
                dts_b,
                1.0,
                True,
                None,
            ),
            kwargs_schema,
            rt_schema,
        )
        self.assertNotEqual(hash(op_schema), hash(op_schema4))
        self.assertFalse(op_schema == op_schema4)

        dts_c5 = DTensorSpec(
            device_mesh, (Partial(),), tensor_meta=TensorMeta(torch.Size([64, 128]), (18, 1), torch.float32)
        )
        op_schema5 = OpSchema(op, args_schema, {"dts": dts_c5, "scalar": 0.0, "bool": False, "other": None}, rt_schema)
        self.assertNotEqual(hash(op_schema), hash(op_schema5))
        self.assertFalse(op_schema == op_schema5)

        # with Shard(<0)
        dts_a_1 = DTensorSpec(device_mesh, (Shard(-1),), tensor_meta=dts_a.tensor_meta)
        dts_a_2 = DTensorSpec(device_mesh, (Shard(-2),), tensor_meta=dts_a.tensor_meta)
        dts_a_3 = DTensorSpec(device_mesh, (Shard(-3),), tensor_meta=dts_a.tensor_meta)
        args_schema_1 = (
            [
                dts_a_1,
            ],
            dts_b,
            1.0,
            True,
            None,
        )
        args_schema_2 = (
            [
                dts_a_2,
            ],
            dts_b,
            1.0,
            True,
            None,
        )
        args_schema_3 = (
            [
                dts_a_3,
            ],
            dts_b,
            1.0,
            True,
            None,
        )
        op_schema_1 = OpSchema(op, args_schema_1, kwargs_schema, rt_schema)
        op_schema_2 = OpSchema(op, args_schema_2, kwargs_schema, rt_schema)
        op_schema_3 = OpSchema(op, args_schema_3, kwargs_schema, rt_schema)
        self.assertNotEqual(hash(op_schema_1), hash(op_schema_2))
        self.assertFalse(op_schema_1 == op_schema_2)
        self.assertNotEqual(hash(op_schema_2), hash(op_schema_3))
        self.assertFalse(op_schema_2 == op_schema_3)
        self.assertNotEqual(hash(op_schema_3), hash(op_schema_1))
        self.assertFalse(op_schema_3 == op_schema_1)

        # with InterleavedShard
        dts_a6 = DTensorSpec(device_mesh, (InterleavedShard(0, 2),), tensor_meta=dts_a.tensor_meta)
        dts_a7 = DTensorSpec(device_mesh, (InterleavedShard(0, 2),), tensor_meta=dts_a.tensor_meta)
        dts_a8 = DTensorSpec(device_mesh, (InterleavedShard(0, 4),), tensor_meta=dts_a.tensor_meta)
        dts_a9 = DTensorSpec(device_mesh, (InterleavedShard(1, 2),), tensor_meta=dts_a.tensor_meta)
        args_schema6 = (
            [
                dts_a6,
            ],
            dts_b,
            1.0,
            True,
            None,
        )
        args_schema7 = (
            [
                dts_a7,
            ],
            dts_b,
            1.0,
            True,
            None,
        )
        args_schema8 = (
            [
                dts_a8,
            ],
            dts_b,
            1.0,
            True,
            None,
        )
        args_schema9 = (
            [
                dts_a9,
            ],
            dts_b,
            1.0,
            True,
            None,
        )
        op_schema6 = OpSchema(op, args_schema6, kwargs_schema, rt_schema)
        op_schema7 = OpSchema(op, args_schema7, kwargs_schema, rt_schema)
        op_schema8 = OpSchema(op, args_schema8, kwargs_schema, rt_schema)
        op_schema9 = OpSchema(op, args_schema9, kwargs_schema, rt_schema)
        self.assertEqual(hash(op_schema6), hash(op_schema6))
        self.assertTrue(op_schema6 == op_schema6)
        self.assertEqual(hash(op_schema6), hash(op_schema7))
        self.assertTrue(op_schema6 == op_schema7)
        self.assertNotEqual(hash(op_schema6), hash(op_schema8))
        self.assertFalse(op_schema6 == op_schema8)
        self.assertNotEqual(hash(op_schema6), hash(op_schema9))
        self.assertFalse(op_schema6 == op_schema9)


if __name__ == "__main__":
    run_tests()
