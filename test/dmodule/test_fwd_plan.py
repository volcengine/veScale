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
from torch import nn

from torch.testing._internal.common_utils import run_tests, instantiate_parametrized_tests, parametrize, subtest

from vescale.dtensor.device_mesh import DeviceMesh
from vescale.dtensor.api import DTensor
from vescale.dtensor.placement_types import Replicate, Shard, Placement
from vescale.dmodule.api import parallelize_module

from common_dtensor import DTensorTestBase, with_comms


class FwdPlanTestBase(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    @property
    def device_mesh(self):
        if getattr(self, "device_mesh_", None) is None:
            self.device_mesh_ = DeviceMesh("cuda", range(self.world_size))
        return self.device_mesh_

    @with_comms
    def run_my_test(self):
        _test_names = [
            func_n for func_n in dir(self) if callable(getattr(self, func_n)) and func_n.startswith("_test_")
        ]
        for t_name in _test_names:
            if t_name == "_test_op":
                continue
            if t_name in dir(super()):
                continue
            f = getattr(self, t_name)
            try:
                f()
            except AssertionError as e:
                print(f"{t_name} failed")
                raise e

    def test(self):
        # save some setup communication
        if type(self) is FwdPlanTestBase:
            return
        self.run_my_test()

    def tearDown(self):
        self.device_mesh_ = None
        super().tearDown()

    def assert_helper(self, out, expected_t):
        self.assertEqual(len(out), len(expected_t))
        for o, e_t in zip(out, expected_t):
            if isinstance(e_t, Placement):
                self.assertIs(type(o), DTensor)
                self.assertEqual(len(o.placements), 1)
                self.assertEqual(o.placements[0], e_t)
            elif e_t is None:
                self.assertIsNone(o)
            else:
                self.assertIs(type(o), e_t)


class FwdPlanTestWNoArgs(FwdPlanTestBase):
    class NoArgs(nn.Module):
        def forward(self):
            return None

    model = NoArgs

    def _test_empty_placements(self):
        fwd_plan = {".input": []}
        dmodule = parallelize_module(self.model(), self.device_mesh, {"parameter": {}, "forward": fwd_plan})
        out = dmodule()
        self.assertIsNone(out)

    def _test_none_placement(self):
        fwd_plan = {".input": [None]}
        dmodule = parallelize_module(self.model(), self.device_mesh, {"parameter": {}, "forward": fwd_plan})
        with self.assertWarns(UserWarning) as _:
            _ = dmodule()

    def _test_incorrect_fwd_plan(self):
        fwd_plan = {".input": [[Shard(0)], [Shard(1)], [Replicate()]]}
        dmodule = parallelize_module(self.model(), self.device_mesh, {"parameter": {}, "forward": fwd_plan})
        with self.assertWarns(UserWarning) as _:
            _ = dmodule()

    def _test_empty_dict(self):
        fwd_plan = {".input": {}}
        dmodule = parallelize_module(self.model(), self.device_mesh, {"parameter": {}, "forward": fwd_plan})
        out = dmodule()
        self.assertIsNone(out)

    def _test_non_empty_dict(self):
        fwd_plan = {".input": {"x": None}}
        dmodule = parallelize_module(self.model(), self.device_mesh, {"parameter": {}, "forward": fwd_plan})
        with self.assertWarns(UserWarning) as _:
            _ = dmodule()

    def _test_non_empty_dict2(self):
        fwd_plan = {".input": {"x": [Shard(0)]}}
        dmodule = parallelize_module(self.model(), self.device_mesh, {"parameter": {}, "forward": fwd_plan})
        with self.assertWarns(UserWarning) as _:
            _ = dmodule()


class FwdPlanTestWSimpleArgs1AndDefaultArgs1(FwdPlanTestBase):
    class SimpleArgs1(nn.Module):
        def forward(self, a):
            return a

    class DefaultArgs1(nn.Module):
        def forward(self, a=None):
            return a

    test_type_candidates = [
        subtest(SimpleArgs1, name="SimpleArgs1"),
        subtest(DefaultArgs1, name="DefaultArgs1"),
    ]

    @with_comms
    @parametrize("model", test_type_candidates)
    def test_seq_fwd_plan(self, model):
        fwd_plan = {".input": [[Shard(0)]]}
        dmodule = parallelize_module(model(), self.device_mesh, {"parameter": {}, "forward": fwd_plan})
        a, b = torch.ones((2, 2)), torch.ones((2, 2)) * 2
        expected_t = [Shard(0)]

        out = dmodule(a)
        self.assert_helper((out,), expected_t)

        out = dmodule(a=a)
        self.assert_helper((out,), expected_t)

        with self.assertRaises(TypeError) as _:
            _ = dmodule(a, b)

        with self.assertRaises(TypeError) as _:
            _ = dmodule(a, b=b)

        with self.assertRaises(TypeError) as _:
            _ = dmodule(b=b, a=a)

    @with_comms
    @parametrize("model", test_type_candidates)
    def test_dict_fwd_plan(self, model):
        fwd_plan = {".input": {"a": [Shard(0)]}}
        dmodule = parallelize_module(model(), self.device_mesh, {"parameter": {}, "forward": fwd_plan})
        a = torch.ones((2, 2))
        expected_t = [Shard(0)]

        out = dmodule(a)
        self.assert_helper((out,), expected_t)

        out = dmodule(a=a)
        self.assert_helper((out,), expected_t)

    @parametrize("model", test_type_candidates)
    def _test_empty_seq_fwd_plan(self, model):
        fwd_plan = {".input": []}
        dmodule = parallelize_module(model(), self.device_mesh, {"parameter": {}, "forward": fwd_plan})
        a, b = torch.ones((2, 2)), torch.ones((2, 2)) * 2
        expected_t = [torch.Tensor]

        out = dmodule(a)
        self.assert_helper((out,), expected_t)

    @parametrize("model", test_type_candidates)
    def _test_empty_dict_fwd_plan(self, model):
        fwd_plan = {".input": {}}
        dmodule = parallelize_module(model(), self.device_mesh, {"parameter": {}, "forward": fwd_plan})
        a, b = torch.ones((2, 2)), torch.ones((2, 2)) * 2
        expected_t = [torch.Tensor]

        out = dmodule(a)
        self.assert_helper((out,), expected_t)

    @parametrize("model", test_type_candidates)
    def _test_incorrect_seq_fwd_plan1(self, model):
        fwd_plan = {".input": {"a": [Shard(0)], "_": None}}

        dmodule = parallelize_module(model(), self.device_mesh, {"parameter": {}, "forward": fwd_plan})
        a, b = torch.ones((2, 2)), torch.ones((2, 2)) * 2
        with self.assertWarns(UserWarning) as _:
            _ = dmodule(a)

    @parametrize("model", test_type_candidates)
    def _test_incorrect_dict_fwd_plan1(self, model):
        fwd_plan = {".input": [[Shard(0)], None]}

        dmodule = parallelize_module(model(), self.device_mesh, {"parameter": {}, "forward": fwd_plan})
        a, b = torch.ones((2, 2)), torch.ones((2, 2)) * 2
        with self.assertWarns(UserWarning) as _:
            _ = dmodule(a)

    @parametrize("model", test_type_candidates)
    def _test_incorrect_seq_fwd_plan2(self, model):
        fwd_plan = {".input": [[Shard(0)], [Shard(1)]]}

        dmodule = parallelize_module(model(), self.device_mesh, {"parameter": {}, "forward": fwd_plan})
        a, b = torch.ones((2, 2)), torch.ones((2, 2)) * 2
        with self.assertWarns(UserWarning) as _:
            _ = dmodule(a)


instantiate_parametrized_tests(FwdPlanTestWSimpleArgs1AndDefaultArgs1)


class FwdPlanTestWKwOnlyArgs1AndDefaultKwOnlyArgs1(FwdPlanTestBase):
    class KwOnlyArgs1(nn.Module):
        def forward(self, *, a):
            return a

    class DefaultKwOnlyArgs1(nn.Module):
        def forward(self, *, a=None):
            return a

    test_type_candidates = [
        subtest(KwOnlyArgs1, name="KwOnlyArgs1"),
        subtest(DefaultKwOnlyArgs1, name="DefaultKwOnlyArgs1"),
    ]

    @with_comms
    @parametrize("model", test_type_candidates)
    def test_seq_fwd_plan(self, model):
        fwd_plan = {".input": [[Shard(0)]]}
        dmodule = parallelize_module(model(), self.device_mesh, {"parameter": {}, "forward": fwd_plan})
        a, b = torch.ones((2, 2)), torch.ones((2, 2)) * 2
        expected_t = [Shard(0)]

        with self.assertRaises(TypeError) as _:
            _ = dmodule(a)

        out = dmodule(a=a)
        self.assert_helper((out,), expected_t)

        with self.assertRaises(TypeError) as _:
            _ = dmodule(a, b)

        with self.assertRaises(TypeError) as _:
            _ = dmodule(a, b=b)

        with self.assertRaises(TypeError) as _:
            _ = dmodule(b=b, a=a)

        if model is self.DefaultKwOnlyArgs1:
            self.assertIsNone(dmodule())

    @with_comms
    @parametrize("model", test_type_candidates)
    def test_dict_fwd_plan(self, model):
        fwd_plan = {".input": {"a": [Shard(0)]}}
        dmodule = parallelize_module(model(), self.device_mesh, {"parameter": {}, "forward": fwd_plan})
        a = torch.ones((2, 2))
        expected_t = [Shard(0)]

        with self.assertRaises(TypeError) as _:
            _ = dmodule(a)

        out = dmodule(a=a)
        self.assert_helper((out,), expected_t)

        if model is self.DefaultKwOnlyArgs1:
            self.assertIsNone(dmodule())

    @parametrize("model", test_type_candidates)
    def _test_empty_seq_fwd_plan(self, model):
        fwd_plan = {".input": []}
        dmodule = parallelize_module(model(), self.device_mesh, {"parameter": {}, "forward": fwd_plan})
        a, b = torch.ones((2, 2)), torch.ones((2, 2)) * 2
        expected_t = [torch.Tensor]

        with self.assertRaises(TypeError) as _:
            _ = dmodule(a)

        out = dmodule(a=a)
        self.assert_helper((out,), expected_t)

        if model is self.DefaultKwOnlyArgs1:
            self.assertIsNone(dmodule())

    @parametrize("model", test_type_candidates)
    def _test_empty_dict_fwd_plan(self, model):
        fwd_plan = {".input": {}}
        dmodule = parallelize_module(model(), self.device_mesh, {"parameter": {}, "forward": fwd_plan})
        a, b = torch.ones((2, 2)), torch.ones((2, 2)) * 2
        expected_t = [torch.Tensor]

        with self.assertRaises(TypeError) as _:
            _ = dmodule(a)

        out = dmodule(a=a)
        self.assert_helper((out,), expected_t)

        if model is self.DefaultKwOnlyArgs1:
            self.assertIsNone(dmodule())

    @parametrize("model", test_type_candidates)
    def _test_incorrect_seq_fwd_plan1(self, model):
        fwd_plan = {".input": [[Shard(0)], None]}

        dmodule = parallelize_module(model(), self.device_mesh, {"parameter": {}, "forward": fwd_plan})
        a, b = torch.ones((2, 2)), torch.ones((2, 2)) * 2
        with self.assertRaises(TypeError) as _:
            _ = dmodule(a)

        with self.assertWarns(UserWarning) as _:
            _ = dmodule(a=a)

    @parametrize("model", test_type_candidates)
    def _test_incorrect_seq_fwd_plan2(self, model):
        fwd_plan = {".input": [[Shard(0)], None]}

        dmodule = parallelize_module(model(), self.device_mesh, {"parameter": {}, "forward": fwd_plan})
        a, b = torch.ones((2, 2)), torch.ones((2, 2)) * 2
        with self.assertRaises(TypeError) as _:
            _ = dmodule(a)

        with self.assertWarns(UserWarning) as _:
            _ = dmodule(a=a)

    @parametrize("model", test_type_candidates)
    def _test_incorrect_seq_fwd_plan3(self, model):
        fwd_plan = {".input": [[Shard(0)], [Shard(1)]]}

        dmodule = parallelize_module(model(), self.device_mesh, {"parameter": {}, "forward": fwd_plan})
        a, b = torch.ones((2, 2)), torch.ones((2, 2)) * 2
        with self.assertRaises(TypeError) as _:
            _ = dmodule(a)

        with self.assertWarns(UserWarning) as _:
            _ = dmodule(a=a)


instantiate_parametrized_tests(FwdPlanTestWKwOnlyArgs1AndDefaultKwOnlyArgs1)


class FwdPlanTestWVarPositionAndSimpleArgs1WithVarPosition(FwdPlanTestBase):
    class VarPosition(nn.Module):
        def forward(self, *args):
            return args

    class SimpleArgs1WithVarPosition(nn.Module):
        def forward(self, a, *args):
            return a, *args

    test_type_candidates = [
        subtest(VarPosition, name="VarPosition"),
        subtest(SimpleArgs1WithVarPosition, name="SimpleArgs1WithVarPosition"),
    ]

    @parametrize("model", test_type_candidates)
    def _test_seq_fwd_plan_no_var_pos(self, model):
        fwd_plan = {".input": [[Shard(0)]]}
        dmodule = parallelize_module(model(), self.device_mesh, {"parameter": {}, "forward": fwd_plan})
        a, b, c = torch.ones((2, 2)), torch.ones((2, 2)) * 2, torch.ones((2, 2)) * 3
        expected_t = [Shard(0)]

        out = dmodule(a)
        self.assert_helper(out, expected_t)

    @parametrize("model", test_type_candidates)
    def _test_seq_fwd_plan(self, model):
        fwd_plan = {".input": [[Shard(0)], [Shard(1)]]}
        dmodule = parallelize_module(model(), self.device_mesh, {"parameter": {}, "forward": fwd_plan})
        a, b, c = torch.ones((2, 2)), torch.ones((2, 2)) * 2, torch.ones((2, 2)) * 3
        expected_t = [Shard(0), Shard(1)]

        out = dmodule(a, b, c)
        self.assert_helper(out, expected_t + [torch.Tensor])

        with self.assertRaises(TypeError) as _:
            out = dmodule(a, b, c=c)

    @parametrize("model", test_type_candidates)
    def _test_dict_fwd_plan(self, model):
        if model is self.VarPosition:
            fwd_plan = {".input": {"args": [[Shard(0)], [Shard(1)]]}}
        else:
            fwd_plan = {".input": {"a": [Shard(0)], "args": [[Shard(1)]]}}
        dmodule = parallelize_module(model(), self.device_mesh, {"parameter": {}, "forward": fwd_plan})
        a, b, c = torch.ones((2, 2)), torch.ones((2, 2)) * 2, torch.ones((2, 2)) * 3
        expected_t = [Shard(0), Shard(1)]

        out = dmodule(a, b, c)
        self.assert_helper(out, expected_t + [torch.Tensor])

        with self.assertWarns(UserWarning) as _:
            _ = dmodule(a)

    @parametrize("model", test_type_candidates)
    def _test_dict_fwd_plan(self, model):
        if model is self.VarPosition:
            fwd_plan = {".input": {"args": [[Shard(0)]]}}
        else:
            fwd_plan = {
                ".input": {
                    "a": [Shard(0)],
                }
            }
        dmodule = parallelize_module(model(), self.device_mesh, {"parameter": {}, "forward": fwd_plan})
        a, b, c = torch.ones((2, 2)), torch.ones((2, 2)) * 2, torch.ones((2, 2)) * 3
        expected_t = [Shard(0)]

        out = dmodule(a)
        self.assert_helper(out, expected_t)

    @parametrize("model", test_type_candidates)
    def _test_mixed_input_type_w_seq_plan(self, model):
        fwd_plan = {".input": [None, [Shard(0)], None, [Shard(1)]]}
        dmodule = parallelize_module(model(), self.device_mesh, {"parameter": {}, "forward": fwd_plan})
        a, b, c = torch.ones((2, 2)), torch.ones((2, 2)) * 2, torch.ones((2, 2)) * 3
        expected_t = [str, Shard(0), float, Shard(1), type(c), type(bool)]
        out = dmodule("string", a, 1.0, b, c, bool)
        self.assert_helper(out, expected_t)

    @parametrize("model", test_type_candidates)
    def _test_mixed_input_type_w_dict_plan(self, model):
        if model is self.VarPosition:
            fwd_plan = {".input": {"args": [None, [Shard(0)], None, [Shard(1)]]}}
        else:
            fwd_plan = {".input": {"a": None, "args": [[Shard(0)], None, [Shard(1)]]}}
        dmodule = parallelize_module(model(), self.device_mesh, {"parameter": {}, "forward": fwd_plan})
        a, b, c = torch.ones((2, 2)), torch.ones((2, 2)) * 2, torch.ones((2, 2)) * 3
        expected_t = [str, Shard(0), float, Shard(1), type(c), type(bool)]
        out = dmodule("string", a, 1.0, b, c, bool)
        self.assert_helper(out, expected_t)


instantiate_parametrized_tests(FwdPlanTestWVarPositionAndSimpleArgs1WithVarPosition)


class FwdPlanTestWVarKeywordAndSimpleArgs1WithVarKeyword(FwdPlanTestBase):
    class VarKeyword(nn.Module):
        def forward(self, **kwargs):
            return kwargs.values()

    class SimpleArgs1WithVarKeyword(nn.Module):
        def forward(self, a, **kwargs):
            return a, *(kwargs.values())

    test_type_candidates = [
        subtest(VarKeyword, name="VarKeyword"),
        subtest(SimpleArgs1WithVarKeyword, name="SimpleArgs1WithVarKeyword"),
    ]

    @parametrize("model", test_type_candidates)
    def _test_seq_fwd_plan_nokwargs(self, model):
        fwd_plan = {".input": [[Shard(0)]]}
        dmodule = parallelize_module(model(), self.device_mesh, {"parameter": {}, "forward": fwd_plan})
        a, b, c = torch.ones((2, 2)), torch.ones((2, 2)) * 2, torch.ones((2, 2)) * 3
        expected_t = [Shard(0)]

        if model is self.SimpleArgs1WithVarKeyword:
            out = dmodule(a)
        else:
            out = dmodule(a=a)
        self.assert_helper(out, expected_t)

    @parametrize("model", test_type_candidates)
    def _test_seq_fwd_plan(self, model):
        fwd_plan = {".input": [[Shard(0)], [Shard(1)]]}
        dmodule = parallelize_module(model(), self.device_mesh, {"parameter": {}, "forward": fwd_plan})
        a, b, c = torch.ones((2, 2)), torch.ones((2, 2)) * 2, torch.ones((2, 2)) * 3
        expected_t = [Shard(0), Shard(1), torch.Tensor]

        if model is self.SimpleArgs1WithVarKeyword:
            out = dmodule(a, b=b, c=c)
        else:
            out = dmodule(a=a, b=b, c=c)
        self.assert_helper(out, expected_t)

    @parametrize("model", test_type_candidates)
    def _test_dict_fwd_plan_nokwarg(self, model):
        fwd_plan = {
            ".input": {
                "a": [Shard(0)],
            }
        }
        dmodule = parallelize_module(model(), self.device_mesh, {"parameter": {}, "forward": fwd_plan})
        a, b, c = torch.ones((2, 2)), torch.ones((2, 2)) * 2, torch.ones((2, 2)) * 3
        expected_t = [Shard(0)]

        out = dmodule(a=a)
        self.assert_helper(out, expected_t)

    @parametrize("model", test_type_candidates)
    def _test_dict_fwd_plan(self, model):
        fwd_plan = {".input": {"a": [Shard(0)], "b": [Shard(1)]}}
        dmodule = parallelize_module(model(), self.device_mesh, {"parameter": {}, "forward": fwd_plan})
        a, b, c = torch.ones((2, 2)), torch.ones((2, 2)) * 2, torch.ones((2, 2)) * 3
        expected_t = [Shard(0), Shard(1), torch.Tensor]

        out = dmodule(a=a, b=b, c=c)
        self.assert_helper(out, expected_t)

        with self.assertWarns(UserWarning) as _:
            _ = dmodule(a=a)

    @parametrize("model", test_type_candidates)
    def _test_mixed_input_type_w_seq_plan(self, model):
        fwd_plan = {".input": [[Shard(0)], None, None, [Shard(1)]]}
        dmodule = parallelize_module(model(), self.device_mesh, {"parameter": {}, "forward": fwd_plan})
        a, b, c = torch.ones((2, 2)), torch.ones((2, 2)) * 2, torch.ones((2, 2)) * 3
        expected_t = [Shard(0), str, float, Shard(1), type(c), type(bool)]
        out = dmodule(a=a, s="string", i=1.0, b=b, c=c, bo=bool)
        self.assert_helper(out, expected_t)

    @parametrize("model", test_type_candidates)
    def _test_mixed_input_type_w_dict_plan(self, model):
        fwd_plan = {".input": {"a": [Shard(0)], "b": [Shard(1)]}}
        dmodule = parallelize_module(model(), self.device_mesh, {"parameter": {}, "forward": fwd_plan})
        a, b, c = torch.ones((2, 2)), torch.ones((2, 2)) * 2, torch.ones((2, 2)) * 3
        expected_t = [Shard(0), str, float, Shard(1), type(c), type(bool)]
        out = dmodule(a=a, s="string", i=1.0, b=b, c=c, bo=bool)
        self.assert_helper(out, expected_t)


instantiate_parametrized_tests(FwdPlanTestWVarKeywordAndSimpleArgs1WithVarKeyword)


class FwdPlanTestWVarPositionDefaultArgs(FwdPlanTestBase):
    class VarPositionDefaultArgs(nn.Module):
        def forward(self, *args, a=None):
            return *args, a

    model = VarPositionDefaultArgs

    def _test_seq_fwd_plan0(self):
        fwd_plan = {".input": [[Shard(0)], [Shard(1)]]}
        dmodule = parallelize_module(self.model(), self.device_mesh, {"parameter": {}, "forward": fwd_plan})
        a, b, c = torch.ones((2, 2)), torch.ones((2, 2)) * 2, torch.ones((2, 2)) * 3
        expected_t = [Shard(0), Shard(1), torch.Tensor]

        out = dmodule(b, c, a=a)
        self.assert_helper(out, expected_t)

    def _test_seq_fwd_plan1(self):
        fwd_plan = {".input": [[Shard(0)], [Shard(1)]]}
        dmodule = parallelize_module(self.model(), self.device_mesh, {"parameter": {}, "forward": fwd_plan})
        a, b, c = torch.ones((2, 2)), torch.ones((2, 2)) * 2, torch.ones((2, 2)) * 3
        expected_t = [Shard(0), Shard(1), torch.Tensor]

        out = dmodule(b, c, a=a)
        self.assert_helper(out, expected_t)

    def _test_dict_fwd_plan(self):
        fwd_plan = {".input": {"a": None, "args": [[Shard(0)], [Shard(1)]]}}
        dmodule = parallelize_module(self.model(), self.device_mesh, {"parameter": {}, "forward": fwd_plan})
        a, b, c = torch.ones((2, 2)), torch.ones((2, 2)) * 2, torch.ones((2, 2)) * 3
        expected_t = [Shard(0), Shard(1), torch.Tensor]

        out = dmodule(b, c, a=a)
        self.assert_helper(out, expected_t)

        with self.assertWarns(UserWarning) as _:
            _ = dmodule(a=a)


class FwdPlanTestWDefaultArgsVarPosition(FwdPlanTestBase):
    class DefaultArgsVarPosition(nn.Module):
        def forward(self, a=None, *args):
            return a, *args

    model = DefaultArgsVarPosition

    def _test_seq_fwd_plan0(self):
        fwd_plan = {".input": [[Shard(0)], [Shard(1)]]}
        dmodule = parallelize_module(self.model(), self.device_mesh, {"parameter": {}, "forward": fwd_plan})
        a, b, c = torch.ones((2, 2)), torch.ones((2, 2)) * 2, torch.ones((2, 2)) * 3
        expected_t = [Shard(0), Shard(1), torch.Tensor]

        out = dmodule(a, b, c)
        self.assert_helper(out, expected_t)

    def _test_seq_fwd_plan1(self):
        fwd_plan = {".input": [None, [Shard(1)]]}
        dmodule = parallelize_module(self.model(), self.device_mesh, {"parameter": {}, "forward": fwd_plan})
        a, b, c = torch.ones((2, 2)), torch.ones((2, 2)) * 2, torch.ones((2, 2)) * 3
        expected_t = [torch.Tensor, Shard(1), torch.Tensor]

        out = dmodule(a, b, c)
        self.assert_helper(out, expected_t)

    def _test_dict_fwd_plan(self):
        fwd_plan = {".input": {"a": [Shard(0)], "args": [[Shard(1)]]}}
        dmodule = parallelize_module(self.model(), self.device_mesh, {"parameter": {}, "forward": fwd_plan})
        a, b, c = torch.ones((2, 2)), torch.ones((2, 2)) * 2, torch.ones((2, 2)) * 3
        expected_t = [Shard(0), Shard(1), torch.Tensor]

        out = dmodule(a, b, c)
        self.assert_helper(out, expected_t)

        with self.assertWarns(UserWarning) as _:
            _ = dmodule(a=a)


class FwdPlanTestWDefaultArgsVarkwarg(FwdPlanTestBase):
    class DefaultArgsVarkwarg(nn.Module):
        def forward(self, a=None, **kwarg):
            return a, *(kwarg.values())

    model = DefaultArgsVarkwarg

    def _test_seq_fwd_plan(self):
        fwd_plan = {".input": [[Shard(0)], [Shard(1)]]}
        dmodule = parallelize_module(self.model(), self.device_mesh, {"parameter": {}, "forward": fwd_plan})
        a, b, c = torch.ones((2, 2)), torch.ones((2, 2)) * 2, torch.ones((2, 2)) * 3
        expected_t = [Shard(0), Shard(1), torch.Tensor]

        out = dmodule(a, b=b, c=c)
        self.assert_helper(out, expected_t)

    def _test_dict_fwd_plan(self):
        fwd_plan = {".input": {"a": [Shard(0)], "b": [Shard(1)]}}
        dmodule = parallelize_module(self.model(), self.device_mesh, {"parameter": {}, "forward": fwd_plan})
        a, b, c = torch.ones((2, 2)), torch.ones((2, 2)) * 2, torch.ones((2, 2)) * 3
        expected_t = [Shard(0), Shard(1), torch.Tensor]

        out = dmodule(a, b=b, c=c)
        self.assert_helper(out, expected_t)

        with self.assertWarns(UserWarning) as _:
            _ = dmodule(a=a)


class FwdPlanTestWSimpleArgs3(FwdPlanTestBase):
    class SimpleArgs3(nn.Module):
        def forward(self, a, b, c):
            return a, b, c

    model = SimpleArgs3

    @with_comms
    def test_seq_fwd_plan(self):
        fwd_plan = {".input": [[Shard(0)], [Shard(1)], [Replicate()]]}
        dmodule = parallelize_module(self.model(), self.device_mesh, {"parameter": {}, "forward": fwd_plan})
        a, b, c = torch.ones((2, 2)), torch.ones((2, 2)) * 2, torch.ones((2, 2)) * 3
        expected_t = [Shard(0), Shard(1), Replicate()]

        out = dmodule(a, b, c)
        self.assert_helper(out, expected_t)

        out = dmodule(a, b, c=c)
        self.assert_helper(out, expected_t)

        out = dmodule(a, c=c, b=b)
        self.assert_helper(out, expected_t)

        out = dmodule(a=a, c=c, b=b)
        self.assert_helper(out, expected_t)

        out = dmodule(a=a, c=c, b=b)
        self.assert_helper(out, expected_t)

        out = dmodule(a=a, c=c, b=b)
        self.assert_helper(out, expected_t)

        out = dmodule(b=b, c=c, a=a)
        self.assert_helper(out, expected_t)


class FwdPlanTestWMixedArgs1(FwdPlanTestBase):
    class MixedArgs1(nn.Module):
        def forward(self, a, b, c, d=1.0, e=None, *, f, g="str", **kwargs):
            return a, b, c, d, e, f, g, *(kwargs.values())

    model = MixedArgs1

    def _test_seq_fwd_plan(self):
        fwd_plan = {
            ".input": [[Shard(0)], [Shard(1)], None, None, [Replicate()], None, [Shard(0)], [Replicate()], [Shard(1)]]
        }
        dmodule = parallelize_module(self.model(), self.device_mesh, {"parameter": {}, "forward": fwd_plan})
        base = torch.ones((2, 2))
        a, b, c, d, e, f, g, h, i, j, k = (base * (i + 1) for i in range(11))
        expected_t = [Shard(0), Shard(1), torch.Tensor, float, Replicate(), None, Shard(0), Replicate(), Shard(1)]

        out = dmodule(a, b, c=c, e=e, f=None, g=g, h=h, i=i)
        self.assert_helper(out, expected_t)


"""
    def _test_dict_fwd_plan(self):
        fwd_plan = {
            ".input": {
                "a": [Shard(0)],
                "b": [Shard(1)],
                "c": None,
                "d": None,
                "e": [Replicate()],
                "f": [Replicate()],
                "g": [Shard(0)],
                "h": [Replicate()],
                "i": [Shard(1)],
            }
        }
        dmodule = parallelize_module(self.model(), self.device_mesh, {"parameter": {}, "forward": fwd_plan})
        base = torch.ones((2, 2))
        a, b, c, d, e, f, g, h, i, j, k = (base * (i + 1) for i in range(11))
        expected_t = [Shard(0), Shard(1), torch.Tensor, float, Replicate(), None, Shard(0), Replicate(), Shard(1)]

        out = dmodule(a, b, c=c, e=e, f=None, g=g, h=h, i=i)
        self.assert_helper(out, expected_t)
"""


class FwdPlanTestWMixedArgs2(FwdPlanTestBase):
    class MixedArgs2(nn.Module):
        def forward(self, a, b, c, d=1.0, e=None, *args, f, g="str", **kwargs):
            return a, b, c, d, e, *args, f, g, *(kwargs.values())

    model = MixedArgs2

    def _test_seq_fwd_plan(self):
        fwd_plan = {
            ".input": [
                [Shard(0)],  # a
                [Shard(1)],  # b
                None,  # c
                None,  # d
                [Replicate()],  # e
                [Shard(0)],  # args[0]
                [Shard(1)],  # args[1]
                [Replicate()],  # args[2]
                None,  # f
                None,  # g
                [Replicate()],  # kwargs[0]
                [Shard(1)],  # kwargs[1]
            ]
        }
        dmodule = parallelize_module(self.model(), self.device_mesh, {"parameter": {}, "forward": fwd_plan})
        base = torch.ones((2, 2))
        a, b, c, d, e, f, g, h, i, j, k = (base * (i + 1) for i in range(11))
        expected_t = [
            Shard(0),  # a
            Shard(1),  # b
            torch.Tensor,  # c    no placement
            float,  # d   float
            Replicate(),  # e
            Shard(0),  # args[0]
            Shard(1),  # args[1]
            Replicate(),  # args[2]
            None,  # f
            str,  # g
            Replicate(),  # kwargs[0]
            Shard(1),  # kwargs[1]
        ]

        out = dmodule(a, b, c, 1.0, e, i, k, d, f=None, h=h, i=i)
        self.assert_helper(out, expected_t)

    def _test_dict_fwd_plan(self):
        fwd_plan = {".input": {"a": [Shard(0)], "b": [Shard(1)]}}

    def _test_seq_fwd_plan(self):
        fwd_plan = {
            ".input": {
                "a": [Shard(0)],  # a
                "b": [Shard(1)],  # b
                "c": None,  # c
                "d": None,  # d
                "e": [Replicate()],  # e
                "args": [[Shard(0)], [Shard(1)], [Replicate()]],  # args
                "f": None,  # f
                "g": None,  # g
                "h": [Replicate()],  # kwargs[0]
                "i": [Shard(1)],  # kwargs[1]
            }
        }
        dmodule = parallelize_module(self.model(), self.device_mesh, {"parameter": {}, "forward": fwd_plan})
        base = torch.ones((2, 2))
        a, b, c, d, e, f, g, h, i, j, k = (base * (i + 1) for i in range(11))
        expected_t = [
            Shard(0),  # a
            Shard(1),  # b
            torch.Tensor,  # c    no placement
            float,  # d   float
            Replicate(),  # e
            Shard(0),  # args[0]
            Shard(1),  # args[1]
            Replicate(),  # args[2]
            None,  # f
            str,  # g
            Replicate(),  # kwargs[0]
            Shard(1),  # kwargs[1]
        ]

        out = dmodule(a, b, c, 1.0, e, i, k, d, f=None, h=h, i=i)
        self.assert_helper(out, expected_t)


if __name__ == "__main__":
    run_tests()
