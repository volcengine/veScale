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
from torch.testing._internal.common_utils import run_tests

from common_dtensor import DTensorTestBase, with_comms

from vescale import dtensor
from vescale.dtensor import DTensor
from vescale.dtensor.api import distribute_tensor
from vescale.dtensor.placement_types import Replicate, Shard
from vescale.dtensor.device_mesh import DeviceMesh
from vescale.dmodule import _factory
from vescale.dmodule.api import parallelize_module
from vescale.dmodule.placements_interface import PlacementsInterface as PI
from vescale.dtensor.random import manual_seed

HIDDEN_SIZE = 4


class DFactoryTest(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    # def _seeding(self):
    #     import os

    #     os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    #     torch.use_deterministic_algorithms(True)
    #     torch.manual_seed(0)
    #     torch.random.manual_seed(0)
    #     torch.cuda.manual_seed(0)
    #     self._rng_state = torch.random.get_rng_state()

    # def _reset_rng(self):
    #     torch.random.set_rng_state(self._rng_state)

    def _match_factory_dfactory(self, factory, dfactory, global_shape, placements, device_mesh):
        aten_dfactory_pi = _factory._provide_args(device_mesh, {factory: PI.from_placements(placements)})

        fill_value = 1.0
        dtype = torch.float32
        layout = torch.strided
        requires_grad = False

        if factory == torch.arange:
            start, end, step = 0, global_shape[0], 1
            assert not placements[0].is_shard() or placements[0].is_shard(0)

            with _factory.FactoryDispatchMode(device_mesh=device_mesh, aten_dfactory_pi=aten_dfactory_pi):
                actual1 = torch.arange(end, dtype=dtype, layout=layout, requires_grad=requires_grad)
                actual2 = torch.arange(start, end, dtype=dtype, layout=layout, requires_grad=requires_grad)
                actual3 = torch.arange(start, end, step, dtype=dtype, layout=layout, requires_grad=requires_grad)
                actuals = (actual1, actual2, actual3)
            golden1 = dfactory(
                end,
                dtype=dtype,
                layout=layout,
                requires_grad=requires_grad,
                device_mesh=device_mesh,
                placements=placements,
            )
            golden2 = dfactory(
                start,
                end,
                dtype=dtype,
                layout=layout,
                requires_grad=requires_grad,
                device_mesh=device_mesh,
                placements=placements,
            )
            golden3 = dfactory(
                start,
                end,
                step,
                dtype=dtype,
                layout=layout,
                requires_grad=requires_grad,
                device_mesh=device_mesh,
                placements=placements,
            )
            goldens = (golden1, golden2, golden3)
        elif factory == torch.full:
            with _factory.FactoryDispatchMode(device_mesh=device_mesh, aten_dfactory_pi=aten_dfactory_pi):
                actual = torch.full(global_shape, fill_value, dtype=dtype, layout=layout, requires_grad=requires_grad)
            golden = dfactory(
                global_shape,
                fill_value,
                dtype=dtype,
                layout=layout,
                requires_grad=requires_grad,
                device_mesh=device_mesh,
                placements=placements,
            )
            actuals = (actual,)
            goldens = (golden,)
        elif factory in [torch.zeros, torch.ones, torch.empty, torch.randn]:
            if factory == torch.randn:
                manual_seed(0, device_mesh)
            with _factory.FactoryDispatchMode(device_mesh=device_mesh, aten_dfactory_pi=aten_dfactory_pi):
                actual = factory(global_shape, dtype=dtype, layout=layout, requires_grad=requires_grad)
            if factory == torch.randn:
                manual_seed(0, device_mesh)
            golden = dfactory(
                global_shape,
                dtype=dtype,
                layout=layout,
                requires_grad=requires_grad,
                device_mesh=device_mesh,
                placements=placements,
            )
            actuals = (actual,)
            goldens = (golden,)
        else:
            raise ValueError

        for actual, golden in zip(actuals, goldens):
            self.assertTrue(isinstance(actual, DTensor))
            self.assertTrue(isinstance(golden, DTensor))
            if factory in [torch.empty]:
                is_match = dtensor._utils._equal_meta_data(actual, golden, exact_device=True)
            else:
                is_match = dtensor.equal(actual, golden)
            if not is_match and self.rank == 0:
                print(f"actual = {actual}")
                print(f"golden = {golden}")
            self.assertTrue(
                is_match, msg=f"mismatch: {factory}, {dfactory}, {global_shape}, {placements}, {device_mesh}"
            )

    @with_comms
    def test_match_factory_dfactory(self):
        device_mesh = DeviceMesh(self.device_type, range(self.world_size))

        factory_dfactory = {
            torch.zeros: dtensor.zeros,
            torch.ones: dtensor.ones,
            torch.empty: dtensor.empty,
            torch.full: dtensor.full,
            torch.randn: dtensor.randn,
            torch.arange: dtensor.arange,
        }

        # self._seeding()
        for factory, dfactory in factory_dfactory.items():
            for global_shape in [(4, 4), (5, 4), (5, 7, 9)]:
                for placements in ([Replicate()], [Shard(0)]):
                    self._match_factory_dfactory(factory, dfactory, global_shape, placements, device_mesh)

    @with_comms
    def test_api(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        class M1(nn.Module):
            sharding_plan = {
                "forward": {
                    "input": [[Replicate()]],
                    "output": [[Replicate()]],
                }
            }

            def forward(self, x):
                assert isinstance(x, DTensor)
                a = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
                assert isinstance(a, DTensor)
                b = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
                assert isinstance(b, DTensor)
                return a + b

        class M2(nn.Module):
            sharding_plan = {
                "forward": {
                    "input": [[Replicate()]],
                    "output": [[Replicate()]],
                }
            }

            def forward(self, x):
                assert isinstance(x, DTensor)
                a = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
                assert isinstance(a, DTensor)
                b = torch.ones(x.shape, dtype=x.dtype, device=x.device)
                assert isinstance(b, DTensor)
                return a + b

        class M3(nn.Module):
            sharding_plan = {
                "forward": {
                    "input": [[Replicate()]],
                    "m1.input": [[Replicate()]],
                    "m1.output": [[Replicate()]],
                    "m2.input": [[Replicate()]],
                    "m2.output": [[Replicate()]],
                    "output": [[Replicate()]],
                }
            }

            def __init__(self):
                super().__init__()
                self.m1 = M1()
                self.m2 = M2()

            def forward(self, x):
                assert isinstance(x, DTensor)
                a = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
                assert isinstance(a, DTensor)
                b = self.m1(x)
                assert isinstance(b, DTensor)
                c = self.m2(x)
                assert isinstance(c, DTensor)
                return a + b + c

        class M4(nn.Module):
            sharding_plan = {
                "forward": {
                    "input": [[Replicate()]],
                    "m1.input": [[Replicate()]],
                    "m1.output": [[Replicate()]],
                    "m2.input": [[Replicate()]],
                    "m2.output": [[Replicate()]],
                    "output": [[Replicate()]],
                }
            }

            def __init__(self):
                super().__init__()
                self.m1 = M1()
                self.m2 = M2()

            def forward(self, x):
                assert isinstance(x, DTensor)
                b = self.m1(x)
                assert isinstance(b, DTensor)
                c = self.m2(x)
                assert isinstance(c, DTensor)
                return b + c

        data = torch.ones(HIDDEN_SIZE, device=self.device_type)
        zero_replicate = distribute_tensor(torch.zeros(data.shape), device_mesh, [Replicate()])
        ones_replicate = distribute_tensor(torch.ones(data.shape), device_mesh, [Replicate()])

        # factory = True
        dm = parallelize_module(M1(), device_mesh, M1.sharding_plan, factory=True)
        out = dm(data)
        self.assertTrue(dtensor.equal(out, zero_replicate))

        dm = parallelize_module(M2(), device_mesh, M2.sharding_plan, factory=True)
        out = dm(data)
        self.assertTrue(dtensor.equal(out, ones_replicate))

        dm = parallelize_module(M3(), device_mesh, M3.sharding_plan, factory=True)
        out = dm(data)
        self.assertTrue(dtensor.equal(out, ones_replicate))

        # factory={ M2 : True }
        dm = parallelize_module(M2(), device_mesh, M2.sharding_plan, factory={M2: True})
        out = dm(data)
        self.assertTrue(dtensor.equal(out, ones_replicate))
        # factory={ M3 : True }
        dm = parallelize_module(M3(), device_mesh, M3.sharding_plan, factory={M3: True})
        out = dm(data)
        self.assertTrue(dtensor.equal(out, ones_replicate))
        # factory={ M1 : True, M2 : True }
        dm = parallelize_module(M4(), device_mesh, M4.sharding_plan, factory={M1: True, M2: True})
        out = dm(data)
        self.assertTrue(dtensor.equal(out, ones_replicate))

        # factory={ M2 : { torch.zeros : [Shard(0)], torch.ones : [Shard(0)] } }
        dm = parallelize_module(
            M2(), device_mesh, M2.sharding_plan, factory={M2: {torch.zeros: [Shard(0)], torch.ones: [Shard(0)]}}
        )
        out = dm(data)
        self.assertTrue(dtensor.equal(out, ones_replicate))
        # factory={ M3 : { torch.zeros : [Shard(0)], torch.ones : [Shard(0)] } }
        plan = {
            "forward": {
                "input": [[Replicate()]],
                "m1.input": [[Replicate()]],
                "m2.input": [[Replicate()]],
                "output": [[Replicate()]],
            }
        }
        dm = parallelize_module(
            M3(), device_mesh, plan, factory={M3: {torch.zeros: [Shard(0)], torch.ones: [Shard(0)]}}
        )
        out = dm(data)
        self.assertTrue(dtensor.equal(out, ones_replicate), msg=f"out:{out}")
        # factory={ M1 : { torch.zeros : [Shard(0)] },
        #           M2: { torch.zeros : [Replicate()], torch.ones : [Replicate()] } }
        dm = parallelize_module(
            M4(),
            device_mesh,
            M4.sharding_plan,
            factory={M1: {torch.zeros: [Shard(0)]}, M2: {torch.zeros: [Replicate()], torch.ones: [Replicate()]}},
        )
        out = dm(data)
        self.assertTrue(dtensor.equal(out, ones_replicate))

    @with_comms
    def test_with_fwd_hook(self):
        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))
        data = torch.ones(HIDDEN_SIZE, device=self.device_type)
        zero_replicate = distribute_tensor(torch.zeros(HIDDEN_SIZE * self.world_size), device_mesh, [Replicate()])

        # simple case
        class SimpleArgs1(nn.Module):
            def forward(self, a):
                b = torch.zeros(a.shape, dtype=a.dtype, device=a.device)
                return a, b

        class DefaultArgs1(nn.Module):
            def forward(self, a=None):
                b = torch.zeros(a.shape, dtype=a.dtype, device=a.device)
                return a, b

        for mcls in [SimpleArgs1, DefaultArgs1]:
            for fwd_plan in [{"input": [[Shard(0)]]}, {"input": {"a": [Shard(0)]}}]:
                dm = parallelize_module(mcls(), device_mesh, {"forward": fwd_plan}, factory=True)
                out1, out2 = dm(data)
                self.assertTrue(isinstance(out1, DTensor))
                self.assertTrue(out1.placements[0].is_shard(0))
                self.assertTrue(isinstance(out2, DTensor))
                self.assertTrue(dtensor.equal(out2, zero_replicate))

        # complex case
        class MixedArgs2(nn.Module):
            def forward(self, a, b, c, d=1.0, e=None, *args, f, g="str", **kwargs):
                z = torch.zeros(a.shape, dtype=a.dtype, device=a.device)
                return a, b, c, d, e, *args, f, g, *(kwargs.values()), z

        fwd_plan1 = {
            "input": [
                [Shard(0)],  # a
                [Shard(0)],  # b
                None,  # c
                None,  # d
                [Replicate()],  # e
                [Shard(0)],  # args[0]
                [Shard(0)],  # args[1]
                [Replicate()],  # args[2]
                None,  # f
                None,  # g
                [Replicate()],  # kwargs[0]
                [Shard(0)],  # kwargs[1]
            ]
        }

        fwd_plan2 = {
            "input": {
                "a": [Shard(0)],  # a
                "b": [Shard(0)],  # b
                "c": None,  # c
                "d": None,  # d
                "e": [Replicate()],  # e
                "args": [[Shard(0)], [Shard(0)], [Replicate()]],  # args
                "f": None,  # f
                "g": None,  # g
                "h": [Replicate()],  # kwargs[0]
                "i": [Shard(0)],  # kwargs[1]
            }
        }

        for fwd_plan in [fwd_plan1, fwd_plan2]:
            dm = parallelize_module(MixedArgs2(), device_mesh, {"forward": fwd_plan}, factory=True)
            a, b, c, d, e, f, g, h, i, j, k = (data for _ in range(11))
            out = dm(a, b, c, 1.0, e, i, k, d, f=None, h=h, i=i)
            self.assertTrue(isinstance(out[0], DTensor))
            self.assertTrue(out[0].placements[0].is_shard(0))
            self.assertTrue(isinstance(out[-1], DTensor))
            self.assertTrue(dtensor.equal(out[-1], zero_replicate))

    @with_comms
    def test_with_model_patch(self):  # TODO: support nested factory False
        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
                self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)

            def forward(self, x):
                x = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
                x = self.fc1(x)
                x = self.fc2(x)
                return x

        device_mesh = DeviceMesh(self.device_type, list(range(self.world_size)))

        sharding_plan = {
            "parameter": {
                "fc1.weight": [Shard(0)],
                "fc1.bias": [Shard(0)],
                "fc2.weight": [Shard(1)],
                "fc2.bias": [Replicate()],
            },
            "forward": {
                "input": [[Replicate()]],
                "fc2.output": [[Replicate()]],
                "output": [[Shard(0)]],
            },
        }

        dmlp = parallelize_module(MLP(), device_mesh, sharding_plan, factory=True)

        data = torch.ones((HIDDEN_SIZE, HIDDEN_SIZE), device=self.device_type)
        out = dmlp(data)
        self.assertTrue(isinstance(out, DTensor))
        self.assertTrue(out.placements[0].is_shard(0))


if __name__ == "__main__":
    run_tests()
