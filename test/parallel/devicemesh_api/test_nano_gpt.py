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
from torch.testing._internal.common_utils import run_tests
import torch
import vescale
from vescale.devicemesh_api import veDeviceMesh
from vescale.dtensor.placement_types import Replicate
from vescale.ddp.distributed_data_parallel import DistributedDataParallel as DDP
from parallel.devicemesh_api._build import build_gpt_model_and_optimizer, prepare_config_and_data, system_setup
from parallel.devicemesh_api._model import GPT, GPTConfig
from parallel.devicemesh_api._sharding_plan import nanoGPT_plan, nanoGPT_tp_only_plan
from vescale.dmodule.api import parallelize_module
from common_dtensor import DTensorTestBase, with_comms_device


class TestNanoGPTTwoDimDMAPI(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @property
    def init_method(self):
        # If the value is "scratch", the GPT is trained from scratch
        # If the value is "gpt2", "gpt2-medium", "gpt2-large", or "gpt2-xl"
        # the GPT loads pretrained weights from OpenAI GPT2 repository on Huggingface
        return "scratch"

    @with_comms_device("cpu")
    def test_2d_dp_tp_doptim_gpt_cpu(self):
        """
        Test 3-dimensional strategy demo on CPU.
        When the demo runs on CPU, it uses gloo as backend.
        """
        self._test_2d_dp_tp_doptim_gpt()

    @with_comms_device("cuda")
    def test_2d_dp_tp_doptim_gpt_cuda(self):
        """
        Test 3-dimensional strategy demo on CUDA.
        When the demo runs on CUDA, it uses nccl as backend.
        """
        self._test_2d_dp_tp_doptim_gpt()

    @with_comms_device("cpu")
    def test_2d_dp_tp_sp_doptim_gpt_cpu(self):
        """
        Test 4-dimensional strategy demo on CPU.
        When the demo runs on CPU, it uses gloo as backend.
        """
        self._test_2d_dp_tp_sp_doptim_gpt()

    @with_comms_device("cuda")
    def test_2d_dp_tp_sp_doptim_gpt_cuda(self):
        """
        Test 4-dimensional strategy demo on CUDA.
        When the demo runs on CUDA, it uses nccl as backend.
        """
        self._test_2d_dp_tp_sp_doptim_gpt()

    def _test_2d_dp_tp_doptim_gpt(self):
        """
        Demo test with 3-dimensional strategy (data, tensor, distributed optimizer parallel)
        with 2-dimensional global DeviceMesh.
        """
        system_setup()
        # DP=2 TP=2, distributed optimizer
        task_config = {
            "init_method": self.init_method,
            "dp_size": 2,
            "tp_size": 2,
            "use_dist_optimizer": True,
            "sharding_plan": nanoGPT_tp_only_plan,
        }
        self._test_gpt(task_config)

    def _test_2d_dp_tp_sp_doptim_gpt(self):
        """
        Demo test with 4-dimensional strategy (data, tensor, sequence, distributed optimizer parallel)
        with 2-dimensional global DeviceMesh.
        """
        system_setup()
        # DP=2 TP=2, distributed optimizer
        task_config = {
            "init_method": self.init_method,
            "dp_size": 2,
            "tp_size": 2,
            "use_dist_optimizer": True,
            "sharding_plan": nanoGPT_plan,
        }
        self._test_gpt(task_config)

    @with_comms_device("cpu")
    def test_2d_dp_tp_base_optimizer_gpt_cpu(self):
        """
        Test 3-dimensional strategy (data, tensor, sequence) demo on CPU.
        When the demo runs on CPU, it uses gloo as backend.
        """
        self._test_2d_dp_tp_base_optimizer_gpt()

    @with_comms_device("cuda")
    def test_2d_dp_tp_base_optimizer_gpt_cuda(self):
        """
        Test 3-dimensional strategy (data, tensor, sequence) demo on CUDA.
        """
        self._test_2d_dp_tp_base_optimizer_gpt()

    def _test_2d_dp_tp_base_optimizer_gpt(self):
        """
        Demo test with 3-dimensional strategy (data, tensor, sequence)
        with 2-dimensional global DeviceMesh.
        """
        system_setup()
        # DP=2 TP=2, basic optimizer
        task_config = {
            "init_method": self.init_method,
            "dp_size": 2,
            "tp_size": 2,
            "use_dist_optimizer": False,
            "sharding_plan": nanoGPT_plan,
        }
        self._test_gpt(task_config)

    def _test_gpt(self, task_config):
        model_args, data_set = prepare_config_and_data()
        task_config["gptconf"] = GPTConfig(**model_args)
        model, optimizer, global_device_mesh = build_gpt_model_and_optimizer(**task_config)

        # Do fwd+bwd+step on the first data
        for X, Y in data_set[:1]:
            input, output = self._process_data(X, Y)
            optimizer.zero_grad()
            _, output = model(input, output)
            loss = output.mean()
            loss.backward()
            model.finish_grad_sync()
            optimizer.step()

    def _process_data(self, x, y):
        if veDeviceMesh.get_strategy_size("TP") > 1:
            tp_mesh = veDeviceMesh.get_tensor_parallel_mesh()
            x = vescale.distribute_tensor(x, tp_mesh, [Replicate()])
            y = vescale.distribute_tensor(y, tp_mesh, [Replicate()])
        return x, y


class TestNanoGPTOneDimDMAPI(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    @property
    def init_method(self):
        # If the value is "scratch", the GPT is trained from scratch
        # If the value is "gpt2", "gpt2-medium", "gpt2-large", or "gpt2-xl"
        # the GPT loads pretrained weights from OpenAI GPT2 repository on Huggingface
        return "scratch"

    @with_comms_device("cpu")
    def test_1d_dp_gpt_cpu(self):
        """
        Test data parallel strategy demo on CPU.
        When the demo runs on CPU, it uses gloo as backend.
        """
        self._test_1d_dp_gpt()

    @with_comms_device("cuda")
    def test_1d_dp_gpt_cuda(self):
        """
        Test data parallel strategy demo on CUDA.
        When the demo runs on CUDA, it uses nccl as backend.
        """
        self._test_1d_dp_gpt()

    def _test_1d_dp_gpt(self):
        """
        Demo test with data parallel strategy with 1-dimensional global DeviceMesh.
        """
        system_setup()
        # Prepare model and data
        dp_size = 2
        model, data_set = self._prepare()
        model.to("cuda")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        # Initialize global DeviceMesh
        device_mesh = veDeviceMesh.init_device_mesh("cuda", mesh_shape=(dp_size,), mesh_dim_names=("DP",))
        # Wrap model with DDP module. Since 1D global DeviceMesh cannot slice sub-DeviceMesh. we have to rely on get_data_parallel_dim_groups()
        dp_comm = veDeviceMesh["DP"] if veDeviceMesh.ndim > 1 else veDeviceMesh.get_data_parallel_dim_groups()
        model = DDP(
            model,
            data_pg_or_device_mesh=dp_comm,
            accumulate_allreduce_grads_in_fp32=True,
            overlap_grad_reduce=False,
            use_distributed_optimizer=True,
        )
        # Train model
        self.train(model, optimizer, data_set, use_dist_tensor=False)

    @with_comms_device("cpu")
    def test_1d_tpsp_gpt_cpu(self):
        """
        Test tensor and sequence parallel strategy demo on CPU.
        When the demo runs on CPU, it uses gloo as backend.
        """
        self._test_1d_tpsp_gpt()

    @with_comms_device("cuda")
    def test_1d_tpsp_gpt_cuda(self):
        """
        Test tensor and sequence parallel strategy demo on CUDA.
        When the demo runs on CUDA, it uses nccl as backend.
        """
        self._test_1d_tpsp_gpt()

    def _test_1d_tpsp_gpt(self):
        """
        Demo test with 2-dimensional (tensor parallel and sequence parallel)
        strategy with 1-dimensional global DeviceMesh.
        """
        system_setup()
        # Prepare model and data
        tp_size = 2
        model, data_set = self._prepare()
        # Initialize global DeviceMesh
        device_mesh = veDeviceMesh.init_device_mesh("cuda", mesh_shape=(tp_size,), mesh_dim_names=("TP",))
        model = parallelize_module(model, device_mesh, nanoGPT_plan)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        # Train model
        self.train(model, optimizer, data_set, use_dist_tensor=True)

    def train(self, model, optimizer, dataset, use_dist_tensor=False):
        for X, Y in dataset[:1]:
            input, output = self._process_data(X, Y, use_dist_tensor=use_dist_tensor)
            optimizer.zero_grad()
            _, output = model(input, output)
            loss = output.mean()
            loss.backward()
            model.finish_grad_sync()
            optimizer.step()

    def _prepare(self):
        model_args, data_set = prepare_config_and_data()
        gptconf = GPTConfig(**model_args)
        if self.init_method == "scratch":
            model = GPT(gptconf).bfloat16()
        else:
            model = GPT.from_pretrained(self.init_method, dict(dropout=0.0)).bfloat16()
        return model, data_set

    def _process_data(self, x, y, use_dist_tensor=False):
        if use_dist_tensor:
            tp_mesh = veDeviceMesh.get()
            x = vescale.distribute_tensor(x, tp_mesh, [Replicate()])
            y = vescale.distribute_tensor(y, tp_mesh, [Replicate()])
        return x, y


if __name__ == "__main__":
    run_tests()
