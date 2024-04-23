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
import torch.distributed as dist
from common_dtensor import DTensorTestBase, with_comms, skip_unless_torch_gpu
from torch.testing._internal.common_utils import run_tests
from vescale.devicemesh_api.device_mesh_api import veDeviceMesh
import vescale
from vescale.dtensor.placement_types import Replicate


from checkpoint.common_func import build_gpt_model_optimizer_and_dataset, flatten_dict

TMP_CKPT_DIR = "open_source_gpt_load_save_checkpoint_dir"


class TestNanoGPT1(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @property
    def init_method(self):
        # If the value is "scratch", the GPT is trained from scratch
        # It the value is "gpt2", "gpt2-medium", "gpt2-large", or "gpt2-xl"
        # the GPT loads pretrained weights from OpenAI GPT2 repository on Huggingface
        return "scratch"

    @skip_unless_torch_gpu
    @with_comms
    def test_save(self):
        ddp_gpt, dist_optimizer, data_set = build_gpt_model_optimizer_and_dataset(
            self.init_method, dp_size=2, tp_size=2
        )

        # turn off 'check_uniqueness' to allow multiple updates of global device mesh during runtime
        device_mesh = veDeviceMesh.init_device_mesh(
            device_type="cuda",
            mesh_shape=(1, 2, 2),
            mesh_dim_names=("PP", "DP", "TP"),
        )
        tp_sub_mesh = device_mesh["TP"]

        # Do fwd+bwd+step on the first data
        for X, Y in data_set[:1]:
            input = vescale.distribute_tensor(X, device_mesh["TP"], [Replicate()])
            output = vescale.distribute_tensor(Y, device_mesh["TP"], [Replicate()])
            dist_optimizer.zero_grad()
            _, output = ddp_gpt(input, output)
            loss = output.mean()
            loss.backward()
            ddp_gpt.finish_grad_sync()
            dist_optimizer.step()

        # Save the model and optimizer before second data foward

        # OmniStore Style API
        ckpt_state = {"model": ddp_gpt, "optimizer": dist_optimizer}
        vescale.checkpoint.save(TMP_CKPT_DIR, ckpt_state)

        # Dump model state_dict
        dumped_model_sd = {}
        for k, v in ddp_gpt.state_dict().items():
            dumped_model_sd[k] = v._local_tensor
        torch.save(dumped_model_sd, f"gpt_load_save_model_{dist.get_rank()}.pt")
        # Dump optimizer state_dict
        optimizer_state = dist_optimizer.state_dict()

        dumped_optimizer_sd = {}
        for k, v in flatten_dict(optimizer_state[torch.float32]).items():
            if "step" not in k:
                dumped_optimizer_sd[k] = v.local_tensor

        torch.save(dumped_optimizer_sd, f"gpt_load_save_optimizer_{dist.get_rank()}.pt")


class TestNanoGPT2(DTensorTestBase):
    @property
    def world_size(self):
        return 4

    @property
    def init_method(self):
        # If the value is "scratch", the GPT is trained from scratch
        # It the value is "gpt2", "gpt2-medium", "gpt2-large", or "gpt2-xl"
        # the GPT loads pretrained weights from OpenAI GPT2 repository on Huggingface
        return "scratch"

    @skip_unless_torch_gpu
    @with_comms
    def test_load(self):
        ddp_gpt, dist_optimizer, _ = build_gpt_model_optimizer_and_dataset(
            self.init_method, dp_size=2, tp_size=2
        )

        # Load the model and optimizer after first data

        # OmniStore Style API
        # One line function, model and optimizer will be loaded automatically
        ckpt_state = {"model": ddp_gpt, "optimizer": dist_optimizer}
        vescale.checkpoint.load(TMP_CKPT_DIR, ckpt_state)

        # Load model state dict and verify it
        dumped_model_sd = torch.load(f"gpt_load_save_model_{dist.get_rank()}.pt")

        current_model_sd = ddp_gpt.state_dict()
        for k, v in current_model_sd.items():
            if not torch.allclose(dumped_model_sd[k], v._local_tensor):
                print(f"k={k} truth={dumped_model_sd[k]} tensor in current model={v}")
                raise AssertionError()

        # Load optimizer state dict and verfify
        dumped_optimizer_sd = torch.load(f"gpt_load_save_optimizer_{dist.get_rank()}.pt")

        current_optimizer_sd = dist_optimizer.state_dict()
        for k, v in flatten_dict(current_optimizer_sd[torch.float32]).items():
            if "step" not in k:
                if not torch.allclose(dumped_optimizer_sd[k], v.local_tensor):
                    print(f"k={k} truth={dumped_optimizer_sd[k]} tensor in optim={v.local_tensor}")
                    raise AssertionError()


if __name__ == "__main__":
    run_tests()
