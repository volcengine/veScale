################################################################################
#
# Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
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
from torch.testing._internal.common_utils import run_tests

from common_dtensor import DTensorTestBase, with_comms

import vescale

from ..common_func import flatten_dict, get_open_llama_model_optimizer

TMP_CKPT_DIR = "./open_llama_load_save_checkpoint_dir"
NUM_OF_LAYERS = 4  # Limit number of transformer layers to avoid OOM in unit tests


class OpenLLaMa2Test1(DTensorTestBase):
    @property
    def world_size(self):
        return 8

    @with_comms
    def test_open_llama2_with_ddp(self):
        bsz = 6
        s = 18
        ddp_decoder, ve_optimizer, config = get_open_llama_model_optimizer(
            dp_size=2, tp_size=4, layer_number=NUM_OF_LAYERS
        )
        input = torch.randint(low=0, high=config.vocab_size, size=(bsz, s)).cuda()
        # d_input = distribute_tensor(input.detach(), tp_sub_mesh, [Shard(1)])

        ve_optimizer.zero_grad()
        vescale_output = ddp_decoder(input.detach()).last_hidden_state
        # vescale_output = vescale_output.redistribute(placements = [Replicate()]* tp_sub_mesh.ndim)
        vescale_loss = vescale_output.mean()
        vescale_loss.backward()
        ddp_decoder.finish_grad_sync()
        ve_optimizer.step()

        ckpt_state = {"model": ddp_decoder, "optimizer": ve_optimizer}
        vescale.checkpoint.save(TMP_CKPT_DIR, ckpt_state)
        # Clean up writing futures (For unit test only)
        vescale.checkpoint.VeScaleCheckpointer._VeScaleCheckpointer__cleanup()

        # Dump model state_dict
        dumped_model_sd = {}
        for k, v in ddp_decoder.state_dict().items():
            dumped_model_sd[k] = v._local_tensor
        torch.save(dumped_model_sd, f"open_llama_load_save_model_{dist.get_rank()}.pt")
        # Dump optimizer state_dict
        optimizer_state = ve_optimizer.state_dict()

        dumped_optimizer_sd = {}
        for k, v in flatten_dict(optimizer_state[torch.float32]).items():
            if "step" not in k:
                dumped_optimizer_sd[k] = v.local_tensor

        torch.save(dumped_optimizer_sd, f"open_llama_load_save_optimizer_{dist.get_rank()}.pt")


class OpenLLaMa2Test2(DTensorTestBase):
    @property
    def world_size(self):
        return 8

    @with_comms
    def test_open_llama2_with_ddp(self):
        ddp_decoder, ve_optimizer, _ = get_open_llama_model_optimizer(dp_size=2, tp_size=4, layer_number=NUM_OF_LAYERS)

        ckpt_state = {"model": ddp_decoder, "optimizer": ve_optimizer}
        vescale.checkpoint.load(TMP_CKPT_DIR, ckpt_state)

        # Load model state dict and verify it
        dumped_model_sd = torch.load(f"open_llama_load_save_model_{dist.get_rank()}.pt")

        current_model_sd = ddp_decoder.state_dict()
        for k, v in current_model_sd.items():
            if not torch.allclose(dumped_model_sd[k], v._local_tensor):
                print(f"k={k} truth={dumped_model_sd[k]} tensor in current model={v}")
                raise AssertionError()

        # Load optimizer state dict and verify
        dumped_optimizer_sd = torch.load(f"open_llama_load_save_optimizer_{dist.get_rank()}.pt")

        current_optimizer_sd = ve_optimizer.state_dict()
        for k, v in flatten_dict(current_optimizer_sd[torch.float32]).items():
            if "step" not in k:
                if not torch.allclose(dumped_optimizer_sd[k], v.local_tensor):
                    print(f"k={k} truth={dumped_optimizer_sd[k]} tensor in optim={v.local_tensor}")
                    raise AssertionError()


if __name__ == "__main__":
    run_tests()
