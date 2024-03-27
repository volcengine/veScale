################################################################################
# Copyright (c) 2022 Andrej Karpathy

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################

out_dir = "out-shakespeare"
eval_interval = 5
eval_iters = 5

dataset = "shakespeare"
init_from = "gpt2"  # this is the smallest GPT-2 model

wandb_log = True  # feel free to turn on
wandb_project = f"{init_from}_finetune_{dataset}"

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 4 batch_size * 8 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 32
gradient_accumulation_steps = 1
max_iters = 20

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False
