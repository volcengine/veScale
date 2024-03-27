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

import os
import re


def parse_train_loss(log_fn, name=None):
    lines = open(log_fn).readlines()
    train_losses = []
    for line in lines:
        if "val loss" in line:
            token = line.split()[line.split().index("val") - 1]
            match = re.match(r"\d+(\.\d+)?", token)
            train_loss = float(match.group())
            train_losses.append(train_loss)
    if name is None:
        name = log_fn
    print(f'"{name}": {train_losses},')


def parse(log_fn, name=None):
    lines = open(log_fn).readlines()
    val_losses = []
    for line in lines:
        if "val loss" in line:
            token = line.split()[line.split().index("val") + 2]
            match = re.match(r"\d+(\.\d+)?", token)
            val_loss = float(match.group())
            val_losses.append(val_loss)
    if name is None:
        name = log_fn
    print(f'"{name}": {val_losses},')


def run_exps(max_iters, dtypes, run=True):
    os.makedirs("logs", exist_ok=True)
    if run:
        for dtype in dtypes:
            dt = "bfloat16" if dtype == "bf16" else "float32"
            cmd = f"python3 base_train.py config/finetune_shakespeare.py --compile=False --wandb_log=False --max_iters={max_iters} --dtype='{dt}'"
            log_fn = f"logs/1gpu_{dtype}_max_iters_{max_iters}.log"
            print(f"run {cmd} > {log_fn} 2> {log_fn}.err")
            os.system(f"{cmd} > {log_fn} 2> {log_fn}.err")
        for dp_size in [1, 2, 4]:
            tp_size = 4 // dp_size
            for dtype in dtypes:
                dt = "bfloat16" if dtype == "bf16" else "float32"
                cmd = f"torchrun --standalone --nproc_per_node=4 finetune_4D.py config/finetune_shakespeare.py --compile=False --use_DO=True --wandb_log=False --dp_size={dp_size} --tp_size={tp_size} --max_iters={max_iters} --dtype='{dt}'"
                log_fn = f"logs/4gpu_dp{dp_size}_tp{tp_size}_{dtype}_max_iters_{max_iters}.log"
                print(f"run {cmd} > {log_fn} 2> {log_fn}.err")
                os.system(f"{cmd} > {log_fn} 2> {log_fn}.err")

    print("train_loss = {")
    for dtype in dtypes:
        parse_train_loss(f"logs/1gpu_{dtype}_max_iters_{max_iters}.log", f"1GPU_{dtype}")
        for dp_size in [1, 2, 4]:
            tp_size = 4 // dp_size
            log_fn = f"logs/4gpu_dp{dp_size}_tp{tp_size}_{dtype}_max_iters_{max_iters}.log"
            parse_train_loss(log_fn, f"4GPU_DP{dp_size}_TP{tp_size}_{dtype}")
    print("}")

    print("val_loss = {")
    for dtype in dtypes:
        parse(f"logs/1gpu_{dtype}_max_iters_{max_iters}.log", f"1GPU_{dtype}")
        for dp_size in [1, 2, 4]:
            tp_size = 4 // dp_size
            log_fn = f"logs/4gpu_dp{dp_size}_tp{tp_size}_{dtype}_max_iters_{max_iters}.log"
            parse(log_fn, f"4GPU_DP{dp_size}_TP{tp_size}_{dtype}")
    print("}")


if __name__ == "__main__":
    run_exps(200, ["bf16", "fp32"], run=True)
