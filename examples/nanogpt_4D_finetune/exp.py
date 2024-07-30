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

import os
import re


def parse_train_loss_per_iter(log_fn, name=None):
    lines = open(log_fn).readlines()
    train_losses = []
    for line in lines:
        if "iter" in line and "loss" in line:
            token = line.split()[line.split().index("loss") + 1]
            match = re.match(r"\d+(\.\d+)?", token)
            train_loss = float(match.group())
            train_losses.append(train_loss)
    if name is None:
        name = log_fn
    print(f'"{name}": {train_losses},')


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


GPU_CNT = 4
# DP_SIZES = [1, 2, 4]
DP_SIZES = [1]
SINGLE_GPU_RUN = "python3"
MULTI_GPU_RUN = f"torchrun --standalone --nproc_per_node={GPU_CNT}"
CONFIG = "config/finetune_shakespeare.py --dropout=0.9 --learning_rate=1e-5 --grad_clip=0.0 --eval_interval=10000"
LOG_PREFIX = "no_attn_dropout_lr1e-5_dropout9_gc0"


def run_exps(max_iters, dtypes, run=True):
    os.makedirs("logs", exist_ok=True)
    if run:
        for dtype in dtypes:
            dt = "bfloat16" if dtype == "bf16" else "float32"
            cmd = f"{SINGLE_GPU_RUN} base_train.py {CONFIG} --compile=False --max_iters={max_iters} --dtype='{dt}'"
            log_fn = f"logs/{LOG_PREFIX}_1gpu_{dtype}_max_iters_{max_iters}.log"
            print(f"run {cmd} > {log_fn} 2> {log_fn}.err")
            os.system(f"{cmd} > {log_fn} 2> {log_fn}.err")
            for dp_size in DP_SIZES:
                tp_size = GPU_CNT // dp_size
                dt = "bfloat16" if dtype == "bf16" else "float32"

                cmd = f"{MULTI_GPU_RUN} finetune_4D.py {CONFIG} --compile=False --dp_size={dp_size} --tp_size={tp_size} --max_iters={max_iters} --dtype='{dt}'"
                log_fn = f"logs/{LOG_PREFIX}_{GPU_CNT}gpu_dp{dp_size}_tp{tp_size}_{dtype}_max_iters_{max_iters}_old_drand.log"
                print(f"run {cmd} > {log_fn} 2> {log_fn}.err")
                os.system(f'export VESCALE_SINGLE_DEVICE_RAND="0"; {cmd} > {log_fn} 2> {log_fn}.err')

                cmd = f"{MULTI_GPU_RUN} finetune_4D.py {CONFIG} --compile=False --dp_size={dp_size} --tp_size={tp_size} --max_iters={max_iters} --dtype='{dt}'"
                log_fn = f"logs/{LOG_PREFIX}_{GPU_CNT}gpu_dp{dp_size}_tp{tp_size}_{dtype}_max_iters_{max_iters}.log"
                print(f"run {cmd} > {log_fn} 2> {log_fn}.err")
                os.system(f"{cmd} > {log_fn} 2> {log_fn}.err")

    print("train_loss = {")
    for dtype in dtypes:
        # parse_train_loss(f"logs/{LOG_PREFIX}_1gpu_{dtype}_max_iters_{max_iters}.log", f"1GPU_{dtype}")
        parse_train_loss_per_iter(f"logs/{LOG_PREFIX}_1gpu_{dtype}_max_iters_{max_iters}.log", f"1GPU_{dtype}")
        for dp_size in DP_SIZES:
            tp_size = GPU_CNT // dp_size
            log_fn = f"logs/{LOG_PREFIX}_{GPU_CNT}gpu_dp{dp_size}_tp{tp_size}_{dtype}_max_iters_{max_iters}.log"
            parse_train_loss_per_iter(log_fn, f"{GPU_CNT}GPU_DP{dp_size}_TP{tp_size}_{dtype}")
            # parse_train_loss(log_fn, f"{GPU_CNT}GPU_DP{dp_size}_TP{tp_size}_{dtype}")
            log_fn = (
                f"logs/{LOG_PREFIX}_{GPU_CNT}gpu_dp{dp_size}_tp{tp_size}_{dtype}_max_iters_{max_iters}_old_drand.log"
            )
            parse_train_loss_per_iter(log_fn, f"{GPU_CNT}GPU_DP{dp_size}_TP{tp_size}_{dtype}_Old_DRand")
            # parse_train_loss(log_fn, f"{GPU_CNT}GPU_DP{dp_size}_TP{tp_size}_{dtype}_Old_DRand")
    print("}")

    print("train_obj = {")
    for dtype in dtypes:
        parse_train_loss(f"logs/{LOG_PREFIX}_1gpu_{dtype}_max_iters_{max_iters}.log", f"1GPU_{dtype}")
        # parse_train_loss_per_iter(f"logs/{LOG_PREFIX}_1gpu_{dtype}_max_iters_{max_iters}.log", f"1GPU_{dtype}")
        for dp_size in DP_SIZES:
            tp_size = GPU_CNT // dp_size
            log_fn = f"logs/{LOG_PREFIX}_{GPU_CNT}gpu_dp{dp_size}_tp{tp_size}_{dtype}_max_iters_{max_iters}.log"
            # parse_train_loss_per_iter(log_fn, f"{GPU_CNT}GPU_DP{dp_size}_TP{tp_size}_{dtype}")
            parse_train_loss(log_fn, f"{GPU_CNT}GPU_DP{dp_size}_TP{tp_size}_{dtype}")
            log_fn = (
                f"logs/{LOG_PREFIX}_{GPU_CNT}gpu_dp{dp_size}_tp{tp_size}_{dtype}_max_iters_{max_iters}_old_drand.log"
            )
            # parse_train_loss_per_iter(log_fn, f"{GPU_CNT}GPU_DP{dp_size}_TP{tp_size}_{dtype}_Old_DRand")
            parse_train_loss(log_fn, f"{GPU_CNT}GPU_DP{dp_size}_TP{tp_size}_{dtype}_Old_DRand")
    print("}")

    print("val_loss = {")
    for dtype in dtypes:
        parse(f"logs/{LOG_PREFIX}_1gpu_{dtype}_max_iters_{max_iters}.log", f"1GPU_{dtype}")
        for dp_size in DP_SIZES:
            tp_size = GPU_CNT // dp_size
            log_fn = f"logs/{LOG_PREFIX}_{GPU_CNT}gpu_dp{dp_size}_tp{tp_size}_{dtype}_max_iters_{max_iters}.log"
            parse(log_fn, f"{GPU_CNT}GPU_DP{dp_size}_TP{tp_size}_{dtype}")
            log_fn = (
                f"logs/{LOG_PREFIX}_{GPU_CNT}gpu_dp{dp_size}_tp{tp_size}_{dtype}_max_iters_{max_iters}_old_drand.log"
            )
            parse(log_fn, f"{GPU_CNT}GPU_DP{dp_size}_TP{tp_size}_{dtype}_Old_DRand")
    print("}")


if __name__ == "__main__":
    run_exps(200, ["fp32"], run=True)
