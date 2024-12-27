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


def parse_train_loss(log_fn, name=None):
    lines = open(log_fn).readlines()
    train_losses = []
    for line in lines:
        if "loss" in line and "iter" in line:
            token = line.split()[line.split().index("loss") + 1]
            train_loss = float(token)
            train_losses.append(train_loss)
    if name is None:
        name = log_fn
    print(f'"{name}": {train_losses},')


def parse_grad_norm(log_fn, name=None):
    lines = open(log_fn).readlines()
    grad_norms = []
    for line in lines:
        if "|g|" in line:
            token = line.split()[line.split().index("|g|") + 1]
            grad_norm = float(token)
            grad_norms.append(grad_norm)
    if name is None:
        name = log_fn
    print(f'"{name}": {grad_norms},')


GPU_CNT = 4
DP_SIZES = [1, 2]
SINGLE_GPU_RUN = "python3"
MULTI_GPU_RUN = f"torchrun --standalone --nproc_per_node={GPU_CNT}"
CODE = "mixtral_train.py"
LOG_PREFIX = "mixtral_new_MOE"
TRAIN_BIN_PATH = "data/shakespeare/train.bin"


def run_exps(max_iters, dtypes, run=True):
    if not os.path.isfile(TRAIN_BIN_PATH):
        os.system("cd data/shakespeare/ && python3 prepare.py && cd ../..")
    os.makedirs("logs", exist_ok=True)
    if run:
        for dtype in dtypes:
            dt = "bfloat16" if dtype == "bf16" else "float32"
            cmd = f"{SINGLE_GPU_RUN} {CODE} --dp=1 --tp=1 --max_iters={max_iters} --dtype='{dt}'"
            log_fn = f"logs/{LOG_PREFIX}_1gpu_{dtype}_max_iters_{max_iters}.log"
            print(f"run {cmd} > {log_fn} 2> {log_fn}.err")
            os.system(f"{cmd} > {log_fn} 2> {log_fn}.err")
            for dp_size in DP_SIZES:
                tp_size = GPU_CNT // dp_size
                dt = "bfloat16" if dtype == "bf16" else "float32"
                cmd = f"{MULTI_GPU_RUN} {CODE} --dp={dp_size} --tp={tp_size} --max_iters={max_iters} --dtype='{dt}'"
                log_fn = f"logs/{LOG_PREFIX}_{GPU_CNT}gpu_dp{dp_size}_tp{tp_size}_{dtype}_max_iters_{max_iters}.log"
                print(f"run {cmd} > {log_fn} 2> {log_fn}.err")
                os.system(f"{cmd} > {log_fn} 2> {log_fn}.err")

    print("train_loss = {")
    for dtype in dtypes:
        parse_train_loss(f"logs/{LOG_PREFIX}_1gpu_{dtype}_max_iters_{max_iters}.log", f"1GPU_{dtype}")
        for dp_size in DP_SIZES:
            tp_size = GPU_CNT // dp_size
            log_fn = f"logs/{LOG_PREFIX}_{GPU_CNT}gpu_dp{dp_size}_tp{tp_size}_{dtype}_max_iters_{max_iters}.log"
            parse_train_loss(log_fn, f"{GPU_CNT}GPU_DP{dp_size}_TP{tp_size}_{dtype}")
    print("}")

    print("grad_norm = {")
    for dtype in dtypes:
        parse_grad_norm(f"logs/{LOG_PREFIX}_1gpu_{dtype}_max_iters_{max_iters}.log", f"1GPU_{dtype}")
        for dp_size in DP_SIZES:
            tp_size = GPU_CNT // dp_size
            log_fn = f"logs/{LOG_PREFIX}_{GPU_CNT}gpu_dp{dp_size}_tp{tp_size}_{dtype}_max_iters_{max_iters}.log"
            parse_grad_norm(log_fn, f"{GPU_CNT}GPU_DP{dp_size}_TP{tp_size}_{dtype}")
    print("}")


if __name__ == "__main__":
    run_exps(1000, ["bf16"], run=True)
