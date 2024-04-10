# veScale Mixtral Example

## Overview

In this directory, we provides an 4D parallelism example of using veScale to run 
a [Mixtral](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) that is directly imported
from HuggingFace without any model code modifications.


## Run

### Single Machine 8 cards
```
torchrun --nproc-per-node=8 --nnodes=1 --master-port=42516  -- python/example/mixtral_4D_benchmark/mixtral_train.py --num_hidden_layers=16
```
This will start a 8-cards MFU benchmark for Mixtral with veScale with dp=1 and tp=8.

### Distributed Environment (2 Machine 16 cards example)
```
# You may need to pull up a suitable distributed cluster environment
torchrun --nproc-per-node=8 --nnodes=1 python/example/mixtral_4D_benchmark/mixtral_train.py  --tp 8 --dp 2
```
This will start a 16 cards MFU benchmark for Mixtral with veScale with dp=2 and tp=8.

### Options
1. `--bsz`: the total number of batch size for one iteration. The default is 16.
2. `--seqlen`: the sequence lengtht of the input. The default value is 256.
3. `--dp`: the amount of data parallelism (DDP). The default is 1.
4. `--tp`: the amount of tensor parallelism. The default is 8.


## Caveats
1. The scripts are purely for demonstration propose and mfu calculation. You need to write your own training script 
   it in order to fine-tune Mixtral with your data.
