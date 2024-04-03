# veScale Open Llama Example
## Overview
In this directory, we provides an 4D parallelism example of using veScale to run 
a [open llama model](https://huggingface.co/openlm-research/open_llama_7b) that is directly imported
from HuggingFace without any model code modifications.


## Run
### Single Machine 8 cards
```
torchrun --standalone --nnodes=1 --nproc-per-node=8 ./run_open_llama_w_vescale.py --dp=4 --tp=2 --warmup=10 --iter=40
```
This will start a 8-cards MFU benchmark for open Llama with veScale with dp=4 and tp=2.

### Distributed Environment (4 Machine 32 cards example)
```
torchrun --nnodes=4 --nproc-per-node=8 --node_rank=$node_rank --master_addr=$master_addr  --master_port=$master_port ./run_open_llama_w_vescale.py --dp=16 --tp=2 --warmup=10 --iter=40
```
This will start a 32 cards MFU benchmark for open Llama with veScale with dp=16 and tp=2.

### Options
1. `--total_bsz`: the total number of batch size for one iteration. The default is 16.
2. `--dp`: the amount of data parallelism (DDP). This arg has no default value.
3. `--tp`: the amount of tensor parallelism. This arg has no default value.
4. `--warmup`: the number of warmup iteration performed. The default is 5.
5. `--iter`: the number of iteration used for calculating the MFU. The default is 10.
6. `--no-ckpt"`: This arg turn off loading check points from Huggingface.

## Caveats
1. The scripts are purely for demonstration propose and mfu calculation. You need to write your own training script 
   it in order to fine-tune open llama with your data.
2. This is a known issue with transformer version greater than 4.37.2. We will be fixing it later.