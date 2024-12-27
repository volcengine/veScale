# Expert Parallelism in veScale

## Overview

veScale provides an efficient framework for training Mixture of Experts (MoE) models using expert parallelism. Expert parallelism can be deployed with the `parallelize_experts()` function, which simplifies the process of distributing and managing workload during MoE training.

### Function Signature

```python
model = parallelize_experts(
    module: nn.Module,
    experts_expr: Union[str, List[str]],
    experts_allocator: vescale.moe.ExpertsAllocator,
    token_dispatcher: vescale.moe.TokenDispatcher,
    config: Dict,
)
```

### Parameters
- **`module`**: The training model (an instance of `nn.Module`) to be parallelized.
- **`experts_expr`**: Specifies the paths to the expert modules. Can be a string or a list of strings.
- **`experts_allocator`**: An instance of `ExpertsAllocator`, used for managing expert parameter allocation.
- **`token_dispatcher`**: An instance of `TokenDispatcher`, responsible for token scheduling and distribution.
- **`config`**: A dictionary containing the MoE training configuration, including layer count, number of experts, and other relevant settings.


## Custom Scheduling

veScale allows users to define custom scheduling strategies for expert parallelism by implementing the following components:

- **`ExpertsAllocator`**: Manages expert parameter allocation. It can use `collect_performance()` to profile and dynamically adjust the DP x TP device mesh for each expert. By default, veScale shards all expert parameters across devices using tensor parallelism.

- **`TokenDispatcher`**: Handles token distribution. Using `assign_task()`, it determines workload allocation (e.g., expert IDs and token weights) and adjusts scheduling with `collect_performance()`. The default implementation randomly assigns tokens to a single DP rank for the selected expert.

## Optimizer Support

Since veScale supports dynamic placement of expert parameters, a dedicated optimizer, `MoEOptimizer`, is required. This optimizer handles the redistribution of expert parameters and their states efficiently.
Future updates will integrate these functionalities into optimizers for static parameters to streamline the process.


## Getting Started

### Data Preparation
Prepare the Shakespeare dataset by running:

```bash
cd data/shakespeare/
python3 prepare.py
cd ../..
```

### Training Command

```
cd data/shakespeare/ && python3 prepare.py && cd ../..
torchrun --standalone --nproc_per_node={GPU_CNT} mixtral_train.py --dp={dp_size} --tp={tp_size} --max_iters={max_iters}
```
