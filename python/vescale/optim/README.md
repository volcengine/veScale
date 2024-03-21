# veScale Optimizers

## Overview

In distributed training, optimizers also need to be adjusted accordingly. We provide two options:

### `BasicOptimizer`

A simple optimizer warpper plus some utilities for distributed training, such as recover flattened gradient from `DDP` and trigger gradient all-reduce for LayerNorm (or some other similar) blocks in Sequence Parallel. `BasicOptimizer` is not a ZeRO optimizer.

### `DistributedOptimizer`

A "ZeRO 2+" optimizer. Simliar to `DDP`, veScale `DistributedOptimizer` is primarily inherited from [Megatron-LM's DistributedOptimizer](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/optimizer/distrib_optimizer.py). We extend compatibility of its implementation with our DTensor.

## Implementation

### `BasicOptimizer`

`BasicOptimizer`'s implementation is quite simple. See the docstring of `BasicOptimizer` at `<repo>/python/vescale/optim/base_optimizer.py`.

### `DistributedOptimizer`

`DistributedOptimizer`'s implementation is complex. Different from `DDP`, in `DistributedOptimizer`, the model parameters and gradients are further split. Each DP rank only obtains the corresponding gradient, updates the corresponding parameters, maintaining the corresponding optimizer states. Therefore, a typical optimizer initialization and step process of `DistributedOptimizer` includes the following stages:

1. At initialzation, model parameters need to be split across all DP ranks, but this is not a `real` split. Each DP rank actually owns a partial view of the original model parameters. Note that this split does not respect parameter boundaries, which means that a parameter could be split into two halves and belong to two DP ranks. Therefore, a complex mapping between the dp-sharded parameters and the original parameters needs to be established, which is mostly done in the init function. At last, we replace the optimizer's param_groups with the dp-sharded parameter.

2. At step, copy `main_grad` attached at original parameter by `DDP` to the dp-sharded parameters.

3. Run `optimizer.step()`.

4. Copy updated dp-sharded parameters to a specific param buffer. To avoid the overhead of sending each parameter individually through an allgather operation, we reused the gradient buffer's space as a parameter buffer. This allow us to store the updated parameters temporarily before they are all-gathered back to their original form before the next forward execution. This strategy helped us save GPU memory. And this introduce a further optimization.

    - We further overlap the param all-gather with the forward, which means we trigger next part's param all-gather when we are doing the current part's forward.

## Compatibility of Optimizer with `DDP`

The compatibility of these two optimizers and `DDP` strategy is shown as follows:

|          | `BasicOptimizer` | `DistributedOptimizer` |
| -------- | ---------------- | ---------------------- |
| `DDP`    |      yes         |        yes             |
| NO `DDP` |      yes         |         no             |

## Example

### `BasicOptimizer`

See `<repo>/test/parallel/ddp_optim/test_ddp.py`.

### `DistributedOptimizer`

A simple usage case is here. For more tests, see `<repo>/test/parallel/ddp_optim/test_doptimizer.py`.

```python
from vescale.ddp.distributed_data_parallel import DistributedDataParallel as DDP
from vescale.optim.distributed_optimizer import DistributedOptimizer
from vescale.dmodule.api import parallelize_module
from vescale.dtensor.device_mesh import DeviceMesh

# create an torch-native model
mlp = MLP()

# create 2-dim DeviceMesh, the first for data-parallel, while the second for tensor-parallel.
device_mesh = DeviceMesh("cuda", [[0, 1], [2, 3]], mesh_dim_names=("DP", "TP"))

# parallelize torch-native model into TP model (see: `<repo>/python/vescale/dmodule/README.md`)
tp_mlp = parallelize_module(mlp, device_mesh["TP"], param_and_fwd_sharding_plan)

# wrap TP model with `DDP` (see: `<repo>/python/vescale/ddp/README.md`)
dp_tp_mlp = DDP(
    module=tp_mlp,
    data_pg_or_device_mesh=device_mesh["DP"],
    overlap_grad_reduce=False,
    use_distributed_optimizer=True
)

# create DistributedOptimizer
doptim = DistributedOptimizer(
    # choose core optimizer class
    torch.optim.Adam,
    # feed model
    models=[dp_tp_mlp],
    # choose whether overlap the param all-gather with the next forward for speeding up
    overlap_param_gather=True or False,
    # feed core optimizer kwargs
    optimizer_kwargs={"lr": 0.01},
)

# training current iteration
dp_tp_mlp(torch.rand(...)).sum().bakward()
# reduce-scatter the gradient across the DP world.
dp_tp_mlp.finish_grad_sync()
# update model
doptim.step()

# training next iteration
doptim.zero_grad()
# <repeat above>

```
