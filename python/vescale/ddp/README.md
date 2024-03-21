# veScale Distributed Data Parallel (DDP)

## Overview

`Distributed Data Parallel` (`DDP`) is a distributed training strategy that partitions the input data across multiple devices, such as multiple GPUs, and replicates the model on each device. On top of this, various ZeRO features can be implemented.

veScale `DDP` is primarily inherited from [Megatron-LM's DDP](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/distributed/distributed_data_parallel.py). We extend the compatibility of the `DDP` implementation with our DTensor.

## Implementation

`DDP` is a module wrapper that creates a flattened grad buffer to store the gradients produced by the model backwarding. This is achieved by adding a hook to the grad_fn of the parameters, which fill DTensor gradient outputed by PyTorch Autograd engine to the pre-allocated grad buffer. The purpose of grad buffer is to accelerate the all-reduce process for gradient updates during distributed training, as it only needs to be performed once for the entire buffer, rather than once per parameter.

On the basis of this, there are some optimizations can be achieved:

1. Overlap gradient all-reduce with backwarding procedure. We can further split the grad buffer into several buckets. Once all gradient in a bucket is ready, we can immediately trigger the gradient all-reduce rather than waiting until the whole grad buffer is ready.

2. Reduce-scatter the gradient rather than all-reduce gradient if we have a veScale `DistributedOptimizer` (a ZeRO 2+ optimizer) installed.

## Example

Following shows a simple code. For more examples, see `<repo>/test/parallel/ddp_optim/*.py`

```python
from vescale.ddp.distributed_data_parallel import DistributedDataParallel as DDP
from vescale.dmodule.api import parallelize_module
from vescale.dtensor.device_mesh import DeviceMesh

# create an torch-native model
mlp = MLP()

# create 2-dim DeviceMesh, the first for data-parallel, while the second for tensor-parallel.
device_mesh = DeviceMesh("cuda", [[0, 1], [2, 3]], mesh_dim_names=("DP", "TP"))

# parallelize torch-native model into TP model (see: `<repo>/python/vescale/dmodule/README.md`)
tp_mlp = parallelize_module(mlp, device_mesh["TP"], param_and_fwd_sharding_plan)

# wrap TP model with `DDP`
dp_tp_mlp = DDP(
    # feed the paralellized module
    module=tp_mlp,
    # feed DP's sub-mesh or just `device_mesh` (i.e., by default we treat the first dim of devicemesh as data-parallelism).
    data_pg_or_device_mesh=device_mesh["DP"],
    # choose whether overlap gradient all-reduce with backwarding procedure for speeding up
    overlap_grad_reduce=True or False,
    # choose whether used `DistributedOptimizer`
    #   if True, `DDP` will be used with `DistributedOptimizer`, so `DDP` reduce-scatter the gradient along data-parallel ranks.
    #   if False, `DDP` just all-reduce the gradient along data-parallel ranks.
    use_distributed_optimizer=True or False
)

# train model
dp_tp_mlp(torch.rand(...)).sum().bakward()
# all-reduce / reduce-scatter the gradient across the DP world.
dp_tp_mlp.finish_grad_sync()
```
