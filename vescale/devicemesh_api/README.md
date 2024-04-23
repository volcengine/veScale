# veScale nD Device Mesh

## Overview
`veDeviceMesh` is an advanced API that is built on top of PyTorch upstream’s higher level abstraction [`DeviceMesh`](https://pytorch.org/tutorials/recipes/distributed_device_mesh.html). This API enhances the existing capabilities of DeviceMesh, enabling effective 5D parallelization strategies and easy-to-use APIs. 

## Implementation
Designed to seamlessly integrate with veScale’s Distributed `Data Parallel`, `Tensor/Sequence` (TP/SP), `DistributedOptimizer` and `Pipeline Parallel` APIs, veDeviceMesh ensures superior compatibility and performance by meticulously managing sub-DeviceMeshes and process groups. Additionally, veDeviceMesh provides user-friendly tools for querying strategy coordinates, attributes of parallel dimensions, and overall `DeviceMesh` configurations, making it a highly accessible and efficient solution for developers.

veDeviceMesh embraces following user practices:
1. “A DeviceMesh, but better”
2. One “Mesh” fits all: users don’t need to worry about meddling with DeviceMesh and ProcessGroups’ throughout the course of training. Additionally, users make the most out of the same DeviceMesh to enable hybrid parallelization training.
3. Easy to extend: for more refined capabilities for imminent parallelization methods in the future, veDeviceMesh provides mature APIs to extend new functionalities without breaking the semantics of communication

## Example
Below is a simple demo of veDeviceMesh API.

```python
from vescale.dmodule.api import parallelize_module
from vescale.devicemesh_api import veDeviceMesh
from vescale.ddp.distributed_data_parallel import DistributedDataParallel as DDP
from vescale.optim.distributed_optimizer import DistributedOptimizer
from ... import GPT


dp_size = tp_size = 2
data_set = ...
sharding_plan = ...

# load GPT-2 model from pretrained weights
model = GPT()

# initialize veDeviceMesh API with a global DeviceMesh of size (2, 2)
veDeviceMesh.init_device_mesh(
    "cuda", mesh_shape=(dp_size, tp_size), mesh_dim_names=("DP", "TP"),
)
...
# wrap DModule (TP/SP)
if veDeviceMesh.get_strategy_size("TP") > 1:
    # use veDeviceMesh to obtain global DeviceMesh's tensor parallelism view
    model = parallelize_module(model, device_mesh["TP"], shardin_plan, ...)

# wrap DDP module
if veDeviceMesh.get_strategy_size("DP") > 1:
    # use veDeviceMesh to obtain ProcessGroup for data parallelism
    model = DDP(
        model,
        veDeviceMesh["DP"],
        ...
    )

# build base optimizer
optimizer = ...

# build distributed optimizer
if veDeviceMesh.get_strategy_size("DP") > 1:
    optimizer = DistributedOptimizer(
        optimizer,
        models=[model],
    )

# Train model with fwd+bwd+step
for X, Y in data_set:
    # use veDeviceMesh to tensor parallel dimension size
    tp_mesh = veDeviceMesh.get_tensor_parallel_mesh()
    ...
    optimizer.zero_grad()
    _, output = model(X, Y)
    loss = ...
    loss.backward()
    ...
    optimizer.step()
```

- More examples can be found under `<repo>/test/parallel/devicemesh_api/*.py`
