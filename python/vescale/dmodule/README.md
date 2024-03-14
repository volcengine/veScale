# veScale DModule (Distributed Module)

## Why veScale DModule?

- `nn.Module` lacks the semantic of being distributed across multiple devices and running distributed operators

- Manually managing `DTensor` and `Tensor` within a `nn.Module` in distributed settings is painful and error-prone.

## What is veScale DModule?

- `DModule (Distributed Module)` provides a single-device abstraction for multiple-device `nn.Module` and empowers user to write distributed training/inference code as if on a single device (i.e., SPMD)

- `DModule` unifies Module-level Tensor Parallelism and Sequence Parallelism by transparently handling distributed logic under the hood:
    - convert `Tensor` to `DTensor` within a `nn.Module` 
    - manage `DTensor` sharding and resharding during forward and backward
    - configure (re)sharding of `DTensor` via Module-level API `parallelize_module()` with given `sharding_plan`
    - allow `sharding_plan` to be either:
        - imported from a pre-defined "plan zoo" 
        - given by "manually written" json
        - [experimental] given by "automatical plan generation" of veScale
    - handles gradient synchronization automatically in backward
    - support deferred initialization with `deferred_init()` (i.e., initialize with `Fake` Tensor without allocating memory, then shard Fake Tensor with TP, and then materialize only a shard of `Tensor` on device)
    - support third-party plug-in Module (e.g. APEX)
    - provide patch interface for customized Module-level hacking
    - extend to optional DP, optional FSDP, and optional EP (in the future)
    - provide debuggability for easy dumping, printing, listing, etc.

## Difference of veScale DModule from PyTorch `parallelize_module`?

- veScale `DModule` is inspired by PyTorch's [`parallelize_module`](https://pytorch.org/docs/stable/_modules/torch/distributed/tensor/parallel/api.html#parallelize_module), but is developed with explicit Module-level abstraction with complete features for our production usage.

- veScale `DModule` extends PyTorch `parallelize_module` with extra features as below (i.e., major differences from PyTorch-v2.2.0): 
    - nD Tensor Parallelism
    - Sequence Parallelism
    - auto gradient synchronization
    - deferred initialization
    - third-party plug-in Module
    - module-level patch interface
    - [experimental] automatical plan generation

## How to use veScale DModule ``manually''?

- Example of `MLP`:

    ``` python
    # torch native code on single device
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 8)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(8, 4)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    # deferred initialization
    mlp = deferred_init(MLP) 
    # or regular initialization
    # mlp = MLP()
    # parallelize model into DModule with "maunal plans"
    dmlp = parallelize_module(mlp, 
                            DeviceMesh("cuda", [0, 1, 2, 3]), 
                            param_sharding_plan={
                                "fc1.weight": [Shard(0)],
                                "fc1.bias": [Shard(0)],
                                "fc2.weight": [Shard(1)],
                                "fc2.bias": [Replicate()],
                            }, 
                            fwd_resharding_plan={
                                "fc1.input": [[Replicate()]], # change to Shard(<dim>) for SP/DP
                                "fc2.output": [[Replicate()]],
                            })
    # forward in TP
    output = dmlp(input)
    # backward in TP
    output.sum().backward()
    # wait for gradient synchronization (which can be hidden when using veScale optimizer)
    dmlp.finish_grad_sync()

    ```

- More details can be found in `<repo>/python/vescale/dmodule/api.py`

- More examples can be found under `<repo>/test/dmodule/*.py`
