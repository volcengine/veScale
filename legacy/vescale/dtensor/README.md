# DTensor (Distributed Tensor)

## TLDR

<img src="../../docs/pictures/dtensor.png" alt="DTensor" width="350"/>

## Why DTensor?

- `torch.Tensor` lacks the semantic of being distributed across multiple devices and running distributed operators

- Manually managing `torch.Tensor` in distributed settings is painful and error-prone, as it demands the manual handling of the sharded storage on each device, the collective communication among devices, and the operator kernel split across devices, all with great care.

## What is DTensor?

- `DTensor (Distributed Tensor)` provides a single-device abstraction for multiple-device `torch.Tensor` and empowers user to write distributed training/inference code as if on a single device (i.e., SPMD)

- `DTensor` transparently handles all distributed logic under the hood (sharded storage on each device, the collective communication among devices, and the operator kernel split across devices)

- `DTensor` is implemented by a wrapper class on `torch.Tensor` with a meta data `DTensorSpec` describing:

    - which multiple devices (`DeviceMesh`) is distributed upon

        - it can be 1D mesh of two GPUs: `DeviceMesh("cuda", [0, 1])`
        - it can be 2D mesh of four GPUs: `DeviceMesh("cuda", [[0, 1], [2, 3]])`

    - how is `DTensor` placed (`Placement`) on the `DeviceMesh`:
    
        - there are three main `Placement`:

            - `Shard(<tensor_dim>)`: `DTensor`'s `<tensor_dim>` is sharded on the `DeviceMesh`
            - `Replicate`: `DTensor` is replicated on the `DeviceMesh`
            - `Partial`: `DTensor` is a partial product on the `DeviceMesh` with pending sum (`AllReduce`) to be a total product
        
        - where a list of `Placement` is needed to define the `placements` of a `DTensor`:

            - `placements = [Shard(1)]` means `DTensor`'s tensor dim #1 is sharded along `DeviceMesh`'s dim #0 (i.e., the #0 element in the list)

            - `placements = [Shard(1), Shard(0)]` means `DTensor`'s tensor dim #1 is sharded along `DeviceMesh`'s dim #0 and `DTensor`'s tensor dim #0 is sharded along `DeviceMesh`'s dim #1

            - `placements = [Shard(1), Replicate()]` means `DTensor`'s tensor dim #1 is sharded along `DeviceMesh`'s dim #0 and `DTensor`'s rest tensor dim #0 is replicated along `DeviceMesh`'s dim #1

    - what is the global tensor shape & stride (`TensorMeta`) of this `DTensor`

- `DTensor` operators (e.g., `torch.add`) are implemented by `ShardingPropagator` which propagates `placements` from input to output for each operator with pre-registered sharding rules and strategies

## What is veScale DTensor? How's different from PyTorch DTensor?

- veScale is a PyTorch-native framework rooted in _**PyTorch DTensor**_

- _**veScale DTensor**_ extends and enhances the _**PyTorch DTensor**_ for our production standard with extra features as below:

    - enabled "correct random ops" under abitrary sharding and uneven sharding, i.e., always guarantee random op sharded on multi device is equal to random op on a single device.

    - enabled DTensor support for third-party plug-in ops (e.g., `APEX`) by unleashing `DTensor.data_ptr` and handling asynchronous collective tensors (e.g., in `from_local`, `to_local`, `redistribute`)

    - make implicit `_Partial` to explicit `Partial` placement for optimized initialization, output, and checkpoint (with an extra dispatch mode)

    - enabled DTensor ops that were not implemented in PyTorch for forward or/and backward:
        - `argmax` 
        - `argmin`
        - `topk`
        - `_unique2`
        - `scatter_` 
        - `scatter` 
        - `select`
        - `alias`
        - `index_put_` 
        - `index_put` 
        - `index_add_`
        - `_scaled_dot_product_flash_attention`
        - `_scaled_dot_product_efficient_attention`
        - `expand_as`
        - `one_hot`
        - `where`
        - `Embedding` in vocabular parallel
    
    - support uneven sharding in conversion between `DTensor` and `torch.Tensor`

    - decoupled special op handling that bypasses DTensor dispatching (`_bypass_for_dispatch`)

    - enabled patching before (`_pre_patch_for_dispatch`) and after (`_post_patch_for_dispatch`) DTensor dispatch, for adding user's custom dispatching logic without coupling original dispatch logic

    - enabled short-cut for ops to bypass sharding propagation entirely (`_bypass_for_sharding_prop`):

    - bypassed `tensor_meta` propagation for ops:
        - with output DTensor as pure `Replicate`, by using local output Tensor's `tensor_meta`
        - with registered `tensor_meta` propagation under `dtensor/ops` (e.g., `conv`, `slice`, `copy`, `clone`, `bucketize`, `t`) 
        - excluding ops in `recompute_tensor_meta_list` (e.g., `clone`, `native_dropout`, `nll_loss_forward`)
        
    - enabled `DeviceMesh` on `meta` device type
    
    - enabled `DeviceMesh` initialization from an existing processs group

    - enabled `DeviceMesh` being split into a list of sub meshes

    - disabled redistributed input:
        - torch DTensor allows each op select its best sharding _strategy_ for input-output sharding based on a cost model capturing input redistribution communication and then redistributes input DTensor to selected input-sharding. 
        - But we currently disable this feature (via environment var `VESCALE_DISABLE_REDISTRIBUTE`), as we don't expect uncontrollable resharding and implicit communication in DTensor dispatch for production. (Ideally, all resharding and communication should be controlled by the end users.) 

    - support deferred initiailization and materialization for DTensor with extended `torchdistx`

    - [experimental] developed `InterleavedShard` placement to support merged QKV in MHA

    - [experimental] extreme performance with C++ DTensor

    - [experimental] extreme performance with dispatching-free DTensor

## How to use veScale DTensor manually?

- Example of `matmul`:

    ``` python
    # create a four-device mesh
    device_mesh = DeviceMesh("cuda", [0, 1, 2, 3])

    # single device matmul
    t1 = torch.ones(12, 8, device="cuda")
    t2 = torch.ones(8, 16, device="cuda")
    t3 = torch.mm(t1, t2)

    # multiple device matmul
    dt1 = distribute_tensor(t1, device_mesh, [Shard(dim=1)]) # colwise shard (tensor dim 1) t1 along device mesh's dim 0
    dt2 = distribute_tensor(t2, device_mesh, [Shard(dim=0)]) # rowwise shard (tensor dim 0) t2 along device mesh's dim 0
    dt3 = torch.mm(dt1, dt2)
    assert isinstance(dt3, DTensor)
    assert dt3.placements[0].is_partial() # product t3 is partial sharded on device mesh
    dt4 = dt3.redistribute(device_mesh, [Replicate()]) # reshard t3 with allreduce to replicate

    # match DTensor and Tensor result
    assert torch.equal(dt4.to_local(), t3) 
    ```

- APIs can be found under `<repo>/vescale/dtensor/api.py`

- More examples can be found under `<repo>/test/dtensor/*/*.py`

- Original examples can be found in PyTorch [DTensor](https://github.com/pytorch/pytorch/tree/main/torch/distributed/_tensor).


## What if encountering an operator that is not supported by DTensor yet?

-- _Register DTensor "Ops" for Sharding Propagation!_

### Why register DTensor Ops for sharding propagation?

Sharding propagation is an important step in DTensor dispatch. It is responsible for inferring the output sharding info (i.e., `DTensorSpec`) from the input sharding info at each operator. So that the all ops of an entire model can be expressed in DTensor.

### How to register a DTensor Op for sharding propagation?

There are two ways to register sharding propagation, namely:

- **rule-based** way (deprecated by upstream, will be converted to strategy-based for all ops in future)
- **strategy-based** way

They're the same thing intrinsically. But the difference between the **rule-based** and **strategy-based** way is that the former only needs to consider the current input `DTensorSpec` while the later requires enumerating all valid (input `DTensorSpec`, output `DTensorSpec`) pair for a single op.

The pros of the rule-based way is the ease of use, while pros of the strategy-based way is having all possible combinations of input-output sharding -- a context info necessary for automatically selecting the best _strategy_ for input-output sharding (e.g., the one with the minimal DTensor redistribution cost). 

It's recommended to use **strategy-based** way to register sharding propagation. But if you encounter a really complex custom op, **rule-based** way might be the better choice.

### Example of the rule-based sharding propagation registration

``` python
@register_prop_rule(
    [torch.ops.aten.native_layer_norm.default], # specify the op you want to register sharding propagation
    schema_info=RuntimeSchemaInfo(1) # see docstring of class ``RuntimeSchemaInfo``
)
# the arguments for every operator sharding propagation is the same.
# `op_shcema`: OpSchema object, storing the input DTensorSpec of current  operator.
def prop_layer_norm_rule(op_schema: OpSchema) -> OutputSharding:
    # extract input DTensorSpec from op_schema
    (
        input_dtensor_spec,
        normalized_shape,
        weight_dtensor_spec,
        bias_dtensor_spec,
        _
    ) = op_schema.args_schema

    # optional: type check
    assert isinstance(input_dtensor_spec, DTensorSpec)
    assert isinstance(normalized_shape, (List, Sequence, torch.Size))
    assert isinstance(weight_dtensor_spec, DTensorSpec)
    assert isinstance(bias_dtensor_spec, DTensorSpec)
    
    # input DTensorSpec validation check
    assert all(isintance(p, Replicate) for p in weight_dtensor_spec.placements)
    assert all(isintance(p, Replicate) for p in bias_dtensor_spec.placements)

    # calculate the output DTensorSpec
    # for native_layer_norm, output placements is just the same as the input placements.
    output_placements = input_dtensor_spec.placements

    # return OutputSharding object.
    return OutputSharding(
        output_spec=DTensorSpec(
            mesh=weight_dtensor_spec.mesh,
            placements=output_placements,
            tensor_meta=input_dtensor_spec.tensor_meta
        )
    )

```

### Example of the strategy-based sharding propagation registration

``` python
@register_op_strategy(
    [torch.ops.aten.native_layer_norm.default],
    schema_info=RuntimeSchemaInfo(1),
)
# the arguments for every operator sharding propagation is the same.
# `mesh`: DeviceMesh that the current operator is running.
# `op_shcema`: OpSchema object, storing the input OpStrategy of current  operator.
def layer_norm_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> OpStrategy:
    # extract placement strategy of op_schema
    (
        input_strategy,
        normalized_shape,
        weight_strategy,
        bias_strategy,
        _,
    ) = op_schema.args_schema
    
    output_strategy = OpStrategy([])

    # enumerate input placement strategies.
    for idx, input_placement_strategy in enumerate(input_strategy.strategies):
        op_args_specs = []
        # the output DTensorSpecs of arguments are the inputs of current op
        input_spec = input_placement_strategy.output_spec
        weight_spec = weight_strategy.output_spec
        bias_spec = bias_strategy.output_spec

        # DTensorSpec validation check
        ...

        op_args_specs.append(input_spec)
        op_args_specs.append(weight_spec)
        op_args_specs.append(bias_spec)

        output_spec = input_spec
        # generate all valid strategies, i.e., (input DTensorSpec, output DTensorSpec) pairs.
        output_strategy.strategies.append(
            PlacementStrategy(
                output_spec=output_spec,
                input_specs=op_args_specs,
            )
        )
    
    # return OpStrategy object containing a list of strategies, where one strategy will be selected during sharding propagation
    return output_strategy
    
```

## How to generate random numbers in DTensor as if it's from a single GPU?

### Motivation

Ideally, DTensor should provide single-device abstraction even for random ops (e.g. `dtensor.randn`, `nn.Dropout`, and `<any random ops>`), i.e., random value generated on single device should be identical to collective of random shard on multiple devices.


### Problem 

PyTorch DTensor (i.e., `OffsetBasedRNGTracker`) does not produce the random values on multiple devices identical to single GPU execution for random operators (e.g. `dtensor.randn`, `nn.Dropout`, and `<any random ops>`).

The key problem lies in that the CUDA random numbers are not generated "sequentially" and cannot be simply offsetted by rank ids, but instead are generated "simultaneously" by multiple CUDA threads and only be sharded by CUDA thread ids! 

### Solution

In veScale, we introduce a `ThreadBasedRNGTracker` for correcting the RNG states across different GPUs, enabling generation of correct DTensor that are identical to the ones from single GPUs for any random ops.

To use the feature, build and install a patched PyTorch of veScale and set the environment variable `VESCALE_SINGLE_DEVICE_RAND=1`.

### Details

Whenever invoking a randomized operation on a DTensor, `ThreadBasedRNGTracker` passes its sharding info to the C++/Cuda side of PyTorch through the RNG state.
This resolves the issue that PyTorch DTensor's `OffsetBasedRNGTracker` does not produce the output identical to single GPU executions.

For example, consider generating `x = torch.rand(4)` given the current random seed and
a global offset. In Cuda's RNG implementation, random numbers are accessed via a triple
`(seed, thread id, offset)`.

On a single GPU, 4 GPU threads is created and the i-th thread fills the entry `x[i]`
with `rand(seed, i, offset)`. That is, we have
```
    | Thread 0        | Thread 1        | Thread 2        | Thread 3        |
x = | rand(0, offset) | rand(1, offset) | rand(2, offset) | rand(3, offset) |
```
After the execution of `torch.rand(4)`, the global offset increments by 4, which is the
granularity of cuda's RNG offsets.

The global offset increments by the size of the randomness used in each thread, rounded
up to the nearest multiple of 4. For instance, if 1000 GPU threads is used to generate
7000 random numbers, each thread takes 7 random numbers from Cuda RNG and the global offset increases by 8 afterward.

However, using `OffsetBasedRNGTracker`, it outputs a different tensor given 2 GPUs.
```
    | GPU 0                                 | GPU 1                                     |
    | Thread 0 of GPU 0 | Thread 1 of GPU 0 | Thread 0 of GPU 1   | Thread 1 of GPU 1   |
x = | rand(0, offset)   | rand(1, offset)   | rand(0, offset + 4) | rand(1, offset + 4) |
```
Furthermore, after the execution, the global offset increments by 8 instead of 4.

To resolve the issue, each physical thread of each GPU should fill the entry using the
thread id as if there is only one GPU. In the previous example, the output should be
```
    | GPU 0                                         | GPU 1                                         |
    | Thread 0 of GPU 0     | Thread 1 of GPU 0     | Thread 0 of GPU 1     | Thread 1 of GPU 1     |
x = | rand(seed, 0, offset) | rand(seed, 1, offset) | rand(seed, 2, offset) | rand(seed, 3, offset) |
```
And after the execution, the global offset should increment by 4.
This can be done if we pass the sharding info into Cuda functions that generate these
outputs.


## Acknowledgement

We would like to acknowledge the assistance of and collaboration with
the [PyTorch DTensor team](https://github.com/pytorch/pytorch/tree/main/torch/distributed/_tensor).