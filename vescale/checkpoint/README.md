# veScale Checkpoint

`vescale.checkpoint` is an automatic distributed checkpointing system for LLM training and inference.

## Why `vescale.checkpoint`?

1. Manually managing distributed checkpointing, such as writing model saving/loading/resharding scripts under complex distributed environments, is painful and error-prone.

2. `torch.save` and `torch.load` lacks the capability of managing checkpointing in distributed settings, let alone resharding checkpoints for different distributed settings. 
Although existing systems extend `torch.save` for saving checkpoints on multiple GPUs or machines, the saved checkpoints are heavily coupled with a single distributed setting like the degrees of data, tensor and pipeline parallelism. Consequently, existing systems with `torch.load` fail to load checkpoints with varying degrees of parallelism, which is common in elastic training or switching between training and fine-tuning.

3. `PyTorch Distirbuted Checkpoint` indeed supports checkpoint resharding to some extent. Nonetheless, it currently only supports resharding for the simplest data parallelism, but not for the complex tensor nor pipeline parallelism, which are commonly used in 3D parallelism of LLM training. Furthermore, it does not support load-time resharding for Distributed Optimizer, nor provide decent performance optimizations.

## What is `vescale.checkpoint`?

`vescale.checkpoint` offers simple and straightforward APIs,
enabling users to load and save distributed model (e.g., `DModule`) and optimizer (e.g., `DistributedOptimizer`) seamlessly, 
abstracting away the complexities of underlying details such as process rank and device mesh.  

`vescale.checkpoint` supports load-time checkpoint resharding when varying the degrees of data, tensor, or pipeline (TODO) parallelism for both veScale model (e.g., `DModule`) and optimizer (e.g., `DistributedOptimizer`).  

`vescale.checkpoint` incorporates [fast checkpointing](https://arxiv.org/abs/2402.15627) and various I/O optimization techinques, enhancing I/O efficiency during LLM training.  

`vescale.checkpoint` is built on top of `PyTorch Distributed Checkpoint` with significant differences as discussed above.

## How to use `vescale.checkpoint`?

- Saving checkpoint: 

```python
# prepare checkpoint state for the model and optimizer
checkpoint_state = { "model": distributed_model, "optimizer": distributed_optimizer }
# save the checkpoint
vescale.checkpoint.save("/user/vescale/gpt/", checkpoint_state)
```

- Loading checkpoint (under different world size or 3D parallelism degrees):

```python
# prepare checkpoint state for the model and optimizer
checkpoint_state = { "model": distributed_model, "optimizer": distributed_optimizer }
# load the checkpoint
vescale.checkpoint.load("/user/vescale/gpt/", checkpoint_state)
```

- APIs can be found in: `<repo>/vescale/checkpoint/__init__.py`

- More examples can be found under `<repo>/test/checkpoint/*.py` and `<repo>/examples/`

- Original examples can be found in PyTorch [Distributed Checkpoint](https://github.com/pytorch/pytorch/tree/main/torch/distributed/checkpoint)