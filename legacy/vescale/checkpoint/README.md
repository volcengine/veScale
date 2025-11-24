# veScale Checkpoint

`vescale.checkpoint` is an automatic distributed checkpointing system for LLM training and inference.

## New Features

[05/30/2024] We improved `vescale.checkpoint` with the following new features for fast checkpointing (where front three features are built-in techniques without necessitating manual activation):

- **Saving Plan Caching**: During training, the program may save model and optimizer checkpoints every n steps. Once a saving plan is created, it remains unchanged as long as the model does. We implemented plan caching to avoid regenerating the plan when checkpointing a model or optimizer multiple times, reducing unnecessary compute and communication costs. As of 05/30/2024, PyTorch DCP does not support plan caching.

- **Saving Plan Load-Balancing**: In data parallel training, models are replicated across GPUs with different data parallel ranks but the same pipeline and tensor parallel ranks. Existing PyTorch DCP (as of 05/30/2024) deduplicates replicated tensors using a simple algorithm, causing GPUs with data parallel rank 0 to save the entire model, leading to load imbalance. We implemented a load-balancing algorithm to address this issue when deduplicating model tensors.

- **D2H Tensor Copying via Pinned Memory**: When copying tensors from GPU to host memory, `vescale.checkpoint` uses pinned host memory, reducing memory allocation costs each time a checkpoint is saved. As of 05/30/2024, PyTorch DCP does not support pinned memory.

- **Checkpoint Broadcasting**: In data parallel training, models are replicated across GPUs with different data parallel ranks but the same pipeline and tensor parallel ranks. If `broadcast_checkpoint` is enabled, `vescale.checkpoint.load` lets GPUs with data parallel rank 0 to load the model and broadcast it to other GPUs with higher data parallel ranks. If GPUs are connected with NCCL, broadcasting model tensors speeds up checkpoint loading compared to all GPUs loading models from persistent storage. E.g.:

    ```python
    # prepare checkpoint state for the model and optimizer
    checkpoint_state = { "model": distributed_model, "optimizer": distributed_optimizer }
    # load the checkpoint
    vescale.checkpoint.load("/user/vescale/gpt/", checkpoint_state, broadcast_checkpoint=True)
    ```

- **Asynchronous Checkpointing**: When `vescale.checkpoint.save` is called, it first generates a saving plan and then synchronously copies tensors from GPU to host memory. If `async_checkpoint` is enabled, the training program can continue after the D2H copying, while `vescale.checkpoint.save` continues to serialize tensors and dump the checkpoint to persistent storage asynchronously without blocking training. As of 05/30/2024, PyTorch DCP does not support asynchronous checkpointing. E.g.:

    ```python
    # prepare checkpoint state for the model and optimizer
    checkpoint_state = { "model": distributed_model, "optimizer": distributed_optimizer }
    # save the checkpoint asynchronuously
    vescale.checkpoint.save("/user/vescale/gpt/", checkpoint_state, async_checkpoint=True)
    ```

## Why `vescale.checkpoint`?

1. Manually managing distributed checkpointing, such as writing model saving/loading/resharding scripts under complex distributed environments, is painful and error-prone.

2. `torch.save` and `torch.load` lacks the capability of managing checkpointing in distributed settings, let alone resharding checkpoints for different distributed settings. 
Although existing systems extend `torch.save` for saving checkpoints on multiple GPUs or machines, the saved checkpoints are heavily coupled with a single distributed setting like the degrees of data, tensor and pipeline parallelism. Consequently, existing systems with `torch.load` fail to load checkpoints with varying degrees of parallelism, which is common in elastic training or switching between training and fine-tuning.

3. `PyTorch Distirbuted Checkpoint` indeed supports checkpoint resharding to some extent. Nonetheless, it currently only supports resharding for the simplest data parallelism, but not for the complex tensor nor pipeline parallelism, which are commonly used in 3D parallelism of LLM training. Furthermore, it does not support load-time resharding for Distributed Optimizer, nor provide decent performance optimizations.

## What is `vescale.checkpoint`?

`vescale.checkpoint` offers simple and straightforward APIs,
enabling users to load and save distributed model (e.g., `DModule`) and optimizer (e.g., `DistributedOptimizer`) seamlessly, abstracting away the complexities of underlying details such as process rank and device mesh.  

`vescale.checkpoint` supports load-time checkpoint resharding when varying the degrees of data, tensor, or pipeline parallelism for both veScale model (e.g., `DModule`) and optimizer (e.g., `DistributedOptimizer`).  

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

- Loading checkpoint (under different world size or 3D parallel sizes):

    ```python
    # prepare checkpoint state for the model and optimizer
    checkpoint_state = { "model": distributed_model, "optimizer": distributed_optimizer }
    # load the checkpoint
    vescale.checkpoint.load("/user/vescale/gpt/", checkpoint_state)
    ```

- APIs can be found in: `<repo>/vescale/checkpoint/__init__.py`

- End-to-end example can be found in: `<repo>/examples/nanogpt_4D_finetune/finetune_4D.py`

- More examples can be found under `<repo>/test/checkpoint/*.py` and `<repo>/examples/`

- Original examples can be found in PyTorch [Distributed Checkpoint](https://github.com/pytorch/pytorch/tree/main/torch/distributed/checkpoint)
