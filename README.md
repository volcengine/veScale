# veScale: A PyTorch Native LLM Training Framework

## Coming Soon

We are refactoring our [internal LLM training system](https://arxiv.org/abs/2402.15627) components to meet open source standard. The tentative timeline is as follows:

1. by mid April, 4D parallelism (tensor parallelism, sequence parallelism, data parallelism and ZERO) examples for nanoGPT, Llama2 and Mixtral models
2. by end of May, fast checkpointing system
3. by end of July, CUDA event monitor, pipeline parallelism and supporting components for large-scale training

## Installation

### From Source

#### Install a Patched Version of PyTorch (optional)

```bash
bash patches/build_pytorch_w_patch.sh
```

This will compile and install a patched version of PyTorch (based on v2.2.1_rc3).
The patch code can be found here: [PyTorch-Patch](patches/patched_pytorch_v2.2.1_rc3.patch)

#### Install a Patched Version of TorchDistX

```bash
bash patches/build_torchdistX_w_patch.sh
```

This will compile and install a patched version of TorchdistX (based on its master).
The patch code can be found here: [TorchDistX-Patch](patches/patched_torchdistX_9c1b9f.patch)

#### Install veScale

```bash
pushd python && pip3 install -r requirements.txt && pip3 install -e . && popd
```

This will install veScale and its dependencies.

### Docker Image

#### Build the Docker Image

Make sure it is in the Vescale directory.

```bash
docker build .
```
It may take a while to build the image.

Once the building process is finished, you can `docker run` with the id.



## [License](./LICENSE)

The veScale Project is under the Apache License v2.0.

## Acknowledgement

veScale team would like to sincerely acknowledge the assistance of and collaboration with
the [PyTorch DTensor team](https://github.com/pytorch/pytorch/tree/main/torch/distributed/_tensor).
