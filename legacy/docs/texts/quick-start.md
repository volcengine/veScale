# Quick Start

First, find the **veScale** [**repo**](https://github.com/volcengine/veScale/tree/main).

## Installation

### From Source

#### Install a Patched Version of PyTorch

```bash
bash [repo]/patches/build_pytorch_w_patch.sh
```

This will compile and install a patched version of PyTorch.

#### Install a Patched Version of TorchDistX

```bash
bash [repo]/patches/build_torchdistX_w_patch.sh
```

This will compile and install a patched version of TorchdistX (based on its master).

#### Install veScale

```bash
pushd python && pip3 install -r requirements.txt && pip3 install -e . && popd
```

This will install **veScale** and its dependencies.

### Docker Image

#### Build the Docker Image

Make sure it is in the veScale directory.

```bash
docker build .
```
It may take a while to build the image.

Once the building process is finished, you can `docker run` with the id.

## Run Examples

- Nano GPT: `<repo>/examples/nanogpt_4D_finetune` 

- Open LLAMA: `<repo>/examples/open_llama_4D_benchmark`

- Mixtral: `<repo>/examples/mixtral_4D_benchmark`