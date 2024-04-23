# torchdistX - Torch Distributed Experimental

[**Installation**](#installation) | [**Getting Started**](#getting-started) | [**Documentation**](#documentation)

Torch Distributed Experimental, or in short torchdistX, contains a collection of
experimental features for which our team wants to gather feedback from our users
before introducing them in the core PyTorch Distributed package. In a sense
features included in torchdistX can be considered in an incubation period.

Please be advised though that all features in torchdistX are subject to change
and, although our team will make its best effort, we do not guarantee any API
or ABI compatibility between releases. This means you should exercise caution if
you plan to use torchdistX in production.

As of today the following features are available in torchdistX:

- [Fake Tensor](https://pytorch.org/torchdistx/latest/fake_tensor.html)
- [Deferred Module Initialization](https://pytorch.org/torchdistx/latest/deferred_init.html)

## Dependencies
torchdistX versions corresponding to each PyTorch release:

| `torch`      | `torchdistx` | `python`          |
| ------------ | ------------ | ----------------- |
| `main`       | `main`       | `>=3.8`, `<=3.10` |
| `1.12.0`     | `0.2.0`      | `>=3.7`, `<=3.10` |
| `1.11.0`     | `0.1.0`      | `>=3.7`, `<=3.9`  |

## Installation
As of today only Linux and macOS operating systems are supported. Please note
that pre-built Conda and PyPI packages are *only* available for Linux though.
For installation on macOS you can follow the instructions in the [From Source](#from-source)
section. At this time there are no plans to introduce Windows support.

### Conda
Conda is the recommended way to install torchdistX. Running the following
command in a Conda environment will install torchdistX and all its dependencies.

**Stable**

For PyTorch CPU:
```
conda install -c pytorch -c conda-forge torchdistx cpuonly
```

For PyTorch with CUDA 10.2:
```
conda install -c pytorch -c conda-forge torchdistx cudatoolkit=10.2
```

For PyTorch with CUDA 11.3:
```
conda install -c pytorch -c conda-forge torchdistx cudatoolkit=11.3
```

For PyTorch with CUDA 11.6:
```
conda install -c pytorch -c conda-forge torchdistx cudatoolkit=11.6
```

**Nightly**

For PyTorch CPU
```
conda install -c pytorch-nightly -c conda-forge torchdistx cpuonly
```

For PyTorch with CUDA 10.2
```
conda install -c pytorch-nightly -c conda-forge torchdistx cudatoolkit=10.2
```

For PyTorch with CUDA 11.3
```
conda install -c pytorch-nightly -c conda-forge torchdistx cudatoolkit=11.3
```

For PyTorch with CUDA 11.6
```
conda install -c pytorch-nightly -c conda-forge torchdistx cudatoolkit=11.6
```

In fact torchdistX offers several Conda packages that you can install
independently based on your needs:

| Package                                                                 | Description                                      |
|-------------------------------------------------------------------------|--------------------------------------------------|
| [torchdistx](https://anaconda.org/pytorch/torchdistx)                   | torchdistX Python Library                        |
| [torchdistx-cc](https://anaconda.org/pytorch/torchdistx-cc)             | torchdistX C++ Runtime Library                   |
| [torchdistx-cc-devel](https://anaconda.org/pytorch/torchdistx-cc-devel) | torchdistX C++ Runtime Library Development Files |
| [torchdistx-cc-debug](https://anaconda.org/pytorch/torchdistx-cc-debug) | torchdistX C++ Runtime Library Debug Symbols     |

### PyPI

**Stable**

For PyTorch CPU:
```
pip install torchdistx --extra-index-url https://download.pytorch.org/whl/cpu
```

For PyTorch with CUDA 10.2:
```
pip install torchdistx --extra-index-url https://download.pytorch.org/whl/cu102
```

For PyTorch with CUDA 11.3:
```
pip install torchdistx --extra-index-url https://download.pytorch.org/whl/cu113
```

For PyTorch with CUDA 11.6:
```
pip install torchdistx --extra-index-url https://download.pytorch.org/whl/cu116
```

**Nightly**

For PyTorch CPU:
```
pip install torchdistx --pre --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

For PyTorch with CUDA 10.2:
```
pip install torchdistx --pre --extra-index-url https://download.pytorch.org/whl/nightly/cu102
```

For PyTorch with CUDA 11.3:
```
pip install torchdistx --pre --extra-index-url https://download.pytorch.org/whl/nightly/cu113
```

For PyTorch with CUDA 11.6:
```
pip install torchdistx --pre --extra-index-url https://download.pytorch.org/whl/nightly/cu116
```

### From Source

#### Prerequisites
- After cloning the repository make sure to initialize all submodules by
  executing `git submodule update --init --recursive`.
- Create a Python virtual environment and install the build dependencies:
 ```
# Build against PyTorch CPU
pip install --upgrade -r requirements.txt -r use-cpu.txt

# Build against PyTorch with CUDA 10.2
pip install --upgrade -r requirements.txt -r use-cu102.txt

# Build against PyTorch with CUDA 11.3
pip install --upgrade -r requirements.txt -r use-cu113.txt

# Build against PyTorch with CUDA 11.6
pip install --upgrade -r requirements.txt -r use-cu116.txt
```
- The build process requires CMake 3.21 or later. You can install an up-to-date
  version by executing `pip install cmake`. For other environments please refer
  to your package manager or [cmake.org](https://cmake.org/download/).

Once you have all prerequisites run the following commands to install the
torchdistX Python package:

```
cmake -DTORCHDIST_INSTALL_STANDALONE=ON -B build
cmake --build build
pip install .
```

For advanced build options you can check out [CMakeLists.txt](./CMakeLists.txt).

#### Development
In case you would like to contribute to the project you can slightly modify the
commands listed above:

```
cmake -B build
cmake --build build
pip install -e .
```

With `pip install -e .` you enable the edit mode (a.k.a. develop mode) that
allows you to modify the Python files in-place without requiring to repeatedly
install the package. If you are working in C++, whenever you modify a header or
implementation file, executing `cmake --build build` alone is sufficient. You do
not have to call `pip install` again.

The project also comes with a [requirements-devel.txt](./requirements-devel.txt)
to set up a Python virtual environment for development.

```
# Build against PyTorch CPU
pip install --upgrade -r requirements-devel.txt -r use-cpu.txt

# Build against PyTorch with CUDA 10.2
pip install --upgrade -r requirements-devel.txt -r use-cu102.txt

# Build against PyTorch with CUDA 11.3
pip install --upgrade -r requirements-devel.txt -r use-cu113.txt

# Build against PyTorch with CUDA 11.6
pip install --upgrade -r requirements-devel.txt -r use-cu116.txt
```

#### Tip
Note that using the Ninja build system and the ccache tool can significatly
speed up your build times. To use them you can replace the initial CMake command
listed above with the following version:

```
cmake -GNinja -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -B build
```

## Getting Started

### Fake Tensor
Fake tensors, similar to meta tensors, carry no data; however, unlike meta
tensors which report `meta` as their device, fake tensors act as if they were
allocated on a real device. In the example below we construct two fake tensors
with the `fake_mode` context manager.

```python
>>> import torch
>>> from torchdistx import fake
>>>
>>> with fake.fake_mode():
...    a = torch.ones([10])
...    b = torch.ones([20], device="cuda")
...
>>> a
tensor(..., size=(10,), fake=True)
>>> b
tensor(..., size=(20,), device=cuda, fake=True)
```

### Deferred Module Initialization
This feature forces all tensors of a module to be constructed as fake while also
recording all operations performed on them. The module, its submodules, and its
tensors can later be materialized by calling the `materialize_module()` and
`materialize_tensor()` functions.

```python
>>> import torch
>>> from torchdistx import deferred_init
>>>
>>> m = deferred_init.deferred_init(torch.nn.Linear, 10, 20)
>>> m.weight
Parameter containing:
tensor(..., size=(20, 10), requires_grad=True, fake=True)
>>>
>>> deferred_init.materialize_module(m)
>>> m.weight
Parameter containing:
tensor([[-0.1838, -0.0080,  0.0747, -0.1663, -0.0936,  0.0587,  0.1988, -0.0977,
         -0.1433,  0.2620],
       ..., requires_grad=True)
```

## Documentation
For more documentation, see [our docs website](https://pytorch.org/torchdistx/latest).

## Contributing
Please refer to [CONTRIBUTING.md](./CONTRIBUTING.md).

## License
This project is BSD licensed, as found in the [LICENSE](LICENSE) file.
