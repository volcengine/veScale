FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ARG MAX_JOBS=32

WORKDIR /app

COPY ./patches /app

ENV TZ=Etc/UTC \
    DEBIAN_FRONTEND=noninteractive \
    NINJA_MAX_JOBS=$MAX_JOBS \
    MAX_JOBS=$MAX_JOBS \
    TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"\
    TORCH_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=1"\
    _GLIBCXX_USE_CXX11_ABI=1\
    PYTORCH_VERSION="2.2.0a0+git6c8c5ad"


RUN apt-get update \
     && apt-get install -y software-properties-common \
     && apt-get install -y python3.10 python3-pip git git-lfs cmake ninja-build build-essential gcc g++ \
     && apt-get clean \
     && rm -rf /var/lib/apt/lists/* \
     && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
     && ln -sf /usr/bin/pip3.10 /usr/bin/pip3 \
     && pip3 install --upgrade pip setuptools wheel \
     && apt update \
     && apt install -y python-is-python3 openssh-server \
     && apt clean

RUN bash build_pytorch_w_patch.sh

RUN bash build_torchvision.sh

RUN bash build_torchdistX_w_patch.sh


RUN pip3 install --no-cache-dir packaging \
    && pip3 install --no-cache-dir mpmath==1.3.0 \
    && pip3 install --no-cache-dir "setuptools>=69.0.0" \
    && pip3 install --no-cache-dir regex \
    && pip3 install --no-cache-dir pillow \
    && pip3 install --no-cache-dir pybind11 \
    && pip3 install --no-cache-dir einops \
    && pip3 install --no-cache-dir expecttest \
    && pip3 install --no-cache-dir hypothesis \
    && pip3 install --no-cache-dir pytest \
    && pip3 install --no-cache-dir tqdm \
    && pip3 install --no-cache-dir optree \
    && pip3 install --no-cache-dir psutil \
    && pip3 install --no-cache-dir transformers==4.37.2 \
    && pip3 install --no-cache-dir accelerate \
    && pip3 install --no-cache-dir grpcio \
    && pip3 install --no-cache-dir grpcio-tools