// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "module.h"

#include <ATen/Context.h>
#include <ATen/Tensor.h>
#include <torch/csrc/utils/cuda_lazy_init.h>
#include <torch/csrc/utils/pybind.h>
#include <torchdistx/fake.h>

namespace torchdistx::python {
namespace {

void pyEnterFakeMode(bool fake_cuda) {
  enterFakeMode(fake_cuda);

  // If CUDA is not available, suppress PyTorch's attempt to initialize its CUDA
  // subsystem which would fail and prevent us from instantiating CUDA devices.
  if (fake_cuda) {
    if (!at::hasCUDA()) {
      torch::utils::set_requires_cuda_init(false);
    }
  }
}

void pyLeaveFakeMode() {
  leaveFakeMode();

  if (!isFakeModeActive() && !at::hasCUDA()) {
    torch::utils::set_requires_cuda_init(true);
  }
}

}  // namespace

void initFakeFunctions(pybind11::module& m) {
  m.def("enter_fake_mode", pyEnterFakeMode);
  m.def("leave_fake_mode", pyLeaveFakeMode);

  m.def("is_fake", [](const at::Tensor& tensor) {
    return isFake(tensor);  // cast to `TensorBase`.
  });

  m.def("meta_like", [](const at::Tensor& fake) {
    return FakeTensor{fake}.toMeta();
  });
}

}  // namespace torchdistx::python
