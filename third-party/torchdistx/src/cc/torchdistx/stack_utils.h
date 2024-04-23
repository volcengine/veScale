// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>
#include <functional>

#include <ATen/core/stack.h>

namespace at {

class Tensor;

}  // namespace at

namespace torchdistx::detail {

using TensorProcessor = std::function<bool(const at::Tensor&)>;

// Calls `processor` for all tensors in the last `n` entries of `s`.
void processTensors(const torch::jit::Stack& s, std::size_t n, const TensorProcessor& processor);

using TensorConverter = std::function<void(at::Tensor&)>;

// Calls `converter` for all tensors in the last `n` entries of `s`.
void convertTensors(torch::jit::Stack& s, std::size_t n, const TensorConverter& converter);

}  // namespace torchdistx::detail
