// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "stack_utils.h"

#include <ATen/core/ivalue.h>
#include <c10/util/irange.h>

namespace torchdistx {

using at::irange;
using at::IValue;

using torch::jit::Stack;

}  // namespace torchdistx

namespace torchdistx::detail {

void processTensors(const Stack& s, std::size_t n, const TensorProcessor& processor) {
  for (auto i : irange(n)) {
    const IValue& value = torch::jit::peek(s, i, n);
    if (value.isTensor()) {
      if (processor(value.toTensor())) {
        return;
      }
    } else if (value.isList()) {
      for (const IValue& elem : value.toListRef()) {
        if (elem.isTensor()) {
          if (processor(elem.toTensor())) {
            return;
          }
        }
      }
    }
  }
}

void convertTensors(Stack& s, std::size_t n, const TensorConverter& converter) {
  for (auto i : irange(n)) {
    IValue& value = torch::jit::peek(s, i, n);
    if (value.isTensor()) {
      converter(value.toTensor());
    } else if (value.isList()) {
      for (const IValue& elem : value.toListRef()) {
        if (elem.isTensor()) {
          // Although technically not mandatory, `ArrayRef` only allows const
          // access to the underlying elements.
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          converter(const_cast<IValue&>(elem).toTensor());
        }
      }
    }
  }
}

}  // namespace torchdistx::detail
