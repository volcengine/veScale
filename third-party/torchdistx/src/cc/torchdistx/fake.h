// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <memory>

#include <c10/core/DispatchKey.h>
#include <c10/core/Storage.h>

#include "macros.h"

namespace at {

class Tensor;
class TensorBase;

}  // namespace at

namespace torchdistx {
namespace detail {

class FakeTensorImpl;

}  // namespace detail

// Forces all newly-constructed tensors on the calling thread to be fake.
//
// When `fake_cuda` is set to true, allows constructing fake CUDA tensors even
// if CUDA is not available.
TDX_API void enterFakeMode(bool fake_cuda = false);

// Leaves the fake mode in the calling thread.
TDX_API void leaveFakeMode() noexcept;

// Indicates whether the calling thread is in fake mode.
TDX_API bool isFakeModeActive() noexcept;

// Indicates whether `tensor` is fake.
TDX_API bool isFake(const at::TensorBase& tensor) noexcept;

// Provides access to the properties of a fake tensor.
class TDX_API FakeTensor {
 public:
  explicit FakeTensor(const at::TensorBase& tensor, bool unsafe = false);

 public:
  // Returns a meta tensor with the same properties.
  at::Tensor toMeta() const;

  void setData(at::DispatchKey key, std::shared_ptr<void> data);

  bool hasData(at::DispatchKey key) const noexcept;

  std::shared_ptr<void> getData(at::DispatchKey key) const;

  template <typename T>
  inline auto getData(at::DispatchKey key) const {
    return std::static_pointer_cast<T>(getData(key));
  }

  void* unsafeGetData(at::DispatchKey key) const;

  template <typename T>
  inline auto unsafeGetData(at::DispatchKey key) const {
    return static_cast<T*>(unsafeGetData(key));
  }

 public:
  const at::Storage& meta_storage() const noexcept;

 private:
  detail::FakeTensorImpl* impl_;
};

// Treats `tensor` as fake.
TDX_API FakeTensor asFake(const at::TensorBase& tensor);

// Treats `tensor` as fake without performing any type checks.
TDX_API FakeTensor unsafeAsFake(const at::TensorBase& tensor) noexcept;

}  // namespace torchdistx
