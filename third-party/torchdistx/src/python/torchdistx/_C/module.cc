// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "module.h"

#include <exception>

#include <torch/csrc/Exceptions.h>

namespace py = pybind11;

namespace torchdistx::python {
namespace {

void registerExceptionTranslator() {
  // NOLINTNEXTLINE(performance-unnecessary-value-param)
  py::register_exception_translator([](std::exception_ptr ex) {
    try {
      if (ex) {
        std::rethrow_exception(ex);
      }
    }
    CATCH_TH_ERRORS()  // NOLINT
  });
}

}  // namespace

// NOLINTNEXTLINE(clang-diagnostic-reserved-identifier)
PYBIND11_MODULE(_C, m) {
  registerExceptionTranslator();

  initDeferredInitFunctions(m);

  initFakeFunctions(m);
}

}  // namespace torchdistx::python
