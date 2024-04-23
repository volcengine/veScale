// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <pybind11/pybind11.h>

namespace torchdistx::python {

void initDeferredInitFunctions(pybind11::module& m);

void initFakeFunctions(pybind11::module& m);

}  // namespace torchdistx::python
