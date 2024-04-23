// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "module.h"

#include <ATen/Tensor.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/SymIntArrayRef.h>
#include <torch/csrc/PyInterpreter.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/utils/pybind.h>
#include <torchdistx/deferred_init.h>

namespace py = pybind11;

namespace torchdistx {

using at::MaybeOwned;
using at::Tensor;

using c10::impl::PyInterpreterStatus;

using torch::TypeError;

}  // namespace torchdistx

namespace torchdistx::python {
namespace {

// Creates a new Python variable (i.e. tensor) that holds `data`.
py::object makeVariable(PyTypeObject* type, Tensor data) {
  PyObject* naked_obj = type->tp_alloc(type, 0);

  TORCH_CHECK(naked_obj != nullptr,
      "Failed to construct the `Variable` object.");

  auto obj = py::reinterpret_steal<py::object>(naked_obj);

  constexpr auto s = PyInterpreterStatus::DEFINITELY_UNINITIALIZED;

  // Associate ATen and Python tensor instances.
  data.unsafeGetTensorImpl()->pyobj_slot()->init_pyobj(getPyInterpreter(), naked_obj, s);

  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  auto* var = reinterpret_cast<THPVariable*>(naked_obj);

  // `THPVariable` is a plain C struct, so we need to use placement new to
  // construct `cdata`.
  new (&var->cdata) MaybeOwned<Tensor>{};

  var->cdata = MaybeOwned<Tensor>::owned(std::move(data));

  return obj;
}

// Materializing a tensor in Python requires an extra step. We need to ensure
// that the materialized tensor has the same Python class (e.g. `Variable` or
// `Parameter`) as the original tensor.
py::object materializeVariable(const py::object& var) {
  PyObject* naked_var = var.ptr();

  if (!THPVariable_Check(naked_var)) {
    throw TypeError{"`var` has to be a `Variable`, but got `%s`.", Py_TYPE(naked_var)->tp_name};
  }

  const Tensor& data = THPVariable_Unpack(naked_var);

  auto materialize = [](const Tensor& tensor) {
    py::gil_scoped_release guard{};

    return materializeTensor(tensor);
  };

  Tensor materialized_data = materialize(data);

  // Check if we have really materialized `data`. Materializing a regular tensor
  // is a no-op, so we can simply return.
  if (materialized_data.is_same(data)) {
    return var;
  }

  // We might have already materialized `data`. Make sure that we preserve its
  // identity on the Python side and avoid creating a new Python tensor.
  c10::optional<PyObject*> opt_materialized_var =
      materialized_data.unsafeGetTensorImpl()->pyobj_slot()->check_pyobj(getPyInterpreter());
  if (opt_materialized_var.has_value()) {
    return py::reinterpret_borrow<py::object>(*opt_materialized_var);
  }

  // Otherwise ensure that our materialized tensor has the same Python class as
  // the original tensor.
  return makeVariable(Py_TYPE(naked_var), std::move(materialized_data));
}


// Materializing a tensor in Python requires an extra step. We need to ensure
// that the materialized tensor has the same Python class (e.g. `Variable` or
// `Parameter`) as the original tensor.
// and with dtensor case we need to change the parallized tensor shape
py::object materializeVariableWithLocalShape(const py::object& var, const py::object &shape, const c10::optional<c10::Device> device) {
  PyObject* naked_var = var.ptr();
  auto c_shape = shape.cast<std::vector<int64_t>>();

  if (!THPVariable_Check(naked_var)) {
    throw TypeError{"`var` has to be a `Variable`, but got `%s`.", Py_TYPE(naked_var)->tp_name};
  }

  const Tensor& data = THPVariable_Unpack(naked_var);

  auto materialize = [=](const Tensor& tensor, c10::IntArrayRef sp) {
    py::gil_scoped_release guard{};

    return materializeTensorWithLocalShape(tensor, sp, device);
  };

  Tensor materialized_data = materialize(data, at::IntArrayRef(c_shape));

  // Check if we have really materialized `data`. Materializing a regular tensor
  // is a no-op, so we can simply return.
  if (materialized_data.is_same(data)) {
    return var;
  }

  // We might have already materialized `data`. Make sure that we preserve its
  // identity on the Python side and avoid creating a new Python tensor.
  c10::optional<PyObject*> opt_materialized_var =
      materialized_data.unsafeGetTensorImpl()->pyobj_slot()->check_pyobj(getPyInterpreter());
  if (opt_materialized_var.has_value()) {
    return py::reinterpret_borrow<py::object>(*opt_materialized_var);
  }

  // Otherwise ensure that our materialized tensor has the same Python class as
  // the original tensor.
  return makeVariable(Py_TYPE(naked_var), std::move(materialized_data));
}


}  // namespace

void initDeferredInitFunctions(py::module& m) {
  m.def("enter_deferred_init", enterDeferredInit);
  m.def("leave_deferred_init", leaveDeferredInit);
  m.def("can_materialize", canMaterialize);
  m.def("is_gen_by_random_op", isGenByRandomOp);
  m.def("materialize_tensor", materializeVariable);
  m.def("materialize_tensor_with_local_shape", materializeVariableWithLocalShape);
}

}  // namespace torchdistx::python
