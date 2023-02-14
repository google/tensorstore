// Copyright 2020 The TensorStore Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_PY_TENSORSTORE_WRITE_FUTURES_H_
#define THIRD_PARTY_PY_TENSORSTORE_WRITE_FUTURES_H_

/// \file
///
/// Defines `tensorstore.WriteFutures`.

#include <pybind11/pybind11.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include <memory>
#include <string>
#include <utility>

#include "python/tensorstore/future.h"
#include "tensorstore/progress.h"

namespace tensorstore {
namespace internal_python {

/// Python object type corresponding to `tensorstore::WriteFutures`.
struct PythonWriteFuturesObject {
  /// Python type object corresponding to this object type.
  ///
  /// This is initialized during the tensorstore module initialization by
  /// `RegisterWriteFuturesBindings`.
  static PyTypeObject* python_type;

  constexpr static const char python_type_name[] = "tensorstore.WriteFutures";

  // clang-format off
  PyObject_HEAD
  PyObject *weakrefs;
  PyObject *copy_future;
  PyObject *commit_future;
  // clang-format on

  PythonFutureObject& copy_future_obj() const {
    return *reinterpret_cast<PythonFutureObject*>(copy_future);
  }

  PythonFutureObject& commit_future_obj() const {
    return *reinterpret_cast<PythonFutureObject*>(commit_future);
  }
};

/// Interface for converting a `WriteFutures` object to a newly-allocated
/// `PythonWriteFuturesObject`.
///
/// This may be used as a return type in pybind11-bound functions.
struct PythonWriteFutures {
  pybind11::object value;

  /// Enables pybind11 type_caster support.
  constexpr static auto tensorstore_pybind11_type_name_override =
      pybind11::detail::_(PythonWriteFuturesObject::python_type_name);

  PythonWriteFutures() = default;
  explicit PythonWriteFutures(pybind11::object value)
      : value(std::move(value)) {}

  explicit PythonWriteFutures(WriteFutures write_futures,
                              const PythonObjectReferenceManager& manager);
};

}  // namespace internal_python
}  // namespace tensorstore

namespace pybind11 {
namespace detail {

template <>
struct type_caster<tensorstore::internal_python::PythonWriteFuturesObject>
    : public tensorstore::internal_python::StaticHeapTypeCaster<
          tensorstore::internal_python::PythonWriteFuturesObject> {};

}  // namespace detail
}  // namespace pybind11

#endif  // THIRD_PARTY_PY_TENSORSTORE_WRITE_FUTURES_H_
