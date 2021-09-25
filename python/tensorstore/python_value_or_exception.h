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

#ifndef THIRD_PARTY_PY_TENSORSTORE_PYTHON_VALUE_OR_EXCEPTION_H_
#define THIRD_PARTY_PY_TENSORSTORE_PYTHON_VALUE_OR_EXCEPTION_H_

#include "python/tensorstore/gil_safe.h"
#include "python/tensorstore/status.h"
#include "pybind11/pybind11.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_python {

/// Special type capable of holding any Python value or exception.  This
/// (wrapped by `GilSafeHolder`) is used as the result type for
/// `Promise`/`Future` pairs created by Python.
///
/// Note: This type cannot safely be copied or destroyed unless the GIL is held.
/// To avoid that restriction, use `GilSafePythonValueOrException`.
struct PythonValueOrException {
  /// Constructs from an arbitrary C++ value.
  ///
  /// If conversion to a Python object fails, the returned
  /// `PythonValueOrException` holds the resultant error.
  template <typename T>
  static PythonValueOrException FromValue(const T& value) {
    PythonValueOrException v;
    if (internal_python::CallAndSetErrorIndicator(
            [&] { v.value = pybind11::cast(value); })) {
      v = FromErrorIndicator();
    }
    return v;
  }

  /// Constructs from the current error indicator, which must be set.
  static PythonValueOrException FromErrorIndicator();

  /// Attempts to convert to an arbitrary C++ type.
  template <typename T>
  explicit operator Result<T>() const {
    if (value.ptr()) {
      Result<T> obj;
      if (internal_python::CallAndSetErrorIndicator(
              [&] { obj = pybind11::cast<T>(value); })) {
        obj = GetStatusFromPythonException();
      }
      return obj;
    }
    return GetStatusFromPythonException(error_value);
  }

  /// If this holds a value, `value` is non-null and `error_type`,
  /// `error_value`, and `error_traceback` are null.
  pybind11::object value;

  /// If this holds an exception, `error_type`, `error_value`, and
  /// `error_traceback` are set, an `value` is null.
  pybind11::object error_type;
  pybind11::object error_value;
  pybind11::object error_traceback;
};

using GilSafePythonValueOrException = GilSafeHolder<PythonValueOrException>;

}  // namespace internal_python
}  // namespace tensorstore

namespace pybind11 {
namespace detail {

/// Defines automatic mapping of
/// `tensorstore::internal_python::PythonValueOrException` to the contained
/// Python value or exception.
template <>
struct type_caster<
    tensorstore::internal_python::GilSafePythonValueOrException> {
  PYBIND11_TYPE_CASTER(
      tensorstore::internal_python::GilSafePythonValueOrException, _("Any"));
  static handle cast(
      const tensorstore::internal_python::GilSafePythonValueOrException& result,
      return_value_policy policy, handle parent);
};

}  // namespace detail
}  // namespace pybind11

#endif  // THIRD_PARTY_PY_TENSORSTORE_PYTHON_VALUE_OR_EXCEPTION_H_
