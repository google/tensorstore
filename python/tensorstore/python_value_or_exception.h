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

#include <pybind11/pybind11.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include "python/tensorstore/garbage_collection.h"
#include "python/tensorstore/gil_safe.h"
#include "python/tensorstore/status.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_python {

template <typename Object>
struct PythonValueOrExceptionBase {
  /// If this holds a value, `value` is non-null and `error_type`,
  /// `error_value`, and `error_traceback` are null.
  Object value;

  /// If this holds an exception, `error_type`, `error_value`, and
  /// `error_traceback` are set, an `value` is null.
  Object error_type;
  Object error_value;
  Object error_traceback;

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.value, x.error_type, x.error_value, x.error_traceback);
  };
};

/// Special type capable of holding any Python value or exception.  This
/// (wrapped by `GilSafeHolder`) is used as the result type for
/// `Promise`/`Future` pairs created by Python.
///
/// Note: This type cannot safely be copied or destroyed unless the GIL is held.
/// To avoid that restriction, use `GilSafePythonValueOrException`.
struct PythonValueOrException
    : public PythonValueOrExceptionBase<pybind11::object> {
  PythonValueOrException() = default;

  /// Constructs from a Python value.
  explicit PythonValueOrException(pybind11::object obj) {
    this->value = std::move(obj);
  }

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
};

/// Same as `PythonValueOrException`, but holds weak references rather than
/// strong references.
///
/// When wrapped by `GilSafeHolder`, this is suitable for use with
/// `PythonFutureObject`, whereas `PythonValueOrException` is not.
struct PythonValueOrExceptionWeakRef
    : public PythonValueOrExceptionBase<PythonWeakRef> {
  PythonValueOrExceptionWeakRef() = default;
  explicit PythonValueOrExceptionWeakRef(PythonObjectReferenceManager& manager,
                                         const PythonValueOrException& obj) {
    if (obj.value) {
      value = PythonWeakRef(manager, obj.value);
      return;
    }
    error_type = PythonWeakRef(manager, obj.error_type);
    error_value = PythonWeakRef(manager, obj.error_value);
    if (obj.error_traceback) {
      error_traceback = PythonWeakRef(manager, obj.error_traceback);
    }
  }
};

/// Used to represent an arbitrary Python value or exception as the value type
/// of a `PythonFutureObject`.
using GilSafePythonValueOrExceptionWeakRef =
    GilSafeHolder<PythonValueOrExceptionWeakRef>;

}  // namespace internal_python
}  // namespace tensorstore

namespace pybind11 {
namespace detail {

/// Defines automatic mapping of
/// `tensorstore::internal_python::PythonValueOrExceptionWeakRef` to the
/// contained Python value or exception.
template <>
struct type_caster<
    tensorstore::internal_python::PythonValueOrExceptionWeakRef> {
  PYBIND11_TYPE_CASTER(
      tensorstore::internal_python::PythonValueOrExceptionWeakRef, _("Any"));
  static handle cast(
      const tensorstore::internal_python::PythonValueOrExceptionWeakRef& obj,
      return_value_policy policy, handle parent);
};

}  // namespace detail
}  // namespace pybind11

#endif  // THIRD_PARTY_PY_TENSORSTORE_PYTHON_VALUE_OR_EXCEPTION_H_
