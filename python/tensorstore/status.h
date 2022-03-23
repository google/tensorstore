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

#ifndef THIRD_PARTY_PY_TENSORSTORE_STATUS_H_
#define THIRD_PARTY_PY_TENSORSTORE_STATUS_H_

/// \file
///
/// Defines mapping from `absl::Status` to Python exception types.

#include <pybind11/pybind11.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include "absl/status/status.h"

namespace tensorstore {
namespace internal_python {

/// Specifies how to convert `absl::StatusCode::kInvalidArgument` to a Python
/// exception.  By default it converts to `ValueError` but for subscript
/// operators it must convert to `IndexError` for consistency with builtin
/// Python subscript operators.
enum class StatusExceptionPolicy {
  /// kInvalidArgument -> ValueError
  kDefault,
  /// kInvalidArgument -> IndexError
  kIndexError,
};

/// Throws an exception that will map to the corresponding Python exception type
/// if `!status.ok()`.
///
/// Requires GIL.  Furthermore, the GIL must be held until any exception that is
/// thrown is handled.
void ThrowStatusException(
    const absl::Status& status,
    StatusExceptionPolicy policy = StatusExceptionPolicy::kDefault);

/// Sets the Python error indicator (i.e. `PyErr_Occurred()`) from the specified
/// error status.
///
/// Requires GIL.
///
/// \dchecks `!status.ok()`
void SetErrorIndicatorFromStatus(
    const absl::Status& status,
    StatusExceptionPolicy policy = StatusExceptionPolicy::kDefault);

/// Returns the Python class type corresponding to the specified status code.
pybind11::handle GetExceptionType(
    absl::StatusCode error_code,
    StatusExceptionPolicy policy = StatusExceptionPolicy::kDefault);

/// Returns the Python exception object corresponding to the specified
/// `absl::Status` object.
pybind11::object GetStatusPythonException(
    const absl::Status& status,
    StatusExceptionPolicy policy = StatusExceptionPolicy::kDefault);

/// Returns a status that encodes a Python exception.
///
/// `ThrowStatusException` may be used to re-throw it.
///
/// The pickled representation of the exception is stored as an additional
/// payload in the returned `absl::Status` object.
///
/// \param exc Python exception, or `nullptr` to indicate that the currently set
///     Python exception should be converted.
absl::Status GetStatusFromPythonException(pybind11::handle exc = {}) noexcept;

/// Calls `func` and converts any exception thrown into the Python error
/// indicator.
template <typename Func>
inline bool CallAndSetErrorIndicator(Func&& func) {
  try {
    std::forward<Func>(func)();
    return false;
  } catch (pybind11::error_already_set& e) {
    e.restore();
    return true;
  } catch (pybind11::builtin_exception& e) {
    e.set_error();
    return true;
  } catch (...) {
    PyErr_SetString(PyExc_RuntimeError, "Unknown Python exception");
    return true;
  }
}

}  // namespace internal_python
}  // namespace tensorstore

#endif  // THIRD_PARTY_PY_TENSORSTORE_STATUS_H_
