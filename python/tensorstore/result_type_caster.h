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

#ifndef THIRD_PARTY_PY_TENSORSTORE_RESULT_TYPE_CASTER_H_
#define THIRD_PARTY_PY_TENSORSTORE_RESULT_TYPE_CASTER_H_

/// \files
///
/// Defines mapping of `tensorstore::Result` to Python values/exceptions.
///
/// Since Python uses exceptions for error handling, we don't define a Python
/// type corresponding to `tensorstore::Result`; instead, we just convert
/// `Result` objects to values or exceptions.

#include <pybind11/pybind11.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include <type_traits>

#include "python/tensorstore/status.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_python {

/// Returns the contained `T` or throws an exception that will map to the
/// corresponding Python exception.  This is used to unwrap `Result` values in
/// pybind11-exposed function implementations.
///
/// Requires GIL.
template <typename T>
std::add_lvalue_reference_t<const T> ValueOrThrow(
    const tensorstore::Result<T>& result,
    StatusExceptionPolicy policy = StatusExceptionPolicy::kDefault) {
  if (!result.ok()) {
    ThrowStatusException(result.status(), policy);
  }
  if constexpr (!std::is_void_v<T>) {
    return *result;
  }
}

/// Same as above, but for rvalue Result.
///
/// Requires GIL.
template <typename T>
T ValueOrThrow(tensorstore::Result<T>&& result,
               StatusExceptionPolicy policy = StatusExceptionPolicy::kDefault) {
  if (!result.ok()) {
    ThrowStatusException(result.status(), policy);
  }
  if constexpr (!std::is_void_v<T>) {
    return *std::move(result);
  }
}

}  // namespace internal_python
}  // namespace tensorstore

namespace pybind11 {
namespace detail {

/// Defines automatic conversion of `Result<T>` return values to Python
/// values/exceptions.
template <typename T>
struct type_caster<tensorstore::Result<T>> {
  using value_conv = make_caster<T>;
  // Use the type caster for `T` to handle conversion of the value.
  PYBIND11_TYPE_CASTER(tensorstore::Result<T>, value_conv::name);

  template <typename R>
  static handle cast(R&& result, return_value_policy policy, handle parent) {
    return value_conv::cast(
        tensorstore::internal_python::ValueOrThrow(static_cast<R&&>(result)),
        policy, parent);
  }
};

template <>
struct type_caster<tensorstore::Result<void>> {
  PYBIND11_TYPE_CASTER(tensorstore::Result<void>, _("None"));

  static handle cast(const tensorstore::Result<void>& result,
                     return_value_policy policy, handle parent) {
    if (!result.ok()) {
      tensorstore::internal_python::ThrowStatusException(result.status());
    }
    return none().release();
  }
};

}  // namespace detail
}  // namespace pybind11

#endif  // THIRD_PARTY_PY_TENSORSTORE_RESULT_TYPE_CASTER_H_
