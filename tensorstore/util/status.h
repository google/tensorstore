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

#ifndef TENSORSTORE_STATUS_H_
#define TENSORSTORE_STATUS_H_

#include <iosfwd>
#include <memory>
#include <string>
#include <system_error>  // NOLINT
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/optimization.h"
#include "absl/memory/memory.h"
#include "absl/meta/type_traits.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tensorstore/internal/preprocessor.h"
#include "tensorstore/internal/source_location.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/util/assert_macros.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {

using Status = absl::Status;

// If status is not `ok()`, then annotate the status message.
Status MaybeAnnotateStatus(const Status& status, absl::string_view message);

/// Returns the first error Status among all of the arguments.
template <typename... T>
Status GetFirstErrorStatus(Status&& a, T&&... other);

inline Status GetFirstErrorStatus() { return {}; }
inline Status GetFirstErrorStatus(const Status& status) { return status; }
inline Status GetFirstErrorStatus(Status&& status) { return std::move(status); }

template <typename... T>
inline Status GetFirstErrorStatus(const Status& a, T&&... other) {
  if (!a.ok()) return a;
  return GetFirstErrorStatus(std::forward<T>(other)...);
}

template <typename... T>
inline Status GetFirstErrorStatus(Status&& a, T&&... other) {
  if (!a.ok()) return std::move(a);
  return GetFirstErrorStatus(std::forward<T>(other)...);
}

/// Overload for the case of a bare Status argument.
/// \returns `status`
inline const Status& GetStatus(const Status& status) { return status; }
inline Status GetStatus(Status&& status) { return std::move(status); }


/// Returns `f(args...)`, converting a `void` return to `Status`.
template <typename F, typename... Args>
inline Status InvokeForStatus(F&& f, Args&&... args) {
  using R = std::invoke_result_t<F&&, Args&&...>;
  static_assert(std::is_void_v<R> ||
                std::is_same_v<internal::remove_cvref_t<R>, Status>);
  if constexpr (std::is_void_v<R>) {
    std::invoke(static_cast<F&&>(f), static_cast<Args&&>(args)...);
    return absl::OkStatus();
  } else {
    return std::invoke(static_cast<F&&>(f), static_cast<Args&&>(args)...);
  }
}

/// Converts `kInvalidArgument` and `kOutOfRange` errors to
/// `kFailedPrecondition` errors.
inline absl::Status ConvertInvalidArgumentToFailedPrecondition(
    absl::Status status) {
  if (status.code() == absl::StatusCode::kInvalidArgument ||
      status.code() == absl::StatusCode::kOutOfRange) {
    return absl::FailedPreconditionError(status.message());
  }
  return status;
}

namespace internal {
[[noreturn]] void FatalStatus(const char* message, const Status& status,
                              SourceLocation loc
                                  TENSORSTORE_LOC_CURRENT_DEFAULT_ARG);

}  // namespace internal
}  // namespace tensorstore

/// Returns the specified Status value if it is an error value.
///
/// Example:
///
///     Status GetSomeStatus();
///
///     Status Bar() {
///       TENSORSTORE_RETURN_IF_ERROR(GetSomeStatus());
///       // More code
///       return absl::OkStatus();
///     }
///
/// An optional second argument specifies the return expression in the case of
/// an error.  A variable `_` is bound to the value of the first expression is
/// in scope within this expression.  For example:
///
///     TENSORSTORE_RETURN_IF_ERROR(GetSomeStatus(),
///                                 MaybeAnnotateStatus(_, "In Bar"));
///
///     TENSORSTORE_RETURN_IF_ERROR(GetSomeStatus(),
///                                 MakeReadyFuture(_));
///
/// \remark You must ensure that the Status expression does not contain any
///     commas outside parentheses (such as in a template argument list), by
///     adding extra parentheses as needed.
#define TENSORSTORE_RETURN_IF_ERROR(...) \
  TENSORSTORE_PP_EXPAND(                 \
      TENSORSTORE_INTERNAL_RETURN_IF_ERROR_IMPL(__VA_ARGS__, _))
// Note: the use of `TENSORSTORE_PP_EXPAND` above is a workaround for MSVC 2019
// preprocessor limitations.

#define TENSORSTORE_INTERNAL_RETURN_IF_ERROR_IMPL(expr, error_expr, ...) \
  for (auto&& tensorstore_return_if_error_value = (expr);                \
       ABSL_PREDICT_FALSE(!tensorstore_return_if_error_value.ok());)     \
    for (auto _ = ::tensorstore::GetStatus(                              \
             static_cast<decltype(tensorstore_return_if_error_value)&&>( \
                 tensorstore_return_if_error_value));                    \
         ;)                                                              \
  return error_expr

/// This macro should be called with a single C++ expression.  We use a variadic
/// macro to allow calls like TENSORSTORE_CHECK_OK(foo<1,2>()).
///
// We use a lambda in the definition below to ensure that all uses of the
// condition argument occurs within a single top-level expression.  This ensures
// that temporary lifetime extension applies while we evaluate the condition.
#define TENSORSTORE_CHECK_OK(...)                                            \
  do {                                                                       \
    [](const ::tensorstore::Status& tensorstore_check_ok_condition) {        \
      if (ABSL_PREDICT_FALSE(!tensorstore_check_ok_condition.ok())) {        \
        ::tensorstore::internal::FatalStatus("Status not ok: " #__VA_ARGS__, \
                                             tensorstore_check_ok_condition, \
                                             TENSORSTORE_LOC);               \
      }                                                                      \
    }(::tensorstore::GetStatus((__VA_ARGS__)));                              \
  } while (false)

#endif  // TENSORSTORE_STATUS_H_
