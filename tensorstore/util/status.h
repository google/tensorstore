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

#include <optional>
#include <string_view>
#include <type_traits>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/optimization.h"
#include "absl/status/status.h"
#include "tensorstore/internal/preprocessor.h"
#include "tensorstore/internal/source_location.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/util/assert_macros.h"

namespace tensorstore {
namespace internal {

/// Returns `f(args...)`, converting a `void` return to `absl::Status`.
template <typename F, typename... Args>
inline absl::Status InvokeForStatus(F&& f, Args&&... args) {
  using R = std::invoke_result_t<F&&, Args&&...>;
  static_assert(std::is_void_v<R> ||
                std::is_same_v<internal::remove_cvref_t<R>, absl::Status>);
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

[[noreturn]] void FatalStatus(const char* message, const absl::Status& status,
                              SourceLocation loc);

absl::Status MaybeAnnotateStatusImpl(absl::Status source,
                                     std::string_view message,
                                     std::optional<SourceLocation> loc);

}  // namespace internal

/// If status is not `absl::StatusCode::kOk`, then annotate the status message.
///
/// \ingroup error handling
inline absl::Status MaybeAnnotateStatus(
    absl::Status source, std::string_view message,
    SourceLocation loc TENSORSTORE_LOC_CURRENT_DEFAULT_ARG) {
  return internal::MaybeAnnotateStatusImpl(std::move(source), message, loc);
}
#if !TENSORSTORE_HAVE_SOURCE_LOCATION_CURRENT
inline absl::Status MaybeAnnotateStatus(absl::Status source,
                                        std::string_view message) {
  return internal::MaybeAnnotateStatusImpl(std::move(source), message,
                                           std::nullopt);
}
#endif

/// Overload for the case of a bare absl::Status argument.
///
/// \returns `status`
/// \relates Result
/// \id status
inline const absl::Status& GetStatus(const absl::Status& status) {
  return status;
}
inline absl::Status GetStatus(absl::Status&& status) {
  return std::move(status);
}

}  // namespace tensorstore

/// Causes the containing function to return the specified `absl::Status` value
/// if it is an error status.
///
/// Example::
///
///     absl::Status GetSomeStatus();
///
///     absl::Status Bar() {
///       TENSORSTORE_RETURN_IF_ERROR(GetSomeStatus());
///       // More code
///       return absl::OkStatus();
///     }
///
/// An optional second argument specifies the return expression in the case of
/// an error.  A variable ``_`` is bound to the value of the first expression
/// is in scope within this expression.  For example::
///
///     TENSORSTORE_RETURN_IF_ERROR(GetSomeStatus(),
///                                 MaybeAnnotateStatus(_, "In Bar"));
///
///     TENSORSTORE_RETURN_IF_ERROR(GetSomeStatus(),
///                                 MakeReadyFuture(_));
///
/// .. warning::
///
///    The `absl::Status` expression must not contain any commas outside
///    parentheses (such as in a template argument list); if necessary, to
///    ensure this, it may be wrapped in additional parentheses as needed.
///
/// \ingroup error handling
#define TENSORSTORE_RETURN_IF_ERROR(...) \
  TENSORSTORE_PP_EXPAND(                 \
      TENSORSTORE_INTERNAL_RETURN_IF_ERROR_IMPL(__VA_ARGS__, _))
// Note: the use of `TENSORSTORE_PP_EXPAND` above is a workaround for MSVC 2019
// preprocessor limitations.

#define TENSORSTORE_INTERNAL_RETURN_IF_ERROR_IMPL(expr, error_expr, ...) \
  for (absl::Status _ = ::tensorstore::GetStatus(expr);                  \
       ABSL_PREDICT_FALSE(!_.ok());)                                     \
  return error_expr /**/

/// Logs an error and terminates the program if the specified `absl::Status` is
/// an error status.
///
/// .. note::
///
///    This macro should be called with a single C++ expression.  We use a
///    variadic macro to allow calls like
///    ``TENSORSTORE_CHECK_OK(foo<1,2>())``.
///
/// \ingroup error handling
#define TENSORSTORE_CHECK_OK(...)                                            \
  do {                                                                       \
    [](const ::absl::Status& tensorstore_check_ok_condition) {               \
      if (ABSL_PREDICT_FALSE(!tensorstore_check_ok_condition.ok())) {        \
        ::tensorstore::internal::FatalStatus("Status not ok: " #__VA_ARGS__, \
                                             tensorstore_check_ok_condition, \
                                             TENSORSTORE_LOC);               \
      }                                                                      \
    }(::tensorstore::GetStatus((__VA_ARGS__)));                              \
  } while (false)
// We use a lambda in the definition above to ensure that all uses of the
// condition argument occurs within a single top-level expression.  This ensures
// that temporary lifetime extension applies while we evaluate the condition.

#endif  // TENSORSTORE_STATUS_H_
