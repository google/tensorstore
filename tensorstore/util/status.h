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

#include <stddef.h>

#include <cassert>
#include <type_traits>
#include <utility>

#include "absl/base/optimization.h"
#include "absl/meta/type_traits.h"
#include "absl/status/status.h"
#include "tensorstore/internal/source_location.h"
#include "tensorstore/util/status_builder.h"

namespace tensorstore {
namespace internal {

/// Logs a fatal error with the `status` and `message`, then terminates.
[[noreturn]] void FatalStatus(const char* message, const absl::Status& status,
                              SourceLocation loc);

/// Returns `f(args...)`, converting a `void` return to `absl::Status`.
template <typename F, typename... Args>
inline absl::Status InvokeForStatus(F&& f, Args&&... args) {
  using R = std::invoke_result_t<F&&, Args&&...>;
  static_assert(std::is_void_v<R> ||
                std::is_same_v<absl::remove_cvref_t<R>, absl::Status>);
  if constexpr (std::is_void_v<R>) {
    std::invoke(static_cast<F&&>(f), static_cast<Args&&>(args)...);
    return absl::OkStatus();
  } else {
    return std::invoke(static_cast<F&&>(f), static_cast<Args&&>(args)...);
  }
}

/// Converts `absl::StatusCode::kInvalidArgument` and
/// `absl::StatusCode::kOutOfRange` errors to
/// `absl::StatusCode::kFailedPrecondition` errors.
inline absl::Status ConvertInvalidArgumentToFailedPrecondition(
    StatusBuilder s) {
  if (s.code() == absl::StatusCode::kInvalidArgument ||
      s.code() == absl::StatusCode::kOutOfRange) {
    s.SetCode(absl::StatusCode::kFailedPrecondition);
  }
  return std::move(s).BuildStatus();
}

}  // namespace internal

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

namespace internal_status {

// A helper class for implementing `TENSORSTORE_RETURN_IF_ERROR`.
//
// This class holds a status value to avoid creating a
// `tensorstore::StatusBuilder` when the result is ``OK``.
class ReturnIfErrorAdaptor {
 public:
  explicit ReturnIfErrorAdaptor(const absl::Status& status) : status_(status) {}
  explicit ReturnIfErrorAdaptor(absl::Status&& status)
      : status_(std::move(status)) {}

  ReturnIfErrorAdaptor() = delete;
  ReturnIfErrorAdaptor(const ReturnIfErrorAdaptor&) = delete;
  ReturnIfErrorAdaptor& operator=(const ReturnIfErrorAdaptor&) = delete;

  ~ReturnIfErrorAdaptor() { status_.~Status(); }

  explicit operator bool() const { return ABSL_PREDICT_TRUE(status_.ok()); }

  StatusBuilder Consume(
      SourceLocation loc = ::tensorstore::SourceLocation::current()) {
    return StatusBuilder(std::move(status_), loc);
  }

 private:
  union {  // Wrap in union to prevent implicit destruction
    absl::Status status_;
    char nothing_[1];
  };
};

// A helper class for implementing `TENSORSTORE_ASSIGN_OR_RETURN`.
//
// This class holds a pre-existing `tensorstore::StatusBuilder`.
class StatusBuilderAdaptor {
 public:
  explicit StatusBuilderAdaptor(const StatusBuilder& status_builder)
      : status_builder_(status_builder) {}
  explicit StatusBuilderAdaptor(StatusBuilder&& status_builder)
      : status_builder_(std::move(status_builder)) {}

  StatusBuilderAdaptor() = delete;
  StatusBuilderAdaptor(const StatusBuilderAdaptor&) = delete;
  StatusBuilderAdaptor& operator=(const StatusBuilderAdaptor&) = delete;

  explicit operator bool() const {
    return ABSL_PREDICT_FALSE(status_builder_.ok());
  }

  StatusBuilder&& Consume() { return std::move(status_builder_); }

 private:
  StatusBuilder status_builder_;
};

// MacroBuilderAdaptor overloads select the correct adaptor class for the
// argument type.
inline StatusBuilderAdaptor MacroBuilderAdaptor(const StatusBuilder& s) {
  return StatusBuilderAdaptor(s);
}
inline StatusBuilderAdaptor MacroBuilderAdaptor(StatusBuilder&& s) {
  return StatusBuilderAdaptor(std::move(s));
}

template <typename T>
inline std::enable_if_t<!std::is_same_v<absl::remove_cvref_t<T>, StatusBuilder>,
                        ReturnIfErrorAdaptor>
MacroBuilderAdaptor(const T& s) {
  return ReturnIfErrorAdaptor(GetStatus(s));
}
template <typename T>
inline std::enable_if_t<!std::is_same_v<absl::remove_cvref_t<T>, StatusBuilder>,
                        ReturnIfErrorAdaptor>
MacroBuilderAdaptor(T&& s) {
  return ReturnIfErrorAdaptor(GetStatus(std::forward<T>(s)));
}

}  // namespace internal_status
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
/// The `TENSORSTORE_RETURN_IF_ERROR` macro implicitly returns a
/// `tensorstore::StatusBuilder` object which may be used to further modify the
/// status. A `tensorstore::StatusBuilder` is implicitly convertible to an
/// `absl::Status`, however when used in lambdas the return type may need to be
/// explicitly specified as `absl::Status`.
///
/// To explicitly convert to `absl::Status` use ``.BuildStatus()``, or use
/// the ``.With()`` method return a different type (including void).
///
/// Example::
///
///     TENSORSTORE_RETURN_IF_ERROR(GetSomeStatus())
///         .Format("In Bar");
///
///     TENSORSTORE_RETURN_IF_ERROR(GetSomeStatus())
///         .Format("In Bar")
///         .With([&](absl::Status s) { future.SetResult(std::move(s)); });
///
///     auto my_lambda = []() -> absl::Status {
///       TENSORSTORE_RETURN_IF_ERROR(GetSomeStatus());
///       return absl::OkStatus();
///     };
///
/// There is also a 2-argument form of `TENSORSTORE_RETURN_IF_ERROR` where
/// the second argument is an error expression which is evaluated only if
/// the first argument is an error.  When invoked with two arguments, ``_`` is
/// bound to the `tensorstore::StatusBuilder`. The second argument must be a
/// valid expression for the right-hand side of a ``return`` statement.
///
/// Example::
///
///     TENSORSTORE_RETURN_IF_ERROR(GetSomeStatus(),
///                                 _.Format("In Bar"));
///
/// .. warning::
///
///    The first argument must not contain any commas outside
///    parentheses (such as in a template argument list); if necessary, to
///    ensure this, it may be wrapped in additional parentheses as needed.
///
/// \ingroup error handling
#define TENSORSTORE_RETURN_IF_ERROR(...)                       \
  TENSORSTORE_INTERNAL_RIE_SELECT_OVERLOAD(                    \
      (__VA_ARGS__, TENSORSTORE_INTERNAL_RETURN_IF_ERROR_2ARG, \
       TENSORSTORE_INTERNAL_RETURN_IF_ERROR_1ARG))(__VA_ARGS__)

// Implementation details of TENSORSTORE_RETURN_IF_ERROR
#define TENSORSTORE_INTERNAL_RIE_SELECT_OVERLOAD_HELPER(_1, _2, OVERLOAD, ...) \
  OVERLOAD

#define TENSORSTORE_INTERNAL_RIE_SELECT_OVERLOAD(args) \
  TENSORSTORE_INTERNAL_RIE_SELECT_OVERLOAD_HELPER args

#define TENSORSTORE_INTERNAL_RIE_ELSE_BLOCKER_ \
  switch (0)                                   \
  case 0:                                      \
  default:  // NOLINT

#define TENSORSTORE_INTERNAL_RETURN_IF_ERROR_1ARG(expr)                  \
  TENSORSTORE_INTERNAL_RIE_ELSE_BLOCKER_                                 \
  if (auto return_if_error_adaptor =                                     \
          ::tensorstore::internal_status::MacroBuilderAdaptor((expr))) { \
  } else                                                                 \
    return return_if_error_adaptor.Consume()

#define TENSORSTORE_INTERNAL_RETURN_IF_ERROR_2ARG(expr, error_expr)      \
  TENSORSTORE_INTERNAL_RIE_ELSE_BLOCKER_                                 \
  if (auto return_if_error_adaptor =                                     \
          ::tensorstore::internal_status::MacroBuilderAdaptor((expr))) { \
  } else if (auto&& _ = return_if_error_adaptor.Consume(); true)         \
  return error_expr

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
#define TENSORSTORE_CHECK_OK(...)                                           \
  do {                                                                      \
    [](const ::absl::Status& tensorstore_check_ok_condition) {              \
      if (ABSL_PREDICT_FALSE(!tensorstore_check_ok_condition.ok())) {       \
        ::tensorstore::internal::FatalStatus(                               \
            "Status not ok: " #__VA_ARGS__, tensorstore_check_ok_condition, \
            ::tensorstore::SourceLocation::current());                      \
      }                                                                     \
    }(::tensorstore::GetStatus((__VA_ARGS__)));                             \
  } while (false)

// The lambda in the definition above ensures that all uses of the condition
// argument occurs within a single top-level expression.  This ensures that
// temporary lifetime extension applies while we evaluate the condition.

#endif  // TENSORSTORE_STATUS_H_
