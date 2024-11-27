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

#ifndef TENSORSTORE_UTIL_EXECUTOR_H_
#define TENSORSTORE_UTIL_EXECUTOR_H_

/// \file
/// Minimal task executor support.

#include <functional>
#include <type_traits>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/functional/any_invocable.h"
#include "absl/meta/type_traits.h"
#include "tensorstore/internal/poly/poly.h"
#include "tensorstore/internal/tracing/trace_context.h"

namespace tensorstore {

/// Executor concept:
///
/// An executor is a function object type that is callable with nullary
/// functions that do not throw exceptions.  Any function with which the
/// executor is called must either be invoked immediately (inline) or in another
/// thread.  The return value of the supplied function is ignored.

/// Type-erased nullary function type that supports move construction.
///
/// \relates Executor
using ExecutorTask = absl::AnyInvocable<void() &&>;

/// Function object that is callable with nullary noexcept functions.
///
/// Any `ExecutorTask` with which the executor is called must either be invoked
/// immediately (inline) or in another thread.
///
/// The return value of the `ExecutorTask` is ignored.
///
/// \ingroup async
using Executor = poly::Poly<0, /*Copyable=*/true, void(ExecutorTask) const>;

/// Executor that simply executes functions immediately in the current thread.
///
/// \relates Executor
class InlineExecutor {
 public:
  template <typename Func>
  void operator()(Func&& func) const {
    std::forward<Func>(func)();
  }
};

/// Function object that invokes a given function using a given executor.  Any
/// arguments are forwarded to the contained function.
///
/// \relates Executor
template <typename ExecutorType, typename FunctionType>
class ExecutorBoundFunction {
 public:
  /// Executor type.
  using Executor = ExecutorType;

  /// Function type.
  using Function = FunctionType;

  template <typename... T>
  std::enable_if_t<std::is_invocable_v<Function&, T...>>  //
  operator()(T&&... arg) {
    internal_tracing::SwapCurrentTraceContext(&tc_);
    executor(std::bind(std::move(function), std::forward<T>(arg)...));
    internal_tracing::SwapCurrentTraceContext(&tc_);
  }
  template <typename... T>
  std::enable_if_t<std::is_invocable_v<const Function&, T...>> operator()(
      T&&... arg) const {
    internal_tracing::SwapCurrentTraceContext(&tc_);
    executor(std::bind(function, std::forward<T>(arg)...));
    internal_tracing::SwapCurrentTraceContext(&tc_);
  }

  /// Executor object.
  ABSL_ATTRIBUTE_NO_UNIQUE_ADDRESS Executor executor;

  /// Function object.
  ABSL_ATTRIBUTE_NO_UNIQUE_ADDRESS Function function;

  /// Trace context.
  ABSL_ATTRIBUTE_NO_UNIQUE_ADDRESS mutable internal_tracing::TraceContext tc_;
};

/// Returns an instance of `ExecutorBoundFunction` that invokes the given
/// function `function` in the specified executor.
///
/// Any arguments are forwarded.
///
/// \relates Executor
template <typename Executor, typename Function>
std::enable_if_t<
    !std::is_same_v<absl::remove_cvref_t<Executor>, InlineExecutor>,
    ExecutorBoundFunction<absl::remove_cvref_t<Executor>,
                          absl::remove_cvref_t<Function>>>
WithExecutor(Executor&& executor, Function&& function) {
  return {
      std::forward<Executor>(executor), std::forward<Function>(function),
      internal_tracing::TraceContext(internal_tracing::TraceContext::kThread)};
}
template <typename Executor, typename Function>
std::enable_if_t<std::is_same_v<absl::remove_cvref_t<Executor>, InlineExecutor>,
                 Function&&>
WithExecutor(Executor&& executor, Function&& function) {
  return std::forward<Function>(function);
}

}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_EXECUTOR_H_
