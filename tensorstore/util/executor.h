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
#include <tuple>
#include <type_traits>
#include <utility>

#include "tensorstore/internal/attributes.h"
#include "tensorstore/internal/poly.h"
#include "tensorstore/internal/type_traits.h"

namespace tensorstore {

/// Executor concept:
///
/// An executor is a function object type that is callable with nullary
/// functions that do not throw exceptions.  Any function with which the
/// executor is called must either be invoked immediately (inline) or in another
/// thread.  The return value of the supplied function is ignored.

/// Type-erased executor type.
using ExecutorTask = internal::Poly<0, /*Copyable=*/false, void()>;
using Executor = internal::Poly<0, /*Copyable=*/true, void(ExecutorTask) const>;

/// Executor that simply executes functions immediately in the current thread.
class InlineExecutor {
 public:
  template <typename Func>
  void operator()(Func&& func) const {
    std::forward<Func>(func)();
  }
};

/// Function object that invokes a given function using a given executor.  Any
/// arguments are forwarded to the contained function.
template <typename ExecutorType, typename FunctionType>
class ExecutorBoundFunction {
 public:
  using Executor = ExecutorType;
  using Function = FunctionType;

  /// Non-const overload.
  template <typename... T>
  std::enable_if_t<std::is_invocable_v<Function&, T...>>  //
  operator()(T&&... arg) {
    executor(std::bind(std::move(function), std::forward<T>(arg)...));
  }

  /// Const overload.
  template <typename... T>
  std::enable_if_t<std::is_invocable_v<const Function&, T...>> operator()(
      T&&... arg) const {
    executor(std::bind(function, std::forward<T>(arg)...));
  }

  TENSORSTORE_ATTRIBUTE_NO_UNIQUE_ADDRESS Executor executor;
  TENSORSTORE_ATTRIBUTE_NO_UNIQUE_ADDRESS Function function;
};

/// Returns an instance of FunctionWithExecutor that invokes the given function
/// `func` in the specified executor.  Any arguments are forwarded.
template <typename Executor, typename Function>
ExecutorBoundFunction<internal::remove_cvref_t<Executor>,
                      internal::remove_cvref_t<Function>>
WithExecutor(Executor&& executor, Function&& function) {
  return {std::forward<Executor>(executor), std::forward<Function>(function)};
}

}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_EXECUTOR_H_
