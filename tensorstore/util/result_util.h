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

#ifndef TENSORSTORE_RESULT_UTIL_H_
#define TENSORSTORE_RESULT_UTIL_H_

/// \file
/// Utility functions for working with Result<T>.

#include <type_traits>
#include <utility>

#include "tensorstore/internal/type_traits.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal_result_util {

/// Result type traits helper structs for UnwrapResultType / FlatResultType.
template <typename T>
struct UnwrapResultHelper {
  static_assert(std::is_same<T, internal::remove_cvref_t<T>>::value,
                "Type argument to UnwrapResultType must be unqualified.");
  using type = T;
  using result_type = Result<T>;
};

template <typename T>
struct UnwrapResultHelper<Result<T>> {
  using type = T;
  using result_type = Result<T>;
};

template <>
struct UnwrapResultHelper<Status> {
  using type = void;
  using result_type = Result<void>;
};

}  // namespace internal_result_util

/// UnwrapResultType<T> maps
///
///   Result<T> -> T
///   Status -> void
///   T -> T
template <typename T>
using UnwrapResultType =
    typename internal_result_util::UnwrapResultHelper<T>::type;

/// As above, preserving const / volatile / reference qualifiers.
template <typename T>
using UnwrapQualifiedResultType =
    internal::CopyQualifiers<T, UnwrapResultType<internal::remove_cvref_t<T>>>;

/// FlatResult<T> maps
///
///     T -> Result<T>
///     Result<T> -> Result<T>
///     Status -> Result<void>
///
template <typename T>
using FlatResult =
    typename internal_result_util::UnwrapResultHelper<T>::result_type;

/// Type alias that maps `Result<T>` to `Result<U>`, where `U = MapType<U>`.
template <template <typename...> class MapType, typename... T>
using FlatMapResultType = Result<MapType<UnwrapResultType<T>...>>;

/// Tries to call `func` with `Result`-wrapped arguments.
///
/// The return value of `func` is wrapped in a `Result` if it not already a
/// `Result` instance.
///
/// \returns `std::forward<Func>(func)(UnwrapResult(std::forward<T>(arg))...)`
///     if no `Result`-wrapped `arg` is an in error state.  Otherwise, returns
///     the error Status of the first `Result`-wrapped `arg` in an error state.
template <typename Func, typename... T>
FlatResult<std::invoke_result_t<Func&&, UnwrapQualifiedResultType<T>...>>
MapResult(Func&& func, T&&... arg) {
  TENSORSTORE_RETURN_IF_ERROR(
      GetFirstErrorStatus(GetStatus(std::forward<T>(arg))...));
  return std::forward<Func>(func)(UnwrapResult(std::forward<T>(arg))...);
}

namespace internal_result_util {

// Helper struct for ChainResultType.
template <typename T, typename... Func>
struct ChainResultTypeHelper;

template <typename T>
struct ChainResultTypeHelper<T> {
  using type =
      typename UnwrapResultHelper<internal::remove_cvref_t<T>>::result_type;
};

template <typename T, typename Func0, typename... Func>
struct ChainResultTypeHelper<T, Func0, Func...>
    : ChainResultTypeHelper<
          std::invoke_result_t<Func0&&, UnwrapQualifiedResultType<T>>,
          Func...> {};

/// Template alias that evaluates to the result of calling `ChainResult`.
///
/// For example:
///
///     ChainResultType<int>
///         -> Result<int>
///     ChainResultType<Result<int>>
///         -> Result<int>
///     ChainResultType<int, float(*)(int)>
///         -> Result<float>
///     ChainResultType<Result<int>, float(*)(int)>
///         -> Result<float>
///     ChainResultType<int, Result<float>(*)(int)>
///         -> Result<float>
///     ChainResultType<int, float(*)(int), std::string(*)(float)>
///         -> Result<std::string>
template <typename T, typename... Func>
using ChainResultType = typename ChainResultTypeHelper<T, Func...>::type;

}  // namespace internal_result_util

/// ChainResult() applies a sequence of functions, which may optionally
/// return `Result`-wrapped values, to an optionally `Result`-wrapped value.
///
/// For example:
///
///     float func1(int x);
///     Result<std::string> func2(float x);
///     bool func3(absl::string_view x);
///
///     Result<bool> y1 = ChainResult(Result<int>(3), func1, func2, func3);
///     Result<bool> y2 = ChainResult(3, func1, func2, func3);

/// This overload handles the base case of zero functions.
template <typename T>
internal_result_util::ChainResultType<T> ChainResult(T&& arg) {
  return std::forward<T>(arg);
}

/// Overload that handles the case of at least one function.
template <typename T, typename Func0, typename... Func>
internal_result_util::ChainResultType<T, Func0, Func...> ChainResult(
    T&& arg, Func0&& func0, Func&&... func) {
  return ChainResult(
      MapResult(std::forward<Func0>(func0), std::forward<T>(arg)),
      std::forward<Func>(func)...);
}

}  // namespace tensorstore

#endif  // TENSORSTORE_RESULT_UTIL_H_
