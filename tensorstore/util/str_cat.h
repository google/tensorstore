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

#ifndef TENSORSTORE_UTIL_STR_CAT_H_
#define TENSORSTORE_UTIL_STR_CAT_H_

/// \file
/// Provides generic conversion to string representation.

#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include "absl/strings/str_cat.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_strcat {

// Requires is a sfinae helper that detects callability.
// Example:
//  if constexpr (Requires<T>([](auto&& v) -> decltype(v.begin()){})) {
//     t.begin();
//  }
template <typename... T, typename F>
constexpr bool Requires(F) {
  return std::is_invocable_v<F, T...>;
}

/// Converts arbitrary input values to a type supported by `absl::StrCat`.
template <typename T>
auto ToAlphaNumOrString(const T& x);

/// Converts the argument to a string representation using `operator<<`.
template <typename T>
std::string StringifyUsingOstream(const T& x) {
  std::ostringstream ostr;
  ostr << x;
  return ostr.str();
}

/// Converts std::tuple<...> values to strings.
template <typename... T>
std::string StringifyTuple(const std::tuple<T...>& x) {
  return std::apply(
      [](const auto&... item) {
        std::string result = "{";
        size_t i = 0;
        (absl::StrAppend(&result, ToAlphaNumOrString(item),
                         (++i == sizeof...(item) ? "}" : ", ")),
         ...);
        return result;
      },
      x);
}

/// Converts std::pair<...> values to strings.
template <typename A, typename B>
std::string StringifyPair(const std::pair<A, B>& x) {
  return absl::StrCat("{", ToAlphaNumOrString(x.first), ", ",
                      ToAlphaNumOrString(x.second), "}");
}

/// Converts container<T> values to strings.
template <typename Iterator>
std::string StringifyContainer(Iterator begin, Iterator end) {
  /// NOTE: Consider a PrintableContainer wrapper type.
  std::string result = "{";
  if (begin != end) {
    absl::StrAppend(&result, ToAlphaNumOrString(*begin++));
  }
  for (; begin != end; ++begin) {
    absl::StrAppend(&result, ", ", ToAlphaNumOrString(*begin));
  }
  absl::StrAppend(&result, "}");
  return result;
}

/// Converts arbitrary input values to a type supported by `absl::StrCat`.
template <typename T>
auto ToAlphaNumOrString(const T& x) {
  if constexpr (std::is_same_v<T, std::nullptr_t>) {
    return "null";
  } else if constexpr (std::is_convertible_v<T, absl::AlphaNum> &&
                       !std::is_enum_v<T>) {
    return x;
  } else if constexpr (internal::IsOstreamable<T>) {
    return StringifyUsingOstream(x);
  } else if constexpr (Requires<const T>(
                           [](auto&& v) -> decltype(StringifyPair(v)) {})) {
    return StringifyPair(x);
  } else if constexpr (Requires<const T>(
                           [](auto&& v) -> decltype(StringifyTuple(v)) {})) {
    return StringifyTuple(x);
  } else if constexpr (Requires<const T>(
                           [](auto&& v) -> decltype(v.begin(), v.end()) {})) {
    return StringifyContainer(x.begin(), x.end());
  } else if constexpr (std::is_enum_v<T>) {
    // Non-streamable enum
    using I = typename std::underlying_type<T>::type;
    return static_cast<I>(x);
  } else {
    // Fallback to streamed output to generate an error.
    return StringifyUsingOstream(x);
  }
}

}  // namespace internal_strcat

/// Prints a string representation of a span.
///
/// \requires `Element` supports ostream insertion.
/// \relates span
/// \id span
template <typename Element, std::ptrdiff_t N>
std::enable_if_t<internal::IsOstreamable<Element>, std::ostream&> operator<<(
    std::ostream& os, ::tensorstore::span<Element, N> s) {
  os << "{";
  std::ptrdiff_t size = s.size();
  for (std::ptrdiff_t i = 0; i < size; ++i) {
    if (i != 0) os << ", ";
    os << s[i];
  }
  return os << "}";
}

/// Concatenates the string representation of `arg...` and returns the result.
///
/// The string conversion is done exactly the same as for `StrAppend`.
///
/// \ingroup string-utilities
template <typename... Arg>
std::string StrCat(const Arg&... arg) {
  return absl::StrCat(internal_strcat::ToAlphaNumOrString(arg)...);
}

/// Appends a string representation of arg... to `*result`.
///
/// The arguments are converted to a string representation as follows:
///
/// `std::string` and `std::string_view`
///   Appended directly
///
/// Numerical types
///   Converted to their usual string representation
///
/// All other types
///   Converted using ostream `operator<<`.
///
/// Specifying an argument type that does not support `operator<<` results in a
/// compile-time error.
///
/// \ingroup string-utilities
template <typename... Arg>
void StrAppend(std::string* result, const Arg&... arg) {
  return absl::StrAppend(result, internal_strcat::ToAlphaNumOrString(arg)...);
}

}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_STR_CAT_H_
