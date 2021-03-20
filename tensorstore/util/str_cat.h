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
#include <string_view>
#include <type_traits>

#include "absl/strings/str_cat.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/util/span.h"

namespace tensorstore {

/// Converts the argument to a string representation using `operator<<`.
template <typename T>
std::string ToStringUsingOstream(const T& x) {
  std::ostringstream ostr;
  ostr << x;
  return ostr.str();
}

namespace internal {

/// Converts arbitrary input values to a type supported by `absl::StrCat`.
template <typename T>
auto ToAlphaNumOrString(const T& x) {
  if constexpr (std::is_same_v<T, std::nullptr_t>) {
    return "null";
  } else if constexpr (std::is_convertible_v<T, absl::AlphaNum> &&
                       (!std::is_enum_v<T> || !IsOstreamable<T>::value)) {
    return absl::AlphaNum(x);
  } else if constexpr (IsOstreamable<T>::value) {
    // Note: Return type is `std::string` in this case.  If it were
    // `absl::AlphaNum`, the `AlphaNum` would hold an invalid reference to a
    // temporary string.
    return tensorstore::ToStringUsingOstream(x);
  } else {
    return "<unprintable>";
  }
}

}  // namespace internal

/// Prints a string representation of a span.
template <typename Element, std::ptrdiff_t N>
std::enable_if_t<internal::IsOstreamable<Element>::value, std::ostream&>
operator<<(std::ostream& os, span<Element, N> s) {
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
template <typename... Arg>
std::string StrCat(const Arg&... arg) {
  return absl::StrCat(internal::ToAlphaNumOrString(arg)...);
}

/// Appends a string representation of arg... to `*result`.
///
/// The arguments are converted to a string representation using
/// `absl::AlphaNum` if supported, or otherwise `ToStringUsingOstream`.
template <typename... Arg>
void StrAppend(std::string* result, const Arg&... arg) {
  return absl::StrAppend(result, internal::ToAlphaNumOrString(arg)...);
}

}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_STR_CAT_H_
