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

#ifndef TENSORSTORE_UTIL_CONSTANT_VECTOR_H_
#define TENSORSTORE_UTIL_CONSTANT_VECTOR_H_

/// \file
/// Facilities for obtaining a `span<const T>` of a given length containing
/// specific constant values.

#include <array>
#include <string>

#include "tensorstore/rank.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_constant_vector {

template <typename T>
constexpr std::array<T, kMaxRank> GetConstantArray(T value) {
  std::array<T, kMaxRank> array = {};
  for (auto& x : array) x = value;
  return array;
}

template <typename T, T Value>
constexpr inline std::array<T, kMaxRank> kConstantArray =
    GetConstantArray(Value);

extern const std::string kStringArray[kMaxRank];

}  // namespace internal_constant_vector

/// Returns a `span<const T>` of length `length` filled with `Value`.
///
/// This overload is for a length specified at run time.  The overload below is
/// for a length specified at compile time.
///
/// Example:
///
///     span<const int> five_3s = GetConstantVector<int, 3>(5);
///     EXPECT_THAT(five_3s, ElementsAre(3, 3, 3, 3, 3));
///
/// The returned `span` is valid for the duration of the program.
///
/// \param length The length of the vector to return.
/// \dchecks `IsValidRank(length)`
template <typename T, T Value>
constexpr span<const T> GetConstantVector(std::ptrdiff_t length) {
  assert(IsValidRank(length));
  return {internal_constant_vector::kConstantArray<T, Value>.data(), length};
}

/// Returns a pointer to a constant array of length `kMaxRank` filled with
/// `Value`.
template <typename T, T Value>
constexpr const T* GetConstantVector() {
  return internal_constant_vector::kConstantArray<T, Value>.data();
}

/// Returns a `span<const T, Length>` filled with `Value`.
///
/// This overload is for a length specified at compile time.  The overload above
/// is for a length specified at run time.
///
/// Example:
///
///     span<const int, 5> five_3s = GetConstantVector<int, 3, 5>();
///     EXPECT_THAT(five_3s, ElementsAre(3, 3, 3, 3, 3));
///
///     span<const int, 4> four_2s = GetConstantVector<int, 2>(StaticRank<4>{});
///     EXPECT_THAT(four_2s, ElementsAre(2, 2, 2, 2));
///
/// The returned `span` is valid for the duration of the program.
template <typename T, T Value, std::ptrdiff_t Length>
constexpr span<const T, Length> GetConstantVector(
    std::integral_constant<std::ptrdiff_t, Length> = {}) {
  static_assert(IsValidRank(Length));
  return {internal_constant_vector::kConstantArray<T, Value>.data(), Length};
}

/// Returns a vector of `length` default-initialized `std::string` instances.
inline constexpr span<const std::string> GetDefaultStringVector(
    std::ptrdiff_t length) {
  assert(IsValidRank(length));
  return {internal_constant_vector::kStringArray, length};
}

inline constexpr const std::string* GetDefaultStringVector() {
  return internal_constant_vector::kStringArray;
}

/// Returns a `span<const std::string, Length>` filled with default-initialized
/// `std::string` instances.
template <std::ptrdiff_t Length>
inline constexpr span<const std::string, Length> GetDefaultStringVector(
    std::integral_constant<std::ptrdiff_t, Length> = {}) {
  static_assert(IsValidRank(Length));
  return {internal_constant_vector::kStringArray, Length};
}

}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_CONSTANT_VECTOR_H_
