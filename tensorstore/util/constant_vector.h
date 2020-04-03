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

#include <algorithm>
#include <atomic>
#include <mutex>  // NOLINT
#include <string>
#include <type_traits>

#include "absl/base/macros.h"
#include "absl/debugging/leak_check.h"
#include "absl/utility/utility.h"
#include "tensorstore/index.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_constant_vector {

/// Maximum run time-specified array length that is supported by static
/// constexpr initialization.  If a longer length is specified at run time,
/// dynamic allocation is used.  Note that this limit does not apply to lengths
/// specified at compile time; for compile time-specified array lengths, static
/// consexpr initialization (using the GetStatic function defined below) is
/// always used.
constexpr DimensionIndex kInitialLength = 32;

template <typename T, T Value, std::size_t... Is>
struct StaticVectorStorage {
  constexpr static T vector[] = {(Is, Value)...};
};

template <typename T, T Value, std::size_t... Is>
constexpr T StaticVectorStorage<T, Value, Is...>::vector[];

/// Helper function that returns a pointer to a constexpr static array of `T`
/// containing `Value` repeated `sizeof...(Is)` times.
template <typename T, T Value, std::size_t... Is>
constexpr const T* GetStatic(absl::index_sequence<Is...>) {
  return StaticVectorStorage<T, Value, Is...>::vector;
}

/// Specialization for a length of 0.
template <typename T, T Value>
constexpr const T* GetStatic(absl::index_sequence<>) {
  return nullptr;
}

/// Helper class that manages a single array of type `T` filled with `Value` as
/// a static member variable, used by the GetConstantVector overload for a
/// length specified at run time, defined below.
///
/// Initially, a constexpr static array of capacity `kInitialLength` is used.
/// If an array of length greater than the existing capacity is requested, the
/// capacity is doubled until it exceeds the requested length, and a new array
/// of that capacity is dynamically allocated.  Because the returned arrays must
/// remain valid for the duration of the program, they are never freed.  This
/// means that the total memory used may be up to 4x the maximum size in bytes
/// that is requested (one factor of 2 is due to the repeated doubling, the
/// other factor of 2 accounts for the possibility that the maximum requested
/// length may be one more than a power of 2).
///
/// This class exists solely for the purpose of defining its static member
/// variables for each `T` and `Value` combination for which it is instantiated.
/// No objects of this class type are created.
template <typename T, T Value>
struct ConstantVectorData {
  /// Pointer to array of length at least `allocated_length` filled with
  /// `Value`.  Initially, this points to a constexpr static array of length
  /// `kInitialLength`.
  static std::atomic<const T*> allocated_vector;

  /// Specifies the length of the array to which `allocated_vector` points.
  static std::atomic<DimensionIndex> allocated_length;

  /// TODO(jbms): Use absl::Mutex once it supports static initialization.
  static std::mutex mutex;

  /// Ensures that `allocated_length` is at least `required_length`.
  static void EnsureLength(DimensionIndex required_length) {
    std::lock_guard<std::mutex> lock(mutex);
    DimensionIndex length = allocated_length.load(std::memory_order_relaxed);
    if (length >= required_length) return;
    do {
      length *= 2;
    } while (length < required_length);
    T* new_pointer = absl::IgnoreLeak(new T[length]);
    std::fill_n(new_pointer, length, Value);

    // We set allocated_vector before setting allocated_length.  This ensures
    // that allocated_length is always <= the length of allocated_vector.
    allocated_vector.store(new_pointer, std::memory_order_release);
    allocated_length.store(length, std::memory_order_release);
  }
};

template <typename T, T Value>
std::atomic<const T*> ConstantVectorData<T, Value>::allocated_vector{
    GetStatic<T, Value>(absl::make_index_sequence<kInitialLength>())};

template <typename T, T Value>
std::atomic<DimensionIndex> ConstantVectorData<T, Value>::allocated_length{
    kInitialLength};

template <typename T, T Value>
std::mutex ConstantVectorData<T, Value>::mutex;

extern template struct ConstantVectorData<Index, 0>;
extern template struct ConstantVectorData<Index, kInfIndex>;
extern template struct ConstantVectorData<Index, -kInfIndex>;
extern template struct ConstantVectorData<Index, kInfSize>;

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
/// \dchecks `length >= 0`
template <typename T, T Value>
span<const T> GetConstantVector(std::ptrdiff_t length) {
  ABSL_ASSERT(length >= 0);
  using V = internal_constant_vector::ConstantVectorData<T, Value>;
  if (length == 0) {
    return {};
  }
  if (length > V::allocated_length.load(std::memory_order_acquire)) {
    V::EnsureLength(length);
  }
  return span(V::allocated_vector.load(std::memory_order_acquire), length);
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
  static_assert(Length >= 0, "Length must be non-negative.");
  return span<const T, Length>(internal_constant_vector::GetStatic<T, Value>(
                                   absl::make_index_sequence<Length>()),
                               Length);
}

/// Returns a vector of `length` default-initialized `std::string` instances.
span<const std::string> GetDefaultStringVector(std::ptrdiff_t length);

/// Returns a `span<const std::string, Length>` filled with default-initialized
/// `std::string` instances.
template <std::ptrdiff_t Length>
inline span<const std::string, Length> GetDefaultStringVector(
    std::integral_constant<std::ptrdiff_t, Length> = {}) {
  static_assert(Length >= 0, "Length must be non-negative.");
  return span<const std::string, Length>(GetDefaultStringVector(Length).data(),
                                         Length);
}

}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_CONSTANT_VECTOR_H_
