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

#ifndef TENSORSTORE_INTERNAL_ESTIMATE_HEAP_USAGE_ESTIMATE_HEAP_USAGE_H_
#define TENSORSTORE_INTERNAL_ESTIMATE_HEAP_USAGE_ESTIMATE_HEAP_USAGE_H_

/// \file
///
/// Defines `EstimateHeapUsage` for use by `Cache` implementations to estimate
/// memory usage.

#include <cstddef>
#include <string>
#include <type_traits>

#include "absl/strings/cord.h"
#include "tensorstore/util/apply_members/apply_members.h"

namespace tensorstore {
namespace internal {

template <typename T, typename SFINAE = void>
struct HeapUsageEstimator;

template <typename T, typename SFINAE = void>
constexpr inline bool MayUseHeapMemory = true;

template <typename T>
constexpr inline bool MayUseHeapMemory<
    T, std::enable_if_t<
           !std::is_trivially_destructible_v<T>,
           std::void_t<decltype(&HeapUsageEstimator<T>::MayUseHeapMemory)>>> =
    HeapUsageEstimator<T>::MayUseHeapMemory();

template <typename T>
constexpr inline bool
    MayUseHeapMemory<T, std::enable_if_t<std::is_trivially_destructible_v<T>>> =
        false;

/// Returns an estimate of the total heap memory owned by `x`, following owned
/// pointers up to a depth of `max_depth`.
///
/// The returned size should only include heap memory owned by `x`; it should
/// not include `sizeof(T)`.
///
/// Ideally it should include any per-allocation overhead.
template <typename T>
size_t EstimateHeapUsage(const T& x, size_t max_depth = -1) {
  if constexpr (!MayUseHeapMemory<T>) {
    return 0;
  } else {
    return HeapUsageEstimator<T>::EstimateHeapUsage(x, max_depth);
  }
}

/// Returns `true` if any argument may use heap memory.
struct MayAnyUseHeapMemory {
  template <typename... T>
  constexpr auto operator()(const T&... arg) const {
    return std::integral_constant<bool, (MayUseHeapMemory<T> || ...)>{};
  }
};

/// Specialization of `HeapUsageEstimator` for types that support
/// `ApplyMembers`.
template <typename T>
struct HeapUsageEstimator<T, std::enable_if_t<SupportsApplyMembers<T>>> {
  static size_t EstimateHeapUsage(const T& v, size_t max_depth) {
    return ApplyMembers<T>::Apply(v, [&](auto&&... x) {
      // Note: `max_depth` is passed through since these are directly contained
      // objects, we aren't following any pointers here.
      return (internal::EstimateHeapUsage(x, max_depth) + ... +
              static_cast<size_t>(0));
    });
  }

  static constexpr bool MayUseHeapMemory() {
    return decltype(ApplyMembers<T>::Apply(std::declval<const T&>(),
                                           MayAnyUseHeapMemory{}))::value;
  }
};

template <>
struct HeapUsageEstimator<std::string> {
  static size_t EstimateHeapUsage(const std::string& x, size_t max_depth) {
    // FIXME: include heap allocation overhead
    return x.capacity();
  }
};

template <>
struct HeapUsageEstimator<absl::Cord> {
  static size_t EstimateHeapUsage(const absl::Cord& x, size_t max_depth) {
    return x.size();
  }
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_ESTIMATE_HEAP_USAGE_H_
