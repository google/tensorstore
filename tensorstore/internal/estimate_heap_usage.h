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

#ifndef TENSORSTORE_INTERNAL_ESTIMATE_HEAP_USAGE_H_
#define TENSORSTORE_INTERNAL_ESTIMATE_HEAP_USAGE_H_

/// \file
///
/// Defines `EstimateHeapUsage` for use by `Cache` implementations to estimate
/// memory usage.
///
/// The `include_children` parameter can be set to `false` to prevent recursion
/// into child objects of containers in case they are accounted for
/// separately/incrementally.

#include <optional>
#include <string>
#include <vector>

#include "absl/strings/cord.h"

namespace tensorstore {
namespace internal {

/// Assume no heap memory usage by default.
template <typename T>
size_t EstimateHeapUsage(const T& x, bool include_children = true) {
  return 0;
}

inline size_t EstimateHeapUsage(const std::string& x,
                                bool include_children = true) {
  return x.capacity();
}

inline size_t EstimateHeapUsage(const absl::Cord& x,
                                bool include_children = true) {
  return x.size();
}

template <typename T>
inline size_t EstimateHeapUsage(const std::vector<T>& x,
                                bool include_children = true) {
  size_t total = sizeof(T) * x.capacity();
  if (include_children) {
    for (const auto& child : x) {
      total += EstimateHeapUsage(child, include_children);
    }
  }
  return x.capacity();
}

template <typename T>
inline size_t EstimateHeapUsage(const std::optional<T>& x,
                                bool include_children = true) {
  if (!x) return 0;
  return EstimateHeapUsage(*x, include_children);
}

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_ESTIMATE_HEAP_USAGE_H_
