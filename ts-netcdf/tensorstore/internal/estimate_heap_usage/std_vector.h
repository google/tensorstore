// Copyright 2022 The TensorStore Authors
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

#ifndef TENSORSTORE_INTERNAL_ESTIMATE_HEAP_USAGE_STD_VECTOR_H_
#define TENSORSTORE_INTERNAL_ESTIMATE_HEAP_USAGE_STD_VECTOR_H_

#include <vector>

#include "tensorstore/internal/estimate_heap_usage/estimate_heap_usage.h"

namespace tensorstore {
namespace internal {

template <typename T>
struct HeapUsageEstimator<std::vector<T>> {
  static size_t EstimateHeapUsage(const std::vector<T>& v, size_t max_depth) {
    size_t total = sizeof(T) * v.capacity();
    if constexpr (MayUseHeapMemory<T>) {
      if (max_depth > 0) {
        for (auto& x : v) {
          total += internal::EstimateHeapUsage(x, max_depth - 1);
        }
      }
    }
    return total;
  }
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_ESTIMATE_HEAP_USAGE_STD_VECTOR_H_
