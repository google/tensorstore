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

#ifndef TENSORSTORE_INTERNAL_ESTIMATE_HEAP_USAGE_STD_VARIANT_H_
#define TENSORSTORE_INTERNAL_ESTIMATE_HEAP_USAGE_STD_VARIANT_H_

#include <variant>

#include "tensorstore/internal/estimate_heap_usage/estimate_heap_usage.h"

namespace tensorstore {
namespace internal {

template <typename... T>
struct HeapUsageEstimator<std::variant<T...>> {
  static size_t EstimateHeapUsage(const std::variant<T...>& v,
                                  size_t max_depth) {
    if (v.valueless_by_exception()) return 0;
    return std::visit(
        [&](auto& x) { return internal::EstimateHeapUsage(x, max_depth); }, v);
  }
  static constexpr bool MayUseHeapMemory() {
    return (internal::MayUseHeapMemory<T> || ...);
  }
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_ESTIMATE_HEAP_USAGE_STD_VARIANT_H_
