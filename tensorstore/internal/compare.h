// Copyright 2024 The TensorStore Authors
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

#ifndef TENSORSTORE_INTERNAL_COMPARE_H_
#define TENSORSTORE_INTERNAL_COMPARE_H_

#include <type_traits>

#include "absl/types/compare.h"

namespace tensorstore {
namespace internal {

// Helper functions to do a three-way comparison of two keys given a boolean or
// three-way comparator.
// SFINAE prevents implicit conversions to int (such as from bool).
template <typename Int,
          std::enable_if_t<std::is_same<int, Int>::value, int> = 0>
constexpr absl::weak_ordering CompareResultAsWeakOrdering(const Int c) {
  return (c < 0)    ? absl::weak_ordering::less
         : (c == 0) ? absl::weak_ordering::equivalent
                    : absl::weak_ordering::greater;
}
constexpr inline absl::weak_ordering CompareResultAsWeakOrdering(
    const absl::weak_ordering c) {
  return c;
}

// Helper function to return an absl::weak_ordering from a comparison function;
// `Compare` can return a boolean, such as `std::less<>` or it may return an
// integer value, as in `std::string_view::compare`.
template <
    typename Compare, typename K, typename LK,
    std::enable_if_t<!std::is_same<bool, std::result_of_t<Compare(
                                             const K &, const LK &)>>::value,
                     int> = 0>
constexpr absl::weak_ordering DoThreeWayComparison(const Compare &compare,
                                                   const K &x, const LK &y) {
  return CompareResultAsWeakOrdering(compare(x, y));
}

template <
    typename Compare, typename K, typename LK,
    std::enable_if_t<std::is_same<bool, std::result_of_t<Compare(
                                            const K &, const LK &)>>::value,
                     int> = 0>
constexpr absl::weak_ordering DoThreeWayComparison(const Compare &compare,
                                                   const K &x, const LK &y) {
  return compare(x, y)   ? absl::weak_ordering::less
         : compare(y, x) ? absl::weak_ordering::greater
                         : absl::weak_ordering::equivalent;
}

// Inverts an absl::weak_ordering.
constexpr inline absl::weak_ordering InvertWeakOrdering(
    absl::weak_ordering order) {
  return (order < 0)   ? absl::weak_ordering::greater
         : (order > 0) ? absl::weak_ordering::less
                       : absl::weak_ordering::equivalent;
}

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_COMPARE_H_
