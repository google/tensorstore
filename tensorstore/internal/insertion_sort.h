// Copyright 2026 The TensorStore Authors
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

#ifndef TENSORSTORE_INTERNAL_INSERTION_SORT_H_
#define TENSORSTORE_INTERNAL_INSERTION_SORT_H_

#include <stddef.h>
#include <stdint.h>

#include <functional>
#include <iterator>
#include <type_traits>
#include <utility>

#include "absl/base/attributes.h"
#include "tensorstore/internal/meta/attributes.h"

namespace tensorstore {
namespace internal {

/// Sorts `[first, last)` in place using stable insertion sort.
///
/// Preferred over `std::sort` on tiny ranges where stability or minimal code
/// size matters.
template <typename RandomIt, typename Compare = std::less<>>
ABSL_ATTRIBUTE_ALWAYS_INLINE constexpr void InsertionSort(RandomIt first,
                                                          RandomIt last,
                                                          Compare comp = {}) {
  using Value = typename std::iterator_traits<RandomIt>::value_type;
  if (first == last) return;
#if defined(__clang__)
#pragma clang loop unroll(disable)
#endif
  for (RandomIt it = first + 1; it != last; ++it) {
    if constexpr (std::is_trivially_copyable_v<Value>) {
      Value val = *it;
      RandomIt j = it;
      while (j != first && comp(val, *(j - 1))) {
        *j = *(j - 1);
        --j;
      }
      *j = val;
    } else {
      // For non-trivially-copyable types (e.g., `std::string`,
      // `nlohmann::json`), ADL `swap` avoids constructing/destroying a
      // temporary `Value` and bypasses copy-and-swap assignment destructors.
      using std::swap;
      for (RandomIt j = it; j != first && comp(*j, *(j - 1)); --j) {
        swap(*j, *(j - 1));
      }
    }
  }
}

namespace internal_sort {
// Ciura's (2001) gap sequence up to 1750, extended by `floor(2.25 * gap) | 1`.
constexpr int32_t kShellSortGaps[] = {
    1,        4,         10,        23,        57,        132,      301,
    701,      1750,      3937,      8859,      19933,     44849,    100911,
    227049,   510861,    1149437,   2586233,   5819025,   13092807, 29458815,
    66282335, 149135253, 335554319, 754997217, 1698743739};
}  // namespace internal_sort

/// Sorts `[first, last)` in place using unstable Shellsort with Ciura's gap
/// sequence.
///
/// Requires only random-access iterators, move-assignable elements, and a
/// strict weak ordering `comp`; avoids the code size of `std::sort`'s
/// introsort/heapsort on moderate-sized ranges.
template <typename RandomIt, typename Compare = std::less<>>
TENSORSTORE_INTERNAL_ATTRIBUTE_HIDDEN constexpr void ShellSort(
    RandomIt first, RandomIt last, Compare comp = {}) {
  using Diff = typename std::iterator_traits<RandomIt>::difference_type;
  using Value = typename std::iterator_traits<RandomIt>::value_type;
  const Diff n = last - first;
  // Delegate small ranges (`n <= 23`) and the final `gap == 1` pass to
  // `InsertionSort` so the unit-stride pass compiles with constant pointer
  // stride rather than variable `gap` indexing.
  if (n > 23) {
    constexpr size_t kNumGaps = sizeof(internal_sort::kShellSortGaps) /
                                sizeof(internal_sort::kShellSortGaps[0]);
    size_t gap_idx = 3;  // `kShellSortGaps[3] == 23 < n`.
#if defined(__clang__)
#pragma clang loop unroll(disable)
#endif
    while (gap_idx + 1 < kNumGaps &&
           static_cast<Diff>(internal_sort::kShellSortGaps[gap_idx + 1]) < n) {
      ++gap_idx;
    }
#if defined(__clang__)
#pragma clang loop unroll(disable)
#endif
    do {
      const Diff gap =
          static_cast<Diff>(internal_sort::kShellSortGaps[gap_idx]);
#if defined(__clang__)
#pragma clang loop unroll(disable)
#endif
      for (Diff i = gap; i < n; ++i) {
        if constexpr (std::is_trivially_copyable_v<Value>) {
          Value val = *(first + i);
          Diff j = i;
          while (j >= gap && comp(val, *(first + (j - gap)))) {
            *(first + j) = *(first + (j - gap));
            j -= gap;
          }
          *(first + j) = val;
        } else {
          using std::swap;
          for (Diff j = i; j >= gap && comp(*(first + j), *(first + (j - gap)));
               j -= gap) {
            swap(*(first + j), *(first + (j - gap)));
          }
        }
      }
    } while (--gap_idx > 0);
  }
  InsertionSort(first, last, std::move(comp));
}

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_INSERTION_SORT_H_
