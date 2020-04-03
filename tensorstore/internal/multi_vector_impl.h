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

#ifndef TENSORSTORE_INTERNAL_MULTI_VECTOR_IMPL_H_
#define TENSORSTORE_INTERNAL_MULTI_VECTOR_IMPL_H_

/// \file
/// Implementation details for `MultiVectorStorage`.
///
/// MultiVectorStorage stores multiple vectors in a contiguous block of memory
/// using the following layout:
///
///     T0 a0[extent];
///     <padding if needed to align T1>
///     T1 a1[extent];
///     <padding if needed to align T2>
///     T2 a2[extent];
///     ...
///
/// On most platforms this will be the same layout used by the compiler for a
/// struct with members:
///
///     T0 a0[extent];
///     T1 a1[extent];
///     T2 a2[extent];
///     ...
///
/// However, in the case that the extent is specified at run time, we cannot
/// rely on the compiler to compute the layout, and must instead compute it
/// ourselves.  This functionality is provided by the `GetVectorOffset` function
/// defined below.

// IWYU pragma: private, include "third_party/tensorstore/internal/multi_vector.h"

#include <algorithm>
#include <cstddef>

#include "tensorstore/internal/meta.h"
#include "tensorstore/util/division.h"

namespace tensorstore {
namespace internal_multi_vector {

/// Rounds `offset` up to a multiple of `next_alignment`, assuming it is already
/// a multiple of `prev_alignment`.
///
/// The seemingly unnecessary `prev_alignment` parameter allows better code
/// generation since the compiler may be unable to optimize out the RoundUpTo
/// operation otherwise.
inline constexpr std::ptrdiff_t GetAlignedOffset(
    std::ptrdiff_t offset, std::ptrdiff_t prev_alignment,
    std::ptrdiff_t next_alignment) {
  return prev_alignment >= next_alignment ? offset
                                          : RoundUpTo(offset, next_alignment);
}

/// Returns the byte offset of `array_i` within a multi vector of the specified
/// `extent` and with element types `Ts...`.
///
/// \param sizes Pointer to an array of size `array_i` specifying the element
///     size of each of the arrays.
/// \param alignments Pointer to an array of size `array_i+1` specifying the
///     alignment required for the element type of each of the arrays.
/// \param extent The extent of the multi vector.
/// \param array_i The index of the vector within the multi vector for which to
///     compute the starting offset.  If set to the total number of vectors, and
///     a `0` is appended to the `alignments` array, the return value is equal
///     to the total storage required by the multi vector.
inline constexpr std::ptrdiff_t GetVectorOffset(
    const std::ptrdiff_t sizes[], const std::ptrdiff_t alignments[],
    std::ptrdiff_t extent, std::size_t array_i) {
  if (array_i == 0) return 0;
  return GetAlignedOffset(
      GetVectorOffset(sizes, alignments, extent, array_i - 1) +
          sizes[array_i - 1] * extent,
      alignments[array_i - 1], alignments[array_i]);
}

/// Helper class for computing a multi vector storage layout.
///
/// \tparam `Ts...` The element types.
template <typename... Ts>
struct PackStorageOffsets {
  /// The alignments of `Ts...`.  An extra `0` is appended as required by
  /// `GetVectorOffset`.
  constexpr static std::ptrdiff_t kAlignments[] = {alignof(Ts)..., 0};

  /// The sizes of `Ts...`.
  constexpr static std::ptrdiff_t kSizes[] = {sizeof(Ts)...};

  /// Alignment required for the multi vector, equal to the maximum alignment of
  /// the element types `Ts...`.
  constexpr static std::size_t kAlignment =
      *std::max_element(std::begin(kAlignments), std::end(kAlignments));

  /// Returns the byte offset of vector `array_i`.
  static constexpr std::ptrdiff_t GetVectorOffset(std::ptrdiff_t extent,
                                                  std::size_t array_i) {
    return internal_multi_vector::GetVectorOffset(kSizes, kAlignments, extent,
                                                  array_i);
  }

  /// Returns the total number of bytes required for the multi vector.
  static constexpr std::ptrdiff_t GetTotalSize(std::ptrdiff_t extent) {
    return GetVectorOffset(extent, sizeof...(Ts));
  }
};

}  // namespace internal_multi_vector
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_MULTI_VECTOR_IMPL_H_
