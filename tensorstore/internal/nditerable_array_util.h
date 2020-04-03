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

#ifndef TENSORSTORE_INTERNAL_NDITERABLE_ARRAY_UTIL_H_
#define TENSORSTORE_INTERNAL_NDITERABLE_ARRAY_UTIL_H_

/// \file
/// Utility functions for defining array-based NDIterable implementations.
///
/// These functions are used by `nditerable_array.cc` and
/// nditerable_transformed_array.cc`.

#include <cmath>

#include "tensorstore/index.h"
#include "tensorstore/internal/integer_overflow.h"
#include "tensorstore/internal/nditerable.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal {

/// Returns `true` if iterating over dimension `i` as an outer loop, and then
/// iterating over dimension `j` as an inner loop, is equivalent to just
/// iterating over dimension `j`.
///
/// \param byte_stride_i Byte stride value of dimension `i`.
/// \param dir_i Direction of dimension `i`, must be `-1` or `1`.
/// \param byte_stride_j Byte stride value of dimension `j`.
/// \param dir_j Direction of dimension `j`, must be `-1` or `1`.
/// \param size_j Extent of dimension `j`.
inline bool CanCombineStridedArrayDimensions(Index byte_stride_i, int dir_i,
                                             Index byte_stride_j, int dir_j,
                                             Index size_j) {
  return wrap_on_overflow::Multiply(byte_stride_i, static_cast<Index>(dir_i)) ==
         wrap_on_overflow::Multiply(
             wrap_on_overflow::Multiply(byte_stride_j,
                                        static_cast<Index>(dir_j)),
             size_j);
}

/// Returns the `DirectionPref` corresponding to an array dimension with a given
/// byte stride.
inline NDIterable::DirectionPref DirectionPrefFromStride(Index byte_stride) {
  return byte_stride == 0
             ? NDIterable::DirectionPref::kCanSkip
             : byte_stride > 0 ? NDIterable::DirectionPref::kForward
                               : NDIterable::DirectionPref::kBackward;
}

/// Merges the `DirectionPref` values corresponding to `byte_strides` into
/// `prefs`.
///
/// \param byte_strides Array byte strides.
/// \param prefs[in,out] Pointer to array of length `byte_strides.size()` to
///     update via `NDIterable::CombineDirectionPrefs`.
inline void UpdateDirectionPrefsFromByteStrides(
    span<const Index> byte_strides, NDIterable::DirectionPref* prefs) {
  for (DimensionIndex i = 0; i < byte_strides.size(); ++i) {
    prefs[i] = NDIterable::CombineDirectionPrefs(
        prefs[i], DirectionPrefFromStride(byte_strides[i]));
  }
}

/// Determines the most efficient relative iteration order for a pair of array
/// dimensions.
///
/// Returns `-1` (meaning the first dimension should be the outer dimension),
/// `0` (meaning no preference), or `+1` (meaning the second dimension should be
/// the outer dimension) if the magnitude of `byte_stride_i` is less than, equal
/// to, or greater than, the magnitude of `byte_stride_j`, respectively.
inline int GetDimensionOrderFromByteStrides(Index byte_stride_i,
                                            Index byte_stride_j) {
  byte_stride_i = std::abs(byte_stride_i);
  byte_stride_j = std::abs(byte_stride_j);
  return byte_stride_i < byte_stride_j
             ? 1
             : byte_stride_i == byte_stride_j ? 0 : -1;
}

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_NDITERABLE_ARRAY_UTIL_H_
