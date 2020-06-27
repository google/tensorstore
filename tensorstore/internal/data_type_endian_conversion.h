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

#ifndef TENSORSTORE_INTERNAL_DATA_TYPE_ENDIAN_CONVERSION_H_
#define TENSORSTORE_INTERNAL_DATA_TYPE_ENDIAN_CONVERSION_H_

#include <array>

#include "absl/strings/cord.h"
#include "tensorstore/array.h"
#include "tensorstore/data_type.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/endian.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal {

/// Functions for endian conversion and unaligned copying for a particular data
/// type.
struct UnalignedDataTypeFunctions {
  /// Swaps endianness in place.  No alignment requirement.
  ///
  /// If the data type does not require endian conversion (e.g. uint8), equal to
  /// `nullptr`.
  const internal::ElementwiseFunction<1, Status*>* swap_endian_inplace =
      nullptr;

  /// Swaps endianness, copying from first argument to second.  No alignment
  /// requirement.
  ///
  /// If the data type does not require endian conversion (e.g. uint8), equal to
  /// `copy`.
  const internal::ElementwiseFunction<2, Status*>* swap_endian = nullptr;

  /// Copies potentially unaligned data from first argument to second.
  ///
  /// If the data type is a non-trivial type (e.g. `string` or `json`), equal to
  /// `nullptr`.
  const internal::ElementwiseFunction<2, Status*>* copy = nullptr;
};

/// Functions for each canonical data type.
extern const std::array<UnalignedDataTypeFunctions, kNumDataTypeIds>
    kUnalignedDataTypeFunctions;

/// Copies `source` to `target` with the specified `target_endian`.
///
/// \param source The source array, assumed to be in the native endianness.
/// \param target Target array, does not have to be suitably aligned for its
///     data type.
/// \param target_endian The target array endianness.
/// \pre `source.data_type()` is a trivial data type.
/// \dchecks `source.data_type() == target.data_type()`
/// \dchecks `source.shape() == target.shape()`
void EncodeArray(ArrayView<const void> source, ArrayView<void> target,
                 endian target_endian);

/// Copies `source` to `target` with the specified `source_endian`.
///
/// This performs endian conversion if necessary.  Additionally, for `bool` data
/// it ensures the value is exactly `0` or `1` (i.e. masks out all but the
/// lowest bit).
///
/// \param source The source array, in `source_endian` byte order.  Does not
///     have to be suitably aligned for its data type.
/// \param source_endian The source array endianness.
/// \param target Target array, assumed to be in the native endianness.
/// \pre `source.data_type()` is a trivial data type.
/// \dchecks `source.data_type() == target.data_type()`
/// \dchecks `source.shape() == target.shape()`
void DecodeArray(ArrayView<const void> source, endian source_endian,
                 ArrayView<void> target);

/// Decodes `*source` from `source_endian` into the native endianness.
///
/// If `*source` is suitably aligned for its data type, it is transformed in
/// place.  Otherwise, sets
/// `*source = CopyAndDecodeArray(*source, source_endian, decoded_layout)`.
///
/// This performs endian conversion if necessary.  Additionally, for `bool` data
/// it ensures the value is exactly `0` or `1` (i.e. masks out all but the
/// lowest bit).
///
/// \param source[in,out] Non-null pointer to source array, which does not have
///     to be suitably aligned for its data type.
/// \param source_endian The source array endianness.
/// \param decoded_layout Layout to use if `*source` is not suitably aligned and
///     a new array must be allocated.  Must be equivalent to a permutation of a
///     C order layout.
/// \dchecks `decoded_layout.shape() == source->shape()`
void DecodeArray(SharedArrayView<void>* source, endian source_endian,
                 StridedLayoutView<> decoded_layout);

/// Returns a decoded copy of `source` in a newly allocated array with
/// `decoded_layout`.
///
/// \param source Source array, does not have to be suitably aligned for its
///     data type.
/// \param source_endian The source array endianness.
/// \param decoded_layout Layout to use for copy.  Must be equivalent to a
///     permutation of a C order layout.
SharedArrayView<void> CopyAndDecodeArray(ArrayView<const void> source,
                                         endian source_endian,
                                         StridedLayoutView<> decoded_layout);

/// Attempts to view the substring of `source` starting at `offset` as an array
/// of the specified `data_type` and `layout`.
///
/// If `source` is already flattened, suitably aligned, and requires no
/// conversion, then an array view that shares ownership with `source` is
/// returned.
///
/// Otherwise, returns a null array.
SharedArrayView<const void> TryViewCordAsArray(const absl::Cord& source,
                                               Index offset, DataType dtype,
                                               endian source_endian,
                                               StridedLayoutView<> layout);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_DATA_TYPE_ENDIAN_CONVERSION_H_
