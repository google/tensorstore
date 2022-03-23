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

#ifndef TENSORSTORE_INTERNAL_UNALIGNED_DATA_TYPE_FUNCTIONS_H_
#define TENSORSTORE_INTERNAL_UNALIGNED_DATA_TYPE_FUNCTIONS_H_

#include <array>

#include "absl/status/status.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/serialization/fwd.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal {

/// Functions for endian conversion, unaligned copying, and encoding/decoding
/// for a particular data type.
struct UnalignedDataTypeFunctions {
  /// Swaps endianness in place.  No alignment requirement.
  ///
  /// If the data type does not require endian conversion (e.g. uint8), equal to
  /// `nullptr`.
  const internal::ElementwiseFunction<1, absl::Status*>* swap_endian_inplace =
      nullptr;

  /// Swaps endianness, copying from first argument to second.  No alignment
  /// requirement.
  ///
  /// If the data type does not require endian conversion (e.g. uint8), equal to
  /// `copy`.
  const internal::ElementwiseFunction<2, absl::Status*>* swap_endian = nullptr;

  /// Copies potentially unaligned data from first argument to second.
  ///
  /// If the data type is a non-trivial type (e.g. `string` or `json`), equal to
  /// `nullptr`.
  const internal::ElementwiseFunction<2, absl::Status*>* copy = nullptr;

  /// For trivial types, writes to a `riegeli::Writer` without swapping byte
  /// order.  For non-trivial types, writes to a `riegeli::Writer` using a
  /// canonical binary encoding (varint length-delimited for strings, CBOR for
  /// json).
  ///
  /// The `context` points to a `riegeli::Writer`.  The `absl::Status` parameter
  /// is ignored.
  internal::ElementwiseFunction<1, absl::Status*> write_native_endian;

  /// For trivial types, writes with swapped endianness.  For non-trivial
  /// types, same as `write_native_endian`.
  ///
  /// The `context` points to a `riegeli::Writer`.  The `absl::Status` parameter
  /// is ignored.
  internal::ElementwiseFunction<1, absl::Status*> write_swapped_endian;

  /// Reads the result of `write_native_endian` from a `riegeli::Reader`.
  ///
  /// The `context` points to a `riegeli::Reader`.  The `absl::Status` parameter
  /// is ignored.
  internal::ElementwiseFunction<1, absl::Status*> read_native_endian;

  /// Decodes the result of `write_swap_endian` from a `riegeli::Reader`.
  ///
  /// The `context` points to a `riegeli::Reader`.  The `absl::Status` parameter
  /// is ignored.
  internal::ElementwiseFunction<1, absl::Status*> read_swapped_endian;
};

/// Functions for each canonical data type.
extern const std::array<UnalignedDataTypeFunctions, kNumDataTypeIds>
    kUnalignedDataTypeFunctions;

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_UNALIGNED_DATA_TYPE_FUNCTIONS_H_
