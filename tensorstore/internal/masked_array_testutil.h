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

#ifndef TENSORSTORE_INTERNAL_MASKED_ARRAY_TESTUTIL_H_
#define TENSORSTORE_INTERNAL_MASKED_ARRAY_TESTUTIL_H_

#include "absl/status/status.h"
#include "tensorstore/box.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/transformed_array.h"
#include "tensorstore/internal/element_copy_function.h"
#include "tensorstore/internal/masked_array.h"
#include "tensorstore/util/element_pointer.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal {

/// Return type of WriteToMaskedArray.  This is used in place of `Result<bool>`
/// because the array may have been modified even in the case of an error, or
/// may have not been modified even in the case of success.
struct MaskedArrayWriteResult {
  absl::Status status;
  bool modified;
};

/// Copies the contents of `source` to an "output" array, and updates `*mask` to
/// include all positions that were modified.
///
/// \param output_ptr[out] Pointer to the origin (not the zero position) of a
///     C-order contiguous "output" array with domain `output_box`.
/// \param mask[in,out] Non-null pointer to mask with domain `output_box`.
/// \param input_to_output Transform to apply to the "output" array.  Must be
///     valid.
/// \param source Source array to copy to the transformed output array.
/// \param copy_function Element-wise function to use for copying.
/// \param copy_context_ptr Context pointer to pass to `copy_function`.
/// \pre The range of `input_to_output` must be a subset of `output_box`.
/// \returns A MaskedArrayWriteResult with `status` indicating if an error
///     occurred, and `modified` indicating if the "output" array was modified.
/// \error `absl::StatusCode::kInvalidArgument` if the transformed arrays do not
///     all have the same rank.
/// \error `absl::StatusCode::kInvalidArgument` if the transformed arrays do not
///     have compatible domains.
/// \error `absl::StatusCode::kOutOfRange` if an index array contains an
///     out-of-bounds index.
/// \error `absl::StatusCode::kInvalidArgument` if integer overflow occurs
///     computing output indices.
MaskedArrayWriteResult WriteToMaskedArray(
    ElementPointer<void> output_ptr, MaskData* mask, BoxView<> output_box,
    IndexTransformView<> input_to_output, TransformedArray<const void> source,
    ElementCopyFunction::Closure copy_function);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_MASKED_ARRAY_TESTUTIL_H_
