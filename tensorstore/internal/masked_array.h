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

#ifndef TENSORSTORE_INTERNAL_MASKED_ARRAY_H_
#define TENSORSTORE_INTERNAL_MASKED_ARRAY_H_

/// \file
/// Functions for tracking modifications to an array using a mask array or
/// bounding box.

#include <algorithm>
#include <memory>

#include "tensorstore/array.h"
#include "tensorstore/box.h"
#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/internal/arena.h"
#include "tensorstore/internal/memory.h"
#include "tensorstore/internal/nditerable.h"
#include "tensorstore/util/element_pointer.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal {

/// Represents a binary mask over a hyperrectangle.
///
/// The actual hyperrectangle `mask_box` over which the mask is defined is
/// stored separately.
///
/// There are two possible representations of the mask:
///
/// If the region of the mask set to `true` happens to be a hyperrectangle, it
/// is represented simply as a `Box`.  Otherwise, it is represented using a
/// `bool` array.
struct MaskData {
  /// Initializes a mask in which no elements are included in the mask.
  explicit MaskData(DimensionIndex rank);

  void Reset() {
    num_masked_elements = 0;
    mask_array.reset();
    region.Fill(IndexInterval::UncheckedSized(0, 0));
  }

  /// If not `nullptr`, stores a mask array of size `mask_box.shape()` in C
  /// order, where all elements outside `region` are `false`.  If `nullptr`,
  /// indicates that all elements within `region` are masked.
  std::unique_ptr<bool[], FreeDeleter> mask_array;

  /// Number of `true` values in `mask_array`, or `region.num_elements()` if
  /// `mask_array` is `nullptr`.  As a special case, if `region.rank() == 0`,
  /// `num_masked_elements` may equal `0` even if `mask_array` is `nullptr` to
  /// indicate that the singleton element is not included in the mask.
  Index num_masked_elements = 0;

  /// Subregion of `mask_box` for which the mask is `true`.
  Box<> region;
};

/// Updates `*mask` to include all positions within `layout` up to (but not
/// including) `write_end_position`.
///
/// \param mask[in,out] Non-null pointer to mask with domain `output_box`.
/// \param output_box Domain of the `mask`.
/// \param input_to_output Transform that specifies the mapping from `layout` to
///     `output_box`.  Must be valid.
/// \param layout Iteration layout for the input space of `input_to_output`.
///     Must be valid for `input_to_output` applied to a contiguous C-order
///     array with domain `output_box`.
/// \param write_end_position One past the last position within `layout` to
///     include in the mask.
/// \param arena Allocation arena that may be used.
/// \returns `true` if the specified mask update region is non-empty.
bool WriteToMask(MaskData* mask, BoxView<> output_box,
                 IndexTransformView<> input_to_output,
                 NDIterable::IterationLayoutView layout,
                 span<const Index> write_end_position, Arena* arena);

/// Copies unmasked elements from `source_data` to `data_ptr`.
///
/// \param box Domain over which the mask is defined.
/// \param source[in] Source array with `source.shape() == box.shape()`.
/// \param dest_ptr[in,out] Pointer to the C-order contiguous destination array
///     of shape `source.shape()`.
/// \param mask[in] The mask specifying the positions of the destination array
///     not to modify.
/// \dchecks `source.dtype() == dest_ptr.dtype()`.
/// \dchecks `source.shape() == box.shape()`.
void RebaseMaskedArray(BoxView<> box, ArrayView<const void> source,
                       ElementPointer<void> dest_ptr, const MaskData& mask);

/// Assigns `*mask_a` to represent the union of two masks.
///
/// May modify `*mask_b`.
///
/// \param box The region over which the two masks are defined.
void UnionMasks(BoxView<> box, MaskData* mask_a, MaskData* mask_b);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_MASKED_ARRAY_H_
