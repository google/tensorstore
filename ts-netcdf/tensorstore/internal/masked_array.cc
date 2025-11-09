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

#include "tensorstore/internal/masked_array.h"

#include <algorithm>
#include <cassert>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "tensorstore/array.h"
#include "tensorstore/box.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/internal/arena.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/internal/integer_overflow.h"
#include "tensorstore/internal/nditerable_buffer_management.h"
#include "tensorstore/internal/nditerable_transformed_array.h"
#include "tensorstore/internal/unowned_to_shared.h"
#include "tensorstore/rank.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/byte_strided_pointer.h"
#include "tensorstore/util/element_pointer.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal {
namespace {

/// Unary function that sets every element to `true`.
struct SetMask {
  void operator()(bool* x, void*) const { *x = true; }
};

/// Unary function that sets every element to `true`, and increments
/// `num_changed` each time a `false` value is encountered.  The count is used
/// to update `MaskData::num_masked_elements`.
struct SetMaskAndCountChanged {
  Index num_changed = 0;
  void operator()(bool* x) {
    if (!*x) {
      ++num_changed;
      *x = true;
    }
  }
};

bool IsHullEqualToUnion(BoxView<> a, BoxView<> b) {
  assert(a.rank() == b.rank());

  Index hull_num_elements = 1, a_num_elements = 1, b_num_elements = 1,
        intersection_num_elements = 1;
  for (DimensionIndex i = 0; i < a.rank(); ++i) {
    IndexInterval a_interval = a[i], b_interval = b[i];
    IndexInterval hull = Hull(a_interval, b_interval);
    IndexInterval intersection = Intersect(a_interval, b_interval);

    // Overflow cannot occur because total number of elements in must fit in
    // memory.
    hull_num_elements *= hull.size();
    a_num_elements *= a_interval.size();
    b_num_elements *= b_interval.size();
    intersection_num_elements *= intersection.size();
  }

  return (hull_num_elements ==
          a_num_elements + b_num_elements - intersection_num_elements);
}

void Hull(BoxView<> a, BoxView<> b, MutableBoxView<> out) {
  const DimensionIndex rank = out.rank();
  assert(a.rank() == rank && b.rank() == rank);
  for (DimensionIndex i = 0; i < rank; ++i) {
    out[i] = Hull(a[i], b[i]);
  }
}

void Intersect(BoxView<> a, BoxView<> b, MutableBoxView<> out) {
  const DimensionIndex rank = out.rank();
  assert(a.rank() == rank && b.rank() == rank);
  for (DimensionIndex i = 0; i < rank; ++i) {
    out[i] = Intersect(a[i], b[i]);
  }
}

Index GetRelativeOffset(tensorstore::span<const Index> base,
                        tensorstore::span<const Index> position,
                        tensorstore::span<const Index> strides) {
  const DimensionIndex rank = base.size();
  assert(rank == position.size());
  assert(rank == strides.size());
  Index result = 0;
  for (DimensionIndex i = 0; i < rank; ++i) {
    result = internal::wrap_on_overflow::Add(
        result, internal::wrap_on_overflow::Multiply(
                    strides[i], internal::wrap_on_overflow::Subtract(
                                    position[i], base[i])));
  }
  return result;
}

void RemoveMaskArrayIfNotNeeded(MaskData* mask) {
  if (mask->num_masked_elements == mask->region.num_elements()) {
    mask->mask_array.element_pointer() = {};
  }
}
}  // namespace

MaskData::MaskData(DimensionIndex rank) : region(rank) {
  region.Fill(IndexInterval::UncheckedSized(0, 0));
}

SharedArray<bool> CreateMaskArray(BoxView<> box, BoxView<> mask_region,
                                  ContiguousLayoutPermutation<> layout_order) {
  auto array = AllocateArray<bool>(box.shape(), layout_order, value_init);
  ByteStridedPointer<bool> start = array.data();
  start += GetRelativeOffset(box.origin(), mask_region.origin(),
                             array.byte_strides());
  internal::IterateOverArrays(
      internal::SimpleElementwiseFunction<SetMask(bool), void*>{},
      /*arg=*/nullptr,
      /*constraints=*/skip_repeated_elements,
      ArrayView<bool>(start.get(), StridedLayoutView<>(mask_region.shape(),
                                                       array.byte_strides())));
  return array;
}

void CreateMaskArrayFromRegion(BoxView<> box, MaskData* mask,
                               ContiguousLayoutPermutation<> layout_order) {
  assert(mask->num_masked_elements == mask->region.num_elements());
  assert(layout_order.size() == mask->region.rank());
  mask->mask_array = CreateMaskArray(box, mask->region, layout_order);
}

void UnionMasks(BoxView<> box, MaskData* mask_a, MaskData* mask_b,
                ContiguousLayoutPermutation<> layout_order) {
  assert(mask_a != mask_b);  // May work but not supported.
  if (mask_a->num_masked_elements == 0) {
    std::swap(*mask_a, *mask_b);
    return;
  } else if (mask_b->num_masked_elements == 0) {
    return;
  }
  assert(mask_a->region.rank() == box.rank());
  assert(mask_b->region.rank() == box.rank());

  if (mask_a->mask_array.valid() && mask_b->mask_array.valid()) {
    Index num_masked_elements = 0;
    IterateOverArrays(
        [&](bool* a, bool* b) {
          if ((*a |= *b) == true) {
            ++num_masked_elements;
          }
        },
        /*constraints=*/{}, mask_a->mask_array, mask_b->mask_array);
    mask_a->num_masked_elements = num_masked_elements;
    Hull(mask_a->region, mask_b->region, mask_a->region);
    RemoveMaskArrayIfNotNeeded(mask_a);
    return;
  }

  if (!mask_a->mask_array.valid() && !mask_b->mask_array.valid()) {
    if (IsHullEqualToUnion(mask_a->region, mask_b->region)) {
      // The combined mask can be specified by the region alone.
      Hull(mask_a->region, mask_b->region, mask_a->region);
      mask_a->num_masked_elements = mask_a->region.num_elements();
      return;
    }
  } else if (!mask_a->mask_array.valid()) {
    std::swap(*mask_a, *mask_b);
  }

  if (!mask_a->mask_array.valid()) {
    CreateMaskArrayFromRegion(box, mask_a, layout_order);
  }

  // Copy in mask_b.
  ByteStridedPointer<bool> start = mask_a->mask_array.data();
  start += GetRelativeOffset(box.origin(), mask_b->region.origin(),
                             mask_a->mask_array.byte_strides());
  IterateOverArrays(
      [&](bool* ptr) {
        if (!*ptr) ++mask_a->num_masked_elements;
        *ptr = true;
      },
      /*constraints=*/{},
      ArrayView<bool>(start.get(),
                      StridedLayoutView<>(mask_b->region.shape(),
                                          mask_a->mask_array.byte_strides())));
  Hull(mask_a->region, mask_b->region, mask_a->region);
  RemoveMaskArrayIfNotNeeded(mask_a);
}

void RebaseMaskedArray(BoxView<> box, ArrayView<const void> source,
                       ArrayView<void> dest, const MaskData& mask) {
  assert(source.dtype() == dest.dtype());
  assert(internal::RangesEqual(box.shape(), source.shape()));
  assert(internal::RangesEqual(box.shape(), dest.shape()));
  const Index num_elements = box.num_elements();
  if (mask.num_masked_elements == num_elements) return;
  DataType dtype = source.dtype();
  if (mask.num_masked_elements == 0) {
    [[maybe_unused]] const auto success = internal::IterateOverArrays(
        {&dtype->copy_assign, /*context=*/nullptr},
        /*arg=*/nullptr, skip_repeated_elements, source, dest);
    assert(success);
    return;
  }

  // Materialize mask array.
  ArrayView<bool> mask_array_view;
  SharedArray<bool> mask_array;
  if (mask.mask_array.valid()) {
    mask_array_view = mask.mask_array;
  } else {
    DimensionIndex layout_order[kMaxRank];
    tensorstore::span<DimensionIndex> layout_order_span(layout_order,
                                                        dest.rank());
    SetPermutationFromStrides(dest.byte_strides(), layout_order_span);
    mask_array = CreateMaskArray(
        box, mask.region, ContiguousLayoutPermutation<>(layout_order_span));
    mask_array_view = mask_array;
  }
  [[maybe_unused]] const auto success = internal::IterateOverArrays(
      {&dtype->copy_assign_unmasked, /*context=*/nullptr},
      /*arg=*/nullptr, skip_repeated_elements, source, dest, mask_array_view);
  assert(success);
}

void WriteToMask(MaskData* mask, BoxView<> output_box,
                 IndexTransformView<> input_to_output,
                 ContiguousLayoutPermutation<> layout_order, Arena* arena) {
  assert(input_to_output.output_rank() == output_box.rank());

  if (input_to_output.domain().box().is_empty()) {
    return;
  }

  const DimensionIndex output_rank = output_box.rank();
  Box<dynamic_rank(kNumInlinedDims)> output_range(output_rank);
  // Supplied `input_to_output` transform must have already been validated.
  const bool range_is_exact =
      GetOutputRange(input_to_output, output_range).value();
  Intersect(output_range, output_box, output_range);

  const bool use_mask_array =
      output_box.rank() != 0 &&
      mask->num_masked_elements != output_box.num_elements() &&
      (mask->mask_array.valid() ||
       (!Contains(mask->region, output_range) &&
        (!range_is_exact || !IsHullEqualToUnion(mask->region, output_range))));
  if (use_mask_array && !mask->mask_array.valid()) {
    CreateMaskArrayFromRegion(output_box, mask, layout_order);
  }
  Hull(mask->region, output_range, mask->region);

  if (use_mask_array) {
    // Cannot fail, because `input_to_output` must have already been validated.
    StridedLayoutView<dynamic_rank, offset_origin> mask_layout(
        output_box, mask->mask_array.byte_strides());
    auto mask_iterable =
        GetTransformedArrayNDIterable(
            ArrayView<Shared<bool>, dynamic_rank, offset_origin>(
                AddByteOffset(SharedElementPointer<bool>(
                                  UnownedToShared(mask->mask_array.data())),
                              -IndexInnerProduct(output_box.origin(),
                                                 mask_layout.byte_strides())),
                mask_layout),
            input_to_output, arena)
            .value();
    SetMaskAndCountChanged set_mask_context;
    constexpr ElementwiseFunction<1> set_mask_func =
        internal::SimpleElementwiseFunction<SetMaskAndCountChanged(bool)>();

    auto status = internal::IterateOverNDIterables<1, /*Update=*/true>(
        input_to_output.input_shape(), skip_repeated_elements,
        {{mask_iterable.get()}}, arena, {&set_mask_func, &set_mask_context});
    mask->num_masked_elements += set_mask_context.num_changed;
    status.IgnoreError();
    assert(status.ok());
    // We could call RemoveMaskArrayIfNotNeeded here.  However, that would
    // introduce the potential to repeatedly allocate and free the mask array
    // under certain write patterns.  Therefore, we don't remove the mask array
    // once it has been allocated.
  } else {
    // No mask array is needed because all elements within the region are
    // masked.
    mask->num_masked_elements = mask->region.num_elements();
  }
}

}  // namespace internal
}  // namespace tensorstore
