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
#include <new>
#include <type_traits>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/container/fixed_array.h"
#include "absl/status/status.h"
#include "tensorstore/array.h"
#include "tensorstore/box.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/transformed_array.h"
#include "tensorstore/internal/arena.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/internal/integer_overflow.h"
#include "tensorstore/internal/memory.h"
#include "tensorstore/internal/nditerable.h"
#include "tensorstore/internal/nditerable_transformed_array.h"
#include "tensorstore/internal/nditerable_util.h"
#include "tensorstore/internal/unowned_to_shared.h"
#include "tensorstore/rank.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/byte_strided_pointer.h"
#include "tensorstore/util/element_pointer.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal {
namespace {

/// Unary function that sets every element to `true`.
struct SetMask {
  void operator()(bool* x, absl::Status*) const { *x = true; }
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

Index GetRelativeOffset(span<const Index> base, span<const Index> position,
                        span<const Index> strides) {
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
    mask->mask_array.reset();
  }
}
}  // namespace

MaskData::MaskData(DimensionIndex rank) : region(rank) {
  region.Fill(IndexInterval::UncheckedSized(0, 0));
}

std::unique_ptr<bool[], FreeDeleter> CreateMaskArray(
    BoxView<> box, BoxView<> mask_region, span<const Index> byte_strides) {
  std::unique_ptr<bool[], FreeDeleter> result(
      static_cast<bool*>(std::calloc(box.num_elements(), sizeof(bool))));
  ByteStridedPointer<bool> start = result.get();
  start += GetRelativeOffset(box.origin(), mask_region.origin(), byte_strides);
  internal::IterateOverArrays(
      internal::SimpleElementwiseFunction<SetMask(bool), absl::Status*>{},
      /*status=*/nullptr,
      /*constraints=*/skip_repeated_elements,
      ArrayView<bool>(start.get(),
                      StridedLayoutView<>(mask_region.shape(), byte_strides)));
  return result;
}

void CreateMaskArrayFromRegion(BoxView<> box, MaskData* mask,
                               span<const Index> byte_strides) {
  assert(mask->num_masked_elements == mask->region.num_elements());
  mask->mask_array = CreateMaskArray(box, mask->region, byte_strides);
}

void UnionMasks(BoxView<> box, MaskData* mask_a, MaskData* mask_b) {
  assert(mask_a != mask_b);  // May work but not supported.
  if (mask_a->num_masked_elements == 0) {
    std::swap(*mask_a, *mask_b);
    return;
  } else if (mask_b->num_masked_elements == 0) {
    return;
  }
  const DimensionIndex rank = box.rank();
  assert(mask_a->region.rank() == rank);
  assert(mask_b->region.rank() == rank);

  if (mask_a->mask_array && mask_b->mask_array) {
    const Index size = box.num_elements();
    mask_a->num_masked_elements = 0;
    for (Index i = 0; i < size; ++i) {
      if ((mask_a->mask_array[i] |= mask_b->mask_array[i])) {
        ++mask_a->num_masked_elements;
      }
    }
    Hull(mask_a->region, mask_b->region, mask_a->region);
    RemoveMaskArrayIfNotNeeded(mask_a);
    return;
  }

  if (!mask_a->mask_array && !mask_b->mask_array) {
    if (IsHullEqualToUnion(mask_a->region, mask_b->region)) {
      // The combined mask can be specified by the region alone.
      Hull(mask_a->region, mask_b->region, mask_a->region);
      mask_a->num_masked_elements = mask_a->region.num_elements();
      return;
    }
  } else if (!mask_a->mask_array) {
    std::swap(*mask_a, *mask_b);
  }

  absl::FixedArray<Index, kNumInlinedDims> byte_strides(rank);
  ComputeStrides(ContiguousLayoutOrder::c, sizeof(bool), box.shape(),
                 byte_strides);
  if (!mask_a->mask_array) {
    CreateMaskArrayFromRegion(box, mask_a, byte_strides);
  }

  // Copy in mask_b.
  ByteStridedPointer<bool> start = mask_a->mask_array.get();
  start +=
      GetRelativeOffset(box.origin(), mask_b->region.origin(), byte_strides);
  IterateOverArrays(
      [&](bool* ptr) {
        if (!*ptr) ++mask_a->num_masked_elements;
        *ptr = true;
      },
      /*constraints=*/{},
      ArrayView<bool>(start.get(), StridedLayoutView<>(mask_b->region.shape(),
                                                       byte_strides)));
  Hull(mask_a->region, mask_b->region, mask_a->region);
  RemoveMaskArrayIfNotNeeded(mask_a);
}

void RebaseMaskedArray(BoxView<> box, ArrayView<const void> source,
                       ElementPointer<void> dest_ptr, const MaskData& mask) {
  assert(source.dtype() == dest_ptr.dtype());
  assert(internal::RangesEqual(box.shape(), source.shape()));
  const Index num_elements = box.num_elements();
  if (mask.num_masked_elements == num_elements) return;
  DataType r = source.dtype();
  absl::FixedArray<Index, kNumInlinedDims> dest_byte_strides(box.rank());
  ComputeStrides(ContiguousLayoutOrder::c, r->size, box.shape(),
                 dest_byte_strides);
  ArrayView<void> dest_array(
      dest_ptr, StridedLayoutView<>(box.shape(), dest_byte_strides));
  if (mask.num_masked_elements == 0) {
    [[maybe_unused]] const auto iterate_result = internal::IterateOverArrays(
        {&r->copy_assign,
         /*context=*/nullptr},
        /*status=*/nullptr, skip_repeated_elements, source, dest_array);
    assert(iterate_result.success);
    return;
  }
  absl::FixedArray<Index, kNumInlinedDims> mask_byte_strides(box.rank());
  ComputeStrides(ContiguousLayoutOrder::c, sizeof(bool), box.shape(),
                 mask_byte_strides);
  std::unique_ptr<bool[], FreeDeleter> mask_owner;
  bool* mask_array_ptr;
  if (!mask.mask_array) {
    mask_owner = CreateMaskArray(box, mask.region, mask_byte_strides);
    mask_array_ptr = mask_owner.get();
  } else {
    mask_array_ptr = mask.mask_array.get();
  }
  ArrayView<const bool> mask_array(
      mask_array_ptr, StridedLayoutView<>(box.shape(), mask_byte_strides));
  [[maybe_unused]] const auto iterate_result = internal::IterateOverArrays(
      {&r->copy_assign_unmasked, /*context=*/nullptr}, /*status=*/nullptr,
      skip_repeated_elements, source, dest_array, mask_array);
  assert(iterate_result.success);
}

bool WriteToMask(MaskData* mask, BoxView<> output_box,
                 IndexTransformView<> input_to_output,
                 NDIterable::IterationLayoutView layout,
                 span<const Index> write_end_position, Arena* arena) {
  assert(write_end_position.size() == layout.iteration_rank());
  assert(layout.full_rank() == input_to_output.input_rank());
  assert(layout.iteration_rank() > 0);

  if (std::all_of(write_end_position.begin(), write_end_position.end(),
                  [](Index x) { return x == 0; })) {
    // No elements were written, therefore the mask does not need to be updated.
    return false;
  }

  const bool partial_copy =
      write_end_position.front() != layout.iteration_shape.front();
  assert(partial_copy ||
         std::all_of(write_end_position.begin() + 1, write_end_position.end(),
                     [](Index x) { return x == 0; }));

  const DimensionIndex output_rank = output_box.rank();
  Box<dynamic_rank(kNumInlinedDims)> output_range(output_rank);
  // Supplied `input_to_output` transform must have already been validated.
  const bool range_is_exact =
      GetOutputRange(input_to_output, output_range).value();
  Intersect(output_range, output_box, output_range);

  absl::FixedArray<Index, kNumInlinedDims> mask_byte_strides(output_rank);
  ComputeStrides(ContiguousLayoutOrder::c, sizeof(bool), output_box.shape(),
                 mask_byte_strides);
  StridedLayoutView<dynamic_rank, offset_origin> mask_layout(output_box,
                                                             mask_byte_strides);

  const bool use_mask_array =
      output_box.rank() != 0 &&
      mask->num_masked_elements != output_box.num_elements() &&
      (static_cast<bool>(mask->mask_array) || partial_copy ||
       (!Contains(mask->region, output_range) &&
        (!range_is_exact || !IsHullEqualToUnion(mask->region, output_range))));
  if (use_mask_array && !mask->mask_array) {
    CreateMaskArrayFromRegion(output_box, mask, mask_byte_strides);
  }
  Hull(mask->region, output_range, mask->region);

  if (use_mask_array) {
    // Cannot fail, because `input_to_output` must have already been validated.
    auto mask_iterable =
        GetTransformedArrayNDIterable(
            ArrayView<Shared<bool>, dynamic_rank, offset_origin>(
                AddByteOffset(SharedElementPointer<bool>(
                                  UnownedToShared(mask->mask_array.get())),
                              -IndexInnerProduct(output_box.origin(),
                                                 span(mask_byte_strides))),
                mask_layout),
            input_to_output, arena)
            .value();
    const auto mask_buffer_kind =
        mask_iterable->GetIterationBufferConstraint(layout).min_buffer_kind;
    const Index mask_block_size =
        GetNDIterationBlockSize(*mask_iterable, layout, mask_buffer_kind);
    auto mask_iterator = mask_iterable->GetIterator(
        {{layout, mask_block_size}, mask_buffer_kind});

    NDIterationPositionStepper stepper(layout.iteration_shape, mask_block_size);

    absl::Status mask_copy_status;

    SetMaskAndCountChanged set_mask_context;
    constexpr ElementwiseFunction<1> set_mask_func =
        internal::SimpleElementwiseFunction<SetMaskAndCountChanged(bool)>();

    // Update the mask using a backward iteration order, starting from the
    // `write_end_position`.  This ensures that we update the mask at exactly
    // the positions that were written.
    std::copy(write_end_position.begin(), write_end_position.end(),
              stepper.position().begin());
    while (Index block_size = stepper.StepBackward()) {
      IterationBufferPointer mask_pointer;
      const bool get_block_ok = mask_iterator->GetBlock(
          stepper.position(), block_size, &mask_pointer, &mask_copy_status);
      static_cast<void>(get_block_ok);
      assert(get_block_ok);
      set_mask_func[mask_buffer_kind](&set_mask_context, block_size,
                                      mask_pointer);
    }
    mask->num_masked_elements += set_mask_context.num_changed;
    // We could call RemoveMaskArrayIfNotNeeded here.  However, that would
    // introduce the potential to repeatedly allocate and free the mask array
    // under certain write patterns.  Therefore, we don't remove the mask array
    // once it has been allocated.
  } else {
    // No mask array is needed because all elements within the region are
    // masked.
    mask->num_masked_elements = mask->region.num_elements();
  }
  return true;
}

}  // namespace internal
}  // namespace tensorstore
