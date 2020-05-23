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

#include "tensorstore/internal/async_write_array.h"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "tensorstore/array.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/masked_array.h"
#include "tensorstore/internal/nditerable.h"
#include "tensorstore/internal/nditerable_transformed_array.h"
#include "tensorstore/internal/unowned_to_shared.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/element_pointer.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal {

AsyncWriteArray::Spec::Spec(SharedArray<const void> fill_value,
                            Box<> component_bounds)
    : fill_value(std::move(fill_value)),
      component_bounds(std::move(component_bounds)) {
  assert(this->fill_value.rank() == this->component_bounds.rank());
  c_order_byte_strides.resize(this->rank());
  tensorstore::ComputeStrides(c_order, this->data_type()->size, this->shape(),
                              c_order_byte_strides);
}

Index AsyncWriteArray::Spec::chunk_num_elements(
    span<const Index> origin) const {
  assert(origin.size() == this->rank());
  Index product = 1;
  for (DimensionIndex i = 0; i < origin.size(); ++i) {
    product *= Intersect(IndexInterval::UncheckedSized(origin[i],
                                                       fill_value.shape()[i]),
                         component_bounds[i])
                   .size();
  }
  return product;
}

Result<NDIterable::Ptr> AsyncWriteArray::Spec::GetReadNDIterable(
    SharedArrayView<const void> array, span<const Index> origin,
    IndexTransform<> chunk_transform, Arena* arena) const {
  if (!array.valid()) array = fill_value;
  assert(internal::RangesEqual(array.shape(), this->shape()));
  StridedLayoutView<dynamic_rank, offset_origin> data_layout(
      origin, shape(), array.byte_strides());
  TENSORSTORE_ASSIGN_OR_RETURN(
      chunk_transform,
      ComposeLayoutAndTransform(data_layout, std::move(chunk_transform)));
  return GetNormalizedTransformedArrayNDIterable(
      {AddByteOffset(std::move(array.element_pointer()),
                     -data_layout.origin_byte_offset()),
       std::move(chunk_transform)},
      arena);
}

AsyncWriteArray::MaskedArray::MaskedArray(DimensionIndex rank) : mask(rank) {}

void AsyncWriteArray::MaskedArray::WriteFillValue(const Spec& spec,
                                                  span<const Index> origin) {
  data = nullptr;
  mask.Reset();
  mask.num_masked_elements = spec.num_elements();
  mask.region = BoxView(origin, spec.shape());
}

AsyncWriteArray::WritebackData
AsyncWriteArray::MaskedArray::GetArrayForWriteback(
    const Spec& spec, span<const Index> origin,
    const SharedArrayView<const void>& read_array,
    bool read_state_already_integrated) {
  assert(origin.size() == spec.rank());
  WritebackData writeback;
  writeback.unconditional = IsFullyOverwritten(spec, origin);
  if (!data) {
    // No data has been allocated for the write array.  This is only possible in
    // two cases:
    if (writeback.unconditional) {
      // Case 1: array was fully overwritten by the fill value using
      // `WriteFillValue`.
      writeback.array = spec.fill_value;
    } else {
      // Case 2: array is unmodified.
      assert(IsUnmodified());
      if (read_array.data()) {
        writeback.array = read_array;
      } else {
        writeback.array = spec.fill_value;
      }
    }
  } else {
    ElementPointer<void> writeback_element_pointer(data, spec.data_type());
    writeback.array = SharedArrayView<void>(
        UnownedToShared(writeback_element_pointer), spec.write_layout());
    if (!read_state_already_integrated &&
        // If any elements in the array haven't been written, fill them from
        // `read_array` or `spec.fill_value`.  Compare
        // `mask.num_masked_elements` to `spec.num_elements()` rather than
        // `spec.chunk_num_elements(origin)`, because even if the only positions
        // not written are outside `spec.component_bounds`, we still need to
        // ensure we don't leak the contents of uninitialized memory, in case
        // the consumer of the `WritebackData` stores out-of-bounds data as
        // well.
        mask.num_masked_elements != spec.num_elements()) {
      // Array was only partially written.
      RebaseMaskedArray(BoxView<>(origin, spec.shape()),
                        read_array.data()
                            ? ArrayView<const void>(read_array)
                            : ArrayView<const void>(spec.fill_value),
                        writeback_element_pointer, mask);
    }
  }
  writeback.equals_fill_value = (writeback.array == spec.fill_value);

  if (writeback.equals_fill_value && writeback.unconditional) {
    data = nullptr;
    writeback.array = spec.fill_value;
  }
  return writeback;
}

std::size_t AsyncWriteArray::MaskedArray::EstimateSizeInBytes(
    const Spec& spec) const {
  std::size_t total = 0;
  const Index num_elements = ProductOfExtents(spec.shape());
  if (data) {
    total += num_elements * spec.fill_value.data_type()->size;
  }
  if (mask.mask_array) {
    total += num_elements * sizeof(bool);
  }
  return total;
}

Result<NDIterable::Ptr> AsyncWriteArray::MaskedArray::BeginWrite(
    const Spec& spec, span<const Index> origin,
    IndexTransform<> chunk_transform, Arena* arena) {
  bool allocated_data = false;
  if (!data) {
    data = AllocateAndConstructSharedElements(spec.num_elements(), default_init,
                                              spec.data_type())
               .pointer();
    allocated_data = true;
  }
  ArrayView<void> write_array(ElementPointer<void>(data, spec.data_type()),
                              spec.write_layout());
  if (allocated_data) {
    if (IsFullyOverwritten(spec, origin)) {
      // Previously, there was no data array allocated for the array but it was
      // considered to have been implicitly overwritten with the fill value.
      // Now that the data array has been allocated, it must actually be
      // initialized with the fill value.
      CopyArray(spec.fill_value, write_array);
    } else {
      assert(IsUnmodified());
    }
  }

  StridedLayoutView<dynamic_rank, offset_origin> data_layout{
      origin, spec.shape(), spec.c_order_byte_strides};
  TENSORSTORE_ASSIGN_OR_RETURN(
      chunk_transform,
      ComposeLayoutAndTransform(data_layout, std::move(chunk_transform)));

  return GetNormalizedTransformedArrayNDIterable(
      {UnownedToShared(AddByteOffset(write_array.element_pointer(),
                                     -data_layout.origin_byte_offset())),
       std::move(chunk_transform)},
      arena);
}

bool AsyncWriteArray::MaskedArray::EndWrite(
    const Spec& spec, span<const Index> origin,
    IndexTransformView<> chunk_transform,
    NDIterable::IterationLayoutView layout,
    span<const Index> write_end_position, Arena* arena) {
  return WriteToMask(&mask, BoxView<>(origin, spec.shape()), chunk_transform,
                     layout, write_end_position, arena);
}

void AsyncWriteArray::MaskedArray::Clear() {
  mask.Reset();
  data = nullptr;
}

void AsyncWriteArray::MaskedArray::RebaseOnto(
    const Spec& spec, span<const Index> origin,
    MaskedArray&& prior_masked_array) {
  if (data) {
    if (prior_masked_array.data) {
      RebaseMaskedArray(
          BoxView(origin, spec.shape()),
          ArrayView<const void>(ElementPointer<const void>(
                                    prior_masked_array.data, spec.data_type()),
                                spec.write_layout()),
          ElementPointer<void>(data, spec.data_type()), mask);
    } else if (prior_masked_array.IsFullyOverwritten(spec, origin)) {
      // `prior_masked_array` implicitly contains the fill value.
      RebaseMaskedArray(BoxView(origin, spec.shape()), spec.fill_value,
                        ElementPointer<void>(data, spec.data_type()), mask);
    } else {
      assert(prior_masked_array.IsUnmodified());
    }
    UnionMasks(BoxView(origin, spec.shape()), &mask, &prior_masked_array.mask);
  } else if (IsUnmodified()) {
    data = std::move(prior_masked_array.data);
    mask = std::move(prior_masked_array.mask);
  } else {
    assert(IsFullyOverwritten(spec, origin));
    // Implicitly contains the fill value.
  }
  prior_masked_array.Clear();
}

AsyncWriteArray::AsyncWriteArray(DimensionIndex rank)
    : write_state(rank), writeback_state(rank) {}

std::size_t AsyncWriteArray::EstimateReadStateSizeInBytes(
    const Spec& spec) const {
  std::size_t total = 0;
  const Index num_elements = ProductOfExtents(spec.shape());
  if (read_array.data()) {
    total += num_elements * spec.fill_value.data_type()->size;
  }
  return total;
}

std::size_t AsyncWriteArray::EstimateWriteStateSizeInBytes(
    const Spec& spec) const {
  return write_state.EstimateSizeInBytes(spec) +
         writeback_state.EstimateSizeInBytes(spec);
}

void AsyncWriteArray::AfterWritebackCompletes(const Spec& spec,
                                              span<const Index> origin,
                                              bool success) {
  if (success) {
    // Writeback was successful.  Reset the prior mask.
    writeback_state.mask.Reset();
    if (writeback_state.data) {
      read_array = SharedArrayView<const void>(
          SharedElementPointer<const void>(
              std::exchange(writeback_state.data, nullptr), spec.data_type()),
          spec.write_layout());
    } else {
      read_array = nullptr;
    }
  } else {
    // Writeback failed or was aborted.  Combine the current mask into the
    // prior mask.
    write_state.RebaseOnto(spec, origin, std::move(writeback_state));
  }
}

AsyncWriteArray::WritebackData AsyncWriteArray::GetArrayForWriteback(
    const Spec& spec, span<const Index> origin) {
  AfterWritebackCompletes(spec, origin, /*success=*/false);
  auto writeback = write_state.GetArrayForWriteback(spec, origin, read_array);

  // Save the current `write_state` in `writeback_state` and clear `write_state`
  // so that new writes are recorded separately.  If writeback fails,
  // `write_state` will be rebased onto `writeback_state`.  If writeback
  // succeeds, `writeback_state` will be discarded.
  writeback_state = std::move(write_state);
  write_state.Clear();

  return writeback;
}

Result<NDIterable::Ptr> AsyncWriteArray::GetReadNDIterable(
    const Spec& spec, span<const Index> origin,
    IndexTransform<> chunk_transform, Arena* arena) {
  return spec.GetReadNDIterable(GetReadArray(spec), origin,
                                std::move(chunk_transform), arena);
}

}  // namespace internal
}  // namespace tensorstore
