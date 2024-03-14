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

#include <stddef.h>

#include <cassert>
#include <memory>
#include <utility>
#include <vector>

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
#include "tensorstore/internal/masked_array.h"
#include "tensorstore/internal/nditerable.h"
#include "tensorstore/internal/nditerable_transformed_array.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/rank.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/element_pointer.h"
#include "tensorstore/util/extents.h"
#include "tensorstore/util/iterate.h"
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
  tensorstore::ComputeStrides(c_order, this->dtype()->size, this->shape(),
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

std::shared_ptr<void> AsyncWriteArray::Spec::AllocateAndConstructBuffer()
    const {
  return AllocateAndConstructSharedElements(this->num_elements(), default_init,
                                            this->dtype())
      .pointer();
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
  return GetTransformedArrayNDIterable(
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

  const auto must_store = [&](ArrayView<const void> array) {
    if (spec.store_if_equal_to_fill_value) return true;
    return !AreArraysEqual(array, spec.fill_value,
                           spec.fill_value_comparison_kind);
  };

  const auto get_writeback_from_array = [&] {
    WritebackData writeback;
    writeback.array = SharedArrayView<void>(
        SharedElementPointer<void>(data, spec.dtype()), spec.write_layout());
    writeback.must_store = must_store(writeback.array);
    if (!writeback.must_store) {
      data = nullptr;
      writeback.array = spec.fill_value;
    }
    return writeback;
  };

  if (!data) {
    // No data has been allocated for the write array.  There are 3 possible
    // cases:

    // Case 1: array was fully overwritten by the fill value using
    // `WriteFillValue`.
    if (IsFullyOverwritten(spec, origin)) {
      WritebackData writeback;
      writeback.array = spec.fill_value;
      writeback.must_store = false;
      return writeback;
    }

    // Case 2: array is unmodified.
    if (IsUnmodified()) {
      WritebackData writeback;
      writeback.must_store = read_array.valid() && must_store(read_array);
      if (writeback.must_store) {
        writeback.array = read_array;
      } else {
        writeback.array = spec.fill_value;
      }
      return writeback;
    }

    // Case 3: Array was only partially written, but a previous call to
    // `GetArrayForWriteback` determined that all values that were written, and
    // all values in `read_array` that were unmodified, were equal the fill

    // Case 3a: New `read_array` is specified.  It is possible that in the new
    // `read_array`, some of elements at unmodified positions are no longer
    // equal to the fill value.
    if (!read_state_already_integrated && read_array.valid()) {
      data = tensorstore::MakeCopy(spec.fill_value,
                                   {c_order, include_repeated_elements})
                 .pointer();
      RebaseMaskedArray(BoxView<>(origin, spec.shape()),
                        ArrayView<const void>(read_array), {data, spec.dtype()},
                        mask);
      return get_writeback_from_array();
    }

    // Case 3b: No new read array since the previous call to
    // `GetArrayForWriteback`.  The writeback array must, therefore, still be
    // equal to the fill value.
    WritebackData writeback;
    writeback.array = spec.fill_value;
    writeback.must_store = false;
    return writeback;
  }

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
    EnsureWritable(spec);
    // Array was only partially written.
    RebaseMaskedArray(BoxView<>(origin, spec.shape()),
                      read_array.data()
                          ? ArrayView<const void>(read_array)
                          : ArrayView<const void>(spec.fill_value),
                      {data, spec.dtype()}, mask);
  }
  return get_writeback_from_array();
}

size_t AsyncWriteArray::MaskedArray::EstimateSizeInBytes(
    const Spec& spec) const {
  size_t total = 0;
  const Index num_elements = ProductOfExtents(spec.shape());
  if (data) {
    total += num_elements * spec.fill_value.dtype()->size;
  }
  if (mask.mask_array) {
    total += num_elements * sizeof(bool);
  }
  return total;
}

void AsyncWriteArray::MaskedArray::EnsureWritable(const Spec& spec) {
  assert(data);
  auto dtype = spec.dtype();
  auto new_data = spec.AllocateAndConstructBuffer();
  dtype->copy_assign[IterationBufferKind::kContiguous](
      /*context=*/nullptr, {1, spec.num_elements()},
      IterationBufferPointer(data.get(), 0, dtype.size()),
      IterationBufferPointer(new_data.get(), 0, dtype.size()),
      /*status=*/nullptr);
  data = std::move(new_data);
}

Result<NDIterable::Ptr> AsyncWriteArray::MaskedArray::BeginWrite(
    const Spec& spec, span<const Index> origin,
    IndexTransform<> chunk_transform, Arena* arena) {
  bool allocated_data = false;
  if (!data) {
    data = spec.AllocateAndConstructBuffer();
    allocated_data = true;
  }
  ArrayView<void> write_array(ElementPointer<void>(data, spec.dtype()),
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

  return GetTransformedArrayNDIterable(
      {UnownedToShared(AddByteOffset(write_array.element_pointer(),
                                     -data_layout.origin_byte_offset())),
       std::move(chunk_transform)},
      arena);
}

void AsyncWriteArray::MaskedArray::EndWrite(
    const Spec& spec, span<const Index> origin,
    IndexTransformView<> chunk_transform, Arena* arena) {
  WriteToMask(&mask, BoxView<>(origin, spec.shape()), chunk_transform, arena);
}

void AsyncWriteArray::MaskedArray::Clear() {
  mask.Reset();
  data = nullptr;
}

AsyncWriteArray::AsyncWriteArray(DimensionIndex rank) : write_state(rank) {}

AsyncWriteArray::WritebackData AsyncWriteArray::GetArrayForWriteback(
    const Spec& spec, span<const Index> origin,
    const SharedArrayView<const void>& read_array,
    const StorageGeneration& read_generation) {
  auto writeback_data = write_state.GetArrayForWriteback(
      spec, origin, read_array, read_generation == this->read_generation);
  if (write_state.data) this->read_generation = read_generation;
  return writeback_data;
}

Result<NDIterable::Ptr> AsyncWriteArray::GetReadNDIterable(
    const Spec& spec, span<const Index> origin,
    SharedArrayView<const void> read_array,
    const StorageGeneration& read_generation, IndexTransform<> chunk_transform,
    Arena* arena) {
  if (!read_array.valid()) read_array = spec.fill_value;
  if (!write_state.IsUnmodified()) {
    if (write_state.IsFullyOverwritten(spec, origin)) {
      if (!write_state.data) {
        // Fully overwritten with fill value.
        read_array = spec.fill_value;
      }
    } else if (this->read_generation != read_generation) {
      assert(write_state.data);
      RebaseMaskedArray(BoxView<>(origin, spec.shape()), read_array,
                        ElementPointer<void>(write_state.data, spec.dtype()),
                        write_state.mask);
      this->read_generation = read_generation;
    }
    if (write_state.data) {
      read_array = SharedArrayView<const void>(
          SharedElementPointer<const void>(write_state.data, spec.dtype()),
          spec.write_layout());
    }
  }
  return spec.GetReadNDIterable(std::move(read_array), origin,
                                std::move(chunk_transform), arena);
}

Result<NDIterable::Ptr> AsyncWriteArray::BeginWrite(
    const Spec& spec, span<const Index> origin,
    IndexTransform<> chunk_transform, Arena* arena) {
  return write_state.BeginWrite(spec, origin, std::move(chunk_transform),
                                arena);
}

void AsyncWriteArray::EndWrite(const Spec& spec, span<const Index> origin,
                               IndexTransformView<> chunk_transform,
                               bool success, Arena* arena) {
  if (!success) {
    InvalidateReadState();
    return;
  }
  write_state.EndWrite(spec, origin, chunk_transform, arena);
}

}  // namespace internal
}  // namespace tensorstore
