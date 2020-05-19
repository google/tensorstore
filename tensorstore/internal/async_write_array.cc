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

AsyncWriteArray::Spec::Spec(SharedArray<const void> fill_value)
    : fill_value(std::move(fill_value)) {
  c_order_byte_strides.resize(this->rank());
  tensorstore::ComputeStrides(c_order, this->data_type()->size, this->shape(),
                              c_order_byte_strides);
}

AsyncWriteArray::AsyncWriteArray(DimensionIndex rank)
    : write_mask(rank), write_mask_prior_to_writeback(rank) {}

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
  std::size_t total = 0;
  const Index num_elements = ProductOfExtents(spec.shape());
  if (write_data) {
    total += num_elements * spec.fill_value.data_type()->size;
  }
  if (write_data_prior_to_writeback) {
    total += num_elements * spec.fill_value.data_type()->size;
  }
  if (write_mask.mask_array) {
    total += num_elements * sizeof(bool);
  }
  if (write_mask_prior_to_writeback.mask_array) {
    total += num_elements * sizeof(bool);
  }
  return total;
}
void AsyncWriteArray::AfterWritebackCompletes(const Spec& spec,
                                              span<const Index> origin,
                                              bool success) {
  if (success) {
    // Writeback was successful.  Reset the prior mask.
    write_mask_prior_to_writeback.Reset();
    if (write_data_prior_to_writeback) {
      read_array = SharedArrayView<const void>(
          SharedElementPointer<const void>(
              std::exchange(write_data_prior_to_writeback, nullptr),
              spec.data_type()),
          spec.write_layout());
    } else {
      read_array = nullptr;
    }
  } else {
    // Writeback failed or was aborted.  Combine the current mask into the
    // prior mask.
    UnionMasks(BoxView(origin, spec.shape()), &write_mask,
               &write_mask_prior_to_writeback);
    write_mask_prior_to_writeback.Reset();
    if (!write_data) {
      write_data = std::move(write_data_prior_to_writeback);
    } else {
      write_data_prior_to_writeback = nullptr;
    }
  }
}

void AsyncWriteArray::WriteFillValue(const Spec& spec,
                                     span<const Index> origin) {
  write_data = nullptr;
  write_mask.Reset();
  write_mask.num_masked_elements = spec.num_elements();
  write_mask.region = BoxView(origin, spec.shape());
}

AsyncWriteArray::WritebackData AsyncWriteArray::GetArrayForWriteback(
    const Spec& spec, span<const Index> origin) {
  assert(origin.size() == spec.rank());
  WritebackData writeback;
  AfterWritebackCompletes(spec, origin, /*success=*/false);
  writeback.unconditional = IsFullyOverwritten(spec);
  if (!write_data) {
    // No data has been allocated for the write array.  This is only possible in
    // two cases:
    if (writeback.unconditional) {
      // Case 1: array was fully overwritten by the fill value using
      // `WriteFillValue`.
      writeback.array = spec.fill_value;
    } else {
      // Case 2: array is unmodified.
      assert(IsUnmodified());
      writeback.array = GetReadArray(spec);
    }
  } else {
    ElementPointer<void> writeback_element_pointer(write_data,
                                                   spec.data_type());
    writeback.array = SharedArrayView<void>(
        UnownedToShared(writeback_element_pointer), spec.write_layout());
    if (!writeback.unconditional) {
      // Array was only partially written.
      RebaseMaskedArray(BoxView<>(origin, spec.shape()), GetReadArray(spec),
                        writeback_element_pointer, write_mask);
    }
  }
  writeback.equals_fill_value = (writeback.array == spec.fill_value);

  if (writeback.equals_fill_value && writeback.unconditional) {
    write_data = nullptr;
    writeback.array = spec.fill_value;
  }

  // Save the current `write_mask` in `write_mask_prior_to_writeback`, and
  // reset `write_mask` so that new writes are recorded separately.  If
  // writeback fails, `write_mask_prior_to_writeback` will be combined into
  // `write_mask`.  If writeback succeeds, it will be discarded.
  std::swap(write_mask, write_mask_prior_to_writeback);
  write_mask.Reset();

  write_data_prior_to_writeback = std::exchange(write_data, nullptr);

  return writeback;
}

Result<NDIterable::Ptr> AsyncWriteArray::BeginWrite(
    const Spec& spec, span<const Index> origin,
    IndexTransform<> chunk_transform, Arena* arena) {
  bool allocated_data = false;
  if (!write_data) {
    write_data = AllocateAndConstructSharedElements(
                     spec.num_elements(), default_init, spec.data_type())
                     .pointer();
    allocated_data = true;
  }
  ArrayView<void> write_array(
      ElementPointer<void>(write_data, spec.data_type()), spec.write_layout());
  if (allocated_data) {
    if (write_data_prior_to_writeback) {
      CopyArray(
          ArrayView<void>(ElementPointer<void>(write_data_prior_to_writeback,
                                               spec.data_type()),
                          spec.write_layout()),
          write_array);
    } else if (IsFullyOverwritten(spec)) {
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

bool AsyncWriteArray::EndWrite(const Spec& spec, span<const Index> origin,
                               IndexTransformView<> chunk_transform,
                               NDIterable::IterationLayoutView layout,
                               span<const Index> write_end_position,
                               Arena* arena) {
  return WriteToMask(&write_mask, BoxView<>(origin, spec.shape()),
                     chunk_transform, layout, write_end_position, arena);
}

Result<NDIterable::Ptr> AsyncWriteArray::GetReadNDIterable(
    const Spec& spec, span<const Index> origin,
    IndexTransform<> chunk_transform, Arena* arena) {
  auto array = GetReadArray(spec);
  StridedLayoutView<dynamic_rank, offset_origin> data_layout(
      origin, spec.shape(), array.byte_strides());
  TENSORSTORE_ASSIGN_OR_RETURN(
      chunk_transform,
      ComposeLayoutAndTransform(data_layout, std::move(chunk_transform)));
  return GetNormalizedTransformedArrayNDIterable(
      {AddByteOffset(std::move(array.element_pointer()),
                     -data_layout.origin_byte_offset()),
       std::move(chunk_transform)},
      arena);
}

}  // namespace internal
}  // namespace tensorstore
