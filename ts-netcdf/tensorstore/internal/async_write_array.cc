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

#include <algorithm>
#include <cassert>
#include <utility>

#include "absl/base/optimization.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "tensorstore/array.h"
#include "tensorstore/box.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/output_index_method.h"
#include "tensorstore/index_space/transformed_array.h"
#include "tensorstore/internal/arena.h"
#include "tensorstore/internal/integer_overflow.h"
#include "tensorstore/internal/masked_array.h"
#include "tensorstore/internal/memory.h"
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
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal {

Index AsyncWriteArray::Spec::GetNumInBoundsElements(BoxView<> domain) const {
  const DimensionIndex rank = this->rank();
  assert(domain.rank() == rank);
  Index product = 1;
  const BoxView<> bounds = this->valid_data_bounds;
  for (DimensionIndex i = 0; i < rank; ++i) {
    product *= Intersect(bounds[i], domain[i]).size();
  }
  return product;
}

SharedArrayView<const void> AsyncWriteArray::Spec::GetFillValueForDomain(
    BoxView<> domain) const {
  const DimensionIndex rank = domain.rank();
  assert(Contains(overall_fill_value.domain(), domain));
  return SharedArrayView<const void>(
      AddByteOffset(
          overall_fill_value.element_pointer(),
          IndexInnerProduct(rank, overall_fill_value.byte_strides().data(),
                            domain.origin().data())),
      StridedLayoutView<>(rank, domain.shape().data(),
                          overall_fill_value.byte_strides().data()));
}

Result<NDIterable::Ptr> AsyncWriteArray::Spec::GetReadNDIterable(
    SharedArrayView<const void> array, BoxView<> domain,
    IndexTransform<> chunk_transform, Arena* arena) const {
  if (!array.valid()) array = GetFillValueForDomain(domain);
  assert(internal::RangesEqual(array.shape(), domain.shape()));
  StridedLayoutView<dynamic_rank, offset_origin> data_layout(
      domain, array.byte_strides());
  TENSORSTORE_ASSIGN_OR_RETURN(
      chunk_transform,
      ComposeLayoutAndTransform(data_layout, std::move(chunk_transform)));
  return GetTransformedArrayNDIterable(
      {AddByteOffset(std::move(array.element_pointer()),
                     -data_layout.origin_byte_offset()),
       std::move(chunk_transform)},
      arena);
}

SharedArray<void> AsyncWriteArray::Spec::AllocateArray(
    span<const Index> shape) const {
  return tensorstore::AllocateArray(shape, layout_order(), default_init,
                                    this->dtype());
}

AsyncWriteArray::MaskedArray::MaskedArray(DimensionIndex rank) : mask(rank) {}

void AsyncWriteArray::MaskedArray::WriteFillValue(const Spec& spec,
                                                  BoxView<> domain) {
  array = {};
  mask.Reset();
  mask.num_masked_elements = domain.num_elements();
  mask.region = domain;
}

AsyncWriteArray::WritebackData
AsyncWriteArray::MaskedArray::GetArrayForWriteback(
    const Spec& spec, BoxView<> domain,
    const SharedArrayView<const void>& read_array,
    bool read_state_already_integrated) {
  assert(domain.rank() == spec.rank());

  const auto must_store = [&](ArrayView<const void> array) {
    if (this->store_if_equal_to_fill_value) return true;
    return !AreArraysEqual(array, spec.GetFillValueForDomain(domain),
                           spec.fill_value_comparison_kind);
  };

  const auto get_writeback_from_array = [&] {
    WritebackData writeback;
    writeback.array = array;
    writeback.must_store = must_store(writeback.array);
    if (!writeback.must_store) {
      array = {};
      writeback.array = spec.GetFillValueForDomain(domain);
      writeback.may_retain_reference_to_array_indefinitely = true;
    } else {
      writeback.may_retain_reference_to_array_indefinitely =
          (array_capabilities <= kImmutableAndCanRetainIndefinitely);
    }
    return writeback;
  };

  if (!array.valid()) {
    // No data has been allocated for the write array.  There are 3 possible
    // cases:

    // Case 1: array was fully overwritten by the fill value using
    // `WriteFillValue`.
    if (IsFullyOverwritten(spec, domain)) {
      WritebackData writeback;
      writeback.array = spec.GetFillValueForDomain(domain);
      writeback.must_store = false;
      writeback.may_retain_reference_to_array_indefinitely = true;
      return writeback;
    }

    // Case 2: array is unmodified.
    if (IsUnmodified()) {
      WritebackData writeback;
      writeback.must_store = read_array.valid() && must_store(read_array);
      if (writeback.must_store) {
        writeback.array = read_array;
      } else {
        writeback.array = spec.GetFillValueForDomain(domain);
      }
      writeback.may_retain_reference_to_array_indefinitely = true;
      return writeback;
    }

    // Case 3: Array was only partially written, but a previous call to
    // `GetArrayForWriteback` determined that all values that were written, and
    // all values in `read_array` that were unmodified, were equal the fill
    // value, and therefore `array` did not need to be stored.

    // Case 3a: New `read_array` is specified.  It is possible that in the new
    // `read_array`, some of elements at unmodified positions are no longer
    // equal to the fill value.
    if (!read_state_already_integrated && read_array.valid()) {
      array_capabilities = kMutableArray;
      array = tensorstore::MakeCopy(spec.GetFillValueForDomain(domain),
                                    {c_order, include_repeated_elements});
      RebaseMaskedArray(domain, ArrayView<const void>(read_array), array, mask);
      return get_writeback_from_array();
    }

    // Case 3b: No new read array since the previous call to
    // `GetArrayForWriteback`.  The writeback array must, therefore, still be
    // equal to the fill value.
    WritebackData writeback;
    writeback.array = spec.GetFillValueForDomain(domain);
    writeback.must_store = false;
    writeback.may_retain_reference_to_array_indefinitely = true;
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
      mask.num_masked_elements != domain.num_elements()) {
    EnsureWritable(spec);
    // Array was only partially written.
    RebaseMaskedArray(
        domain,
        read_array.valid()
            ? ArrayView<const void>(read_array)
            : ArrayView<const void>(spec.GetFillValueForDomain(domain)),
        array, mask);
  }
  return get_writeback_from_array();
}

size_t AsyncWriteArray::MaskedArray::EstimateSizeInBytes(
    const Spec& spec, tensorstore::span<const Index> shape) const {
  size_t total = 0;
  if (array.valid()) {
    total += GetByteExtent(array);
  }
  if (mask.mask_array.valid()) {
    const Index num_elements = ProductOfExtents(shape);
    total += num_elements * sizeof(bool);
  }
  return total;
}

void AsyncWriteArray::MaskedArray::EnsureWritable(const Spec& spec) {
  assert(array.valid());
  auto new_array = spec.AllocateArray(array.shape());
  CopyArray(array, new_array);
  array = std::move(new_array);
  array_capabilities = kMutableArray;
}

Result<TransformedSharedArray<void>>
AsyncWriteArray::MaskedArray::GetWritableTransformedArray(
    const Spec& spec, BoxView<> domain, IndexTransform<> chunk_transform) {
  // TODO(jbms): Could avoid copies when the output range of `chunk_transform`
  // is known to fully cover ``domain`.
  if (!array.valid()) {
    this->array = spec.AllocateArray(domain.shape());
    array_capabilities = kMutableArray;
    if (IsFullyOverwritten(spec, domain)) {
      // Previously, there was no data array allocated for the array but it
      // was considered to have been implicitly overwritten with the fill
      // value. Now that the data array has been allocated, it must actually
      // be initialized with the fill value.
      CopyArray(spec.GetFillValueForDomain(domain), this->array);
    } else {
      assert(IsUnmodified());
    }
  } else if (array_capabilities != kMutableArray) {
    EnsureWritable(spec);
  }

  StridedLayoutView<dynamic_rank, offset_origin> data_layout{
      domain, this->array.byte_strides()};
  TENSORSTORE_ASSIGN_OR_RETURN(
      chunk_transform,
      ComposeLayoutAndTransform(data_layout, std::move(chunk_transform)));

  return {std::in_place,
          UnownedToShared(
              AddByteOffset(ElementPointer<void>(this->array.element_pointer()),
                            -data_layout.origin_byte_offset())),
          std::move(chunk_transform)};
}

Result<NDIterable::Ptr> AsyncWriteArray::MaskedArray::BeginWrite(
    const Spec& spec, BoxView<> domain, IndexTransform<> chunk_transform,
    Arena* arena) {
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto transformed_array,
      GetWritableTransformedArray(spec, domain, std::move(chunk_transform)));
  return GetTransformedArrayNDIterable(std::move(transformed_array), arena);
}

void AsyncWriteArray::MaskedArray::EndWrite(
    const Spec& spec, BoxView<> domain, IndexTransformView<> chunk_transform,
    Arena* arena) {
  WriteToMask(&mask, domain, chunk_transform, spec.layout_order(), arena);
}

void AsyncWriteArray::MaskedArray::Clear() {
  mask.Reset();
  array = {};
}

AsyncWriteArray::AsyncWriteArray(DimensionIndex rank) : write_state(rank) {}

AsyncWriteArray::WritebackData AsyncWriteArray::GetArrayForWriteback(
    const Spec& spec, BoxView<> domain,
    const SharedArrayView<const void>& read_array,
    const StorageGeneration& read_generation) {
  auto writeback_data = write_state.GetArrayForWriteback(
      spec, domain, read_array, read_generation == this->read_generation);
  if (write_state.array.valid()) this->read_generation = read_generation;
  return writeback_data;
}

Result<NDIterable::Ptr> AsyncWriteArray::GetReadNDIterable(
    const Spec& spec, BoxView<> domain, SharedArrayView<const void> read_array,
    const StorageGeneration& read_generation, IndexTransform<> chunk_transform,
    Arena* arena) {
  if (!read_array.valid()) read_array = spec.GetFillValueForDomain(domain);
  if (!write_state.IsUnmodified()) {
    if (write_state.IsFullyOverwritten(spec, domain)) {
      if (!write_state.array.valid()) {
        // Fully overwritten with fill value.
        read_array = spec.GetFillValueForDomain(domain);
      }
    } else if (this->read_generation != read_generation) {
      assert(write_state.array.valid());
      if (write_state.array_capabilities != MaskedArray::kMutableArray) {
        write_state.EnsureWritable(spec);
      }
      RebaseMaskedArray(domain, read_array, write_state.array,
                        write_state.mask);
      this->read_generation = read_generation;
    }
    if (write_state.array.valid()) {
      read_array = write_state.array;
    }
  }
  return spec.GetReadNDIterable(std::move(read_array), domain,
                                std::move(chunk_transform), arena);
}

namespace {
// Zero-copies `source_array` into
// `write_state transformed by `chunk_transform`.
//
// Preconditions:
//
// - `source_capabilities != kCannotRetain`
// - output range of `chunk_transform` is exactly the domain of the array.
//
// Returns: `true` on success, `false` if not supported.
bool ZeroCopyToWriteArray(
    const AsyncWriteArray::Spec& spec, BoxView<> domain,
    IndexTransformView<> chunk_transform,
    TransformedSharedArray<const void> source_array,
    AsyncWriteArray::WriteArraySourceCapabilities source_capabilities,
    AsyncWriteArray::MaskedArray& write_state) {
  assert(source_capabilities !=
         AsyncWriteArray::WriteArraySourceCapabilities::kCannotRetain);
  const DimensionIndex dest_rank = domain.rank();
  assert(spec.rank() == dest_rank);
  assert(chunk_transform.output_rank() == dest_rank);
  IndexTransformView<> source_transform = source_array.transform();
  const DimensionIndex input_rank = chunk_transform.input_rank();

  assert(source_transform.input_rank() == input_rank);
  assert(source_transform.domain().box() == chunk_transform.domain().box());

  // source_array(pos) = source_array.data()[pos[i] + ]

  Index new_byte_strides[kMaxRank];

  DimensionIndex dest_dim_for_input_dim[kMaxRank];
  std::fill_n(dest_dim_for_input_dim, input_rank, DimensionIndex(-1));
  std::fill_n(new_byte_strides, dest_rank, Index(0));

  for (DimensionIndex dest_dim = 0; dest_dim < dest_rank; ++dest_dim) {
    if (domain.shape()[dest_dim] == 1) continue;
    auto map = chunk_transform.output_index_map(dest_dim);
    if (map.method() != OutputIndexMethod::single_input_dimension) {
      // Must be a constant dimension map (possibly represented as an array
      // map), since output range has already been confirmed to be exact.
      continue;
    }
    [[maybe_unused]] DimensionIndex prev_dest_dim =
        std::exchange(dest_dim_for_input_dim[map.input_dimension()], dest_dim);
    // It is not possible for more than one input dimension to map to the same
    // dest dim, given that the output range has already been confirmed to be
    // exact.
    assert(prev_dest_dim == -1);
  }

  const DimensionIndex source_output_rank = source_transform.output_rank();

  Index source_offset = 0;

  for (DimensionIndex source_output_dim = 0;
       source_output_dim < source_output_rank; ++source_output_dim) {
    auto map = source_transform.output_index_map(source_output_dim);
    source_offset =
        internal::wrap_on_overflow::Add(source_offset, map.offset());
    switch (map.method()) {
      case OutputIndexMethod::constant:
        break;
      case OutputIndexMethod::single_input_dimension: {
        const DimensionIndex input_dim = map.input_dimension();
        const DimensionIndex dest_dim = dest_dim_for_input_dim[input_dim];
        const Index source_stride = map.stride();
        if (dest_dim == -1) {
          // Singleton dimension that will be ignored in the resultant layout.
          // Consequently, `source_offset` must be adjusted if it has a non-zero
          // origin.
          assert(source_transform.input_shape()[input_dim] == 1);
          const Index source_origin =
              source_transform.input_origin()[input_dim];
          source_offset = internal::wrap_on_overflow::Add(
              source_offset, internal::wrap_on_overflow::Multiply(
                                 source_origin, source_stride));
          break;
        }
        const auto dest_map = chunk_transform.output_index_map(dest_dim);
        const Index dest_stride = dest_map.stride();

        // Consider dest position of `x` and corresponding input position `y`.
        // We have:
        //     x = dest_offset + dest_stride * y
        //     y = (x - dest_offset) * dest_stride
        //       = x * dest_stride - dest_offset * dest_stride
        //
        // The source array byte_stride contribution is:
        //
        //     source_stride * y + source_offset
        //       = x * source_stride * dest_stride
        //       - source_stride * dest_offset * dest_stride

        assert(dest_stride == 1 || dest_stride == -1);
        new_byte_strides[dest_dim] = internal::wrap_on_overflow::Add(
            new_byte_strides[dest_dim],
            internal::wrap_on_overflow::Multiply(source_stride, dest_stride));
        break;
      }
      case OutputIndexMethod::array:
        return false;
    }
  }

  for (DimensionIndex dest_dim = 0; dest_dim < dest_rank; ++dest_dim) {
    auto map = chunk_transform.output_index_map(dest_dim);
    source_offset = internal::wrap_on_overflow::Subtract(
        source_offset, internal::wrap_on_overflow::Multiply(
                           new_byte_strides[dest_dim], map.offset()));
  }

  auto& new_array = write_state.array;
  new_array.layout() =
      StridedLayoutView<>(dest_rank, domain.shape().data(), new_byte_strides);

  source_offset = internal::wrap_on_overflow::Add(
      source_offset,
      IndexInnerProduct(dest_rank, domain.origin().data(), new_byte_strides));

  new_array.element_pointer() = AddByteOffset(
      SharedElementPointer<void>(internal::const_pointer_cast<void>(std::move(
                                     source_array.element_pointer().pointer())),
                                 spec.dtype()),
      source_offset);

  using WriteArraySourceCapabilities =
      AsyncWriteArray::WriteArraySourceCapabilities;
  using MaskedArray = AsyncWriteArray::MaskedArray;
  switch (source_capabilities) {
    case WriteArraySourceCapabilities::kCannotRetain:
      ABSL_UNREACHABLE();
    case WriteArraySourceCapabilities::kMutable:
      write_state.array_capabilities = MaskedArray::kMutableArray;
      break;
    case WriteArraySourceCapabilities::kImmutableAndCanRetainIndefinitely:
      write_state.array_capabilities =
          MaskedArray::kImmutableAndCanRetainIndefinitely;
      break;
    case WriteArraySourceCapabilities::kImmutableAndCanRetainUntilCommit:
      write_state.array_capabilities =
          MaskedArray::kImmutableAndCanRetainUntilCommit;
      break;
  }

  return true;
}

}  // namespace

absl::Status AsyncWriteArray::WriteArray(
    const Spec& spec, BoxView<> domain, IndexTransformView<> chunk_transform,
    absl::FunctionRef<Result<std::pair<TransformedSharedArray<const void>,
                                       WriteArraySourceCapabilities>>()>
        get_source_array) {
  [[maybe_unused]] const DimensionIndex dest_rank = spec.rank();
  assert(domain.rank() == dest_rank);
  assert(chunk_transform.output_rank() == dest_rank);
  // Check if `chunk_transform` has an output range exactly equal to the domain
  // of the `AsyncWriteArray`.  The array can be used by reference only in this
  // case.
  Box<dynamic_rank(kMaxRank)> output_range(spec.rank());
  TENSORSTORE_ASSIGN_OR_RETURN(
      bool output_range_exact,
      tensorstore::GetOutputRange(chunk_transform, output_range));
  if (!output_range_exact || output_range != domain) {
    // Output range of `chunk_transform` does not match the domain of the
    // `AsyncWriteArray`.
    return absl::CancelledError();
  }
  TENSORSTORE_ASSIGN_OR_RETURN(auto source_array_info, get_source_array());

  auto source_capabilities = std::get<1>(source_array_info);
  if (source_capabilities == WriteArraySourceCapabilities::kCannotRetain) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto dest_transformed_array,
        write_state.GetWritableTransformedArray(spec, domain, chunk_transform));
    TENSORSTORE_RETURN_IF_ERROR(CopyTransformedArray(
        std::get<0>(source_array_info), dest_transformed_array));
  } else {
    if (!ZeroCopyToWriteArray(spec, domain, chunk_transform,
                              std::get<0>(source_array_info),
                              source_capabilities, write_state)) {
      return absl::CancelledError();
    }
  }
  write_state.mask.Reset();
  write_state.mask.num_masked_elements = domain.num_elements();
  write_state.mask.region = domain;
  return absl::OkStatus();
}

Result<NDIterable::Ptr> AsyncWriteArray::BeginWrite(
    const Spec& spec, BoxView<> domain, IndexTransform<> chunk_transform,
    Arena* arena) {
  return write_state.BeginWrite(spec, domain, std::move(chunk_transform),
                                arena);
}

void AsyncWriteArray::EndWrite(const Spec& spec, BoxView<> domain,
                               IndexTransformView<> chunk_transform,
                               bool success, Arena* arena) {
  if (!success) {
    InvalidateReadState();
    return;
  }
  write_state.EndWrite(spec, domain, chunk_transform, arena);
}

}  // namespace internal
}  // namespace tensorstore
