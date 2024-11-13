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

#ifndef TENSORSTORE_INTERNAL_ASYNC_WRITE_ARRAY_H_
#define TENSORSTORE_INTERNAL_ASYNC_WRITE_ARRAY_H_

/// \file
///
/// Define `AsyncWriteArray`, which is used by TensorStore `Driver`
/// implementations to track uncommitted writes.

#include <stddef.h>

#include <utility>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "tensorstore/array.h"
#include "tensorstore/box.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/transformed_array.h"
#include "tensorstore/internal/arena.h"
#include "tensorstore/internal/masked_array.h"
#include "tensorstore/internal/nditerable.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/util/extents.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal {

/// Represents an array that may have uncommitted writes to a subset of the
/// domain.
///
/// This is used by `ChunkCache` and other TensorStore drivers that need to
/// track uncommitted writes.
///
/// Each `AsyncWriteArray` object is implicitly associated with a corresponding
/// `AsyncWriteArray::Spec` object, and an `origin` vector.
///
/// The same `spec` and `origin` vector must be used consistently (for `origin`,
/// it need only be equivalent, but not necessarily the same memory).
struct AsyncWriteArray {
  /// Constructs an array of the specified rank (must equal the rank of the
  /// corresponding `Spec`).
  ///
  /// Does not actually allocate any data arrays.
  explicit AsyncWriteArray(DimensionIndex rank);
  AsyncWriteArray(AsyncWriteArray&& other)
      : write_state(std::move(other.write_state)),
        read_generation(std::move(other.read_generation)) {}

  struct Spec {
    Spec() = default;

    template <typename LayoutOrder = ContiguousLayoutOrder,
              typename = std::enable_if_t<IsContiguousLayoutOrder<LayoutOrder>>>
    explicit Spec(SharedOffsetArray<const void> overall_fill_value,
                  Box<> valid_data_bounds, LayoutOrder order = c_order)
        : overall_fill_value(std::move(overall_fill_value)),
          valid_data_bounds(std::move(valid_data_bounds)) {
      ConvertToContiguousLayoutPermutation(
          order, span(layout_order_buffer, this->rank()));
    }

    /// The overall fill value.  Every individual chunk must be contained within
    /// `overall_fill_value.domain()`.
    SharedOffsetArray<const void> overall_fill_value;

    /// Bounds on the valid data.  The domain of a chunk may extend beyond
    /// `valid_data_bounds` (but must still be contained within
    /// `overall_fill_value.domain()`).  However, values outside of
    /// `valid_data_bounds` are neither read nor written and do not need to be
    /// preserved.
    Box<> valid_data_bounds;

    /// Buffer containing permutation specifying the storage order to use when
    /// allocating a new array.
    ///
    /// Note that this order is used when allocating a new order but does not
    /// apply when the zero-copy `WriteArray` method is called.
    ///
    /// Only the first `rank()` elements are meaningful.
    ///
    /// For example, ``0, 1, 2`` denotes C order for rank 3, while ``2, 1, 0``
    /// denotes F order.
    DimensionIndex layout_order_buffer[kMaxRank];

    /// Comparison kind to use for fill value.
    EqualityComparisonKind fill_value_comparison_kind =
        EqualityComparisonKind::identical;

    /// Returns the rank of this array.
    DimensionIndex rank() const { return overall_fill_value.rank(); }

    /// Returns the data type of this array.
    DataType dtype() const { return overall_fill_value.dtype(); }

    /// Returns the number of elements that are contained within `bounds` and
    /// `domain`.
    Index GetNumInBoundsElements(BoxView<> domain) const;

    /// Returns the portion of `fill_value_and_bounds` contained within
    /// `domain`, translated to have a zero origin.
    SharedArrayView<const void> GetFillValueForDomain(BoxView<> domain) const;

    /// Storage order to use when allocating a new array.
    ContiguousLayoutPermutation<> layout_order() const {
      return ContiguousLayoutPermutation<>(span(layout_order_buffer, rank()));
    }

    /// Returns an `NDIterable` for that may be used for reading the specified
    /// `array`, using the specified `chunk_transform`.
    ///
    /// \param array The array to read. If `!array.valid()`, then the fill value
    ///     is used instead.
    /// \param domain The associated domain of the array.
    /// \param chunk_transform Transform to use for reading, the output rank
    ///     must equal `rank()`.
    Result<NDIterable::Ptr> GetReadNDIterable(SharedArrayView<const void> array,
                                              BoxView<> domain,
                                              IndexTransform<> chunk_transform,
                                              Arena* arena) const;

    size_t EstimateReadStateSizeInBytes(
        bool valid, tensorstore::span<const Index> shape) const {
      if (!valid) return 0;
      return ProductOfExtents(shape) * dtype()->size;
    }

    /// Allocates an array of the specified `shape`, for `this->dtype()` and
    /// `this->layout_order`.
    SharedArray<void> AllocateArray(span<const Index> shape) const;
  };

  /// Return type of `GetArrayForWriteback`.
  struct WritebackData {
    /// Array with shape and data type matching that of the associated `Spec`.
    SharedArrayView<const void> array;

    /// Indicates that the array must be stored.
    ///
    /// The conditions under which this is set to `true` depend on the value of
    /// `store_if_equal_to_fill_value`:
    ///
    /// - If `store_if_equal_to_fill_value == false`, this is `true` if, and
    ///   only if, `array` is not equal to the `fill_value`.
    ///
    /// - If `store_if_equal_to_fill_value == true`, this is `true` if there is
    ///   an existing read value or any writes have been performed that were not
    ///   overwritten by an explicit call to `WriteFillValue`.
    bool must_store;

    /// If `true`, a reference to `array` may be retained indefinitely.  If
    /// `false`, a reference to `array` may not be retained after the
    /// transaction is committed.
    bool may_retain_reference_to_array_indefinitely;
  };

  /// Represents an array with an associated mask indicating the positions that
  /// are valid.
  ///
  /// Each `MaskedArray` object is implicitly associated with a corresponding
  /// `Spec` object, and an `origin` vector.
  ///
  /// The same `spec` and `origin` vector must be used consistently (for
  /// `origin`, it need only be equivalent, but not necessarily the same
  /// memory).
  struct MaskedArray {
    /// Initializes the array to the specified rank.
    explicit MaskedArray(DimensionIndex rank);

    /// Returns an estimate of the memory required.
    size_t EstimateSizeInBytes(const Spec& spec,
                               tensorstore::span<const Index> shape) const;

    /// Array with data type of `spec.dtype()` and shape equal to
    /// `spec.shape()`.  If `array.data() == nullptr`, no data has been written
    /// yet, or the current value is equal to the fill value.
    SharedArray<void> array;

    /// If `mask` is all `true` (`num_masked_elements` is equal to the total
    /// number of elements in the `data` array), `data == nullptr` represents
    /// the same state as `data` containing the fill value.
    MaskData mask;

    enum ArrayCapabilities {
      kMutableArray,
      kImmutableAndCanRetainIndefinitely,
      kImmutableAndCanRetainUntilCommit,
    };

    /// Specifies how `array` may be used.  Only meaningful if
    /// `array.data() != nullptr`.
    ArrayCapabilities array_capabilities;

    /// If `true`, indicates that the array should be stored even if it equals
    /// the fill value. By default (when set to `false`), when preparing a
    /// writeback snapshot, if the value of the array is equal to the fill
    /// value, a null array is substituted. Note that even if set to `true`, if
    /// the array is never written, or explicitly set to the fill value via a
    /// call to `WriteFillValue`, then it won't be stored.
    bool store_if_equal_to_fill_value = false;

    SharedArrayView<const void> shared_array_view(const Spec& spec) {
      return array;
    }

    /// Returns a writable transformed array corresponding to `chunk_transform`.
    ///
    /// \param spec The associated `Spec`.
    /// \param domain The associated domain of the array.
    /// \param chunk_transform Transform to use for writing, the output rank
    ///     must equal `spec.rank()`.
    Result<TransformedSharedArray<void>> GetWritableTransformedArray(
        const Spec& spec, BoxView<> domain, IndexTransform<> chunk_transform);

    /// Returns an `NDIterable` that may be used for writing to this array using
    /// the specified `chunk_transform`.
    ///
    /// \param spec The associated `Spec`.
    /// \param domain The associated domain of the array.
    /// \param chunk_transform Transform to use for writing, the output rank
    ///     must equal `spec.rank()`.
    /// \param arena Arena Non-null pointer to allocation arena that may be used
    ///     for allocating memory.
    Result<NDIterable::Ptr> BeginWrite(const Spec& spec, BoxView<> domain,
                                       IndexTransform<> chunk_transform,
                                       Arena* arena);

    /// Must be called after writing to the `NDIterable` returned by
    /// `BeginWrite`, even if an error occurs.
    ///
    /// \param spec The associated `Spec`.
    /// \param domain The associated domain of the array.
    /// \param chunk_transform Same transform supplied to prior call to
    ///     `BeginWrite`, the output rank must equal `spec.rank()`.
    /// \param arena Arena Non-null pointer to allocation arena that may be used
    ///     for allocating memory.
    void EndWrite(const Spec& spec, BoxView<> domain,
                  IndexTransformView<> chunk_transform, Arena* arena);

    /// Write the fill value.
    ///
    /// \param spec The associated `Spec`.
    /// \param origin The associated origin of the array.
    void WriteFillValue(const Spec& spec, BoxView<> domain);

    /// Returns `true` if the array has been fully overwritten.
    bool IsFullyOverwritten(const Spec& spec, BoxView<> domain) const {
      return mask.num_masked_elements >= spec.GetNumInBoundsElements(domain);
    }

    /// Returns `true` if the array is unmodified.
    bool IsUnmodified() const { return mask.num_masked_elements == 0; }

    /// Resets to unmodified state.
    void Clear();

    /// Returns a snapshot to use to write back the current modifications.
    ///
    /// The returned array references `array`, so the caller should ensure that
    /// no concurrent modifications are made to `array` while the returned
    /// snapshot is in use.
    ///
    /// \param spec The associated `Spec`.
    /// \param domain The associated domain of the array.
    WritebackData GetArrayForWriteback(
        const Spec& spec, BoxView<> domain,
        const SharedArrayView<const void>& read_array,
        bool read_state_already_integrated = false);

   private:
    friend struct AsyncWriteArray;

    /// Copies `array`, which must already exist.
    void EnsureWritable(const Spec& spec);
  };

  /// Modifications to the read state.
  MaskedArray write_state;

  void InvalidateReadState() { read_generation = StorageGeneration::Invalid(); }

  /// Read generation on which `write_state` is based.
  StorageGeneration read_generation = StorageGeneration::Invalid();

  /// Returns an `NDIterable` for that may be used for reading the current write
  /// state of this array, using the specified `chunk_transform`.
  ///
  /// \param spec The associated `Spec`.
  /// \param domain The associated domain of the array.
  /// \param read_array The last read state.  If `read_array.data() == nullptr`,
  ///     implies `spec.fill_value()`.  Must match `spec.shape()` and
  ///     `spec.dtype()`, but the layout does not necessarily have to match
  ///     `spec.write_layout()`.
  /// \param read_generation The read generation corresponding to `read_array`.
  ///     This is compared to `this->read_generation` to determine if the read
  ///     state is up to date.
  /// \param chunk_transform Transform to use for reading, the output rank must
  ///     equal `spec.rank()`.
  Result<NDIterable::Ptr> GetReadNDIterable(
      const Spec& spec, BoxView<> domain,
      SharedArrayView<const void> read_array,
      const StorageGeneration& read_generation,
      IndexTransform<> chunk_transform, Arena* arena);

  enum class WriteArraySourceCapabilities {
    // The `AsyncWriteArray` must not retain a reference to the source array
    // after `WriteArray` returns.  The source data is guaranteed to remain
    // valid and unchanged until `WriteArray` returns.
    kCannotRetain,

    // Unique (mutable) ownership of the array data is transferred to the
    // `AsyncWriteArray` if `WriteArray` returns successfully.
    kMutable,

    // The `AsyncWriteArray` may retain a reference to the source array data
    // indefinitely.  The source data is guaranteed to remain valid and
    // unchanged until all references are released.
    kImmutableAndCanRetainIndefinitely,

    // The `AsyncWriteArray` may retain a reference to the source array data
    // until the transaction with which this `AsyncWriteArray` is associated is
    // fully committed or fully aborted, at which point all references must be
    // released.  The source data is guaranteed to remain valid and unchanged
    // until all references are released.
    kImmutableAndCanRetainUntilCommit,
  };

  /// Returns an `NDIterable` that may be used for writing to this array using
  /// the specified `chunk_transform`.
  ///
  /// \param spec The associated `Spec`.
  /// \param domain The associated domain of the array.
  /// \param chunk_transform Transform to use for writing, the output rank
  ///     must equal `spec.rank()`.
  /// \param arena Arena Non-null pointer to allocation arena that may be used
  ///     for allocating memory.
  Result<NDIterable::Ptr> BeginWrite(const Spec& spec, BoxView<> domain,
                                     IndexTransform<> chunk_transform,
                                     Arena* arena);

  /// Must be called after writing to the `NDIterable` returned by `BeginWrite`,
  /// even if an error occurs.
  ///
  /// \param spec The associated `Spec`.
  /// \param domain The associated domain of the array.
  /// \param chunk_transform Same transform supplied to prior call to
  ///     `BeginWrite`, the output rank must equal `spec.rank()`.
  /// \param success Indicates if all positions in the range of
  ///     `chunk_transform` were successfully updated.
  /// \param arena Arena Non-null pointer to allocation arena that may be used
  ///     for allocating memory.
  void EndWrite(const Spec& spec, BoxView<> domain,
                IndexTransformView<> chunk_transform, bool success,
                Arena* arena);

  /// Assigns this array from an existing source array, potentially without
  /// actually copying the data.
  ///
  /// \param spec The associated `Spec`.
  /// \param domain The associated domain of the array.
  /// \param chunk_transform Transform to use for writing, the output rank
  ///     must equal `spec.rank()`.
  /// \param get_source_array Returns the source array information.  Provided as
  ///     a callback to avoid potentially expensive work in constructing the
  ///     source array if it cannot be used anyway.
  /// \error `absl::StatusCode::kCancelled` if the output range of
  ///     `chunk_transform` is not exactly equal to the domain of this
  ///     `AsyncWriteArray`, or the source array is not compatible.  In this
  ///     case, callers should fall back to writing via `BeginWrite` and
  ///     `EndWrite`.
  absl::Status WriteArray(
      const Spec& spec, BoxView<> domain, IndexTransformView<> chunk_transform,
      absl::FunctionRef<Result<std::pair<TransformedSharedArray<const void>,
                                         WriteArraySourceCapabilities>>()>
          get_source_array);

  /// Returns an array to write back the current modifications.
  ///
  /// Moves `write_state` to `writeback_state`, which is referenced by the
  /// returned snapshot.  The caller should ensure that `writeback_state` is not
  /// modified while the returned snapshot is in use.
  ///
  /// \param spec The associated `Spec`.
  /// \param domain The associated domain of the array.
  WritebackData GetArrayForWriteback(
      const Spec& spec, BoxView<> domain,
      const SharedArrayView<const void>& read_array,
      const StorageGeneration& read_generation);
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_ASYNC_WRITE_ARRAY_H_
