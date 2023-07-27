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

#include <memory>
#include <vector>

#include "tensorstore/array.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/masked_array.h"
#include "tensorstore/internal/nditerable.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/strided_layout.h"
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

    //// Constructs an async array specification.
    ///
    /// \param fill_value The fill value to use for each chunk, specifies the
    ///     shape of the chunk.  There are no constraints on
    ///     `fill_value.layout()`.
    /// \param component_bounds Specifies the non-resizable bounds of the
    ///     overall array.  Resizable bound should be specified as
    ///     +/-`kInfIndex`.  Must have the same rank as `fill_value`.  This is
    ///     used to determine whether a chunk has been fully overwritten (and
    ///     thereby allow an unconditional write to be used), even if part of
    ///     the chunk is outside the bounds of the overall component.
    explicit Spec(SharedArray<const void> fill_value, Box<> component_bounds);

    /// The fill value of the array.  Must be non-null.  This also specifies the
    /// shape of the "chunk".
    SharedArray<const void> fill_value;

    /// The bounds of the overall array.  Must have the same rank as
    /// `fill_value`.
    Box<> component_bounds;

    /// If `true`, indicates that the array should be stored even if it equals
    /// the fill value.  By default (when set to `false`), when preparing a
    /// writeback snapshot, if the value of the array is equal to the fill
    /// value, a null array is substituted.  Note that even if set to `true`, if
    /// the array is never written, or explicitly set to the fill value via a
    /// call to `WriteFillValue`, then it won't be stored.
    bool store_if_equal_to_fill_value = false;

    /// Comparison kind to use for fill value.
    EqualityComparisonKind fill_value_comparison_kind =
        EqualityComparisonKind::identical;

    /// Returns the shape of the array.
    span<const Index> shape() const { return fill_value.shape(); }

    Index num_elements() const { return fill_value.num_elements(); }

    /// Returns the number of elements in the chunk starting at `origin`.  This
    /// is less than or equal to `num_elements()`.
    Index chunk_num_elements(span<const Index> origin) const;

    /// Returns the rank of this array.
    DimensionIndex rank() const { return fill_value.rank(); }

    /// Returns the data type of this array.
    DataType dtype() const { return fill_value.dtype(); }

    /// C-order byte strides for `fill_value.shape()`.
    std::vector<Index> c_order_byte_strides;

    /// Returns the `StridedLayout` used for `write_data`.  May not be the
    /// layout of `fill_value` or `read_array`.
    StridedLayoutView<> write_layout() const {
      return StridedLayoutView<>(fill_value.shape(), c_order_byte_strides);
    }

    /// Allocates and constructs an array of `num_elements()` elements of type
    /// `dtype()`.
    std::shared_ptr<void> AllocateAndConstructBuffer() const;

    /// Returns an `NDIterable` for that may be used for reading the specified
    /// `array`, using the specified `chunk_transform`.
    ///
    /// \param array The array to read, must have shape equal to `shape()`.
    /// \param origin The associated origin of the array.
    /// \param chunk_transform Transform to use for reading, the output rank
    ///     must equal `rank()`.
    Result<NDIterable::Ptr> GetReadNDIterable(SharedArrayView<const void> array,
                                              span<const Index> origin,
                                              IndexTransform<> chunk_transform,
                                              Arena* arena) const;

    std::size_t EstimateReadStateSizeInBytes(bool valid) const {
      if (!valid) return 0;
      return num_elements() * dtype()->size;
    }
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
    std::size_t EstimateSizeInBytes(const Spec& spec) const;

    /// Optional pointer to C-order multi-dimensional array of data type and
    /// shape given by the `dtype` and `shape`, respectively, of the `Spec`.
    /// If equal to `nullptr`, no data has been written yet, or the current
    /// value is equal to the fill value.
    std::shared_ptr<void> data;

    /// If `mask` is all `true` (`num_masked_elements` is equal to the total
    /// number of elements in the `data` array), `data == nullptr` represents
    /// the same state as `data` containing the fill value.
    MaskData mask;

    SharedArrayView<const void> shared_array_view(const Spec& spec) {
      return SharedArrayView<const void>(
          SharedElementPointer<const void>(data, spec.dtype()),
          spec.write_layout());
    }

    /// Returns an `NDIterable` that may be used for writing to this array using
    /// the specified `chunk_transform`.
    ///
    /// \param spec The associated `Spec`.
    /// \param origin The associated origin of the array.
    /// \param chunk_transform Transform to use for writing, the output rank
    ///     must equal `spec.rank()`.
    /// \param arena Arena Non-null pointer to allocation arena that may be used
    ///     for allocating memory.
    Result<NDIterable::Ptr> BeginWrite(const Spec& spec,
                                       span<const Index> origin,
                                       IndexTransform<> chunk_transform,
                                       Arena* arena);

    /// Must be called after writing to the `NDIterable` returned by
    /// `BeginWrite`, even if an error occurs.
    ///
    /// \param spec The associated `Spec`.
    /// \param origin The associated origin of the array.
    /// \param chunk_transform Same transform supplied to prior call to
    ///     `BeginWrite`, the output rank must equal `spec.rank()`.
    /// \param layout The layout used for iterating over the `NDIterable`
    ///     returned by `BeginWrite`.
    /// \param write_end_position One past the last position (with respect to
    ///     `layout`) that was modified.
    /// \param arena Arena Non-null pointer to allocation arena that may be used
    ///     for allocating memory.
    bool EndWrite(const Spec& spec, span<const Index> origin,
                  IndexTransformView<> chunk_transform,
                  NDIterable::IterationLayoutView layout,
                  span<const Index> write_end_position, Arena* arena);

    /// Write the fill value.
    ///
    /// \param spec The associated `Spec`.
    /// \param origin The associated origin of the array.
    void WriteFillValue(const Spec& spec, span<const Index> origin);

    /// Returns `true` if the array has been fully overwritten.
    bool IsFullyOverwritten(const Spec& spec, span<const Index> origin) const {
      return mask.num_masked_elements >= spec.chunk_num_elements(origin);
    }

    /// Returns `true` if the array is unmodified.
    bool IsUnmodified() const { return mask.num_masked_elements == 0; }

    /// Resets to unmodified state.
    void Clear();

    /// Returns a snapshot to use to write back the current modifications.
    ///
    /// The returned array references `data`, so the caller should ensure that
    /// no concurrent modifications are made to `data` while the returned
    /// snapshot is in use.
    ///
    /// \param spec The associated `Spec`.
    /// \param origin The associated origin of the array.
    WritebackData GetArrayForWriteback(
        const Spec& spec, span<const Index> origin,
        const SharedArrayView<const void>& read_array,
        bool read_state_already_integrated = false);

   private:
    /// Ensures that `data` can be written, copying it if necessary.
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
  /// \param origin The associated origin of the array.
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
      const Spec& spec, span<const Index> origin,
      SharedArrayView<const void> read_array,
      const StorageGeneration& read_generation,
      IndexTransform<> chunk_transform, Arena* arena);

  /// Returns an `NDIterable` that may be used for writing to this array using
  /// the specified `chunk_transform`.
  ///
  /// \param spec The associated `Spec`.
  /// \param origin The associated origin of the array.
  /// \param chunk_transform Transform to use for writing, the output rank
  ///     must equal `spec.rank()`.
  /// \param arena Arena Non-null pointer to allocation arena that may be used
  ///     for allocating memory.
  Result<NDIterable::Ptr> BeginWrite(const Spec& spec, span<const Index> origin,
                                     IndexTransform<> chunk_transform,
                                     Arena* arena);

  bool EndWrite(const Spec& spec, span<const Index> origin,
                IndexTransformView<> chunk_transform,
                NDIterable::IterationLayoutView layout,
                span<const Index> write_end_position, Arena* arena);

  /// Returns an array to write back the current modifications.
  ///
  /// Moves `write_state` to `writeback_state`, which is referenced by the
  /// returned snapshot.  The caller should ensure that `writeback_state` is not
  /// modified while the returned snapshot is in use.
  ///
  /// \param spec The associated `Spec`.
  /// \param origin The associated origin of the array.
  WritebackData GetArrayForWriteback(
      const Spec& spec, span<const Index> origin,
      const SharedArrayView<const void>& read_array,
      const StorageGeneration& read_generation);
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_ASYNC_WRITE_ARRAY_H_
