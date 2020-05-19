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

  struct Spec {
    explicit Spec(SharedArray<const void> fill_value);

    /// The fill value of the array.  Must be non-null.  This also specifies the
    /// shape.
    SharedArray<const void> fill_value;

    /// Returns the shape of the array.
    span<const Index> shape() const { return fill_value.shape(); }

    Index num_elements() const { return fill_value.num_elements(); }

    /// Returns the rank of this array.
    DimensionIndex rank() const { return fill_value.rank(); }

    /// Returns the data type of this array.
    DataType data_type() const { return fill_value.data_type(); }

    /// C-order byte strides for `fill_value.shape()`.
    std::vector<Index> c_order_byte_strides;

    /// Returns the `StridedLayout` used for `write_data`.  May not be the
    /// layout of `fill_value` or `read_array`.
    StridedLayoutView<> write_layout() const {
      return StridedLayoutView<>(fill_value.shape(), c_order_byte_strides);
    }
  };

  /// Optional pointer to array with `shape` and `data_type` matching that of
  /// the `Spec`.  If equal to `nullptr`, no data has been read yet or the value
  /// is equal to the fill value.
  ///
  /// The array data must be immutable.
  SharedArrayView<const void> read_array;

  /// Optional pointer to C-order multi-dimensional array of data type and shape
  /// given by the `data_type` and `shape`, respectively, of the `Spec`.  If
  /// equal to `nullptr`, no data has been written yet, or the current value is
  /// equal to the fill value.
  std::shared_ptr<void> write_data;

  /// If `write_mask` is all `true` (`num_masked_elements` is equal to the
  /// total number of elements in the `write_data` array),
  /// `write_data == nullptr` represents the same state as `write_data`
  /// containing the fill value.
  MaskData write_mask;

  /// Value of `write_mask` prior to the current in-progress writeback
  /// operation.  If the writeback succeeds, this is discarded.  Otherwise,
  /// it is merged with `write_mask` when the writeback fails (e.g. due to a
  /// generation mismatch).
  MaskData write_mask_prior_to_writeback;

  /// Value of `write_data` prior to the current in-progress writeback
  /// operation.  If the writeback succeeds, this becomes the new
  /// `read_array` value.
  std::shared_ptr<void> write_data_prior_to_writeback;

  /// Returns an estimate of the memory required.
  std::size_t EstimateSizeInBytes(const Spec& spec) const;

  /// Returns the array that should be used for reading.
  SharedArrayView<const void> GetReadArray(const Spec& spec) const {
    if (read_array.data()) return read_array;
    return spec.fill_value;
  }

  /// Returns an `NDIterable` for that may be used for reading the committed
  /// state of this array (i.e. `read_array`), using the specified
  /// `chunk_transform`.
  ///
  /// \param spec The associated `Spec`.
  /// \param origin The associated origin of the array.
  /// \param chunk_transform Transform to use for reading, the output rank must
  ///     equal `spec.rank()`.
  Result<NDIterable::Ptr> GetReadNDIterable(const Spec& spec,
                                            span<const Index> origin,
                                            IndexTransform<> chunk_transform,
                                            Arena* arena);

  /// Returns an `NDIterable` that may be used for writing to this array using
  /// the specified `chunk_transform`.
  ///
  /// \param spec The associated `Spec`.
  /// \param origin The associated origin of the array.
  /// \param chunk_transform Transform to use for writing, the output rank must
  ///     equal `spec.rank()`.
  /// \param arena Arena Non-null pointer to allocation arena that may be used
  ///     for allocating memory.
  Result<NDIterable::Ptr> BeginWrite(const Spec& spec, span<const Index> origin,
                                     IndexTransform<> chunk_transform,
                                     Arena* arena);

  /// Must be called after writing to the `NDIterable` returned by `BeginWrite`,
  /// even if an error occurs.
  ///
  /// \param spec The associated `Spec`.
  /// \param origin The associated origin of the array.
  /// \param chunk_transform Same transform supplied to prior call to
  ///     `BeginWrite`, the output rank must equal `spec.rank()`.
  /// \param layout The layout used for iterating over the `NDIterable` returned
  ///     by `BeginWrite`.
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

  /// Return type of `GetArrayForWriteback`.
  struct WritebackData {
    /// Array with shape and data type matching that of the associated `Spec`.
    SharedArrayView<const void> array;
    /// If `true`, `array` is equal to the `fill_value of the associated `Spec`.
    bool equals_fill_value;
    /// If `true`, `array` updates all positions and can therefore potentially
    /// be written back unconditionally.
    bool unconditional;
  };

  /// Returns an array to write back the current modifications.
  ///
  /// The returned array references `write_data`, so the caller should ensure
  /// that no concurrent modifications are made to `write_data` until after
  /// `AfterWritebackStarts` is called.
  ///
  /// \param spec The associated `Spec`.
  /// \param origin The associated origin of the array.
  WritebackData GetArrayForWriteback(const Spec& spec,
                                     span<const Index> origin);

  /// Must be called after starting writeback with the data returned by
  /// `GetArrayForWriteback`.
  void AfterWritebackStarts(const Spec& spec);

  /// Should be called after writeback completes successfully or with an error
  /// to update the state.
  ///
  /// \param spec The associated `Spec`.
  /// \param origin The associated origin of the array.
  /// \param success Specifies whether writeback was successful.
  void AfterWritebackCompletes(const Spec& spec, span<const Index> origin,
                               bool success);

  /// Returns `true` if the array has been fully overwritten.
  bool IsFullyOverwritten(const Spec& spec) const {
    return write_mask.num_masked_elements == spec.num_elements();
  }

  /// Returns `true` if the array is unmodified.
  bool IsUnmodified() const { return write_mask.num_masked_elements == 0; }
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_ASYNC_WRITE_ARRAY_H_
