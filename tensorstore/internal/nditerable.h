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

#ifndef TENSORSTORE_INTERNAL_NDITERABLE_H_
#define TENSORSTORE_INTERNAL_NDITERABLE_H_

/// \file
/// Defines the low-level NDIterable and NDIterator interfaces for iterating
/// over the contents of multi-dimensional array-like views.
///
/// This file defines the basic interfaces.  See `nditerable_util.h` for
/// utilities for composing and consuming NDIterable objects.  See
/// `nditerable_array.h` for obtaining an NDIterable corresponding to an
/// `Array`, and `nditerable_transformed_array.h` for obtaining an NDIterable
/// corresponding to a `TransformedArray`, and
/// `nditerable_elementwise_{input,output}_transform.h` for obtaining an
/// NDIterable objects that are elementwise-transformed views.

#include <memory>

#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/arena.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/internal/unique_with_intrusive_allocator.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal {

/// Interface for retrieving 1-dimensional buffers within a multi-dimensional
/// NDIterable.
///
/// NDIterable instances are always allocated using an `ArenaAllocator` and
/// managed using an `NDIterator::Ptr`.
///
/// Implementations of the `NDIterable` interface must define a suitable
/// `NDIterator` implementation to return from `NDIterable::GetIterator`.
///
/// While not explicitly exposed as part of the interface, the following
/// additional attributes are implicitly associated with an `NDIterator`:
///
///   - `DataType data_type`: the type of the data yielded by this iterator.
///
///   - `IterationBufferKind buffer_kind`: the type of buffer supported by this
///     iterator.
///
///   - `bool external`: If `true`, the user of this `NDIterator` is responsible
///     for providing an external buffer to store the data at a given position.
///     If `false`, the `NDIterator` provides its own buffer.
///
///   - `DimensionIndex rank`, with `rank >= 1`: the rank of the iteration
///     space.
///
///   - `span<const Index> iteration_shape`: The shape of the iteration space,
///     of size `rank`.
///
///   - `Index max_block_size`: the maximum supported block size.
///
///   - The supported read/write/read-write mode of the iterator.
class NDIterator {
 public:
  using allocator_type = ArenaAllocator<>;
  virtual ArenaAllocator<> get_allocator() const = 0;

  /// A `Derived` class implementation of `NDIterator` must inherit from
  /// `NDIterator::Base<Derived>`, override the `get_allocator` method, and
  /// override at least one of `GetBlock` and `UpdateBlock`.
  template <typename Derived>
  using Base = IntrusiveAllocatorBase<Derived, NDIterator>;

  /// Unique pointer to an `NDIterator`.
  using Ptr = std::unique_ptr<NDIterator, VirtualDestroyDeleter>;

  /// Gets the block at the specified location.
  ///
  /// \param indices Vector of length equal to the implicitly-associated `rank`.
  ///     Must satisfy `0 >= indices[i] && indices[i] <= iteration_shape[i]` for
  ///     `0 <= i < rank`, and
  ///     `indices[rank-1] + block_size <= iteration_shape[rank-1]`.
  /// \param block_size The extent along the last iteration dimension, must be
  ///     `<= max_block_size`.
  /// \param pointer[in,out] If the implicit attribute `external == true`,
  ///     `*pointer` must be an already-constructed buffer of kind
  ///     `buffer_kind`, data type `data_type`, and length `block_size` to fill
  ///     with the data starting at the specified position.  If
  ///     `external == false`, the existing value of `*pointer` is ignored and
  ///     `*pointer` must be assigned to a buffer of kind `buffer_kind`, length
  ///     `block_size`, and data type `data_type` containing the data starting
  ///     at `indices`.  The returned buffer is invalidated upon the next call
  ///     to `GetBlock`, `UpdateBlock`, or if the `NDIterator` is destroyed.
  /// \param status[out] Non-null pointer to location in which an error may be
  ///     returned if the return value is less than `block_size`.
  /// \returns The number of elements, starting at the beginning of the buffer,
  ///     that were successfully filled with data.  If `external == true`,
  ///     positions in the `*pointer` buffer after the returned number of
  ///     elements must not be modified.  If no error occurred, `block_size`
  ///     should be returned.  Otherwise, a number less than `block_size` may be
  ///     returned and `*status` may optionally be set to an error.  If
  ///     `*status` is left unchanged, a default error status is used.
  /// \remark The default implementation just returns `block_size`.  Except for
  ///     write-only iterators with `external == true`, it must be overridden.
  virtual Index GetBlock(span<const Index> indices, Index block_size,
                         IterationBufferPointer* pointer, Status* status);

  /// Updates the block at the specified location.
  ///
  /// Must be called after modifying the buffer obtained/filled by the prior
  /// call to `GetBlock`.
  ///
  /// \param indices Must specify same index vector (same contents, not
  ///     necessarily the same address) as prior call to `GetBlock`.
  /// \param block_size Number of elements, starting at the beginning of the
  ///     buffer, that were modified.  Must be <= return value from prior call
  ///     to `GetBlock`.
  /// \param pointer[in] Must equal to the pointer passed to (if
  ///     `external == true`) or returned by (if `external == false`) prior call
  ///     to `GetBlock`.
  /// \param status[out] Non-null pointer to location in which an error may be
  ///     returned if the return value is less than `block_size`.
  /// \returns The number of elements, starting at the beginning of the buffer,
  ///     that were successfully updated.
  /// \remark The default implementation just returns `block_size`, and need not
  ///     be overridden by read-only iterators.
  virtual Index UpdateBlock(span<const Index> indices, Index block_size,
                            IterationBufferPointer pointer, Status* status);

  /// Needed by `VirtualDestroyDeleter`.
  virtual void Destroy() = 0;

  /// Virtual destructor not needed, but added to silence warnings.
  virtual ~NDIterator();
};

/// Specifies constraints on iteration order for iterating over a collection of
/// NDIterable instances.
///
/// This interface is used by `GetNDIterationLayoutInfo` to construct efficient
/// iteration layouts.  For example, a contiguous array (with dimensions
/// optionally transposed or reversed) can be treated as a contiguous 1-d array.
///
/// For example, consider the case of an iterable for a `tensorstore::Array`
/// with shape `{2, 3, 4, 5}` and byte strides `{4, -40, 0, -8}`:
///
/// Note that despite the fact that the layout is neither a C order nor
/// Fortran order layout, the array data is in fact simply a dimension
/// permutation and reflection of a contiguous C order array of size
/// `{3, 5, 2}` with byte strides `{40, 8, 4}` (for a 4-byte data type), with
/// an extra dummy dimension of size 4 added.
///
/// The NDIterable interface makes it possible for `GetNDIterationLayoutInfo` to
/// discover this structure so that the data can be efficiently iterated over as
/// a 1-D array.
///
/// The `GetNDIterationLayoutInfo` function operates as follows:
///
/// 1. Computes a permutation of the dimensions, which it does by sorting the
///    dimensions using `GetDimensionOrder`.
///
/// 2. Discards dimensions that can be skipped as specified by their
///    `DirectionPref`.
///
/// 3. For the remaining dimensions, chooses an iteration direction according to
///    their `DirectionPref` values and combines consecutive dimensions as
///    permitted by `CanCombineDimensions`.
///
/// The NDIterable implementation for `tensorstore::Array` is defined such that
/// `GetDimensionOrder` orders the dimensions by the magnitudes of their
/// strides, in decreasing order, and the `DirectionPref` is determined from the
/// byte stride, where:
///
///     stride == 0: kCanSkip
///     stride >  0: kForward
///     stride <  0: kBackward
///
/// The `DirectionPref::kEither` value is intended to be used for special cases
/// where combining with other dimensions may be possible regardless of the
/// iteration direction, e.g. data generated on the fly during iteration.
///
/// For the example above, the result is:
///
///     | Dimension index | Size | Byte stride | DirectionPref |
///     | --------------- | ---- | ----------- | ------------- |
///     |       1         |   3  |     -40     |   kBackward   |
///     |       3         |   5  |      -8     |   kBackward   |
///     |       0         |   2  |       4     |   kForward    |
///     |       2         |   4  |       0     |   kCanSkip    |
///
/// Note that for the purpose of combining dimensions it is equally valid to
/// reverse the directions, as long as it is done consistently within a given
/// iterable.
class NDIterableLayoutConstraint {
 public:
  /// Specifies preferences for iterating over a given dimension.
  enum class DirectionPref {
    /// Forward is more likely to permit combining.  This dimension cannot be
    /// skipped.
    kForward = 1,
    /// Backward is more likely to permit combining.  This dimension cannot be
    /// skipped.
    kBackward = -1,
    /// Either direction is equally good.  This dimension cannot be skipped.
    kEither = 0,
    /// Either direction is equally good.  This dimension can be skipped.
    kCanSkip = 2,
  };

  /// Returns the combined `DirectionPref`.
  static inline DirectionPref CombineDirectionPrefs(DirectionPref a,
                                                    DirectionPref b) {
    switch (a) {
      case DirectionPref::kForward:
      case DirectionPref::kBackward:
        return a;
      case DirectionPref::kCanSkip:
        return b;
      case DirectionPref::kEither:
        return (b == DirectionPref::kCanSkip) ? DirectionPref::kEither : b;
    }
    TENSORSTORE_UNREACHABLE;  // COV_NF_LINE
  }

  /// Returns the iteration direction (`+1` for forward, `-1` for backward) for
  /// a given `DirectionPref`.
  static inline int GetDirection(DirectionPref x) {
    return (x == DirectionPref::kBackward) ? -1 : 1;
  }

  /// Returns a negative value if `dim_i` should be outer, positive
  /// value if `dim_j` should be outer, `0` if no preference.  When
  /// iterating over multiple iterables, the largest magnitude value
  /// takes precedence.
  virtual int GetDimensionOrder(DimensionIndex dim_i,
                                DimensionIndex dim_j) const = 0;

  /// Updates `prefs[i]` for each dimension `i`.
  virtual void UpdateDirectionPrefs(DirectionPref* prefs) const = 0;

  /// Returns true if we can iterate over `(dim_i, dim_j)` with the given
  /// directions by iterating over `dim_j` for `size_i * size_j`.
  virtual bool CanCombineDimensions(DimensionIndex dim_i, int dir_i,
                                    DimensionIndex dim_j, int dir_j,
                                    Index size_j) const = 0;

  virtual ~NDIterableLayoutConstraint();
};

/// Specifies constraints on iteration order and buffer kind/size for iterating
/// over a collection of NDIterable instances.
class NDIterableBufferConstraint : public NDIterableLayoutConstraint {
 public:
  struct IterationBufferConstraint {
    /// Minimum buffer kind supported.
    IterationBufferKind min_buffer_kind;

    /// If `true`, an external buffer must be provided to the
    /// `NDIterator::GetBlock` method.
    bool external;
  };

  /// Specifies a sequence of iteration dimensions, directions, and extents.
  struct IterationLayoutView {
    /// Specifies the extent of each original dimension (must have the same
    /// length as `iteration_directions`).
    span<const Index> shape;

    /// The iteration direction for each original dimension.  A value of `+1`
    /// means forward, a value of `-1` means reverse.  A value of `0` is only
    /// valid when it corresponds to a dummy iteration dimension of `-1`.
    span<const int> directions;

    /// The sequence of simplified dimensions.  The first dimension is the
    /// outermost dimension when iterating (changes slowest), while the last
    /// dimension is the innermost (changes fastest).  A dimension of `-1` means
    /// a dummy dimension (elements are repeated over this dimension).
    ///
    /// A valid iteration layout always has at least one iteration dimension
    /// (using a dummy dimension of `-1` if it would otherwise have a rank of
    /// 0).
    ///
    /// All dimensions in `iteration_dimensions` must either be `-1` or a unique
    /// dimension index in the range `[0, shape.size())`.
    ///
    /// Note that `iteration_dimensions` and `iteration_shape` are indexed by
    /// the iteration dimension index, while `shape` and `directions` are
    /// indexed by the original dimension index.
    span<const DimensionIndex> iteration_dimensions;

    /// Specifies the simplified iteration shape for each dimension in
    /// `iteration_dimensions`.  Must have the same length as
    /// `iteration_dimensions`.
    span<const Index> iteration_shape;

    /// Returns the full (original) number of dimensions.
    DimensionIndex full_rank() const { return shape.size(); }

    /// Returns the number of simplified iteration dimensions.
    DimensionIndex iteration_rank() const {
      return iteration_dimensions.size();
    }
  };

  /// Returns the minimum supported iteration buffer kind.
  virtual IterationBufferConstraint GetIterationBufferConstraint(
      IterationLayoutView layout) const = 0;

  /// Returns the number of working memory bytes per element given the
  /// specified iteration dimensions and buffer kind.
  virtual std::ptrdiff_t GetWorkingMemoryBytesPerElement(
      IterationLayoutView layout, IterationBufferKind buffer_kind) const = 0;

  virtual ~NDIterableBufferConstraint();
};

/// Interface for iterating over multi-dimensional array-like views as a
/// sequence of 1-d contiguous/strided/indexed arrays.
///
/// Instances of `NDIterable` are always allocated using `ArenaAllocator<>` and
/// managed using an `NDIterable::Ptr`.
///
/// The `NDIterable` implicitly has an associated `span<const Index> shape`, but
/// for efficiency there is no accessor for retrieving the `shape` or even the
/// rank.  (This permits implementations of `NDIterable` that do not store the
/// shape.)  Instead, the user is responsible for ensuring that an `NDIterable`
/// is only used with the correct shape.
class NDIterable : public NDIterableBufferConstraint {
 public:
  /// A `Derived` class implementation of `NDIterable` must inherit from
  /// `NDIterable::Base<Derived>` and override all of the pure virtual methods.
  template <typename Derived>
  using Base = IntrusiveAllocatorBase<Derived, NDIterable>;

  using Ptr = std::unique_ptr<NDIterable, VirtualDestroyDeleter>;

  using allocator_type = ArenaAllocator<>;

  virtual ArenaAllocator<> get_allocator() const = 0;

  /// Returns the data type.
  virtual DataType data_type() const = 0;

  /// Combines an `IterationLayoutView` with a block size value.
  struct IterationBufferLayoutView : public IterationLayoutView {
    /// Block size to use when iterating.
    Index block_size;
  };

  /// Combines an `IterationBufferLayoutView` with an `IterationBufferKind`.
  struct IterationBufferKindLayoutView : public IterationBufferLayoutView {
    /// The buffer kind to use when iterating.
    IterationBufferKind buffer_kind;
  };

  /// Returns an iterator for the specified buffer parameters, dimension order,
  /// and directions.
  ///
  /// \pre `block_size >= 0`.
  /// \pre `buffer_kind >= constraint.min_buffer_kind`, where `constraint`
  ///     equals `GetIterationBufferConstraint(iteration_dimensions)`.
  /// \pre `iteration_dimensions.size() == iteration_directions.size()`.
  /// \returns A non-null `NDIterator` with an associated `iteration_shape`
  ///     equal to the implicitly associated `shape`, indexed by
  ///     `layout.iteration_dimensions`, and associated `data_type` equal to
  ///     `this->data_type()`, an associated `buffer_kind` of
  ///     `layout.buffer_kind`, and an associated `max_block_size` of
  ///     `layout.block_size`.
  virtual NDIterator::Ptr GetIterator(
      IterationBufferKindLayoutView layout) const = 0;

  /// Needed by `VirtualDestroyDeleter`.
  virtual void Destroy() = 0;

  /// Virtual destructor not needed, but added to silence warnings.
  virtual ~NDIterable();
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_NDITERABLE_H_
