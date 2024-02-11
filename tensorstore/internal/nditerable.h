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

#include "absl/status/status.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/arena.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/internal/unique_with_intrusive_allocator.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal {

/// Interface for retrieving 2-dimensional buffers within a multi-dimensional
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
///   - `DataType dtype`: the type of the data yielded by this iterator.
///
///   - `IterationBufferKind buffer_kind`: the type of buffer supported by this
///     iterator.
///
///   - `bool external`: If `true`, the user of this `NDIterator` is responsible
///     for providing an external buffer to store the data at a given position.
///     If `false`, the `NDIterator` provides its own buffer.
///
///   - `DimensionIndex rank`, with `rank >= 2`: the rank of the iteration
///     space.
///
///   - `span<const Index> iteration_shape`: The shape of the iteration space,
///     of size `rank`.
///
///   - `IterationBufferShape max_block_shape`: the maximum supported block
///     shape, equal to the `block_shape` field of the
///     `IterationBufferKindLayoutView` passed to `NDIterable::GetIterator`.
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
  ///     `0 <= i < rank`,
  ///     `indices[rank-2] + block_outer_size <= iteration_shape[rank-2]`, and
  ///     `indices[rank-1] + block_inner_size <= iteration_shape[rank-1]`.
  /// \param block_shape The extent along the last 2 dimensions, must be
  ///     elementwise `<= max_block_shape`.
  /// \param pointer[in,out] If the implicit attribute `external == true`,
  ///     `*pointer` must be an already-constructed buffer of kind
  ///     `buffer_kind`, data type `dtype`, and shape `block_shape` to fill with
  ///     the data starting at the specified position.  If `external == false`,
  ///     the existing value of `*pointer` is ignored and `*pointer` must be
  ///     assigned to a buffer of kind `buffer_kind`, shape `block_shape`, and
  ///     data type `dtype` containing the data starting at `indices`.  The
  ///     returned buffer is invalidated upon the next call to `GetBlock`,
  ///     `UpdateBlock`, or if the `NDIterator` is destroyed.
  /// \param status[out] Non-null pointer to location in which an error may be
  ///     returned if the return value is `false`.
  /// \returns `true` in the case of success, `false` to indicate an error.  If
  ///     `false` is returned, `*status` may optionally be set to an error.  If
  ///     `*status` is left unchanged, a default error status is used.
  /// \remark The default implementation just returns `true`.  Except for
  ///     write-only iterators with `external == true`, it must be overridden.
  virtual bool GetBlock(span<const Index> indices,
                        internal::IterationBufferShape block_shape,
                        IterationBufferPointer* pointer, absl::Status* status);

  /// Updates the block at the specified location.
  ///
  /// Must be called after modifying the buffer obtained/filled by the prior
  /// call to `GetBlock`.
  ///
  /// \param indices Must specify same index vector (same contents, not
  ///     necessarily the same address) as prior call to `GetBlock`.
  /// \param block_shape The shape of the block, must be the same as passed to
  ///     the prior call to `GetBlock`.
  /// \param pointer[in] Must equal the pointerpassed to (if `external == true`)
  ///     or returned by (if `external == false`) the prior call to `GetBlock`.
  /// \param status[out] Non-null pointer to location in which an error may be
  ///     returned if the return value is `false`.
  /// \returns `true` on success, or `false` to indicate an error.
  /// \remark The default implementation just returns `true`, and need not be
  ///     overridden by read-only iterators.
  virtual bool UpdateBlock(span<const Index> indices,
                           internal::IterationBufferShape block_shape,
                           IterationBufferPointer pointer,
                           absl::Status* status);

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
/// an extra inert dimension of size 4 added.
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
/// The `DirectionPref::kForwardRequired` value is intended to be used for
/// special cases where the NDIterable implementation may be simplified by not
/// having to consider skipped / backward iteration for a dimension.
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
  /// Specifies preferences for iterating over a given dimension.  Precedence is
  /// according to magnitude.
  enum class DirectionPref {
    /// Forward is more likely to permit combining.  This dimension cannot be
    /// skipped.
    kForward = 2,
    /// Forward direction is required for iteration.  This dimension cannot be
    /// skipped, even if the extent is 1.
    kForwardRequired = 3,
    /// Backward is more likely to permit combining.  This dimension cannot be
    /// skipped.
    kBackward = -2,
    /// Either direction is equally good.  This dimension cannot be skipped.
    kEither = 1,
    /// Either direction is equally good.  This dimension can be skipped.
    kCanSkip = 0,
  };

  /// Returns the most restrictive `DirectionPref`, or `a` if both are equally
  /// restrictive.
  static inline DirectionPref CombineDirectionPrefs(DirectionPref a,
                                                    DirectionPref b) {
    return (std::abs(static_cast<int>(a)) >= std::abs(static_cast<int>(b))) ? a
                                                                            : b;
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
    /// valid when it corresponds to a skipped (but not a combined) dimension.
    span<const int> directions;

    /// The sequence of simplified dimensions.  The first dimension is the
    /// outermost dimension when iterating (changes slowest), while the last
    /// dimension is the innermost (changes fastest).  A dimension of `-1` means
    /// an inert dimension (elements are repeated over this dimension).
    ///
    /// A valid iteration layout always has at least one iteration dimension
    /// (using an inert dimension of `-1` if it would otherwise have a rank of
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
  virtual ptrdiff_t GetWorkingMemoryBytesPerElement(
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
  virtual DataType dtype() const = 0;

  /// Combines an `IterationLayoutView` with a block size value.
  struct IterationBufferLayoutView : public IterationLayoutView {
    /// Block size to use when iterating.
    internal::IterationBufferShape block_shape;
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
  ///     `layout.iteration_dimensions`, and associated `dtype` equal to
  ///     `this->dtype()`, an associated `buffer_kind` of `layout.buffer_kind`,
  ///     and an associated `max_block_shape` of `layout.block_shape`.
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
