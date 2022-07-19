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

#ifndef TENSORSTORE_INTERNAL_NDITERABLE_UTIL_H_
#define TENSORSTORE_INTERNAL_NDITERABLE_UTIL_H_

/// \file
/// Utilities for consuming and composing NDIterable objects.

#include <algorithm>
#include <cassert>
#include <type_traits>
#include <utility>

#include "absl/container/fixed_array.h"
#include "absl/container/inlined_vector.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/arena.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/internal/integer_overflow.h"
#include "tensorstore/internal/nditerable.h"
#include "tensorstore/util/byte_strided_pointer.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal {

struct NDIterationFullLayoutInfo;
struct NDIterationSimplifiedLayoutInfo;

/// Convenience alias that evaluates to either `NDIterationSimplifiedLayoutInfo`
/// or `NDIterationFullLayoutInfo` depending on the value of `Full`.
template <bool Full = false>
using NDIterationLayoutInfo =
    std::conditional_t<Full, NDIterationFullLayoutInfo,
                       NDIterationSimplifiedLayoutInfo>;

/// Computes the simplified iteration layout for a given `iterable` and `shape`.
///
/// \param iterable The layout constraints.
/// \param shape The shape implicitly associated with `iterable`.
/// \param constraints Constraints on the iteration order.
/// \param info[out] Non-null pointer to location where computed layout is
///     stored.
void GetNDIterationLayoutInfo(const NDIterableLayoutConstraint& iterable,
                              span<const Index> shape,
                              IterationConstraints constraints,
                              NDIterationSimplifiedLayoutInfo* info);

/// Same as above, but also computes the corresponding "full" layout.
void GetNDIterationLayoutInfo(const NDIterableLayoutConstraint& iterable,
                              span<const Index> shape,
                              IterationConstraints constraints,
                              NDIterationFullLayoutInfo* info);

/// Specifies a simplified iteration layout for an `NDIterable`.
struct NDIterationSimplifiedLayoutInfo {
  NDIterationSimplifiedLayoutInfo() = default;

  NDIterationSimplifiedLayoutInfo(const NDIterableLayoutConstraint& iterable,
                                  span<const Index> shape,
                                  IterationConstraints constraints) {
    GetNDIterationLayoutInfo(iterable, shape, constraints, this);
  }

  NDIterable::IterationLayoutView layout_view() const {
    return {/*.shape=*/shape, /*.directions=*/directions,
            /*.iteration_dimensions=*/iteration_dimensions,
            /*.iteration_shape=*/iteration_shape};
  }

  /// Specifies whether the domain is empty, meaning iteration would cover 0
  /// elements (not the same as rank 0).
  bool empty;

  /// Specifies the extent of each original dimension.
  absl::InlinedVector<Index, kNumInlinedDims> shape;

  /// Specifies the iteration direction for each original dimension (`1` means
  /// forward, `-1` means backward, `0` means skipped).
  absl::InlinedVector<int, kNumInlinedDims> directions;

  /// Simplified iteration dimensions.  Dimensions that are skipped or combined
  /// into another dimensions are excluded.  The special value of `-1` indicates
  /// a dummy dimension, not corresponding to any real dimension, which is used
  /// for zero rank iterables; the corresponding value in `iteration_shape` is
  /// guaranteed to be `1`.
  ///
  /// All dimensions in `iteration_dimensions` must either be `-1` or a unique
  /// dimension index in the range `[0, shape.size())`.
  ///
  /// Note that `iteration_dimensions` and `iteration_shape` are indexed by the
  /// iteration dimension index, while `shape` and `directions` are indexed by
  /// the original dimension index.
  absl::InlinedVector<DimensionIndex, kNumInlinedDims> iteration_dimensions;

  /// Iteration extent for each simplified iteration dimension in
  /// `iteration_dimensions`.
  absl::InlinedVector<Index, kNumInlinedDims> iteration_shape;
};

/// Augments `NDIterationSimplifiedLayoutInfo` with the corresponding "full"
/// layout.
struct NDIterationFullLayoutInfo : public NDIterationSimplifiedLayoutInfo {
  NDIterationFullLayoutInfo() = default;

  NDIterationFullLayoutInfo(const NDIterableLayoutConstraint& iterable,
                            span<const Index> shape,
                            IterationConstraints constraints) {
    GetNDIterationLayoutInfo(iterable, shape, constraints, this);
  }

  /// Full iteration dimensions (including skipped and combined dimensions)
  /// equivalent to the simplified `iteration_dimensions`.
  absl::InlinedVector<DimensionIndex, kNumInlinedDims>
      full_iteration_dimensions;
};

/// Specifies a buffer kind and block size for use in conjunction with an
/// iteration layout.
struct NDIterationBufferInfo {
  /// Minimum buffer kind supported by all iterables.
  IterationBufferKind buffer_kind;

  /// Recommended block size to use for iteration.
  Index block_size;
};

/// Computes the block size to use for iteration that is L1-cache efficient.
///
/// For testing purposes, the behavior may be overridden to always return 1 by
/// calling `SetNDIterableTestUnitBlockSize(true)` or defining the
/// `TENSORSTORE_INTERNAL_NDITERABLE_TEST_UNIT_BLOCK_SIZE` preprocessor macro.
///
/// \param working_memory_bytes_per_element The number of bytes of temporary
///     buffer space required for each block element.
/// \param iteration_shape The simplified iteration shape, must have
///     `size() >= 1`.
/// \returns The block size, which is `<= iteration_shape.back()`.
Index GetNDIterationBlockSize(std::ptrdiff_t working_memory_bytes_per_element,
                              span<const Index> iteration_shape);

/// Computes the block size to use for iteration, based on
/// `iterable.GetWorkingMemoryBytesPerElement`.
///
/// \param iterable The buffer constraint.
/// \param iteration_dimensions The simplified iteration dimensions.
/// \param iteration_shape The simplified iteration shape (must have same length
///     as `iteration_dimensions`).
/// \param buffer_kind The buffer kind.
Index GetNDIterationBlockSize(const NDIterableBufferConstraint& iterable,
                              NDIterable::IterationLayoutView layout,
                              IterationBufferKind buffer_kind);

/// Computes the buffer kind and block size to use for iteration, based on
/// `iterable.GetIterationBufferConstraint` and
/// `iterable.GetWorkingMemoryBytesPerElement`.
///
/// \param layout The simplified iteration layout.
/// \param buffer_info[out] Non-null pointer to location where computed buffer
///     info will be stored.
void GetNDIterationBufferInfo(const NDIterableBufferConstraint& iterable,
                              NDIterable::IterationLayoutView layout,
                              NDIterationBufferInfo* buffer_info);

/// Specifies an iteration layout, buffer kind, and block size.
template <bool Full = false>
struct NDIterationInfo : public NDIterationLayoutInfo<Full>,
                         public NDIterationBufferInfo {
  NDIterationInfo() = default;

  explicit NDIterationInfo(const NDIterableBufferConstraint& iterable,
                           span<const Index> shape,
                           IterationConstraints constraints) {
    GetNDIterationLayoutInfo(iterable, shape, constraints, this);
    GetNDIterationBufferInfo(iterable, this->layout_view(), this);
  }

  NDIterable::IterationBufferKindLayoutView buffer_layout_view() const {
    return {{this->layout_view(), this->block_size}, this->buffer_kind};
  }
};

/// Combines the layout constraints of a sequence of NDIterableLayoutConstraint
/// objects.
///
/// \tparam Iterables Sequence type, must support `begin()`, `end()`,
///     `operator[]` and `size()`, and the `value_type` must be a pointer-like
///     type with a pointee type that inherits from
///     `NDIterableLayoutConstraint`.
/// \tparam Base Base class type, may be any derived class of
///     `NDIterableLayoutConstraint`.
template <typename Iterables, typename Base = NDIterableLayoutConstraint>
struct CompositeNDIterableLayoutConstraint : public Base {
  CompositeNDIterableLayoutConstraint(Iterables iterables)
      : iterables(std::move(iterables)) {}

  Iterables iterables;

  /// Returns the highest magnitude order preference.  In the case of a tie, the
  /// first iterable in `iterables` wins.
  int GetDimensionOrder(DimensionIndex dim_i,
                        DimensionIndex dim_j) const override {
    int max_magnitude_order = 0;
    for (const auto& iterable : iterables) {
      int order = iterable->GetDimensionOrder(dim_i, dim_j);
      if (std::abs(order) > std::abs(max_magnitude_order)) {
        max_magnitude_order = order;
      }
    }
    return max_magnitude_order;
  }

  /// Combines the direction preferences of all iterables in `iterables`.
  void UpdateDirectionPrefs(NDIterable::DirectionPref* prefs) const override {
    for (const auto& iterable : iterables) {
      iterable->UpdateDirectionPrefs(prefs);
    }
  }

  /// Two dimensions may be combined if, and only if, they can be combined in
  /// all iterables in `iterables`.
  bool CanCombineDimensions(DimensionIndex dim_i, int dir_i,
                            DimensionIndex dim_j, int dir_j,
                            Index size_j) const override {
    for (const auto& iterable : iterables) {
      if (!iterable->CanCombineDimensions(dim_i, dir_i, dim_j, dir_j, size_j)) {
        return false;
      }
    }
    return true;
  }
};

/// Advances `position` forward by `step` positions in the last dimension (C
/// order), with carrying.
///
/// Example usage:
///
///     std::vector<Index> position(shape.size());
///     for (Index block_size = max_block_size; block_size;
///          block_size = StepBufferPositionForward(shape, block_size,
///                                                 max_block_size,
///                                                 position.data())) {
///       // Use position
///     }
///
/// \param shape Specifies the shape bounds.
/// \param step Number of positions to step in the last dimension.
/// \param max_buffer_size Maximum block size that may be returned.
/// \param position[in,out] Non-null pointer to vector of length `shape.size()`.
/// \dchecks `shape.size() > 0`
/// \dchecks `step >= 0`
/// \dchecks `position[shape.size()-1] + step <= shape[shape.size()-1]`
/// \dchecks `max_buffer_size >= 0 && max_buffer_size <= shape[shape.size()-1]`
/// \pre `position[0] >= 0 && position[0] <= shape[0]`.
/// \pre `position[i] >= 0 && position[i] < shape[i]` for
///     `0 < i < shape.size()`.
/// \returns The remaining number of elements in the last dimension, after
///     advancing by `step` and carrying if necessary.
inline Index StepBufferPositionForward(span<const Index> shape, Index step,
                                       Index max_buffer_size, Index* position) {
  const DimensionIndex rank = shape.size();
  assert(rank > 0);
  assert(step >= 0);
  assert(max_buffer_size >= 0 && max_buffer_size <= shape[rank - 1]);

  const Index remainder = shape[rank - 1] - position[rank - 1];
  assert(remainder >= step);
  position[rank - 1] += step;
  if (remainder != step) {
    return std::min(max_buffer_size, remainder - step);
  }
  for (DimensionIndex i = rank - 1; i > 0;) {
    position[i] = 0;
    --i;
    if (++position[i] < shape[i]) {
      return max_buffer_size;
    }
  }
  return 0;
}

/// Advances `position` backward by up to `max_buffer_size` positions in the
/// last dimension (C order), with carrying.
///
/// If `position[shape.size()-1] > 0`, steps by
/// `min(max_buffer_size, position[shape.size()-1])` positions.  Otherwise,
/// steps by `max_buffer_size` positions (with carrying).
///
/// Example usage:
///
///     std::vector<Index> position(shape.size());
///     ResetBufferPositionAtEnd(shape, max_block_size, position.data());
///     for (Index block_size = max_block_size; block_size;
///          block_size = StepBufferPositionBackward(shape, max_block_size,
///                                                  position.data())) {
///       // Use position
///     }
///
/// \param shape Specifies the shape bounds.
/// \param max_buffer_size Maximum number of positions to steps.
/// \param position[in,out] Non-null pointer to vector of length `shape.size()`.
/// \dchecks `shape.size() > 0`
/// \dchecks `max_buffer_size > 0 && max_buffer_size <= shape[shape.size()-1]`
/// \pre `position[0] >= 0 && position[0] <= shape[0]`.
/// \pre `position[i] >= 0 && position[i] < shape[i]` for
///     `0 < i < shape.size()`.
/// \returns The number of (backward) positions advanced.
inline Index StepBufferPositionBackward(span<const Index> shape,
                                        Index max_buffer_size,
                                        Index* position) {
  const DimensionIndex rank = shape.size();
  assert(rank > 0);
  assert(max_buffer_size > 0);
  assert(max_buffer_size <= shape[rank - 1]);
  const Index remainder = position[rank - 1];
  if (remainder != 0) {
    const Index buffer_size = std::min(max_buffer_size, remainder);
    position[rank - 1] -= buffer_size;
    return buffer_size;
  }
  DimensionIndex i = rank - 2;
  while (true) {
    if (i < 0) return 0;
    if (position[i] != 0) {
      position[i]--;
      break;
    }
    --i;
  }
  ++i;
  while (i < rank - 1) {
    position[i] = shape[i] - 1;
    ++i;
  }
  position[rank - 1] = shape[rank - 1] - max_buffer_size;
  return max_buffer_size;
}

/// Resets `position` to a zero vector.
inline void ResetBufferPositionAtBeginning(span<Index> position) {
  std::fill_n(position.begin(), position.size(), Index(0));
}

/// Resets `position` to `step` positions before the end (C order).
///
/// \param shape Specifies the shape bounds.
/// \param step Number of elements before the end.
/// \param position[out] Non-null pointer to vector of length `shape.size()`.
/// \dchecks `shape.size() > 0`
/// \dchecks `step >= 0 && step <= shape[shape.size()-1]`
inline void ResetBufferPositionAtEnd(span<const Index> shape, Index step,
                                     Index* position) {
  const DimensionIndex rank = shape.size();
  assert(rank > 0);
  assert(step >= 0);
  assert(step <= shape[rank - 1]);
  for (DimensionIndex i = 0; i < rank - 1; ++i) {
    position[i] = shape[i] - 1;
  }
  position[rank - 1] = shape[rank - 1] - step;
}

/// Fills `offsets_array` with an index array corresponding to the specified
/// byte `stride`.
///
/// This is used to provide an `IterationBufferKind::kIndexed` view of a strided
/// array.
///
/// Sets `offsets_array[i] = i * stride` for `0 <= i < offsets_array.size()`.
inline void FillOffsetsArrayFromStride(Index stride,
                                       span<Index> offsets_array) {
  for (Index i = 0; i < offsets_array.size(); ++i) {
    offsets_array[i] = wrap_on_overflow::Multiply(stride, i);
  }
}

/// Returns an `IterationBufferPointer` of kind `buffer_kind` specified at run
/// time.
///
/// \param buffer_kind The buffer kind to return.
/// \param pointer The base pointer value.
/// \param byte_stride The byte stride value to use if
///     `buffer_kind != IterationBufferKind::kIndexed`.
/// \param byte_offsets Pointer to the byte offsets array to use if
///     `buffer_kind == IterationBufferKind::kIndexed`.
inline IterationBufferPointer GetIterationBufferPointer(
    IterationBufferKind buffer_kind, ByteStridedPointer<void> pointer,
    Index byte_stride, Index* byte_offsets) {
  IterationBufferPointer result;
  result.pointer = pointer;
  if (buffer_kind == IterationBufferKind::kIndexed) {
    result.byte_offsets = byte_offsets;
  } else {
    result.byte_stride = byte_stride;
  }
  return result;
}

/// Stores a buffer position vector, a block size, and a `shape` span.
///
/// Provides a convenience interface for advancing the position
/// forward/backward.
///
/// Example usage (forward):
///
///     NDIterationPositionStepper stepper(shape, max_block_size);
///     for (Index block_size = stepper.ResetAtBeginning();
///          block_size;
///          block_size = stepper.StepForward(block_size)) {
///       // Use `stepper.position()`
///     }
///
/// Example usage (backward):
///
///     NDIterationPositionStepper stepper(shape, max_block_size);
///     for (Index block_size = stepper.ResetAtEnd();
///          block_size;
///          block_size = stepper.StepBackward()) {
///       // Use `stepper.position()`
///     }
class NDIterationPositionStepper {
 public:
  /// Constructs the stepper with the specified `shape` and `block_size`.
  ///
  /// \param shape The shape vector, must outlive this object.
  /// \param block_size The maximum block size.
  NDIterationPositionStepper(span<const Index> shape, Index block_size)
      : position_(shape.size()),
        shape_(shape.data()),
        block_size_(block_size) {}

  Index ResetAtBeginning() {
    ResetBufferPositionAtBeginning(position_);
    return block_size_;
  }

  Index ResetAtEnd() {
    ResetBufferPositionAtEnd(shape(), block_size_, position_.data());
    return block_size_;
  }

  /// Advances `position()` forward by `step` positions (in C order),
  Index StepForward(Index step) {
    return StepBufferPositionForward(shape(), step, block_size_,
                                     position_.data());
  }

  Index StepBackward() {
    return StepBufferPositionBackward(shape(), block_size_, position_.data());
  }

  /// Returns the position vector.
  span<Index> position() { return position_; }
  span<const Index> position() const { return position_; }

  /// Returns the shape vector.
  span<const Index> shape() const { return span(shape_, position_.size()); }

  /// Returns the block size.
  Index block_size() const { return block_size_; }

  /// Returns a mutable reference to the block size, which may be modified.
  Index& block_size() { return block_size_; }

 private:
  absl::FixedArray<Index, kNumInlinedDims> position_;
  const Index* shape_;
  Index block_size_;
};

/// Arena with a stack-allocated buffer of 32 KiB.
class DefaultNDIterableArena {
 public:
  DefaultNDIterableArena()
      : arena_(/*Workaround gcc -Wuninitialized*/ (buffer_[0] = 0, buffer_)) {}

  operator Arena*() { return &arena_; }

  template <typename T>
  operator ArenaAllocator<T>() {
    return &arena_;
  }

 private:
  unsigned char buffer_[32 * 1024];
  tensorstore::internal::Arena arena_;
};

#ifndef NDEBUG
/// If `value` is `true`, forces `GetNDIterationBlockSize` to always return 1.
///
/// Note that if `TENSORSTORE_INTERNAL_NDITERABLE_TEST_UNIT_BLOCK_SIZE` is
/// defined, `GetNDIterationBlockSize` always returns 1 and calling this
/// function has no effect.
///
/// Should only be used for testing purposes, and should not be called
/// concurrently from multiple threads, or while any other thread is executing
/// NDIterable code.
void SetNDIterableTestUnitBlockSize(bool value);
#endif

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_NDITERABLE_UTIL_H_
