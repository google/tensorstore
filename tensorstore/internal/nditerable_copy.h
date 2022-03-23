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

#ifndef TENSORSTORE_INTERNAL_NDITERABLE_COPY_H_
#define TENSORSTORE_INTERNAL_NDITERABLE_COPY_H_

/// \file
/// Utilities for efficiently copying from one `NDIterable` to another.
///
/// The high-level interface is `NDIterableCopier`, which copies one
/// `NDIterable` to another in entirety.  The lower-level
/// `NDIterableCopyManager` and `NDIteratorCopyManager` classes can be used for
/// to perform partial copies or for greater control over the iteration order.

#include <array>

#include "absl/status/status.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/arena.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/internal/nditerable.h"
#include "tensorstore/internal/nditerable_buffer_management.h"
#include "tensorstore/internal/nditerable_util.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal {

/// Constraints the layout and buffer parameters to use for efficiently copying
/// from one NDIterable to another, while minimizing the use of extra buffers
/// and copies.
///
/// For usage, see the implementation of `NDIterableCopier`.
struct NDIterableCopyManager
    : public CompositeNDIterableLayoutConstraint<
          std::array<const NDIterable*, 2>, NDIterableLayoutConstraint> {
  using Base =
      CompositeNDIterableLayoutConstraint<std::array<const NDIterable*, 2>,
                                          NDIterableLayoutConstraint>;

  enum class BufferSource {
    /// Elements are separately copied from the buffer provided by `input` to
    /// the buffer provided by `output`.
    kBoth,
    /// Buffer provided by `input` is directly used to update `output`.
    kInput,

    /// Buffer provided by `output` is directly filled by `input`.
    kOutput,

    /// External buffer is separately allocated, filled by `input`, and then
    /// directly used to update `output`.
    kExternal,
  };

  struct BufferParameters {
    /// Buffering method to use.
    BufferSource buffer_source;

    /// The input buffer kind.
    IterationBufferKind input_buffer_kind;
    IterationBufferKind output_buffer_kind;
  };

  /// Constructs from the specified iterables, which must have a lifetime at
  /// least as long as this object.
  ///
  /// \param input Non-null pointer to input iterable.
  /// \param output Non-null pointer to output iterable.
  /// \dchecks `input.dtype() == output.dtype()`.
  /// \pre `input` and `output` have the same implicit `shape`.
  NDIterableCopyManager(const NDIterable* input, const NDIterable* output);

  BufferParameters GetBufferParameters(
      NDIterable::IterationLayoutView layout) const;

  std::ptrdiff_t GetWorkingMemoryBytesPerElement(
      NDIterable::IterationLayoutView layout) const;

  const NDIterable* input() const { return this->iterables[0]; }
  const NDIterable* output() const { return this->iterables[1]; }
};

/// Holds an `input` and `output` iterator and supports efficiently copying
/// between them.
struct NDIteratorCopyManager {
 public:
  /// Obtains an `input` and `output` iterator from the input and output
  /// iterables in `iterable` according to `layout`, and allocates an external
  /// buffer if needed.
  ///
  /// \param iterable Specifies the input and output iterables.
  /// \param layout Specifies the iteration layout.
  /// \param allocator Specifies the allocator to use.
  NDIteratorCopyManager(const NDIterableCopyManager& iterable,
                        NDIterable::IterationBufferLayoutView layout,
                        ArenaAllocator<> allocator);

  /// Copies a single block.
  ///
  /// \param indices Position vector of length `layout.rank()`, where `layout`
  ///     is the constructor argument.
  /// \param block_size Size of block to copy starting at `indices`.  Must be
  ///     `<= layout.block_size`, where `layout` is the constructor argument.
  /// \param status[out] Non-null pointer to location where error status may be
  ///     set if the return value is less than `block_size`.
  /// \returns The number of elements successfully copied (equal to `block_size`
  ///     on success).  If less than `block_size`, `*status` may be set to an
  ///     error, or may be left unchanged to indicate a default/unspecified
  ///     error.
  Index Copy(span<const Index> indices, Index block_size,
             absl::Status* status) {
    return copy_impl_(this, indices, block_size, status);
  }

 private:
  NDIterator::Ptr input_;
  NDIterator::Ptr output_;
  using CopyImpl = Index (*)(NDIteratorCopyManager* self,
                             span<const Index> indices, Index block_size,
                             absl::Status* status);
  CopyImpl copy_impl_;
  SpecializedElementwiseFunctionPointer<2, absl::Status*>
      copy_elements_function_;
  NDIteratorExternalBufferManager<1, 2> buffer_manager_;
};

/// Convenience interface for copying from one `NDIterable` to another.
///
/// Example usage:
///
///     NDIterable::Ptr input = ...;
///     NDIterable::Ptr output = ...;
///     NDIterableCopier copier(input, output, shape, constraints, arena);
///     auto status = copier.Copy();
///     if (!status) {
///       // In case of an error, `copier.stepper()` points to one past the
///       // last position fully copied.
///     }
struct NDIterableCopier {
  /// Constructs a copier.
  ///
  /// \param input The input (source) iterable.
  /// \param output The output (destination) iterable.
  /// \param shape The implicitly-associated shape of both `input` and `output`.
  /// \param constraints Constraints on the iteration order.
  /// \param arena Arena to use for memory allocation.  Must be non-null.
  /// \dchecks `input.dtype() == output.dtype()`.
  NDIterableCopier(const NDIterable& input, const NDIterable& output,
                   span<const Index> shape, IterationConstraints constraints,
                   Arena* arena);

  /// Same as above, but sets `constraints = skip_repeated_elements`.
  NDIterableCopier(const NDIterable& input, const NDIterable& output,
                   span<const Index> shape, Arena* arena)
      : NDIterableCopier(input, output, shape, skip_repeated_elements, arena) {}

  /// Attempts to copy from `source` to `dest`.
  ///
  /// Leaves `stepper()` at one past the last position copied.
  absl::Status Copy();

  /// Returns the layout used for copying.
  const NDIterationLayoutInfo<>& layout_info() const { return layout_info_; }

  /// Returns the stepper used for copying.  Set to one past the last position
  /// successfully copied.
  const NDIterationPositionStepper& stepper() const { return stepper_; }

  /// Provides access to the iterators obtained from the `input` and `output`
  /// iterables.
  NDIteratorCopyManager& iterator_copy_manager() {
    return iterator_copy_manager_;
  }

 private:
  NDIterableCopier(const NDIterableCopyManager& iterable_copy_manager,
                   span<const Index> shape, IterationConstraints constraints,
                   Arena* arena);

  NDIterationLayoutInfo<> layout_info_;
  NDIterationPositionStepper stepper_;
  NDIteratorCopyManager iterator_copy_manager_;
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_NDITERABLE_COPY_H_
