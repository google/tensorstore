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

#ifndef TENSORSTORE_INTERNAL_NDITERABLE_BUFFER_MANAGEMENT_H_
#define TENSORSTORE_INTERNAL_NDITERABLE_BUFFER_MANAGEMENT_H_

/// \file
/// Utilities for managing external buffers for NDIterator objects.

#include <algorithm>
#include <array>
#include <utility>

#include "absl/status/status.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/arena.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/internal/nditerable.h"
#include "tensorstore/internal/nditerable_util.h"
#include "tensorstore/util/division.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal {

/// Extends `CompositeNDIterableLayoutConstraint` to implement the
/// `NDIterableBufferConstraint` interface.
///
/// The buffer constraints are computed under the assumption that
/// `NDIteratorExternalBufferManager` will be used to manage any external
/// buffers needed by the iterables.
///
/// \tparam Iterables Sequence type, must support `begin()`, `end()`,
///     `operator[]` and `size()`, and the `value_type` must be a pointer-like
///     type with a pointee type that inherits from
///     `NDIterableBufferConstraint`.
/// \tparam BaseT Base class type, may be any derived class of
///     `NDIterableBufferConstraint`.
template <typename Iterables, typename BaseT = NDIterableBufferConstraint>
class NDIterablesWithManagedBuffers
    : public CompositeNDIterableLayoutConstraint<Iterables, BaseT> {
 public:
  NDIterablesWithManagedBuffers(Iterables iterables)
      : CompositeNDIterableLayoutConstraint<Iterables, BaseT>(
            std::move(iterables)) {}

  NDIterable::IterationBufferConstraint GetIterationBufferConstraint(
      NDIterable::IterationLayoutView layout) const override {
    auto buffer_kind = IterationBufferKind::kContiguous;
    for (const auto& iterable : this->iterables) {
      auto constraint = iterable->GetIterationBufferConstraint(layout);
      buffer_kind = std::max(buffer_kind, constraint.external
                                              ? IterationBufferKind::kContiguous
                                              : constraint.min_buffer_kind);
    }
    return NDIterable::IterationBufferConstraint{buffer_kind, true};
  }

  std::ptrdiff_t GetWorkingMemoryBytesPerElement(
      NDIterable::IterationLayoutView layout,
      IterationBufferKind buffer_kind) const override {
    std::ptrdiff_t num_bytes = 0;
    for (size_t i = 0; i < this->iterables.size(); ++i) {
      const auto& iterable = this->iterables[i];
      const auto constraint = iterable->GetIterationBufferConstraint(layout);
      if (constraint.external) {
        const auto dtype = iterable->dtype();
        num_bytes += dtype->size;
        if (constraint.min_buffer_kind == IterationBufferKind::kIndexed ||
            buffer_kind == IterationBufferKind::kIndexed) {
          num_bytes += sizeof(Index);
        }
      }
      num_bytes += iterable->GetWorkingMemoryBytesPerElement(
          layout,
          constraint.external ? constraint.min_buffer_kind : buffer_kind);
    }
    return num_bytes;
  }
};

/// Manages external buffers needed for `NDIterator` objects.
///
/// This is used by `NDIteratorsWithManagedBuffers` and by
/// `NDIteratorCopyManager`.
///
/// \tparam Arity Number of iterators to support (possibly including iterators
///     that do not require external buffers).
/// \param NumBufferKinds Number of separate buffer kind views (of the same
///     underlying buffer) to support for each iterator, e.g. to provide both a
///     `kContiguous` or `kStrided` and a `kIndexed` view of the same buffer.
template <std::size_t Arity, std::size_t NumBufferKinds>
class NDIteratorExternalBufferManager {
 public:
  using allocator_type = ArenaAllocator<>;

  /// Constructs the buffer manager without initializing any buffers.
  ///
  /// \param allocator Allocator to use to obtain any buffers that are needed.
  NDIteratorExternalBufferManager(ArenaAllocator<> allocator)
      : allocator_(allocator), buffer_(nullptr), buffer_size_(0) {}

  NDIteratorExternalBufferManager(
      const NDIteratorExternalBufferManager& other) = delete;
  NDIteratorExternalBufferManager(NDIteratorExternalBufferManager&& other) =
      delete;

  /// Initializes any necessary buffers.
  ///
  /// Frees any previously-allocated buffers.
  ///
  /// \param block_size The block size.
  /// \param data_types For each `0 <= i < Arity` for which
  ///     `data_types[i].valid() == true`, allocates a buffer of size
  ///     `block_size` of the specified data type, and sets
  ///     `buffer_pointers()[i][j]` to a buffer pointer of kind
  ///     `buffer_kinds[j][i]` that refers to it for `0 <= j < NumBufferKinds`.
  /// \param buffer_kinds For each `0 <= i < Arity` for which
  ///     `data_types[i].valid() == true`, specifies the buffer kinds.  If
  ///     `data_types[i].valid() == false`, `data_types[j][i]` is ignored.
  void Initialize(
      Index block_size, std::array<DataType, Arity> data_types,
      std::array<std::array<IterationBufferKind, NumBufferKinds>, Arity>
          buffer_kinds) {
    Free();
    data_types_ = data_types;
    block_size_ = block_size;
    ptrdiff_t buffer_bytes_needed = 0;
    ptrdiff_t num_offset_arrays = 0;
    ptrdiff_t alignment = 0;
    for (size_t i = 0; i < Arity; ++i) {
      const auto dtype = data_types_[i];
      if (!dtype.valid()) continue;
      buffer_bytes_needed = RoundUpTo(buffer_bytes_needed, dtype->alignment);
      buffer_bytes_needed += block_size * dtype->size;
      alignment = std::max(alignment, dtype->alignment);
      for (const auto buffer_kind : buffer_kinds[i]) {
        if (buffer_kind == IterationBufferKind::kIndexed) {
          ++num_offset_arrays;
          break;
        }
      }
    }
    ptrdiff_t next_offset_array_offset = buffer_bytes_needed;

    if (num_offset_arrays) {
      buffer_bytes_needed = RoundUpTo(buffer_bytes_needed,
                                      static_cast<ptrdiff_t>(alignof(Index)));
      buffer_bytes_needed += block_size * sizeof(Index) * num_offset_arrays;
      alignment = std::max(alignment, static_cast<ptrdiff_t>(alignof(Index)));
    }

    if (!buffer_bytes_needed) return;

    buffer_ = allocator_.arena()->allocate(buffer_bytes_needed, alignment);
    buffer_size_ = buffer_bytes_needed;
    buffer_alignment_ = alignment;

    ptrdiff_t buffer_offset = 0;

    for (size_t i = 0; i < Arity; ++i) {
      const auto dtype = data_types_[i];
      if (!dtype.valid()) continue;
      buffer_offset = RoundUpTo(buffer_offset, dtype->alignment);
      void* buffer_ptr = buffer_ + buffer_offset;
      dtype->construct(block_size, buffer_ptr);
      buffer_offset += block_size * dtype->size;
      Index* byte_offsets = nullptr;
      for (const auto buffer_kind : buffer_kinds[i]) {
        if (buffer_kind == IterationBufferKind::kIndexed) {
          byte_offsets =
              reinterpret_cast<Index*>(buffer_ + next_offset_array_offset);
          next_offset_array_offset += block_size * sizeof(Index);
          FillOffsetsArrayFromStride(dtype->size,
                                     span(byte_offsets, block_size));
          break;
        }
      }
      for (size_t j = 0; j < NumBufferKinds; ++j) {
        buffer_pointers_[j][i] = GetIterationBufferPointer(
            buffer_kinds[i][j], buffer_ptr, dtype->size, byte_offsets);
      }
    }
  }

  /// Returns a copy of the allocator used to allocate the buffers.
  ArenaAllocator<> get_allocator() const { return allocator_; }

  /// Returns the data types of the buffers that are allocated.
  span<const DataType, Arity> data_types() const { return data_types_; }

  /// Returns the block size of the buffers that are allocated.
  Index block_size() const { return block_size_; }

  /// Returns the buffer pointers.
  ///
  /// If `Initialize` has been called and `data_types()[i].valid() == true`,
  /// `buffer_pointers()[j][i]` is a buffer pointer to the corresponding buffer
  /// of size `block_size()` of kind `buffer_kinds[i][j]`, where `buffer_kinds`
  /// is the argument passed to `Initialize`.  Otherwise,
  /// `buffer_pointers()[j][i]` is unspecified.
  span<std::array<IterationBufferPointer, Arity>, NumBufferKinds>
  buffer_pointers() {
    return buffer_pointers_;
  }

  span<const std::array<IterationBufferPointer, Arity>, NumBufferKinds>
  buffer_pointers() const {
    return buffer_pointers_;
  }

  ~NDIteratorExternalBufferManager() { Free(); }

 private:
  void Free() {
    if (!buffer_) return;

    for (size_t i = 0; i < Arity; ++i) {
      if (data_types_[i].valid()) {
        data_types_[i]->destroy(block_size_, buffer_pointers_[0][i].pointer);
      }
    }
    allocator_.arena()->deallocate(buffer_, buffer_size_, buffer_alignment_);
    buffer_ = nullptr;
  }

  std::array<DataType, Arity> data_types_;
  Index block_size_;
  ArenaAllocator<> allocator_;
  unsigned char* buffer_;
  size_t buffer_size_;
  size_t buffer_alignment_;
  std::array<std::array<IterationBufferPointer, Arity>, NumBufferKinds>
      buffer_pointers_;
};

/// Stores `Arity` `NDIterator` objects and an associated
/// `NDIteratorExternalBufferManager`.
///
/// This can be used to apply an elementwise operation to a collection of
/// read-only/write-only/read-write iterables.
template <std::size_t Arity>
struct NDIteratorsWithManagedBuffers {
  /// Obtains iterators and allocates any necessary external buffers.
  ///
  /// \param iterables Range of size `Arity` supporting `operator[]` with a
  ///     `value_type` that is pointer-like with a pointee that inherits from
  ///     `NDIterable`.
  /// \param layout The layout to use for iteration.  Must be compatible with
  ///     `iterables`, e.g. as computed from an `NDIterablesWithManagedBuffers`
  ///     object with the same `iterables`.
  /// \param allocator The allocator to use to obtain iterators and allocate any
  ///     necessary external buffers.
  template <typename Iterables>
  explicit NDIteratorsWithManagedBuffers(
      const Iterables& iterables,
      NDIterable::IterationBufferKindLayoutView layout,
      ArenaAllocator<> allocator)
      : buffer_manager_(allocator) {
    std::array<NDIterable::IterationBufferConstraint, Arity> buffer_constraints;
    std::array<DataType, Arity> data_types;
    std::array<std::array<IterationBufferKind, 2>, Arity> buffer_kinds;
    for (size_t i = 0; i < Arity; ++i) {
      buffer_constraints[i] =
          iterables[i]->GetIterationBufferConstraint(layout);
      if (buffer_constraints[i].external) {
        data_types[i] = iterables[i]->dtype();
        buffer_kinds[i][0] = buffer_constraints[i].min_buffer_kind;
        buffer_kinds[i][1] = layout.buffer_kind;
      }
    }

    buffer_manager_.Initialize(layout.block_size, data_types, buffer_kinds);
    for (size_t i = 0; i < Arity; ++i) {
      iterators_[i] = iterables[i]->GetIterator(
          {static_cast<const NDIterable::IterationBufferLayoutView&>(layout),
           buffer_constraints[i].external
               ? buffer_constraints[i].min_buffer_kind
               : layout.buffer_kind});
      get_block_pointers_[i] =
          &buffer_manager_
               .buffer_pointers()[buffer_constraints[i].external ? 0 : 1][i];
    }
  }

  /// Returns the allocator.
  ArenaAllocator<> get_allocator() const {
    return buffer_manager_.get_allocator();
  }

  /// Obtains the block of the specified size at the specified `indices` by
  /// calling `GetBlock` on each iterator.
  ///
  /// After calling this method, the block pointers may be obtained by calling
  /// `block_pointers()`.
  ///
  /// \param indices Specifies the position of the block.  Vector of length
  ///     `layout.rank()`, where `layout` is the constructor parameter.
  /// \param block_size The size of the block, must be `<= layout.block_size`,
  ///     where `layout` is the constructor parameter.
  /// \param status[out] Non-null pointer to location where error status may be
  ///     stored if the return value is `false`.  It may be left unmodified even
  ///     if the return value is `false`, in which case a default error should
  ///     be assumed.
  /// \returns `true` if `GetBlock` succeeded (returned `block_size`) for all of
  ///     the iterators, `false` otherwise.
  bool GetBlock(span<const Index> indices, DimensionIndex block_size,
                absl::Status* status) {
    for (size_t i = 0; i < Arity; ++i) {
      if (iterators_[i]->GetBlock(indices, block_size, get_block_pointers_[i],
                                  status) != block_size) {
        return false;
      }
    }
    return true;
  }

  /// Calls `UpdateBlock` on each iterator.
  ///
  /// \param indices Equivalent vector as passed to prior call to `GetBlock`.
  /// \param block_size Same `block_size` as passed to prior call to `GetBlock`.
  /// \param status[out] Non-null pointer to location where error status may be
  ///     stored if the return value is less than `block_size`.  It may be left
  ///     unmodified even if the return value is `false`, in which case a
  ///     default error should be assumed.
  /// \returns The minimum number of elements successfully updated in all
  ///     iterators (equal to `block_size` on success).
  Index UpdateBlock(span<const Index> indices, DimensionIndex block_size,
                    absl::Status* status) {
    Index n = block_size;
    for (size_t i = 0; i < Arity; ++i) {
      n = std::min(n,
                   iterators_[i]->UpdateBlock(indices, block_size,
                                              *get_block_pointers_[i], status));
    }
    return n;
  }

  /// Returns block pointers corresponding to the prior successful call to
  /// `GetBlock`.
  span<const IterationBufferPointer, Arity> block_pointers() const {
    return buffer_manager_.buffer_pointers()[1];
  }

 private:
  NDIteratorExternalBufferManager<Arity, 2> buffer_manager_;
  std::array<NDIterator::Ptr, Arity> iterators_;
  std::array<IterationBufferPointer*, Arity> get_block_pointers_;
};

/// Convenience class that combines an `NDIterationInfo`,
/// `NDIteratorsWithManagedBuffers` and `NDIterationPositionStepper`.
///
/// Example usage:
///
///     Arena arena;
///     NDIterable::Ptr iterable_a = ...;
///     NDIterable::Ptr iterable_b = ...;
///     ElementwiseClosure<2, absl::Status*> closure = ...;
///     MultiNDIterator<2> multi_iterator(
///         shape, constraints, {{ iterable_a.get(), iterable_b.get() }},
///         &arena);
///     absl::Status status;
///     for (Index block_size = multi_iterator.ResetAtBeginning();
///          block_size;
///          block_size = multi_iterator.StepForward(block_size)) {
///       if (!multi_iterator.GetBlock(block_size, &status)) {
///         // handle error
///         break;
///       }
///       Index n = InvokeElementwiseClosure(closure, block_size,
///                                         multi_iterator.block_pointers(),
///                                         &status);
///       if (multi_iterator.UpdateBlock(n, &status) != n) {
///         // handle error
///         break;
///       }
///       if (n != block_size) {
///         // handle error
///         break;
///       }
///     }
template <std::ptrdiff_t Arity, bool Full = false>
struct MultiNDIterator : public NDIterationInfo<Full>,
                         public NDIteratorsWithManagedBuffers<Arity>,
                         public NDIterationPositionStepper {
  using Iterables = std::array<const NDIterable*, Arity>;

  using NDIterationInfo<Full>::block_size;
  using NDIterationInfo<Full>::shape;

  /// Computes a layout for the specified `iterables`, obtains iterators, and
  /// allocates any necessary external buffers.
  MultiNDIterator(span<const Index> shape, IterationConstraints constraints,
                  Iterables iterables, ArenaAllocator<> allocator)
      : NDIterationInfo<Full>(
            NDIterablesWithManagedBuffers<Iterables>{iterables}, shape,
            constraints),
        NDIteratorsWithManagedBuffers<Arity>(
            iterables, this->buffer_layout_view(), allocator),
        NDIterationPositionStepper(this->iteration_shape, this->block_size) {}

  bool GetBlock(Index cur_size, absl::Status* status) {
    return NDIteratorsWithManagedBuffers<Arity>::GetBlock(
        this->position(), this->block_size, status);
  }

  Index UpdateBlock(Index cur_size, absl::Status* status) {
    return NDIteratorsWithManagedBuffers<Arity>::UpdateBlock(
        this->position(), this->block_size, status);
  }
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_NDITERABLE_BUFFER_MANAGEMENT_H_
