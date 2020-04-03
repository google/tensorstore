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

#ifndef TENSORSTORE_INTERNAL_CHUNK_H_
#define TENSORSTORE_INTERNAL_CHUNK_H_

/// \file
/// Defines the ReadChunk and WriteChunk abstractions, which represent a portion
/// of a TensorStore that may be read/written atomically.
///
/// Both ReadChunk and WriteChunk are defined in terms of a "cell domain".
/// Typically, a transform from a known index space to this "cell domain" is
/// provided along with the ReadChunk/WriteChunk object.

#include "tensorstore/data_type.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/internal/arena.h"
#include "tensorstore/internal/element_copy_function.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/internal/nditerable.h"
#include "tensorstore/internal/poly.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/sender.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal {

struct ReadChunk {
  struct AcquireReadLock {};
  struct ReleaseReadLock {};
  using Impl =
      Poly<sizeof(void*) * 2,
           /*Copyable=*/true,  //
           /// Acquires a read lock on the chunk, and returns a readable view of
           /// the data.
           ///
           /// \param chunk_transform Transform with a range that is a subset of
           ///     `transform`.
           /// \param arena Non-null pointer to allocation arena.  Must remain
           ///     valid until after `ReleaseReadLock` is called.
           /// \returns An NDIterable with a shape of
           ///     `chunk_transform.input_shape()`.
           Result<NDIterable::Ptr>(
               AcquireReadLock, IndexTransform<> chunk_transform, Arena* arena)
               const,

           /// Release the read lock acquired by a prior call to the
           /// AcquireReadLock method.  This must be called exactly once after
           /// each successful call to `AcquireReadLock`.  If `AcquireReadLock`
           /// returns an error `Status`, this method must not be called.
           ///
           /// The `NDIterable::Ptr` returned by `AcquireReadLock` must be
           /// destroyed before calling this method.
           void(ReleaseReadLock) const>;

  struct ReadLockReleaserDeleter {
    void operator()(Impl* impl) const { (*impl)(ReleaseReadLock{}); }
  };

  using ReadLockReleaser = std::unique_ptr<Impl, ReadLockReleaserDeleter>;

  /// Type-erased chunk implementation.  In the case of the chunks produced by
  /// `ChunkCache::Read`, for example, the contained object holds a
  /// reference-counted pointer to the underlying cache entry from which the
  /// data may be read.
  Impl impl;

  /// Transform from the "cell domain" to the chunk.
  IndexTransform<> transform;
};

struct WriteChunk {
  struct AcquireWriteLock {};
  struct ReleaseWriteLock {};
  using Impl =
      Poly<sizeof(void*) * 2,
           /*Copyable=*/true,  //

           /// Acquires a write lock on the chunk, and returns a write-only
           /// iterable that may be used to write data to the chunk.
           ///
           /// The returned `NDIterable` may be iterated using any single
           /// compatible layout, but must be written in such a way the set of
           /// modified positions is equal to a prefix of positions accessed by
           /// a single compatible layout.
           ///
           /// \param chunk_transform Transform with a range that is a subset of
           ///     `transform`.
           /// \param arena Non-null pointer to allocation arena that may be
           ///     used for allocating memory.  Must remain valid until after
           ///     `ReleaseWriteLock` is called.
           /// \returns An NDIterable with a shape of
           ///     `chunk_transform.input_shape()`.
           Result<NDIterable::Ptr>(
               AcquireWriteLock, IndexTransform<> chunk_transform, Arena* area)
               const,

           /// Release the write lock acquired by a prior call to the
           /// `AcquireWriteLock` method.  This must be called exactly once
           /// after each successful call to `AcquireWriteLock`.  If
           /// `AcquireWriteLock` returns an error `Status`, this method must
           /// not be called.
           ///
           /// The `NDIterable::Ptr` returned by `AcquireWriteLock` must be
           /// destroyed before calling this method.
           ///
           /// \param chunk_transform Same transform supplied to prior call to
           ///     `AcquireWriteLock`.
           /// \param layout Iteration layout used for writing to the
           ///     `NDIterable` returned by `AcquireWriteLock`.
           /// \param write_end_position One past the last position (with
           ///     respect to `layout`) that was modified.
           /// \param arena Non-null pointer to allocation arena that may be
           ///     used for allocating memory.
           /// \returns A Future that becomes ready when the write has been
           /// committed
           ///     (specifically, when durability is guaranteed), or an error
           ///     occurs. In the case of an error, the write may be lost.
           ///     `Force` must be called to ensure the future eventually
           ///     becomes ready.  A return value of a null (invalid) `Future`
           ///     is equivalent to a ready future in the success state.
           Future<const void>(
               ReleaseWriteLock, IndexTransformView<> chunk_transform,
               NDIterable::IterationLayoutView layout,
               span<const Index> write_end_position, Arena* arena) const>;

  /// Type-erased chunk implementation.  In the case of the chunks produced by
  /// `ChunkCache::Write`, for example, the contained object holds a
  /// reference-counted pointer to the underlying cache entry to which the data
  /// may be written.
  Impl impl;

  /// Transform from the "cell domain" to the chunk.
  IndexTransform<> transform;
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_CHUNK_H_
