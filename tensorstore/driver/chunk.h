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

#include <mutex>

#include "absl/status/status.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/internal/arena.h"
#include "tensorstore/internal/lock_collection.h"
#include "tensorstore/internal/nditerable.h"
#include "tensorstore/internal/poly/poly.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal {

struct ReadChunk {
  struct BeginRead {};
  using Impl = poly::Poly<
      sizeof(void*) * 2,
      /*Copyable=*/true,  //
      /// Registers any necessary locks that must be acquired before
      /// calling the `BeginRead` overload.
      ///
      /// When operating on multiple chunks at once (e.g. reading from one
      /// chunk and writing to another), this method will first be called
      /// on all chunks to collect the required locks, then all requested
      /// locks will be acquired in a deadlock-free way.
      absl::Status(LockCollection& lock_collection),

      /// Returns a readable view of the data.
      ///
      /// The locks registered by the `LockCollection` overload above will
      /// be held when this function is called and won't be released until
      /// after the returned `NDIterable` is destroyed.
      ///
      /// \param chunk_transform Transform with a range that is a subset of
      ///     `transform`.
      /// \param arena Non-null pointer to allocation arena.  Must remain
      ///     valid until the returned `NDIterable` is destroyed.
      /// \returns An NDIterable with a shape of
      ///     `chunk_transform.input_shape()`.
      Result<NDIterable::Ptr>(BeginRead, IndexTransform<> chunk_transform,
                              Arena* arena)>;

  /// Type-erased chunk implementation.  In the case of the chunks produced by
  /// `ChunkCache::Read`, for example, the contained object holds a
  /// reference-counted pointer to the underlying cache entry from which the
  /// data may be read.
  Impl impl;

  /// Transform from the "cell domain" to the chunk.
  IndexTransform<> transform;
};

struct WriteChunk {
  struct BeginWrite {};
  struct EndWrite {};

  struct [[nodiscard]] EndWriteResult {
    /// Indicates an error recording write operation in memory.  Errors
    /// committing the write operation should instead be reported using
    /// `commit_future`.
    absl::Status copy_status;

    /// For write operations without an explicit transaction, specifies a Future
    /// that becomes ready when the write has been committed (specifically, when
    /// durability is guaranteed), or an error occurs.  In the case of an error,
    /// the write may be lost.  `Force` must be called to ensure the future
    /// eventually becomes ready.
    ///
    /// A null (invalid) `Future` is equivalent to `MakeReadyFuture()`, and may
    /// be returned if this write operation had no effect and no changes need to
    /// be committed.
    ///
    /// For write operations with an explicit transaction, this is assumed to
    /// equal the future associated with the transaction, and is ignored.
    Future<const void> commit_future;
  };

  using Impl = poly::Poly<
      sizeof(void*) * 2,
      /*Copyable=*/true,  //

      /// Registers any necessary locks that must be acquired before
      /// calling the `BeginWrite` overload.
      ///
      /// When operating on multiple chunks at once (e.g. reading from one
      /// chunk and writing to another), this method will first be called
      /// on all chunks to collect the required locks, then all requested
      /// locks will be acquired in a deadlock-free way.
      absl::Status(LockCollection& lock_collection),

      /// Returns a write-only iterable that may be used to write data to
      /// the chunk.
      ///
      /// The locks registered by the `LockCollection` overload above will
      /// be held when this function is called and won't be released until
      /// after `EndWrite` is called.
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
      ///     `EndWrite` is called.
      /// \returns An NDIterable with a shape of
      ///     `chunk_transform.input_shape()`.
      Result<NDIterable::Ptr>(BeginWrite, IndexTransform<> chunk_transform,
                              Arena* area),

      /// Finalize the write started by a successful call to `BeginWrite`.
      /// This must be called exactly once after each successful call to
      /// `BeginWrite`.  If `BeginWrite` returns an error `absl::Status`, this
      /// method must not be called.
      ///
      /// The `NDIterable::Ptr` returned by `BeginWrite` must be destroyed
      /// before calling this method.
      ///
      /// \param chunk_transform Same transform supplied to prior call to
      ///     `BeginWrite`.
      /// \param layout Iteration layout used for writing to the
      ///     `NDIterable` returned by `BeginWrite`.
      /// \param write_end_position One past the last position (with
      ///     respect to `layout`) that was modified.
      /// \param arena Non-null pointer to allocation arena that may be
      ///     used for allocating memory.
      EndWriteResult(EndWrite, IndexTransformView<> chunk_transform,
                     NDIterable::IterationLayoutView layout,
                     span<const Index> write_end_position, Arena* arena)>;

  /// Type-erased chunk implementation.  In the case of the chunks produced by
  /// `ChunkCache::Write`, for example, the contained object holds a
  /// reference-counted pointer to the underlying cache entry to which the data
  /// may be written.
  Impl impl;

  /// Transform from the "cell domain" to the chunk.
  IndexTransform<> transform;
};

/// Attempts to lock one or more `ReadChunk`/`WriteChunk` objects.
///
/// If registering a chunk with the lock collection fails, the error propagates
/// immediately.
///
/// If the lock collection fails to acquire a lock, we retry.
///
/// \tparam ChunkImpl Must be either `ReadChunk::Impl` or `WriteChunk::Impl`.
template <typename... ChunkImpl>
Result<std::unique_lock<LockCollection>> LockChunks(
    LockCollection& lock_collection, ChunkImpl&... chunk_impl) {
  std::unique_lock<LockCollection> guard(lock_collection, std::defer_lock);
  while (true) {
    // Attempt to register each chunk with the `lock_collection`.
    if (absl::Status status;
        !((status = chunk_impl(lock_collection)).ok() && ...)) {
      return status;
    }
    if (guard.try_lock()) return guard;
    // Locking failed.  Clear the lock collection and re-register the chunks,
    // because the locks to be registered may have changed (in order to avoid
    // failing again).  For example, failure may be due to an
    // `AsyncCache::TransactionNode` having been revoked, and in that case a new
    // transaction node will be obtained when the chunk is re-registered.  The
    // lock function registered with the `lock_collection` cannot itself obtain
    // a new TransactionNode, because that could lead to deadlock, since the
    // lock ordering and deduplication was based on the original transaction
    // node.
    lock_collection.clear();
  }
}

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_CHUNK_H_
