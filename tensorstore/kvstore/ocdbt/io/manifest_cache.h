// Copyright 2022 The TensorStore Authors
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

#ifndef TENSORSTORE_KVSTORE_OCDBT_IO_MANIFEST_CACHE_H_
#define TENSORSTORE_KVSTORE_OCDBT_IO_MANIFEST_CACHE_H_

#include <stddef.h>

#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "tensorstore/internal/cache/async_cache.h"
#include "tensorstore/internal/estimate_heap_usage/estimate_heap_usage.h"
#include "tensorstore/internal/estimate_heap_usage/std_vector.h"
#include "tensorstore/kvstore/ocdbt/format/manifest.h"
#include "tensorstore/kvstore/ocdbt/format/version_tree.h"
#include "tensorstore/kvstore/ocdbt/io_handle.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace internal_ocdbt {

/// Writeback cache used for reading and writing the manifest.
class ManifestCache : public internal::AsyncCache {
  using Base = internal::AsyncCache;

 public:
  using ReadData = Manifest;

  explicit ManifestCache(kvstore::DriverPtr kvstore_driver, Executor executor)
      : kvstore_driver_(std::move(kvstore_driver)),
        executor_(std::move(executor)) {}

  class Entry : public Base::Entry {
   public:
    using OwningCache = ManifestCache;

    size_t ComputeReadDataSizeInBytes(const void* read_data) final;

    void DoRead(AsyncCacheReadRequest request) final;

    // Performs an atomic read-modify-write operation on the manifest.
    //
    // If the existing manifest (as of `staleness_bound`) is not equal to
    // `old_manifest`, resolves with `TryUpdateManifestResult::success = false`.
    //
    // Otherwise, atomically writes `new_manifest` in its place (subject to the
    // atomic limitations of the underlying base kvstore).
    //
    // Fails with an error status if an I/O error occurs in the underlying base
    // kvstore.
    Future<TryUpdateManifestResult> TryUpdate(
        std::shared_ptr<const Manifest> old_manifest,
        std::shared_ptr<const Manifest> new_manifest,
        absl::Time staleness_bound);
  };

  class TransactionNode : public Base::TransactionNode {
   public:
    using OwningCache = ManifestCache;

    using Base::TransactionNode::TransactionNode;

    absl::Status DoInitialize(internal::OpenTransactionPtr& transaction) final;
    void DoRead(AsyncCacheReadRequest request) final;
    void Commit() final;

    void WritebackSuccess(ReadState&& read_state) final;

    std::shared_ptr<const Manifest> old_manifest, new_manifest;
    absl::Time staleness_bound;
    Promise<TryUpdateManifestResult> promise;
  };

  Entry* DoAllocateEntry() final;
  size_t DoGetSizeofEntry() final;
  TransactionNode* DoAllocateTransactionNode(AsyncCache::Entry& entry) final;

  kvstore::DriverPtr kvstore_driver_;
  Executor executor_;

  const Executor& executor() { return executor_; }
};

// Cache for the `manifest.XXXXXXXXXXXXXXXX` files.
class NumberedManifestCache : public internal::AsyncCache {
  using Base = internal::AsyncCache;

 public:
  struct NumberedManifest {
    // Most recent manifest observed.
    std::shared_ptr<const Manifest> manifest;
    // List of versions present, needed to delete the previous version.
    std::vector<GenerationNumber> versions_present;

    constexpr static auto ApplyMembers = [](auto&& x, auto f) {
      return f(x.manifest, x.versions_present);
    };
  };

  using ReadData = NumberedManifest;

  explicit NumberedManifestCache(kvstore::DriverPtr kvstore_driver,
                                 Executor executor)
      : kvstore_driver_(std::move(kvstore_driver)),
        executor_(std::move(executor)) {}

  class Entry : public Base::Entry {
   public:
    using OwningCache = NumberedManifestCache;

    size_t ComputeReadDataSizeInBytes(const void* read_data) final;

    void DoRead(AsyncCacheReadRequest request) final;

    // Attempts to write a new manifest.
    //
    // The returned `Future` becomes ready when the attempt completes, but a
    // successful return does not indicate that the update was successful.  The
    // caller must determine that by re-reading the new manifest.
    //
    // A previous read request on this entry must have already completed.
    //
    // Args:
    //   new_manifest: New manifest, must be non-null.
    Future<TryUpdateManifestResult> TryUpdate(
        std::shared_ptr<const Manifest> new_manifest);
  };

  class TransactionNode : public Base::TransactionNode {
   public:
    using OwningCache = NumberedManifestCache;

    using Base::TransactionNode::TransactionNode;

    absl::Status DoInitialize(internal::OpenTransactionPtr& transaction) final;
    void DoRead(AsyncCacheReadRequest request) final;
    void Commit() final;

    std::shared_ptr<const Manifest> new_manifest;
    Promise<TryUpdateManifestResult> promise;
  };

  Entry* DoAllocateEntry() final;
  size_t DoGetSizeofEntry() final;
  TransactionNode* DoAllocateTransactionNode(AsyncCache::Entry& entry) final;

  kvstore::DriverPtr kvstore_driver_;
  Executor executor_;

  const Executor& executor() { return executor_; }
};

}  // namespace internal_ocdbt
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_OCDBT_IO_MANIFEST_CACHE_H_
