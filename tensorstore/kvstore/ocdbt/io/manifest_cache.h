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

#include <functional>
#include <memory>
#include <optional>
#include <utility>

#include "absl/strings/cord.h"
#include "tensorstore/internal/cache/async_cache.h"
#include "tensorstore/internal/cache/kvs_backed_cache.h"
#include "tensorstore/internal/estimate_heap_usage/estimate_heap_usage.h"
#include "tensorstore/kvstore/ocdbt/format/manifest.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace internal_ocdbt {

/// Writeback cache used for reading and writing the manifest.
class ManifestCache
    : public internal::KvsBackedCache<ManifestCache, internal::AsyncCache> {
  using Base = internal::KvsBackedCache<ManifestCache, internal::AsyncCache>;

 public:
  using ReadData = Manifest;

  explicit ManifestCache(kvstore::DriverPtr kvstore_driver, Executor executor)
      : Base(std::move(kvstore_driver)), executor_(std::move(executor)) {}

  // Function that asynchronously computes a new manifest from an existing one.
  using UpdateFunction = std::function<Future<std::shared_ptr<const Manifest>>(
      std::shared_ptr<const Manifest> existing)>;

  class Entry : public Base::Entry {
   public:
    using OwningCache = ManifestCache;

    std::size_t ComputeReadDataSizeInBytes(const void* read_data) final;

    void DoDecode(std::optional<absl::Cord> value,
                  DecodeReceiver receiver) final;

    void DoEncode(std::shared_ptr<const ReadData> read_data,
                  EncodeReceiver receiver) final;

    // Performs an atomic read-modify-write operation on the manifest.
    //
    // The `update_function` will be called at least once to compute the
    // modified manifest.  In the case that the cached manifest is out-of-date
    // or there are concurrent modifications, the `update_function` may be
    // called multiple times.
    //
    // The returned `Future` resolves to the new manifest and its timestamp on
    // success.
    Future<const ManifestWithTime> Update(UpdateFunction update_function);
  };

  class TransactionNode : public Base::TransactionNode {
   public:
    using OwningCache = ManifestCache;

    using Base::TransactionNode::TransactionNode;

    void WritebackSuccess(ReadState&& read_state) final;
    void DoApply(ApplyOptions options, ApplyReceiver receiver) final;

    UpdateFunction update_function;
    Promise<ManifestWithTime> promise;
  };

  Entry* DoAllocateEntry() final;
  std::size_t DoGetSizeofEntry() final;
  TransactionNode* DoAllocateTransactionNode(AsyncCache::Entry& entry) final;

  Executor executor_;

  const Executor& executor() { return executor_; }
};

}  // namespace internal_ocdbt
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_OCDBT_IO_MANIFEST_CACHE_H_
