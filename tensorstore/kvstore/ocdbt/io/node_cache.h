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

#ifndef TENSORSTORE_KVSTORE_OCDBT_IO_NODE_CACHE_H_
#define TENSORSTORE_KVSTORE_OCDBT_IO_NODE_CACHE_H_

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/time/time.h"
#include "tensorstore/context.h"
#include "tensorstore/internal/cache/cache.h"
#include "tensorstore/internal/cache/kvs_backed_cache.h"
#include "tensorstore/internal/cache_key/cache_key.h"
#include "tensorstore/internal/concurrency_resource.h"
#include "tensorstore/internal/data_copy_concurrency_resource.h"
#include "tensorstore/internal/estimate_heap_usage/estimate_heap_usage.h"
#include "tensorstore/internal/estimate_heap_usage/std_variant.h"
#include "tensorstore/internal/estimate_heap_usage/std_vector.h"
#include "tensorstore/kvstore/ocdbt/format/btree.h"
#include "tensorstore/kvstore/ocdbt/format/indirect_data_reference.h"
#include "tensorstore/kvstore/ocdbt/format/version_tree.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_ocdbt {

template <typename Derived, typename T>
class DecodedIndirectDataCache
    : public internal::KvsBackedCache<DecodedIndirectDataCache<Derived, T>,
                                      internal::AsyncCache> {
  using Base = internal::KvsBackedCache<DecodedIndirectDataCache<Derived, T>,
                                        internal::AsyncCache>;

 public:
  explicit DecodedIndirectDataCache(kvstore::DriverPtr kvstore_driver,
                                    Executor executor)
      : Base(std::move(kvstore_driver)), executor_(std::move(executor)) {}

  using ReadData = T;

  class Entry : public Base::Entry {
   public:
    using OwningCache = Derived;
    using typename Base::Entry::DecodeReceiver;

    std::size_t ComputeReadDataSizeInBytes(const void* read_data) override {
      return internal::EstimateHeapUsage(
          *static_cast<const ReadData*>(read_data));
    }

    void DoDecode(std::optional<absl::Cord> value,
                  typename Base::Entry::DecodeReceiver receiver) override {
      if (!value) {
        execution::set_error(receiver, absl::NotFoundError(""));
        return;
      }
      IndirectDataReference ref;
      ABSL_CHECK(ref.DecodeCacheKey(this->key()));

      GetOwningCache(*this).executor()(
          [value = std::move(*value), base_path = ref.file_id.base_path,
           receiver = std::move(receiver)]() mutable {
            auto read_data = std::make_shared<T>();
            TENSORSTORE_ASSIGN_OR_RETURN(
                *read_data, Derived::Decode(value, base_path),
                static_cast<void>(execution::set_error(receiver, _)));
            execution::set_value(receiver, std::move(read_data));
          });
    }
  };

  using typename Base::TransactionNode;

  Entry* DoAllocateEntry() final { return new Entry; }
  std::size_t DoGetSizeofEntry() final { return sizeof(Entry); }
  typename Base::TransactionNode* DoAllocateTransactionNode(
      internal::AsyncCache::Entry& entry) final {
    return new TransactionNode(static_cast<Entry&>(entry));
  }

  internal::PinnedCacheEntry<DecodedIndirectDataCache<Derived, T>> GetEntry(
      const IndirectDataReference& ref) {
    return GetCacheEntry(this, ref.EncodeCacheKey());
  }

  Future<const std::shared_ptr<const T>> ReadEntry(
      const IndirectDataReference& ref,
      absl::Time staleness_bound = absl::InfinitePast()) {
    auto entry = GetEntry(ref);
    auto* entry_ptr = entry.get();
    return PromiseFuturePair<std::shared_ptr<const T>>::LinkValue(
               [entry = std::move(entry)](
                   Promise<std::shared_ptr<const T>> promise,
                   ReadyFuture<const void> future) {
                 promise.SetResult(
                     internal::AsyncCache::ReadLock<T>(*entry).shared_data());
               },
               entry_ptr->Read({staleness_bound}))
        .future;
  }

  const Executor& executor() { return executor_; }

  Executor executor_;
};

template <typename Derived>
internal::CachePtr<Derived> GetDecodedIndirectDataCache(
    internal::CachePool* pool, const kvstore::DriverPtr& kvstore_driver,
    const Context::Resource<internal::DataCopyConcurrencyResource>&
        data_copy_concurrency) {
  std::string cache_identifier;
  internal::EncodeCacheKey(&cache_identifier, data_copy_concurrency,
                           kvstore_driver);
  return internal::GetCache<Derived>(pool, cache_identifier, [&] {
    return std::make_unique<Derived>(kvstore_driver,
                                     data_copy_concurrency->executor);
  });
}

class BtreeNodeCache
    : public DecodedIndirectDataCache<BtreeNodeCache, BtreeNode> {
  using Base = DecodedIndirectDataCache<BtreeNodeCache, BtreeNode>;

 public:
  using Base::Base;

  static Result<BtreeNode> Decode(const absl::Cord& encoded,
                                  const BasePath& base_path) {
    return DecodeBtreeNode(encoded, base_path);
  }
};

extern template class DecodedIndirectDataCache<BtreeNodeCache, BtreeNode>;

class VersionTreeNodeCache
    : public DecodedIndirectDataCache<VersionTreeNodeCache, VersionTreeNode> {
  using Base = DecodedIndirectDataCache<VersionTreeNodeCache, VersionTreeNode>;

 public:
  using Base::Base;

  static Result<VersionTreeNode> Decode(const absl::Cord& encoded,
                                        const BasePath& base_path) {
    return DecodeVersionTreeNode(encoded, base_path);
  }
};

extern template class DecodedIndirectDataCache<VersionTreeNodeCache,
                                               VersionTreeNode>;

}  // namespace internal_ocdbt
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_OCDBT_IO_NODE_CACHE_H_
