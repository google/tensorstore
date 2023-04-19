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

#include "tensorstore/kvstore/ocdbt/io/io_handle_impl.h"

#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

#include "absl/strings/cord.h"
#include "absl/time/time.h"
#include "tensorstore/context.h"
#include "tensorstore/internal/cache/async_cache.h"
#include "tensorstore/internal/cache/cache.h"
#include "tensorstore/internal/cache_key/cache_key.h"
#include "tensorstore/internal/cache_key/std_optional.h"
#include "tensorstore/internal/concurrency_resource.h"
#include "tensorstore/internal/data_copy_concurrency_resource.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/ocdbt/config.h"
#include "tensorstore/kvstore/ocdbt/format/btree.h"
#include "tensorstore/kvstore/ocdbt/format/indirect_data_reference.h"
#include "tensorstore/kvstore/ocdbt/format/manifest.h"
#include "tensorstore/kvstore/ocdbt/format/version_tree.h"
#include "tensorstore/kvstore/ocdbt/io/coalesce_kvstore.h"
#include "tensorstore/kvstore/ocdbt/io/indirect_data_kvstore_driver.h"
#include "tensorstore/kvstore/ocdbt/io/indirect_data_writer.h"
#include "tensorstore/kvstore/ocdbt/io/manifest_cache.h"
#include "tensorstore/kvstore/ocdbt/io/node_cache.h"
#include "tensorstore/kvstore/ocdbt/io_handle.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal_ocdbt {

class IoHandleImpl : public IoHandle {
 public:
  using Ptr = internal::IntrusivePtr<const IoHandleImpl>;
  KvStore base_kvstore_;
  internal::PinnedCacheEntry<ManifestCache> manifest_cache_entry_;
  internal::CachePtr<BtreeNodeCache> btree_node_cache_;
  internal::CachePtr<VersionTreeNodeCache> version_tree_node_cache_;
  IndirectDataWriterPtr indirect_data_writer_;
  kvstore::DriverPtr indirect_data_kvstore_driver_;

  Future<const std::shared_ptr<const BtreeNode>> GetBtreeNode(
      const IndirectDataReference& ref) const final {
    return btree_node_cache_->ReadEntry(ref);
  }

  Future<const std::shared_ptr<const VersionTreeNode>> GetVersionTreeNode(
      const IndirectDataReference& ref) const final {
    return version_tree_node_cache_->ReadEntry(ref);
  }

  Future<const ManifestWithTime> GetManifest(
      absl::Time staleness_bound) const final {
    return PromiseFuturePair<ManifestWithTime>::LinkValue(
               [self = Ptr(this)](Promise<ManifestWithTime> promise,
                                  ReadyFuture<const void> future) {
                 ManifestWithTime manifest_with_time;
                 {
                   internal::AsyncCache::ReadLock<Manifest> lock{
                       *self->manifest_cache_entry_};
                   manifest_with_time.manifest = lock.shared_data();
                   manifest_with_time.time = lock.stamp().time;
                 }
                 if (manifest_with_time.manifest) {
                   TENSORSTORE_RETURN_IF_ERROR(
                       self->config_state->ValidateNewConfig(
                           manifest_with_time.manifest->config),
                       static_cast<void>(promise.SetResult(_)));
                 }
                 promise.SetResult(std::move(manifest_with_time));
               },
               manifest_cache_entry_->Read(staleness_bound))
        .future;
  }

  Future<kvstore::ReadResult> ReadIndirectData(
      const IndirectDataReference& ref,
      kvstore::ReadOptions read_options) const final {
    return indirect_data_kvstore_driver_->Read(ref.EncodeCacheKey(),
                                               std::move(read_options));
  }

  Future<const ManifestWithTime> ReadModifyWriteManifest(
      ManifestUpdateFunction update_function) const final {
    return manifest_cache_entry_->Update(
        [config_state = this->config_state,
         update_function = std::move(update_function)](
            std::shared_ptr<const Manifest> existing_manifest)
            -> Future<std::shared_ptr<const Manifest>> {
          if (existing_manifest) {
            TENSORSTORE_RETURN_IF_ERROR(
                config_state->ValidateNewConfig(existing_manifest->config));
          }
          return update_function(std::move(existing_manifest));
        });
  }

  Future<const void> WriteData(absl::Cord data,
                               IndirectDataReference& ref) const final {
    return internal_ocdbt::Write(*indirect_data_writer_, std::move(data), ref);
  }

  std::string DescribeLocation() const final {
    return base_kvstore_.driver->DescribeKey(base_kvstore_.path);
  }
};

IoHandle::Ptr MakeIoHandle(
    const Context::Resource<tensorstore::internal::DataCopyConcurrencyResource>&
        data_copy_concurrency,
    internal::CachePool& cache_pool, const KvStore& base_kvstore,
    ConfigStatePtr config_state,
    std::optional<int64_t> max_read_coalescing_overhead_bytes_per_request) {
  // Maybe wrap the base driver in CoalesceKvStoreDriver.
  kvstore::DriverPtr driver_with_optional_coalescing =
      max_read_coalescing_overhead_bytes_per_request.has_value()
          ? MakeCoalesceKvStoreDriver(
                base_kvstore.driver,
                *max_read_coalescing_overhead_bytes_per_request)
          : base_kvstore.driver;
  auto impl = internal::MakeIntrusivePtr<IoHandleImpl>();
  impl->base_kvstore_ = base_kvstore;
  impl->config_state = std::move(config_state);
  impl->executor = data_copy_concurrency->executor;
  auto data_kvstore =
      kvstore::KvStore(driver_with_optional_coalescing, base_kvstore.path);
  impl->indirect_data_writer_ =
      internal_ocdbt::MakeIndirectDataWriter(data_kvstore);
  impl->indirect_data_kvstore_driver_ =
      internal_ocdbt::MakeIndirectDataKvStoreDriver(data_kvstore);
  impl->btree_node_cache_ =
      internal_ocdbt::GetDecodedIndirectDataCache<BtreeNodeCache>(
          cache_pool, impl->indirect_data_kvstore_driver_,
          data_copy_concurrency);
  impl->version_tree_node_cache_ =
      tensorstore::internal_ocdbt::GetDecodedIndirectDataCache<
          tensorstore::internal_ocdbt::VersionTreeNodeCache>(
          cache_pool, impl->indirect_data_kvstore_driver_,
          data_copy_concurrency);
  std::string manifest_cache_identifier;
  internal::EncodeCacheKey(&manifest_cache_identifier, data_copy_concurrency,
                           base_kvstore.driver,
                           max_read_coalescing_overhead_bytes_per_request);
  auto manifest_cache =
      cache_pool.GetCache<tensorstore::internal_ocdbt::ManifestCache>(
          manifest_cache_identifier, [&] {
            return std::make_unique<ManifestCache>(
                base_kvstore.driver, data_copy_concurrency->executor);
          });
  impl->manifest_cache_entry_ = tensorstore::internal::GetCacheEntry(
      manifest_cache,
      tensorstore::internal_ocdbt::GetManifestPath(base_kvstore.path));
  return impl;
}

}  // namespace internal_ocdbt
}  // namespace tensorstore
