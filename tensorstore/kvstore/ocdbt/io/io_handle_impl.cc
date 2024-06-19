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

#include <stddef.h>

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tensorstore/context.h"
#include "tensorstore/internal/cache/async_cache.h"
#include "tensorstore/internal/cache/cache.h"
#include "tensorstore/internal/cache_key/cache_key.h"
#include "tensorstore/internal/cache_key/std_optional.h"  // IWYU pragma: keep
#include "tensorstore/internal/concurrency_resource.h"
#include "tensorstore/internal/data_copy_concurrency_resource.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/ocdbt/config.h"
#include "tensorstore/kvstore/ocdbt/format/btree.h"
#include "tensorstore/kvstore/ocdbt/format/config.h"
#include "tensorstore/kvstore/ocdbt/format/indirect_data_reference.h"
#include "tensorstore/kvstore/ocdbt/format/manifest.h"
#include "tensorstore/kvstore/ocdbt/format/version_tree.h"
#include "tensorstore/kvstore/ocdbt/io/coalesce_kvstore.h"
#include "tensorstore/kvstore/ocdbt/io/indirect_data_kvstore_driver.h"
#include "tensorstore/kvstore/ocdbt/io/indirect_data_writer.h"
#include "tensorstore/kvstore/ocdbt/io/manifest_cache.h"
#include "tensorstore/kvstore/ocdbt/io/node_cache.h"
#include "tensorstore/kvstore/ocdbt/io_handle.h"
#include "tensorstore/kvstore/ocdbt/non_distributed/read_version.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal_ocdbt {

namespace {
ABSL_CONST_INIT internal_log::VerboseFlag ocdbt_logging("ocdbt");

}  // namespace

class IoHandleImpl : public IoHandle {
 public:
  using Ptr = internal::IntrusivePtr<const IoHandleImpl>;
  KvStore base_kvstore_;
  internal::PinnedCacheEntry<ManifestCache> manifest_cache_entry_;
  internal::PinnedCacheEntry<NumberedManifestCache>
      numbered_manifest_cache_entry_;
  internal::CachePtr<BtreeNodeCache> btree_node_cache_;
  internal::CachePtr<VersionTreeNodeCache> version_tree_node_cache_;
  IndirectDataWriterPtr indirect_data_writer_[kNumIndirectDataKinds];
  kvstore::DriverPtr indirect_data_kvstore_driver_;

  mutable absl::Mutex manifest_mutex_;
  mutable ManifestWithTime cached_top_level_manifest_{nullptr,
                                                      absl::InfinitePast()};
  mutable ManifestWithTime cached_numbered_manifest_{nullptr,
                                                     absl::InfinitePast()};
  Future<const std::shared_ptr<const BtreeNode>> GetBtreeNode(
      const IndirectDataReference& ref) const final {
    return btree_node_cache_->ReadEntry(ref);
  }

  Future<const std::shared_ptr<const VersionTreeNode>> GetVersionTreeNode(
      const IndirectDataReference& ref) const final {
    return version_tree_node_cache_->ReadEntry(ref);
  }

  // Returns the cached "top-level" manifest at `manifest.ocdbt`.
  absl::Status GetCachedTopLevelManifest(
      ManifestWithTime& manifest_with_time) const {
    ManifestWithTime new_manifest_with_time;
    {
      internal::AsyncCache::ReadLock<Manifest> lock{*manifest_cache_entry_};
      new_manifest_with_time.manifest = lock.shared_data();
      new_manifest_with_time.time = lock.stamp().time;
    }

    if (new_manifest_with_time.time == absl::InfinitePast()) {
      manifest_with_time = std::move(new_manifest_with_time);
      return absl::OkStatus();
    }

    absl::MutexLock lock(&manifest_mutex_);
    // Only validate the new manifest if it is not the same as the existing
    // manifest.
    if (new_manifest_with_time.manifest !=
        cached_top_level_manifest_.manifest) {
      auto* new_manifest = new_manifest_with_time.manifest.get();
      if (new_manifest) {
        TENSORSTORE_RETURN_IF_ERROR(
            config_state->ValidateNewConfig(new_manifest->config));
      }
    }
    cached_top_level_manifest_ = new_manifest_with_time;
    manifest_with_time = std::move(new_manifest_with_time);
    return absl::OkStatus();
  }

  absl::Status GetCachedNumberedManifest(
      ManifestWithTime& manifest_with_time) const {
    ManifestWithTime new_manifest_with_time;
    {
      internal::AsyncCache::ReadLock<NumberedManifestCache::NumberedManifest>
          lock{*numbered_manifest_cache_entry_};
      if (auto* data = lock.data()) {
        new_manifest_with_time.manifest = data->manifest;
      }
      new_manifest_with_time.time = lock.stamp().time;
    }

    if (new_manifest_with_time.time == absl::InfinitePast()) {
      manifest_with_time = std::move(new_manifest_with_time);
      return absl::OkStatus();
    }

    absl::MutexLock lock(&manifest_mutex_);
    if (new_manifest_with_time.manifest != cached_numbered_manifest_.manifest) {
      auto* new_manifest = new_manifest_with_time.manifest.get();
      if (new_manifest) {
        TENSORSTORE_RETURN_IF_ERROR(
            config_state->ValidateNewConfig(new_manifest->config));
      }
    }
    cached_numbered_manifest_ = new_manifest_with_time;
    manifest_with_time = std::move(new_manifest_with_time);
    return absl::OkStatus();
  }

  struct GetManifestOp {
    static void Start(const IoHandleImpl* self,
                      Promise<ManifestWithTime> promise,
                      absl::Time staleness_bound) {
      // Retrieve the cached manifest to see if it can be used.
      ManifestWithTime manifest_with_time;
      TENSORSTORE_RETURN_IF_ERROR(
          self->GetCachedTopLevelManifest(manifest_with_time),
          static_cast<void>(promise.SetResult(_)));

      if (manifest_with_time.manifest &&
          manifest_with_time.manifest->config.manifest_kind !=
              ManifestKind::kSingle) {
        ABSL_LOG_IF(INFO, ocdbt_logging)
            << "GetManifestOp::Start: using cached non-single manifest";
        HandleNonSingleManifest(IoHandleImpl::Ptr(self), std::move(promise),
                                staleness_bound);
        return;
      }

      if (manifest_with_time.time >= staleness_bound &&
          manifest_with_time.time != absl::InfinitePast()) {
        promise.SetResult(std::move(manifest_with_time));
        return;
      }

      auto read_future = self->manifest_cache_entry_->Read({staleness_bound});
      LinkValue(
          [self = IoHandleImpl::Ptr(self), staleness_bound](
              Promise<ManifestWithTime> promise,
              ReadyFuture<const void> future) mutable {
            ManifestWithTime manifest_with_time;
            TENSORSTORE_RETURN_IF_ERROR(
                self->GetCachedTopLevelManifest(manifest_with_time),
                static_cast<void>(promise.SetResult(_)));
            if (manifest_with_time.manifest &&
                manifest_with_time.manifest->config.manifest_kind !=
                    ManifestKind::kSingle) {
              HandleNonSingleManifest(std::move(self), std::move(promise),
                                      staleness_bound);
              return;
            }

            promise.SetResult(std::move(manifest_with_time));
          },
          std::move(promise), std::move(read_future));
    }

    static void HandleNonSingleManifest(Ptr self,
                                        Promise<ManifestWithTime> promise,
                                        absl::Time staleness_bound) {
      ManifestWithTime manifest_with_time;
      TENSORSTORE_RETURN_IF_ERROR(
          self->GetCachedNumberedManifest(manifest_with_time),
          static_cast<void>(promise.SetResult(_)));
      if (manifest_with_time.time >= staleness_bound &&
          manifest_with_time.time != absl::InfinitePast()) {
        ABSL_LOG_IF(INFO, ocdbt_logging)
            << "GetManifestOp::Start: using cached numbered manifest: time="
            << manifest_with_time.time
            << ", staleness_bound=" << staleness_bound << ", latest_generation="
            << GetLatestGeneration(manifest_with_time.manifest.get());
        promise.SetResult(std::move(manifest_with_time));
        return;
      }
      auto read_future =
          self->numbered_manifest_cache_entry_->Read({staleness_bound});
      LinkValue(
          [self = std::move(self)](Promise<ManifestWithTime> promise,
                                   ReadyFuture<const void> future) {
            ManifestWithTime manifest_with_time;
            TENSORSTORE_RETURN_IF_ERROR(
                self->GetCachedNumberedManifest(manifest_with_time),
                static_cast<void>(promise.SetResult(_)));
            promise.SetResult(std::move(manifest_with_time));
          },
          std::move(promise), std::move(read_future));
    }
  };

  Future<const ManifestWithTime> GetManifest(
      absl::Time staleness_bound) const final {
    auto [promise, future] = PromiseFuturePair<ManifestWithTime>::Make();
    GetManifestOp::Start(this, std::move(promise), staleness_bound);
    return std::move(future);
  }

  Future<kvstore::ReadResult> ReadIndirectData(
      const IndirectDataReference& ref,
      kvstore::ReadOptions read_options) const final {
    return indirect_data_kvstore_driver_->Read(ref.EncodeCacheKey(),
                                               std::move(read_options));
  }

  struct TryUpdateManifestOp {
    using PromiseType = Promise<TryUpdateManifestResult>;
    static Future<TryUpdateManifestResult> Start(
        Ptr self, std::shared_ptr<const Manifest> old_manifest,
        std::shared_ptr<const Manifest> new_manifest, absl::Time time) {
      ABSL_CHECK(new_manifest);
      if (old_manifest == new_manifest) {
        return MapFutureValue(
            InlineExecutor{},
            [new_manifest = std::move(new_manifest)](
                const ManifestWithTime& value) -> TryUpdateManifestResult {
              return {/*.time=*/value.time,
                      /*.success=*/value.manifest == new_manifest};
            },
            self->GetManifest(time));
      }
      if (new_manifest->config.manifest_kind == ManifestKind::kSingle) {
        return self->manifest_cache_entry_->TryUpdate(
            std::move(old_manifest), std::move(new_manifest), time);
      }
      auto [promise, future] =
          PromiseFuturePair<TryUpdateManifestResult>::Make();

      if (!old_manifest) {
        WriteConfigManifest(std::move(self), std::move(promise),
                            std::move(new_manifest), time);
      } else {
        WriteNewNumberedManifest(std::move(self), std::move(promise),
                                 std::move(old_manifest),
                                 std::move(new_manifest));
      }

      return std::move(future);
    }

    static void WriteConfigManifest(
        Ptr self, PromiseType promise,
        std::shared_ptr<const Manifest> new_manifest, absl::Time time) {
      {
        // Check for existing cached config manifest.  This may be present
        // even if `old_manifest == nullptr`, e.g. in the case that the
        // `manifest.ocdbt` file is written but then an error/failure occurs
        // that prevents the `manifest.0000000000000001` file from being
        // written.
        ManifestWithTime manifest_with_time;
        TENSORSTORE_RETURN_IF_ERROR(
            self->GetCachedTopLevelManifest(manifest_with_time),
            static_cast<void>(promise.SetResult(_)));
        if (manifest_with_time.manifest && manifest_with_time.time >= time) {
          // Config-only `manifest.ocdbt` already present.  Note that
          // `GetCachedTopLevelManifest` already checked the `config`.
          WriteNewNumberedManifest(std::move(self), std::move(promise),
                                   /*old_manifest=*/{},
                                   std::move(new_manifest));
          return;
        }
      }

      auto config_manifest = std::make_shared<Manifest>();
      config_manifest->config = new_manifest->config;

      auto config_manifest_future = self->manifest_cache_entry_->TryUpdate(
          {}, std::move(config_manifest), time);
      LinkValue(
          [self = std::move(self), new_manifest = std::move(new_manifest)](
              PromiseType promise,
              ReadyFuture<TryUpdateManifestResult> future) mutable {
            auto& result = future.value();
            if (!result.success) {
              promise.SetResult(result);
              return;
            }
            WriteNewNumberedManifest(std::move(self), std::move(promise),
                                     /*old_manifest=*/{},
                                     std::move(new_manifest));
          },
          std::move(promise), std::move(config_manifest_future));
    }

    static void WriteNewNumberedManifest(
        Ptr self, PromiseType promise,
        std::shared_ptr<const Manifest> old_manifest,
        std::shared_ptr<const Manifest> new_manifest) {
      auto future =
          self->numbered_manifest_cache_entry_->TryUpdate(new_manifest);
      LinkValue(
          [self = std::move(self), new_manifest = std::move(new_manifest)](
              PromiseType promise,
              ReadyFuture<TryUpdateManifestResult> future) {
            auto& result = future.value();
            if (!result.success) {
              promise.SetResult(result);
              return;
            }
            ValidateNewNumberedManifest(std::move(self), std::move(promise),
                                        std::move(new_manifest), result.time);
          },
          std::move(promise), std::move(future));
    }

    static void ValidateNewNumberedManifest(
        Ptr self, PromiseType promise,
        std::shared_ptr<const Manifest> new_manifest, absl::Time time) {
      ABSL_LOG_IF(INFO, ocdbt_logging)
          << "ValidateNewNumberedManifest: generation="
          << new_manifest->latest_generation();
      auto read_future = internal_ocdbt::ReadVersion(
          self, new_manifest->latest_generation(), time);
      LinkValue(
          [self = std::move(self), new_manifest = std::move(new_manifest)](
              PromiseType promise,
              ReadyFuture<BtreeGenerationReference> future) {
            auto& ref = future.value();
            bool success = (ref == new_manifest->latest_version());

            absl::Time time;
            std::shared_ptr<const NumberedManifestCache::NumberedManifest>
                numbered_manifest;
            {
              internal::AsyncCache::ReadLock<
                  NumberedManifestCache::NumberedManifest>
                  lock(*self->numbered_manifest_cache_entry_);
              time = lock.stamp().time;
              numbered_manifest = lock.shared_data();
            }

            if (!numbered_manifest->manifest) {
              promise.SetResult(absl::FailedPreconditionError(
                  "Manifest was unexpectedly deleted"));
              return;
            }

            TENSORSTORE_RETURN_IF_ERROR(
                self->config_state->ValidateNewConfig(
                    numbered_manifest->manifest->config),
                static_cast<void>(promise.SetResult(_)));

            promise.SetResult(TryUpdateManifestResult{time, success});
          },
          std::move(promise), std::move(read_future));
    }
  };

  virtual Future<TryUpdateManifestResult> TryUpdateManifest(
      std::shared_ptr<const Manifest> old_manifest,
      std::shared_ptr<const Manifest> new_manifest, absl::Time time) const {
    return TryUpdateManifestOp::Start(IoHandleImpl::Ptr(this),
                                      std::move(old_manifest),
                                      std::move(new_manifest), time);
  }

  Future<const void> WriteData(IndirectDataKind kind, absl::Cord data,
                               IndirectDataReference& ref) const final {
    return internal_ocdbt::Write(
        *indirect_data_writer_[static_cast<size_t>(kind)], std::move(data),
        ref);
  }

  std::string DescribeLocation() const final {
    return base_kvstore_.driver->DescribeKey(base_kvstore_.path);
  }
};

IoHandle::Ptr MakeIoHandle(
    const Context::Resource<tensorstore::internal::DataCopyConcurrencyResource>&
        data_copy_concurrency,
    internal::CachePool* cache_pool, const KvStore& base_kvstore,
    const KvStore& manifest_kvstore, ConfigStatePtr config_state,
    const DataFilePrefixes& data_file_prefixes, size_t write_target_size,
    std::optional<ReadCoalesceOptions> read_coalesce_options) {
  // Maybe wrap the base driver in CoalesceKvStoreDriver.
  kvstore::DriverPtr driver_with_optional_coalescing =
      read_coalesce_options.has_value()
          ? MakeCoalesceKvStoreDriver(
                base_kvstore.driver,
                read_coalesce_options->max_overhead_bytes_per_request,
                read_coalesce_options->max_merged_bytes_per_request,
                read_coalesce_options->max_interval,
                data_copy_concurrency->executor)
          : base_kvstore.driver;
  auto impl = internal::MakeIntrusivePtr<IoHandleImpl>();
  impl->base_kvstore_ = base_kvstore;
  impl->config_state = std::move(config_state);
  impl->executor = data_copy_concurrency->executor;
  auto data_kvstore =
      kvstore::KvStore(driver_with_optional_coalescing, base_kvstore.path);
  {
    std::string_view data_prefix_array[kNumIndirectDataKinds];
    data_prefix_array[static_cast<size_t>(IndirectDataKind::kValue)] =
        data_file_prefixes.value;
    data_prefix_array[static_cast<size_t>(IndirectDataKind::kBtreeNode)] =
        data_file_prefixes.btree_node;
    data_prefix_array[static_cast<size_t>(IndirectDataKind::kVersionNode)] =
        data_file_prefixes.version_tree_node;
    for (size_t i = 0; i < kNumIndirectDataKinds; ++i) {
      size_t match_i = std::find(&data_prefix_array[0], &data_prefix_array[i],
                                 data_prefix_array[i]) -
                       &data_prefix_array[0];
      if (match_i == i) {
        impl->indirect_data_writer_[i] = internal_ocdbt::MakeIndirectDataWriter(
            data_kvstore, std::string(data_prefix_array[i]), write_target_size);
      } else {
        impl->indirect_data_writer_[i] = impl->indirect_data_writer_[match_i];
      }
    }
  }
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
                           manifest_kvstore.driver);
  {
    auto manifest_cache =
        internal::GetCache<tensorstore::internal_ocdbt::ManifestCache>(
            cache_pool, manifest_cache_identifier, [&] {
              return std::make_unique<ManifestCache>(
                  manifest_kvstore.driver, data_copy_concurrency->executor);
            });
    impl->manifest_cache_entry_ = tensorstore::internal::GetCacheEntry(
        manifest_cache, manifest_kvstore.path);
  }
  {
    auto numbered_manifest_cache =
        internal::GetCache<tensorstore::internal_ocdbt::NumberedManifestCache>(
            cache_pool, manifest_cache_identifier, [&] {
              return std::make_unique<NumberedManifestCache>(
                  manifest_kvstore.driver, data_copy_concurrency->executor);
            });
    impl->numbered_manifest_cache_entry_ = tensorstore::internal::GetCacheEntry(
        numbered_manifest_cache, manifest_kvstore.path);
  }
  return impl;
}

}  // namespace internal_ocdbt
}  // namespace tensorstore
