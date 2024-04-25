// Copyright 2023 The TensorStore Authors
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

#include <stddef.h>

#include <algorithm>
#include <cassert>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_format.h"
#include "riegeli/bytes/cord_reader.h"
#include "tensorstore/context.h"
#include "tensorstore/internal/cache/async_cache.h"
#include "tensorstore/internal/cache/cache.h"
#include "tensorstore/internal/cache/cache_pool_resource.h"
#include "tensorstore/internal/cache_key/cache_key.h"
#include "tensorstore/internal/compression/zip_details.h"
#include "tensorstore/internal/data_copy_concurrency_resource.h"
#include "tensorstore/internal/estimate_heap_usage/estimate_heap_usage.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/registry.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/kvstore/supported_features.h"
#include "tensorstore/kvstore/zip/zip_dir_cache.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/garbage_collection/fwd.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

/// specializations
#include "absl/base/attributes.h"
#include "tensorstore/internal/cache_key/std_vector.h"  // IWYU pragma: keep
#include "tensorstore/serialization/absl_time.h"  // IWYU pragma: keep
#include "tensorstore/serialization/serialization.h"  // IWYU pragma: keep
#include "tensorstore/serialization/std_vector.h"  // IWYU pragma: keep
#include "tensorstore/util/garbage_collection/std_vector.h"  // IWYU pragma: keep

using ::tensorstore::internal_zip_kvstore::Directory;
using ::tensorstore::internal_zip_kvstore::ZipDirectoryCache;
using ::tensorstore::kvstore::ListEntry;
using ::tensorstore::kvstore::ListReceiver;

namespace tensorstore {
namespace {

namespace jb = tensorstore::internal_json_binding;

ABSL_CONST_INIT internal_log::VerboseFlag zip_logging("zip");

// -----------------------------------------------------------------------------

struct ZipKvStoreSpecData {
  kvstore::Spec base;
  Context::Resource<internal::CachePoolResource> cache_pool;
  Context::Resource<internal::DataCopyConcurrencyResource>
      data_copy_concurrency;

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.base, x.cache_pool, x.data_copy_concurrency);
  };

  constexpr static auto default_json_binder = jb::Object(
      jb::Member("base", jb::Projection<&ZipKvStoreSpecData::base>()),
      jb::Member(internal::CachePoolResource::id,
                 jb::Projection<&ZipKvStoreSpecData::cache_pool>()),
      jb::Member(
          internal::DataCopyConcurrencyResource::id,
          jb::Projection<&ZipKvStoreSpecData::data_copy_concurrency>()) /**/
  );
};

class ZipKvStoreSpec
    : public internal_kvstore::RegisteredDriverSpec<ZipKvStoreSpec,
                                                    ZipKvStoreSpecData> {
 public:
  static constexpr char id[] = "zip";

  Future<kvstore::DriverPtr> DoOpen() const override;

  absl::Status ApplyOptions(kvstore::DriverSpecOptions&& options) override {
    return data_.base.driver.Set(std::move(options));
  }

  Result<kvstore::Spec> GetBase(std::string_view path) const override {
    return data_.base;
  }
};

/// Defines the "zip" key value store.
class ZipKvStore
    : public internal_kvstore::RegisteredDriver<ZipKvStore, ZipKvStoreSpec> {
 public:
  Future<ReadResult> Read(Key key, ReadOptions options) override;

  std::string DescribeKey(std::string_view key) override {
    return tensorstore::StrCat(QuoteString(key), " in ",
                               base_.driver->DescribeKey(base_.path));
  }

  void ListImpl(ListOptions options, ListReceiver receiver) override;

  absl::Status GetBoundSpecData(ZipKvStoreSpecData& spec) const {
    spec = spec_data_;
    return absl::OkStatus();
  }

  kvstore::SupportedFeatures GetSupportedFeatures(
      const KeyRange& key_range) const final {
    return base_.driver->GetSupportedFeatures(KeyRange::Singleton(base_.path));
  }

  Result<KvStore> GetBase(std::string_view path,
                          const Transaction& transaction) const override {
    return KvStore(base_.driver, base_.path, transaction);
  }

  const Executor& executor() const {
    return spec_data_.data_copy_concurrency->executor;
  }

  ZipKvStoreSpecData spec_data_;
  kvstore::KvStore base_;
  internal::PinnedCacheEntry<ZipDirectoryCache> cache_entry_;
};

Future<kvstore::DriverPtr> ZipKvStoreSpec::DoOpen() const {
  return MapFutureValue(
      InlineExecutor{},
      [spec = internal::IntrusivePtr<const ZipKvStoreSpec>(this)](
          kvstore::KvStore& base_kvstore) mutable
      -> Result<kvstore::DriverPtr> {
        std::string cache_key;
        internal::EncodeCacheKey(&cache_key, base_kvstore.driver,
                                 base_kvstore.path,
                                 spec->data_.data_copy_concurrency);
        auto& cache_pool = *spec->data_.cache_pool;
        auto directory_cache = internal::GetCache<ZipDirectoryCache>(
            cache_pool.get(), cache_key, [&] {
              return std::make_unique<ZipDirectoryCache>(
                  base_kvstore.driver,
                  spec->data_.data_copy_concurrency->executor);
            });

        auto driver = internal::MakeIntrusivePtr<ZipKvStore>();
        driver->base_ = std::move(base_kvstore);
        driver->spec_data_ = std::move(spec->data_);
        driver->cache_entry_ =
            GetCacheEntry(directory_cache, driver->base_.path);
        return driver;
      },
      kvstore::Open(data_.base));
}

// Implements ZipKvStore::Read
struct ReadState : public internal::AtomicReferenceCount<ReadState> {
  internal::IntrusivePtr<ZipKvStore> owner_;
  kvstore::Key key_;
  kvstore::ReadOptions options_;

  // The cache read has completed, so the zip directory entries are available.
  void OnDirectoryReady(Promise<kvstore::ReadResult> promise) {
    TimestampedStorageGeneration stamp;

    // Set options for the entry request.
    kvstore::ReadOptions options;
    options.staleness_bound = options_.staleness_bound;
    options.byte_range = OptionalByteRangeRequest{};
    size_t seek_pos = 0;

    {
      ZipDirectoryCache::ReadLock<ZipDirectoryCache::ReadData> lock(
          *(owner_->cache_entry_));
      stamp = lock.stamp();

      // Find key in the directory.
      assert(lock.data());
      const ZipDirectoryCache::ReadData& dir = *lock.data();
      ABSL_LOG_IF(INFO, zip_logging) << dir;

      auto it = std::lower_bound(
          dir.entries.begin(), dir.entries.end(), key_,
          [](const auto& e, const std::string& k) { return e.filename < k; });

      if (it == dir.entries.end() || it->filename != key_) {
        // Missing value.
        promise.SetResult(kvstore::ReadResult::Missing(std::move(stamp)));
        return;
      }

      // Check if_equal and if_not_equal conditions.
      // This happens after searching the directory in order to correctly handle
      // IsNoValue matches, above.
      if (!options_.generation_conditions.Matches(stamp.generation)) {
        promise.SetResult(kvstore::ReadResult::Unspecified(std::move(stamp)));
        return;
      }

      // Setup a read for the key.
      if (dir.full_read) {
        seek_pos = it->local_header_offset;
      } else {
        seek_pos = 0;
        options.byte_range = OptionalByteRangeRequest::Range(
            it->local_header_offset,
            it->local_header_offset + it->estimated_size);
      }
    }

    options.generation_conditions.if_equal = stamp.generation;
    Link(WithExecutor(owner_->executor(),
                      [self = internal::IntrusivePtr<ReadState>(this),
                       seek_pos](Promise<kvstore::ReadResult> promise,
                                 ReadyFuture<kvstore::ReadResult> ready) {
                        self->OnValueRead(std::move(promise), std::move(ready),
                                          seek_pos);
                      }),
         std::move(promise),
         kvstore::Read(owner_->base_, {}, std::move(options)));
  }

  void OnValueRead(Promise<kvstore::ReadResult> promise,
                   ReadyFuture<kvstore::ReadResult> ready, size_t seek_pos) {
    if (!promise.result_needed()) return;
    if (!ready.status().ok()) {
      promise.SetResult(ready.status());
      return;
    }

    internal_zip::ZipEntry local_header{};
    auto result = [&]() -> Result<kvstore::ReadResult> {
      kvstore::ReadResult read_result = std::move(ready.value());
      if (!read_result.has_value()) {
        return read_result;
      }
      absl::Cord source = std::move(read_result.value);
      riegeli::CordReader reader(&source);
      reader.Seek(seek_pos);

      TENSORSTORE_RETURN_IF_ERROR(ReadLocalEntry(reader, local_header));
      TENSORSTORE_RETURN_IF_ERROR(ValidateEntryIsSupported(local_header));

      TENSORSTORE_ASSIGN_OR_RETURN(
          auto byte_range,
          options_.byte_range.Validate(local_header.uncompressed_size));

      TENSORSTORE_ASSIGN_OR_RETURN(
          auto entry_reader, internal_zip::GetReader(&reader, local_header));

      // NOTE: To handle range requests efficiently we'd need a cache.
      if (byte_range.inclusive_min > 0) {
        // This should, IMO, be Seek, however when the reader is only
        // wrapped in a LimitingReader<>, Seek appear appears to seek the
        // underlying reader.  Maybe zip_details should use a WrappingReader?
        entry_reader->Skip(byte_range.inclusive_min);
      }

      if (!entry_reader->Read(byte_range.size(), read_result.value)) {
        // This should not happen unless there's some underlying corruption,
        // since the range has already been validated.
        if (entry_reader->status().ok()) {
          return absl::OutOfRangeError("Failed to read range");
        }
        return entry_reader->status();
      }
      return read_result;
    }();

    ABSL_LOG_IF(INFO, zip_logging && !result.ok()) << result.status() << "\n"
                                                   << local_header;

    promise.SetResult(std::move(result));
  }
};

Future<kvstore::ReadResult> ZipKvStore::Read(Key key, ReadOptions options) {
  auto state = internal::MakeIntrusivePtr<ReadState>();
  state->owner_ = internal::IntrusivePtr<ZipKvStore>(this);
  state->key_ = std::move(key);
  state->options_ = options;

  return PromiseFuturePair<kvstore::ReadResult>::LinkValue(
             WithExecutor(
                 executor(),
                 [state = std::move(state)](Promise<ReadResult> promise,
                                            ReadyFuture<const void>) {
                   if (!promise.result_needed()) return;
                   state->OnDirectoryReady(std::move(promise));
                 }),
             cache_entry_->Read({options.staleness_bound}))
      .future;
}

// Implements ZipKvStore::List
struct ListState : public internal::AtomicReferenceCount<ListState> {
  internal::IntrusivePtr<ZipKvStore> owner_;
  kvstore::ListOptions options_;
  ListReceiver receiver_;
  Promise<void> promise_;
  Future<void> future_;

  ListState(internal::IntrusivePtr<ZipKvStore>&& owner,
            kvstore::ListOptions&& options, ListReceiver&& receiver)
      : owner_(std::move(owner)),
        options_(std::move(options)),
        receiver_(std::move(receiver)) {
    auto [promise, future] = PromiseFuturePair<void>::Make(MakeResult());
    this->promise_ = std::move(promise);
    this->future_ = std::move(future);
    future_.Force();
    execution::set_starting(receiver_, [promise = promise_] {
      promise.SetResult(absl::CancelledError(""));
    });
  }

  ~ListState() {
    auto& r = promise_.raw_result();
    if (r.ok()) {
      execution::set_done(receiver_);
    } else {
      execution::set_error(receiver_, r.status());
    }
    execution::set_stopping(receiver_);
  }

  void OnDirectoryReady() {
    // ephemeral lock to acquire the ZIP directory.
    auto dir = ZipDirectoryCache::ReadLock<ZipDirectoryCache::ReadData>(
                   *(owner_->cache_entry_))
                   .shared_data();
    assert(dir);

    auto it = std::lower_bound(
        dir->entries.begin(), dir->entries.end(), options_.range.inclusive_min,
        [](const auto& e, const std::string& k) { return e.filename < k; });
    for (; it != dir->entries.end(); ++it) {
      if (KeyRange::CompareKeyAndExclusiveMax(
              it->filename, options_.range.exclusive_max) >= 0) {
        break;
      }
      if (it->filename.size() >= options_.strip_prefix_length) {
        execution::set_value(
            receiver_,
            ListEntry{it->filename.substr(options_.strip_prefix_length),
                      ListEntry::checked_size(it->uncompressed_size)});
      }
    }
  }
};

void ZipKvStore::ListImpl(ListOptions options, ListReceiver receiver) {
  auto state = internal::MakeIntrusivePtr<ListState>(
      internal::IntrusivePtr<ZipKvStore>(this), std::move(options),
      std::move(receiver));
  auto* state_ptr = state.get();
  LinkValue(WithExecutor(executor(),
                         [state = std::move(state)](Promise<void> promise,
                                                    ReadyFuture<const void>) {
                           state->OnDirectoryReady();
                         }),
            state_ptr->promise_,
            cache_entry_->Read({state_ptr->options_.staleness_bound}));
}

}  // namespace
}  // namespace tensorstore

TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED(tensorstore::ZipKvStore)

// Registers the driver.
namespace {
const tensorstore::internal_kvstore::DriverRegistration<
    tensorstore::ZipKvStoreSpec>
    registration;

}  // namespace
