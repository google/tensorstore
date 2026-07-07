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
#include <stdint.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "riegeli/bytes/cord_reader.h"
#include "tensorstore/context.h"
#include "tensorstore/internal/cache/async_cache.h"
#include "tensorstore/internal/cache/cache.h"
#include "tensorstore/internal/cache/cache_pool_resource.h"
#include "tensorstore/internal/cache_key/cache_key.h"
#include "tensorstore/internal/compression/zip_details.h"
#include "tensorstore/internal/data_copy_concurrency_resource.h"
#include "tensorstore/internal/estimate_heap_usage/estimate_heap_usage.h"
#include "tensorstore/internal/global_initializer.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/internal/uri/parse.h"
#include "tensorstore/internal/uri/percent_coder.h"
#include "tensorstore/kvstore/auto_detect.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/common_metrics.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/registry.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/kvstore/supported_features.h"
#include "tensorstore/kvstore/url_registry.h"
#include "tensorstore/kvstore/zip/cached_dir.h"
#include "tensorstore/kvstore/zip/zip_dir_cache.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/garbage_collection/fwd.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_builder.h"

// IWYU: needed for serialization/cache_key/GC specializations.
#include "tensorstore/internal/cache_key/std_vector.h"  // IWYU pragma: keep
#include "tensorstore/serialization/absl_time.h"  // IWYU pragma: keep
#include "tensorstore/serialization/serialization.h"  // IWYU pragma: keep
#include "tensorstore/serialization/std_vector.h"  // IWYU pragma: keep
#include "tensorstore/util/garbage_collection/std_vector.h"  // IWYU pragma: keep

using ::tensorstore::internal_zip_kvstore::ZipDirectoryCache;
using ::tensorstore::kvstore::ListEntry;
using ::tensorstore::kvstore::ListReceiver;

namespace tensorstore {
namespace internal_zip_kvstore {
namespace {

namespace jb = tensorstore::internal_json_binding;

ABSL_CONST_INIT internal_log::VerboseFlag zip_logging("zip");

static internal_kvstore::CommonReadMetrics zip_metrics;

TENSORSTORE_GLOBAL_INITIALIZER {
  TENSORSTORE_KVSTORE_REGISTER_COMMON_READ_METRICS(&zip_metrics, zip);
}

// Threshold for using two-phase reads (read local header first, then data).
constexpr int64_t kTwoPhaseReadThreshold = 2 * 1024 * 1024;
constexpr int64_t kZipMinLocalHeaderSize = 30;

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

  Result<std::string> ToUrl(std::string_view path) const override {
    TENSORSTORE_ASSIGN_OR_RETURN(auto base_url,
                                 data_.base.driver->ToUrl(data_.base.path));
    return absl::StrCat(base_url, "|", id, ":",
                        internal_uri::PercentEncodeKvStoreUriPath(path));
  }
};

// Defines the "zip" key value store.
class ZipKvStore
    : public internal_kvstore::RegisteredDriver<ZipKvStore, ZipKvStoreSpec> {
 public:
  Future<ReadResult> Read(Key key, ReadOptions options) override;

  std::string DescribeKey(std::string_view key) override {
    return absl::StrCat(QuoteString(key), " in ",
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

  void InvalidateDirectoryCache(absl::Time time) {
    if (cache_entry_) {
      cache_entry_->MarkStale(time);
    }
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
  CachedDir::Entry entry_;

  struct ResolvedRead {
    // The timestamped generation of the ZIP file container.
    TimestampedStorageGeneration stamp;
    // Read options to be sent to the base ZIP file container.
    kvstore::ReadOptions options;
    // True if a two-phase read is required (reads local header first).
    bool optimized_read = false;
    // True if the local header size was already parsed and cached.
    bool local_header_parsed = false;
    // Byte offset in the returned read result where the local entry begins.
    size_t seek_pos = 0;
  };

  std::optional<ResolvedRead> ResolveEntry(
      Promise<kvstore::ReadResult>& promise) {
    ResolvedRead resolved;
    bool found = false;
    bool match = false;
    bool full_read = false;
    uint64_t local_header_offset = 0;
    uint64_t local_header_and_data_size = 0;

    {
      ZipDirectoryCache::ReadLock<ZipDirectoryCache::ReadData> lock(
          *(owner_->cache_entry_));

      resolved.stamp = lock.stamp();
      resolved.options.staleness_bound = options_.staleness_bound;
      resolved.options.batch = std::move(options_.batch);

      const auto* dir = lock.data();
      if (!dir) {
        promise.SetResult(
            kvstore::ReadResult::Missing(std::move(resolved.stamp)));
        return std::nullopt;
      }

      full_read = dir->full_read;
      ABSL_LOG_IF(INFO, zip_logging) << *dir;

      auto it = std::lower_bound(
          dir->entries.begin(), dir->entries.end(), key_,
          [](const auto& e, const std::string& k) { return e.filename < k; });

      if (it != dir->entries.end() && it->filename == key_) {
        found = true;
        entry_ = *it;
        local_header_offset = it->local_header_offset;
        local_header_and_data_size = it->local_header_and_data_size;
        // NOTE: A member-specific generation could be derived from
        // (crc, uncompressed_size, compressed_size, mtime) to reduce re-reads
        // when only other members change. Currently, we use the base container
        // generation for simplicity.
        match =
            options_.generation_conditions.Matches(resolved.stamp.generation);
      }
    }

    if (!found) {
      promise.SetResult(
          kvstore::ReadResult::Missing(std::move(resolved.stamp)));
      return std::nullopt;
    }

    if (!match) {
      promise.SetResult(
          kvstore::ReadResult::Unspecified(std::move(resolved.stamp)));
      return std::nullopt;
    }

    // Seek directly to range for large uncompressed entries to avoid
    // read amplification.
    if (!options_.byte_range.IsFull() &&
        entry_.compression_method == internal_zip::ZipCompression::kStore &&
        entry_.uncompressed_size > kTwoPhaseReadThreshold) {
      auto validated_range_result =
          options_.byte_range.Validate(entry_.uncompressed_size);
      if (!validated_range_result.ok()) {
        promise.SetResult(validated_range_result.status());
        return std::nullopt;
      }
      ByteRange validated_range = *validated_range_result;

      // Check if we have the local header size cached.
      // entry_ is a local copy; this reads the snapshot value.
      uint64_t local_header_size =
          entry_.local_header_size.load(std::memory_order_relaxed);
      if (local_header_size > 0) {
        uint64_t start = entry_.local_header_offset + local_header_size +
                         validated_range.inclusive_min;
        resolved.options.byte_range = OptionalByteRangeRequest::Range(
            start, start + validated_range.size());
        resolved.optimized_read = false;
        resolved.local_header_parsed = true;
        return resolved;
      }

      uint64_t header_size = kZipMinLocalHeaderSize + entry_.filename.size() +
                             entry_.extra_field_length;
      uint64_t start = entry_.local_header_offset + header_size +
                       validated_range.inclusive_min;
      resolved.options.byte_range = OptionalByteRangeRequest::Range(
          start, start + validated_range.size());
      resolved.optimized_read = true;
      return resolved;
    }

    // Must read from the start for compressed/small/full-range entries.
    if (full_read) {
      resolved.seek_pos = local_header_offset;
    } else {
      resolved.options.byte_range = OptionalByteRangeRequest::Range(
          local_header_offset,
          local_header_offset + local_header_and_data_size);
    }
    return resolved;
  }

  void OnDirectoryReady(Promise<kvstore::ReadResult> promise) {
    auto resolved_opt = ResolveEntry(promise);
    if (!resolved_opt) {
      return;
    }
    auto& resolved = *resolved_opt;

    // Guard base read against directory staleness; stamp.generation holds the
    // base container generation.
    resolved.options.generation_conditions.if_equal = resolved.stamp.generation;

    if (resolved.optimized_read) {
      kvstore::ReadOptions header_options;
      header_options.generation_conditions.if_equal =
          resolved.options.generation_conditions.if_equal;
      header_options.byte_range = OptionalByteRangeRequest::Range(
          entry_.local_header_offset,
          entry_.local_header_offset + kZipMinLocalHeaderSize +
              entry_.filename.size() + entry_.extra_field_length);
      header_options.batch = resolved.options.batch;

      auto read_header_future =
          kvstore::Read(owner_->base_, {}, std::move(header_options));

      // Consider a per-entry Future so that if we get multiple concurrent
      // requests before the local header size is known, we avoid redundant
      // reads of the local header. Could also be skipped, though.
      LinkValue(
          WithExecutor(owner_->executor(),
                       [self = internal::IntrusivePtr<ReadState>(this),
                        resolved = std::move(resolved)](
                           Promise<kvstore::ReadResult> promise,
                           ReadyFuture<kvstore::ReadResult> ready) mutable {
                         self->OnLocalHeaderReadComplete(std::move(promise),
                                                         std::move(ready),
                                                         std::move(resolved));
                       }),
          std::move(promise), std::move(read_header_future));
      return;
    }

    auto read_future =
        kvstore::Read(owner_->base_, {}, std::move(resolved.options));

    LinkValue(WithExecutor(owner_->executor(),
                           [self = internal::IntrusivePtr<ReadState>(this),
                            optimized = resolved.local_header_parsed,
                            seek_pos = resolved.seek_pos](
                               Promise<kvstore::ReadResult> promise,
                               ReadyFuture<kvstore::ReadResult> ready) {
                             self->OnBaseReadComplete(std::move(promise),
                                                      std::move(ready),
                                                      optimized, seek_pos);
                           }),
              std::move(promise), std::move(read_future));
  }

  void OnLocalHeaderReadComplete(Promise<kvstore::ReadResult> promise,
                                 ReadyFuture<kvstore::ReadResult> ready,
                                 ResolvedRead resolved) {
    if (!promise.result_needed()) return;
    auto& r = ready.result();
    if (!r.ok()) {
      promise.SetResult(
          tensorstore::StatusBuilder(std::move(r).status())
              .With(internal::ConvertInvalidArgumentToFailedPrecondition));
      return;
    }
    auto& read_result = *r;
    if (read_result.aborted()) {
      owner_->InvalidateDirectoryCache(read_result.stamp.time);
      auto retry_options = options_;
      retry_options.staleness_bound = absl::Now();
      auto retry_future = owner_->Read(key_, std::move(retry_options));
      LinkResult(std::move(promise), std::move(retry_future));
      return;
    }
    if (read_result.not_found() || !ready.value().has_value()) {
      promise.SetResult(
          absl::InvalidArgumentError("Failed to read ZIP local header"));
      return;
    }

    auto options_result = [&]() -> Result<kvstore::ReadOptions> {
      riegeli::CordReader reader(&read_result.value);
      internal_zip::ZipEntry local_header{};
      TENSORSTORE_RETURN_IF_ERROR(ReadLocalEntry(reader, local_header));
      TENSORSTORE_RETURN_IF_ERROR(ValidateEntryIsSupported(local_header));
      size_t header_size = kZipMinLocalHeaderSize +
                           local_header.filename.size() +
                           local_header.extra_field_length;
      {
        // Update the cached local header size for future reads. This mutates
        // the mutable atomic through the ReadLock.
        ZipDirectoryCache::ReadLock<ZipDirectoryCache::ReadData> lock(
            *(owner_->cache_entry_));
        const auto* dir = lock.data();
        if (dir) {
          auto it =
              std::lower_bound(dir->entries.begin(), dir->entries.end(), key_,
                               [](const auto& e, const std::string& k) {
                                 return e.filename < k;
                               });
          if (it != dir->entries.end() && it->filename == key_) {
            it->local_header_size.store(header_size, std::memory_order_relaxed);
          }
        }
      }
      TENSORSTORE_ASSIGN_OR_RETURN(
          auto byte_range,
          options_.byte_range.Validate(entry_.uncompressed_size));

      kvstore::ReadOptions sub_options = resolved.options;
      uint64_t start =
          entry_.local_header_offset + header_size + byte_range.inclusive_min;
      sub_options.byte_range =
          OptionalByteRangeRequest::Range(start, start + byte_range.size());
      return sub_options;
    }();

    if (!options_result.ok()) {
      promise.SetResult(options_result.status());
      return;
    }

    auto read_future =
        kvstore::Read(owner_->base_, {}, std::move(*options_result));

    LinkValue(WithExecutor(owner_->executor(),
                           [self = internal::IntrusivePtr<ReadState>(this)](
                               Promise<kvstore::ReadResult> promise,
                               ReadyFuture<kvstore::ReadResult> ready) {
                             self->OnBaseReadComplete(std::move(promise),
                                                      std::move(ready),
                                                      /*optimized=*/true,
                                                      /*seek_pos=*/0);
                           }),
              std::move(promise), std::move(read_future));
  }

  void OnBaseReadComplete(Promise<kvstore::ReadResult> promise,
                          ReadyFuture<kvstore::ReadResult> ready,
                          bool optimized, size_t seek_pos) {
    if (!promise.result_needed()) return;
    assert(ready.status().ok());

    const auto& read_result = ready.value();
    if (read_result.aborted()) {
      owner_->InvalidateDirectoryCache(read_result.stamp.time);
      auto retry_options = options_;
      retry_options.staleness_bound = absl::Now();
      auto retry_future = owner_->Read(key_, std::move(retry_options));
      LinkResult(std::move(promise), std::move(retry_future));
      return;
    }

    auto result = [&]() -> Result<kvstore::ReadResult> {
      kvstore::ReadResult rr = std::move(ready.value());
      if (!rr.has_value()) {
        rr.stamp.generation = StorageGeneration::NoValue();
        return rr;
      }

      if (!optimized) {
        absl::Cord source = std::move(rr.value);
        riegeli::CordReader reader(&source);
        reader.Seek(seek_pos);

        internal_zip::ZipEntry local_header{};
        TENSORSTORE_RETURN_IF_ERROR(ReadLocalEntry(reader, local_header));
        TENSORSTORE_RETURN_IF_ERROR(ValidateEntryIsSupported(local_header));

        TENSORSTORE_ASSIGN_OR_RETURN(
            auto byte_range,
            options_.byte_range.Validate(local_header.uncompressed_size));

        TENSORSTORE_ASSIGN_OR_RETURN(
            auto entry_reader, internal_zip::GetReader(&reader, local_header));

        if (byte_range.inclusive_min > 0) {
          entry_reader->Skip(byte_range.inclusive_min);
        }

        if (!entry_reader->Read(byte_range.size(), rr.value)) {
          if (entry_reader->status().ok()) {
            return absl::OutOfRangeError("Failed to read range");
          }
          return entry_reader->status();
        }
      }
      return rr;
    }();

    promise.SetResult(std::move(result));
  }
};

Future<kvstore::ReadResult> ZipKvStore::Read(Key key, ReadOptions options) {
  auto staleness_bound = options.staleness_bound;
  auto state = internal::MakeIntrusivePtr<ReadState>();
  state->owner_ = internal::IntrusivePtr<ZipKvStore>(this);
  state->key_ = std::move(key);
  state->options_ = std::move(options);
  zip_metrics.read.Increment();
  return PromiseFuturePair<kvstore::ReadResult>::LinkValue(
             WithExecutor(
                 executor(),
                 [state = std::move(state)](Promise<ReadResult> promise,
                                            ReadyFuture<const void>) {
                   if (!promise.result_needed()) return;
                   state->OnDirectoryReady(std::move(promise));
                 }),
             cache_entry_->Read({std::move(staleness_bound)}))
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
    if (!dir) return;

    // Directory entries are sorted; binary-search to the range start.
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
  zip_metrics.list.Increment();

  LinkValue(WithExecutor(executor(),
                         [state = std::move(state)](Promise<void> promise,
                                                    ReadyFuture<const void>) {
                           state->OnDirectoryReady();
                         }),
            state_ptr->promise_,
            cache_entry_->Read({state_ptr->options_.staleness_bound}));
}

Result<kvstore::Spec> ParseZipUrl(std::string_view url, kvstore::Spec base) {
  auto parsed = internal_uri::ParseGenericUri(url);
  if (parsed.scheme != ZipKvStoreSpec::id || parsed.has_authority_delimiter) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Scheme \"", ZipKvStoreSpec::id, ":\" not present in url"));
  }
  TENSORSTORE_RETURN_IF_ERROR(EnsureNoQueryOrFragment(parsed));
  TENSORSTORE_ASSIGN_OR_RETURN(std::string path,
                               internal_uri::PercentDecode(parsed.path));
  auto driver_spec = internal::MakeIntrusivePtr<ZipKvStoreSpec>();
  driver_spec->data_.base = std::move(base);
  driver_spec->data_.cache_pool =
      Context::Resource<internal::CachePoolResource>::DefaultSpec();
  driver_spec->data_.data_copy_concurrency =
      Context::Resource<internal::DataCopyConcurrencyResource>::DefaultSpec();
  return {std::in_place, std::move(driver_spec), std::move(path)};
}

std::vector<internal_kvstore::AutoDetectMatch> MatchZipFormat(
    const internal_kvstore::AutoDetectFileOptions& options) {
  riegeli::CordReader reader{options.suffix};
  internal_zip::ZipEOCD zip_eocd;
  auto result =
      internal_zip::TryReadFullEOCD(reader, zip_eocd, /*offset_adjustment=*/0);
  std::vector<internal_kvstore::AutoDetectMatch> matches;
  if (auto* status = std::get_if<absl::Status>(&result);
      status && status->ok()) {
    matches.push_back(internal_kvstore::AutoDetectMatch{ZipKvStoreSpec::id});
  }
  return matches;
}

internal_kvstore::AutoDetectFileSpec GetAutoDetectSpec() {
  internal_kvstore::AutoDetectFileSpec spec;
  // To avoid excessive reads for auto-detection, only support comments up to
  // 4096 bytes rather than the full 65535 bytes permitted.
  spec.suffix_length = 4096 + 48;
  spec.match = MatchZipFormat;
  return spec;
}

}  // namespace
}  // namespace internal_zip_kvstore
}  // namespace tensorstore

TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED(
    tensorstore::internal_zip_kvstore::ZipKvStore)

// Registers the driver.
namespace {
const tensorstore::internal_kvstore::DriverRegistration<
    tensorstore::internal_zip_kvstore::ZipKvStoreSpec>
    registration;

const tensorstore::internal_kvstore::UrlSchemeRegistration
    url_scheme_registration{
        tensorstore::internal_zip_kvstore::ZipKvStoreSpec::id,
        tensorstore::internal_zip_kvstore::ParseZipUrl};

const tensorstore::internal_kvstore::AutoDetectRegistration
    auto_detect_registration{
        tensorstore::internal_zip_kvstore::GetAutoDetectSpec()};

}  // namespace
