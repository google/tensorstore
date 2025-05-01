// Copyright 2025 The TensorStore Authors
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

#include "tensorstore/kvstore/tiff/tiff_key_value_store.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_format.h"
#include "absl/strings/strip.h"
#include "tensorstore/context.h"
#include "tensorstore/internal/cache/async_cache.h"
#include "tensorstore/internal/cache/cache.h"
#include "tensorstore/internal/cache/cache_pool_resource.h"
#include "tensorstore/internal/cache_key/cache_key.h"
#include "tensorstore/internal/data_copy_concurrency_resource.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/registry.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/kvstore/tiff/tiff_details.h"
#include "tensorstore/kvstore/tiff/tiff_dir_cache.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore::kvstore::tiff_kvstore {
namespace jb = ::tensorstore::internal_json_binding;

using ::tensorstore::internal_tiff_kvstore::ImageDirectory;
using ::tensorstore::internal_tiff_kvstore::TiffDirectoryCache;
using ::tensorstore::internal_tiff_kvstore::TiffParseResult;
using ::tensorstore::kvstore::ListEntry;
using ::tensorstore::kvstore::ListReceiver;

namespace {

ABSL_CONST_INIT internal_log::VerboseFlag tiff_logging("tiff");

// Expected key: "chunk/<ifd>/<chunk_index>"
absl::Status ParseChunkKey(std::string_view key, uint32_t& ifd,
                           uint32_t& linear_index) {
  auto eat_number = [&](std::string_view& s, uint32_t& out) -> bool {
    if (s.empty()) return false;
    uint32_t v = 0;
    size_t i = 0;
    while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
      v = v * 10 + (s[i] - '0');
      ++i;
    }
    if (i == 0) return false;  // no digits
    out = v;
    s.remove_prefix(i);
    return true;
  };

  if (!absl::ConsumePrefix(&key, "chunk/")) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Invalid chunk key format: expected prefix 'chunk/' in '", key, "'"));
  }

  // Parse IFD index
  if (!eat_number(key, ifd)) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Invalid chunk key format: expected numeric IFD index in '", key, "'"));
  }

  // Consume separator '/'
  if (!absl::ConsumePrefix(&key, "/")) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Invalid chunk key format: expected '/' after IFD index in '", key,
        "'"));
  }

  // Parse linear index
  if (!eat_number(key, linear_index)) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Invalid chunk key format: expected numeric linear chunk index in '",
        key, "'"));
  }

  // Ensure no trailing characters remain
  if (!key.empty()) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Invalid chunk key format: unexpected trailing characters '", key,
        "'"));
  }

  return absl::OkStatus();
}

struct TiffKvStoreSpecData {
  kvstore::Spec base;
  Context::Resource<internal::CachePoolResource> cache_pool;
  Context::Resource<internal::DataCopyConcurrencyResource>
      data_copy_concurrency;

  constexpr static auto ApplyMembers = [](auto& x, auto f) {
    return f(x.base, x.cache_pool, x.data_copy_concurrency);
  };

  constexpr static auto default_json_binder = jb::Object(
      jb::Member("base", jb::Projection<&TiffKvStoreSpecData::base>()),
      jb::Member(internal::CachePoolResource::id,
                 jb::Projection<&TiffKvStoreSpecData::cache_pool>()),
      jb::Member(
          internal::DataCopyConcurrencyResource::id,
          jb::Projection<&TiffKvStoreSpecData::data_copy_concurrency>()));
};

struct Spec
    : public internal_kvstore::RegisteredDriverSpec<Spec, TiffKvStoreSpecData> {
  static constexpr char id[] = "tiff";

  Future<kvstore::DriverPtr> DoOpen() const override;

  absl::Status ApplyOptions(kvstore::DriverSpecOptions&& o) override {
    return data_.base.driver.Set(std::move(o));
  }
  Result<kvstore::Spec> GetBase(std::string_view) const override {
    return data_.base;
  }
};

class TiffKeyValueStore
    : public internal_kvstore::RegisteredDriver<TiffKeyValueStore, Spec> {
 public:
  Future<ReadResult> Read(Key key, ReadOptions options) override;

  void ListImpl(ListOptions options, ListReceiver receiver) override;

  std::string DescribeKey(std::string_view key) override {
    return StrCat(QuoteString(key), " in ",
                  base_.driver->DescribeKey(base_.path));
  }

  SupportedFeatures GetSupportedFeatures(const KeyRange& r) const override {
    return base_.driver->GetSupportedFeatures(
        KeyRange::AddPrefix(base_.path, r));
  }

  Result<KvStore> GetBase(std::string_view,
                          const Transaction& t) const override {
    return KvStore(base_.driver, base_.path, t);
  }

  const Executor& executor() const {
    return spec_data_.data_copy_concurrency->executor;
  }

  absl::Status GetBoundSpecData(TiffKvStoreSpecData& spec) const {
    spec = spec_data_;
    return absl::OkStatus();
  }

  TiffKvStoreSpecData spec_data_;
  kvstore::KvStore base_;
  internal::PinnedCacheEntry<TiffDirectoryCache> cache_entry_;
};

// Implements TiffKeyValueStore::Read
struct ReadState : public internal::AtomicReferenceCount<ReadState> {
  internal::IntrusivePtr<TiffKeyValueStore> owner_;
  kvstore::Key key_;
  kvstore::ReadOptions options_;
  uint32_t ifd_;
  uint32_t linear_index_;

  void OnDirectoryReady(Promise<kvstore::ReadResult> promise) {
    TimestampedStorageGeneration dir_stamp;
    uint64_t chunk_offset;
    uint64_t chunk_byte_count;

    {
      TiffDirectoryCache::ReadLock<TiffDirectoryCache::ReadData> lock(
          *(owner_->cache_entry_));

      if (!lock.data()) {
        promise.SetResult(owner_->cache_entry_->AnnotateError(
            absl::FailedPreconditionError(
                "TIFF directory cache data is null after read attempt"),
            true));
        return;
      }
      dir_stamp = lock.stamp();
      const auto& parse_result = *lock.data();

      if (ifd_ >= parse_result.image_directories.size()) {
        promise.SetResult(absl::NotFoundError(
            absl::StrFormat("IFD %d not found, only %d IFDs available", ifd_,
                            lock.data()->image_directories.size())));
        return;
      }

      const auto& dir = parse_result.image_directories[ifd_];

      if (linear_index_ >= dir.chunk_offsets.size() ||
          linear_index_ >= dir.chunk_bytecounts.size()) {
        promise.SetResult(absl::OutOfRangeError(
            absl::StrFormat("Linear chunk index %d out of range for IFD %d "
                            "(valid range [0, %d))",
                            linear_index_, ifd_, dir.chunk_offsets.size())));
        return;
      }

      chunk_offset = dir.chunk_offsets[linear_index_];
      chunk_byte_count = dir.chunk_bytecounts[linear_index_];

      if (!options_.generation_conditions.Matches(dir_stamp.generation)) {
        promise.SetResult(
            kvstore::ReadResult::Unspecified(std::move(dir_stamp)));
        return;
      }
    }

    kvstore::ReadOptions chunk_read_options;
    chunk_read_options.staleness_bound = options_.staleness_bound;
    chunk_read_options.byte_range = options_.byte_range;
    chunk_read_options.generation_conditions = options_.generation_conditions;

    // Calculate the absolute byte range needed from the base store
    Result<ByteRange> absolute_byte_range_result =
        chunk_read_options.byte_range.Validate(chunk_byte_count);
    if (!absolute_byte_range_result.ok()) {
      promise.SetResult(std::move(absolute_byte_range_result).status());
      return;
    }
    ByteRange absolute_byte_range = absolute_byte_range_result.value();
    absolute_byte_range.inclusive_min += chunk_offset;
    absolute_byte_range.exclusive_max += chunk_offset;
    chunk_read_options.byte_range = absolute_byte_range;

    // Issue read for the chunk data bytes from the base kvstore
    auto future = owner_->base_.driver->Read(owner_->base_.path,
                                             std::move(chunk_read_options));
    future.Force();
    future.ExecuteWhenReady(
        [self = internal::IntrusivePtr<ReadState>(this),
         promise = std::move(promise)](
            ReadyFuture<kvstore::ReadResult> ready) mutable {
          if (!ready.result().ok()) {
            promise.SetResult(std::move(ready.result()));
            return;
          }

          auto read_result = std::move(ready.result().value());
          if (!read_result.has_value()) {
            promise.SetResult(std::move(read_result));
            return;
          }

          promise.SetResult(std::move(read_result));
        });
  }
};

// Implements TiffKeyValueStore::List
struct ListState : public internal::AtomicReferenceCount<ListState> {
  internal::IntrusivePtr<TiffKeyValueStore> owner_;
  kvstore::ListOptions options_;
  ListReceiver receiver_;
  Promise<void> promise_;
  Future<void> future_;

  ListState(internal::IntrusivePtr<TiffKeyValueStore>&& owner,
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
    TiffDirectoryCache::ReadLock<TiffDirectoryCache::ReadData> lock(
        *(owner_->cache_entry_));

    if (!lock.data()) {
      promise_.SetResult(owner_->cache_entry_->AnnotateError(
          absl::FailedPreconditionError(
              "TIFF directory cache data is null after read attempt"),
          true));
      return;
    }

    const auto& parse_result = *lock.data();
    for (size_t ifd_index = 0;
         ifd_index < parse_result.image_directories.size(); ++ifd_index) {
      const auto& dir = parse_result.image_directories[ifd_index];

      const size_t num_chunks = dir.chunk_offsets.size();
      if (num_chunks != dir.chunk_bytecounts.size()) {
        promise_.SetResult(absl::InternalError(absl::StrFormat(
            "Inconsistent chunk offset/bytecount array sizes for IFD %d",
            ifd_index)));
        return;
      }

      for (uint64_t linear_index = 0; linear_index < num_chunks;
           ++linear_index) {
        std::string key =
            absl::StrFormat("chunk/%d/%d", ifd_index, linear_index);

        if (tensorstore::Contains(options_.range, key)) {
          size_t chunk_size = dir.chunk_bytecounts[linear_index];

          // Apply prefix stripping if requested
          std::string_view adjusted_key = key;
          if (options_.strip_prefix_length > 0 &&
              options_.strip_prefix_length <= key.size()) {
            adjusted_key =
                std::string_view(key).substr(options_.strip_prefix_length);
          } else if (options_.strip_prefix_length > key.size()) {
            adjusted_key = "";  // Strip entire key
          }

          // Send the entry to the receiver
          execution::set_value(receiver_,
                               ListEntry{std::string(adjusted_key),
                                         ListEntry::checked_size(chunk_size)});

          // Check if cancellation was requested by the receiver downstream
          if (!promise_.result_needed()) {
            return;
          }
        } else if (key >= options_.range.exclusive_max &&
                   !options_.range.exclusive_max.empty()) {
          // If current key is already past the requested range's end,
          // we can potentially optimize by stopping early for this IFD,
          // assuming keys are generated in lexicographical order.
          break;
        }
      }

      // Check again for cancellation after processing an IFD
      if (!promise_.result_needed()) {
        return;
      }

    }  // End loop over IFDs

    promise_.SetResult(absl::OkStatus());
  }
};

Future<kvstore::DriverPtr> Spec::DoOpen() const {
  return MapFutureValue(
      InlineExecutor{},
      [spec = internal::IntrusivePtr<const Spec>(this)](
          kvstore::KvStore& base_kvstore) mutable
          -> Result<kvstore::DriverPtr> {
        // Create cache key from base kvstore and executor
        std::string cache_key;
        internal::EncodeCacheKey(&cache_key, base_kvstore.driver,
                                 base_kvstore.path,
                                 spec->data_.data_copy_concurrency);

        // Get or create the directory cache
        auto& cache_pool = *spec->data_.cache_pool;
        auto directory_cache = internal::GetCache<TiffDirectoryCache>(
            cache_pool.get(), cache_key, [&] {
              return std::make_unique<TiffDirectoryCache>(
                  base_kvstore.driver,
                  spec->data_.data_copy_concurrency->executor);
            });

        // Create the driver and set its fields
        auto driver = internal::MakeIntrusivePtr<TiffKeyValueStore>();
        driver->base_ = std::move(base_kvstore);
        driver->spec_data_ = std::move(spec->data_);
        driver->cache_entry_ =
            GetCacheEntry(directory_cache, driver->base_.path);

        return driver;
      },
      kvstore::Open(data_.base));
}

Future<ReadResult> TiffKeyValueStore::Read(Key key, ReadOptions options) {
  uint32_t ifd, linear_index;
  if (auto st = ParseChunkKey(key, ifd, linear_index); !st.ok()) {
    // Instead of returning the error, return a "missing" result
    return MakeReadyFuture<ReadResult>(
        kvstore::ReadResult::Missing(TimestampedStorageGeneration{
            StorageGeneration::NoValue(), absl::Now()}));
  }

  auto state = internal::MakeIntrusivePtr<ReadState>();
  state->owner_ = internal::IntrusivePtr<TiffKeyValueStore>(this);
  state->key_ = std::move(key);
  state->options_ = options;
  state->ifd_ = ifd;
  state->linear_index_ = linear_index;

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

void TiffKeyValueStore::ListImpl(ListOptions options, ListReceiver receiver) {
  auto state = internal::MakeIntrusivePtr<ListState>(
      internal::IntrusivePtr<TiffKeyValueStore>(this), std::move(options),
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

Result<DriverPtr> GetTiffKeyValueStoreDriver(
    DriverPtr base_kvstore, std::string path,
    const Context::Resource<internal::CachePoolResource>& cache_pool_res,
    const Context::Resource<internal::DataCopyConcurrencyResource>&
        data_copy_res,
    const internal::PinnedCacheEntry<internal_tiff_kvstore::TiffDirectoryCache>&
        dir_cache_entry) {
  // Check if resources are valid before dereferencing
  if (!cache_pool_res.has_resource()) {
    return absl::InvalidArgumentError("Cache pool resource is not available");
  }
  if (!data_copy_res.has_resource()) {
    return absl::InvalidArgumentError(
        "Data copy concurrency resource is not available");
  }
  if (!dir_cache_entry) {
    return absl::InvalidArgumentError(
        "TIFF directory cache entry is not valid");
  }

  auto driver = internal::MakeIntrusivePtr<TiffKeyValueStore>();
  driver->base_ = KvStore(base_kvstore, std::move(path));

  // Assign the provided *resolved* resource handles
  driver->spec_data_.cache_pool = cache_pool_res;
  driver->spec_data_.data_copy_concurrency = data_copy_res;

  // Assign the provided cache entry
  driver->cache_entry_ = dir_cache_entry;

  return DriverPtr(std::move(driver));
}

}  // namespace tensorstore::kvstore::tiff_kvstore

TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED(
    tensorstore::kvstore::tiff_kvstore::TiffKeyValueStore)

namespace {
const tensorstore::internal_kvstore::DriverRegistration<
    tensorstore::kvstore::tiff_kvstore::Spec>
    registration;
}  // namespace
