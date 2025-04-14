// Copyright 2024 The TensorStore Authors
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

// -----------------------------------------------------------------------------
// TIFF key‑value‑store adapter
//   * read‑only
//   * uses TiffDirectoryCache for parsing TIFF file structure
//   * supports tile or strip-based TIFF files
// -----------------------------------------------------------------------------

#include "tensorstore/kvstore/tiff/tiff_key_value_store.h"

#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_format.h"
#include "absl/strings/strip.h"  // For ConsumePrefix
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
using ::tensorstore::kvstore::ListEntry;
using ::tensorstore::kvstore::ListReceiver;

namespace {

ABSL_CONST_INIT internal_log::VerboseFlag tiff_logging("tiff");

// Expected key: "tile/<ifd>/<row>/<col>"
absl::Status ParseTileKey(std::string_view key,
                          uint32_t& ifd, uint32_t& row, uint32_t& col) {
  auto eat_number = [&](std::string_view& s, uint32_t& out) -> bool {
    if (s.empty()) return false;
    uint32_t v = 0;
    size_t i = 0;
    while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
      v = v * 10 + (s[i] - '0');
      ++i;
    }
    if (i == 0) return false;           // no digits
    out = v;
    s.remove_prefix(i);
    return true;
  };

  if (!absl::ConsumePrefix(&key, "tile/")) {
    return absl::InvalidArgumentError("Key must start with \"tile/\"");
  }
  if (!eat_number(key, ifd) || !absl::ConsumePrefix(&key, "/") ||
      !eat_number(key, row) || !absl::ConsumePrefix(&key, "/") ||
      !eat_number(key, col) || !key.empty()) {
    return absl::InvalidArgumentError("Bad tile key format");
  }
  return absl::OkStatus();
}

// ─────────────────────────────────────────────────────────────────────────────
//  Spec‑data (JSON parameters)
// ─────────────────────────────────────────────────────────────────────────────
struct TiffKvStoreSpecData {
  kvstore::Spec base;
  Context::Resource<internal::CachePoolResource> cache_pool;
  Context::Resource<internal::DataCopyConcurrencyResource> data_copy_concurrency;

  constexpr static auto ApplyMembers = [](auto& x, auto f) {
    return f(x.base, x.cache_pool, x.data_copy_concurrency);
  };

  constexpr static auto default_json_binder = jb::Object(
      jb::Member("base", jb::Projection<&TiffKvStoreSpecData::base>()),
      jb::Member(internal::CachePoolResource::id,
                 jb::Projection<&TiffKvStoreSpecData::cache_pool>()),
      jb::Member(internal::DataCopyConcurrencyResource::id,
                 jb::Projection<&TiffKvStoreSpecData::data_copy_concurrency>()));
};

// ─────────────────────────────────────────────────────────────────────────────
//  Spec
// ─────────────────────────────────────────────────────────────────────────────
struct Spec
    : public internal_kvstore::RegisteredDriverSpec<Spec,
                                                    TiffKvStoreSpecData> {
  static constexpr char id[] = "tiff";

  Future<kvstore::DriverPtr> DoOpen() const override;

  absl::Status ApplyOptions(kvstore::DriverSpecOptions&& o) override {
    return data_.base.driver.Set(std::move(o));
  }
  Result<kvstore::Spec> GetBase(std::string_view) const override {
    return data_.base;
  }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Driver
// ─────────────────────────────────────────────────────────────────────────────
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
  
  Result<KvStore> GetBase(std::string_view, const Transaction& t) const override {
    return KvStore(base_.driver, base_.path, t);
  }
  
  const Executor& executor() const { return spec_data_.data_copy_concurrency->executor; }

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
  uint32_t ifd_, row_, col_;

  void OnDirectoryReady(Promise<kvstore::ReadResult> promise) {
    TimestampedStorageGeneration stamp;

    // Set options for the chunk read request
    kvstore::ReadOptions options;
    options.staleness_bound = options_.staleness_bound;
    options.byte_range = OptionalByteRangeRequest{};
    
    // Store original byte range for later use
    OptionalByteRangeRequest original_byte_range = options_.byte_range;

    {
      TiffDirectoryCache::ReadLock<TiffDirectoryCache::ReadData> lock(
          *(owner_->cache_entry_));
      stamp = lock.stamp();

      // Get directory data and verify ifd_ is valid (only ifd 0 for now)
      assert(lock.data());
      const auto& dir = lock.data()->image_directory;
      
      if (ifd_ != 0) {
        promise.SetResult(absl::UnimplementedError("Only IFD 0 implemented"));
        return;
      }

      // Check if tile/strip indices are in bounds
      uint32_t chunk_rows, chunk_cols;
      uint64_t offset, byte_count;
      
      if (dir.tile_width > 0) {
        // Tiled TIFF
        chunk_rows = (dir.height + dir.tile_height - 1) / dir.tile_height;
        chunk_cols = (dir.width + dir.tile_width - 1) / dir.tile_width;
        
        if (row_ >= chunk_rows || col_ >= chunk_cols) {
          promise.SetResult(absl::OutOfRangeError("Tile index out of range"));
          return;
        }
        
        // Calculate tile index and get offset/size
        size_t tile_index = row_ * chunk_cols + col_;
        if (tile_index >= dir.tile_offsets.size()) {
          promise.SetResult(absl::OutOfRangeError("Tile index out of range"));
          return;
        }
        
        offset = dir.tile_offsets[tile_index];
        byte_count = dir.tile_bytecounts[tile_index];
      } else {
        // Strip-based TIFF
        chunk_rows = dir.strip_offsets.size();
        chunk_cols = 1;
        
        if (row_ >= chunk_rows || col_ != 0) {
          promise.SetResult(absl::OutOfRangeError("Strip index out of range"));
          return;
        }
        
        // Get strip offset/size
        offset = dir.strip_offsets[row_];
        byte_count = dir.strip_bytecounts[row_];
      }

      // Check if_equal and if_not_equal conditions
      if (!options_.generation_conditions.Matches(stamp.generation)) {
        promise.SetResult(kvstore::ReadResult::Unspecified(std::move(stamp)));
        return;
      }
      
      options.byte_range = OptionalByteRangeRequest::Range(
          offset, offset + byte_count);
    }

    options.generation_conditions.if_equal = stamp.generation;
    
    // Issue read for the tile/strip data
    auto future = owner_->base_.driver->Read(owner_->base_.path, std::move(options));
    future.Force();
    future.ExecuteWhenReady(
        [self = internal::IntrusivePtr<ReadState>(this), 
         original_byte_range = std::move(original_byte_range),
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
          
          // Apply byte range to the result if needed
          if (!original_byte_range.IsFull()) {
            // Validate the byte range against the actual size of the data
            auto size = read_result.value.size();
            auto byte_range_result = original_byte_range.Validate(size);
            
            if (!byte_range_result.ok()) {
              promise.SetResult(std::move(byte_range_result.status()));
              return;
            }
            
            // Apply the validated byte range
            ByteRange byte_range = byte_range_result.value();
            if (byte_range.inclusive_min > 0 || byte_range.exclusive_max < size) {
              read_result.value = read_result.value.Subcord(
                  byte_range.inclusive_min, byte_range.size());
            }
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
      
    // Get directory information
    assert(lock.data());
    const auto& dir = lock.data()->image_directory;
    
    // Currently only support IFD 0
    // Determine number of tiles/strips
    uint32_t chunk_rows, chunk_cols;
    if (dir.tile_width > 0) {
      // Tiled TIFF
      chunk_rows = (dir.height + dir.tile_height - 1) / dir.tile_height;
      chunk_cols = (dir.width + dir.tile_width - 1) / dir.tile_width;
    } else {
      // Strip-based TIFF
      chunk_rows = dir.strip_offsets.size();
      chunk_cols = 1;
    }
    
    // Generate tile/strip keys that match our range constraints
    for (uint32_t row = 0; row < chunk_rows; ++row) {
      for (uint32_t col = 0; col < chunk_cols; ++col) {
        // Create key in "tile/0/%d/%d" format
        std::string key = absl::StrFormat("tile/0/%d/%d", row, col);
        
        // Check if key is in the requested range
        if (tensorstore::Contains(options_.range, key)) {
          // For strips, get size from strip_bytecounts
          // For tiles, get size from tile_bytecounts
          size_t size;
          if (dir.tile_width > 0) {
            size_t index = row * chunk_cols + col;
            size = dir.tile_bytecounts[index];
          } else {
            size = dir.strip_bytecounts[row];
          }
          
          // Strip prefix if needed
          std::string adjusted_key = key;
          if (options_.strip_prefix_length > 0 && 
              options_.strip_prefix_length < key.size()) {
            adjusted_key = key.substr(options_.strip_prefix_length);
          }
          
          execution::set_value(receiver_, 
                              ListEntry{adjusted_key, ListEntry::checked_size(size)});
        }
      }
    }
  }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Spec::DoOpen
// ─────────────────────────────────────────────────────────────────────────────
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
  uint32_t ifd, row, col;
  if (auto st = ParseTileKey(key, ifd, row, col); !st.ok()) {
    // Instead of returning the error, return a "missing" result
    return MakeReadyFuture<ReadResult>(kvstore::ReadResult::Missing(
        TimestampedStorageGeneration{StorageGeneration::NoValue(), absl::Now()}));
  }

  auto state = internal::MakeIntrusivePtr<ReadState>();
  state->owner_ = internal::IntrusivePtr<TiffKeyValueStore>(this);
  state->key_ = std::move(key);
  state->options_ = options;
  state->ifd_ = ifd;
  state->row_ = row;
  state->col_ = col;

  return PromiseFuturePair<kvstore::ReadResult>::LinkValue(
      WithExecutor(executor(),
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

// GetTiffKeyValueStore factory function implementation
DriverPtr GetTiffKeyValueStore(DriverPtr base_kvstore) {
  auto driver = internal::MakeIntrusivePtr<TiffKeyValueStore>();
  driver->base_ = KvStore(base_kvstore);
  driver->spec_data_.data_copy_concurrency = Context::Resource<internal::DataCopyConcurrencyResource>::DefaultSpec();
  driver->spec_data_.cache_pool = Context::Resource<internal::CachePoolResource>::DefaultSpec();
  
  auto& cache_pool = *driver->spec_data_.cache_pool;
  std::string cache_key;
  internal::EncodeCacheKey(&cache_key, driver->base_.driver,
                         driver->base_.path,
                         driver->spec_data_.data_copy_concurrency);
                         
  auto directory_cache = internal::GetCache<TiffDirectoryCache>(
      cache_pool.get(), cache_key, [&] {
        return std::make_unique<TiffDirectoryCache>(
            driver->base_.driver,
            driver->spec_data_.data_copy_concurrency->executor);
      });
      
  driver->cache_entry_ =
      GetCacheEntry(directory_cache, driver->base_.path);
      
  return driver;
}

}  // namespace tensorstore::kvstore::tiff_kvstore

TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED(
    tensorstore::kvstore::tiff_kvstore::TiffKeyValueStore)

// ─────────────────────────────────────────────────────────────────────────────
//  Registration
// ─────────────────────────────────────────────────────────────────────────────
namespace {
const tensorstore::internal_kvstore::DriverRegistration<
    tensorstore::kvstore::tiff_kvstore::Spec>
    registration;
}  // namespace
