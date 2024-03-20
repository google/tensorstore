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

#ifndef TENSORSTORE_DRIVER_ZARR3_CHUNK_CACHE_H_
#define TENSORSTORE_DRIVER_ZARR3_CHUNK_CACHE_H_

#include <stddef.h>

#include <cstddef>
#include <memory>
#include <string>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tensorstore/array.h"
#include "tensorstore/driver/chunk.h"
#include "tensorstore/driver/read_request.h"
#include "tensorstore/driver/write_request.h"
#include "tensorstore/driver/zarr3/codec/codec.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/internal/cache/cache.h"
#include "tensorstore/internal/cache/chunk_cache.h"
#include "tensorstore/internal/cache/kvs_backed_chunk_cache.h"
#include "tensorstore/internal/chunk_grid_specification.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/lexicographical_grid_index_key.h"
#include "tensorstore/internal/storage_statistics.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_zarr3 {

/// Abstract base class for a chunked zarr array or sub-array.
///
/// This is used at the top level to operate on the entire array.  Additionally,
/// it is used to operate on a (possibly-nested) shard.
class ZarrChunkCache {
 public:
  virtual internal::Cache& cache() = 0;

  using Ptr = internal::CachePtr<ZarrChunkCache>;

  virtual const internal::ChunkGridSpecification& grid() const = 0;
  virtual const Executor& executor() const = 0;

  struct ReadRequest : internal::DriverReadRequest {
    absl::Time staleness_bound;
  };

  virtual void Read(ReadRequest request,
                    AnyFlowReceiver<absl::Status, internal::ReadChunk,
                                    IndexTransform<>>&& receiver) = 0;

  using WriteRequest = internal::DriverWriteRequest;

  virtual void Write(WriteRequest request,
                     AnyFlowReceiver<absl::Status, internal::WriteChunk,
                                     IndexTransform<>>&& receiver) = 0;

  struct GetStorageStatisticsRequest {
    internal::OpenTransactionPtr transaction;
    span<const Index> shape;
    IndexTransform<> transform;
    absl::Time staleness_bound;
  };

  virtual void GetStorageStatistics(
      internal::IntrusivePtr<internal::GetStorageStatisticsAsyncOperationState>
          state,
      GetStorageStatisticsRequest request) = 0;

  virtual const internal::LexicographicalGridIndexKeyParser&
  GetChunkStorageKeyParser() = 0;

  virtual kvstore::Driver* GetKvStoreDriver() = 0;

  virtual ~ZarrChunkCache();

  // If this is a nested chunk cache corresponding to a shard, points to parent
  // entry for the shard that holds a strong reference to this cache.  This is
  // used to initialize the weak reference owned by each entry of this cache.
  //
  // Due to the way cache eviction is handled, the weak references must be owned
  // by each entry within this cache individually, rather than by the cache
  // itself.  That way, the parent chunk can be evicted once all entries of this
  // cache are evicted.  If `parent_chunk_` were itself a
  // `WeakPinnedCacheEntry`, then there would be a reference cycle.
  //
  // `parent_chunk_` will be initialized when this cache is first constructed.
  // It must only be accessed while holding (directly or indirectly) a strong
  // reference to the parent entry, because this will be a dangling pointer
  // while destruction of the parent entry is in progress.
  internal::Cache::Entry* parent_chunk_ = nullptr;
};

/// Chunk cache for (a portion of) a Zarr array where each chunk has no further
/// sharding.
class ZarrLeafChunkCache : public internal::KvsBackedChunkCache,
                           public ZarrChunkCache {
  using Base = internal::KvsBackedChunkCache;

 public:
  using Base::executor;
  using Base::grid;

  class Entry : public Base::Entry {
   public:
    using OwningCache = ZarrLeafChunkCache;
    // Weak reference to the parent chunk, if applicable.
    internal::WeakPinnedCacheEntry parent_chunk;

    void DoInitialize() override {
      Base::Entry::DoInitialize();
      if (auto* parent_chunk_ptr = GetOwningCache(*this).parent_chunk_;
          parent_chunk_ptr) {
        parent_chunk = parent_chunk_ptr->AcquireWeakReference();
      }
    }
  };

  Entry* DoAllocateEntry() override { return new Entry; }
  size_t DoGetSizeofEntry() override { return sizeof(Entry); }

  explicit ZarrLeafChunkCache(kvstore::DriverPtr store,
                              ZarrCodecChain::PreparedState::Ptr codec_state);

  void Read(ZarrChunkCache::ReadRequest request,
            AnyFlowReceiver<absl::Status, internal::ReadChunk,
                            IndexTransform<>>&& receiver) override;

  void Write(ZarrChunkCache::WriteRequest request,
             AnyFlowReceiver<absl::Status, internal::WriteChunk,
                             IndexTransform<>>&& receiver) override;

  void GetStorageStatistics(
      internal::IntrusivePtr<internal::GetStorageStatisticsAsyncOperationState>
          state,
      ZarrChunkCache::GetStorageStatisticsRequest request) override;

  std::string GetChunkStorageKey(span<const Index> cell_indices) final;

  Result<absl::InlinedVector<SharedArray<const void>, 1>> DecodeChunk(
      span<const Index> chunk_indices, absl::Cord data) override;

  Result<absl::Cord> EncodeChunk(
      span<const Index> chunk_indices,
      span<const SharedArrayView<const void>> component_arrays) override;

  kvstore::Driver* GetKvStoreDriver() override;

  ZarrCodecChain::PreparedState::Ptr codec_state_;
};

/// Chunk cache for a Zarr array where each chunk is a shard.
class ZarrShardedChunkCache : public internal::Cache, public ZarrChunkCache {
  using Base = ZarrChunkCache;

 public:
  explicit ZarrShardedChunkCache(
      kvstore::DriverPtr store, ZarrCodecChain::PreparedState::Ptr codec_state);

  const ZarrShardingCodec::PreparedState& sharding_codec_state() const {
    return static_cast<const ZarrShardingCodec::PreparedState&>(
        *codec_state_->array_to_bytes);
  }

  void Read(ZarrChunkCache::ReadRequest request,
            AnyFlowReceiver<absl::Status, internal::ReadChunk,
                            IndexTransform<>>&& receiver) override;

  void Write(ZarrChunkCache::WriteRequest request,
             AnyFlowReceiver<absl::Status, internal::WriteChunk,
                             IndexTransform<>>&& receiver) override;

  void GetStorageStatistics(
      internal::IntrusivePtr<internal::GetStorageStatisticsAsyncOperationState>
          state,
      ZarrChunkCache::GetStorageStatisticsRequest request) override;

  Future<const void> DeleteCell(span<const Index> grid_cell_indices,
                                internal::OpenTransactionPtr transaction);

  class Entry : public internal::Cache::Entry {
   public:
    using OwningCache = ZarrShardedChunkCache;
    // Indicates whether the entry was initialized successfully.  This serves as
    // the error return channel for `DoInitialize`.
    absl::Status sharding_error;
    ZarrChunkCache::Ptr sub_chunk_cache;
    // Weak reference to the parent chunk, if applicable.
    internal::WeakPinnedCacheEntry parent_chunk;

    /// Returns the grid cell index vector corresponding to this cache entry.
    span<const Index> cell_indices() {
      return {reinterpret_cast<const Index*>(key().data()),
              static_cast<std::ptrdiff_t>(key().size() / sizeof(Index))};
    }

    void DoInitialize() override;
  };

  Entry* DoAllocateEntry() final { return new Entry; }
  size_t DoGetSizeofEntry() final { return sizeof(Entry); }

  kvstore::Driver* GetKvStoreDriver() override;

  kvstore::DriverPtr base_kvstore_;
  ZarrCodecChain::PreparedState::Ptr codec_state_;
};

/// Chunk cache mixin for a chunk cache where the entire chunk cache corresponds
/// to a single shard.
template <typename ChunkCacheImpl>
class ZarrShardSubChunkCache : public ChunkCacheImpl {
 public:
  explicit ZarrShardSubChunkCache(
      kvstore::DriverPtr store, Executor executor,
      ZarrShardingCodec::PreparedState::Ptr sharding_state)
      : ChunkCacheImpl(std::move(store),
                       ZarrCodecChain::PreparedState::Ptr(
                           sharding_state->sub_chunk_codec_state)),
        sharding_state_(std::move(sharding_state)),
        executor_(std::move(executor)) {}

  const internal::LexicographicalGridIndexKeyParser& GetChunkStorageKeyParser()
      override {
    return sharding_state_->GetSubChunkStorageKeyParser();
  }

  const internal::ChunkGridSpecification& grid() const override {
    return *sharding_state_->sub_chunk_grid;
  }
  const Executor& executor() const override { return executor_; }

  internal::Cache& cache() final { return *this; }

  ZarrShardingCodec::PreparedState::Ptr sharding_state_;
  Executor executor_;
};

// Creates a `ZarrChunkCache` for the specified `codec_chain`.
//
// If `codec_chain` is a sharding codec chain, then returns an instance of
// `CacheWrapper<ZarrShardedChunkCache>`.  Otherwise, returns an instance of
// `CacheWrapper<ZarrLeafChunkCache>`.
//
// \tparam ChunkCacheType Base class of both
//     `CacheWrapper<ZarrShardedChunkCache>` and
//     `CacheWrapper<ZarrLeafChunkCache>`.  This can be either `ZarrChunkCache`,
//     or another type that `CacheWrapper<T>` inherits from.
// \tparam CacheWrapper Cache type template.  `CacheWrapper<T>` must inherit
//     from `T`.  This is either `ZarrDataCache` (defined in `driver.cc)` to
//     create the top-level cache, or `ZarrShardSubChunkCache`, to create the
//     inner cache for a shard.
// \param codec_chain Codec chain that determine the type of cache to create.
// \param args Arguments to forward to the cache constructor.
template <typename ChunkCacheType, template <typename> class CacheWrapper,
          typename... U>
std::unique_ptr<ChunkCacheType> MakeZarrChunkCache(
    const ZarrCodecChain& codec_chain, U&&... args) {
  if (codec_chain.array_to_bytes->is_sharding_codec()) {
    return std::make_unique<CacheWrapper<ZarrShardedChunkCache>>(
        std::forward<U>(args)...);
  } else {
    return std::make_unique<CacheWrapper<ZarrLeafChunkCache>>(
        std::forward<U>(args)...);
  }
}

}  // namespace internal_zarr3
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_ZARR3_CHUNK_CACHE_H_
