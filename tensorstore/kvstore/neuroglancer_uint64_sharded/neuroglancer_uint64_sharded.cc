// Copyright 2020 The TensorStore Authors
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

#include "tensorstore/kvstore/neuroglancer_uint64_sharded/neuroglancer_uint64_sharded.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/internal/endian.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include <nlohmann/json.hpp>
#include "tensorstore/batch.h"
#include "tensorstore/context.h"
#include "tensorstore/internal/cache/async_cache.h"
#include "tensorstore/internal/cache/cache.h"
#include "tensorstore/internal/cache/cache_pool_resource.h"
#include "tensorstore/internal/cache/kvs_backed_cache.h"
#include "tensorstore/internal/data_copy_concurrency_resource.h"
#include "tensorstore/internal/estimate_heap_usage/estimate_heap_usage.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/mutex.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/kvstore/batch_util.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/key_range.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/neuroglancer_uint64_sharded/uint64_sharded.h"
#include "tensorstore/kvstore/neuroglancer_uint64_sharded/uint64_sharded_decoder.h"
#include "tensorstore/kvstore/neuroglancer_uint64_sharded/uint64_sharded_encoder.h"
#include "tensorstore/kvstore/operations.h"
#include "tensorstore/kvstore/read_modify_write.h"
#include "tensorstore/kvstore/read_result.h"
#include "tensorstore/kvstore/registry.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/kvstore/supported_features.h"
#include "tensorstore/kvstore/transaction.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/garbage_collection/fwd.h"
#include "tensorstore/util/garbage_collection/garbage_collection.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

// specializations
#include "tensorstore/internal/estimate_heap_usage/std_vector.h"  // IWYU pragma: keep
#include "tensorstore/util/execution/result_sender.h"  // IWYU pragma: keep

namespace tensorstore {
namespace neuroglancer_uint64_sharded {
namespace {

using ::tensorstore::internal::ConvertInvalidArgumentToFailedPrecondition;
using ::tensorstore::internal::IntrusivePtr;
using ::tensorstore::kvstore::ListEntry;
using ::tensorstore::kvstore::ListReceiver;
using ::tensorstore::kvstore::SupportedFeatures;

/// Read-only KeyValueStore for retrieving a minishard index
///
/// The key is a `ChunkCombinedShardInfo` (in native memory layout).  The value
/// is the encoded minishard index.
///
/// This is used by `MinishardIndexCache`, which decodes and caches the
/// minishard indices.  By using a separate `KeyValueStore` rather than just
/// including this logic directly in `MinishardIndexCache`, we are able to take
/// advantage of `KvsBackedCache` to define `MinishardIndexCache`.
class MinishardIndexKeyValueStore : public kvstore::Driver {
 public:
  explicit MinishardIndexKeyValueStore(kvstore::DriverPtr base,
                                       Executor executor,
                                       std::string key_prefix,
                                       const ShardingSpec& sharding_spec)
      : base_(std::move(base)),
        executor_(std::move(executor)),
        key_prefix_(key_prefix),
        sharding_spec_(sharding_spec) {}

  Future<ReadResult> Read(Key key, ReadOptions options) override;

  std::string DescribeKey(std::string_view key) override {
    ChunkCombinedShardInfo combined_info;
    if (key.size() != sizeof(combined_info)) {
      return tensorstore::StrCat("invalid key ", tensorstore::QuoteString(key));
    }
    std::memcpy(&combined_info, key.data(), sizeof(combined_info));
    auto split_info = GetSplitShardInfo(sharding_spec_, combined_info);
    return tensorstore::StrCat(
        "minishard ", split_info.minishard, " in ",
        base_->DescribeKey(
            GetShardKey(sharding_spec_, key_prefix_, split_info.shard)));
  }

  void GarbageCollectionVisit(
      garbage_collection::GarbageCollectionVisitor& visitor) const final {
    // No-op
  }

  kvstore::Driver* base() { return base_.get(); }
  const ShardingSpec& sharding_spec() { return sharding_spec_; }
  const std::string& key_prefix() const { return key_prefix_; }
  const Executor& executor() const { return executor_; }

  kvstore::DriverPtr base_;
  Executor executor_;
  std::string key_prefix_;
  ShardingSpec sharding_spec_;
};

namespace {

using ShardIndex = uint64_t;
using MinishardIndex = uint64_t;

// Reading a minishard index proceeds as follows:
//
// 1. Request the shard index.
//
//    a. If not found, the minishard is empty.  Done.
//
//    b. If the minishard index was cached and the generation has not
//       changed, the cached data is up to date.  Done.
//
//    c. Otherwise, decode the byte range of the minishard index.
//
// 2. Request the minishard index from the byte range specified in the
//    shard index.
//
//    a. If the generation has changed (concurrent modification of the
//       shard), retry starting at step 1.
//
//    b. If not found, the minishard is empty.  (This can only happens in
//       the case of concurrent modification of the shard).  Done.
//
//    c.  Otherwise, return the encoded minishard index.
class MinishardIndexReadOperationState;
using MinishardIndexReadOperationStateBase =
    internal_kvstore_batch::BatchReadEntry<
        MinishardIndexKeyValueStore,
        internal_kvstore_batch::ReadRequest<MinishardIndex>,
        // BatchEntryKey members:
        ShardIndex, kvstore::ReadGenerationConditions>;
;
class MinishardIndexReadOperationState
    : public MinishardIndexReadOperationStateBase,
      public internal::AtomicReferenceCount<MinishardIndexReadOperationState> {
 public:
  explicit MinishardIndexReadOperationState(BatchEntryKey&& batch_entry_key_)
      : MinishardIndexReadOperationStateBase(std::move(batch_entry_key_)),
        // Initial reference count that will be implicitly transferred to
        // `Submit`.
        internal::AtomicReferenceCount<MinishardIndexReadOperationState>(
            /*initial_ref_count=*/1) {}

 private:
  Batch retry_batch_{no_batch};

  void Submit(Batch::View batch) override {
    // Note: Submit is responsible for arranging to delete `this` eventually,
    // which it does via reference counting. Prior to `Submit` being called the
    // reference count isn't used.
    const auto& executor = driver().executor();
    executor(
        [this, batch = Batch(batch)] { this->ProcessBatch(std::move(batch)); });
  }

  void ProcessBatch(Batch batch) {
    // Take explicit ownership of the initial reference count.
    internal::IntrusivePtr<MinishardIndexReadOperationState> self(
        this, internal::adopt_object_ref);
    retry_batch_ = Batch::New();

    auto minishard_fetch_batch = Batch::New();

    for (auto& request : request_batch.requests) {
      ProcessMinishard(batch, request, minishard_fetch_batch);
    }
  }

  std::string ShardKey() {
    const auto& sharding_spec = driver().sharding_spec();
    return GetShardKey(sharding_spec, driver().key_prefix(),
                       std::get<ShardIndex>(batch_entry_key));
  }

  void ProcessMinishard(Batch::View batch, Request& request,
                        Batch minishard_fetch_batch) {
    kvstore::ReadOptions kvstore_read_options;
    kvstore_read_options.generation_conditions =
        std::get<kvstore::ReadGenerationConditions>(this->batch_entry_key);
    kvstore_read_options.staleness_bound = this->request_batch.staleness_bound;
    auto key = std::get<MinishardIndex>(request);
    kvstore_read_options.byte_range = OptionalByteRangeRequest{
        static_cast<int64_t>(key * 16), static_cast<int64_t>((key + 1) * 16)};
    kvstore_read_options.batch = batch;
    auto shard_index_read_future = this->driver().base()->Read(
        this->ShardKey(), std::move(kvstore_read_options));
    shard_index_read_future.Force();
    shard_index_read_future.ExecuteWhenReady(
        [self = internal::IntrusivePtr<MinishardIndexReadOperationState>(this),
         minishard_fetch_batch = std::move(minishard_fetch_batch),
         &request](ReadyFuture<kvstore::ReadResult> future) mutable {
          const auto& executor = self->driver().executor();
          executor([self = std::move(self), &request,
                    minishard_fetch_batch = std::move(minishard_fetch_batch),
                    future = std::move(future)] {
            OnShardIndexReady(std::move(self), request,
                              std::move(minishard_fetch_batch),
                              std::move(future.result()));
          });
        });
  }

  static void OnShardIndexReady(
      internal::IntrusivePtr<MinishardIndexReadOperationState> self,
      Request& request, Batch minishard_fetch_batch,
      Result<kvstore::ReadResult>&& result) {
    auto& byte_range_request =
        std::get<internal_kvstore_batch::ByteRangeReadRequest>(request);
    const auto set_error = [&](absl::Status status) {
      byte_range_request.promise.SetResult(MaybeAnnotateStatus(
          ConvertInvalidArgumentToFailedPrecondition(std::move(status)),
          "Error retrieving shard index entry"));
    };
    TENSORSTORE_ASSIGN_OR_RETURN(auto&& read_result, result,
                                 set_error(std::move(_)));
    if (  // Shard is empty (case 1a above)
        read_result.aborted() ||
        // Existing data is up to date (case 1b above).
        read_result.not_found()) {
      byte_range_request.promise.SetResult(std::move(read_result));
      return;
    }
    // Read was successful (case 1c above).
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto byte_range, DecodeShardIndexEntry(read_result.value.Flatten()),
        set_error(std::move(_)));
    TENSORSTORE_ASSIGN_OR_RETURN(
        byte_range,
        GetAbsoluteShardByteRange(byte_range, self->driver().sharding_spec()),
        set_error(std::move(_)));
    if (byte_range.size() == 0) {
      // Minishard index is 0 bytes, which means the minishard is empty.
      read_result.value.Clear();
      read_result.state = kvstore::ReadResult::kMissing;
      byte_range_request.promise.SetResult(std::move(read_result));
      return;
    }
    kvstore::ReadOptions kvstore_read_options;
    // The `if_equal` condition ensure that an "aborted" `ReadResult` is
    // returned in the case of a concurrent modification (case 2a above).
    kvstore_read_options.generation_conditions.if_equal =
        std::move(read_result.stamp.generation);
    kvstore_read_options.staleness_bound = self->request_batch.staleness_bound;
    kvstore_read_options.byte_range = byte_range;
    kvstore_read_options.batch = std::move(minishard_fetch_batch);
    auto read_future = self->driver().base()->Read(
        self->ShardKey(), std::move(kvstore_read_options));
    read_future.Force();
    read_future.ExecuteWhenReady(
        [self = std::move(self),
         &request](ReadyFuture<kvstore::ReadResult> future) mutable {
          const auto& executor = self->driver().executor();
          executor([self = std::move(self), &request,
                    future = std::move(future)]() mutable {
            self->OnMinishardIndexReadReady(request,
                                            std::move(future.result()));
          });
        });
  }

  void OnMinishardIndexReadReady(Request& request,
                                 Result<kvstore::ReadResult>&& result) {
    auto& byte_range_request =
        std::get<internal_kvstore_batch::ByteRangeReadRequest>(request);
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto&& read_result, result,
        static_cast<void>(byte_range_request.promise.SetResult(
            internal::ConvertInvalidArgumentToFailedPrecondition(
                std::move(_)))));
    if (read_result.aborted()) {
      // Shard was modified since the index was read (case 2a above).
      // Retry.
      MakeRequest<MinishardIndexReadOperationState>(
          driver(), std::get<ShardIndex>(batch_entry_key),
          kvstore::ReadGenerationConditions(
              std::get<kvstore::ReadGenerationConditions>(batch_entry_key)),
          retry_batch_, read_result.stamp.time, std::move(request));
      return;
    }

    // Shard was modified since the index was read, but minishard nonetheless
    // does not exist (case 2b above).
    //
    // or
    //
    // read was successful (case 2c above).
    byte_range_request.promise.SetResult(std::move(read_result));
  }
};
}  // namespace

Future<kvstore::ReadResult> MinishardIndexKeyValueStore::Read(
    Key key, ReadOptions options) {
  ChunkCombinedShardInfo combined_info;
  if (key.size() != sizeof(combined_info)) {
    return absl::InvalidArgumentError("Key does not specify a minishard");
  }
  std::memcpy(&combined_info, key.data(), sizeof(combined_info));
  auto split_info = GetSplitShardInfo(sharding_spec_, combined_info);
  if (options.byte_range != OptionalByteRangeRequest()) {
    // Byte range requests are not useful for minishard indices.
    return absl::InvalidArgumentError("Byte ranges not supported");
  }
  auto [promise, future] = PromiseFuturePair<ReadResult>::Make();
  MinishardIndexReadOperationState::MakeRequest<
      MinishardIndexReadOperationState>(
      *this, split_info.shard, std::move(options.generation_conditions),
      options.batch, options.staleness_bound,
      MinishardIndexReadOperationState::Request{{std::move(promise)},
                                                split_info.minishard});
  return std::move(future);
}

/// Caches minishard indexes.
///
/// Each cache entry corresponds to a particular minishard within a particular
/// shard.  The entry keys directly encode `ChunkCombinedShardInfo` values
/// (via `memcpy`), specifying a shard and minishard number.
///
/// This cache is only used for reading.
class MinishardIndexCache
    : public internal::KvsBackedCache<MinishardIndexCache,
                                      internal::AsyncCache> {
  using Base =
      internal::KvsBackedCache<MinishardIndexCache, internal::AsyncCache>;

 public:
  using ReadData = std::vector<MinishardIndexEntry>;

  class Entry : public Base::Entry {
   public:
    using OwningCache = MinishardIndexCache;

    ChunkSplitShardInfo shard_info() {
      ChunkCombinedShardInfo combined_info;
      assert(this->key().size() == sizeof(combined_info));
      std::memcpy(&combined_info, this->key().data(), sizeof(combined_info));
      return GetSplitShardInfo(GetOwningCache(*this).sharding_spec(),
                               combined_info);
    }

    size_t ComputeReadDataSizeInBytes(const void* read_data) override {
      return internal::EstimateHeapUsage(
          *static_cast<const ReadData*>(read_data));
    }

    void DoDecode(std::optional<absl::Cord> value,
                  DecodeReceiver receiver) override {
      GetOwningCache(*this).executor()(
          [this, value = std::move(value),
           receiver = std::move(receiver)]() mutable {
            std::shared_ptr<ReadData> read_data;
            if (value) {
              if (auto result = DecodeMinishardIndexAndAdjustByteRanges(
                      *value, GetOwningCache(*this).sharding_spec());
                  result.ok()) {
                read_data = std::make_shared<ReadData>(std::move(*result));
              } else {
                execution::set_error(receiver,
                                     ConvertInvalidArgumentToFailedPrecondition(
                                         std::move(result).status()));
                return;
              }
            }
            execution::set_value(receiver, std::move(read_data));
          });
    }
  };

  Entry* DoAllocateEntry() final { return new Entry; }
  size_t DoGetSizeofEntry() final { return sizeof(Entry); }
  TransactionNode* DoAllocateTransactionNode(AsyncCache::Entry& entry) final {
    return new TransactionNode(static_cast<Entry&>(entry));
  }

  explicit MinishardIndexCache(kvstore::DriverPtr base_kvstore,
                               Executor executor, std::string key_prefix,
                               const ShardingSpec& sharding_spec)
      : Base(kvstore::DriverPtr(new MinishardIndexKeyValueStore(
            std::move(base_kvstore), executor, std::move(key_prefix),
            sharding_spec))) {}

  MinishardIndexKeyValueStore* kvstore_driver() {
    return static_cast<MinishardIndexKeyValueStore*>(
        this->Base::kvstore_driver());
  }

  const ShardingSpec& sharding_spec() {
    return kvstore_driver()->sharding_spec();
  }

  kvstore::Driver* base_kvstore_driver() { return kvstore_driver()->base(); }
  const Executor& executor() { return kvstore_driver()->executor(); }
  const std::string& key_prefix() { return kvstore_driver()->key_prefix(); }
};

MinishardAndChunkId GetMinishardAndChunkId(std::string_view key) {
  assert(key.size() == 16);
  return {absl::big_endian::Load64(key.data()),
          {absl::big_endian::Load64(key.data() + 8)}};
}

/// Cache used to buffer writes to the KeyValueStore.
///
/// Each cache entry correspond to a particular shard.  The entry key directly
/// encodes the uint64 shard number (via `memcpy`).
///
/// This cache is used only for writing, not for reading.  However, in order to
/// update existing non-empty shards, it does read the full contents of the
/// existing shard and store it within the cache entry.  This data is discarded
/// once writeback completes.
class ShardedKeyValueStoreWriteCache
    : public internal::KvsBackedCache<ShardedKeyValueStoreWriteCache,
                                      internal::AsyncCache> {
  using Base = internal::KvsBackedCache<ShardedKeyValueStoreWriteCache,
                                        internal::AsyncCache>;

 public:
  using ReadData = EncodedChunks;

  static std::string ShardToKey(ShardIndex shard) {
    std::string key;
    key.resize(sizeof(ShardIndex));
    absl::big_endian::Store64(key.data(), shard);
    return key;
  }

  static ShardIndex KeyToShard(std::string_view key) {
    assert(key.size() == sizeof(ShardIndex));
    return absl::big_endian::Load64(key.data());
  }

  class Entry : public Base::Entry {
   public:
    using OwningCache = ShardedKeyValueStoreWriteCache;

    ShardIndex shard() { return KeyToShard(key()); }

    size_t ComputeReadDataSizeInBytes(const void* data) override {
      return internal::EstimateHeapUsage(*static_cast<const ReadData*>(data));
    }

    void DoDecode(std::optional<absl::Cord> value,
                  DecodeReceiver receiver) override {
      GetOwningCache(*this).executor()(
          [this, value = std::move(value),
           receiver = std::move(receiver)]() mutable {
            EncodedChunks chunks;
            if (value) {
              if (auto result =
                      SplitShard(GetOwningCache(*this).sharding_spec(), *value);
                  result.ok()) {
                chunks = std::move(*result);
              } else {
                execution::set_error(receiver,
                                     ConvertInvalidArgumentToFailedPrecondition(
                                         std::move(result).status()));
                return;
              }
            }
            execution::set_value(
                receiver, std::make_shared<EncodedChunks>(std::move(chunks)));
          });
    }

    void DoEncode(std::shared_ptr<const EncodedChunks> data,
                  EncodeReceiver receiver) override {
      // Can call `EncodeShard` synchronously without using our executor since
      // `DoEncode` is already guaranteed to be called from our executor.
      execution::set_value(
          receiver, EncodeShard(GetOwningCache(*this).sharding_spec(), *data));
    }

    std::string GetKeyValueStoreKey() override {
      auto& cache = GetOwningCache(*this);
      return GetShardKey(cache.sharding_spec(), cache.key_prefix(),
                         this->shard());
    }
  };

  class TransactionNode : public Base::TransactionNode,
                          public internal_kvstore::AtomicMultiPhaseMutation {
   public:
    using OwningCache = ShardedKeyValueStoreWriteCache;
    using Base::TransactionNode::TransactionNode;

    absl::Mutex& mutex() override { return this->mutex_; }

    void PhaseCommitDone(size_t next_phase) override {}

    internal::TransactionState::Node& GetTransactionNode() override {
      return *this;
    }

    void Abort() override {
      this->AbortRemainingPhases();
      Base::TransactionNode::Abort();
    }

    std::string DescribeKey(std::string_view key) override {
      auto& entry = GetOwningEntry(*this);
      auto& cache = GetOwningCache(entry);
      auto minishard_and_chunk_id = GetMinishardAndChunkId(key);
      return tensorstore::StrCat(
          "chunk ", minishard_and_chunk_id.chunk_id.value, " in minishard ",
          minishard_and_chunk_id.minishard, " in ",
          cache.kvstore_driver()->DescribeKey(entry.GetKeyValueStoreKey()));
    }

    void DoApply(ApplyOptions options, ApplyReceiver receiver) override;
    void AllEntriesDone(
        internal_kvstore::SinglePhaseMutation& single_phase_mutation) override;
    void RecordEntryWritebackError(
        internal_kvstore::ReadModifyWriteEntry& entry,
        absl::Status error) override {
      absl::MutexLock lock(&mutex_);
      if (apply_status_.ok()) {
        apply_status_ = std::move(error);
      }
    }

    void Revoke() override {
      Base::TransactionNode::Revoke();
      { UniqueWriterLock(*this); }
      // At this point, no new entries may be added and we can safely traverse
      // the list of entries without a lock.
      this->RevokeAllEntries();
    }

    void WritebackSuccess(ReadState&& read_state) override;
    void WritebackError() override;

    void InvalidateReadState() override;

    bool MultiPhaseReadsCommitted() override { return this->reads_committed_; }

    /// Handles transactional read requests for single chunks.
    ///
    /// Always reads the full shard, and then decodes the individual chunk
    /// within it.
    void Read(
        internal_kvstore::ReadModifyWriteEntry& entry,
        kvstore::ReadModifyWriteTarget::TransactionalReadOptions&& options,
        kvstore::ReadModifyWriteTarget::ReadReceiver&& receiver) override {
      this->AsyncCache::TransactionNode::Read({options.staleness_bound})
          .ExecuteWhenReady(WithExecutor(
              GetOwningCache(*this).executor(),
              [&entry,
               if_not_equal =
                   std::move(options.generation_conditions.if_not_equal),
               receiver = std::move(receiver)](
                  ReadyFuture<const void> future) mutable {
                if (!future.result().ok()) {
                  execution::set_error(receiver, future.result().status());
                  return;
                }
                execution::submit(HandleShardReadSuccess(entry, if_not_equal),
                                  receiver);
              }));
    }

    /// Called asynchronously from `Read` when the full shard is ready.
    static Result<kvstore::ReadResult> HandleShardReadSuccess(
        internal_kvstore::ReadModifyWriteEntry& entry,
        const StorageGeneration& if_not_equal) {
      auto& self = static_cast<TransactionNode&>(entry.multi_phase());
      TimestampedStorageGeneration stamp;
      std::shared_ptr<const EncodedChunks> encoded_chunks;
      {
        AsyncCache::ReadLock<EncodedChunks> lock{self};
        stamp = lock.stamp();
        encoded_chunks = lock.shared_data();
      }
      if (!StorageGeneration::IsUnknown(stamp.generation) &&
          stamp.generation == if_not_equal) {
        return kvstore::ReadResult::Unspecified(std::move(stamp));
      }
      if (StorageGeneration::IsDirty(stamp.generation)) {
        // Add layer to generation in order to make it possible to
        // distinguish:
        //
        // 1. the shard being modified by a predecessor
        //    `ReadModifyWrite` operation on the underlying
        //    KeyValueStore.
        //
        // 2. the chunk being modified by a `ReadModifyWrite`
        //    operation attached to this transaction node.
        stamp.generation =
            StorageGeneration::AddLayer(std::move(stamp.generation));
      }
      auto* chunk =
          FindChunk(*encoded_chunks, GetMinishardAndChunkId(entry.key_));
      if (!chunk) {
        return kvstore::ReadResult::Missing(std::move(stamp));
      } else {
        TENSORSTORE_ASSIGN_OR_RETURN(
            absl::Cord value,
            DecodeData(chunk->encoded_data,
                       GetOwningCache(self).sharding_spec().data_encoding));
        return kvstore::ReadResult::Value(std::move(value), std::move(stamp));
      }
    }

    void Writeback(internal_kvstore::ReadModifyWriteEntry& entry,
                   kvstore::ReadResult&& read_result) override {
      auto& value = read_result.value;
      if (read_result.state == kvstore::ReadResult::kValue) {
        value = EncodeData(value,
                           GetOwningCache(*this).sharding_spec().data_encoding);
      }
      internal_kvstore::AtomicMultiPhaseMutation::Writeback(
          entry, std::move(read_result));
    }

    ApplyReceiver apply_receiver_;
    ApplyOptions apply_options_;
    absl::Status apply_status_;
  };

  Entry* DoAllocateEntry() final { return new Entry; }
  size_t DoGetSizeofEntry() final { return sizeof(Entry); }
  TransactionNode* DoAllocateTransactionNode(AsyncCache::Entry& entry) final {
    return new TransactionNode(static_cast<Entry&>(entry));
  }

  explicit ShardedKeyValueStoreWriteCache(
      internal::CachePtr<MinishardIndexCache> minishard_index_cache,
      GetMaxChunksPerShardFunction get_max_chunks_per_shard)
      : Base(kvstore::DriverPtr(minishard_index_cache->base_kvstore_driver())),
        minishard_index_cache_(std::move(minishard_index_cache)),
        get_max_chunks_per_shard_(std::move(get_max_chunks_per_shard)) {}

  const ShardingSpec& sharding_spec() const {
    return minishard_index_cache()->sharding_spec();
  }

  const std::string& key_prefix() const {
    return minishard_index_cache()->key_prefix();
  }

  const internal::CachePtr<MinishardIndexCache>& minishard_index_cache() const {
    return minishard_index_cache_;
  }

  const Executor& executor() { return minishard_index_cache()->executor(); }

  internal::CachePtr<MinishardIndexCache> minishard_index_cache_;

  GetMaxChunksPerShardFunction get_max_chunks_per_shard_;
};

void ShardedKeyValueStoreWriteCache::TransactionNode::InvalidateReadState() {
  Base::TransactionNode::InvalidateReadState();
  internal_kvstore::InvalidateReadState(phases_);
}

void ShardedKeyValueStoreWriteCache::TransactionNode::WritebackSuccess(
    ReadState&& read_state) {
  for (auto& entry : phases_.entries_) {
    internal_kvstore::WritebackSuccess(
        static_cast<internal_kvstore::ReadModifyWriteEntry&>(entry),
        read_state.stamp);
  }
  internal_kvstore::DestroyPhaseEntries(phases_);
  Base::TransactionNode::WritebackSuccess(std::move(read_state));
}

void ShardedKeyValueStoreWriteCache::TransactionNode::WritebackError() {
  internal_kvstore::WritebackError(phases_);
  internal_kvstore::DestroyPhaseEntries(phases_);
  Base::TransactionNode::WritebackError();
}

namespace {
void StartApply(ShardedKeyValueStoreWriteCache::TransactionNode& node) {
  RetryAtomicWriteback(node.phases_, node.apply_options_.staleness_bound);
}

/// Attempts to compute the new encoded shard state that merges any mutations
/// into the existing state.
///
/// This is used by the implementation of
/// `ShardedKeyValueStoreWriteCache::TransactionNode::DoApply`.
///
/// If all conditional mutations are based on a consistent existing state
/// generation, sends the new state to `node.apply_receiver_`.  Otherwise, calls
/// `StartApply` to retry with an updated existing state.
void MergeForWriteback(ShardedKeyValueStoreWriteCache::TransactionNode& node,
                       bool conditional) {
  TimestampedStorageGeneration stamp;
  std::shared_ptr<const EncodedChunks> shared_existing_chunks;
  span<const EncodedChunk> existing_chunks;
  if (conditional) {
    // The new shard state depends on the existing shard state.  We will need to
    // merge the mutations with the existing chunks.  Additionally, any
    // conditional mutations must be consistent with `stamp.generation`.
    auto lock = internal::AsyncCache::ReadLock<EncodedChunks>{node};
    stamp = lock.stamp();
    shared_existing_chunks = lock.shared_data();
    existing_chunks = *shared_existing_chunks;
  } else {
    // The new shard state is guaranteed not to depend on the existing shard
    // state.  We will merge the mutations into an empty set of existing chunks.
    stamp = TimestampedStorageGeneration::Unconditional();
  }

  std::vector<EncodedChunk> chunks;
  // Index of next chunk in `existing_chunks` not yet merged into `chunks`.
  size_t existing_index = 0;
  // Indicates that inconsistent conditional mutations were observed.
  bool mismatch = false;
  // Indicates that the new shard state is not identical to the existing shard
  // state.
  bool changed = false;
  // Due to the encoding of minishard and chunk id into the key, entries are
  // guaranteed to be ordered by minishard and then by chunk id, which is the
  // order required for encoding.
  for (auto& entry : node.phases_.entries_) {
    auto& buffered_entry =
        static_cast<internal_kvstore::AtomicMultiPhaseMutation::
                        BufferedReadModifyWriteEntry&>(entry);
    if (StorageGeneration::IsConditional(
            buffered_entry.read_result_.stamp.generation) &&
        StorageGeneration::Clean(
            buffered_entry.read_result_.stamp.generation) !=
            StorageGeneration::Clean(stamp.generation)) {
      // This mutation is conditional, and is inconsistent with a prior
      // conditional mutation or with `existing_chunks`.
      mismatch = true;
      break;
    }
    if (buffered_entry.read_result_.state ==
            kvstore::ReadResult::kUnspecified ||
        !StorageGeneration::IsInnerLayerDirty(
            buffered_entry.read_result_.stamp.generation)) {
      // This is a no-op mutation; ignore it, which has the effect of retaining
      // the existing chunk with this id, if present in `existing_chunks`.
      continue;
    }
    auto minishard_and_chunk_id = GetMinishardAndChunkId(buffered_entry.key_);
    // Merge in chunks from `existing_chunks` that do not occur (in the
    // (minishard id, chunk id) ordering) after this chunk.
    while (existing_index < static_cast<size_t>(existing_chunks.size())) {
      auto& existing_chunk = existing_chunks[existing_index];
      if (existing_chunk.minishard_and_chunk_id < minishard_and_chunk_id) {
        // Include the existing chunk.
        chunks.push_back(existing_chunk);
        ++existing_index;
      } else if (existing_chunk.minishard_and_chunk_id ==
                 minishard_and_chunk_id) {
        // Skip the existing chunk.
        changed = true;
        ++existing_index;
        break;
      } else {
        // All remaining existing chunks occur after this chunk.
        break;
      }
    }
    if (buffered_entry.read_result_.state == kvstore::ReadResult::kValue) {
      // The mutation specifies a new value (rather than a deletion).
      chunks.push_back(EncodedChunk{minishard_and_chunk_id,
                                    buffered_entry.read_result_.value});
      changed = true;
    }
  }
  if (mismatch) {
    // We can't proceed, because the existing chunks (if applicable) and the
    // conditional mutations are not all based on a consistent existing shard
    // generation.  Retry, requesting that all mutations be based on a new
    // up-to-date existing shard generation, which will normally lead to a
    // consistent set.  In the rare case where there are both concurrent
    // external modifications to the existing shard, and also concurrent
    // requests for updated writeback states, multiple retries may be required.
    // However, even in that case, the extra retries aren't really an added
    // cost, because those retries would still be needed anyway due to the
    // concurrent modifications.
    node.apply_options_.staleness_bound = absl::Now();
    StartApply(node);
    return;
  }
  // Merge in any remaining existing chunks that occur after all mutated chunks.
  chunks.insert(chunks.end(), existing_chunks.begin() + existing_index,
                existing_chunks.end());
  internal::AsyncCache::ReadState update;
  update.stamp = std::move(stamp);
  if (changed) {
    update.stamp.generation.MarkDirty();
  }
  update.data = std::make_shared<EncodedChunks>(std::move(chunks));
  execution::set_value(std::exchange(node.apply_receiver_, {}),
                       std::move(update));
}

}  // namespace

void ShardedKeyValueStoreWriteCache::TransactionNode::DoApply(
    ApplyOptions options, ApplyReceiver receiver) {
  apply_receiver_ = std::move(receiver);
  apply_options_ = options;
  apply_status_ = absl::Status();

  GetOwningCache(*this).executor()([this] { StartApply(*this); });
}

void ShardedKeyValueStoreWriteCache::TransactionNode::AllEntriesDone(
    internal_kvstore::SinglePhaseMutation& single_phase_mutation) {
  if (!apply_status_.ok()) {
    execution::set_error(std::exchange(apply_receiver_, {}),
                         std::exchange(apply_status_, {}));
    return;
  }
  auto& self = *this;
  GetOwningCache(*this).executor()([&self] {
    TimestampedStorageGeneration stamp;
    bool mismatch = false;
    bool modified = false;

    size_t num_chunks = 0;

    // Determine if the writeback result depends on the existing shard, and if
    // all chunks are conditioned on the same generation.
    for (auto& entry : self.phases_.entries_) {
      auto& buffered_entry =
          static_cast<AtomicMultiPhaseMutation::BufferedReadModifyWriteEntry&>(
              entry);
      if (buffered_entry.read_result_.state !=
          kvstore::ReadResult::kUnspecified) {
        modified = true;
        ++num_chunks;
      }
      auto& entry_stamp = buffered_entry.read_result_.stamp;
      if (StorageGeneration::IsConditional(entry_stamp.generation)) {
        if (!StorageGeneration::IsUnknown(stamp.generation) &&
            StorageGeneration::Clean(stamp.generation) !=
                StorageGeneration::Clean(entry_stamp.generation)) {
          mismatch = true;
          break;
        } else {
          stamp = entry_stamp;
        }
      }
    }

    if (mismatch) {
      self.apply_options_.staleness_bound = absl::Now();
      StartApply(self);
      return;
    }
    auto& cache = GetOwningCache(self);
    if (!modified && StorageGeneration::IsUnknown(stamp.generation) &&
        self.apply_options_.apply_mode !=
            ApplyOptions::ApplyMode::kSpecifyUnchanged) {
      internal::AsyncCache::ReadState update;
      update.stamp = TimestampedStorageGeneration::Unconditional();
      execution::set_value(std::exchange(self.apply_receiver_, {}),
                           std::move(update));
      return;
    }
    if (!StorageGeneration::IsUnknown(stamp.generation) ||
        !cache.get_max_chunks_per_shard_ ||
        cache.get_max_chunks_per_shard_(GetOwningEntry(self).shard()) !=
            num_chunks) {
      self.internal::AsyncCache::TransactionNode::Read(
              {self.apply_options_.staleness_bound})
          .ExecuteWhenReady([&self](ReadyFuture<const void> future) {
            if (!future.result().ok()) {
              execution::set_error(std::exchange(self.apply_receiver_, {}),
                                   future.result().status());
              return;
            }
            GetOwningCache(self).executor()(
                [&self] { MergeForWriteback(self, /*conditional=*/true); });
          });
      return;
    }
    MergeForWriteback(self, /*conditional=*/false);
  });
}

Result<ChunkId> KeyToChunkIdOrError(std::string_view key) {
  if (auto chunk_id = KeyToChunkId(key)) {
    return *chunk_id;
  }
  return absl::InvalidArgumentError(
      tensorstore::StrCat("Invalid key: ", tensorstore::QuoteString(key)));
}

}  // namespace

struct ShardedKeyValueStoreSpecData {
  Context::Resource<internal::CachePoolResource> cache_pool;
  Context::Resource<internal::DataCopyConcurrencyResource>
      data_copy_concurrency;
  kvstore::Spec base;
  ShardingSpec metadata;
  TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(ShardedKeyValueStoreSpecData,
                                          internal_json_binding::NoOptions,
                                          IncludeDefaults,
                                          ::nlohmann::json::object_t)

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.cache_pool, x.data_copy_concurrency, x.base, x.metadata);
  };
};

namespace jb = ::tensorstore::internal_json_binding;

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(
    ShardedKeyValueStoreSpecData,
    jb::Object(
        jb::Member("base",
                   jb::Projection<&ShardedKeyValueStoreSpecData::base>()),
        jb::Initialize([](auto* obj) {
          internal::EnsureDirectoryPath(obj->base.path);
          return absl::OkStatus();
        }),
        jb::Member("metadata",
                   jb::Projection<&ShardedKeyValueStoreSpecData::metadata>(
                       jb::DefaultInitializedValue())),
        jb::Member(internal::CachePoolResource::id,
                   jb::Projection<&ShardedKeyValueStoreSpecData::cache_pool>()),
        jb::Member(
            internal::DataCopyConcurrencyResource::id,
            jb::Projection<
                &ShardedKeyValueStoreSpecData::data_copy_concurrency>())));

class ShardedKeyValueStoreSpec
    : public internal_kvstore::RegisteredDriverSpec<
          ShardedKeyValueStoreSpec, ShardedKeyValueStoreSpecData> {
 public:
  static constexpr char id[] = "neuroglancer_uint64_sharded";
  Future<kvstore::DriverPtr> DoOpen() const override;

  Result<kvstore::Spec> GetBase(std::string_view path) const override {
    return data_.base;
  }
};

class ShardedKeyValueStore
    : public internal_kvstore::RegisteredDriver<ShardedKeyValueStore,
                                                ShardedKeyValueStoreSpec> {
 public:
  explicit ShardedKeyValueStore(
      kvstore::DriverPtr base_kvstore, Executor executor,
      std::string key_prefix, const ShardingSpec& sharding_spec,
      internal::CachePool::WeakPtr cache_pool,
      GetMaxChunksPerShardFunction get_max_chunks_per_shard = {})
      : write_cache_(internal::GetCache<ShardedKeyValueStoreWriteCache>(
            cache_pool.get(), "",
            [&] {
              return std::make_unique<ShardedKeyValueStoreWriteCache>(
                  internal::GetCache<MinishardIndexCache>(
                      cache_pool.get(), "",
                      [&] {
                        return std::make_unique<MinishardIndexCache>(
                            std::move(base_kvstore), std::move(executor),
                            std::move(key_prefix), sharding_spec);
                      }),
                  std::move(get_max_chunks_per_shard));
            })),
        is_raw_encoding_(sharding_spec.data_encoding ==
                         ShardingSpec::DataEncoding::raw) {}

  Future<ReadResult> Read(Key key, ReadOptions options) override;

  void ListImpl(ListOptions options, ListReceiver receiver) override {
    struct State {
      ListReceiver receiver_;
      Promise<void> promise_;
      Future<void> future_;
      ListOptions options_;

      explicit State(ListReceiver&& receiver, ListOptions&& options)
          : receiver_(std::move(receiver)), options_(std::move(options)) {
        auto [promise, future] = PromiseFuturePair<void>::Make(MakeResult());
        this->promise_ = std::move(promise);
        this->future_ = std::move(future);
        future_.Force();
        execution::set_starting(receiver_, [promise = promise_] {
          promise.SetResult(absl::CancelledError(""));
        });
      }
      ~State() {
        auto& r = promise_.raw_result();
        if (r.ok()) {
          execution::set_done(receiver_);
        } else {
          execution::set_error(receiver_, r.status());
        }
        execution::set_stopping(receiver_);
      }
    };
    auto state =
        std::make_shared<State>(std::move(receiver), std::move(options));
    // Inefficient, but only used for testing.
    ShardIndex num_shards = ShardIndex{1} << sharding_spec().shard_bits;
    for (ShardIndex shard = 0; shard < num_shards; ++shard) {
      auto entry = GetCacheEntry(
          write_cache_, ShardedKeyValueStoreWriteCache::ShardToKey(shard));
      LinkValue(
          [state, entry, is_raw_encoding = is_raw_encoding_](
              Promise<void> promise, ReadyFuture<const void> future) {
            auto chunks = internal::AsyncCache::ReadLock<EncodedChunks>(*entry)
                              .shared_data();
            if (!chunks) return;
            for (auto& chunk : *chunks) {
              auto key = ChunkIdToKey(chunk.minishard_and_chunk_id.chunk_id);
              if (!Contains(state->options_.range, key)) continue;
              key.erase(0, state->options_.strip_prefix_length);
              execution::set_value(
                  state->receiver_,
                  ListEntry{
                      std::move(key),
                      is_raw_encoding
                          ? ListEntry::checked_size(chunk.encoded_data.size())
                          : -1,
                  });
            }
          },
          state->promise_, entry->Read({absl::InfiniteFuture()}));
    }
  }

  Future<TimestampedStorageGeneration> Write(Key key,
                                             std::optional<Value> value,
                                             WriteOptions options) override {
    return internal_kvstore::WriteViaTransaction(
        this, std::move(key), std::move(value), std::move(options));
  }

  absl::Status ReadModifyWrite(internal::OpenTransactionPtr& transaction,
                               size_t& phase, Key key,
                               ReadModifyWriteSource& source) override {
    TENSORSTORE_ASSIGN_OR_RETURN(ChunkId chunk_id, KeyToChunkIdOrError(key));
    const auto& sharding_spec = this->sharding_spec();
    const auto shard_info = GetSplitShardInfo(
        sharding_spec, GetChunkShardInfo(sharding_spec, chunk_id));
    const ShardIndex shard = shard_info.shard;
    auto entry = GetCacheEntry(
        write_cache_, ShardedKeyValueStoreWriteCache::ShardToKey(shard));
    std::string key_within_shard;
    key_within_shard.resize(16);
    absl::big_endian::Store64(key_within_shard.data(), shard_info.minishard);
    absl::big_endian::Store64(key_within_shard.data() + 8, chunk_id.value);
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto node, GetWriteLockedTransactionNode(*entry, transaction));
    node->ReadModifyWrite(phase, std::move(key_within_shard), source);
    if (!transaction) {
      transaction.reset(node.unlock()->transaction());
    }
    return absl::OkStatus();
  }

  absl::Status TransactionalDeleteRange(
      const internal::OpenTransactionPtr& transaction,
      KeyRange range) override {
    return absl::UnimplementedError("DeleteRange not supported");
  }

  std::string DescribeKey(std::string_view key) override {
    auto chunk_id_opt = KeyToChunkId(key);
    if (!chunk_id_opt) {
      return tensorstore::StrCat("invalid key ", tensorstore::QuoteString(key));
    }
    const auto& sharding_spec = this->sharding_spec();
    const auto shard_info = GetSplitShardInfo(
        sharding_spec, GetChunkShardInfo(sharding_spec, *chunk_id_opt));
    return tensorstore::StrCat(
        "chunk ", chunk_id_opt->value, " in minishard ", shard_info.minishard,
        " in ",
        base_kvstore_driver()->DescribeKey(
            GetShardKey(sharding_spec, key_prefix(), shard_info.shard)));
  }

  SupportedFeatures GetSupportedFeatures(
      const KeyRange& key_range) const final {
    return base_kvstore_driver()->GetSupportedFeatures(
        KeyRange::Prefix(key_prefix()));
  }

  Result<KvStore> GetBase(std::string_view path,
                          const Transaction& transaction) const override {
    return KvStore(kvstore::DriverPtr(base_kvstore_driver()), key_prefix(),
                   transaction);
  }

  kvstore::Driver* base_kvstore_driver() const {
    return minishard_index_cache()->base_kvstore_driver();
  }
  const ShardingSpec& sharding_spec() const {
    return minishard_index_cache()->sharding_spec();
  }
  const Executor& executor() const {
    return minishard_index_cache()->executor();
  }
  const std::string& key_prefix() const {
    return minishard_index_cache()->key_prefix();
  }

  const internal::CachePtr<MinishardIndexCache>& minishard_index_cache() const {
    return write_cache_->minishard_index_cache_;
  }

  absl::Status GetBoundSpecData(ShardedKeyValueStoreSpecData& spec) const;

  internal::CachePtr<ShardedKeyValueStoreWriteCache> write_cache_;
  Context::Resource<internal::CachePoolResource> cache_pool_resource_;
  Context::Resource<internal::DataCopyConcurrencyResource>
      data_copy_concurrency_resource_;
  bool is_raw_encoding_ = false;
};

namespace {
class ReadOperationState;
using ReadOperationStateBase = internal_kvstore_batch::BatchReadEntry<
    ShardedKeyValueStore,
    internal_kvstore_batch::ReadRequest<MinishardAndChunkId,
                                        kvstore::ReadGenerationConditions>,
    // BatchEntryKey members:
    ShardIndex>;
class ReadOperationState
    : public ReadOperationStateBase,
      public internal::AtomicReferenceCount<ReadOperationState> {
 public:
  explicit ReadOperationState(BatchEntryKey&& batch_entry_key_)
      : ReadOperationStateBase(std::move(batch_entry_key_)),
        // Initial reference that will be transferred to `Submit`.
        internal::AtomicReferenceCount<ReadOperationState>(
            /*initial_ref_count=*/1) {}

 private:
  Batch retry_batch_{no_batch};

  void Submit(Batch::View batch) override {
    // Note: Submit is responsible for arranging to delete `this` eventually,
    // which it does via reference counting. Prior to `Submit` being called the
    // reference count isn't used.
    const auto& executor = driver().executor();
    executor(
        [this, batch = Batch(batch)] { this->ProcessBatch(std::move(batch)); });
  }

  void ProcessBatch(Batch batch) {
    // Take ownership of initial reference.
    internal::IntrusivePtr<ReadOperationState> self(this,
                                                    internal::adopt_object_ref);
    span<Request> requests = request_batch.requests;
    std::sort(request_batch.requests.begin(), request_batch.requests.end(),
              [](const Request& a, const Request& b) {
                return std::get<MinishardAndChunkId>(a) <
                       std::get<MinishardAndChunkId>(b);
              });

    if (ShouldReadEntireShard()) {
      ReadEntireShard(std::move(self), std::move(batch));
      return;
    }

    retry_batch_ = Batch::New();

    Batch data_fetch_batch{no_batch};

    for (size_t minishard_start_i = 0; minishard_start_i < requests.size();) {
      size_t minishard_end_i = minishard_start_i + 1;
      auto minishard =
          std::get<MinishardAndChunkId>(requests[minishard_start_i]).minishard;
      while (
          minishard_end_i < requests.size() &&
          std::get<MinishardAndChunkId>(requests[minishard_end_i]).minishard ==
              minishard) {
        ++minishard_end_i;
      }
      ProcessMinishard(batch, minishard,
                       requests.subspan(minishard_start_i,
                                        minishard_end_i - minishard_start_i),
                       data_fetch_batch);
      minishard_start_i = minishard_end_i;
    }
  }

  bool ShouldReadEntireShard() {
    const auto& get_max_chunks_per_shard =
        driver().write_cache_->get_max_chunks_per_shard_;
    if (!get_max_chunks_per_shard) return false;
    uint64_t max_chunks =
        get_max_chunks_per_shard(std::get<ShardIndex>(batch_entry_key));
    if (request_batch.requests.size() < max_chunks) {
      // The requests can't possibly cover all of the chunks.
      return false;
    }
    const auto& first_request = request_batch.requests[0];

    uint64_t num_chunks_covered = 0;
    std::optional<uint64_t> prev_chunk_covered;
    for (const auto& request : request_batch.requests) {
      if (std::get<kvstore::ReadGenerationConditions>(request) !=
          std::get<kvstore::ReadGenerationConditions>(first_request)) {
        // Generation constraints are not all the same.
        return false;
      }
      if (std::get<internal_kvstore_batch::ByteRangeReadRequest>(request)
              .byte_range.IsFull()) {
        uint64_t chunk_id =
            std::get<MinishardAndChunkId>(request).chunk_id.value;
        if (chunk_id != prev_chunk_covered) {
          prev_chunk_covered = chunk_id;
          ++num_chunks_covered;
        }
      }
    }
    return (num_chunks_covered == max_chunks);
  }

  std::string ShardKey() {
    const auto& sharding_spec = driver().sharding_spec();
    return GetShardKey(sharding_spec, driver().key_prefix(),
                       std::get<ShardIndex>(batch_entry_key));
  }

  static void ReadEntireShard(internal::IntrusivePtr<ReadOperationState> self,
                              Batch batch) {
    auto& first_request = self->request_batch.requests[0];
    kvstore::ReadOptions read_options;
    read_options.batch = std::move(batch);
    read_options.generation_conditions =
        std::move(std::get<kvstore::ReadGenerationConditions>(first_request));
    read_options.staleness_bound = self->request_batch.staleness_bound;
    auto& driver = self->driver();
    auto read_future = driver.base_kvstore_driver()->Read(
        GetShardKey(driver.sharding_spec(), driver.key_prefix(),
                    std::get<ShardIndex>(self->batch_entry_key)),
        std::move(read_options));
    read_future.Force();
    std::move(read_future)
        .ExecuteWhenReady([self = std::move(self)](
                              ReadyFuture<kvstore::ReadResult> future) mutable {
          const auto& executor = self->driver().executor();
          executor([self = std::move(self), future = std::move(future)] {
            OnEntireShardReady(std::move(self), std::move(future.result()));
          });
        });
  }

  static void OnEntireShardReady(
      internal::IntrusivePtr<ReadOperationState> self,
      Result<kvstore::ReadResult>&& result) {
    if (!result.ok() || !result->has_value()) {
      internal_kvstore_batch::SetCommonResult(self->request_batch.requests,
                                              std::move(result));
      return;
    }
    auto& read_result = *result;
    const auto& sharding_spec = self->driver().sharding_spec();
    TENSORSTORE_ASSIGN_OR_RETURN(auto encoded_chunks,
                                 SplitShard(sharding_spec, read_result.value),
                                 internal_kvstore_batch::SetCommonResult(
                                     self->request_batch.requests, _));
    span<Request> requests = self->request_batch.requests;
    size_t request_i = 0;

    const auto complete_not_found = [&](Request& request) {
      std::get<internal_kvstore_batch::ByteRangeReadRequest>(request)
          .promise.SetResult(kvstore::ReadResult::Missing(read_result.stamp));
    };

    for (const auto& encoded_chunk : encoded_chunks) {
      auto decoded_data_result =
          DecodeData(encoded_chunk.encoded_data, sharding_spec.data_encoding);

      const auto complete_request =
          [&](Request& request) -> Result<kvstore::ReadResult> {
        TENSORSTORE_ASSIGN_OR_RETURN(
            const auto& decoded_data, decoded_data_result,
            internal::ConvertInvalidArgumentToFailedPrecondition(_));
        TENSORSTORE_ASSIGN_OR_RETURN(
            auto validated_byte_range,
            std::get<internal_kvstore_batch::ByteRangeReadRequest>(request)
                .byte_range.Validate(decoded_data.size()));
        kvstore::ReadResult request_read_result;
        request_read_result.stamp = read_result.stamp;
        request_read_result.state = kvstore::ReadResult::kValue;
        request_read_result.value =
            internal::GetSubCord(decoded_data, validated_byte_range);
        return request_read_result;
      };

      auto decoded_key = encoded_chunk.minishard_and_chunk_id;
      for (; request_i < requests.size(); ++request_i) {
        auto& request = requests[request_i];
        auto request_key = std::get<MinishardAndChunkId>(request);
        if (request_key < decoded_key) {
          complete_not_found(request);
        } else if (request_key == decoded_key) {
          std::get<internal_kvstore_batch::ByteRangeReadRequest>(request)
              .promise.SetResult(complete_request(request));
        } else {
          break;
        }
      }
    }

    for (; request_i < requests.size(); ++request_i) {
      complete_not_found(requests[request_i]);
    }
  }

  void ProcessMinishard(Batch::View batch, MinishardIndex minishard,
                        span<Request> requests, Batch& data_fetch_batch) {
    ChunkSplitShardInfo split_shard_info;
    split_shard_info.shard = std::get<ShardIndex>(batch_entry_key);
    split_shard_info.minishard = minishard;
    auto shard_info =
        GetCombinedShardInfo(driver().sharding_spec(), split_shard_info);
    auto minishard_index_cache_entry = GetCacheEntry(
        driver().minishard_index_cache(),
        std::string_view(reinterpret_cast<const char*>(&shard_info),
                         sizeof(shard_info)));
    auto minishard_index_read_future = minishard_index_cache_entry->Read(
        {request_batch.staleness_bound, batch});
    Batch successor_batch{no_batch};
    if (batch) {
      if (minishard_index_read_future.ready()) {
        successor_batch = batch;
      } else {
        if (!data_fetch_batch) {
          data_fetch_batch = Batch::New();
        }
        successor_batch = data_fetch_batch;
      }
    }
    const auto& executor = driver().executor();
    std::move(minishard_index_read_future)
        .ExecuteWhenReady(WithExecutor(
            executor,
            [self = internal::IntrusivePtr<ReadOperationState>(this), requests,
             minishard_index_cache_entry =
                 std::move(minishard_index_cache_entry),
             successor_batch = std::move(successor_batch)](
                ReadyFuture<const void> future) mutable {
              const auto& status = future.status();
              if (!status.ok()) {
                internal_kvstore_batch::SetCommonResult<Request>(requests,
                                                                 {status});
                return;
              }
              OnMinishardIndexReady(std::move(self), requests,
                                    std::move(successor_batch),
                                    std::move(minishard_index_cache_entry));
            }));
  }

  static void OnMinishardIndexReady(
      internal::IntrusivePtr<ReadOperationState> self, span<Request> requests,
      Batch successor_batch,
      internal::PinnedCacheEntry<MinishardIndexCache>
          minishard_index_cache_entry) {
    std::shared_ptr<const std::vector<MinishardIndexEntry>> minishard_index;
    TimestampedStorageGeneration stamp;
    {
      auto lock = internal::AsyncCache::ReadLock<MinishardIndexCache::ReadData>(
          *minishard_index_cache_entry);
      stamp = lock.stamp();
      minishard_index = lock.shared_data();
    }

    assert(!StorageGeneration::IsUnknown(stamp.generation));

    if (!minishard_index) {
      internal_kvstore_batch::SetCommonResult(
          requests, kvstore::ReadResult::Missing(std::move(stamp)));
      return;
    }

    const auto& sharding_spec = self->driver().sharding_spec();

    const auto process_chunk = [&](ChunkId chunk_id,
                                   span<Request> chunk_requests) {
      auto byte_range = FindChunkInMinishard(*minishard_index, chunk_id);
      if (!byte_range) {
        internal_kvstore_batch::SetCommonResult(
            chunk_requests, kvstore::ReadResult::Missing(std::move(stamp)));
        return;
      }

      int64_t size = byte_range->size();

      chunk_requests = chunk_requests.first(
          std::remove_if(
              chunk_requests.begin(), chunk_requests.end(),
              [&](Request& request) {
                return !internal_kvstore_batch::ValidateRequestGeneration(
                    request, stamp);
              }) -
          chunk_requests.begin());

      if (sharding_spec.data_encoding == ShardingSpec::DataEncoding::raw) {
        // Can apply requested byte range directly.
        const auto process_request = [&](Request& request) {
          auto& byte_range_request =
              std::get<internal_kvstore_batch::ByteRangeReadRequest>(request);
          // Note: We can't use
          // `internal_kvstore_batch::ValidateRequestGenerationAndByteRange`
          // because that mutates `request.byte_range` and the resolved byte
          // range should not be used in the event of a retry due to generation
          // mismatch.
          TENSORSTORE_ASSIGN_OR_RETURN(
              auto sub_byte_range, byte_range_request.byte_range.Validate(size),
              static_cast<void>(byte_range_request.promise.SetResult(_)));
          kvstore::ReadOptions kvstore_read_options;
          kvstore_read_options.generation_conditions.if_equal =
              stamp.generation;
          kvstore_read_options.staleness_bound =
              self->request_batch.staleness_bound;
          kvstore_read_options.byte_range = ByteRange{
              byte_range->inclusive_min + sub_byte_range.inclusive_min,
              byte_range->inclusive_min + sub_byte_range.exclusive_max};
          kvstore_read_options.batch = successor_batch;
          auto value_read_future = self->driver().base_kvstore_driver()->Read(
              self->ShardKey(), std::move(kvstore_read_options));
          value_read_future.Force();
          std::move(value_read_future)
              .ExecuteWhenReady([self,
                                 &request](ReadyFuture<kvstore::ReadResult>
                                               future) mutable {
                TENSORSTORE_ASSIGN_OR_RETURN(
                    auto&& read_result, std::move(future.result()),
                    static_cast<void>(
                        std::get<internal_kvstore_batch::ByteRangeReadRequest>(
                            request)
                            .promise.SetResult(_)));
                self->OnRawValueReady(request, std::move(read_result));
              });
        };
        for (auto& request : chunk_requests) {
          process_request(request);
        }
      } else {
        kvstore::ReadOptions kvstore_read_options;
        kvstore_read_options.generation_conditions.if_equal = stamp.generation;
        kvstore_read_options.staleness_bound =
            self->request_batch.staleness_bound;
        kvstore_read_options.byte_range = *byte_range;
        kvstore_read_options.batch = successor_batch;
        auto value_read_future = self->driver().base_kvstore_driver()->Read(
            self->ShardKey(), std::move(kvstore_read_options));
        value_read_future.Force();
        std::move(value_read_future)
            .ExecuteWhenReady(
                [self, chunk_requests](
                    ReadyFuture<kvstore::ReadResult> future) mutable {
                  const auto& executor = self->driver().executor();
                  executor([self = std::move(self), chunk_requests,
                            future = std::move(future)] {
                    TENSORSTORE_ASSIGN_OR_RETURN(
                        auto&& read_result, std::move(future.result()),
                        internal_kvstore_batch::SetCommonResult(chunk_requests,
                                                                _));
                    self->OnEncodedValueReady(chunk_requests,
                                              std::move(read_result));
                  });
                });
      }
    };

    for (size_t request_i = 0; request_i < requests.size();) {
      ChunkId chunk_id =
          std::get<MinishardAndChunkId>(requests[request_i]).chunk_id;
      size_t end_request_i;
      for (end_request_i = request_i + 1; end_request_i < requests.size();
           ++end_request_i) {
        if (std::get<MinishardAndChunkId>(requests[end_request_i]).chunk_id !=
            chunk_id)
          break;
      }
      process_chunk(chunk_id,
                    requests.subspan(request_i, end_request_i - request_i));
      request_i = end_request_i;
    }
  }

  void OnRawValueReady(Request& request, kvstore::ReadResult&& read_result) {
    if (read_result.aborted()) {
      // Concurrent modification.  Retry.
      MakeRequest<ReadOperationState>(
          driver(), std::get<ShardIndex>(batch_entry_key), retry_batch_,
          read_result.stamp.time, std::move(request));
      return;
    }
    std::get<internal_kvstore_batch::ByteRangeReadRequest>(request)
        .promise.SetResult(std::move(read_result));
  }

  void OnEncodedValueReady(span<Request> chunk_requests,
                           kvstore::ReadResult&& read_result) {
    if (read_result.aborted()) {
      // Concurrent modification.  Retry.
      for (auto& request : chunk_requests) {
        MakeRequest<ReadOperationState>(
            driver(), std::get<ShardIndex>(batch_entry_key), retry_batch_,
            read_result.stamp.time, std::move(request));
      }
      return;
    }

    if (!read_result.has_value()) {
      internal_kvstore_batch::SetCommonResult(chunk_requests,
                                              std::move(read_result));
      return;
    }

    TENSORSTORE_ASSIGN_OR_RETURN(
        auto decoded_value,
        DecodeData(read_result.value, driver().sharding_spec().data_encoding),
        internal_kvstore_batch::SetCommonResult(
            chunk_requests,
            internal::ConvertInvalidArgumentToFailedPrecondition(_)));

    const auto process_request =
        [&](Request& request) -> Result<kvstore::ReadResult> {
      auto& byte_range_request =
          std::get<internal_kvstore_batch::ByteRangeReadRequest>(request);
      TENSORSTORE_ASSIGN_OR_RETURN(
          auto byte_range,
          byte_range_request.byte_range.Validate(decoded_value.size()));
      return kvstore::ReadResult::Value(
          internal::GetSubCord(decoded_value, byte_range), read_result.stamp);
    };

    for (auto& request : chunk_requests) {
      std::get<internal_kvstore_batch::ByteRangeReadRequest>(request)
          .promise.SetResult(process_request(request));
    }
  }
};
}  // namespace

Future<kvstore::ReadResult> ShardedKeyValueStore::Read(Key key,
                                                       ReadOptions options) {
  TENSORSTORE_ASSIGN_OR_RETURN(ChunkId chunk_id, KeyToChunkIdOrError(key));
  const auto& sharding_spec = this->sharding_spec();
  auto shard_info = GetChunkShardInfo(sharding_spec, chunk_id);
  auto split_shard_info = GetSplitShardInfo(sharding_spec, shard_info);
  auto [promise, future] = PromiseFuturePair<kvstore::ReadResult>::Make();
  ReadOperationState::MakeRequest<ReadOperationState>(
      *this, split_shard_info.shard, options.batch, options.staleness_bound,
      ReadOperationState::Request{
          {std::move(promise), options.byte_range},
          MinishardAndChunkId{split_shard_info.minishard, chunk_id},
          std::move(options.generation_conditions)});
  return std::move(future);
}

}  // namespace neuroglancer_uint64_sharded

namespace garbage_collection {
template <>
struct GarbageCollection<neuroglancer_uint64_sharded::ShardedKeyValueStore> {
  static void Visit(
      GarbageCollectionVisitor& visitor,
      const neuroglancer_uint64_sharded::ShardedKeyValueStore& value) {
    garbage_collection::GarbageCollectionVisit(visitor,
                                               *value.base_kvstore_driver());
  }
};
}  // namespace garbage_collection

namespace neuroglancer_uint64_sharded {

absl::Status ShardedKeyValueStore::GetBoundSpecData(
    ShardedKeyValueStoreSpecData& spec) const {
  TENSORSTORE_ASSIGN_OR_RETURN(spec.base.driver,
                               base_kvstore_driver()->GetBoundSpec());
  spec.base.path = key_prefix();
  if (!data_copy_concurrency_resource_.has_resource() ||
      !cache_pool_resource_.has_resource()) {
    return absl::InternalError("JSON representation not supported");
  }
  spec.data_copy_concurrency = data_copy_concurrency_resource_;
  spec.cache_pool = cache_pool_resource_;
  spec.metadata = sharding_spec();
  return absl::Status();
}

Future<kvstore::DriverPtr> ShardedKeyValueStoreSpec::DoOpen() const {
  return MapFutureValue(
      InlineExecutor{},
      [spec = internal::IntrusivePtr<const ShardedKeyValueStoreSpec>(this)](
          kvstore::KvStore& base_kvstore) -> Result<kvstore::DriverPtr> {
        auto driver = internal::MakeIntrusivePtr<ShardedKeyValueStore>(
            std::move(base_kvstore.driver),
            spec->data_.data_copy_concurrency->executor,
            std::move(base_kvstore.path), spec->data_.metadata,
            *spec->data_.cache_pool);
        driver->data_copy_concurrency_resource_ =
            spec->data_.data_copy_concurrency;
        driver->cache_pool_resource_ = spec->data_.cache_pool;
        return driver;
      },
      kvstore::Open(data_.base));
}

kvstore::DriverPtr GetShardedKeyValueStore(
    kvstore::DriverPtr base_kvstore, Executor executor, std::string key_prefix,
    const ShardingSpec& sharding_spec, internal::CachePool::WeakPtr cache_pool,
    GetMaxChunksPerShardFunction get_max_chunks_per_shard) {
  return kvstore::DriverPtr(new ShardedKeyValueStore(
      std::move(base_kvstore), std::move(executor), std::move(key_prefix),
      sharding_spec, std::move(cache_pool),
      std::move(get_max_chunks_per_shard)));
}

std::string ChunkIdToKey(ChunkId chunk_id) {
  std::string key;
  key.resize(sizeof(uint64_t));
  absl::big_endian::Store64(key.data(), chunk_id.value);
  return key;
}

std::optional<ChunkId> KeyToChunkId(std::string_view key) {
  if (key.size() != 8) return std::nullopt;
  return ChunkId{absl::big_endian::Load64(key.data())};
}

}  // namespace neuroglancer_uint64_sharded
}  // namespace tensorstore

namespace {
const tensorstore::internal_kvstore::DriverRegistration<
    tensorstore::neuroglancer_uint64_sharded::ShardedKeyValueStoreSpec>
    registration;
}  // namespace
