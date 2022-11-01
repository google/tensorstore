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

#include <cstdint>
#include <optional>
#include <string>

#include "absl/algorithm/container.h"
#include "absl/base/internal/endian.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "tensorstore/internal/cache/async_cache.h"
#include "tensorstore/internal/cache/cache_pool_resource.h"
#include "tensorstore/internal/cache/kvs_backed_cache.h"
#include "tensorstore/internal/data_copy_concurrency_resource.h"
#include "tensorstore/internal/estimate_heap_usage/estimate_heap_usage.h"
#include "tensorstore/internal/estimate_heap_usage/std_vector.h"  // iwyu: keep
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/mutex.h"
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/neuroglancer_uint64_sharded/uint64_sharded.h"
#include "tensorstore/kvstore/neuroglancer_uint64_sharded/uint64_sharded_decoder.h"
#include "tensorstore/kvstore/neuroglancer_uint64_sharded/uint64_sharded_encoder.h"
#include "tensorstore/kvstore/registry.h"
#include "tensorstore/kvstore/transaction.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/execution/result_sender.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace neuroglancer_uint64_sharded {
namespace {

using ::tensorstore::internal::ConvertInvalidArgumentToFailedPrecondition;
using ::tensorstore::internal::IntrusivePtr;

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

  Future<ReadResult> Read(Key key, ReadOptions options) override {
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
    DoRead(std::move(promise), split_info, std::move(options));
    return std::move(future);
  }

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

 private:
  /// Asynchronously recursive implementation of `Read`, to handle retrying as
  /// may be required in the case of concurrent modifications, as described
  /// below.
  void DoRead(Promise<ReadResult> promise, ChunkSplitShardInfo split_info,
              ReadOptions options) {
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
    struct MinishardIndexReadyCallback {
      internal::IntrusivePtr<MinishardIndexKeyValueStore> self;
      ChunkSplitShardInfo split_info;

      void operator()(Promise<kvstore::ReadResult> promise,
                      ReadyFuture<kvstore::ReadResult> future) {
        auto& r = future.result();
        if (!r) {
          promise.SetResult(
              ConvertInvalidArgumentToFailedPrecondition(r.status()));
          return;
        }
        if (r->aborted()) {
          // Shard was modified since the index was read (case 2a above).
          // Retry.
          ReadOptions options;
          options.staleness_bound = r->stamp.time;
          self->DoRead(std::move(promise), split_info, std::move(options));
          return;
        }

        // Shard was modified since the index was read, but minishard
        // nonetheless does not exist (case 2b above).
        //
        // or
        //
        // read was successful (case 2c above).
        promise.SetResult(std::move(r));
      }
    };

    struct ShardIndexReadyCallback {
      IntrusivePtr<MinishardIndexKeyValueStore> self;
      ChunkSplitShardInfo split_info;
      absl::Time staleness_bound;
      static void SetError(const Promise<kvstore::ReadResult>& promise,
                           absl::Status error) {
        promise.SetResult(MaybeAnnotateStatus(
            ConvertInvalidArgumentToFailedPrecondition(std::move(error)),
            "Error retrieving shard index entry"));
      }

      void operator()(Promise<kvstore::ReadResult> promise,
                      ReadyFuture<kvstore::ReadResult> future) {
        auto& r = future.result();
        if (!r) {
          return SetError(promise, r.status());
        }
        if (  // Shard is empty (case 1a above)
            r->aborted() ||
            // Existing data is up to date (case 1b above).
            r->not_found()) {
          promise.SetResult(std::move(r));
          return;
        }
        // Read was successful (case 1c above).
        ByteRange byte_range;
        if (auto byte_range_result = DecodeShardIndexEntry(r->value.Flatten());
            byte_range_result.ok()) {
          byte_range = *byte_range_result;
        } else {
          SetError(promise, std::move(byte_range_result).status());
          return;
        }
        if (auto byte_range_result =
                GetAbsoluteShardByteRange(byte_range, self->sharding_spec_);
            byte_range_result.ok()) {
          byte_range = *byte_range_result;
        } else {
          SetError(promise, std::move(byte_range_result).status());
          return;
        }
        if (byte_range.size() == 0) {
          // Minishard index is 0 bytes, which means the minishard is empty.
          r->value.Clear();
          r->state = kvstore::ReadResult::kMissing;
          promise.SetResult(std::move(r));
          return;
        }
        kvstore::ReadOptions kvs_read_options;
        // The `if_equal` condition ensure that an "aborted" `ReadResult` is
        // returned in the case of a concurrent modification (case 2a above).
        kvs_read_options.if_equal = std::move(r->stamp.generation);
        kvs_read_options.staleness_bound = staleness_bound;
        kvs_read_options.byte_range = byte_range;
        auto read_future =
            self->base_->Read(GetShardKey(self->sharding_spec_,
                                          self->key_prefix_, split_info.shard),
                              std::move(kvs_read_options));
        Link(WithExecutor(self->executor_,
                          MinishardIndexReadyCallback{self, split_info}),
             std::move(promise), std::move(read_future));
      }
    };
    options.byte_range = {split_info.minishard * 16,
                          (split_info.minishard + 1) * 16};
    const auto staleness_bound = options.staleness_bound;
    Link(WithExecutor(executor_,
                      ShardIndexReadyCallback{
                          IntrusivePtr<MinishardIndexKeyValueStore>(this),
                          split_info, staleness_bound}),
         std::move(promise),
         base_->Read(GetShardKey(sharding_spec_, key_prefix_, split_info.shard),
                     std::move(options)));
  }

  kvstore::DriverPtr base_;
  Executor executor_;
  std::string key_prefix_;
  ShardingSpec sharding_spec_;
};

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

    std::size_t ComputeReadDataSizeInBytes(const void* read_data) override {
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
  std::size_t DoGetSizeofEntry() final { return sizeof(Entry); }
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

  static std::string ShardToKey(uint64_t shard) {
    std::string key;
    key.resize(sizeof(uint64_t));
    absl::big_endian::Store64(key.data(), shard);
    return key;
  }

  static uint64_t KeyToShard(std::string_view key) {
    assert(key.size() == sizeof(uint64_t));
    return absl::big_endian::Load64(key.data());
  }

  class Entry : public Base::Entry {
   public:
    using OwningCache = ShardedKeyValueStoreWriteCache;

    std::uint64_t shard() { return KeyToShard(key()); }

    size_t ComputeReadDataSizeInBytes(const void* data) override {
      return internal::EstimateHeapUsage(*static_cast<const absl::Cord*>(data));
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
      this->AsyncCache::TransactionNode::Read(options.staleness_bound)
          .ExecuteWhenReady(WithExecutor(
              GetOwningCache(*this).executor(),
              [&entry, if_not_equal = std::move(options.if_not_equal),
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
      kvstore::ReadResult read_result;
      std::shared_ptr<const EncodedChunks> encoded_chunks;
      {
        AsyncCache::ReadLock<EncodedChunks> lock{self};
        read_result.stamp = lock.stamp();
        encoded_chunks = lock.shared_data();
      }
      if (!StorageGeneration::IsUnknown(read_result.stamp.generation) &&
          read_result.stamp.generation == if_not_equal) {
        read_result.state = kvstore::ReadResult::kUnspecified;
      } else {
        auto* chunk =
            FindChunk(*encoded_chunks, GetMinishardAndChunkId(entry.key_));
        if (!chunk) {
          read_result.state = kvstore::ReadResult::kMissing;
        } else {
          read_result.state = kvstore::ReadResult::kValue;
          TENSORSTORE_ASSIGN_OR_RETURN(
              read_result.value,
              DecodeData(chunk->encoded_data,
                         GetOwningCache(self).sharding_spec().data_encoding));
        }
        if (StorageGeneration::IsDirty(read_result.stamp.generation)) {
          // Add layer to generation in order to make it possible to
          // distinguish:
          //
          // 1. the shard being modified by a predecessor
          //    `ReadModifyWrite` operation on the underlying
          //    KeyValueStore.
          //
          // 2. the chunk being modified by a `ReadModifyWrite`
          //    operation attached to this transaction node.
          read_result.stamp.generation = StorageGeneration::AddLayer(
              std::move(read_result.stamp.generation));
        }
      }
      return read_result;
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

    absl::Time apply_staleness_bound_;
    ApplyReceiver apply_receiver_;
    absl::Status apply_status_;
  };

  Entry* DoAllocateEntry() final { return new Entry; }
  std::size_t DoGetSizeofEntry() final { return sizeof(Entry); }
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
  RetryAtomicWriteback(node.phases_, node.apply_staleness_bound_);
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
    node.apply_staleness_bound_ = absl::Now();
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
  apply_staleness_bound_ = options.staleness_bound;
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

    size_t num_chunks = 0;

    // Determine if the writeback result depends on the existing shard, and if
    // all chunks are conditioned on the same generation.
    for (auto& entry : self.phases_.entries_) {
      auto& buffered_entry =
          static_cast<AtomicMultiPhaseMutation::BufferedReadModifyWriteEntry&>(
              entry);
      auto& entry_stamp = buffered_entry.read_result_.stamp;
      ++num_chunks;
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
      self.apply_staleness_bound_ = absl::Now();
      StartApply(self);
      return;
    }
    auto& cache = GetOwningCache(self);
    if (!StorageGeneration::IsUnknown(stamp.generation) ||
        !cache.get_max_chunks_per_shard_ ||
        cache.get_max_chunks_per_shard_(GetOwningEntry(self).shard()) !=
            num_chunks) {
      self.internal::AsyncCache::TransactionNode::Read(
              self.apply_staleness_bound_)
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

/// Asynchronous callback invoked (indirectly) from `ShardedKeyValueStore::Read`
/// when the cache entry for a given minishard index is ready.
struct MinishardIndexCacheEntryReadyCallback {
  using ReadResult = kvstore::ReadResult;
  using ReadOptions = kvstore::ReadOptions;

  internal::PinnedCacheEntry<MinishardIndexCache> entry_;
  ChunkId chunk_id_;
  ReadOptions options_;
  void operator()(Promise<ReadResult> promise, ReadyFuture<const void>) {
    std::optional<ByteRange> byte_range;
    TimestampedStorageGeneration stamp;
    kvstore::ReadResult::State state;
    {
      auto lock = internal::AsyncCache::ReadLock<MinishardIndexCache::ReadData>(
          *entry_);
      stamp = lock.stamp();
      if (!StorageGeneration::IsNoValue(stamp.generation) &&
          (options_.if_not_equal == stamp.generation ||
           (!StorageGeneration::IsUnknown(options_.if_equal) &&
            options_.if_equal != stamp.generation))) {
        state = kvstore::ReadResult::kUnspecified;
      } else {
        span<const MinishardIndexEntry> minishard_index;
        if (lock.data()) minishard_index = *lock.data();
        byte_range = FindChunkInMinishard(minishard_index, chunk_id_);
        state = kvstore::ReadResult::kMissing;
      }
    }
    if (!byte_range) {
      promise.SetResult(ReadResult{state, {}, std::move(stamp)});
      return;
    }
    assert(!StorageGeneration::IsUnknown(stamp.generation));
    auto& cache = GetOwningCache(*entry_);
    ReadOptions kvs_read_options;
    kvs_read_options.if_equal = stamp.generation;
    kvs_read_options.staleness_bound = options_.staleness_bound;
    assert(options_.byte_range.SatisfiesInvariants());
    OptionalByteRangeRequest post_decode_byte_range;
    const auto data_encoding = cache.sharding_spec().data_encoding;
    if (data_encoding == ShardingSpec::DataEncoding::raw) {
      // Can apply requested byte range request directly.
      if (auto result = options_.byte_range.Validate(byte_range->size())) {
        kvs_read_options.byte_range =
            ByteRange{byte_range->inclusive_min + result->inclusive_min,
                      byte_range->inclusive_min + result->exclusive_max};
      } else {
        promise.SetResult(std::move(result).status());
        return;
      }
    } else {
      post_decode_byte_range = options_.byte_range;
      kvs_read_options.byte_range = *byte_range;
    }
    const auto shard = entry_->shard_info().shard;
    LinkValue(
        [entry = std::move(entry_), chunk_id = chunk_id_,
         options = std::move(options_), post_decode_byte_range,
         data_encoding](Promise<ReadResult> promise,
                        ReadyFuture<ReadResult> future) mutable {
          auto& r = future.result();
          if (r->aborted()) {
            // Concurrent modification.  Retry.
            auto& cache = GetOwningCache(*entry);
            auto minishard_index_read_future =
                entry->Read(/*staleness_bound=*/absl::InfiniteFuture());
            LinkValue(WithExecutor(
                          cache.executor(),
                          MinishardIndexCacheEntryReadyCallback{
                              std::move(entry), chunk_id, std::move(options)}),
                      std::move(promise),
                      std::move(minishard_index_read_future));
            return;
          }
          if (r->not_found()) {
            // Shard was concurrently modified, but a value is not present
            // in any case.
            promise.SetResult(std::move(r));
            return;
          }
          if (data_encoding != ShardingSpec::DataEncoding::raw) {
            auto result = DecodeData(r->value, data_encoding);
            if (!result) {
              promise.SetResult(
                  absl::FailedPreconditionError(result.status().message()));
              return;
            }
            r->value = *result;
          }
          auto byte_range_result =
              post_decode_byte_range.Validate(r->value.size());
          if (!byte_range_result.ok()) {
            promise.SetResult(std::move(byte_range_result).status());
            return;
          }
          r->value =
              internal::GetSubCord(std::move(r->value), *byte_range_result);
          promise.SetResult(std::move(r));
        },
        std::move(promise),
        cache.base_kvstore_driver()->Read(
            GetShardKey(cache.sharding_spec(), cache.key_prefix(), shard),
            std::move(kvs_read_options)));
  }
};

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
      : write_cache_(
            cache_pool->GetCache<ShardedKeyValueStoreWriteCache>("", [&] {
              return std::make_unique<ShardedKeyValueStoreWriteCache>(
                  cache_pool->GetCache<MinishardIndexCache>(
                      "",
                      [&] {
                        return std::make_unique<MinishardIndexCache>(
                            std::move(base_kvstore), std::move(executor),
                            std::move(key_prefix), sharding_spec);
                      }),
                  std::move(get_max_chunks_per_shard));
            })) {}

  Future<ReadResult> Read(Key key, ReadOptions options) override {
    TENSORSTORE_ASSIGN_OR_RETURN(ChunkId chunk_id, KeyToChunkIdOrError(key));
    auto shard_info = GetChunkShardInfo(sharding_spec(), chunk_id);

    auto minishard_index_cache_entry = GetCacheEntry(
        minishard_index_cache(),
        std::string_view(reinterpret_cast<const char*>(&shard_info),
                         sizeof(shard_info)));

    auto minishard_index_read_future =
        minishard_index_cache_entry->Read(options.staleness_bound);
    return PromiseFuturePair<ReadResult>::LinkValue(
               WithExecutor(executor(),
                            MinishardIndexCacheEntryReadyCallback{
                                std::move(minishard_index_cache_entry),
                                chunk_id, std::move(options)}),
               std::move(minishard_index_read_future))
        .future;
  }

  void ListImpl(ListOptions options,
                AnyFlowReceiver<absl::Status, Key> receiver) override {
    struct State {
      explicit State(AnyFlowReceiver<absl::Status, Key>&& receiver,
                     ListOptions options)
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
      AnyFlowReceiver<absl::Status, Key> receiver_;
      Promise<void> promise_;
      Future<void> future_;
      ListOptions options_;
    };
    auto state =
        std::make_shared<State>(std::move(receiver), std::move(options));
    // Inefficient, but only used for testing.
    uint64_t num_shards = uint64_t(1) << sharding_spec().shard_bits;
    for (uint64_t shard = 0; shard < num_shards; ++shard) {
      auto entry = GetCacheEntry(
          write_cache_, ShardedKeyValueStoreWriteCache::ShardToKey(shard));
      LinkValue(
          [state, entry](Promise<void> promise,
                         ReadyFuture<const void> future) {
            auto chunks = internal::AsyncCache::ReadLock<EncodedChunks>(*entry)
                              .shared_data();
            if (!chunks) return;
            for (auto& chunk : *chunks) {
              auto key = ChunkIdToKey(chunk.minishard_and_chunk_id.chunk_id);
              if (!Contains(state->options_.range, key)) continue;
              key.erase(0, state->options_.strip_prefix_length);
              execution::set_value(state->receiver_, std::move(key));
            }
          },
          state->promise_, entry->Read(absl::InfiniteFuture()));
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
    const std::uint64_t shard = shard_info.shard;
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
};
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
