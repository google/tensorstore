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

#include "tensorstore/driver/neuroglancer_precomputed/uint64_sharded_key_value_store.h"

#include <cstdint>
#include <optional>
#include <string>

#include "absl/algorithm/container.h"
#include "tensorstore/driver/neuroglancer_precomputed/uint64_sharded.h"
#include "tensorstore/driver/neuroglancer_precomputed/uint64_sharded_decoder.h"
#include "tensorstore/driver/neuroglancer_precomputed/uint64_sharded_encoder.h"
#include "tensorstore/internal/async_storage_backed_cache.h"
#include "tensorstore/internal/compression/zlib.h"
#include "tensorstore/internal/mutex.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace neuroglancer_uint64_sharded {

namespace {

/// Decodes a minishard and adjusts the byte ranges to account for the implicit
/// offset of the end of the shard index.
///
/// \returns The decoded minishard index on success.
/// \error `absl::StatusCode::kInvalidArgument` if the encoded minishard is
///   invalid.
Result<std::vector<MinishardIndexEntry>>
DecodeMinishardIndexAndAdjustByteRanges(absl::string_view encoded,
                                        const ShardingSpec& sharding_spec) {
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto minishard_index,
      DecodeMinishardIndex(encoded, sharding_spec.minishard_index_encoding));
  for (auto& entry : minishard_index) {
    auto result = GetAbsoluteShardByteRange(entry.byte_range, sharding_spec);
    if (!result.ok()) {
      return MaybeAnnotateStatus(
          result.status(),
          StrCat("Error decoding minishard index entry for chunk ",
                 entry.chunk_id.value));
    }
    entry.byte_range = std::move(result).value();
  }
  return minishard_index;
}

/// Caches minishard indexes.
///
/// Each cache entry corresponds to a particular minishard within a particular
/// shard.  The entry keys directly encode `ChunkCombinedShardInfo` values (via
/// `memcpy`), specifying a shard and minishard number.
///
/// This cache is only used for reading.  `FinishWrite` must never be called.
class MinishardIndexCache : public internal::AsyncStorageBackedCache {
  using Base = internal::AsyncStorageBackedCache;

 public:
  class Entry : public Base::Entry {
   public:
    using Cache = MinishardIndexCache;
    Mutex mutex_;
    TimestampedStorageGeneration generation_;           // Guarded by `mutex_`
    std::vector<MinishardIndexEntry> minishard_index_;  // Guarded by `mutex_`

    ChunkSplitShardInfo shard_info() {
      ChunkCombinedShardInfo combined_info;
      assert(this->key().size() == sizeof(combined_info));
      std::memcpy(&combined_info, this->key().data(), sizeof(combined_info));
      return GetSplitShardInfo(GetOwningCache(this)->sharding_spec_,
                               combined_info);
    }
  };

  explicit MinishardIndexCache(KeyValueStore::Ptr base_kv_store,
                               Executor executor, std::string key_prefix,
                               ShardingSpec sharding_spec)
      : base_kv_store_(std::move(base_kv_store)),
        executor_(std::move(executor)),
        key_prefix_(std::move(key_prefix)),
        sharding_spec_(sharding_spec) {}

  void DoDeleteEntry(internal::Cache::Entry* base_entry) override {
    Entry* entry = static_cast<Entry*>(base_entry);
    delete entry;
  }

  internal::Cache::Entry* DoAllocateEntry() override { return new Entry; }

  std::size_t DoGetSizeInBytes(Cache::Entry* base_entry) override {
    Entry* entry = static_cast<Entry*>(base_entry);
    return sizeof(Entry) + Base::DoGetSizeInBytes(base_entry) +
           entry->minishard_index_.capacity() * sizeof(MinishardIndexEntry);
  }

  void DoRead(ReadOptions options, ReadReceiver receiver) override {
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
    // 2. Request the minishard index from the byte range specified in the shard
    //    index.
    //
    //    a. If the generation has changed (concurrent modification of the
    //       shard), retry starting at step 1.
    //
    //    b. If not found, the minishard is empty.  (This can only happens in
    //       the case of concurrent modification of the shard).  Done.
    //
    //    c.  Otherwise, decode the minishard index.
    struct MinishardIndexReadyCallback {
      ReadReceiver receiver;

      void SetError(Status error) {
        if (error.code() == absl::StatusCode::kOutOfRange ||
            error.code() == absl::StatusCode::kInvalidArgument) {
          error = absl::FailedPreconditionError(error.message());
        }
        auto& entry = static_cast<Entry&>(*receiver.entry());
        auto split_info = entry.shard_info();
        receiver.NotifyDone(
            /*size_update=*/{},
            MaybeAnnotateStatus(
                error,
                StrCat("Error retrieving minishard index for shard ",
                       split_info.shard, " minishard ", split_info.minishard)));
      }

      void operator()(ReadyFuture<KeyValueStore::ReadResult> future) {
        auto& r = future.result();
        if (!r) {
          return SetError(r.status());
        }
        auto& entry = static_cast<Entry&>(*receiver.entry());
        auto* cache = GetOwningCache(&entry);
        if (r->aborted()) {
          // Shard was modified since the index was read (case 2a above).
          // Retry.
          ReadOptions options;
          options.staleness_bound = r->generation.time;
          return cache->DoRead(std::move(options), std::move(receiver));
        }
        if (r->not_found()) {
          // Shard was modified since the index was read, but minishard
          // nonetheless does not exist (case 2b above).
          std::unique_lock<Mutex> lock(entry.mutex_);
          entry.minishard_index_.clear();
          entry.generation_ = r->generation;
          receiver.NotifyDone({std::move(lock),
                               /*new_size=*/cache->DoGetSizeInBytes(&entry)},
                              std::move(r->generation));
          return;
        }
        // Read was successful (case 2c above).
        TENSORSTORE_ASSIGN_OR_RETURN(auto minishard_index,
                                     DecodeMinishardIndexAndAdjustByteRanges(
                                         *r->value, cache->sharding_spec()),
                                     SetError(_));
        std::unique_lock<Mutex> lock(entry.mutex_);
        entry.minishard_index_ = std::move(minishard_index);
        entry.generation_ = r->generation;
        receiver.NotifyDone({std::move(lock), cache->DoGetSizeInBytes(&entry)},
                            std::move(r->generation));
      }
    };
    struct ShardIndexReadyCallback {
      ReadReceiver receiver;
      absl::Time staleness_bound;

      void SetError(Status error) {
        if (error.code() == absl::StatusCode::kOutOfRange) {
          error = absl::FailedPreconditionError(error.message());
        }
        auto& entry = static_cast<Entry&>(*receiver.entry());
        auto split_info = entry.shard_info();
        receiver.NotifyDone(
            /*size_update=*/{},
            MaybeAnnotateStatus(
                error,
                StrCat("Error retrieving shard index entry for shard ",
                       split_info.shard, " minishard ", split_info.minishard)));
      }

      void operator()(ReadyFuture<KeyValueStore::ReadResult> future) {
        auto& r = future.result();
        if (!r) {
          return SetError(r.status());
        }
        auto& entry = static_cast<Entry&>(*receiver.entry());
        auto* cache = GetOwningCache(&entry);
        if (r->aborted()) {
          // Existing data is up to date (case 1b above).
          std::unique_lock<Mutex> lock(entry.mutex_);
          entry.generation_.time = r->generation.time;
          receiver.NotifyDone({std::move(lock)}, std::move(r->generation));
          return;
        }
        if (r->not_found()) {
          // Shard is empty (case 1a above).
          std::unique_lock<Mutex> lock(entry.mutex_);
          entry.minishard_index_.clear();
          entry.generation_ = r->generation;
          receiver.NotifyDone({std::move(lock),
                               /*new_size=*/cache->DoGetSizeInBytes(&entry)},
                              std::move(r->generation));
          return;
        }
        // Read was successful (case 1c above).
        TENSORSTORE_ASSIGN_OR_RETURN(
            auto byte_range, DecodeShardIndexEntry(*r->value), SetError(_));
        TENSORSTORE_ASSIGN_OR_RETURN(
            byte_range,
            GetAbsoluteShardByteRange(byte_range, cache->sharding_spec_),
            SetError(_));
        if (byte_range.size() == 0) {
          // Minishard index is 0 bytes, which means the minishard is empty.
          std::unique_lock<Mutex> lock(entry.mutex_);
          entry.minishard_index_.clear();
          entry.generation_ = r->generation;
          receiver.NotifyDone({std::move(lock),
                               /*new_size=*/cache->DoGetSizeInBytes(&entry)},
                              std::move(r->generation));
          return;
        }
        auto split_info = entry.shard_info();
        KeyValueStore::ReadOptions kvs_read_options;
        // The `if_equal` condition ensure that an "aborted" `ReadResult` is
        // returned in the case of a concurrent modification (case 2a above).
        kvs_read_options.if_equal = std::move(r->generation.generation);
        kvs_read_options.staleness_bound = staleness_bound;
        kvs_read_options.byte_range = byte_range;

        cache->base_kv_store_
            ->Read(GetShardKey(cache->sharding_spec_, cache->key_prefix_,
                               split_info.shard),
                   std::move(kvs_read_options))
            .ExecuteWhenReady(
                WithExecutor(cache->executor_,
                             MinishardIndexReadyCallback{std::move(receiver)}));
      }
    };
    auto& entry = static_cast<Entry&>(*receiver.entry());
    auto split_info = entry.shard_info();
    KeyValueStore::ReadOptions kvs_read_options;
    kvs_read_options.if_not_equal = options.existing_generation;
    kvs_read_options.staleness_bound = options.staleness_bound;
    kvs_read_options.byte_range = {split_info.minishard * 16,
                                   (split_info.minishard + 1) * 16};
    base_kv_store_
        ->Read(GetShardKey(sharding_spec_, key_prefix_, split_info.shard),
               std::move(kvs_read_options))
        .ExecuteWhenReady(WithExecutor(
            executor_, ShardIndexReadyCallback{std::move(receiver),
                                               options.staleness_bound}));
  }
  // COV_NF_START
  void DoWriteback(TimestampedStorageGeneration existing_generation,
                   WritebackReceiver receiver) override {
    // Writes are handled by `ShardedKeyValueStoreWriteCache`.
    TENSORSTORE_UNREACHABLE;
  }
  // COV_NF_END

  KeyValueStore* base_kv_store() const { return base_kv_store_.get(); }
  const ShardingSpec& sharding_spec() const { return sharding_spec_; }
  const Executor& executor() const { return executor_; }
  const std::string& key_prefix() const { return key_prefix_; }

  KeyValueStore::Ptr base_kv_store_;
  Executor executor_;
  std::string key_prefix_;
  ShardingSpec sharding_spec_;
};

/// Specifies a pending write/delete operation for a chunk.
struct PendingChunkWrite {
  std::uint64_t minishard;
  ChunkId chunk_id;
  /// Specifies the new value for the chunk, or `std::nullopt` to request that
  /// the chunk be deleted.
  std::optional<std::string> data;

  /// If not equal to `StorageGeneration::Unknown()`, apply the write/delete
  /// only if the existing generation matches.
  StorageGeneration if_equal;

  enum class WriteStatus {
    /// The write or delete was successfully applied.
    kSuccess,

    /// The write or delete was aborted due to a failed `if_equal` condition.
    kAborted,

    /// The `if_equal` condition was satisfied, but the chunk was overwritten by
    /// a subsequent write.
    kOverwritten,
  };

  /// Set to indicate the status of the operation.
  WriteStatus write_status;

  /// Specifies the promise to be marked ready when the write/delete operation
  /// completes successfully or with an error.
  Promise<TimestampedStorageGeneration> promise;
};

/// Class used to implement `MergeShard`.
class MergeShardImpl {
 public:
  using WriteStatus = PendingChunkWrite::WriteStatus;

  explicit MergeShardImpl(const ShardingSpec& sharding_spec,
                          const StorageGeneration& existing_generation,
                          span<PendingChunkWrite> new_chunks,
                          std::string* new_shard)
      : new_shard_(new_shard),
        encoder_(sharding_spec, new_shard),
        existing_generation_(existing_generation) {
    shard_index_offset_ = new_shard->size();
    new_shard->resize(shard_index_offset_ + ShardIndexSize(sharding_spec));
    shard_data_offset_ = new_shard->size();
    chunk_it_ = new_chunks.data();
    chunk_end_ = new_chunks.data() + new_chunks.size();
  }

  /// Merges the existing shard data with the new chunks.
  Status ProcessExistingShard(absl::string_view existing_shard) {
    const std::uint64_t num_minishards =
        encoder_.sharding_spec().num_minishards();
    if (!existing_shard.empty() &&
        existing_shard.size() < num_minishards * 16) {
      return absl::FailedPreconditionError(
          StrCat("Existing shard index has size ", existing_shard.size(),
                 ", but expected at least: ", num_minishards * 16));
    }
    for (std::uint64_t minishard = 0; minishard < num_minishards; ++minishard) {
      TENSORSTORE_RETURN_IF_ERROR(WriteNewChunksWhile(
          [&](std::uint64_t new_minishard, ChunkId new_chunk_id) {
            return new_minishard < minishard;
          }));
      const auto GetMinishardIndexByteRange = [&]() -> Result<ByteRange> {
        TENSORSTORE_ASSIGN_OR_RETURN(
            auto minishard_index_byte_range,
            DecodeShardIndexEntry(existing_shard.substr(16 * minishard, 16)));
        TENSORSTORE_ASSIGN_OR_RETURN(
            minishard_index_byte_range,
            GetAbsoluteShardByteRange(minishard_index_byte_range,
                                      encoder_.sharding_spec()));
        TENSORSTORE_RETURN_IF_ERROR(
            OptionalByteRangeRequest(minishard_index_byte_range)
                .Validate(existing_shard.size()));
        return minishard_index_byte_range;
      };
      auto minishard_ibr_result = GetMinishardIndexByteRange();
      if (!minishard_ibr_result.ok()) {
        return MaybeAnnotateStatus(
            minishard_ibr_result.status(),
            StrCat("Error decoding existing shard index entry for minishard ",
                   minishard));
      }
      if (minishard_ibr_result.value().size() == 0) {
        continue;
      }
      auto minishard_index_result = DecodeMinishardIndexAndAdjustByteRanges(
          internal::GetSubStringView(existing_shard,
                                     minishard_ibr_result.value()),
          encoder_.sharding_spec());
      if (!minishard_index_result.ok()) {
        return MaybeAnnotateStatus(
            minishard_index_result.status(),
            StrCat("Error decoding existing minishard index for minishard ",
                   minishard));
      }
      TENSORSTORE_RETURN_IF_ERROR(ProcessExistingMinishard(
          existing_shard, minishard, minishard_index_result.value()));
    }
    return absl::OkStatus();
  }

  /// Writes any remaining new chunks that follow any existing minishards.
  Status Finalize() {
    TENSORSTORE_RETURN_IF_ERROR(
        WriteNewChunksWhile([](std::uint64_t new_minishard,
                               ChunkId new_chunk_id) { return true; }));
    if (!modified_) {
      return absl::AbortedError("");
    }
    TENSORSTORE_ASSIGN_OR_RETURN(auto shard_index, encoder_.Finalize());
    if (new_shard_->size() == shard_data_offset_) {
      // Empty shard.
      new_shard_->resize(shard_index_offset_);
    } else {
      assert(shard_index.size() == shard_data_offset_ - shard_index_offset_);
      std::memcpy(new_shard_->data() + shard_index_offset_, shard_index.data(),
                  shard_index.size());
    }
    return absl::OkStatus();
  }

 private:
  /// Writes a single new chunk.
  Status WriteNewChunk(PendingChunkWrite* chunk_to_write) {
    chunk_to_write->write_status = WriteStatus::kSuccess;
    if (chunk_to_write->data) {
      TENSORSTORE_RETURN_IF_ERROR(encoder_.WriteIndexedEntry(
          chunk_to_write->minishard, chunk_to_write->chunk_id,
          *chunk_to_write->data, /*compress=*/true));
    }
    return absl::OkStatus();
  }

  /// Writes new chunks in order while `predicate(minishard, chunk_id)` is
  /// `true`.
  template <typename Predicate>
  Status WriteNewChunksWhile(Predicate predicate) {
    while (chunk_it_ != chunk_end_ &&
           predicate(chunk_it_->minishard, chunk_it_->chunk_id)) {
      PendingChunkWrite* chunk_to_write = nullptr;
      ChunkId chunk_id = chunk_it_->chunk_id;
      while (true) {
        // Returns `true` if the `if_equal` condition is satisfied.
        const auto ConditionsSatisfied = [&] {
          if (StorageGeneration::IsUnknown(chunk_it_->if_equal)) {
            // `if_equal` condition was not specified.
            return true;
          }
          if (chunk_to_write && chunk_to_write->data) {
            // Chunk was just written (not deleted) as part of this same
            // writeback.  The new generation is not yet known, and it is
            // therefore impossible for `if_equal` to match it.
            return false;
          }
          // Chunk is not already present.
          if (StorageGeneration::IsNoValue(chunk_it_->if_equal)) {
            return true;
          }
          return chunk_it_->if_equal == existing_generation_;
        };
        if (!ConditionsSatisfied()) {
          chunk_it_->write_status = WriteStatus::kAborted;
        } else {
          // Set to `kOverwritten` initially.  If this chunk is actually
          // written, the `write_status` is updated to `kSuccess`.
          chunk_it_->write_status = WriteStatus::kOverwritten;
          chunk_to_write = chunk_it_;
        }
        ++chunk_it_;
        if (chunk_it_ == chunk_end_ ||
            chunk_it_->chunk_id.value != chunk_id.value) {
          break;
        }
      }
      if (!chunk_to_write) continue;
      chunk_to_write->write_status = WriteStatus::kSuccess;
      if (chunk_to_write->data) modified_ = true;
      TENSORSTORE_RETURN_IF_ERROR(WriteNewChunk(chunk_to_write));
    }
    return absl::OkStatus();
  }

  /// Merges a single existing minishard with any new chunks in that minishard.
  ///
  /// This is called by `ProcessExistingShard` for each minishard in order.
  Status ProcessExistingMinishard(
      absl::string_view existing_shard, std::uint64_t minishard,
      span<const MinishardIndexEntry> minishard_index) {
    std::optional<ChunkId> prev_chunk_id;
    for (const auto& existing_entry : minishard_index) {
      if (prev_chunk_id &&
          existing_entry.chunk_id.value == prev_chunk_id->value) {
        return absl::FailedPreconditionError(
            StrCat("Chunk ", existing_entry.chunk_id.value,
                   " occurs more than once in the minishard index "
                   "for minishard ",
                   minishard));
      }
      prev_chunk_id = existing_entry.chunk_id;
      TENSORSTORE_RETURN_IF_ERROR(WriteNewChunksWhile(
          [&](std::uint64_t new_minishard, ChunkId new_chunk_id) {
            return new_minishard == minishard &&
                   new_chunk_id.value < existing_entry.chunk_id.value;
          }));
      PendingChunkWrite* chunk_to_write = nullptr;
      for (; chunk_it_ != chunk_end_ &&
             chunk_it_->chunk_id.value == existing_entry.chunk_id.value;
           ++chunk_it_) {
        // Returns `true` if the `if_equal` condition is satisfied.
        const auto ConditionsSatisfied = [&] {
          if (StorageGeneration::IsUnknown(chunk_it_->if_equal)) {
            // `if_equal` condition was not specified.
            return true;
          }
          if (chunk_to_write) {
            // Chunk was already updated (re-written or deleted) as part of
            // this writeback.  The existing entry is irrelevant.
            if (chunk_to_write->data) {
              // Chunk was re-written.  The new generation is not yet known,
              // and it is
              // therefore impossible for `if_equal` to match it.
              return false;
            }
            // Chunk was deleted.
            return StorageGeneration::IsNoValue(chunk_it_->if_equal);
          }
          // The existing entry has not yet been updated.
          return chunk_it_->if_equal == existing_generation_;
        };
        if (!ConditionsSatisfied()) {
          chunk_it_->write_status = WriteStatus::kAborted;
        } else {
          chunk_it_->write_status = WriteStatus::kOverwritten;
          chunk_to_write = chunk_it_;
        }
      }
      if (chunk_to_write) {
        modified_ = true;
        TENSORSTORE_RETURN_IF_ERROR(WriteNewChunk(chunk_to_write));
      } else {
        // Keep existing data.
        const auto GetChunkByteRange = [&]() -> Result<ByteRange> {
          TENSORSTORE_RETURN_IF_ERROR(
              OptionalByteRangeRequest(existing_entry.byte_range)
                  .Validate(existing_shard.size()));
          return existing_entry.byte_range;
        };
        auto chunk_byte_range_result = GetChunkByteRange();
        if (!chunk_byte_range_result.ok()) {
          return MaybeAnnotateStatus(
              chunk_byte_range_result.status(),
              StrCat("Invalid existing byte range for chunk ",
                     existing_entry.chunk_id.value));
        }
        TENSORSTORE_RETURN_IF_ERROR(encoder_.WriteIndexedEntry(
            minishard, existing_entry.chunk_id,
            internal::GetSubStringView(existing_shard,
                                       chunk_byte_range_result.value()),
            /*compress=*/false));
      }
    }
    return absl::OkStatus();
  }

  std::size_t shard_index_offset_;
  std::size_t shard_data_offset_;
  std::string* new_shard_;
  ShardEncoder encoder_;
  const StorageGeneration& existing_generation_;
  bool modified_ = false;
  PendingChunkWrite* chunk_it_;
  PendingChunkWrite* chunk_end_;
};

/// Applies write/delete operations to an existing or empty shard.
///
/// \param sharding_spec The sharding spec.
/// \param existing_generation The generation associated with the existing
///     shard.  This value is compared to the `if_equal` values of the
///     write/delete operations.
/// \param existing_shard The existing encoded shard data.
/// \param new_chunks[in,out] The write/delete operations to apply.  On
///     successful return, or if `absl::StatusCode::kAborted` is returned, the
///     `write_status` member of each `PendingChunkWrite` is updated to indicate
///     whether the operation was applied and whether it was subsequently
///     overwritten.
/// \param new_shard[out] Pointer to string to which the new encoded shard data
///     will be appended.
/// \returns `Status()` on success.
/// \error `absl::StatusCode::kAborted` if no changes would be made.
/// \error `absl::StatusCode::kFailedPrecondition` or
///     `absl::StatusCode::kInvalidArgument` if the existing data is invalid.
Status MergeShard(const ShardingSpec& sharding_spec,
                  const StorageGeneration& existing_generation,
                  absl::string_view existing_shard,
                  span<PendingChunkWrite> new_chunks, std::string* new_shard) {
  absl::c_sort(new_chunks,
               [&](const PendingChunkWrite& a, const PendingChunkWrite& b) {
                 return std::tuple(a.minishard, a.chunk_id.value) <
                        std::tuple(b.minishard, b.chunk_id.value);
               });
  MergeShardImpl merge_shard_impl(sharding_spec, existing_generation,
                                  new_chunks, new_shard);

  if (!existing_shard.empty()) {
    TENSORSTORE_RETURN_IF_ERROR(
        merge_shard_impl.ProcessExistingShard(existing_shard));
  }
  return merge_shard_impl.Finalize();
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
    : public internal::AsyncStorageBackedCache {
  using Base = internal::AsyncStorageBackedCache;

 public:
  class Entry : public Base::Entry {
   public:
    using Cache = ShardedKeyValueStoreWriteCache;
    Mutex mutex_;

    std::uint64_t shard() {
      std::uint64_t shard;
      assert(key().size() == sizeof(std::uint64_t));
      std::memcpy(&shard, key().data(), sizeof(std::uint64_t));
      return shard;
    }

    using Chunks = std::vector<PendingChunkWrite>;
    std::size_t chunk_memory_usage = 0;
    Chunks pending_writes_;
    Chunks issued_writes_;
    absl::Time last_write_time_ = absl::InfinitePast();
    std::optional<std::string> full_shard_data_;
    bool full_shard_discarded_ = false;
  };

  explicit ShardedKeyValueStoreWriteCache(
      internal::CachePtr<MinishardIndexCache> minishard_index_cache)
      : minishard_index_cache_(std::move(minishard_index_cache)) {}

  void DoDeleteEntry(internal::Cache::Entry* base_entry) override {
    Entry* entry = static_cast<Entry*>(base_entry);
    delete entry;
  }
  internal::Cache::Entry* DoAllocateEntry() override { return new Entry; }
  std::size_t DoGetSizeInBytes(Cache::Entry* base_entry) override {
    Entry* entry = static_cast<Entry*>(base_entry);
    // TODO: use better estimate
    return sizeof(Entry) + Base::DoGetSizeInBytes(base_entry) +
           (entry->pending_writes_.capacity() +
            entry->issued_writes_.capacity()) *
               sizeof(Entry::Chunks::value_type) +
           (entry->full_shard_data_ ? entry->full_shard_data_->size() : 0) +
           entry->chunk_memory_usage;
  }
  void DoRead(ReadOptions options, ReadReceiver receiver) override {
    auto* entry = static_cast<Entry*>(receiver.entry());
    const std::uint64_t shard = entry->shard();
    KeyValueStore::ReadOptions kvs_read_options;
    kvs_read_options.if_not_equal = std::move(options.existing_generation);
    kvs_read_options.staleness_bound = options.staleness_bound;

    base_kv_store()
        ->Read(GetShardKey(sharding_spec(), key_prefix(), shard),
               std::move(kvs_read_options))
        .ExecuteWhenReady(WithExecutor(
            executor(),
            [receiver = std::move(receiver)](
                ReadyFuture<KeyValueStore::ReadResult> future) mutable {
              auto& r = future.result();
              if (!r) {
                receiver.NotifyDone(/*size_update=*/{}, std::move(r).status());
                return;
              }
              auto* entry = static_cast<Entry*>(receiver.entry());
              if (r->aborted()) {
                return receiver.NotifyDone(/*size_update=*/{},
                                           std::move(r->generation));
              }
              std::unique_lock<Mutex> lock(entry->mutex_);
              entry->full_shard_discarded_ = false;
              if (r->not_found()) {
                entry->full_shard_data_ = std::nullopt;
              } else {
                entry->full_shard_data_.emplace(std::move(*r->value));
              }
              receiver.NotifyDone(
                  /*size_update=*/{std::move(lock),
                                   GetOwningCache(entry)->DoGetSizeInBytes(
                                       entry)},
                  std::move(r->generation));
            }));
  }
  void DoWriteback(TimestampedStorageGeneration existing_generation,
                   WritebackReceiver receiver) override;

  KeyValueStore* base_kv_store() const {
    return minishard_index_cache()->base_kv_store();
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
    return minishard_index_cache_;
  }

  internal::CachePtr<MinishardIndexCache> minishard_index_cache_;
};

std::vector<PendingChunkWrite> CompleteWriteback(
    ShardedKeyValueStoreWriteCache::WritebackReceiver receiver,
    internal::Cache::Entry::SizeUpdate::Lock lock,
    Result<TimestampedStorageGeneration> result) {
  auto* entry =
      static_cast<ShardedKeyValueStoreWriteCache::Entry*>(receiver.entry());
  std::vector<PendingChunkWrite> issued_writes;
  auto* cache = GetOwningCache(entry);
  if (result && StorageGeneration::IsUnknown(result->generation)) {
    // Retry issued writes and combine with pending writes.
    entry->issued_writes_.swap(entry->pending_writes_);
    entry->pending_writes_.insert(
        entry->pending_writes_.end(),
        std::make_move_iterator(entry->issued_writes_.begin()),
        std::make_move_iterator(entry->issued_writes_.end()));
    entry->issued_writes_.clear();
  } else {
    issued_writes = std::move(entry->issued_writes_);
    entry->full_shard_discarded_ = true;
    entry->full_shard_data_ = std::nullopt;
  }
  receiver.NotifyDone(
      {/*lock=*/std::move(lock), cache->DoGetSizeInBytes(entry)}, result);
  return issued_writes;
}

void CompletePendingChunkWrites(span<PendingChunkWrite> issued_writes,
                                Result<TimestampedStorageGeneration> result) {
  for (auto& chunk : issued_writes) {
    chunk.promise.SetResult([&]() -> Result<TimestampedStorageGeneration> {
      if (!result) {
        return result;
      } else if (chunk.write_status ==
                 PendingChunkWrite::WriteStatus::kAborted) {
        return {std::in_place, StorageGeneration::Unknown(), result->time};
      } else if (chunk.write_status ==
                 PendingChunkWrite::WriteStatus::kOverwritten) {
        return {std::in_place, StorageGeneration::Invalid(), result->time};
      } else {
        return *result;
      }
    }());
  }
}

void ShardedKeyValueStoreWriteCache::DoWriteback(
    TimestampedStorageGeneration existing_generation,
    WritebackReceiver receiver) {
  auto* entry = static_cast<Entry*>(receiver.entry());
  bool full_shard_discarded = false;
  {
    absl::MutexLock lock(&entry->mutex_);
    full_shard_discarded = entry->full_shard_discarded_;
  }
  if (full_shard_discarded) {
    receiver.NotifyStarted({});
    receiver.NotifyDone(/*size_update=*/{},
                        {std::in_place, StorageGeneration::Unknown(),
                         existing_generation.time});
    return;
  }
  executor()([receiver = std::move(receiver),
              existing_generation = std::move(existing_generation)]() mutable {
    auto* entry = static_cast<Entry*>(receiver.entry());
    const std::uint64_t shard = entry->shard();

    auto* cache = GetOwningCache(entry);
    std::string new_shard;
    const auto& sharding_spec = cache->sharding_spec();
    Status merge_status;
    absl::Time last_write_time;
    {
      std::unique_lock<Mutex> lock(entry->mutex_);
      last_write_time = entry->last_write_time_;
      merge_status = MergeShard(
          cache->sharding_spec(), existing_generation.generation,
          entry->full_shard_data_ ? absl::string_view(*entry->full_shard_data_)
                                  : absl::string_view(),
          entry->pending_writes_, &new_shard);
      if (!merge_status.ok() &&
          merge_status.code() != absl::StatusCode::kAborted) {
        merge_status = absl::FailedPreconditionError(merge_status.message());
        merge_status = MaybeAnnotateStatus(
            merge_status, StrCat("Error decoding existing shard ", shard));
      }
      entry->issued_writes_ = std::move(entry->pending_writes_);
      receiver.NotifyStarted({/*lock=*/std::move(lock)});
    }

    if (!merge_status.ok()) {
      Result<TimestampedStorageGeneration> writeback_result = merge_status;
      Result<TimestampedStorageGeneration> pending_chunk_result = merge_status;
      if (merge_status.code() == absl::StatusCode::kAborted) {
        // No changes were made to the shard.
        if (last_write_time > existing_generation.time) {
          // At least one write request is newer than the cached shard data,
          // which means the cached shard data may be out of date.  Complete
          // with `StorageGeneration::Unknown()` to force a re-read.
          writeback_result.emplace(StorageGeneration::Unknown(),
                                   last_write_time);
        } else {
          // Complete with `absl::StatusCode::kAborted` to indicate that no
          // writeback is needed.
          pending_chunk_result = existing_generation;
        }
      }
      auto issued_writes = CompleteWriteback(
          std::move(receiver), std::unique_lock<Mutex>(entry->mutex_),
          std::move(writeback_result));
      CompletePendingChunkWrites(issued_writes,
                                 std::move(pending_chunk_result));
      return;
    }
    Future<TimestampedStorageGeneration> future;
    auto shard_key = GetShardKey(sharding_spec, cache->key_prefix(), shard);
    StorageGeneration generation_condition =
        StorageGeneration::IsUnknown(existing_generation.generation)
            ? StorageGeneration::NoValue()
            : std::move(existing_generation.generation);
    if (new_shard.empty()) {
      future = cache->base_kv_store()->Delete(
          std::move(shard_key), {std::move(generation_condition)});
    } else {
      future = cache->base_kv_store()->Write(std::move(shard_key),
                                             std::move(new_shard),
                                             {std::move(generation_condition)});
    }
    future.Force();
    std::move(future).ExecuteWhenReady(WithExecutor(
        cache->executor(),
        [receiver = std::move(receiver)](
            ReadyFuture<TimestampedStorageGeneration> future) mutable {
          auto* entry = static_cast<Entry*>(receiver.entry());
          std::unique_lock<Mutex> lock(entry->mutex_);
          auto issued_writes = CompleteWriteback(
              std::move(receiver), std::move(lock), future.result());
          CompletePendingChunkWrites(issued_writes, std::move(future.result()));
        }));
  });
}

Result<ChunkId> ParseKey(absl::string_view key) {
  if (key.size() != sizeof(ChunkId)) {
    return absl::InvalidArgumentError("Invalid key");
  }
  ChunkId chunk_id;
  std::memcpy(&chunk_id, key.data(), sizeof(chunk_id));
  return chunk_id;
}

/// Asynchronous callback invoked (indirectly) from `ShardedKeyValueStore::Read`
/// when the cache entry for a given minishard index is ready.
struct MinishardIndexCacheEntryReadyCallback {
  using ReadResult = KeyValueStore::ReadResult;
  using ReadOptions = KeyValueStore::ReadOptions;

  internal::PinnedCacheEntry<MinishardIndexCache> entry_;
  ChunkId chunk_id_;
  ReadOptions options_;
  void operator()(Promise<ReadResult> promise, ReadyFuture<const void>) {
    std::optional<ByteRange> byte_range;
    TimestampedStorageGeneration generation;
    {
      absl::ReaderMutexLock lock(&entry_->mutex_);
      generation.time = entry_->generation_.time;
      if (!StorageGeneration::IsNoValue(entry_->generation_.generation) &&
          (options_.if_not_equal == entry_->generation_.generation ||
           (!StorageGeneration::IsUnknown(options_.if_equal) &&
            options_.if_equal != entry_->generation_.generation))) {
        generation.generation = StorageGeneration::Unknown();
      } else {
        byte_range = FindChunkInMinishard(entry_->minishard_index_, chunk_id_);
        generation.generation = entry_->generation_.generation;
      }
    }
    if (!byte_range) {
      promise.SetResult(ReadResult{/*value=*/std::nullopt,
                                   /*generation=*/std::move(generation)});
      return;
    }
    assert(!StorageGeneration::IsUnknown(generation.generation));
    auto* cache = GetOwningCache(entry_);
    ReadOptions kvs_read_options;
    kvs_read_options.if_equal = generation.generation;
    kvs_read_options.staleness_bound = options_.staleness_bound;
    assert(options_.byte_range.SatisfiesInvariants());
    OptionalByteRangeRequest post_decode_byte_range;
    const auto data_encoding = cache->sharding_spec().data_encoding;
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
            auto* cache = GetOwningCache(entry);
            auto minishard_index_read_future =
                entry->Read(/*staleness_bound=*/absl::InfiniteFuture());
            LinkValue(WithExecutor(
                          cache->executor(),
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
            auto result = DecodeData(*r->value, data_encoding);
            if (!result) {
              promise.SetResult(
                  absl::FailedPreconditionError(result.status().message()));
              return;
            } else {
              *r->value = *result;
            }
          }
          TENSORSTORE_ASSIGN_OR_RETURN(
              auto byte_range,
              post_decode_byte_range.Validate(r->value->size()),
              static_cast<void>(promise.SetResult(_)));
          *r->value = internal::GetSubString(std::move(*r->value), byte_range);
          promise.SetResult(std::move(r));
        },
        std::move(promise),
        cache->base_kv_store_->Read(
            GetShardKey(cache->sharding_spec(), cache->key_prefix(), shard),
            std::move(kvs_read_options)));
  }
};

class ShardedKeyValueStore : public KeyValueStore {
 public:
  explicit ShardedKeyValueStore(KeyValueStore::Ptr base_kv_store,
                                Executor executor, std::string key_prefix,
                                const ShardingSpec& sharding_spec,
                                internal::CachePool::WeakPtr cache_pool)
      : write_cache_(
            cache_pool->GetCache<ShardedKeyValueStoreWriteCache>("", [&] {
              return std::make_unique<ShardedKeyValueStoreWriteCache>(
                  cache_pool->GetCache<MinishardIndexCache>("", [&] {
                    return std::make_unique<MinishardIndexCache>(
                        std::move(base_kv_store), std::move(executor),
                        std::move(key_prefix), sharding_spec);
                  }));
            })) {}

  Future<ReadResult> Read(Key key, ReadOptions options) override {
    TENSORSTORE_ASSIGN_OR_RETURN(ChunkId chunk_id, ParseKey(key));
    auto shard_info = GetChunkShardInfo(sharding_spec(), chunk_id);

    auto minishard_index_cache_entry = GetCacheEntry(
        minishard_index_cache(),
        absl::string_view(reinterpret_cast<const char*>(&shard_info),
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

  Future<TimestampedStorageGeneration> Write(Key key, Value value,
                                             WriteOptions options) override {
    return WriteImpl(std::move(key), std::move(value), std::move(options));
  }

  Future<TimestampedStorageGeneration> Delete(Key key,
                                              DeleteOptions options) override {
    return WriteImpl(std::move(key), std::nullopt, std::move(options));
  }

  void ListImpl(const ListOptions& options,
                AnyFlowReceiver<Status, Key> receiver) override {
    execution::submit(
        FlowSingleSender{ErrorSender{absl::UnimplementedError("")}},
        std::move(receiver));
  }

  Future<std::int64_t> DeletePrefix(Key prefix) override {
    if (!prefix.empty()) {
      return absl::InvalidArgumentError("Only empty prefix is supported");
    }
    const auto& key_prefix = this->key_prefix();
    return base_kv_store()->DeletePrefix(key_prefix.empty() ? std::string()
                                                            : key_prefix + "/");
  }

  Future<TimestampedStorageGeneration> WriteImpl(
      Key key, std::optional<std::string> value, WriteOptions options) {
    TENSORSTORE_ASSIGN_OR_RETURN(ChunkId chunk_id, ParseKey(key));
    const auto& sharding_spec = this->sharding_spec();
    const auto shard_info = GetSplitShardInfo(
        sharding_spec, GetChunkShardInfo(sharding_spec, chunk_id));
    const std::uint64_t shard = shard_info.shard;
    auto entry = GetCacheEntry(
        write_cache_, absl::string_view(reinterpret_cast<const char*>(&shard),
                                        sizeof(shard)));
    auto [promise, future] =
        PromiseFuturePair<TimestampedStorageGeneration>::Make();
    std::unique_lock<Mutex> lock(entry->mutex_);
    {
      auto& chunk = entry->pending_writes_.emplace_back();
      entry->last_write_time_ = absl::Now();
      chunk.minishard = shard_info.minishard;
      chunk.chunk_id = chunk_id;
      chunk.promise = promise;
      chunk.data = std::move(value);
      chunk.if_equal = std::move(options.if_equal);
    }
    LinkError(std::move(promise),
              entry->FinishWrite(
                  {std::move(lock),
                   /*new_size=*/write_cache_->DoGetSizeInBytes(entry.get())},
                  internal::AsyncStorageBackedCache::WriteFlags::
                      kUnconditionalWriteback));
    return future;
  }

  KeyValueStore* base_kv_store() const {
    return minishard_index_cache()->base_kv_store();
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

  internal::CachePtr<ShardedKeyValueStoreWriteCache> write_cache_;
};

}  // namespace

KeyValueStore::Ptr GetShardedKeyValueStore(
    KeyValueStore::Ptr base_kv_store, Executor executor, std::string key_prefix,
    const ShardingSpec& sharding_spec,
    internal::CachePool::WeakPtr cache_pool) {
  return KeyValueStore::Ptr(new ShardedKeyValueStore(
      std::move(base_kv_store), std::move(executor), std::move(key_prefix),
      sharding_spec, std::move(cache_pool)));
}

}  // namespace neuroglancer_uint64_sharded
}  // namespace tensorstore
