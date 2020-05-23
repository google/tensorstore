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
#include "tensorstore/internal/aggregate_writeback_cache.h"
#include "tensorstore/internal/async_storage_backed_cache.h"
#include "tensorstore/internal/compression/zlib.h"
#include "tensorstore/internal/key_value_store_cache.h"
#include "tensorstore/internal/mutex.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/to_string.h"

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

/// Read-only KeyValueStore for retrieving a minishard index
///
/// The key is a `ChunkCombinedShardInfo` (in native memory layout).  The value
/// is the encoded minishard index.
///
/// This is used by `MinishardIndexCache`, which decodes and caches the
/// minishard indices.  By using a separate `KeyValueStore` rather than just
/// including this logic directly in `MinishardIndexCache`, we are able to take
/// advantage of `KeyValueStoreCache` to define `MinishardIndexCache`.
class MinishardIndexKeyValueStore : public KeyValueStore {
 public:
  explicit MinishardIndexKeyValueStore(KeyValueStore::Ptr base,
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
    return future;
  }

  std::string DescribeKey(absl::string_view key) override {
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

  KeyValueStore* base() { return base_.get(); }
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
    // 2. Request the minishard index from the byte range specified in the shard
    //    index.
    //
    //    a. If the generation has changed (concurrent modification of the
    //       shard), retry starting at step 1.
    //
    //    b. If not found, the minishard is empty.  (This can only happens in
    //       the case of concurrent modification of the shard).  Done.
    //
    //    c.  Otherwise, return the encoded minishard index.
    struct MinishardIndexReadyCallback {
      KeyValueStore::PtrT<MinishardIndexKeyValueStore> self;
      ChunkSplitShardInfo split_info;

      void operator()(Promise<KeyValueStore::ReadResult> promise,
                      ReadyFuture<KeyValueStore::ReadResult> future) {
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
          options.staleness_bound = r->generation.time;
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
      KeyValueStore::PtrT<MinishardIndexKeyValueStore> self;
      ChunkSplitShardInfo split_info;
      absl::Time staleness_bound;
      static void SetError(const Promise<KeyValueStore::ReadResult>& promise,
                           Status error) {
        promise.SetResult(MaybeAnnotateStatus(
            ConvertInvalidArgumentToFailedPrecondition(std::move(error)),
            StrCat("Error retrieving shard index entry")));
      }

      void operator()(Promise<KeyValueStore::ReadResult> promise,
                      ReadyFuture<KeyValueStore::ReadResult> future) {
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
        TENSORSTORE_ASSIGN_OR_RETURN(auto byte_range,
                                     DecodeShardIndexEntry(*r->value),
                                     SetError(promise, _));
        TENSORSTORE_ASSIGN_OR_RETURN(
            byte_range,
            GetAbsoluteShardByteRange(byte_range, self->sharding_spec_),
            SetError(promise, _));
        if (byte_range.size() == 0) {
          // Minishard index is 0 bytes, which means the minishard is empty.
          r->value = std::nullopt;
          promise.SetResult(std::move(r));
          return;
        }
        KeyValueStore::ReadOptions kvs_read_options;
        // The `if_equal` condition ensure that an "aborted" `ReadResult` is
        // returned in the case of a concurrent modification (case 2a above).
        kvs_read_options.if_equal = std::move(r->generation.generation);
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
    Link(
        WithExecutor(executor_,
                     ShardIndexReadyCallback{
                         KeyValueStore::PtrT<MinishardIndexKeyValueStore>(this),
                         split_info, staleness_bound}),
        std::move(promise),
        base_->Read(GetShardKey(sharding_spec_, key_prefix_, split_info.shard),
                    std::move(options)));
  }

  KeyValueStore::Ptr base_;
  Executor executor_;
  std::string key_prefix_;
  ShardingSpec sharding_spec_;
};

/// Caches minishard indexes.
///
/// Each cache entry corresponds to a particular minishard within a particular
/// shard.  The entry keys directly encode `ChunkCombinedShardInfo` values (via
/// `memcpy`), specifying a shard and minishard number.
///
/// This cache is only used for reading.  `FinishWrite` must never be called.
class MinishardIndexCache
    : public internal::CacheBase<
          MinishardIndexCache,
          internal::KeyValueStoreCache<internal::AsyncStorageBackedCache>> {
  using Base = internal::CacheBase<
      MinishardIndexCache,
      internal::KeyValueStoreCache<internal::AsyncStorageBackedCache>>;

 public:
  class Entry : public Base::Entry {
   public:
    using Cache = MinishardIndexCache;
    std::vector<MinishardIndexEntry> minishard_index_;

    ChunkSplitShardInfo shard_info() {
      ChunkCombinedShardInfo combined_info;
      assert(this->key().size() == sizeof(combined_info));
      std::memcpy(&combined_info, this->key().data(), sizeof(combined_info));
      return GetSplitShardInfo(GetOwningCache(this)->sharding_spec(),
                               combined_info);
    }
  };

  explicit MinishardIndexCache(KeyValueStore::Ptr base_kvstore,
                               Executor executor, std::string key_prefix,
                               const ShardingSpec& sharding_spec)
      : Base(KeyValueStore::Ptr(new MinishardIndexKeyValueStore(
                 std::move(base_kvstore), executor, std::move(key_prefix),
                 sharding_spec)),
             executor) {}

  MinishardIndexKeyValueStore* kvstore() {
    return static_cast<MinishardIndexKeyValueStore*>(this->Base::kvstore());
  }

  std::size_t DoGetReadStateSizeInBytes(
      internal::Cache::Entry* base_entry) override {
    Entry* entry = static_cast<Entry*>(base_entry);
    return entry->minishard_index_.capacity() * sizeof(MinishardIndexEntry);
  }

  void DoDecode(internal::Cache::PinnedEntry base_entry,
                std::optional<std::string> value) override {
    auto* entry = static_cast<Entry*>(base_entry.get());
    std::vector<MinishardIndexEntry> minishard_index;
    if (value) {
      TENSORSTORE_ASSIGN_OR_RETURN(
          minishard_index,
          DecodeMinishardIndexAndAdjustByteRanges(*value, sharding_spec()),
          this->NotifyReadError(entry,
                                ConvertInvalidArgumentToFailedPrecondition(_)));
    }
    auto lock = entry->AcquireReadStateWriterLock();
    entry->minishard_index_ = std::move(minishard_index);
    this->NotifyReadSuccess(entry, std::move(lock));
  }

  const ShardingSpec& sharding_spec() { return kvstore()->sharding_spec(); }

  KeyValueStore* base_kvstore() { return kvstore()->base(); }
  const Executor& executor() { return kvstore()->executor(); }
  const std::string& key_prefix() { return kvstore()->key_prefix(); }
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
  ///
  /// \dchecks `!existing_shard.empty()`
  Status ProcessExistingShard(absl::string_view existing_shard) {
    assert(!existing_shard.empty());
    const std::uint64_t num_minishards =
        encoder_.sharding_spec().num_minishards();
    if (existing_shard.size() < num_minishards * 16) {
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
  Status Finalize(size_t* total_chunks, bool* unconditional) {
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
    *total_chunks = total_chunks_;
    *unconditional = unconditional_;
    return absl::OkStatus();
  }

 private:
  /// Writes a single new chunk.
  Status WriteNewChunk(PendingChunkWrite* chunk_to_write) {
    if (!StorageGeneration::IsUnknown(chunk_to_write->if_equal)) {
      unconditional_ = false;
    }
    chunk_to_write->write_status = WriteStatus::kSuccess;
    if (chunk_to_write->data) {
      TENSORSTORE_RETURN_IF_ERROR(encoder_.WriteIndexedEntry(
          chunk_to_write->minishard, chunk_to_write->chunk_id,
          *chunk_to_write->data, /*compress=*/true));
      ++total_chunks_;
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
              // Chunk was re-written.  The new generation is not yet known, and
              // it is therefore impossible for `if_equal` to match it.
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
        ++total_chunks_;
        unconditional_ = false;
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
  size_t total_chunks_ = 0;
  bool unconditional_ = true;
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
/// \param total_chunks[out] Set to total number of chunks in `new_shard`.
/// \param unconditional[out] Set to `true` if the result is not conditioned on
///     `existing_shard`.
/// \returns `Status()` on success.
/// \error `absl::StatusCode::kAborted` if no changes would be made.
/// \error `absl::StatusCode::kFailedPrecondition` or
///     `absl::StatusCode::kInvalidArgument` if the existing data is invalid.
Status MergeShard(const ShardingSpec& sharding_spec,
                  const StorageGeneration& existing_generation,
                  absl::string_view existing_shard,
                  span<PendingChunkWrite> new_chunks, std::string* new_shard,
                  size_t* total_chunks, bool* unconditional) {
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
  return merge_shard_impl.Finalize(total_chunks, unconditional);
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
    : public internal::CacheBase<ShardedKeyValueStoreWriteCache,
                                 internal::AggregateWritebackCache<
                                     ShardedKeyValueStoreWriteCache,
                                     internal::KeyValueStoreCache<
                                         internal::AsyncStorageBackedCache>>> {
  using Base = internal::CacheBase<
      ShardedKeyValueStoreWriteCache,
      internal::AggregateWritebackCache<
          ShardedKeyValueStoreWriteCache,
          internal::KeyValueStoreCache<internal::AsyncStorageBackedCache>>>;

 public:
  using PendingWrite = PendingChunkWrite;

  class Entry : public Base::Entry {
   public:
    using Cache = ShardedKeyValueStoreWriteCache;

    std::uint64_t shard() {
      std::uint64_t shard;
      assert(key().size() == sizeof(std::uint64_t));
      std::memcpy(&shard, key().data(), sizeof(std::uint64_t));
      return shard;
    }

    // The following members are part of the read state:

    // Set to the full encoded shard that was read.  After writeback, this will
    // be reset to save memory, and `full_shard_discarded_` will be set to
    // `true`.
    std::optional<std::string> full_shard_data_;
    // Set to `true` to indicate that there is likely an existing shard that
    // should be read before re-writing.
    bool full_shard_discarded_ = false;
  };

  explicit ShardedKeyValueStoreWriteCache(
      internal::CachePtr<MinishardIndexCache> minishard_index_cache,
      GetMaxChunksPerShardFunction get_max_chunks_per_shard)
      : Base(KeyValueStore::Ptr(minishard_index_cache->base_kvstore()),
             minishard_index_cache->executor()),
        minishard_index_cache_(std::move(minishard_index_cache)),
        get_max_chunks_per_shard_(std::move(get_max_chunks_per_shard)) {}

  size_t DoGetReadStateSizeInBytes(
      internal::Cache::Entry* base_entry) override {
    auto* entry = static_cast<Entry*>(base_entry);
    if (entry->full_shard_data_) return entry->full_shard_data_->size();
    return 0;
  }

  void DoDecode(internal::Cache::PinnedEntry base_entry,
                std::optional<std::string> value) override {
    auto* entry = static_cast<Entry*>(base_entry.get());
    auto lock = entry->AcquireReadStateWriterLock();
    entry->full_shard_data_ = std::move(value);
    entry->full_shard_discarded_ = false;
    this->NotifyReadSuccess(entry, std::move(lock));
  }

  std::string GetKeyValueStoreKey(internal::Cache::Entry* base_entry) override {
    auto* entry = static_cast<Entry*>(base_entry);
    const std::uint64_t shard = entry->shard();
    return GetShardKey(sharding_spec(), key_prefix(), shard);
  }

  void DoWriteback(internal::Cache::PinnedEntry entry) override;

  void NotifyWritebackSuccess(internal::Cache::Entry* entry,
                              WriteAndReadStateLock lock) override;

  const ShardingSpec& sharding_spec() const {
    return minishard_index_cache()->sharding_spec();
  }

  const std::string& key_prefix() const {
    return minishard_index_cache()->key_prefix();
  }

  const internal::CachePtr<MinishardIndexCache>& minishard_index_cache() const {
    return minishard_index_cache_;
  }

  internal::CachePtr<MinishardIndexCache> minishard_index_cache_;

  GetMaxChunksPerShardFunction get_max_chunks_per_shard_;
};

void ShardedKeyValueStoreWriteCache::NotifyWritebackSuccess(
    internal::Cache::Entry* base_entry, WriteAndReadStateLock lock) {
  auto* entry = static_cast<Entry*>(base_entry);
  std::vector<PendingChunkWrite> issued_writes;
  std::swap(issued_writes, entry->issued_writes);

  // Timestamp and generation were set by `KeyValueStoreCache`.
  TimestampedStorageGeneration generation;
  generation.time = entry->last_read_time;
  generation.generation = std::move(entry->last_read_generation);

  entry->full_shard_discarded_ = true;
  entry->full_shard_data_ = std::nullopt;
  entry->last_read_generation = StorageGeneration::Unknown();
  Base::NotifyWritebackSuccess(entry, std::move(lock));

  for (auto& chunk : issued_writes) {
    TimestampedStorageGeneration chunk_result;
    chunk_result.time = generation.time;
    switch (chunk.write_status) {
      case PendingChunkWrite::WriteStatus::kAborted:
        chunk_result.generation = StorageGeneration::Unknown();
        break;
      case PendingChunkWrite::WriteStatus::kOverwritten:
        chunk_result.generation = StorageGeneration::Invalid();
        break;
      case PendingChunkWrite::WriteStatus::kSuccess:
        chunk_result.generation = generation.generation;
        break;
    }
    chunk.promise.SetResult(std::move(chunk_result));
  }
}

void ShardedKeyValueStoreWriteCache::DoWriteback(
    internal::Cache::PinnedEntry base_entry) {
  executor()([entry =
                  internal::static_pointer_cast<Entry>(base_entry)]() mutable {
    auto* cache = GetOwningCache(entry);
    std::string new_shard;
    Status merge_status;
    size_t total_chunks = 0;
    bool unconditional = false;
    absl::Time last_write_time;
    {
      auto lock = entry->AcquireWriteStateLock();
      last_write_time = entry->last_pending_write_time;
      merge_status = MergeShard(
          cache->sharding_spec(), entry->last_read_generation,
          entry->full_shard_data_ ? absl::string_view(*entry->full_shard_data_)
                                  : absl::string_view(),
          entry->pending_writes, &new_shard, &total_chunks, &unconditional);
      if (!merge_status.ok() &&
          merge_status.code() != absl::StatusCode::kAborted) {
        merge_status = absl::FailedPreconditionError(merge_status.message());
      }
      cache->NotifyWritebackStarted(entry.get(), std::move(lock));
    }

    if (unconditional &&
        (!cache->get_max_chunks_per_shard_ ||
         cache->get_max_chunks_per_shard_(entry->shard()) != total_chunks)) {
      // If we aren't fully overwriting the shard, a conditional write is
      // required.
      unconditional = false;
    }

    if (!merge_status.ok()) {
      Result<TimestampedStorageGeneration> writeback_result = merge_status;
      Result<TimestampedStorageGeneration> pending_chunk_result = merge_status;
      if (merge_status.code() == absl::StatusCode::kAborted) {
        // No changes were made to the shard.
        if (last_write_time > entry->last_read_time ||
            entry->full_shard_discarded_) {
          // At least one write request is newer than the cached shard data,
          // which means the cached shard data may be out of date.
          cache->NotifyWritebackNeedsRead(
              entry.get(), entry->AcquireWriteStateLock(), last_write_time);
        } else {
          cache->NotifyWritebackSuccess(entry.get(),
                                        entry->AcquireWriteAndReadStateLock());
        }
      } else {
        cache->NotifyWritebackError(entry.get(), entry->AcquireWriteStateLock(),
                                    std::move(merge_status));
      }
      return;
    }

    if (entry->full_shard_discarded_ && !unconditional) {
      cache->NotifyWritebackNeedsRead(
          entry.get(), entry->AcquireWriteStateLock(), entry->last_read_time);
      return;
    }

    std::optional<std::string> new_value;
    if (!new_shard.empty()) new_value = std::move(new_shard);
    cache->Writeback(std::move(entry), std::move(new_value),
                     /*unconditional=*/unconditional);
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
      auto lock = entry_->AcquireReadStateReaderLock();
      generation.time = entry_->last_read_time;
      if (!StorageGeneration::IsNoValue(entry_->last_read_generation) &&
          (options_.if_not_equal == entry_->last_read_generation ||
           (!StorageGeneration::IsUnknown(options_.if_equal) &&
            options_.if_equal != entry_->last_read_generation))) {
        generation.generation = StorageGeneration::Unknown();
      } else {
        byte_range = FindChunkInMinishard(entry_->minishard_index_, chunk_id_);
        generation.generation = entry_->last_read_generation;
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
        cache->base_kvstore()->Read(
            GetShardKey(cache->sharding_spec(), cache->key_prefix(), shard),
            std::move(kvs_read_options)));
  }
};

class ShardedKeyValueStore : public KeyValueStore {
 public:
  explicit ShardedKeyValueStore(
      KeyValueStore::Ptr base_kvstore, Executor executor,
      std::string key_prefix, const ShardingSpec& sharding_spec,
      internal::CachePool::WeakPtr cache_pool,
      GetMaxChunksPerShardFunction get_max_chunks_per_shard)
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

  Future<std::int64_t> DeletePrefix(Key prefix) override {
    if (!prefix.empty()) {
      return absl::InvalidArgumentError("Only empty prefix is supported");
    }
    const auto& key_prefix = this->key_prefix();
    return base_kvstore()->DeletePrefix(key_prefix.empty() ? std::string()
                                                           : key_prefix + "/");
  }

  Future<TimestampedStorageGeneration> Write(Key key,
                                             std::optional<std::string> value,
                                             WriteOptions options) override {
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
    PendingChunkWrite chunk;
    chunk.promise = promise;
    chunk.minishard = shard_info.minishard;
    chunk.chunk_id = chunk_id;
    chunk.data = std::move(value);
    chunk.if_equal = std::move(options.if_equal);
    LinkError(std::move(promise),
              entry->AddPendingWrite(std::move(chunk),
                                     internal::AsyncStorageBackedCache::
                                         WriteFlags::kUnconditionalWriteback));
    return future;
  }

  std::string DescribeKey(absl::string_view key) override {
    TENSORSTORE_ASSIGN_OR_RETURN(
        ChunkId chunk_id, ParseKey(key),
        tensorstore::StrCat("invalid key ", tensorstore::QuoteString(key)));
    const auto& sharding_spec = this->sharding_spec();
    const auto shard_info = GetSplitShardInfo(
        sharding_spec, GetChunkShardInfo(sharding_spec, chunk_id));
    return tensorstore::StrCat(
        "chunk ", chunk_id.value, " in minishard ", shard_info.minishard,
        " in ",
        base_kvstore()->DescribeKey(
            GetShardKey(sharding_spec, key_prefix(), shard_info.shard)));
  }

  KeyValueStore* base_kvstore() const {
    return minishard_index_cache()->base_kvstore();
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
    KeyValueStore::Ptr base_kvstore, Executor executor, std::string key_prefix,
    const ShardingSpec& sharding_spec, internal::CachePool::WeakPtr cache_pool,
    GetMaxChunksPerShardFunction get_max_chunks_per_shard) {
  return KeyValueStore::Ptr(new ShardedKeyValueStore(
      std::move(base_kvstore), std::move(executor), std::move(key_prefix),
      sharding_spec, std::move(cache_pool),
      std::move(get_max_chunks_per_shard)));
}

}  // namespace neuroglancer_uint64_sharded
}  // namespace tensorstore
