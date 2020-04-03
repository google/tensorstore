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

#ifndef TENSORSTORE_DRIVER_KVS_BACKED_CHUNK_DRIVER_IMPL_H_
#define TENSORSTORE_DRIVER_KVS_BACKED_CHUNK_DRIVER_IMPL_H_

/// \file
/// Implementation details of `kvs_backed_chunk_driver.h`.

#include <memory>
#include <string>

#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"
#include "tensorstore/box.h"
#include "tensorstore/driver/kvs_backed_chunk_driver.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/internal/async_storage_backed_cache.h"
#include "tensorstore/internal/chunk_cache.h"
#include "tensorstore/kvstore/key_value_store.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_kvs_backed_chunk_driver {

/// Caches metadata associated with a KeyValueStore-backed chunk driver.
///
/// There is one entry in the `MetadataCache` for each `DataCache` (but there
/// may be multiple `DataCache` objects associated with a single entry).
class MetadataCache : public internal::AsyncStorageBackedCache {
 public:
  using MetadataPtr = std::shared_ptr<const void>;

  class Entry : public internal::AsyncStorageBackedCache::Entry {
   public:
    using Cache = MetadataCache;

    MetadataPtr GetMetadata() {
      absl::ReaderMutexLock lock(&metadata_mutex);
      return metadata;
    }

    /// Function invoked to atomically modify a metadata entry (e.g. to create
    /// or resize an array).
    ///
    /// This function may be called multiple times (even after returning an
    /// error status) due to retries to handle concurrent modifications to the
    /// metadata.
    ///
    /// \param existing_metadata Specifies the existing metadata of type
    ///     `Metadata`, or `nullptr` to indicate no existing metadata.
    /// \returns The new metadata on success, or an error result for the
    /// request.
    using UpdateFunction =
        std::function<Result<MetadataPtr>(const void* existing_metadata)>;

    /// Requests an atomic metadata update.
    ///
    /// \param update Update function to apply.
    /// \param update_constraint Specifies additional constraints on the atomic
    ///     update.
    /// \returns Future that becomes ready when the request has completed
    ///     (either successfully or with an error).  Any error returned from the
    ///     last call to `update` (i.e. the last retry) is returned as the
    ///     Future result.  Additionally, any error that occurs with the
    ///     underlying KeyValueStore is also returned.
    Future<const void> RequestAtomicUpdate(
        UpdateFunction update, AtomicUpdateConstraint update_constraint,
        absl::Time request_time = absl::Now());

    /// Specifies a request to atomically read-modify-write a metadata entry.
    struct UpdateRequest {
      absl::Time request_time;
      UpdateFunction update;
      Promise<void> promise;
      AtomicUpdateConstraint update_constraint;
    };

    Mutex metadata_mutex;
    MetadataPtr metadata;

    /// Requests that have been enqueued but not yet attempted.
    std::vector<UpdateRequest> pending_requests;

    /// Requests that have already been attempted but not completed, and may be
    /// retried.
    std::vector<UpdateRequest> issued_requests;
  };

  MetadataCache(MetadataCacheState::Ptr state, KeyValueStore::Ptr base_store,
                KeyValueStore::Ptr store,
                Context::Resource<internal::DataCopyConcurrencyResource>
                    data_copy_concurrency,
                Context::Resource<internal::CachePoolResource> cache_pool);

  MetadataCacheState* state() { return state_.get(); }
  KeyValueStore* base_store() { return base_store_.get(); }
  KeyValueStore* store() { return store_.get(); }
  const Executor& executor() { return data_copy_concurrency_->executor; }

  void DoDeleteEntry(internal::Cache::Entry* base_entry) override;

  internal::Cache::Entry* DoAllocateEntry() override;
  std::size_t DoGetSizeInBytes(Cache::Entry* base_entry) override;
  void DoRead(ReadOptions options, ReadReceiver receiver) override;
  void DoWriteback(TimestampedStorageGeneration existing_generation,
                   WritebackReceiver receiver) override;

  MetadataCacheState::Ptr state_;
  /// KeyValueStore from which `store_` was derived.  Used only by
  /// `DriverBase::GetBoundSpecData`.  A driver implementation may apply some
  /// type of adapter to the `KeyValueStore` in order to retrieve metadata by
  /// overriding the default implementation of
  /// `OpenState::GetMetadataKeyValueStore`.
  KeyValueStore::Ptr base_store_;
  /// KeyValueStore used for retrieving the metadata.
  KeyValueStore::Ptr store_;
  Context::Resource<internal::DataCopyConcurrencyResource>
      data_copy_concurrency_;
  Context::Resource<internal::CachePoolResource> cache_pool_;
};

inline MetadataCache::Entry* GetMetadataCacheEntry(const OpenState& state) {
  auto& base = (PrivateOpenState&)state;  // Cast to private base
  return static_cast<MetadataCache::Entry*>(base.metadata_cache_entry_.get());
}

inline MetadataCache* GetMetadataCache(const OpenState& state) {
  auto& base = (PrivateOpenState&)state;  // Cast to private base
  return static_cast<MetadataCache*>(
      GetOwningCache(base.metadata_cache_entry_));
}

inline internal::PinnedCacheEntry<MetadataCache> GetMetadataCacheEntry(
    OpenState&& state) {
  auto& base = (PrivateOpenState&)state;  // Cast to private base
  return internal::static_pointer_cast<MetadataCache::Entry>(
      std::move(base.metadata_cache_entry_));
}

class DataCache : public internal::ChunkCache {
 public:
  using MetadataPtr = MetadataCache::MetadataPtr;
  explicit DataCache(
      DataCacheState::Ptr state, KeyValueStore::Ptr store,
      internal::PinnedCacheEntry<MetadataCache> metadata_cache_entry,
      MetadataPtr metadata);

  void DoRead(ReadOptions options, ReadReceiver receiver) override;

  void DoWriteback(TimestampedStorageGeneration existing_generation,
                   WritebackReceiver receiver) override;

  MetadataCache* metadata_cache() {
    return GetOwningCache(metadata_cache_entry_);
  }
  KeyValueStore* store() { return store_.get(); }
  const Executor& executor() { return metadata_cache()->executor(); }

  std::string GetChunkStorageKey(Entry* entry) {
    return state_->GetChunkStorageKey(initial_metadata_.get(),
                                      entry->cell_indices());
  }

  MetadataPtr validated_metadata() {
    absl::ReaderMutexLock lock(&mutex_);
    return validated_metadata_;
  }

  const DataCacheState::Ptr state_;
  KeyValueStore::Ptr store_;
  const internal::PinnedCacheEntry<MetadataCache> metadata_cache_entry_;
  const MetadataPtr initial_metadata_;
  absl::Mutex mutex_;
  MetadataPtr validated_metadata_ ABSL_GUARDED_BY(mutex_);
};

/// Validates that the resize operation specified by
/// `new_{inclusive_min,exclusive_max}` can be applied to `current_domaian`
/// subject to the constraints of `{inclusive_min,exclusive_max}_constraint` and
/// `{expand,shrink}_only`.
///
/// For each value in `{inclusive_min,exclusive_max}_constraint` that is not
/// `kImplicit`, the corresponding bound of `current_domain` must be equal.
///
/// \param new_inclusive_min The new inclusive min bounds of length
///     `current_domain.rank()`, where a value of `kImplicit` indicates no
///     change.
/// \param new_exclusive_max The new exclusive max bounds of length
///     `current_domain.rank()`, where a value of `kImplicit` indicates no
///     change.
/// \param inclusive_min_constraint The inclusive min constraint vector of
///     length `current_domain.rank()`.
/// \param exclusive_max_constraint The inclusive max constraint vector of
///     length `current_domain.rank()`.
/// \param expand_only If `true`, the bounds must not shrink.
/// \param shrink_only If `true`, the bounds must not expand.
/// \returns `OkStatus()` if the constraints are satisfied.
/// \error `absl::StatusCode::kFailedPrecondition` if the constraints are not
///     satisfied.
/// \dchecks `current_domain.rank() == new_inclusive_min.size()`
/// \dchecks `current_domain.rank() == new_exclusive_max.size()`
/// \dchecks `current_domain.rank() == inclusive_min_constraint.size()`
/// \dchecks `current_domain.rank() == exclusive_max_constraint.size()`
Status ValidateResizeConstraints(BoxView<> current_domain,
                                 span<const Index> new_inclusive_min,
                                 span<const Index> new_exclusive_max,
                                 span<const Index> inclusive_min_constraint,
                                 span<const Index> exclusive_max_constraint,
                                 bool expand_only, bool shrink_only);

/// Specifies how to resize a DataCache.
struct ResizeParameters {
  /// `new_inclusive_min[i]` and `new_exclusive_max[i]` specify the new lower
  /// and upper bounds for dimension `i`, or may be `kImplicit` to indicate
  /// that the existing value should be retained.
  std::vector<Index> new_inclusive_min;
  std::vector<Index> new_exclusive_max;

  /// If `inclusive_min_constraint[i]` or `exclusive_max_constraint[i]` is not
  /// `kImplicit`, the existing lower/upper bound must match it in order for
  /// the resize to succeed.
  std::vector<Index> inclusive_min_constraint;
  std::vector<Index> exclusive_max_constraint;

  /// Fail if any bounds would be reduced.
  bool expand_only;

  /// Fail if any bounds would be increased.
  bool shrink_only;
};

/// Propagates bounds from `new_metadata` for component `component_index` to
/// `transform`.
///
/// \param state The data cache state.
///     `state.GetChunkGridSpecification(new_metadata)`.
/// \param new_metadata Non-null pointer to the new metadata, of the type
///     expected by `state`.
/// \param component_index The component index.
/// \param transform The existing transform.
/// \param options Resolve options.
Result<IndexTransform<>> ResolveBoundsFromMetadata(
    DataCacheState& state, const internal::ChunkGridSpecification& grid,
    const void* new_metadata, std::size_t component_index,
    IndexTransform<> transform, ResolveBoundsOptions options);

/// Validates a resize request for consistency with `transform` and `metadata`.
///
/// \param state The data cache state.
/// \param grid The chunk grid, must equal
///     `state.GetChunkGridSpecification(metadata)`.
/// \param metadata Non-null pointer to the existing metadata, of the type
///     expected by `state`.
/// \param component_index The component index.
/// \param transform The existing transform.
/// \param inclusive_min The new inclusive min bounds for the input domain of
///     `transform`, or `kImplicit` for no change.
/// \param exclusive_min The new exclusive max bounds for the input domain of
///     `transform`, or `kImplicit` for no change.
/// \param options The resize options.
/// \returns The computed resize parameters for the output index space if the
///     resize request is valid.
/// \error `absl::StatusCode::kAborted` if the resize would be a no-op.
/// \error `absl::StatusCode::kFailedPrecondition` if the resize is not
///     compatible with `metadata`.
/// \error `absl::StatusCode::kInvalidArgument` if the resize is invalid
///     irrespective of `metadata`.
/// \remark Even in the case this function returns successfully, the request may
///     fail later due to concurrent modification of the stored metadata.
Result<ResizeParameters> GetResizeParameters(
    DataCacheState& state, const internal::ChunkGridSpecification& grid,
    const void* metadata, size_t component_index,
    IndexTransformView<> transform, span<const Index> inclusive_min,
    span<const Index> exclusive_max, ResizeOptions options);

}  // namespace internal_kvs_backed_chunk_driver
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_KVS_BACKED_CHUNK_DRIVER_IMPL_H_
