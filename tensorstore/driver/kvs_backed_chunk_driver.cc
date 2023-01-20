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

#include "tensorstore/driver/kvs_backed_chunk_driver.h"

#include "absl/container/fixed_array.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "tensorstore/driver/kvs_backed_chunk_driver_impl.h"
#include "tensorstore/internal/box_difference.h"
#include "tensorstore/internal/cache/async_initialized_cache_mixin.h"
#include "tensorstore/internal/cache/cache_pool_resource.h"
#include "tensorstore/internal/cache_key/cache_key.h"
#include "tensorstore/internal/data_copy_concurrency_resource.h"
#include "tensorstore/internal/json_binding/staleness_bound.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/internal/unowned_to_shared.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/util/iterate_over_index_range.h"
#include "tensorstore/util/str_cat.h"

#ifndef TENSORSTORE_KVS_DRIVER_DEBUG
#define TENSORSTORE_KVS_DRIVER_DEBUG 0
#endif

namespace tensorstore {
namespace internal_kvs_backed_chunk_driver {

OpenState::~OpenState() = default;

Result<IndexTransform<>> DataCache::GetExternalToInternalTransform(
    const void* metadata, std::size_t component_index) {
  return IndexTransform<>();
}

OpenState::OpenState(Initializer initializer)
    : PrivateOpenState{std::move(initializer.transaction),
                       std::move(initializer.spec),
                       initializer.read_write_mode} {
  request_time_ = absl::Now();
}

std::string OpenState::GetMetadataCacheKey() { return {}; }

Result<kvstore::DriverPtr> OpenState::GetMetadataKeyValueStore(
    kvstore::DriverPtr base_kv_store) {
  return base_kv_store;
}

Result<kvstore::DriverPtr> OpenState::GetDataKeyValueStore(
    kvstore::DriverPtr base_kv_store, const void* metadata) {
  return base_kv_store;
}

ReadWriteMode OpenState::GetReadWriteMode(const void* metadata) {
  return ReadWriteMode::read_write;
}

AtomicUpdateConstraint OpenState::GetCreateConstraint() {
  return AtomicUpdateConstraint::kRequireMissing;
}

MetadataCache::MetadataCache(Initializer initializer)
    : Base(kvstore::DriverPtr()),
      data_copy_concurrency_(std::move(initializer.data_copy_concurrency)),
      cache_pool_(std::move(initializer.cache_pool)) {}

DataCache::DataCache(Initializer initializer,
                     internal::ChunkGridSpecification grid)
    : Base(std::move(initializer.store), std::move(grid),
           GetOwningCache(*initializer.metadata_cache_entry).executor()),
      metadata_cache_entry_(std::move(initializer.metadata_cache_entry)),
      initial_metadata_(std::move(initializer.metadata)) {}

Result<ChunkLayout> DataCache::GetChunkLayout(const void* metadata_ptr,
                                              std::size_t component_index) {
  return ChunkCache::GetChunkLayout(component_index);
}

Result<CodecSpec> DataCache::GetCodec(const void* metadata,
                                      std::size_t component_index) {
  return CodecSpec{};
}

namespace {

// Address of this variable is used to signal an invalid metadata value.
const char invalid_metadata = 0;

/// Returns an error status indicating that a resize request would implicitly
/// affect a region of dimension `output_dim`, or an out-of-bounds region.
///
/// If `affected_inclusive_min <= affected_exclusive_max`, then the error
/// indicates that the resize would affect the region
/// `[affected_inclusive_min, affected_exclusive_max)`.  If
/// `affected_inclusive_min > affected_exclusive_max`, then the error indicates
/// that the resize request was made with a view containing an out-of-bounds
/// region.
///
/// \param output_dim The output dimension number to be included in the error
///     message.
/// \param affected_inclusive_min Either the inclusive lower bound of the
///     affected region, or the exclusive upper bound of the out-of-bounds
///     region.
/// \param affected_exclusive_max Either the exclusive upper bound of the
///     affected region, or the inclusive lower bound of the out-of-bounds
///     region.
/// \dchecks `affected_inclusive_min != affected_exclusive_max`.
absl::Status ShapeConstraintError(DimensionIndex output_dim,
                                  DimensionIndex affected_inclusive_min,
                                  DimensionIndex affected_exclusive_max) {
  assert(affected_inclusive_min != affected_exclusive_max);
  if (affected_inclusive_min < affected_exclusive_max) {
    return absl::FailedPreconditionError(tensorstore::StrCat(
        "Resize operation would also affect output dimension ", output_dim,
        " over the interval ",
        IndexInterval::UncheckedHalfOpen(affected_inclusive_min,
                                         affected_exclusive_max),
        " but `resize_tied_bounds` was not specified"));
  }
  return absl::FailedPreconditionError(tensorstore::StrCat(
      "Resize operation would also affect output dimension ", output_dim,
      " over the out-of-bounds interval ",
      IndexInterval::UncheckedHalfOpen(affected_exclusive_max,
                                       affected_inclusive_min)));
}

IndexInterval GetNewIndexInterval(IndexInterval existing,
                                  Index new_inclusive_min,
                                  Index new_exclusive_max) {
  return IndexInterval::UncheckedHalfOpen(
      ExplicitIndexOr(new_inclusive_min, existing.inclusive_min()),
      ExplicitIndexOr(new_exclusive_max, existing.exclusive_max()));
}

/// Validates that `current_domain` is compatible with
/// `{inclusive_min,exclusive_max}_constraint`.
///
/// For each value in `{inclusive_min,exclusive_max}_constraint` that is not
/// `kImplicit`, the corresponding bound of `current_domain` must be equal.
///
/// \param current_domain The current bounds.
/// \param inclusive_min_constraint The inclusive min constraint vector of
///     length `current_domain.rank()`.
/// \param exclusive_max_constraint The inclusive max constraint vector of
///     length `current_domain.rank()`.
/// \dchecks `current_domain.rank() == inclusive_min_constraint.size()`
/// \dchecks `current_domain.rank() == exclusive_max_constraint.size()`
/// \return `absl::Status()` if compatible.
/// \error `absl::StatusCode::kFailedPrecondition` if not compatible.
absl::Status ValidateResizeDomainConstraint(
    BoxView<> current_domain, span<const Index> inclusive_min_constraint,
    span<const Index> exclusive_max_constraint) {
  assert(current_domain.rank() == inclusive_min_constraint.size());
  assert(current_domain.rank() == exclusive_max_constraint.size());
  for (DimensionIndex i = 0; i < current_domain.rank(); ++i) {
    const IndexInterval cur_interval = current_domain[i];
    if (!ImplicitOrEqual(inclusive_min_constraint[i],
                         cur_interval.inclusive_min())) {
      return ShapeConstraintError(i, cur_interval.inclusive_min(),
                                  inclusive_min_constraint[i]);
    }
    if (!ImplicitOrEqual(exclusive_max_constraint[i],
                         cur_interval.exclusive_max())) {
      return ShapeConstraintError(i, exclusive_max_constraint[i],
                                  cur_interval.exclusive_max());
    }
  }
  return absl::OkStatus();
}

/// Validates that `new_{inclusive_min,exclusive_max}` differ from
/// `current_domain` only as allowed by the `expand_only` and `shrink_only`
/// constraints.
///
/// \param current_domain The existing domain.
/// \param new_inclusive_min The new inclusive min bounds of length
///     `current_domain.rank()`, where a value of `kImplicit` indicates no
///     change.
/// \param new_exclusive_max The new exclusive max bounds of length
///     `current_domain.rank()`, where a value of `kImplicit` indicates no
///     change.
/// \param expand_only If `true`, the bounds must not shrink.
/// \param shrink_only If `true`, the bounds must not expand.
/// \returns `OkStatus()` if the constraints are satisfied.
/// \error `absl::StatusCode::kFailedPrecondition` if the constraints are not
///     satisfied.
/// \dchecks `current_domain.rank() == new_inclusive_min.size()`
/// \dchecks `current_domain.rank() == new_exclusive_max.size()`
absl::Status ValidateExpandShrinkConstraints(
    BoxView<> current_domain, span<const Index> new_inclusive_min,
    span<const Index> new_exclusive_max, bool expand_only, bool shrink_only) {
  assert(current_domain.rank() == new_inclusive_min.size());
  assert(current_domain.rank() == new_exclusive_max.size());
  for (DimensionIndex i = 0; i < current_domain.rank(); ++i) {
    const IndexInterval cur_interval = current_domain[i];
    const IndexInterval new_interval = GetNewIndexInterval(
        cur_interval, new_inclusive_min[i], new_exclusive_max[i]);
    if (shrink_only && !Contains(cur_interval, new_interval)) {
      return absl::FailedPreconditionError(
          tensorstore::StrCat("Resize operation would expand output dimension ",
                              i, " from ", cur_interval, " to ", new_interval,
                              " but `shrink_only` was specified"));
    }
    if (expand_only && !Contains(new_interval, cur_interval)) {
      return absl::FailedPreconditionError(
          tensorstore::StrCat("Resize operation would shrink output dimension ",
                              i, " from ", cur_interval, " to ", new_interval,
                              " but `expand_only` was specified"));
    }
  }
  return absl::OkStatus();
}

std::string GetMetadataMissingErrorMessage(
    MetadataCache::Entry* metadata_cache_entry) {
  return tensorstore::StrCat(
      "Metadata at ",
      GetOwningCache(*metadata_cache_entry)
          .kvstore_driver()
          ->DescribeKey(metadata_cache_entry->GetKeyValueStoreKey()),
      " does not exist");
}

/// Validates that the parsed metadata in the metadata cache entry associated
/// with `cache` is compatible with the existing metadata from which `cache` was
/// constructed.
///
/// If the metadata has changed in an incompatible way (e.g. a change to the
/// chunk shape), returns an error.
absl::Status ValidateNewMetadata(DataCache* cache, const void* new_metadata) {
  if (!new_metadata) {
    return absl::FailedPreconditionError(
        GetMetadataMissingErrorMessage(cache->metadata_cache_entry_.get()));
  }
  TENSORSTORE_RETURN_IF_ERROR(cache->ValidateMetadataCompatibility(
      cache->initial_metadata_.get(), new_metadata));
  return absl::OkStatus();
}

/// Returns the updated metadata for `cache` in the context of `transaction`.
///
/// If the metadata has changed in an incompatible way, returns an error.
Result<std::shared_ptr<const void>> ValidateNewMetadata(
    DataCache* cache, internal::OpenTransactionPtr transaction) {
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto new_metadata,
      cache->metadata_cache_entry_->GetMetadata(std::move(transaction)));
  TENSORSTORE_RETURN_IF_ERROR(ValidateNewMetadata(cache, new_metadata.get()));
  return new_metadata;
}

Result<IndexTransform<>> GetInitialTransform(DataCache* cache,
                                             const void* metadata,
                                             size_t component_index) {
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto new_transform, cache->GetExternalToInternalTransform(
                              cache->initial_metadata_.get(), component_index));

  return ResolveBoundsFromMetadata(cache, metadata, component_index,
                                   std::move(new_transform),
                                   /*options=*/{});
}

void GetComponentBounds(DataCache* data_cache, const void* metadata,
                        std::size_t component_index, MutableBoxView<> bounds,
                        DimensionSet& implicit_lower_bounds,
                        DimensionSet& implicit_upper_bounds) {
  const auto& grid = data_cache->grid();
  const auto& component_spec = grid.components[component_index];
  assert(bounds.rank() == component_spec.rank());
  Box<dynamic_rank(internal::kNumInlinedDims)> grid_bounds(
      grid.chunk_shape.size());
  DimensionSet grid_implicit_lower_bounds;
  DimensionSet grid_implicit_upper_bounds;
  data_cache->GetChunkGridBounds(metadata, grid_bounds,
                                 grid_implicit_lower_bounds,
                                 grid_implicit_upper_bounds);
  span<const DimensionIndex> chunked_to_cell_dimensions =
      component_spec.chunked_to_cell_dimensions;
  bounds.DeepAssign(component_spec.fill_value.domain());
  implicit_lower_bounds = false;
  implicit_upper_bounds = false;
  for (DimensionIndex grid_dim = 0; grid_dim < grid_bounds.rank(); ++grid_dim) {
    const DimensionIndex cell_dim = chunked_to_cell_dimensions[grid_dim];
    bounds[cell_dim] = grid_bounds[grid_dim];
    implicit_lower_bounds[cell_dim] = grid_implicit_lower_bounds[grid_dim];
    implicit_upper_bounds[cell_dim] = grid_implicit_upper_bounds[grid_dim];
  }
}

}  // namespace

Result<ChunkLayout> DataCache::GetChunkLayout(size_t component_index) {
  return GetChunkLayout(initial_metadata_.get(), component_index);
}

Future<IndexTransform<>> KvsDriverBase::ResolveBounds(
    internal::OpenTransactionPtr transaction, IndexTransform<> transform,
    ResolveBoundsOptions options) {
  return ResolveBounds(std::move(transaction), std::move(transform),
                       metadata_staleness_bound_, options);
}

Future<IndexTransform<>> KvsDriverBase::ResolveBounds(
    internal::OpenTransactionPtr transaction, IndexTransform<> transform,
    StalenessBound metadata_staleness_bound, ResolveBoundsOptions options) {
  auto* cache = this->cache();
  const bool skip_read = assume_metadata_time_ >= metadata_staleness_bound.time;
  const auto handle_assume_metadata =
      [&](auto& entry_or_node) -> Result<IndexTransform<>> {
    // Don't issue a read request, but check if there is already newer cached
    // metadata available.
    std::shared_ptr<const void> new_metadata;
    bool has_cache_entry = false;
    if (MetadataCache::ReadLock<void> lock(entry_or_node);
        lock.stamp().time > assume_metadata_time_) {
      new_metadata = lock.shared_data();
      has_cache_entry = true;
    }
    if (has_cache_entry) {
      if constexpr (std::is_same_v<
                        internal::remove_cvref_t<decltype(entry_or_node)>,
                        MetadataCache::TransactionNode>) {
        TENSORSTORE_ASSIGN_OR_RETURN(
            new_metadata,
            entry_or_node.GetUpdatedMetadata(std::move(new_metadata)),
            cache->metadata_cache_entry_->AnnotateError(_,
                                                        /*reading=*/false));
      }
    }
    // If there is a newer negative cache entry (i.e. an entry that indicates
    // that the metadata is not present), ignore it.  This allows the following
    // use case
    //
    //   1. Open TensorStore with `OpenMode::assume_metadata` specified, and the
    //      metadata does not actually exist.
    //
    //   2. Write data to the TensorStore opened in step 1 (this results in
    //      calls to `ResolveBounds`).
    //
    //   3. Concurrent with step 2, open the same spec using the same cache,
    //      with `OpenMode::create` specified and `OpenMode::assume_metadata`
    //      not specified.  This first checks if the metadata exists, and then
    //      writes it if not present.
    //
    // If negative cache entries were taken into account, then after the initial
    // check in step 3 that finds the metadata missing, a subsequent call to
    // `ResolveBounds` in step 2 will fail.
    if (new_metadata) {
      return ResolveBoundsFromMetadata(cache, new_metadata.get(),
                                       component_index(), std::move(transform),
                                       options);
    } else {
      return ResolveBoundsFromMetadata(cache, cache->initial_metadata_.get(),
                                       component_index(), std::move(transform),
                                       options);
    }
  };
  if (transaction) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto node,
        GetTransactionNode(*cache->metadata_cache_entry_, transaction));
    if (skip_read) return handle_assume_metadata(*node);
    auto read_future = node->Read(metadata_staleness_bound.time);
    return MapFuture(
        cache->executor(),
        [cache = internal::CachePtr<DataCache>(cache), node = std::move(node),
         transform = std::move(transform),
         component_index = this->component_index(),
         options = std::move(options)](
            const Result<void>& result) -> Result<IndexTransform<>> {
          TENSORSTORE_RETURN_IF_ERROR(result);
          TENSORSTORE_ASSIGN_OR_RETURN(
              auto new_metadata, node->GetUpdatedMetadata(),
              cache->metadata_cache_entry_->AnnotateError(_,
                                                          /*reading=*/false));
          TENSORSTORE_RETURN_IF_ERROR(
              ValidateNewMetadata(cache.get(), new_metadata.get()));
          return ResolveBoundsFromMetadata(cache.get(), new_metadata.get(),
                                           component_index,
                                           std::move(transform), options);
        },
        std::move(read_future));
  }
  if (skip_read) return handle_assume_metadata(*cache->metadata_cache_entry_);
  return MapFuture(
      cache->executor(),
      [cache = internal::CachePtr<DataCache>(cache),
       transform = std::move(transform),
       component_index = this->component_index(), options = std::move(options)](
          const Result<void>& result) -> Result<IndexTransform<>> {
        TENSORSTORE_RETURN_IF_ERROR(result);
        auto new_metadata = cache->metadata_cache_entry_->GetMetadata();
        TENSORSTORE_RETURN_IF_ERROR(
            ValidateNewMetadata(cache.get(), new_metadata.get()));
        return ResolveBoundsFromMetadata(cache.get(), new_metadata.get(),
                                         component_index, std::move(transform),
                                         options);
      },
      cache->metadata_cache_entry_->Read(metadata_staleness_bound.time));
}

namespace {

/// Enqueues a request to resize the chunked dimensions of a DataCache.
///
/// \param cache The DataCache to resize.
/// \param transaction The transaction to use.
/// \param parameters Specifies the resize request.
/// \returns A `Future` that becomes ready when the request completes
///     successfully or with an error.  Must call `Force` to ensure the request
///     is actually issued.
Future<const void> RequestResize(DataCache* cache,
                                 internal::OpenTransactionPtr transaction,
                                 ResizeParameters parameters) {
  return cache->metadata_cache_entry_->RequestAtomicUpdate(
      transaction,
      /*update=*/
      [parameters = std::move(parameters),
       cache = internal::CachePtr<DataCache>(cache),
       metadata_constraint = cache->initial_metadata_](
          const MetadataCache::MetadataPtr& current_metadata)
          -> Result<std::shared_ptr<const void>> {
        if (!current_metadata) {
          return absl::NotFoundError("Metadata was deleted");
        }
        TENSORSTORE_RETURN_IF_ERROR(cache->ValidateMetadataCompatibility(
            metadata_constraint.get(), current_metadata.get()));
        Box<dynamic_rank(internal::kNumInlinedDims)> bounds(
            parameters.new_inclusive_min.size());
        DimensionSet implicit_lower_bounds;
        DimensionSet implicit_upper_bounds;
        cache->GetChunkGridBounds(current_metadata.get(), bounds,
                                  implicit_lower_bounds, implicit_upper_bounds);
        // The resize request has already been validated against explicit grid
        // bounds (i.e. bounds corresponding to `false` values in
        // `implicit_{lower,upper}_bounds`), so we don't need to check again
        // here.
        TENSORSTORE_RETURN_IF_ERROR(ValidateResizeConstraints(
            bounds, parameters.new_inclusive_min, parameters.new_exclusive_max,
            parameters.inclusive_min_constraint,
            parameters.exclusive_max_constraint, parameters.expand_only,
            parameters.shrink_only));

        return cache->GetResizedMetadata(current_metadata.get(),
                                         parameters.new_inclusive_min,
                                         parameters.new_exclusive_max);
      },
      AtomicUpdateConstraint::kRequireExisting);
}

struct ResizeContinuation {
  internal::CachePtr<DataCache> cache;
  internal::OpenTransactionPtr transaction;
  std::size_t component_index;
  IndexTransform<> transform;
  Result<IndexTransform<>> GetResult() {
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto new_metadata,
        ValidateNewMetadata(cache.get(), std::move(transaction)));
    return ResolveBoundsFromMetadata(cache.get(), new_metadata.get(),
                                     component_index, std::move(transform),
                                     /*options=*/{});
  }

  void operator()(Promise<IndexTransform<>> promise, ReadyFuture<const void>) {
    promise.SetResult(GetResult());
  }
};

struct ResizeState {
  internal::CachePtr<DataCache> cache;
  internal::OpenTransactionPtr transaction;
  std::size_t component_index;
  IndexTransform<> transform;
  ResizeParameters resize_parameters;
};

void SubmitResizeRequest(Promise<IndexTransform<>> promise, ResizeState state) {
  auto* cache_ptr = state.cache.get();
  LinkValue(
      WithExecutor(cache_ptr->executor(),
                   ResizeContinuation{std::move(state.cache), state.transaction,
                                      state.component_index,
                                      std::move(state.transform)}),
      std::move(promise),
      RequestResize(cache_ptr, state.transaction,
                    std::move(state.resize_parameters)));
}

struct DeleteChunksForResizeContinuation {
  std::unique_ptr<ResizeState> state;
  void operator()(Promise<IndexTransform<>> promise, ReadyFuture<const void>) {
    SubmitResizeRequest(std::move(promise), std::move(*state));
  }
};

Future<const void> DeleteChunksForResize(
    internal::CachePtr<DataCache> cache, BoxView<> current_bounds,
    span<const Index> new_inclusive_min, span<const Index> new_exclusive_max,
    internal::OpenTransactionPtr transaction) {
  span<const Index> chunk_shape = cache->grid().chunk_shape;
  const DimensionIndex rank = chunk_shape.size();
  assert(current_bounds.rank() == rank);
  assert(new_inclusive_min.size() == rank);
  assert(new_exclusive_max.size() == rank);
  auto pair = PromiseFuturePair<void>::Make(MakeResult(absl::Status()));
  pair.future.Force();
  Box<dynamic_rank(internal::kNumInlinedDims)> current_grid_bounds(rank);
  Box<dynamic_rank(internal::kNumInlinedDims)> new_grid_bounds(rank);
  for (DimensionIndex i = 0; i < rank; ++i) {
    const IndexInterval cur_dim_bounds = current_bounds[i];
    const IndexInterval new_dim_bounds = IndexInterval::UncheckedHalfOpen(
        ExplicitIndexOr(new_inclusive_min[i], cur_dim_bounds.inclusive_min()),
        ExplicitIndexOr(new_exclusive_max[i], cur_dim_bounds.exclusive_max()));
    const Index chunk_size = chunk_shape[i];
    current_grid_bounds[i] = DividePositiveRoundOut(cur_dim_bounds, chunk_size);
    new_grid_bounds[i] = DividePositiveRoundOut(new_dim_bounds, chunk_size);
  }
  internal::BoxDifference box_difference(current_grid_bounds, new_grid_bounds);
  Box<dynamic_rank(internal::kNumInlinedDims)> part(rank);
  for (Index box_i = 0; box_i < box_difference.num_sub_boxes(); ++box_i) {
    box_difference.GetSubBox(box_i, part);
    IterateOverIndexRange(part, [&](span<const Index> cell_indices) {
      auto entry = cache->GetEntryForCell(cell_indices);
      LinkError(pair.promise, entry->Delete(transaction));
    });
  }
  return pair.future;
}

struct ResolveBoundsForDeleteAndResizeContinuation {
  std::unique_ptr<ResizeState> state;
  void operator()(Promise<IndexTransform<>> promise, ReadyFuture<const void>) {
    std::shared_ptr<const void> new_metadata;
    if (auto result =
            ValidateNewMetadata(state->cache.get(), state->transaction);
        result.ok()) {
      new_metadata = std::move(*result);
    } else {
      promise.SetResult(std::move(result).status());
      return;
    }
    // Chunks should never be deleted if `expand_only==false`.
    const DimensionIndex grid_rank = state->cache->grid().chunk_shape.size();
    assert(!state->resize_parameters.expand_only);
    Box<dynamic_rank(internal::kNumInlinedDims)> bounds(grid_rank);
    DimensionSet implicit_lower_bounds;
    DimensionSet implicit_upper_bounds;
    state->cache->GetChunkGridBounds(new_metadata.get(), bounds,
                                     implicit_lower_bounds,
                                     implicit_upper_bounds);
    // The resize request has already been validated against explicit grid
    // bounds (i.e. bounds corresponding to `false` values in
    // `implicit_{lower,upper}_bounds`), so we don't need to check again here.
    if (auto status = ValidateResizeConstraints(
            bounds, state->resize_parameters.new_inclusive_min,
            state->resize_parameters.new_exclusive_max,
            state->resize_parameters.inclusive_min_constraint,
            state->resize_parameters.exclusive_max_constraint,
            /*expand_only=*/false,
            /*shrink_only=*/state->resize_parameters.shrink_only);
        !status.ok()) {
      promise.SetResult(std::move(status));
      return;
    }
    auto* state_ptr = state.get();
    LinkValue(
        WithExecutor(state_ptr->cache->executor(),
                     DeleteChunksForResizeContinuation{std::move(state)}),
        std::move(promise),
        DeleteChunksForResize(state_ptr->cache, bounds,
                              state_ptr->resize_parameters.new_inclusive_min,
                              state_ptr->resize_parameters.new_exclusive_max,
                              state_ptr->transaction));
  }
};
}  // namespace

Future<IndexTransform<>> KvsDriverBase::Resize(
    internal::OpenTransactionPtr transaction, IndexTransform<> transform,
    span<const Index> inclusive_min, span<const Index> exclusive_max,
    ResizeOptions options) {
  auto* cache = this->cache();
  auto resize_parameters = GetResizeParameters(
      cache, cache->initial_metadata_.get(), component_index(), transform,
      inclusive_min, exclusive_max, options,
      transaction ? transaction->mode() : TransactionMode::no_transaction_mode);
  if (!resize_parameters) {
    if (resize_parameters.status().code() == absl::StatusCode::kAborted) {
      // Requested resize is a no-op.  Currently there is no resize option
      // corresponding to the `fix_resizable_bounds` resolve option, so we
      // don't specify it.
      return ResolveBounds(std::move(transaction), std::move(transform),
                           /*metadata_staleness_bound=*/{},
                           /*options=*/{});
    }
    return resize_parameters.status();
  }

  auto pair = PromiseFuturePair<IndexTransform<>>::Make();
  ResizeState resize_state{
      /*.cache=*/internal::CachePtr<DataCache>(cache),
      /*.transaction=*/std::move(transaction),
      /*.component_index=*/component_index(),
      /*.transform=*/std::move(transform),
      /*.resize_parameters=*/std::move(*resize_parameters),
  };
  if ((options.mode & resize_metadata_only) == resize_metadata_only ||
      (options.mode & expand_only) == expand_only) {
    // No existing data chunks need to be deleted.  Just update the metadata.
    SubmitResizeRequest(std::move(pair.promise), std::move(resize_state));
  } else {
    // Delete any out-of-bounds data chunks before updating the metadata.
    LinkValue(WithExecutor(
                  cache->executor(),
                  ResolveBoundsForDeleteAndResizeContinuation{
                      std::make_unique<ResizeState>(std::move(resize_state))}),
              std::move(pair.promise),
              cache->metadata_cache_entry_->Read(absl::Now()));
  }
  return std::move(pair.future);
}

Result<IndexTransform<>> KvsDriverBase::GetBoundSpecData(
    internal::OpenTransactionPtr transaction, KvsDriverSpec& spec,
    IndexTransformView<> transform_view) {
  auto* cache = this->cache();
  auto* metadata_cache = cache->metadata_cache();
  TENSORSTORE_ASSIGN_OR_RETURN(spec.store.driver,
                               metadata_cache->base_store()->GetBoundSpec());
  spec.store.path = cache->GetBaseKvstorePath();
  spec.data_copy_concurrency = metadata_cache->data_copy_concurrency_;
  spec.cache_pool = metadata_cache->cache_pool_;
  spec.delete_existing = false;
  spec.open = true;
  spec.create = false;
  spec.staleness.metadata = this->metadata_staleness_bound();
  spec.staleness.data = this->data_staleness_bound();
  spec.schema.Set(RankConstraint{this->rank()}).IgnoreError();
  spec.schema.Set(this->dtype()).IgnoreError();

  TENSORSTORE_ASSIGN_OR_RETURN(
      auto validated_metadata,
      ValidateNewMetadata(cache, std::move(transaction)));
  TENSORSTORE_RETURN_IF_ERROR(cache->GetBoundSpecData(
      spec, validated_metadata.get(), this->component_index()));

  IndexTransform<> transform(transform_view);

  TENSORSTORE_ASSIGN_OR_RETURN(
      auto external_to_internal_transform,
      cache->GetExternalToInternalTransform(validated_metadata.get(),
                                            component_index()));
  if (external_to_internal_transform.valid()) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto internal_to_external_transform,
        InverseTransform(external_to_internal_transform));
    TENSORSTORE_ASSIGN_OR_RETURN(
        transform,
        ComposeTransforms(internal_to_external_transform, transform));
  }

  return transform;
}

absl::Status KvsDriverSpec::ApplyOptions(SpecOptions&& options) {
  if (options.recheck_cached_data.specified()) {
    staleness.data = StalenessBound(options.recheck_cached_data);
  }
  if (options.recheck_cached_metadata.specified()) {
    staleness.metadata = StalenessBound(options.recheck_cached_metadata);
  }
  if (options.kvstore.valid()) {
    if (store.valid()) {
      return absl::InvalidArgumentError("\"kvstore\" is already specified");
    }
    store = std::move(options.kvstore);
  }
  TENSORSTORE_RETURN_IF_ERROR(schema.Set(static_cast<Schema&&>(options)));
  return OpenModeSpec::ApplyOptions(options);
}

Result<CodecSpec> KvsDriverBase::GetCodec() {
  auto* cache = this->cache();
  return cache->GetCodec(cache->initial_metadata_.get(), component_index());
}

kvstore::Spec KvsDriverSpec::GetKvstore() const { return store; }

KvStore KvsDriverBase::GetKvstore() {
  auto* cache = this->cache();
  auto* metadata_cache = cache->metadata_cache();
  return KvStore{kvstore::DriverPtr(metadata_cache->base_store()),
                 cache->GetBaseKvstorePath()};
}

namespace {
/// Validates that the open request specified by `state` can be applied to
/// `metadata`.
Result<std::size_t> ValidateOpenRequest(OpenState* state,
                                        const void* metadata) {
  auto& base = *(PrivateOpenState*)state;  // Cast to private base
  if (!metadata) {
    return absl::NotFoundError(
        GetMetadataMissingErrorMessage(base.metadata_cache_entry_.get()));
  }
  return state->GetComponentIndex(metadata, base.spec_->open_mode());
}

/// \pre `component_index` is the result of a previous call to
///     `state->GetComponentIndex` with the same `metadata`.
/// \pre `metadata != nullptr`
Result<internal::Driver::Handle> CreateTensorStoreFromMetadata(
    OpenState::Ptr state, std::shared_ptr<const void> metadata,
    std::size_t component_index) {
  ABSL_LOG_IF(INFO, TENSORSTORE_KVS_DRIVER_DEBUG)
      << "CreateTensorStoreFromMetadata: state=" << state.get();
  auto& base = *(PrivateOpenState*)state.get();  // Cast to private base
  // TODO(jbms): The read-write mode should be determined based on the kvstore
  // mode, once that is exposed.
  auto read_write_mode = state->GetReadWriteMode(metadata.get());
  if (base.read_write_mode_ != ReadWriteMode::dynamic) {
    TENSORSTORE_RETURN_IF_ERROR(internal::ValidateSupportsModes(
        read_write_mode, base.read_write_mode_));
    read_write_mode = base.read_write_mode_;
  }

  std::string chunk_cache_identifier;
  if (!base.metadata_cache_key_.empty()) {
    auto data_cache_key = state->GetDataCacheKey(metadata.get());
    if (!data_cache_key.empty()) {
      internal::EncodeCacheKey(&chunk_cache_identifier, data_cache_key,
                               base.metadata_cache_key_);
    }
  }
  absl::Status data_key_value_store_status;
  auto chunk_cache =
      (*state->cache_pool())
          ->GetCache<DataCache>(
              chunk_cache_identifier, [&]() -> std::unique_ptr<DataCache> {
                auto store_result = state->GetDataKeyValueStore(
                    GetOwningCache(*base.metadata_cache_entry_).base_store_,
                    metadata.get());
                if (!store_result) {
                  data_key_value_store_status =
                      std::move(store_result).status();
                  return nullptr;
                }
                return state->GetDataCache({std::move(*store_result),
                                            base.metadata_cache_entry_,
                                            metadata});
              });
  TENSORSTORE_RETURN_IF_ERROR(data_key_value_store_status);
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto new_transform,
      GetInitialTransform(chunk_cache.get(), metadata.get(), component_index));

  if (base.transaction_ && !base.spec_->assume_metadata) {
    // Add consistency check.
    chunk_cache->metadata_cache_entry_
        ->RequestAtomicUpdate(
            base.transaction_,
            [cache = chunk_cache, transform = new_transform, component_index](
                const MetadataCache::MetadataPtr& existing_metadata)
                -> Result<MetadataCache::MetadataPtr> {
              TENSORSTORE_RETURN_IF_ERROR(
                  ValidateNewMetadata(cache.get(), existing_metadata.get()));
              TENSORSTORE_ASSIGN_OR_RETURN(
                  auto new_transform,
                  GetInitialTransform(cache.get(), existing_metadata.get(),
                                      component_index));
              if (transform != new_transform) {
                return absl::AbortedError("Metadata is inconsistent");
              }
              return existing_metadata;
            },
            AtomicUpdateConstraint::kRequireExisting)
        .IgnoreFuture();
  }

  internal::DriverPtr driver(
      state->AllocateDriver(
          {std::move(chunk_cache), component_index,
           base.spec_->staleness.BoundAtOpen(base.request_time_)}),
      read_write_mode);
  if (base.spec_->assume_metadata) {
    static_cast<KvsDriverBase&>(*driver).assume_metadata_time_ =
        base.request_time_;
  }
  return internal::Driver::Handle{
      std::move(driver), std::move(new_transform),
      internal::TransactionState::ToTransaction(std::move(base.transaction_))};
}

/// Called when the metadata has been written (successfully or unsuccessfully).
struct HandleWroteMetadata {
  OpenState::Ptr state;
  void operator()(Promise<internal::Driver::Handle> promise,
                  ReadyFuture<const void> future) {
    auto& base = *(PrivateOpenState*)state.get();  // Cast to private base
    auto& result = future.result();
    ABSL_LOG_IF(INFO, TENSORSTORE_KVS_DRIVER_DEBUG)
        << "HandleWroteMetadata: state=" << state.get()
        << ", status=" << result.status();
    if (!result) {
      // Creation of new array metadata failed.
      if (result.status().code() != absl::StatusCode::kAlreadyExists ||
          !base.spec_->open) {
        promise.SetResult(result.status());
        return;
      }
      // Creation of the array failed due to it already existing.  Attempt to
      // open the existing array.
    }
    promise.SetResult([&]() -> Result<internal::Driver::Handle> {
      TENSORSTORE_ASSIGN_OR_RETURN(
          auto metadata,
          base.metadata_cache_entry_->GetMetadata(base.transaction_));
      TENSORSTORE_ASSIGN_OR_RETURN(
          std::size_t component_index,
          ValidateOpenRequest(state.get(), metadata.get()));
      return CreateTensorStoreFromMetadata(
          std::move(state), std::move(metadata), component_index);
    }());
  }
};

/// Attempts to create new array.
void CreateMetadata(OpenState::Ptr state,
                    Promise<internal::Driver::Handle> promise) {
  ABSL_LOG_IF(INFO, TENSORSTORE_KVS_DRIVER_DEBUG)
      << "CreateMetadata: state=" << state.get();
  auto state_ptr = state.get();
  auto& base = *(PrivateOpenState*)state.get();  // Cast to private base
  internal::OpenTransactionPtr transaction = base.transaction_;
  auto state_copy = state;
  Link(WithExecutor(state_ptr->executor(),
                    HandleWroteMetadata{std::move(state)}),
       std::move(promise),
       base.metadata_cache_entry_->RequestAtomicUpdate(
           transaction,
           [state = std::move(state_copy)](
               const MetadataCache::MetadataPtr& existing_metadata)
               -> Result<MetadataCache::MetadataPtr> {
             return state->Create(existing_metadata.get());
           },
           state_ptr->GetCreateConstraint(), base.request_time_));
}

/// Called when the metadata has been read (successfully or not found).
struct HandleReadMetadata {
  OpenState::Ptr state;
  void operator()(Promise<internal::Driver::Handle> promise,
                  ReadyFuture<const void> metadata_future) {
    auto& base = *(PrivateOpenState*)state.get();  // Cast to private base
    std::shared_ptr<const void> metadata;
    if (auto result =
            base.metadata_cache_entry_->GetMetadata(base.transaction_);
        result.ok()) {
      metadata = std::move(*result);
    } else {
      promise.SetResult(std::move(result).status());
      return;
    }
    auto component_index_result =
        ValidateOpenRequest(state.get(), metadata.get());
    if (component_index_result) {
      promise.SetResult(CreateTensorStoreFromMetadata(
          std::move(state), std::move(metadata), *component_index_result));
      return;
    }
    if (component_index_result.status().code() == absl::StatusCode::kNotFound) {
      if (base.spec_->create) {
        CreateMetadata(std::move(state), std::move(promise));
        return;
      }
    }
    promise.SetResult(std::move(component_index_result).status());
  }
};

/// Called when the metadata should be requested or created.
struct GetMetadataForOpen {
  OpenState::Ptr state;
  void operator()(Promise<internal::Driver::Handle> promise) {
    ABSL_LOG_IF(INFO, TENSORSTORE_KVS_DRIVER_DEBUG)
        << "GetMetadataForOpen: state=" << state.get();
    auto& base = *(PrivateOpenState*)state.get();  // Cast to private base
    auto state_ptr = state.get();
    if (base.spec_->open) {
      if (base.spec_->assume_metadata) {
        TENSORSTORE_ASSIGN_OR_RETURN(auto metadata, state->Create(nullptr),
                                     static_cast<void>(promise.SetResult(_)));
        TENSORSTORE_ASSIGN_OR_RETURN(
            std::size_t component_index,
            ValidateOpenRequest(state.get(), metadata.get()),
            static_cast<void>(promise.SetResult(_)));
        promise.SetResult(CreateTensorStoreFromMetadata(
            std::move(state), std::move(metadata), component_index));
        return;
      }
      LinkValue(
          WithExecutor(state_ptr->executor(),
                       HandleReadMetadata{std::move(state)}),
          std::move(promise),
          base.metadata_cache_entry_->Read(
              base.spec_->staleness.metadata.BoundAtOpen(base.request_time_)
                  .time));
      return;
    }
    // `tensorstore::Open` ensures that at least one of `OpenMode::create` and
    // `OpenMode::open` is specified.
    assert(base.spec_->create);
    CreateMetadata(std::move(state), std::move(promise));
  }
};

/// Called when the kvstore has been successfully opened.
struct HandleKeyValueStoreReady {
  OpenState::Ptr state;
  void operator()(Promise<internal::Driver::Handle> promise,
                  ReadyFuture<const void> store) {
    ABSL_LOG_IF(INFO, TENSORSTORE_KVS_DRIVER_DEBUG)
        << "Metadata kvstore ready: state=" << state.get();
    auto& base = *(PrivateOpenState*)state.get();  // Cast to private base
    auto* state_ptr = state.get();
    if (base.spec_->delete_existing) {
      // Delete all keys starting with the key prefix.
      KeyRange range_to_delete =
          KeyRange::Prefix(state->GetPrefixForDeleteExisting());
      auto* kvstore =
          GetOwningCache(*base.metadata_cache_entry_).base_store_.get();
      if (!base.transaction_) {
        LinkValue(std::bind(WithExecutor(state_ptr->executor(),
                                         GetMetadataForOpen{std::move(state)}),
                            std::placeholders::_1),
                  std::move(promise),
                  kvstore->DeleteRange(std::move(range_to_delete)));
        return;
      }
      if (auto status = kvstore->TransactionalDeleteRange(
              base.transaction_, std::move(range_to_delete));
          !status.ok()) {
        promise.SetResult(status);
        return;
      }
      base.transaction_->Barrier();
    }
    // Immediately proceed with reading/creating the metadata.
    GetMetadataForOpen{std::move(state)}(std::move(promise));
  }
};

}  // namespace

Future<const void> MetadataCache::Entry::RequestAtomicUpdate(
    const internal::OpenTransactionPtr& transaction, UpdateFunction update,
    AtomicUpdateConstraint update_constraint,
    std::optional<absl::Time> read_time) {
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto node, GetWriteLockedTransactionNode(*this, transaction));
  node->updated_metadata_base_state_ =
      internal::UnownedToShared(&invalid_metadata);
  node->updated_metadata_ = nullptr;
  if (node->transaction()->implicit_transaction()) {
    auto [promise, future] = PromiseFuturePair<void>::Make();
    node->AddPendingWrite(
        PendingWrite{std::move(update), update_constraint, promise});
    LinkError(std::move(promise), node.unlock()->transaction()->future());
    return std::move(future);
  }
  node->AddPendingWrite(PendingWrite{std::move(update), update_constraint});
  if (read_time) {
    return node->Read(*read_time);
  }
  return MakeReadyFuture();
}

Result<MetadataCache::MetadataPtr> MetadataCache::Entry::GetMetadata(
    internal::OpenTransactionPtr transaction) {
  if (!transaction) return GetMetadata();
  TENSORSTORE_ASSIGN_OR_RETURN(auto node,
                               GetTransactionNode(*this, transaction));
  TENSORSTORE_ASSIGN_OR_RETURN(auto metadata, node->GetUpdatedMetadata(),
                               this->AnnotateError(_, /*reading=*/false));
  return metadata;
}

Result<MetadataCache::MetadataPtr>
MetadataCache::TransactionNode::GetUpdatedMetadata(MetadataPtr metadata) {
  UniqueWriterLock lock(*this);
  if (this->updated_metadata_base_state_ == metadata) {
    return this->updated_metadata_;
  }
  this->updated_metadata_base_state_ = metadata;
  for (const auto& request : this->pending_writes) {
    auto result = request.update(metadata);
    if (result) {
      assert(*result);
      assert(request.update_constraint !=
                 AtomicUpdateConstraint::kRequireMissing ||
             metadata == nullptr);
      assert(request.update_constraint !=
                 AtomicUpdateConstraint::kRequireExisting ||
             metadata != nullptr);
      metadata = std::move(*result);
      if (!request.promise.null()) {
        request.promise.raw_result() = MakeResult();
      }
    } else {
      if (!request.promise.null()) {
        request.promise.raw_result() = GetOwningEntry(*this).AnnotateError(
            result.status(), /*reading=*/false);
      } else {
        this->updated_metadata_ = result.status();
        return std::move(result).status();
      }
    }
  }
  this->updated_metadata_ = metadata;
  return metadata;
}

Result<MetadataCache::MetadataPtr>
MetadataCache::TransactionNode::GetUpdatedMetadata() {
  auto metadata = ReadLock<void>(*this).shared_data();
  return GetUpdatedMetadata(std::move(metadata));
}

void MetadataCache::Entry::DoDecode(std::optional<absl::Cord> value,
                                    DecodeReceiver receiver) {
  GetOwningCache(*this).executor()([this, value = std::move(value),
                                    receiver = std::move(receiver)]() mutable {
    MetadataPtr new_metadata;
    if (value) {
      if (auto result =
              GetOwningCache(*this).DecodeMetadata(this->key(), *value);
          result.ok()) {
        new_metadata = std::move(*result);
      } else {
        execution::set_error(
            receiver, internal::ConvertInvalidArgumentToFailedPrecondition(
                          std::move(result).status()));
        return;
      }
    }
    execution::set_value(receiver, std::move(new_metadata));
  });
}

std::string MetadataCache::Entry::GetKeyValueStoreKey() {
  return GetOwningCache(*this).GetMetadataStorageKey(this->key());
}

void MetadataCache::TransactionNode::DoApply(ApplyOptions options,
                                             ApplyReceiver receiver) {
  auto continuation = [this, receiver = std::move(receiver)](
                          ReadyFuture<const void> future) mutable {
    if (!future.result().ok()) {
      return execution::set_error(receiver, future.result().status());
    }
    ABSL_LOG_IF(INFO, TENSORSTORE_ASYNC_CACHE_DEBUG)
        << *this << "Apply metadata";
    auto read_state = AsyncCache::ReadLock<void>(*this).read_state();
    std::shared_ptr<const void> new_data;
    if (auto result = this->GetUpdatedMetadata(read_state.data); result.ok()) {
      new_data = std::move(*result);
    } else {
      execution::set_error(receiver, std::move(result).status());
      return;
    }
    if (new_data != read_state.data) {
      read_state.stamp.generation.MarkDirty();
      read_state.data = std::move(new_data);
    }
    execution::set_value(receiver, std::move(read_state));
  };
  this->Read(options.staleness_bound)
      .ExecuteWhenReady(WithExecutor(GetOwningCache(*this).executor(),
                                     std::move(continuation)));
}

void MetadataCache::TransactionNode::InvalidateReadState() {
  Base::TransactionNode::InvalidateReadState();
  this->updated_metadata_base_state_ =
      internal::UnownedToShared(&invalid_metadata);
  this->updated_metadata_ = nullptr;
}

void MetadataCache::Entry::DoEncode(std::shared_ptr<const void> data,
                                    EncodeReceiver receiver) {
  ABSL_LOG_IF(INFO, TENSORSTORE_ASYNC_CACHE_DEBUG)
      << *this << "Encoding metadata";
  auto& entry = GetOwningEntry(*this);
  auto& cache = GetOwningCache(entry);
  if (auto encoded_result = cache.EncodeMetadata(entry.key(), data.get());
      encoded_result.ok()) {
    execution::set_value(receiver, std::move(*encoded_result));
  } else {
    execution::set_error(receiver, std::move(encoded_result).status());
  }
}

std::string DataCache::Entry::GetKeyValueStoreKey() {
  auto& cache = GetOwningCache(*this);
  return cache.GetChunkStorageKey(cache.initial_metadata_.get(),
                                  this->cell_indices());
}

void DataCache::Entry::DoDecode(std::optional<absl::Cord> value,
                                DecodeReceiver receiver) {
  GetOwningCache(*this).executor()([this, value = std::move(value),
                                    receiver = std::move(receiver)]() mutable {
    if (!value) {
      execution::set_value(receiver, nullptr);
      return;
    }
    auto& cache = GetOwningCache(*this);
    auto decoded_result = cache.DecodeChunk(
        cache.initial_metadata_.get(), this->cell_indices(), std::move(*value));
    if (!decoded_result.ok()) {
      execution::set_error(receiver,
                           internal::ConvertInvalidArgumentToFailedPrecondition(
                               std::move(decoded_result).status()));
      return;
    }
    const size_t num_components = this->component_specs().size();
    auto new_read_data =
        internal::make_shared_for_overwrite<ReadData[]>(num_components);
    assert(decoded_result->size() == num_components);
    std::copy_n(decoded_result->begin(), num_components, new_read_data.get());
    execution::set_value(
        receiver, std::static_pointer_cast<ReadData>(std::move(new_read_data)));
  });
}

void DataCache::Entry::DoEncode(std::shared_ptr<const ReadData> data,
                                EncodeReceiver receiver) {
  if (!data) {
    execution::set_value(receiver, std::nullopt);
    return;
  }
  auto& entry = GetOwningEntry(*this);
  auto& cache = GetOwningCache(entry);
  // Convert from array of `SharedArrayView<const void>` to array of
  // `ArrayView<const void>`.
  auto* components = data.get();
  const auto component_specs = this->component_specs();
  absl::FixedArray<SharedArrayView<const void>, 2> component_arrays(
      component_specs.size());
  for (size_t i = 0; i < component_arrays.size(); ++i) {
    if (components[i].valid()) {
      component_arrays[i] = components[i];
    } else {
      component_arrays[i] = component_specs[i].fill_value;
    }
  }
  auto encoded_result = cache.EncodeChunk(
      cache.initial_metadata_.get(), entry.cell_indices(), component_arrays);
  if (!encoded_result.ok()) {
    execution::set_error(receiver, std::move(encoded_result).status());
    return;
  }
  execution::set_value(receiver, std::move(*encoded_result));
}

namespace {
/// Returns the metadata cache for `state`, creating it if it doesn't already
/// exist.
///
/// The key used to lookup the cache depends on the `kvstore::DriverSpec`; the
/// actual `kvstore::Driver` has not yet been opened.
///
/// The returned `metadata_cache` must not be used for read or write operations
/// until the `metadata_cache->initialized_` future becomes ready.  This
/// asynchronous initialization pattern is needed in order to asynchronously
/// open the `kvstore::Driver` when the metadata cache is created.
internal::CachePtr<MetadataCache> GetOrCreateMetadataCache(OpenState* state) {
  auto& base = *(PrivateOpenState*)state;  // Cast to private base

  auto& spec = *base.spec_;
  internal::EncodeCacheKey(&base.metadata_cache_key_, spec.store.driver,
                           typeid(*state), state->GetMetadataCacheKey());
  return internal::GetOrCreateAsyncInitializedCache<MetadataCache>(
      **state->cache_pool(), base.metadata_cache_key_,
      [&] {
        ABSL_LOG_IF(INFO, TENSORSTORE_KVS_DRIVER_DEBUG)
            << "Creating metadata cache: open_state=" << state;
        return state->GetMetadataCache(
            {base.spec_->data_copy_concurrency, base.spec_->cache_pool});
      },
      [&](Promise<void> initialized,
          internal::CachePtr<MetadataCache> metadata_cache) {
        ABSL_LOG_IF(INFO, TENSORSTORE_KVS_DRIVER_DEBUG)
            << "Opening metadata kvstore: open_state=" << state;
        // The cache didn't previously exist.  Open the kvstore.
        LinkValue(
            [state = OpenState::Ptr(state),
             metadata_cache = std::move(metadata_cache)](
                Promise<void> metadata_cache_promise,
                ReadyFuture<kvstore::DriverPtr> future) {
              metadata_cache->base_store_ = *future.result();
              if (auto result = state->GetMetadataKeyValueStore(
                      metadata_cache->base_store_);
                  result.ok()) {
                metadata_cache->SetKvStoreDriver(std::move(*result));
              } else {
                metadata_cache_promise.SetResult(std::move(result).status());
              }
            },
            initialized, kvstore::Open(spec.store.driver));
      });
}
}  // namespace

Future<internal::Driver::Handle> OpenDriver(OpenState::Ptr state) {
  ABSL_LOG_IF(INFO, TENSORSTORE_KVS_DRIVER_DEBUG)
      << "OpenDriver: open_state=" << state.get();
  // TODO(jbms): possibly determine these options from the open options.
  auto& base = *(PrivateOpenState*)state.get();  // Cast to private base
  auto& spec = *base.spec_;
  TENSORSTORE_RETURN_IF_ERROR(
      spec.OpenModeSpec::Validate(base.read_write_mode_));
  if (!spec.store.valid()) {
    return absl::InvalidArgumentError("\"kvstore\" must be specified");
  }
  auto* state_ptr = state.get();
  auto metadata_cache = GetOrCreateMetadataCache(state_ptr);
  base.metadata_cache_entry_ =
      GetCacheEntry(metadata_cache, state->GetMetadataCacheEntryKey());
  return PromiseFuturePair<internal::Driver::Handle>::LinkValue(
             HandleKeyValueStoreReady{std::move(state)},
             metadata_cache->initialized_)
      .future;
}

Result<IndexTransform<>> ResolveBoundsFromMetadata(
    DataCache* data_cache, const void* new_metadata,
    std::size_t component_index, IndexTransform<> transform,
    ResolveBoundsOptions options) {
  auto& grid = data_cache->grid();
  const DimensionIndex base_rank = grid.components[component_index].rank();
  DimensionSet base_implicit_lower_bounds;
  DimensionSet base_implicit_upper_bounds;
  Box<dynamic_rank(internal::kNumInlinedDims)> base_bounds(base_rank);
  GetComponentBounds(data_cache, new_metadata, component_index, base_bounds,
                     base_implicit_lower_bounds, base_implicit_upper_bounds);
  if ((options.mode & fix_resizable_bounds) == fix_resizable_bounds) {
    base_implicit_lower_bounds = false;
    base_implicit_upper_bounds = false;
  }
  return PropagateBoundsToTransform(
      BoxView<>(base_bounds), base_implicit_lower_bounds,
      base_implicit_upper_bounds, std::move(transform));
}

absl::Status ValidateResizeConstraints(
    BoxView<> current_domain, span<const Index> new_inclusive_min,
    span<const Index> new_exclusive_max,
    span<const Index> inclusive_min_constraint,
    span<const Index> exclusive_max_constraint, bool expand_only,
    bool shrink_only) {
  TENSORSTORE_RETURN_IF_ERROR(ValidateResizeDomainConstraint(
      current_domain, inclusive_min_constraint, exclusive_max_constraint));
  TENSORSTORE_RETURN_IF_ERROR(ValidateExpandShrinkConstraints(
      current_domain, new_inclusive_min, new_exclusive_max, expand_only,
      shrink_only));
  return absl::OkStatus();
}

Result<ResizeParameters> GetResizeParameters(
    DataCache* data_cache, const void* metadata, size_t component_index,
    IndexTransformView<> transform, span<const Index> inclusive_min,
    span<const Index> exclusive_max, ResizeOptions options,
    TransactionMode transaction_mode) {
  assert(transform.input_rank() == inclusive_min.size());
  assert(transform.input_rank() == exclusive_max.size());
  const DimensionIndex output_rank = transform.output_rank();

  const auto& grid = data_cache->grid();
  const DimensionIndex base_rank = grid.components[component_index].rank();
  DimensionSet base_implicit_lower_bounds;
  DimensionSet base_implicit_upper_bounds;
  Box<dynamic_rank(internal::kNumInlinedDims)> base_bounds(base_rank);
  GetComponentBounds(data_cache, metadata, component_index, base_bounds,
                     base_implicit_lower_bounds, base_implicit_upper_bounds);

  const DimensionIndex grid_rank = grid.grid_rank();

  using FixedIndexVec = absl::FixedArray<Index, internal::kNumInlinedDims>;

  FixedIndexVec new_output_inclusive_min(output_rank);
  FixedIndexVec new_output_exclusive_max(output_rank);
  FixedIndexVec output_inclusive_min_constraint(output_rank);
  FixedIndexVec output_exclusive_max_constraint(output_rank);

  bool is_noop;
  TENSORSTORE_RETURN_IF_ERROR(PropagateInputDomainResizeToOutput(
      transform, inclusive_min, exclusive_max,
      /*can_resize_tied_bounds=*/(options.mode & resize_tied_bounds) ==
          resize_tied_bounds,
      output_inclusive_min_constraint, output_exclusive_max_constraint,
      new_output_inclusive_min, new_output_exclusive_max, &is_noop));

  if (is_noop) return absl::AbortedError("");

  if (grid.components.size() != 1 && !(options.mode & resize_tied_bounds)) {
    return absl::FailedPreconditionError(
        "Resize operation would affect other fields but "
        "`resize_tied_bounds` was not specified");
  }

  // Validate that new bounds and constraints are compatible with non-resizable
  // bounds.
  for (DimensionIndex output_dim = 0; output_dim < output_rank; ++output_dim) {
    const IndexInterval dim_bounds = base_bounds[output_dim];
    if (!base_implicit_lower_bounds[output_dim]) {
      const Index min_constraint = output_inclusive_min_constraint[output_dim];
      if (!ImplicitOrEqual(min_constraint, dim_bounds.inclusive_min())) {
        return ShapeConstraintError(output_dim, dim_bounds.inclusive_min(),
                                    min_constraint);
      }
      const Index new_inclusive_min = new_output_inclusive_min[output_dim];
      if (!ImplicitOrEqual(new_inclusive_min, dim_bounds.inclusive_min())) {
        return absl::FailedPreconditionError(tensorstore::StrCat(
            "Cannot change inclusive lower bound of output dimension ",
            output_dim, ", which is fixed at ", dim_bounds.inclusive_min(),
            ", to ", new_inclusive_min));
      }
    }
    if (!base_implicit_upper_bounds[output_dim]) {
      const Index max_constraint = output_exclusive_max_constraint[output_dim];
      if (!ImplicitOrEqual(max_constraint, dim_bounds.exclusive_max())) {
        return ShapeConstraintError(output_dim, max_constraint,
                                    dim_bounds.exclusive_max());
      }
      const Index new_exclusive_max = new_output_exclusive_max[output_dim];
      if (!ImplicitOrEqual(new_exclusive_max, dim_bounds.exclusive_max())) {
        return absl::FailedPreconditionError(tensorstore::StrCat(
            "Cannot change exclusive upper bound of output dimension ",
            output_dim, ", which is fixed at ", dim_bounds.exclusive_max(),
            ", to ", new_exclusive_max));
      }
    }
    if (transaction_mode == TransactionMode::atomic_isolated &&
        !(options.mode & resize_metadata_only) &&
        !(options.mode & expand_only)) {
      // Since chunks will be deleted, there must not be another concurrent
      // resize to ensure consistency.
      output_inclusive_min_constraint[output_dim] = dim_bounds.inclusive_min();
      output_exclusive_max_constraint[output_dim] = dim_bounds.exclusive_max();
    }
  }

  // Convert resize request on component dimensions to chunk dimensions.
  span<const DimensionIndex> chunked_to_cell_dimensions =
      grid.components[component_index].chunked_to_cell_dimensions;

  std::vector<Index> new_grid_inclusive_min(grid_rank);
  std::vector<Index> new_grid_exclusive_max(grid_rank);
  std::vector<Index> grid_inclusive_min_constraint(grid_rank);
  std::vector<Index> grid_exclusive_max_constraint(grid_rank);

  for (DimensionIndex i = 0; i < grid_rank; ++i) {
    const DimensionIndex j = chunked_to_cell_dimensions[i];
    new_grid_inclusive_min[i] = new_output_inclusive_min[j];
    new_grid_exclusive_max[i] = new_output_exclusive_max[j];
    grid_inclusive_min_constraint[i] = output_inclusive_min_constraint[j];
    grid_exclusive_max_constraint[i] = output_exclusive_max_constraint[j];
  }

  return ResizeParameters{
      /*.new_inclusive_min=*/new_grid_inclusive_min,
      /*.new_exclusive_max=*/new_grid_exclusive_max,
      /*.inclusive_min_constraint=*/grid_inclusive_min_constraint,
      /*.exclusive_max_constraint=*/grid_exclusive_max_constraint,
      /*.expand_only=*/(options.mode & expand_only) == expand_only,
      /*.shrink_only=*/(options.mode & shrink_only) == shrink_only};
}

KvsDriverBase::KvsDriverBase(Initializer&& initializer)
    : internal::ChunkCacheDriver(std::move(initializer.cache),
                                 initializer.component_index,
                                 initializer.staleness_bounds.data),
      metadata_staleness_bound_(initializer.staleness_bounds.metadata) {}

DataCache* KvsDriverBase::cache() const {
  return static_cast<DataCache*>(internal::ChunkCacheDriver::cache());
}

void KvsDriverBase::GarbageCollectionBase::Visit(
    garbage_collection::GarbageCollectionVisitor& visitor,
    const KvsDriverBase& value) {
  auto* cache = value.cache();
  auto* metadata_cache = cache->metadata_cache();
  garbage_collection::GarbageCollectionVisit(visitor,
                                             *metadata_cache->base_store());
}

namespace jb = tensorstore::internal_json_binding;
TENSORSTORE_DEFINE_JSON_BINDER(
    SpecJsonBinder,
    jb::Sequence(
        jb::Member(internal::DataCopyConcurrencyResource::id,
                   jb::Projection<&KvsDriverSpec::data_copy_concurrency>()),
        jb::Member(internal::CachePoolResource::id,
                   jb::Projection<&KvsDriverSpec::cache_pool>()),
        jb::Projection<&KvsDriverSpec::store>(jb::KvStoreSpecAndPathJsonBinder),
        jb::Initialize([](auto* obj) {
          internal::EnsureDirectoryPath(obj->store.path);
          return absl::OkStatus();
        }),
        jb::Projection<&KvsDriverSpec::staleness>(jb::Sequence(
            jb::Member("recheck_cached_metadata",
                       jb::Projection(&StalenessBounds::metadata,
                                      jb::DefaultValue([](auto* obj) {
                                        obj->bounded_by_open_time = true;
                                      }))),
            jb::Member("recheck_cached_data",
                       jb::Projection(&StalenessBounds::data,
                                      jb::DefaultInitializedValue())))),
        internal::OpenModeSpecJsonBinder));

}  // namespace internal_kvs_backed_chunk_driver
}  // namespace tensorstore
