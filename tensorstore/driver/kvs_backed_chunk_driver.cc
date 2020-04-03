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

#include "tensorstore/driver/kvs_backed_chunk_driver_impl.h"
#include "tensorstore/internal/box_difference.h"
#include "tensorstore/internal/cache_key.h"
#include "tensorstore/internal/cache_pool_resource.h"
#include "tensorstore/internal/data_copy_concurrency_resource.h"
#include "tensorstore/util/bit_vec.h"
#include "tensorstore/util/iterate_over_index_range.h"
#include "tensorstore/util/quote_string.h"

namespace tensorstore {
namespace internal_kvs_backed_chunk_driver {

OpenState::~OpenState() = default;
DataCacheState::~DataCacheState() = default;
MetadataCacheState::~MetadataCacheState() = default;

Result<IndexTransform<>> DataCacheState::GetExternalToInternalTransform(
    const void* metadata, std::size_t component_index) {
  return IndexTransform<>();
}

OpenState::OpenState(Initializer initializer)
    : PrivateOpenState{std::move(initializer.spec),
                       initializer.read_write_mode} {
  request_time_ = absl::Now();
}

std::string OpenState::GetMetadataCacheKey() { return {}; }

Result<KeyValueStore::Ptr> OpenState::GetMetadataKeyValueStore(
    KeyValueStore::Ptr base_kv_store) {
  return base_kv_store;
}

Result<KeyValueStore::Ptr> OpenState::GetDataKeyValueStore(
    KeyValueStore::Ptr base_kv_store, const void* metadata) {
  return base_kv_store;
}

ReadWriteMode OpenState::GetReadWriteMode(const void* metadata) {
  return ReadWriteMode::read_write;
}

AtomicUpdateConstraint OpenState::GetCreateConstraint() {
  return AtomicUpdateConstraint::kRequireMissing;
}

MetadataCache::MetadataCache(
    MetadataCacheState::Ptr state, KeyValueStore::Ptr base_store,
    KeyValueStore::Ptr store,
    Context::Resource<internal::DataCopyConcurrencyResource>
        data_copy_concurrency,
    Context::Resource<internal::CachePoolResource> cache_pool)
    : state_(std::move(state)),
      base_store_(std::move(base_store)),
      store_(std::move(store)),
      data_copy_concurrency_(std::move(data_copy_concurrency)),
      cache_pool_(std::move(cache_pool)) {}

DataCache::DataCache(
    DataCacheState::Ptr state, KeyValueStore::Ptr store,
    internal::PinnedCacheEntry<MetadataCache> metadata_cache_entry,
    MetadataPtr metadata)
    : ChunkCache(state->GetChunkGridSpecification(metadata.get())),
      state_(std::move(state)),
      store_(std::move(store)),
      metadata_cache_entry_(std::move(metadata_cache_entry)),
      initial_metadata_(metadata),
      validated_metadata_(metadata) {}

namespace {

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
Status ShapeConstraintError(DimensionIndex output_dim,
                            DimensionIndex affected_inclusive_min,
                            DimensionIndex affected_exclusive_max) {
  assert(affected_inclusive_min != affected_exclusive_max);
  if (affected_inclusive_min < affected_exclusive_max) {
    return absl::FailedPreconditionError(
        StrCat("Resize operation would also affect output dimension ",
               output_dim, " over the interval ",
               IndexInterval::UncheckedHalfOpen(affected_inclusive_min,
                                                affected_exclusive_max),
               " but `resize_tied_bounds` was not specified"));
  }
  return absl::FailedPreconditionError(
      StrCat("Resize operation would also affect output dimension ", output_dim,
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
/// \return `Status()` if compatible.
/// \error `absl::StatusCode::kFailedPrecondition` if not compatible.
Status ValidateResizeDomainConstraint(
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
Status ValidateExpandShrinkConstraints(BoxView<> current_domain,
                                       span<const Index> new_inclusive_min,
                                       span<const Index> new_exclusive_max,
                                       bool expand_only, bool shrink_only) {
  assert(current_domain.rank() == new_inclusive_min.size());
  assert(current_domain.rank() == new_exclusive_max.size());
  for (DimensionIndex i = 0; i < current_domain.rank(); ++i) {
    const IndexInterval cur_interval = current_domain[i];
    const IndexInterval new_interval = GetNewIndexInterval(
        cur_interval, new_inclusive_min[i], new_exclusive_max[i]);
    if (shrink_only && !Contains(cur_interval, new_interval)) {
      return absl::FailedPreconditionError(
          StrCat("Resize operation would expand output dimension ", i, " from ",
                 cur_interval, " to ", new_interval,
                 " but `shrink_only` was specified"));
    }
    if (expand_only && !Contains(new_interval, cur_interval)) {
      return absl::FailedPreconditionError(
          StrCat("Resize operation would shrink output dimension ", i, " from ",
                 cur_interval, " to ", new_interval,
                 " but `expand_only` was specified"));
    }
  }
  return absl::OkStatus();
}

/// Validates that the parsed metadata in the metadata cache entry associated
/// with `cache` is compatible with the existing metadata from which `cache` was
/// constructed.
///
/// If the metadata has changed in an incompatible way (e.g. a change to the
/// chunk shape), returns an error.  Otherwise, sets
/// `cache->validated_metadata_` to the new parsed metadata.
Result<std::shared_ptr<const void>> ValidateNewMetadata(DataCache* cache) {
  auto new_metadata = cache->metadata_cache_entry_->GetMetadata();
  absl::MutexLock lock(&cache->mutex_);
  TENSORSTORE_RETURN_IF_ERROR(cache->state_->ValidateMetadataCompatibility(
      cache->validated_metadata_.get(), new_metadata.get()));
  cache->validated_metadata_ = new_metadata;
  return new_metadata;
}

void GetComponentBounds(DataCacheState& state,
                        const internal::ChunkGridSpecification& grid,
                        const void* metadata, std::size_t component_index,
                        MutableBoxView<> bounds,
                        BitSpan<std::uint64_t> implicit_lower_bounds,
                        BitSpan<std::uint64_t> implicit_upper_bounds) {
  const auto& component_spec = grid.components[component_index];
  assert(bounds.rank() == component_spec.rank());
  assert(implicit_lower_bounds.size() == bounds.rank());
  assert(implicit_upper_bounds.size() == bounds.rank());
  Box<dynamic_rank(internal::kNumInlinedDims)> grid_bounds(
      grid.chunk_shape.size());
  BitVec<> grid_implicit_lower_bounds(grid_bounds.rank());
  BitVec<> grid_implicit_upper_bounds(grid_bounds.rank());
  state.GetChunkGridBounds(metadata, grid_bounds, grid_implicit_lower_bounds,
                           grid_implicit_upper_bounds);
  span<const DimensionIndex> chunked_to_cell_dimensions =
      component_spec.chunked_to_cell_dimensions;
  bounds.DeepAssign(component_spec.fill_value.domain());
  implicit_lower_bounds.fill(false);
  implicit_upper_bounds.fill(false);
  for (DimensionIndex grid_dim = 0; grid_dim < grid_bounds.rank(); ++grid_dim) {
    const DimensionIndex cell_dim = chunked_to_cell_dimensions[grid_dim];
    bounds[cell_dim] = grid_bounds[grid_dim];
    implicit_lower_bounds[cell_dim] = grid_implicit_lower_bounds[grid_dim];
    implicit_upper_bounds[cell_dim] = grid_implicit_upper_bounds[grid_dim];
  }
}

Result<IndexTransform<>> ResolveBoundsFromMetadata(
    DataCache* cache, const void* new_metadata, std::size_t component_index,
    IndexTransform<> transform, ResolveBoundsOptions options) {
  return ResolveBoundsFromMetadata(*cache->state_, cache->grid(), new_metadata,
                                   component_index, std::move(transform),
                                   options);
}

struct ResolveBoundsContinuation {
  internal::CachePtr<DataCache> cache;
  IndexTransform<> transform;
  std::size_t component_index;
  ResolveBoundsOptions options;
  Result<IndexTransform<>> operator()(const Result<void>& result) {
    TENSORSTORE_RETURN_IF_ERROR(result);
    TENSORSTORE_ASSIGN_OR_RETURN(auto new_metadata,
                                 ValidateNewMetadata(cache.get()));
    return ResolveBoundsFromMetadata(cache.get(), new_metadata.get(),
                                     component_index, std::move(transform),
                                     options);
  }
};

}  // namespace

Future<IndexTransform<>> DriverBase::ResolveBounds(
    IndexTransform<> transform, ResolveBoundsOptions options) {
  return ResolveBounds(transform, metadata_staleness_bound_, options);
}

Future<IndexTransform<>> DriverBase::ResolveBounds(
    IndexTransform<> transform, StalenessBound metadata_staleness_bound,
    ResolveBoundsOptions options) {
  auto* cache = this->cache();

  return MapFuture(
      cache->executor(),
      ResolveBoundsContinuation{internal::CachePtr<DataCache>(cache),
                                std::move(transform), component_index(),
                                options},
      cache->metadata_cache_entry_->Read(metadata_staleness_bound));
}

namespace {

/// Enqueues a request to resize the chunked dimensions of a DataCache.
///
/// \param cache The DataCache to resize.
/// \param parameters Specifies the resize request.
/// \param request_time Time at which the request was initiated (affects
///     retrying in the case of concurrent modifications).
/// \returns A `Future` that becomes ready when the request completes
///     successfully or with an error.  Must call `Force` to ensure the request
///     is actually issued.
Future<const void> RequestResize(DataCache* cache, ResizeParameters parameters,
                                 absl::Time request_time) {
  return cache->metadata_cache_entry_->RequestAtomicUpdate(
      /*update=*/
      [parameters = std::move(parameters), data_cache_state = cache->state_,
       metadata_constraint = cache->initial_metadata_](
          const void* current_metadata) -> Result<std::shared_ptr<const void>> {
        if (!current_metadata) {
          return absl::NotFoundError("Metadata was deleted");
        }
        TENSORSTORE_RETURN_IF_ERROR(
            data_cache_state->ValidateMetadataCompatibility(
                metadata_constraint.get(), current_metadata));
        Box<dynamic_rank(internal::kNumInlinedDims)> bounds(
            parameters.new_inclusive_min.size());
        BitVec<> implicit_lower_bounds(bounds.rank());
        BitVec<> implicit_upper_bounds(bounds.rank());
        data_cache_state->GetChunkGridBounds(current_metadata, bounds,
                                             implicit_lower_bounds,
                                             implicit_upper_bounds);
        // The resize request has already been validated against explicit grid
        // bounds (i.e. bounds corresponding to `false` values in
        // `implicit_{lower,upper}_bounds`), so we don't need to check again
        // here.
        TENSORSTORE_RETURN_IF_ERROR(ValidateResizeConstraints(
            bounds, parameters.new_inclusive_min, parameters.new_exclusive_max,
            parameters.inclusive_min_constraint,
            parameters.exclusive_max_constraint, parameters.expand_only,
            parameters.shrink_only));

        return data_cache_state->GetResizedMetadata(
            current_metadata, parameters.new_inclusive_min,
            parameters.new_exclusive_max);
      },
      AtomicUpdateConstraint::kRequireExisting, request_time);
}

struct ResizeContinuation {
  internal::CachePtr<DataCache> cache;
  std::size_t component_index;
  IndexTransform<> transform;
  Result<IndexTransform<>> GetResult() {
    TENSORSTORE_ASSIGN_OR_RETURN(auto new_metadata,
                                 ValidateNewMetadata(cache.get()));
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
  std::size_t component_index;
  absl::Time request_time;
  IndexTransform<> transform;
  ResizeParameters resize_parameters;
};

void SubmitResizeRequest(Promise<IndexTransform<>> promise, ResizeState state) {
  auto* cache_ptr = state.cache.get();
  LinkValue(WithExecutor(cache_ptr->executor(),
                         ResizeContinuation{std::move(state.cache),
                                            state.component_index,
                                            std::move(state.transform)}),
            std::move(promise),
            RequestResize(cache_ptr, std::move(state.resize_parameters),
                          state.request_time));
}

struct DeleteChunksForResizeContinuation {
  std::unique_ptr<ResizeState> state;
  void operator()(Promise<IndexTransform<>> promise, ReadyFuture<const void>) {
    SubmitResizeRequest(std::move(promise), std::move(*state));
  }
};

Future<const void> DeleteChunksForResize(internal::CachePtr<DataCache> cache,
                                         BoxView<> current_bounds,
                                         span<const Index> new_inclusive_min,
                                         span<const Index> new_exclusive_max) {
  span<const Index> chunk_shape = cache->grid().chunk_shape;
  const DimensionIndex rank = chunk_shape.size();
  assert(current_bounds.rank() == rank);
  assert(new_inclusive_min.size() == rank);
  assert(new_exclusive_max.size() == rank);
  auto pair = PromiseFuturePair<void>::Make(MakeResult(Status()));
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
  if (!box_difference.valid()) {
    return absl::InvalidArgumentError(StrCat("Resize would require more than ",
                                             std::numeric_limits<Index>::max(),
                                             " chunk regions to be deleted"));
  }
  for (Index box_i = 0; box_i < box_difference.num_sub_boxes(); ++box_i) {
    box_difference.GetSubBox(box_i, part);
    IterateOverIndexRange(part, [&](span<const Index> cell_indices) {
      auto entry = cache->GetEntryForCell(cell_indices);
      LinkError(pair.promise, entry->Delete());
    });
  }
  return pair.future;
}

struct ResolveBoundsForDeleteAndResizeContinuation {
  std::unique_ptr<ResizeState> state;
  void operator()(Promise<IndexTransform<>> promise, ReadyFuture<const void>) {
    TENSORSTORE_ASSIGN_OR_RETURN(auto new_metadata,
                                 ValidateNewMetadata(state->cache.get()),
                                 static_cast<void>(promise.SetResult(_)));
    // Chunks should never be deleted if `expand_only==false`.
    const DimensionIndex grid_rank = state->cache->grid().chunk_shape.size();
    assert(!state->resize_parameters.expand_only);
    Box<dynamic_rank(internal::kNumInlinedDims)> bounds(grid_rank);
    BitVec<> implicit_lower_bounds(grid_rank);
    BitVec<> implicit_upper_bounds(grid_rank);
    state->cache->state_->GetChunkGridBounds(new_metadata.get(), bounds,
                                             implicit_lower_bounds,
                                             implicit_upper_bounds);
    // The resize request has already been validated against explicit grid
    // bounds (i.e. bounds corresponding to `false` values in
    // `implicit_{lower,upper}_bounds`), so we don't need to check again here.
    TENSORSTORE_RETURN_IF_ERROR(
        ValidateResizeConstraints(
            bounds, state->resize_parameters.new_inclusive_min,
            state->resize_parameters.new_exclusive_max,
            state->resize_parameters.inclusive_min_constraint,
            state->resize_parameters.exclusive_max_constraint,
            /*expand_only=*/false,
            /*shrink_only=*/state->resize_parameters.shrink_only),
        static_cast<void>(promise.SetResult(_)));
    auto* state_ptr = state.get();
    LinkValue(
        WithExecutor(state_ptr->cache->executor(),
                     DeleteChunksForResizeContinuation{std::move(state)}),
        std::move(promise),
        DeleteChunksForResize(state_ptr->cache, bounds,
                              state_ptr->resize_parameters.new_inclusive_min,
                              state_ptr->resize_parameters.new_exclusive_max));
  }
};
}  // namespace

Future<IndexTransform<>> DriverBase::Resize(IndexTransform<> transform,
                                            span<const Index> inclusive_min,
                                            span<const Index> exclusive_max,
                                            ResizeOptions options) {
  auto* cache = this->cache();
  auto resize_parameters = GetResizeParameters(
      *cache->state_, cache->grid(), cache->initial_metadata_.get(),
      component_index(), transform, inclusive_min, exclusive_max, options);
  if (!resize_parameters) {
    if (resize_parameters.status().code() == absl::StatusCode::kAborted) {
      // Requested resize is a no-op.  Currently there is no resize option
      // corresponding to the `fix_resizable_bounds` resolve option, so we
      // don't specify it.
      return ResolveBounds(std::move(transform), /*staleness=*/{},
                           /*options=*/{});
    }
    return resize_parameters.status();
  }

  auto pair = PromiseFuturePair<IndexTransform<>>::Make();
  const absl::Time request_time = absl::Now();
  ResizeState resize_state{
      /*.cache=*/internal::CachePtr<DataCache>(cache),
      /*.component_index=*/component_index(),
      /*.request_time=*/request_time,
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
              cache->metadata_cache_entry_->Read(request_time));
  }
  return std::move(pair.future);
}

Result<IndexTransformSpec> DriverBase::GetBoundSpecData(
    SpecT<internal::ContextBound>* spec, IndexTransformView<> transform_view) {
  auto* cache = this->cache();
  auto* metadata_cache = cache->metadata_cache();
  TENSORSTORE_ASSIGN_OR_RETURN(spec->store,
                               metadata_cache->base_store()->GetBoundSpec());
  spec->data_copy_concurrency = metadata_cache->data_copy_concurrency_;
  spec->cache_pool = metadata_cache->cache_pool_;
  spec->delete_existing = false;
  spec->open = true;
  spec->create = false;
  spec->allow_metadata_mismatch = false;
  spec->staleness.metadata = this->metadata_staleness_bound();
  spec->staleness.data = this->data_staleness_bound();
  spec->rank = this->rank();
  spec->data_type = this->data_type();

  std::shared_ptr<const void> validated_metadata;
  {
    absl::ReaderMutexLock lock(&cache->mutex_);
    validated_metadata = cache->validated_metadata_;
  }

  TENSORSTORE_RETURN_IF_ERROR(cache->state_->GetBoundSpecData(
      spec, validated_metadata.get(), this->component_index()));

  IndexTransform<> transform(transform_view);

  TENSORSTORE_ASSIGN_OR_RETURN(
      auto external_to_internal_transform,
      cache->state_->GetExternalToInternalTransform(validated_metadata.get(),
                                                    component_index()));
  if (external_to_internal_transform.valid()) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto internal_to_external_transform,
        InverseTransform(external_to_internal_transform));
    TENSORSTORE_ASSIGN_OR_RETURN(
        transform,
        ComposeTransforms(internal_to_external_transform, transform));
  }

  return IndexTransformSpec{transform};
}

Status DriverBase::ConvertSpec(SpecT<internal::ContextUnbound>* spec,
                               const SpecRequestOptions& options) {
  if (options.staleness) {
    spec->staleness = *options.staleness;
  }
  if (options.open_mode) {
    const OpenMode open_mode = *options.open_mode;
    spec->open = (open_mode & OpenMode::open) == OpenMode::open;
    spec->create = (open_mode & OpenMode::create) == OpenMode::create;
    spec->allow_metadata_mismatch =
        (open_mode & OpenMode::allow_option_mismatch) ==
        OpenMode::allow_option_mismatch;
    spec->delete_existing =
        (open_mode & OpenMode::delete_existing) == OpenMode::delete_existing;
  }
  return Status();
}

namespace {
/// Validates that the open request specified by `state` can be applied to
/// `metadata`.
Result<std::size_t> ValidateOpenRequest(OpenState* state,
                                        const void* metadata) {
  if (!metadata) {
    return absl::NotFoundError(StrCat(
        "Metadata key ",
        QuoteString(GetMetadataCache(*state)->state_->GetMetadataStorageKey(
            GetMetadataCacheEntry(*state)->key())),
        " does not exist"));
  }
  auto& base = *(PrivateOpenState*)state;  // Cast to private base
  return state->GetComponentIndex(metadata, base.spec_->open_mode());
}

/// \pre `component_index` is the result of a previous call to
///     `state->GetComponentIndex` with the same `metadata`.
/// \pre `metadata != nullptr`
Result<internal::Driver::ReadWriteHandle> CreateTensorStoreFromMetadata(
    std::unique_ptr<OpenState> state, std::shared_ptr<const void> metadata,
    std::size_t component_index) {
  auto& base = *(PrivateOpenState*)state.get();  // Cast to private base
  // TODO(jbms): The read-write mode should be determined based on the
  // KeyValueStore mode, once that is exposed.
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
  Status data_key_value_store_status;
  auto chunk_cache =
      (*state->cache_pool())
          ->GetCache<DataCache>(
              chunk_cache_identifier, [&]() -> std::unique_ptr<DataCache> {
                auto store_result =
                    state->GetDataKeyValueStore(base.store_, metadata.get());
                if (!store_result) {
                  data_key_value_store_status =
                      std::move(store_result).status();
                  return nullptr;
                }
                return std::make_unique<DataCache>(
                    state->GetDataCacheState(metadata.get()),
                    std::move(*store_result),
                    GetMetadataCacheEntry(std::move(*state)), metadata);
              });
  TENSORSTORE_RETURN_IF_ERROR(data_key_value_store_status);
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto new_transform,
      chunk_cache->state_->GetExternalToInternalTransform(
          chunk_cache->initial_metadata_.get(), component_index));

  TENSORSTORE_ASSIGN_OR_RETURN(
      new_transform,
      ResolveBoundsFromMetadata(chunk_cache.get(), metadata.get(),
                                component_index, std::move(new_transform),
                                /*options=*/{}));
  internal::Driver::Ptr driver(state->AllocateDriver(
      {std::move(chunk_cache), component_index,
       base.spec_->staleness.BoundAtOpen(base.request_time_)}));
  return internal::Driver::ReadWriteHandle{
      std::move(driver), std::move(new_transform), read_write_mode};
}

/// Called when the metadata has been written (successfully or unsuccessfully).
struct HandleWroteMetadata {
  std::unique_ptr<OpenState> state;
  void operator()(Promise<internal::Driver::ReadWriteHandle> promise,
                  ReadyFuture<const void> future) {
    auto& base = *(PrivateOpenState*)state.get();  // Cast to private base
    auto& result = future.result();
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
    promise.SetResult([&]() -> Result<internal::Driver::ReadWriteHandle> {
      auto metadata = GetMetadataCacheEntry(*state)->GetMetadata();
      TENSORSTORE_ASSIGN_OR_RETURN(
          std::size_t component_index,
          ValidateOpenRequest(state.get(), metadata.get()));
      return CreateTensorStoreFromMetadata(
          std::move(state), std::move(metadata), component_index);
    }());
  }
};

/// Attempts to create new array.
void CreateMetadata(std::unique_ptr<OpenState> state,
                    Promise<internal::Driver::ReadWriteHandle> promise) {
  auto state_ptr = state.get();
  auto& base = *(PrivateOpenState*)state.get();  // Cast to private base
  LinkValue(
      WithExecutor(state_ptr->executor(),
                   HandleWroteMetadata{std::move(state)}),
      std::move(promise),
      GetMetadataCacheEntry(*state_ptr)
          ->RequestAtomicUpdate(
              [state = state_ptr](const void* existing_metadata)
                  -> Result<std::shared_ptr<const void>> {
                auto result = state->Create(existing_metadata);
                if (result) return result;
                return MaybeAnnotateStatus(
                    result.status(),
                    StrCat("Error creating array with metadata key ",
                           QuoteString(
                               GetMetadataCache(*state)
                                   ->state_->GetMetadataStorageKey(
                                       GetMetadataCacheEntry(*state)->key()))));
              },
              state_ptr->GetCreateConstraint(), base.request_time_));
}

/// Called when the metadata has been read (successfully or not found).
struct HandleReadMetadata {
  std::unique_ptr<OpenState> state;
  void operator()(Promise<internal::Driver::ReadWriteHandle> promise,
                  ReadyFuture<const void> metadata_future) {
    auto metadata = GetMetadataCacheEntry(*state)->GetMetadata();
    auto& base = *(PrivateOpenState*)state.get();  // Cast to private base
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
    promise.SetResult(component_index_result.status());
  }
};

/// Called when the metadata should be requested or created.
struct GetMetadataForOpen {
  OpenState::Ptr state;
  void operator()(Promise<internal::Driver::ReadWriteHandle> promise) {
    auto& base = *(PrivateOpenState*)state.get();  // Cast to private base
    Status metadata_key_value_store_status;
    auto metadata_cache =
        (*state->cache_pool())
            ->GetCache<MetadataCache>(
                base.metadata_cache_key_,
                [&]() -> std::unique_ptr<MetadataCache> {
                  auto store_result =
                      state->GetMetadataKeyValueStore(base.store_);
                  if (!store_result) {
                    promise.SetResult(std::move(store_result).status());
                    return nullptr;
                  }
                  return std::make_unique<MetadataCache>(
                      state->GetMetadataCacheState(), base.store_,
                      std::move(*store_result),
                      base.spec_->data_copy_concurrency,
                      base.spec_->cache_pool);
                });
    if (!metadata_cache) return;
    base.metadata_cache_entry_ =
        GetCacheEntry(metadata_cache, state->GetMetadataCacheEntryKey());
    auto state_ptr = state.get();
    if (base.spec_->open) {
      LinkValue(WithExecutor(state_ptr->executor(),
                             HandleReadMetadata{std::move(state)}),
                std::move(promise),
                GetMetadataCacheEntry(*state_ptr)
                    ->Read(base.spec_->staleness.metadata));
      return;
    }
    // `tensorstore::Open` ensures that at least one of `OpenMode::create` and
    // `OpenMode::open` is specified.
    assert(base.spec_->create);
    CreateMetadata(std::move(state), std::move(promise));
  }
};

/// Called when the KeyValueStore has been successfully opened.
struct HandleKeyValueStoreReady {
  OpenState::Ptr state;
  void operator()(Promise<internal::Driver::ReadWriteHandle> promise,
                  ReadyFuture<KeyValueStore::Ptr> store) {
    auto& base = *(PrivateOpenState*)state.get();  // Cast to private base
    base.store_ = std::move(store.value());
    auto* state_ptr = state.get();
    internal::EncodeCacheKey(&base.metadata_cache_key_, base.store_,
                             typeid(*state_ptr), state->GetMetadataCacheKey());
    if (base.spec_->delete_existing) {
      // Delete all keys starting with the key prefix.
      auto prefix_for_delete = state->GetPrefixForDeleteExisting();
      LinkValue(std::bind(WithExecutor(state_ptr->executor(),
                                       GetMetadataForOpen{std::move(state)}),
                          std::placeholders::_1),
                std::move(promise),
                base.store_->DeletePrefix(std::move(prefix_for_delete)));
      return;
    }
    // Immediately proceed with reading/creating the metadata.
    GetMetadataForOpen callback{std::move(state)};
    callback(std::move(promise));
  }
};

}  // namespace

Future<const void> MetadataCache::Entry::RequestAtomicUpdate(
    UpdateFunction update, AtomicUpdateConstraint update_constraint,
    absl::Time request_time) {
  std::unique_lock<Mutex> lock(metadata_mutex);
  auto [promise, future] = PromiseFuturePair<void>::Make();
  pending_requests.push_back({request_time, std::move(update), promise});
  // If an error occurs with the underlying KeyValueStore, propagate that to the
  // returned future.  Otherwise, the `MetadataCache::DoWriteback`
  // implementation ensures that `promise.raw_result()` is updated to the result
  // returned by the `update` function before the writeback completes.
  LinkValue([](Promise<void> promise,
               ReadyFuture<const void> future) { promise.SetReady(); },
            std::move(promise),
            FinishWrite(
                {std::move(lock)},
                (update_constraint == AtomicUpdateConstraint::kRequireExisting)
                    ? WriteFlags::kConditionalWriteback
                    : WriteFlags::kUnconditionalWriteback));
  return std::move(future);
}

void MetadataCache::DoDeleteEntry(internal::Cache::Entry* base_entry) {
  Entry* entry = static_cast<Entry*>(base_entry);
  delete entry;
}

internal::Cache::Entry* MetadataCache::DoAllocateEntry() { return new Entry; }

std::size_t MetadataCache::DoGetSizeInBytes(Cache::Entry* base_entry) {
  // FIXME: include size of heap-allocated data
  return sizeof(Entry) + Cache::DoGetSizeInBytes(base_entry);
}

void MetadataCache::DoRead(ReadOptions options, ReadReceiver receiver) {
  struct ReadyCallback {
    ReadReceiver receiver;
    void operator()(ReadyFuture<KeyValueStore::ReadResult> future) {
      auto& entry = static_cast<Entry&>(*receiver.entry());
      auto* cache = GetOwningCache(&entry);
      auto& r = future.result();
      if (!r) {
        receiver.NotifyDone(/*size_update=*/{}, r.status());
        return;
      }
      if (r->aborted()) {
        receiver.NotifyDone(/*size_update=*/{}, std::move(r->generation));
        return;
      }
      if (r->not_found()) {
        std::unique_lock<Mutex> lock(entry.metadata_mutex);
        entry.metadata = nullptr;
        receiver.NotifyDone({std::move(lock),
                             /*new_size=*/cache->DoGetSizeInBytes(&entry)},
                            std::move(r->generation));
        return;
      }
      auto metadata_result =
          cache->state_->DecodeMetadata(entry.key(), *r->value);
      if (!metadata_result) {
        auto status = MaybeAnnotateStatus(
            metadata_result.status(),
            StrCat("Error decoding metadata from ",
                   QuoteString(cache->state_->GetMetadataStorageKey(
                       receiver.entry()->key()))));
        if (status.code() == absl::StatusCode::kInvalidArgument) {
          status = absl::FailedPreconditionError(status.message());
        }
        metadata_result = status;
      }
      std::unique_lock<Mutex> lock(entry.metadata_mutex);
      if (metadata_result) {
        entry.metadata = std::move(*metadata_result);
      }
      receiver.NotifyDone(
          {std::move(lock), cache->DoGetSizeInBytes(&entry)},
          metadata_result
              ? Result<TimestampedStorageGeneration>(r->generation)
              : Result<TimestampedStorageGeneration>(metadata_result.status()));
    }
  };
  KeyValueStore::ReadOptions kvs_read_options;
  kvs_read_options.if_not_equal = std::move(options.existing_generation);
  kvs_read_options.staleness_bound = options.staleness_bound;
  auto future =
      store_->Read(state_->GetMetadataStorageKey(receiver.entry()->key()),
                   std::move(kvs_read_options));
  std::move(future).ExecuteWhenReady(
      WithExecutor(executor(), ReadyCallback{std::move(receiver)}));
}

void MetadataCache::DoWriteback(
    TimestampedStorageGeneration existing_generation,
    WritebackReceiver receiver) {
  struct FinishWritebackTask {
    WritebackReceiver receiver;
    MetadataPtr new_metadata;
    void operator()(ReadyFuture<TimestampedStorageGeneration> future) {
      auto& entry = static_cast<Entry&>(*receiver.entry());
      std::unique_lock<Mutex> lock(entry.metadata_mutex);
      auto& r = future.result();
      if (r) {
        if (!StorageGeneration::IsUnknown(r->generation)) {
          entry.metadata = std::move(new_metadata);
        } else {
          if (absl::c_all_of(entry.issued_requests,
                             [](const Entry::UpdateRequest& request) {
                               return request.update_constraint ==
                                      AtomicUpdateConstraint::kRequireMissing;
                             })) {
            // All requests require that the metadata not exist.  Return
            // `kAborted` to indicate that the writeback was not needed.
            r = absl::AbortedError("");
            entry.issued_requests.clear();
          }

          // Retry issued requests by adding them to the beginning of the
          // pending request list.
          if (!entry.issued_requests.empty()) {
            std::swap(entry.issued_requests, entry.pending_requests);
            entry.pending_requests.insert(
                entry.pending_requests.end(),
                std::make_move_iterator(entry.issued_requests.begin()),
                std::make_move_iterator(entry.issued_requests.end()));
          }
        }
      }

      entry.issued_requests.clear();
      receiver.NotifyDone({std::move(lock)}, std::move(r));
    }
  };

  struct StartWritebackTask {
    WritebackReceiver receiver;
    TimestampedStorageGeneration existing_generation;
    void operator()() {
      auto& entry = static_cast<Entry&>(*receiver.entry());
      MetadataPtr new_metadata;
      // Indicates whether there is an update request newer than the metadata.
      absl::Time newest_request_time = existing_generation.time;
      {
        std::unique_lock<Mutex> lock(entry.metadata_mutex);
        const void* existing_metadata = entry.metadata.get();
        for (const auto& request : entry.pending_requests) {
          newest_request_time =
              std::max(newest_request_time, request.request_time);
          auto result = request.update(existing_metadata);
          if (result) {
            assert(*result);
            assert(request.update_constraint !=
                       AtomicUpdateConstraint::kRequireMissing ||
                   existing_metadata == nullptr);
            assert(request.update_constraint !=
                       AtomicUpdateConstraint::kRequireExisting ||
                   existing_metadata != nullptr);
            new_metadata = std::move(*result);
            existing_metadata = new_metadata.get();
            request.promise.raw_result() = MakeResult();
          } else {
            request.promise.raw_result() = std::move(result).status();
          }
        }
        if (new_metadata) {
          // Mark all pending requests as issued.
          entry.issued_requests = std::move(entry.pending_requests);
          entry.pending_requests.clear();
        } else if (newest_request_time > existing_generation.time) {
          // Requests will be retried after re-reading the existing metadata.
          // Leave the pending requests marked pending.
        } else {
          // Pending requests will not be retried.
          entry.pending_requests.clear();
        }
        receiver.NotifyStarted({std::move(lock)});
      }

      if (!new_metadata) {
        // None of the requested changes are compatible with the current state.
        // If at least one update request is newer than the current metadata, we
        // specify `StorageGeneration::Unknown()` to force a re-read.
        // Otherwise, we fail with `absl::StatusCode::kAborted` to indicate that
        // no writeback was needed.
        receiver.NotifyDone(
            /*size_update=*/{},
            (newest_request_time > existing_generation.time)
                ? Result<TimestampedStorageGeneration>(
                      std::in_place, StorageGeneration::Unknown(),
                      newest_request_time)
                : Result<TimestampedStorageGeneration>(absl::AbortedError("")));
        return;
      }
      auto* cache = GetOwningCache(&entry);
      auto future = cache->store_->Write(
          cache->state_->GetMetadataStorageKey(entry.key()),
          cache->state_->EncodeMetadata(entry.key(), new_metadata.get()),
          {StorageGeneration::IsUnknown(existing_generation.generation)
               ? StorageGeneration::NoValue()
               : existing_generation.generation});
      future.Force();
      std::move(future).ExecuteWhenReady(
          WithExecutor(cache->executor(),
                       FinishWritebackTask{std::move(receiver), new_metadata}));
    }
  };
  executor()(
      StartWritebackTask{std::move(receiver), std::move(existing_generation)});
}

void DataCache::DoRead(ReadOptions options, ReadReceiver receiver) {
  struct ReadyCallback {
    ReadReceiver receiver;
    void operator()(ReadyFuture<KeyValueStore::ReadResult> future) {
      auto* cache = static_cast<DataCache*>(GetOwningCache(receiver.entry()));
      auto& r = future.result();
      if (!r) {
        receiver.NotifyDone(r.status());
        return;
      }
      if (!r->value) {
        // Aborted or not found.
        receiver.NotifyDone(ReadReceiver::ComponentsWithGeneration{
            {}, std::move(r->generation)});
        return;
      }
      const auto validated_metadata = cache->validated_metadata();
      Result<absl::InlinedVector<SharedArrayView<const void>, 1>> decoded =
          cache->state_->DecodeChunk(validated_metadata.get(),
                                     receiver.entry()->cell_indices(),
                                     std::move(*r->value));
      if (!decoded) {
        decoded = MaybeAnnotateStatus(
            decoded.status(),
            StrCat("Error decoding chunk ",
                   QuoteString(cache->GetChunkStorageKey(receiver.entry()))));
      }
      Result<absl::InlinedVector<ArrayView<const void>, 1>> decoded_ref =
          MapResult(
              [](span<const SharedArrayView<const void>> data) {
                return absl::InlinedVector<ArrayView<const void>, 1>(
                    data.begin(), data.end());
              },
              decoded);
      receiver.NotifyDone(MapResult(
          [&](const absl::InlinedVector<ArrayView<const void>, 1>& data)
              -> ReadReceiver::ComponentsWithGeneration {
            return {data, std::move(r->generation)};
          },
          decoded_ref));
    }
  };
  KeyValueStore::ReadOptions kvs_read_options;
  kvs_read_options.if_not_equal = std::move(options.existing_generation);
  kvs_read_options.staleness_bound = options.staleness_bound;

  auto future = store()->Read(GetChunkStorageKey(receiver.entry()),
                              std::move(kvs_read_options));
  std::move(future).ExecuteWhenReady(
      WithExecutor(executor(), ReadyCallback{std::move(receiver)}));
}

void DataCache::DoWriteback(TimestampedStorageGeneration existing_generation,
                            WritebackReceiver receiver) {
  struct WriteDoneCallback {
    WritebackReceiver receiver;
    void operator()(ReadyFuture<TimestampedStorageGeneration> future) {
      receiver.NotifyDone(std::move(future.result()));
    }
  };
  struct ExecutorCallback {
    WritebackReceiver receiver;
    StorageGeneration existing_generation;
    void operator()() {
      ChunkCache::Entry* entry = receiver.entry();
      auto* cache = static_cast<DataCache*>(GetOwningCache(entry));
      std::string encoded;
      bool equals_fill_value;
      {
        ChunkCache::WritebackSnapshot snapshot(receiver);
        equals_fill_value = snapshot.equals_fill_value();
        if (!equals_fill_value) {
          const auto validated_metadata = cache->validated_metadata();
          auto status = cache->state_->EncodeChunk(
              validated_metadata.get(), entry->cell_indices(),
              snapshot.component_arrays(), &encoded);
          if (!status.ok()) {
            receiver.NotifyDone(std::move(status));
            return;
          }
        }
      }
      auto write_future =
          equals_fill_value
              ? cache->store()->Delete(cache->GetChunkStorageKey(entry),
                                       {std::move(existing_generation)})
              : cache->store()->Write(cache->GetChunkStorageKey(entry),
                                      std::move(encoded),
                                      {std::move(existing_generation)});
      write_future.Force();
      std::move(write_future)
          .ExecuteWhenReady(WithExecutor(
              cache->executor(), WriteDoneCallback{std::move(receiver)}));
    }
  };
  executor()(ExecutorCallback{std::move(receiver),
                              std::move(existing_generation.generation)});
}

Future<internal::Driver::ReadWriteHandle> OpenDriver(OpenState::Ptr state) {
  // TODO(jbms): possibly determine these options from the open options.
  auto& base = *(PrivateOpenState*)state.get();  // Cast to private base

  auto& spec = *base.spec_;
  if (spec.delete_existing && !spec.create) {
    return Status(absl::StatusCode::kInvalidArgument,
                  "Cannot specify an open mode of `delete_existing` "
                  "without `create`");
  }

  if (spec.delete_existing && spec.open) {
    return Status(absl::StatusCode::kInvalidArgument,
                  "Cannot specify an open mode of `delete_existing` "
                  "with `open`");
  }

  if (spec.create && (base.read_write_mode_ != ReadWriteMode::dynamic &&
                      !(base.read_write_mode_ & ReadWriteMode::write))) {
    return Status(absl::StatusCode::kInvalidArgument,
                  "Cannot specify an open mode of `create` "
                  "without `write`");
  }

  auto* state_ptr = state.get();
  return PromiseFuturePair<internal::Driver::ReadWriteHandle>::LinkValue(
             WithExecutor(state_ptr->executor(),
                          HandleKeyValueStoreReady{std::move(state)}),
             base.spec_->store->Open())
      .future;
}

Result<IndexTransform<>> ResolveBoundsFromMetadata(
    DataCacheState& state, const internal::ChunkGridSpecification& grid,
    const void* new_metadata, std::size_t component_index,
    IndexTransform<> transform, ResolveBoundsOptions options) {
  const DimensionIndex base_rank = grid.components[component_index].rank();
  BitVec<> base_implicit_lower_bounds(base_rank);
  BitVec<> base_implicit_upper_bounds(base_rank);
  Box<dynamic_rank(internal::kNumInlinedDims)> base_bounds(base_rank);
  GetComponentBounds(state, grid, new_metadata, component_index, base_bounds,
                     base_implicit_lower_bounds, base_implicit_upper_bounds);
  if ((options.mode & fix_resizable_bounds) == fix_resizable_bounds) {
    base_implicit_lower_bounds.fill(false);
    base_implicit_upper_bounds.fill(false);
  }
  return PropagateBoundsToTransform(
      BoxView<>(base_bounds),
      BitSpan<const std::uint64_t>(base_implicit_lower_bounds),
      BitSpan<const std::uint64_t>(base_implicit_upper_bounds),
      std::move(transform));
}

Status ValidateResizeConstraints(BoxView<> current_domain,
                                 span<const Index> new_inclusive_min,
                                 span<const Index> new_exclusive_max,
                                 span<const Index> inclusive_min_constraint,
                                 span<const Index> exclusive_max_constraint,
                                 bool expand_only, bool shrink_only) {
  TENSORSTORE_RETURN_IF_ERROR(ValidateResizeDomainConstraint(
      current_domain, inclusive_min_constraint, exclusive_max_constraint));
  TENSORSTORE_RETURN_IF_ERROR(ValidateExpandShrinkConstraints(
      current_domain, new_inclusive_min, new_exclusive_max, expand_only,
      shrink_only));
  return absl::OkStatus();
}

Result<ResizeParameters> GetResizeParameters(
    DataCacheState& state, const internal::ChunkGridSpecification& grid,
    const void* metadata, size_t component_index,
    IndexTransformView<> transform, span<const Index> inclusive_min,
    span<const Index> exclusive_max, ResizeOptions options) {
  assert(transform.input_rank() == inclusive_min.size());
  assert(transform.input_rank() == exclusive_max.size());
  const DimensionIndex output_rank = transform.output_rank();

  const DimensionIndex base_rank = grid.components[component_index].rank();
  BitVec<> base_implicit_lower_bounds(base_rank);
  BitVec<> base_implicit_upper_bounds(base_rank);
  Box<dynamic_rank(internal::kNumInlinedDims)> base_bounds(base_rank);
  GetComponentBounds(state, grid, metadata, component_index, base_bounds,
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
        return absl::FailedPreconditionError(
            StrCat("Cannot change inclusive lower bound of output dimension ",
                   output_dim, ", which is fixed at ",
                   dim_bounds.inclusive_min(), ", to ", new_inclusive_min));
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
        return absl::FailedPreconditionError(
            StrCat("Cannot change exclusive upper bound of output dimension ",
                   output_dim, ", which is fixed at ",
                   dim_bounds.exclusive_max(), ", to ", new_exclusive_max));
      }
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

DriverBase::DriverBase(Initializer&& initializer)
    : internal::ChunkCacheDriver(std::move(initializer.cache),
                                 initializer.component_index,
                                 initializer.staleness_bounds.data),
      metadata_staleness_bound_(initializer.staleness_bounds.metadata) {}

DataCache* DriverBase::cache() const {
  return static_cast<DataCache*>(internal::ChunkCacheDriver::cache());
}

Executor DriverBase::data_copy_executor() { return cache()->executor(); }

namespace jb = tensorstore::internal::json_binding;
namespace {
struct MaybeOpenCreate {
  std::optional<bool> open;
  std::optional<bool> create;
};

/// JSON binder for `tensorstore::StalenessBound`.
inline constexpr auto StalenessBoundJsonBinder =
    [](auto is_loading, const auto& options, auto* obj,
       ::nlohmann::json* j) -> absl::Status {
  if constexpr (is_loading) {
    if (const auto* b = j->get_ptr<const bool*>()) {
      *obj = *b ? absl::InfiniteFuture() : absl::InfinitePast();
    } else if (j->is_number()) {
      const double t = static_cast<double>(*j);
      *obj = absl::UnixEpoch() + absl::Seconds(t);
    } else if (*j == "open") {
      static_cast<absl::Time&>(*obj) = absl::InfiniteFuture();
      obj->bounded_by_open_time = true;
    } else {
      return internal_json::ExpectedError(*j, "boolean, number, or \"open\"");
    }
  } else {
    if (obj->bounded_by_open_time) {
      *j = "open";
    } else {
      const absl::Time& t = *obj;
      if (t == absl::InfiniteFuture()) {
        *j = true;
      } else if (t == absl::InfinitePast()) {
        *j = false;
      } else {
        *j = absl::ToDoubleSeconds(t - absl::UnixEpoch());
      }
    }
  }
  return absl::OkStatus();
};
}  // namespace

TENSORSTORE_DEFINE_JSON_BINDER(
    SpecJsonBinder,
    jb::Sequence(
        jb::Member(internal::DataCopyConcurrencyResource::id,
                   jb::Projection(&SpecT<>::data_copy_concurrency)),
        jb::Member(internal::CachePoolResource::id,
                   jb::Projection(&SpecT<>::cache_pool)),
        jb::Member("kvstore", jb::Projection(&SpecT<>::store)),
        jb::Projection(
            &SpecT<>::staleness,
            jb::Sequence(
                jb::Member("recheck_cached_metadata",
                           jb::Projection(&StalenessBounds::metadata,
                                          jb::DefaultValue(
                                              [](auto* obj) {
                                                obj->bounded_by_open_time =
                                                    true;
                                              },
                                              StalenessBoundJsonBinder))),
                jb::Member("recheck_cached_data",
                           jb::Projection(&StalenessBounds::data,
                                          jb::DefaultInitializedValue(
                                              StalenessBoundJsonBinder))))),
        jb::GetterSetter(
            [](auto& obj) {
              return MaybeOpenCreate{(obj.open == true && obj.create == true)
                                         ? std::optional<bool>(true)
                                         : std::optional<bool>(),
                                     (obj.create == true)
                                         ? std::optional<bool>(true)
                                         : std::optional<bool>()};
            },
            [](auto& obj, const auto& x) {
              obj.open = x.open || x.create ? x.open.value_or(false) : true;
              obj.create = x.create.value_or(false);
            },
            jb::Sequence(jb::Member("open",
                                    jb::Projection(&MaybeOpenCreate::open)),
                         jb::Member("create",
                                    jb::Projection(&MaybeOpenCreate::create)))),
        jb::Member("delete_existing",
                   jb::Projection(&SpecT<>::delete_existing,
                                  jb::DefaultValue([](bool* v) {
                                    *v = false;
                                  }))),
        jb::Member("allow_metadata_mismatch",
                   jb::Projection(&SpecT<>::allow_metadata_mismatch,
                                  jb::DefaultValue([](bool* v) {
                                    *v = false;
                                  })))));

}  // namespace internal_kvs_backed_chunk_driver
}  // namespace tensorstore
