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

#include "tensorstore/driver/driver.h"

#include <cassert>
#include <memory>    // For std::shared_ptr, std::move
#include <optional>  // For std::optional
#include <string>
#include <utility>  // For std::move

#include "absl/log/absl_log.h"  // For logging
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "tensorstore/array.h"
#include "tensorstore/chunk_layout.h"
#include "tensorstore/driver/chunk_cache_driver.h"  // For ChunkGridSpecificationDriver, ChunkCacheReadWriteDriverMixin, ChunkCacheDriverInitializer
#include "tensorstore/driver/driver_spec.h"
#include "tensorstore/driver/kvs_backed_chunk_driver.h"  // For KvsDriverSpec, SpecJsonBinder
#include "tensorstore/driver/registry.h"
#include "tensorstore/driver/tiff/metadata.h"  // For TiffMetadata, DecodeChunk
#include "tensorstore/index_space/internal/propagate_bounds.h"  // For PropagateBoundsToTransform
#include "tensorstore/internal/cache/async_cache.h"  // For AsyncCache, AsyncCache::Entry, ReadData
#include "tensorstore/internal/cache/cache.h"  // For CachePool, GetOwningCache
#include "tensorstore/internal/cache/kvs_backed_chunk_cache.h"  // For KvsBackedCache base class
#include "tensorstore/kvstore/driver.h"      // For kvstore::DriverPtr
#include "tensorstore/kvstore/generation.h"  // For TimestampedStorageGeneration
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/tiff/tiff_key_value_store.h"
#include "tensorstore/util/execution/any_receiver.h"  // For DecodeReceiver etc.
#include "tensorstore/util/execution/execution.h"  // For execution::set_value/error
#include "tensorstore/util/garbage_collection/fwd.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal_tiff {

// Avoid anonymous namespace to workaround MSVC bug.
//
// https://developercommunity.visualstudio.com/t/Bug-involving-virtual-functions-templat/10424129
#ifndef _MSC_VER
namespace {
#endif

namespace jb = tensorstore::internal_json_binding;

using ::tensorstore::internal::AsyncCache;
using ::tensorstore::internal::GetOwningCache;
using ::tensorstore::internal_kvs_backed_chunk_driver::KvsDriverSpec;

// Define the TIFF-specific chunk cache, inheriting from KvsBackedChunkCache.
// This cache handles reading raw tile/strip data from the TiffKeyValueStore
// and decoding it.
class TiffChunkCache : public internal::KvsBackedChunkCache {
 public:
  using Base = internal::KvsBackedChunkCache;
  using ReadData = ChunkCache::ReadData;

  explicit TiffChunkCache(kvstore::DriverPtr tiff_kv_store_driver,
                          std::shared_ptr<const TiffMetadata> resolved_metadata,
                          internal::ChunkGridSpecification grid,
                          Executor executor)
      : Base(std::move(tiff_kv_store_driver)),
        resolved_metadata_(std::move(resolved_metadata)),
        grid_(std::move(grid)),
        executor_(std::move(executor)) {
    assert(resolved_metadata_ && "Resolved metadata cannot be null.");
  }

  // Returns the grid specification provided during construction.
  const internal::ChunkGridSpecification& grid() const override {
    return grid_;
  }

  const Executor& executor() const override { return executor_; }

  // TODO(hsidky): Refactor this out into metadata. Especially when we change
  // the kvstore to index based.
  std::string GetChunkStorageKey(span<const Index> cell_indices) override {
    const auto& metadata = *resolved_metadata_;
    const auto& grid = grid_;  // Get the grid spec stored in the cache

    const DimensionIndex grid_rank = grid.grid_rank();
    ABSL_CHECK(cell_indices.size() == grid_rank);
    ABSL_CHECK(grid.components.size() == 1);  // Expect single component view

    // Get dimension mapping information from the helper
    TiffGridMappingInfo mapping_info = GetTiffGridMappingInfo(metadata);

    uint32_t ifd = 0;
    uint32_t row_idx = 0;
    uint32_t col_idx = 0;

    const auto& chunked_to_cell = grid.components[0].chunked_to_cell_dimensions;
    ABSL_CHECK(chunked_to_cell.size() == grid_rank);

    // Find the grid dimensions corresponding to the logical dimensions
    DimensionIndex grid_dim_for_y = -1;
    DimensionIndex grid_dim_for_x = -1;
    DimensionIndex grid_dim_for_ifd = -1;

    for (DimensionIndex grid_i = 0; grid_i < grid_rank; ++grid_i) {
      DimensionIndex ts_dim = chunked_to_cell[grid_i];
      if (ts_dim == mapping_info.ts_y_dim) grid_dim_for_y = grid_i;
      if (ts_dim == mapping_info.ts_x_dim) grid_dim_for_x = grid_i;
      if (ts_dim == mapping_info.ts_ifd_dim) grid_dim_for_ifd = grid_i;
    }

    // Extract indices based on the mapping found
    if (metadata.num_ifds == 1) {
      ifd = metadata.ifd_index;
      // Grid must map Y (if rank >= 2) and X dimensions
      ABSL_CHECK(grid_rank >= 1);  // Must have at least X dimension chunked
      ABSL_CHECK(metadata.rank < 2 || grid_dim_for_y != -1)
          << "Grid mapping for Y dim missing in single IFD mode";
      ABSL_CHECK(grid_dim_for_x != -1)
          << "Grid mapping for X dim missing in single IFD mode";

      row_idx = (grid_dim_for_y != -1)
                    ? static_cast<uint32_t>(cell_indices[grid_dim_for_y])
                    : 0;
      col_idx = static_cast<uint32_t>(cell_indices[grid_dim_for_x]);

    } else {  // Multi-IFD case
      ABSL_CHECK(grid_rank == 3) << "Expected grid rank 3 for multi-IFD mode";
      ABSL_CHECK(grid_dim_for_ifd != -1)
          << "Grid mapping for IFD/Z dim missing in multi-IFD mode";
      ABSL_CHECK(grid_dim_for_y != -1)
          << "Grid mapping for Y dim missing in multi-IFD mode";
      ABSL_CHECK(grid_dim_for_x != -1)
          << "Grid mapping for X dim missing in multi-IFD mode";

      ifd = static_cast<uint32_t>(cell_indices[grid_dim_for_ifd]);
      row_idx = static_cast<uint32_t>(cell_indices[grid_dim_for_y]);
      col_idx = static_cast<uint32_t>(cell_indices[grid_dim_for_x]);
    }

    // Handle stripped images: column index is always 0
    if (!mapping_info.is_tiled) {
      // Grid dim for X must exist if rank > 0
      ABSL_CHECK(grid_dim_for_x != -1);
      // Check grid configuration consistency for strips
      ABSL_CHECK(grid.chunk_shape[grid_dim_for_x] == 1)
          << "Grid shape for X dimension should be 1 for stripped TIFF";
      ABSL_CHECK(cell_indices[grid_dim_for_x] == 0)
          << "Cell index for X dimension should be 0 for stripped TIFF";
      col_idx = 0;
    }

    // Format the final key
    return absl::StrFormat("tile/%d/%d/%d", ifd, row_idx, col_idx);
  }

  // Decodes chunk data (called by Entry::DoDecode indirectly).
  Result<absl::InlinedVector<SharedArray<const void>, 1>> DecodeChunk(
      span<const Index> chunk_indices, absl::Cord data) override {
    // This method is required by the base class. We delegate to the
    // already-existing global DecodeChunk function.
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto decoded_chunk,
        internal_tiff::DecodeChunk(*resolved_metadata_, std::move(data)));
    absl::InlinedVector<SharedArray<const void>, 1> components;
    components.emplace_back(std::move(decoded_chunk));
    return components;
  }

  // Encodes chunk data (called by Entry::DoEncode indirectly). Read-only.
  Result<absl::Cord> EncodeChunk(
      span<const Index> chunk_indices,
      span<const SharedArray<const void>> component_arrays) override {
    return absl::UnimplementedError("TIFF driver is read-only");
  }

  // Defines cache entry behavior, specifically decoding.
  class Entry : public Base::Entry {
   public:
    using OwningCache = TiffChunkCache;
    using KvsEntryBase = OwningCache::Base::Entry;
    using DecodeReceiver = typename Base::Entry::DecodeReceiver;
    using EncodeReceiver = typename Base::Entry::EncodeReceiver;

    // Encodes data for writing back to KvStore. Not supported for read-only.
    void DoEncode(std::shared_ptr<const ReadData> read_data,
                  EncodeReceiver receiver) override {
      execution::set_error(
          receiver, absl::UnimplementedError("TIFF driver is read-only"));
    }

    // Override description for debugging/logging.
    std::string DescribeChunk() override {
      auto& cache = GetOwningCache(*this);
      auto cell_indices = this->cell_indices();
      return tensorstore::StrCat("TIFF chunk ", cell_indices, " (key=",
                                 cache.GetChunkStorageKey(cell_indices), ")");
    }

    absl::Status AnnotateError(const absl::Status& error, bool reading) {
      return GetOwningCache(*this).kvstore_driver_->AnnotateError(
          this->GetKeyValueStoreKey(), reading ? "reading" : "writing", error);
    }

  };  // End Entry definition

  // --- Required Allocation Methods ---
  Entry* DoAllocateEntry() final { return new Entry; }
  size_t DoGetSizeofEntry() final { return sizeof(Entry); }

  // Allocate the base transaction node type from KvsBackedChunkCache.
  TransactionNode* DoAllocateTransactionNode(AsyncCache::Entry& entry) final {
    return new Base::TransactionNode(static_cast<Base::Entry&>(entry));
  }

 private:
  std::shared_ptr<const TiffMetadata> resolved_metadata_;
  internal::ChunkGridSpecification grid_;
  Executor executor_;

};  // End TiffChunkCache definition

// TiffDriverSpec: Defines the specification for opening a TIFF TensorStore.
class TiffDriverSpec
    : public internal::RegisteredDriverSpec<TiffDriverSpec, KvsDriverSpec> {
 public:
  constexpr static char id[] = "tiff";
  using Base = internal::RegisteredDriverSpec<TiffDriverSpec, KvsDriverSpec>;

  // --- Members ---
  TiffSpecOptions tiff_options;  // e.g. ifd_index
  TiffMetadataConstraints
      metadata_constraints;  // e.g. shape, dtype constraints

  constexpr static auto ApplyMembers = [](auto& x, auto f) {
    return f(internal::BaseCast<KvsDriverSpec>(x), x.tiff_options,
             x.metadata_constraints);
  };

  // Inherited members from KvsDriverSpec:
  // kvstore::Spec store;
  // Schema schema;
  // Context::Resource<...> data_copy_concurrency;
  // Context::Resource<...> cache_pool;
  // std::optional<Context::Resource<...>> metadata_cache_pool;
  // StalenessBounds staleness;
  // internal_kvs_backed_chunk_driver::FillValueMode fill_value_mode;
  // (Also OpenModeSpec members: open, create, delete_existing, etc.)

  static inline const auto default_json_binder = jb::Sequence(
      jb::Validate(
          [](const auto& options, auto* obj) {
            if (obj->schema.dtype().valid()) {
              return ValidateDataType(obj->schema.dtype());
            }
            return absl::OkStatus();
          },
          internal_kvs_backed_chunk_driver::SpecJsonBinder),
      jb::Member(
          "metadata",
          jb::Validate(
              [](const auto& options, auto* obj) {
                TENSORSTORE_RETURN_IF_ERROR(obj->schema.Set(
                    obj->metadata_constraints.dtype.value_or(DataType())));
                TENSORSTORE_RETURN_IF_ERROR(obj->schema.Set(
                    RankConstraint{obj->metadata_constraints.rank}));
                return absl::OkStatus();
              },
              jb::Projection<&TiffDriverSpec::metadata_constraints>(
                  jb::DefaultInitializedValue()))),
      jb::Member("tiff", jb::Projection<&TiffDriverSpec::tiff_options>(
                             jb::DefaultValue([](auto* v) { *v = {}; }))));

  // --- Overrides from DriverSpec ---
  Result<IndexDomain<>> GetDomain() const override {
    return internal_tiff::GetEffectiveDomain(tiff_options, metadata_constraints,
                                             schema);
  }

  Result<CodecSpec> GetCodec() const override {
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto codec_spec_ptr, internal_tiff::GetEffectiveCodec(
                                 tiff_options, metadata_constraints, schema));
    // Wrap the driver-specific spec ptr in the generic CodecSpec
    return CodecSpec(std::move(codec_spec_ptr));
  }

  Result<ChunkLayout> GetChunkLayout() const override {
    return internal_tiff::GetEffectiveChunkLayout(tiff_options,
                                                  metadata_constraints, schema);
  }

  Result<SharedArray<const void>> GetFillValue(
      IndexTransformView<> transform) const override {
    // Respect schema's fill value if set, otherwise default (nullptr).
    return schema.fill_value().valid()
               ? tensorstore::Result<SharedArray<const void>>(
                     schema.fill_value())
               : tensorstore::Result<SharedArray<const void>>{std::in_place};
  }

  Result<DimensionUnitsVector> GetDimensionUnits() const override {
    return internal_tiff::GetEffectiveDimensionUnits(
        tiff_options, metadata_constraints, schema);
  }

  absl::Status ApplyOptions(SpecOptions&& options) override {
    if (options.minimal_spec) {
      // Reset constraints if minimal_spec is requested
      metadata_constraints = {};
      tiff_options = {};
    }
    // Apply options to base KvsDriverSpec members (includes Schema options)
    TENSORSTORE_RETURN_IF_ERROR(Base::ApplyOptions(std::move(options)));
    return absl::OkStatus();
  }

  // --- Open method ---
  // Implementation will be provided later, after TiffDriver is defined.
  Future<internal::Driver::Handle> Open(
      internal::DriverOpenRequest request) const override;

};  // End TiffDriverSpec

// Initializer structure for TiffDriver
struct TiffDriverInitializer {
  internal::CachePtr<TiffChunkCache> cache;
  size_t component_index;  // Always 0 for TIFF.
  StalenessBound data_staleness_bound;
  StalenessBound metadata_staleness_bound;
  internal::PinnedCacheEntry<internal_tiff_kvstore::TiffDirectoryCache>
      metadata_cache_entry;
  internal_kvs_backed_chunk_driver::FillValueMode fill_value_mode;
  std::shared_ptr<const TiffMetadata> initial_metadata;
  TiffSpecOptions tiff_options;
  Schema schema;
  Context::Resource<internal::DataCopyConcurrencyResource>
      data_copy_concurrency;
  Context::Resource<internal::CachePoolResource> cache_pool;
  // Use optional for metadata pool resource, as it might be the same as
  // cache_pool
  std::optional<Context::Resource<internal::CachePoolResource>>
      metadata_cache_pool;
};

// Forward declare TiffDriver if needed before the using alias
class TiffDriver;

using TiffDriverBase = internal::RegisteredDriver<
    TiffDriver,
    internal::ChunkGridSpecificationDriver<TiffChunkCache, internal::Driver>>;

class TiffDriver final : public TiffDriverBase {
 public:
  using Base = TiffDriverBase;

  explicit TiffDriver(TiffDriverInitializer&& initializer)
      : Base({std::move(initializer.cache),
              initializer.component_index,  // Should be 0
              initializer.data_staleness_bound}),
        metadata_staleness_bound_(initializer.metadata_staleness_bound),
        metadata_cache_entry_(std::move(initializer.metadata_cache_entry)),
        fill_value_mode_(initializer.fill_value_mode),
        initial_metadata_(std::move(initializer.initial_metadata)),
        tiff_options_(std::move(initializer.tiff_options)),
        schema_(std::move(initializer.schema)),
        data_copy_concurrency_(std::move(initializer.data_copy_concurrency)),
        cache_pool_(std::move(initializer.cache_pool)),
        metadata_cache_pool_(std::move(initializer.metadata_cache_pool)) {
    ABSL_CHECK(component_index() == 0);
    ABSL_CHECK(metadata_cache_entry_);
  }

  Result<std::shared_ptr<const TiffMetadata>> GetMetadata() const {
    return initial_metadata_;
  }

  // --- Overrides from internal::Driver ---

  // dtype() and rank() are provided by ChunkGridSpecificationDriver base

  Result<internal::TransformedDriverSpec> GetBoundSpec(
      internal::OpenTransactionPtr transaction,
      IndexTransformView<> transform) override {
    auto spec = internal::DriverSpec::Make<TiffDriverSpec>();

    // Call the helper function to populate the spec and get the transform
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto final_transform,
        GetBoundSpecData(std::move(transaction), *spec, transform));

    return internal::TransformedDriverSpec{std::move(spec),
                                           std::move(final_transform)};
  }

  void GarbageCollectionVisit(
      garbage_collection::GarbageCollectionVisitor& visitor) const override {
    // Visit the base members (includes the cache ptr)
    Base::GarbageCollectionVisit(visitor);
    // Visit the metadata cache entry
    garbage_collection::GarbageCollectionVisit(visitor, metadata_cache_entry_);
  }

  Result<ChunkLayout> GetChunkLayout(IndexTransformView<> transform) override {
    // initial_metadata_ holds the snapshot from Open, which includes the base
    // chunk layout.
    const auto& metadata = *initial_metadata_;

    // Apply the inverse transform to the driver's base chunk layout
    // to get the layout corresponding to the input space of the transform.
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto layout,
        ApplyInverseIndexTransform(transform, metadata.chunk_layout));

    TENSORSTORE_RETURN_IF_ERROR(layout.Finalize());
    return layout;
  }

  Result<CodecSpec> GetCodec() override {
    TENSORSTORE_ASSIGN_OR_RETURN(auto metadata, GetMetadata());
    // TODO(hsidky): Create TiffCodecSpec based on
    // metadata->compressor/compression_type
    //             and return CodecSpec(std::move(tiff_codec_spec_ptr))
    // For now, return default/unimplemented.
    auto codec_spec = internal::CodecDriverSpec::Make<TiffCodecSpec>();
    codec_spec->compression_type = metadata->compression_type;
    return CodecSpec(std::move(codec_spec));
  }

  Result<SharedArray<const void>> GetFillValue(
      IndexTransformView<> transform) override {
    // TIFF doesn't intrinsically have a fill value. Return default (null).
    return SharedArray<const void>();
  }

  Result<DimensionUnitsVector> GetDimensionUnits() override {
    TENSORSTORE_ASSIGN_OR_RETURN(auto metadata, GetMetadata());
    // Return the dimension units stored in the resolved metadata.
    // Ensure the rank matches.
    if (metadata->dimension_units.size() != rank()) {
      return absl::InternalError("Metadata dimension_units rank mismatch");
    }
    return metadata->dimension_units;
  }

  KvStore GetKvstore(const Transaction& transaction) override {
    auto& dir_cache = internal::GetOwningCache(*metadata_cache_entry_);
    std::string path(metadata_cache_entry_->key());
    return KvStore(kvstore::DriverPtr(dir_cache.kvstore_driver_),
                   std::move(path), transaction);
  }

  Result<internal::DriverHandle> GetBase(
      ReadWriteMode read_write_mode, IndexTransformView<> transform,
      const Transaction& transaction) override {
    // TIFF driver is not an adapter.
    return internal::DriverHandle();
  }

  // data_copy_executor() is provided by ChunkGridSpecificationDriver base

  void Read(ReadRequest request, ReadChunkReceiver receiver) override {
    // Replicate logic from ChunkCacheReadWriteDriverMixin
    cache()->Read({std::move(request), component_index(),
                   this->data_staleness_bound().time,
                   fill_value_mode_.fill_missing_data_reads},
                  std::move(receiver));
  }

  void Write(WriteRequest request, WriteChunkReceiver receiver) override {
    // Fail explicitly for read-only driver
    execution::set_error(receiver,
                         absl::UnimplementedError("TIFF driver is read-only"));
  }

  Future<IndexTransform<>> ResolveBounds(
      ResolveBoundsRequest request) override {
    // Asynchronously resolve the metadata first.
    return MapFuture(
        this->data_copy_executor(),
        // Capture the necessary parts of the request.
        [transform = std::move(request.transform),
         options = std::move(request.options)](
            const Result<std::shared_ptr<const TiffMetadata>>&
                metadata_result) mutable -> Result<IndexTransform<>> {
          // Check if metadata resolution was successful.
          TENSORSTORE_RETURN_IF_ERROR(metadata_result);
          const auto& metadata = *metadata_result.value();

          // The authoritative domain is defined by the metadata's shape.
          // TIFF files inherently have a zero origin.
          BoxView<> domain(metadata.shape);

          // Determine implicit bounds. TIFF dimensions are usually fixed
          // by the file format, so upper bounds are explicit unless
          // fix_resizable_bounds is requested.
          DimensionSet implicit_lower_bounds(
              false);  // Always explicit 0 lower bound
          DimensionSet implicit_upper_bounds(
              false);  // Assume fixed upper bounds initially

          if (!(options.mode & fix_resizable_bounds)) {
            // If fix_resizable_bounds is *not* set, treat upper bounds
            // as implicit, For TIFF, where bounds are usually fixed, this
            // might be debatable, but let's follow the pattern unless
            // fix_resizable_bounds is set.
            implicit_upper_bounds = true;
          }

          // Propagate the domain bounds from metadata to the transform.
          return PropagateBoundsToTransform(domain, implicit_lower_bounds,
                                            implicit_upper_bounds,
                                            std::move(transform));
        },
        // Call the helper to get the metadata future.
        ResolveMetadata(std::move(request.transaction)));
  }

  Future<IndexTransform<>> Resize(ResizeRequest request) override {
    return absl::UnimplementedError("Resize is not supported by TIFF driver");
  }

  Future<ArrayStorageStatistics> GetStorageStatistics(
      GetStorageStatisticsRequest request) override {
    // TODO(hsidky): Implement GetStorageStatistics if desired.
    // Might involve iterating keys in TiffKvStore? Complex.
    return absl::UnimplementedError("GetStorageStatistics not implemented");
  }

  // --- Helper for potentially stale metadata access ---
  Future<std::shared_ptr<const TiffMetadata>> ResolveMetadata(
      internal::OpenTransactionPtr transaction) {
    // Asynchronously read the directory cache entry, respecting staleness
    // bounds. Note: Transactions are not currently applied to metadata cache
    // reads here,
    //       pass `transaction` to Read if/when supported. For now, use nullptr.
    auto read_future =
        metadata_cache_entry_->Read({metadata_staleness_bound_.time});

    // Chain the metadata resolution logic onto the future.
    return MapFuture(
        this->data_copy_executor(),
        // Capture necessary members by value for the lambda.
        [this, tiff_options = this->tiff_options_,
         schema = this->schema_](const Result<void>& read_result)
            -> Result<std::shared_ptr<const TiffMetadata>> {
          // Check if the directory cache read succeeded.
          TENSORSTORE_RETURN_IF_ERROR(read_result);

          // Lock the directory cache entry to get the TiffParseResult.
          // Use the correct ReadData type for TiffDirectoryCache.
          auto lock = AsyncCache::ReadLock<
              const internal_tiff_kvstore::TiffParseResult>(
              *this->metadata_cache_entry_);
          auto parse_result_ptr = lock.shared_data();

          if (!parse_result_ptr) {
            return absl::NotFoundError(
                "TIFF parse result not found or failed to load.");
          }

          // Call the metadata resolution function using the (potentially
          // updated) parse result and the original options/schema stored in the
          // driver.
          TENSORSTORE_ASSIGN_OR_RETURN(
              auto resolved_metadata,
              internal_tiff::ResolveMetadata(*parse_result_ptr, tiff_options,
                                             schema));

          // TODO: Optionally compare resolved_metadata with initial_metadata_
          //       and return an error if incompatible changes occurred?
          //       For read-only, this might not be strictly necessary unless
          //       bounds changed in an unsupported way.

          return resolved_metadata;
        },
        std::move(read_future));
  }

  // Returns the transform from the external user view to the internal driver
  // view. For the base TIFF driver, this is typically identity.
  Result<IndexTransform<>> GetExternalToInternalTransform(
      const TiffMetadata& metadata, size_t component_index) const {
    ABSL_CHECK(component_index == 0);  // Expect only one component
    return IdentityTransform(metadata.rank);
  }

 private:
  friend class TiffDriverSpec;  // Allow Spec to call constructor/access members

  // Add as a private method to TiffDriver class:
  Result<IndexTransform<>> GetBoundSpecData(
      internal::OpenTransactionPtr transaction, TiffDriverSpec& spec,
      IndexTransformView<> transform) {
    // Get the metadata snapshot associated with this driver instance.
    // For generating a spec, using the initial metadata snapshot is
    // appropriate. Note: `GetMetadata()` uses `initial_metadata_` and is
    // synchronous.
    TENSORSTORE_ASSIGN_OR_RETURN(auto metadata, GetMetadata());

    // --- Populate Base KvsDriverSpec Members ---
    spec.context_binding_state_ = ContextBindingState::bound;

    // Get base KvStore spec from the TiffDirectoryCache driver
    // The TiffDirectoryCache holds the driver for the *underlying* store (e.g.,
    // file)
    auto& dir_cache = internal::GetOwningCache(*metadata_cache_entry_);
    TENSORSTORE_ASSIGN_OR_RETURN(spec.store.driver,
                                 dir_cache.kvstore_driver_->GetBoundSpec());
    // Use the directory cache entry's key as the base path for the spec.
    // This assumes the key represents the logical path to the TIFF data.
    spec.store.path = metadata_cache_entry_->key();

    // Copy stored context resources into the spec
    spec.data_copy_concurrency = this->data_copy_concurrency_;
    spec.cache_pool = this->cache_pool_;
    spec.metadata_cache_pool =
        this->metadata_cache_pool_;  // Copy optional resource

    // Copy staleness bounds and fill mode from driver state
    spec.staleness.data = this->data_staleness_bound();
    spec.staleness.metadata = this->metadata_staleness_bound_;
    spec.fill_value_mode = this->fill_value_mode_;

    // Set basic schema constraints from the resolved metadata
    // Only rank and dtype are typically set directly; others are derived via
    // GetEffective... methods when the spec is used/resolved.
    TENSORSTORE_RETURN_IF_ERROR(
        spec.schema.Set(RankConstraint{metadata->rank}));
    TENSORSTORE_RETURN_IF_ERROR(spec.schema.Set(metadata->dtype));
    // Copy the fill_value constraint from the driver's schema snapshot
    if (this->schema_.fill_value().valid()) {
      TENSORSTORE_RETURN_IF_ERROR(
          spec.schema.Set(Schema::FillValue(this->schema_.fill_value())));
    }
    // Note: We don't copy chunk_layout, codec, units directly here. They are
    // part of the overall schema constraints potentially stored in
    // `this->schema_` but are usually better represented via the
    // `GetChunkLayout()`, etc. overrides on the spec itself, which use the
    // `GetEffective...` functions.

    // --- Populate Derived TiffDriverSpec Members ---
    spec.tiff_options =
        this->tiff_options_;  // Copy original TIFF-specific options

    // Populate metadata constraints based on the *resolved* metadata state
    // This ensures the spec reflects the actual properties of the opened
    // driver.
    spec.metadata_constraints.rank = metadata->rank;
    spec.metadata_constraints.shape = metadata->shape;
    spec.metadata_constraints.dtype = metadata->dtype;
    // Note: Other constraints (chunking, units) aren't typically back-filled
    // from resolved metadata into the constraints section of the spec.

    // --- Calculate Final Transform ---
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto external_to_internal,
        GetExternalToInternalTransform(*metadata, component_index()));

    IndexTransform<> final_transform = transform;  // Create mutable copy

    // If the driver uses an internal transform (e.g., due to origin offsets
    // or dimension reordering not captured by the base TensorStore view),
    // compose the inverse of that transform with the input transform.
    if (external_to_internal.valid()) {
      TENSORSTORE_ASSIGN_OR_RETURN(auto internal_to_external,
                                   InverseTransform(external_to_internal));
      TENSORSTORE_ASSIGN_OR_RETURN(
          final_transform,
          ComposeTransforms(internal_to_external, std::move(final_transform)));
    }

    // Return the adjusted transform that maps from the user-specified domain
    // to the domain represented by the populated `driver_spec`.
    return final_transform;
  }

  StalenessBound metadata_staleness_bound_;
  internal::PinnedCacheEntry<internal_tiff_kvstore::TiffDirectoryCache>
      metadata_cache_entry_;
  internal_kvs_backed_chunk_driver::FillValueMode fill_value_mode_;
  std::shared_ptr<const TiffMetadata> initial_metadata_;
  TiffSpecOptions tiff_options_;
  Schema schema_;
  Context::Resource<internal::DataCopyConcurrencyResource>
      data_copy_concurrency_;
  Context::Resource<internal::CachePoolResource> cache_pool_;
  std::optional<Context::Resource<internal::CachePoolResource>>
      metadata_cache_pool_;
};  // End TiffDriver

// Helper function to create the ChunkGridSpecification from metadata.
// Constructs the grid based on logical dimensions identified by mapping_info.
Result<internal::ChunkGridSpecification> GetGridSpec(
    const TiffMetadata& metadata, const TiffGridMappingInfo& mapping_info) {
  internal::ChunkGridSpecification::ComponentList components;
  const DimensionIndex metadata_rank = metadata.rank;

  // --- Determine mapping from grid dimensions to component dimensions ---
  std::vector<DimensionIndex> chunked_to_cell_dims_vector;

  // Build chunked_to_cell_dims_vector based on identified logical dims
  // Order matters here: determines the order of grid dimensions
  if (mapping_info.ts_ifd_dim != -1) {  // IFD/Z dimension (if present)
    ABSL_CHECK(metadata.num_ifds > 1);
    chunked_to_cell_dims_vector.push_back(mapping_info.ts_ifd_dim);
  }
  if (mapping_info.ts_y_dim != -1) {  // Y dimension (if present)
    chunked_to_cell_dims_vector.push_back(mapping_info.ts_y_dim);
  }
  if (mapping_info.ts_x_dim != -1) {  // X dimension (if present)
    chunked_to_cell_dims_vector.push_back(mapping_info.ts_x_dim);
  } else if (metadata_rank > 0 && mapping_info.ts_y_dim == -1) {
    // Handle Rank 1 case where X is the only dimension
    chunked_to_cell_dims_vector.push_back(0);
  }
  // Rank 0 case results in empty chunked_to_cell_dims_vector (grid_rank = 0)

  // --- Prepare Component Specification ---

  // Create the fill value array
  SharedArray<const void> fill_value;
  if (metadata.fill_value.valid()) {
    fill_value = metadata.fill_value;
  } else {
    // Create a default (value-initialized) scalar fill value
    fill_value = AllocateArray(/*shape=*/span<const Index>{}, c_order,
                               value_init, metadata.dtype);
  }
  // Broadcast fill value to the full metadata shape
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto fill_value_array,  // SharedArray<const void>
      BroadcastArray(std::move(fill_value), BoxView<>(metadata.shape)));

  // Convert fill_value_array (zero-origin) to SharedOffsetArray
  SharedOffsetArray<const void> offset_fill_value(std::move(fill_value_array));

  // Determine layout order for the component data within chunks
  ContiguousLayoutOrder component_layout_order = metadata.layout_order;

  // Create the AsyncWriteArray::Spec
  internal::AsyncWriteArray::Spec array_spec{
      std::move(offset_fill_value),
      Box<>(metadata_rank),  // Component bounds (unbounded)
      component_layout_order};

  // Create the component's full chunk shape vector
  std::vector<Index> component_chunk_shape_vec(
      metadata.chunk_layout.read_chunk_shape().begin(),
      metadata.chunk_layout.read_chunk_shape().end());

  // Add the single component to the list
  components.emplace_back(
      std::move(array_spec), std::move(component_chunk_shape_vec),
      std::move(chunked_to_cell_dims_vector)  // Pass the mapping
  );

  // Construct ChunkGridSpecification using the single-argument constructor
  // It will deduce the grid's chunk_shape from the component list.
  return internal::ChunkGridSpecification(std::move(components));
}

struct TiffOpenState : public internal::AtomicReferenceCount<TiffOpenState> {
  internal::DriverOpenRequest request_;  // Move request in
  kvstore::Spec store_;
  Context::Resource<internal::DataCopyConcurrencyResource>
      data_copy_concurrency_;
  Context::Resource<internal::CachePoolResource> cache_pool_;
  std::optional<Context::Resource<internal::CachePoolResource>>
      metadata_cache_pool_;
  StalenessBounds staleness_;
  internal_kvs_backed_chunk_driver::FillValueMode fill_value_mode_;
  TiffSpecOptions tiff_options_;
  TiffMetadataConstraints metadata_constraints_;
  Schema schema_;
  absl::Time open_time_;
  Promise<internal::Driver::Handle> promise_;  // Final promise

  // Constructor captures spec members
  TiffOpenState(const TiffDriverSpec* spec, internal::DriverOpenRequest request)
      : request_(std::move(request)),
        store_(spec->store),
        data_copy_concurrency_(spec->data_copy_concurrency),
        cache_pool_(spec->cache_pool),
        metadata_cache_pool_(spec->metadata_cache_pool),
        staleness_(spec->staleness),
        fill_value_mode_(spec->fill_value_mode),
        tiff_options_(spec->tiff_options),
        metadata_constraints_(spec->metadata_constraints),
        schema_(spec->schema),
        open_time_(absl::Now()) {}

  // Initiates the open process
  void Start(Promise<internal::Driver::Handle> promise);

  // Callback when base KvStore is ready
  void OnKvStoreOpen(ReadyFuture<KvStore> future);

  // Callback when TiffDirectoryCache entry read is complete
  void OnDirCacheRead(
      KvStore base_kvstore,  // Pass needed results explicitly
      internal::PinnedCacheEntry<internal_tiff_kvstore::TiffDirectoryCache>
          metadata_cache_entry,
      ReadyFuture<const void> future);
};

void TiffOpenState::Start(Promise<internal::Driver::Handle> promise) {
  promise_ = std::move(promise);  // Store the final promise

  // Use LinkValue to link OnKvStoreOpen to the KvStore::Open future
  LinkValue(
      WithExecutor(
          data_copy_concurrency_->executor,  // Ensure callback runs on executor
          [self = internal::IntrusivePtr<TiffOpenState>(this)](
              Promise<internal::Driver::Handle> promise,  // Not used here
              ReadyFuture<KvStore> future) {
            // Note: promise passed to LinkValue is the final one,
            // which we stored in self->promise_.
            self->OnKvStoreOpen(std::move(future));
          }),
      promise_,  // Link potential errors from Open to final promise
      kvstore::Open(store_));
}

void TiffOpenState::OnKvStoreOpen(ReadyFuture<KvStore> future) {
  ABSL_LOG(INFO) << "TiffOpenState::OnKvStoreOpen";
  // Check if opening the base KvStore failed.
  Result<KvStore> base_kvstore_result = future.result();
  if (!base_kvstore_result.ok()) {
    promise_.SetResult(std::move(base_kvstore_result).status());
    return;
  }
  KvStore base_kvstore = *std::move(base_kvstore_result);

  // Determine the cache pool for metadata.
  const auto& metadata_pool_res =
      metadata_cache_pool_.has_value() ? *metadata_cache_pool_ : cache_pool_;

  auto* pool_ptr = metadata_pool_res->get();
  if (!pool_ptr) {
    promise_.SetResult(
        absl::InvalidArgumentError("Cache pool resource is null or invalid"));
    return;
  }

  // Create the cache key for the TiffDirectoryCache.
  std::string directory_cache_key;
  internal::EncodeCacheKey(&directory_cache_key, base_kvstore.driver,
                           data_copy_concurrency_);

  // Get or create the TiffDirectoryCache instance.
  auto directory_cache =
      internal::GetCache<internal_tiff_kvstore::TiffDirectoryCache>(
          pool_ptr, directory_cache_key, [&] {
            return std::make_unique<internal_tiff_kvstore::TiffDirectoryCache>(
                base_kvstore.driver, data_copy_concurrency_->executor);
          });

  // Get the specific cache entry for the TIFF file path.
  auto metadata_cache_entry =
      internal::GetCacheEntry(directory_cache, base_kvstore.path);

  // Initiate an asynchronous read on the directory cache entry.
  StalenessBound metadata_staleness_bound =
      staleness_.metadata.BoundAtOpen(open_time_);
  auto read_future =
      metadata_cache_entry->Read({metadata_staleness_bound.time});

  // Link the next step (OnDirCacheRead) to the completion of the read.
  LinkValue(
      WithExecutor(data_copy_concurrency_->executor,
                   // ---- FIX 2: Capture metadata_cache_entry by move ----
                   [self = internal::IntrusivePtr<TiffOpenState>(this),
                    base_kvstore = std::move(base_kvstore),
                    metadata_cache_entry = std::move(metadata_cache_entry)](
                       Promise<internal::Driver::Handle> promise,
                       ReadyFuture<const void> future) mutable {
                     self->OnDirCacheRead(std::move(base_kvstore),
                                          std::move(metadata_cache_entry),
                                          std::move(future));
                   }),
      promise_,  // Link errors to the final promise
      std::move(read_future));
}

void TiffOpenState::OnDirCacheRead(
    KvStore base_kvstore,
    internal::PinnedCacheEntry<internal_tiff_kvstore::TiffDirectoryCache>
        metadata_cache_entry,
    ReadyFuture<const void> future) {
  ABSL_LOG(INFO) << "TiffOpenState::OnDirCacheRead";

  // 1. Check if reading the directory cache failed.
  //    (Error already propagated by LinkError/LinkValue, but check anyway)
  if (!future.result().ok()) {
    // Error should have already been set on promise_, but double-check.
    if (promise_.result_needed()) {
      promise_.SetResult(metadata_cache_entry->AnnotateError(
          future.result().status(), /*reading=*/true));
    }
    return;
  }

  // 2. Lock the cache entry to access the parsed TiffParseResult.
  internal::AsyncCache::ReadLock<const internal_tiff_kvstore::TiffParseResult>
      lock(*metadata_cache_entry);
  auto parse_result = lock.shared_data();

  if (!parse_result) {
    // This case indicates an internal issue if the future succeeded.
    promise_.SetResult(absl::DataLossError(
        "TIFF directory cache entry data is null after successful read"));
    return;
  }

  // 3. Resolve the final TiffMetadata
  Result<std::shared_ptr<const TiffMetadata>> metadata_result =
      internal_tiff::ResolveMetadata(*parse_result, tiff_options_, schema_);
  if (!metadata_result.ok()) {
    promise_.SetResult(std::move(metadata_result).status());
    return;
  }
  std::shared_ptr<const TiffMetadata> metadata = *std::move(metadata_result);

  // 4. Validate the resolved metadata against user-provided constraints.
  absl::Status validate_status =
      internal_tiff::ValidateResolvedMetadata(*metadata, metadata_constraints_);
  if (!validate_status.ok()) {
    promise_.SetResult(internal::ConvertInvalidArgumentToFailedPrecondition(
        std::move(validate_status)));
    return;
  }

  // 5. Validate against read/write mode (TIFF is read-only for now)
  if (request_.read_write_mode != ReadWriteMode::read &&
      request_.read_write_mode != ReadWriteMode::dynamic) {
    promise_.SetResult(
        absl::InvalidArgumentError("TIFF driver only supports read mode"));
    return;
  }
  ReadWriteMode driver_read_write_mode = ReadWriteMode::read;  // Hardcoded

  // ---- 6. Create TiffChunkCache ----

  // 6a. Get the TiffKeyValueStore driver instance.
  auto tiff_kvstore_driver =
      kvstore::tiff_kvstore::GetTiffKeyValueStore(base_kvstore.driver);
  if (!tiff_kvstore_driver) {
    promise_.SetResult(
        absl::InternalError("Failed to get TiffKeyValueStore driver"));
    return;
  }

  // 6b. Get the ChunkGridSpecification.
  TiffGridMappingInfo mapping_info = GetTiffGridMappingInfo(*metadata);
  Result<internal::ChunkGridSpecification> grid_spec_result =
      GetGridSpec(*metadata, mapping_info);
  if (!grid_spec_result.ok()) {
    promise_.SetResult(std::move(grid_spec_result).status());
    return;
  }
  internal::ChunkGridSpecification grid_spec = *std::move(grid_spec_result);

  // 6c. Create the cache key for TiffChunkCache.
  std::string chunk_cache_key;
  // Simple key based on the metadata cache entry key and metadata properties.

  std::string metadata_compat_key = absl::StrFormat(
      "ifd%d_dtype%s_comp%d_planar%d_spp%d", metadata->ifd_index,
      metadata->dtype.name(), static_cast<int>(metadata->compression_type),
      static_cast<int>(metadata->planar_config), metadata->samples_per_pixel);

  internal::EncodeCacheKey(
      &chunk_cache_key,
      metadata_cache_entry->key(),  // Use original path key
      metadata_compat_key,
      cache_pool_->get());  // Include data cache pool

  // 6d. Get or create the TiffChunkCache.
  auto chunk_cache = internal::GetCache<TiffChunkCache>(
      cache_pool_->get(), chunk_cache_key, [&] {
        // Factory to create the TiffChunkCache.
        // Pass copies/moved values needed by the cache constructor.
        return std::make_unique<TiffChunkCache>(
            tiff_kvstore_driver,  // Use the specific TIFF KvStore driver
            metadata,             // Pass the resolved metadata
            grid_spec,            // Pass the generated grid spec
            data_copy_concurrency_->executor);
      });
  if (!chunk_cache) {
    promise_.SetResult(
        absl::InternalError("Failed to get or create TiffChunkCache"));
    return;
  }

  // ---- 7. Create TiffDriver ----
  TiffDriverInitializer driver_initializer{
      /*.cache=*/std::move(chunk_cache),
      /*.component_index=*/0,  // Always 0 for TIFF
      /*.data_staleness_bound=*/staleness_.data.BoundAtOpen(open_time_),
      /*.metadata_staleness_bound=*/staleness_.metadata.BoundAtOpen(open_time_),
      /*.metadata_cache_entry=*/std::move(metadata_cache_entry),  // Move
                                                                  // ownership
      /*.fill_value_mode=*/fill_value_mode_,
      /*.initial_metadata=*/metadata,  // Store the resolved metadata
      /*.tiff_options=*/tiff_options_,
      /*.schema=*/schema_,  // Store original schema constraints
      /*.data_copy_concurrency=*/data_copy_concurrency_,
      /*.cache_pool=*/cache_pool_,
      /*.metadata_cache_pool=*/metadata_cache_pool_};

  // Use MakeIntrusivePtr for the driver
  auto driver =
      internal::MakeIntrusivePtr<TiffDriver>(std::move(driver_initializer));

  // ---- 8. Finalize: Get Transform and Set Promise ----

  // Get the initial transform (likely identity for TIFF base driver).
  // Use the resolved metadata stored within the newly created driver instance.
  Result<IndexTransform<>> transform_result =
      driver->GetExternalToInternalTransform(
          *metadata, 0);  // Use metadata passed to driver
  if (!transform_result.ok()) {
    promise_.SetResult(std::move(transform_result).status());
    return;
  }

  // Fulfill the final promise with the driver handle.
  internal::Driver::Handle handle{internal::ReadWritePtr<internal::Driver>(
                                      driver.get(), driver_read_write_mode),
                                  std::move(*transform_result),
                                  internal::TransactionState::ToTransaction(
                                      std::move(request_.transaction))};

  promise_.SetResult(std::move(handle));
}

Future<internal::Driver::Handle> TiffDriverSpec::Open(
    internal::DriverOpenRequest request) const {
  if (!store.valid()) {
    return absl::InvalidArgumentError("\"kvstore\" must be specified");
  }
  TENSORSTORE_RETURN_IF_ERROR(
      this->OpenModeSpec::Validate(request.read_write_mode));

  // Create the state object, transferring ownership of spec parts.
  // MakeIntrusivePtr handles the reference counting.
  auto state =
      internal::MakeIntrusivePtr<TiffOpenState>(this, std::move(request));

  // Create the final promise/future pair.
  auto [promise, future] = PromiseFuturePair<internal::Driver::Handle>::Make();

  // Start the asynchronous open process by calling the first step function.
  state->Start(std::move(promise));

  // Return the future to the caller.
  return std::move(future);
}

#ifndef _MSC_VER
}  // namespace
#endif

}  // namespace internal_tiff
}  // namespace tensorstore

TENSORSTORE_DECLARE_GARBAGE_COLLECTION_SPECIALIZATION(
    tensorstore::internal_tiff::TiffDriver)

TENSORSTORE_DEFINE_GARBAGE_COLLECTION_SPECIALIZATION(
    tensorstore::internal_tiff::TiffDriver,
    tensorstore::garbage_collection::PolymorphicGarbageCollection<
        tensorstore::internal_tiff::TiffDriver>)

namespace {
const tensorstore::internal::DriverRegistration<
    tensorstore::internal_tiff::TiffDriverSpec>
    registration;
}  // namespace