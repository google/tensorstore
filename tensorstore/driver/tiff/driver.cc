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
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/base/optimization.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "tensorstore/array.h"
#include "tensorstore/chunk_layout.h"
#include "tensorstore/driver/chunk_cache_driver.h"
#include "tensorstore/driver/driver_spec.h"
#include "tensorstore/driver/kvs_backed_chunk_driver.h"
#include "tensorstore/driver/registry.h"
#include "tensorstore/driver/tiff/metadata.h"
#include "tensorstore/index_space/index_domain_builder.h"
#include "tensorstore/index_space/internal/propagate_bounds.h"
#include "tensorstore/internal/cache/async_cache.h"
#include "tensorstore/internal/cache/cache.h"
#include "tensorstore/internal/cache/kvs_backed_chunk_cache.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_binding/staleness_bound.h"  // IWYU: pragma keep
#include "tensorstore/kvstore/driver.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/tiff/tiff_key_value_store.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/execution/execution.h"
#include "tensorstore/util/garbage_collection/fwd.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

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
  // Hot‑path data we compute once and then reuse for every call.
  struct FastPath {
    DimensionIndex y_grid_dim = -1;
    DimensionIndex x_grid_dim = -1;
    DimensionIndex sample_grid_dim = -1;

    //  Stack label to grid dimension
    absl::flat_hash_map<std::string_view, DimensionIndex> stack_to_grid;

    //  Stack label to size
    absl::flat_hash_map<std::string_view, Index> stack_size;

    //  Stack label to stride
    absl::flat_hash_map<std::string_view, uint64_t> stack_stride;

    //  Geometry derived from metadata
    Index num_cols = 0;              // tiles/strips per row
    Index num_chunks_per_plane = 0;  // planar‑config adjustment
  };

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

  void InitFastPath() {
    fast_ = std::make_unique<FastPath>();
    const auto& metadata = *resolved_metadata_;
    const auto& grid_spec = this->grid();
    const auto& mapping = metadata.dimension_mapping;
    const DimensionIndex grid_rank = grid_spec.grid_rank();

    const auto& chunked_to_cell =
        grid_spec.components[0].chunked_to_cell_dimensions;

    // Helper lambda to find index of a label in a vector
    auto find_index = [](const std::vector<std::string>& vec,
                         std::string_view label) {
      return static_cast<size_t>(std::find(vec.begin(), vec.end(), label) -
                                 vec.begin());
    };

    // Classify grid dimensions
    for (DimensionIndex g = 0; g < grid_rank; ++g) {
      const DimensionIndex ts_dim = chunked_to_cell[g];
      if (mapping.ts_y_dim == ts_dim) {
        fast_->y_grid_dim = g;
      } else if (mapping.ts_x_dim == ts_dim) {
        fast_->x_grid_dim = g;
      } else if (mapping.ts_sample_dim == ts_dim) {
        fast_->sample_grid_dim = g;
      } else {
        std::string_view label = mapping.labels_by_ts_dim[ts_dim];
        fast_->stack_to_grid[label] = g;
      }
    }

    // Pre‑compute strides for stacked dimensions
    if (metadata.stacking_info) {
      const auto& stacking_info = *metadata.stacking_info;
      const auto& sizes = *stacking_info.dimension_sizes;
      const auto& order = stacking_info.ifd_sequence_order
                              ? *stacking_info.ifd_sequence_order
                              : stacking_info.dimensions;

      uint64_t stride = 1;
      for (int i = static_cast<int>(order.size()) - 1; i >= 0; --i) {
        std::string_view label = order[i];
        fast_->stack_stride[label] = stride;
        size_t idx = find_index(stacking_info.dimensions, label);
        fast_->stack_size[label] = sizes[idx];
        stride *= static_cast<uint64_t>(sizes[idx]);
      }
    }

    // Geometry that never changes
    const Index chunk_width = metadata.ifd0_chunk_width;
    const Index chunk_height = metadata.ifd0_chunk_height;
    const Index image_width = metadata.shape[*mapping.ts_x_dim];
    const Index image_height = metadata.shape[*mapping.ts_y_dim];

    fast_->num_cols = (image_width + chunk_width - 1) / chunk_width;
    if (metadata.is_tiled) {
      const Index num_rows = (image_height + chunk_height - 1) / chunk_height;
      fast_->num_chunks_per_plane = num_rows * fast_->num_cols;
    } else {
      fast_->num_chunks_per_plane =
          (image_height + chunk_height - 1) / chunk_height;
    }
  }

  std::string GetChunkStorageKey(span<const Index> cell_indices) override {
    using internal_tiff_kvstore::PlanarConfigType;
    if (!fast_) {
      InitFastPath();
    }

    const FastPath& fast = *fast_;
    const auto& metadata = *resolved_metadata_;

    // Determine the target IFD index.
    uint32_t target_ifd_index = metadata.base_ifd_index;

    if (metadata.stacking_info) {
      const auto& stacking_info = *metadata.stacking_info;
      const auto& ifd_iteration_order =
          stacking_info.ifd_sequence_order.value_or(stacking_info.dimensions);

      for (std::string_view stack_label : ifd_iteration_order) {
        auto grid_dim_it = fast.stack_to_grid.find(stack_label);
        if (ABSL_PREDICT_FALSE(grid_dim_it == fast.stack_to_grid.end())) {
          ABSL_LOG(FATAL) << "Stacking dimension label '" << stack_label
                          << "' not found in grid specification.";
        }

        DimensionIndex grid_dimension_index = grid_dim_it->second;
        uint64_t dimension_stride = fast.stack_stride.find(stack_label)->second;

        target_ifd_index += static_cast<uint32_t>(
            cell_indices[grid_dimension_index] * dimension_stride);
      }
    }

    // Compute the linear chunk index within the chosen IFD.
    Index y_chunk_index =
        (fast.y_grid_dim >= 0) ? cell_indices[fast.y_grid_dim] : 0;
    Index x_chunk_index =
        (fast.x_grid_dim >= 0) ? cell_indices[fast.x_grid_dim] : 0;

    uint64_t linear_chunk_index =
        metadata.is_tiled
            ? static_cast<uint64_t>(y_chunk_index) * fast.num_cols +
                  x_chunk_index
            : static_cast<uint64_t>(y_chunk_index);

    // Planar‑configuration adjustment: add an offset for the sample plane.
    if (metadata.planar_config == PlanarConfigType::kPlanar &&
        metadata.samples_per_pixel > 1) {
      Index sample_plane_index = cell_indices[fast.sample_grid_dim];
      linear_chunk_index +=
          static_cast<uint64_t>(sample_plane_index) * fast.num_chunks_per_plane;
    }

    // Assemble the final storage‑key string.
    auto storage_key = tensorstore::StrCat("chunk/", target_ifd_index, "/",
                                           linear_chunk_index);
    return storage_key;
  }

  // Decodes chunk data (called by Entry::DoDecode indirectly).
  Result<absl::InlinedVector<SharedArray<const void>, 1>> DecodeChunk(
      span<const Index> chunk_indices, absl::Cord data) override {
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
  };

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
  std::unique_ptr<FastPath> fast_;
};

// Validator function for positive integers
template <typename T>
absl::Status ValidatePositive(const T& value) {
  if (value <= 0) {
    return absl::InvalidArgumentError("Value must be positive");
  }
  return absl::OkStatus();
}

// TiffDriverSpec: Defines the specification for opening a TIFF TensorStore.
class TiffDriverSpec
    : public internal::RegisteredDriverSpec<TiffDriverSpec, KvsDriverSpec> {
 public:
  constexpr static char id[] = "tiff";
  using Base = internal::RegisteredDriverSpec<TiffDriverSpec, KvsDriverSpec>;

  TiffSpecOptions tiff_options;
  TiffMetadataConstraints metadata_constraints;

  constexpr static auto ApplyMembers = [](auto& x, auto f) {
    return f(internal::BaseCast<KvsDriverSpec>(x), x.tiff_options,
             x.metadata_constraints);
  };

  static inline const auto default_json_binder =
      jb::Sequence(
          // Copied from kvs_backed_chunk_driver::KvsDriverSpec because
          // KvsDriverSpec::store initializer was enforcing directory path.
          jb::Member(internal::DataCopyConcurrencyResource::id,
                     jb::Projection<&KvsDriverSpec::data_copy_concurrency>()),
          jb::Member(internal::CachePoolResource::id,
                     jb::Projection<&KvsDriverSpec::cache_pool>()),
          jb::Member("metadata_cache_pool",
                     jb::Projection<&KvsDriverSpec::metadata_cache_pool>()),
          jb::Projection<&KvsDriverSpec::store>(
              jb::KvStoreSpecAndPathJsonBinder),
          jb::Initialize([](auto* obj) { return absl::OkStatus(); }),
          jb::Projection<&KvsDriverSpec::staleness>(jb::Sequence(
              jb::Member("recheck_cached_metadata",
                         jb::Projection(&StalenessBounds::metadata,
                                        jb::DefaultValue([](auto* obj) {
                                          obj->bounded_by_open_time = true;
                                        }))),
              jb::Member("recheck_cached_data",
                         jb::Projection(&StalenessBounds::data,
                                        jb::DefaultInitializedValue())))),
          jb::Projection<&KvsDriverSpec::fill_value_mode>(
              jb::Sequence(
                  jb::Member(
                      "fill_missing_data_reads",
                      jb::Projection<
                          &internal_kvs_backed_chunk_driver::FillValueMode::
                              fill_missing_data_reads>(
                          jb::DefaultValue([](auto* obj) { *obj = true; }))),
                  jb::Member(
                      "store_data_equal_to_fill_value",
                      jb::Projection<
                          &internal_kvs_backed_chunk_driver::FillValueMode::
                              store_data_equal_to_fill_value>(
                          jb::DefaultInitializedValue())))),
          internal::OpenModeSpecJsonBinder,
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
          jb::Member("tiff", jb::
                                 Projection<&TiffDriverSpec::tiff_options>(
                                     jb::DefaultValue(
                                         [](auto* v) { *v = {}; }))) /*,
  // Final validation combining spec parts
  jb::Validate([](const auto& options, auto* obj) -> absl::Status {
    // Enforce mutual exclusion: if ifd_stacking is present, ifd_index must
    // be 0. Note: binder for "ifd" already ensures it's >= 0.
    if (obj->tiff_options.ifd_stacking &&
        obj->tiff_options.ifd_index != 0) {
      return absl::InvalidArgumentError(
          "Cannot specify both \"ifd\" (non-zero) and \"ifd_stacking\" in "
          "\"tiff\" options");
    }
    // Validate sample_dimension_label against stacking dimensions
    if (obj->tiff_options.ifd_stacking &&
        obj->tiff_options.sample_dimension_label) {
      const auto& stack_dims = obj->tiff_options.ifd_stacking->dimensions;
      if (std::find(stack_dims.begin(), stack_dims.end(),
                    *obj->tiff_options.sample_dimension_label) !=
          stack_dims.end()) {
        return absl::InvalidArgumentError(tensorstore::StrCat(
            "\"sample_dimension_label\" (\"",
            *obj->tiff_options.sample_dimension_label,
            "\") conflicts with a label in \"ifd_stacking.dimensions\""));
      }
    }
    // Validate schema dtype if specified
    if (obj->schema.dtype().valid()) {
      TENSORSTORE_RETURN_IF_ERROR(ValidateDataType(obj->schema.dtype()));
    }
    return absl::OkStatus();
  })*/);

  Result<IndexDomain<>> GetDomain() const override {
    return internal_tiff::GetEffectiveDomain(metadata_constraints, schema);
  }

  Result<CodecSpec> GetCodec() const override {
    CodecSpec codec_constraint = schema.codec();
    auto tiff_codec = internal::CodecDriverSpec::Make<TiffCodecSpec>();

    if (codec_constraint.valid()) {
      TENSORSTORE_RETURN_IF_ERROR(
          tiff_codec->MergeFrom(codec_constraint),
          MaybeAnnotateStatus(
              _, "Cannot merge schema codec constraints with tiff driver"));
    }
    return CodecSpec(std::move(tiff_codec));
  }

  Result<ChunkLayout> GetChunkLayout() const override {
    return schema.chunk_layout();
  }

  Result<SharedArray<const void>> GetFillValue(
      IndexTransformView<> transform) const override {
    return schema.fill_value().valid()
               ? tensorstore::Result<SharedArray<const void>>(
                     schema.fill_value())
               : tensorstore::Result<SharedArray<const void>>{std::in_place};
  }

  Result<DimensionUnitsVector> GetDimensionUnits() const override {
    DimensionIndex rank = schema.rank().rank;
    if (metadata_constraints.rank != dynamic_rank) {
      if (rank != dynamic_rank && rank != metadata_constraints.rank) {
        return absl::InvalidArgumentError(tensorstore::StrCat(
            "Rank specified in schema (", rank,
            ") conflicts with rank specified in metadata constraints (",
            metadata_constraints.rank, ")"));
      }
      rank = metadata_constraints.rank;
    }
    if (rank == dynamic_rank && metadata_constraints.shape.has_value()) {
      rank = metadata_constraints.shape->size();
    }
    if (rank == dynamic_rank && schema.domain().valid()) {
      rank = schema.domain().rank();
    }
    return internal_tiff::GetEffectiveDimensionUnits(rank, schema);
  }

  absl::Status ApplyOptions(SpecOptions&& options) override {
    if (options.minimal_spec) {
      metadata_constraints = {};
      tiff_options = {};
    }
    TENSORSTORE_RETURN_IF_ERROR(Base::ApplyOptions(std::move(options)));
    return absl::OkStatus();
  }

  Future<internal::Driver::Handle> Open(
      internal::DriverOpenRequest request) const override;
};

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
  std::optional<Context::Resource<internal::CachePoolResource>>
      metadata_cache_pool;
};

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

  Result<internal::TransformedDriverSpec> GetBoundSpec(
      internal::OpenTransactionPtr transaction,
      IndexTransformView<> transform) override {
    auto spec = internal::DriverSpec::Make<TiffDriverSpec>();

    TENSORSTORE_ASSIGN_OR_RETURN(
        auto final_transform,
        GetBoundSpecData(std::move(transaction), *spec, transform));

    return internal::TransformedDriverSpec{std::move(spec),
                                           std::move(final_transform)};
  }

  void GarbageCollectionVisit(
      garbage_collection::GarbageCollectionVisitor& visitor) const override {
    Base::GarbageCollectionVisit(visitor);
    garbage_collection::GarbageCollectionVisit(visitor, metadata_cache_entry_);
  }

  Result<ChunkLayout> GetChunkLayout(IndexTransformView<> transform) override {
    const auto& metadata = *initial_metadata_;

    TENSORSTORE_ASSIGN_OR_RETURN(
        auto layout,
        ApplyInverseIndexTransform(transform, metadata.chunk_layout));

    TENSORSTORE_RETURN_IF_ERROR(layout.Finalize());
    return layout;
  }

  Result<CodecSpec> GetCodec() override {
    const auto& metadata = *initial_metadata_;
    auto codec_spec = internal::CodecDriverSpec::Make<TiffCodecSpec>();
    codec_spec->compressor = metadata.compressor;
    return CodecSpec(std::move(codec_spec));
  }

  Result<SharedArray<const void>> GetFillValue(
      IndexTransformView<> transform) override {
    return {std::in_place};
  }

  Result<DimensionUnitsVector> GetDimensionUnits() override {
    TENSORSTORE_ASSIGN_OR_RETURN(auto metadata, GetMetadata());
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

  void Read(ReadRequest request, ReadChunkReceiver receiver) override {
    // Replicate logic from ChunkCacheReadWriteDriverMixin
    cache()->Read({std::move(request), component_index(),
                   this->data_staleness_bound().time,
                   fill_value_mode_.fill_missing_data_reads},
                  std::move(receiver));
  }

  void Write(WriteRequest request, WriteChunkReceiver receiver) override {
    execution::set_error(receiver,
                         absl::UnimplementedError("TIFF driver is read-only"));
  }

  Future<IndexTransform<>> ResolveBounds(
      ResolveBoundsRequest request) override {
    // Asynchronously resolve the metadata first.
    return MapFuture(
        this->data_copy_executor(),
        [transform = std::move(request.transform),
         options = std::move(request.options)](
            const Result<std::shared_ptr<const TiffMetadata>>&
                metadata_result) mutable -> Result<IndexTransform<>> {
          TENSORSTORE_RETURN_IF_ERROR(metadata_result);
          const auto& metadata = *metadata_result.value();

          // The authoritative domain is defined by the metadata's shape.
          // TIFF files inherently have a zero origin.
          BoxView<> domain(metadata.shape);

          DimensionSet implicit_lower_bounds(
              false);  // Always explicit 0 lower bound
          DimensionSet implicit_upper_bounds(
              false);  // Assume fixed upper bounds initially

          if (!(options.mode & fix_resizable_bounds)) {
            // If fix_resizable_bounds is *not* set, treat upper bounds
            // as implicit. Questionable for TIFF...
            implicit_upper_bounds = true;
          }

          return PropagateBoundsToTransform(domain, implicit_lower_bounds,
                                            implicit_upper_bounds,
                                            std::move(transform));
        },
        ResolveMetadata(std::move(request.transaction)));
  }

  Future<IndexTransform<>> Resize(ResizeRequest request) override {
    return absl::UnimplementedError("Resize is not supported by TIFF driver");
  }

  Future<ArrayStorageStatistics> GetStorageStatistics(
      GetStorageStatisticsRequest request) override {
    // TODO(hsidky): Implement GetStorageStatistics.
    // Might involve iterating keys in TiffKvStore? Complex.
    return absl::UnimplementedError("GetStorageStatistics not implemented");
  }

  // --- Helper for potentially stale metadata access ---
  Future<std::shared_ptr<const TiffMetadata>> ResolveMetadata(
      internal::OpenTransactionPtr transaction) {
    // TODO: Transactions are not currently applied to metadata cache
    auto read_future =
        metadata_cache_entry_->Read({metadata_staleness_bound_.time});

    // Chain the metadata resolution logic onto the future.
    return MapFuture(
        this->data_copy_executor(),
        [this, tiff_options = this->tiff_options_,
         schema = this->schema_](const Result<void>& read_result)
            -> Result<std::shared_ptr<const TiffMetadata>> {
          TENSORSTORE_RETURN_IF_ERROR(read_result);

          // Lock the directory cache entry to get the TiffParseResult.
          auto lock = AsyncCache::ReadLock<
              const internal_tiff_kvstore::TiffParseResult>(
              *this->metadata_cache_entry_);
          auto parse_result_ptr = lock.shared_data();

          if (!parse_result_ptr) {
            return absl::NotFoundError(
                "TIFF parse result not found or failed to load.");
          }

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
    ABSL_CHECK(component_index == 0);
    // Assumes zero origin, adjust if needed for OME-TIFF etc. later.
    TENSORSTORE_ASSIGN_OR_RETURN(auto domain,
                                 IndexDomainBuilder(metadata.rank)
                                     .shape(metadata.shape)
                                     .labels(metadata.dimension_labels)
                                     .Finalize());
    return IdentityTransform(domain);
  }

 private:
  friend class TiffDriverSpec;

  Result<IndexTransform<>> GetBoundSpecData(
      internal::OpenTransactionPtr transaction, TiffDriverSpec& spec,
      IndexTransformView<> transform) {
    TENSORSTORE_ASSIGN_OR_RETURN(auto metadata, GetMetadata());

    spec.context_binding_state_ = ContextBindingState::bound;
    auto& dir_cache = internal::GetOwningCache(*metadata_cache_entry_);
    TENSORSTORE_ASSIGN_OR_RETURN(spec.store.driver,
                                 dir_cache.kvstore_driver_->GetBoundSpec());
    // Use the directory cache entry's key as the base path for the spec.
    // This assumes the key represents the logical path to the TIFF data.
    spec.store.path = metadata_cache_entry_->key();

    spec.data_copy_concurrency = this->data_copy_concurrency_;
    spec.cache_pool = this->cache_pool_;
    spec.metadata_cache_pool = this->metadata_cache_pool_;
    spec.staleness.data = this->data_staleness_bound();
    spec.staleness.metadata = this->metadata_staleness_bound_;
    spec.fill_value_mode = this->fill_value_mode_;

    TENSORSTORE_RETURN_IF_ERROR(
        spec.schema.Set(RankConstraint{metadata->rank}));
    TENSORSTORE_RETURN_IF_ERROR(spec.schema.Set(metadata->dtype));
    if (this->schema_.fill_value().valid()) {
      TENSORSTORE_RETURN_IF_ERROR(
          spec.schema.Set(Schema::FillValue(this->schema_.fill_value())));
    }

    // Copy original TIFF-specific options
    spec.tiff_options = this->tiff_options_;
    spec.metadata_constraints.rank = metadata->rank;
    spec.metadata_constraints.shape = metadata->shape;
    spec.metadata_constraints.dtype = metadata->dtype;

    TENSORSTORE_ASSIGN_OR_RETURN(
        auto external_to_internal,
        GetExternalToInternalTransform(*metadata, component_index()));

    IndexTransform<> final_transform = transform;

    // If the driver uses an internal transform compose the inverse of that
    // transform with the input transform.
    if (external_to_internal.valid()) {
      TENSORSTORE_ASSIGN_OR_RETURN(auto internal_to_external,
                                   InverseTransform(external_to_internal));
      TENSORSTORE_ASSIGN_OR_RETURN(
          final_transform,
          ComposeTransforms(internal_to_external, std::move(final_transform)));
    }

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
};

/// Creates the ChunkGridSpecification based on the resolved TIFF metadata.
Result<internal::ChunkGridSpecification> GetGridSpec(
    const TiffMetadata& metadata) {
  using internal::AsyncWriteArray;
  using internal::ChunkGridSpecification;
  using internal_tiff_kvstore::PlanarConfigType;

  const DimensionIndex rank = metadata.rank;
  if (rank == dynamic_rank) {
    return absl::InvalidArgumentError(
        "Cannot determine grid with unknown rank");
  }

  ChunkGridSpecification::ComponentList components;
  std::vector<DimensionIndex> chunked_to_cell_dimensions;

  // Determine which final dimensions correspond to the grid axes.
  // Order: Stacked dims, Y, X, Sample (if planar)
  if (metadata.stacking_info) {
    // Use the sequence order if specified, otherwise use dimension order
    const auto& stack_dims_in_final_order = metadata.stacking_info->dimensions;
    const auto& sequence = metadata.stacking_info->ifd_sequence_order.value_or(
        stack_dims_in_final_order);
    for (const auto& label : sequence) {
      auto it = metadata.dimension_mapping.ts_stacked_dims.find(label);
      if (it != metadata.dimension_mapping.ts_stacked_dims.end()) {
        chunked_to_cell_dimensions.push_back(it->second);
      } else {
        return absl::InternalError(tensorstore::StrCat(
            "Stacking dimension '", label,
            "' specified in sequence_order/dimensions not found in "
            "final mapping"));
      }
    }
  }
  if (metadata.dimension_mapping.ts_y_dim.has_value()) {
    chunked_to_cell_dimensions.push_back(*metadata.dimension_mapping.ts_y_dim);
  }
  if (metadata.dimension_mapping.ts_x_dim.has_value()) {
    chunked_to_cell_dimensions.push_back(*metadata.dimension_mapping.ts_x_dim);
  }
  // Add Sample dimension to the grid ONLY if Planar
  if (metadata.planar_config == PlanarConfigType::kPlanar &&
      metadata.dimension_mapping.ts_sample_dim.has_value()) {
    chunked_to_cell_dimensions.push_back(
        *metadata.dimension_mapping.ts_sample_dim);
  }

  const DimensionIndex grid_rank = chunked_to_cell_dimensions.size();
  if (grid_rank == 0 && rank > 0) {
    // Check if the only dimension is a non-grid Sample dimension (chunky, spp >
    // 1, rank 1)
    if (rank == 1 && metadata.dimension_mapping.ts_sample_dim.has_value() &&
        metadata.planar_config == PlanarConfigType::kChunky) {
      // This is valid (e.g., just a list of RGB values), grid rank is 0
    } else {
      return absl::InternalError(
          "Calculated grid rank is 0 but overall rank > 0 and not solely a "
          "sample dimension");
    }
  }
  if (grid_rank > rank) {
    // Sanity check
    return absl::InternalError("Calculated grid rank exceeds overall rank");
  }

  SharedArray<const void> fill_value;
  if (metadata.fill_value.valid()) {
    fill_value = metadata.fill_value;
  } else {
    fill_value = AllocateArray(/*shape=*/span<const Index>{}, c_order,
                               value_init, metadata.dtype);
  }
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto fill_value_array,
      BroadcastArray(std::move(fill_value), BoxView<>(metadata.shape)));
  SharedOffsetArray<const void> offset_fill_value(std::move(fill_value_array));

  Box<> component_bounds(rank);

  ContiguousLayoutOrder component_layout_order = metadata.layout_order;

  AsyncWriteArray::Spec array_spec{std::move(offset_fill_value),
                                   std::move(component_bounds),
                                   component_layout_order};

  std::vector<Index> component_chunk_shape_vec(
      metadata.chunk_layout.read_chunk_shape().begin(),
      metadata.chunk_layout.read_chunk_shape().end());

  components.emplace_back(std::move(array_spec),
                          std::move(component_chunk_shape_vec),
                          std::move(chunked_to_cell_dimensions));

  return ChunkGridSpecification(std::move(components));
}

struct TiffOpenState : public internal::AtomicReferenceCount<TiffOpenState> {
  internal::DriverOpenRequest request_;
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
  Promise<internal::Driver::Handle> promise_;

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
      KvStore base_kvstore,
      internal::PinnedCacheEntry<internal_tiff_kvstore::TiffDirectoryCache>
          metadata_cache_entry,
      ReadyFuture<const void> future);
};

void TiffOpenState::Start(Promise<internal::Driver::Handle> promise) {
  promise_ = std::move(promise);

  LinkValue(WithExecutor(data_copy_concurrency_->executor,
                         [self = internal::IntrusivePtr<TiffOpenState>(this)](
                             Promise<internal::Driver::Handle> promise,
                             ReadyFuture<KvStore> future) {
                           self->OnKvStoreOpen(std::move(future));
                         }),
            promise_, kvstore::Open(store_));
}

void TiffOpenState::OnKvStoreOpen(ReadyFuture<KvStore> future) {
  Result<KvStore> base_kvstore_result = future.result();
  if (!base_kvstore_result.ok()) {
    promise_.SetResult(std::move(base_kvstore_result).status());
    return;
  }
  KvStore base_kvstore = *std::move(base_kvstore_result);

  const auto& metadata_pool_res =
      metadata_cache_pool_.has_value() ? *metadata_cache_pool_ : cache_pool_;

  auto* pool_ptr = metadata_pool_res->get();
  if (!pool_ptr) {
    promise_.SetResult(
        absl::InvalidArgumentError("Cache pool resource is null or invalid"));
    return;
  }

  std::string directory_cache_key;
  internal::EncodeCacheKey(&directory_cache_key, base_kvstore.driver,
                           data_copy_concurrency_);

  auto directory_cache =
      internal::GetCache<internal_tiff_kvstore::TiffDirectoryCache>(
          pool_ptr, directory_cache_key, [&] {
            return std::make_unique<internal_tiff_kvstore::TiffDirectoryCache>(
                base_kvstore.driver, data_copy_concurrency_->executor);
          });

  auto metadata_cache_entry =
      internal::GetCacheEntry(directory_cache, base_kvstore.path);

  StalenessBound metadata_staleness_bound =
      staleness_.metadata.BoundAtOpen(open_time_);
  auto read_future =
      metadata_cache_entry->Read({metadata_staleness_bound.time});

  LinkValue(
      WithExecutor(data_copy_concurrency_->executor,
                   [self = internal::IntrusivePtr<TiffOpenState>(this),
                    base_kvstore = std::move(base_kvstore),
                    metadata_cache_entry = std::move(metadata_cache_entry)](
                       Promise<internal::Driver::Handle> promise,
                       ReadyFuture<const void> future) mutable {
                     self->OnDirCacheRead(std::move(base_kvstore),
                                          std::move(metadata_cache_entry),
                                          std::move(future));
                   }),
      promise_, std::move(read_future));
}

void TiffOpenState::OnDirCacheRead(
    KvStore base_kvstore,
    internal::PinnedCacheEntry<internal_tiff_kvstore::TiffDirectoryCache>
        metadata_cache_entry,
    ReadyFuture<const void> future) {
  // 1. Check if reading the directory cache failed.
  if (!future.result().ok()) {
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

  // 5. Validate against read/write mode (TIFF is read-only)
  if (request_.read_write_mode != ReadWriteMode::read &&
      request_.read_write_mode != ReadWriteMode::dynamic) {
    promise_.SetResult(
        absl::InvalidArgumentError("TIFF driver only supports read mode"));
    return;
  }
  ReadWriteMode driver_read_write_mode = ReadWriteMode::read;  // Hardcoded

  //  6. Create TiffChunkCache
  Result<kvstore::DriverPtr> tiff_kvstore_driver_result =
      kvstore::tiff_kvstore::GetTiffKeyValueStoreDriver(
          base_kvstore.driver, base_kvstore.path, cache_pool_,
          data_copy_concurrency_, metadata_cache_entry);

  if (!tiff_kvstore_driver_result.ok()) {
    promise_.SetResult(std::move(tiff_kvstore_driver_result).status());
    return;
  }
  kvstore::DriverPtr tiff_kvstore_driver =
      *std::move(tiff_kvstore_driver_result);

  auto grid_spec_result = GetGridSpec(*metadata);

  if (!grid_spec_result.ok()) {
    promise_.SetResult(std::move(grid_spec_result).status());
    return;
  }
  internal::ChunkGridSpecification grid_spec = *std::move(grid_spec_result);

  std::string chunk_cache_key;
  std::string metadata_compat_part;
  std::string read_shape_str = tensorstore::StrCat(
      tensorstore::span(metadata->chunk_layout.read_chunk_shape()));

  if (metadata->stacking_info) {
    auto json_result = jb::ToJson(*metadata->stacking_info);
    if (!json_result.ok()) {
      promise_.SetResult(std::move(json_result).status());
      return;
    }
    auto stacking_json = *std::move(json_result);

    metadata_compat_part = absl::StrCat(
        "stack",
        stacking_json.dump(-1, ' ', false,
                           nlohmann::json::error_handler_t::replace),
        "_dtype", metadata->dtype.name(), "_comp",
        static_cast<int>(metadata->compression_type), "_planar",
        static_cast<int>(metadata->planar_config), "_spp",
        metadata->samples_per_pixel, "_endian",
        static_cast<int>(metadata->endian), "_readshape", read_shape_str);
  } else {
    metadata_compat_part = absl::StrFormat(
        "ifd%d_dtype%s_comp%d_planar%d_spp%d_endian%d_readshape%s",
        metadata->base_ifd_index, metadata->dtype.name(),
        static_cast<int>(metadata->compression_type),
        static_cast<int>(metadata->planar_config), metadata->samples_per_pixel,
        static_cast<int>(metadata->endian), read_shape_str);
  }

  internal::EncodeCacheKey(&chunk_cache_key, metadata_cache_entry->key(),
                           metadata_compat_part, cache_pool_->get());

  // 6d. Get or create the TiffChunkCache.
  auto chunk_cache = internal::GetCache<TiffChunkCache>(
      cache_pool_->get(), chunk_cache_key, [&] {
        return std::make_unique<TiffChunkCache>(
            tiff_kvstore_driver, metadata, grid_spec,
            data_copy_concurrency_->executor);
      });
  if (!chunk_cache) {
    promise_.SetResult(
        absl::InternalError("Failed to get or create TiffChunkCache"));
    return;
  }

  // 7. Create TiffDriver
  TiffDriverInitializer driver_initializer{
      /*.cache=*/std::move(chunk_cache),
      /*.component_index=*/0,  // Always 0 for TIFF
      /*.data_staleness_bound=*/staleness_.data.BoundAtOpen(open_time_),
      /*.metadata_staleness_bound=*/staleness_.metadata.BoundAtOpen(open_time_),
      /*.metadata_cache_entry=*/std::move(metadata_cache_entry),
      /*.fill_value_mode=*/fill_value_mode_,
      /*.initial_metadata=*/metadata,  // resolved metadata
      /*.tiff_options=*/tiff_options_,
      /*.schema=*/schema_,  // original schema constraints
      /*.data_copy_concurrency=*/data_copy_concurrency_,
      /*.cache_pool=*/cache_pool_,
      /*.metadata_cache_pool=*/metadata_cache_pool_};

  auto driver =
      internal::MakeIntrusivePtr<TiffDriver>(std::move(driver_initializer));

  // 8. Finalize: Get Transform and Set Promise
  Result<IndexTransform<>> transform_result =
      driver->GetExternalToInternalTransform(*metadata, 0);
  if (!transform_result.ok()) {
    promise_.SetResult(std::move(transform_result).status());
    return;
  }

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

  auto state =
      internal::MakeIntrusivePtr<TiffOpenState>(this, std::move(request));
  auto [promise, future] = PromiseFuturePair<internal::Driver::Handle>::Make();
  state->Start(std::move(promise));

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