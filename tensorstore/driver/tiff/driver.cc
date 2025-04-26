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
#include "tensorstore/driver/driver_spec.h"         // For SharedArray
#include "tensorstore/driver/kvs_backed_chunk_driver.h"  // For KvsDriverSpec, SpecJsonBinder
#include "tensorstore/driver/tiff/metadata.h"  // For TiffMetadata, DecodeChunk
#include "tensorstore/internal/cache/async_cache.h"  // For AsyncCache, AsyncCache::Entry, ReadData
#include "tensorstore/internal/cache/cache.h"  // For CachePool, GetOwningCache
#include "tensorstore/internal/cache/kvs_backed_chunk_cache.h"  // For KvsBackedCache base class
#include "tensorstore/kvstore/driver.h"      // For kvstore::DriverPtr
#include "tensorstore/kvstore/generation.h"  // For TimestampedStorageGeneration
#include "tensorstore/util/execution/any_receiver.h"  // For DecodeReceiver etc.
#include "tensorstore/util/execution/execution.h"  // For execution::set_value/error
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
                          internal::ChunkGridSpecification grid)
      : Base(std::move(tiff_kv_store_driver)),
        resolved_metadata_(std::move(resolved_metadata)),
        grid_(std::move(grid)) {
    assert(resolved_metadata_ && "Resolved metadata cannot be null.");
  }

  // Returns the grid specification provided during construction.
  const internal::ChunkGridSpecification& grid() const override {
    return grid_;
  }

  std::string GetChunkStorageKey(span<const Index> cell_indices) override {
    ABSL_CHECK(resolved_metadata_ != nullptr);
    const auto& metadata = *resolved_metadata_;
    const auto& grid = grid_;  // Get the grid spec stored in the cache

    const DimensionIndex grid_rank = grid.grid_rank();
    const DimensionIndex metadata_rank = metadata.rank;

    // --- Determine logical Y and X dimensions in the TensorStore array ---
    // Same logic as before to find ts_y_dim and ts_x_dim based on inner_order
    DimensionIndex ts_y_dim = -1, ts_x_dim = -1;
    const auto& inner_order = metadata.chunk_layout.inner_order();

    if (!inner_order.empty()) {
      DimensionIndex x_perm_val = metadata_rank - 1;
      DimensionIndex y_perm_val = metadata_rank - 2;  // Only valid if rank >= 2
      for (DimensionIndex i = 0; i < metadata_rank; ++i) {
        if (inner_order[i] == x_perm_val) ts_x_dim = i;
        if (metadata_rank >= 2 && inner_order[i] == y_perm_val) ts_y_dim = i;
      }
    } else {
      // Fallback: Assume C-order if inner_order is not set
      if (metadata_rank >= 2) {
        ts_y_dim = metadata_rank - 2;
        ts_x_dim = metadata_rank - 1;
      } else if (metadata_rank == 1) {
        ts_y_dim = -1;
        ts_x_dim = 0;
      }
    }
    ABSL_CHECK(metadata_rank == 0 || ts_x_dim != -1)
        << "Could not determine X dimension index";
    ABSL_CHECK(metadata_rank < 2 || ts_y_dim != -1)
        << "Could not determine Y dimension index";

    // --- Determine if Tiled or Stripped ---
    const auto& read_chunk_shape = metadata.chunk_layout.read_chunk_shape();
    // Handle rank 0 or 1 cases where there might not be an X dimension
    bool is_tiled = false;
    if (ts_x_dim != -1) {
      const Index chunk_width = read_chunk_shape[ts_x_dim];
      const Index image_width = metadata.shape[ts_x_dim];
      is_tiled = (chunk_width < image_width);
    }  // else: if rank < 2, it's effectively stripped (or a single point)

    // --- Map grid indices to IFD, Row, Col based on num_ifds ---
    uint32_t ifd = 0;
    uint32_t row_idx = 0;
    uint32_t col_idx = 0;

    const auto& chunked_to_cell = grid.components[0].chunked_to_cell_dimensions;
    ABSL_CHECK(chunked_to_cell.size() == grid_rank);

    if (metadata.num_ifds == 1) {
      // --- Single IFD Mode ---
      ifd = metadata.ifd_index;  // IFD is fixed by the metadata context

      // Grid dimensions must correspond to the spatial dimensions Y and X.
      // Grid rank should be 1 (if rank 1 image) or 2 (if rank >= 2 image)
      ABSL_CHECK(grid_rank >= 1 && grid_rank <= 2)
          << "Expected grid rank 1 or 2 for single IFD mode, got " << grid_rank;
      ABSL_CHECK(metadata_rank >= grid_rank)
          << "Metadata rank cannot be less than grid rank";

      DimensionIndex grid_dim_for_y = -1;
      DimensionIndex grid_dim_for_x = -1;

      // Find which grid dimension maps to ts_y_dim and ts_x_dim
      if (ts_y_dim != -1) {  // Should exist if metadata_rank >= 2
        for (DimensionIndex grid_i = 0; grid_i < grid_rank; ++grid_i) {
          if (chunked_to_cell[grid_i] == ts_y_dim) {
            grid_dim_for_y = grid_i;
            break;
          }
        }
        ABSL_CHECK(grid_dim_for_y != -1) << "Grid dimension for Y not found";
        row_idx = static_cast<uint32_t>(cell_indices[grid_dim_for_y]);
      } else {
        // Handle rank 1 case (only X dimension) - no row index conceptually
        row_idx = 0;
      }

      for (DimensionIndex grid_i = 0; grid_i < grid_rank; ++grid_i) {
        if (chunked_to_cell[grid_i] == ts_x_dim) {
          grid_dim_for_x = grid_i;
          break;
        }
      }
      ABSL_CHECK(grid_dim_for_x != -1) << "Grid dimension for X not found";
      col_idx = static_cast<uint32_t>(cell_indices[grid_dim_for_x]);

      // For stripped images, the column index in the key is always 0.
      if (!is_tiled) {
        ABSL_CHECK(grid.chunk_shape[grid_dim_for_x] == 1)
            << "Grid shape for X dimension should be 1 for stripped TIFF in "
               "single IFD mode";
        ABSL_CHECK(cell_indices[grid_dim_for_x] == 0)
            << "Cell index for X dimension should be 0 for stripped TIFF in "
               "single IFD mode";
        col_idx = 0;
      }

    } else {
      // --- Multi IFD Mode (Stacking - Future Scenario) ---
      // Grid rank must be 3 (IFD/Z, Y, X).
      ABSL_CHECK(grid_rank == 3)
          << "Expected grid rank 3 for multi-IFD mode, got " << grid_rank;
      ABSL_CHECK(metadata_rank >= 2)
          << "Metadata rank must be >= 2 for multi-IFD stack";

      DimensionIndex grid_dim_for_y = -1;
      DimensionIndex grid_dim_for_x = -1;
      DimensionIndex grid_dim_for_ifd =
          -1;  // The grid dim mapping to the IFD/Z stack

      // Find grid dims for Y and X (must exist)
      for (DimensionIndex grid_i = 0; grid_i < grid_rank; ++grid_i) {
        if (chunked_to_cell[grid_i] == ts_y_dim) grid_dim_for_y = grid_i;
        if (chunked_to_cell[grid_i] == ts_x_dim) grid_dim_for_x = grid_i;
      }
      ABSL_CHECK(grid_dim_for_y != -1)
          << "Grid dimension for Y not found in multi-IFD";
      ABSL_CHECK(grid_dim_for_x != -1)
          << "Grid dimension for X not found in multi-IFD";

      // Find the remaining grid dimension, assume it maps to IFD/Z
      for (DimensionIndex grid_i = 0; grid_i < grid_rank; ++grid_i) {
        if (grid_i != grid_dim_for_y && grid_i != grid_dim_for_x) {
          grid_dim_for_ifd = grid_i;
          break;
        }
      }
      ABSL_CHECK(grid_dim_for_ifd != -1)
          << "Grid dimension for IFD/Z not found";

      // Assign values from cell_indices based on discovered grid dimension
      // mappings
      ifd = static_cast<uint32_t>(cell_indices[grid_dim_for_ifd]);
      row_idx = static_cast<uint32_t>(cell_indices[grid_dim_for_y]);
      col_idx = static_cast<uint32_t>(cell_indices[grid_dim_for_x]);

      // For stripped images, the column index in the key is always 0.
      if (!is_tiled) {
        ABSL_CHECK(grid.chunk_shape[grid_dim_for_x] == 1)
            << "Grid shape for X dimension should be 1 for stripped TIFF in "
               "multi-IFD mode";
        ABSL_CHECK(cell_indices[grid_dim_for_x] == 0)
            << "Cell index for X dimension should be 0 for stripped TIFF in "
               "multi-IFD mode";
        col_idx = 0;
      }
    }

    // --- Format the key ---
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
};

class TiffDriver final
    : public internal::ChunkGridSpecificationDriver<TiffChunkCache,
                                                    internal::Driver> {
 public:
  using Base =
      internal::ChunkGridSpecificationDriver<TiffChunkCache, internal::Driver>;

  explicit TiffDriver(TiffDriverInitializer&& initializer)
      : Base({std::move(initializer.cache),
              initializer.component_index,  // Should be 0
              initializer.data_staleness_bound}),
        metadata_staleness_bound_(initializer.metadata_staleness_bound),
        metadata_cache_entry_(std::move(initializer.metadata_cache_entry)),
        fill_value_mode_(initializer.fill_value_mode),
        initial_metadata_(std::move(initializer.initial_metadata)) {
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
    // TODO(user): Implement GetBoundSpec using TiffMetadata
    return absl::UnimplementedError("GetBoundSpec not implemented");
  }

  // Define GarbageCollectionBase struct inside TiffDriver
  struct GarbageCollectionBase {
    static void Visit(garbage_collection::GarbageCollectionVisitor& visitor,
                      const TiffDriver& value) {
      // Visit the base class members (including cache ptr)
      value.Base::GarbageCollectionVisit(visitor);
      // Visit TiffDriver specific members
      garbage_collection::GarbageCollectionVisit(visitor,
                                                 value.metadata_cache_entry_);
    }
  };

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
    // TODO(user): Create TiffCodecSpec based on
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
    // The relevant KvStore is the base store used by the TiffDirectoryCache.
    // We can access the TiffDirectoryCache via the metadata_cache_entry_.
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
    cache()->Read(
        {std::move(request), component_index(),
         this->data_staleness_bound().time,
         /* Use member directly */ fill_value_mode_.fill_missing_data_reads},
        std::move(receiver));
  }

  void Write(WriteRequest request, WriteChunkReceiver receiver) override {
    // Fail explicitly for read-only driver
    execution::set_error(receiver,
                         absl::UnimplementedError("TIFF driver is read-only"));
  }

  Future<IndexTransform<>> ResolveBounds(
      ResolveBoundsRequest request) override {
    // TODO(user): Implement ResolveBounds using TiffMetadata
    // Needs to get potentially updated metadata via ResolveMetadata helper.
    // For now, return error or identity based on metadata.
    return absl::UnimplementedError("ResolveBounds not implemented");
    // Example structure:
    // return MapFutureValue(
    //      executor(),
    //      [transform = std::move(request.transform)](const MetadataPtr& md)
    //          -> Result<IndexTransform<>> {
    //          // Use md to resolve bounds in transform
    //      },
    //      ResolveMetadata(std::move(request.transaction)));
  }

  Future<IndexTransform<>> Resize(ResizeRequest request) override {
    return absl::UnimplementedError("Resize is not supported by TIFF driver");
  }

  Future<ArrayStorageStatistics> GetStorageStatistics(
      GetStorageStatisticsRequest request) override {
    // TODO(user): Implement GetStorageStatistics if desired.
    // Might involve iterating keys in TiffKvStore? Complex.
    return absl::UnimplementedError("GetStorageStatistics not implemented");
  }

  // --- Helper for potentially stale metadata access ---
  Future<std::shared_ptr<const TiffMetadata>> ResolveMetadata(
      internal::OpenTransactionPtr transaction) {
    // Use the metadata cache entry to read potentially updated metadata
    // respecting the transaction and staleness bound.
    // return MapFuture(
    //     this->data_copy_executor(),
    //     [this](const Result<void>& read_result)
    //         -> Result<std::shared_ptr<const TiffMetadata>> {
    //       TENSORSTORE_RETURN_IF_ERROR(read_result);
    //       // Use ReadLock to get the data associated with the completed read
    //       auto lock = AsyncCache::ReadLock<const TiffMetadata>(
    //           *this->metadata_cache_entry_);
    //       auto data_ptr = lock.shared_data();
    //       if (!data_ptr) {
    //         return absl::NotFoundError(
    //             "TIFF metadata not found or failed to load.");
    //       }
    //       return data_ptr;
    //     },
    //     metadata_cache_entry_->Read({metadata_staleness_bound_.time}));
  }

  // --- Required by ChunkCacheReadWriteDriverMixin ---
  const StalenessBound& metadata_staleness_bound() const {
    return metadata_staleness_bound_;
  }
  bool fill_missing_data_reads() const {
    return fill_value_mode_.fill_missing_data_reads;
  }
  bool store_data_equal_to_fill_value() const {
    return fill_value_mode_.store_data_equal_to_fill_value;
  }

 private:
  friend class TiffDriverSpec;  // Allow Spec to call constructor/access members

  StalenessBound metadata_staleness_bound_;
  internal::PinnedCacheEntry<internal_tiff_kvstore::TiffDirectoryCache>
      metadata_cache_entry_;
  internal_kvs_backed_chunk_driver::FillValueMode fill_value_mode_;
  std::shared_ptr<const TiffMetadata> initial_metadata_;
};  // End TiffDriver

// --- TiffDriverSpec::Open Implementation ---
Future<internal::Driver::Handle> TiffDriverSpec::Open(
    internal::DriverOpenRequest request) const {
  // TODO(user): Implement the full Open logic:
  // 1. Validate OpenModeSpec against request.read_write_mode.
  // 2. Check store.valid().
  // 3. Get or create TiffDirectoryCache entry using metadata_cache_pool.
  // 4. Read TiffParseResult from directory cache entry, handling staleness.
  // 5. Call ResolveMetadata(parse_result, tiff_options, schema) -> metadata.
  // 6. Validate metadata against metadata_constraints.
  // 7. Create TiffKvStore driver instance.
  // 8. Create ChunkGridSpecification from metadata.
  // 9. Get or create TiffChunkCache using cache_pool, appropriate key,
  //    passing TiffKvStore driver, metadata ptr, and grid to factory.
  // 10. Create TiffDriverInitializer.
  // 11. Create TiffDriver instance.
  // 12. Create DriverHandle with appropriate transform (likely identity or
  //     based on resolved bounds).
  // Return...
  return absl::UnimplementedError("TiffDriverSpec::Open not implemented");
}

#ifndef _MSC_VER
}  // namespace
#endif

}  // namespace internal_tiff
}  // namespace tensorstore

// --- Garbage Collection ---
// Add near the top of driver.cc or relevant header if missing
TENSORSTORE_DECLARE_GARBAGE_COLLECTION_SPECIALIZATION(
    tensorstore::internal_tiff::TiffDriver)

TENSORSTORE_DEFINE_GARBAGE_COLLECTION_SPECIALIZATION(
    tensorstore::internal_tiff::TiffDriver,
    tensorstore::internal_tiff::TiffDriver::GarbageCollectionBase)

// --- Registration (Placeholder) ---
// TODO(user): Add registration using
// internal::DriverRegistration<TiffDriverSpec>
