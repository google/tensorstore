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

#include "tensorstore/driver/driver.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <charconv>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "tensorstore/array.h"
#include "tensorstore/array_storage_statistics.h"
#include "tensorstore/batch.h"
#include "tensorstore/box.h"
#include "tensorstore/chunk_layout.h"
#include "tensorstore/codec_spec.h"
#include "tensorstore/context.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/chunk.h"
#include "tensorstore/driver/chunk_cache_driver.h"
#include "tensorstore/driver/chunk_receiver_utils.h"
#include "tensorstore/driver/driver_spec.h"
#include "tensorstore/driver/kvs_backed_chunk_driver.h"
#include "tensorstore/driver/neuroglancer_precomputed/chunk_encoding.h"
#include "tensorstore/driver/neuroglancer_precomputed/metadata.h"
#include "tensorstore/driver/registry.h"
#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/index_space/dimension_permutation.h"
#include "tensorstore/index_space/dimension_units.h"
#include "tensorstore/index_space/index_domain.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/internal/cache_key/cache_key.h"
#include "tensorstore/internal/chunk_grid_specification.h"
#include "tensorstore/internal/grid_chunk_key_ranges_base10.h"
#include "tensorstore/internal/grid_partition.h"
#include "tensorstore/internal/grid_storage_statistics.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_fwd.h"
#include "tensorstore/internal/lexicographical_grid_index_key.h"
#include "tensorstore/kvstore/kvstore.h"
#include "tensorstore/kvstore/neuroglancer_uint64_sharded/neuroglancer_uint64_sharded.h"
#include "tensorstore/kvstore/spec.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/open_options.h"
#include "tensorstore/rank.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/constant_vector.h"
#include "tensorstore/util/dimension_set.h"
#include "tensorstore/util/division.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/garbage_collection/fwd.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"
#include "tensorstore/util/unit.h"

namespace tensorstore {
namespace internal_neuroglancer_precomputed {

// Avoid anonymous namespace to workaround MSVC bug.
//
// https://developercommunity.visualstudio.com/t/Bug-involving-virtual-functions-templat/10424129
#ifndef _MSC_VER
namespace {
#endif

namespace jb = tensorstore::internal_json_binding;

using ::tensorstore::internal_kvs_backed_chunk_driver::KvsDriverSpec;

class NeuroglancerPrecomputedDriverSpec
    : public internal::RegisteredDriverSpec<NeuroglancerPrecomputedDriverSpec,
                                            KvsDriverSpec> {
 public:
  using Base = internal::RegisteredDriverSpec<NeuroglancerPrecomputedDriverSpec,
                                              KvsDriverSpec>;
  constexpr static char id[] = "neuroglancer_precomputed";

  OpenConstraints open_constraints;

  constexpr static auto ApplyMembers = [](auto& x, auto f) {
    return f(internal::BaseCast<KvsDriverSpec>(x), x.open_constraints);
  };

  static inline const auto default_json_binder = jb::Sequence(
      internal_kvs_backed_chunk_driver::SpecJsonBinder,
      [](auto is_loading, auto options, auto* obj, auto* j) {
        options.Set(obj->schema.dtype());
        return jb::DefaultBinder<>(is_loading, options, &obj->open_constraints,
                                   j);
      },
      jb::Initialize([](auto* obj) {
        TENSORSTORE_RETURN_IF_ERROR(obj->schema.Set(RankConstraint{4}));
        TENSORSTORE_RETURN_IF_ERROR(
            obj->schema.Set(obj->open_constraints.multiscale.dtype));
        return absl::OkStatus();
      }));

  absl::Status ApplyOptions(SpecOptions&& options) override {
    if (options.minimal_spec) {
      open_constraints.scale = ScaleMetadataConstraints{};
      open_constraints.multiscale = MultiscaleMetadataConstraints{};
    }
    return Base::ApplyOptions(std::move(options));
  }

  Result<IndexDomain<>> GetDomain() const override {
    return GetEffectiveDomain(/*existing_metadata=*/nullptr, open_constraints,
                              schema);
  }

  Result<CodecSpec> GetCodec() const override {
    TENSORSTORE_ASSIGN_OR_RETURN(auto codec,
                                 GetEffectiveCodec(open_constraints, schema));
    return CodecSpec(std::move(codec));
  }

  Result<ChunkLayout> GetChunkLayout() const override {
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto domain_and_chunk_layout,
        GetEffectiveDomainAndChunkLayout(/*existing_metadata=*/nullptr,
                                         open_constraints, schema));
    return domain_and_chunk_layout.second;
  }

  Result<SharedArray<const void>> GetFillValue(
      IndexTransformView<> transform) const override {
    return {std::in_place};
  }

  Result<DimensionUnitsVector> GetDimensionUnits() const override {
    return GetEffectiveDimensionUnits(open_constraints, schema);
  }

  Future<internal::Driver::Handle> Open(
      internal::DriverOpenRequest request) const override;
};

Result<std::shared_ptr<const MultiscaleMetadata>> ParseEncodedMetadata(
    std::string_view encoded_value) {
  nlohmann::json raw_data = nlohmann::json::parse(encoded_value, nullptr,
                                                  /*allow_exceptions=*/false);
  if (raw_data.is_discarded()) {
    return absl::FailedPreconditionError("Invalid JSON");
  }
  TENSORSTORE_ASSIGN_OR_RETURN(auto metadata,
                               MultiscaleMetadata::FromJson(raw_data));
  return std::make_shared<MultiscaleMetadata>(std::move(metadata));
}

class MetadataCache : public internal_kvs_backed_chunk_driver::MetadataCache {
  using Base = internal_kvs_backed_chunk_driver::MetadataCache;

 public:
  using Base::Base;

  std::string GetMetadataStorageKey(std::string_view entry_key) override {
    return tensorstore::StrCat(entry_key, kMetadataKey);
  }

  Result<MetadataPtr> DecodeMetadata(std::string_view entry_key,
                                     absl::Cord encoded_metadata) override {
    return ParseEncodedMetadata(encoded_metadata.Flatten());
  }

  Result<absl::Cord> EncodeMetadata(std::string_view entry_key,
                                    const void* metadata) override {
    return absl::Cord(
        ::nlohmann::json(*static_cast<const MultiscaleMetadata*>(metadata))
            .dump());
  }
};

/// Defines common DataCache behavior for the Neuroglancer precomputed driver
/// for both the unsharded and sharded formats.
///
/// In the metadata `"size"` and `"chunk_sizes"` fields, dimensions are listed
/// in `(x, y, z)` order, and in the chunk keys, dimensions are also listed in
/// `(x, y, z)` order.  Within encoded chunks, data is stored in
/// `(x, y, z, channel)` Fortran order.  For consistency, the default dimension
/// order exposed to users is also `(x, y, z, channel)`.  Because the chunk
/// cache always stores each chunk component in C order, we use the reversed
/// `(channel, z, y, x)` order for the component, and then permute the
/// dimensions in `{Ex,In}ternalizeTransform`.
class DataCacheBase : public internal_kvs_backed_chunk_driver::DataCache {
  using Base = internal_kvs_backed_chunk_driver::DataCache;

 public:
  explicit DataCacheBase(Initializer initializer, std::string_view key_prefix,
                         const MultiscaleMetadata& metadata, size_t scale_index,
                         std::array<Index, 3> chunk_size_xyz)
      : Base(std::move(initializer),
             GetChunkGridSpecification(metadata, scale_index, chunk_size_xyz)),
        key_prefix_(key_prefix),
        scale_index_(scale_index) {
    chunk_layout_czyx_.shape()[0] = metadata.num_channels;
    for (int i = 0; i < 3; ++i) {
      chunk_layout_czyx_.shape()[1 + i] = chunk_size_xyz[2 - i];
    }
    ComputeStrides(c_order, metadata.dtype.size(), chunk_layout_czyx_.shape(),
                   chunk_layout_czyx_.byte_strides());
  }

  const MultiscaleMetadata& metadata() const {
    return *static_cast<const MultiscaleMetadata*>(initial_metadata().get());
  }

  /// Returns the chunk size in the external (xyz) order.
  std::array<Index, 3> chunk_size_xyz() const {
    return {{
        chunk_layout_czyx_.shape()[3],
        chunk_layout_czyx_.shape()[2],
        chunk_layout_czyx_.shape()[1],
    }};
  }

  absl::Status ValidateMetadataCompatibility(
      const void* existing_metadata_ptr,
      const void* new_metadata_ptr) override {
    const auto& existing_metadata =
        *static_cast<const MultiscaleMetadata*>(existing_metadata_ptr);
    const auto& new_metadata =
        *static_cast<const MultiscaleMetadata*>(new_metadata_ptr);
    return internal_neuroglancer_precomputed::ValidateMetadataCompatibility(
        existing_metadata, new_metadata, scale_index_, chunk_size_xyz());
  }

  void GetChunkGridBounds(const void* metadata_ptr, MutableBoxView<> bounds,
                          DimensionSet& implicit_lower_bounds,
                          DimensionSet& implicit_upper_bounds) override {
    // Chunk grid dimension order is `[x, y, z]`.
    const auto& metadata =
        *static_cast<const MultiscaleMetadata*>(metadata_ptr);
    assert(3 == bounds.rank());
    std::fill(bounds.origin().begin(), bounds.origin().end(), Index(0));
    const auto& scale_metadata = metadata.scales[scale_index_];
    absl::c_copy(scale_metadata.box.shape(), bounds.shape().begin());
    implicit_lower_bounds = false;
    implicit_upper_bounds = false;
  }

  Result<std::shared_ptr<const void>> GetResizedMetadata(
      const void* existing_metadata, span<const Index> new_inclusive_min,
      span<const Index> new_exclusive_max) override {
    return absl::UnimplementedError("");
  }

  static internal::ChunkGridSpecification GetChunkGridSpecification(
      const MultiscaleMetadata& metadata, size_t scale_index,
      span<Index, 3> chunk_size_xyz) {
    std::array<Index, 4> chunk_shape_czyx;
    chunk_shape_czyx[0] = metadata.num_channels;
    for (DimensionIndex i = 0; i < 3; ++i) {
      chunk_shape_czyx[3 - i] = chunk_size_xyz[i];
    }
    // Component dimension order is `[channel, z, y, x]`.
    SharedArray<const void> fill_value(
        internal::AllocateAndConstructSharedElements(1, value_init,
                                                     metadata.dtype),
        StridedLayout<>(chunk_shape_czyx, GetConstantVector<Index, 0, 4>()));
    // Resizing is not supported.  Specifying the `component_bounds` permits
    // partial chunks at the upper bounds to be written unconditionally (which
    // may be more efficient) if fully overwritten.
    Box<> component_bounds_czyx(4);
    component_bounds_czyx.origin()[0] = 0;
    component_bounds_czyx.shape()[0] = metadata.num_channels;
    const auto& box_xyz = metadata.scales[scale_index].box;
    for (DimensionIndex i = 0; i < 3; ++i) {
      // The `ChunkCache` always translates the origin to `0`.
      component_bounds_czyx[3 - i] =
          IndexInterval::UncheckedSized(0, box_xyz[i].size());
    }
    internal::ChunkGridSpecification::ComponentList components;
    components.emplace_back(std::move(fill_value),
                            std::move(component_bounds_czyx),
                            std::vector<DimensionIndex>{3, 2, 1});
    return internal::ChunkGridSpecification(std::move(components));
  }

  Result<absl::InlinedVector<SharedArray<const void>, 1>> DecodeChunk(
      span<const Index> chunk_indices, absl::Cord data) override {
    if (auto result = internal_neuroglancer_precomputed::DecodeChunk(
            chunk_indices, metadata(), scale_index_, chunk_layout_czyx_,
            std::move(data))) {
      absl::InlinedVector<SharedArray<const void>, 1> components;
      components.emplace_back(std::move(*result));
      return components;
    } else {
      return absl::FailedPreconditionError(result.status().message());
    }
  }

  Result<absl::Cord> EncodeChunk(
      span<const Index> chunk_indices,
      span<const SharedArrayView<const void>> component_arrays) override {
    assert(component_arrays.size() == 1);
    return internal_neuroglancer_precomputed::EncodeChunk(
        chunk_indices, metadata(), scale_index_, component_arrays[0]);
  }

  Result<IndexTransform<>> GetExternalToInternalTransform(
      const void* metadata_ptr, size_t component_index) override {
    assert(component_index == 0);
    const auto& metadata =
        *static_cast<const MultiscaleMetadata*>(metadata_ptr);
    const auto& scale = metadata.scales[scale_index_];
    const auto& box = scale.box;
    auto builder = IndexTransformBuilder<>(4, 4);
    auto input_origin = builder.input_origin();
    std::copy(box.origin().begin(), box.origin().end(), input_origin.begin());
    input_origin[3] = 0;
    auto input_shape = builder.input_shape();
    std::copy(box.shape().begin(), box.shape().end(), input_shape.begin());
    input_shape[3] = metadata.num_channels;
    builder.input_labels({"x", "y", "z", "channel"});
    builder.output_single_input_dimension(0, 3);
    for (int i = 0; i < 3; ++i) {
      builder.output_single_input_dimension(3 - i, -box.origin()[i], 1, i);
    }
    return builder.Finalize();
  }

  absl::Status GetBoundSpecData(
      KvsDriverSpec& spec_base, const void* metadata_ptr,
      [[maybe_unused]] size_t component_index) override {
    assert(component_index == 0);
    auto& spec = static_cast<NeuroglancerPrecomputedDriverSpec&>(spec_base);
    const auto& metadata =
        *static_cast<const MultiscaleMetadata*>(metadata_ptr);
    const auto& scale = metadata.scales[scale_index_];
    spec.open_constraints.scale_index = scale_index_;
    auto& scale_constraints = spec.open_constraints.scale;
    scale_constraints.chunk_size = chunk_size_xyz();
    scale_constraints.key = scale.key;
    scale_constraints.resolution = scale.resolution;
    scale_constraints.box = scale.box;
    scale_constraints.encoding = scale.encoding;
    if (scale.encoding == ScaleMetadata::Encoding::compressed_segmentation) {
      scale_constraints.compressed_segmentation_block_size =
          scale.compressed_segmentation_block_size;
    }
    scale_constraints.sharding = scale.sharding;
    auto& multiscale_constraints = spec.open_constraints.multiscale;
    multiscale_constraints.num_channels = metadata.num_channels;
    multiscale_constraints.type = metadata.type;
    return absl::OkStatus();
  }

  Result<ChunkLayout> GetBaseChunkLayout(const MultiscaleMetadata& metadata,
                                         ChunkLayout::Usage base_usage) {
    ChunkLayout layout;
    // Leave origin set at zero; origin is accounted for by the index transform.
    TENSORSTORE_RETURN_IF_ERROR(
        layout.Set(ChunkLayout::GridOrigin(GetConstantVector<Index, 0>(4))));
    const auto& scale = metadata.scales[scale_index_];
    {
      DimensionIndex inner_order[4];
      SetPermutation(c_order, inner_order);
      TENSORSTORE_RETURN_IF_ERROR(
          layout.Set(ChunkLayout::InnerOrder(inner_order)));
    }
    TENSORSTORE_RETURN_IF_ERROR(layout.Set(ChunkLayout::Chunk(
        ChunkLayout::ChunkShape(chunk_layout_czyx_.shape()), base_usage)));
    if (scale.encoding == ScaleMetadata::Encoding::compressed_segmentation) {
      TENSORSTORE_RETURN_IF_ERROR(layout.Set(ChunkLayout::CodecChunkShape(
          {1, scale.compressed_segmentation_block_size[2],
           scale.compressed_segmentation_block_size[1],
           scale.compressed_segmentation_block_size[0]})));
    }
    return layout;
  }

  std::string GetBaseKvstorePath() override { return key_prefix_; }

  virtual Future<ArrayStorageStatistics> GetStorageStatistics(
      internal::Driver::GetStorageStatisticsRequest request,
      absl::Time staleness_bound) = 0;

  std::string key_prefix_;
  size_t scale_index_;
  // channel, z, y, x
  StridedLayout<4> chunk_layout_czyx_;
};

class UnshardedDataCache : public DataCacheBase {
 public:
  explicit UnshardedDataCache(Initializer initializer,
                              std::string_view key_prefix,
                              const MultiscaleMetadata& metadata,
                              size_t scale_index,
                              std::array<Index, 3> chunk_size_xyz)
      : DataCacheBase(std::move(initializer), key_prefix, metadata, scale_index,
                      chunk_size_xyz) {
    const auto& scale = metadata.scales[scale_index];
    scale_key_prefix_ = ResolveScaleKey(key_prefix, scale.key);
  }

  class KeyFormatter : public internal::LexicographicalGridIndexKeyParser {
   public:
    explicit KeyFormatter(const UnshardedDataCache& cache) {
      const auto& metadata = *static_cast<const MultiscaleMetadata*>(
          cache.initial_metadata_.get());
      const auto& scale = metadata.scales[cache.scale_index_];
      box_ = scale.box;
      std::copy_n(cache.chunk_layout_czyx_.shape().data() + 1, 3,
                  chunk_shape_zyx_.begin());
    }

    std::string FormatKey(span<const Index> grid_indices) const final {
      std::string key;
      internal::FormatGridIndexKeyWithDimensionSeparator(
          key, '_',
          [this](std::string& out, DimensionIndex dim, Index grid_index) {
            FormatGridIndex(out, dim, grid_index);
          },
          /*rank=*/3, grid_indices);
      return key;
    }

    bool ParseKey(std::string_view key, span<Index> grid_indices) const final {
      return internal::ParseGridIndexKeyWithDimensionSeparator(
          '_',
          [this](std::string_view part, DimensionIndex dim, Index& grid_index) {
            return ParseGridIndex(part, dim, grid_index);
          },
          key, grid_indices);
    }

    void FormatGridIndex(std::string& out, DimensionIndex dim,
                         Index grid_index) const {
      const Index chunk_size = chunk_shape_zyx_[2 - dim];
      absl::StrAppend(
          &out, box_.origin()[dim] + chunk_size * grid_index, "-",
          box_.origin()[dim] +
              std::min(chunk_size * (grid_index + 1), box_.shape()[dim]));
    }

    bool ParseGridIndex(std::string_view key, DimensionIndex dim,
                        Index& grid_index) const {
      Index start_index, end_index;
      // Start and end bounds are separated by '-'.
      size_t sep = key.find('-');
      if (sep == std::string_view::npos) return false;
      if (auto result =
              std::from_chars(key.data(), key.data() + sep, start_index);
          result.ptr != key.data() + sep || result.ec != std::errc{}) {
        return false;
      }
      if (auto result = std::from_chars(key.data() + sep + 1,
                                        key.data() + key.size(), end_index);
          result.ptr != key.data() + key.size() || result.ec != std::errc{}) {
        return false;
      }
      if (!Contains(box_[dim], start_index)) return false;
      const Index chunk_size = chunk_shape_zyx_[2 - dim];
      const Index origin_relative_start = start_index - box_.origin()[dim];
      if (origin_relative_start % chunk_size != 0) return false;
      grid_index = origin_relative_start / chunk_size;
      if (end_index !=
          std::min(start_index + chunk_size, box_[dim].exclusive_max())) {
        return false;
      }
      return true;
    }

    Index MinGridIndexForLexicographicalOrder(
        DimensionIndex dim, IndexInterval grid_interval) const final {
      const Index chunk_size = chunk_shape_zyx_[2 - dim];
      Index max_index = box_[dim].exclusive_max();
      Index min_lex_index =
          max_index > 0
              ? internal::MinValueWithMaxBase10Digits(box_[dim].exclusive_max())
              : 0;
      return tensorstore::CeilOfRatio(min_lex_index - box_.origin()[dim],
                                      chunk_size);
    }

   private:
    Box<3> box_;
    std::array<Index, 3> chunk_shape_zyx_;
  };

  std::string GetChunkStorageKey(span<const Index> cell_indices) override {
    std::string key = scale_key_prefix_;
    if (!key.empty()) key += '/';
    KeyFormatter key_formatter(*this);
    for (int i = 0; i < 3; ++i) {
      if (i != 0) key += '_';
      key_formatter.FormatGridIndex(key, i, cell_indices[i]);
    }
    return key;
  }

  Result<ChunkLayout> GetChunkLayoutFromMetadata(
      const void* metadata_ptr, size_t component_index) override {
    const auto& metadata =
        *static_cast<const MultiscaleMetadata*>(metadata_ptr);
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto layout, GetBaseChunkLayout(metadata, ChunkLayout::kWrite));
    TENSORSTORE_RETURN_IF_ERROR(layout.Finalize());
    return layout;
  }

  virtual Future<ArrayStorageStatistics> GetStorageStatistics(
      internal::Driver::GetStorageStatisticsRequest request,
      absl::Time staleness_bound) override {
    const auto& metadata =
        *static_cast<const MultiscaleMetadata*>(this->initial_metadata_.get());
    const auto& scale = metadata.scales[scale_index_];
    auto& grid = this->grid();
    Box<3> grid_bounds;
    for (DimensionIndex i = 0; i < 3; ++i) {
      const Index chunk_size = chunk_layout_czyx_.shape()[3 - i];
      grid_bounds[i] = IndexInterval::UncheckedSized(
          0, tensorstore::CeilOfRatio(scale.box.shape()[i], chunk_size));
    }
    const auto& component = grid.components[0];
    std::string path = tensorstore::StrCat(this->GetBaseKvstorePath(),
                                           this->scale_key_prefix_);
    if (!path.empty()) {
      path += '/';
    }
    return internal::
        GetStorageStatisticsForRegularGridWithSemiLexicographicalKeys(
            KvStore{kvstore::DriverPtr(this->kvstore_driver()), std::move(path),
                    internal::TransactionState::ToTransaction(
                        std::move(request.transaction))},
            request.transform, /*grid_output_dimensions=*/
            component.chunked_to_cell_dimensions,
            /*chunk_shape=*/grid.chunk_shape, grid_bounds,
            std::make_unique<KeyFormatter>(*this), staleness_bound,
            request.options);
  }

 private:
  /// Resolved key prefix for the scale.
  std::string scale_key_prefix_;
};

class ShardedDataCache : public DataCacheBase {
 public:
  explicit ShardedDataCache(Initializer initializer,
                            std::string_view key_prefix,
                            const MultiscaleMetadata& metadata,
                            size_t scale_index,
                            std::array<Index, 3> chunk_size_xyz)
      : DataCacheBase(std::move(initializer), key_prefix, metadata, scale_index,
                      chunk_size_xyz) {
    const auto& scale = metadata.scales[scale_index];
    compressed_z_index_bits_ =
        GetCompressedZIndexBits(scale.box.shape(), chunk_size_xyz);
  }

  std::string GetChunkStorageKey(span<const Index> cell_indices) override {
    assert(cell_indices.size() == 3);
    const uint64_t chunk_key = EncodeCompressedZIndex({cell_indices.data(), 3},
                                                      compressed_z_index_bits_);
    return neuroglancer_uint64_sharded::ChunkIdToKey({chunk_key});
  }

  Result<ChunkLayout> GetChunkLayoutFromMetadata(
      const void* metadata_ptr, size_t component_index) override {
    const auto& metadata = this->metadata();
    const auto& scale = metadata.scales[scale_index_];
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto layout, GetBaseChunkLayout(metadata, ChunkLayout::kRead));
    // Each shard does not correspond to a rectangular region.  The write
    // chunk shape is equal to the full domain.
    Index write_chunk_shape[4];
    write_chunk_shape[0] = metadata.num_channels;
    for (int dim = 0; dim < 3; ++dim) {
      write_chunk_shape[3 - dim] =
          RoundUpTo(scale.box.shape()[dim], scale.chunk_sizes[0][dim]);
    }
    TENSORSTORE_RETURN_IF_ERROR(
        layout.Set(ChunkLayout::WriteChunkShape(write_chunk_shape)));
    TENSORSTORE_RETURN_IF_ERROR(layout.Finalize());
    return layout;
  }

  virtual Future<ArrayStorageStatistics> GetStorageStatistics(
      internal::Driver::GetStorageStatisticsRequest request,
      absl::Time staleness_bound) override {
    // Not yet implemented for sharded format.
    return absl::UnimplementedError("");
  }

  std::array<int, 3> compressed_z_index_bits_;
};

// DataCache for sharded format in the case that shards correspond to
// rectangular regions.
class RegularlyShardedDataCache : public ShardedDataCache {
 public:
  RegularlyShardedDataCache(Initializer initializer,
                            std::string_view key_prefix,
                            const MultiscaleMetadata& metadata,
                            size_t scale_index,
                            std::array<Index, 3> chunk_size_xyz,
                            ShardChunkHierarchy hierarchy)
      : ShardedDataCache(std::move(initializer), key_prefix, metadata,
                         scale_index, chunk_size_xyz),
        hierarchy_(hierarchy) {}

  Result<ChunkLayout> GetChunkLayoutFromMetadata(
      const void* metadata_ptr, size_t component_index) override {
    const auto& metadata = this->metadata();
    const auto& scale = metadata.scales[scale_index_];
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto layout, GetBaseChunkLayout(metadata, ChunkLayout::kRead));
    // Each shard corresponds to a rectangular region.
    Index write_chunk_shape[4];
    write_chunk_shape[0] = metadata.num_channels;
    for (int dim = 0; dim < 3; ++dim) {
      const Index chunk_size = scale.chunk_sizes[0][dim];
      const Index volume_size = scale.box.shape()[dim];
      write_chunk_shape[3 - dim] =
          RoundUpTo(std::min(hierarchy_.shard_shape_in_chunks[dim] * chunk_size,
                             volume_size),
                    chunk_size);
    }
    TENSORSTORE_RETURN_IF_ERROR(
        layout.Set(ChunkLayout::WriteChunkShape(write_chunk_shape)));
    TENSORSTORE_RETURN_IF_ERROR(layout.Finalize());
    return layout;
  }

  void Read(ReadRequest request,
            AnyFlowReceiver<absl::Status, internal::ReadChunk, IndexTransform<>>
                receiver) override {
    return ShardedReadOrWrite(
        std::move(request.transform), std::move(receiver),
        [&](IndexTransform<> transform,
            AnyFlowReceiver<absl::Status, internal::ReadChunk, IndexTransform<>>
                receiver) {
          Batch shard_batch = request.batch;
          if (!shard_batch) shard_batch = Batch::New();
          return ShardedDataCache::Read(
              {{request.transaction, std::move(transform),
                std::move(shard_batch)},
               request.component_index,
               request.staleness_bound},
              std::move(receiver));
        });
  }

  void Write(
      WriteRequest request,
      AnyFlowReceiver<absl::Status, internal::WriteChunk, IndexTransform<>>
          receiver) override {
    return ShardedReadOrWrite(
        std::move(request.transform), std::move(receiver),
        [&](IndexTransform<> transform,
            AnyFlowReceiver<absl::Status, internal::WriteChunk,
                            IndexTransform<>>
                receiver) {
          internal::OpenTransactionPtr shard_transaction = request.transaction;
          if (!shard_transaction) {
            shard_transaction = internal::TransactionState::MakeImplicit();
            shard_transaction->RequestCommit();
          }
          return ShardedDataCache::Write(
              {{std::move(shard_transaction), std::move(transform)},
               request.component_index},
              std::move(receiver));
        });
  }

 private:
  template <typename ChunkType, typename Callback>
  void ShardedReadOrWrite(
      IndexTransform<> transform,
      AnyFlowReceiver<absl::Status, ChunkType, IndexTransform<>> receiver,
      Callback callback) {
    const auto& metadata = this->metadata();
    const auto& scale = metadata.scales[scale_index_];
    const DimensionIndex chunked_to_cell_dimensions[] = {3, 2, 1};
    Index shard_shape_in_elements[3];
    for (DimensionIndex dim = 0; dim < 3; ++dim) {
      shard_shape_in_elements[dim] =
          scale.chunk_sizes[0][dim] * hierarchy_.shard_shape_in_chunks[dim];
    }
    using State = internal::ChunkOperationState<ChunkType>;
    using ForwardingReceiver =
        internal::ForwardingChunkOperationReceiver<State>;
    auto state = internal::MakeIntrusivePtr<State>(std::move(receiver));
    auto status = internal::PartitionIndexTransformOverRegularGrid(
        chunked_to_cell_dimensions, shard_shape_in_elements, transform,
        [&](span<const Index> grid_cell_indices,
            IndexTransformView<> cell_transform) -> absl::Status {
          if (state->cancelled()) {
            return absl::CancelledError("");
          }
          TENSORSTORE_ASSIGN_OR_RETURN(
              auto cell_to_source,
              ComposeTransforms(transform, cell_transform));
          callback(std::move(cell_to_source),
                   ForwardingReceiver{state, cell_transform});
          return absl::OkStatus();
        });
  }

  ShardChunkHierarchy hierarchy_;
};

class NeuroglancerPrecomputedDriver;
using NeuroglancerPrecomputedDriverBase =
    internal_kvs_backed_chunk_driver::RegisteredKvsDriver<
        NeuroglancerPrecomputedDriver, NeuroglancerPrecomputedDriverSpec,
        DataCacheBase,
        internal::ChunkCacheReadWriteDriverMixin<
            NeuroglancerPrecomputedDriver,
            internal_kvs_backed_chunk_driver::KvsChunkedDriverBase>>;

class NeuroglancerPrecomputedDriver : public NeuroglancerPrecomputedDriverBase {
  using Base = NeuroglancerPrecomputedDriverBase;

 public:
  using Base::Base;

  class OpenState;

  const MultiscaleMetadata& metadata() const {
    return *static_cast<const MultiscaleMetadata*>(
        cache()->initial_metadata().get());
  }

  Result<CodecSpec> GetCodec() override {
    auto* cache = static_cast<DataCacheBase*>(this->cache());
    return GetCodecFromMetadata(metadata(), cache->scale_index_);
  }

  Result<DimensionUnitsVector> GetDimensionUnits() override {
    auto* cache = static_cast<DataCacheBase*>(this->cache());
    const auto& metadata = this->metadata();
    const auto& scale = metadata.scales[cache->scale_index_];
    DimensionUnitsVector units(4);
    for (int i = 0; i < 3; ++i) {
      units[3 - i] = Unit(scale.resolution[i], "nm");
    }
    return units;
  }

  Future<ArrayStorageStatistics> GetStorageStatistics(
      GetStorageStatisticsRequest request) override {
    auto* cache = static_cast<DataCacheBase*>(this->cache());
    return cache->GetStorageStatistics(std::move(request),
                                       this->data_staleness_bound().time);
  }
};

class NeuroglancerPrecomputedDriver::OpenState
    : public NeuroglancerPrecomputedDriver::OpenStateBase {
 public:
  using NeuroglancerPrecomputedDriver::OpenStateBase::OpenStateBase;

  std::string GetPrefixForDeleteExisting() override {
    // TODO(jbms): Possibly change behavior in the future to allow deleting
    // just a single scale.
    return spec().store.path;
  }

  std::string GetMetadataCacheEntryKey() override { return spec().store.path; }

  std::unique_ptr<internal_kvs_backed_chunk_driver::MetadataCache>
  GetMetadataCache(MetadataCache::Initializer initializer) override {
    return std::make_unique<MetadataCache>(std::move(initializer));
  }

  std::string GetDataCacheKey(const void* metadata) override {
    std::string result;
    const auto& spec = this->spec();
    internal::EncodeCacheKey(
        &result, spec.store.path,
        GetMetadataCompatibilityKey(
            *static_cast<const MultiscaleMetadata*>(metadata),
            scale_index_ ? *scale_index_ : *spec.open_constraints.scale_index,
            chunk_size_xyz_));
    return result;
  }

  internal_kvs_backed_chunk_driver::AtomicUpdateConstraint GetCreateConstraint()
      override {
    // `Create` can modify an existing `info` file, but can also create a new
    // `info` file if one does not already exist.
    return internal_kvs_backed_chunk_driver::AtomicUpdateConstraint::kNone;
  }

  Result<std::shared_ptr<const void>> Create(const void* existing_metadata,
                                             CreateOptions options) override {
    const auto* metadata =
        static_cast<const MultiscaleMetadata*>(existing_metadata);
    if (auto result = CreateScale(metadata, spec().open_constraints,
                                  spec().schema, options.assume_metadata)) {
      scale_index_ = result->second;
      return result->first;
    } else {
      scale_index_ = std::nullopt;
      return std::move(result).status();
    }
  }

  std::unique_ptr<internal_kvs_backed_chunk_driver::DataCacheBase> GetDataCache(
      internal_kvs_backed_chunk_driver::DataCacheInitializer&& initializer)
      override {
    const auto& metadata =
        *static_cast<const MultiscaleMetadata*>(initializer.metadata.get());
    assert(scale_index_);
    const auto& scale = metadata.scales[scale_index_.value()];
    if (std::holds_alternative<ShardingSpec>(scale.sharding)) {
      if (ShardChunkHierarchy hierarchy; GetShardChunkHierarchy(
              std::get<ShardingSpec>(scale.sharding), scale.box.shape(),
              scale.chunk_sizes[0], hierarchy)) {
        return std::make_unique<RegularlyShardedDataCache>(
            std::move(initializer), spec().store.path, metadata,
            scale_index_.value(), chunk_size_xyz_, hierarchy);
      } else {
        return std::make_unique<ShardedDataCache>(
            std::move(initializer), spec().store.path, metadata,
            scale_index_.value(), chunk_size_xyz_);
      }
    } else {
      return std::make_unique<UnshardedDataCache>(
          std::move(initializer), spec().store.path, metadata,
          scale_index_.value(), chunk_size_xyz_);
    }
  }

  Result<size_t> GetComponentIndex(const void* metadata_ptr,
                                   OpenMode open_mode) override {
    const auto& metadata =
        *static_cast<const MultiscaleMetadata*>(metadata_ptr);
    // FIXME: avoid copy by changing OpenScale to take separate arguments
    auto open_constraints = spec().open_constraints;
    if (scale_index_) {
      if (spec().open_constraints.scale_index) {
        assert(*spec().open_constraints.scale_index == *scale_index_);
      } else {
        open_constraints.scale_index = *scale_index_;
      }
    }
    TENSORSTORE_ASSIGN_OR_RETURN(
        size_t scale_index,
        OpenScale(metadata, open_constraints, spec().schema));
    const auto& scale = metadata.scales[scale_index];
    if (spec().open_constraints.scale.chunk_size &&
        absl::c_linear_search(scale.chunk_sizes,
                              *spec().open_constraints.scale.chunk_size)) {
      // Use the specified chunk size.
      chunk_size_xyz_ = *spec().open_constraints.scale.chunk_size;
    } else {
      // Chunk size was unspecified.
      assert(!spec().open_constraints.scale.chunk_size);
      chunk_size_xyz_ = scale.chunk_sizes[0];
    }

    TENSORSTORE_RETURN_IF_ERROR(ValidateMetadataSchema(
        metadata, scale_index, chunk_size_xyz_, spec().schema));
    scale_index_ = scale_index;
    // Component index is always 0.
    return 0;
  }

  Result<kvstore::DriverPtr> GetDataKeyValueStore(
      kvstore::DriverPtr base_kv_store, const void* metadata_ptr) override {
    const auto& metadata =
        *static_cast<const MultiscaleMetadata*>(metadata_ptr);
    assert(scale_index_);
    const auto& scale = metadata.scales[*scale_index_];
    if (auto* sharding_spec = std::get_if<ShardingSpec>(&scale.sharding)) {
      assert(scale.chunk_sizes.size() == 1);
      return neuroglancer_uint64_sharded::GetShardedKeyValueStore(
          std::move(base_kv_store), executor(),
          ResolveScaleKey(spec().store.path, scale.key), *sharding_spec,
          *cache_pool(),
          GetChunksPerVolumeShardFunction(*sharding_spec, scale.box.shape(),
                                          scale.chunk_sizes[0]));
    }
    return base_kv_store;
  }

  // Set by `Create` or `GetComponentIndex` to indicate the scale index that
  // has been determined.
  std::optional<size_t> scale_index_;
  // Set by `GetComponentIndex` to indicate the chunk size that has been
  // determined.
  std::array<Index, 3> chunk_size_xyz_;
};

Future<internal::Driver::Handle> NeuroglancerPrecomputedDriverSpec::Open(
    internal::DriverOpenRequest request) const {
  return NeuroglancerPrecomputedDriver::Open(this, std::move(request));
}

#ifndef _MSC_VER
}  // namespace
#endif

}  // namespace internal_neuroglancer_precomputed
}  // namespace tensorstore

TENSORSTORE_DECLARE_GARBAGE_COLLECTION_SPECIALIZATION(
    tensorstore::internal_neuroglancer_precomputed::
        NeuroglancerPrecomputedDriver)
// Use default garbage collection implementation provided by
// kvs_backed_chunk_driver (just handles the kvstore)
TENSORSTORE_DEFINE_GARBAGE_COLLECTION_SPECIALIZATION(
    tensorstore::internal_neuroglancer_precomputed::
        NeuroglancerPrecomputedDriver,
    tensorstore::internal_neuroglancer_precomputed::
        NeuroglancerPrecomputedDriver::GarbageCollectionBase)

namespace {
const tensorstore::internal::DriverRegistration<
    tensorstore::internal_neuroglancer_precomputed::
        NeuroglancerPrecomputedDriverSpec>
    registration;
}  // namespace
