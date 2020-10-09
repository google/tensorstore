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

#include "absl/strings/str_cat.h"
#include "tensorstore/context.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/kvs_backed_chunk_driver.h"
#include "tensorstore/driver/neuroglancer_precomputed/chunk_encoding.h"
#include "tensorstore/driver/neuroglancer_precomputed/metadata.h"
#include "tensorstore/driver/neuroglancer_precomputed/uint64_sharded_key_value_store.h"
#include "tensorstore/driver/registry.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/internal/cache_key.h"
#include "tensorstore/internal/chunk_cache.h"
#include "tensorstore/internal/json.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/kvstore/key_value_store.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/util/constant_vector.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace internal_neuroglancer_precomputed {

namespace {

namespace jb = tensorstore::internal::json_binding;

template <template <typename> class MaybeBound = internal::ContextUnbound>
struct SpecT : public internal_kvs_backed_chunk_driver::SpecT<MaybeBound> {
  std::string key_prefix;
  OpenConstraints open_constraints;

  constexpr static auto ApplyMembers = [](auto& x, auto f) {
    return f(
        internal::BaseCast<internal_kvs_backed_chunk_driver::SpecT<MaybeBound>>(
            x),
        x.key_prefix, x.open_constraints);
  };
};

Result<std::shared_ptr<const MultiscaleMetadata>> ParseEncodedMetadata(
    absl::string_view encoded_value) {
  nlohmann::json raw_data = nlohmann::json::parse(encoded_value, nullptr,
                                                  /*allow_exceptions=*/false);
  if (raw_data.is_discarded()) {
    return absl::FailedPreconditionError("Invalid JSON");
  }
  TENSORSTORE_ASSIGN_OR_RETURN(auto metadata,
                               MultiscaleMetadata::Parse(raw_data));
  return std::make_shared<MultiscaleMetadata>(std::move(metadata));
}

class MetadataCache : public internal_kvs_backed_chunk_driver::MetadataCache {
  using Base = internal_kvs_backed_chunk_driver::MetadataCache;

 public:
  using Base::Base;

  std::string GetMetadataStorageKey(absl::string_view entry_key) override {
    return internal::JoinPath(entry_key, kMetadataKey);
  }

  Result<MetadataPtr> DecodeMetadata(absl::string_view entry_key,
                                     absl::Cord encoded_metadata) override {
    return ParseEncodedMetadata(encoded_metadata.Flatten());
  }

  Result<absl::Cord> EncodeMetadata(absl::string_view entry_key,
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
  explicit DataCacheBase(Initializer initializer, absl::string_view key_prefix,
                         const MultiscaleMetadata& metadata,
                         std::size_t scale_index,
                         std::array<Index, 3> chunk_size_xyz)
      : Base(std::move(initializer),
             GetChunkGridSpecification(metadata, scale_index, chunk_size_xyz)),
        key_prefix_(key_prefix),
        scale_index_(scale_index) {
    chunk_layout_czyx_.shape()[0] = metadata.num_channels;
    for (int i = 0; i < 3; ++i) {
      chunk_layout_czyx_.shape()[1 + i] = chunk_size_xyz[2 - i];
    }
    ComputeStrides(c_order, metadata.data_type.size(),
                   chunk_layout_czyx_.shape(),
                   chunk_layout_czyx_.byte_strides());
  }

  /// Returns the chunk size in the external (xyz) order.
  std::array<Index, 3> chunk_size_xyz() const {
    return {{
        chunk_layout_czyx_.shape()[3],
        chunk_layout_czyx_.shape()[2],
        chunk_layout_czyx_.shape()[1],
    }};
  }

  Status ValidateMetadataCompatibility(const void* existing_metadata_ptr,
                                       const void* new_metadata_ptr) override {
    const auto& existing_metadata =
        *static_cast<const MultiscaleMetadata*>(existing_metadata_ptr);
    const auto& new_metadata =
        *static_cast<const MultiscaleMetadata*>(new_metadata_ptr);
    return internal_neuroglancer_precomputed::ValidateMetadataCompatibility(
        existing_metadata, new_metadata, scale_index_, chunk_size_xyz());
  }

  void GetChunkGridBounds(
      const void* metadata_ptr, MutableBoxView<> bounds,
      BitSpan<std::uint64_t> implicit_lower_bounds,
      BitSpan<std::uint64_t> implicit_upper_bounds) override {
    // Chunk grid dimension order is `[x, y, z]`.
    const auto& metadata =
        *static_cast<const MultiscaleMetadata*>(metadata_ptr);
    assert(3 == bounds.rank());
    assert(3 == implicit_lower_bounds.size());
    assert(3 == implicit_upper_bounds.size());
    std::fill(bounds.origin().begin(), bounds.origin().end(), Index(0));
    const auto& scale_metadata = metadata.scales[scale_index_];
    absl::c_copy(scale_metadata.box.shape(), bounds.shape().begin());
    implicit_lower_bounds.fill(false);
    implicit_upper_bounds.fill(false);
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
                                                     metadata.data_type),
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
    return internal::ChunkGridSpecification(
        {internal::ChunkGridSpecification::Component(
            std::move(fill_value), std::move(component_bounds_czyx),
            {3, 2, 1})});
  }

  Result<absl::InlinedVector<SharedArrayView<const void>, 1>> DecodeChunk(
      const void* metadata, span<const Index> chunk_indices,
      absl::Cord data) override {
    if (auto result = internal_neuroglancer_precomputed::DecodeChunk(
            chunk_indices, *static_cast<const MultiscaleMetadata*>(metadata),
            scale_index_, chunk_layout_czyx_, std::move(data))) {
      return absl::InlinedVector<SharedArrayView<const void>, 1>{
          std::move(*result)};
    } else {
      return absl::FailedPreconditionError(result.status().message());
    }
  }

  Result<absl::Cord> EncodeChunk(
      const void* metadata, span<const Index> chunk_indices,
      span<const ArrayView<const void>> component_arrays) override {
    assert(component_arrays.size() == 1);
    return internal_neuroglancer_precomputed::EncodeChunk(
        chunk_indices, *static_cast<const MultiscaleMetadata*>(metadata),
        scale_index_, component_arrays[0]);
  }

  Result<IndexTransform<>> GetExternalToInternalTransform(
      const void* metadata_ptr, std::size_t component_index) override {
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

  Status GetBoundSpecData(
      internal_kvs_backed_chunk_driver::SpecT<internal::ContextBound>*
          spec_base,
      const void* metadata_ptr,
      [[maybe_unused]] std::size_t component_index) override {
    assert(component_index == 0);
    auto& spec = static_cast<SpecT<internal::ContextBound>&>(*spec_base);
    const auto& metadata =
        *static_cast<const MultiscaleMetadata*>(metadata_ptr);
    const auto& scale = metadata.scales[scale_index_];
    spec.key_prefix = key_prefix_;
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

  std::string key_prefix_;
  std::size_t scale_index_;
  // channel, z, y, x
  StridedLayout<4> chunk_layout_czyx_;
};

class UnshardedDataCache : public DataCacheBase {
 public:
  explicit UnshardedDataCache(Initializer initializer,
                              absl::string_view key_prefix,
                              const MultiscaleMetadata& metadata,
                              std::size_t scale_index,
                              std::array<Index, 3> chunk_size_xyz)
      : DataCacheBase(std::move(initializer), key_prefix, metadata, scale_index,
                      chunk_size_xyz) {
    const auto& scale = metadata.scales[scale_index];
    scale_key_prefix_ = ResolveScaleKey(key_prefix, scale.key);
  }

  std::string GetChunkStorageKey(const void* metadata_ptr,
                                 span<const Index> cell_indices) override {
    const auto& metadata =
        *static_cast<const MultiscaleMetadata*>(metadata_ptr);
    std::string key = scale_key_prefix_;
    if (!key.empty()) key += '/';
    const auto& scale = metadata.scales[scale_index_];
    for (int i = 0; i < 3; ++i) {
      const Index chunk_size = chunk_layout_czyx_.shape()[3 - i];
      if (i != 0) key += '_';
      absl::StrAppend(
          &key, scale.box.origin()[i] + chunk_size * cell_indices[i], "-",
          scale.box.origin()[i] + std::min(chunk_size * (cell_indices[i] + 1),
                                           scale.box.shape()[i]));
    }
    return key;
  }

 private:
  /// Resolved key prefix for the scale.
  std::string scale_key_prefix_;
};

class ShardedDataCache : public DataCacheBase {
 public:
  explicit ShardedDataCache(Initializer initializer,
                            absl::string_view key_prefix,
                            const MultiscaleMetadata& metadata,
                            std::size_t scale_index,
                            std::array<Index, 3> chunk_size_xyz)
      : DataCacheBase(std::move(initializer), key_prefix, metadata, scale_index,
                      chunk_size_xyz) {
    const auto& scale = metadata.scales[scale_index];
    compressed_z_index_bits_ =
        GetCompressedZIndexBits(scale.box.shape(), chunk_size_xyz);
  }

  std::string GetChunkStorageKey(const void* metadata_ptr,
                                 span<const Index> cell_indices) override {
    assert(cell_indices.size() == 3);
    const std::uint64_t chunk_key = EncodeCompressedZIndex(
        {cell_indices.data(), 3}, compressed_z_index_bits_);
    return neuroglancer_uint64_sharded::ChunkIdToKey({chunk_key});
  }

  std::array<int, 3> compressed_z_index_bits_;
};

class NeuroglancerPrecomputedDriver
    : public internal_kvs_backed_chunk_driver::RegisteredKvsDriver<
          NeuroglancerPrecomputedDriver> {
  using Base = internal_kvs_backed_chunk_driver::RegisteredKvsDriver<
      NeuroglancerPrecomputedDriver>;

 public:
  using Base::Base;

  constexpr static char id[] = "neuroglancer_precomputed";

  template <template <typename> class MaybeBound = internal::ContextUnbound>
  using SpecT = internal_neuroglancer_precomputed::SpecT<MaybeBound>;

  static inline const auto json_binder = jb::Sequence(
      internal_kvs_backed_chunk_driver::SpecJsonBinder,
      jb::Member("path", jb::Projection(&SpecT<>::key_prefix,
                                        jb::DefaultInitializedValue())),
      [](auto is_loading, const auto& options, auto* obj, auto* j) {
        if constexpr (is_loading) {
          // TODO(jbms): Convert to use JSON binding framework for loading as
          // well.
          TENSORSTORE_ASSIGN_OR_RETURN(
              obj->open_constraints,
              OpenConstraints::Parse(*j, obj->data_type));
          // Erase members that were parsed to prevent error about extra
          // members.
          j->erase("scale_metadata");
          j->erase("multiscale_metadata");
          j->erase("scale_index");
          return absl::OkStatus();
        } else {
          return jb::Projection(
              &SpecT<>::open_constraints,
              jb::Sequence(
                  jb::Member("scale_index",
                             jb::Projection(&OpenConstraints::scale_index)),
                  jb::Member(
                      "scale_metadata",
                      jb::Projection(
                          &OpenConstraints::scale,
                          jb::DefaultInitializedValue<
                              /*DisallowIncludeDefaults=*/true>(jb::Object(
                              jb::Member("key",
                                         jb::Projection(
                                             &ScaleMetadataConstraints::key)),
                              jb::Member(
                                  "resolution",
                                  jb::Projection(
                                      &ScaleMetadataConstraints::resolution)),
                              jb::Member(
                                  "chunk_size",
                                  jb::Projection(
                                      &ScaleMetadataConstraints::chunk_size)),
                              jb::Member(
                                  "voxel_offset",
                                  jb::Projection(
                                      &ScaleMetadataConstraints::box,
                                      jb::Optional(jb::Projection([](auto& b) {
                                        return b.origin();
                                      })))),
                              jb::Member(
                                  "size",
                                  jb::Projection(
                                      &ScaleMetadataConstraints::box,
                                      jb::Optional(jb::Projection(
                                          [](auto& b) { return b.shape(); })))),
                              jb::Member(
                                  "encoding",
                                  jb::Projection(
                                      &ScaleMetadataConstraints::encoding,
                                      jb::Optional([](auto is_loading,
                                                      const auto& options,
                                                      auto* obj, auto* j) {
                                        if constexpr (!is_loading) {
                                          *j = to_string(*obj);
                                        }
                                        return absl::OkStatus();
                                      }))),
                              jb::Member(
                                  "compressed_segmentation_block_size",
                                  jb::Projection(
                                      &ScaleMetadataConstraints::
                                          compressed_segmentation_block_size)),
                              jb::Member(
                                  "sharding",
                                  jb::Projection(
                                      &ScaleMetadataConstraints::sharding,
                                      jb::Optional([](auto is_loading,
                                                      const auto& options,
                                                      auto* obj, auto* j) {
                                        if constexpr (!is_loading) {
                                          *j = ::nlohmann::json(*obj);
                                        }
                                        return absl::OkStatus();
                                      }))))))),
                  jb::Member(
                      "multiscale_metadata",
                      jb::Projection(
                          &OpenConstraints::multiscale,
                          jb::DefaultInitializedValue<
                              /*DisallowIncludeDefaults=*/true>(jb::Object(
                              jb::Member("num_channels",
                                         jb::Projection(
                                             &MultiscaleMetadataConstraints::
                                                 num_channels)),
                              jb::Member(kTypeId,
                                         jb::Projection(
                                             &MultiscaleMetadataConstraints::
                                                 type))))))))(is_loading,
                                                              options, obj, j);
        }
      });

  class OpenState;

  static Status ConvertSpec(SpecT<>* spec, const SpecRequestOptions& options) {
    if (options.minimal_spec()) {
      spec->open_constraints.scale = ScaleMetadataConstraints{};
      spec->open_constraints.multiscale = MultiscaleMetadataConstraints{};
    }
    return Base::ConvertSpec(spec, options);
  }
};

class NeuroglancerPrecomputedDriver::OpenState
    : public NeuroglancerPrecomputedDriver::OpenStateBase {
 public:
  using NeuroglancerPrecomputedDriver::OpenStateBase::OpenStateBase;

  std::string GetPrefixForDeleteExisting() override {
    // TODO(jbms): Possibly change behavior in the future to allow deleting
    // just a single scale.
    return spec().key_prefix.empty() ? std::string()
                                     : StrCat(spec().key_prefix, "/");
  }

  std::string GetMetadataCacheEntryKey() override { return spec().key_prefix; }

  std::unique_ptr<internal_kvs_backed_chunk_driver::MetadataCache>
  GetMetadataCache(MetadataCache::Initializer initializer) override {
    return std::make_unique<MetadataCache>(std::move(initializer));
  }

  std::string GetDataCacheKey(const void* metadata) override {
    std::string result;
    const auto& spec = this->spec();
    internal::EncodeCacheKey(
        &result, spec.key_prefix,
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

  Result<std::shared_ptr<const void>> Create(
      const void* existing_metadata) override {
    const auto* metadata =
        static_cast<const MultiscaleMetadata*>(existing_metadata);
    if (auto result = CreateScale(metadata, spec().open_constraints)) {
      scale_index_ = result->second;
      return result->first;
    } else {
      scale_index_ = std::nullopt;
      return std::move(result).status();
    }
  }

  std::unique_ptr<internal_kvs_backed_chunk_driver::DataCache> GetDataCache(
      internal_kvs_backed_chunk_driver::DataCache::Initializer initializer)
      override {
    const auto& metadata =
        *static_cast<const MultiscaleMetadata*>(initializer.metadata.get());
    assert(scale_index_);
    const auto& scale = metadata.scales[scale_index_.value()];
    if (std::holds_alternative<ShardingSpec>(scale.sharding)) {
      return std::make_unique<ShardedDataCache>(
          std::move(initializer), spec().key_prefix, metadata,
          scale_index_.value(), chunk_size_xyz_);
    } else {
      return std::make_unique<UnshardedDataCache>(
          std::move(initializer), spec().key_prefix, metadata,
          scale_index_.value(), chunk_size_xyz_);
    }
  }

  Result<std::size_t> GetComponentIndex(const void* metadata_ptr,
                                        OpenMode open_mode) override {
    const auto& metadata =
        *static_cast<const MultiscaleMetadata*>(metadata_ptr);
    // Check for compatibility
    if (spec().data_type.valid() && spec().data_type != metadata.data_type) {
      return absl::FailedPreconditionError(
          StrCat("Expected data type of ", spec().data_type,
                 " but received: ", metadata.data_type));
    }
    // FIXME: avoid copy by changing OpenScale to take separate arguments
    auto open_constraints = spec().open_constraints;
    if (scale_index_) {
      if (spec().open_constraints.scale_index) {
        assert(*spec().open_constraints.scale_index == *scale_index_);
      } else {
        open_constraints.scale_index = *scale_index_;
      }
    }
    if (auto result = OpenScale(metadata, open_constraints, open_mode)) {
      scale_index_ = *result;
      const auto& scale = metadata.scales[*result];
      if (spec().open_constraints.scale.chunk_size &&
          absl::c_linear_search(scale.chunk_sizes,
                                *spec().open_constraints.scale.chunk_size)) {
        // Use the specified chunk size.
        chunk_size_xyz_ = *spec().open_constraints.scale.chunk_size;
      } else {
        // Chunk size was unspecified or not found.  It is only possible for a
        // specified chunk size not to be found if `open_mode` specifies
        // `allow_option_mismatch`.
        assert(!spec().open_constraints.scale.chunk_size ||
               !!(open_mode & OpenMode::allow_option_mismatch));
        chunk_size_xyz_ = scale.chunk_sizes[0];
      }
      // Component index is always 0.
      return 0;
    } else {
      return std::move(result).status();
    }
  }

  Result<KeyValueStore::Ptr> GetDataKeyValueStore(
      KeyValueStore::Ptr base_kv_store, const void* metadata_ptr) override {
    const auto& metadata =
        *static_cast<const MultiscaleMetadata*>(metadata_ptr);
    assert(scale_index_);
    const auto& scale = metadata.scales[*scale_index_];
    if (auto* sharding_spec = std::get_if<ShardingSpec>(&scale.sharding)) {
      assert(scale.chunk_sizes.size() == 1);
      return neuroglancer_uint64_sharded::GetShardedKeyValueStore(
          std::move(base_kv_store), executor(),
          ResolveScaleKey(spec().key_prefix, scale.key), *sharding_spec,
          *cache_pool(),
          GetChunksPerVolumeShardFunction(*sharding_spec, scale.box.shape(),
                                          scale.chunk_sizes[0]));
    }
    return base_kv_store;
  }

  // Set by `Create` or `GetComponentIndex` to indicate the scale index that
  // has been determined.
  std::optional<std::size_t> scale_index_;
  // Set by `GetComponentIndex` to indicate the chunk size that has been
  // determined.
  std::array<Index, 3> chunk_size_xyz_;
};

const internal::DriverRegistration<NeuroglancerPrecomputedDriver> registration;

}  // namespace

}  // namespace internal_neuroglancer_precomputed
}  // namespace tensorstore
