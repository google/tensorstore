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

#include "absl/status/status.h"
#include "tensorstore/context.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/kvs_backed_chunk_driver.h"
#include "tensorstore/driver/n5/metadata.h"
#include "tensorstore/driver/registry.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/internal/cache/chunk_cache.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/util/constant_vector.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_n5 {

namespace {

namespace jb = tensorstore::internal_json_binding;

using ::tensorstore::internal_kvs_backed_chunk_driver::KvsDriverSpec;

constexpr const char kMetadataKey[] = "attributes.json";

class N5DriverSpec
    : public internal::RegisteredDriverSpec<N5DriverSpec,
                                            /*Parent=*/KvsDriverSpec> {
 public:
  constexpr static char id[] = "n5";

  using Base = internal::RegisteredDriverSpec<N5DriverSpec,
                                              /*Parent=*/KvsDriverSpec>;

  N5MetadataConstraints metadata_constraints;

  constexpr static auto ApplyMembers = [](auto& x, auto f) {
    return f(internal::BaseCast<KvsDriverSpec>(x), x.metadata_constraints);
  };

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
              jb::Projection<&N5DriverSpec::metadata_constraints>(
                  jb::DefaultInitializedValue()))));

  absl::Status ApplyOptions(SpecOptions&& options) override {
    if (options.minimal_spec) {
      metadata_constraints = N5MetadataConstraints{};
    }
    return Base::ApplyOptions(std::move(options));
  }

  Result<IndexDomain<>> GetDomain() const override {
    return GetEffectiveDomain(metadata_constraints, schema);
  }

  Result<CodecSpec> GetCodec() const override {
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto codec, GetEffectiveCodec(metadata_constraints, schema));
    return CodecSpec(std::move(codec));
  }

  Result<ChunkLayout> GetChunkLayout() const override {
    return GetEffectiveChunkLayout(metadata_constraints, schema);
  }

  Result<SharedArray<const void>> GetFillValue(
      IndexTransformView<> transform) const override {
    return {std::in_place};
  }

  Result<DimensionUnitsVector> GetDimensionUnits() const override {
    return GetEffectiveDimensionUnits(metadata_constraints.rank,
                                      metadata_constraints.units_and_resolution,
                                      schema.dimension_units());
  }

  Future<internal::Driver::Handle> Open(
      internal::OpenTransactionPtr transaction,
      ReadWriteMode read_write_mode) const override;
};

Result<std::shared_ptr<const N5Metadata>> ParseEncodedMetadata(
    std::string_view encoded_value) {
  nlohmann::json raw_data = nlohmann::json::parse(encoded_value, nullptr,
                                                  /*allow_exceptions=*/false);
  if (raw_data.is_discarded()) {
    return absl::FailedPreconditionError("Invalid JSON");
  }
  TENSORSTORE_ASSIGN_OR_RETURN(auto metadata,
                               N5Metadata::FromJson(std::move(raw_data)));
  return std::make_shared<N5Metadata>(std::move(metadata));
}

class MetadataCache : public internal_kvs_backed_chunk_driver::MetadataCache {
  using Base = internal_kvs_backed_chunk_driver::MetadataCache;

 public:
  using Base::Base;

  // Metadata is stored as JSON under the `attributes.json` key.
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
        ::nlohmann::json(*static_cast<const N5Metadata*>(metadata)).dump());
  }
};

class DataCache : public internal_kvs_backed_chunk_driver::DataCache {
  using Base = internal_kvs_backed_chunk_driver::DataCache;

 public:
  explicit DataCache(Initializer initializer, std::string key_prefix)
      : Base(initializer,
             GetChunkGridSpecification(
                 *static_cast<const N5Metadata*>(initializer.metadata.get()))),
        key_prefix_(std::move(key_prefix)) {}

  absl::Status ValidateMetadataCompatibility(
      const void* existing_metadata_ptr,
      const void* new_metadata_ptr) override {
    const auto& existing_metadata =
        *static_cast<const N5Metadata*>(existing_metadata_ptr);
    const auto& new_metadata =
        *static_cast<const N5Metadata*>(new_metadata_ptr);
    auto existing_key = existing_metadata.GetCompatibilityKey();
    auto new_key = new_metadata.GetCompatibilityKey();
    if (existing_key == new_key) return absl::OkStatus();
    return absl::FailedPreconditionError(tensorstore::StrCat(
        "Updated N5 metadata ", new_key,
        " is incompatible with existing metadata ", existing_key));
  }

  void GetChunkGridBounds(const void* metadata_ptr, MutableBoxView<> bounds,
                          DimensionSet& implicit_lower_bounds,
                          DimensionSet& implicit_upper_bounds) override {
    const auto& metadata = *static_cast<const N5Metadata*>(metadata_ptr);
    assert(bounds.rank() == static_cast<DimensionIndex>(metadata.shape.size()));
    std::fill(bounds.origin().begin(), bounds.origin().end(), Index(0));
    std::copy(metadata.shape.begin(), metadata.shape.end(),
              bounds.shape().begin());
    implicit_lower_bounds = false;
    implicit_upper_bounds = true;
  }

  Result<std::shared_ptr<const void>> GetResizedMetadata(
      const void* existing_metadata, span<const Index> new_inclusive_min,
      span<const Index> new_exclusive_max) override {
    auto new_metadata = std::make_shared<N5Metadata>(
        *static_cast<const N5Metadata*>(existing_metadata));
    const DimensionIndex rank = new_metadata->shape.size();
    assert(rank == new_inclusive_min.size());
    assert(rank == new_exclusive_max.size());
    for (DimensionIndex i = 0; i < rank; ++i) {
      assert(ExplicitIndexOr(new_inclusive_min[i], 0) == 0);
      const Index new_size = new_exclusive_max[i];
      if (new_size == kImplicit) continue;
      new_metadata->shape[i] = new_size;
    }
    return new_metadata;
  }

  static internal::ChunkGridSpecification GetChunkGridSpecification(
      const N5Metadata& metadata) {
    SharedArray<const void> fill_value(
        internal::AllocateAndConstructSharedElements(1, value_init,
                                                     metadata.dtype),
        StridedLayout<>(metadata.chunk_layout.shape(),
                        GetConstantVector<Index, 0>(metadata.rank)));
    return internal::ChunkGridSpecification(
        {internal::ChunkGridSpecification::Component(std::move(fill_value),
                                                     // Since all dimensions are
                                                     // resizable, just specify
                                                     // unbounded
                                                     // `component_bounds`.
                                                     Box<>(metadata.rank))});
  }

  Result<absl::InlinedVector<SharedArrayView<const void>, 1>> DecodeChunk(
      const void* metadata, span<const Index> chunk_indices,
      absl::Cord data) override {
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto array,
        internal_n5::DecodeChunk(*static_cast<const N5Metadata*>(metadata),
                                 std::move(data)));
    return absl::InlinedVector<SharedArrayView<const void>, 1>{
        std::move(array)};
  }

  Result<absl::Cord> EncodeChunk(
      const void* metadata, span<const Index> chunk_indices,
      span<const SharedArrayView<const void>> component_arrays) override {
    assert(component_arrays.size() == 1);
    return internal_n5::EncodeChunk(chunk_indices,
                                    *static_cast<const N5Metadata*>(metadata),
                                    component_arrays[0]);
  }

  std::string GetChunkStorageKey(const void* metadata,
                                 span<const Index> cell_indices) override {
    // Use "0" for rank 0 as a special case.
    std::string key = tensorstore::StrCat(
        key_prefix_, cell_indices.empty() ? 0 : cell_indices[0]);
    for (DimensionIndex i = 1; i < cell_indices.size(); ++i) {
      tensorstore::StrAppend(&key, "/", cell_indices[i]);
    }
    return key;
  }

  Result<IndexTransform<>> GetExternalToInternalTransform(
      const void* metadata_ptr, std::size_t component_index) override {
    assert(component_index == 0);
    const auto& metadata = *static_cast<const N5Metadata*>(metadata_ptr);
    const auto& axes = metadata.axes;
    const DimensionIndex rank = metadata.axes.size();
    auto builder = tensorstore::IndexTransformBuilder<>(rank, rank)
                       .input_shape(metadata.shape)
                       .input_labels(axes);
    builder.implicit_upper_bounds(true);
    for (DimensionIndex i = 0; i < rank; ++i) {
      builder.output_single_input_dimension(i, i);
    }
    return builder.Finalize();
  }

  absl::Status GetBoundSpecData(KvsDriverSpec& spec_base,
                                const void* metadata_ptr,
                                std::size_t component_index) override {
    assert(component_index == 0);
    auto& spec = static_cast<N5DriverSpec&>(spec_base);
    const auto& metadata = *static_cast<const N5Metadata*>(metadata_ptr);
    auto& constraints = spec.metadata_constraints;
    constraints.shape = metadata.shape;
    constraints.axes = metadata.axes;
    constraints.dtype = metadata.dtype;
    constraints.compressor = metadata.compressor;
    constraints.units_and_resolution = metadata.units_and_resolution;
    constraints.extra_attributes = metadata.extra_attributes;
    constraints.chunk_shape =
        std::vector<Index>(metadata.chunk_layout.shape().begin(),
                           metadata.chunk_layout.shape().end());
    return absl::OkStatus();
  }

  Result<ChunkLayout> GetChunkLayout(const void* metadata_ptr,
                                     std::size_t component_index) override {
    const auto& metadata = *static_cast<const N5Metadata*>(metadata_ptr);
    ChunkLayout chunk_layout;
    TENSORSTORE_RETURN_IF_ERROR(SetChunkLayoutFromMetadata(
        metadata.rank, metadata.chunk_shape, chunk_layout));
    TENSORSTORE_RETURN_IF_ERROR(chunk_layout.Finalize());
    return chunk_layout;
  }

  Result<CodecSpec> GetCodec(const void* metadata,
                             std::size_t component_index) override {
    assert(component_index == 0);
    return GetCodecFromMetadata(*static_cast<const N5Metadata*>(metadata));
  }

  std::string GetBaseKvstorePath() override { return key_prefix_; }

  std::string key_prefix_;
};

class N5Driver : public internal_kvs_backed_chunk_driver::RegisteredKvsDriver<
                     N5Driver, N5DriverSpec> {
  using Base =
      internal_kvs_backed_chunk_driver::RegisteredKvsDriver<N5Driver,
                                                            N5DriverSpec>;

 public:
  using Base::Base;

  class OpenState;

  Result<DimensionUnitsVector> GetDimensionUnits() override {
    auto* cache = static_cast<DataCache*>(this->cache());
    const auto& metadata =
        *static_cast<const N5Metadata*>(cache->initial_metadata_.get());
    return internal_n5::GetDimensionUnits(metadata.rank,
                                          metadata.units_and_resolution);
  }
};

class N5Driver::OpenState : public N5Driver::OpenStateBase {
 public:
  using N5Driver::OpenStateBase::OpenStateBase;

  std::string GetPrefixForDeleteExisting() override {
    return spec().store.path;
  }

  std::string GetMetadataCacheEntryKey() override { return spec().store.path; }

  // The metadata cache isn't parameterized by anything other than the
  // KeyValueStore; therefore, we don't need to override `GetMetadataCacheKey`
  // to encode the state.
  std::unique_ptr<internal_kvs_backed_chunk_driver::MetadataCache>
  GetMetadataCache(MetadataCache::Initializer initializer) override {
    return std::make_unique<MetadataCache>(std::move(initializer));
  }

  std::string GetDataCacheKey(const void* metadata) override {
    std::string result;
    internal::EncodeCacheKey(
        &result, spec().store.path,
        static_cast<const N5Metadata*>(metadata)->GetCompatibilityKey());
    return result;
  }

  Result<std::shared_ptr<const void>> Create(
      const void* existing_metadata) override {
    if (existing_metadata) {
      return absl::AlreadyExistsError("");
    }
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto metadata,
        internal_n5::GetNewMetadata(spec().metadata_constraints, spec().schema),
        tensorstore::MaybeAnnotateStatus(
            _, "Cannot create using specified \"metadata\" and schema"));
    return metadata;
  }

  std::unique_ptr<internal_kvs_backed_chunk_driver::DataCache> GetDataCache(
      DataCache::Initializer initializer) override {
    return std::make_unique<DataCache>(std::move(initializer),
                                       spec().store.path);
  }

  Result<std::size_t> GetComponentIndex(const void* metadata_ptr,
                                        OpenMode open_mode) override {
    const auto& metadata = *static_cast<const N5Metadata*>(metadata_ptr);
    TENSORSTORE_RETURN_IF_ERROR(
        ValidateMetadata(metadata, spec().metadata_constraints));
    TENSORSTORE_RETURN_IF_ERROR(
        ValidateMetadataSchema(metadata, spec().schema));
    return 0;
  }
};

Future<internal::Driver::Handle> N5DriverSpec::Open(
    internal::OpenTransactionPtr transaction,
    ReadWriteMode read_write_mode) const {
  return N5Driver::Open(std::move(transaction), this, read_write_mode);
}

}  // namespace
}  // namespace internal_n5
}  // namespace tensorstore

TENSORSTORE_DECLARE_GARBAGE_COLLECTION_SPECIALIZATION(
    tensorstore::internal_n5::N5Driver)
// Use default garbage collection implementation provided by
// kvs_backed_chunk_driver (just handles the kvstore)
TENSORSTORE_DEFINE_GARBAGE_COLLECTION_SPECIALIZATION(
    tensorstore::internal_n5::N5Driver,
    tensorstore::internal_n5::N5Driver::GarbageCollectionBase)

namespace {
const tensorstore::internal::DriverRegistration<
    tensorstore::internal_n5::N5DriverSpec>
    registration;
}  // namespace
