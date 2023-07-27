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

#include <vector>

#include "absl/status/status.h"
#include "tensorstore/context.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/kvs_backed_chunk_driver.h"
#include "tensorstore/driver/registry.h"
#include "tensorstore/driver/zarr/driver_impl.h"
#include "tensorstore/driver/zarr/metadata.h"
#include "tensorstore/driver/zarr/spec.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/index_domain_builder.h"
#include "tensorstore/index_space/transform_broadcastable_array.h"
#include "tensorstore/internal/cache/chunk_cache.h"
#include "tensorstore/internal/cache_key/cache_key.h"
#include "tensorstore/internal/grid_storage_statistics.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_zarr {

namespace {

namespace jb = tensorstore::internal_json_binding;

using ::tensorstore::internal_kvs_backed_chunk_driver::KvsDriverSpec;

constexpr const char kDefaultMetadataKey[] = ".zarray";

inline char GetDimensionSeparatorChar(DimensionSeparator dimension_separator) {
  return dimension_separator == DimensionSeparator::kDotSeparated ? '.' : '/';
}

DimensionSeparator GetDimensionSeparator(
    const ZarrPartialMetadata& partial_metadata, const ZarrMetadata& metadata) {
  if (metadata.dimension_separator) {
    return *metadata.dimension_separator;
  } else if (partial_metadata.dimension_separator) {
    return *partial_metadata.dimension_separator;
  }
  return DimensionSeparator::kDotSeparated;
}

Result<ZarrMetadataPtr> ParseEncodedMetadata(std::string_view encoded_value) {
  nlohmann::json raw_data = nlohmann::json::parse(encoded_value, nullptr,
                                                  /*allow_exceptions=*/false);
  if (raw_data.is_discarded()) {
    return absl::FailedPreconditionError("Invalid JSON");
  }
  auto metadata = std::make_shared<ZarrMetadata>();
  TENSORSTORE_ASSIGN_OR_RETURN(*metadata,
                               ZarrMetadata::FromJson(std::move(raw_data)));
  return metadata;
}

}  // namespace

std::string MetadataCache::GetMetadataStorageKey(std::string_view entry_key) {
  return std::string(entry_key);
}

Result<MetadataCache::MetadataPtr> MetadataCache::DecodeMetadata(
    std::string_view entry_key, absl::Cord encoded_metadata) {
  return ParseEncodedMetadata(encoded_metadata.Flatten());
}

Result<absl::Cord> MetadataCache::EncodeMetadata(std::string_view entry_key,
                                                 const void* metadata) {
  return absl::Cord(
      ::nlohmann::json(*static_cast<const ZarrMetadata*>(metadata)).dump());
}

absl::Status ZarrDriverSpec::ApplyOptions(SpecOptions&& options) {
  if (options.minimal_spec) {
    partial_metadata = ZarrPartialMetadata{};
  }
  return Base::ApplyOptions(std::move(options));
}

Result<SpecRankAndFieldInfo> ZarrDriverSpec::GetSpecInfo() const {
  return GetSpecRankAndFieldInfo(partial_metadata, selected_field, schema);
}

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(
    ZarrDriverSpec,
    jb::Sequence(
        internal_kvs_backed_chunk_driver::SpecJsonBinder,
        jb::Member("metadata",
                   jb::Projection<&ZarrDriverSpec::partial_metadata>(
                       jb::DefaultInitializedValue())),
        jb::Member("metadata_key",
                   jb::Projection<&ZarrDriverSpec::metadata_key>(
                       jb::DefaultValue<jb::kNeverIncludeDefaults>(
                           [](auto* obj) { *obj = kDefaultMetadataKey; }))),
        // Deprecated `key_encoding` property.
        jb::LoadSave(jb::OptionalMember(
            "key_encoding",
            jb::Compose<DimensionSeparator>(
                [](auto is_loading, const auto& options, auto* obj,
                   DimensionSeparator* value) {
                  auto& sep = obj->partial_metadata.dimension_separator;
                  if (sep && *sep != *value) {
                    return absl::InvalidArgumentError(tensorstore::StrCat(
                        "value (", ::nlohmann::json(*value).dump(),
                        ") does not match value in metadata (",
                        ::nlohmann::json(*sep).dump(), ")"));
                  }
                  sep = *value;
                  return absl::OkStatus();
                },
                DimensionSeparatorJsonBinder))),
        jb::Member("field", jb::Projection<&ZarrDriverSpec::selected_field>(
                                jb::DefaultValue<jb::kNeverIncludeDefaults>(
                                    [](auto* obj) { *obj = std::string{}; }))),
        jb::Initialize([](auto* obj) {
          TENSORSTORE_ASSIGN_OR_RETURN(auto info, obj->GetSpecInfo());
          if (info.full_rank != dynamic_rank) {
            TENSORSTORE_RETURN_IF_ERROR(
                obj->schema.Set(RankConstraint(info.full_rank)));
          }
          if (info.field) {
            TENSORSTORE_RETURN_IF_ERROR(obj->schema.Set(info.field->dtype));
          }
          return absl::OkStatus();
        })));

Result<IndexDomain<>> ZarrDriverSpec::GetDomain() const {
  TENSORSTORE_ASSIGN_OR_RETURN(auto info, GetSpecInfo());
  return GetDomainFromMetadata(info, partial_metadata.shape, schema);
}

Result<CodecSpec> ZarrDriverSpec::GetCodec() const {
  auto codec_spec = internal::CodecDriverSpec::Make<ZarrCodecSpec>();
  codec_spec->compressor = partial_metadata.compressor;
  TENSORSTORE_RETURN_IF_ERROR(codec_spec->MergeFrom(schema.codec()));
  return codec_spec;
}

Result<ChunkLayout> ZarrDriverSpec::GetChunkLayout() const {
  auto chunk_layout = schema.chunk_layout();
  TENSORSTORE_ASSIGN_OR_RETURN(auto info, GetSpecInfo());
  TENSORSTORE_RETURN_IF_ERROR(SetChunkLayoutFromMetadata(
      info, partial_metadata.chunks, partial_metadata.order, chunk_layout));
  return chunk_layout;
}

Result<SharedArray<const void>> ZarrDriverSpec::GetFillValue(
    IndexTransformView<> transform) const {
  SharedArrayView<const void> fill_value = schema.fill_value();

  const auto& metadata = partial_metadata;
  if (metadata.dtype && metadata.fill_value) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        size_t field_index, GetFieldIndex(*metadata.dtype, selected_field));
    fill_value = (*metadata.fill_value)[field_index];
  }

  if (!fill_value.valid() || !transform.valid()) {
    return SharedArray<const void>(fill_value);
  }

  const DimensionIndex output_rank = transform.output_rank();
  if (output_rank < fill_value.rank()) {
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Transform with output rank ", output_rank,
                            " is not compatible with metadata"));
  }
  Index pseudo_shape[kMaxRank];
  std::fill_n(pseudo_shape, output_rank - fill_value.rank(), kInfIndex + 1);
  for (DimensionIndex i = 0; i < fill_value.rank(); ++i) {
    Index size = fill_value.shape()[i];
    if (size == 1) size = kInfIndex + 1;
    pseudo_shape[output_rank - fill_value.rank() + i] = size;
  }
  return TransformOutputBroadcastableArray(
      transform, std::move(fill_value),
      IndexDomain(span(pseudo_shape, output_rank)));
}

DataCache::DataCache(Initializer&& initializer, std::string key_prefix,
                     DimensionSeparator dimension_separator,
                     std::string metadata_key)
    : Base(std::move(initializer),
           GetChunkGridSpecification(
               *static_cast<const ZarrMetadata*>(initializer.metadata.get()))),
      key_prefix_(std::move(key_prefix)),
      dimension_separator_(dimension_separator),
      metadata_key_(std::move(metadata_key)) {}

absl::Status DataCache::ValidateMetadataCompatibility(
    const void* existing_metadata_ptr, const void* new_metadata_ptr) {
  assert(existing_metadata_ptr);
  assert(new_metadata_ptr);
  const auto& existing_metadata =
      *static_cast<const ZarrMetadata*>(existing_metadata_ptr);
  const auto& new_metadata =
      *static_cast<const ZarrMetadata*>(new_metadata_ptr);
  if (IsMetadataCompatible(existing_metadata, new_metadata)) {
    return absl::OkStatus();
  }
  return absl::FailedPreconditionError(tensorstore::StrCat(
      "Updated zarr metadata ", ::nlohmann::json(new_metadata).dump(),
      " is incompatible with existing metadata ",
      ::nlohmann::json(existing_metadata).dump()));
}

void DataCache::GetChunkGridBounds(const void* metadata_ptr,
                                   MutableBoxView<> bounds,
                                   DimensionSet& implicit_lower_bounds,
                                   DimensionSet& implicit_upper_bounds) {
  const auto& metadata = *static_cast<const ZarrMetadata*>(metadata_ptr);
  assert(bounds.rank() == static_cast<DimensionIndex>(metadata.shape.size()));
  std::fill(bounds.origin().begin(), bounds.origin().end(), Index(0));
  std::copy(metadata.shape.begin(), metadata.shape.end(),
            bounds.shape().begin());
  implicit_lower_bounds = false;
  implicit_upper_bounds = true;
}

Result<std::shared_ptr<const void>> DataCache::GetResizedMetadata(
    const void* existing_metadata, span<const Index> new_inclusive_min,
    span<const Index> new_exclusive_max) {
  auto new_metadata = std::make_shared<ZarrMetadata>(
      *static_cast<const ZarrMetadata*>(existing_metadata));
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

internal::ChunkGridSpecification DataCache::GetChunkGridSpecification(
    const ZarrMetadata& metadata) {
  internal::ChunkGridSpecification::ComponentList components;
  components.reserve(metadata.dtype.fields.size());
  std::vector<DimensionIndex> chunked_to_cell_dimensions(
      metadata.chunks.size());
  std::iota(chunked_to_cell_dimensions.begin(),
            chunked_to_cell_dimensions.end(), static_cast<DimensionIndex>(0));
  for (std::size_t field_i = 0; field_i < metadata.dtype.fields.size();
       ++field_i) {
    const auto& field = metadata.dtype.fields[field_i];
    const auto& field_layout = metadata.chunk_layout.fields[field_i];
    auto fill_value = metadata.fill_value[field_i];
    const bool fill_value_specified = fill_value.valid();
    if (!fill_value.valid()) {
      // Use value-initialized rank-0 fill value.
      fill_value = AllocateArray(span<const Index, 0>{}, c_order, value_init,
                                 field.dtype);
    }
    assert(fill_value.rank() <=
           static_cast<DimensionIndex>(field.field_shape.size()));
    const DimensionIndex cell_rank = field_layout.full_chunk_shape().size();
    SharedArray<const void> chunk_fill_value;
    chunk_fill_value.layout().set_rank(cell_rank);
    chunk_fill_value.element_pointer() = fill_value.element_pointer();
    const DimensionIndex fill_value_start_dim = cell_rank - fill_value.rank();
    for (DimensionIndex cell_dim = 0; cell_dim < fill_value_start_dim;
         ++cell_dim) {
      chunk_fill_value.shape()[cell_dim] =
          field_layout.full_chunk_shape()[cell_dim];
      chunk_fill_value.byte_strides()[cell_dim] = 0;
    }
    for (DimensionIndex cell_dim = fill_value_start_dim; cell_dim < cell_rank;
         ++cell_dim) {
      const Index size = field_layout.full_chunk_shape()[cell_dim];
      assert(fill_value.shape()[cell_dim - fill_value_start_dim] == size);
      chunk_fill_value.shape()[cell_dim] = size;
      chunk_fill_value.byte_strides()[cell_dim] =
          fill_value.byte_strides()[cell_dim - fill_value_start_dim];
    }
    components.emplace_back(std::move(chunk_fill_value),
                            // Since all chunked dimensions are resizable in
                            // zarr, just specify unbounded
                            // `component_bounds`.
                            Box<>(cell_rank), chunked_to_cell_dimensions);
    components.back().store_if_equal_to_fill_value = !fill_value_specified;
  }
  return internal::ChunkGridSpecification{std::move(components)};
}

Result<absl::InlinedVector<SharedArray<const void>, 1>> DataCache::DecodeChunk(
    span<const Index> chunk_indices, absl::Cord data) {
  return internal_zarr::DecodeChunk(metadata(), std::move(data));
}

Result<absl::Cord> DataCache::EncodeChunk(
    span<const Index> chunk_indices,
    span<const SharedArrayView<const void>> component_arrays) {
  return internal_zarr::EncodeChunk(metadata(), component_arrays);
}

std::string DataCache::GetChunkStorageKey(span<const Index> cell_indices) {
  return tensorstore::StrCat(
      key_prefix_, EncodeChunkIndices(cell_indices, dimension_separator_));
}

absl::Status DataCache::GetBoundSpecData(
    internal_kvs_backed_chunk_driver::KvsDriverSpec& spec_base,
    const void* metadata_ptr, std::size_t component_index) {
  auto& spec = static_cast<ZarrDriverSpec&>(spec_base);
  const auto& metadata = *static_cast<const ZarrMetadata*>(metadata_ptr);
  spec.selected_field = EncodeSelectedField(component_index, metadata.dtype);
  spec.metadata_key = metadata_key_;
  auto& pm = spec.partial_metadata;
  pm.rank = metadata.rank;
  pm.zarr_format = metadata.zarr_format;
  pm.shape = metadata.shape;
  pm.chunks = metadata.chunks;
  pm.compressor = metadata.compressor;
  pm.filters = metadata.filters;
  pm.order = metadata.order;
  pm.dtype = metadata.dtype;
  pm.fill_value = metadata.fill_value;
  pm.dimension_separator = dimension_separator_;
  return absl::OkStatus();
}

Result<ChunkLayout> DataCache::GetChunkLayoutFromMetadata(
    const void* metadata_ptr, size_t component_index) {
  const auto& metadata = *static_cast<const ZarrMetadata*>(metadata_ptr);
  ChunkLayout chunk_layout;
  TENSORSTORE_RETURN_IF_ERROR(internal_zarr::SetChunkLayoutFromMetadata(
      GetSpecRankAndFieldInfo(metadata, component_index), metadata.chunks,
      metadata.order, chunk_layout));
  TENSORSTORE_RETURN_IF_ERROR(chunk_layout.Finalize());
  return chunk_layout;
}

std::string DataCache::GetBaseKvstorePath() { return key_prefix_; }
Result<CodecSpec> ZarrDriver::GetCodec() {
  return internal_zarr::GetCodecSpecFromMetadata(metadata());
}

Result<SharedArray<const void>> ZarrDriver::GetFillValue(
    IndexTransformView<> transform) {
  const auto& metadata = this->metadata();
  const auto& fill_value = metadata.fill_value[this->component_index()];
  if (!fill_value.valid()) return {std::in_place};
  const auto& field = metadata.dtype.fields[this->component_index()];
  IndexDomainBuilder builder(field.field_shape.size() + metadata.rank);
  span<Index> shape = builder.shape();
  std::fill_n(shape.begin(), metadata.rank, kInfIndex + 1);
  std::copy(field.field_shape.begin(), field.field_shape.end(),
            shape.end() - field.field_shape.size());
  TENSORSTORE_ASSIGN_OR_RETURN(auto output_domain, builder.Finalize());
  return TransformOutputBroadcastableArray(transform, fill_value,
                                           output_domain);
}

Future<internal::Driver::Handle> ZarrDriverSpec::Open(
    internal::OpenTransactionPtr transaction,
    ReadWriteMode read_write_mode) const {
  return ZarrDriver::Open(std::move(transaction), this, read_write_mode);
}

Future<ArrayStorageStatistics> ZarrDriver::GetStorageStatistics(
    internal::OpenTransactionPtr transaction, IndexTransform<> transform,
    GetArrayStorageStatisticsOptions options) {
  auto* cache = static_cast<DataCache*>(this->cache());
  auto [promise, future] = PromiseFuturePair<ArrayStorageStatistics>::Make();
  auto metadata_future =
      ResolveMetadata(transaction, metadata_staleness_bound_.time);
  LinkValue(
      WithExecutor(
          cache->executor(),
          [cache = internal::CachePtr<DataCache>(cache),
           transform = std::move(transform),
           component_index = this->component_index(),
           transaction = std::move(transaction),
           staleness_bound = this->data_staleness_bound().time,
           options](Promise<ArrayStorageStatistics> promise,
                    ReadyFuture<MetadataCache::MetadataPtr> future) {
            auto* metadata =
                static_cast<const ZarrMetadata*>(future.value().get());
            auto& grid = cache->grid();
            auto& component = grid.components[component_index];
            LinkResult(
                std::move(promise),
                internal::GetStorageStatisticsForRegularGridWithBase10Keys(
                    KvStore{kvstore::DriverPtr(cache->kvstore_driver()),
                            cache->GetBaseKvstorePath(),
                            internal::TransactionState::ToTransaction(
                                std::move(transaction))},
                    transform, /*grid_output_dimensions=*/
                    component.chunked_to_cell_dimensions,
                    /*chunk_shape=*/grid.chunk_shape,
                    /*shape=*/metadata->shape,
                    /*dimension_separator=*/
                    GetDimensionSeparatorChar(cache->dimension_separator_),
                    staleness_bound, options));
          }),
      std::move(promise), std::move(metadata_future));
  return std::move(future);
}

class ZarrDriver::OpenState : public ZarrDriver::OpenStateBase {
 public:
  using ZarrDriver::OpenStateBase::OpenStateBase;

  std::string GetPrefixForDeleteExisting() override {
    return spec().store.path;
  }

  std::string GetMetadataCacheEntryKey() override {
    return tensorstore::StrCat(spec().store.path, spec().metadata_key);
  }

  std::unique_ptr<internal_kvs_backed_chunk_driver::MetadataCache>
  GetMetadataCache(MetadataCache::Initializer initializer) override {
    return std::make_unique<MetadataCache>(std::move(initializer));
  }

  Result<std::shared_ptr<const void>> Create(
      const void* existing_metadata) override {
    if (existing_metadata) {
      return absl::AlreadyExistsError("");
    }
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto metadata,
        internal_zarr::GetNewMetadata(spec().partial_metadata,
                                      spec().selected_field, spec().schema),
        tensorstore::MaybeAnnotateStatus(
            _, "Cannot create using specified \"metadata\" and schema"));
    return metadata;
  }

  std::string GetDataCacheKey(const void* metadata) override {
    std::string result;
    const auto& spec = this->spec();
    const auto& zarr_metadata = *static_cast<const ZarrMetadata*>(metadata);
    internal::EncodeCacheKey(
        &result, spec.store.path,
        GetDimensionSeparator(spec.partial_metadata, zarr_metadata),
        zarr_metadata, spec.metadata_key);
    return result;
  }

  std::unique_ptr<internal_kvs_backed_chunk_driver::DataCacheBase> GetDataCache(
      DataCache::Initializer&& initializer) override {
    const auto& metadata =
        *static_cast<const ZarrMetadata*>(initializer.metadata.get());
    return std::make_unique<DataCache>(
        std::move(initializer), spec().store.path,
        GetDimensionSeparator(spec().partial_metadata, metadata),
        spec().metadata_key);
  }

  Result<std::size_t> GetComponentIndex(const void* metadata_ptr,
                                        OpenMode open_mode) override {
    const auto& metadata = *static_cast<const ZarrMetadata*>(metadata_ptr);
    TENSORSTORE_RETURN_IF_ERROR(
        ValidateMetadata(metadata, spec().partial_metadata));
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto field_index, GetFieldIndex(metadata.dtype, spec().selected_field));
    TENSORSTORE_RETURN_IF_ERROR(
        ValidateMetadataSchema(metadata, field_index, spec().schema));
    return field_index;
  }
};

std::string EncodeChunkIndices(span<const Index> indices,
                               DimensionSeparator dimension_separator) {
  // Use "0" for rank 0 as a special case.
  const char separator = GetDimensionSeparatorChar(dimension_separator);
  std::string key = (indices.empty() ? "0" : tensorstore::StrCat(indices[0]));
  for (DimensionIndex i = 1; i < indices.size(); ++i) {
    tensorstore::StrAppend(&key, separator, indices[i]);
  }
  return key;
}

}  // namespace internal_zarr
}  // namespace tensorstore

// Use default garbage collection implementation provided by
// kvs_backed_chunk_driver (just handles the kvstore)
TENSORSTORE_DEFINE_GARBAGE_COLLECTION_SPECIALIZATION(
    tensorstore::internal_zarr::ZarrDriver,
    tensorstore::internal_zarr::ZarrDriver::GarbageCollectionBase)

namespace {
const tensorstore::internal::DriverRegistration<
    tensorstore::internal_zarr::ZarrDriverSpec>
    registration;
}  // namespace
