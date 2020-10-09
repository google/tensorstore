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

#include "tensorstore/context.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/kvs_backed_chunk_driver.h"
#include "tensorstore/driver/registry.h"
#include "tensorstore/driver/zarr/driver_impl.h"
#include "tensorstore/driver/zarr/metadata.h"
#include "tensorstore/driver/zarr/spec.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/cache_key.h"
#include "tensorstore/internal/chunk_cache.h"
#include "tensorstore/internal/json.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/kvstore/key_value_store.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace internal_zarr {

namespace {
constexpr const char kZarrMetadataKey[] = ".zarray";

inline char GetChunkKeyEncodingSeparator(ChunkKeyEncoding key_encoding) {
  return key_encoding == ChunkKeyEncoding::kDotSeparated ? '.' : '/';
}

Result<ZarrMetadataPtr> ParseEncodedMetadata(absl::string_view encoded_value) {
  nlohmann::json raw_data = nlohmann::json::parse(encoded_value, nullptr,
                                                  /*allow_exceptions=*/false);
  if (raw_data.is_discarded()) {
    return absl::FailedPreconditionError("Invalid JSON");
  }
  auto metadata = std::make_shared<ZarrMetadata>();
  TENSORSTORE_RETURN_IF_ERROR(ParseMetadata(raw_data, metadata.get()));
  return metadata;
}

class MetadataCache : public internal_kvs_backed_chunk_driver::MetadataCache {
  using Base = internal_kvs_backed_chunk_driver::MetadataCache;

 public:
  using Base::Base;
  std::string GetMetadataStorageKey(absl::string_view entry_key) override {
    return internal::JoinPath(entry_key, kZarrMetadataKey);
  }

  Result<MetadataPtr> DecodeMetadata(std::string_view entry_key,
                                     absl::Cord encoded_metadata) override {
    return ParseEncodedMetadata(encoded_metadata.Flatten());
  }

  Result<absl::Cord> EncodeMetadata(absl::string_view entry_key,
                                    const void* metadata) override {
    return absl::Cord(
        ::nlohmann::json(*static_cast<const ZarrMetadata*>(metadata)).dump());
  }
};

namespace jb = tensorstore::internal::json_binding;

constexpr auto PartialMetadataBinder = [](auto is_loading, const auto& options,
                                          auto* obj, auto* j) -> Status {
  // TODO(jbms): Convert `ParsePartialMetadata` to use JSON binding framework.
  // For now this is a bit redundant.
  if constexpr (is_loading) {
    TENSORSTORE_ASSIGN_OR_RETURN(*obj, ParsePartialMetadata(*j));
    return absl::OkStatus();
  } else {
    return jb::Object(
        jb::Member("zarr_format",
                   jb::Projection(&ZarrPartialMetadata::zarr_format)),
        jb::Member("shape", jb::Projection(&ZarrPartialMetadata::shape)),
        jb::Member("chunks", jb::Projection(&ZarrPartialMetadata::chunks)),
        jb::Member("compressor",
                   jb::Projection(&ZarrPartialMetadata::compressor)),
        jb::Member("order", jb::Projection(&ZarrPartialMetadata::order,
                                           jb::Optional([](auto is_loading,
                                                           const auto& options,
                                                           auto* obj, auto* j) {
                                             if constexpr (!is_loading) {
                                               *j = EncodeOrder(*obj);
                                             }
                                             return absl::OkStatus();
                                           }))),
        jb::Member("dtype", jb::Projection(&ZarrPartialMetadata::dtype,
                                           jb::Optional([](auto is_loading,
                                                           const auto& options,
                                                           auto* obj, auto* j) {
                                             if constexpr (!is_loading) {
                                               *j = ::nlohmann::json(*obj);
                                             }
                                             return absl::OkStatus();
                                           }))),
        jb::Member("filters", jb::Constant([] { return nullptr; })),
        jb::Member("fill_value",
                   jb::Projection(&ZarrPartialMetadata::fill_value)))(
        is_loading, options, obj, j);
  }
};

class ZarrDriver
    : public internal_kvs_backed_chunk_driver::RegisteredKvsDriver<ZarrDriver> {
  using Base =
      internal_kvs_backed_chunk_driver::RegisteredKvsDriver<ZarrDriver>;

 public:
  using Base::Base;

  class OpenState;

  constexpr static char id[] = "zarr";

  template <template <typename> class MaybeBound = internal::ContextUnbound>
  struct SpecT : public internal_kvs_backed_chunk_driver::SpecT<MaybeBound> {
    std::string key_prefix;
    ChunkKeyEncoding key_encoding;
    std::optional<ZarrPartialMetadata> partial_metadata;
    SelectedField selected_field;

    constexpr static auto ApplyMembers = [](auto& x, auto f) {
      return f(internal::BaseCast<
                   internal_kvs_backed_chunk_driver::SpecT<MaybeBound>>(x),
               x.key_prefix, x.key_encoding, x.partial_metadata,
               x.selected_field);
    };
  };

  static inline const auto json_binder = jb::Sequence(
      internal_kvs_backed_chunk_driver::SpecJsonBinder,
      jb::Member("path", jb::Projection(&SpecT<>::key_prefix,
                                        jb::DefaultValue([](auto* obj) {
                                          *obj = std::string{};
                                        }))),
      jb::Member(
          "key_encoding",
          jb::Projection(
              &SpecT<>::key_encoding,
              jb::DefaultValue(
                  [](auto* obj) { *obj = ChunkKeyEncoding::kDotSeparated; },
                  [](auto is_loading, const auto& options, auto* obj, auto* j) {
                    if constexpr (is_loading) {
                      TENSORSTORE_ASSIGN_OR_RETURN(*obj, ParseKeyEncoding(*j));
                    } else {
                      *j = ::nlohmann::json(*obj);
                    }
                    return absl::OkStatus();
                  }))),
      jb::Member("metadata",
                 jb::Projection(&SpecT<>::partial_metadata,
                                jb::Optional(PartialMetadataBinder))),
      jb::Member("field", jb::Projection(
                              &SpecT<>::selected_field,
                              jb::DefaultValue</*DisallowIncludeDefault=*/true>(
                                  [](auto* obj) { *obj = std::string{}; }))));

  static Status ConvertSpec(SpecT<>* spec, const SpecRequestOptions& options) {
    if (options.minimal_spec()) {
      spec->partial_metadata = std::nullopt;
    }
    return Base::ConvertSpec(spec, options);
  }
};

class DataCache : public internal_kvs_backed_chunk_driver::DataCache {
  using Base = internal_kvs_backed_chunk_driver::DataCache;

 public:
  explicit DataCache(Initializer initializer, std::string key_prefix,
                     ChunkKeyEncoding key_encoding)
      : Base(initializer,
             GetChunkGridSpecification(*static_cast<const ZarrMetadata*>(
                 initializer.metadata.get()))),
        key_prefix_(std::move(key_prefix)),
        key_encoding_(key_encoding) {}

  Status ValidateMetadataCompatibility(const void* existing_metadata_ptr,
                                       const void* new_metadata_ptr) override {
    assert(existing_metadata_ptr);
    assert(new_metadata_ptr);
    const auto& existing_metadata =
        *static_cast<const ZarrMetadata*>(existing_metadata_ptr);
    const auto& new_metadata =
        *static_cast<const ZarrMetadata*>(new_metadata_ptr);
    if (IsMetadataCompatible(existing_metadata, new_metadata)) {
      return absl::OkStatus();
    }
    return absl::FailedPreconditionError(
        StrCat("Updated zarr metadata ", ::nlohmann::json(new_metadata).dump(),
               " is incompatible with existing metadata ",
               ::nlohmann::json(existing_metadata).dump()));
  }

  void GetChunkGridBounds(
      const void* metadata_ptr, MutableBoxView<> bounds,
      BitSpan<std::uint64_t> implicit_lower_bounds,
      BitSpan<std::uint64_t> implicit_upper_bounds) override {
    const auto& metadata = *static_cast<const ZarrMetadata*>(metadata_ptr);
    assert(bounds.rank() == static_cast<DimensionIndex>(metadata.shape.size()));
    assert(bounds.rank() == implicit_lower_bounds.size());
    assert(bounds.rank() == implicit_upper_bounds.size());
    std::fill(bounds.origin().begin(), bounds.origin().end(), Index(0));
    std::copy(metadata.shape.begin(), metadata.shape.end(),
              bounds.shape().begin());
    implicit_lower_bounds.fill(false);
    implicit_upper_bounds.fill(true);
  }

  Result<std::shared_ptr<const void>> GetResizedMetadata(
      const void* existing_metadata, span<const Index> new_inclusive_min,
      span<const Index> new_exclusive_max) override {
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

  /// Returns the ChunkCache grid to use for the given metadata.
  static internal::ChunkGridSpecification GetChunkGridSpecification(
      const ZarrMetadata& metadata) {
    internal::ChunkGridSpecification::Components components;
    components.reserve(metadata.dtype.fields.size());
    std::vector<DimensionIndex> chunked_to_cell_dimensions(
        metadata.chunks.size());
    std::iota(chunked_to_cell_dimensions.begin(),
              chunked_to_cell_dimensions.end(), static_cast<DimensionIndex>(0));
    for (std::size_t field_i = 0; field_i < metadata.dtype.fields.size();
         ++field_i) {
      const auto& field = metadata.dtype.fields[field_i];
      const auto& field_layout = metadata.chunk_layout.fields[field_i];
      auto fill_value = metadata.fill_values[field_i];
      if (!fill_value.valid()) {
        // Use value-initialized rank-0 fill value.
        fill_value = AllocateArray(span<const Index, 0>{}, c_order, value_init,
                                   field.data_type);
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
    }
    return internal::ChunkGridSpecification{std::move(components)};
  }

  Result<absl::InlinedVector<SharedArrayView<const void>, 1>> DecodeChunk(
      const void* metadata, span<const Index> chunk_indices,
      absl::Cord data) override {
    return internal_zarr::DecodeChunk(
        *static_cast<const ZarrMetadata*>(metadata), std::move(data));
  }

  Result<absl::Cord> EncodeChunk(
      const void* metadata, span<const Index> chunk_indices,
      span<const ArrayView<const void>> component_arrays) override {
    return internal_zarr::EncodeChunk(
        *static_cast<const ZarrMetadata*>(metadata), component_arrays);
  }

  std::string GetChunkStorageKey(const void* metadata,
                                 span<const Index> cell_indices) override {
    return internal::JoinPath(key_prefix_,
                              EncodeChunkIndices(cell_indices, key_encoding_));
  }

  Status GetBoundSpecData(
      internal_kvs_backed_chunk_driver::SpecT<internal::ContextBound>*
          spec_base,
      const void* metadata_ptr, std::size_t component_index) override {
    auto& spec =
        static_cast<ZarrDriver::SpecT<internal::ContextBound>&>(*spec_base);
    const auto& metadata = *static_cast<const ZarrMetadata*>(metadata_ptr);
    spec.key_prefix = key_prefix_;
    spec.selected_field = EncodeSelectedField(component_index, metadata.dtype);
    spec.key_encoding = key_encoding_;
    auto& pm = spec.partial_metadata.emplace();
    pm.zarr_format = metadata.zarr_format;
    pm.shape = metadata.shape;
    pm.chunks = metadata.chunks;
    pm.compressor = metadata.compressor;
    pm.order = metadata.order;
    pm.dtype = metadata.dtype;
    pm.fill_value = EncodeFillValue(metadata.dtype, metadata.fill_values);
    return absl::OkStatus();
  }

 private:
  std::string key_prefix_;
  ChunkKeyEncoding key_encoding_;
};

class ZarrDriver::OpenState : public ZarrDriver::OpenStateBase {
 public:
  using ZarrDriver::OpenStateBase::OpenStateBase;

  std::string GetPrefixForDeleteExisting() override {
    return spec().key_prefix.empty() ? std::string()
                                     : StrCat(spec().key_prefix, "/");
  }

  std::string GetMetadataCacheEntryKey() override { return spec().key_prefix; }

  std::unique_ptr<internal_kvs_backed_chunk_driver::MetadataCache>
  GetMetadataCache(MetadataCache::Initializer initializer) override {
    return std::make_unique<MetadataCache>(std::move(initializer));
  }

  Result<std::shared_ptr<const void>> Create(
      const void* existing_metadata) override {
    if (existing_metadata) {
      return absl::AlreadyExistsError("");
    }
    if (!spec().partial_metadata) {
      return Status(absl::StatusCode::kInvalidArgument,
                    "Cannot create array without specifying \"metadata\"");
    }
    if (auto result = internal_zarr::GetNewMetadata(*spec().partial_metadata,
                                                    spec().data_type)) {
      return result;
    } else {
      return tensorstore::MaybeAnnotateStatus(
          result.status(), "Cannot create array from specified \"metadata\"");
    }
  }

  std::string GetDataCacheKey(const void* metadata) override {
    std::string result;
    const auto& spec = this->spec();
    internal::EncodeCacheKey(&result, spec.key_prefix, spec.key_encoding,
                             *static_cast<const ZarrMetadata*>(metadata));
    return result;
  }

  std::unique_ptr<internal_kvs_backed_chunk_driver::DataCache> GetDataCache(
      DataCache::Initializer initializer) override {
    return std::make_unique<DataCache>(std::move(initializer),
                                       spec().key_prefix, spec().key_encoding);
  }

  Result<std::size_t> GetComponentIndex(const void* metadata_ptr,
                                        OpenMode open_mode) override {
    const auto& metadata = *static_cast<const ZarrMetadata*>(metadata_ptr);
    if (!(open_mode & OpenMode::allow_option_mismatch) &&
        spec().partial_metadata) {
      TENSORSTORE_RETURN_IF_ERROR(
          ValidateMetadata(metadata, *spec().partial_metadata));
    }
    return GetCompatibleField(metadata.dtype, spec().data_type,
                              spec().selected_field);
  }
};

const internal::DriverRegistration<ZarrDriver> registration;

}  // namespace

std::string EncodeChunkIndices(span<const Index> indices,
                               ChunkKeyEncoding key_encoding) {
  const char separator = GetChunkKeyEncodingSeparator(key_encoding);
  std::string key;
  for (DimensionIndex i = 0; i < indices.size(); ++i) {
    if (i != 0) {
      StrAppend(&key, separator, indices[i]);
    } else {
      StrAppend(&key, indices[i]);
    }
  }
  return key;
}

}  // namespace internal_zarr
}  // namespace tensorstore
