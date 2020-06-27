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
#include "tensorstore/driver/n5/metadata.h"
#include "tensorstore/driver/registry.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/chunk_cache.h"
#include "tensorstore/internal/json.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/kvstore/key_value_store.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/util/constant_vector.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace internal_n5 {

namespace {

namespace jb = tensorstore::internal::json_binding;

constexpr const char kMetadataKey[] = "attributes.json";

template <template <typename> class MaybeBound = internal::ContextUnbound>
struct SpecT : public internal_kvs_backed_chunk_driver::SpecT<MaybeBound> {
  std::string key_prefix;
  N5MetadataConstraints metadata_constraints;

  constexpr static auto ApplyMembers = [](auto& x, auto f) {
    return f(
        internal::BaseCast<internal_kvs_backed_chunk_driver::SpecT<MaybeBound>>(
            x),
        x.key_prefix, x.metadata_constraints);
  };
};

Result<std::shared_ptr<const N5Metadata>> ParseEncodedMetadata(
    absl::string_view encoded_value) {
  nlohmann::json raw_data = nlohmann::json::parse(encoded_value, nullptr,
                                                  /*allow_exceptions=*/false);
  if (raw_data.is_discarded()) {
    return absl::FailedPreconditionError("Invalid JSON");
  }
  TENSORSTORE_ASSIGN_OR_RETURN(auto metadata, N5Metadata::Parse(raw_data));
  return std::make_shared<N5Metadata>(std::move(metadata));
}

class MetadataCache : public internal_kvs_backed_chunk_driver::MetadataCache {
  using Base = internal_kvs_backed_chunk_driver::MetadataCache;

 public:
  using Base::Base;

  // Metadata is stored as JSON under the `attributes.json` key.
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

  Status ValidateMetadataCompatibility(const void* existing_metadata_ptr,
                                       const void* new_metadata_ptr) override {
    const auto& existing_metadata =
        *static_cast<const N5Metadata*>(existing_metadata_ptr);
    const auto& new_metadata =
        *static_cast<const N5Metadata*>(new_metadata_ptr);
    auto existing_key = existing_metadata.GetCompatibilityKey();
    auto new_key = new_metadata.GetCompatibilityKey();
    if (existing_key == new_key) return absl::OkStatus();
    return absl::FailedPreconditionError(
        StrCat("Updated N5 metadata ", new_key,
               " is incompatible with existing metadata ", existing_key));
  }

  void GetChunkGridBounds(
      const void* metadata_ptr, MutableBoxView<> bounds,
      BitSpan<std::uint64_t> implicit_lower_bounds,
      BitSpan<std::uint64_t> implicit_upper_bounds) override {
    const auto& metadata = *static_cast<const N5Metadata*>(metadata_ptr);
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
    new_metadata->attributes["dimensions"] = new_metadata->shape;
    return new_metadata;
  }

  static internal::ChunkGridSpecification GetChunkGridSpecification(
      const N5Metadata& metadata) {
    SharedArray<const void> fill_value(
        internal::AllocateAndConstructSharedElements(1, value_init,
                                                     metadata.data_type),
        StridedLayout<>(metadata.chunk_layout.shape(),
                        GetConstantVector<Index, 0>(metadata.rank())));
    return internal::ChunkGridSpecification(
        {internal::ChunkGridSpecification::Component(std::move(fill_value),
                                                     // Since all dimensions are
                                                     // resizable, just specify
                                                     // unbounded
                                                     // `component_bounds`.
                                                     Box<>(metadata.rank()))});
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
      span<const ArrayView<const void>> component_arrays) override {
    assert(component_arrays.size() == 1);
    return internal_n5::EncodeChunk(chunk_indices,
                                    *static_cast<const N5Metadata*>(metadata),
                                    component_arrays[0]);
  }

  std::string GetChunkStorageKey(const void* metadata,
                                 span<const Index> cell_indices) override {
    std::string key = key_prefix_;
    for (const Index x : cell_indices) {
      absl::StrAppend(&key, key.empty() ? "" : "/", x);
    }
    return key;
  }

  Status GetBoundSpecData(
      internal_kvs_backed_chunk_driver::SpecT<internal::ContextBound>*
          spec_base,
      const void* metadata_ptr, std::size_t component_index) override {
    assert(component_index == 0);
    auto& spec = static_cast<SpecT<internal::ContextBound>&>(*spec_base);
    spec.key_prefix = key_prefix_;
    const auto& metadata = *static_cast<const N5Metadata*>(metadata_ptr);
    auto& constraints = spec.metadata_constraints;
    constraints.shape = metadata.shape;
    constraints.data_type = metadata.data_type;
    constraints.compressor = metadata.compressor;
    constraints.attributes = metadata.attributes;
    constraints.chunk_shape =
        std::vector<Index>(metadata.chunk_layout.shape().begin(),
                           metadata.chunk_layout.shape().end());
    return absl::OkStatus();
  }

  std::string key_prefix_;
};

class N5Driver
    : public internal_kvs_backed_chunk_driver::RegisteredKvsDriver<N5Driver> {
  using Base = internal_kvs_backed_chunk_driver::RegisteredKvsDriver<N5Driver>;

 public:
  using Base::Base;

  constexpr static char id[] = "n5";

  template <template <typename> class MaybeBound = internal::ContextUnbound>
  using SpecT = internal_n5::SpecT<MaybeBound>;

  static inline const auto json_binder = jb::Sequence(
      jb::Validate(
          [](const auto& options, auto* obj) {
            if (obj->data_type.valid()) {
              return ValidateDataType(obj->data_type);
            }
            return absl::OkStatus();
          },
          internal_kvs_backed_chunk_driver::SpecJsonBinder),
      jb::Member("path", jb::Projection(&SpecT<>::key_prefix,
                                        jb::DefaultValue([](auto* obj) {
                                          *obj = std::string{};
                                        }))),
      jb::Member(
          "metadata",
          jb::Validate(
              [](const auto& options, auto* obj) {
                if (obj->data_type.valid()) {
                  if (!tensorstore::IsPossiblySameDataType(
                          obj->data_type,
                          obj->metadata_constraints.data_type)) {
                    return absl::InvalidArgumentError(StrCat(
                        "Mismatch between data type in TensorStore Spec (",
                        obj->data_type, ") and \"metadata\" (",
                        obj->metadata_constraints.data_type, ")"));
                  }
                  obj->metadata_constraints.data_type = obj->data_type;
                  obj->metadata_constraints.attributes.emplace(
                      "dataType", obj->data_type.name());
                }
                return absl::OkStatus();
              },
              jb::Projection(&SpecT<>::metadata_constraints,
                             jb::DefaultValue([](auto* obj) {
                               *obj = N5MetadataConstraints{};
                             })))));

  class OpenState;

  static Status ConvertSpec(SpecT<>* spec, const SpecRequestOptions& options) {
    if (options.minimal_spec()) {
      spec->metadata_constraints = N5MetadataConstraints{};
    }
    return Base::ConvertSpec(spec, options);
  }
};

class N5Driver::OpenState : public N5Driver::OpenStateBase {
 public:
  using N5Driver::OpenStateBase::OpenStateBase;

  std::string GetPrefixForDeleteExisting() override {
    return spec().key_prefix.empty() ? std::string()
                                     : StrCat(spec().key_prefix, "/");
  }

  std::string GetMetadataCacheEntryKey() override { return spec().key_prefix; }

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
        &result, spec().key_prefix,
        static_cast<const N5Metadata*>(metadata)->GetCompatibilityKey());
    return result;
  }

  Result<std::shared_ptr<const void>> Create(
      const void* existing_metadata) override {
    if (existing_metadata) {
      return absl::AlreadyExistsError("");
    }
    if (auto result =
            internal_n5::GetNewMetadata(spec().metadata_constraints)) {
      return result;
    } else {
      return tensorstore::MaybeAnnotateStatus(
          result.status(), "Cannot create array from specified \"metadata\"");
    }
  }

  std::unique_ptr<internal_kvs_backed_chunk_driver::DataCache> GetDataCache(
      DataCache::Initializer initializer) override {
    return std::make_unique<DataCache>(std::move(initializer),
                                       spec().key_prefix);
  }

  Result<std::size_t> GetComponentIndex(const void* metadata_ptr,
                                        OpenMode open_mode) override {
    const auto& metadata = *static_cast<const N5Metadata*>(metadata_ptr);
    // Check for compatibility
    if (spec().data_type.valid() && spec().data_type != metadata.data_type) {
      return absl::InvalidArgumentError(
          StrCat("Expected data type of ", spec().data_type,
                 " but received: ", metadata.data_type));
    }
    if (!(open_mode & OpenMode::allow_option_mismatch)) {
      TENSORSTORE_RETURN_IF_ERROR(
          ValidateMetadata(metadata, spec().metadata_constraints));
    }
    return 0;
  }
};

const internal::DriverRegistration<N5Driver> registration;

}  // namespace
}  // namespace internal_n5
}  // namespace tensorstore
