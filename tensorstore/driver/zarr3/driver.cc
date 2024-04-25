// Copyright 2023 The TensorStore Authors
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

#include <algorithm>
#include <cassert>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/cord.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include <nlohmann/json.hpp>
#include "tensorstore/array.h"
#include "tensorstore/array_storage_statistics.h"
#include "tensorstore/box.h"
#include "tensorstore/chunk_layout.h"
#include "tensorstore/codec_spec.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/chunk.h"
#include "tensorstore/driver/driver_spec.h"
#include "tensorstore/driver/kvs_backed_chunk_driver.h"
#include "tensorstore/driver/registry.h"
#include "tensorstore/driver/zarr3/chunk_cache.h"
#include "tensorstore/driver/zarr3/metadata.h"
#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/index_space/dimension_units.h"
#include "tensorstore/index_space/index_domain.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/internal/cache/cache.h"
#include "tensorstore/internal/cache_key/cache_key.h"
#include "tensorstore/internal/chunk_grid_specification.h"
#include "tensorstore/internal/grid_chunk_key_ranges_base10.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/lexicographical_grid_index_key.h"
#include "tensorstore/internal/storage_statistics.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/open_options.h"
#include "tensorstore/rank.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/dimension_set.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/garbage_collection/fwd.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_zarr3 {

// Avoid anonymous namespace to workaround MSVC bug.
//
// https://developercommunity.visualstudio.com/t/Bug-involving-virtual-functions-templat/10424129
#ifndef _MSC_VER
namespace {
#endif

namespace jb = tensorstore::internal_json_binding;

using ::tensorstore::internal_kvs_backed_chunk_driver::KvsDriverSpec;

constexpr const char kMetadataKey[] = "zarr.json";

class ZarrDriverSpec
    : public internal::RegisteredDriverSpec<ZarrDriverSpec,
                                            /*Parent=*/KvsDriverSpec> {
 public:
  constexpr static char id[] = "zarr3";

  using Base = internal::RegisteredDriverSpec<ZarrDriverSpec,
                                              /*Parent=*/KvsDriverSpec>;

  ZarrMetadataConstraints metadata_constraints;

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
                    obj->metadata_constraints.data_type.value_or(DataType())));
                TENSORSTORE_RETURN_IF_ERROR(obj->schema.Set(
                    RankConstraint{obj->metadata_constraints.rank}));
                return absl::OkStatus();
              },
              jb::Projection<&ZarrDriverSpec::metadata_constraints>(
                  jb::DefaultInitializedValue()))));

  absl::Status ApplyOptions(SpecOptions&& options) override {
    if (options.minimal_spec) {
      metadata_constraints = ZarrMetadataConstraints{};
    }
    return Base::ApplyOptions(std::move(options));
  }

  Result<IndexDomain<>> GetDomain() const override {
    return GetEffectiveDomain(metadata_constraints, schema);
  }

  Result<SharedArray<const void>> GetFillValue(
      IndexTransformView<> transform) const override {
    SharedArray<const void> fill_value{schema.fill_value()};

    const auto& metadata = metadata_constraints;
    if (metadata.fill_value) {
      fill_value = *metadata.fill_value;
    }

    return fill_value;
  }

  Result<DimensionUnitsVector> GetDimensionUnits() const override {
    return GetEffectiveDimensionUnits(metadata_constraints.rank,
                                      metadata_constraints.dimension_units,
                                      schema.dimension_units());
  }

  Result<CodecSpec> GetCodec() const override {
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto codec, GetEffectiveCodec(metadata_constraints, schema));
    return CodecSpec(std::move(codec));
  }

  Result<ChunkLayout> GetChunkLayout() const override {
    return GetEffectiveChunkLayout(metadata_constraints, schema);
  }

  Future<internal::Driver::Handle> Open(
      internal::DriverOpenRequest request) const override;
};

Result<std::shared_ptr<const ZarrMetadata>> ParseEncodedMetadata(
    std::string_view encoded_value) {
  nlohmann::json raw_data = nlohmann::json::parse(encoded_value, nullptr,
                                                  /*allow_exceptions=*/false);
  if (raw_data.is_discarded()) {
    return absl::DataLossError("Invalid JSON");
  }
  TENSORSTORE_ASSIGN_OR_RETURN(auto metadata,
                               ZarrMetadata::FromJson(std::move(raw_data)));
  return std::make_shared<ZarrMetadata>(std::move(metadata));
}

class MetadataCache : public internal_kvs_backed_chunk_driver::MetadataCache {
  using Base = internal_kvs_backed_chunk_driver::MetadataCache;

 public:
  using Base::Base;

  // Metadata is stored as JSON under the `zarr.json` key.
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
        ::nlohmann::json(*static_cast<const ZarrMetadata*>(metadata)).dump());
  }
};

class DataCacheBase
    : public internal_kvs_backed_chunk_driver::ChunkedDataCacheBase,
      public internal::LexicographicalGridIndexKeyParser {
  using Base = internal_kvs_backed_chunk_driver::ChunkedDataCacheBase;

 public:
  explicit DataCacheBase(Initializer&& initializer, std::string key_prefix)
      : Base(std::move(initializer)), key_prefix_(std::move(key_prefix)) {}

  const ZarrMetadata& metadata() const {
    return *static_cast<const ZarrMetadata*>(initial_metadata().get());
  }

  virtual ZarrChunkCache& zarr_chunk_cache() = 0;

  absl::Status ValidateMetadataCompatibility(
      const void* existing_metadata_ptr,
      const void* new_metadata_ptr) override {
    const auto& existing_metadata =
        *static_cast<const ZarrMetadata*>(existing_metadata_ptr);
    const auto& new_metadata =
        *static_cast<const ZarrMetadata*>(new_metadata_ptr);
    auto existing_key = existing_metadata.GetCompatibilityKey();
    auto new_key = new_metadata.GetCompatibilityKey();
    if (existing_key == new_key) return absl::OkStatus();
    return absl::FailedPreconditionError(tensorstore::StrCat(
        "Updated zarr metadata ", new_key,
        " is incompatible with existing metadata ", existing_key));
  }

  void GetChunkGridBounds(const void* metadata_ptr, MutableBoxView<> bounds,
                          DimensionSet& implicit_lower_bounds,
                          DimensionSet& implicit_upper_bounds) override {
    const auto& metadata = *static_cast<const ZarrMetadata*>(metadata_ptr);
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

  static internal::ChunkGridSpecification GetChunkGridSpecification(
      const ZarrMetadata& metadata) {
    auto fill_value =
        BroadcastArray(metadata.fill_value, metadata.chunk_shape).value();
    internal::ChunkGridSpecification::ComponentList components;
    auto& component =
        components.emplace_back(std::move(fill_value),
                                // Since all dimensions are resizable, just
                                // specify unbounded `component_bounds`.
                                Box<>(metadata.rank));
    component.fill_value_comparison_kind = EqualityComparisonKind::identical;
    return internal::ChunkGridSpecification(std::move(components));
  }

  std::string FormatKey(span<const Index> grid_indices) const final {
    std::string key;
    const auto& metadata =
        *static_cast<const ZarrMetadata*>(initial_metadata().get());
    const DimensionIndex rank = metadata.rank;
    char separator = metadata.chunk_key_encoding.separator;
    if (metadata.chunk_key_encoding.kind == ChunkKeyEncoding::kDefault) {
      key = tensorstore::StrCat(
          key_prefix_, "c",
          rank != 0 ? std::string_view(&separator, 1) : std::string_view());
    } else {
      if (rank == 0) {
        return tensorstore::StrCat(key_prefix_, "0");
      }
      key = key_prefix_;
    }
    internal::FormatGridIndexKeyWithDimensionSeparator(
        key, separator,
        [](std::string& out, DimensionIndex dim, Index grid_index) {
          absl::StrAppend(&out, grid_index);
        },
        rank, grid_indices);
    return key;
  }

  bool ParseKey(std::string_view key, span<Index> grid_indices) const final {
    const auto& metadata =
        *static_cast<const ZarrMetadata*>(initial_metadata().get());
    assert(key.size() > key_prefix_.size());
    assert(!grid_indices.empty());
    key.remove_prefix(
        key_prefix_.size() +
        (metadata.chunk_key_encoding.kind == ChunkKeyEncoding::kDefault ? 2
                                                                        : 0));
    return internal::ParseGridIndexKeyWithDimensionSeparator(
        metadata.chunk_key_encoding.separator,
        [](std::string_view part, DimensionIndex dim, Index& grid_index) {
          if (part.empty() || !absl::ascii_isdigit(part.front()) ||
              !absl::ascii_isdigit(part.back()) ||
              !absl::SimpleAtoi(part, &grid_index)) {
            return false;
          }
          return true;
        },
        key, grid_indices);
  }

  Index MinGridIndexForLexicographicalOrder(
      DimensionIndex dim, IndexInterval grid_interval) const final {
    return internal::MinValueWithMaxBase10Digits(grid_interval.exclusive_max());
  }

  std::string GetChunkStorageKey(span<const Index> cell_indices) override {
    const auto& metadata =
        *static_cast<const ZarrMetadata*>(initial_metadata().get());
    if (metadata.chunk_key_encoding.kind == ChunkKeyEncoding::kDefault) {
      std::string key = tensorstore::StrCat(key_prefix_, "c");
      for (DimensionIndex i = 0; i < cell_indices.size(); ++i) {
        tensorstore::StrAppend(
            &key, std::string_view(&metadata.chunk_key_encoding.separator, 1),
            cell_indices[i]);
      }
      return key;
    }
    // Use "0" for rank 0 as a special case.
    std::string key = tensorstore::StrCat(
        key_prefix_, cell_indices.empty() ? 0 : cell_indices[0]);
    for (DimensionIndex i = 1; i < cell_indices.size(); ++i) {
      tensorstore::StrAppend(
          &key, std::string_view(&metadata.chunk_key_encoding.separator, 1),
          cell_indices[i]);
    }
    return key;
  }

  Result<IndexTransform<>> GetExternalToInternalTransform(
      const void* metadata_ptr, size_t component_index) override {
    assert(component_index == 0);
    const auto& metadata = *static_cast<const ZarrMetadata*>(metadata_ptr);
    const DimensionIndex rank = metadata.rank;
    std::string_view normalized_dimension_names[kMaxRank];
    for (DimensionIndex i = 0; i < rank; ++i) {
      if (const auto& name = metadata.dimension_names[i]; name.has_value()) {
        normalized_dimension_names[i] = *name;
      }
    }
    auto builder =
        tensorstore::IndexTransformBuilder<>(rank, rank)
            .input_shape(metadata.shape)
            .input_labels(span(&normalized_dimension_names[0], rank));
    builder.implicit_upper_bounds(true);
    for (DimensionIndex i = 0; i < rank; ++i) {
      builder.output_single_input_dimension(i, i);
    }
    return builder.Finalize();
  }

  absl::Status GetBoundSpecData(KvsDriverSpec& spec_base,
                                const void* metadata_ptr,
                                size_t component_index) override {
    assert(component_index == 0);
    auto& spec = static_cast<ZarrDriverSpec&>(spec_base);
    const auto& metadata = *static_cast<const ZarrMetadata*>(metadata_ptr);
    spec.metadata_constraints = ZarrMetadataConstraints(metadata);
    return absl::OkStatus();
  }

  Result<ChunkLayout> GetChunkLayoutFromMetadata(
      const void* metadata_ptr, size_t component_index) override {
    const auto& metadata = *static_cast<const ZarrMetadata*>(metadata_ptr);
    ChunkLayout chunk_layout;
    TENSORSTORE_RETURN_IF_ERROR(SetChunkLayoutFromMetadata(
        metadata.data_type, metadata.rank, metadata.chunk_shape,
        &metadata.codec_specs, chunk_layout));
    TENSORSTORE_RETURN_IF_ERROR(chunk_layout.Finalize());
    return chunk_layout;
  }

  std::string GetBaseKvstorePath() override { return key_prefix_; }

  std::string key_prefix_;
};

using internal_kvs_backed_chunk_driver::DataCacheInitializer;

template <typename ChunkCacheImpl>
class ZarrDataCache : public ChunkCacheImpl, public DataCacheBase {
 public:
  template <typename... U>
  explicit ZarrDataCache(DataCacheInitializer&& initializer,
                         std::string key_prefix, U&&... arg)
      : ChunkCacheImpl(std::move(initializer.store), std::forward<U>(arg)...),
        DataCacheBase(std::move(initializer), std::move(key_prefix)),
        grid_(DataCacheBase::GetChunkGridSpecification(metadata())) {}

  const internal::LexicographicalGridIndexKeyParser& GetChunkStorageKeyParser()
      final {
    return *this;
  }

  internal::Cache& cache() final { return *this; }

  ZarrChunkCache& zarr_chunk_cache() final { return *this; }

  const internal::ChunkGridSpecification& grid() const override {
    return grid_;
  }

  Future<const void> DeleteCell(
      span<const Index> grid_cell_indices,
      internal::OpenTransactionPtr transaction) final {
    return ChunkCacheImpl::DeleteCell(grid_cell_indices,
                                      std::move(transaction));
  }

  const Executor& executor() const override {
    return DataCacheBase::executor();
  }

  internal::ChunkGridSpecification grid_;
};

class ZarrDriver;
using ZarrDriverBase = internal_kvs_backed_chunk_driver::RegisteredKvsDriver<
    ZarrDriver, ZarrDriverSpec, DataCacheBase,
    internal_kvs_backed_chunk_driver::KvsChunkedDriverBase>;

class ZarrDriver : public ZarrDriverBase {
  using Base = ZarrDriverBase;

 public:
  using Base::Base;
  const ZarrMetadata& metadata() const {
    return *static_cast<const ZarrMetadata*>(cache()->initial_metadata().get());
  }

  Result<SharedArray<const void>> GetFillValue(
      IndexTransformView<> transform) override {
    const auto& metadata = this->metadata();
    return metadata.fill_value;
  }

  Future<ArrayStorageStatistics> GetStorageStatistics(
      GetStorageStatisticsRequest request) override;

  Result<CodecSpec> GetCodec() override {
    return GetCodecFromMetadata(metadata());
  }

  Result<DimensionUnitsVector> GetDimensionUnits() override {
    const auto& metadata = this->metadata();
    return internal_zarr3::GetDimensionUnits(metadata.rank,
                                             metadata.dimension_units);
  }

  void Read(ReadRequest request,
            AnyFlowReceiver<absl::Status, internal::ReadChunk, IndexTransform<>>
                receiver) override {
    return cache()->zarr_chunk_cache().Read(
        {std::move(request), GetCurrentDataStalenessBound()},
        std::move(receiver));
  }

  void Write(
      WriteRequest request,
      AnyFlowReceiver<absl::Status, internal::WriteChunk, IndexTransform<>>
          receiver) override {
    return cache()->zarr_chunk_cache().Write(std::move(request),
                                             std::move(receiver));
  }

  absl::Time GetCurrentDataStalenessBound() {
    absl::Time bound = this->data_staleness_bound().time;
    if (bound != absl::InfinitePast()) {
      bound = std::min(bound, absl::Now());
    }
    return bound;
  }

  class OpenState;
};

Future<ArrayStorageStatistics> ZarrDriver::GetStorageStatistics(
    GetStorageStatisticsRequest request) {
  Future<ArrayStorageStatistics> future;
  // Note: `future` is an output parameter.
  auto state = internal::MakeIntrusivePtr<
      internal::GetStorageStatisticsAsyncOperationState>(future,
                                                         request.options);
  auto* state_ptr = state.get();
  auto* cache = this->cache();
  auto transaction = request.transaction;
  LinkValue(
      WithExecutor(cache->executor(),
                   [state = std::move(state),
                    cache = internal::CachePtr<DataCacheBase>(cache),
                    transform = std::move(request.transform),
                    transaction = std::move(request.transaction),
                    staleness_bound = this->GetCurrentDataStalenessBound()](
                       Promise<ArrayStorageStatistics> promise,
                       ReadyFuture<MetadataCache::MetadataPtr> future) mutable {
                     auto* metadata =
                         static_cast<const ZarrMetadata*>(future.value().get());
                     cache->zarr_chunk_cache().GetStorageStatistics(
                         std::move(state),
                         {std::move(transaction), metadata->shape,
                          std::move(transform), staleness_bound});
                   }),
      state_ptr->promise,
      ResolveMetadata(std::move(transaction), metadata_staleness_bound_.time));
  return future;
}

class ZarrDriver::OpenState : public ZarrDriver::OpenStateBase {
 public:
  using ZarrDriver::OpenStateBase::OpenStateBase;

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
        static_cast<const ZarrMetadata*>(metadata)->GetCompatibilityKey());
    return result;
  }

  Result<std::shared_ptr<const void>> Create(const void* existing_metadata,
                                             CreateOptions options) override {
    if (existing_metadata) {
      return absl::AlreadyExistsError("");
    }
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto metadata,
        internal_zarr3::GetNewMetadata(spec().metadata_constraints,
                                       spec().schema),
        tensorstore::MaybeAnnotateStatus(
            _, "Cannot create using specified \"metadata\" and schema"));
    return metadata;
  }

  std::unique_ptr<internal_kvs_backed_chunk_driver::DataCacheBase> GetDataCache(
      DataCacheInitializer&& initializer) override {
    const auto& metadata =
        *static_cast<const ZarrMetadata*>(initializer.metadata.get());
    return internal_zarr3::MakeZarrChunkCache<DataCacheBase, ZarrDataCache>(
        *metadata.codecs, std::move(initializer), spec().store.path,
        metadata.codec_state);
  }

  Result<size_t> GetComponentIndex(const void* metadata_ptr,
                                   OpenMode open_mode) override {
    const auto& metadata = *static_cast<const ZarrMetadata*>(metadata_ptr);
    TENSORSTORE_RETURN_IF_ERROR(
        ValidateMetadata(metadata, spec().metadata_constraints));
    TENSORSTORE_RETURN_IF_ERROR(
        ValidateMetadataSchema(metadata, spec().schema));
    return 0;
  }
};

Future<internal::Driver::Handle> ZarrDriverSpec::Open(
    internal::DriverOpenRequest request) const {
  return ZarrDriver::Open(this, std::move(request));
}

#ifndef _MSC_VER
}  // namespace
#endif

}  // namespace internal_zarr3
}  // namespace tensorstore

TENSORSTORE_DECLARE_GARBAGE_COLLECTION_SPECIALIZATION(
    tensorstore::internal_zarr3::ZarrDriver)
// Use default garbage collection implementation provided by
// kvs_backed_chunk_driver (just handles the kvstore)
TENSORSTORE_DEFINE_GARBAGE_COLLECTION_SPECIALIZATION(
    tensorstore::internal_zarr3::ZarrDriver,
    tensorstore::internal_zarr3::ZarrDriver::GarbageCollectionBase)

namespace {
const tensorstore::internal::DriverRegistration<
    tensorstore::internal_zarr3::ZarrDriverSpec>
    registration;
}  // namespace
