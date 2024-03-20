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

#ifndef TENSORSTORE_DRIVER_ZARR_DRIVER_IMPL_H_
#define TENSORSTORE_DRIVER_ZARR_DRIVER_IMPL_H_

#include <string>
#include <string_view>

#include "tensorstore/driver/kvs_backed_chunk_driver.h"
#include "tensorstore/driver/zarr/metadata.h"
#include "tensorstore/driver/zarr/spec.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/cache/chunk_cache.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/util/garbage_collection/fwd.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_zarr {

/// Encodes a chunk grid index vector as a storage key suffix.
std::string EncodeChunkIndices(span<const Index> indices,
                               DimensionSeparator dimension_separator);

class MetadataCache : public internal_kvs_backed_chunk_driver::MetadataCache {
  using Base = internal_kvs_backed_chunk_driver::MetadataCache;

 public:
  using Base::Base;
  std::string GetMetadataStorageKey(std::string_view entry_key) override;

  Result<MetadataPtr> DecodeMetadata(std::string_view entry_key,
                                     absl::Cord encoded_metadata) override;

  Result<absl::Cord> EncodeMetadata(std::string_view entry_key,
                                    const void* metadata) override;
};

class ZarrDriverSpec
    : public internal::RegisteredDriverSpec<
          ZarrDriverSpec,
          /*Parent=*/internal_kvs_backed_chunk_driver::KvsDriverSpec> {
 public:
  using Base = internal::RegisteredDriverSpec<
      ZarrDriverSpec,
      /*Parent=*/internal_kvs_backed_chunk_driver::KvsDriverSpec>;
  constexpr static char id[] = "zarr";

  ZarrPartialMetadata partial_metadata;
  SelectedField selected_field;
  std::string metadata_key;

  constexpr static auto ApplyMembers = [](auto& x, auto f) {
    return f(internal::BaseCast<KvsDriverSpec>(x), x.partial_metadata,
             x.selected_field, x.metadata_key);
  };
  absl::Status ApplyOptions(SpecOptions&& options) override;

  Result<SpecRankAndFieldInfo> GetSpecInfo() const;

  TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(ZarrDriverSpec,
                                          JsonSerializationOptions,
                                          JsonSerializationOptions,
                                          ::nlohmann::json::object_t)

  Result<IndexDomain<>> GetDomain() const override;

  Result<CodecSpec> GetCodec() const override;

  Result<ChunkLayout> GetChunkLayout() const override;

  Result<SharedArray<const void>> GetFillValue(
      IndexTransformView<> transform) const override;

  Future<internal::Driver::Handle> Open(
      DriverOpenRequest request) const override;
};

class DataCache : public internal_kvs_backed_chunk_driver::DataCache {
  using Base = internal_kvs_backed_chunk_driver::DataCache;

 public:
  explicit DataCache(Initializer&& initializer, std::string key_prefix,
                     DimensionSeparator dimension_separator,
                     std::string metadata_key);

  const ZarrMetadata& metadata() {
    return *static_cast<const ZarrMetadata*>(initial_metadata().get());
  }

  absl::Status ValidateMetadataCompatibility(
      const void* existing_metadata_ptr, const void* new_metadata_ptr) override;

  void GetChunkGridBounds(const void* metadata_ptr, MutableBoxView<> bounds,
                          DimensionSet& implicit_lower_bounds,
                          DimensionSet& implicit_upper_bounds) override;

  Result<std::shared_ptr<const void>> GetResizedMetadata(
      const void* existing_metadata, span<const Index> new_inclusive_min,
      span<const Index> new_exclusive_max) override;

  /// Returns the ChunkCache grid to use for the given metadata.
  static internal::ChunkGridSpecification GetChunkGridSpecification(
      const ZarrMetadata& metadata);

  Result<absl::InlinedVector<SharedArray<const void>, 1>> DecodeChunk(
      span<const Index> chunk_indices, absl::Cord data) override;

  Result<absl::Cord> EncodeChunk(
      span<const Index> chunk_indices,
      span<const SharedArrayView<const void>> component_arrays) override;

  std::string GetChunkStorageKey(span<const Index> cell_indices) override;

  absl::Status GetBoundSpecData(
      internal_kvs_backed_chunk_driver::KvsDriverSpec& spec_base,
      const void* metadata_ptr, std::size_t component_index) override;

  Result<ChunkLayout> GetChunkLayoutFromMetadata(
      const void* metadata_ptr, size_t component_index) override;

  std::string GetBaseKvstorePath() override;

  std::string key_prefix_;
  DimensionSeparator dimension_separator_;
  std::string metadata_key_;
};

class ZarrDriver;
using ZarrDriverBase = internal_kvs_backed_chunk_driver::RegisteredKvsDriver<
    ZarrDriver, ZarrDriverSpec, DataCache,
    internal::ChunkCacheReadWriteDriverMixin<
        ZarrDriver, internal_kvs_backed_chunk_driver::KvsChunkedDriverBase>>;

class ZarrDriver : public ZarrDriverBase {
  using Base = ZarrDriverBase;

 public:
  using Base::Base;

  class OpenState;

  const ZarrMetadata& metadata() const {
    return *static_cast<const ZarrMetadata*>(
        this->cache()->initial_metadata().get());
  }

  Result<CodecSpec> GetCodec() override;

  Result<SharedArray<const void>> GetFillValue(
      IndexTransformView<> transform) override;

  Future<ArrayStorageStatistics> GetStorageStatistics(
      GetStorageStatisticsRequest request) override;
};

}  // namespace internal_zarr
}  // namespace tensorstore

TENSORSTORE_DECLARE_GARBAGE_COLLECTION_SPECIALIZATION(
    tensorstore::internal_zarr::ZarrDriver)

#endif  // TENSORSTORE_DRIVER_ZARR_DRIVER_IMPL_H_
