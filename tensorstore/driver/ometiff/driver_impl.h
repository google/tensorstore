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

#ifndef TENSORSTORE_DRIVER_OMETIFF_DRIVER_IMPL_H_
#define TENSORSTORE_DRIVER_OMETIFF_DRIVER_IMPL_H_

#include <string>
#include <string_view>

#include "tensorstore/driver/kvs_backed_chunk_driver.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/cache/chunk_cache.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/kvstore/ometiff/ometiff_spec.h"
#include "tensorstore/serialization/fwd.h"
#include "tensorstore/serialization/json_bindable.h"
#include "tensorstore/util/garbage_collection/fwd.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_ometiff {

class MetadataCache : public internal_kvs_backed_chunk_driver::MetadataCache {
  using Base = internal_kvs_backed_chunk_driver::MetadataCache;

 public:
  using Base::Base;
  std::string GetMetadataStorageKey(std::string_view entry_key) override;

  Result<MetadataPtr> DecodeMetadata(std::string_view entry_key,
                                     absl::Cord encoded_metadata) override;

  Result<absl::Cord> EncodeMetadata(std::string_view entry_key,
                                    const void* metadata) override;

  class Entry : public Base::Entry {
   public:
    using OwningCache = MetadataCache;
  };
};

class OMETiffDriverSpec
    : public internal::RegisteredDriverSpec<
          OMETiffDriverSpec,
          /*Parent=*/internal_kvs_backed_chunk_driver::KvsDriverSpec> {
 public:
  using Base = internal::RegisteredDriverSpec<
      OMETiffDriverSpec,
      /*Parent=*/internal_kvs_backed_chunk_driver::KvsDriverSpec>;
  constexpr static char id[] = "ometiff";

  ometiff::OMETiffImageInfo metadata;
  constexpr static auto ApplyMembers = [](auto& x, auto f) {
    return f(internal::BaseCast<KvsDriverSpec>(x), x.metadata);
  };

  TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(OMETiffDriverSpec,
                                          JsonSerializationOptions,
                                          JsonSerializationOptions,
                                          ::nlohmann::json::object_t)
  Future<internal::Driver::Handle> Open(
      internal::OpenTransactionPtr transaction,
      ReadWriteMode read_write_mode) const override;
};

class DataCache : public internal_kvs_backed_chunk_driver::DataCache {
  using Base = internal_kvs_backed_chunk_driver::DataCache;

 public:
  explicit DataCache(Initializer&& initializer, std::string key);

  const ometiff::OMETiffImageInfo& metadata() {
    return *static_cast<const ometiff::OMETiffImageInfo*>(
        initial_metadata().get());
  }

  std::string GetChunkStorageKey(span<const Index> cell_indices) override {
    return key_;
  }

  OptionalByteRangeRequest GetChunkByteRange(
      span<const Index> cell_indices) override;

  absl::Status ValidateMetadataCompatibility(
      const void* existing_metadata_ptr, const void* new_metadata_ptr) override;

  Result<std::shared_ptr<const void>> GetResizedMetadata(
      const void* existing_metadata, span<const Index> new_inclusive_min,
      span<const Index> new_exclusive_max) override;

  void GetChunkGridBounds(const void* metadata_ptr, MutableBoxView<> bounds,
                          DimensionSet& implicit_lower_bounds,
                          DimensionSet& implicit_upper_bounds) override;

  absl::Status GetBoundSpecData(
      internal_kvs_backed_chunk_driver::KvsDriverSpec& spec_base,
      const void* metadata_ptr, std::size_t component_index) override;

  /// Returns the ChunkCache grid to use for the given metadata.
  static internal::ChunkGridSpecification GetChunkGridSpecification(
      const ometiff::OMETiffImageInfo& metadata);

  Result<absl::InlinedVector<SharedArray<const void>, 1>> DecodeChunk(
      span<const Index> chunk_indices, absl::Cord data) override;

  Result<absl::Cord> EncodeChunk(
      span<const Index> chunk_indices,
      span<const SharedArrayView<const void>> component_arrays) override;

  Result<ChunkLayout> GetChunkLayoutFromMetadata(
      const void* metadata_ptr, size_t component_index) override;

  std::string GetBaseKvstorePath() override { return key_; }

  std::string key_;
};

class OMETiffDriver;
using OMETiffDriverBase = internal_kvs_backed_chunk_driver::RegisteredKvsDriver<
    OMETiffDriver, OMETiffDriverSpec, DataCache,
    internal::ChunkCacheReadWriteDriverMixin<
        OMETiffDriver, internal_kvs_backed_chunk_driver::KvsChunkedDriverBase>>;

class OMETiffDriver : public OMETiffDriverBase {
  using Base = OMETiffDriverBase;

 public:
  using Base::Base;

  class OpenState;

  const ometiff::OMETiffImageInfo& metadata() const {
    return *static_cast<const ometiff::OMETiffImageInfo*>(
        this->cache()->initial_metadata().get());
  }
};

}  // namespace internal_ometiff
}  // namespace tensorstore

TENSORSTORE_DECLARE_GARBAGE_COLLECTION_SPECIALIZATION(
    tensorstore::internal_ometiff::OMETiffDriver)

#endif  // TENSORSTORE_DRIVER_OMETIFF_DRIVER_IMPL_H_
