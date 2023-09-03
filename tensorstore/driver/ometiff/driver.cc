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

#include <string>
#include <string_view>

#include "riegeli/bytes/cord_reader.h"
#include "tensorstore/driver/ometiff/driver_impl.h"
#include "tensorstore/driver/registry.h"
#include "tensorstore/internal/cache_key/cache_key.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/kvstore/ometiff/ometiff_key_value_store.h"
#include "tensorstore/kvstore/ometiff/ometiff_spec.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/util/endian.h"

namespace tensorstore {
namespace internal_ometiff {

namespace {
namespace jb = tensorstore::internal_json_binding;
using ::tensorstore::ometiff::OMETiffImageInfo;

template <typename T>
uint32_t TIFFhowmany_32(T x, T y) {
  return (((uint32_t)x < (0xffffffff - (uint32_t)(y - 1)))
              ? ((((uint32_t)(x)) + (((uint32_t)(y)) - 1)) / ((uint32_t)(y)))
              : 0U);
}

Result<std::shared_ptr<const OMETiffImageInfo>> ParseEncodedMetadata(
    std::string_view encoded_value) {
  nlohmann::json raw_data = nlohmann::json::parse(encoded_value, nullptr,
                                                  /*allow_exceptions=*/false);
  if (raw_data.is_discarded()) {
    return absl::FailedPreconditionError("Invalid JSON");
  }
  TENSORSTORE_ASSIGN_OR_RETURN(auto metadata,
                               OMETiffImageInfo::FromJson(std::move(raw_data)));
  return std::make_shared<OMETiffImageInfo>(std::move(metadata));
}

uint32_t TIFFComputeTile(const OMETiffImageInfo& tiff, uint32_t x, uint32_t y,
                         uint32_t z, uint16_t s) {
  uint32_t dx = tiff.tile_width;
  uint32_t dy = tiff.tile_height;
  if (!tiff.is_tiled) {
    dx = tiff.width;
    dy = tiff.rows_per_strip;
  }

  uint32_t dz = 1;
  uint32_t tile = 1;

  uint32_t depth = 1;  // TODO: Generalize.
  if (depth == 1) z = 0;
  if (dx != 0 && dy != 0 && dz != 0) {
    uint32_t xpt = TIFFhowmany_32(tiff.width, dx);
    uint32_t ypt = TIFFhowmany_32(tiff.height, dy);
    uint32_t zpt = TIFFhowmany_32(depth, dz);
    tile = (xpt * ypt) * z + xpt * y + x;
  }
  return (tile);
}

}  // namespace

std::string MetadataCache::GetMetadataStorageKey(std::string_view entry_key) {
  ABSL_LOG(INFO) << "Get metadata storage key: " << entry_key;
  return std::string(entry_key);
}

Result<MetadataCache::MetadataPtr> MetadataCache::DecodeMetadata(
    std::string_view entry_key, absl::Cord encoded_metadata) {
  ABSL_LOG(INFO) << "Parsing metadata";
  return ParseEncodedMetadata(std::move(encoded_metadata.Flatten()));
}

Result<absl::Cord> MetadataCache::EncodeMetadata(std::string_view entry_key,
                                                 const void* metadata) {
  return absl::Cord(
      ::nlohmann::json(*static_cast<const OMETiffImageInfo*>(metadata)).dump());
}

Future<internal::Driver::Handle> OMETiffDriverSpec::Open(
    internal::OpenTransactionPtr transaction,
    ReadWriteMode read_write_mode) const {
  if (read_write_mode == ReadWriteMode::write) {
    return absl::InvalidArgumentError("Writing not supported");
  }
  return OMETiffDriver::Open(std::move(transaction), this, read_write_mode);
}

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(
    OMETiffDriverSpec,
    jb::Sequence(internal_kvs_backed_chunk_driver::SpecJsonBinder,
                 jb::Initialize([](auto* obj) {
                   // Base kvs chunk driver forces path. Undo.
                   internal::EnsureNonDirectoryPath(obj->store.path);
                   return absl::OkStatus();
                 })))

DataCache::DataCache(Initializer&& initializer, std::string key)
    : Base(std::move(initializer),
           GetChunkGridSpecification(*static_cast<const OMETiffImageInfo*>(
               initializer.metadata.get()))),
      key_(std::move(key)) {}

OptionalByteRangeRequest DataCache::GetChunkByteRange(
    span<const Index> cell_indices) {
  ABSL_LOG(INFO) << "Requested cell indices: " << cell_indices;

  auto& metadata = this->metadata();
  auto rank = 2;
  auto chunk_elements = metadata.rows_per_strip * metadata.width;
  auto chunk_index =
      TIFFComputeTile(metadata, cell_indices[1], cell_indices[0], 0, 0);

  // Adjust final chunk if needed.

  if (metadata.is_tiled) {
    ABSL_LOG(INFO) << "IMPLEMENT ME!!!!";
  } else {
    chunk_elements =
        std::min(metadata.height - static_cast<uint32_t>(cell_indices[0]) *
                                       metadata.rows_per_strip,
                 metadata.rows_per_strip) *
        metadata.width;
  }
  // Map to byte offset.
  int64_t start = metadata.chunk_offset + chunk_index * metadata.chunk_size;

  ABSL_LOG(INFO) << "Calculated chunk offset: " << start;

  return ByteRange{start, start + chunk_elements * metadata.dtype.size()};
}

absl::Status DataCache::ValidateMetadataCompatibility(
    const void* existing_metadata_ptr, const void* new_metadata_ptr) {
  assert(existing_metadata_ptr);
  assert(new_metadata_ptr);
  // const auto& existing_metadata =
  //     *static_cast<const OMEMetadata*>(existing_metadata_ptr);
  // const auto& new_metadata =
  //     *static_cast<const OMEMetadata*>(new_metadata_ptr);
  ABSL_LOG(INFO) << "Validate metadata compatibility";
  return absl::OkStatus();
}

Result<std::shared_ptr<const void>> DataCache::GetResizedMetadata(
    const void* existing_metadata, span<const Index> new_inclusive_min,
    span<const Index> new_exclusive_max) {
  ABSL_LOG(INFO) << "Getting resized metadata";
  auto new_metadata = std::make_shared<OMETiffImageInfo>(
      *static_cast<const OMETiffImageInfo*>(existing_metadata));
  const DimensionIndex rank = 2;  // TODO: fix me.
  assert(rank == new_inclusive_min.size());
  assert(rank == new_exclusive_max.size());
  for (DimensionIndex i = 0; i < rank; ++i) {
    assert(ExplicitIndexOr(new_inclusive_min[i], 0) == 0);
    const Index new_size = new_exclusive_max[i];
    if (new_size == kImplicit) continue;
    // new_metadata->shape[i] = new_size;
  }
  return new_metadata;
}

internal::ChunkGridSpecification DataCache::GetChunkGridSpecification(
    const OMETiffImageInfo& metadata) {
  uint32_t rank = 2;

  ABSL_LOG(INFO) << "Get chunk grid specification";

  std::vector<Index> chunk_shape(rank);
  if (metadata.is_tiled) {
    chunk_shape[1] = metadata.tile_width;
    chunk_shape[0] = metadata.tile_height;
  } else {
    chunk_shape[1] = metadata.width;
    chunk_shape[0] = metadata.rows_per_strip;
  }

  ChunkLayout chunk_layout;
  chunk_layout.Set(tensorstore::ChunkLayout::InnerOrder({0, 1}));
  chunk_layout.Set(tensorstore::ChunkLayout::ReadChunkShape(chunk_shape));
  chunk_layout.Set(RankConstraint(2));
  chunk_layout.Set(ChunkLayout::GridOrigin(GetConstantVector<Index, 0>(2)));

  IndexDomain<> domain = IndexDomain<>(rank);
  domain = WithImplicitDimensions(std::move(domain),
                                  /*implicit_lower_bounds=*/false,
                                  /*implicit_upper_bounds=*/false);

  Box<> chunk_template(rank);
  SharedArray<const void> fill_value;
  fill_value.layout().set_rank(rank);
  std::fill_n(fill_value.byte_strides().begin(), rank, 0);

  internal::ChooseReadWriteChunkGrid(chunk_layout, domain.box(),
                                     chunk_template);

  for (DimensionIndex component_dim = 0; component_dim < rank;
       ++component_dim) {
    const DimensionIndex external_dim =
        chunk_layout.inner_order()[component_dim];
    fill_value.shape()[component_dim] = chunk_template.shape()[external_dim];
  }
  fill_value.element_pointer() = internal::AllocateAndConstructSharedElements(
      1, value_init, metadata.dtype);

  ABSL_LOG(INFO) << "Chunk template: " << chunk_template;
  internal::ChunkGridSpecification::ComponentList components;
  components.emplace_back(std::move(fill_value), std::move(chunk_template));
  return components;
}

Result<absl::InlinedVector<SharedArray<const void>, 1>> DataCache::DecodeChunk(
    span<const Index> chunk_indices, absl::Cord data) {
  auto& dtype = metadata().dtype;
  std::vector<Index> chunk_shape(2);
  if (metadata().is_tiled) {
    chunk_shape[1] = metadata().tile_width;
    chunk_shape[0] = metadata().tile_height;
  } else {
    chunk_shape[1] = metadata().width;
    chunk_shape[0] = metadata().rows_per_strip;
  }

  ABSL_LOG(INFO) << "Decoding " << chunk_indices << " into shape ("
                 << chunk_shape[0] << "," << chunk_shape[1] << ")";

  auto array = AllocateArray(chunk_shape, c_order, default_init, dtype);
  ABSL_LOG(INFO) << "Expecting: " << array.num_elements() * dtype.size()
                 << ", got " << data.size();
  // assert(array.num_elements() * dtype.size() == data.size());

  auto data_flat = data.Flatten();
  memcpy(array.data(), data_flat.data(), data.size());
  absl::InlinedVector<SharedArray<const void>, 1> components;
  components.emplace_back(std::move(array));
  return components;
}

Result<absl::Cord> DataCache::EncodeChunk(
    span<const Index> chunk_indices,
    span<const SharedArrayView<const void>> component_arrays) {
  return absl::UnimplementedError("Writing is not supported for OME TIFF");
}

void DataCache::GetChunkGridBounds(const void* metadata_ptr,
                                   MutableBoxView<> bounds,
                                   DimensionSet& implicit_lower_bounds,
                                   DimensionSet& implicit_upper_bounds) {
  ABSL_LOG(INFO) << "GetChunkGridBounds";
  const auto& metadata = *static_cast<const OMETiffImageInfo*>(metadata_ptr);
  assert(bounds.rank() == static_cast<DimensionIndex>(2));
  std::vector<Index> shape{metadata.width, metadata.height};
  std::fill(bounds.origin().begin(), bounds.origin().end(), Index(0));
  std::copy(shape.begin(), shape.end(), bounds.shape().begin());
  implicit_lower_bounds = false;
  implicit_upper_bounds = false;
}

absl::Status DataCache::GetBoundSpecData(
    internal_kvs_backed_chunk_driver::KvsDriverSpec& spec_base,
    const void* metadata_ptr, std::size_t component_index) {
  return absl::OkStatus();
}

Result<ChunkLayout> DataCache::GetChunkLayoutFromMetadata(
    const void* metadata_ptr, size_t component_index) {
  ABSL_LOG(INFO) << "Getting chunk layout from metadata";
  const auto& metadata = *static_cast<const OMETiffImageInfo*>(metadata_ptr);
  uint32_t rank = 2;  // metadata.rank;

  std::vector<Index> chunk_shape(rank);
  if (metadata.is_tiled) {
    chunk_shape[0] = metadata.tile_width;
    chunk_shape[1] = metadata.tile_height;
  } else {
    chunk_shape[0] = metadata.width;
    chunk_shape[1] = metadata.rows_per_strip;
  }

  ChunkLayout chunk_layout;
  chunk_layout.Set(tensorstore::ChunkLayout::InnerOrder({1, 0}));
  chunk_layout.Set(tensorstore::ChunkLayout::ReadChunkShape(chunk_shape));

  // Move the stuff below to a seaprate function later. Maybe
  // spec.cc.
  TENSORSTORE_RETURN_IF_ERROR(chunk_layout.Set(RankConstraint(2)));
  TENSORSTORE_RETURN_IF_ERROR(chunk_layout.Set(
      ChunkLayout::GridOrigin(GetConstantVector<Index, 0>(2))));

  ABSL_LOG(INFO) << "Calculated chunk layout: " << chunk_layout << std::endl;

  TENSORSTORE_RETURN_IF_ERROR(chunk_layout.Finalize());
  return chunk_layout;
}

class OMETiffDriver::OpenState : public OMETiffDriver::OpenStateBase {
 public:
  using OMETiffDriver::OpenStateBase::OpenStateBase;

  std::string GetPrefixForDeleteExisting() override {
    return spec().store.path;
  }

  std::string GetMetadataCacheEntryKey() override { return spec().store.path; }

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
        Result<MetadataCache::MetadataPtr>(
            std::make_shared<OMETiffImageInfo>(spec().metadata)),
        tensorstore::MaybeAnnotateStatus(
            _, "Cannot create using specified \"metadata\" and schema"));
    return metadata;
  }

  std::string GetDataCacheKey(const void* metadata) override {
    std::string result;
    const auto& spec = this->spec();
    internal::EncodeCacheKey(&result, spec.store.path);
    return result;
  }

  std::unique_ptr<internal_kvs_backed_chunk_driver::DataCacheBase> GetDataCache(
      DataCache::Initializer&& initializer) override {
    return std::make_unique<DataCache>(std::move(initializer),
                                       spec().store.path);
  }

  Result<std::size_t> GetComponentIndex(const void* metadata_ptr,
                                        OpenMode open_mode) override {
    ABSL_LOG(INFO) << "Getting component index";
    // const auto& metadata = *static_cast<const OMEMetadata*>(metadata_ptr);
    //  TENSORSTORE_RETURN_IF_ERROR(
    //      ValidateMetadataSchema(metadata, spec().schema));
    return 0;
  }
  Result<kvstore::DriverPtr> GetMetadataKeyValueStore(
      kvstore::DriverPtr base_kv_store) override {
    return ometiff::GetOMETiffKeyValueStore(base_kv_store, spec().store.path);
  }
};

}  // namespace internal_ometiff
}  // namespace tensorstore

TENSORSTORE_DEFINE_GARBAGE_COLLECTION_SPECIALIZATION(
    tensorstore::internal_ometiff::OMETiffDriver,
    tensorstore::internal_ometiff::OMETiffDriver::GarbageCollectionBase)

namespace {
const tensorstore::internal::DriverRegistration<
    tensorstore::internal_ometiff::OMETiffDriverSpec>
    registration;
}  // namespace