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
#include "riegeli/bytes/read_all.h"
#include "tensorstore/driver/ometiff/driver_impl.h"
#include "tensorstore/driver/ometiff/metadata.h"
#include "tensorstore/driver/registry.h"
#include "tensorstore/internal/cache_key/cache_key.h"
#include "tensorstore/internal/compression/zstd_compressor.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/path.h"
#include "tensorstore/kvstore/ometiff/ometiff_key_value_store.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/util/endian.h"

namespace tensorstore {
namespace internal_ometiff {

namespace {
namespace jb = tensorstore::internal_json_binding;

Result<std::shared_ptr<const OMETiffMetadata>> ParseEncodedMetadata(
    std::string_view encoded_value) {
  nlohmann::json raw_data = nlohmann::json::parse(encoded_value, nullptr,
                                                  /*allow_exceptions=*/false);
  if (raw_data.is_discarded()) {
    return absl::FailedPreconditionError("Invalid JSON");
  }
  TENSORSTORE_ASSIGN_OR_RETURN(auto metadata,
                               OMETiffMetadata::FromJson(std::move(raw_data)));
  return std::make_shared<OMETiffMetadata>(std::move(metadata));
}

Index ComputeChunkIndex(const OMETiffMetadata& metadata,
                        const span<const Index>& cell_indices) {
  auto rank = metadata.rank;

  std::vector<Index> num_chunks(rank);
  for (Index i = 0; i < rank; ++i) {
    num_chunks[i] = metadata.shape[i] / metadata.chunk_shape[i];
  }

  Index index = 0;
  for (Index i = 0; i < rank; ++i) {
    index *= num_chunks[i];
    index += cell_indices[i];
  }
  return index;
}

int64_t CalculateChunkElements(const OMETiffMetadata& metadata,
                               const span<const Index>& cell_indices) {
  int64_t elements = 1;
  auto rank = metadata.rank;
  auto& chunk_shape = metadata.chunk_shape;
  auto& shape = metadata.shape;
  for (Index i = 0; i < rank; ++i) {
    elements *=
        std::min(chunk_shape[i], shape[i] - chunk_shape[i] * cell_indices[i]);
  }
  return elements;
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
      ::nlohmann::json(*static_cast<const OMETiffMetadata*>(metadata)).dump());
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
           GetChunkGridSpecification(*static_cast<const OMETiffMetadata*>(
               initializer.metadata.get()))),
      key_(std::move(key)) {}

OptionalByteRangeRequest DataCache::GetChunkByteRange(
    span<const Index> cell_indices) {
  auto& metadata = this->metadata();
  ABSL_LOG(INFO) << "Requested cell indices: " << cell_indices << " mapping to "
                 << ComputeChunkIndex(metadata, cell_indices);

  auto& chunk_info =
      metadata.chunk_info[ComputeChunkIndex(metadata, cell_indices)];
  return ByteRange{static_cast<int64_t>(chunk_info.offset),
                   static_cast<int64_t>(chunk_info.offset + chunk_info.size)};
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
  auto new_metadata = std::make_shared<OMETiffMetadata>(
      *static_cast<const OMETiffMetadata*>(existing_metadata));
  const DimensionIndex rank = new_metadata->rank;  // TODO: fix me.
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
    const OMETiffMetadata& metadata) {
  // TODO: Add multiple components (resolutions) here.

  ABSL_LOG(INFO) << "Get chunk grid specification";

  SharedArray<const void> fill_value =
      AllocateArray(metadata.chunk_shape, c_order, value_init, metadata.dtype);
  internal::ChunkGridSpecification::ComponentList components;
  components.emplace_back(std::move(fill_value), Box<>(metadata.chunk_shape),
                          std::vector<DimensionIndex>{0, 1});

  // ChunkLayout chunk_layout;
  // chunk_layout.Set(tensorstore::ChunkLayout::InnerOrder({0, 1}));
  // chunk_layout.Set(tensorstore::ChunkLayout::ReadChunkShape(metadata.chunk_shape));
  // chunk_layout.Set(RankConstraint(2));
  // chunk_layout.Set(ChunkLayout::GridOrigin(GetConstantVector<Index, 0>(2)));

  // IndexDomain<> domain = IndexDomain<>(rank);
  // domain = WithImplicitDimensions(std::move(domain),
  //                                 /*implicit_lower_bounds=*/false,
  //                                 /*implicit_upper_bounds=*/false);

  // Box<> chunk_template(rank);
  // SharedArray<const void> fill_value;
  // fill_value.layout().set_rank(rank);
  // std::fill_n(fill_value.byte_strides().begin(), rank, 0);

  // internal::ChooseReadWriteChunkGrid(chunk_layout, domain.box(),
  //                                      chunk_template);

  // for (DimensionIndex component_dim = 0; component_dim < rank;
  //      ++component_dim) {
  //   const DimensionIndex external_dim =
  //       chunk_layout.inner_order()[component_dim];
  //   fill_value.shape()[component_dim] = chunk_template.shape()[external_dim];
  // }
  // fill_value.element_pointer() =
  // internal::AllocateAndConstructSharedElements(
  //     1, value_init, metadata.dtype);

  // ABSL_LOG(INFO) << "Chunk template: " << chunk_template;
  // internal::ChunkGridSpecification::ComponentList components;
  // components.emplace_back(std::move(fill_value), std::move(chunk_template));
  return components;
}

Result<absl::InlinedVector<SharedArray<const void>, 1>> DataCache::DecodeChunk(
    span<const Index> chunk_indices, absl::Cord data) {
  auto& dtype = metadata().dtype;

  auto array = AllocateArray(metadata().chunk_shape, c_order, default_init,
                             metadata().dtype);

  absl::InlinedVector<SharedArray<const void>, 1> components;
  if (metadata().compressor) {
    ABSL_LOG(INFO) << "Data is compressed, attempting to decode...";
    std::unique_ptr<riegeli::Reader> reader =
        std::make_unique<riegeli::CordReader<absl::Cord>>(std::move(data));
    reader = metadata().compressor->GetReader(std::move(reader), data.size());
    TENSORSTORE_RETURN_IF_ERROR(riegeli::ReadAll(std::move(reader), data));
  }

  // Tile chunks are always fixed size but strips are not.
  auto expected_bytes =
      metadata().is_tiled
          ? array.num_elements() * dtype.size()
          : CalculateChunkElements(metadata(), chunk_indices) * dtype.size();
  if (static_cast<Index>(data.size()) != expected_bytes) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Uncompressed chunk is ", data.size(), " bytes, but should be ",
        expected_bytes, " bytes"));
  }

  auto data_flat = data.Flatten();
  memcpy(array.data(), data_flat.data(), data.size());
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
  const auto& metadata = *static_cast<const OMETiffMetadata*>(metadata_ptr);
  assert(bounds.rank() == static_cast<DimensionIndex>(2));
  std::fill(bounds.origin().begin(), bounds.origin().end(), Index(0));
  std::copy(metadata.shape.begin(), metadata.shape.end(),
            bounds.shape().begin());
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
  const auto& metadata = *static_cast<const OMETiffMetadata*>(metadata_ptr);
  ChunkLayout chunk_layout;
  TENSORSTORE_RETURN_IF_ERROR(SetChunkLayoutFromMetadata(
      metadata.rank, metadata.chunk_shape, chunk_layout));
  TENSORSTORE_RETURN_IF_ERROR(chunk_layout.Finalize());

  ABSL_LOG(INFO) << "Calculated chunk layout: " << chunk_layout << std::endl;

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
            std::make_shared<OMETiffMetadata>(spec().metadata)),
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
    return ometiff::GetOMETiffMetadataKeyValueStore(base_kv_store,
                                                    spec().store.path);
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