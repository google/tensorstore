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

#include "tensorstore/driver/ometiff/metadata.h"
#include "tensorstore/internal/kvs_read_streambuf.h"
#include "tensorstore/kvstore/registry.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace ometiff {
namespace {

using internal_ometiff::GetOMETiffMetadata;

Result<absl::Cord> DecodeTiffChunk(std::istream& istream, Index chunk_index);

class OMETiffMetadataKeyValueStore : public kvstore::Driver {
 public:
  explicit OMETiffMetadataKeyValueStore(kvstore::DriverPtr base,
                                        std::string key_prefix)
      : base_(std::move(base)), key_prefix_(key_prefix) {}

  Future<ReadResult> Read(Key key, ReadOptions options) override {
    ReadResult result;
    if (options.byte_range != OptionalByteRangeRequest()) {
      // Metadata doesn't need byte range request.
      return absl::InvalidArgumentError("Byte ranges not supported");
    }
    // TODO: plumb in buffer size.
    auto streambuf = internal::KvsReadStreambuf(base_, key, 100);
    std::istream stream(&streambuf);
    TENSORSTORE_ASSIGN_OR_RETURN(auto image_info, GetOMETiffMetadata(stream));
    ABSL_LOG(INFO) << image_info;
    result.stamp = TimestampedStorageGeneration{
        StorageGeneration::FromString(key), absl::Now()};
    result.state = ReadResult::kValue;
    result.value = absl::Cord(image_info.dump());
    return result;
  }

  void GarbageCollectionVisit(
      garbage_collection::GarbageCollectionVisitor& visitor) const final {
    // No-op
  }

  kvstore::Driver* base() { return base_.get(); }

 private:
  kvstore::DriverPtr base_;
  std::string key_prefix_;
};

class OMETiffDataKeyValueStore : public kvstore::Driver {
 public:
  // Need to plumb in metadata.
  explicit OMETiffDataKeyValueStore(kvstore::DriverPtr base,
                                    std::string key_prefix)
      : base_(std::move(base)), key_prefix_(key_prefix) {}

  Future<ReadResult> Read(Key key, ReadOptions options) override {
    ReadResult result;
    if (options.byte_range != OptionalByteRangeRequest()) {
      // Metadata doesn't need byte range request.
      return absl::InvalidArgumentError("Byte ranges not supported");
    }
    // TODO: plumb in buffer size.
    auto streambuf = internal::KvsReadStreambuf(base_, key_prefix_, 100);
    std::istream stream(&streambuf);
    TENSORSTORE_ASSIGN_OR_RETURN(auto read_result,
                                 DecodeTiffChunk(stream, KeyToChunk(key)));
    result.stamp = TimestampedStorageGeneration{
        StorageGeneration::FromString(key), absl::Now()};
    result.state = ReadResult::kValue;
    result.value = std::move(read_result);
    return result;
  }

  void GarbageCollectionVisit(
      garbage_collection::GarbageCollectionVisitor& visitor) const final {
    // No-op
  }

  static std::string ChunkToKey(uint64_t chunk) {
    std::string key;
    key.resize(sizeof(uint64_t));
    absl::big_endian::Store64(key.data(), chunk);
    return key;
  }

  static uint64_t KeyToChunk(std::string_view key) {
    assert(key.size() == sizeof(uint64_t));
    return absl::big_endian::Load64(key.data());
  }

  kvstore::Driver* base() { return base_.get(); }

 private:
  kvstore::DriverPtr base_;
  std::string key_prefix_;
};

// Result<absl::Cord> DecodeTiffChunk(std::istream& istream, Index chunk_index)
// {
//   ABSL_LOG(INFO) << "Opening TIFF";
//   TIFF* tiff = TIFFStreamOpen("ts", &istream);

//   std::unique_ptr<TIFF, void (*)(TIFF*)> tiff_scope(tiff, [](TIFF* tiff) {
//     if (tiff != nullptr) {
//       TIFFClose(tiff);
//     }
//   });

//   if (tiff == nullptr) {
//     return absl::DataLossError("Unable to read TIFF file");
//   }

//   if (TIFFIsTiled(tiff)) {
//     const int tile_bytes = TIFFTileSize(tiff);
//     uint64_t bytecount = TIFFGetStrileByteCount(tiff, chunk_index);
//     ABSL_LOG(INFO) << "Allocating " << tile_bytes
//                    << " bytes for true bytecount of " << bytecount;
//     std::unique_ptr<unsigned char[]> tile_buffer(new unsigned
//     char[tile_bytes]); if (TIFFReadEncodedTile(tiff, chunk_index,
//     tile_buffer.get(), tile_bytes) ==
//         -1) {
//       return absl::DataLossError("TIFF read tile failed");
//     }

//     // TODO: This seems wrong to me...
//     return absl::Cord(absl::string_view(
//         reinterpret_cast<char*>(tile_buffer.release()), tile_bytes));
//   } else {
//     const int strip_bytes = TIFFStripSize(tiff);
//     uint32_t rows_per_strip = 1;
//     TIFFGetFieldDefaulted(tiff, TIFFTAG_ROWSPERSTRIP, &rows_per_strip);
//     std::unique_ptr<unsigned char[]> strip_buffer(
//         new unsigned char[strip_bytes]);
//     if (TIFFReadEncodedStrip(tiff, chunk_index, strip_buffer.get(),
//                              strip_bytes) == -1) {
//       return absl::DataLossError("Tiff read strip failed");
//     }
//     // TODO: This seems wrong to me...
//     return absl::Cord(absl::string_view(
//         reinterpret_cast<char*>(strip_buffer.release()), strip_bytes));
//   }
// }

}  // namespace
kvstore::DriverPtr GetOMETiffMetadataKeyValueStore(
    kvstore::DriverPtr base_kvstore, std::string key_prefix) {
  return kvstore::DriverPtr(new OMETiffMetadataKeyValueStore(
      std::move(base_kvstore), std::move(key_prefix)));
}

// kvstore::DriverPtr GetOMETiffDataKeyValueStore(kvstore::DriverPtr
// base_kvstore,
//                                                std::string key_prefix) {
//   return kvstore::DriverPtr(new OMETiffDataKeyValueStore(
//       std::move(base_kvstore), std::move(key_prefix)));
// }

}  // namespace ometiff
}  // namespace tensorstore