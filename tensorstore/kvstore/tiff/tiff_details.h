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

#ifndef TENSORSTORE_KVSTORE_TIFF_TIFF_DETAILS_H_
#define TENSORSTORE_KVSTORE_TIFF_TIFF_DETAILS_H_

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "riegeli/bytes/reader.h"

namespace tensorstore {
namespace internal_tiff_kvstore {

enum class Endian {
  kLittle,
  kBig,
};

enum Tag : uint16_t {
  kImageWidth        = 256,
  kImageLength       = 257,
  kBitsPerSample     = 258,
  kCompression       = 259,
  kPhotometric       = 262,
  kStripOffsets      = 273,
  kRowsPerStrip      = 278,
  kStripByteCounts   = 279,
  kTileWidth         = 322,
  kTileLength        = 323,
  kTileOffsets       = 324,
  kTileByteCounts    = 325,
};

// TIFF data types
enum class TiffDataType : uint16_t {
  kByte = 1,      // 8-bit unsigned integer
  kAscii = 2,     // 8-bit bytes with last byte null
  kShort = 3,     // 16-bit unsigned integer
  kLong = 4,      // 32-bit unsigned integer
  kRational = 5,  // Two 32-bit unsigned integers
  kSbyte = 6,     // 8-bit signed integer
  kUndefined = 7, // 8-bit byte
  kSshort = 8,    // 16-bit signed integer
  kSlong = 9,     // 32-bit signed integer
  kSrational = 10,// Two 32-bit signed integers
  kFloat = 11,    // 32-bit IEEE floating point
  kDouble = 12,   // 64-bit IEEE floating point
  kIfd = 13,      // 32-bit unsigned integer (offset)
  kLong8 = 16,    // BigTIFF 64-bit unsigned integer
  kSlong8 = 17,   // BigTIFF 64-bit signed integer
  kIfd8 = 18,     // BigTIFF 64-bit unsigned integer (offset)
};

// IFD entry in a TIFF file
struct IfdEntry {
  Tag      tag;
  TiffDataType type;
  uint64_t count;
  uint64_t value_or_offset;  // For values that fit in 4/8 bytes, this is the value
                            // Otherwise, this is an offset to the data
  
  // Flag to indicate if this entry references an external array
  bool is_external_array = false;
};

// Represents a TIFF Image File Directory (IFD)
struct TiffDirectory {
  // Basic header info
  Endian endian;
  uint64_t directory_offset;  // Offset to this IFD from start of file
  uint64_t next_ifd_offset;   // Offset to next IFD (0 if none)

  // Entries in this IFD
  std::vector<IfdEntry> entries;
};

struct ImageDirectory {
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t tile_width = 0;
  uint32_t tile_height = 0;
  uint32_t rows_per_strip = 0;
  std::vector<uint64_t> strip_offsets;
  std::vector<uint64_t> strip_bytecounts;
  std::vector<uint64_t> tile_offsets;
  std::vector<uint64_t> tile_bytecounts;
};

// Parse the TIFF header at the current position
absl::Status ParseTiffHeader(
    riegeli::Reader& reader,
    Endian& endian,
    uint64_t& first_ifd_offset);

// Parse a TIFF directory at the given offset
absl::Status ParseTiffDirectory(
    riegeli::Reader& reader,
    Endian endian,
    uint64_t directory_offset,
    size_t available_size,
    TiffDirectory& out);

// Parse IFD entries into an ImageDirectory structure
absl::Status ParseImageDirectory(
    const std::vector<IfdEntry>& entries,
    ImageDirectory& out);

// Parse an external array from a reader
absl::Status ParseExternalArray(
    riegeli::Reader& reader,
    Endian endian,
    uint64_t offset,
    uint64_t count,
    TiffDataType data_type,
    std::vector<uint64_t>& out);

// Determine if an IFD entry represents an external array based on type and count
bool IsExternalArray(TiffDataType type, uint64_t count);

// Get the size in bytes for a given TIFF data type
size_t GetTiffDataTypeSize(TiffDataType type);

}  // namespace internal_tiff_kvstore
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_TIFF_TIFF_DETAILS_H_