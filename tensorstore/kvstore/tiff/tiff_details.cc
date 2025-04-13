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

#include "tensorstore/kvstore/tiff/tiff_details.h"

#include <stddef.h>
#include <stdint.h>

#include <cassert>
#include <limits>

#include "absl/log/absl_log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "riegeli/bytes/reader.h"
#include "riegeli/endian/endian_reading.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/util/status.h"  // for TENSORSTORE_RETURN_IF_ERROR
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_tiff_kvstore {

namespace {

using ::riegeli::ReadBigEndian16;
using ::riegeli::ReadBigEndian32;
using ::riegeli::ReadBigEndian64;
using ::riegeli::ReadLittleEndian16;
using ::riegeli::ReadLittleEndian32;
using ::riegeli::ReadLittleEndian64;

ABSL_CONST_INIT internal_log::VerboseFlag tiff_logging("tiff_details");

// Helper function to read a value based on endianness
template <typename T>
bool ReadEndian(riegeli::Reader& reader, Endian endian, T& value) {
  if (endian == Endian::kLittle) {
    if constexpr (sizeof(T) == 2) return ReadLittleEndian16(reader, value);
    if constexpr (sizeof(T) == 4) return ReadLittleEndian32(reader, value);
    if constexpr (sizeof(T) == 8) return ReadLittleEndian64(reader, value);
  } else {
    if constexpr (sizeof(T) == 2) return ReadBigEndian16(reader, value);
    if constexpr (sizeof(T) == 4) return ReadBigEndian32(reader, value);
    if constexpr (sizeof(T) == 8) return ReadBigEndian64(reader, value);
  }
  return false;
}

// Helper to find an IFD entry by tag
const IfdEntry* GetIfdEntry(Tag tag, const std::vector<IfdEntry>& entries) {
  const IfdEntry* found = nullptr;
  for (const auto& entry : entries) {
    if (entry.tag == tag) {
      if (found) {
        return nullptr;  // Duplicate tag
      }
      found = &entry;
    }
  }
  return found;
}

// Helper to parse a uint32 value from an IFD entry
absl::Status ParseUint32Value(const IfdEntry* entry, uint32_t& out) {
  if (!entry) {
    return absl::NotFoundError("Required tag missing");
  }
  if (entry->count != 1) {
    return absl::InvalidArgumentError("Expected count of 1");
  }
  if (entry->type != TiffDataType::kShort && entry->type != TiffDataType::kLong) {
    return absl::InvalidArgumentError("Expected SHORT or LONG type");
  }
  out = static_cast<uint32_t>(entry->value_or_offset);
  return absl::OkStatus();
}

// Helper to parse array of uint64 values from an IFD entry
absl::Status ParseUint64Array(const IfdEntry* entry, std::vector<uint64_t>& out) {
  if (!entry) {
    return absl::NotFoundError("Required tag missing");
  }
  
  if (entry->type != TiffDataType::kShort && 
      entry->type != TiffDataType::kLong &&
      entry->type != TiffDataType::kLong8) {
    return absl::InvalidArgumentError("Expected SHORT, LONG, or LONG8 type");
  }

  // If this is an external array, it must be loaded separately
  if (entry->is_external_array) {
    // Initialize the output array with the correct size
    out.resize(entry->count);
    return absl::OkStatus();
  } else {
    // Inline value - parse it directly
    out.resize(entry->count);
    if (entry->count == 1) {
      out[0] = entry->value_or_offset;
      return absl::OkStatus();
    } else {
      // This shouldn't happen as we've checked is_external_array above
      return absl::InternalError("Inconsistent state: multi-value array marked as inline");
    }
  }
}

}  // namespace

// Get the size in bytes for a given TIFF data type
size_t GetTiffDataTypeSize(TiffDataType type) {
  switch (type) {
    case TiffDataType::kByte:
    case TiffDataType::kAscii:
    case TiffDataType::kSbyte:
    case TiffDataType::kUndefined:
      return 1;
    case TiffDataType::kShort:
    case TiffDataType::kSshort:
      return 2;
    case TiffDataType::kLong:
    case TiffDataType::kSlong:
    case TiffDataType::kFloat:
    case TiffDataType::kIfd:
      return 4;
    case TiffDataType::kRational:
    case TiffDataType::kSrational:
    case TiffDataType::kDouble:
    case TiffDataType::kLong8:
    case TiffDataType::kSlong8:
    case TiffDataType::kIfd8:
      return 8;
    default:
      return 0;  // Unknown type
  }
}

// Determine if an entry represents an external array based on type and count
bool IsExternalArray(TiffDataType type, uint64_t count) {
  // Calculate how many bytes the value would take
  size_t type_size = GetTiffDataTypeSize(type);
  size_t total_size = type_size * count;
  
  // If the total size is more than 4 bytes, it's stored externally
  // (4 bytes is the size of the value_or_offset field in standard TIFF)
  return (total_size > 4);
}

absl::Status ParseTiffHeader(
    riegeli::Reader& reader,
    Endian& endian,
    uint64_t& first_ifd_offset) {
  
  // Pull first 8 bytes which contain the header info
  if (!reader.Pull(8)) {
    return absl::InvalidArgumentError(
        "Failed to read TIFF header: insufficient data");
  }

  // Read byte order mark (II or MM)
  char byte_order[2];
  if (!reader.Read(2, byte_order)) {
    return absl::InvalidArgumentError(
        "Failed to read TIFF header byte order mark");
  }

  if (byte_order[0] == 'I' && byte_order[1] == 'I') {
    endian = Endian::kLittle;
  } else if (byte_order[0] == 'M' && byte_order[1] == 'M') {
    endian = Endian::kBig;
  } else {
    return absl::InvalidArgumentError(
        "Invalid TIFF byte order mark");
  }

  // Read magic number (42 for standard TIFF)
  uint16_t magic;
  if (!ReadEndian(reader, endian, magic) || magic != 42) {
    return absl::InvalidArgumentError(
        "Invalid TIFF magic number");
  }

  // Read offset to first IFD
  uint32_t offset32;
  if (!ReadEndian(reader, endian, offset32)) {
    return absl::InvalidArgumentError(
        "Failed to read first IFD offset");
  }
  first_ifd_offset = offset32;

  ABSL_LOG_IF(INFO, tiff_logging)
      << "TIFF header: endian=" << (endian == Endian::kLittle ? "little" : "big")
      << " first_ifd_offset=" << first_ifd_offset;

  return absl::OkStatus();
}

absl::Status ParseTiffDirectory(
    riegeli::Reader& reader,
    Endian endian,
    uint64_t directory_offset,
    size_t available_size,
    TiffDirectory& out) {
  
  // Position reader at directory offset
  if (!reader.Seek(directory_offset)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Failed to seek to IFD at offset %d", directory_offset));
  }

  // Read number of directory entries (2 bytes)
  if (available_size < 2) {
    return absl::DataLossError("Insufficient data to read IFD entry count");
  }

  uint16_t num_entries;
  if (!ReadEndian(reader, endian, num_entries)) {
    return absl::InvalidArgumentError("Failed to read IFD entry count");
  }

  // Each entry is 12 bytes, plus 2 bytes for count and 4 bytes for next IFD offset
  size_t required_size = 2 + (num_entries * 12) + 4;
  if (available_size < required_size) {
    return absl::DataLossError(absl::StrFormat(
        "Insufficient data to read complete IFD: need %d bytes, have %d",
        required_size, available_size));
  }

  // Initialize directory fields
  out.endian = endian;
  out.directory_offset = directory_offset;
  out.entries.clear();
  out.entries.reserve(num_entries);

  // Read each entry
  for (uint16_t i = 0; i < num_entries; ++i) {
    IfdEntry entry;
    
    // Read tag
    uint16_t tag_value;  // Temporary variable for reading the tag
    if (!ReadEndian(reader, endian, tag_value)) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Failed to read tag for IFD entry %d", i));
    }
    entry.tag = static_cast<Tag>(tag_value);  // Assign to enum

    // Read type
    uint16_t type_raw;
    if (!ReadEndian(reader, endian, type_raw)) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Failed to read type for IFD entry %d", i));
    }
    entry.type = static_cast<TiffDataType>(type_raw);

    // Read count
    uint32_t count32;
    if (!ReadEndian(reader, endian, count32)) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Failed to read count for IFD entry %d", i));
    }
    entry.count = count32;

    // Read value/offset
    uint32_t value32;
    if (!ReadEndian(reader, endian, value32)) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Failed to read value/offset for IFD entry %d", i));
    }
    entry.value_or_offset = value32;

    // Determine if this is an external array
    entry.is_external_array = IsExternalArray(entry.type, entry.count);

    ABSL_LOG_IF(INFO, tiff_logging)
        << absl::StrFormat("IFD entry %d: tag=0x%x type=%d count=%d value=%d external=%d",
                          i, entry.tag, static_cast<int>(entry.type),
                          entry.count, entry.value_or_offset, entry.is_external_array);

    out.entries.push_back(entry);
  }

  // Read offset to next IFD
  uint32_t next_ifd;
  if (!ReadEndian(reader, endian, next_ifd)) {
    return absl::InvalidArgumentError("Failed to read next IFD offset");
  }
  out.next_ifd_offset = next_ifd;

  ABSL_LOG_IF(INFO, tiff_logging)
      << "Read IFD with " << num_entries << " entries"
      << ", next_ifd_offset=" << out.next_ifd_offset;

  return absl::OkStatus();
}

absl::Status ParseExternalArray(
    riegeli::Reader& reader,
    Endian endian,
    uint64_t offset,
    uint64_t count,
    TiffDataType data_type,
    std::vector<uint64_t>& out) {
  
  // Ensure output vector has the right size
  out.resize(count);
  
  // Seek to the offset
  if (!reader.Seek(offset)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Failed to seek to external array at offset %llu", offset));
  }
  
  // Read based on data type
  for (uint64_t i = 0; i < count; ++i) {
    switch (data_type) {
      case TiffDataType::kShort: {
        uint16_t value;
        if (!ReadEndian(reader, endian, value)) {
          return absl::DataLossError(absl::StrFormat(
              "Failed to read SHORT value %llu in external array", i));
        }
        out[i] = value;
        break;
      }
      case TiffDataType::kLong: {
        uint32_t value;
        if (!ReadEndian(reader, endian, value)) {
          return absl::DataLossError(absl::StrFormat(
              "Failed to read LONG value %llu in external array", i));
        }
        out[i] = value;
        break;
      }
      case TiffDataType::kLong8: {
        uint64_t value;
        if (!ReadEndian(reader, endian, value)) {
          return absl::DataLossError(absl::StrFormat(
              "Failed to read LONG8 value %llu in external array", i));
        }
        out[i] = value;
        break;
      }
      default:
        return absl::InvalidArgumentError(absl::StrFormat(
            "Unsupported data type %d for external array",
            static_cast<int>(data_type)));
    }
  }
  
  ABSL_LOG_IF(INFO, tiff_logging)
      << absl::StrFormat("Read external array: offset=%llu, count=%llu",
                        offset, count);
  
  return absl::OkStatus();
}

absl::Status ParseImageDirectory(
    const std::vector<IfdEntry>& entries,
    ImageDirectory& out) {
  // Required fields for all TIFF files
  TENSORSTORE_RETURN_IF_ERROR(
      ParseUint32Value(GetIfdEntry(Tag::kImageWidth, entries), out.width));
  TENSORSTORE_RETURN_IF_ERROR(
      ParseUint32Value(GetIfdEntry(Tag::kImageLength, entries), out.height));

  // Check for tile-based organization
  const IfdEntry* tile_offsets = GetIfdEntry(Tag::kTileOffsets, entries);
  if (tile_offsets) {
    // Tiled TIFF
    TENSORSTORE_RETURN_IF_ERROR(
        ParseUint32Value(GetIfdEntry(Tag::kTileWidth, entries), out.tile_width));
    TENSORSTORE_RETURN_IF_ERROR(
        ParseUint32Value(GetIfdEntry(Tag::kTileLength, entries), out.tile_height));
    TENSORSTORE_RETURN_IF_ERROR(
        ParseUint64Array(tile_offsets, out.tile_offsets));
    
    const IfdEntry* tile_bytecounts = GetIfdEntry(Tag::kTileByteCounts, entries);
    TENSORSTORE_RETURN_IF_ERROR(
        ParseUint64Array(tile_bytecounts, out.tile_bytecounts));
  } else {
    // Strip-based TIFF
    TENSORSTORE_RETURN_IF_ERROR(
        ParseUint32Value(GetIfdEntry(Tag::kRowsPerStrip, entries), out.rows_per_strip));
    
    const IfdEntry* strip_offsets = GetIfdEntry(Tag::kStripOffsets, entries);
    TENSORSTORE_RETURN_IF_ERROR(
        ParseUint64Array(strip_offsets, out.strip_offsets));
    
    const IfdEntry* strip_bytecounts = GetIfdEntry(Tag::kStripByteCounts, entries);
    TENSORSTORE_RETURN_IF_ERROR(
        ParseUint64Array(strip_bytecounts, out.strip_bytecounts));
  }

  return absl::OkStatus();
}

}  // namespace internal_tiff_kvstore
}  // namespace tensorstore