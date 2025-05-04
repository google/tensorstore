// Copyright 2025 The TensorStore Authors
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
#include "tensorstore/util/status.h"
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
  if (entry->type != TiffDataType::kShort &&
      entry->type != TiffDataType::kLong) {
    return absl::InvalidArgumentError("Expected SHORT or LONG type");
  }
  out = static_cast<uint32_t>(entry->value_or_offset);
  return absl::OkStatus();
}

// Helper to parse array of uint64 values from an IFD entry
absl::Status ParseUint64Array(const IfdEntry* entry,
                              std::vector<uint64_t>& out) {
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
    out.resize(entry->count);
    return absl::OkStatus();
  } else {
    // Inline value - parse it directly
    out.resize(entry->count);
    if (entry->count == 1) {
      out[0] = entry->value_or_offset;
      return absl::OkStatus();
    } else {
      return absl::InternalError(
          "Inconsistent state: multi-value array marked as inline");
    }
  }
}

// Helper to parse a uint16 value from an IFD entry
absl::Status ParseUint16Value(const IfdEntry* entry, uint16_t& out) {
  if (!entry) {
    return absl::NotFoundError("Required tag missing");
  }
  if (entry->count != 1) {
    return absl::InvalidArgumentError("Expected count of 1");
  }
  if (entry->type != TiffDataType::kShort) {
    return absl::InvalidArgumentError("Expected SHORT type");
  }
  out = static_cast<uint16_t>(entry->value_or_offset);
  return absl::OkStatus();
}

// Helper function to parse array of uint16 values from an IFD entry
absl::Status ParseUint16Array(const IfdEntry* entry,
                              std::vector<uint16_t>& out) {
  if (!entry) {
    return absl::NotFoundError("Required tag missing");
  }

  if (entry->type != TiffDataType::kShort) {
    return absl::InvalidArgumentError("Expected SHORT type");
  }

  // If this is an external array, it must be loaded separately
  if (entry->is_external_array) {
    out.resize(entry->count);
    return absl::OkStatus();
  } else {
    // Inline value - parse it directly
    out.resize(entry->count);
    if (entry->count == 1) {
      out[0] = static_cast<uint16_t>(entry->value_or_offset);
      return absl::OkStatus();
    } else {
      return absl::InternalError(
          "Inconsistent state: multi-value array marked as inline");
    }
  }
}

// Helper to calculate the number of chunks/tiles/strips
std::tuple<uint64_t, uint32_t, uint32_t> CalculateChunkCounts(
    uint32_t image_width, uint32_t image_height, uint32_t chunk_width,
    uint32_t chunk_height) {
  if (chunk_width == 0 || chunk_height == 0) {
    return {0, 0, 0};
  }
  uint32_t num_cols = (image_width + chunk_width - 1) / chunk_width;
  uint32_t num_rows = (image_height + chunk_height - 1) / chunk_height;
  uint64_t num_chunks = static_cast<uint64_t>(num_rows) * num_cols;
  return {num_chunks, num_rows, num_cols};
}

}  // namespace

absl::Status ParseUint16Array(riegeli::Reader& reader, Endian endian,
                              uint64_t offset, uint64_t count,
                              std::vector<uint16_t>& out) {
  out.resize(count);

  if (!reader.Seek(offset)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Failed to seek to external array at offset %llu", offset));
  }

  for (uint64_t i = 0; i < count; ++i) {
    uint16_t value;
    if (!ReadEndian(reader, endian, value)) {
      return absl::DataLossError(absl::StrFormat(
          "Failed to read SHORT value %llu in external array", i));
    }
    out[i] = value;
  }

  ABSL_LOG_IF(INFO, tiff_logging) << absl::StrFormat(
      "Read uint16 external array: offset=%llu, count=%llu", offset, count);

  return absl::OkStatus();
}

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
  size_t type_size = GetTiffDataTypeSize(type);
  size_t total_size = type_size * count;

  // If the total size is more than 4 bytes, it's stored externally
  // (4 bytes is the size of the value_or_offset field in standard TIFF)
  return (total_size > 4);
}

absl::Status ParseTiffHeader(riegeli::Reader& reader, Endian& endian,
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
    return absl::InvalidArgumentError("Invalid TIFF byte order mark");
  }

  // Read magic number (42 for standard TIFF)
  uint16_t magic;
  if (!ReadEndian(reader, endian, magic) || magic != 42) {
    return absl::InvalidArgumentError("Invalid TIFF magic number");
  }

  // Read offset to first IFD
  uint32_t offset32;
  if (!ReadEndian(reader, endian, offset32)) {
    return absl::InvalidArgumentError("Failed to read first IFD offset");
  }
  first_ifd_offset = offset32;

  ABSL_LOG_IF(INFO, tiff_logging)
      << "TIFF header: endian="
      << (endian == Endian::kLittle ? "little" : "big")
      << " first_ifd_offset=" << first_ifd_offset;

  return absl::OkStatus();
}

absl::Status ParseTiffDirectory(riegeli::Reader& reader, Endian endian,
                                uint64_t directory_offset,
                                size_t available_size, TiffDirectory& out) {
  if (!reader.Seek(directory_offset)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Failed to seek to IFD at offset %d", directory_offset));
  }

  if (available_size < 2) {
    return absl::DataLossError("Insufficient data to read IFD entry count");
  }

  uint16_t num_entries;
  if (!ReadEndian(reader, endian, num_entries)) {
    return absl::InvalidArgumentError("Failed to read IFD entry count");
  }

  // Each entry is 12 bytes, plus 2 bytes for count and 4 bytes for next IFD
  // offset
  size_t required_size = 2 + (num_entries * 12) + 4;
  if (available_size < required_size) {
    return absl::DataLossError(absl::StrFormat(
        "Insufficient data to read complete IFD: need %d bytes, have %d",
        required_size, available_size));
  }

  out.endian = endian;
  out.directory_offset = directory_offset;
  out.entries.clear();
  out.entries.reserve(num_entries);

  for (uint16_t i = 0; i < num_entries; ++i) {
    IfdEntry entry;

    // Read tag
    uint16_t tag_value;
    if (!ReadEndian(reader, endian, tag_value)) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Failed to read tag for IFD entry %d", i));
    }
    entry.tag = static_cast<Tag>(tag_value);

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

    ABSL_LOG_IF(INFO, tiff_logging) << absl::StrFormat(
        "IFD entry %d: tag=0x%x type=%d count=%d value=%d external=%d", i,
        entry.tag, static_cast<int>(entry.type), entry.count,
        entry.value_or_offset, entry.is_external_array);

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

absl::Status ParseExternalArray(riegeli::Reader& reader, Endian endian,
                                uint64_t offset, uint64_t count,
                                TiffDataType data_type,
                                std::vector<uint64_t>& out) {
  out.resize(count);

  if (!reader.Seek(offset)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Failed to seek to external array at offset %llu", offset));
  }

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
        return absl::InvalidArgumentError(
            absl::StrFormat("Unsupported data type %d for external array",
                            static_cast<int>(data_type)));
    }
  }

  ABSL_LOG_IF(INFO, tiff_logging) << absl::StrFormat(
      "Read external array: offset=%llu, count=%llu", offset, count);

  return absl::OkStatus();
}

absl::Status ParseImageDirectory(const std::vector<IfdEntry>& entries,
                                 ImageDirectory& out) {
  // Required fields for all TIFF files
  TENSORSTORE_RETURN_IF_ERROR(
      ParseUint32Value(GetIfdEntry(Tag::kImageWidth, entries), out.width));
  TENSORSTORE_RETURN_IF_ERROR(
      ParseUint32Value(GetIfdEntry(Tag::kImageLength, entries), out.height));

  // Parse optional fields

  // Samples Per Pixel (defaults to 1 if missing)
  const IfdEntry* spp_entry = GetIfdEntry(Tag::kSamplesPerPixel, entries);
  if (spp_entry) {
    TENSORSTORE_RETURN_IF_ERROR(
        ParseUint16Value(spp_entry, out.samples_per_pixel));
  } else {
    out.samples_per_pixel = 1;
  }

  // Bits Per Sample (defaults to 1 bit per sample if missing)
  const IfdEntry* bps_entry = GetIfdEntry(Tag::kBitsPerSample, entries);
  if (bps_entry) {
    TENSORSTORE_RETURN_IF_ERROR(
        ParseUint16Array(bps_entry, out.bits_per_sample));
    if (out.bits_per_sample.size() != out.samples_per_pixel &&
        out.bits_per_sample.size() !=
            1) {  // Allow single value for all samples
      return absl::InvalidArgumentError(
          "BitsPerSample count does not match SamplesPerPixel");
    }
    // If only one value provided, replicate it for all samples
    if (out.bits_per_sample.size() == 1 && out.samples_per_pixel > 1) {
      out.bits_per_sample.resize(out.samples_per_pixel, out.bits_per_sample[0]);
    }
  } else {
    out.bits_per_sample.assign(out.samples_per_pixel, 1);
  }

  // Compression (defaults to None if missing)
  const IfdEntry* comp_entry = GetIfdEntry(Tag::kCompression, entries);
  if (comp_entry) {
    TENSORSTORE_RETURN_IF_ERROR(ParseUint16Value(comp_entry, out.compression));
  } else {
    out.compression = static_cast<uint16_t>(CompressionType::kNone);
  }

  // Photometric Interpretation (defaults to 0 if missing)
  const IfdEntry* photo_entry = GetIfdEntry(Tag::kPhotometric, entries);
  if (photo_entry) {
    TENSORSTORE_RETURN_IF_ERROR(ParseUint16Value(photo_entry, out.photometric));
  } else {
    out.photometric = 0;  // Default WhiteIsZero
  }

  // Planar Configuration (defaults to Chunky if missing)
  const IfdEntry* planar_entry = GetIfdEntry(Tag::kPlanarConfig, entries);
  if (planar_entry) {
    TENSORSTORE_RETURN_IF_ERROR(
        ParseUint16Value(planar_entry, out.planar_config));
  } else {
    out.planar_config = static_cast<uint16_t>(PlanarConfigType::kChunky);
  }

  // Sample Format (defaults to uint if missing)
  const IfdEntry* format_entry = GetIfdEntry(Tag::kSampleFormat, entries);
  if (format_entry) {
    TENSORSTORE_RETURN_IF_ERROR(
        ParseUint16Array(format_entry, out.sample_format));
    // Validate size matches SamplesPerPixel
    if (out.sample_format.size() != out.samples_per_pixel &&
        out.sample_format.size() != 1) {
      return absl::InvalidArgumentError(
          "SampleFormat count does not match SamplesPerPixel");
    }
    // If only one value provided, replicate it for all samples
    if (out.sample_format.size() == 1 && out.samples_per_pixel > 1) {
      out.sample_format.resize(out.samples_per_pixel, out.sample_format[0]);
    }
  } else {
    out.sample_format.assign(
        out.samples_per_pixel,
        static_cast<uint16_t>(SampleFormatType::kUnsignedInteger));
  }

  // Determine tiled vs. stripped and parse chunk info
  const IfdEntry* tile_width_entry = GetIfdEntry(Tag::kTileWidth, entries);
  const IfdEntry* rows_per_strip_entry =
      GetIfdEntry(Tag::kRowsPerStrip, entries);

  if (tile_width_entry) {
    out.is_tiled = true;
    if (rows_per_strip_entry) {
      ABSL_LOG_IF(WARNING, tiff_logging)
          << "Both TileWidth and RowsPerStrip present; ignoring RowsPerStrip.";
    }

    TENSORSTORE_RETURN_IF_ERROR(
        ParseUint32Value(tile_width_entry, out.chunk_width));
    TENSORSTORE_RETURN_IF_ERROR(ParseUint32Value(
        GetIfdEntry(Tag::kTileLength, entries), out.chunk_height));

    const IfdEntry* offsets_entry = GetIfdEntry(Tag::kTileOffsets, entries);
    const IfdEntry* counts_entry = GetIfdEntry(Tag::kTileByteCounts, entries);

    if (!offsets_entry)
      return absl::NotFoundError("TileOffsets tag missing for tiled image");
    if (!counts_entry)
      return absl::NotFoundError("TileByteCounts tag missing for tiled image");

    TENSORSTORE_RETURN_IF_ERROR(
        ParseUint64Array(offsets_entry, out.chunk_offsets));
    TENSORSTORE_RETURN_IF_ERROR(
        ParseUint64Array(counts_entry, out.chunk_bytecounts));

    // Validate counts
    auto [num_chunks, num_rows, num_cols] = CalculateChunkCounts(
        out.width, out.height, out.chunk_width, out.chunk_height);
    if (out.chunk_offsets.size() != num_chunks) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "TileOffsets count (%d) does not match expected number of tiles (%d)",
          out.chunk_offsets.size(), num_chunks));
    }
    if (out.chunk_bytecounts.size() != num_chunks) {
      return absl::InvalidArgumentError(
          absl::StrFormat("TileByteCounts count (%d) does not match expected "
                          "number of tiles (%d)",
                          out.chunk_bytecounts.size(), num_chunks));
    }

  } else {
    // Stripped Mode
    out.is_tiled = false;
    if (!rows_per_strip_entry) {
      // Neither TileWidth nor RowsPerStrip found
      return absl::NotFoundError(
          "Neither TileWidth nor RowsPerStrip tag found");
    }

    TENSORSTORE_RETURN_IF_ERROR(
        ParseUint32Value(rows_per_strip_entry, out.chunk_height));
    // Strip width is always the image width
    out.chunk_width = out.width;

    const IfdEntry* offsets_entry = GetIfdEntry(Tag::kStripOffsets, entries);
    const IfdEntry* counts_entry = GetIfdEntry(Tag::kStripByteCounts, entries);

    if (!offsets_entry)
      return absl::NotFoundError("StripOffsets tag missing for stripped image");
    if (!counts_entry)
      return absl::NotFoundError(
          "StripByteCounts tag missing for stripped image");

    TENSORSTORE_RETURN_IF_ERROR(
        ParseUint64Array(offsets_entry, out.chunk_offsets));
    TENSORSTORE_RETURN_IF_ERROR(
        ParseUint64Array(counts_entry, out.chunk_bytecounts));

    // Validate counts
    auto [num_chunks, num_rows, num_cols] = CalculateChunkCounts(
        out.width, out.height, out.chunk_width, out.chunk_height);

    if (out.chunk_offsets.size() != out.chunk_bytecounts.size()) {
      return absl::InvalidArgumentError(
          "StripOffsets and StripByteCounts have different counts");
    }
    if (out.chunk_offsets.size() != num_chunks) {
      ABSL_LOG_IF(WARNING, tiff_logging) << absl::StrFormat(
          "StripOffsets/Counts size (%d) does not match expected number of "
          "strips (%d) based on RowsPerStrip",
          out.chunk_offsets.size(), num_chunks);
    }
  }

  return absl::OkStatus();
}

}  // namespace internal_tiff_kvstore
}  // namespace tensorstore