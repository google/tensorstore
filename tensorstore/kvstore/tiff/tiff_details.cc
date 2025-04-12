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

}  // namespace

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
    if (!ReadEndian(reader, endian, entry.tag)) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Failed to read tag for IFD entry %d", i));
    }

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

    ABSL_LOG_IF(INFO, tiff_logging)
        << absl::StrFormat("IFD entry %d: tag=0x%x type=%d count=%d value=%d",
                          i, entry.tag, static_cast<int>(entry.type),
                          entry.count, entry.value_or_offset);

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

}  // namespace internal_tiff_kvstore
}  // namespace tensorstore