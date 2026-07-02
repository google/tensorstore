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

#include "tensorstore/internal/compression/zip_details.h"

#include <stdint.h>

#include <algorithm>
#include <cassert>
#include <ctime>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/log/absl_log.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/time/time.h"
#include "riegeli/bytes/cord_writer.h"
#include "riegeli/bytes/limiting_reader.h"
#include "riegeli/bytes/prefix_limiting_reader.h"
#include "riegeli/bytes/reader.h"
#include "riegeli/bzip2/bzip2_reader.h"
#include "riegeli/digests/crc32_digester.h"
#include "riegeli/endian/endian_reading.h"
#include "riegeli/endian/endian_writing.h"
#include "riegeli/xz/xz_reader.h"
#include "riegeli/xz/xz_writer.h"
#include "riegeli/zlib/zlib_reader.h"
#include "riegeli/zlib/zlib_writer.h"
#include "riegeli/zstd/zstd_reader.h"
#include "riegeli/zstd/zstd_writer.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/internal/riegeli/find.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_builder.h"

namespace tensorstore {
namespace internal_zip {
namespace {

using ::riegeli::ReadLittleEndian;
using ::riegeli::WriteLittleEndian;

// 32-bit signatures (little-endian representations of PK.. literals)
constexpr uint32_t kLocalHeaderSignature = 0x04034b50;
constexpr uint32_t kCentralHeaderSignature = 0x02014b50;
constexpr uint32_t kEOCDSignature = 0x06054b50;
constexpr uint32_t kEOCD64LocatorSignature = 0x07064b50;
constexpr uint32_t kEOCD64Signature = 0x06064b50;
constexpr uint32_t kDataDescriptorSignature = 0x08074b50;

// ZIP versions (spec section 4.4.3)
constexpr uint16_t kVersionDefault = 20;  // Version 2.0
constexpr uint16_t kVersionZip64 = 45;    // Version 4.5

// Extra field tags (spec section 4.5)
enum class ZipExtraFieldId : uint16_t {
  kZip64 = 0x0001,
  kUnix = 0x000d,
  kNtfs = 0x000a,
  kUnixExtendedTimestamp = 0x5455,
  kUnixUidGid = 0x7875,
  kUnicodePath = 0x7075,
};

enum class TimestampPrecisionLevel {
  kMSDOS = 0,
  kUnix = 1,
  kNTFS = 2,
};

struct TimestampPrecision {
  TimestampPrecisionLevel mtime = TimestampPrecisionLevel::kMSDOS;
  TimestampPrecisionLevel atime = TimestampPrecisionLevel::kMSDOS;
  TimestampPrecisionLevel ctime = TimestampPrecisionLevel::kMSDOS;
};

// Unix extended timestamp flags for 0x5455.
enum class UnixExtendedFlags : uint8_t {
  kMtime = 0x01,
  kAtime = 0x02,
  kCtime = 0x04,
};

constexpr bool HasFlag(uint8_t flags, UnixExtendedFlags flag) {
  return (flags & static_cast<uint8_t>(flag)) != 0;
}

// Size of EOCD64 record minus the signature and size fields.
constexpr uint64_t kEOCD64RecordSizeWithoutFixedHeader = 44;

// Estimated size of the data descriptor if present (for estimating read size).
constexpr int64_t kDataDescriptorEstimatedSize = 12;

ABSL_CONST_INIT internal_log::VerboseFlag zip_logging("zip_details");

constexpr char kLocalHeaderLiteral[4] = {'P', 'K', 0x03, 0x04};
constexpr char kCentralHeaderLiteral[4] = {'P', 'K', 0x01, 0x02};
constexpr char kEOCDLiteral[4] = {'P', 'K', 0x05, 0x06};
constexpr char kEOCD64LocatorLiteral[4] = {'P', 'K', 0x06, 0x07};
constexpr char kEOCD64Literal[4] = {'P', 'K', 0x06, 0x06};
constexpr char kDataDescriptorLiteral[4] = {'P', 'K', 0x07, 0x08};

constexpr uint32_t GetSignatureFromBytes(const char bytes[4]) {
  return (static_cast<uint32_t>(static_cast<unsigned char>(bytes[3])) << 24) |
         (static_cast<uint32_t>(static_cast<unsigned char>(bytes[2])) << 16) |
         (static_cast<uint32_t>(static_cast<unsigned char>(bytes[1])) << 8) |
         static_cast<uint32_t>(static_cast<unsigned char>(bytes[0]));
}

static_assert(GetSignatureFromBytes(kLocalHeaderLiteral) ==
              kLocalHeaderSignature);
static_assert(GetSignatureFromBytes(kCentralHeaderLiteral) ==
              kCentralHeaderSignature);
static_assert(GetSignatureFromBytes(kEOCDLiteral) == kEOCDSignature);
static_assert(GetSignatureFromBytes(kEOCD64LocatorLiteral) ==
              kEOCD64LocatorSignature);
static_assert(GetSignatureFromBytes(kEOCD64Literal) == kEOCD64Signature);
static_assert(GetSignatureFromBytes(kDataDescriptorLiteral) ==
              kDataDescriptorSignature);

// Windows epoch 1601-01-01T00:00:00Z is 11644473600 seconds before
// Unix epoch 1970-01-01T00:00:00Z.
constexpr int64_t kWindowsEpochSeconds = 11644473600ULL;
const absl::Time kWindowsEpoch =
    ::absl::UnixEpoch() - ::absl::Seconds(kWindowsEpochSeconds);

uint64_t ToWindowsTicks(absl::Time t) {
  if (t < kWindowsEpoch) return 0;
  absl::Duration d = t - kWindowsEpoch;
  int64_t seconds = absl::ToInt64Seconds(d);
  int64_t nanoseconds = absl::ToInt64Nanoseconds(d - absl::Seconds(seconds));
  return static_cast<uint64_t>(seconds) * 10000000ULL + nanoseconds / 100;
}

absl::Time FromWindowsTicks(uint64_t ticks) {
  uint64_t seconds = ticks / 10000000ULL;
  uint64_t remainder = ticks % 10000000ULL;
  return kWindowsEpoch + absl::Seconds(seconds) +
         absl::Nanoseconds(remainder * 100);
}

constexpr int kMSDOSYearEpoch = 80;
constexpr int kMSDOSMaxYearOffset = 127;

absl::Status ValidateAsciiString(std::string_view filename) {
  for (char c : filename) {
    if (static_cast<unsigned char>(c) > 127) {
      return absl::InvalidArgumentError(
          "Filename contains non-ASCII characters");
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::Time MakeMSDOSTime(MSDOSTime dos_time) {
  // Like DosDateTimeToVariantTime;
  struct tm dos_tm = {};
  dos_tm.tm_mday = static_cast<uint16_t>(dos_time.date & 0x1f);
  dos_tm.tm_mon = static_cast<uint16_t>((dos_time.date >> 5) & 0xf) - 1;
  dos_tm.tm_year = static_cast<uint16_t>(dos_time.date >> 9) + kMSDOSYearEpoch;
  dos_tm.tm_hour = static_cast<uint16_t>(dos_time.time >> 11);
  dos_tm.tm_min = static_cast<uint16_t>((dos_time.time >> 5) & 0x3f);
  dos_tm.tm_sec = static_cast<uint16_t>(2 * (dos_time.time & 0x1f));
  dos_tm.tm_isdst = -1;

  // TODO: Time zone?
  return absl::FromTM(dos_tm, absl::UTCTimeZone());
}

MSDOSTime ValueToMSDOSTime(absl::Time time) {
  struct tm dos_tm = absl::ToTM(time, absl::UTCTimeZone());
  uint16_t date = 0;
  date |= (dos_tm.tm_mday & 0x1f);
  date |= (((dos_tm.tm_mon + 1) & 0xf) << 5);
  int year_offset = dos_tm.tm_year - kMSDOSYearEpoch;
  if (year_offset < 0) year_offset = 0;
  if (year_offset > kMSDOSMaxYearOffset) year_offset = kMSDOSMaxYearOffset;
  date |= ((year_offset & 0x7f) << 9);

  uint16_t dos_time = 0;
  dos_time |= ((dos_tm.tm_sec / 2) & 0x1f);
  dos_time |= ((dos_tm.tm_min & 0x3f) << 5);
  dos_time |= ((dos_tm.tm_hour & 0x1f) << 11);

  return {date, dos_time};
}

namespace {

constexpr uint64_t kMaxUncompressedSize = 2ULL << 30;  // 2 GB
constexpr uint64_t kMaxCompressionRatio = 1024;
constexpr uint64_t kMinSizeForRatioCheck = 1024 * 1024;  // 1 MB

absl::Status ValidateEntrySizes(uint64_t uncompressed_size,
                                uint64_t compressed_size) {
  if (uncompressed_size > kMaxUncompressedSize) {
    return absl::InvalidArgumentError(
        absl::StrFormat("ZIP entry uncompressed size (%d) exceeds limit of %d",
                        uncompressed_size, kMaxUncompressedSize));
  }
  if (compressed_size > 0 && uncompressed_size > kMinSizeForRatioCheck) {
    if (uncompressed_size > compressed_size * kMaxCompressionRatio) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "ZIP entry compression ratio (%f) exceeds limit of %d",
          static_cast<double>(uncompressed_size) / compressed_size,
          kMaxCompressionRatio));
    }
  }
  return absl::OkStatus();
}

// These could have different implementations for central headers vs.
// local headers.
absl::Status ReadExtraField_Zip64_0001(riegeli::Reader& reader,
                                       uint16_t tag_size, ZipEntry& entry) {
  if (tag_size < 8) {
    return absl::InvalidArgumentError("ZIP64 extra field too small");
  }

  static constexpr auto kReadError = "Failed to read ZIP64 extra field";

  entry.is_zip64 = true;
  // Only read a field if the corresponding 32-bit field is UINT32_MAX,
  // indicating the real value is in the ZIP64 extra field.
  if (tag_size >= 8 &&
      entry.uncompressed_size == std::numeric_limits<uint32_t>::max()) {
    if (!ReadLittleEndian<uint64_t>(reader, entry.uncompressed_size)) {
      return absl::InvalidArgumentError(kReadError);
    }
    tag_size -= 8;
  }
  if (tag_size >= 8 &&
      entry.compressed_size == std::numeric_limits<uint32_t>::max()) {
    if (!ReadLittleEndian<uint64_t>(reader, entry.compressed_size)) {
      return absl::InvalidArgumentError(kReadError);
    }
    tag_size -= 8;
  }
  if (tag_size >= 8 &&
      entry.local_header_offset == std::numeric_limits<uint32_t>::max()) {
    if (!ReadLittleEndian<uint64_t>(reader, entry.local_header_offset)) {
      return absl::InvalidArgumentError(kReadError);
    }
    tag_size -= 8;
  }
  // Remaining bytes (e.g. disk number) are ignored.
  return absl::OkStatus();
}

uint16_t Zip64ExtraFieldBodySize(const ZipEntry& entry, bool is_local) {
  uint16_t size = 0;
  if (entry.uncompressed_size >= std::numeric_limits<uint32_t>::max()) {
    size += 8;
  }
  if (entry.compressed_size >= std::numeric_limits<uint32_t>::max()) {
    size += 8;
  }
  if (!is_local &&
      entry.local_header_offset >= std::numeric_limits<uint32_t>::max()) {
    size += 8;
  }
  return size;
}

absl::Status WriteExtraField_Zip64_0001(riegeli::Writer& writer,
                                        const ZipEntry& entry, bool is_local) {
  uint16_t body_size = Zip64ExtraFieldBodySize(entry, is_local);
  if (body_size == 0) return absl::OkStatus();

  WriteLittleEndian<uint16_t>(static_cast<uint16_t>(ZipExtraFieldId::kZip64),
                              writer);             // Tag
  WriteLittleEndian<uint16_t>(body_size, writer);  // Size
  if (entry.uncompressed_size >= std::numeric_limits<uint32_t>::max()) {
    WriteLittleEndian<uint64_t>(entry.uncompressed_size, writer);
  }
  if (entry.compressed_size >= std::numeric_limits<uint32_t>::max()) {
    WriteLittleEndian<uint64_t>(entry.compressed_size, writer);
  }
  if (!is_local &&
      entry.local_header_offset >= std::numeric_limits<uint32_t>::max()) {
    WriteLittleEndian<uint64_t>(entry.local_header_offset, writer);
  }
  if (!writer.ok()) return writer.status();
  return absl::OkStatus();
}

absl::Status ReadExtraField_Unix_000D(riegeli::Reader& reader,
                                      uint16_t tag_size, ZipEntry& entry,
                                      TimestampPrecision& precision) {
  if (tag_size < 12) {
    return absl::InvalidArgumentError("UNIX extra field too small");
  }
  uint32_t ignored32;
  uint32_t mtime;
  uint32_t atime;
  if (!ReadLittleEndian<uint32_t>(reader, atime) ||
      !ReadLittleEndian<uint32_t>(reader, mtime) ||
      !ReadLittleEndian<uint32_t>(reader, ignored32) /* uid+gid */) {
    return absl::InvalidArgumentError("Failed to read UNIX extra field");
  }
  // convert atime/mtime.
  if (precision.atime < TimestampPrecisionLevel::kUnix) {
    entry.atime = absl::FromUnixSeconds(atime);
    precision.atime = TimestampPrecisionLevel::kUnix;
  }
  if (precision.mtime < TimestampPrecisionLevel::kUnix) {
    entry.mtime = absl::FromUnixSeconds(mtime);
    precision.mtime = TimestampPrecisionLevel::kUnix;
  }
  // Ignore linkname.
  return absl::OkStatus();
}

// Unix 000d is considered superceded by 5455, so we don't write it.

absl::Status ReadExtraField_NTFS_000A(riegeli::Reader& reader,
                                      uint16_t tag_size, ZipEntry& entry,
                                      TimestampPrecision& precision) {
  if (tag_size < 8) {
    return absl::InvalidArgumentError("NTFS extra field too small");
  }
  uint32_t ignored32;
  if (!ReadLittleEndian<uint32_t>(reader, ignored32)) {
    return absl::InvalidArgumentError("Failed to read NTFS extra field");
  }
  tag_size -= 4;
  uint16_t ntfs_tag, ntfs_size;
  while (tag_size > 4) {
    if (!ReadLittleEndian<uint16_t>(reader, ntfs_tag) ||
        !ReadLittleEndian<uint16_t>(reader, ntfs_size)) {
      break;
    }
    tag_size -= 4;
    if (tag_size < ntfs_size) {
      return absl::InvalidArgumentError("NTFS extra field size mismatch");
    }
    tag_size -= ntfs_size;
    if (ntfs_tag == 0x0001 && ntfs_size == 24) {
      uint64_t mtime;
      uint64_t atime;
      uint64_t ctime;
      if (!ReadLittleEndian<uint64_t>(reader, mtime) ||
          !ReadLittleEndian<uint64_t>(reader, atime) ||
          !ReadLittleEndian<uint64_t>(reader, ctime)) {
        return absl::InvalidArgumentError("Failed to read NTFS extra field");
      }

      if (precision.mtime < TimestampPrecisionLevel::kNTFS) {
        entry.mtime = FromWindowsTicks(mtime);
        precision.mtime = TimestampPrecisionLevel::kNTFS;
      }
      if (precision.atime < TimestampPrecisionLevel::kNTFS) {
        entry.atime = FromWindowsTicks(atime);
        precision.atime = TimestampPrecisionLevel::kNTFS;
      }
      if (precision.ctime < TimestampPrecisionLevel::kNTFS) {
        entry.ctime = FromWindowsTicks(ctime);
        precision.ctime = TimestampPrecisionLevel::kNTFS;
      }
    } else {
      reader.Skip(ntfs_size);
    }
  }
  return absl::OkStatus();
}

absl::Status WriteExtraField_NTFS_000A(riegeli::Writer& writer,
                                       const ZipEntry& entry) {
  WriteLittleEndian<uint16_t>(static_cast<uint16_t>(ZipExtraFieldId::kNtfs),
                              writer);      // Tag
  WriteLittleEndian<uint16_t>(32, writer);  // Size of body
  WriteLittleEndian<uint32_t>(0, writer);   // Reserved
  WriteLittleEndian<uint16_t>(1, writer);   // Sub-tag
  WriteLittleEndian<uint16_t>(24, writer);  // Sub-size

  uint64_t mtime_ticks = ToWindowsTicks(entry.mtime);
  auto is_valid_time = [](absl::Time t) {
    return t != absl::Time() && t != absl::InfinitePast() &&
           t != absl::InfiniteFuture();
  };
  uint64_t atime_ticks =
      is_valid_time(entry.atime) ? ToWindowsTicks(entry.atime) : mtime_ticks;
  uint64_t ctime_ticks =
      is_valid_time(entry.ctime) ? ToWindowsTicks(entry.ctime) : mtime_ticks;

  WriteLittleEndian<uint64_t>(mtime_ticks, writer);
  WriteLittleEndian<uint64_t>(atime_ticks, writer);
  WriteLittleEndian<uint64_t>(ctime_ticks, writer);

  if (!writer.ok()) return writer.status();
  return absl::OkStatus();
}

uint16_t UnixExtendedTimestampExtraFieldBodySize(const ZipEntry& entry,
                                                 bool is_local) {
  auto is_valid_time = [](absl::Time t) {
    return t != absl::Time() && t != absl::InfinitePast() &&
           t != absl::InfiniteFuture();
  };
  uint8_t flags = 0;
  if (is_valid_time(entry.mtime)) {
    flags |= static_cast<uint8_t>(UnixExtendedFlags::kMtime);
  }
  if (is_valid_time(entry.atime)) {
    flags |= static_cast<uint8_t>(UnixExtendedFlags::kAtime);
  }
  if (is_valid_time(entry.ctime)) {
    flags |= static_cast<uint8_t>(UnixExtendedFlags::kCtime);
  }
  if (flags == 0) return 0;
  uint16_t size = 1;
  if (is_local) {
    if (flags & static_cast<uint8_t>(UnixExtendedFlags::kMtime)) size += 4;
    if (flags & static_cast<uint8_t>(UnixExtendedFlags::kAtime)) size += 4;
    if (flags & static_cast<uint8_t>(UnixExtendedFlags::kCtime)) size += 4;
  } else {
    if (flags & static_cast<uint8_t>(UnixExtendedFlags::kMtime)) size += 4;
  }
  return size;
}

absl::Status ReadExtraField_Unix_5455(riegeli::Reader& reader,
                                      uint16_t tag_size, ZipEntry& entry,
                                      TimestampPrecision& precision) {
  if (tag_size < 1) {
    return absl::InvalidArgumentError("Unix timestamp extra field too small");
  }

  static constexpr auto kReadError =
      "Failed to read unix timestamp extra field";

  uint8_t flags = 0;
  uint32_t tstamp = 0;
  if (!reader.ReadByte(flags)) {
    return absl::InvalidArgumentError(kReadError);
  }
  --tag_size;
  if (HasFlag(flags, UnixExtendedFlags::kMtime) && tag_size >= 4) {  // mtime
    if (!ReadLittleEndian<uint32_t>(reader, tstamp)) {
      return absl::InvalidArgumentError(kReadError);
    }
    tag_size -= 4;
    if (precision.mtime < TimestampPrecisionLevel::kUnix) {
      entry.mtime = absl::FromUnixSeconds(tstamp);
      precision.mtime = TimestampPrecisionLevel::kUnix;
    }
  }
  if (HasFlag(flags, UnixExtendedFlags::kAtime) && tag_size >= 4) {  // atime
    if (!ReadLittleEndian<uint32_t>(reader, tstamp)) {
      return absl::InvalidArgumentError(kReadError);
    }
    tag_size -= 4;
    if (precision.atime < TimestampPrecisionLevel::kUnix) {
      entry.atime = absl::FromUnixSeconds(tstamp);
      precision.atime = TimestampPrecisionLevel::kUnix;
    }
  }
  if (HasFlag(flags, UnixExtendedFlags::kCtime) && tag_size >= 4) {
    if (!ReadLittleEndian<uint32_t>(reader, tstamp)) {
      return absl::InvalidArgumentError(kReadError);
    }
    tag_size -= 4;
    if (precision.ctime < TimestampPrecisionLevel::kUnix) {
      entry.ctime = absl::FromUnixSeconds(tstamp);
      precision.ctime = TimestampPrecisionLevel::kUnix;
    }
  }
  return absl::OkStatus();
}

absl::Status WriteExtraField_Unix_5455(riegeli::Writer& writer,
                                       const ZipEntry& entry, bool is_local) {
  uint16_t body_size = UnixExtendedTimestampExtraFieldBodySize(entry, is_local);
  if (body_size == 0) return absl::OkStatus();

  WriteLittleEndian<uint16_t>(
      static_cast<uint16_t>(ZipExtraFieldId::kUnixExtendedTimestamp), writer);
  WriteLittleEndian<uint16_t>(body_size, writer);

  auto is_valid_time = [](absl::Time t) {
    return t != absl::Time() && t != absl::InfinitePast() &&
           t != absl::InfiniteFuture();
  };
  uint8_t flags = 0;
  if (is_valid_time(entry.mtime)) {
    flags |= static_cast<uint8_t>(UnixExtendedFlags::kMtime);
  }
  if (is_valid_time(entry.atime)) {
    flags |= static_cast<uint8_t>(UnixExtendedFlags::kAtime);
  }
  if (is_valid_time(entry.ctime)) {
    flags |= static_cast<uint8_t>(UnixExtendedFlags::kCtime);
  }

  writer.WriteByte(flags);

  if (flags & static_cast<uint8_t>(UnixExtendedFlags::kMtime)) {
    WriteLittleEndian<uint32_t>(
        static_cast<uint32_t>(absl::ToUnixSeconds(entry.mtime)), writer);
  }
  if (is_local) {
    if (flags & static_cast<uint8_t>(UnixExtendedFlags::kAtime)) {
      WriteLittleEndian<uint32_t>(
          static_cast<uint32_t>(absl::ToUnixSeconds(entry.atime)), writer);
    }
    if (flags & static_cast<uint8_t>(UnixExtendedFlags::kCtime)) {
      WriteLittleEndian<uint32_t>(
          static_cast<uint32_t>(absl::ToUnixSeconds(entry.ctime)), writer);
    }
  }

  if (!writer.ok()) return writer.status();
  return absl::OkStatus();
}

absl::Status ReadExtraField_InfoZipUnicodePath_7075(riegeli::Reader& reader,
                                                    uint16_t tag_size,
                                                    ZipEntry& entry) {
  if (tag_size < 5) {
    return absl::InvalidArgumentError("Unicode Path extra field too small");
  }
  uint8_t version;
  if (!reader.ReadByte(version)) {
    return absl::InvalidArgumentError("Failed to read Unicode Path version");
  }
  if (version != 1) {
    reader.Skip(tag_size - 1);
    return absl::OkStatus();
  }
  uint32_t name_crc32;
  if (!ReadLittleEndian<uint32_t>(reader, name_crc32)) {
    return absl::InvalidArgumentError("Failed to read Unicode Path NameCRC32");
  }
  std::string unicode_name;
  if (!reader.Read(tag_size - 5, unicode_name)) {
    return absl::InvalidArgumentError(
        "Failed to read Unicode Path UnicodeName");
  }
  riegeli::Crc32Digester digester;
  digester.Write(entry.filename);
  uint32_t actual_crc = digester.Digest();
  if (actual_crc != name_crc32) {
    return absl::InvalidArgumentError(
        "Info-ZIP Unicode Path CRC32 mismatch with standard filename");
  }
  if (unicode_name != entry.filename) {
    return absl::InvalidArgumentError(
        absl::StrFormat("ZIP entry filename '%s' is inconsistent with "
                        "Unicode Path extra field filename '%s'",
                        entry.filename, unicode_name));
  }
  entry.filename = std::move(unicode_name);
  return absl::OkStatus();
}

absl::Status ReadExtraField(riegeli::Reader& reader, ZipEntry& entry) {
  // These could have different implementations for central headers vs.
  // local headers.
  uint16_t tag, tag_size;
  absl::Status status;
  TimestampPrecision precision;
  while (reader.ok()) {
    if (!ReadLittleEndian<uint16_t>(reader, tag) ||
        !ReadLittleEndian<uint16_t>(reader, tag_size)) {
      // No more extra fields.
      break;
    }
    ABSL_LOG_IF(INFO, zip_logging)
        << absl::StrFormat("extra tag %04x size %d", tag, tag_size);
    auto pos = reader.pos();
    switch (static_cast<ZipExtraFieldId>(tag)) {
      case ZipExtraFieldId::kZip64:
        status.Update(ReadExtraField_Zip64_0001(reader, tag_size, entry));
        break;
      case ZipExtraFieldId::kUnix:
        status.Update(
            ReadExtraField_Unix_000D(reader, tag_size, entry, precision));
        break;
      case ZipExtraFieldId::kNtfs:
        status.Update(
            ReadExtraField_NTFS_000A(reader, tag_size, entry, precision));
        break;
      case ZipExtraFieldId::kUnixExtendedTimestamp:
        status.Update(
            ReadExtraField_Unix_5455(reader, tag_size, entry, precision));
        break;
      case ZipExtraFieldId::kUnicodePath:
        status.Update(
            ReadExtraField_InfoZipUnicodePath_7075(reader, tag_size, entry));
        break;
      case ZipExtraFieldId::kUnixUidGid:
        break;
      default:
        break;
    }
    if (reader.pos() > pos + tag_size) {
      return absl::InvalidArgumentError(
          "ZIP extra field parser read too much data");
    }
    reader.Seek(pos + tag_size);
  }
  return status;
}

absl::Status WriteExtraFields(riegeli::Writer& writer, const ZipEntry& entry,
                              bool is_local) {
  TENSORSTORE_RETURN_IF_ERROR(
      WriteExtraField_Zip64_0001(writer, entry, is_local));
  TENSORSTORE_RETURN_IF_ERROR(
      WriteExtraField_Unix_5455(writer, entry, is_local));
  TENSORSTORE_RETURN_IF_ERROR(WriteExtraField_NTFS_000A(writer, entry));
  return absl::OkStatus();
}

}  // namespace

absl::Status ReadEOCD64Locator(riegeli::Reader& reader,
                               ZipEOCD64Locator& locator) {
  if (!reader.Pull(ZipEOCD64Locator::kRecordSize)) {
    return absl::InvalidArgumentError(
        "ZIP EOCD64 Locator Entry insufficient data available");
  }

  uint32_t signature;
  ReadLittleEndian<uint32_t>(reader, signature);
  if (signature != kEOCD64LocatorSignature) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Failed to read ZIP64 End of Central Directory Locator signature %08x",
        signature));
  }

  uint32_t ignored32;
  ReadLittleEndian<uint32_t>(reader, locator.disk_number_with_cd);
  ReadLittleEndian<int64_t>(reader, locator.cd_offset);
  ReadLittleEndian<uint32_t>(reader, ignored32);
  if (locator.cd_offset < 0) {
    ABSL_LOG_IF(INFO, zip_logging && !reader.ok()) << reader.status();
    return absl::InvalidArgumentError(
        "Failed to read ZIP64 End of Central Directory Locator");
  }
  return absl::OkStatus();
}

absl::Status ReadEOCD64(riegeli::Reader& reader, ZipEOCD& eocd) {
  if (!reader.Pull(ZipEOCD::kEOCD64RecordSize)) {
    return absl::InvalidArgumentError(
        "ZIP EOCD Entry insufficient data available");
  }

  auto eocd_pos = reader.pos();
  uint32_t signature;
  ReadLittleEndian<uint32_t>(reader, signature);
  if (signature != kEOCD64Signature) {
    return absl::InvalidArgumentError(
        "Failed to read ZIP64 Central Directory Entry signature");
  }

  // Size = SizeOfFixedFields + SizeOfVariableData - 12.
  uint64_t eocd_size;
  ReadLittleEndian<uint64_t>(reader, eocd_size);
  if (eocd_size < kEOCD64RecordSizeWithoutFixedHeader ||
      !reader.Pull(eocd_size)) {
    return absl::InvalidArgumentError(
        "Failed to read ZIP64 End of Central Directory");
  }

  // Read remainder of EOCD64.
  riegeli::LimitingReader eocd64_reader(
      &reader,
      riegeli::LimitingReaderBase::Options().set_exact_length(eocd_size));

  uint16_t version_madeby;
  uint16_t version_needed_to_extract;
  uint32_t disk_number;
  uint32_t disk_number_with_cd;
  uint64_t total_num_entries;
  ReadLittleEndian<uint16_t>(eocd64_reader, version_madeby);
  ReadLittleEndian<uint16_t>(eocd64_reader, version_needed_to_extract);
  ReadLittleEndian<uint32_t>(eocd64_reader, disk_number);
  ReadLittleEndian<uint32_t>(eocd64_reader, disk_number_with_cd);
  ReadLittleEndian<uint64_t>(eocd64_reader, eocd.num_entries);
  ReadLittleEndian<uint64_t>(eocd64_reader, total_num_entries);
  ReadLittleEndian<int64_t>(eocd64_reader, eocd.cd_size);
  ReadLittleEndian<int64_t>(eocd64_reader, eocd.cd_offset);

  if (disk_number != disk_number_with_cd ||
      eocd.num_entries != total_num_entries ||
      eocd.num_entries == std::numeric_limits<uint64_t>::max() ||
      eocd.cd_size == std::numeric_limits<int64_t>::max() ||
      eocd.cd_offset == std::numeric_limits<int64_t>::max() ||
      eocd.cd_size < 0 || eocd.cd_offset < 0) {
    return absl::InvalidArgumentError(
        "Failed to read ZIP64 End of Central Directory");
  }

  eocd64_reader.Seek(eocd64_reader.max_pos());
  eocd.record_offset = eocd_pos;
  if (!eocd64_reader.Close()) {
    return eocd64_reader.status();
  }
  return absl::OkStatus();
}

absl::Status ReadEOCD(riegeli::Reader& reader, ZipEOCD& eocd) {
  if (!reader.Pull(ZipEOCD::kEOCDRecordSize)) {
    return absl::InvalidArgumentError(
        "ZIP EOCD Entry insufficient data available");
  }
  auto eocd_pos = reader.pos();
  uint32_t signature;
  ReadLittleEndian<uint32_t>(reader, signature);
  if (signature != kEOCDSignature) {
    return absl::InvalidArgumentError(
        "Failed to read ZIP Central Directory Entry signature");
  }
  uint16_t disk_number;
  uint16_t disk_number_with_cd;
  uint16_t num_entries;
  uint16_t total_num_entries;
  uint32_t cd_size;
  uint32_t cd_offset;
  uint16_t comment_length;
  ReadLittleEndian<uint16_t>(reader, disk_number);
  ReadLittleEndian<uint16_t>(reader, disk_number_with_cd);
  ReadLittleEndian<uint16_t>(reader, num_entries);
  ReadLittleEndian<uint16_t>(reader, total_num_entries);
  ReadLittleEndian<uint32_t>(reader, cd_size);
  ReadLittleEndian<uint32_t>(reader, cd_offset);
  ReadLittleEndian<uint16_t>(reader, comment_length);
  if (num_entries != total_num_entries) {
    ABSL_LOG_IF(INFO, zip_logging) << "ZIP num_entries mismatch " << num_entries
                                   << " vs " << total_num_entries;
    return absl::InvalidArgumentError(
        "Failed to read ZIP End of Central Directory");
  }
  if (disk_number != disk_number_with_cd) {
    ABSL_LOG_IF(INFO, zip_logging) << "ZIP disk_number mismatch " << disk_number
                                   << " vs " << disk_number_with_cd;
    return absl::InvalidArgumentError(
        "Failed to read ZIP End of Central Directory");
  }
  if (comment_length > 0 && !reader.Read(comment_length, eocd.comment)) {
    return absl::InvalidArgumentError(
        "Failed to read ZIP End of Central Directory");
  }
  // Validate that the reader is at EOF.
  reader.VerifyEnd();
  if (!reader.status().ok()) {
    return absl::InvalidArgumentError(
        "Failed to read ZIP End of Central Directory");
  }

  eocd.record_offset = eocd_pos;
  eocd.num_entries = num_entries;
  eocd.cd_size = cd_size;
  eocd.cd_offset = cd_offset;

  // Is this ZIP64?
  if (total_num_entries == std::numeric_limits<uint16_t>::max() ||
      cd_offset == std::numeric_limits<uint32_t>::max()) {
    eocd.cd_offset = std::numeric_limits<uint32_t>::max();
  }
  return absl::OkStatus();
}

// TODO: Modify kvstore::ReadResult to include the returned range as well
// as the size of the file.

std::variant<absl::Status, int64_t> TryReadFullEOCD(riegeli::Reader& reader,
                                                    ZipEOCD& eocd,
                                                    int64_t offset_adjustment) {
  // Try and find the EOCD, which should exist in all ZIP files.
  if (!internal::FindLast(
          reader, std::string_view(reinterpret_cast<const char*>(kEOCDLiteral),
                                   sizeof(kEOCDLiteral)))) {
    return absl::InvalidArgumentError("Failed to find valid ZIP EOCD");
  }

  int64_t eocd_start = reader.pos();
  ZipEOCD last_eocd{};
  TENSORSTORE_RETURN_IF_ERROR(ReadEOCD(reader, last_eocd));

  int64_t locator_start = eocd_start - ZipEOCD64Locator::kRecordSize;
  bool has_locator = false;
  if (eocd_start >= ZipEOCD64Locator::kRecordSize) {
    if (reader.Seek(locator_start)) {
      uint32_t signature;
      if (ReadLittleEndian<uint32_t>(reader, signature) &&
          signature == kEOCD64LocatorSignature) {
        has_locator = true;
      }
    }
  }

  if (last_eocd.cd_offset != std::numeric_limits<uint32_t>::max() &&
      !has_locator) {
    // Not a ZIP64 archive.
    eocd = last_eocd;
    reader.Seek(eocd_start + 4);
    return absl::OkStatus();
  }

  // Otherwise there must be an EOCD64. First, attempt to read the record
  // locator which must be located immediately prior to the EOCD. The reader
  // block request size will include the necessary extra 20 bytes.
  if (eocd_start < ZipEOCD64Locator::kRecordSize) {
    return absl::InvalidArgumentError("Block does not contain EOCD64 Locator");
  }

  if (!reader.Seek(locator_start)) {
    if (!reader.ok()) {
      return StatusBuilder(reader.status())
          .Format("Failed to read EOCD64 Locator");
    }
    return absl::InvalidArgumentError("Failed to read EOCD64 Locator");
  }

  ZipEOCD64Locator locator;
  TENSORSTORE_RETURN_IF_ERROR(ReadEOCD64Locator(reader, locator));

  if (offset_adjustment < 0) {
    // It's possible to seek for the location of the EOCD64, but that's
    // messy. Instead return the position; the caller needs to provide
    // a better location.
    return locator.cd_offset;
  }

  // The offset of the reader with respect to the file is known
  // (and maybe the whole file), so seeking to the EOCD64 is computable.
  auto target_pos = locator.cd_offset - offset_adjustment;
  if (target_pos < 0) {
    // RETRY: Read cd_offset should be available.
    assert(offset_adjustment > 0);
    return locator.cd_offset;
  }
  if (!reader.Seek(target_pos)) {
    if (!reader.ok()) {
      return StatusBuilder(reader.status()).Format("Failed to read EOCD64");
    }
    return absl::InvalidArgumentError("Failed to read EOCD64");
  }

  TENSORSTORE_RETURN_IF_ERROR(ReadEOCD64(reader, last_eocd));
  if (reader.pos() != locator_start) {
    return absl::InvalidArgumentError("Inconsistent ZIP64 EOCD locator offset");
  }
  eocd = last_eocd;
  reader.Seek(eocd_start + 4);
  return absl::OkStatus();
}

// --------------------------------------------------------------------------

// 4.3.12
absl::Status ReadCentralDirectoryEntry(riegeli::Reader& reader,
                                       ZipEntry& entry) {
  if (!reader.Pull(ZipEntry::kCentralRecordSize)) {
    return absl::InvalidArgumentError(
        "ZIP Central Directory Entry insufficient data available");
  }

  uint32_t signature;
  ReadLittleEndian<uint32_t>(reader, signature);
  if (signature != kCentralHeaderSignature) {
    return absl::InvalidArgumentError(
        "Failed to read ZIP Central Directory Entry signature");
  }

  uint32_t uncompressed_size = 0;
  uint32_t compressed_size;
  uint32_t relative_header_offset = 0;
  uint16_t file_name_length = 0;
  uint16_t extra_field_length = 0;
  uint16_t file_comment_length = 0;
  MSDOSTime last_mod;
  uint16_t ignored16;
  uint16_t compression_method;
  ReadLittleEndian<uint16_t>(reader, entry.version_madeby);
  ReadLittleEndian<uint16_t>(reader, ignored16);  // version needed
  ReadLittleEndian<uint16_t>(reader, entry.flags);
  ReadLittleEndian<uint16_t>(reader, compression_method);
  ReadLittleEndian<uint16_t>(reader, last_mod.time);
  ReadLittleEndian<uint16_t>(reader, last_mod.date);
  ReadLittleEndian<uint32_t>(reader, entry.crc);
  ReadLittleEndian<uint32_t>(reader, compressed_size);
  ReadLittleEndian<uint32_t>(reader, uncompressed_size);
  ReadLittleEndian<uint16_t>(reader, file_name_length);
  ReadLittleEndian<uint16_t>(reader, extra_field_length);
  ReadLittleEndian<uint16_t>(reader, file_comment_length);
  ReadLittleEndian<uint16_t>(reader, ignored16);  // start disk_number
  ReadLittleEndian<uint16_t>(reader, entry.internal_fa);
  ReadLittleEndian<uint32_t>(reader, entry.external_fa);
  ReadLittleEndian<uint32_t>(reader, relative_header_offset);

  entry.compressed_size = compressed_size;
  entry.uncompressed_size = uncompressed_size;
  entry.local_header_offset = relative_header_offset;
  entry.mtime = MakeMSDOSTime(last_mod);
  entry.compression_method = static_cast<ZipCompression>(compression_method);
  entry.extra_field_length = extra_field_length;

  if (file_name_length > 0 && !reader.Read(file_name_length, entry.filename)) {
    return absl::InvalidArgumentError(
        "Failed to read ZIP Central Directory Entry (filename)");
  }
  assert(entry.filename.size() == file_name_length);

  // Read extra field.
  if (extra_field_length > 0) {
    if (extra_field_length <= 4) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Invalid ZIP Central Directory Entry extra field length %d",
          extra_field_length));
    }
    riegeli::LimitingReader extra_reader(
        &reader, riegeli::LimitingReaderBase::Options().set_exact_length(
                     extra_field_length));
    extra_reader.SetReadAllHint(true);
    if (auto status = ReadExtraField(extra_reader, entry); !status.ok()) {
      return status;
    }
    extra_reader.Seek(extra_reader.max_pos());
  }

  // Read central directory file comment.
  if (file_comment_length > 0 &&
      !reader.Read(file_comment_length, entry.comment)) {
    return absl::InvalidArgumentError(
        "Failed to read ZIP Central Directory Entry (comment)");
  }

  entry.end_of_header_offset = reader.pos();
  entry.estimated_read_size =
      std::max(entry.compressed_size, entry.uncompressed_size) +
      file_name_length + extra_field_length + ZipEntry::kLocalRecordSize +
      /*data descriptor size*/
      (HasZipGeneralFlag(entry.flags, ZipGeneralFlags::kHasDataDescriptor)
           ? kDataDescriptorEstimatedSize
           : 0);

  return absl::OkStatus();
}

// 4.3.7
absl::Status ReadLocalEntry(riegeli::Reader& reader, ZipEntry& entry) {
  if (!reader.Pull(ZipEntry::kLocalRecordSize)) {
    return absl::InvalidArgumentError(
        "ZIP Local Entry insufficient data available");
  }

  uint32_t signature;
  ReadLittleEndian<uint32_t>(reader, signature);
  if (signature != kLocalHeaderSignature) {
    return absl::InvalidArgumentError(
        "Failed to read ZIP Local Entry signature");
  }
  uint16_t ignored16;
  uint16_t compression_method;
  MSDOSTime last_mod;
  uint32_t uncompressed_size;
  uint32_t compressed_size;
  uint16_t file_name_length = 0;
  uint16_t extra_field_length = 0;
  uint32_t crc = 0;
  ReadLittleEndian<uint16_t>(reader, ignored16);  // version needed
  ReadLittleEndian<uint16_t>(reader, entry.flags);
  ReadLittleEndian<uint16_t>(reader, compression_method);
  ReadLittleEndian<uint16_t>(reader, last_mod.time);
  ReadLittleEndian<uint16_t>(reader, last_mod.date);
  ReadLittleEndian<uint32_t>(reader, crc);
  ReadLittleEndian<uint32_t>(reader, compressed_size);
  ReadLittleEndian<uint32_t>(reader, uncompressed_size);
  ReadLittleEndian<uint16_t>(reader, file_name_length);
  ReadLittleEndian<uint16_t>(reader, extra_field_length);

  entry.version_madeby = 0;
  entry.internal_fa = 0;
  entry.external_fa = 0;
  entry.local_header_offset = 0;
  entry.estimated_read_size = 0;
  bool has_data_descriptor =
      HasZipGeneralFlag(entry.flags, ZipGeneralFlags::kHasDataDescriptor);
  if (!has_data_descriptor) {
    entry.compressed_size = compressed_size;
    entry.uncompressed_size = uncompressed_size;
    entry.crc = crc;
  } else {
    if (compressed_size != 0 || entry.compressed_size == 0) {
      entry.compressed_size = compressed_size;
    }
    if (uncompressed_size != 0 || entry.uncompressed_size == 0) {
      entry.uncompressed_size = uncompressed_size;
    }
    if (crc != 0 || entry.crc == 0) {
      entry.crc = crc;
    }
  }
  entry.mtime = MakeMSDOSTime(last_mod);
  entry.compression_method = static_cast<ZipCompression>(compression_method);
  entry.extra_field_length = extra_field_length;

  if (file_name_length > 0 && !reader.Read(file_name_length, entry.filename)) {
    return absl::InvalidArgumentError(
        "Failed to read ZIP Local Entry (filename)");
  }
  assert(entry.filename.size() == file_name_length);
  entry.end_of_header_offset = reader.pos() + extra_field_length;

  // Read extra field.
  if (extra_field_length > 0) {
    if (extra_field_length <= 4) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Invalid ZIP Local Entry extra field length %d", extra_field_length));
    }
    riegeli::LimitingReader extra_reader(
        &reader, riegeli::LimitingReaderBase::Options().set_exact_length(
                     extra_field_length));
    extra_reader.SetReadAllHint(true);
    if (auto status = ReadExtraField(extra_reader, entry); !status.ok()) {
      return status;
    }
    extra_reader.Seek(extra_reader.max_pos());
  }

  return absl::OkStatus();
}

// Returns whether the ZIP entry can be read.
absl::Status ValidateEntryIsSupported(const ZipEntry& entry) {
  if (HasZipGeneralFlag(entry.flags, ZipGeneralFlags::kEncrypted) ||
      HasZipGeneralFlag(entry.flags, ZipGeneralFlags::kStrongEncryption) ||
      HasZipGeneralFlag(entry.flags, ZipGeneralFlags::kHeaderEncryption) ||
      entry.compression_method == ZipCompression::kAes) {
    return absl::InvalidArgumentError("ZIP encryption is not supported");
  }
  if (entry.compression_method != ZipCompression::kStore &&
      entry.compression_method != ZipCompression::kDeflate &&
      entry.compression_method != ZipCompression::kBzip2 &&
      entry.compression_method != ZipCompression::kZStd &&
      entry.compression_method != ZipCompression::kXZ) {
    return absl::InvalidArgumentError(
        absl::StrFormat("ZIP compression method %d is not supported",
                        static_cast<int>(entry.compression_method)));
  }
  if (absl::EndsWith(entry.filename, "/")) {
    return absl::InvalidArgumentError("ZIP directory entries cannot be read");
  }
  if (absl::StartsWith(entry.filename, "/")) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "ZIP entry filename cannot be absolute: %s", entry.filename));
  }
  std::vector<std::string_view> components =
      absl::StrSplit(entry.filename, absl::ByAnyChar("/\\"));
  for (auto component : components) {
    if (component == "..") {
      return absl::InvalidArgumentError(absl::StrFormat(
          "ZIP entry filename contains path traversal component: %s",
          entry.filename));
    }
  }

  return ValidateEntrySizes(entry.uncompressed_size, entry.compressed_size);
}

tensorstore::Result<std::unique_ptr<riegeli::Reader>> GetRawReader(
    riegeli::Reader* reader, ZipEntry& entry) {
  assert(reader != nullptr);

  // reader position should be at the beginning of the compressed file data.
  // entry.flags indicates whether the actual sizes are stored in a
  // ZIP Data Descriptor, which follows the compressed data. If so, that
  // needs to be read.
  if (HasZipGeneralFlag(entry.flags, ZipGeneralFlags::kHasDataDescriptor)) {
    const auto start_pos = reader->pos();
    if (!reader->Skip(entry.compressed_size)) {
      return reader->status();
    }

    // There are 8 bytes of guaranteed data; then there is a variable length
    // section depending on whether the entry is a ZIP or ZIP64.
    if (!reader->Pull(entry.is_zip64 ? kZip64DataDescriptorSize
                                     : kZipDataDescriptorSize)) {
      return absl::DataLossError("Failed to read ZIP DataDescriptor");
    }

    // 4.3.9  Data descriptor
    uint32_t signature, crc32;
    ReadLittleEndian<uint32_t>(*reader, signature);
    ReadLittleEndian<uint32_t>(*reader, crc32);
    if (signature != kDataDescriptorSignature) {
      return absl::DataLossError(absl::StrFormat(
          "Failed to read ZIP DataDescriptor signature %08x", signature));
    }
    if (entry.crc == 0) {
      entry.crc = crc32;
    }
    if (entry.is_zip64) {
      uint64_t compressed_size, uncompressed_size;
      ReadLittleEndian<uint64_t>(*reader, compressed_size);
      ReadLittleEndian<uint64_t>(*reader, uncompressed_size);
      if (entry.compressed_size == 0) {
        entry.compressed_size = compressed_size;
      }
      if (entry.uncompressed_size == 0) {
        entry.uncompressed_size = uncompressed_size;
      }
    } else {
      uint32_t compressed_size, uncompressed_size;
      ReadLittleEndian<uint32_t>(*reader, compressed_size);
      ReadLittleEndian<uint32_t>(*reader, uncompressed_size);
      if (entry.compressed_size == 0) {
        entry.compressed_size = compressed_size;
      }
      if (entry.uncompressed_size == 0) {
        entry.uncompressed_size = uncompressed_size;
      }
    }
    TENSORSTORE_RETURN_IF_ERROR(
        ValidateEntrySizes(entry.uncompressed_size, entry.compressed_size));
    if (!reader->Seek(start_pos)) {
      return reader->status();
    }
  }

  using Reader = riegeli::LimitingReader<riegeli::Reader*>;
  return std::make_unique<Reader>(
      reader, riegeli::LimitingReaderBase::Options().set_exact_length(
                  entry.compressed_size));
}

tensorstore::Result<std::unique_ptr<riegeli::Reader>> GetReader(
    riegeli::Reader* reader, ZipEntry& entry) {
  TENSORSTORE_ASSIGN_OR_RETURN(std::unique_ptr<riegeli::Reader> base_reader,
                               GetRawReader(reader, entry));

  // TODO: consider wrapping in a crc32 digesting reader?
  switch (entry.compression_method) {
    case ZipCompression::kStore: {
      using PLReader =
          riegeli::PrefixLimitingReader<std::unique_ptr<riegeli::Reader>>;
      return std::make_unique<PLReader>(
          std::move(base_reader),
          PLReader::Options().set_base_pos(reader->pos()));
    }
    case ZipCompression::kDeflate: {
      using DeflateReader =
          riegeli::ZlibReader<std::unique_ptr<riegeli::Reader>>;
      return std::make_unique<DeflateReader>(
          std::move(base_reader),
          DeflateReader::Options().set_header(DeflateReader::Header::kRaw));
    }
    case ZipCompression::kBzip2: {
      using Bzip2Reader =
          riegeli::Bzip2Reader<std::unique_ptr<riegeli::Reader>>;
      return std::make_unique<Bzip2Reader>(std::move(base_reader));
    }
    case ZipCompression::kZStd: {
      using ZStdReader = riegeli::ZstdReader<std::unique_ptr<riegeli::Reader>>;
      return std::make_unique<ZStdReader>(std::move(base_reader));
    }
    case ZipCompression::kXZ: {
      using XzReader = riegeli::XzReader<std::unique_ptr<riegeli::Reader>>;
      return std::make_unique<XzReader>(
          std::move(base_reader), XzReader::Options()
                                      .set_container(XzReader::Container::kXz)
                                      .set_concatenate(true));
    }
    // case ZipCompression::kLZMA:
    // To unpack ZIP LZMA we need a modified stream header and the ability to
    // set the stream format to "lzma alone", which doesn't exist in riegeli.
    // See, for example, how libzip handles the format:
    // https://github.com/nih-at/libzip/blob/main/lib/zip_algorithm_xz.c
    default:
      break;
  }
  return absl::InvalidArgumentError(absl::StrFormat(
      "Unsupported ZIP compression method %d", entry.compression_method));
}

tensorstore::Result<absl::Cord> CompressWithMethod(const absl::Cord& data,
                                                   ZipCompression method) {
  absl::Cord compressed;
  riegeli::CordWriter writer(&compressed);
  switch (method) {
    case ZipCompression::kStore:
      return data;
    case ZipCompression::kDeflate: {
      using DeflateWriter = riegeli::ZlibWriter<riegeli::Writer*>;
      DeflateWriter deflate_writer(&writer, DeflateWriter::Options().set_header(
                                                DeflateWriter::Header::kRaw));
      if (!deflate_writer.Write(data) || !deflate_writer.Close()) {
        return deflate_writer.status();
      }
      break;
    }
    case ZipCompression::kZStd: {
      using ZstdWriter = riegeli::ZstdWriter<riegeli::Writer*>;
      ZstdWriter zstd_writer(&writer);
      if (!zstd_writer.Write(data) || !zstd_writer.Close()) {
        return zstd_writer.status();
      }
      break;
    }
    case ZipCompression::kXZ: {
      using XzWriter = riegeli::XzWriter<riegeli::Writer*>;
      XzWriter xz_writer(
          &writer, XzWriter::Options().set_container(XzWriter::Container::kXz));
      if (!xz_writer.Write(data) || !xz_writer.Close()) {
        return xz_writer.status();
      }
      break;
    }
    default:
      return absl::InvalidArgumentError("Unsupported compression method");
  }
  if (!writer.Close()) {
    return writer.status();
  }
  return compressed;
}

tensorstore::Result<CompressionResult> Compress(
    const absl::Cord& data, tensorstore::span<const ZipCompression> methods) {
  if (methods.empty()) {
    return absl::InvalidArgumentError("No compression methods specified");
  }
  CompressionResult best_result;
  bool first = true;
  for (ZipCompression method : methods) {
    auto result = CompressWithMethod(data, method);
    if (!result.ok()) {
      return result.status();
    }
    if (first || result->size() < best_result.data.size()) {
      best_result.data = std::move(*result);
      best_result.method = method;
      first = false;
    }
  }
  return best_result;
}

absl::Status WriteLocalEntry(riegeli::Writer& writer, ZipEntry& entry) {
  if (entry.filename.size() > std::numeric_limits<uint16_t>::max()) {
    return absl::InvalidArgumentError("Filename too long");
  }
  // Automatically set the language encoding flag if the filename is not ASCII.
  if (!HasZipGeneralFlag(entry.flags, ZipGeneralFlags::kLanguageEncoding) &&
      !ValidateAsciiString(entry.filename).ok()) {
    entry.flags |= static_cast<uint16_t>(ZipGeneralFlags::kLanguageEncoding);
  }

  uint32_t uncompressed_size_header =
      entry.uncompressed_size >= std::numeric_limits<uint32_t>::max()
          ? std::numeric_limits<uint32_t>::max()
          : entry.uncompressed_size;
  uint32_t compressed_size_header =
      entry.compressed_size >= std::numeric_limits<uint32_t>::max()
          ? std::numeric_limits<uint32_t>::max()
          : entry.compressed_size;
  uint16_t extra_field_length = 0;
  uint16_t zip64_body_size = Zip64ExtraFieldBodySize(entry, /*is_local=*/true);
  if (zip64_body_size > 0) {
    extra_field_length += 4 + zip64_body_size;
    entry.is_zip64 = true;
  }
  extra_field_length += 36;  // NTFS extra field
  uint16_t unix_body_size =
      UnixExtendedTimestampExtraFieldBodySize(entry, /*is_local=*/true);
  if (unix_body_size > 0) {
    extra_field_length += 4 + unix_body_size;
  }

  WriteLittleEndian<uint32_t>(kLocalHeaderSignature, writer);  // Signature
  WriteLittleEndian<uint16_t>(
      zip64_body_size > 0 ? kVersionZip64 : kVersionDefault,
      writer);  // Version needed
  WriteLittleEndian<uint16_t>(entry.flags, writer);
  WriteLittleEndian<uint16_t>(static_cast<uint16_t>(entry.compression_method),
                              writer);

  auto [date, time] = ValueToMSDOSTime(entry.mtime);
  WriteLittleEndian<uint16_t>(time, writer);
  WriteLittleEndian<uint16_t>(date, writer);

  WriteLittleEndian<uint32_t>(entry.crc, writer);
  WriteLittleEndian<uint32_t>(compressed_size_header, writer);
  WriteLittleEndian<uint32_t>(uncompressed_size_header, writer);

  WriteLittleEndian<uint16_t>(static_cast<uint16_t>(entry.filename.size()),
                              writer);
  WriteLittleEndian<uint16_t>(extra_field_length, writer);

  writer.Write(entry.filename);

  TENSORSTORE_RETURN_IF_ERROR(
      WriteExtraFields(writer, entry, /*is_local=*/true));

  if (!writer.ok()) {
    return writer.status();
  }
  entry.end_of_header_offset = writer.pos();
  return absl::OkStatus();
}

absl::Status WriteCentralDirectoryEntry(riegeli::Writer& writer,
                                        ZipEntry& entry) {
  if (!HasZipGeneralFlag(entry.flags, ZipGeneralFlags::kLanguageEncoding)) {
    TENSORSTORE_RETURN_IF_ERROR(ValidateAsciiString(entry.filename));
  }
  if (entry.filename.size() > std::numeric_limits<uint16_t>::max()) {
    return absl::InvalidArgumentError("Filename too long");
  }
  if (entry.comment.size() > std::numeric_limits<uint16_t>::max()) {
    return absl::InvalidArgumentError("Comment too long");
  }

  uint32_t uncompressed_size_header =
      entry.uncompressed_size >= std::numeric_limits<uint32_t>::max()
          ? std::numeric_limits<uint32_t>::max()
          : entry.uncompressed_size;
  uint32_t compressed_size_header =
      entry.compressed_size >= std::numeric_limits<uint32_t>::max()
          ? std::numeric_limits<uint32_t>::max()
          : entry.compressed_size;
  uint32_t local_header_offset_header =
      entry.local_header_offset >= std::numeric_limits<uint32_t>::max()
          ? std::numeric_limits<uint32_t>::max()
          : entry.local_header_offset;
  uint16_t extra_field_length = 0;
  uint16_t zip64_body_size = Zip64ExtraFieldBodySize(entry, /*is_local=*/false);
  if (zip64_body_size > 0) {
    extra_field_length += 4 + zip64_body_size;
    entry.is_zip64 = true;
  }
  extra_field_length += 36;  // NTFS extra field
  uint16_t unix_body_size =
      UnixExtendedTimestampExtraFieldBodySize(entry, /*is_local=*/false);
  if (unix_body_size > 0) {
    extra_field_length += 4 + unix_body_size;
  }

  WriteLittleEndian<uint32_t>(kCentralHeaderSignature, writer);  // Signature
  WriteLittleEndian<uint16_t>(entry.version_madeby, writer);
  WriteLittleEndian<uint16_t>(
      zip64_body_size > 0 ? kVersionZip64 : kVersionDefault,
      writer);  // Version needed
  WriteLittleEndian<uint16_t>(entry.flags, writer);
  WriteLittleEndian<uint16_t>(static_cast<uint16_t>(entry.compression_method),
                              writer);

  auto [date, time] = ValueToMSDOSTime(entry.mtime);
  WriteLittleEndian<uint16_t>(time, writer);
  WriteLittleEndian<uint16_t>(date, writer);

  WriteLittleEndian<uint32_t>(entry.crc, writer);
  WriteLittleEndian<uint32_t>(compressed_size_header, writer);
  WriteLittleEndian<uint32_t>(uncompressed_size_header, writer);

  WriteLittleEndian<uint16_t>(static_cast<uint16_t>(entry.filename.size()),
                              writer);
  WriteLittleEndian<uint16_t>(extra_field_length, writer);
  WriteLittleEndian<uint16_t>(static_cast<uint16_t>(entry.comment.size()),
                              writer);
  WriteLittleEndian<uint16_t>(0, writer);  // Disk number start
  WriteLittleEndian<uint16_t>(entry.internal_fa, writer);
  WriteLittleEndian<uint32_t>(entry.external_fa, writer);
  WriteLittleEndian<uint32_t>(local_header_offset_header, writer);

  writer.Write(entry.filename);

  TENSORSTORE_RETURN_IF_ERROR(
      WriteExtraFields(writer, entry, /*is_local=*/false));

  writer.Write(entry.comment);

  if (!writer.ok()) {
    return writer.status();
  }
  return absl::OkStatus();
}

absl::Status WriteEOCD64(riegeli::Writer& writer, const ZipEOCD& eocd) {
  WriteLittleEndian<uint32_t>(kEOCD64Signature, writer);  // Signature
  WriteLittleEndian<uint64_t>(kEOCD64RecordSizeWithoutFixedHeader,
                              writer);                 // Size of EOCD64
  WriteLittleEndian<uint16_t>(kVersionZip64, writer);  // Version made by
  WriteLittleEndian<uint16_t>(kVersionZip64, writer);  // Version needed
  WriteLittleEndian<uint32_t>(0, writer);              // Disk number
  WriteLittleEndian<uint32_t>(0, writer);              // Disk with CD
  WriteLittleEndian<uint64_t>(eocd.num_entries,
                              writer);                    // Num entries on disk
  WriteLittleEndian<uint64_t>(eocd.num_entries, writer);  // Total num entries
  WriteLittleEndian<uint64_t>(static_cast<uint64_t>(eocd.cd_size), writer);
  WriteLittleEndian<uint64_t>(static_cast<uint64_t>(eocd.cd_offset), writer);
  if (!writer.ok()) return writer.status();
  return absl::OkStatus();
}

absl::Status WriteEOCD64Locator(riegeli::Writer& writer,
                                uint64_t zip64_eocd_offset) {
  WriteLittleEndian<uint32_t>(kEOCD64LocatorSignature, writer);  // Signature
  WriteLittleEndian<uint32_t>(0, writer);  // Disk with ZIP64 EOCD
  WriteLittleEndian<uint64_t>(zip64_eocd_offset, writer);
  WriteLittleEndian<uint32_t>(1, writer);  // Total number of disks
  if (!writer.ok()) return writer.status();
  return absl::OkStatus();
}

bool UseZip64(const ZipEntry& entry, bool is_local) {
  return Zip64ExtraFieldBodySize(entry, is_local) > 0;
}

bool UseZip64(const ZipEOCD& eocd) {
  return eocd.num_entries >= std::numeric_limits<uint16_t>::max() ||
         eocd.cd_size >= std::numeric_limits<uint32_t>::max() ||
         eocd.cd_offset >= std::numeric_limits<uint32_t>::max();
}

// Writes the End of Central Directory (EOCD) records.
absl::Status WriteEOCD(riegeli::Writer& writer, const ZipEOCD& eocd) {
  if (eocd.comment.size() > std::numeric_limits<uint16_t>::max()) {
    return absl::InvalidArgumentError("Comment too long");
  }

  bool use_zip64 = UseZip64(eocd);

  if (use_zip64) {
    // +---------------------------------------------+
    // | ZIP64 End of Central Directory Record       | (56 bytes)
    // +---------------------------------------------+
    // | ZIP64 End of Central Directory Locator      | (20 bytes)
    // +---------------------------------------------+
    // | End of Central Directory Record (Standard)  | (22 bytes + comment)
    // | ... comment                                 |
    // +---------------------------------------------+
    uint64_t zip64_eocd_offset = writer.pos();
    TENSORSTORE_RETURN_IF_ERROR(WriteEOCD64(writer, eocd));
    TENSORSTORE_RETURN_IF_ERROR(WriteEOCD64Locator(writer, zip64_eocd_offset));

    // 3. Write Standard EOCD placeholders
    WriteLittleEndian<uint32_t>(kEOCDSignature, writer);  // Signature
    WriteLittleEndian<uint16_t>(0, writer);               // Disk number
    WriteLittleEndian<uint16_t>(0, writer);               // Disk with CD
    WriteLittleEndian<uint16_t>(0xFFFF, writer);
    WriteLittleEndian<uint16_t>(0xFFFF, writer);
    WriteLittleEndian<uint32_t>(0xFFFFFFFF, writer);
    WriteLittleEndian<uint32_t>(0xFFFFFFFF, writer);
  } else {
    // +---------------------------------------------+
    // | End of Central Directory Record (Standard)  | (22 bytes + comment)
    // | ... comment                                 |
    // +---------------------------------------------+
    WriteLittleEndian<uint32_t>(kEOCDSignature, writer);  // Signature
    WriteLittleEndian<uint16_t>(0, writer);               // Disk number
    WriteLittleEndian<uint16_t>(0, writer);               // Disk with CD
    WriteLittleEndian<uint16_t>(static_cast<uint16_t>(eocd.num_entries),
                                writer);
    WriteLittleEndian<uint16_t>(static_cast<uint16_t>(eocd.num_entries),
                                writer);
    WriteLittleEndian<uint32_t>(static_cast<uint32_t>(eocd.cd_size), writer);
    WriteLittleEndian<uint32_t>(static_cast<uint32_t>(eocd.cd_offset), writer);
  }

  WriteLittleEndian<uint16_t>(static_cast<uint16_t>(eocd.comment.size()),
                              writer);
  writer.Write(eocd.comment);

  if (!writer.ok()) {
    return writer.status();
  }
  return absl::OkStatus();
}

}  // namespace internal_zip
}  // namespace tensorstore
