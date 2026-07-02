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

#ifndef TENSORSTORE_INTERNAL_COMPRESSION_ZIP_DETAILS_H_
#define TENSORSTORE_INTERNAL_COMPRESSION_ZIP_DETAILS_H_

#include <stddef.h>
#include <stdint.h>

#include <memory>
#include <string>
#include <variant>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "riegeli/bytes/reader.h"
#include "riegeli/bytes/writer.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

// NOTE: Currently tensorstore does not use a third-party zip library such
// as minizip-ng, libzip, or libarchive since they don't play well with async
// operation which is the base of the kvstore abstraction. As such, a minimal
// set of features is currently supported. This may be worth revisiting if
// a reasonable async integration can be made for such a library.
//
// Based on the ZIP spec.
// https://pkware.cachefly.net/webdocs/APPNOTE/APPNOTE-6.3.10.TXT
//
// 4.3.6: ZIP file layout is
//
//  [local file header 1]
//  [encryption header 1]
//  [file data 1]
//  [data descriptor 1]
//   ...
//  [local file header n]
//  [encryption header n]
//  [file data n]
//  [data descriptor n]
//  [archive decryption header]
//  [archive extra data record]
//  [central directory header 1]
//  ...
//  [central directory header n]
//  [zip64 end of central directory record] (56 bytes)
//  [zip64 end of central directory locator] (20 bytes)
//  [end of central directory record] (22 bytes + comment)
//  .. [comment]
//
namespace tensorstore {
namespace internal_zip {

/// 4.3.1.  A ZIP file MUST contain one and only one "end of central directory
/// record". The EOCD record size is 22 bytes, and may include a comment field
/// of up to 2^16-1 bytes, so the final 64K + 24 bytes should be examined to
/// find the EOCD. If an EOCD64 is needed, the locator is 20 bytes, and appears
/// immediately prior to the EOCD.
constexpr size_t kEOCDBlockSize =
    65536 + /*EOCD*/ 22 + /*EOCD64 Locator*/ 20 + /*EOCD64*/ 56;

// 4.4.5  Compression method.
enum class ZipCompression : uint16_t {
  kStore = 0,
  kDeflate = 8,
  kBzip2 = 12,
  kLZMA = 14,  // unsupported
  kZStd = 93,
  kXZ = 95,
  kAes = 99,  // unsupported
};

// 4.3.15  Zip64 end of central directory locator
struct ZipEOCD64Locator {
  uint32_t disk_number_with_cd = 0;
  int64_t cd_offset = 0;

  static constexpr int64_t kRecordSize = 20;
};

/// Read an EOCD64 Locator record at the current reader position.
absl::Status ReadEOCD64Locator(riegeli::Reader& reader,
                               ZipEOCD64Locator& locator);

// 4.3.16 End of central directory
// 4.3.14 Zip64 end of central directory
struct ZipEOCD {
  uint64_t num_entries = 0;
  int64_t cd_size = 0;
  int64_t cd_offset = 0;  // offset from start of file.

  // for additional bookkeeping.
  uint64_t record_offset = 0;

  std::string comment;

  static constexpr int64_t kEOCDRecordSize = 22;
  static constexpr int64_t kEOCD64RecordSize = 48;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const ZipEOCD& entry) {
    absl::Format(&sink,
                 "EOCD{num_entries=%d, cd_size=%d, cd_offset=%d, "
                 "record_offset=%d, comment=\"%s\"}",
                 entry.num_entries, entry.cd_size, entry.cd_offset,
                 entry.record_offset, entry.comment);
  }
};

/// Read an EOCD at the current reader position.
absl::Status ReadEOCD(riegeli::Reader& reader, ZipEOCD& eocd);

/// Read an EOCD64 at the current reader position.
absl::Status ReadEOCD64(riegeli::Reader& reader, ZipEOCD& eocd);

/// Attempts to read a full EOCD from the provided reader, including the
/// optional EOCD64 data fields. The reader must provide either the entire file,
/// or a suffix of the file which is a minimum of kEOCDBlockSize bytes long.
///
/// According to the ZIP spec, the EOCD records should appear in this order:
///    [zip64 end of central directory record]
///    [zip64 end of central directory locator]
///    [end of central directory record]
///
/// Returns an int64_t when an offset must be read from the input.
std::variant<absl::Status, int64_t> TryReadFullEOCD(riegeli::Reader& reader,
                                                    ZipEOCD& eocd,
                                                    int64_t offset_adjustment);

/// Represents MS-DOS date and time fields (each 16 bits), which are used
/// by the ZIP format for entry modification times (with 2-second resolution).
struct MSDOSTime {
  uint16_t date;
  uint16_t time;
};

/// Converts an MSDOSTime structure to an absl::Time.
absl::Time MakeMSDOSTime(MSDOSTime dos_time);

/// Converts an absl::Time structure to MSDOSTime (truncating to 2 seconds).
MSDOSTime ValueToMSDOSTime(absl::Time time);

// 4.4.4 General purpose bit flag
enum class ZipGeneralFlags : uint16_t {
  kEncrypted = 0x0001,
  kHasDataDescriptor = 0x0008,
  kLanguageEncoding = 0x0800,
  kStrongEncryption = 0x0040,
  kHeaderEncryption = 0x2000,
};

constexpr bool HasZipGeneralFlag(uint16_t flags, ZipGeneralFlags flag) {
  return (flags & static_cast<uint16_t>(flag)) != 0;
}

// Size of a standard ZIP Data Descriptor (4-byte signature, 4-byte CRC,
// 4-byte compressed size, 4-byte uncompressed size).
constexpr int64_t kZipDataDescriptorSize = 16;

// Size of a ZIP64 Data Descriptor (4-byte signature, 4-byte CRC,
// 8-byte compressed size, 8-byte uncompressed size).
constexpr int64_t kZip64DataDescriptorSize = 24;

// 4.3.7   Local file header:
// 4.3.12  Central directory structure:
struct ZipEntry {
  uint16_t version_madeby = 0;  // central-only
  uint16_t flags = 0;
  ZipCompression compression_method = ZipCompression::kStore;
  uint32_t crc = 0;
  uint64_t compressed_size = 0;
  uint64_t uncompressed_size = 0;
  uint16_t internal_fa = 0;          // central-only
  uint32_t external_fa = 0;          // central-only
  uint64_t local_header_offset = 0;  // central-only
  uint64_t estimated_read_size = 0;  // central-only
  uint16_t extra_field_length = 0;   // central-only

  // for additional bookkeeping.
  uint64_t end_of_header_offset;

  absl::Time mtime;
  absl::Time atime;
  absl::Time ctime;

  std::string filename;
  std::string comment;
  bool is_zip64 = false;

  static constexpr int64_t kCentralRecordSize = 46;
  static constexpr int64_t kLocalRecordSize = 30;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const ZipEntry& entry) {
    absl::Format(&sink,
                 "ZipEntry{\n"
                 "  version_madeby=%v\n"
                 "  flags=%x\n"
                 "  compression_method=%v\n"
                 "  crc=%08x\n"
                 "  compressed_size=%d\n"
                 "  uncompressed_size=%d\n"
                 "  internal_fa=%x\n"
                 "  external_fa=%x\n"
                 "  local_header_offset=%v\n"
                 "  estimated_read_size=%v\n"
                 "  extra_field_length=%v\n"
                 "  mtime=%s\n"
                 "  atime=%s\n"
                 "  ctime=%s\n"
                 "  filename=\"%s\"\n"
                 "  comment=\"%s\"\n"
                 "}",
                 entry.version_madeby, entry.flags, entry.compression_method,
                 entry.crc, entry.compressed_size, entry.uncompressed_size,
                 entry.internal_fa, entry.external_fa,
                 entry.local_header_offset, entry.estimated_read_size,
                 entry.extra_field_length, absl::FormatTime(entry.mtime),
                 absl::FormatTime(entry.atime), absl::FormatTime(entry.ctime),
                 entry.filename, entry.comment);
  }
};

/// Read a ZIP Central Directory Entry at the current reader position.
absl::Status ReadCentralDirectoryEntry(riegeli::Reader& reader,
                                       ZipEntry& entry);

/// Read a ZIP Local Directory Entry at the current reader position.
absl::Status ReadLocalEntry(riegeli::Reader& reader, ZipEntry& entry);

/// Write a ZIP Local Directory Entry at the current writer position.
absl::Status WriteLocalEntry(riegeli::Writer& writer, ZipEntry& entry);

/// Write a ZIP Central Directory Entry at the current writer position.
absl::Status WriteCentralDirectoryEntry(riegeli::Writer& writer,
                                        ZipEntry& entry);

/// Write a ZIP EOCD at the current writer position.
absl::Status WriteEOCD(riegeli::Writer& writer, const ZipEOCD& eocd);

/// Write a ZIP64 EOCD Record at the current writer position.
absl::Status WriteEOCD64(riegeli::Writer& writer, const ZipEOCD& eocd);

/// Write a ZIP64 EOCD Locator at the current writer position.
absl::Status WriteEOCD64Locator(riegeli::Writer& writer,
                                uint64_t zip64_eocd_offset);

/// Returns true if the entry requires ZIP64 extra field.
bool UseZip64(const ZipEntry& entry, bool is_local);

/// Returns true if the EOCD requires ZIP64.
bool UseZip64(const ZipEOCD& eocd);

/// Returns an error when the zip entry cannot be read.
absl::Status ValidateEntryIsSupported(const ZipEntry& entry);

/// Return a riegeli::Reader for the raw entry data.
/// \pre reader is positioned after the LocalHeader, at the beginning of
///   the compressed data, which can be accomplished by calling
///   reader.Seek(entry.end_of_header_offset)
tensorstore::Result<std::unique_ptr<riegeli::Reader>> GetRawReader(
    riegeli::Reader* reader, ZipEntry& entry);

/// Return a riegeli::Reader for the decompressed entry data.
/// \pre reader is positioned after the LocalHeader, which
///   can be accomplished by calling reader.Seek(entry.end_of_header_offset)
tensorstore::Result<std::unique_ptr<riegeli::Reader>> GetReader(
    riegeli::Reader* reader, ZipEntry& entry);

/// Compress data using the specified method.
tensorstore::Result<absl::Cord> CompressWithMethod(const absl::Cord& data,
                                                   ZipCompression method);

/// Result of compression.
struct CompressionResult {
  absl::Cord data;
  ZipCompression method;
};

/// Compress data using the specified methods and selects the smallest.
tensorstore::Result<CompressionResult> Compress(
    const absl::Cord& data, tensorstore::span<const ZipCompression> methods);

}  // namespace internal_zip
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_COMPRESSION_ZIP_DETAILS_H_
