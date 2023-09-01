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
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "riegeli/bytes/reader.h"
#include "tensorstore/util/result.h"

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
//  [zip64 end of central directory record]
//  [zip64 end of central directory locator]
//  [end of central directory record]
//

namespace tensorstore {
namespace internal_zip {

/// 4.3.1.  A ZIP file MUST contain one and only one "end of central directory
/// record". The EOCD record size is 22 bytes, and may include a comment field
/// of up to 2^16-1 bytes, so the final 64K + 24 bytes should be examined to
/// find the EOCD. If an EOCD64 is needed, the locator is 20 bytes, and appears
/// immediately prior to the EOCD.
constexpr size_t kEOCDBlockSize = 65536 + 48;

constexpr const unsigned char kLocalHeaderLiteral[4] = {'P', 'K', 0x03, 0x04};
constexpr const unsigned char kCentralHeaderLiteral[4] = {'P', 'K', 0x01, 0x02};
constexpr const unsigned char kEOCDLiteral[4] = {'P', 'K', 0x05, 0x06};
constexpr const unsigned char kEOCD64LocatorLiteral[4] = {'P', 'K', 0x06, 0x07};
constexpr const unsigned char kEOCD64Literal[4] = {'P', 'K', 0x06, 0x06};
constexpr const unsigned char kDataDescriptorLiteral[4] = {'P', 'K', 0x07,
                                                           0x08};

constexpr const uint16_t kHasDataDescriptor = 0x08;

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
  uint32_t disk_number_with_cd;
  int64_t cd_offset;

  static constexpr int64_t kRecordSize = 20;
};

/// Read an EOCD64 Locator record at the current reader position.
absl::Status ReadEOCD64Locator(riegeli::Reader &reader,
                               ZipEOCD64Locator &locator);

// 4.3.16 End of central directory
// 4.3.14 Zip64 end of central directory
struct ZipEOCD {
  uint64_t num_entries;
  int64_t cd_size;
  int64_t cd_offset;  // offset from start of file.

  // for additional bookkeeping.
  uint64_t record_offset;

  std::string comment;

  static constexpr int64_t kEOCDRecordSize = 22;
  static constexpr int64_t kEOCD64RecordSize = 48;

  template <typename Sink>
  friend void AbslStringify(Sink &sink, const ZipEOCD &entry) {
    absl::Format(&sink,
                 "EOCD{num_entries=%d, cd_size=%d, cd_offset=%d, "
                 "record_offset=%d, comment=\"%s\"}",
                 entry.num_entries, entry.cd_size, entry.cd_offset,
                 entry.record_offset, entry.comment);
  }
};

/// Read an EOCD at the current reader position.
absl::Status ReadEOCD(riegeli::Reader &reader, ZipEOCD &eocd);

/// Read an EOCD64 at the current reader position.
absl::Status ReadEOCD64(riegeli::Reader &reader, ZipEOCD &eocd);

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
std::variant<absl::Status, int64_t> TryReadFullEOCD(riegeli::Reader &reader,
                                                    ZipEOCD &eocd,
                                                    int64_t offset_adjustment);

// 4.3.7   Local file header:
// 4.3.12  Central directory structure:
struct ZipEntry {
  uint16_t version_madeby;  // central-only
  uint16_t flags;
  ZipCompression compression_method;
  uint32_t crc;
  uint64_t compressed_size;
  uint64_t uncompressed_size;
  uint16_t internal_fa;          // central-only
  uint32_t external_fa;          // central-only
  uint64_t local_header_offset;  // central-only
  uint64_t estimated_read_size;  // central-only

  // for additional bookkeeping.
  uint64_t end_of_header_offset;

  absl::Time mtime;
  absl::Time atime;

  std::string filename;
  std::string comment;
  bool is_zip64 = false;

  static constexpr int64_t kCentralRecordSize = 46;
  static constexpr int64_t kLocalRecordSize = 30;

  template <typename Sink>
  friend void AbslStringify(Sink &sink, const ZipEntry &entry) {
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
                 "  mtime=%s\n"
                 "  atime=%s\n"
                 "  filename=\"%s\"\n"
                 "  comment=\"%s\"\n"
                 "}",
                 entry.version_madeby, entry.flags, entry.compression_method,
                 entry.crc, entry.compressed_size, entry.uncompressed_size,
                 entry.internal_fa, entry.external_fa,
                 entry.local_header_offset, entry.estimated_read_size,
                 absl::FormatTime(entry.mtime), absl::FormatTime(entry.atime),
                 entry.filename, entry.comment);
  }
};

/// Read a ZIP Central Directory Entry at the current reader position.
absl::Status ReadCentralDirectoryEntry(riegeli::Reader &reader,
                                       ZipEntry &entry);

/// Read a ZIP Local Directory Entry at the current reader position.
absl::Status ReadLocalEntry(riegeli::Reader &reader, ZipEntry &entry);

/// Returns an error when the zip entry cannot be read.
absl::Status ValidateEntryIsSupported(const ZipEntry &entry);

/// Return a riegeli::Reader for the raw entry data.
/// \pre reader is positioned after the LocalHeader, at the beginning of
///   the compressed data, which can be accomplished by calling
///   reader.Seek(entry.end_of_header_offset)
tensorstore::Result<std::unique_ptr<riegeli::Reader>> GetRawReader(
    riegeli::Reader *reader, ZipEntry &entry);

/// Return a riegeli::Reader for the decompressed entry data.
/// \pre reader is positioned after the LocalHeader, which
///   can be accomplished by calling reader.Seek(entry.end_of_header_offset)
tensorstore::Result<std::unique_ptr<riegeli::Reader>> GetReader(
    riegeli::Reader *reader, ZipEntry &entry);

}  // namespace internal_zip
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_COMPRESSION_ZIP_DETAILS_H_
