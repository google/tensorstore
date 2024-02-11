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

#include <stddef.h>
#include <stdint.h>

#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/flags/flag.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/time/time.h"
#include "riegeli/bytes/cord_reader.h"
#include "riegeli/bytes/fd_reader.h"
#include "riegeli/bytes/read_all.h"
#include "riegeli/bytes/reader.h"
#include "riegeli/bytes/string_reader.h"
#include "tensorstore/internal/riegeli/find.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

using ::tensorstore::internal::FindFirst;
using ::tensorstore::internal::StartsWith;
using ::tensorstore::internal_zip::kCentralHeaderLiteral;
using ::tensorstore::internal_zip::kEOCDLiteral;
using ::tensorstore::internal_zip::kLocalHeaderLiteral;
using ::tensorstore::internal_zip::ReadCentralDirectoryEntry;
using ::tensorstore::internal_zip::ReadEOCD;
using ::tensorstore::internal_zip::ReadEOCD64Locator;
using ::tensorstore::internal_zip::ReadLocalEntry;
using ::tensorstore::internal_zip::TryReadFullEOCD;
using ::tensorstore::internal_zip::ZipCompression;
using ::tensorstore::internal_zip::ZipEntry;
using ::tensorstore::internal_zip::ZipEOCD;
using ::tensorstore::internal_zip::ZipEOCD64Locator;

using ::tensorstore::internal_zip::kCentralHeaderLiteral;
using ::tensorstore::internal_zip::kEOCD64Literal;
using ::tensorstore::internal_zip::kEOCD64LocatorLiteral;
using ::tensorstore::internal_zip::kEOCDLiteral;
using ::tensorstore::internal_zip::kLocalHeaderLiteral;

ABSL_FLAG(std::string, tensorstore_test_data, "",
          "Path to internal/compression/testdata/data.zip");

namespace {

absl::Cord GetTestZipFileData() {
  ABSL_CHECK(!absl::GetFlag(FLAGS_tensorstore_test_data).empty());
  absl::Cord filedata;
  TENSORSTORE_CHECK_OK(riegeli::ReadAll(
      riegeli::FdReader(absl::GetFlag(FLAGS_tensorstore_test_data)), filedata));
  ABSL_CHECK_EQ(filedata.size(), 319482);
  return filedata;
}

// 4.3.1  A minimal zip file contains an empty EOCD record.
static constexpr unsigned char kMinimalZip[] = {
    /*signature*/ 0x50, 0x4b, 0x5, 0x6,
    /*disk*/ 0x0,       0x0,  0x0, 0x0,
    /*entries*/ 0x0,    0x0,  0x0, 0x0,
    /*size*/ 0x0,       0x0,  0x0, 0x0,
    /*offset*/ 0x0,     0x0,  0x0, 0x0,
    /*comment*/ 0x0,    0x0};

// clang-format off
static constexpr unsigned char kZip64OneEmptyFile[] = {
    // local header + data
    0x50, 0x4b, 0x03, 0x04, 0x2d, 0x00, 0x00, 0x00, 0x00, 0x00, 0x4f, 0x72,
    0x5b, 0x40, 0x07, 0xa1, 0xea, 0xdd, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0x01, 0x00, 0x14, 0x00, 0x2d, 0x01, 0x00, 0x10, 0x00, 0x02,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x61, 0x0a,
    // central header
    0x50, 0x4b, 0x01, 0x02, 0x1e, 0x03, 0x2d, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x4f, 0x72, 0x5b, 0x40, 0x07, 0xa1, 0xea, 0xdd, 0x02, 0x00, 0x00, 0x00,
    0x02, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x80, 0x11, 0x00, 0x00, 0x00, 0x00, 0x2d,
    // eocd64
    0x50, 0x4b, 0x06, 0x06, 0x2c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x1e, 0x03, 0x2d, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x2f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x35, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    // eocd64 locator
    0x50, 0x4b, 0x06, 0x07, 0x00, 0x00, 0x00, 0x00, 0x64, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
    // eocd
    0x50, 0x4b, 0x05, 0x06, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00,
    0x2f, 0x00, 0x00, 0x00, 0x35, 0x00, 0x00, 0x00, 0x00, 0x00,
};

static constexpr unsigned char kZipTest2[] = {
    // local header + data
    0x50, 0x4b, 0x03, 0x04, 0x0a, 0x00, 0x02, 0x00, 0x00, 0x00, 0xd5, 0x7d,
    0x46, 0x2f, 0xc6, 0x35, 0xb9, 0x3b, 0x05, 0x00, 0x00, 0x00, 0x05, 0x00,
    0x00, 0x00, 0x04, 0x00, 0x15, 0x00, 0x74, 0x65, 0x73, 0x74, 0x55, 0x54,
    0x09, 0x00, 0x03, 0x41, 0x72, 0x81, 0x3f, 0x41, 0x72, 0x81, 0x3f, 0x55,
    0x78, 0x04, 0x00, 0x64, 0x00, 0x14, 0x00, 0x74, 0x65, 0x73, 0x74, 0x0a,
    // local header + data
    0x50, 0x4b, 0x03, 0x04, 0x0a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7b, 0x98,
    0x2b, 0x32, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x08, 0x00, 0x15, 0x00, 0x74, 0x65, 0x73, 0x74, 0x64, 0x69,
    0x72, 0x2f, 0x55, 0x54, 0x09, 0x00, 0x03, 0x09, 0x15, 0xe4, 0x41, 0x9a,
    0x15, 0xe4, 0x41, 0x55, 0x78, 0x04, 0x00, 0xe8, 0x03, 0x64, 0x00,
    // local header + data
    0x50, 0x4b, 0x03, 0x04, 0x0a, 0x00, 0x02, 0x00, 0x00, 0x00, 0xd5, 0x7d,
    0x46, 0x2f, 0xc6, 0x35, 0xb9, 0x3b, 0x05, 0x00, 0x00, 0x00, 0x05, 0x00,
    0x00, 0x00, 0x0d, 0x00, 0x15, 0x00, 0x74, 0x65, 0x73, 0x74, 0x64, 0x69,
    0x72, 0x2f, 0x74, 0x65, 0x73, 0x74, 0x32, 0x55, 0x54, 0x09, 0x00, 0x03,
    0x41, 0x72, 0x81, 0x3f, 0x41, 0x72, 0x81, 0x3f, 0x55, 0x78, 0x04, 0x00,
    0xe8, 0x03, 0x64, 0x00, 0x74, 0x65, 0x73, 0x74, 0x0a,
    // central header
    0x50, 0x4b, 0x01, 0x02, 0x17, 0x03, 0x0a, 0x00, 0x02, 0x00, 0x00, 0x00,
    0xd5, 0x7d, 0x46, 0x2f, 0xc6, 0x35, 0xb9, 0x3b, 0x05, 0x00, 0x00, 0x00,
    0x05, 0x00, 0x00, 0x00, 0x04, 0x00, 0x0d, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0xb4, 0x81, 0x00, 0x00, 0x00, 0x00, 0x74, 0x65,
    0x73, 0x74, 0x55, 0x54, 0x05, 0x00, 0x03, 0x41, 0x72, 0x81, 0x3f, 0x55,
    0x78, 0x00, 0x00,
    // central header
    0x50, 0x4b, 0x01, 0x02, 0x17, 0x03, 0x0a, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x7b, 0x98, 0x2b, 0x32, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x08, 0x00, 0x0d, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x10, 0x00, 0xed, 0x41, 0x3c, 0x00, 0x00, 0x00, 0x74, 0x65,
    0x73, 0x74, 0x64, 0x69, 0x72, 0x2f, 0x55, 0x54, 0x05, 0x00, 0x03, 0x09,
    0x15, 0xe4, 0x41, 0x55, 0x78, 0x00, 0x00,
    // central header
    0x50, 0x4b, 0x01, 0x02, 0x17, 0x03, 0x0a, 0x00, 0x02, 0x00, 0x00, 0x00,
    0xd5, 0x7d, 0x46, 0x2f, 0xc6, 0x35, 0xb9, 0x3b, 0x05, 0x00, 0x00, 0x00,
    0x05, 0x00, 0x00, 0x00, 0x0d, 0x00, 0x0d, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0xb4, 0x81, 0x77, 0x00, 0x00, 0x00, 0x74, 0x65,
    0x73, 0x74, 0x64, 0x69, 0x72, 0x2f, 0x74, 0x65, 0x73, 0x74, 0x32, 0x55,
    0x54, 0x05, 0x00, 0x03, 0x41, 0x72, 0x81, 0x3f, 0x55, 0x78, 0x00, 0x00,
    // eocd
    0x50, 0x4b, 0x05, 0x06, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x03, 0x00,
    0xca, 0x00, 0x00, 0x00, 0xbc, 0x00, 0x00, 0x00, 0x00, 0x00,
};
// clang-format on

template <size_t N>
std::string_view StringViewOf(const unsigned char (&str)[N]) {
  return std::string_view(reinterpret_cast<const char*>(str), N);
}

TEST(ZipDetailsTest, DecodeEOCD) {
  riegeli::StringReader string_reader(StringViewOf(kMinimalZip));

  EXPECT_TRUE(FindFirst(string_reader, StringViewOf(kEOCDLiteral)));  // EOCD

  ZipEOCD eocd;
  ASSERT_THAT(ReadEOCD(string_reader, eocd), ::tensorstore::IsOk());
  EXPECT_EQ(eocd.num_entries, 0);
  EXPECT_EQ(eocd.cd_size, 0);
  EXPECT_EQ(eocd.cd_offset, 0);
}

TEST(ZipDetailsTest, ReadEOCDZip64) {
  riegeli::StringReader string_reader(StringViewOf(kZip64OneEmptyFile));

  EXPECT_TRUE(FindFirst(string_reader, StringViewOf(kEOCDLiteral)));  // EOCD

  ZipEOCD eocd;
  ASSERT_THAT(ReadEOCD(string_reader, eocd), ::tensorstore::IsOk());
  EXPECT_EQ(eocd.num_entries, 1);
  EXPECT_EQ(eocd.cd_size, 47);
  EXPECT_EQ(eocd.cd_offset, 53);
}

TEST(ZipDetailsTest, ReadEOCD6LocatorZip64) {
  riegeli::StringReader string_reader(StringViewOf(kZip64OneEmptyFile));

  EXPECT_TRUE(FindFirst(string_reader, StringViewOf(kEOCD64LocatorLiteral)));

  ZipEOCD64Locator eocd64_locator;
  ASSERT_THAT(ReadEOCD64Locator(string_reader, eocd64_locator),
              ::tensorstore::IsOk());
  EXPECT_EQ(eocd64_locator.disk_number_with_cd, 0);
  EXPECT_EQ(eocd64_locator.cd_offset, 100);
}

TEST(ZipDetailsTest, ReadEOCD64Zip64) {
  riegeli::StringReader string_reader(StringViewOf(kZip64OneEmptyFile));

  EXPECT_TRUE(FindFirst(string_reader, StringViewOf(kEOCD64Literal)));
  EXPECT_EQ(100, string_reader.pos());

  ZipEOCD eocd64;
  ASSERT_THAT(ReadEOCD64(string_reader, eocd64), ::tensorstore::IsOk());
  EXPECT_EQ(eocd64.num_entries, 1);
  EXPECT_EQ(eocd64.cd_size, 47);
  EXPECT_EQ(eocd64.cd_offset, 53);
}

TEST(ZipDetailsTest, TryReadFullEOCDZip64) {
  riegeli::StringReader string_reader(StringViewOf(kZip64OneEmptyFile));

  EXPECT_TRUE(FindFirst(string_reader, StringViewOf(kEOCD64Literal)));
  EXPECT_EQ(100, string_reader.pos());

  ZipEOCD eocd64;
  ASSERT_THAT(TryReadFullEOCD(string_reader, eocd64, 0),
              ::testing::VariantWith<absl::Status>(::tensorstore::IsOk()));
  EXPECT_EQ(eocd64.num_entries, 1);
  EXPECT_EQ(eocd64.cd_size, 47);
  EXPECT_EQ(eocd64.cd_offset, 53);
}

TEST(ZipDetailsTest, ReadCentralHeaderZip64) {
  riegeli::StringReader string_reader(StringViewOf(kZip64OneEmptyFile));

  EXPECT_TRUE(FindFirst(string_reader, StringViewOf(kCentralHeaderLiteral)));
  EXPECT_EQ(53, string_reader.pos());

  ZipEntry central_header;
  ASSERT_THAT(ReadCentralDirectoryEntry(string_reader, central_header),
              ::tensorstore::IsOk());

  EXPECT_EQ(central_header.version_madeby, 798);
  EXPECT_EQ(central_header.flags, 0);
  EXPECT_EQ(central_header.compression_method, ZipCompression::kStore);
  EXPECT_EQ(central_header.crc, 3723141383);
  EXPECT_EQ(central_header.compressed_size, 2);
  EXPECT_EQ(central_header.uncompressed_size, 2);
  EXPECT_EQ(central_header.internal_fa, 1);
  EXPECT_EQ(central_header.external_fa, 293601280);
  EXPECT_EQ(central_header.local_header_offset, 0);
  EXPECT_EQ(central_header.filename, "-");
  EXPECT_EQ(central_header.comment, "");
  EXPECT_GT(central_header.mtime, absl::UnixEpoch());
}

TEST(ZipDetailsTest, ReadLocalHeaderZip64) {
  riegeli::StringReader string_reader(
      reinterpret_cast<const char*>(kZip64OneEmptyFile),
      sizeof(kZip64OneEmptyFile));
  EXPECT_TRUE(FindFirst(string_reader, StringViewOf(kLocalHeaderLiteral)));

  ZipEntry local_header;
  ASSERT_THAT(ReadLocalEntry(string_reader, local_header),
              ::tensorstore::IsOk());

  EXPECT_EQ(local_header.version_madeby, 0);
  EXPECT_EQ(local_header.flags, 0);
  EXPECT_EQ(local_header.compression_method, ZipCompression::kStore);
  EXPECT_EQ(local_header.crc, 3723141383);
  EXPECT_EQ(local_header.compressed_size, 2);
  EXPECT_EQ(local_header.uncompressed_size, 2);
  EXPECT_EQ(local_header.internal_fa, 0);
  EXPECT_EQ(local_header.external_fa, 0);
  EXPECT_EQ(local_header.local_header_offset, 0);
  EXPECT_EQ(local_header.filename, "-");
  EXPECT_EQ(local_header.comment, "");
  EXPECT_GT(local_header.mtime, absl::UnixEpoch());
}

TEST(ZipDetailsTest, Decode) {
  riegeli::StringReader string_reader(reinterpret_cast<const char*>(kZipTest2),
                                      sizeof(kZipTest2));

  EXPECT_TRUE(FindFirst(string_reader, StringViewOf(kEOCDLiteral)));  // EOCD

  ZipEOCD eocd;
  ASSERT_THAT(ReadEOCD(string_reader, eocd), ::tensorstore::IsOk());
  EXPECT_EQ(eocd.num_entries, 3);
  EXPECT_EQ(eocd.cd_size, 202);
  EXPECT_EQ(eocd.cd_offset, 188);

  string_reader.Seek(eocd.cd_offset);
  std::vector<ZipEntry> central_headers;
  for (size_t i = 0; i < eocd.num_entries; ++i) {
    EXPECT_TRUE(StartsWith(string_reader, StringViewOf(kCentralHeaderLiteral)))
        << i;
    ZipEntry header;
    ASSERT_THAT(ReadCentralDirectoryEntry(string_reader, header),
                ::tensorstore::IsOk());
    central_headers.push_back(std::move(header));
  }

  std::vector<ZipEntry> local_headers;
  for (const auto& header : central_headers) {
    ZipEntry local_header;
    string_reader.Seek(header.local_header_offset);
    EXPECT_TRUE(StartsWith(string_reader, StringViewOf(kLocalHeaderLiteral)));
    ASSERT_THAT(ReadLocalEntry(string_reader, local_header),
                ::tensorstore::IsOk());
    local_headers.push_back(std::move(local_header));
    absl::Cord data;
    string_reader.Read(local_headers.back().compressed_size, data);
  }

  ASSERT_THAT(local_headers.size(), 3);
  for (size_t i = 0; i < local_headers.size(); ++i) {
    EXPECT_EQ(local_headers[i].flags, central_headers[i].flags);
    EXPECT_EQ(local_headers[i].compression_method,
              central_headers[i].compression_method);
    EXPECT_EQ(local_headers[i].crc, central_headers[i].crc);
    EXPECT_EQ(local_headers[i].compressed_size,
              central_headers[i].compressed_size);
    EXPECT_EQ(local_headers[i].uncompressed_size,
              central_headers[i].uncompressed_size);
    EXPECT_EQ(local_headers[i].filename, central_headers[i].filename);
  }
}

struct ZipDirectory {
  ZipEOCD eocd;
  std::vector<ZipEntry> entries;
};

absl::Status ReadDirectory(riegeli::Reader& reader, ZipDirectory& directory) {
  int64_t initial_pos = reader.pos();
  auto response =
      tensorstore::internal_zip::TryReadFullEOCD(reader, directory.eocd, -1);
  if (std::holds_alternative<int64_t>(response)) {
    reader.Seek(initial_pos);
    response =
        tensorstore::internal_zip::TryReadFullEOCD(reader, directory.eocd, 0);
  }

  if (auto* status = std::get_if<absl::Status>(&response);
      status != nullptr && !status->ok()) {
    return std::move(*status);
  }
  if (std::holds_alternative<int64_t>(response)) {
    return absl::InternalError("ZIP incomplete");
  }

  // Attempt to read all the entries.
  reader.Seek(directory.eocd.cd_offset);
  std::vector<ZipEntry> central_headers;
  for (size_t i = 0; i < directory.eocd.num_entries; ++i) {
    ZipEntry header{};
    if (auto entry_status = ReadCentralDirectoryEntry(reader, header);
        !entry_status.ok()) {
      return entry_status;
    }
    directory.entries.push_back(std::move(header));
  }

  // The directory should be read at this point.
  return absl::OkStatus();
}

TEST(ZipDetailsTest, ReadDirectory) {
  riegeli::StringReader string_reader(reinterpret_cast<const char*>(kZipTest2),
                                      sizeof(kZipTest2));
  ZipDirectory dir;
  EXPECT_THAT(ReadDirectory(string_reader, dir), ::tensorstore::IsOk());

  std::vector<ZipEntry> local_headers;
  for (const auto& header : dir.entries) {
    ZipEntry local_header;
    string_reader.Seek(header.local_header_offset);
    EXPECT_TRUE(StartsWith(string_reader, StringViewOf(kLocalHeaderLiteral)));
    EXPECT_THAT(ReadLocalEntry(string_reader, local_header),
                ::tensorstore::IsOk());
    local_headers.push_back(std::move(local_header));
  }

  EXPECT_THAT(local_headers.size(), 3);
  for (size_t i = 0; i < local_headers.size(); ++i) {
    EXPECT_EQ(local_headers[i].flags, dir.entries[i].flags);
    EXPECT_EQ(local_headers[i].compression_method,
              dir.entries[i].compression_method);
    EXPECT_EQ(local_headers[i].crc, dir.entries[i].crc);
    EXPECT_EQ(local_headers[i].compressed_size, dir.entries[i].compressed_size);
    EXPECT_EQ(local_headers[i].uncompressed_size,
              dir.entries[i].uncompressed_size);
    EXPECT_EQ(local_headers[i].filename, dir.entries[i].filename);
  }

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto reader,
                                   GetReader(&string_reader, local_headers[0]));
  std::string data;
  EXPECT_THAT(riegeli::ReadAll(*reader, data), ::tensorstore::IsOk());
  EXPECT_EQ(data, "test\n");
  EXPECT_EQ(data.size(), local_headers[0].uncompressed_size);
}

/// Test specific formats.
TEST(ZipDetailsTest, Xz) {
  static constexpr unsigned char kXZ[] = {
      0x50, 0x4b, 0x03, 0x04, 0x14, 0x00, 0x00, 0x00, 0x5f, 0x00, 0x89, 0x8a,
      0x36, 0x4f, 0x28, 0xe2, 0xde, 0xa0, 0x48, 0x00, 0x00, 0x00, 0x40, 0x00,
      0x00, 0x00, 0x0f, 0x00, 0x00, 0x00, 0x61, 0x62, 0x61, 0x63, 0x2d, 0x72,
      0x65, 0x70, 0x65, 0x61, 0x74, 0x2e, 0x74, 0x78, 0x74, 0xfd, 0x37, 0x7a,
      0x58, 0x5a, 0x00, 0x00, 0x00, 0xff, 0x12, 0xd9, 0x41, 0x02, 0x00, 0x21,
      0x01, 0x00, 0x00, 0x00, 0x00, 0x37, 0x27, 0x97, 0xd6, 0xe0, 0x00, 0x3f,
      0x00, 0x11, 0x5e, 0x00, 0x30, 0xec, 0xbd, 0xa0, 0xa3, 0x19, 0xd7, 0x9c,
      0xf2, 0xec, 0x93, 0x6b, 0xfe, 0x81, 0xb3, 0x7a, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x01, 0x25, 0x40, 0x5c, 0x24, 0xa9, 0xbe, 0x06, 0x72, 0x9e,
      0x7a, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x59, 0x5a, 0x50, 0x4b, 0x01,
      0x02, 0x14, 0x00, 0x14, 0x00, 0x00, 0x00, 0x5f, 0x00, 0x89, 0x8a, 0x36,
      0x4f, 0x28, 0xe2, 0xde, 0xa0, 0x48, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00,
      0x00, 0x0f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x20,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x61, 0x62, 0x61, 0x63, 0x2d,
      0x72, 0x65, 0x70, 0x65, 0x61, 0x74, 0x2e, 0x74, 0x78, 0x74, 0x50, 0x4b,
      0x05, 0x06, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x3d, 0x00,
      0x00, 0x00, 0x75, 0x00, 0x00, 0x00, 0x00, 0x00,
  };

  riegeli::StringReader string_reader(reinterpret_cast<const char*>(kXZ),
                                      sizeof(kXZ));
  ZipDirectory dir;
  ASSERT_THAT(ReadDirectory(string_reader, dir), ::tensorstore::IsOk());
  EXPECT_THAT(dir.entries.size(), ::testing::Gt(0));

  ZipEntry local_header;
  string_reader.Seek(dir.entries[0].local_header_offset);
  EXPECT_TRUE(StartsWith(string_reader, StringViewOf(kLocalHeaderLiteral)));
  ASSERT_THAT(ReadLocalEntry(string_reader, local_header),
              ::tensorstore::IsOk());

  EXPECT_EQ(local_header.compression_method, ZipCompression::kXZ);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto reader,
                                   GetReader(&string_reader, local_header));
  std::string data;
  EXPECT_THAT(riegeli::ReadAll(*reader, data), ::tensorstore::IsOk());
  EXPECT_EQ(data,
            "aaaaaaaaaaaaaa\r\nbbbbbbbbbbbbbb\r\naaaaaaaaaaaaaa\r\ncccccccccccc"
            "cc\r\n");
  EXPECT_EQ(data.size(), local_header.uncompressed_size);
}

TEST(ZipDetailsTest, Zstd) {
  static constexpr unsigned char kZStd[] = {
      0x50, 0x4b, 0x03, 0x04, 0x3f, 0x00, 0x00, 0x00, 0x5d, 0x00, 0xa2, 0x69,
      0xf2, 0x50, 0x28, 0xe2, 0xde, 0xa0, 0x20, 0x00, 0x00, 0x00, 0x40, 0x00,
      0x00, 0x00, 0x0f, 0x00, 0x00, 0x00, 0x61, 0x62, 0x61, 0x63, 0x2d, 0x72,
      0x65, 0x70, 0x65, 0x61, 0x74, 0x2e, 0x74, 0x78, 0x74, 0x28, 0xb5, 0x2f,
      0xfd, 0x20, 0x40, 0xbd, 0x00, 0x00, 0x68, 0x61, 0x61, 0x0d, 0x0a, 0x62,
      0x0d, 0x0a, 0x61, 0x0d, 0x0a, 0x63, 0x0d, 0x0a, 0x04, 0x10, 0x00, 0xc7,
      0x38, 0xc6, 0x31, 0x38, 0x2c, 0x50, 0x4b, 0x01, 0x02, 0x3f, 0x00, 0x3f,
      0x00, 0x00, 0x00, 0x5d, 0x00, 0xa2, 0x69, 0xf2, 0x50, 0x28, 0xe2, 0xde,
      0xa0, 0x20, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x0f, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x20, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x61, 0x62, 0x61, 0x63, 0x2d, 0x72, 0x65, 0x70, 0x65,
      0x61, 0x74, 0x2e, 0x74, 0x78, 0x74, 0x50, 0x4b, 0x05, 0x06, 0x00, 0x00,
      0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x3d, 0x00, 0x00, 0x00, 0x4d, 0x00,
      0x00, 0x00, 0x00, 0x00,
  };

  riegeli::StringReader string_reader(StringViewOf(kZStd));
  ZipDirectory dir;
  ASSERT_THAT(ReadDirectory(string_reader, dir), ::tensorstore::IsOk());
  EXPECT_THAT(dir.entries.size(), ::testing::Gt(0));

  ZipEntry local_header;
  string_reader.Seek(dir.entries[0].local_header_offset);
  EXPECT_TRUE(StartsWith(string_reader, StringViewOf(kLocalHeaderLiteral)));
  ASSERT_THAT(ReadLocalEntry(string_reader, local_header),
              ::tensorstore::IsOk());

  EXPECT_EQ(local_header.compression_method, ZipCompression::kZStd);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto reader,
                                   GetReader(&string_reader, local_header));
  std::string data;
  EXPECT_THAT(riegeli::ReadAll(*reader, data), ::tensorstore::IsOk());
  EXPECT_EQ(data,
            "aaaaaaaaaaaaaa\r\nbbbbbbbbbbbbbb\r\naaaaaaaaaaaaaa\r\ncccccccccccc"
            "cc\r\n");
  EXPECT_EQ(data.size(), local_header.uncompressed_size);
}

TEST(ZipDetailsTest, Bzip2) {
  static constexpr unsigned char kBzip2[] = {
      0x50, 0x4b, 0x03, 0x04, 0x0a, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x54, 0x74,
      0x45, 0x3c, 0x48, 0x40, 0x35, 0xb0, 0x2f, 0x00, 0x00, 0x00, 0x3c, 0x00,
      0x00, 0x00, 0x0f, 0x00, 0x00, 0x00, 0x61, 0x62, 0x61, 0x63, 0x2d, 0x72,
      0x65, 0x70, 0x65, 0x61, 0x74, 0x2e, 0x74, 0x78, 0x74, 0x42, 0x5a, 0x68,
      0x39, 0x31, 0x41, 0x59, 0x26, 0x53, 0x59, 0x03, 0x64, 0xc8, 0x04, 0x00,
      0x00, 0x07, 0x41, 0x00, 0x00, 0x10, 0x38, 0x00, 0x20, 0x00, 0x30, 0xcd,
      0x34, 0x12, 0x6a, 0x7a, 0x95, 0x10, 0x26, 0x4e, 0xcd, 0x9f, 0x17, 0x72,
      0x45, 0x38, 0x50, 0x90, 0x03, 0x64, 0xc8, 0x04, 0x50, 0x4b, 0x01, 0x02,
      0x1e, 0x03, 0x0a, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x54, 0x74, 0x45, 0x3c,
      0x48, 0x40, 0x35, 0xb0, 0x2f, 0x00, 0x00, 0x00, 0x3c, 0x00, 0x00, 0x00,
      0x0f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0xfd, 0x81, 0x00, 0x00, 0x00, 0x00, 0x61, 0x62, 0x61, 0x63, 0x2d, 0x72,
      0x65, 0x70, 0x65, 0x61, 0x74, 0x2e, 0x74, 0x78, 0x74, 0x50, 0x4b, 0x05,
      0x06, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x3d, 0x00, 0x00,
      0x00, 0x5c, 0x00, 0x00, 0x00, 0x00, 0x00,
  };

  riegeli::StringReader string_reader(StringViewOf(kBzip2));
  ZipDirectory dir;
  ASSERT_THAT(ReadDirectory(string_reader, dir), ::tensorstore::IsOk());
  EXPECT_THAT(dir.entries.size(), ::testing::Gt(0));

  ZipEntry local_header;
  string_reader.Seek(dir.entries[0].local_header_offset);
  EXPECT_TRUE(StartsWith(string_reader, StringViewOf(kLocalHeaderLiteral)));
  ASSERT_THAT(ReadLocalEntry(string_reader, local_header),
              ::tensorstore::IsOk());

  EXPECT_EQ(local_header.compression_method, ZipCompression::kBzip2);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto reader,
                                   GetReader(&string_reader, local_header));
  std::string data;
  EXPECT_THAT(riegeli::ReadAll(*reader, data), ::tensorstore::IsOk());
  EXPECT_EQ(data,
            "aaaaaaaaaaaaaa\nbbbbbbbbbbbbbb\naaaaaaaaaaaaaa\ncccccccccccccc\n");
  EXPECT_EQ(data.size(), local_header.uncompressed_size);
}

TEST(ZipDetailsTest, Deflate) {
  static constexpr unsigned char kDeflate[] = {
      0x50, 0x4b, 0x03, 0x04, 0x14, 0x00, 0x00, 0x00, 0x08, 0x00, 0x56, 0x5e,
      0x9c, 0x40, 0xb0, 0x91, 0x01, 0x58, 0x12, 0x00, 0x00, 0x00, 0x13, 0x00,
      0x00, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x66, 0x69, 0x72, 0x73, 0x74, 0x73,
      0x65, 0x63, 0x6f, 0x6e, 0x64, 0x4b, 0xcb, 0x2c, 0x2a, 0x2e, 0x29, 0x48,
      0x2c, 0x2a, 0x29, 0x4e, 0x4d, 0xce, 0xcf, 0x4b, 0x01, 0xb1, 0x00, 0x50,
      0x4b, 0x01, 0x02, 0x1e, 0x03, 0x14, 0x00, 0x00, 0x00, 0x08, 0x00, 0x56,
      0x5e, 0x9c, 0x40, 0xb0, 0x91, 0x01, 0x58, 0x12, 0x00, 0x00, 0x00, 0x13,
      0x00, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
      0x00, 0x00, 0x00, 0xb4, 0x81, 0x00, 0x00, 0x00, 0x00, 0x66, 0x69, 0x72,
      0x73, 0x74, 0x73, 0x65, 0x63, 0x6f, 0x6e, 0x64, 0x50, 0x4b, 0x05, 0x06,
      0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x39, 0x00, 0x00, 0x00,
      0x3b, 0x00, 0x00, 0x00, 0x00, 0x00,
  };

  riegeli::StringReader string_reader(StringViewOf(kDeflate));
  ZipDirectory dir;
  ASSERT_THAT(ReadDirectory(string_reader, dir), ::tensorstore::IsOk());
  EXPECT_THAT(dir.entries.size(), ::testing::Gt(0));

  ZipEntry local_header;
  string_reader.Seek(dir.entries[0].local_header_offset);
  EXPECT_TRUE(StartsWith(string_reader, StringViewOf(kLocalHeaderLiteral)));
  ASSERT_THAT(ReadLocalEntry(string_reader, local_header),
              ::tensorstore::IsOk());

  EXPECT_EQ(local_header.compression_method, ZipCompression::kDeflate);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto reader,
                                   GetReader(&string_reader, local_header));
  std::string data;
  EXPECT_THAT(riegeli::ReadAll(*reader, data), ::tensorstore::IsOk());
  EXPECT_EQ(data, "firstpartsecondpart");
  EXPECT_EQ(data.size(), local_header.uncompressed_size);
}

/* zipdetails data.zip
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
00000 LOCAL HEADER #1       04034B50
00004 Extract Zip Spec      14 '2.0'
00005 Extract OS            00 'MS-DOS'
00006 General Purpose Flag  0002
      [Bits 1-2]            1 'Maximum Compression'
00008 Compression Method    0008 'Deflated'
0000A Last Mod Time         5703304B 'Thu Aug  3 06:02:22 2023'
0000E CRC                   94EE1E3E
00012 Compressed Length     00019F62
00016 Uncompressed Length   00019F6F
0001A Filename Length       000A
0001C Extra Length          001C
0001E Filename              'data/a.png'
00028 Extra ID #0001        5455 'UT: Extended Timestamp'
0002A   Length              0009
0002C   Flags               '03 mod access'
0002D   Mod Time            64CB42ED 'Thu Aug  3 06:02:21 2023'
00031   Access Time         64CB434D 'Thu Aug  3 06:03:57 2023'
00035 Extra ID #0002        7875 'ux: Unix Extra Type 3'
00037   Length              000B
00039   Version             01
0003A   UID Size            04
0003B   UID                 0000356C
0003F   GID Size            04
00040   GID                 00015F53
00044 PAYLOAD

19FA6 LOCAL HEADER #2       04034B50
...
19FEB PAYLOAD

33F4D LOCAL HEADER #3       04034B50
...
33F91 PAYLOAD

4DEF3 CENTRAL HEADER #1     02014B50
4DEF7 Created Zip Spec      1E '3.0'
4DEF8 Created OS            03 'Unix'
4DEF9 Extract Zip Spec      14 '2.0'
4DEFA Extract OS            00 'MS-DOS'
4DEFB General Purpose Flag  0002
      [Bits 1-2]            1 'Maximum Compression'
4DEFD Compression Method    0008 'Deflated'
4DEFF Last Mod Time         5703304B 'Thu Aug  3 06:02:22 2023'
4DF03 CRC                   94EE1E3E
4DF07 Compressed Length     00019F62
4DF0B Uncompressed Length   00019F6F
4DF0F Filename Length       000A
4DF11 Extra Length          0018
4DF13 Comment Length        0000
4DF15 Disk Start            0000
4DF17 Int File Attributes   0000
      [Bit 0]               0 'Binary Data'
4DF19 Ext File Attributes   81240001
      [Bit 0]               Read-Only
4DF1D Local Header Offset   00000000
4DF21 Filename              'data/a.png'
4DF2B Extra ID #0001        5455 'UT: Extended Timestamp'
4DF2D   Length              0005
4DF2F   Flags               '03 mod access'
4DF30   Mod Time            64CB42ED 'Thu Aug  3 06:02:21 2023'
4DF34 Extra ID #0002        7875 'ux: Unix Extra Type 3'
4DF36   Length              000B
4DF38   Version             01
4DF39   UID Size            04
4DF3A   UID                 0000356C
4DF3E   GID Size            04
4DF3F   GID                 00015F53

4DF43 CENTRAL HEADER #2     02014B50
...

4DF94 CENTRAL HEADER #3     02014B50
...

4DFE4 END CENTRAL HEADER    06054B50
4DFE8 Number of this disk   0000
4DFEA Central Dir Disk no   0000
4DFEC Entries in this disk  0003
4DFEE Total Entries         0003
4DFF0 Size of Central Dir   000000F1
4DFF4 Offset to Central Dir 0004DEF3
4DFF8 Comment Length        0000
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
*/

TEST(TestdataTest, HeaderPositions) {
  riegeli::CordReader reader(GetTestZipFileData());

  /* Verify that we find the starting positions of each record
  00000 LOCAL HEADER #1       04034B50
  19FA6 LOCAL HEADER #2       04034B50
  33F4D LOCAL HEADER #3       04034B50
  4DEF3 CENTRAL HEADER #1     02014B50
  4DF43 CENTRAL HEADER #2     02014B50
  4DF94 CENTRAL HEADER #3     02014B50
  4DFE4 END CENTRAL HEADER    06054B50
  */

  EXPECT_TRUE(FindFirst(reader, StringViewOf(kLocalHeaderLiteral)));
  EXPECT_TRUE(StartsWith(reader, StringViewOf(kLocalHeaderLiteral)));
  EXPECT_THAT(reader.pos(), 0);
  reader.Skip(4);

  EXPECT_TRUE(FindFirst(reader, StringViewOf(kLocalHeaderLiteral)));
  EXPECT_TRUE(StartsWith(reader, StringViewOf(kLocalHeaderLiteral)));
  EXPECT_THAT(reader.pos(), 0x19FA6);
  reader.Skip(4);

  EXPECT_TRUE(FindFirst(reader, StringViewOf(kLocalHeaderLiteral)));
  EXPECT_TRUE(StartsWith(reader, StringViewOf(kLocalHeaderLiteral)));
  EXPECT_THAT(reader.pos(), 0x33F4D);
  reader.Seek(0);

  EXPECT_TRUE(FindFirst(reader, StringViewOf(kCentralHeaderLiteral)));
  EXPECT_TRUE(StartsWith(reader, StringViewOf(kCentralHeaderLiteral)));
  EXPECT_THAT(reader.pos(), 0x4DEF3);
  reader.Skip(4);

  EXPECT_TRUE(FindFirst(reader, StringViewOf(kCentralHeaderLiteral)));
  EXPECT_TRUE(StartsWith(reader, StringViewOf(kCentralHeaderLiteral)));
  EXPECT_THAT(reader.pos(), 0x4DF43);
  reader.Skip(4);

  EXPECT_TRUE(FindFirst(reader, StringViewOf(kCentralHeaderLiteral)));
  EXPECT_TRUE(StartsWith(reader, StringViewOf(kCentralHeaderLiteral)));
  EXPECT_THAT(reader.pos(), 0x4DF94);
  reader.Seek(0);

  EXPECT_TRUE(FindFirst(reader, StringViewOf(kEOCDLiteral)));
  EXPECT_TRUE(StartsWith(reader, StringViewOf(kEOCDLiteral)));
  EXPECT_THAT(reader.pos(), 0x4DFE4);
}

TEST(TestdataTest, LocalHeaderEntry) {
  riegeli::CordReader reader(GetTestZipFileData());

  ZipEntry header;

  EXPECT_TRUE(StartsWith(reader, StringViewOf(kLocalHeaderLiteral)));
  EXPECT_THAT(reader.pos(), 0);
  /// Reads a ZipFileHeader from the current stream position.
  ASSERT_THAT(ReadLocalEntry(reader, header), ::tensorstore::IsOk());

  EXPECT_THAT(header.version_madeby, 0);
  EXPECT_THAT(header.flags, 0x2);
  EXPECT_THAT(header.compression_method, ZipCompression::kDeflate);
  EXPECT_THAT(header.crc, 0x94EE1E3E);
  EXPECT_THAT(header.compressed_size, 0x00019F62);
  EXPECT_THAT(header.uncompressed_size, 0x00019F6F);
  EXPECT_THAT(header.internal_fa, 0);
  EXPECT_THAT(header.external_fa, 0);
  EXPECT_THAT(header.local_header_offset, 0);

  // for additional bookkeeping.
  EXPECT_THAT(header.end_of_header_offset, 68);
  EXPECT_THAT(header.filename, "data/a.png");
  EXPECT_THAT(header.comment, "");
  EXPECT_THAT(header.is_zip64, false);
}

TEST(TestdataTest, CentralHeaderEntry) {
  riegeli::CordReader reader(GetTestZipFileData());

  reader.Seek(0x4DEF3);
  ASSERT_TRUE(FindFirst(reader, StringViewOf(kCentralHeaderLiteral)));
  EXPECT_TRUE(StartsWith(reader, StringViewOf(kCentralHeaderLiteral)));
  EXPECT_THAT(reader.pos(), 0x4DEF3);

  ZipEntry header{};

  /// Reads a ZipFileHeader from the current stream position.
  ASSERT_THAT(ReadCentralDirectoryEntry(reader, header), ::tensorstore::IsOk());

  EXPECT_THAT(header.flags, 0x2);
  EXPECT_THAT(header.compression_method, ZipCompression::kDeflate);
  EXPECT_THAT(header.crc, 0x94EE1E3E);
  EXPECT_THAT(header.compressed_size, 0x00019F62);
  EXPECT_THAT(header.uncompressed_size, 0x00019F6F);
  EXPECT_THAT(header.local_header_offset, 0);

  // for additional bookkeeping.
  EXPECT_THAT(header.end_of_header_offset, 24);
  EXPECT_THAT(header.filename, "data/a.png");
  EXPECT_THAT(header.comment, "");
  EXPECT_THAT(header.is_zip64, false);

  // central-only
  EXPECT_THAT(header.version_madeby, 0x031E);
  EXPECT_THAT(header.internal_fa, 0);
  EXPECT_THAT(header.external_fa, 0x81240001);
  EXPECT_THAT(header.local_header_offset, 0);
  EXPECT_THAT(header.estimated_read_size, 106415);
}

TEST(TestdataTest, EOCD) {
  riegeli::CordReader reader(GetTestZipFileData());

  ASSERT_TRUE(FindFirst(reader, StringViewOf(kEOCDLiteral)));
  EXPECT_TRUE(StartsWith(reader, StringViewOf(kEOCDLiteral)));
  EXPECT_THAT(reader.pos(), 0x4DFE4);

  ::tensorstore::internal_zip::ZipEOCD eocd{};
  ASSERT_THAT(ReadEOCD(reader, eocd), ::tensorstore::IsOk());

  EXPECT_THAT(eocd.num_entries, 3);
  EXPECT_THAT(eocd.cd_size, 0x000000F1);
  EXPECT_THAT(eocd.cd_offset, 0x0004DEF3);  // offset from start of file.

  // for additional bookkeeping.
  EXPECT_THAT(eocd.comment, "");
}

TEST(TestdataTest, FileData) {
  riegeli::CordReader reader(GetTestZipFileData());

  ZipEntry header;

  /// Reads a ZipFileHeader from the current stream position.
  ASSERT_THAT(ReadLocalEntry(reader, header), ::tensorstore::IsOk());

  EXPECT_THAT(reader.pos(), 0x0044);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto entry_reader, tensorstore::internal_zip::GetReader(&reader, header));

  std::string data;
  EXPECT_THAT(riegeli::ReadAll(*entry_reader, data), ::tensorstore::IsOk());
  EXPECT_EQ(data.size(), header.uncompressed_size);
}

}  // namespace
