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

#include <array>
#include <ctime>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/flags/flag.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/time/civil_time.h"
#include "absl/time/time.h"
#include "riegeli/bytes/cord_reader.h"
#include "riegeli/bytes/cord_writer.h"
#include "riegeli/bytes/fd_reader.h"
#include "riegeli/bytes/limiting_writer.h"
#include "riegeli/bytes/read_all.h"
#include "riegeli/bytes/string_reader.h"
#include "riegeli/bytes/string_writer.h"
#include "tensorstore/internal/compression/zip_easy.h"
#include "tensorstore/internal/riegeli/find.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

using ::tensorstore::IsOk;
using ::tensorstore::internal::FindFirst;
using ::tensorstore::internal::StartsWith;
using ::tensorstore::internal_zip::Compress;
using ::tensorstore::internal_zip::EasyZipReader;
using ::tensorstore::internal_zip::EasyZipWriter;
using ::tensorstore::internal_zip::ReadCentralDirectoryEntry;
using ::tensorstore::internal_zip::ReadEOCD;
using ::tensorstore::internal_zip::ReadEOCD64Locator;
using ::tensorstore::internal_zip::ReadLocalEntry;
using ::tensorstore::internal_zip::TryReadFullEOCD;
using ::tensorstore::internal_zip::ValidateEntryIsSupported;
using ::tensorstore::internal_zip::WriteCentralDirectoryEntry;
using ::tensorstore::internal_zip::WriteEOCD;
using ::tensorstore::internal_zip::WriteLocalEntry;
using ::tensorstore::internal_zip::ZipCompression;
using ::tensorstore::internal_zip::ZipEntry;
using ::tensorstore::internal_zip::ZipEOCD;
using ::tensorstore::internal_zip::ZipEOCD64Locator;

ABSL_FLAG(std::string, tensorstore_test_data, "",
          "Path to internal/compression/testdata/data.zip");

namespace {

constexpr unsigned char kLocalHeaderLiteral[4] = {'P', 'K', 0x03, 0x04};
constexpr unsigned char kCentralHeaderLiteral[4] = {'P', 'K', 0x01, 0x02};
constexpr unsigned char kEOCDLiteral[4] = {'P', 'K', 0x05, 0x06};
constexpr unsigned char kEOCD64LocatorLiteral[4] = {'P', 'K', 0x06, 0x07};
constexpr unsigned char kEOCD64Literal[4] = {'P', 'K', 0x06, 0x06};

// Returns the data inside "testdata/data.zip".
//
// zipinfo output:
// Archive:  testdata/data.zip
// Zip file size: 319482 bytes, number of entries: 3
// -r--r--r--  3.0 unx   106351 bx defX 23-Aug-03 06:02 data/a.png
// -r--r--r--  3.0 unx   106351 bx defX 23-Aug-03 06:03 data/bb.png
// -r--r--r--  3.0 unx   106351 bx defX 23-Aug-03 06:03 data/c.png
// 3 files, 319053 bytes uncompressed, 319014 bytes compressed:  0.0%
absl::Cord GetTestZipFileData() {
  ABSL_CHECK(!absl::GetFlag(FLAGS_tensorstore_test_data).empty());
  absl::Cord filedata;
  TENSORSTORE_CHECK_OK(riegeli::ReadAll(
      riegeli::FdReader(absl::GetFlag(FLAGS_tensorstore_test_data)), filedata));
  ABSL_CHECK_EQ(filedata.size(), 319482);
  return filedata;
}

// Returns the path to any named test zip file.
std::string GetTestZipPath(std::string_view filename) {
  std::string base_path = absl::GetFlag(FLAGS_tensorstore_test_data);
  size_t last_slash = base_path.find_last_of('/');
  if (last_slash != std::string::npos) {
    return base_path.substr(0, last_slash + 1) + std::string(filename);
  }
  return std::string(filename);
}

// Returns the path to "data_descriptor_store_compression.zip".
std::string GetTestDescriptorZipPath() {
  return GetTestZipPath("data_descriptor_store_compression.zip");
}

absl::Cord GetZip64OneEmptyFileData() {
  absl::Cord filedata;
  TENSORSTORE_CHECK_OK(riegeli::ReadAll(
      riegeli::FdReader(GetTestZipPath("zip64_one_empty_file.zip")), filedata));
  return filedata;
}

absl::Cord GetZipTest2Data() {
  absl::Cord filedata;
  TENSORSTORE_CHECK_OK(riegeli::ReadAll(
      riegeli::FdReader(GetTestZipPath("zip_test2.zip")), filedata));
  return filedata;
}

absl::Cord GetZipDeflateData() {
  absl::Cord filedata;
  TENSORSTORE_CHECK_OK(riegeli::ReadAll(
      riegeli::FdReader(GetTestZipPath("zip_deflate.zip")), filedata));
  return filedata;
}

absl::Cord GetZipBzip2Data() {
  absl::Cord filedata;
  TENSORSTORE_CHECK_OK(riegeli::ReadAll(
      riegeli::FdReader(GetTestZipPath("zip_bzip2.zip")), filedata));
  return filedata;
}

absl::Cord GetZipZstdData() {
  absl::Cord filedata;
  TENSORSTORE_CHECK_OK(riegeli::ReadAll(
      riegeli::FdReader(GetTestZipPath("zip_zstd.zip")), filedata));
  return filedata;
}

absl::Cord GetZipXzData() {
  absl::Cord filedata;
  TENSORSTORE_CHECK_OK(riegeli::ReadAll(
      riegeli::FdReader(GetTestZipPath("zip_xz.zip")), filedata));
  return filedata;
}

// Invalid EOCD zip file data
// Size: 363
static constexpr unsigned char kInvalidEocdZip[] = {
    // Local File Header #1 for "boring_file"
    0x50, 0x4b, 0x03, 0x04, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x21, 0x00, 0x4b, 0x95, 0x55, 0x77, 0x0b, 0x00, 0x00, 0x00, 0x0b, 0x00,
    0x00, 0x00, 0x0b, 0x00, 0x00, 0x00,
    // File name: "boring_file"
    0x62, 0x6f, 0x72, 0x69, 0x6e, 0x67, 0x5f, 0x66, 0x69, 0x6c, 0x65,
    // File data: "not python\n"
    0x6e, 0x6f, 0x74, 0x20, 0x70, 0x79, 0x74, 0x68, 0x6f, 0x6e, 0x0a,
    // Central Directory Header #1 for "boring_file"
    0x50, 0x4b, 0x01, 0x02, 0x14, 0x03, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x21, 0x00, 0x4b, 0x95, 0x55, 0x77, 0x0b, 0x00, 0x00, 0x00,
    0x0b, 0x00, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0xb4, 0x01, 0x00, 0x00, 0x00, 0x00,
    // File name: "boring_file"
    0x62, 0x6f, 0x72, 0x69, 0x6e, 0x67, 0x5f, 0x66, 0x69, 0x6c, 0x65,
    // Zip64 End of Central Directory Record #1
    0x50, 0x4b, 0x06, 0x06, 0x2c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x2d, 0x00, 0x2d, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x39, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x34, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    // Local File Header #2 for "py_file"
    0x50, 0x4b, 0x03, 0x04, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x21, 0x00, 0x61, 0xec, 0x85, 0x94, 0x0a, 0x00, 0x00, 0x00, 0x0a, 0x00,
    0x00, 0x00, 0x07, 0x00, 0x00, 0x00,
    // File name: "py_file"
    0x70, 0x79, 0x5f, 0x66, 0x69, 0x6c, 0x65,
    // File data: "is python\n"
    0x69, 0x73, 0x20, 0x70, 0x79, 0x74, 0x68, 0x6f, 0x6e, 0x0a,
    // Central Directory Header #2 for "py_file"
    0x50, 0x4b, 0x01, 0x02, 0x14, 0x03, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x21, 0x00, 0x61, 0xec, 0x85, 0x94, 0x0a, 0x00, 0x00, 0x00,
    0x0a, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0xb4, 0x01, 0xa5, 0x00, 0x00, 0x00,
    // File name: "py_file"
    0x70, 0x79, 0x5f, 0x66, 0x69, 0x6c, 0x65,
    // Zip64 End of Central Directory Record #2
    0x50, 0x4b, 0x06, 0x06, 0x2c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x2d, 0x00, 0x2d, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x35, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0xd4, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    // Zip64 End of Central Directory Locator
    0x50, 0x4b, 0x06, 0x07, 0x00, 0x00, 0x00, 0x00, 0x6d, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
    // End of Central Directory (EOCD)
    0x50, 0x4b, 0x05, 0x06, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00,
    0x39, 0x00, 0x00, 0x00, 0x34, 0x00, 0x00, 0x00, 0x00, 0x00};

// 4.3.1  A minimal zip file contains an empty EOCD record.
static constexpr unsigned char kMinimalZip[] = {
    /*signature*/ 0x50, 0x4b, 0x5, 0x6,
    /*disk*/ 0x0,       0x0,  0x0, 0x0,
    /*entries*/ 0x0,    0x0,  0x0, 0x0,
    /*size*/ 0x0,       0x0,  0x0, 0x0,
    /*offset*/ 0x0,     0x0,  0x0, 0x0,
    /*comment*/ 0x0,    0x0};


template <size_t N>
std::string_view StringViewOf(const unsigned char (&str)[N]) {
  return std::string_view(reinterpret_cast<const char*>(str), N);
}

TEST(ZipDetailsTest, DecodeEOCD) {
  riegeli::StringReader string_reader(StringViewOf(kMinimalZip));

  EXPECT_TRUE(FindFirst(string_reader, StringViewOf(kEOCDLiteral)));  // EOCD

  ZipEOCD eocd;
  ASSERT_THAT(ReadEOCD(string_reader, eocd), IsOk());
  EXPECT_EQ(eocd.num_entries, 0);
  EXPECT_EQ(eocd.cd_size, 0);
  EXPECT_EQ(eocd.cd_offset, 0);
}

TEST(ZipDetailsTest, ReadEOCDZip64) {
  riegeli::CordReader string_reader(GetZip64OneEmptyFileData());

  EXPECT_TRUE(FindFirst(string_reader, StringViewOf(kEOCDLiteral)));  // EOCD

  ZipEOCD eocd;
  ASSERT_THAT(ReadEOCD(string_reader, eocd), IsOk());
  EXPECT_EQ(eocd.num_entries, 1);
  EXPECT_EQ(eocd.cd_size, 47);
  EXPECT_EQ(eocd.cd_offset, 53);
}

TEST(ZipDetailsTest, ReadEOCD6LocatorZip64) {
  riegeli::CordReader string_reader(GetZip64OneEmptyFileData());

  EXPECT_TRUE(FindFirst(string_reader, StringViewOf(kEOCD64LocatorLiteral)));

  ZipEOCD64Locator eocd64_locator;
  ASSERT_THAT(ReadEOCD64Locator(string_reader, eocd64_locator), IsOk());
  EXPECT_EQ(eocd64_locator.disk_number_with_cd, 0);
  EXPECT_EQ(eocd64_locator.cd_offset, 100);
}

TEST(ZipDetailsTest, ReadEOCD64Zip64) {
  riegeli::CordReader string_reader(GetZip64OneEmptyFileData());

  EXPECT_TRUE(FindFirst(string_reader, StringViewOf(kEOCD64Literal)));
  EXPECT_EQ(100, string_reader.pos());

  ZipEOCD eocd64;
  ASSERT_THAT(ReadEOCD64(string_reader, eocd64), IsOk());
  EXPECT_EQ(eocd64.num_entries, 1);
  EXPECT_EQ(eocd64.cd_size, 47);
  EXPECT_EQ(eocd64.cd_offset, 53);
}

TEST(ZipDetailsTest, TryReadFullEOCDZip64) {
  riegeli::CordReader string_reader(GetZip64OneEmptyFileData());

  EXPECT_TRUE(FindFirst(string_reader, StringViewOf(kEOCD64Literal)));
  EXPECT_EQ(100, string_reader.pos());

  ZipEOCD eocd64;
  ASSERT_THAT(TryReadFullEOCD(string_reader, eocd64, 0),
              ::testing::VariantWith<absl::Status>(IsOk()));
  EXPECT_EQ(eocd64.num_entries, 1);
  EXPECT_EQ(eocd64.cd_size, 47);
  EXPECT_EQ(eocd64.cd_offset, 53);
}

TEST(ZipDetailsTest, ReadCentralHeaderZip64) {
  riegeli::CordReader string_reader(GetZip64OneEmptyFileData());

  EXPECT_TRUE(FindFirst(string_reader, StringViewOf(kCentralHeaderLiteral)));
  EXPECT_EQ(53, string_reader.pos());

  ZipEntry central_header;
  ASSERT_THAT(ReadCentralDirectoryEntry(string_reader, central_header), IsOk());

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
  riegeli::CordReader string_reader(GetZip64OneEmptyFileData());
  EXPECT_TRUE(FindFirst(string_reader, StringViewOf(kLocalHeaderLiteral)));

  ZipEntry local_header;
  ASSERT_THAT(ReadLocalEntry(string_reader, local_header), IsOk());

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
  riegeli::CordReader string_reader(GetZipTest2Data());

  EXPECT_TRUE(FindFirst(string_reader, StringViewOf(kEOCDLiteral)));  // EOCD

  ZipEOCD eocd;
  ASSERT_THAT(ReadEOCD(string_reader, eocd), IsOk());
  EXPECT_EQ(eocd.num_entries, 3);
  EXPECT_EQ(eocd.cd_size, 202);
  EXPECT_EQ(eocd.cd_offset, 188);

  string_reader.Seek(eocd.cd_offset);
  std::vector<ZipEntry> central_headers;
  for (size_t i = 0; i < eocd.num_entries; ++i) {
    EXPECT_TRUE(StartsWith(string_reader, StringViewOf(kCentralHeaderLiteral)))
        << i;
    ZipEntry header;
    ASSERT_THAT(ReadCentralDirectoryEntry(string_reader, header), IsOk());
    central_headers.push_back(std::move(header));
  }

  std::vector<ZipEntry> local_headers;
  for (const auto& header : central_headers) {
    ZipEntry local_header;
    string_reader.Seek(header.local_header_offset);
    EXPECT_TRUE(StartsWith(string_reader, StringViewOf(kLocalHeaderLiteral)));
    ASSERT_THAT(ReadLocalEntry(string_reader, local_header), IsOk());
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

TEST(ZipDetailsTest, EasyReadZip) {
  riegeli::CordReader string_reader(GetZipTest2Data());
  EasyZipReader zip_reader(string_reader);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto entries, zip_reader.entries());

  std::vector<ZipEntry> local_headers;
  for (const auto& header : entries) {
    ZipEntry local_header;
    string_reader.Seek(header.local_header_offset);
    EXPECT_TRUE(StartsWith(string_reader, StringViewOf(kLocalHeaderLiteral)));
    EXPECT_THAT(ReadLocalEntry(string_reader, local_header), IsOk());
    local_headers.push_back(std::move(local_header));
  }

  EXPECT_THAT(local_headers.size(), 3);
  for (size_t i = 0; i < local_headers.size(); ++i) {
    EXPECT_EQ(local_headers[i].flags, entries[i].flags);
    EXPECT_EQ(local_headers[i].compression_method,
              entries[i].compression_method);
    EXPECT_EQ(local_headers[i].crc, entries[i].crc);
    EXPECT_EQ(local_headers[i].compressed_size, entries[i].compressed_size);
    EXPECT_EQ(local_headers[i].uncompressed_size, entries[i].uncompressed_size);
    EXPECT_EQ(local_headers[i].filename, entries[i].filename);
  }

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto reader,
                                   GetReader(&string_reader, local_headers[0]));
  std::string data;
  EXPECT_THAT(riegeli::ReadAll(*reader, data), IsOk());
  EXPECT_EQ(data, "test\n");
  EXPECT_EQ(data.size(), local_headers[0].uncompressed_size);
}

// Test specific formats.
TEST(ZipDetailsTest, Xz) {
  riegeli::CordReader string_reader(GetZipXzData());
  EasyZipReader zip_reader(string_reader);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto entries, zip_reader.entries());
  EXPECT_THAT(entries.size(), ::testing::Gt(0));

  ZipEntry local_header;
  string_reader.Seek(entries[0].local_header_offset);
  EXPECT_TRUE(StartsWith(string_reader, StringViewOf(kLocalHeaderLiteral)));
  ASSERT_THAT(ReadLocalEntry(string_reader, local_header), IsOk());

  EXPECT_EQ(local_header.compression_method, ZipCompression::kXZ);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto reader,
                                   GetReader(&string_reader, local_header));
  std::string data;
  EXPECT_THAT(riegeli::ReadAll(*reader, data), IsOk());
  EXPECT_EQ(data,
            "aaaaaaaaaaaaaa\r\nbbbbbbbbbbbbbb\r\naaaaaaaaaaaaaa\r\ncccccccccccc"
            "cc\r\n");
  EXPECT_EQ(data.size(), local_header.uncompressed_size);
}

TEST(ZipDetailsTest, Zstd) {
  riegeli::CordReader string_reader(GetZipZstdData());
  EasyZipReader zip_reader(string_reader);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto entries, zip_reader.entries());
  EXPECT_THAT(entries.size(), ::testing::Gt(0));

  ZipEntry local_header;
  string_reader.Seek(entries[0].local_header_offset);
  EXPECT_TRUE(StartsWith(string_reader, StringViewOf(kLocalHeaderLiteral)));
  ASSERT_THAT(ReadLocalEntry(string_reader, local_header), IsOk());

  EXPECT_EQ(local_header.compression_method, ZipCompression::kZStd);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto reader,
                                   GetReader(&string_reader, local_header));
  std::string data;
  EXPECT_THAT(riegeli::ReadAll(*reader, data), IsOk());
  EXPECT_EQ(data,
            "aaaaaaaaaaaaaa\r\nbbbbbbbbbbbbbb\r\naaaaaaaaaaaaaa\r\ncccccccccccc"
            "cc\r\n");
  EXPECT_EQ(data.size(), local_header.uncompressed_size);
}

TEST(ZipDetailsTest, Bzip2) {
  riegeli::CordReader string_reader(GetZipBzip2Data());
  EasyZipReader zip_reader(string_reader);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto entries, zip_reader.entries());
  EXPECT_THAT(entries.size(), ::testing::Gt(0));

  ZipEntry local_header;
  string_reader.Seek(entries[0].local_header_offset);
  EXPECT_TRUE(StartsWith(string_reader, StringViewOf(kLocalHeaderLiteral)));
  ASSERT_THAT(ReadLocalEntry(string_reader, local_header), IsOk());

  EXPECT_EQ(local_header.compression_method, ZipCompression::kBzip2);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto reader,
                                   GetReader(&string_reader, local_header));
  std::string data;
  EXPECT_THAT(riegeli::ReadAll(*reader, data), IsOk());
  EXPECT_EQ(data,
            "aaaaaaaaaaaaaa\nbbbbbbbbbbbbbb\naaaaaaaaaaaaaa\ncccccccccccccc\n");
  EXPECT_EQ(data.size(), local_header.uncompressed_size);
}

TEST(ZipDetailsTest, Deflate) {
  riegeli::CordReader string_reader(GetZipDeflateData());
  EasyZipReader zip_reader(string_reader);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto entries, zip_reader.entries());
  EXPECT_THAT(entries.size(), ::testing::Gt(0));

  ZipEntry local_header;
  string_reader.Seek(entries[0].local_header_offset);
  EXPECT_TRUE(StartsWith(string_reader, StringViewOf(kLocalHeaderLiteral)));
  ASSERT_THAT(ReadLocalEntry(string_reader, local_header), IsOk());

  EXPECT_EQ(local_header.compression_method, ZipCompression::kDeflate);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto reader,
                                   GetReader(&string_reader, local_header));
  std::string data;
  EXPECT_THAT(riegeli::ReadAll(*reader, data), IsOk());
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
  // Reads a ZipFileHeader from the current stream position.
  ASSERT_THAT(ReadLocalEntry(reader, header), IsOk());

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

  // Reads a ZipFileHeader from the current stream position.
  ASSERT_THAT(ReadCentralDirectoryEntry(reader, header), IsOk());

  EXPECT_THAT(header.flags, 0x2);
  EXPECT_THAT(header.compression_method, ZipCompression::kDeflate);
  EXPECT_THAT(header.crc, 0x94EE1E3E);
  EXPECT_THAT(header.compressed_size, 0x00019F62);
  EXPECT_THAT(header.uncompressed_size, 0x00019F6F);
  EXPECT_THAT(header.local_header_offset, 0);

  // for additional bookkeeping.
  EXPECT_THAT(header.end_of_header_offset, 319299);
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
  ASSERT_THAT(ReadEOCD(reader, eocd), IsOk());

  EXPECT_THAT(eocd.num_entries, 3);
  EXPECT_THAT(eocd.cd_size, 0x000000F1);
  EXPECT_THAT(eocd.cd_offset, 0x0004DEF3);  // offset from start of file.

  // for additional bookkeeping.
  EXPECT_THAT(eocd.comment, "");
}

TEST(TestdataTest, FileData) {
  riegeli::CordReader reader(GetTestZipFileData());

  ZipEntry header;

  // Reads a ZipFileHeader from the current stream position.
  ASSERT_THAT(ReadLocalEntry(reader, header), IsOk());

  EXPECT_THAT(reader.pos(), 0x0044);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto entry_reader, tensorstore::internal_zip::GetReader(&reader, header));

  std::string data;
  EXPECT_THAT(riegeli::ReadAll(*entry_reader, data), IsOk());
  EXPECT_EQ(data.size(), header.uncompressed_size);
}

TEST(ZipDetailsTest, WriteAndReadRoundtrip) {
  absl::Cord cord;
  riegeli::CordWriter writer(&cord);

  ZipEntry entry1;
  entry1.version_madeby = 20;
  entry1.flags = 0;
  entry1.compression_method = ZipCompression::kStore;
  entry1.mtime = absl::FromCivil(absl::CivilSecond(2023, 8, 3, 6, 2, 22),
                                 absl::UTCTimeZone());
  entry1.crc = 0x12345678;
  entry1.compressed_size = 5;
  entry1.uncompressed_size = 5;
  entry1.filename = "file1";
  entry1.comment = "comment1";
  entry1.local_header_offset = 0;

  TENSORSTORE_ASSERT_OK(WriteLocalEntry(writer, entry1));
  writer.Write("hello");

  entry1.local_header_offset = 0;

  auto cd_offset = writer.pos();
  TENSORSTORE_ASSERT_OK(WriteCentralDirectoryEntry(writer, entry1));
  auto cd_size = writer.pos() - cd_offset;

  ZipEOCD eocd;
  eocd.num_entries = 1;
  eocd.cd_size = cd_size;
  eocd.cd_offset = cd_offset;
  eocd.comment = "zipcomment";
  TENSORSTORE_ASSERT_OK(WriteEOCD(writer, eocd));

  EXPECT_TRUE(writer.Close()) << writer.status();

  riegeli::CordReader reader(&cord);

  EXPECT_TRUE(FindFirst(reader, StringViewOf(kEOCDLiteral)));
  ZipEOCD read_eocd;
  ASSERT_THAT(ReadEOCD(reader, read_eocd), IsOk());
  EXPECT_EQ(read_eocd.num_entries, 1);
  EXPECT_EQ(read_eocd.cd_size, cd_size);
  EXPECT_EQ(read_eocd.cd_offset, cd_offset);
  EXPECT_EQ(read_eocd.comment, "zipcomment");

  reader.Seek(read_eocd.cd_offset);
  EXPECT_TRUE(StartsWith(reader, StringViewOf(kCentralHeaderLiteral)));
  ZipEntry read_central_entry;
  ASSERT_THAT(ReadCentralDirectoryEntry(reader, read_central_entry), IsOk());
  EXPECT_EQ(read_central_entry.version_madeby, entry1.version_madeby);
  EXPECT_EQ(read_central_entry.flags, entry1.flags);
  EXPECT_EQ(read_central_entry.compression_method, entry1.compression_method);
  EXPECT_EQ(read_central_entry.crc, entry1.crc);
  EXPECT_EQ(read_central_entry.compressed_size, entry1.compressed_size);
  EXPECT_EQ(read_central_entry.uncompressed_size, entry1.uncompressed_size);
  EXPECT_EQ(read_central_entry.filename, entry1.filename);
  EXPECT_EQ(read_central_entry.comment, entry1.comment);
  EXPECT_EQ(read_central_entry.local_header_offset, entry1.local_header_offset);
  EXPECT_EQ(read_central_entry.mtime, entry1.mtime);

  reader.Seek(read_central_entry.local_header_offset);
  EXPECT_TRUE(StartsWith(reader, StringViewOf(kLocalHeaderLiteral)));
  ZipEntry read_local_entry;
  ASSERT_THAT(ReadLocalEntry(reader, read_local_entry), IsOk());
  EXPECT_EQ(read_local_entry.flags, entry1.flags);
  EXPECT_EQ(read_local_entry.compression_method, entry1.compression_method);
  EXPECT_EQ(read_local_entry.crc, entry1.crc);
  EXPECT_EQ(read_local_entry.compressed_size, entry1.compressed_size);
  EXPECT_EQ(read_local_entry.uncompressed_size, entry1.uncompressed_size);
  EXPECT_EQ(read_local_entry.filename, entry1.filename);
  EXPECT_EQ(read_local_entry.mtime, entry1.mtime);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto data_reader,
                                   GetReader(&reader, read_local_entry));
  std::string data;
  EXPECT_THAT(riegeli::ReadAll(*data_reader, data), IsOk());
  EXPECT_EQ(data, "hello");
}

TEST(ZipDetailsTest, ReadCentralHeaderMalformedExtraFieldLengthAssertion) {
  static constexpr unsigned char kBytes[] = {
      0x50, 0x4b, 0x01, 0x02,  // Signature
      0x14, 0x00,              // Version made by (20)
      0x14, 0x00,              // Version needed (20)
      0x00, 0x00,              // Flags
      0x00, 0x00,              // Compression method (store)
      0x00, 0x00,              // Last mod time
      0x00, 0x00,              // Last mod date
      0x00, 0x00, 0x00, 0x00,  // CRC
      0x00, 0x00, 0x00, 0x00,  // Compressed size
      0x00, 0x00, 0x00, 0x00,  // Uncompressed size
      0x04, 0x00,              // File name length (4)
      0x03, 0x00,              // Extra field length (3)
      0x00, 0x00,              // File comment length
      0x00, 0x00,              // Disk number start
      0x00, 0x00,              // Internal attr
      0x00, 0x00, 0x00, 0x00,  // External attr
      0x00, 0x00, 0x00, 0x00,  // Local header offset
      't',  'e',  's',  't',   // Filename ("test")
      0x01, 0x02, 0x03         // Extra field (3 bytes)
  };
  riegeli::StringReader reader(StringViewOf(kBytes));
  ZipEntry entry;
  auto status = ReadCentralDirectoryEntry(reader, entry);
  EXPECT_FALSE(status.ok());
}

TEST(ZipDetailsTest, ReadLocalHeaderMalformedExtraFieldLengthAssertion) {
  static constexpr unsigned char kBytes[] = {
      0x50, 0x4b, 0x03, 0x04,  // Signature
      0x14, 0x00,              // Version needed (20)
      0x00, 0x00,              // Flags
      0x00, 0x00,              // Compression method (store)
      0x00, 0x00,              // Last mod time
      0x00, 0x00,              // Last mod date
      0x00, 0x00, 0x00, 0x00,  // CRC
      0x00, 0x00, 0x00, 0x00,  // Compressed size
      0x00, 0x00, 0x00, 0x00,  // Uncompressed size
      0x04, 0x00,              // File name length (4)
      0x03, 0x00,              // Extra field length (3)
      't',  'e',  's',  't',   // Filename ("test")
      0x01, 0x02, 0x03         // Extra field (3 bytes)
  };
  riegeli::StringReader reader(StringViewOf(kBytes));
  ZipEntry entry;
  auto status = ReadLocalEntry(reader, entry);
  EXPECT_FALSE(status.ok());
}

TEST(ZipDetailsTest, ReadExtraFieldZip64MalformedTagSizeAssertion) {
  static constexpr unsigned char kBytes[] = {
      0x50, 0x4b, 0x03, 0x04,  // Signature
      0x14, 0x00,              // Version needed (20)
      0x00, 0x00,              // Flags
      0x00, 0x00,              // Compression method (store)
      0x00, 0x00,              // Last mod time
      0x00, 0x00,              // Last mod date
      0x00, 0x00, 0x00, 0x00,  // CRC
      0x00, 0x00, 0x00, 0x00,  // Compressed size
      0x00, 0x00, 0x00, 0x00,  // Uncompressed size
      0x04, 0x00,              // File name length (4)
      0x08, 0x00,              // Extra field length (8)
      't', 'e', 's', 't',      // Filename ("test")
      // Extra field:
      0x01, 0x00,  // Tag ZIP64 (0x0001)
      0x02, 0x00,  // Tag size (2)
      0x34, 0x12   // 2 bytes of data (0x1234)
  };
  riegeli::StringReader reader(StringViewOf(kBytes));
  ZipEntry entry;
  auto status = ReadLocalEntry(reader, entry);
  EXPECT_FALSE(status.ok());
}
TEST(ZipDetailsTest, ValidateEntryIsSupportedSizeLimit) {
  ZipEntry entry;
  entry.flags = 0;
  entry.compression_method = ZipCompression::kStore;
  entry.filename = "test";

  // 3 GB, exceeds 2 GB limit.
  entry.uncompressed_size = 3ULL << 30;
  entry.compressed_size = 3ULL << 30;
  EXPECT_FALSE(ValidateEntryIsSupported(entry).ok());
}

TEST(ZipDetailsTest, ValidateEntryIsSupportedZipBomb) {
  ZipEntry entry;
  entry.flags = 0;
  entry.compression_method = ZipCompression::kDeflate;
  entry.filename = "test";

  // 100 MB uncompressed, 10 KB compressed. Ratio = 10240, exceeds 1024 limit.
  entry.uncompressed_size = 100 * 1024 * 1024;
  entry.compressed_size = 10 * 1024;
  EXPECT_FALSE(ValidateEntryIsSupported(entry).ok());

  // 100 MB uncompressed, 1 MB compressed. Ratio = 100, within limit.
  entry.uncompressed_size = 100 * 1024 * 1024;
  entry.compressed_size = 1024 * 1024;
  EXPECT_TRUE(ValidateEntryIsSupported(entry).ok());

  // 500 KB uncompressed, 10 bytes compressed. Ratio = 51200.
  // But uncompressed is < 1 MB check threshold, so it should be allowed.
  entry.uncompressed_size = 500 * 1024;
  entry.compressed_size = 10;
  EXPECT_TRUE(ValidateEntryIsSupported(entry).ok());

  // Exactly 1024 ratio.
  entry.uncompressed_size = 1024 * 1024;
  entry.compressed_size = 1024;
  EXPECT_TRUE(ValidateEntryIsSupported(entry).ok());

  // Slightly above 1024 ratio.
  entry.uncompressed_size = 1024 * 1024 + 1;
  entry.compressed_size = 1024;
  EXPECT_FALSE(ValidateEntryIsSupported(entry).ok());
}

TEST(ZipDetailsTest, ValidateEntryIsSupportedPathTraversal) {
  ZipEntry entry;
  entry.flags = 0;
  entry.compression_method = ZipCompression::kStore;
  entry.uncompressed_size = 0;
  entry.compressed_size = 0;

  // Safe path.
  entry.filename = "foo/bar/baz";
  EXPECT_TRUE(ValidateEntryIsSupported(entry).ok());

  // Safe path containing two dots.
  entry.filename = "foo..bar/baz";
  EXPECT_TRUE(ValidateEntryIsSupported(entry).ok());

  // Safe path containing dot.
  entry.filename = "foo.bar/baz";
  EXPECT_TRUE(ValidateEntryIsSupported(entry).ok());

  // Path traversal: parent directory at start.
  entry.filename = "../foo/bar";
  EXPECT_FALSE(ValidateEntryIsSupported(entry).ok());

  // Path traversal: parent directory in middle.
  entry.filename = "foo/../bar";
  EXPECT_FALSE(ValidateEntryIsSupported(entry).ok());

  // Path traversal: parent directory at end.
  entry.filename = "foo/bar/..";
  EXPECT_FALSE(ValidateEntryIsSupported(entry).ok());

  // Path traversal: absolute path.
  entry.filename = "/foo/bar";
  EXPECT_FALSE(ValidateEntryIsSupported(entry).ok());

  // Path traversal: Windows style parent directory.
  entry.filename = "foo\\..\\bar";
  EXPECT_FALSE(ValidateEntryIsSupported(entry).ok());
}

TEST(ZipDetailsTest, WriteLocalEntryExceeds4GB) {
  std::string bytes;
  riegeli::StringWriter writer(&bytes);
  ZipEntry entry;
  entry.filename = "test";
  entry.uncompressed_size = 4ULL << 30;  // 4 GB
  entry.compressed_size = 3ULL << 30;    // 3 GB
  entry.crc = 0x12345678;
  ASSERT_TRUE(WriteLocalEntry(writer, entry).ok());
  ASSERT_TRUE(writer.Close());
  EXPECT_EQ(entry.end_of_header_offset, bytes.size());

  // Read it back
  riegeli::StringReader reader(bytes);
  ZipEntry read_entry;
  ASSERT_TRUE(ReadLocalEntry(reader, read_entry).ok());
  EXPECT_EQ(read_entry.filename, "test");
  EXPECT_EQ(read_entry.uncompressed_size, 4ULL << 30);
  EXPECT_EQ(read_entry.compressed_size, 3ULL << 30);
  EXPECT_EQ(read_entry.crc, 0x12345678);
  EXPECT_TRUE(read_entry.is_zip64);
}

TEST(ZipDetailsTest, WriteCentralDirectoryEntryExceeds4GB) {
  std::string bytes;
  riegeli::StringWriter writer(&bytes);
  ZipEntry entry;
  entry.filename = "test";
  entry.uncompressed_size = 4ULL << 30;
  entry.compressed_size = 3ULL << 30;
  entry.local_header_offset = 5ULL << 30;
  entry.crc = 0x12345678;
  ASSERT_TRUE(WriteCentralDirectoryEntry(writer, entry).ok());
  ASSERT_TRUE(writer.Close());

  // Read it back
  riegeli::StringReader reader(bytes);
  ZipEntry read_entry;
  ASSERT_TRUE(ReadCentralDirectoryEntry(reader, read_entry).ok());
  EXPECT_EQ(read_entry.filename, "test");
  EXPECT_EQ(read_entry.uncompressed_size, 4ULL << 30);
  EXPECT_EQ(read_entry.compressed_size, 3ULL << 30);
  EXPECT_EQ(read_entry.local_header_offset, 5ULL << 30);
  EXPECT_EQ(read_entry.crc, 0x12345678);
  EXPECT_TRUE(read_entry.is_zip64);
}

TEST(ZipDetailsTest, WriteAndReadCentralDirectoryEntryZip64WithComment) {
  std::string bytes;
  riegeli::StringWriter writer(&bytes);
  ZipEntry entry;
  entry.filename = "test";
  entry.comment = "this is a comment";
  entry.uncompressed_size = 4ULL << 30;
  entry.compressed_size = 3ULL << 30;
  entry.local_header_offset = 5ULL << 30;
  entry.crc = 0x12345678;
  ASSERT_TRUE(WriteCentralDirectoryEntry(writer, entry).ok());
  ASSERT_TRUE(writer.Close());

  // Read it back
  riegeli::StringReader reader(bytes);
  ZipEntry read_entry;
  ASSERT_TRUE(ReadCentralDirectoryEntry(reader, read_entry).ok());
  EXPECT_EQ(read_entry.filename, "test");
  EXPECT_EQ(read_entry.comment, "this is a comment");
  EXPECT_EQ(read_entry.uncompressed_size, 4ULL << 30);
  EXPECT_EQ(read_entry.compressed_size, 3ULL << 30);
  EXPECT_EQ(read_entry.local_header_offset, 5ULL << 30);
  EXPECT_EQ(read_entry.crc, 0x12345678);
  EXPECT_TRUE(read_entry.is_zip64);
}

TEST(ZipDetailsTest, ReadEOCD64BoundaryValues) {
  // Case 1: num_entries is exactly 65535
  {
    std::string bytes;
    riegeli::StringWriter writer(&bytes);
    ZipEOCD eocd;
    eocd.num_entries = 65535;
    eocd.cd_size = 100;
    eocd.cd_offset = 100;
    ASSERT_TRUE(WriteEOCD(writer, eocd).ok());
    ASSERT_TRUE(writer.Close());

    riegeli::StringReader reader(bytes);
    ZipEOCD read_eocd;
    auto response = TryReadFullEOCD(reader, read_eocd, 0);
    ASSERT_TRUE(std::holds_alternative<absl::Status>(response));
    EXPECT_THAT(std::get<absl::Status>(response), IsOk());
    EXPECT_EQ(read_eocd.num_entries, 65535);
  }

  // Case 2: cd_size is exactly 65535
  {
    std::string bytes;
    riegeli::StringWriter writer(&bytes);
    ZipEOCD eocd;
    eocd.num_entries = 70000;  // Force ZIP64
    eocd.cd_size = 65535;
    eocd.cd_offset = 100;
    ASSERT_TRUE(WriteEOCD(writer, eocd).ok());
    ASSERT_TRUE(writer.Close());

    riegeli::StringReader reader(bytes);
    ZipEOCD read_eocd;
    auto response = TryReadFullEOCD(reader, read_eocd, 0);
    ASSERT_TRUE(std::holds_alternative<absl::Status>(response));
    EXPECT_THAT(std::get<absl::Status>(response), IsOk());
    EXPECT_EQ(read_eocd.cd_size, 65535);
  }

  // Case 3: cd_offset is exactly 4GB (0xFFFFFFFF)
  {
    std::string bytes;
    riegeli::StringWriter writer(&bytes);
    ZipEOCD eocd;
    eocd.num_entries = 1;
    eocd.cd_size = 100;
    eocd.cd_offset = 0xFFFFFFFF;  // Force ZIP64
    ASSERT_TRUE(WriteEOCD(writer, eocd).ok());
    ASSERT_TRUE(writer.Close());

    riegeli::StringReader reader(bytes);
    ZipEOCD read_eocd;
    auto response = TryReadFullEOCD(reader, read_eocd, 0);
    ASSERT_TRUE(std::holds_alternative<absl::Status>(response));
    EXPECT_THAT(std::get<absl::Status>(response), IsOk());
    EXPECT_EQ(read_eocd.cd_offset, 0xFFFFFFFF);
  }

  // Case 4: num_entries is UINT64_MAX sentinel
  {
    std::string bytes;
    riegeli::StringWriter writer(&bytes);
    ZipEOCD eocd;
    eocd.num_entries = std::numeric_limits<uint64_t>::max();
    eocd.cd_size = 100;
    eocd.cd_offset = 100;
    ASSERT_TRUE(WriteEOCD(writer, eocd).ok());
    ASSERT_TRUE(writer.Close());

    riegeli::StringReader reader(bytes);
    ZipEOCD read_eocd;
    auto response = TryReadFullEOCD(reader, read_eocd, 0);
    ASSERT_TRUE(std::holds_alternative<absl::Status>(response));
    EXPECT_FALSE(std::get<absl::Status>(response).ok());
  }

  // Case 5: cd_size is INT64_MAX sentinel
  {
    std::string bytes;
    riegeli::StringWriter writer(&bytes);
    ZipEOCD eocd;
    eocd.num_entries = 70000;
    eocd.cd_size = std::numeric_limits<int64_t>::max();
    eocd.cd_offset = 100;
    ASSERT_TRUE(WriteEOCD(writer, eocd).ok());
    ASSERT_TRUE(writer.Close());

    riegeli::StringReader reader(bytes);
    ZipEOCD read_eocd;
    auto response = TryReadFullEOCD(reader, read_eocd, 0);
    ASSERT_TRUE(std::holds_alternative<absl::Status>(response));
    EXPECT_FALSE(std::get<absl::Status>(response).ok());
  }

  // Case 6: cd_offset is INT64_MAX sentinel
  {
    std::string bytes;
    riegeli::StringWriter writer(&bytes);
    ZipEOCD eocd;
    eocd.num_entries = 70000;
    eocd.cd_size = 100;
    eocd.cd_offset = std::numeric_limits<int64_t>::max();
    ASSERT_TRUE(WriteEOCD(writer, eocd).ok());
    ASSERT_TRUE(writer.Close());

    riegeli::StringReader reader(bytes);
    ZipEOCD read_eocd;
    auto response = TryReadFullEOCD(reader, read_eocd, 0);
    ASSERT_TRUE(std::holds_alternative<absl::Status>(response));
    EXPECT_FALSE(std::get<absl::Status>(response).ok());
  }
}

TEST(ZipDetailsTest, WriteEOCDExceedsLimits) {
  std::string bytes;
  riegeli::StringWriter writer(&bytes);
  ZipEOCD eocd;
  eocd.num_entries = 70000;     // Exceeds 65535
  eocd.cd_size = 4ULL << 30;    // Exceeds 4GB
  eocd.cd_offset = 5ULL << 30;  // Exceeds 4GB
  eocd.comment = "EOCD comment";
  ASSERT_TRUE(WriteEOCD(writer, eocd).ok());
  ASSERT_TRUE(writer.Close());

  // Read it back
  riegeli::StringReader reader(bytes);
  ZipEOCD read_eocd;
  auto response = TryReadFullEOCD(reader, read_eocd, 0);
  ASSERT_TRUE(std::holds_alternative<absl::Status>(response));
  ASSERT_TRUE(std::get<absl::Status>(response).ok());

  EXPECT_EQ(read_eocd.num_entries, 70000);
  EXPECT_EQ(read_eocd.cd_size, 4ULL << 30);
  EXPECT_EQ(read_eocd.cd_offset, 5ULL << 30);
  EXPECT_EQ(read_eocd.comment, "EOCD comment");
}

TEST(ZipDetailsTest, AnalyzeInvalidEocdZip) {
  riegeli::StringReader reader(StringViewOf(kInvalidEocdZip));

  EasyZipReader zip_reader(reader);
  absl::Status status = zip_reader.Initialize();

  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(),
              ::testing::HasSubstr("Inconsistent ZIP64 EOCD locator offset"));
}

TEST(ZipDetailsTest, CompressRoundtrip) {
  std::string original_data =
      "Hello, world! This is a test string to be compressed and decompressed.";
  absl::Cord original_cord(original_data);

  for (auto method : {ZipCompression::kStore, ZipCompression::kDeflate,
                      ZipCompression::kZStd, ZipCompression::kXZ}) {
    std::array<ZipCompression, 1> methods = {method};
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto compress_result,
                                     Compress(original_cord, methods));
    if (method != ZipCompression::kStore) {
      EXPECT_NE(original_cord, compress_result.data);
    }
    EXPECT_EQ(compress_result.method, method);

    ZipEntry entry;
    entry.compression_method = compress_result.method;
    entry.compressed_size = compress_result.data.size();
    entry.uncompressed_size = original_cord.size();
    entry.flags = 0;

    riegeli::CordReader reader(&compress_result.data);
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto decompressed_reader,
                                     GetReader(&reader, entry));
    std::string decompressed_data;
    EXPECT_THAT(riegeli::ReadAll(*decompressed_reader, decompressed_data),
                IsOk());
    EXPECT_EQ(decompressed_data, original_data);
  }
}

TEST(ZipDetailsTest, CompressMulti) {
  // For very short data, Store should be smaller than Deflate/ZStd/XZ
  // because of header/metadata overhead.
  std::string short_data = "a";
  absl::Cord short_cord(short_data);

  std::array<ZipCompression, 4> all_methods = {
      ZipCompression::kStore, ZipCompression::kDeflate, ZipCompression::kZStd,
      ZipCompression::kXZ};

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto short_result,
                                   Compress(short_cord, all_methods));
  EXPECT_EQ(short_result.method, ZipCompression::kStore);
  EXPECT_EQ(short_result.data, short_cord);

  // For highly repetitive data, XZ or ZStd or Deflate should be smaller than
  // Store.
  std::string long_data(1000, 'a');
  absl::Cord long_cord(long_data);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto long_result,
                                   Compress(long_cord, all_methods));
  EXPECT_NE(long_result.method, ZipCompression::kStore);
  EXPECT_LT(long_result.data.size(), long_cord.size());

  // Verify we can decompress it
  ZipEntry entry;
  entry.compression_method = long_result.method;
  entry.compressed_size = long_result.data.size();
  entry.uncompressed_size = long_cord.size();
  entry.flags = 0;

  riegeli::CordReader reader(&long_result.data);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto decompressed_reader,
                                   GetReader(&reader, entry));
  std::string decompressed_data;
  EXPECT_THAT(riegeli::ReadAll(*decompressed_reader, decompressed_data),
              IsOk());
  EXPECT_EQ(decompressed_data, long_data);
}

TEST(ZipEasyTest, Roundtrip) {
  absl::flat_hash_map<std::string, absl::Cord> entry_data;
  entry_data["file1.txt"] = absl::Cord("Hello World");
  entry_data["file2.txt"] = absl::Cord(std::string(100, 'a'));

  absl::Cord zip_data;
  riegeli::CordWriter writer(&zip_data);
  EasyZipWriter zip_writer(writer);

  ZipEntry entry1;
  entry1.filename = "file1.txt";
  entry1.compression_method = ZipCompression::kStore;
  TENSORSTORE_ASSERT_OK(zip_writer.WriteEntry(entry1, entry_data["file1.txt"]));

  ZipEntry entry2;
  entry2.filename = "file2.txt";
  entry2.compression_method = ZipCompression::kDeflate;
  TENSORSTORE_ASSERT_OK(zip_writer.WriteEntry(entry2, entry_data["file2.txt"]));

  TENSORSTORE_ASSERT_OK(zip_writer.Finalize());
  ASSERT_TRUE(writer.Close());

  // Now read it back.
  riegeli::CordReader reader(&zip_data);
  EasyZipReader zip_reader(reader);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto entries, zip_reader.entries());

  ASSERT_EQ(entries.size(), 2);
  EXPECT_EQ(entries[0].filename, "file1.txt");
  EXPECT_EQ(entries[0].compression_method, ZipCompression::kStore);
  EXPECT_EQ(entries[0].uncompressed_size, entry_data["file1.txt"].size());

  EXPECT_EQ(entries[1].filename, "file2.txt");
  EXPECT_EQ(entries[1].compression_method, ZipCompression::kDeflate);
  EXPECT_EQ(entries[1].uncompressed_size, entry_data["file2.txt"].size());

  // Verify content.
  for (size_t i = 0; i < entries.size(); ++i) {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto content,
                                     zip_reader.ReadEntry(entries[i].filename));
    EXPECT_EQ(content, entry_data[entries[i].filename]);
  }
}

TEST(ZipDetailsTest, EasyZipWriter) {
  absl::Cord zip_data;
  riegeli::CordWriter writer(&zip_data);
  EasyZipWriter zip_writer(writer);

  absl::Time custom_time = absl::FromUnixSeconds(1234567890);
  TENSORSTORE_ASSERT_OK(zip_writer.WriteEntry("file1.txt", absl::Cord("Hello"),
                                              ZipCompression::kStore,
                                              custom_time, "Comment 1"));
  TENSORSTORE_ASSERT_OK(zip_writer.WriteEntry("file2.txt", absl::Cord("World"),
                                              ZipCompression::kDeflate));
  TENSORSTORE_ASSERT_OK(zip_writer.Finalize());
  ASSERT_TRUE(writer.Close());

  // Read back and verify.
  riegeli::CordReader reader(&zip_data);
  EasyZipReader zip_reader(reader);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto entries, zip_reader.entries());

  ASSERT_EQ(entries.size(), 2);
  EXPECT_EQ(entries[0].filename, "file1.txt");
  EXPECT_EQ(entries[0].compression_method, ZipCompression::kStore);
  EXPECT_EQ(entries[0].mtime, custom_time);
  EXPECT_EQ(entries[0].comment, "Comment 1");

  EXPECT_EQ(entries[1].filename, "file2.txt");
  EXPECT_EQ(entries[1].compression_method, ZipCompression::kDeflate);
  EXPECT_EQ(entries[1].comment, "");

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto content1,
                                   zip_reader.ReadEntry("file1.txt"));
  EXPECT_EQ(content1, "Hello");

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto content2,
                                   zip_reader.ReadEntry("file2.txt"));
  EXPECT_EQ(content2, "World");
}

TEST(ZipDetailsTest, EasyZipReader) {
  absl::Cord zip_data;
  riegeli::CordWriter writer(&zip_data);
  EasyZipWriter zip_writer(writer);

  TENSORSTORE_ASSERT_OK(zip_writer.WriteEntry("file1.txt", absl::Cord("Hello"),
                                              ZipCompression::kStore));
  TENSORSTORE_ASSERT_OK(zip_writer.WriteEntry("file2.txt", absl::Cord("World"),
                                              ZipCompression::kDeflate));
  TENSORSTORE_ASSERT_OK(zip_writer.Finalize());
  ASSERT_TRUE(writer.Close());

  // Read back and verify with EasyZipReader.
  riegeli::CordReader reader(&zip_data);
  EasyZipReader zip_reader(reader);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto entries, zip_reader.entries());
  ASSERT_EQ(entries.size(), 2);
  EXPECT_EQ(entries[0].filename, "file1.txt");
  EXPECT_EQ(entries[1].filename, "file2.txt");

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto content1,
                                   zip_reader.ReadEntry("file1.txt"));
  EXPECT_EQ(content1, "Hello");

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto content2,
                                   zip_reader.ReadEntry("file2.txt"));
  EXPECT_EQ(content2, "World");

  // Read with invalid file name should error.
  auto not_found_result = zip_reader.ReadEntry("nonexistent.txt");
  EXPECT_FALSE(not_found_result.ok());
  EXPECT_EQ(not_found_result.status().code(), absl::StatusCode::kNotFound);
}

TEST(ZipEasyTest, Zip64) {
  std::string bytes;
  riegeli::StringWriter writer(&bytes);

  // Write a ZIP64 local entry.
  ZipEntry entry;
  entry.filename = "file.txt";
  entry.compression_method = ZipCompression::kStore;
  entry.uncompressed_size = 10;
  entry.compressed_size = 10;
  entry.crc = 12345;
  entry.local_header_offset = 0;
  ASSERT_TRUE(WriteLocalEntry(writer, entry).ok());
  writer.Write("0123456789");
  EXPECT_EQ(entry.end_of_header_offset, writer.pos() - 10);

  // Write a central directory entry (forcing ZIP64 offsets/sizes).
  uint64_t cd_offset = writer.pos();
  entry.uncompressed_size = 5ULL << 30;  // 5 GB, forces ZIP64
  entry.compressed_size = 10;
  ASSERT_TRUE(WriteCentralDirectoryEntry(writer, entry).ok());
  uint64_t cd_size = writer.pos() - cd_offset;

  // Write EOCD record (with ZIP64 because the offset or size exceeds limit).
  ZipEOCD eocd;
  eocd.num_entries = 1;
  eocd.cd_size = cd_size;
  eocd.cd_offset = cd_offset;
  ASSERT_TRUE(WriteEOCD(writer, eocd).ok());
  ASSERT_TRUE(writer.Close());

  // Parse and read using EasyZipReader.
  riegeli::StringReader string_reader(bytes);
  EasyZipReader zip_reader(string_reader);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto entries, zip_reader.entries());

  ASSERT_EQ(entries.size(), 1);
  EXPECT_EQ(entries[0].filename, "file.txt");
  EXPECT_EQ(entries[0].uncompressed_size, 5ULL << 30);
  EXPECT_TRUE(entries[0].is_zip64);
}

TEST(ZipDetailsTest, CompressEmptyData) {
  absl::Cord original_cord;

  for (auto method : {ZipCompression::kStore, ZipCompression::kDeflate,
                      ZipCompression::kZStd, ZipCompression::kXZ}) {
    std::array<ZipCompression, 1> methods = {method};
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto compress_result,
                                     Compress(original_cord, methods));
    EXPECT_EQ(compress_result.method, method);

    ZipEntry entry;
    entry.compression_method = compress_result.method;
    entry.compressed_size = compress_result.data.size();
    entry.uncompressed_size = original_cord.size();
    entry.flags = 0;

    riegeli::CordReader reader(&compress_result.data);
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto decompressed_reader,
                                     GetReader(&reader, entry));
    std::string decompressed_data;
    EXPECT_THAT(riegeli::ReadAll(*decompressed_reader, decompressed_data),
                IsOk());
    EXPECT_EQ(decompressed_data, "");
  }
}

TEST(ZipDetailsTest, DataDescriptor) {
  std::string file_path = GetTestDescriptorZipPath();
  riegeli::FdReader reader(file_path);
  ASSERT_TRUE(reader.ok()) << reader.status();

  EasyZipReader zip_reader(reader);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto entries, zip_reader.entries());
  ASSERT_EQ(entries.size(), 1);
  EXPECT_EQ(entries[0].filename, "test_file");
  EXPECT_EQ(entries[0].uncompressed_size, 11);
  EXPECT_EQ(entries[0].compressed_size, 11);
  EXPECT_EQ(entries[0].crc, 0xf73eae91);
  EXPECT_EQ(entries[0].extra_field_length, 0);

  // Seek and read the entry
  reader.Seek(entries[0].local_header_offset);
  ZipEntry local_header = entries[0];
  ASSERT_THAT(ReadLocalEntry(reader, local_header), IsOk());
  EXPECT_EQ(local_header.filename, "test_file");
  EXPECT_EQ(local_header.uncompressed_size, 11);
  EXPECT_EQ(local_header.compressed_size, 11);
  EXPECT_EQ(local_header.extra_field_length, 0);

  // Now read decompressed entry using GetReader.
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto decompressed_reader,
                                   GetReader(&reader, local_header));
  std::string decompressed_data;
  EXPECT_THAT(riegeli::ReadAll(*decompressed_reader, decompressed_data),
              IsOk());
  EXPECT_EQ(decompressed_data, "hello_world");
  // Verify that GetRawReader / GetReader correctly resolved properties back
  // into local_header.
  EXPECT_EQ(local_header.uncompressed_size, 11);
  EXPECT_EQ(local_header.compressed_size, 11);
  EXPECT_EQ(local_header.crc, 0xf73eae91);
}

TEST(ZipEasyTest, EasyZipWriterWriteFailure) {
  std::string bytes;
  riegeli::StringWriter base_writer(&bytes);
  riegeli::LimitingWriter writer(
      &base_writer, riegeli::LimitingWriterBase::Options().set_exact_length(5));
  EasyZipWriter zip_writer(writer);
  EXPECT_FALSE(
      zip_writer
          .WriteEntry("file1.txt", absl::Cord("Hello"), ZipCompression::kStore)
          .ok());
}

TEST(ZipDetailsTest, MSDOSTimeRoundTrip) {
  using ::tensorstore::internal_zip::MakeMSDOSTime;
  using ::tensorstore::internal_zip::MSDOSTime;
  using ::tensorstore::internal_zip::ValueToMSDOSTime;

  for (int minute : {0, 15, 30, 45, 59}) {
    absl::Time t = absl::FromCivil(
        absl::CivilSecond(2026, 6, 25, 10, minute, 12), absl::UTCTimeZone());
    MSDOSTime dos_time = ValueToMSDOSTime(t);
    absl::Time t_roundtrip = MakeMSDOSTime(dos_time);
    struct tm decoded_tm = absl::ToTM(t_roundtrip, absl::UTCTimeZone());
    EXPECT_EQ(decoded_tm.tm_min, minute) << "for minute " << minute;
    EXPECT_EQ(decoded_tm.tm_sec, 12);
  }
}

TEST(ZipSecurityTest, SameZipDuplicateFilenamesRejected) {
  std::string bytes;
  ZipEOCD eocd;
  {
    riegeli::StringWriter writer(&bytes);
    EasyZipWriter zip_writer(writer);

    ZipEntry entry1;
    entry1.filename = "file1.txt";
    entry1.compression_method = ZipCompression::kStore;
    TENSORSTORE_ASSERT_OK(zip_writer.WriteEntry(entry1, absl::Cord("data1")));

    ZipEntry entry2;
    entry2.filename = "file2.txt";
    entry2.compression_method = ZipCompression::kStore;
    TENSORSTORE_ASSERT_OK(zip_writer.WriteEntry(entry2, absl::Cord("data2")));

    TENSORSTORE_ASSERT_OK(zip_writer.Finalize(&eocd));
    ASSERT_TRUE(writer.Close());
  }

  // Find the central directory entry signature for CDH #1 and CDH #2.
  size_t cdh1_pos = bytes.find("PK\x01\x02", eocd.cd_offset);
  ASSERT_NE(cdh1_pos, std::string::npos);
  size_t cdh2_pos = bytes.find("PK\x01\x02", cdh1_pos + 4);
  ASSERT_NE(cdh2_pos, std::string::npos);

  // CDH #2 filename starts exactly 46 bytes after cdh2_pos.
  ASSERT_LT(cdh2_pos + 46 + 9, bytes.size());
  bytes.replace(cdh2_pos + 46, 9, "file1.txt");

  riegeli::StringReader string_reader(bytes);
  EasyZipReader zip_reader(string_reader);
  auto entries_result = zip_reader.entries();
  EXPECT_FALSE(entries_result.ok());
  EXPECT_THAT(entries_result.status().message(),
              ::testing::HasSubstr("Duplicate filename"));
}

TEST(ZipSecurityTest, ThreeNamesZipMismatchedFilesRejected) {
  // 3names.zip: contains different names under CDH, LFH and
  // Unicode Path Extra Field
  static constexpr unsigned char kThreeNamesZip[] = {
      // local header #1 + data
      0x50, 0x4b, 0x03, 0x04, 0x0a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x4b, 0x1e, 0xa7, 0x9e, 0x1b, 0x00, 0x00, 0x00, 0x1b, 0x00,
      0x00, 0x00, 0x1a, 0x00, 0x3a, 0x00, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x5f,
      0x66, 0x69, 0x6c, 0x65, 0x5f, 0x68, 0x65, 0x61, 0x64, 0x65, 0x72, 0x5f,
      0x6e, 0x61, 0x6d, 0x65, 0x2e, 0x74, 0x78, 0x74, 0x75, 0x70, 0x19, 0x00,
      0x01, 0xfc, 0x2d, 0x09, 0x78, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x5f, 0x69,
      0x6e, 0x66, 0x6f, 0x5f, 0x7a, 0x69, 0x70, 0x5f, 0x31, 0x2e, 0x74, 0x78,
      0x74, 0x75, 0x70, 0x19, 0x00, 0x01, 0x2c, 0x57, 0xa9, 0x3f, 0x6c, 0x6f,
      0x63, 0x61, 0x6c, 0x5f, 0x69, 0x6e, 0x66, 0x6f, 0x5f, 0x7a, 0x69, 0x70,
      0x5f, 0x32, 0x2e, 0x74, 0x78, 0x74, 0x4a, 0x75, 0x73, 0x74, 0x20, 0x65,
      0x6e, 0x74, 0x65, 0x72, 0x20, 0x61, 0x6e, 0x79, 0x20, 0x64, 0x61, 0x74,
      0x61, 0x20, 0x68, 0x65, 0x72, 0x65, 0x2e, 0x0d, 0x0a,
      // central header #1
      0x50, 0x4b, 0x01, 0x02, 0x1e, 0x03, 0x0a, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x4b, 0x1e, 0xa7, 0x9e, 0x1b, 0x00, 0x00, 0x00,
      0x1b, 0x00, 0x00, 0x00, 0x1a, 0x00, 0x3e, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x63, 0x65,
      0x6e, 0x74, 0x72, 0x61, 0x6c, 0x5f, 0x64, 0x69, 0x72, 0x65, 0x63, 0x74,
      0x6f, 0x72, 0x79, 0x5f, 0x6e, 0x61, 0x6d, 0x65, 0x2e, 0x74, 0x78, 0x74,
      0x75, 0x70, 0x1b, 0x00, 0x01, 0x8a, 0xc8, 0xdc, 0xf8, 0x63, 0x65, 0x6e,
      0x74, 0x72, 0x61, 0x6c, 0x5f, 0x69, 0x6e, 0x66, 0x6f, 0x5f, 0x7a, 0x69,
      0x70, 0x5f, 0x31, 0x2e, 0x74, 0x78, 0x74, 0x75, 0x70, 0x1b, 0x00, 0x01,
      0x5a, 0xb2, 0x7c, 0xbf, 0x63, 0x65, 0x6e, 0x74, 0x72, 0x61, 0x6c, 0x5f,
      0x69, 0x6e, 0x66, 0x6f, 0x5f, 0x7a, 0x69, 0x70, 0x5f, 0x32, 0x2e, 0x74,
      0x78, 0x74,
      // eocd
      0x50, 0x4b, 0x05, 0x06, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00,
      0x86, 0x00, 0x00, 0x00, 0x8d, 0x00, 0x00, 0x00, 0x00, 0x00};

  riegeli::StringReader string_reader(StringViewOf(kThreeNamesZip));
  EasyZipReader zip_reader(string_reader);
  auto entries_result = zip_reader.entries();
  EXPECT_FALSE(entries_result.ok());
}

TEST(ZipSecurityTest, DifferentZipMismatchedLfhRejected) {
  std::string bytes;
  ZipEOCD eocd;
  {
    riegeli::StringWriter writer(&bytes);
    EasyZipWriter zip_writer(writer);

    ZipEntry entry;
    entry.filename = "file1.txt";
    entry.compression_method = ZipCompression::kStore;
    TENSORSTORE_ASSERT_OK(zip_writer.WriteEntry(entry, absl::Cord("data")));

    TENSORSTORE_ASSERT_OK(zip_writer.Finalize(&eocd));
    ASSERT_TRUE(writer.Close());
  }

  // Modify the central directory entry filename.
  // The filename in CDH starts exactly 46 bytes after cdh_offset.
  ASSERT_LT(eocd.cd_offset + 46 + 9, bytes.size());
  bytes.replace(eocd.cd_offset + 46, 9, "clash.txt");

  riegeli::StringReader string_reader(bytes);
  EasyZipReader zip_reader(string_reader);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto entries, zip_reader.entries());
  ASSERT_GE(entries.size(), 1);

  ZipEntry mutable_entry = entries[0];
  auto read_result = zip_reader.ReadEntry(mutable_entry);
  EXPECT_FALSE(read_result.ok());
  EXPECT_THAT(
      read_result.status().message(),
      ::testing::HasSubstr("does not match central directory filename"));
}

TEST(ZipDetailsTest, WriteNonAsciiEntryAutoSetsEfs) {
  ZipEntry entry;
  entry.filename = "файлы.txt";
  entry.compression_method = ZipCompression::kStore;

  std::string bytes;
  riegeli::StringWriter writer(&bytes);
  TENSORSTORE_ASSERT_OK(WriteLocalEntry(writer, entry));

  EXPECT_TRUE(::tensorstore::internal_zip::HasZipGeneralFlag(
      entry.flags,
      ::tensorstore::internal_zip::ZipGeneralFlags::kLanguageEncoding));
}

TEST(ZipDetailsTest, ReadInfoZipUnicodePathSuccess) {
  std::string file_path = GetTestZipPath("infozip_unicode_path.zip");
  riegeli::FdReader reader(file_path);
  ASSERT_TRUE(reader.ok()) << reader.status();

  EasyZipReader zip_reader(reader);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto entries, zip_reader.entries());
  ASSERT_EQ(entries.size(), 1);
  EXPECT_EQ(entries[0].filename, "utf8_name.txt");
  EXPECT_EQ(entries[0].uncompressed_size, 14);

  ZipEntry mutable_entry = entries[0];
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto content,
                                   zip_reader.ReadEntry(mutable_entry));
  EXPECT_EQ(content, "Hello Unicode!");
}

}  // namespace
