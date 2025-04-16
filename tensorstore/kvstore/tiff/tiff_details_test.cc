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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <stddef.h>
#include <stdint.h>

#include <string>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "riegeli/bytes/cord_reader.h"
#include "riegeli/bytes/string_reader.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::internal_tiff_kvstore::Endian;
using ::tensorstore::internal_tiff_kvstore::GetTiffDataTypeSize;
using ::tensorstore::internal_tiff_kvstore::IfdEntry;
using ::tensorstore::internal_tiff_kvstore::ImageDirectory;
using ::tensorstore::internal_tiff_kvstore::IsExternalArray;
using ::tensorstore::internal_tiff_kvstore::ParseExternalArray;
using ::tensorstore::internal_tiff_kvstore::ParseImageDirectory;
using ::tensorstore::internal_tiff_kvstore::ParseTiffDirectory;
using ::tensorstore::internal_tiff_kvstore::ParseTiffHeader;
using ::tensorstore::internal_tiff_kvstore::ParseUint16Array;
using ::tensorstore::internal_tiff_kvstore::Tag;
using ::tensorstore::internal_tiff_kvstore::TiffDataType;
using ::tensorstore::internal_tiff_kvstore::TiffDirectory;

TEST(TiffDetailsTest, ParseValidTiffHeader) {
  // Create a minimal valid TIFF header (II, 42, offset 8)
  static constexpr unsigned char kHeader[] = {
      'I', 'I',        // Little endian
      42,  0,          // Magic number (little endian)
      8,   0,   0, 0,  // Offset to first IFD (little endian)
  };

  riegeli::StringReader reader(std::string_view(
      reinterpret_cast<const char*>(kHeader), sizeof(kHeader)));

  Endian endian;
  uint64_t first_ifd_offset;
  ASSERT_THAT(ParseTiffHeader(reader, endian, first_ifd_offset),
              ::tensorstore::IsOk());
  EXPECT_EQ(endian, Endian::kLittle);
  EXPECT_EQ(first_ifd_offset, 8);
}

TEST(TiffDetailsTest, ParseBadByteOrder) {
  // Create an invalid TIFF header with wrong byte order marker
  static constexpr unsigned char kHeader[] = {
      'X', 'X',        // Invalid byte order
      42,  0,          // Magic number
      8,   0,   0, 0,  // Offset to first IFD
  };

  riegeli::StringReader reader(std::string_view(
      reinterpret_cast<const char*>(kHeader), sizeof(kHeader)));

  Endian endian;
  uint64_t first_ifd_offset;
  EXPECT_THAT(ParseTiffHeader(reader, endian, first_ifd_offset),
              ::tensorstore::MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST(TiffDetailsTest, ParseBadMagic) {
  // Create an invalid TIFF header with wrong magic number
  static constexpr unsigned char kHeader[] = {
      'I', 'I',        // Little endian
      43,  0,          // Wrong magic number
      8,   0,   0, 0,  // Offset to first IFD
  };

  riegeli::StringReader reader(std::string_view(
      reinterpret_cast<const char*>(kHeader), sizeof(kHeader)));

  Endian endian;
  uint64_t first_ifd_offset;
  EXPECT_THAT(ParseTiffHeader(reader, endian, first_ifd_offset),
              ::tensorstore::MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST(TiffDetailsTest, ParseValidDirectory) {
  // Create a minimal valid IFD with one entry
  static constexpr unsigned char kIfd[] = {
      1,   0,        // Number of entries
      0,   1,        // Tag (ImageWidth = 256)
      3,   0,        // Type (SHORT)
      1,   0, 0, 0,  // Count
      100, 0, 0, 0,  // Value (100)
      0,   0, 0, 0,  // Next IFD offset (0 = no more)
  };

  riegeli::StringReader reader(
      std::string_view(reinterpret_cast<const char*>(kIfd), sizeof(kIfd)));

  TiffDirectory dir;
  ASSERT_THAT(ParseTiffDirectory(reader, Endian::kLittle, 0, sizeof(kIfd), dir),
              ::tensorstore::IsOk());

  EXPECT_EQ(dir.entries.size(), 1);
  EXPECT_EQ(dir.next_ifd_offset, 0);

  const auto& entry = dir.entries[0];
  EXPECT_EQ(entry.tag, Tag::kImageWidth);
  EXPECT_EQ(entry.type, TiffDataType::kShort);
  EXPECT_EQ(entry.count, 1);
  EXPECT_EQ(entry.value_or_offset, 100);
}

TEST(TiffDetailsTest, ParseTruncatedDirectory) {
  // Create a truncated IFD
  static constexpr unsigned char kTruncatedIfd[] = {
      1, 0,  // Number of entries
      1, 0,  // Tag (partial entry)
  };

  riegeli::StringReader reader(std::string_view(
      reinterpret_cast<const char*>(kTruncatedIfd), sizeof(kTruncatedIfd)));

  TiffDirectory dir;
  EXPECT_THAT(ParseTiffDirectory(reader, Endian::kLittle, 0,
                                 sizeof(kTruncatedIfd), dir),
              ::tensorstore::MatchesStatus(absl::StatusCode::kDataLoss));
}

TEST(TiffDetailsTest, ParseImageDirectory_Tiled_InlineOffsets_Success) {
  std::vector<IfdEntry> entries = {
      {Tag::kImageWidth, TiffDataType::kLong, 1, 800},        // ImageWidth
      {Tag::kImageLength, TiffDataType::kLong, 1, 600},       // ImageLength
      {Tag::kTileWidth, TiffDataType::kLong, 1, 256},         // TileWidth
      {Tag::kTileLength, TiffDataType::kLong, 1, 256},        // TileLength
      {Tag::kTileOffsets, TiffDataType::kLong, 1, 1000},      // TileOffsets
      {Tag::kTileByteCounts, TiffDataType::kLong, 1, 65536},  // TileByteCounts
  };

  ImageDirectory dir;
  ASSERT_THAT(ParseImageDirectory(entries, dir), ::tensorstore::IsOk());

  EXPECT_EQ(dir.width, 800);
  EXPECT_EQ(dir.height, 600);
  EXPECT_EQ(dir.tile_width, 256);
  EXPECT_EQ(dir.tile_height, 256);
  ASSERT_EQ(dir.tile_offsets.size(), 1);
  EXPECT_EQ(dir.tile_offsets[0], 1000);
  ASSERT_EQ(dir.tile_bytecounts.size(), 1);
  EXPECT_EQ(dir.tile_bytecounts[0], 65536);
}

TEST(TiffDetailsTest, ParseImageDirectory_Stripped_InlineOffsets_Success) {
  std::vector<IfdEntry> entries = {
      {Tag::kImageWidth, TiffDataType::kLong, 1, 800},        // ImageWidth
      {Tag::kImageLength, TiffDataType::kLong, 1, 600},       // ImageLength
      {Tag::kRowsPerStrip, TiffDataType::kLong, 1, 100},      // RowsPerStrip
      {Tag::kStripOffsets, TiffDataType::kLong, 1, 1000},     // StripOffsets
      {Tag::kStripByteCounts, TiffDataType::kLong, 1, 8192},  // StripByteCounts
  };

  ImageDirectory dir;
  ASSERT_THAT(ParseImageDirectory(entries, dir), ::tensorstore::IsOk());

  EXPECT_EQ(dir.width, 800);
  EXPECT_EQ(dir.height, 600);
  EXPECT_EQ(dir.rows_per_strip, 100);
  ASSERT_EQ(dir.strip_offsets.size(), 1);
  EXPECT_EQ(dir.strip_offsets[0], 1000);
  ASSERT_EQ(dir.strip_bytecounts.size(), 1);
  EXPECT_EQ(dir.strip_bytecounts[0], 8192);
}

TEST(TiffDetailsTest, ParseImageDirectory_DuplicateTags) {
  std::vector<IfdEntry> entries = {
      {Tag::kImageWidth, TiffDataType::kLong, 1, 800},   // ImageWidth
      {Tag::kImageWidth, TiffDataType::kLong, 1, 1024},  // Duplicate ImageWidth
      {Tag::kImageLength, TiffDataType::kLong, 1, 600},  // ImageLength
  };

  ImageDirectory dir;
  EXPECT_THAT(ParseImageDirectory(entries, dir),
              ::tensorstore::MatchesStatus(absl::StatusCode::kNotFound));
}

TEST(TiffDetailsTest, GetTiffDataTypeSize) {
  // Test size of various TIFF data types
  EXPECT_EQ(GetTiffDataTypeSize(TiffDataType::kByte), 1);
  EXPECT_EQ(GetTiffDataTypeSize(TiffDataType::kAscii), 1);
  EXPECT_EQ(GetTiffDataTypeSize(TiffDataType::kShort), 2);
  EXPECT_EQ(GetTiffDataTypeSize(TiffDataType::kLong), 4);
  EXPECT_EQ(GetTiffDataTypeSize(TiffDataType::kRational), 8);
  EXPECT_EQ(GetTiffDataTypeSize(TiffDataType::kSbyte), 1);
  EXPECT_EQ(GetTiffDataTypeSize(TiffDataType::kUndefined), 1);
  EXPECT_EQ(GetTiffDataTypeSize(TiffDataType::kSshort), 2);
  EXPECT_EQ(GetTiffDataTypeSize(TiffDataType::kSlong), 4);
  EXPECT_EQ(GetTiffDataTypeSize(TiffDataType::kSrational), 8);
  EXPECT_EQ(GetTiffDataTypeSize(TiffDataType::kFloat), 4);
  EXPECT_EQ(GetTiffDataTypeSize(TiffDataType::kDouble), 8);
  EXPECT_EQ(GetTiffDataTypeSize(TiffDataType::kIfd), 4);
  EXPECT_EQ(GetTiffDataTypeSize(TiffDataType::kLong8), 8);
  EXPECT_EQ(GetTiffDataTypeSize(TiffDataType::kSlong8), 8);
  EXPECT_EQ(GetTiffDataTypeSize(TiffDataType::kIfd8), 8);

  // Test with invalid type
  EXPECT_EQ(GetTiffDataTypeSize(static_cast<TiffDataType>(999)), 0);
}

TEST(TiffDetailsTest, IsExternalArray) {
  // Test with data that fits in 4 bytes (inline)
  EXPECT_FALSE(IsExternalArray(TiffDataType::kLong, 1));   // 4 bytes
  EXPECT_FALSE(IsExternalArray(TiffDataType::kShort, 2));  // 4 bytes
  EXPECT_FALSE(IsExternalArray(TiffDataType::kByte, 4));   // 4 bytes

  // Test with data that doesn't fit in 4 bytes (external)
  EXPECT_TRUE(IsExternalArray(TiffDataType::kLong, 2));      // 8 bytes
  EXPECT_TRUE(IsExternalArray(TiffDataType::kShort, 3));     // 6 bytes
  EXPECT_TRUE(IsExternalArray(TiffDataType::kByte, 5));      // 5 bytes
  EXPECT_TRUE(IsExternalArray(TiffDataType::kRational, 1));  // 8 bytes
}

TEST(TiffDetailsTest, ParseExternalArray) {
  // Create a buffer with four uint32 values in little-endian format
  static constexpr unsigned char kBuffer[] = {
      100, 0, 0, 0,  // 100 (uint32, little endian)
      200, 0, 0, 0,  // 200
      150, 0, 0, 0,  // 150
      250, 0, 0, 0,  // 250
  };

  riegeli::StringReader reader(std::string_view(
      reinterpret_cast<const char*>(kBuffer), sizeof(kBuffer)));

  std::vector<uint64_t> values;
  ASSERT_THAT(ParseExternalArray(reader, Endian::kLittle, 0, 4,
                                 TiffDataType::kLong, values),
              ::tensorstore::IsOk());

  ASSERT_EQ(values.size(), 4);
  EXPECT_EQ(values[0], 100);
  EXPECT_EQ(values[1], 200);
  EXPECT_EQ(values[2], 150);
  EXPECT_EQ(values[3], 250);
}

TEST(TiffDetailsTest, ParseExternalArray_SeekFail) {
  // Create a small buffer to test seek failure
  static constexpr unsigned char kBuffer[] = {1, 2, 3, 4};

  riegeli::StringReader reader(std::string_view(
      reinterpret_cast<const char*>(kBuffer), sizeof(kBuffer)));

  std::vector<uint64_t> values;
  // Try to seek beyond the buffer size
  EXPECT_THAT(ParseExternalArray(reader, Endian::kLittle, 100, 1,
                                 TiffDataType::kLong, values),
              ::tensorstore::MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST(TiffDetailsTest, ParseExternalArray_ReadFail) {
  // Create a buffer with incomplete data
  static constexpr unsigned char kBuffer[] = {100, 0, 0};  // Only 3 bytes

  riegeli::StringReader reader(std::string_view(
      reinterpret_cast<const char*>(kBuffer), sizeof(kBuffer)));

  std::vector<uint64_t> values;
  // Try to read a uint32 from a 3-byte buffer
  EXPECT_THAT(ParseExternalArray(reader, Endian::kLittle, 0, 1,
                                 TiffDataType::kLong, values),
              ::tensorstore::MatchesStatus(absl::StatusCode::kDataLoss));
}

TEST(TiffDetailsTest, ParseExternalArray_InvalidType) {
  // Create a small valid buffer
  static constexpr unsigned char kBuffer[] = {1, 2, 3, 4};

  riegeli::StringReader reader(std::string_view(
      reinterpret_cast<const char*>(kBuffer), sizeof(kBuffer)));

  std::vector<uint64_t> values;
  // Try with an unsupported type
  EXPECT_THAT(ParseExternalArray(reader, Endian::kLittle, 0, 1,
                                 TiffDataType::kRational, values),
              ::tensorstore::MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST(TiffDetailsTest, ParseUint16Array) {
  // Create a buffer with four uint16 values in little-endian format
  static constexpr unsigned char kBuffer[] = {
      100, 0,  // 100 (uint16, little endian)
      200, 0,  // 200
      150, 0,  // 150
      250, 0,  // 250
  };

  riegeli::StringReader reader(std::string_view(
      reinterpret_cast<const char*>(kBuffer), sizeof(kBuffer)));

  std::vector<uint16_t> values;
  ASSERT_THAT(ParseUint16Array(reader, Endian::kLittle, 0, 4, values),
              ::tensorstore::IsOk());

  ASSERT_EQ(values.size(), 4);
  EXPECT_EQ(values[0], 100);
  EXPECT_EQ(values[1], 200);
  EXPECT_EQ(values[2], 150);
  EXPECT_EQ(values[3], 250);
}

TEST(TiffDetailsTest, ParseUint16Array_SeekFail) {
  // Create a small buffer to test seek failure
  static constexpr unsigned char kBuffer[] = {1, 2, 3, 4};

  riegeli::StringReader reader(std::string_view(
      reinterpret_cast<const char*>(kBuffer), sizeof(kBuffer)));

  std::vector<uint16_t> values;
  // Try to seek beyond the buffer size
  EXPECT_THAT(ParseUint16Array(reader, Endian::kLittle, 100, 1, values),
              ::tensorstore::MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST(TiffDetailsTest, ParseUint16Array_ReadFail) {
  // Create a buffer with incomplete data
  static constexpr unsigned char kBuffer[] = {100};  // Only 1 byte

  riegeli::StringReader reader(std::string_view(
      reinterpret_cast<const char*>(kBuffer), sizeof(kBuffer)));

  std::vector<uint16_t> values;
  // Try to read a uint16 from a 1-byte buffer
  EXPECT_THAT(ParseUint16Array(reader, Endian::kLittle, 0, 1, values),
              ::tensorstore::MatchesStatus(absl::StatusCode::kDataLoss));
}

// Test for ParseImageDirectory with external arrays
TEST(TiffDetailsTest, ParseImageDirectory_ExternalArrays) {
  // Setup IFD entries with external arrays
  std::vector<IfdEntry> entries = {
      {Tag::kImageWidth, TiffDataType::kLong, 1, 800},   // ImageWidth
      {Tag::kImageLength, TiffDataType::kLong, 1, 600},  // ImageLength
      {Tag::kTileWidth, TiffDataType::kLong, 1, 256},    // TileWidth
      {Tag::kTileLength, TiffDataType::kLong, 1, 256},   // TileLength
      // External arrays (is_external_array = true)
      {Tag::kTileOffsets, TiffDataType::kLong, 4, 1000,
       true},  // TileOffsets (external)
      {Tag::kTileByteCounts, TiffDataType::kLong, 4, 2000,
       true},  // TileByteCounts (external)
      {Tag::kBitsPerSample, TiffDataType::kShort, 3, 3000,
       true},  // BitsPerSample (external)
      {Tag::kSamplesPerPixel, TiffDataType::kShort, 1,
       3},  // SamplesPerPixel (inline)
  };

  ImageDirectory dir;
  ASSERT_THAT(ParseImageDirectory(entries, dir), ::tensorstore::IsOk());

  EXPECT_EQ(dir.width, 800);
  EXPECT_EQ(dir.height, 600);
  EXPECT_EQ(dir.tile_width, 256);
  EXPECT_EQ(dir.tile_height, 256);
  EXPECT_EQ(dir.samples_per_pixel, 3);

  // External arrays should have the correct size but not be loaded yet
  ASSERT_EQ(dir.tile_offsets.size(), 4);
  ASSERT_EQ(dir.tile_bytecounts.size(), 4);
  ASSERT_EQ(dir.bits_per_sample.size(), 3);
}

}  // namespace