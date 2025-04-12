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

#include <string>
#include <string_view>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "riegeli/bytes/cord_reader.h"
#include "riegeli/bytes/string_reader.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::internal_tiff_kvstore::Endian;
using ::tensorstore::internal_tiff_kvstore::IfdEntry;
using ::tensorstore::internal_tiff_kvstore::ParseTiffDirectory;
using ::tensorstore::internal_tiff_kvstore::ParseTiffHeader;
using ::tensorstore::internal_tiff_kvstore::TiffDataType;
using ::tensorstore::internal_tiff_kvstore::TiffDirectory;

TEST(TiffDetailsTest, ParseValidTiffHeader) {
  // Create a minimal valid TIFF header (II, 42, offset 8)
  static constexpr unsigned char kHeader[] = {
      'I', 'I',             // Little endian
      42, 0,               // Magic number (little endian)
      8, 0, 0, 0,         // Offset to first IFD (little endian)
  };

  riegeli::StringReader reader(
      std::string_view(reinterpret_cast<const char*>(kHeader), sizeof(kHeader)));

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
      'X', 'X',             // Invalid byte order
      42, 0,               // Magic number
      8, 0, 0, 0,         // Offset to first IFD
  };

  riegeli::StringReader reader(
      std::string_view(reinterpret_cast<const char*>(kHeader), sizeof(kHeader)));

  Endian endian;
  uint64_t first_ifd_offset;
  EXPECT_THAT(ParseTiffHeader(reader, endian, first_ifd_offset),
              ::tensorstore::MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST(TiffDetailsTest, ParseBadMagic) {
  // Create an invalid TIFF header with wrong magic number
  static constexpr unsigned char kHeader[] = {
      'I', 'I',             // Little endian
      43, 0,               // Wrong magic number
      8, 0, 0, 0,         // Offset to first IFD
  };

  riegeli::StringReader reader(
      std::string_view(reinterpret_cast<const char*>(kHeader), sizeof(kHeader)));

  Endian endian;
  uint64_t first_ifd_offset;
  EXPECT_THAT(ParseTiffHeader(reader, endian, first_ifd_offset),
              ::tensorstore::MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST(TiffDetailsTest, ParseValidDirectory) {
  // Create a minimal valid IFD with one entry
  static constexpr unsigned char kIfd[] = {
      1, 0,                // Number of entries
      1, 0,                // Tag (ImageWidth)
      3, 0,                // Type (SHORT)
      1, 0, 0, 0,         // Count
      100, 0, 0, 0,       // Value (100)
      0, 0, 0, 0,         // Next IFD offset (0 = no more)
  };

  riegeli::StringReader reader(
      std::string_view(reinterpret_cast<const char*>(kIfd), sizeof(kIfd)));

  TiffDirectory dir;
  ASSERT_THAT(ParseTiffDirectory(reader, Endian::kLittle, 0, sizeof(kIfd), dir),
              ::tensorstore::IsOk());
  
  EXPECT_EQ(dir.entries.size(), 1);
  EXPECT_EQ(dir.next_ifd_offset, 0);
  
  const auto& entry = dir.entries[0];
  EXPECT_EQ(entry.tag, 1);
  EXPECT_EQ(entry.type, TiffDataType::kShort);
  EXPECT_EQ(entry.count, 1);
  EXPECT_EQ(entry.value_or_offset, 100);
}

TEST(TiffDetailsTest, ParseTruncatedDirectory) {
  // Create a truncated IFD
  static constexpr unsigned char kTruncatedIfd[] = {
      1, 0,                // Number of entries
      1, 0,                // Tag (partial entry)
  };

  riegeli::StringReader reader(
      std::string_view(reinterpret_cast<const char*>(kTruncatedIfd),
                      sizeof(kTruncatedIfd)));

  TiffDirectory dir;
  EXPECT_THAT(
      ParseTiffDirectory(reader, Endian::kLittle, 0, sizeof(kTruncatedIfd), dir),
      ::tensorstore::MatchesStatus(absl::StatusCode::kDataLoss));
}

}  // namespace