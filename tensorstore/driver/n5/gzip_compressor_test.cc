// Copyright 2020 The TensorStore Authors
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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "tensorstore/array.h"
#include "tensorstore/driver/n5/compressor.h"
#include "tensorstore/driver/n5/metadata.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using tensorstore::Index;
using tensorstore::MakeArray;
using tensorstore::MatchesStatus;
using tensorstore::span;
using tensorstore::Status;
using tensorstore::internal_n5::Compressor;
using tensorstore::internal_n5::DecodeChunk;
using tensorstore::internal_n5::N5Metadata;

TEST(GzipCompressionTest, Parse) {
  // Parse without any options.
  {
    auto c = Compressor::FromJson({{"type", "gzip"}});
    EXPECT_EQ(Status(), GetStatus(c));
    EXPECT_EQ(
        ::nlohmann::json({{"type", "gzip"}, {"level", -1}, {"useZlib", false}}),
        ::nlohmann::json(*c));
  }

  // Parse with level option.
  {
    auto c = Compressor::FromJson({{"type", "gzip"}, {"level", 3}});
    EXPECT_EQ(Status(), GetStatus(c));
    EXPECT_EQ(
        ::nlohmann::json({{"type", "gzip"}, {"level", 3}, {"useZlib", false}}),
        ::nlohmann::json(*c));
  }

  // Parse with useZlib=true option.
  {
    auto c = Compressor::FromJson({{"type", "gzip"}, {"useZlib", true}});
    EXPECT_EQ(Status(), GetStatus(c));
    EXPECT_EQ(
        ::nlohmann::json({{"type", "gzip"}, {"level", -1}, {"useZlib", true}}),
        ::nlohmann::json(*c));
  }

  // Parse with level and useZlib options.
  {
    auto c = Compressor::FromJson(
        {{"type", "gzip"}, {"level", 3}, {"useZlib", false}});
    EXPECT_EQ(Status(), GetStatus(c));
    EXPECT_EQ(
        ::nlohmann::json({{"type", "gzip"}, {"level", 3}, {"useZlib", false}}),
        ::nlohmann::json(*c));
  }

  // Invalid level option type
  EXPECT_THAT(Compressor::FromJson({{"type", "gzip"}, {"level", "x"}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));

  // Invalid level option value
  EXPECT_THAT(Compressor::FromJson({{"type", "gzip"}, {"level", -2}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(Compressor::FromJson({{"type", "gzip"}, {"level", 10}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));

  // Invalid useZlib option
  EXPECT_THAT(Compressor::FromJson({{"type", "gzip"}, {"useZlib", "x"}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));

  // Invalid extra option
  EXPECT_THAT(Compressor::FromJson({{"type", "gzip"}, {"extra", "x"}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

// Gzip chunk example from the specification:
// https://github.com/saalfeldlab/n5#file-system-specification-version-203-snapshot
TEST(GzipCompressionTest, Golden) {
  const unsigned char kData[] = {
      0x00, 0x00,              //
      0x00, 0x03,              //
      0x00, 0x00, 0x00, 0x01,  //
      0x00, 0x00, 0x00, 0x02,  //
      0x00, 0x00, 0x00, 0x03,  //
      0x1f, 0x8b, 0x08, 0x00,  //
      0x00, 0x00, 0x00, 0x00,  //
      0x00, 0x00, 0x63, 0x60,  //
      0x64, 0x60, 0x62, 0x60,  //
      0x66, 0x60, 0x61, 0x60,  //
      0x65, 0x60, 0x03, 0x00,  //
      0xaa, 0xea, 0x6d, 0xbf,  //
      0x0c, 0x00, 0x00, 0x00,  //
  };

  std::string encoded_data(std::begin(kData), std::end(kData));
  auto metadata = N5Metadata::Parse({{"dimensions", {10, 11, 12}},
                                     {"blockSize", {1, 2, 3}},
                                     {"dataType", "uint16"},
                                     {"compression", {{"type", "gzip"}}}})
                      .value();
  auto array = MakeArray<std::uint16_t>({{{1, 3, 5}, {2, 4, 6}}});
  EXPECT_EQ(array, DecodeChunk(metadata, absl::Cord(encoded_data)));

  // Verify round trip.
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto buffer,
        EncodeChunk(span<const Index>({0, 0, 0}), metadata, array));
    EXPECT_EQ(array, DecodeChunk(metadata, buffer));
  }
}

}  // namespace
