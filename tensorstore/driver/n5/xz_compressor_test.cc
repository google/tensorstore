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

TEST(XzCompressionTest, Parse) {
  // Parse without any options.
  {
    auto c = Compressor::FromJson({{"type", "xz"}});
    EXPECT_EQ(Status(), GetStatus(c));
    EXPECT_EQ(::nlohmann::json({{"type", "xz"}, {"preset", 6}}),
              ::nlohmann::json(*c));
  }

  // Parse with preset option.
  {
    auto c = Compressor::FromJson({{"type", "xz"}, {"preset", 3}});
    EXPECT_EQ(Status(), GetStatus(c));
    EXPECT_EQ(::nlohmann::json({{"type", "xz"}, {"preset", 3}}),
              ::nlohmann::json(*c));
  }

  // Invalid preset option type
  EXPECT_THAT(Compressor::FromJson({{"type", "xz"}, {"preset", "x"}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));

  // Invalid preset option value
  EXPECT_THAT(Compressor::FromJson({{"type", "xz"}, {"preset", -1}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(Compressor::FromJson({{"type", "xz"}, {"preset", 10}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));

  // Invalid extra option
  EXPECT_THAT(Compressor::FromJson({{"type", "xz"}, {"extra", "x"}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

// Xz chunk example from the specification:
// https://github.com/saalfeldlab/n5#file-system-specification-version-203-snapshot
TEST(XzCompressionTest, Golden) {
  const unsigned char kData[] = {
      0x00, 0x00,              //
      0x00, 0x03,              //
      0x00, 0x00, 0x00, 0x01,  //
      0x00, 0x00, 0x00, 0x02,  //
      0x00, 0x00, 0x00, 0x03,  //
      0xfd, 0x37, 0x7a, 0x58,  //
      0x5a, 0x00, 0x00, 0x04,  //
      0xe6, 0xd6, 0xb4, 0x46,  //
      0x02, 0x00, 0x21, 0x01,  //
      0x16, 0x00, 0x00, 0x00,  //
      0x74, 0x2f, 0xe5, 0xa3,  //
      0x01, 0x00, 0x0b, 0x00,  //
      0x01, 0x00, 0x02, 0x00,  //
      0x03, 0x00, 0x04, 0x00,  //
      0x05, 0x00, 0x06, 0x00,  //
      0x0d, 0x03, 0x09, 0xca,  //
      0x34, 0xec, 0x15, 0xa7,  //
      0x00, 0x01, 0x24, 0x0c,  //
      0xa6, 0x18, 0xd8, 0xd8,  //
      0x1f, 0xb6, 0xf3, 0x7d,  //
      0x01, 0x00, 0x00, 0x00,  //
      0x00, 0x04, 0x59, 0x5a,  //
  };

  std::string encoded_data(std::begin(kData), std::end(kData));
  auto metadata = N5Metadata::Parse({{"dimensions", {10, 11, 12}},
                                     {"blockSize", {1, 2, 3}},
                                     {"dataType", "uint16"},
                                     {"compression", {{"type", "xz"}}}})
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
