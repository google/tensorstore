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

TEST(Bzip2CompressionTest, Parse) {
  // Parse without any options.
  {
    auto c = Compressor::FromJson({{"type", "bzip2"}});
    EXPECT_EQ(Status(), GetStatus(c));
    EXPECT_EQ(::nlohmann::json({{"type", "bzip2"}, {"blockSize", 9}}),
              ::nlohmann::json(*c));
  }

  // Parse with blockSize option.
  {
    auto c = Compressor::FromJson({{"type", "bzip2"}, {"blockSize", 3}});
    EXPECT_EQ(Status(), GetStatus(c));
    EXPECT_EQ(::nlohmann::json({{"type", "bzip2"}, {"blockSize", 3}}),
              ::nlohmann::json(*c));
  }

  // Invalid blockSize option type
  EXPECT_THAT(Compressor::FromJson({{"type", "bzip2"}, {"blockSize", "x"}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));

  // Invalid blockSize option value
  EXPECT_THAT(Compressor::FromJson({{"type", "bzip2"}, {"blockSize", 0}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(Compressor::FromJson({{"type", "bzip2"}, {"blockSize", 10}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));

  // Invalid extra option
  EXPECT_THAT(Compressor::FromJson({{"type", "bzip2"}, {"extra", "x"}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

// Bzip2 chunk example from the specification:
// https://github.com/saalfeldlab/n5#file-system-specification-version-203-snapshot
TEST(Bzip2CompressionTest, Golden) {
  const unsigned char kData[] = {
      0x00, 0x00,              //
      0x00, 0x03,              //
      0x00, 0x00, 0x00, 0x01,  //
      0x00, 0x00, 0x00, 0x02,  //
      0x00, 0x00, 0x00, 0x03,  //
      0x42, 0x5a, 0x68, 0x39,  //
      0x31, 0x41, 0x59, 0x26,  //
      0x53, 0x59, 0x02, 0x3e,  //
      0x0d, 0xd2, 0x00, 0x00,  //
      0x00, 0x40, 0x00, 0x7f,  //
      0x00, 0x20, 0x00, 0x31,  //
      0x0c, 0x01, 0x0d, 0x31,  //
      0xa8, 0x73, 0x94, 0x33,  //
      0x7c, 0x5d, 0xc9, 0x14,  //
      0xe1, 0x42, 0x40, 0x08,  //
      0xf8, 0x37, 0x48,        //
  };

  std::string encoded_data(std::begin(kData), std::end(kData));
  auto metadata = N5Metadata::Parse({{"dimensions", {10, 11, 12}},
                                     {"blockSize", {1, 2, 3}},
                                     {"dataType", "uint16"},
                                     {"compression", {{"type", "bzip2"}}}})
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
