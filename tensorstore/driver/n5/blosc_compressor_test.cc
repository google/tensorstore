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

#include <string_view>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/array.h"
#include "tensorstore/driver/n5/compressor.h"
#include "tensorstore/driver/n5/metadata.h"
#include "tensorstore/internal/json_binding/gtest.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::Index;
using ::tensorstore::MakeArray;
using ::tensorstore::MatchesStatus;
using ::tensorstore::span;
using ::tensorstore::internal_n5::Compressor;
using ::tensorstore::internal_n5::DecodeChunk;
using ::tensorstore::internal_n5::N5Metadata;

TEST(BloscCompressionTest, Parse) {
  for (auto codec : {"lz4", "blosclz", "lz4hc", "snappy", "zlib", "zstd"}) {
    for (int level = 0; level <= 9; ++level) {
      for (int shuffle = 0; shuffle <= 2; ++shuffle) {
        for (int blocksize : {0, 256}) {
          ::nlohmann::json j{{"type", "blosc"},
                             {"cname", codec},
                             {"shuffle", shuffle},
                             {"clevel", level},
                             {"blocksize", blocksize}};
          tensorstore::TestJsonBinderRoundTripJsonOnly<Compressor>({j});
        }
      }
    }
  }

  // Missing codec
  EXPECT_THAT(
      Compressor::FromJson({{"type", "blosc"}, {"shuffle", 0}, {"clevel", 5}}),
      MatchesStatus(absl::StatusCode::kInvalidArgument));

  // Missing shuffle
  EXPECT_THAT(Compressor::FromJson(
                  {{"type", "blosc"}, {"cname", "lz4"}, {"clevel", 5}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));

  // Missing level
  EXPECT_THAT(Compressor::FromJson(
                  {{"type", "blosc"}, {"cname", "lz4"}, {"shuffle", 0}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));

  // Invalid codec option type
  EXPECT_THAT(
      Compressor::FromJson(
          {{"type", "blosc"}, {"cname", 3}, {"shuffle", 0}, {"clevel", 5}}),
      MatchesStatus(absl::StatusCode::kInvalidArgument));

  // Invalid codec option value
  EXPECT_THAT(Compressor::FromJson({{"type", "blosc"},
                                    {"cname", "invalid"},
                                    {"shuffle", 0},
                                    {"clevel", 5}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));

  // Invalid level option value
  EXPECT_THAT(Compressor::FromJson({{"type", "blosc"},
                                    {"cname", "lz4"},
                                    {"shuffle", 0},
                                    {"clevel", -1}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(Compressor::FromJson({{"type", "blosc"},
                                    {"cname", "lz4"},
                                    {"shuffle", 0},
                                    {"clevel", 10}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));

  // Invalid shuffle option value
  EXPECT_THAT(Compressor::FromJson({{"type", "blosc"},
                                    {"cname", "lz4"},
                                    {"shuffle", -1},
                                    {"clevel", 3}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      Compressor::FromJson(
          {{"type", "blosc"}, {"cname", "lz4"}, {"shuffle", 3}, {"clevel", 3}}),
      MatchesStatus(absl::StatusCode::kInvalidArgument));

  // Invalid extra option
  EXPECT_THAT(Compressor::FromJson({{"type", "blosc"},
                                    {"cname", "lz4"},
                                    {"shuffle", 0},
                                    {"clevel", 3},
                                    {"extra", 5}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST(BloscCompressionTest, RoundTrip) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto metadata, N5Metadata::FromJson({{"dimensions", {10, 11, 12}},
                                           {"blockSize", {1, 2, 3}},
                                           {"dataType", "uint16"},
                                           {"compression",
                                            {{"type", "blosc"},
                                             {"cname", "lz4"},
                                             {"clevel", 5},
                                             {"shuffle", 0}}}}));
  auto array = MakeArray<std::uint16_t>({{{1, 2, 3}, {4, 5, 6}}});

  // Verify round trip.
  {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto buffer,
        EncodeChunk(span<const Index>({0, 0, 0}), metadata, array));
    EXPECT_EQ(array, DecodeChunk(metadata, buffer));
  }
}

// Blosc chunk example generated using zarr n5
TEST(BloscCompressionTest, Golden) {
  const unsigned char kData[] = {
      0x00, 0x00,              //
      0x00, 0x03,              //
      0x00, 0x00, 0x00, 0x01,  //
      0x00, 0x00, 0x00, 0x02,  //
      0x00, 0x00, 0x00, 0x03,  //
      0x02, 0x01, 0x96, 0x02, 0x0c, 0x00, 0x00, 0x00, 0x0c, 0x00,
      0x00, 0x00, 0x1c, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x02,
      0x00, 0x03, 0x00, 0x04, 0x00, 0x05, 0x00, 0x06,
  };

  std::string encoded_data(std::begin(kData), std::end(kData));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto metadata,
                                   N5Metadata::FromJson({
                                       {"dimensions", {10, 11, 12}},
                                       {"blockSize", {1, 2, 3}},
                                       {"dataType", "uint16"},
                                       {"compression",
                                        {
                                            {"type", "blosc"},
                                            {"clevel", 3},
                                            {"blocksize", 0},
                                            {"cname", "zstd"},
                                            {"shuffle", 2},
                                        }},
                                   }));
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
