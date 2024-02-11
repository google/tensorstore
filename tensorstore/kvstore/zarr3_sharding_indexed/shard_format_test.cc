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

#include "tensorstore/kvstore/zarr3_sharding_indexed/shard_format.h"

#include <optional>
#include <string_view>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include <nlohmann/json.hpp>
#include "tensorstore/driver/zarr3/codec/codec_chain_spec.h"
#include "tensorstore/driver/zarr3/codec/codec_test_util.h"
#include "tensorstore/index.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::Index;
using ::tensorstore::MatchesStatus;
using ::tensorstore::Result;
using ::tensorstore::internal_zarr3::GetDefaultBytesCodecJson;
using ::tensorstore::internal_zarr3::ZarrCodecChainSpec;
using ::tensorstore::zarr3_sharding_indexed::DecodeShard;
using ::tensorstore::zarr3_sharding_indexed::EncodeShard;
using ::tensorstore::zarr3_sharding_indexed::ShardEntries;
using ::tensorstore::zarr3_sharding_indexed::ShardIndexLocation;
using ::tensorstore::zarr3_sharding_indexed::ShardIndexParameters;

Result<ShardIndexParameters> GetParams(
    ShardIndexLocation index_location, std::vector<Index> grid_shape,
    ::nlohmann::json::array_t index_codecs_json = {GetDefaultBytesCodecJson(),
                                                   {{"name", "crc32c"}}}) {
  TENSORSTORE_ASSIGN_OR_RETURN(auto index_codecs,
                               ZarrCodecChainSpec::FromJson(index_codecs_json));
  ShardIndexParameters p;
  p.index_location = index_location;
  TENSORSTORE_RETURN_IF_ERROR(p.Initialize(index_codecs, grid_shape));
  return p;
}

TEST(InitializeTest, Success) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto p,
                                   GetParams(ShardIndexLocation::kEnd, {2, 3}));
  EXPECT_EQ(6, p.num_entries);
  EXPECT_THAT(p.index_shape, ::testing::ElementsAre(2, 3, 2));
}

TEST(InitializeTest, InvalidIndexCodecs) {
  EXPECT_THAT(
      GetParams(ShardIndexLocation::kEnd, {2, 3},
                {GetDefaultBytesCodecJson(),
                 {{"name", "gzip"}, {"configuration", {{"level", 5}}}}}),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    ".*: only fixed-size encodings are supported"));
}

TEST(InitializeTest, InvalidGridShape) {
  EXPECT_THAT(
      GetParams(ShardIndexLocation::kEnd, {1024 * 1024 * 1024 + 1}),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "grid shape of .* has more than 1073741824 entries"));
}

TEST(EncodeShardTest, RoundTrip) {
  for (auto index_location :
       {ShardIndexLocation::kStart, ShardIndexLocation::kEnd}) {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto p, GetParams(index_location, {2, 3}));

    ShardEntries entries;
    entries.entries = {
        absl::Cord("(0, 0)"), absl::Cord("(0, 1)"), std::nullopt,  //
        std::nullopt,         absl::Cord("(1, 1)"), std::nullopt   //
    };
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto encoded, EncodeShard(entries, p));
    ASSERT_TRUE(encoded.has_value());

    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto decoded_entries,
                                     DecodeShard(*encoded, p));
    EXPECT_THAT(decoded_entries.entries,
                ::testing::ElementsAreArray(entries.entries));
  }
}

TEST(EncodeShardTest, RoundTripEmpty) {
  for (auto index_location :
       {ShardIndexLocation::kStart, ShardIndexLocation::kEnd}) {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto p, GetParams(index_location, {2, 3}));

    ShardEntries entries;
    entries.entries.resize(6);
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto encoded, EncodeShard(entries, p));
    ASSERT_FALSE(encoded.has_value());
  }
}

TEST(DecodeShardTest, TooShort) {
  absl::Cord encoded(std::string{1, 2, 3});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto p,
                                   GetParams(ShardIndexLocation::kEnd, {2}));
  EXPECT_THAT(DecodeShard(encoded, p),
              MatchesStatus(absl::StatusCode::kDataLoss,
                            "Existing shard has size of 3 bytes, but expected "
                            "at least .* bytes"));
}

TEST(DecodeShardTest, ByteRangeOutOfRange) {
  absl::Cord encoded(std::string{
      0, 0, 0, 0, 0, 0, 0, 0,   //
      17, 0, 0, 0, 0, 0, 0, 0,  //
  });
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto p, GetParams(ShardIndexLocation::kEnd, {1},
                        {{{"name", "bytes"},
                          {"configuration", {{"endian", "little"}}}}}));
  EXPECT_THAT(
      DecodeShard(encoded, p),
      MatchesStatus(absl::StatusCode::kDataLoss,
                    "Shard index entry 0 with byte range .* is invalid .*"));
}

TEST(DecodeShardTest, ByteRangeInvalid) {
  unsigned char data[] = {
      0xfe, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,  //
      1,    0,    0,    0,    0,    0,    0,    0,     //
  };
  absl::Cord encoded(
      std::string_view(reinterpret_cast<const char*>(data), sizeof(data)));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto p, GetParams(ShardIndexLocation::kEnd, {1},
                        {{{"name", "bytes"},
                          {"configuration", {{"endian", "little"}}}}}));
  EXPECT_THAT(DecodeShard(encoded, p),
              MatchesStatus(absl::StatusCode::kDataLoss,
                            "Invalid shard index entry 0 with .*"));
}

}  // namespace
