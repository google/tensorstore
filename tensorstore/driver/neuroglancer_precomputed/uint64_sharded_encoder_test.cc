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

#include "tensorstore/driver/neuroglancer_precomputed/uint64_sharded_encoder.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/driver/neuroglancer_precomputed/uint64_sharded.h"
#include "tensorstore/internal/compression/zlib.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

namespace zlib = tensorstore::zlib;
using tensorstore::Status;
using tensorstore::neuroglancer_uint64_sharded::EncodeMinishardIndex;
using tensorstore::neuroglancer_uint64_sharded::EncodeShardIndex;
using tensorstore::neuroglancer_uint64_sharded::MinishardIndexEntry;
using tensorstore::neuroglancer_uint64_sharded::ShardEncoder;
using tensorstore::neuroglancer_uint64_sharded::ShardIndexEntry;
using tensorstore::neuroglancer_uint64_sharded::ShardingSpec;

absl::Cord Bytes(std::vector<unsigned char> bytes) {
  return absl::Cord(std::string_view(
      reinterpret_cast<const char*>(bytes.data()), bytes.size()));
}

TEST(EncodeMinishardIndexTest, Empty) {
  auto out = EncodeMinishardIndex({});
  EXPECT_EQ("", out);
}

TEST(EncodeMinishardIndexTest, SingleEntry) {
  auto out = EncodeMinishardIndex(
      std::vector<MinishardIndexEntry>{{{0x0123456789abcdef}, {0x11, 0x23}}});
  EXPECT_THAT(out, Bytes({
                       0xef, 0xcd, 0xab, 0x89, 0x67, 0x45, 0x23, 0x01,  //
                       0x11, 0,    0,    0,    0,    0,    0,    0,     //
                       0x12, 0,    0,    0,    0,    0,    0,    0,     //
                   }));
}

TEST(EncodeMinishardIndexTest, MultipleEntries) {
  auto out = EncodeMinishardIndex(std::vector<MinishardIndexEntry>{
      {{1}, {3, 10}},
      {{7}, {12, 15}},
  });
  EXPECT_THAT(out, Bytes({
                       1, 0, 0, 0, 0, 0, 0, 0,  // chunk_id[0]=1
                       6, 0, 0, 0, 0, 0, 0, 0,  // chunk_id[1]=1+6
                       3, 0, 0, 0, 0, 0, 0, 0,  // start[0]   =3
                       2, 0, 0, 0, 0, 0, 0, 0,  // start[1]   =10+2
                       7, 0, 0, 0, 0, 0, 0, 0,  // size[0]    =7
                       3, 0, 0, 0, 0, 0, 0, 0,  // size[1]    =3
                   }));
}

TEST(EncodeShardIndexTest, Basic) {
  std::vector<ShardIndexEntry> shard_index{{1, 5}, {7, 10}};
  auto out = EncodeShardIndex(shard_index);
  EXPECT_THAT(out, Bytes({
                       1,  0, 0, 0, 0, 0, 0, 0,  // start[0]
                       5,  0, 0, 0, 0, 0, 0, 0,  // end[0]
                       7,  0, 0, 0, 0, 0, 0, 0,  // start[1]
                       10, 0, 0, 0, 0, 0, 0, 0,  // end[1]
                   }));
}

TEST(ShardEncoderTest, Raw) {
  ::nlohmann::json sharding_spec_json{
      {"@type", "neuroglancer_uint64_sharded_v1"},
      {"hash", "identity"},
      {"preshift_bits", 0},
      {"minishard_bits", 1},
      {"shard_bits", 0},
      {"data_encoding", "raw"},
      {"minishard_index_encoding", "raw"}};
  ShardingSpec sharding_spec =
      ShardingSpec::FromJson(sharding_spec_json).value();
  absl::Cord encoded_shard_data;
  ShardEncoder shard_encoder(sharding_spec, encoded_shard_data);
  TENSORSTORE_ASSERT_OK(shard_encoder.WriteIndexedEntry(0, {2},
                                                        Bytes({1, 2, 3, 4}),
                                                        /*compress=*/false));
  TENSORSTORE_ASSERT_OK(shard_encoder.WriteIndexedEntry(0, {8},
                                                        Bytes({6, 7, 8}),
                                                        /*compress=*/false));
  TENSORSTORE_ASSERT_OK(shard_encoder.WriteIndexedEntry(1, {3}, Bytes({9, 10}),
                                                        /*compress=*/false));
  auto encoded_shard_index = shard_encoder.Finalize().value();
  EXPECT_THAT(encoded_shard_data,
              Bytes({
                  1,  2,  3, 4,              //
                  6,  7,  8,                 //
                  2,  0,  0, 0, 0, 0, 0, 0,  // chunk[0]=2
                  6,  0,  0, 0, 0, 0, 0, 0,  // chunk[1]=8=2+6
                  0,  0,  0, 0, 0, 0, 0, 0,  // start[0]=0
                  0,  0,  0, 0, 0, 0, 0, 0,  // start[1]=0
                  4,  0,  0, 0, 0, 0, 0, 0,  // size[0] =4
                  3,  0,  0, 0, 0, 0, 0, 0,  // size[1] =3
                  9,  10,                    //
                  3,  0,  0, 0, 0, 0, 0, 0,  // chunk[0]=3
                  55, 0,  0, 0, 0, 0, 0, 0,  // start[0]=55
                  2,  0,  0, 0, 0, 0, 0, 0,  // size[0] =2
              }));
  EXPECT_THAT(encoded_shard_index,  //
              Bytes({
                  7,  0, 0, 0, 0, 0, 0, 0,  //
                  55, 0, 0, 0, 0, 0, 0, 0,  //
                  57, 0, 0, 0, 0, 0, 0, 0,  //
                  81, 0, 0, 0, 0, 0, 0, 0,  //
              }));
}

TEST(ShardEncoderTest, Gzip) {
  ::nlohmann::json sharding_spec_json{
      {"@type", "neuroglancer_uint64_sharded_v1"},
      {"hash", "identity"},
      {"preshift_bits", 0},
      {"minishard_bits", 1},
      {"shard_bits", 0},
      {"data_encoding", "gzip"},
      {"minishard_index_encoding", "gzip"}};
  ShardingSpec sharding_spec =
      ShardingSpec::FromJson(sharding_spec_json).value();
  absl::Cord encoded_shard_data;
  ShardEncoder shard_encoder(sharding_spec, encoded_shard_data);
  TENSORSTORE_ASSERT_OK(shard_encoder.WriteIndexedEntry(0, {2},
                                                        Bytes({1, 2, 3, 4}),
                                                        /*compress=*/true));
  TENSORSTORE_ASSERT_OK(shard_encoder.WriteIndexedEntry(0, {8},
                                                        Bytes({6, 7, 8}),
                                                        /*compress=*/true));
  TENSORSTORE_ASSERT_OK(shard_encoder.WriteIndexedEntry(1, {3}, Bytes({9, 10}),
                                                        /*compress=*/false));
  absl::Cord encoded_shard_index = shard_encoder.Finalize().value();
  absl::Cord expected_shard_data;

  zlib::Options options{/*.level=*/9, /*.use_gzip_header=*/true};
  std::vector<ShardIndexEntry> shard_index(2);
  {
    std::vector<MinishardIndexEntry> minishard_index(2);
    minishard_index[0].chunk_id = {2};
    minishard_index[0].byte_range.inclusive_min = expected_shard_data.size();
    zlib::Encode(Bytes({1, 2, 3, 4}), &expected_shard_data, options);
    minishard_index[0].byte_range.exclusive_max = expected_shard_data.size();
    minishard_index[1].chunk_id = {8};
    minishard_index[1].byte_range.inclusive_min = expected_shard_data.size();
    zlib::Encode(Bytes({6, 7, 8}), &expected_shard_data, options);
    minishard_index[1].byte_range.exclusive_max = expected_shard_data.size();
    shard_index[0].inclusive_min = expected_shard_data.size();
    zlib::Encode(EncodeMinishardIndex(minishard_index), &expected_shard_data,
                 options);
    shard_index[0].exclusive_max = expected_shard_data.size();
  }
  {
    std::vector<MinishardIndexEntry> minishard_index(1);
    minishard_index[0].chunk_id = {3};
    minishard_index[0].byte_range.inclusive_min = expected_shard_data.size();
    expected_shard_data.Append(Bytes({9, 10}));
    minishard_index[0].byte_range.exclusive_max = expected_shard_data.size();
    shard_index[1].inclusive_min = expected_shard_data.size();
    zlib::Encode(EncodeMinishardIndex(minishard_index), &expected_shard_data,
                 options);
    shard_index[1].exclusive_max = expected_shard_data.size();
  }
  auto expected_shard_index = EncodeShardIndex(shard_index);

  EXPECT_EQ(expected_shard_data, encoded_shard_data);
  EXPECT_EQ(expected_shard_index, encoded_shard_index);
}

}  // namespace
