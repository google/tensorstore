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

#include "tensorstore/kvstore/neuroglancer_uint64_sharded/uint64_sharded.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

namespace {

using ::tensorstore::MatchesStatus;
using ::tensorstore::neuroglancer_uint64_sharded::MinishardIndexEntry;
using ::tensorstore::neuroglancer_uint64_sharded::ShardingSpec;

TEST(ShardingSpecTest, Comparison) {
  ShardingSpec a{
      /*.hash_function=*/ShardingSpec::HashFunction::identity,
      /*.preshift_bits=*/1,
      /*.minishard_bits=*/2,
      /*.shard_bits=*/3,
      /*.data_encoding=*/ShardingSpec::DataEncoding::raw,
      /*.minishard_index_encoding=*/ShardingSpec::DataEncoding::gzip,
  };

  // Differs from `a` in `hash_function`.
  ShardingSpec b{
      /*.hash_function=*/ShardingSpec::HashFunction::murmurhash3_x86_128,
      /*.preshift_bits=*/1,
      /*.minishard_bits=*/2,
      /*.shard_bits=*/3,
      /*.data_encoding=*/ShardingSpec::DataEncoding::raw,
      /*.minishard_index_encoding=*/ShardingSpec::DataEncoding::gzip,
  };

  // Differs from `a` in `preshift_bits`
  ShardingSpec c{
      /*.hash_function=*/ShardingSpec::HashFunction::identity,
      /*.preshift_bits=*/2,
      /*.minishard_bits=*/2,
      /*.shard_bits=*/3,
      /*.data_encoding=*/ShardingSpec::DataEncoding::raw,
      /*.minishard_index_encoding=*/ShardingSpec::DataEncoding::gzip,
  };

  // Differs from `a` in `minishard_bits`
  ShardingSpec d{
      /*.hash_function=*/ShardingSpec::HashFunction::identity,
      /*.preshift_bits=*/1,
      /*.minishard_bits=*/5,
      /*.shard_bits=*/3,
      /*.data_encoding=*/ShardingSpec::DataEncoding::raw,
      /*.minishard_index_encoding=*/ShardingSpec::DataEncoding::gzip,
  };

  // Differs from `a` in `shard_bits`
  ShardingSpec e{
      /*.hash_function=*/ShardingSpec::HashFunction::identity,
      /*.preshift_bits=*/1,
      /*.minishard_bits=*/2,
      /*.shard_bits=*/9,
      /*.data_encoding=*/ShardingSpec::DataEncoding::raw,
      /*.minishard_index_encoding=*/ShardingSpec::DataEncoding::gzip,
  };

  // Differs from `a` in `data_encoding`
  ShardingSpec f{
      /*.hash_function=*/ShardingSpec::HashFunction::identity,
      /*.preshift_bits=*/1,
      /*.minishard_bits=*/2,
      /*.shard_bits=*/3,
      /*.data_encoding=*/ShardingSpec::DataEncoding::gzip,
      /*.minishard_index_encoding=*/ShardingSpec::DataEncoding::gzip,
  };

  // Differs from `a` in `minishard_index_encoding`
  ShardingSpec g{
      /*.hash_function=*/ShardingSpec::HashFunction::identity,
      /*.preshift_bits=*/1,
      /*.minishard_bits=*/2,
      /*.shard_bits=*/3,
      /*.data_encoding=*/ShardingSpec::DataEncoding::raw,
      /*.minishard_index_encoding=*/ShardingSpec::DataEncoding::raw,
  };

  EXPECT_EQ(a, a);
  EXPECT_EQ(b, b);
  EXPECT_EQ(c, c);
  EXPECT_EQ(d, d);
  EXPECT_EQ(e, e);
  EXPECT_EQ(f, f);
  EXPECT_EQ(g, g);
  EXPECT_NE(a, b);
  EXPECT_NE(a, c);
  EXPECT_NE(a, d);
  EXPECT_NE(a, e);
  EXPECT_NE(a, f);
  EXPECT_NE(a, g);
}

TEST(ShardingSpecTest, ToJson) {
  ShardingSpec a{
      /*.hash_function=*/ShardingSpec::HashFunction::identity,
      /*.preshift_bits=*/1,
      /*.minishard_bits=*/2,
      /*.shard_bits=*/3,
      /*.data_encoding=*/ShardingSpec::DataEncoding::raw,
      /*.minishard_index_encoding=*/ShardingSpec::DataEncoding::gzip,
  };
  EXPECT_EQ(::nlohmann::json({{"@type", "neuroglancer_uint64_sharded_v1"},
                              {"hash", "identity"},
                              {"preshift_bits", 1},
                              {"minishard_bits", 2},
                              {"shard_bits", 3},
                              {"data_encoding", "raw"},
                              {"minishard_index_encoding", "gzip"}}),
            ::nlohmann::json(a));
}

TEST(ShardingSpecTest, Parse) {
  for (auto h : {ShardingSpec::HashFunction::identity,
                 ShardingSpec::HashFunction::murmurhash3_x86_128}) {
    EXPECT_THAT(
        ShardingSpec::FromJson({{"@type", "neuroglancer_uint64_sharded_v1"},
                                {"hash", ::nlohmann::json(h)},
                                {"preshift_bits", 1},
                                {"minishard_bits", 2},
                                {"shard_bits", 3},
                                {"data_encoding", "raw"},
                                {"minishard_index_encoding", "gzip"}}),
        ::testing::Optional(ShardingSpec{
            /*.hash_function=*/h,
            /*.preshift_bits=*/1,
            /*.minishard_bits=*/2,
            /*.shard_bits=*/3,
            /*.data_encoding=*/ShardingSpec::DataEncoding::raw,
            /*.minishard_index_encoding=*/ShardingSpec::DataEncoding::gzip,
        }));
  }

  // Tests that `data_encoding` defaults to `raw`.
  EXPECT_THAT(
      ShardingSpec::FromJson({{"@type", "neuroglancer_uint64_sharded_v1"},
                              {"hash", "murmurhash3_x86_128"},
                              {"preshift_bits", 1},
                              {"minishard_bits", 2},
                              {"shard_bits", 3},
                              {"minishard_index_encoding", "gzip"}}),
      ::testing::Optional(ShardingSpec{
          /*.hash_function=*/ShardingSpec::HashFunction::murmurhash3_x86_128,
          /*.preshift_bits=*/1,
          /*.minishard_bits=*/2,
          /*.shard_bits=*/3,
          /*.data_encoding=*/ShardingSpec::DataEncoding::raw,
          /*.minishard_index_encoding=*/ShardingSpec::DataEncoding::gzip,
      }));

  // Tests that `minishard_index_encoding` defaults to `raw`.
  EXPECT_THAT(
      ShardingSpec::FromJson({{"@type", "neuroglancer_uint64_sharded_v1"},
                              {"hash", "murmurhash3_x86_128"},
                              {"preshift_bits", 1},
                              {"minishard_bits", 2},
                              {"shard_bits", 3},
                              {"data_encoding", "gzip"}}),
      ::testing::Optional(ShardingSpec{
          /*.hash_function=*/ShardingSpec::HashFunction::murmurhash3_x86_128,
          /*.preshift_bits=*/1,
          /*.minishard_bits=*/2,
          /*.shard_bits=*/3,
          /*.data_encoding=*/ShardingSpec::DataEncoding::gzip,
          /*.minishard_index_encoding=*/ShardingSpec::DataEncoding::raw,
      }));

  // Tests that each of the following members is required.
  for (const char* k :
       {"@type", "hash", "preshift_bits", "minishard_bits", "shard_bits"}) {
    ::nlohmann::json j{{"@type", "neuroglancer_uint64_sharded_v1"},
                       {"hash", "murmurhash3_x86_128"},
                       {"preshift_bits", 1},
                       {"minishard_bits", 2},
                       {"shard_bits", 3},
                       {"minishard_index_encoding", "raw"},
                       {"data_encoding", "gzip"}};
    j.erase(k);
    EXPECT_THAT(ShardingSpec::FromJson(j),
                MatchesStatus(absl::StatusCode::kInvalidArgument));
    j[k] = nullptr;
    EXPECT_THAT(ShardingSpec::FromJson(j),
                MatchesStatus(absl::StatusCode::kInvalidArgument));
  }

  // Tests that `@type` must be "neuroglancer_uint64_sharded_v1".
  EXPECT_THAT(
      ShardingSpec::FromJson({{"@type", "neuroglancer_uint64_sharded_v2"},
                              {"hash", "murmurhash3_x86_128"},
                              {"preshift_bits", 1},
                              {"minishard_bits", 2},
                              {"shard_bits", 3},
                              {"minishard_index_encoding", "raw"},
                              {"data_encoding", "gzip"}}),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    ".*\"neuroglancer_uint64_sharded_v2\".*"));

  // Tests that an invalid `hash` leads to an error.
  EXPECT_THAT(
      ShardingSpec::FromJson({{"@type", "neuroglancer_uint64_sharded_v1"},
                              {"hash", "invalid_hash"},
                              {"preshift_bits", 1},
                              {"minishard_bits", 2},
                              {"shard_bits", 3},
                              {"minishard_index_encoding", "raw"},
                              {"data_encoding", "gzip"}}),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    ".*\"invalid_hash\".*"));

  // Tests that an invalid `data_encoding` leads to an error.
  EXPECT_THAT(
      ShardingSpec::FromJson({{"@type", "neuroglancer_uint64_sharded_v1"},
                              {"hash", "identity"},
                              {"preshift_bits", 1},
                              {"minishard_bits", 2},
                              {"shard_bits", 3},
                              {"minishard_index_encoding", "raw"},
                              {"data_encoding", 1234}}),
      MatchesStatus(absl::StatusCode::kInvalidArgument, ".*1234.*"));
  EXPECT_THAT(
      ShardingSpec::FromJson({{"@type", "neuroglancer_uint64_sharded_v1"},
                              {"hash", "identity"},
                              {"preshift_bits", 1},
                              {"minishard_bits", 2},
                              {"shard_bits", 3},
                              {"minishard_index_encoding", "raw"},
                              {"data_encoding", "invalid_encoding"}}),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    ".*\"invalid_encoding\".*"));

  // Tests that `preshift_bits` is limited to `[0, 64]`.
  for (int i : {0, 1, 63, 64}) {
    EXPECT_THAT(
        ShardingSpec::FromJson({{"@type", "neuroglancer_uint64_sharded_v1"},
                                {"hash", "identity"},
                                {"preshift_bits", i},
                                {"minishard_bits", 2},
                                {"shard_bits", 3}}),
        ::testing::Optional(ShardingSpec{
            /*.hash_function=*/ShardingSpec::HashFunction::identity,
            /*.preshift_bits=*/i,
            /*.minishard_bits=*/2,
            /*.shard_bits=*/3,
            /*.data_encoding=*/ShardingSpec::DataEncoding::raw,
            /*.minishard_index_encoding=*/ShardingSpec::DataEncoding::raw,
        }));
  }

  for (int i : {-1, -2, 65, 66}) {
    EXPECT_THAT(
        ShardingSpec::FromJson({{"@type", "neuroglancer_uint64_sharded_v1"},
                                {"hash", "identity"},
                                {"preshift_bits", i},
                                {"minishard_bits", 2},
                                {"shard_bits", 3}}),
        MatchesStatus(absl::StatusCode::kInvalidArgument));
  }

  // Tests that `minishard_bits` is limited to `[0, 60]`.
  for (int i : {0, 1, 59, 60}) {
    EXPECT_THAT(
        ShardingSpec::FromJson({{"@type", "neuroglancer_uint64_sharded_v1"},
                                {"hash", "identity"},
                                {"preshift_bits", 1},
                                {"minishard_bits", i},
                                {"shard_bits", 0}}),
        ::testing::Optional(ShardingSpec{
            /*.hash_function=*/ShardingSpec::HashFunction::identity,
            /*.preshift_bits=*/1,
            /*.minishard_bits=*/i,
            /*.shard_bits=*/0,
            /*.data_encoding=*/ShardingSpec::DataEncoding::raw,
            /*.minishard_index_encoding=*/ShardingSpec::DataEncoding::raw,
        }));
  }

  for (int i : {-1, -2, 61, 62, 63}) {
    EXPECT_THAT(
        ShardingSpec::FromJson({{"@type", "neuroglancer_uint64_sharded_v1"},
                                {"hash", "identity"},
                                {"preshift_bits", 1},
                                {"minishard_bits", i},
                                {"shard_bits", 0}}),
        MatchesStatus(absl::StatusCode::kInvalidArgument));
  }

  // Tests that `shard_bits` is limited to `[0, 64-minishard_bits]`.
  for (int i : {0, 1, 64 - 8, 64 - 7}) {
    EXPECT_THAT(
        ShardingSpec::FromJson({{"@type", "neuroglancer_uint64_sharded_v1"},
                                {"hash", "identity"},
                                {"preshift_bits", 1},
                                {"minishard_bits", 7},
                                {"shard_bits", i}}),
        ::testing::Optional(ShardingSpec{
            /*.hash_function=*/ShardingSpec::HashFunction::identity,
            /*.preshift_bits=*/1,
            /*.minishard_bits=*/7,
            /*.shard_bits=*/i,
            /*.data_encoding=*/ShardingSpec::DataEncoding::raw,
            /*.minishard_index_encoding=*/ShardingSpec::DataEncoding::raw,
        }));
  }

  for (int i : {-1, -2, 64 - 6, 64 - 5, 65, 66}) {
    EXPECT_THAT(
        ShardingSpec::FromJson({{"@type", "neuroglancer_uint64_sharded_v1"},
                                {"hash", "identity"},
                                {"preshift_bits", 1},
                                {"minishard_bits", 7},
                                {"shard_bits", i}}),
        MatchesStatus(absl::StatusCode::kInvalidArgument));
  }

  EXPECT_THAT(ShardingSpec::FromJson("invalid"),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST(MinishardIndexEntryTest, Comparison) {
  MinishardIndexEntry a{{1}, {2, 3}};
  MinishardIndexEntry b{{1}, {3, 4}};
  MinishardIndexEntry c{{2}, {2, 3}};
  MinishardIndexEntry d{{2}, {3, 4}};

  EXPECT_EQ(a, a);
  EXPECT_EQ(b, b);
  EXPECT_EQ(c, c);
  EXPECT_EQ(d, d);
  EXPECT_FALSE(a != a);
  EXPECT_FALSE(a == b);
  EXPECT_NE(a, c);
  EXPECT_NE(a, d);
  EXPECT_NE(b, c);
  EXPECT_NE(b, d);
  EXPECT_NE(c, d);
}

}  // namespace
