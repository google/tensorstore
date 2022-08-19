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

#include "tensorstore/kvstore/neuroglancer_uint64_sharded/uint64_sharded_decoder.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/internal/compression/zlib.h"
#include "tensorstore/kvstore/neuroglancer_uint64_sharded/uint64_sharded.h"
#include "tensorstore/kvstore/neuroglancer_uint64_sharded/uint64_sharded_encoder.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

namespace zlib = tensorstore::zlib;
using ::tensorstore::MatchesStatus;
using ::tensorstore::neuroglancer_uint64_sharded::DecodeMinishardIndex;
using ::tensorstore::neuroglancer_uint64_sharded::EncodeMinishardIndex;
using ::tensorstore::neuroglancer_uint64_sharded::MinishardIndexEntry;
using ::tensorstore::neuroglancer_uint64_sharded::ShardIndexEntry;
using ::tensorstore::neuroglancer_uint64_sharded::ShardingSpec;

void TestEncodeMinishardRoundTrip(
    std::vector<MinishardIndexEntry> minishard_index) {
  auto out = EncodeMinishardIndex(minishard_index);
  absl::Cord compressed;
  zlib::Options options{/*.level=*/9, /*.use_gzip_header=*/true};
  zlib::Encode(out, &compressed, options);
  EXPECT_THAT(
      DecodeMinishardIndex(out, ShardingSpec::DataEncoding::raw),
      ::testing::Optional(::testing::ElementsAreArray(minishard_index)));
  EXPECT_THAT(
      DecodeMinishardIndex(compressed, ShardingSpec::DataEncoding::gzip),
      ::testing::Optional(::testing::ElementsAreArray(minishard_index)));
}

TEST(DecodeMinishardIndexTest, Empty) {  //
  TestEncodeMinishardRoundTrip({});
}

TEST(DecodeMinishardIndexTest, SingleEntry) {
  TestEncodeMinishardRoundTrip({{{0x0123456789abcdef}, {0x11, 0x23}}});
}

TEST(DecodeMinishardIndexTest, MultipleEntries) {
  TestEncodeMinishardRoundTrip({
      {{1}, {3, 10}},
      {{7}, {12, 15}},
  });
}

TEST(DecodeMinishardIndexTest, InvalidGzip) {
  EXPECT_THAT(
      DecodeMinishardIndex(absl::Cord("abc"), ShardingSpec::DataEncoding::gzip),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Error decoding zlib-compressed data"));
}

TEST(DecodeMinishardIndexTest, InvalidSizeRaw) {
  EXPECT_THAT(
      DecodeMinishardIndex(absl::Cord("abc"), ShardingSpec::DataEncoding::raw),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Invalid minishard index length: 3"));
}

TEST(DecodeMinishardIndexTest, InvalidSizeGzip) {
  absl::Cord temp;
  zlib::Options options{/*.level=*/9, /*.use_gzip_header=*/true};
  zlib::Encode(absl::Cord("abc"), &temp, options);
  EXPECT_THAT(DecodeMinishardIndex(temp, ShardingSpec::DataEncoding::gzip),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Invalid minishard index length: 3"));
}

TEST(DecodeMinishardIndexTest, InvalidInterval) {
  std::vector<MinishardIndexEntry> minishard_index{{{3}, {1, 0}}};
  auto encoded = EncodeMinishardIndex(minishard_index);
  EXPECT_THAT(
      DecodeMinishardIndex(encoded, ShardingSpec::DataEncoding::raw),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          "Invalid byte range in minishard index for chunk 3: \\[1, 0\\)"));
}

}  // namespace
