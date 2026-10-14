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

#include <stddef.h>

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "tensorstore/internal/compression/zlib.h"
#include "tensorstore/kvstore/byte_range.h"
#include "tensorstore/kvstore/neuroglancer_uint64_sharded/uint64_sharded.h"
#include "tensorstore/kvstore/neuroglancer_uint64_sharded/uint64_sharded_encoder.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

namespace zlib = tensorstore::zlib;
using ::tensorstore::ByteRange;
using ::tensorstore::StatusIs;
using ::tensorstore::neuroglancer_uint64_sharded::ChunkId;
using ::tensorstore::neuroglancer_uint64_sharded::DecodeMinishardIndex;
using ::tensorstore::neuroglancer_uint64_sharded::DecodeShardIndexEntry;
using ::tensorstore::neuroglancer_uint64_sharded::EncodedChunk;
using ::tensorstore::neuroglancer_uint64_sharded::EncodeMinishardIndex;
using ::tensorstore::neuroglancer_uint64_sharded::EncodeShard;
using ::tensorstore::neuroglancer_uint64_sharded::EncodeShardIndex;
using ::tensorstore::neuroglancer_uint64_sharded::MinishardIndexEntry;
using ::tensorstore::neuroglancer_uint64_sharded::ShardIndexEntry;
using ::tensorstore::neuroglancer_uint64_sharded::ShardingSpec;
using ::tensorstore::neuroglancer_uint64_sharded::SplitShard;
using ::testing::HasSubstr;

absl::Cord MakeFragmentedTestCord(const absl::Cord& input,
                                  std::string& backing_storage) {
  backing_storage = std::string(input);
  if (backing_storage.size() < 2) return input;
  const size_t split = backing_storage.size() / 2;
  absl::Cord fragmented;
  fragmented.Append(absl::MakeCordFromExternal(
      std::string_view(backing_storage.data(), split), [] {}));
  fragmented.Append(absl::MakeCordFromExternal(
      std::string_view(backing_storage.data() + split,
                       backing_storage.size() - split),
      [] {}));
  return fragmented;
}

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

  std::string out_storage;
  absl::Cord fragmented_out = MakeFragmentedTestCord(out, out_storage);
  EXPECT_THAT(
      DecodeMinishardIndex(fragmented_out, ShardingSpec::DataEncoding::raw),
      ::testing::Optional(::testing::ElementsAreArray(minishard_index)));

  std::string compressed_storage;
  absl::Cord fragmented_compressed =
      MakeFragmentedTestCord(compressed, compressed_storage);
  EXPECT_THAT(
      DecodeMinishardIndex(fragmented_compressed,
                           ShardingSpec::DataEncoding::gzip),
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
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Error decoding zlib-compressed data")));
}

TEST(DecodeMinishardIndexTest, InvalidSizeRaw) {
  EXPECT_THAT(
      DecodeMinishardIndex(absl::Cord("abc"), ShardingSpec::DataEncoding::raw),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Invalid minishard index length: 3")));
}

TEST(DecodeMinishardIndexTest, InvalidSizeGzip) {
  absl::Cord temp;
  zlib::Options options{/*.level=*/9, /*.use_gzip_header=*/true};
  zlib::Encode(absl::Cord("abc"), &temp, options);
  EXPECT_THAT(DecodeMinishardIndex(temp, ShardingSpec::DataEncoding::gzip),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid minishard index length: 3")));
}

TEST(DecodeMinishardIndexTest, InvalidInterval) {
  std::vector<MinishardIndexEntry> minishard_index{{{3}, {1, 0}}};
  auto encoded = EncodeMinishardIndex(minishard_index);
  EXPECT_THAT(DecodeMinishardIndex(encoded, ShardingSpec::DataEncoding::raw),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid byte range in minishard index for "
                                 "chunk 3: [1, 0)")));
}

TEST(DecodeShardIndexEntryTest, FlatAndFragmentedCord) {
  std::vector<ShardIndexEntry> entries{{{10, 34}}};
  absl::Cord encoded = EncodeShardIndex(entries);
  EXPECT_THAT(DecodeShardIndexEntry(encoded),
              ::testing::Optional(ByteRange{10, 34}));

  std::string storage;
  absl::Cord fragmented = MakeFragmentedTestCord(encoded, storage);
  ASSERT_FALSE(fragmented.TryFlat().has_value());
  EXPECT_THAT(DecodeShardIndexEntry(fragmented),
              ::testing::Optional(ByteRange{10, 34}));
  EXPECT_FALSE(fragmented.TryFlat().has_value());
}

TEST(DecodeShardIndexEntryTest, InvalidLengthAndRange) {
  std::string short_buf = "0123456789";
  absl::Cord fragmented_short;
  fragmented_short.Append(
      absl::MakeCordFromExternal(std::string_view(short_buf.data(), 5), [] {}));
  fragmented_short.Append(absl::MakeCordFromExternal(
      std::string_view(short_buf.data() + 5, 5), [] {}));
  ASSERT_FALSE(fragmented_short.TryFlat().has_value());
  EXPECT_THAT(DecodeShardIndexEntry(fragmented_short),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("Expected 16 bytes, but received: 10 bytes")));

  std::vector<ShardIndexEntry> invalid_entries{{{20, 10}}};
  std::string invalid_storage;
  absl::Cord fragmented_invalid = MakeFragmentedTestCord(
      EncodeShardIndex(invalid_entries), invalid_storage);
  ASSERT_FALSE(fragmented_invalid.TryFlat().has_value());
  EXPECT_THAT(DecodeShardIndexEntry(fragmented_invalid),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("Shard index specified invalid byte range: "
                                 "[20, 10)")));
}

TEST(SplitShardTest, EmptyAndFragmentedRoundTrip) {
  ShardingSpec spec{
      /*.hash_function=*/ShardingSpec::HashFunction::identity,
      /*.preshift_bits=*/0,
      /*.minishard_bits=*/1,
      /*.shard_bits=*/0,
      /*.data_encoding=*/ShardingSpec::DataEncoding::raw,
      /*.minishard_index_encoding=*/ShardingSpec::DataEncoding::raw,
  };
  EXPECT_THAT(SplitShard(spec, absl::Cord()),
              ::testing::Optional(::testing::IsEmpty()));

  std::vector<EncodedChunk> chunks{
      {{/*minishard=*/0, ChunkId{2}}, absl::Cord("hello")},
      {{/*minishard=*/0, ChunkId{5}}, absl::Cord("world!")},
      {{/*minishard=*/1, ChunkId{3}}, absl::Cord("foo")},
  };
  for (auto index_encoding :
       {ShardingSpec::DataEncoding::raw, ShardingSpec::DataEncoding::gzip}) {
    spec.minishard_index_encoding = index_encoding;
    auto shard_cord = EncodeShard(spec, chunks);
    ASSERT_TRUE(shard_cord.has_value());
    std::string storage;
    absl::Cord fragmented_shard = MakeFragmentedTestCord(*shard_cord, storage);
    ASSERT_FALSE(fragmented_shard.TryFlat().has_value());
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto split_chunks,
                                     SplitShard(spec, fragmented_shard));
    ASSERT_EQ(split_chunks.size(), chunks.size());
    for (size_t i = 0; i < chunks.size(); ++i) {
      EXPECT_EQ(split_chunks[i].minishard_and_chunk_id.minishard,
                chunks[i].minishard_and_chunk_id.minishard);
      EXPECT_EQ(split_chunks[i].minishard_and_chunk_id.chunk_id.value,
                chunks[i].minishard_and_chunk_id.chunk_id.value);
      EXPECT_EQ(split_chunks[i].encoded_data, chunks[i].encoded_data);
    }
  }

  EXPECT_THAT(SplitShard(spec, absl::Cord("too_short")),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("Existing shard has size 9, but expected at "
                                 "least: 32")));
}

}  // namespace
