// Copyright 2022 The TensorStore Authors
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

#include "tensorstore/kvstore/ocdbt/format/manifest.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_format.h"
#include "tensorstore/kvstore/ocdbt/format/btree.h"
#include "tensorstore/kvstore/ocdbt/format/config.h"
#include "tensorstore/kvstore/ocdbt/format/indirect_data_reference.h"
#include "tensorstore/kvstore/ocdbt/format/version_tree.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::MatchesStatus;
using ::tensorstore::Result;
using ::tensorstore::internal_ocdbt::CommitTime;
using ::tensorstore::internal_ocdbt::DecodeManifest;
using ::tensorstore::internal_ocdbt::Manifest;

Result<absl::Time> RoundTripCommitTime(absl::Time time) {
  TENSORSTORE_ASSIGN_OR_RETURN(auto commit_time,
                               CommitTime::FromAbslTime(time));
  return static_cast<absl::Time>(commit_time);
}

TEST(CommitTimeTest, Simple) {
  EXPECT_THAT(RoundTripCommitTime(absl::FromUnixNanos(0)),
              ::testing::Optional(absl::FromUnixNanos(0)));
  EXPECT_THAT(RoundTripCommitTime(absl::FromUnixNanos(-1)),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(RoundTripCommitTime(
                  absl::FromUnixNanos(std::numeric_limits<int64_t>::max())),
              ::testing::Optional(
                  absl::FromUnixNanos(std::numeric_limits<int64_t>::max())));
  EXPECT_THAT(RoundTripCommitTime(
                  absl::FromUnixNanos(std::numeric_limits<int64_t>::max()) +
                  absl::Nanoseconds(1)),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

void TestManifestRoundTrip(const Manifest& manifest) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto encoded, EncodeManifest(manifest));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto decoded, DecodeManifest(encoded));
  EXPECT_EQ(manifest, decoded);
}

Manifest GetSimpleManifest() {
  Manifest manifest;

  auto& x = manifest.versions.emplace_back();
  x.root.location.file_id.base_path = "abc";
  x.root.location.file_id.relative_path = "defgh";
  x.root.location.offset = 10;
  x.root.location.length = 42;
  x.generation_number = 1;
  x.root.statistics.num_indirect_value_bytes = 101;
  x.root.statistics.num_tree_bytes = 220;
  x.root.statistics.num_keys = 8;
  x.root_height = 0;
  x.commit_time = CommitTime{1};
  return manifest;
}

TEST(ManifestTest, RoundTrip) { TestManifestRoundTrip(GetSimpleManifest()); }

TEST(ManifestTest, RoundTripNonZeroHeight) {
  Manifest manifest;
  {
    auto& x = manifest.versions.emplace_back();
    x.root.location.file_id.base_path = "abc";
    x.root.location.file_id.relative_path = "defgh";
    x.root.location.offset = 10;
    x.root.location.length = 42;
    x.generation_number = 1;
    x.root.statistics.num_indirect_value_bytes = 101;
    x.root.statistics.num_tree_bytes = 220;
    x.root.statistics.num_keys = 8;
    x.root_height = 5;
    x.commit_time = CommitTime{1};
  }
  TestManifestRoundTrip(manifest);
}

TEST(ManifestTest, CorruptMagic) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto encoded,
                                   EncodeManifest(GetSimpleManifest()));
  absl::Cord corrupt = encoded;
  corrupt.RemovePrefix(4);
  corrupt.Prepend("abcd");
  EXPECT_THAT(DecodeManifest(corrupt),
              MatchesStatus(
                  absl::StatusCode::kDataLoss,
                  ".*: Expected to start with hex bytes .* but received: .*"));
}

TEST(ManifestTest, CorruptLength) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto encoded,
                                   EncodeManifest(GetSimpleManifest()));
  auto corrupt = encoded;
  corrupt.Append("x");
  EXPECT_THAT(
      DecodeManifest(corrupt),
      MatchesStatus(absl::StatusCode::kDataLoss, ".*: Length in header .*"));
}

TEST(ManifestTest, InvalidVersion) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto encoded,
                                   EncodeManifest(GetSimpleManifest()));
  auto corrupt = encoded.Subcord(0, 12);
  corrupt.Append(std::string(1, 1));
  corrupt.Append(encoded.Subcord(13, -1));
  EXPECT_THAT(
      DecodeManifest(corrupt),
      MatchesStatus(absl::StatusCode::kDataLoss,
                    ".*: Maximum supported version is 0 but received: 1.*"));
}

TEST(ManifestTest, CorruptChecksum) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto encoded,
                                   EncodeManifest(GetSimpleManifest()));
  auto corrupt = encoded;
  auto sv = corrupt.Flatten();
  unsigned char final_char = sv.back();
  ++final_char;
  corrupt.RemoveSuffix(1);
  corrupt.Append(std::string(1, final_char));
  EXPECT_THAT(DecodeManifest(corrupt),
              MatchesStatus(absl::StatusCode::kDataLoss,
                            ".*: CRC-32C checksum verification failed.*"));
}

TEST(ManifestTest, RoundTripMultipleVersions) {
  Manifest manifest;
  manifest.config.version_tree_arity_log2 = 1;
  // Latest generation is 15
  // Manifest directly references versions: 15
  // Manifest references version nodes:
  //   - 13 up to 14 (eventually up to 16) (height 1)
  //   - 9 up to 12 (eventually up to 16) (height 2)
  //   - 1 up to 8 (eventually up to 16) (height 3)
  {
    auto& x = manifest.versions.emplace_back();
    x.root.location.file_id.base_path = "abc";
    x.root.location.file_id.relative_path = "defgh";
    x.root.location.offset = 10;
    x.root.location.length = 42;
    x.generation_number = 15;
    x.root.statistics.num_indirect_value_bytes = 101;
    x.root.statistics.num_tree_bytes = 220;
    x.root.statistics.num_keys = 8;
    x.root_height = 0;
    x.commit_time = CommitTime{10};
  }
  {
    auto& x = manifest.version_tree_nodes.emplace_back();
    x.location.file_id.base_path = "abc";
    x.location.file_id.relative_path = "defgh";
    x.location.offset = 10;
    x.location.length = 42;
    x.generation_number = 8;
    x.height = 3;
    x.commit_time = CommitTime{1};
    x.num_generations = 8;
  }
  {
    auto& x = manifest.version_tree_nodes.emplace_back();
    x.location.file_id.base_path = "abc";
    x.location.file_id.relative_path = "defgh1";
    x.location.offset = 10;
    x.location.length = 42;
    x.generation_number = 12;
    x.height = 2;
    x.commit_time = CommitTime{5};
    x.num_generations = 4;
  }
  {
    auto& x = manifest.version_tree_nodes.emplace_back();
    x.location.file_id.base_path = "abc1";
    x.location.file_id.relative_path = "defgh";
    x.location.offset = 10;
    x.location.length = 42;
    x.generation_number = 14;
    x.height = 1;
    x.commit_time = CommitTime{8};
    x.num_generations = 2;
  }
  TestManifestRoundTrip(manifest);
}

namespace for_each_manifest_version_tree_node_ref {
using ::tensorstore::internal_ocdbt::ForEachManifestVersionTreeNodeRef;
using ::tensorstore::internal_ocdbt::GenerationNumber;
using ::tensorstore::internal_ocdbt::VersionTreeArityLog2;
using R = std::tuple<GenerationNumber, GenerationNumber, int>;

std::vector<R> GetRanges(GenerationNumber generation_number,
                         VersionTreeArityLog2 version_tree_arity_log2) {
  std::vector<R> results;
  ForEachManifestVersionTreeNodeRef(
      generation_number, version_tree_arity_log2,
      [&](GenerationNumber min_generation_number,
          GenerationNumber max_generation_number, VersionTreeArityLog2 height) {
        results.emplace_back(min_generation_number, max_generation_number,
                             height);
      });
  return results;
}

TEST(ForEachManifestVersionTreeNodeRefTest, SimpleCases) {
  EXPECT_THAT(GetRanges(8, 2), ::testing::ElementsAre(R{1, 4, 1}));
  EXPECT_THAT(GetRanges(9, 2), ::testing::ElementsAre(R{1, 8, 1}));
  EXPECT_THAT(GetRanges(17, 2), ::testing::ElementsAre(R{1, 16, 1}));
  EXPECT_THAT(GetRanges(30, 2),
              ::testing::ElementsAre(R{17, 28, 1}, R{1, 16, 2}));
  EXPECT_THAT(GetRanges(43, 2),
              ::testing::ElementsAre(R{33, 40, 1}, R{1, 32, 2}));
  EXPECT_THAT(GetRanges(17, 1),
              ::testing::ElementsAre(R{13, 16, 1}, R{9, 12, 2}, R{1, 8, 3}));
}

class ForEachManifestVersionTreeNodeRefPropertyTest
    : public ::testing::TestWithParam<std::tuple<GenerationNumber, int>> {};

TEST_P(ForEachManifestVersionTreeNodeRefPropertyTest, Properties) {
  auto [generation_number, version_tree_arity_log2] = GetParam();

  auto range = GetRanges(generation_number, version_tree_arity_log2);
  SCOPED_TRACE(
      absl::StrFormat("generation_number=%d, version_tree_arity_log2=%d",
                      generation_number, version_tree_arity_log2));
  SCOPED_TRACE(::testing::PrintToString(range));
  for (size_t i = 0; i < range.size(); ++i) {
    auto [min_gen, max_gen, height] = range[i];
    SCOPED_TRACE(
        absl::StrFormat("i=%d,height=%d,min_generation=%d,max_generation=%d", i,
                        height, min_gen, max_gen));
    EXPECT_EQ(height, i + 1);
    EXPECT_LT(max_gen, generation_number);
    EXPECT_GT(max_gen, 0);
    EXPECT_GT(min_gen, 0);
    EXPECT_LT(min_gen, max_gen);
    EXPECT_EQ(
        0, max_gen % (GenerationNumber(1) << height * version_tree_arity_log2));
    if (i == 0) {
      EXPECT_GE(max_gen + (GenerationNumber(1) << version_tree_arity_log2),
                generation_number);
    }
    if (i > 0) {
      auto [prev_min_gen, prev_max_gen, prev_height] = range[i - 1];
      EXPECT_EQ(prev_min_gen, max_gen + 1);
    }
  }
}

std::string PrintPropertyTestValue(
    const ::testing::TestParamInfo<std::tuple<GenerationNumber, int>>& info) {
  const auto [generation_number, version_tree_arity_log2] = info.param;
  return absl::StrFormat("%d_%d", generation_number, version_tree_arity_log2);
}

INSTANTIATE_TEST_SUITE_P(
    Combinations, ForEachManifestVersionTreeNodeRefPropertyTest,
    ::testing::Combine(::testing::ValuesIn<GenerationNumber>({
                           1,
                           2,
                           101,
                           12345,
                           567890,
                       }),
                       ::testing::ValuesIn<int>({
                           1,
                           2,
                           3,
                           4,
                       })),
    PrintPropertyTestValue);

INSTANTIATE_TEST_SUITE_P(
    Simple, ForEachManifestVersionTreeNodeRefPropertyTest,
    (::testing::ValuesIn<std::tuple<GenerationNumber, int>>({
        {8, 2},
        {9, 2},
        {17, 2},
        {43, 2},
        {17, 1},
    })),
    PrintPropertyTestValue);

}  // namespace for_each_manifest_version_tree_node_ref

}  // namespace
