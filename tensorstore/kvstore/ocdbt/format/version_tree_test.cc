// Copyright 2025 The TensorStore Authors
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

#include "tensorstore/kvstore/ocdbt/format/version_tree.h"

#include <limits>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorstore/kvstore/ocdbt/format/btree.h"
#include "tensorstore/kvstore/ocdbt/format/config.h"
#include "tensorstore/kvstore/ocdbt/format/data_file_id.h"
#include "tensorstore/kvstore/ocdbt/format/indirect_data_reference.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

namespace {
using ::tensorstore::Result;
using ::tensorstore::StatusIs;
using ::tensorstore::internal_ocdbt::CommitTime;
using ::tensorstore::internal_ocdbt::CommitTimeUpperBound;
using ::tensorstore::internal_ocdbt::Config;
using ::tensorstore::internal_ocdbt::DataFileId;
using ::tensorstore::internal_ocdbt::DecodeVersionTreeNode;
using ::tensorstore::internal_ocdbt::EncodeVersionTreeNodeUnchecked;
using ::tensorstore::internal_ocdbt::FormatCommitTimeForUrl;
using ::tensorstore::internal_ocdbt::FormatVersionSpecForUrl;
using ::tensorstore::internal_ocdbt::GenerationNumber;
using ::tensorstore::internal_ocdbt::
    GetVersionTreeLeafNodeRangeContainingGeneration;
using ::tensorstore::internal_ocdbt::IndirectDataReference;
using ::tensorstore::internal_ocdbt::ParseCommitTimeFromUrl;
using ::tensorstore::internal_ocdbt::ParseVersionSpecFromUrl;
using ::tensorstore::internal_ocdbt::VersionNodeReference;
using ::tensorstore::internal_ocdbt::VersionSpec;
using ::tensorstore::internal_ocdbt::VersionTreeNode;
using ::testing::HasSubstr;

void TestVersionSpecUrlRoundtrip(VersionSpec spec,
                                 std::vector<std::string> url_representations) {
  EXPECT_THAT(FormatVersionSpecForUrl(spec),
              ::testing::Eq(url_representations.at(0)));
  for (const auto &rep : url_representations) {
    EXPECT_THAT(ParseVersionSpecFromUrl(rep), ::testing::Eq(spec));
  }
}

TEST(VersionSpecUrl, GenerationRoundtrip) {
  TestVersionSpecUrlRoundtrip(GenerationNumber{1}, {"v1"});
  TestVersionSpecUrlRoundtrip(GenerationNumber{18446744073709551615u},
                              {"v18446744073709551615"});
}

TEST(VersionSpecUrl, CommitTimeRoundtrip) {
  TestVersionSpecUrlRoundtrip(
      CommitTimeUpperBound{CommitTime{0}},
      {"1970-01-01T00:00:00Z", "1970-01-01T00:00:00.00Z"});
  TestVersionSpecUrlRoundtrip(CommitTimeUpperBound{CommitTime{1}},
                              {"1970-01-01T00:00:00.000000001Z"});
  EXPECT_THAT(FormatVersionSpecForUrl(CommitTime{0}),
              ::testing::Eq("1970-01-01T00:00:00Z"));
}

TEST(VersionSpecUrl, Errors) {
  EXPECT_THAT(ParseVersionSpecFromUrl("a"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseVersionSpecFromUrl("v"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseVersionSpecFromUrl("v0"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseVersionSpecFromUrl("1970-01-01T00:00:00"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

void TestCommitTimeUrlRoundtrip(CommitTime t,
                                std::vector<std::string> url_representations) {
  EXPECT_THAT(FormatCommitTimeForUrl(t),
              ::testing::Eq(url_representations.at(0)));
  for (const auto &rep : url_representations) {
    EXPECT_THAT(ParseCommitTimeFromUrl(rep), ::testing::Eq(t));
  }
}

TEST(CommitTimeUrl, Roundtrip) {
  TestCommitTimeUrlRoundtrip(
      CommitTime{0}, {"1970-01-01T00:00:00Z", "1970-01-01T00:00:00.00Z"});
  TestCommitTimeUrlRoundtrip(CommitTime{1}, {"1970-01-01T00:00:00.000000001Z"});
}

TEST(CommitTimeUrl, Errors) {
  EXPECT_THAT(ParseVersionSpecFromUrl("1970-01-01T00:00:00"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(VersionTreeFormatTest, AbslStringify) {
  using ::tensorstore::internal_ocdbt::BtreeGenerationReference;
  using ::tensorstore::internal_ocdbt::BtreeNodeStatistics;
  using ::tensorstore::internal_ocdbt::DataFileId;
  using ::tensorstore::internal_ocdbt::IndirectDataReference;
  using ::tensorstore::internal_ocdbt::VersionNodeReference;
  using ::tensorstore::internal_ocdbt::VersionTreeNode;

  EXPECT_EQ("1970-01-01T00:00:00.000000001+00:00", absl::StrCat(CommitTime{1}));

  BtreeGenerationReference btree_ref{
      {IndirectDataReference{DataFileId{"", "file1"}, 2, 3},
       BtreeNodeStatistics{10, 200, 30}},
      4,
      5,
      CommitTime{6}};
  EXPECT_EQ(
      "{root={location={file_id=\"\"+\"file1\", offset=2, length=3}, "
      "statistics={num_indirect_value_bytes=10, num_tree_bytes=200, "
      "num_keys=30}}, generation_number=4, root_height=5, "
      "commit_time=1970-01-01T00:00:00.000000006+00:00}",
      absl::StrCat(btree_ref));

  VersionNodeReference version_ref{
      IndirectDataReference{DataFileId{"", "file2"}, 5, 6}, 7, 8, 9,
      CommitTime{10}};
  EXPECT_EQ(
      "{location={file_id=\"\"+\"file2\", offset=5, length=6}, "
      "generation_number=7, height=8, "
      "num_generations=9, commit_time=1970-01-01T00:00:00.00000001+00:00}",
      absl::StrCat(version_ref));

  VersionTreeNode node;
  node.height = 1;
  node.entries = VersionTreeNode::InteriorNodeEntries{version_ref};
  EXPECT_EQ(
      "{height=1, entries={{location={file_id=\"\"+\"file2\", offset=5, "
      "length=6}, generation_number=7, height=8, num_generations=9, "
      "commit_time=1970-01-01T00:00:00.00000001+00:00}}}",
      absl::StrCat(node));
}

// Generation numbers near UINT64_MAX must not wrap around to 0 when computing
// `last_generation = next_generation + (1 << arity_log2)`.
TEST(VersionTreeFormatTest, LeafRangeNearMaxUint64) {
  constexpr GenerationNumber kMax =
      std::numeric_limits<GenerationNumber>::max();
  EXPECT_THAT(GetVersionTreeLeafNodeRangeContainingGeneration(
                  /*version_tree_arity_log2=*/4, kMax),
              ::testing::Pair(kMax - 14, kMax));
}

// Leaf version tree nodes must have strictly increasing `commit_time` across
// entries so binary search by `CommitTime` / `CommitTimeUpperBound` remains
// monotonic.
TEST(VersionTreeFormatTest, LeafCommitTimesMustStrictlyIncrease) {
  const auto decode =
      [](CommitTime second_commit_time) -> Result<VersionTreeNode> {
    Config config;
    config.version_tree_arity_log2 = 2;
    VersionTreeNode node;
    node.version_tree_arity_log2 = 2;
    node.height = 0;
    auto& entries = node.entries.emplace<VersionTreeNode::LeafNodeEntries>();
    auto& ref1 = entries.emplace_back();
    ref1.root.location = IndirectDataReference{DataFileId{"", "f1"}, 0, 10};
    ref1.root.statistics = {0, 10, 1};
    ref1.generation_number = 1;
    ref1.root_height = 0;
    ref1.commit_time = CommitTime{20};
    auto& ref2 = entries.emplace_back(ref1);
    ref2.generation_number = 2;
    ref2.commit_time = second_commit_time;
    TENSORSTORE_ASSIGN_OR_RETURN(auto encoded,
                                 EncodeVersionTreeNodeUnchecked(config, node));
    return DecodeVersionTreeNode(encoded, /*base_path=*/{});
  };
  EXPECT_THAT(decode(CommitTime{21}), ::tensorstore::IsOk());
  EXPECT_THAT(decode(CommitTime{20}),
              StatusIs(absl::StatusCode::kDataLoss, HasSubstr("commit_time")));
  EXPECT_THAT(decode(CommitTime{10}),
              StatusIs(absl::StatusCode::kDataLoss, HasSubstr("commit_time")));
}

// Interior version tree nodes must also have strictly increasing `commit_time`
// across child entries.
TEST(VersionTreeFormatTest, InteriorCommitTimesMustStrictlyIncrease) {
  const auto decode =
      [](CommitTime second_commit_time) -> Result<VersionTreeNode> {
    Config config;
    config.version_tree_arity_log2 = 2;
    VersionTreeNode node;
    node.version_tree_arity_log2 = 2;
    node.height = 1;
    auto& entries =
        node.entries.emplace<VersionTreeNode::InteriorNodeEntries>();
    entries.push_back(VersionNodeReference{
        IndirectDataReference{DataFileId{"", "f1"}, 0, 10},
        /*.generation_number=*/4,
        /*.height=*/0,
        /*.num_generations=*/4,
        /*.commit_time=*/CommitTime{20},
    });
    entries.push_back(VersionNodeReference{
        IndirectDataReference{DataFileId{"", "f1"}, 10, 10},
        /*.generation_number=*/8,
        /*.height=*/0,
        /*.num_generations=*/4,
        /*.commit_time=*/second_commit_time,
    });
    TENSORSTORE_ASSIGN_OR_RETURN(auto encoded,
                                 EncodeVersionTreeNodeUnchecked(config, node));
    return DecodeVersionTreeNode(encoded, /*base_path=*/{});
  };
  EXPECT_THAT(decode(CommitTime{21}), ::tensorstore::IsOk());
  EXPECT_THAT(decode(CommitTime{20}),
              StatusIs(absl::StatusCode::kDataLoss, HasSubstr("commit_time")));
  EXPECT_THAT(decode(CommitTime{10}),
              StatusIs(absl::StatusCode::kDataLoss, HasSubstr("commit_time")));
}

}  // namespace
