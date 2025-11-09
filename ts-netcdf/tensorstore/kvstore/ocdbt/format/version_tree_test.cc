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

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {
using ::tensorstore::MatchesStatus;
using ::tensorstore::internal_ocdbt::CommitTime;
using ::tensorstore::internal_ocdbt::CommitTimeUpperBound;
using ::tensorstore::internal_ocdbt::FormatCommitTimeForUrl;
using ::tensorstore::internal_ocdbt::FormatVersionSpecForUrl;
using ::tensorstore::internal_ocdbt::GenerationNumber;
using ::tensorstore::internal_ocdbt::ParseCommitTimeFromUrl;
using ::tensorstore::internal_ocdbt::ParseVersionSpecFromUrl;
using ::tensorstore::internal_ocdbt::VersionSpec;

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
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseVersionSpecFromUrl("v"),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseVersionSpecFromUrl("v0"),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseVersionSpecFromUrl("1970-01-01T00:00:00"),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
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
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

}  // namespace
