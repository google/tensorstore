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

#include "tensorstore/array_testutil.h"

#include <sstream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace {

using ::tensorstore::Index;
using ::tensorstore::MakeArray;
using ::tensorstore::MakeOffsetArray;
using ::tensorstore::MakeScalarArray;
using ::tensorstore::MatchesArray;
using ::tensorstore::MatchesScalarArray;
using ::tensorstore::span;

TEST(MatchesArrayTest, Describe) {
  std::ostringstream ss;
  MatchesArray<std::int32_t>({1, 2}).DescribeTo(&ss);
  EXPECT_EQ(
      R"(has a data type of int32 and a domain of {origin={0}, shape={2}} where
element at {0} is equal to 1,
element at {1} is equal to 2)",
      ss.str());
}

TEST(MatchesArrayTest, DescribeNegation) {
  std::ostringstream ss;
  MatchesArray<std::int32_t>({1, 2}).DescribeNegationTo(&ss);
  EXPECT_EQ(R"(doesn't have a data type of int32, or
doesn't have a domain of {origin={0}, shape={2}}, or
element at {0} isn't equal to 1, or
element at {1} isn't equal to 2)",
            ss.str());
}

TEST(MatchesArrayTest, ExplainDataTypeMismatch) {
  ::testing::StringMatchResultListener listener;
  ::testing::ExplainMatchResult(MatchesArray<std::int32_t>({1, 2, 3}),
                                MakeArray<float>({1, 2}), &listener);
  EXPECT_EQ("which has a data type of float32", listener.str());
}

TEST(MatchesArrayTest, ExplainDomainMismatch) {
  ::testing::StringMatchResultListener listener;
  ::testing::ExplainMatchResult(MatchesArray<int>({1, 2, 3}),
                                MakeArray<int>({1, 2}), &listener);
  EXPECT_EQ("", listener.str());
}

TEST(MatchesArrayTest, ExplainElementMismatch) {
  ::testing::StringMatchResultListener listener;
  ::testing::ExplainMatchResult(MatchesArray<int>({1, 2}),
                                MakeArray<int>({1, 3}), &listener);
  EXPECT_EQ("whose element at {1} doesn't match", listener.str());
}

TEST(MatchesArrayTest, ExplainElementMatch) {
  ::testing::StringMatchResultListener listener;
  ::testing::ExplainMatchResult(
      MatchesArray<std::string>(
          {::testing::Not(::testing::ElementsAre('d')),
           ::testing::Not(::testing::ElementsAre('a', 'b'))}),
      MakeArray<std::string>({"x", "ac"}), &listener);
  EXPECT_EQ(
      "whose element at {0} matches, whose element #0 doesn't match,\n"
      "and whose element at {1} matches, whose element #1 doesn't match",
      listener.str());
}

TEST(MatchesArrayTest, ExplainElementMismatchExplanation) {
  ::testing::StringMatchResultListener listener;
  ::testing::ExplainMatchResult(
      MatchesScalarArray<std::string>(::testing::ElementsAre('a', 'b')),
      MakeScalarArray<std::string>("ac"), &listener);
  EXPECT_EQ("whose element at {} doesn't match, whose element #1 doesn't match",
            listener.str());
}

TEST(MatchesArrayTest, Matches) {
  // Rank 0
  EXPECT_THAT(MakeScalarArray<int>(1), MatchesScalarArray<int>(1));

  // Rank 1 with zero origin.
  EXPECT_THAT(MakeArray<int>({1, 2}), MatchesArray<int>({1, 2}));

  // Rank 2 with zero origin.
  EXPECT_THAT(MakeArray<int>({{1, 2}}), MatchesArray<int>({{1, 2}}));

  // Rank 3 with zero origin.
  EXPECT_THAT(MakeArray<int>({{{1, 2}}}), MatchesArray<int>({{{1, 2}}}));

  // Rank 4 with zero origin.
  EXPECT_THAT(MakeArray<int>({{{{1, 2}}}}), MatchesArray<int>({{{{1, 2}}}}));

  // Rank 5 with zero origin.
  EXPECT_THAT(MakeArray<int>({{{{{1, 2}}}}}),
              MatchesArray<int>({{{{{1, 2}}}}}));

  // Rank 6 with zero origin.
  EXPECT_THAT(MakeArray<int>({{{{{{1, 2}}}}}}),
              MatchesArray<int>({{{{{{1, 2}}}}}}));

  // Rank 1 with braced list offset origin.
  EXPECT_THAT(MakeOffsetArray<int>({3}, {1, 2}),
              MatchesArray<int>({3}, {1, 2}));

  // Rank 2 with braced list offset origin.
  EXPECT_THAT(MakeOffsetArray<int>({3, 4}, {{1, 2}}),
              MatchesArray<int>({3, 4}, {{1, 2}}));

  // Rank 3 with braced list offset origin.
  EXPECT_THAT(MakeOffsetArray<int>({3, 4, 5}, {{{1, 2}}}),
              MatchesArray<int>({3, 4, 5}, {{{1, 2}}}));

  // Rank 4 with braced list offset origin.
  EXPECT_THAT(MakeOffsetArray<int>({3, 4, 5, 6}, {{{{1, 2}}}}),
              MatchesArray<int>({3, 4, 5, 6}, {{{{1, 2}}}}));

  // Rank 5 with braced list offset origin.
  EXPECT_THAT(MakeOffsetArray<int>({3, 4, 5, 6, 7}, {{{{{1, 2}}}}}),
              MatchesArray<int>({3, 4, 5, 6, 7}, {{{{{1, 2}}}}}));

  // Rank 6 with braced list offset origin.
  EXPECT_THAT(MakeOffsetArray<int>({3, 4, 5, 6, 7, 8}, {{{{{{1, 2}}}}}}),
              MatchesArray<int>({3, 4, 5, 6, 7, 8}, {{{{{{1, 2}}}}}}));

  // Rank 1 with span offset origin.
  EXPECT_THAT(MakeOffsetArray<int>({3}, {1, 2}),
              MatchesArray<int>(span<const Index, 1>({3}), {1, 2}));

  // Rank 2 with span offset origin.
  EXPECT_THAT(MakeOffsetArray<int>({3, 4}, {{1, 2}}),
              MatchesArray<int>(span<const Index, 2>({3, 4}), {{1, 2}}));

  // Rank 3 with span offset origin.
  EXPECT_THAT(MakeOffsetArray<int>({3, 4, 5}, {{{1, 2}}}),
              MatchesArray<int>(span<const Index, 3>({3, 4, 5}), {{{1, 2}}}));

  // Rank 4 with span offset origin.
  EXPECT_THAT(
      MakeOffsetArray<int>({3, 4, 5, 6}, {{{{1, 2}}}}),
      MatchesArray<int>(span<const Index, 4>({3, 4, 5, 6}), {{{{1, 2}}}}));

  // Rank 5 with span offset origin.
  EXPECT_THAT(
      MakeOffsetArray<int>({3, 4, 5, 6, 7}, {{{{{1, 2}}}}}),
      MatchesArray<int>(span<const Index, 5>({3, 4, 5, 6, 7}), {{{{{1, 2}}}}}));

  // Rank 6 with span offset origin.
  EXPECT_THAT(MakeOffsetArray<int>({3, 4, 5, 6, 7, 8}, {{{{{{1, 2}}}}}}),
              MatchesArray<int>(span<const Index, 6>({3, 4, 5, 6, 7, 8}),
                                {{{{{{1, 2}}}}}}));

  // Mismatch due to elements.
  EXPECT_THAT(MakeArray<int>({1, 3}),
              ::testing::Not(MatchesArray<int>({1, 2})));

  // Mismatch due to rank.
  EXPECT_THAT(MakeArray<int>({1}), ::testing::Not(MatchesArray<int>({1, 2})));
}

}  // namespace
