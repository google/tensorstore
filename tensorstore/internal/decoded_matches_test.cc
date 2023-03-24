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

#include "tensorstore/internal/decoded_matches.h"

#include <cstddef>
#include <sstream>
#include <string>
#include <string_view>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "tensorstore/util/result.h"

namespace {

using ::tensorstore::internal::DecodedMatches;

tensorstore::Result<std::string> Stride2Decoder(std::string_view input) {
  if (input.size() % 2 != 0) {
    return absl::InvalidArgumentError("");
  }
  std::string output;
  for (std::size_t i = 0; i < input.size(); i += 2) {
    output += input[i];
  }
  return output;
}

TEST(DecodedMatchesTest, Describe) {
  std::ostringstream ss;
  DecodedMatches("x", Stride2Decoder).DescribeTo(&ss);
  EXPECT_EQ("when decoded is equal to \"x\"", ss.str());
}

TEST(DecodedMatchesTest, ExplainValueMatcher) {
  ::testing::StringMatchResultListener listener;
  ::testing::ExplainMatchResult(
      DecodedMatches(::testing::SizeIs(3), Stride2Decoder), absl::Cord("xy"),
      &listener);
  EXPECT_EQ("whose size 1 doesn't match", listener.str());
}

TEST(DecodedMatchesTest, ExplainDecodeError) {
  ::testing::StringMatchResultListener listener;
  ::testing::ExplainMatchResult(DecodedMatches("x", Stride2Decoder),
                                absl::Cord("xyz"), &listener);
  EXPECT_EQ("Failed to decode value: INVALID_ARGUMENT: ", listener.str());
}

TEST(DecodedMatchesTest, Matches) {
  EXPECT_THAT(absl::Cord("abcd"), DecodedMatches("ac", Stride2Decoder));
  EXPECT_THAT(absl::Cord("abc"),
              ::testing::Not(DecodedMatches("ac", Stride2Decoder)));
  EXPECT_THAT(absl::Cord("abcd"),
              ::testing::Not(DecodedMatches("ab", Stride2Decoder)));
  EXPECT_THAT(absl::Cord("abcd"),
              DecodedMatches(::testing::Not("ab"), Stride2Decoder));
}

}  // namespace
