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

#include <sstream>
#include <string>
#include <string_view>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/util/status.h"

namespace {

using ::tensorstore::internal::DecodedMatches;

absl::Status Stride2Decoder(const absl::Cord& input, absl::Cord* dest) {
  if (input.size() % 2 != 0) {
    return absl::InvalidArgumentError("");
  }
  dest->Clear();
  for (std::size_t i = 0; i < input.size(); i += 2) {
    char x = input[i];
    dest->Append(std::string_view(&x, 1));
  }
  return absl::OkStatus();
}

TEST(DecodedMatchesTest, Describe) {
  std::ostringstream ss;
  DecodedMatches(absl::Cord("x"), Stride2Decoder).DescribeTo(&ss);
  EXPECT_EQ("when decoded is equal to x", ss.str());
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
  ::testing::ExplainMatchResult(DecodedMatches(absl::Cord("x"), Stride2Decoder),
                                absl::Cord("xyz"), &listener);
  EXPECT_EQ("Failed to decode value: INVALID_ARGUMENT: ", listener.str());
}

TEST(DecodedMatchesTest, Matches) {
  EXPECT_THAT(absl::Cord("abcd"),
              DecodedMatches(absl::Cord("ac"), Stride2Decoder));
  EXPECT_THAT(absl::Cord("abc"),
              ::testing::Not(DecodedMatches(absl::Cord("ac"), Stride2Decoder)));
  EXPECT_THAT(absl::Cord("abcd"),
              ::testing::Not(DecodedMatches(absl::Cord("ab"), Stride2Decoder)));
  EXPECT_THAT(absl::Cord("abcd"),
              DecodedMatches(::testing::Not(absl::Cord("ab")), Stride2Decoder));
}

}  // namespace
