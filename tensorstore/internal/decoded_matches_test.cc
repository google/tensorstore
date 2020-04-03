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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "tensorstore/util/status.h"

namespace {

using tensorstore::Status;
using tensorstore::internal::DecodedMatches;

Status Stride2Decoder(absl::string_view input, std::string* dest) {
  if (input.size() % 2 != 0) {
    return absl::InvalidArgumentError("");
  }
  dest->clear();
  for (std::size_t i = 0; i < input.size(); i += 2) {
    *dest += input[i];
  }
  return absl::OkStatus();
}

TEST(DecodedMatchesTest, Describe) {
  std::ostringstream ss;
  DecodedMatches("x", Stride2Decoder).DescribeTo(&ss);
  EXPECT_EQ("when decoded is equal to \"x\"", ss.str());
}

TEST(DecodedMatchesTest, ExplainValueMatcher) {
  ::testing::StringMatchResultListener listener;
  ::testing::ExplainMatchResult(
      DecodedMatches(::testing::ElementsAre('y'), Stride2Decoder), "xy",
      &listener);
  EXPECT_EQ("whose element #0 doesn't match", listener.str());
}

TEST(DecodedMatchesTest, ExplainDecodeError) {
  ::testing::StringMatchResultListener listener;
  ::testing::ExplainMatchResult(DecodedMatches("x", Stride2Decoder), "xyz",
                                &listener);
  EXPECT_EQ("Failed to decode value: INVALID_ARGUMENT: ", listener.str());
}

TEST(DecodedMatchesTest, Matches) {
  EXPECT_THAT("abcd", DecodedMatches("ac", Stride2Decoder));
  EXPECT_THAT("abc", ::testing::Not(DecodedMatches("ac", Stride2Decoder)));
  EXPECT_THAT("abcd", ::testing::Not(DecodedMatches("ab", Stride2Decoder)));
  EXPECT_THAT("abcd", DecodedMatches(::testing::Not("ab"), Stride2Decoder));
}

}  // namespace
