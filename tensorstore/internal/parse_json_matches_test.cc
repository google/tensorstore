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

#include "tensorstore/internal/parse_json_matches.h"

#include <sstream>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

namespace {

using tensorstore::internal::ParseJsonMatches;

TEST(ParseJsonMatchesTest, Describe) {
  std::ostringstream ss;
  ParseJsonMatches(::nlohmann::json(true)).DescribeTo(&ss);
  EXPECT_EQ("when parsed as JSON matches json true", ss.str());
}

TEST(ParseJsonMatchesTest, Explain) {
  ::testing::StringMatchResultListener listener;
  ::testing::ExplainMatchResult(ParseJsonMatches(::nlohmann::json(true)),
                                "false", &listener);
  EXPECT_EQ(
      "where the difference is:\n"
      "[\n"
      "  {\n"
      "    \"op\": \"replace\",\n"
      "    \"path\": \"\",\n"
      "    \"value\": false\n"
      "  }\n"
      "]",
      listener.str());
}

TEST(ParseJsonMatchesTest, Matches) {
  EXPECT_THAT("{\"a\":\"b\"}", ParseJsonMatches(::nlohmann::json{{"a", "b"}}));
  EXPECT_THAT("{\"a\":\"b\"}",
              ::testing::Not(ParseJsonMatches(::nlohmann::json{{"a", "c"}})));
  EXPECT_THAT("invalid",
              ::testing::Not(ParseJsonMatches(::nlohmann::json{{"a", "c"}})));
  EXPECT_THAT("{\"a\":\"b\"}",
              ParseJsonMatches(::testing::Not(::nlohmann::json{{"a", "c"}})));
}

}  // namespace
