// Copyright 2023 The TensorStore Authors
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

#include "tensorstore/internal/json_gtest.h"

#include <sstream>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

namespace {

using ::tensorstore::JsonSubValueMatches;
using ::tensorstore::JsonSubValuesMatch;
using ::tensorstore::MatchesJson;

template <typename MatcherType>
std::string Describe(const MatcherType& m) {
  std::ostringstream ss;
  m.DescribeTo(&ss);
  return ss.str();
}

// Returns the reason why x matches, or doesn't match, m.
template <typename MatcherType, typename Value>
std::string Explain(const MatcherType& m, const Value& x) {
  testing::StringMatchResultListener listener;
  ExplainMatchResult(m, x, &listener);
  return listener.str();
}

TEST(JsonSubValueMatchesTest, Example) {
  ::nlohmann::json obj{{"a", 123}, {"b", {{"c", "xyz"}}}};
  EXPECT_THAT(obj, JsonSubValueMatches("/a", 123));
  EXPECT_THAT(obj, JsonSubValueMatches("/b/c", "xyz"));
  EXPECT_THAT(obj,
              JsonSubValueMatches("/b/c", ::testing::Not(MatchesJson("xy"))));

  EXPECT_THAT(Describe(JsonSubValueMatches("/a", 123)),
              "has sub value \"/a\" that matches json 123");
  EXPECT_THAT(Explain(JsonSubValueMatches("/a", 124), obj),
              ::testing::StartsWith(
                  "whose sub value doesn't match, where the difference is:"));
}

TEST(JsonSubValuesMatchTest, Example) {
  ::nlohmann::json obj{{"a", 123}, {"b", {{"c", "xyz"}}}};
  EXPECT_THAT(obj, JsonSubValuesMatch({{"/a", 123}, {"/b/c", "xyz"}}));
}

}  // namespace
