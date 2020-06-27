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

#include <ostream>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "tensorstore/internal/json.h"
#include "tensorstore/internal/json_gtest.h"

namespace tensorstore {
namespace internal {

namespace {
class Matcher : public ::testing::MatcherInterface<std::string> {
 public:
  Matcher(::testing::Matcher<::nlohmann::json> json_matcher)
      : json_matcher_(std::move(json_matcher)) {}

  bool MatchAndExplain(
      std::string value,
      ::testing::MatchResultListener* listener) const override {
    return json_matcher_.MatchAndExplain(
        tensorstore::internal::ParseJson(value), listener);
  }

  void DescribeTo(std::ostream* os) const override {
    *os << "when parsed as JSON ";
    json_matcher_.DescribeTo(os);
  }

 private:
  ::testing::Matcher<::nlohmann::json> json_matcher_;
};

}  // namespace

::testing::Matcher<std::string> ParseJsonMatches(
    ::testing::Matcher<::nlohmann::json> json_matcher) {
  return ::testing::MakeMatcher(new Matcher(std::move(json_matcher)));
}

::testing::Matcher<std::string> ParseJsonMatches(::nlohmann::json json) {
  return ParseJsonMatches(MatchesJson(json));
}

}  // namespace internal
}  // namespace tensorstore
