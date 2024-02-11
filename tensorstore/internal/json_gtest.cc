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

#include "tensorstore/internal/json_gtest.h"

#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "tensorstore/internal/json/same.h"
#include "tensorstore/internal/json_pointer.h"
#include "tensorstore/util/quote_string.h"

namespace tensorstore {

namespace {
class JsonMatcherImpl : public ::testing::MatcherInterface<::nlohmann::json> {
 public:
  JsonMatcherImpl(::nlohmann::json value) : value_(std::move(value)) {}

  bool MatchAndExplain(
      ::nlohmann::json value_untyped,
      ::testing::MatchResultListener* listener) const override {
    if (!internal_json::JsonSame(value_, value_untyped)) {
      if (listener->IsInterested()) {
        *listener << "where the difference is:\n"
                  << ::nlohmann::json::diff(value_, value_untyped).dump(2);
      }
      return false;
    }
    return true;
  }

  void DescribeTo(std::ostream* os) const override {
    *os << "matches json " << value_;
  }

  void DescribeNegationTo(std::ostream* os) const override {
    *os << "does not match json " << value_;
  }

 private:
  ::nlohmann::json value_;
};

}  // namespace

::testing::Matcher<::nlohmann::json> MatchesJson(::nlohmann::json j) {
  return ::testing::MakeMatcher(new JsonMatcherImpl(std::move(j)));
}

namespace {
class JsonPointerMatcherImpl
    : public ::testing::MatcherInterface<::nlohmann::json> {
 public:
  JsonPointerMatcherImpl(std::string sub_value_pointer,
                         ::testing::Matcher<::nlohmann::json> sub_value_matcher)
      : sub_value_pointer_(std::move(sub_value_pointer)),
        sub_value_matcher_(std::move(sub_value_matcher)) {}

  bool MatchAndExplain(
      ::nlohmann::json value_untyped,
      ::testing::MatchResultListener* listener) const override {
    auto sub_value =
        json_pointer::Dereference(value_untyped, sub_value_pointer_);
    if (!sub_value.ok()) {
      if (listener->IsInterested()) {
        *listener << "where the pointer could not be resolved: "
                  << sub_value.status();
      }
      return false;
    }
    if (listener->IsInterested()) {
      ::testing::StringMatchResultListener s;
      if (!sub_value_matcher_.MatchAndExplain(**sub_value, &s)) {
        *listener << "whose sub value doesn't match";
        auto str = s.str();
        if (!str.empty()) {
          *listener << ", " << str;
        }
        return false;
      }
      return true;
    }
    return sub_value_matcher_.Matches(**sub_value);
  }

  void DescribeTo(std::ostream* os) const override {
    *os << "has sub value " << tensorstore::QuoteString(sub_value_pointer_)
        << " that ";
    sub_value_matcher_.DescribeTo(os);
  }

  void DescribeNegationTo(std::ostream* os) const override {
    *os << "does not have sub value "
        << tensorstore::QuoteString(sub_value_pointer_) << " that ";
    sub_value_matcher_.DescribeTo(os);
  }

 private:
  std::string sub_value_pointer_;
  ::testing::Matcher<nlohmann::json> sub_value_matcher_;
};

}  // namespace
::testing::Matcher<::nlohmann::json> JsonSubValueMatches(
    std::string json_pointer,
    ::testing::Matcher<::nlohmann::json> value_matcher) {
  return ::testing::MakeMatcher(new JsonPointerMatcherImpl(
      std::move(json_pointer), std::move(value_matcher)));
}

::testing::Matcher<::nlohmann::json> JsonSubValueMatches(
    std::string json_pointer, ::nlohmann::json value_matcher) {
  return JsonSubValueMatches(std::move(json_pointer),
                             MatchesJson(std::move(value_matcher)));
}

::testing::Matcher<::nlohmann::json> JsonSubValuesMatch(
    std::vector<std::pair<std::string, ::nlohmann::json>> matchers) {
  std::vector<::testing::Matcher<::nlohmann::json>> all;
  all.reserve(matchers.size());
  for (const auto& p : matchers) {
    all.push_back(JsonSubValueMatches(p.first, p.second));
  }
  return ::testing::AllOfArray(all);
}

}  // namespace tensorstore
