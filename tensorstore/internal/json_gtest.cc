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

#include <gmock/gmock.h>
#include "tensorstore/internal/json/json.h"

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

}  // namespace tensorstore
