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

#include <functional>
#include <ostream>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal {

namespace {
using DecodeFunction =
    std::function<Status(absl::string_view source, std::string* dest)>;

class Matcher : public ::testing::MatcherInterface<std::string> {
 public:
  Matcher(::testing::Matcher<std::string> value_matcher, DecodeFunction decoder)
      : value_matcher_(std::move(value_matcher)),
        decoder_(std::move(decoder)) {}

  bool MatchAndExplain(
      std::string value,
      ::testing::MatchResultListener* listener) const override {
    std::string decoded;
    auto status = decoder_(value, &decoded);
    if (!status.ok()) {
      *listener << "Failed to decode value: " << status;
      return false;
    }
    return value_matcher_.MatchAndExplain(decoded, listener);
  }

  void DescribeTo(std::ostream* os) const override {
    *os << "when decoded ";
    value_matcher_.DescribeTo(os);
  }

 private:
  ::testing::Matcher<std::string> value_matcher_;
  DecodeFunction decoder_;
};

}  // namespace

::testing::Matcher<std::string> DecodedMatches(
    ::testing::Matcher<std::string> value_matcher, DecodeFunction decoder) {
  return ::testing::MakeMatcher(
      new Matcher(std::move(value_matcher), std::move(decoder)));
}

}  // namespace internal
}  // namespace tensorstore
