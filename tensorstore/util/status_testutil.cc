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

#include "tensorstore/util/status_testutil.h"

#include <ostream>
#include <regex>  // NOLINT
#include <string>
#include <system_error>  // NOLINT

#include <gtest/gtest.h>
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal_status {

ErrorCodeMatchesStatusMatcher::ErrorCodeMatchesStatusMatcher(
    absl::StatusCode status_code, const std::string& message_pattern)
    : status_code_(status_code), message_pattern_(message_pattern) {}

bool ErrorCodeMatchesStatusMatcher::MatchAndExplain(
    const Status& status, ::testing::MatchResultListener* listener) const {
  auto message = status.message();
  return status.code() == status_code_ &&
         std::regex_match(std::begin(message), std::end(message),
                          std::regex(message_pattern_));
}

void ErrorCodeMatchesStatusMatcher::DescribeTo(::std::ostream* os) const {
  *os << "matches error code " << status_code_;
  if (message_pattern_ != "[^]*") {
    *os << " and message pattern ";
    ::testing::internal::UniversalPrint(message_pattern_, os);
  }
}

void ErrorCodeMatchesStatusMatcher::DescribeNegationTo(
    ::std::ostream* os) const {
  *os << "does not match error code " << status_code_;
  if (message_pattern_ != "[^]*") {
    *os << " and message pattern ";
    ::testing::internal::UniversalPrint(message_pattern_, os);
  }
}

}  // namespace internal_status

::testing::PolymorphicMatcher<internal_status::ErrorCodeMatchesStatusMatcher>
MatchesStatus(absl::StatusCode status_code,
              const std::string& message_pattern) {
  return ::testing::MakePolymorphicMatcher(
      internal_status::ErrorCodeMatchesStatusMatcher(status_code,
                                                     message_pattern));
}

}  // namespace tensorstore
