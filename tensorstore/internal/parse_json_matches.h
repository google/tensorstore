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

#ifndef TENSORSTORE_INTERNAL_PARSE_JSON_MATCHES_H_
#define TENSORSTORE_INTERNAL_PARSE_JSON_MATCHES_H_

#include <string>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "tensorstore/internal/json_gtest.h"

namespace tensorstore {
namespace internal {

/// Returns a GoogleMock `std::string` matcher that matches if `json_matcher`
/// matches the result of `ParseJson` on the specified string.
///
/// Example usage:
///
///     EXPECT_THAT("{\"a\":\"b\"}",
///                 ParseJsonMatches(::nlohmann::json{{"a", "b"}}));
::testing::Matcher<std::string> ParseJsonMatches(
    ::testing::Matcher<::nlohmann::json> json_matcher);

/// Equivalent to `ParseJsonMatches(MatchesJson(json))`.
::testing::Matcher<std::string> ParseJsonMatches(::nlohmann::json json);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_PARSE_JSON_MATCHES_H_
