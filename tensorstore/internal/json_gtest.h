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

#ifndef TENSORSTORE_INTERNAL_JSON_GTEST_H_
#define TENSORSTORE_INTERNAL_JSON_GTEST_H_

#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <nlohmann/json.hpp>

namespace nlohmann {
//
// Google Test uses PrintTo (with many overloads) to print the results of failed
// comparisons. Most of the time the default overloads for PrintTo() provided
// by Google Test magically do the job picking a good way to print things, but
// in this case there is a bug:
//     https://github.com/nlohmann/json/issues/709
// explicitly defining an overload, which must be in the namespace of the class,
// works around the problem, see ADL for why it must be in the right namespace:
//     https://en.wikipedia.org/wiki/Argument-dependent_name_lookup
//
/// Prints json objects to output streams from within Google Test.
inline void PrintTo(json const& j, std::ostream* os) { *os << j.dump(); }

}  // namespace nlohmann

namespace tensorstore {

/// GMock matcher that compares values using internal_json::JsonSame and prints
/// the differences in the case of a mismatch.
///
/// Note that unlike `operator==`, `JsonSame` correctly handles `discarded`.
::testing::Matcher<::nlohmann::json> MatchesJson(::nlohmann::json j);

/// GMock matcher that applies a matcher to the sub-value specified by the json
/// pointer.
///
/// The overload with `::nlohmann::json` as the `value_matcher` simply uses
/// `MatchesJson` to match the sub-value.
///
/// Example:
///
///     ::nlohmann::json obj{{"a", 123}, {"b", {{"c", "xyz"}}}};
///     EXPECT_THAT(obj, JsonSubValueMatches("/a", 123));
///     EXPECT_THAT(obj, JsonSubValueMatches("/b/c", "xyz"));
///     EXPECT_THAT(obj, JsonSubValueMatches("/b/c",
///                                         testing::Not(MatchesJson("xy"))));
::testing::Matcher<::nlohmann::json> JsonSubValueMatches(
    std::string json_pointer,
    ::testing::Matcher<::nlohmann::json> value_matcher);
::testing::Matcher<::nlohmann::json> JsonSubValueMatches(
    std::string json_pointer, ::nlohmann::json value_matcher);

/// GMock matcher that tests that all of the (json_pointer, value_matcher) pairs
/// match.
///
/// Example:
///
///     ::nlohmann::json obj{{"a", 123}, {"b", {{"c", "xyz"}}}};
///     EXPECT_THAT(obj, JsonSubValuesMatch({{"/a", 123}, {"/b/c", "xyz"}}));
::testing::Matcher<::nlohmann::json> JsonSubValuesMatch(
    std::vector<std::pair<std::string, ::nlohmann::json>> matchers);

}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_JSON_GTEST_H_
