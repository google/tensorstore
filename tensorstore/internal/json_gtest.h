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
#include <vector>

#include <gmock/gmock.h>
#include <nlohmann/json.hpp>
#include "tensorstore/internal/json.h"

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

MATCHER_P(MatchesJson, j, "") {
  *result_listener << "where the difference is:\n"
                   << ::nlohmann::json::diff(j, arg).dump(2);
  return tensorstore::internal_json::JsonSame(arg, j);
}

/// Tests that a sequence of examples for json binding of `T` values round
/// trips.
///
/// \param round_trips Sequence of round trip pairs to test.
/// \param binder Optional.  The JSON binder to use.
/// \param to_json_options Optional.  Options for converting to JSON.
/// \param from_json_options Optional.  Options for converting from JSON.
template <typename T,
          typename Binder = decltype(internal::json_binding::DefaultBinder<>),
          typename ToJsonOptions = IncludeDefaults,
          typename FromJsonOptions = internal::json_binding::NoOptions>
void TestJsonBinderRoundTrip(
    std::vector<std::pair<T, ::nlohmann::json>> round_trips,
    Binder binder = internal::json_binding::DefaultBinder<>,
    ToJsonOptions to_json_options = IncludeDefaults{true},
    FromJsonOptions from_json_options = {}) {
  for (const auto& [value, j] : round_trips) {
    SCOPED_TRACE(tensorstore::StrCat("value=", value, ", j=", j));
    EXPECT_THAT(tensorstore::internal::json_binding::ToJson(value, binder,
                                                            to_json_options),
                ::testing::Optional(MatchesJson(j)));
    EXPECT_THAT(tensorstore::internal::json_binding::FromJson<T>(
                    j, binder, from_json_options),
                ::testing::Optional(value));
  }
}

/// Tests a sequence of examples for converting `T` values to JSON.
///
/// \param to_json_cases Sequence of conversions to json to test (useful for
///     error cases or other cases that don't round trip).
/// \param binder Optional.  The JSON binder to use.
/// \param to_json_options Optional.  Options for converting to JSON.
template <typename T,
          typename Binder = decltype(internal::json_binding::DefaultBinder<>),
          typename ToJsonOptions = IncludeDefaults>
void TestJsonBinderToJson(
    std::vector<std::pair<T, ::testing::Matcher<Result<::nlohmann::json>>>>
        to_json_cases,
    Binder binder = internal::json_binding::DefaultBinder<>,
    ToJsonOptions to_json_options = IncludeDefaults{true}) {
  for (const auto& [value, matcher] : to_json_cases) {
    SCOPED_TRACE(tensorstore::StrCat("value=", value));
    EXPECT_THAT(tensorstore::internal::json_binding::ToJson(value, binder,
                                                            to_json_options),
                matcher);
  }
}

/// Tests a sequence of examples for converting `T` values from JSON.
///
/// \param from_json_cases Sequence of conversions from json to test (useful for
///     error cases or other cases that don't round trip).
/// \param binder Optional.  The JSON binder to use.
/// \param from_json_options Optional.  Options for converting from JSON.
template <typename T,
          typename Binder = decltype(internal::json_binding::DefaultBinder<>),
          typename FromJsonOptions = internal::json_binding::NoOptions>
void TestJsonBinderFromJson(
    std::vector<std::pair<::nlohmann::json, ::testing::Matcher<Result<T>>>>
        from_json_cases,
    Binder binder = internal::json_binding::DefaultBinder<>,
    FromJsonOptions from_json_options = {}) {
  for (const auto& [j, matcher] : from_json_cases) {
    SCOPED_TRACE(tensorstore::StrCat("j=", j));
    EXPECT_THAT(tensorstore::internal::json_binding::FromJson<T>(
                    j, binder, from_json_options),
                matcher);
  }
}

}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_JSON_GTEST_H_
