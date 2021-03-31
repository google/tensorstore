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
#include "tensorstore/util/status_testutil.h"

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

/// Tests that a sequence of examples for json binding of `T` values round
/// trips.
///
/// \param round_trips Sequence of round trip pairs to test.
/// \param binder Optional.  The JSON binder to use.
/// \param to_json_options Optional.  Options for converting to JSON.
/// \param from_json_options Optional.  Options for converting from JSON.
template <typename T,
          typename Binder = decltype(internal_json_binding::DefaultBinder<>),
          typename ToJsonOptions = IncludeDefaults,
          typename FromJsonOptions = internal_json_binding::NoOptions>
void TestJsonBinderRoundTrip(
    std::vector<std::pair<T, ::nlohmann::json>> round_trips,
    Binder binder = internal_json_binding::DefaultBinder<>,
    ToJsonOptions to_json_options = IncludeDefaults{false},
    FromJsonOptions from_json_options = {}) {
  for (const auto& [value, j] : round_trips) {
    SCOPED_TRACE(tensorstore::StrCat("value=", value, ", j=", j));
    EXPECT_THAT(tensorstore::internal_json_binding::ToJson(value, binder,
                                                           to_json_options),
                ::testing::Optional(MatchesJson(j)));
    EXPECT_THAT(tensorstore::internal_json_binding::FromJson<T>(
                    j, binder, from_json_options),
                ::testing::Optional(value));
  }
}

/// Tests that a sequence of json values can be parsed as `T` values then
/// converted back to identical json values.
///
/// This may be used in place of `TestJsonBinderRoundTrip` for types `T` for
/// which `operator==` is not defined.
///
/// \param round_trips Sequence of JSON values to test.
/// \param binder Optional.  The JSON binder to use.
/// \param to_json_options Optional.  Options for converting to JSON.
/// \param from_json_options Optional.  Options for converting from JSON.
template <typename T,
          typename Binder = decltype(internal_json_binding::DefaultBinder<>),
          typename ToJsonOptions = IncludeDefaults,
          typename FromJsonOptions = internal_json_binding::NoOptions>
void TestJsonBinderRoundTripJsonOnly(
    std::vector<::nlohmann::json> round_trips,
    Binder binder = internal_json_binding::DefaultBinder<>,
    ToJsonOptions to_json_options = IncludeDefaults{false},
    FromJsonOptions from_json_options = {}) {
  for (const auto& j : round_trips) {
    SCOPED_TRACE(tensorstore::StrCat("j=", j));
    auto result = tensorstore::internal_json_binding::FromJson<T>(
        j, binder, from_json_options);
    TENSORSTORE_EXPECT_OK(result) << "FromJson";
    if (result.ok()) {
      EXPECT_THAT(tensorstore::internal_json_binding::ToJson(*result, binder,
                                                             to_json_options),
                  ::testing::Optional(MatchesJson(j)));
    }
  }
}

/// Tests that for a sequence of pairs `(a, b)` of json values, both `a` and `b`
/// can be parsed as `T` values which convert back to `b`.
///
/// \param round_trips Sequence of JSON values to test.
/// \param binder Optional.  The JSON binder to use.
/// \param to_json_options Optional.  Options for converting to JSON.
/// \param from_json_options Optional.  Options for converting from JSON.
template <typename T,
          typename Binder = decltype(internal_json_binding::DefaultBinder<>),
          typename ToJsonOptions = IncludeDefaults,
          typename FromJsonOptions = internal_json_binding::NoOptions>
void TestJsonBinderRoundTripJsonOnlyInexact(
    std::vector<std::pair<::nlohmann::json, ::nlohmann::json>> round_trips,
    Binder binder = internal_json_binding::DefaultBinder<>,
    ToJsonOptions to_json_options = IncludeDefaults{false},
    FromJsonOptions from_json_options = {}) {
  for (const auto& [a, b] : round_trips) {
    SCOPED_TRACE(tensorstore::StrCat("a=", a, ", b=", b));
    auto a_result = tensorstore::internal_json_binding::FromJson<T>(
        a, binder, from_json_options);
    TENSORSTORE_EXPECT_OK(a_result) << "FromJson: a=" << a;
    auto b_result = tensorstore::internal_json_binding::FromJson<T>(
        b, binder, from_json_options);
    TENSORSTORE_EXPECT_OK(b_result) << "FromJson: b=" << b;
    if (a_result.ok()) {
      EXPECT_THAT(tensorstore::internal_json_binding::ToJson(*a_result, binder,
                                                             to_json_options),
                  ::testing::Optional(MatchesJson(b)));
    }
    if (b_result.ok()) {
      EXPECT_THAT(tensorstore::internal_json_binding::ToJson(*b_result, binder,
                                                             to_json_options),
                  ::testing::Optional(MatchesJson(b)));
    }
  }
}

/// Tests a sequence of examples for converting `T` values to JSON.
///
/// \param to_json_cases Sequence of conversions to json to test (useful for
///     error cases or other cases that don't round trip).
/// \param binder Optional.  The JSON binder to use.
/// \param to_json_options Optional.  Options for converting to JSON.
template <typename T,
          typename Binder = decltype(internal_json_binding::DefaultBinder<>),
          typename ToJsonOptions = IncludeDefaults>
void TestJsonBinderToJson(
    std::vector<std::pair<T, ::testing::Matcher<Result<::nlohmann::json>>>>
        to_json_cases,
    Binder binder = internal_json_binding::DefaultBinder<>,
    ToJsonOptions to_json_options = IncludeDefaults{false}) {
  for (const auto& [value, matcher] : to_json_cases) {
    SCOPED_TRACE(tensorstore::StrCat("value=", value));
    EXPECT_THAT(tensorstore::internal_json_binding::ToJson(value, binder,
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
          typename Binder = decltype(internal_json_binding::DefaultBinder<>),
          typename FromJsonOptions = internal_json_binding::NoOptions>
void TestJsonBinderFromJson(
    std::vector<std::pair<::nlohmann::json, ::testing::Matcher<Result<T>>>>
        from_json_cases,
    Binder binder = internal_json_binding::DefaultBinder<>,
    FromJsonOptions from_json_options = {}) {
  for (const auto& [j, matcher] : from_json_cases) {
    SCOPED_TRACE(tensorstore::StrCat("j=", j));
    EXPECT_THAT(tensorstore::internal_json_binding::FromJson<T>(
                    j, binder, from_json_options),
                matcher);
  }
}

/// Verifies that each `T` value is equal to itself and not equal to any of the
/// other values.
///
/// Also verifies that every JSON value round trips.
template <typename T,
          typename Binder = decltype(internal_json_binding::DefaultBinder<>)>
void TestCompareDistinctFromJson(
    std::vector<::nlohmann::json> specs,
    Binder binder = internal_json_binding::DefaultBinder<>) {
  tensorstore::TestJsonBinderRoundTripJsonOnly<T>(specs, binder);
  std::vector<T> values(specs.size());
  for (size_t i = 0; i < specs.size(); ++i) {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        values[i], internal_json_binding::FromJson<T>(specs[i], binder));
  }
  for (size_t i = 0; i < specs.size(); ++i) {
    EXPECT_EQ(values[i], values[i]) << "specs[" << i << "]=" << specs[i];
    for (size_t j = i + 1; j < specs.size(); ++j) {
      EXPECT_NE(values[i], values[j]) << "specs[" << i << "]=" << specs[i]
                                      << ", specs[" << j << "]=" << specs[j];
    }
  }
}

}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_JSON_GTEST_H_
