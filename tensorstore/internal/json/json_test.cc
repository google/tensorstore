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

#include "tensorstore/internal/json/json.h"

#include <map>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/internal/json_fwd.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::nlohmann::json;
using ::tensorstore::MatchesStatus;
using ::tensorstore::internal::ParseJson;
using ::tensorstore::internal_json::JsonParseArray;
using ::tensorstore::internal_json::JsonValidateArrayLength;

TEST(JsonTest, SimpleParse) {
  const char kArray[] = R"({ "foo": "bar" })";

  auto x = ParseJson("");  // std::string_view
  EXPECT_TRUE(x.is_discarded());

  // Test parsing objects.
  auto y = ParseJson(kArray);  // std::string_view
  EXPECT_FALSE(y.is_discarded());

  auto one = ParseJson("1");  // std::string_view
  EXPECT_FALSE(one.is_discarded());
}

TEST(JsonParseArrayTest, Basic) {
  bool size_received = false;
  std::vector<std::pair<::nlohmann::json, std::ptrdiff_t>> elements;
  EXPECT_EQ(absl::OkStatus(),
            JsonParseArray(
                ::nlohmann::json{1, 2, 3},
                [&](std::ptrdiff_t s) {
                  EXPECT_EQ(3, s);
                  size_received = true;
                  return JsonValidateArrayLength(s, 3);
                },
                [&](const ::nlohmann::json& j, std::ptrdiff_t i) {
                  EXPECT_TRUE(size_received);
                  elements.emplace_back(j, i);
                  return absl::OkStatus();
                }));
  EXPECT_TRUE(size_received);
  EXPECT_THAT(elements, ::testing::ElementsAre(::testing::Pair(1, 0),
                                               ::testing::Pair(2, 1),
                                               ::testing::Pair(3, 2)));
}

TEST(JsonParseArrayTest, NotArray) {
  EXPECT_THAT(JsonParseArray(
                  ::nlohmann::json(3),
                  [&](std::ptrdiff_t s) { return absl::OkStatus(); },
                  [&](const ::nlohmann::json& j, std::ptrdiff_t i) {
                    return absl::OkStatus();
                  }),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Expected array, but received: 3"));
}

TEST(JsonValidateArrayLength, Success) {
  EXPECT_EQ(absl::OkStatus(), JsonValidateArrayLength(3, 3));
}

TEST(JsonValidateArrayLength, Failure) {
  EXPECT_THAT(JsonValidateArrayLength(3, 4),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Array has length 3 but should have length 4"));
}

TEST(JsonParseArrayTest, SizeCallbackError) {
  EXPECT_THAT(
      JsonParseArray(
          ::nlohmann::json{1, 2, 3},
          [&](std::ptrdiff_t s) { return absl::UnknownError("size_callback"); },
          [&](const ::nlohmann::json& j, std::ptrdiff_t i) {
            return absl::OkStatus();
          }),
      MatchesStatus(absl::StatusCode::kUnknown, "size_callback"));
}

TEST(JsonParseArrayTest, ElementCallbackError) {
  EXPECT_THAT(JsonParseArray(
                  ::nlohmann::json{1, 2, 3},
                  [&](std::ptrdiff_t s) { return absl::OkStatus(); },
                  [&](const ::nlohmann::json& j, std::ptrdiff_t i) {
                    if (i == 0) return absl::OkStatus();
                    return absl::UnknownError("element");
                  }),
              MatchesStatus(absl::StatusCode::kUnknown,
                            "Error parsing value at position 1: element"));
}

TEST(JsonSame, Basic) {
  EXPECT_TRUE(tensorstore::internal_json::JsonSame(1.0, 1));
  EXPECT_FALSE(tensorstore::internal_json::JsonSame(
      ::nlohmann::json::value_t::discarded, ::nlohmann::json::value_t::null));
  EXPECT_TRUE(tensorstore::internal_json::JsonSame(
      ::nlohmann::json::value_t::discarded,
      ::nlohmann::json::value_t::discarded));
  EXPECT_TRUE(tensorstore::internal_json::JsonSame({1, 2, 3}, {1, 2, 3}));
  EXPECT_TRUE(tensorstore::internal_json::JsonSame(
      {1, {1, 2, 3, {{"a", 5}, {"b", 7}}}, 3},
      {1, {1, 2, 3, {{"a", 5}, {"b", 7}}}, 3}));
  EXPECT_TRUE(tensorstore::internal_json::JsonSame(
      ::nlohmann::json::array_t{}, ::nlohmann::json::array_t{}));
  EXPECT_TRUE(tensorstore::internal_json::JsonSame(
      ::nlohmann::json::object_t{}, ::nlohmann::json::object_t{}));
  EXPECT_FALSE(tensorstore::internal_json::JsonSame({1, 2, 3}, {1, 2, 4}));
  EXPECT_FALSE(tensorstore::internal_json::JsonSame({1, 2, 3}, {1, 2}));
  EXPECT_TRUE(tensorstore::internal_json::JsonSame(
      {1, ::nlohmann::json::value_t::discarded, 3},
      {1, ::nlohmann::json::value_t::discarded, 3}));
  EXPECT_TRUE(tensorstore::internal_json::JsonSame(
      {{"a", ::nlohmann::json::value_t::discarded}, {"b", 3}},
      {{"a", ::nlohmann::json::value_t::discarded}, {"b", 3}}));
  EXPECT_FALSE(tensorstore::internal_json::JsonSame(
      {{"a", ::nlohmann::json::value_t::discarded}, {"b", 3}},
      {{"a", ::nlohmann::json::value_t::discarded}, {"b", 4}}));
  EXPECT_FALSE(tensorstore::internal_json::JsonSame(
      {{"a", ::nlohmann::json::value_t::discarded}, {"b", 3}},
      {{"a", ::nlohmann::json::value_t::discarded}, {"c", 3}}));
  EXPECT_FALSE(tensorstore::internal_json::JsonSame(
      {{"a", ::nlohmann::json::value_t::discarded}, {"b", 3}},
      {{"a", ::nlohmann::json::value_t::discarded}, {"b", 3}, {"d", 4}}));

  const auto make_nested = [](int depth) {
    ::nlohmann::json value;
    ::nlohmann::json* tail = &value;
    for (int i = 0; i < depth; ++i) {
      *tail = ::nlohmann::json::object_t();
      auto& obj = tail->get_ref<::nlohmann::json::object_t&>();
      tail = &obj["a"];
    }
    return value;
  };
  auto nested = make_nested(10000);
  EXPECT_TRUE(tensorstore::internal_json::JsonSame(nested, nested));
}

}  // namespace
