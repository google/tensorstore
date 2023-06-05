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

#include "tensorstore/internal/json/same.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

namespace {

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
