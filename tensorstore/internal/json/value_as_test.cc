// Copyright 2021 The TensorStore Authors
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

#include "tensorstore/internal/json/value_as.h"

#include <stdint.h>

#include <map>
#include <optional>
#include <set>
#include <string>
#include <type_traits>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::MatchesStatus;
using ::tensorstore::internal_json::JsonRequireInteger;
using ::tensorstore::internal_json::JsonRequireValueAs;
using ::tensorstore::internal_json::JsonValueAs;

template <typename T, bool kStrict = true>
std::optional<T> JsonMemberT(const ::nlohmann::json::object_t& j,
                             const char* member) {
  auto it = j.find(member);
  if (it == j.end()) {
    return std::nullopt;
  }
  return JsonValueAs<T>(it->second, kStrict);
}

template <typename T, bool kStrict = true>
std::optional<T> JsonMemberT(const ::nlohmann::json& j, const char* member) {
  if (const auto* obj = j.get_ptr<const ::nlohmann::json::object_t*>()) {
    return JsonMemberT<T, kStrict>(*obj, member);
  }
  return std::nullopt;
}

TEST(JsonTest, Meta) {
  auto JsonRequireString = [](const ::nlohmann::json& json,
                              const char* member) -> bool {
    auto v = JsonMemberT<std::string>(json, member);
    return v.has_value() && !v->empty();
  };
  auto JsonRequireInt = [](const ::nlohmann::json& json,
                           const char* member) -> bool {
    auto v = JsonMemberT<int64_t, false>(json, member);
    return v.has_value();
  };

  auto meta = ::nlohmann::json::meta();

  EXPECT_TRUE(JsonRequireString(meta, "copyright"));
  EXPECT_TRUE(JsonRequireString(meta, "name"));
  EXPECT_TRUE(JsonRequireString(meta, "url"));
  EXPECT_TRUE(JsonRequireString(meta, "platform"));
  EXPECT_TRUE(JsonRequireString(meta, "copyright"));

  EXPECT_TRUE(meta.find("compiler") != meta.end());

  auto compiler = meta["compiler"];
  EXPECT_TRUE(JsonRequireString(compiler, "c++"));

  EXPECT_FALSE(JsonRequireString(meta, "version"));
  auto version = meta["version"];
  EXPECT_TRUE(JsonRequireInt(version, "major"));
}

::nlohmann::json GetDefaultJSON() {
  return ::nlohmann::json{
      {"bool_true", true}, {"bool_false", false},  {"str_bool", "true"},
      {"signed", 456},     {"neg_signed", -567},   {"unsigned", 565u},
      {"float", 456.789},  {"neg_float", -678.91}, {"int_float", 122.0},
      {"str", "abc"},      {"str_number", "789"},  {"str_float", "123.40"},
      {"nil", nullptr},    {"empty_obj", {}},      {"obj", {"a", 1}},
  };
}

std::set<std::string> GetKeys() {
  return std::set<std::string>{{
      "bool_true",
      "bool_false",
      "str_bool",
      "signed",
      "neg_signed",
      "unsigned",
      "float",
      "neg_float",
      "int_float",
      "str",
      "abc",
      "str_number",
      "str_float",
      "nil",
      "empty_obj",
      "obj",
      "missing",
  }};
}

TEST(JsonTest, JsonParseBool) {
  auto keys = GetKeys();

  auto JsonParseBool = [&keys](const ::nlohmann::json& json,
                               const char* member) {
    keys.erase(member);
    return JsonMemberT<bool, false>(json, member);
  };

  auto result = GetDefaultJSON();
  EXPECT_FALSE(result.is_discarded());

  // Some values can be parsed.
  ASSERT_TRUE(JsonParseBool(result, "bool_true"));
  EXPECT_EQ(true, *JsonParseBool(result, "bool_true"));

  ASSERT_TRUE(JsonParseBool(result, "bool_false"));
  EXPECT_EQ(false, *JsonParseBool(result, "bool_false"));

  ASSERT_TRUE(JsonParseBool(result, "str_bool"));
  EXPECT_EQ(true, *JsonParseBool(result, "str_bool"));

  // Some values cannot be parsed.
  std::set<std::string> remaining = keys;
  for (const std::string& x : remaining) {
    EXPECT_FALSE(JsonParseBool(result, x.c_str())) << x;
  }

  EXPECT_EQ(std::nullopt, JsonValueAs<bool>(::nlohmann::json("a")));
  EXPECT_EQ(false, JsonValueAs<bool>(::nlohmann::json("false")));
  EXPECT_EQ(true, JsonValueAs<bool>(::nlohmann::json("true")));

  const bool kStrict = true;
  EXPECT_EQ(std::nullopt, JsonValueAs<bool>(::nlohmann::json("true"), kStrict));
  EXPECT_EQ(true, JsonValueAs<bool>(::nlohmann::json(true), kStrict));
  EXPECT_EQ(false, JsonValueAs<bool>(::nlohmann::json(false), kStrict));
}

TEST(JsonValueAsTest, Int64FromUint64) {
  EXPECT_EQ(std::nullopt,
            JsonValueAs<int64_t>(::nlohmann::json(0x8fffffffffffffffu)));
  EXPECT_EQ(std::nullopt,
            JsonValueAs<int64_t>(::nlohmann::json(0xffffffffffffffffu)));
  EXPECT_EQ(0x7fffffffffffffff,
            JsonValueAs<int64_t>(::nlohmann::json(0x7fffffffffffffffu)));

  const bool kStrict = true;
  EXPECT_EQ(
      0x7fffffffffffffff,
      JsonValueAs<int64_t>(::nlohmann::json(0x7fffffffffffffffu), kStrict));
}

TEST(JsonValueAsTest, Int64FromDouble) {
  EXPECT_EQ(std::nullopt, JsonValueAs<int64_t>(::nlohmann::json(0.5)));
  EXPECT_EQ(1, JsonValueAs<int64_t>(::nlohmann::json(1.0)));

  // Test smallest positive integer that can be exactly represented as a double
  // but not as an int64.
  EXPECT_EQ(
      std::nullopt,
      JsonValueAs<int64_t>(::nlohmann::json(9223372036854775808.0 /*=2^63*/)));

  // Test largest negative integer that can be exactly represented as a double
  // but not as an int64.
  EXPECT_EQ(std::nullopt,
            JsonValueAs<int64_t>(::nlohmann::json(-9223372036854777856.0)));

  // Test largest integer that can be exactly represented as both a double and
  // an int64.
  EXPECT_EQ(9223372036854774784,
            JsonValueAs<int64_t>(::nlohmann::json(9223372036854774784.0)));

  // Test -2^63, which can be exactly represented as a double.
  EXPECT_EQ(
      -0x8000000000000000,
      JsonValueAs<int64_t>(::nlohmann::json(-9223372036854775808.0 /*=2^63*/)));
}

TEST(JsonValueAsTest, Int64FromString) {
  EXPECT_EQ(-1, JsonValueAs<int64_t>(::nlohmann::json("-1")));
  EXPECT_EQ(-0x8000000000000000,
            JsonValueAs<int64_t>(::nlohmann::json("-9223372036854775808")));
  EXPECT_EQ(0x7fffffffffffffff,
            JsonValueAs<int64_t>(::nlohmann::json("9223372036854775807")));

  EXPECT_FALSE(JsonValueAs<int64_t>(::nlohmann::json("0.0")));
  EXPECT_FALSE(JsonValueAs<int64_t>(::nlohmann::json("0a")));
  EXPECT_FALSE(JsonValueAs<int64_t>(::nlohmann::json("0x0")));
  EXPECT_FALSE(JsonValueAs<int64_t>(::nlohmann::json("0xf")));

  // Values out of bounds.
  EXPECT_FALSE(JsonValueAs<int64_t>(::nlohmann::json("9223372036854775808")));
  EXPECT_FALSE(JsonValueAs<int64_t>(::nlohmann::json("-9223372036854775809")));

  const bool kStrict = true;
  EXPECT_EQ(std::nullopt,
            JsonValueAs<int64_t>(::nlohmann::json("-1"), kStrict));
}

TEST(JsonValueAsTest, Uint64FromDouble) {
  EXPECT_EQ(std::nullopt, JsonValueAs<uint64_t>(::nlohmann::json(0.5)));
  EXPECT_EQ(1, JsonValueAs<uint64_t>(::nlohmann::json(1.0)));

  // Test smallest integer that can be exactly represented as a double but not
  // as a uint64.
  EXPECT_EQ(std::nullopt, JsonValueAs<uint64_t>(::nlohmann::json(
                              18446744073709551616.0 /*=2^64*/)));

  EXPECT_EQ(std::nullopt, JsonValueAs<uint64_t>(::nlohmann::json(-1.0)));

  // Test largest integer that can be exactly represented as a double and a
  // uint64.
  EXPECT_EQ(18446744073709549568u,
            JsonValueAs<uint64_t>(::nlohmann::json(18446744073709549568.0)));
}

TEST(JsonValueAsTest, Uint64FromString) {
  EXPECT_EQ(0xffffffffffffffffu,
            JsonValueAs<uint64_t>(::nlohmann::json("18446744073709551615")));

  EXPECT_FALSE(JsonValueAs<uint64_t>(::nlohmann::json("0.0")));
  EXPECT_FALSE(JsonValueAs<uint64_t>(::nlohmann::json("0a")));
  EXPECT_FALSE(JsonValueAs<uint64_t>(::nlohmann::json("0x0")));
  EXPECT_FALSE(JsonValueAs<uint64_t>(::nlohmann::json("0xf")));
  EXPECT_FALSE(JsonValueAs<uint64_t>(::nlohmann::json("-1")));

  const bool kStrict = true;
  EXPECT_EQ(std::nullopt,
            JsonValueAs<uint64_t>(::nlohmann::json("1"), kStrict));
}

TEST(JsonTest, JsonParseInt) {
  auto keys = GetKeys();
  auto JsonParseInt = [&keys](const ::nlohmann::json& json,
                              const char* member) {
    keys.erase(member);
    return JsonMemberT<int64_t, false>(json, member);
  };

  auto result = GetDefaultJSON();
  EXPECT_FALSE(result.is_discarded());

  // Some values can be parsed.
  ASSERT_TRUE(JsonParseInt(result, "signed"));
  EXPECT_EQ(456, *JsonParseInt(result, "signed"));

  ASSERT_TRUE(JsonParseInt(result, "neg_signed"));
  EXPECT_EQ(-567, *JsonParseInt(result, "neg_signed"));

  ASSERT_TRUE(JsonParseInt(result, "unsigned"));
  EXPECT_EQ(565, *JsonParseInt(result, "unsigned"));

  ASSERT_TRUE(JsonParseInt(result, "int_float"));
  EXPECT_EQ(122, *JsonParseInt(result, "int_float"));

  ASSERT_TRUE(JsonParseInt(result, "str_number"));
  EXPECT_EQ(789, *JsonParseInt(result, "str_number"));

  // Some values cannot be parsed.
  std::set<std::string> remaining = keys;
  for (const std::string& x : remaining) {
    EXPECT_FALSE(JsonParseInt(result, x.c_str())) << x;
  }
}

TEST(JsonTest, JsonParseUnsigned) {
  auto keys = GetKeys();
  auto JsonParseUnsigned = [&keys](const ::nlohmann::json& json,
                                   const char* member) {
    keys.erase(member);
    return JsonMemberT<uint64_t, false>(json, member);
  };

  auto result = GetDefaultJSON();
  EXPECT_FALSE(result.is_discarded());

  // Some values can be parsed.
  ASSERT_TRUE(JsonParseUnsigned(result, "signed"));
  EXPECT_EQ(456, *JsonParseUnsigned(result, "signed"));

  ASSERT_TRUE(JsonParseUnsigned(result, "unsigned"));
  EXPECT_EQ(565, *JsonParseUnsigned(result, "unsigned"));

  ASSERT_TRUE(JsonParseUnsigned(result, "int_float"));
  EXPECT_EQ(122, *JsonParseUnsigned(result, "int_float"));

  ASSERT_TRUE(JsonParseUnsigned(result, "str_number"));
  EXPECT_EQ(789, *JsonParseUnsigned(result, "str_number"));

  // Some values cannot be parsed.
  std::set<std::string> remaining = keys;
  for (const std::string& x : remaining) {
    EXPECT_FALSE(JsonParseUnsigned(result, x.c_str())) << x;
  }
}

TEST(JsonTest, JsonParseDouble) {
  auto keys = GetKeys();
  auto JsonParseDouble = [&keys](const ::nlohmann::json& json,
                                 const char* member) {
    keys.erase(member);
    return JsonMemberT<double, false>(json, member);
  };

  auto result = GetDefaultJSON();
  EXPECT_FALSE(result.is_discarded());

  // Some values can be parsed.
  ASSERT_TRUE(JsonParseDouble(result, "signed"));
  EXPECT_EQ(456, *JsonParseDouble(result, "signed"));

  ASSERT_TRUE(JsonParseDouble(result, "neg_signed"));
  EXPECT_EQ(-567, *JsonParseDouble(result, "neg_signed"));

  ASSERT_TRUE(JsonParseDouble(result, "unsigned"));
  EXPECT_EQ(565, *JsonParseDouble(result, "unsigned"));

  ASSERT_TRUE(JsonParseDouble(result, "float"));
  EXPECT_EQ(456.789, *JsonParseDouble(result, "float"));

  ASSERT_TRUE(JsonParseDouble(result, "neg_float"));
  EXPECT_EQ(-678.91, *JsonParseDouble(result, "neg_float"));

  ASSERT_TRUE(JsonParseDouble(result, "int_float"));
  EXPECT_EQ(122, *JsonParseDouble(result, "int_float"));

  ASSERT_TRUE(JsonParseDouble(result, "str_number"));
  EXPECT_EQ(789, *JsonParseDouble(result, "str_number"));

  ASSERT_TRUE(JsonParseDouble(result, "str_float"));
  EXPECT_EQ(123.4, *JsonParseDouble(result, "str_float"));

  // Some values cannot be parsed.
  std::set<std::string> remaining = keys;
  for (const std::string& x : remaining) {
    EXPECT_FALSE(JsonParseDouble(result, x.c_str())) << x;
  }
}

TEST(JsonTest, JsonParseString) {
  auto keys = GetKeys();
  auto JsonParseString = [&keys](const ::nlohmann::json& json,
                                 const char* member) {
    keys.erase(member);
    return JsonMemberT<std::string>(json, member);
  };

  auto result = GetDefaultJSON();
  EXPECT_FALSE(result.is_discarded());

  // Some values can be parsed
  ASSERT_TRUE(JsonParseString(result, "str_bool"));
  EXPECT_EQ("true", *JsonParseString(result, "str_bool"));

  ASSERT_TRUE(JsonParseString(result, "str"));
  EXPECT_EQ("abc", *JsonParseString(result, "str"));

  ASSERT_TRUE(JsonParseString(result, "str_number"));
  EXPECT_EQ("789", *JsonParseString(result, "str_number"));

  ASSERT_TRUE(JsonParseString(result, "str_float"));
  EXPECT_EQ("123.40", *JsonParseString(result, "str_float"));

  // Some values cannot be parsed.
  std::set<std::string> remaining = keys;
  for (const std::string& x : remaining) {
    EXPECT_FALSE(JsonParseString(result, x.c_str())) << x;
  }
}

TEST(JsonRequireValueAs, Success) {
  {
    bool v;
    EXPECT_TRUE(JsonRequireValueAs(::nlohmann::json(true), &v, true).ok());
    EXPECT_TRUE(v);

    EXPECT_TRUE(JsonRequireValueAs(::nlohmann::json("true"), &v, false).ok());
    EXPECT_TRUE(v);

    EXPECT_TRUE(JsonRequireValueAs(::nlohmann::json("true"), &v, [](bool) {
                  return true;
                }).ok());
    EXPECT_TRUE(v);

    EXPECT_TRUE(
        JsonRequireValueAs<bool>(::nlohmann::json(true), nullptr, true).ok());
  }
  {
    int64_t v;
    EXPECT_TRUE(JsonRequireValueAs(::nlohmann::json(-3), &v, true).ok());
    EXPECT_EQ(-3, v);

    EXPECT_TRUE(JsonRequireValueAs(::nlohmann::json(-4.0), &v, false).ok());
    EXPECT_EQ(-4, v);

    EXPECT_TRUE(JsonRequireValueAs(::nlohmann::json("-5"), &v, false).ok());
    EXPECT_EQ(-5, v);

    EXPECT_TRUE(JsonRequireValueAs(::nlohmann::json("-5"), &v, [](int64_t) {
                  return true;
                }).ok());
    EXPECT_EQ(-5, v);

    EXPECT_TRUE(
        JsonRequireValueAs<int64_t>(::nlohmann::json(-3), nullptr, true).ok());
  }
  {
    uint64_t v;
    EXPECT_TRUE(JsonRequireValueAs(::nlohmann::json(6), &v, true).ok());
    EXPECT_EQ(6, v);

    EXPECT_TRUE(JsonRequireValueAs(::nlohmann::json(7.0), &v, false).ok());
    EXPECT_EQ(7, v);

    EXPECT_TRUE(JsonRequireValueAs(::nlohmann::json("8"), &v, false).ok());
    EXPECT_EQ(8, v);

    EXPECT_TRUE(JsonRequireValueAs(::nlohmann::json("8"), &v, [](uint64_t) {
                  return true;
                }).ok());
    EXPECT_EQ(8, v);

    EXPECT_TRUE(
        JsonRequireValueAs<uint64_t>(::nlohmann::json(3), nullptr, true).ok());
  }
  {
    double v;
    EXPECT_TRUE(JsonRequireValueAs(::nlohmann::json(0.5), &v, true).ok());
    EXPECT_EQ(0.5, v);

    EXPECT_TRUE(JsonRequireValueAs(::nlohmann::json("2.0"), &v, false).ok());
    EXPECT_EQ(2.0, v);

    EXPECT_TRUE(JsonRequireValueAs(::nlohmann::json("2.0"), &v, [](double) {
                  return true;
                }).ok());
    EXPECT_EQ(2.0, v);

    EXPECT_TRUE(
        JsonRequireValueAs<double>(::nlohmann::json(3.0), nullptr, true).ok());
  }
  {
    std::string v;
    EXPECT_TRUE(JsonRequireValueAs(::nlohmann::json("x"), &v, false).ok());
    EXPECT_EQ("x", v);

    EXPECT_TRUE(JsonRequireValueAs(::nlohmann::json("y"), &v, [](std::string) {
                  return true;
                }).ok());
    EXPECT_EQ("y", v);

    EXPECT_TRUE(
        JsonRequireValueAs<std::string>(::nlohmann::json("z"), nullptr, true)
            .ok());
  }
}

TEST(JsonRequireValueAs, Failure) {
  {
    bool v;
    EXPECT_THAT(JsonRequireValueAs(::nlohmann::json("true"), &v, true),
                MatchesStatus(absl::StatusCode::kInvalidArgument,
                              "Expected boolean, but received: \"true\""));
  }

  EXPECT_THAT(JsonRequireValueAs<bool>(::nlohmann::json("true"), nullptr, true),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Expected boolean, but received: \"true\""));

  EXPECT_THAT(JsonRequireValueAs<bool>(::nlohmann::json(true), nullptr,
                                       [](bool) { return false; }),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Validation of boolean failed, received: true"));

  EXPECT_THAT(
      JsonRequireValueAs<int64_t>(::nlohmann::json("true"), nullptr, true),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Expected 64-bit signed integer, but received: \"true\""));

  EXPECT_THAT(
      JsonRequireValueAs<uint64_t>(::nlohmann::json(3.5), nullptr, true),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Expected 64-bit unsigned integer, but received: 3.5"));

  EXPECT_THAT(
      JsonRequireValueAs<std::string>(::nlohmann::json(true), nullptr, true),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Expected string, but received: true"));
}

TEST(JsonRequireIntegerTest, Success) {
  {
    std::int32_t result_int32 = 42;
    EXPECT_EQ(absl::OkStatus(), JsonRequireInteger<std::int32_t>(
                                    ::nlohmann::json(-5), &result_int32,
                                    /*strict=*/true, -7, -3));
    EXPECT_EQ(-5, result_int32);
  }
  {
    std::int32_t result_int32 = 42;
    EXPECT_EQ(absl::OkStatus(), JsonRequireInteger<std::int32_t>(
                                    ::nlohmann::json(-7), &result_int32,
                                    /*strict=*/true, -7, -3));
    EXPECT_EQ(-7, result_int32);
  }
  {
    std::int32_t result_int32 = 42;
    EXPECT_EQ(absl::OkStatus(), JsonRequireInteger<std::int32_t>(
                                    ::nlohmann::json("-7"), &result_int32,
                                    /*strict=*/false, -7, -3));
    EXPECT_EQ(-7, result_int32);
  }
  {
    std::int32_t result_int32 = 42;
    EXPECT_EQ(absl::OkStatus(), JsonRequireInteger<std::int32_t>(
                                    ::nlohmann::json(-3), &result_int32,
                                    /*strict=*/true, -7, -3));
    EXPECT_EQ(-3, result_int32);
  }
  {
    std::uint32_t result_uint32 = 42;
    EXPECT_EQ(absl::OkStatus(),
              JsonRequireInteger(::nlohmann::json(5), &result_uint32,
                                 /*strict=*/true, 2, 7));
    EXPECT_EQ(5u, result_uint32);
  }
  {
    std::int16_t result_int16 = 42;
    EXPECT_EQ(absl::OkStatus(),
              JsonRequireInteger(::nlohmann::json(5), &result_int16,
                                 /*strict=*/true, 2, 7));
    EXPECT_EQ(5, result_int16);
  }
}

TEST(JsonRequireIntegerTest, Failure) {
  {
    std::int32_t result_int32 = 42;
    EXPECT_THAT(
        JsonRequireInteger(::nlohmann::json(-2), &result_int32, /*strict=*/true,
                           -7, -3),
        MatchesStatus(
            absl::StatusCode::kInvalidArgument,
            "Expected integer in the range \\[-7, -3\\], but received: -2"));
    EXPECT_EQ(42, result_int32);
  }
  {
    std::int32_t result_int32 = 42;
    EXPECT_THAT(JsonRequireInteger(::nlohmann::json(true), &result_int32,
                                   /*strict=*/true, -7, -3),
                MatchesStatus(absl::StatusCode::kInvalidArgument,
                              "Expected integer in the range \\[-7, -3\\], but "
                              "received: true"));
    EXPECT_EQ(42, result_int32);
  }
  {
    std::uint32_t result_uint32 = 42;
    EXPECT_THAT(
        JsonRequireInteger(::nlohmann::json(11), &result_uint32,
                           /*strict=*/true, 5, 10),
        MatchesStatus(
            absl::StatusCode::kInvalidArgument,
            "Expected integer in the range \\[5, 10\\], but received: 11"));
    EXPECT_EQ(42u, result_uint32);
  }
}

}  // namespace
