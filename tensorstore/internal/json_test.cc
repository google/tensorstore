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

#include "tensorstore/internal/json.h"

#include <cstdint>
#include <iterator>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/node_hash_set.h"
#include <nlohmann/json.hpp>
#include "tensorstore/box.h"
#include "tensorstore/internal/json_bindable.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/json_serialization_options.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

namespace jb = tensorstore::internal::json_binding;
using ::nlohmann::json;
using tensorstore::MatchesStatus;
using tensorstore::Status;
using tensorstore::internal::JsonHandleObjectMember;
using tensorstore::internal::JsonParseArray;
using tensorstore::internal::JsonRequireInteger;
using tensorstore::internal::JsonRequireObjectMember;
using tensorstore::internal::JsonRequireValueAs;
using tensorstore::internal::JsonValidateArrayLength;
using tensorstore::internal::JsonValidateObjectMembers;
using tensorstore::internal::JsonValueAs;

TEST(JsonTest, SimpleParse) {
  using tensorstore::internal::ParseJson;
  const char kArray[] = R"({ "foo": "bar" })";

  auto x = ParseJson("");  // absl::string_view
  EXPECT_TRUE(x.is_discarded());

  // Test parsing objects.
  auto y = ParseJson(kArray);  // absl::string_view
  EXPECT_FALSE(y.is_discarded());

  auto z = ParseJson(std::begin(kArray), std::end(kArray));  // template
  EXPECT_FALSE(z.is_discarded());

  auto one = ParseJson("1");  // absl::string_view
  EXPECT_FALSE(one.is_discarded());
}

TEST(JsonTest, Meta) {
  auto JsonRequireString = [](const ::nlohmann::json& json,
                              const char* member) {
    return JsonRequireObjectMember(json, member,
                                   [](const ::nlohmann::json& j) {
                                     return JsonRequireValueAs<std::string>(
                                         j, nullptr, [](const std::string& x) {
                                           return !x.empty();
                                         });
                                   })
        .ok();
  };
  auto JsonRequireInt = [](const ::nlohmann::json& json, const char* member) {
    int64_t result;
    return JsonRequireObjectMember(json, member,
                                   [&result](const ::nlohmann::json& j) {
                                     return JsonRequireValueAs(j, &result);
                                   })
        .ok();
  };

  auto meta = ::nlohmann::json::meta();

  EXPECT_TRUE(JsonRequireString(meta, "copyright"));
  EXPECT_TRUE(JsonRequireString(meta, "name"));
  EXPECT_TRUE(JsonRequireString(meta, "url"));
  EXPECT_TRUE(JsonRequireString(meta, "platform"));
  EXPECT_TRUE(JsonRequireString(meta, "copyright"));

  EXPECT_TRUE(
      JsonRequireObjectMember(meta, "compiler", [&](const ::nlohmann::json& j) {
        EXPECT_TRUE(JsonRequireString(j, "c++"));
        return absl::OkStatus();
      }).ok());

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

absl::node_hash_set<std::string> GetKeys() {
  return absl::node_hash_set<std::string>{{
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
    absl::optional<bool> result;
    JsonHandleObjectMember(json, member, [&](const ::nlohmann::json& j) {
      result = JsonValueAs<bool>(j);
      return absl::OkStatus();
    }).IgnoreError();
    return result;
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
  absl::node_hash_set<std::string> remaining = keys;
  for (const std::string& x : remaining) {
    EXPECT_FALSE(JsonParseBool(result, x.c_str())) << x;
  }

  EXPECT_EQ(absl::nullopt, JsonValueAs<bool>(::nlohmann::json("a")));
  EXPECT_EQ(false, JsonValueAs<bool>(::nlohmann::json("false")));
  EXPECT_EQ(true, JsonValueAs<bool>(::nlohmann::json("true")));

  const bool kStrict = true;
  EXPECT_EQ(absl::nullopt,
            JsonValueAs<bool>(::nlohmann::json("true"), kStrict));
  EXPECT_EQ(true, JsonValueAs<bool>(::nlohmann::json(true), kStrict));
  EXPECT_EQ(false, JsonValueAs<bool>(::nlohmann::json(false), kStrict));
}

TEST(JsonValueAsTest, Int64FromUint64) {
  EXPECT_EQ(absl::nullopt,
            JsonValueAs<int64_t>(::nlohmann::json(0x8fffffffffffffffu)));
  EXPECT_EQ(absl::nullopt,
            JsonValueAs<int64_t>(::nlohmann::json(0xffffffffffffffffu)));
  EXPECT_EQ(0x7fffffffffffffff,
            JsonValueAs<int64_t>(::nlohmann::json(0x7fffffffffffffffu)));

  const bool kStrict = true;
  EXPECT_EQ(
      0x7fffffffffffffff,
      JsonValueAs<int64_t>(::nlohmann::json(0x7fffffffffffffffu), kStrict));
}

TEST(JsonValueAsTest, Int64FromDouble) {
  EXPECT_EQ(absl::nullopt, JsonValueAs<int64_t>(::nlohmann::json(0.5)));
  EXPECT_EQ(1, JsonValueAs<int64_t>(::nlohmann::json(1.0)));

  // Test smallest positive integer that can be exactly represented as a double
  // but not as an int64.
  EXPECT_EQ(
      absl::nullopt,
      JsonValueAs<int64_t>(::nlohmann::json(9223372036854775808.0 /*=2^63*/)));

  // Test largest negative integer that can be exactly represented as a double
  // but not as an int64.
  EXPECT_EQ(absl::nullopt,
            JsonValueAs<int64_t>(::nlohmann::json(-9223372036854777856.0)));

  // Test largest integer that can be exactly represented as both a double and
  // an int64.
  EXPECT_EQ(9223372036854774784,
            JsonValueAs<int64_t>(::nlohmann::json(9223372036854774784.0)));

  // Test -2^63, which can be exactly represented as a double.
  EXPECT_EQ(
      -0x8000000000000000,
      JsonValueAs<int64_t>(::nlohmann::json(-9223372036854775808.0 /*=2^63*/)));

  const bool kStrict = true;
  EXPECT_EQ(absl::nullopt,
            JsonValueAs<int64_t>(::nlohmann::json(1.0), kStrict));
}

TEST(JsonValueAsTest, Int64FromString) {
  EXPECT_EQ(-1, JsonValueAs<int64_t>(::nlohmann::json("-1")));
  EXPECT_EQ(-0x8000000000000000,
            JsonValueAs<int64_t>(::nlohmann::json("-9223372036854775808")));
  EXPECT_EQ(0x7fffffffffffffff,
            JsonValueAs<int64_t>(::nlohmann::json("9223372036854775807")));

  // TODO(jbms): Fix this inconsistency (conversion from double is allowed).
  EXPECT_FALSE(JsonValueAs<int64_t>(::nlohmann::json("0.0")));
  EXPECT_FALSE(JsonValueAs<int64_t>(::nlohmann::json("0a")));
  EXPECT_FALSE(JsonValueAs<int64_t>(::nlohmann::json("0x0")));
  EXPECT_FALSE(JsonValueAs<int64_t>(::nlohmann::json("0xf")));

  // Values out of bounds.
  EXPECT_FALSE(JsonValueAs<int64_t>(::nlohmann::json("9223372036854775808")));
  EXPECT_FALSE(JsonValueAs<int64_t>(::nlohmann::json("-9223372036854775809")));

  const bool kStrict = true;
  EXPECT_EQ(absl::nullopt,
            JsonValueAs<int64_t>(::nlohmann::json("-1"), kStrict));
}

TEST(JsonValueAsTest, Uint64FromDouble) {
  EXPECT_EQ(absl::nullopt, JsonValueAs<uint64_t>(::nlohmann::json(0.5)));
  EXPECT_EQ(1, JsonValueAs<uint64_t>(::nlohmann::json(1.0)));

  // Test smallest integer that can be exactly represented as a double but not
  // as a uint64.
  EXPECT_EQ(absl::nullopt, JsonValueAs<uint64_t>(::nlohmann::json(
                               18446744073709551616.0 /*=2^64*/)));

  EXPECT_EQ(absl::nullopt, JsonValueAs<uint64_t>(::nlohmann::json(-1.0)));

  // Test largest integer that can be exactly represented as a double and a
  // uint64.
  EXPECT_EQ(18446744073709549568u,
            JsonValueAs<uint64_t>(::nlohmann::json(18446744073709549568.0)));

  const bool kStrict = true;
  EXPECT_EQ(absl::nullopt,
            JsonValueAs<uint64_t>(::nlohmann::json(1.0), kStrict));
}

TEST(JsonValueAsTest, Uint64FromString) {
  EXPECT_EQ(0xffffffffffffffffu,
            JsonValueAs<uint64_t>(::nlohmann::json("18446744073709551615")));

  // TODO(jbms): Fix this inconsistency (conversion from double is allowed).
  EXPECT_FALSE(JsonValueAs<uint64_t>(::nlohmann::json("0.0")));
  EXPECT_FALSE(JsonValueAs<uint64_t>(::nlohmann::json("0a")));
  EXPECT_FALSE(JsonValueAs<uint64_t>(::nlohmann::json("0x0")));
  EXPECT_FALSE(JsonValueAs<uint64_t>(::nlohmann::json("0xf")));
  EXPECT_FALSE(JsonValueAs<uint64_t>(::nlohmann::json("-1")));

  const bool kStrict = true;
  EXPECT_EQ(absl::nullopt,
            JsonValueAs<uint64_t>(::nlohmann::json("1"), kStrict));
}

TEST(JsonTest, JsonParseInt) {
  auto keys = GetKeys();
  auto JsonParseInt = [&keys](const ::nlohmann::json& json,
                              const char* member) {
    keys.erase(member);
    absl::optional<int64_t> result;
    JsonHandleObjectMember(json, member, [&](const ::nlohmann::json& j) {
      result = JsonValueAs<int64_t>(j);
      return absl::OkStatus();
    }).IgnoreError();
    return result;
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
  absl::node_hash_set<std::string> remaining = keys;
  for (const std::string& x : remaining) {
    EXPECT_FALSE(JsonParseInt(result, x.c_str())) << x;
  }
}

TEST(JsonTest, JsonParseUnsigned) {
  auto keys = GetKeys();
  auto JsonParseUnsigned = [&keys](const ::nlohmann::json& json,
                                   const char* member) {
    keys.erase(member);
    absl::optional<uint64_t> result;
    JsonHandleObjectMember(json, member, [&](const ::nlohmann::json& j) {
      result = JsonValueAs<uint64_t>(j);
      return absl::OkStatus();
    }).IgnoreError();
    return result;
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
  absl::node_hash_set<std::string> remaining = keys;
  for (const std::string& x : remaining) {
    EXPECT_FALSE(JsonParseUnsigned(result, x.c_str())) << x;
  }
}

TEST(JsonTest, JsonParseDouble) {
  auto keys = GetKeys();
  auto JsonParseDouble = [&keys](const ::nlohmann::json& json,
                                 const char* member) {
    keys.erase(member);
    absl::optional<double> result;
    JsonHandleObjectMember(json, member, [&](const ::nlohmann::json& j) {
      result = JsonValueAs<double>(j);
      return absl::OkStatus();
    }).IgnoreError();
    return result;
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
  absl::node_hash_set<std::string> remaining = keys;
  for (const std::string& x : remaining) {
    EXPECT_FALSE(JsonParseDouble(result, x.c_str())) << x;
  }
}

TEST(JsonTest, JsonParseString) {
  auto keys = GetKeys();
  auto JsonParseString = [&keys](const ::nlohmann::json& json,
                                 const char* member) {
    keys.erase(member);
    absl::optional<std::string> result;
    JsonHandleObjectMember(json, member, [&](const ::nlohmann::json& j) {
      result = JsonValueAs<std::string>(j);
      return absl::OkStatus();
    }).IgnoreError();
    return result;
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
  absl::node_hash_set<std::string> remaining = keys;
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
    EXPECT_EQ(Status(), JsonRequireInteger<std::int32_t>(
                            ::nlohmann::json(-5), &result_int32,
                            /*strict=*/true, -7, -3));
    EXPECT_EQ(-5, result_int32);
  }
  {
    std::int32_t result_int32 = 42;
    EXPECT_EQ(Status(), JsonRequireInteger<std::int32_t>(
                            ::nlohmann::json(-7), &result_int32,
                            /*strict=*/true, -7, -3));
    EXPECT_EQ(-7, result_int32);
  }
  {
    std::int32_t result_int32 = 42;
    EXPECT_EQ(Status(), JsonRequireInteger<std::int32_t>(
                            ::nlohmann::json("-7"), &result_int32,
                            /*strict=*/false, -7, -3));
    EXPECT_EQ(-7, result_int32);
  }
  {
    std::int32_t result_int32 = 42;
    EXPECT_EQ(Status(), JsonRequireInteger<std::int32_t>(
                            ::nlohmann::json(-3), &result_int32,
                            /*strict=*/true, -7, -3));
    EXPECT_EQ(-3, result_int32);
  }
  {
    std::uint32_t result_uint32 = 42;
    EXPECT_EQ(Status(), JsonRequireInteger(::nlohmann::json(5), &result_uint32,
                                           /*strict=*/true, 2, 7));
    EXPECT_EQ(5u, result_uint32);
  }
  {
    std::int16_t result_int16 = 42;
    EXPECT_EQ(Status(), JsonRequireInteger(::nlohmann::json(5), &result_int16,
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

TEST(JsonParseArrayTest, Basic) {
  bool size_received = false;
  std::vector<std::pair<::nlohmann::json, std::ptrdiff_t>> elements;
  EXPECT_EQ(Status(), JsonParseArray(
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
  EXPECT_EQ(Status(), JsonValidateArrayLength(3, 3));
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

TEST(JsonValidateObjectMembers, Basic) {
  // Same set
  EXPECT_TRUE(
      JsonValidateObjectMembers(::nlohmann::json{{"name", 3}}, {"name"}).ok());

  // Missing members => ok
  EXPECT_TRUE(JsonValidateObjectMembers(::nlohmann::json{{"name", 3}},
                                        {"name", "birthday"})
                  .ok());
}

TEST(JsonValidateObjectMembers, Failure) {
  // Not an object
  EXPECT_THAT(JsonValidateObjectMembers(::nlohmann::json("true"), {"bar"}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Expected object, but received: \"true\""));

  EXPECT_THAT(JsonValidateObjectMembers(::nlohmann::json{3, 4, 5}, {"name"}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Expected object, but received: \\[3,4,5\\]"));

  // Extra members
  EXPECT_THAT(JsonValidateObjectMembers(
                  ::nlohmann::json{{"name", 3}, {"birthday", 4}}, {"name"}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Object includes extra members: .*"));
}

TEST(JsonRequireObjectMember, Success) {
  auto ok = [](const ::nlohmann::json& value) { return absl::OkStatus(); };

  EXPECT_TRUE(
      JsonRequireObjectMember(::nlohmann::json{{"name", 3}}, "name", ok).ok());
}

TEST(JsonRequireObjectMember, Failure) {
  auto ok = [](const ::nlohmann::json& value) { return absl::OkStatus(); };
  auto fail = [](const ::nlohmann::json& value) {
    return absl::InvalidArgumentError("failure");
  };

  EXPECT_THAT(JsonRequireObjectMember(::nlohmann::json("true"), "bar", ok),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Expected object, but received: \"true\""));

  EXPECT_THAT(JsonRequireObjectMember(::nlohmann::json{{"name", 3}}, "bar", ok),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Missing object member \"bar\""));

  EXPECT_THAT(
      JsonRequireObjectMember(::nlohmann::json{{"name", 3}}, "name", fail),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Error parsing object member \"name\": failure"));
}

TEST(JsonHandleObjectMember, Success) {
  auto ok = [](const ::nlohmann::json& value) { return absl::OkStatus(); };
  auto fail = [](const ::nlohmann::json& value) {
    return absl::InvalidArgumentError("failure");
  };

  EXPECT_TRUE(
      JsonHandleObjectMember(::nlohmann::json{{"name", 3}}, "name", ok).ok());

  // Field not found => success
  EXPECT_TRUE(
      JsonHandleObjectMember(::nlohmann::json{{"name", 3}}, "bar", fail).ok());
}

TEST(JsonHandleObjectMember, Failure) {
  auto ok = [](const ::nlohmann::json& value) { return absl::OkStatus(); };
  auto fail = [](const ::nlohmann::json& value) {
    return absl::InvalidArgumentError("failure");
  };

  EXPECT_THAT(JsonHandleObjectMember(::nlohmann::json("true"), "bar", ok),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Expected object, but received: \"true\""));

  EXPECT_THAT(
      JsonHandleObjectMember(::nlohmann::json{{"name", 3}}, "name", fail),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Error parsing object member \"name\": failure"));
}

TEST(JsonBindingTest, Example) {
  struct Foo {
    int x;
    std::string y;
    std::optional<int> z;
  };

  constexpr auto FooBinder = [] {
    return jb::Object(
        jb::Member("x", jb::Projection(&Foo::x)),
        jb::Member("y", jb::Projection(&Foo::y, jb::DefaultValue([](auto* y) {
                     *y = "default";
                   }))),
        jb::Member("z", jb::Projection(&Foo::z)));
  };

  EXPECT_EQ(::nlohmann::json({{"x", 3}}),
            jb::ToJson(Foo{3, "default", std::nullopt}, FooBinder(),
                       tensorstore::IncludeDefaults{false}));

  auto value =
      jb::FromJson<Foo>({{"x", 3}, {"y", "value"}, {"z", 10}}, FooBinder())
          .value();
  EXPECT_EQ(3, value.x);
  EXPECT_EQ("value", value.y);
  EXPECT_EQ(10, value.z);
}

TEST(JsonBindingTest, GetterSetter) {
  struct Foo {
    int x;
    int get_x() const { return x; }
    void set_x(int value) { this->x = value; }
  };

  const auto FooBinder =
      jb::Object(jb::Member("x", jb::GetterSetter(&Foo::get_x, &Foo::set_x)));

  EXPECT_EQ(::nlohmann::json({{"x", 3}}), jb::ToJson(Foo{3}, FooBinder));
  auto value = jb::FromJson<Foo>({{"x", 3}}, FooBinder).value();
  EXPECT_EQ(3, value.x);
}

TEST(JsonBindingTest, Constant) {
  const auto binder = jb::Constant([] { return 3; });
  EXPECT_THAT(jb::ToJson("ignored", binder),
              ::testing::Optional(::nlohmann::json(3)));
  EXPECT_THAT(jb::FromJson<std::string>(::nlohmann::json(3), binder),
              ::testing::Optional(std::string{}));
  EXPECT_THAT(jb::FromJson<std::string>(::nlohmann::json(4), binder),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Expected 3, but received: 4"));
}

TEST(JsonBindingTest, Optional) {
  EXPECT_THAT(jb::ToJson(std::optional<int>(3)),
              ::testing::Optional(::nlohmann::json(3)));
  EXPECT_THAT(jb::FromJson<std::optional<int>>(::nlohmann::json(3)),
              ::testing::Optional(::testing::Optional(3)));
  EXPECT_THAT(jb::FromJson<std::optional<int>>(
                  ::nlohmann::json(::nlohmann::json::value_t::discarded)),
              ::testing::Optional(std::nullopt));
  EXPECT_THAT(jb::ToJson(std::optional<int>{}),
              ::testing::Optional(tensorstore::MatchesJson(
                  ::nlohmann::json(::nlohmann::json::value_t::discarded))));
}

TEST(JsonBindingTest, OptionalExplicitNullopt) {
  const auto binder =
      jb::Optional(jb::DefaultBinder<>, [] { return "nullopt"; });
  EXPECT_THAT(jb::ToJson(std::optional<int>(3), binder),
              ::testing::Optional(::nlohmann::json(3)));
  EXPECT_THAT(jb::FromJson<std::optional<int>>(::nlohmann::json(3), binder),
              ::testing::Optional(::testing::Optional(3)));
  EXPECT_THAT(
      jb::FromJson<std::optional<int>>(::nlohmann::json("nullopt"), binder),
      ::testing::Optional(std::nullopt));
  EXPECT_THAT(jb::ToJson(std::optional<int>{}, binder),
              ::testing::Optional(::nlohmann::json("nullopt")));
}

TEST(JsonBindingTest, DefaultValueDiscarded) {
  const auto binder =
      jb::DefaultValue([](auto* obj) { *obj = 3; },
                       jb::DefaultValue([](auto* obj) { *obj = 3; }));
  EXPECT_THAT(jb::ToJson(3, binder, tensorstore::IncludeDefaults{false}),
              ::testing::Optional(tensorstore::MatchesJson(
                  ::nlohmann::json(::nlohmann::json::value_t::discarded))));
  EXPECT_THAT(jb::ToJson(3, binder, tensorstore::IncludeDefaults{true}),
              ::testing::Optional(::nlohmann::json(3)));
  EXPECT_THAT(jb::ToJson(4, binder, tensorstore::IncludeDefaults{true}),
              ::testing::Optional(::nlohmann::json(4)));
  EXPECT_THAT(jb::ToJson(4, binder, tensorstore::IncludeDefaults{false}),
              ::testing::Optional(::nlohmann::json(4)));
  EXPECT_THAT(jb::FromJson<int>(::nlohmann::json(4), binder),
              ::testing::Optional(4));
  EXPECT_THAT(jb::FromJson<int>(::nlohmann::json(3), binder),
              ::testing::Optional(3));
  EXPECT_THAT(
      jb::FromJson<int>(::nlohmann::json(::nlohmann::json::value_t::discarded),
                        binder),
      ::testing::Optional(3));
}

TEST(JsonBindingTest, Array) {
  const auto binder = jb::Array();
  EXPECT_THAT(jb::ToJson(std::vector<int>{1, 2, 3}, binder),
              ::testing::Optional(::nlohmann::json({1, 2, 3})));
  EXPECT_THAT(jb::FromJson<std::vector<int>>(::nlohmann::json{1, 2, 3}, binder),
              ::testing::Optional(std::vector<int>{1, 2, 3}));
  EXPECT_THAT(
      jb::FromJson<std::vector<int>>(::nlohmann::json{1, 2, "a"}, binder),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Error parsing value at position 2: Expected integer .*"));
}

TEST(JsonBindingTest, FixedSizeArray) {
  const auto binder = jb::FixedSizeArray();
  EXPECT_THAT(jb::ToJson(std::array<int, 3>{{1, 2, 3}}, binder),
              ::testing::Optional(::nlohmann::json({1, 2, 3})));
  EXPECT_THAT(
      (jb::FromJson<std::array<int, 3>>(::nlohmann::json{1, 2, 3}, binder)),
      ::testing::Optional(std::array<int, 3>{{1, 2, 3}}));
  EXPECT_THAT(
      (jb::FromJson<std::array<int, 3>>(::nlohmann::json{1, 2, 3, 4}, binder)),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Array has length 4 but should have length 3"));
}

// Tests `FixedSizeArray` applied to `tensorstore::span<tensorstore::Index, 3>`.
TEST(JsonBindingTest, StaticRankBox) {
  using Value = tensorstore::Box<3>;
  const auto binder = jb::Object(
      jb::Member("origin", jb::Projection([](auto& x) { return x.origin(); })),
      jb::Member("shape", jb::Projection([](auto& x) { return x.shape(); })));
  const auto value = Value({1, 2, 3}, {4, 5, 6});
  const ::nlohmann::json json{{"origin", {1, 2, 3}}, {"shape", {4, 5, 6}}};
  EXPECT_THAT(jb::ToJson(value, binder), ::testing::Optional(json));
  EXPECT_THAT(jb::FromJson<Value>(json, binder), ::testing::Optional(value));
}

// Tests `FixedSizeArray` applied to `tensorstore::span<tensorstore::Index>`.
TEST(JsonBindingTest, DynamicRankBox) {
  using Value = tensorstore::Box<>;
  const auto binder = jb::Object(
      jb::Member("rank", jb::GetterSetter(
                             [](auto& x) { return x.rank(); },
                             [](auto& x, tensorstore::DimensionIndex rank) {
                               x.set_rank(rank);
                             },
                             jb::Integer(0))),
      jb::Member("origin", jb::Projection([](auto& x) { return x.origin(); })),
      jb::Member("shape", jb::Projection([](auto& x) { return x.shape(); })));
  const auto value = Value({1, 2, 3}, {4, 5, 6});
  const ::nlohmann::json json{
      {"rank", 3}, {"origin", {1, 2, 3}}, {"shape", {4, 5, 6}}};
  EXPECT_THAT(jb::ToJson(value, binder), ::testing::Optional(json));
  EXPECT_THAT(jb::FromJson<Value>(json, binder), ::testing::Optional(value));
}

}  // namespace
