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

#include "tensorstore/internal/json_binding/json_binding.h"

#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/box.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/json/json.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/gtest.h"
#include "tensorstore/internal/json_binding/std_array.h"
#include "tensorstore/internal/json_binding/std_optional.h"
#include "tensorstore/internal/json_fwd.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

namespace {

namespace jb = tensorstore::internal_json_binding;
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

TEST(JsonBindingTest, SequenceOrder) {
  auto binder = jb::Sequence(
      [](auto is_loading, const auto& options, int* obj, auto* j) {
        *obj = 1;
        return absl::OkStatus();
      },
      [](auto is_loading, const auto& options, int* obj, auto* j) {
        *obj = 3;
        return absl::OkStatus();
      });

  int x = 0;
  ::nlohmann::json j({{"x", 3}});

  // Loading, forward order.
  EXPECT_TRUE(binder(std::true_type{}, jb::NoOptions{}, &x, &j).ok());
  EXPECT_EQ(3, x);

  // Saving, reverse order.
  EXPECT_TRUE(binder(std::false_type{}, jb::NoOptions{}, &x, &j).ok());
  EXPECT_EQ(1, x);
}

TEST(JsonBindingTest, ValueAsBinder) {
  tensorstore::TestJsonBinderRoundTrip<bool>(
      {
          {true, ::nlohmann::json(true)},
      },
      jb::ValueAsBinder);
  tensorstore::TestJsonBinderRoundTrip<std::int64_t>(
      {
          {3, ::nlohmann::json(3)},
      },
      jb::ValueAsBinder);
  tensorstore::TestJsonBinderRoundTrip<std::uint64_t>(
      {
          {4, ::nlohmann::json(4)},
      },
      jb::ValueAsBinder);
  tensorstore::TestJsonBinderRoundTrip<double>(
      {
          {5, ::nlohmann::json(5)},
          {5.0, ::nlohmann::json(5.0)},
      },
      jb::ValueAsBinder);
  tensorstore::TestJsonBinderRoundTrip<std::string>(
      {
          {"a", ::nlohmann::json("a")},
          {"", ::nlohmann::json("")},
      },
      jb::ValueAsBinder);
}

TEST(JsonBindingTest, LooseValueAsBinder) {
  using testing::Eq;

  tensorstore::TestJsonBinderFromJson<bool>(
      {
          {::nlohmann::json(true), Eq(true)},
          {::nlohmann::json("true"), Eq(true)},
      },
      jb::LooseValueAsBinder);
  tensorstore::TestJsonBinderFromJson<std::int64_t>(
      {
          {::nlohmann::json(3), Eq(3)},
          {::nlohmann::json(3.0), Eq(3)},
          {::nlohmann::json("3"), Eq(3)},
      },
      jb::LooseValueAsBinder);
  tensorstore::TestJsonBinderFromJson<std::uint64_t>(
      {
          {::nlohmann::json(4), Eq(4)},
          {::nlohmann::json(4.0), Eq(4)},
          {::nlohmann::json("4"), Eq(4)},
      },
      jb::LooseValueAsBinder);
  tensorstore::TestJsonBinderFromJson<double>(
      {
          {::nlohmann::json(5.0), Eq(5.0)},
          {::nlohmann::json(5), Eq(5.0)},
          {::nlohmann::json("5"), Eq(5.0)},
      },
      jb::LooseValueAsBinder);

  // LooseValueAsBinder<string> is the same as ValueAsBinder<string>
  tensorstore::TestJsonBinderRoundTrip<std::string>(
      {
          {"a", ::nlohmann::json("a")},
          {"", ::nlohmann::json("")},
      },
      jb::LooseValueAsBinder);
}

TEST(JsonBindingTest, NonEmptyStringBinder) {
  using testing::Eq;

  tensorstore::TestJsonBinderRoundTrip<std::string>(
      {
          {"a", ::nlohmann::json("a")},
      },
      jb::NonEmptyStringBinder);

  tensorstore::TestJsonBinderFromJson<std::string>(
      {
          {"", MatchesStatus(absl::StatusCode::kInvalidArgument,
                             "Validation of string failed, received: \"\"")},
      },
      jb::NonEmptyStringBinder);
}

TEST(JsonBindingTest, FloatBinders) {
  using testing::Eq;

  tensorstore::TestJsonBinderFromJson<float>(
      {
          {::nlohmann::json(5.0), Eq(5.0f)},
          {::nlohmann::json(5), Eq(5.0f)},
      },
      jb::FloatBinder);
  tensorstore::TestJsonBinderFromJson<double>(
      {
          {::nlohmann::json(5.0), Eq(5.0)},
          {::nlohmann::json(5), Eq(5.0)},
      },
      jb::FloatBinder);

  tensorstore::TestJsonBinderFromJson<float>(
      {
          {::nlohmann::json(5.0), Eq(5.0f)},
          {::nlohmann::json(5), Eq(5.0f)},
          {::nlohmann::json("5"), Eq(5.0f)},
      },
      jb::LooseFloatBinder);
  tensorstore::TestJsonBinderFromJson<double>(
      {
          {::nlohmann::json(5.0), Eq(5.0)},
          {::nlohmann::json(5), Eq(5.0)},
          {::nlohmann::json("5"), Eq(5.0)},
      },
      jb::LooseFloatBinder);
}

TEST(JsonBindingTest, DefaultValueDiscarded) {
  const auto binder =
      jb::DefaultValue([](auto* obj) { *obj = 3; },
                       jb::DefaultValue([](auto* obj) { *obj = 3; }));
  tensorstore::TestJsonBinderRoundTrip<int>(
      {
          {3, ::nlohmann::json(::nlohmann::json::value_t::discarded)},
          {4, 4},
      },
      binder, tensorstore::IncludeDefaults{false});
  tensorstore::TestJsonBinderRoundTrip<int>(
      {
          {3, 3},
          {4, 4},
      },
      binder, tensorstore::IncludeDefaults{true});
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

TEST(JsonBindingTest, ObjectMember) {
  tensorstore::TestJsonBinderRoundTrip<int>(
      {
          {3, ::nlohmann::json({{"x", 3}})},
      },
      jb::Object(jb::Member("x")));
}

TEST(JsonBindingTest, ObjectOptionalMember) {
  struct Foo {
    int x = 1;
  };

  const auto FooBinder =
      jb::Object(jb::OptionalMember("x", jb::Projection(&Foo::x)),
                 jb::DiscardExtraMembers);

  EXPECT_EQ(::nlohmann::json({{"x", 3}}), jb::ToJson(Foo{3}, FooBinder));

  {
    auto value = jb::FromJson<Foo>({{"x", 3}}, FooBinder).value();
    EXPECT_EQ(3, value.x);
  }

  {
    auto value = jb::FromJson<Foo>({{"y", 3}}, FooBinder).value();
    EXPECT_EQ(1, value.x);
  }
}

// Tests `FixedSizeArray` applied to `tensorstore::span<tensorstore::Index, 3>`.
TEST(JsonBindingTest, StaticRankBox) {
  using Value = tensorstore::Box<3>;
  const auto binder = jb::Object(
      jb::Member("origin", jb::Projection([](auto& x) { return x.origin(); })),
      jb::Member("shape", jb::Projection([](auto& x) { return x.shape(); })));
  tensorstore::TestJsonBinderRoundTrip<Value>(
      {
          {Value({1, 2, 3}, {4, 5, 6}),
           {{"origin", {1, 2, 3}}, {"shape", {4, 5, 6}}}},
      },
      binder);
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

  tensorstore::TestJsonBinderRoundTrip<Value>(
      {
          {Value({1, 2, 3}, {4, 5, 6}),
           {{"rank", 3}, {"origin", {1, 2, 3}}, {"shape", {4, 5, 6}}}},
      },
      binder);
}

TEST(JsonBindingTest, Null) {
  tensorstore::TestJsonBinderRoundTrip<std::nullptr_t>({
      {nullptr, nullptr},
  });
  tensorstore::TestJsonBinderFromJson<std::nullptr_t>({
      {42, MatchesStatus(absl::StatusCode::kInvalidArgument,
                         "Expected null, but received: 42")},
  });
}

}  // namespace
