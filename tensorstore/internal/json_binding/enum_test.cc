// Copyright 2022 The TensorStore Authors
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

#include "tensorstore/internal/json_binding/enum.h"

#include <memory>
#include <string_view>
#include <utility>
#include <variant>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorstore/internal/json/json.h"
#include "tensorstore/internal/json_binding/gtest.h"
#include "tensorstore/internal/json_fwd.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/util/status_testutil.h"

using ::tensorstore::MatchesStatus;

namespace jb = tensorstore::internal_json_binding;

namespace {

TEST(JsonBindingTest, Enum) {
  enum class TestEnum { a, b };
  const auto binder = jb::Enum<TestEnum, std::string_view>({
      {TestEnum::a, "a"},
      {TestEnum::b, "b"},
  });
  tensorstore::TestJsonBinderRoundTrip<TestEnum>(
      {
          {TestEnum::a, "a"},
          {TestEnum::b, "b"},
      },
      binder);
  tensorstore::TestJsonBinderFromJson<TestEnum>(
      {
          {"c",
           MatchesStatus(absl::StatusCode::kInvalidArgument,
                         "Expected one of \"a\", \"b\", but received: \"c\"")},
      },
      binder);
}

TEST(JsonBindingTest, MapValue) {
  enum class TestMap { a, b };

  const auto binder = jb::MapValue(
      [](auto...) { return absl::InvalidArgumentError("missing"); },
      std::make_pair(TestMap::a, "a"),  //
      std::make_pair(TestMap::b, "b"),  //
      std::make_pair(TestMap::a, 1),    //
      std::make_pair(TestMap::b, 2));

  tensorstore::TestJsonBinderRoundTrip<TestMap>(
      {
          {TestMap::a, "a"},
          {TestMap::b, "b"},
      },
      binder);

  tensorstore::TestJsonBinderFromJson<TestMap>(
      {
          {"a", ::testing::Eq(TestMap::a)},
          {"b", ::testing::Eq(TestMap::b)},
          {"c",
           MatchesStatus(absl::StatusCode::kInvalidArgument, ".*missing.*")},
          {1, ::testing::Eq(TestMap::a)},
          {2, ::testing::Eq(TestMap::b)},
          {3, MatchesStatus(absl::StatusCode::kInvalidArgument, ".*missing.*")},
      },
      binder);
}

namespace map_variant_test {
struct A {
  [[maybe_unused]] friend bool operator==(const A&, const A&) { return true; }
};
struct B {
  [[maybe_unused]] friend bool operator==(const B&, const B&) { return true; }
};
struct C {
  [[maybe_unused]] friend bool operator==(const C&, const C&) { return true; }
};
}  // namespace map_variant_test

TEST(JsonBindingTest, MapValueVariant) {
  // MapValue can be used to map simple variant types, but more complex types
  // still require custom binding.
  using map_variant_test::A;
  using map_variant_test::B;
  using map_variant_test::C;

  using T = std::variant<A, B, C>;
  const auto binder = jb::MapValue(
      [](auto...) { return absl::InvalidArgumentError("missing"); },
      std::make_pair(T{A{}}, "a"),  //
      std::make_pair(T{B{}}, "b"),  //
      std::make_pair(T{C{}}, 3));

  tensorstore::TestJsonBinderRoundTrip<T>(
      {
          {A{}, "a"},
          {B{}, "b"},
          {C{}, 3},
      },
      binder);
}

}  // namespace
