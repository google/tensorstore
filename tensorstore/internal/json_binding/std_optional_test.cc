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

#include "tensorstore/internal/json_binding/std_optional.h"

#include <optional>
#include <string>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/gtest.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_fwd.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

namespace jb = tensorstore::internal_json_binding;

namespace {

TEST(JsonBindingTest, Optional) {
  tensorstore::TestJsonBinderRoundTrip<std::optional<int>>({
      {3, ::nlohmann::json(3)},
      {std::nullopt, ::nlohmann::json(::nlohmann::json::value_t::discarded)},
  });
}

TEST(JsonBindingTest, OptionalExplicitNullopt) {
  const auto binder =
      jb::Optional(jb::DefaultBinder<>, [] { return "nullopt"; });
  tensorstore::TestJsonBinderRoundTrip<std::optional<int>>(
      {
          {3, 3},
          {std::nullopt, "nullopt"},
      },
      binder);
}

TEST(JsonBindingTest, OptionalResult) {
  // Result doesn't default-initialize, so verify manually.

  ::nlohmann::json j;
  tensorstore::Result<int> x(absl::UnknownError("x"));

  // saving error -> discarded
  j = 3;
  EXPECT_TRUE(jb::Optional()(std::false_type{}, jb::NoOptions{}, &x, &j).ok());
  EXPECT_TRUE(j.is_discarded());

  // loading value -> value
  j = 4;
  EXPECT_TRUE(jb::Optional()(std::true_type{}, jb::NoOptions{}, &x, &j).ok());
  EXPECT_TRUE(x.has_value());
  EXPECT_EQ(4, x.value());

  // saving value -> value
  j = ::nlohmann::json::value_t::discarded;
  EXPECT_TRUE(jb::Optional()(std::false_type{}, jb::NoOptions{}, &x, &j).ok());
  EXPECT_FALSE(j.is_discarded());
  EXPECT_EQ(4, j);

  // loading discarded -> no change.
  j = ::nlohmann::json::value_t::discarded;
  EXPECT_TRUE(jb::Optional()(std::true_type{}, jb::NoOptions{}, &x, &j).ok());
  EXPECT_TRUE(x.has_value());
  EXPECT_EQ(4, x.value());
}

}  // namespace
