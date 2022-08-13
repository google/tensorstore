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

#include "tensorstore/internal/json_binding/std_variant.h"

#include <string>
#include <type_traits>
#include <utility>
#include <variant>

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

namespace jb = ::tensorstore::internal_json_binding;

namespace {

using ::tensorstore::MatchesStatus;

TEST(JsonBindingTest, VariantDefaultBinder) {
  tensorstore::TestJsonBinderRoundTrip<std::variant<int, std::string>>({
      {3, ::nlohmann::json(3)},
      {"abc", ::nlohmann::json("abc")},
  });
}

TEST(JsonBindingTest, VariantDefaultBinderError) {
  EXPECT_THAT(
      (jb::FromJson<std::variant<int, std::string>>(::nlohmann::json(false))),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "No matching value binder: "
                    "Expected integer in the range .*, but received: false; "
                    "Expected string, but received: false"));
}

TEST(JsonBindingTest, VariantExplicitBinder) {
  auto binder = jb::Object(jb::Variant(jb::Member("a"), jb::Member("b")));
  tensorstore::TestJsonBinderRoundTrip<std::variant<int, std::string>>(
      {
          {3, {{"a", 3}}},
          {"abc", {{"b", "abc"}}},
      },
      binder);
}

}  // namespace
