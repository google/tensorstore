// Copyright 2023 The TensorStore Authors
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

#include "tensorstore/internal/json_binding/optional_object.h"

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/internal/json_binding/gtest.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/util/status_testutil.h"

namespace jb = tensorstore::internal_json_binding;

namespace {

using ::tensorstore::MatchesStatus;

TEST(JsonBindingTest, RoundTrip) {
  tensorstore::TestJsonBinderRoundTrip<::nlohmann::json::object_t>(
      {
          {{}, ::nlohmann::json(::nlohmann::json::value_t::discarded)},
          {{{"a", 1}, {"b", 2}}, {{"a", 1}, {"b", 2}}},
      },
      jb::OptionalObject(jb::DefaultBinder<>));
}

TEST(JsonBindingTest, Invalid) {
  tensorstore::TestJsonBinderFromJson<::nlohmann::json::object_t>(
      {
          {"abc", MatchesStatus(absl::StatusCode::kInvalidArgument,
                                "Expected object, but received: \"abc\"")},
      },
      jb::OptionalObject(jb::DefaultBinder<>));
}

}  // namespace
