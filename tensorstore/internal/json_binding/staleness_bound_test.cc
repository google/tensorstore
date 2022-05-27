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

#include "tensorstore/internal/json_binding/staleness_bound.h"

#include <memory>
#include <type_traits>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/time/time.h"
#include <nlohmann/json.hpp>
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/gtest.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/staleness_bound.h"

using ::tensorstore::MatchesJson;
using ::tensorstore::StalenessBound;
using ::testing::Optional;

namespace {

TEST(StalenessBoundJsonBinderTest, RoundTrip) {
  tensorstore::TestJsonBinderToJson<StalenessBound>({
      {StalenessBound{absl::InfinitePast()}, Optional(MatchesJson(false))},
      {StalenessBound{absl::InfiniteFuture()}, Optional(MatchesJson(true))},
      {StalenessBound::BoundedByOpen(), Optional(MatchesJson("open"))},
      {StalenessBound{absl::UnixEpoch()}, Optional(MatchesJson(0))},
      {StalenessBound{absl::UnixEpoch() + absl::Seconds(1)},
       Optional(MatchesJson(1))},
  });
}

TEST(StalenessBoundJsonBinderTest, FromJson) {
  tensorstore::TestJsonBinderFromJson<StalenessBound>({
      {false,
       ::testing::Optional(::testing::AllOf(
           ::testing::Field(&StalenessBound::time, absl::InfinitePast()),
           ::testing::Field(&StalenessBound::bounded_by_open_time, false)))},
      {true,
       ::testing::Optional(::testing::AllOf(
           ::testing::Field(&StalenessBound::time, absl::InfiniteFuture()),
           ::testing::Field(&StalenessBound::bounded_by_open_time, false)))},
      {"open", ::testing::Optional(::testing::Field(
                   &StalenessBound::bounded_by_open_time, true))},
      {0, ::testing::Optional(::testing::AllOf(
              ::testing::Field(&StalenessBound::time, absl::UnixEpoch()),
              ::testing::Field(&StalenessBound::bounded_by_open_time, false)))},
      {1, ::testing::Optional(::testing::AllOf(
              ::testing::Field(&StalenessBound::time,
                               absl::UnixEpoch() + absl::Seconds(1)),
              ::testing::Field(&StalenessBound::bounded_by_open_time, false)))},
      {1u,
       ::testing::Optional(::testing::AllOf(
           ::testing::Field(&StalenessBound::time,
                            absl::UnixEpoch() + absl::Seconds(1)),
           ::testing::Field(&StalenessBound::bounded_by_open_time, false)))},
      {1.5,
       ::testing::Optional(::testing::AllOf(
           ::testing::Field(&StalenessBound::time,
                            absl::UnixEpoch() + absl::Milliseconds(1500)),
           ::testing::Field(&StalenessBound::bounded_by_open_time, false)))},
  });
}

}  // namespace
