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

#include "tensorstore/internal/staleness_bound_json_binder.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/time/time.h"
#include <nlohmann/json.hpp>
#include "tensorstore/internal/json_bindable.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/staleness_bound.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using tensorstore::MatchesJson;
using tensorstore::StalenessBound;
using tensorstore::internal::json_binding::FromJson;
using tensorstore::internal::json_binding::ToJson;

TEST(StalenessBoundJsonBinderTest, ToJson) {
  EXPECT_THAT(ToJson(StalenessBound{absl::InfinitePast()}),
              ::testing::Optional(MatchesJson(false)));
  EXPECT_THAT(ToJson(StalenessBound{absl::InfiniteFuture()}),
              ::testing::Optional(MatchesJson(true)));
  EXPECT_THAT(ToJson(StalenessBound::BoundedByOpen()),
              ::testing::Optional(MatchesJson("open")));
  EXPECT_THAT(ToJson(StalenessBound{absl::UnixEpoch()}),
              ::testing::Optional(MatchesJson(0)));
  EXPECT_THAT(ToJson(StalenessBound{absl::UnixEpoch() + absl::Seconds(1)}),
              ::testing::Optional(MatchesJson(1)));
}

TEST(StalenessBoundJsonBinderTest, FromJson) {
  EXPECT_THAT(
      FromJson<StalenessBound>(false),
      ::testing::Optional(::testing::AllOf(
          StalenessBound{absl::InfinitePast()},
          ::testing::Field(&StalenessBound::bounded_by_open_time, false))));
  EXPECT_THAT(
      FromJson<StalenessBound>(true),
      ::testing::Optional(::testing::AllOf(
          StalenessBound{absl::InfiniteFuture()},
          ::testing::Field(&StalenessBound::bounded_by_open_time, false))));
  EXPECT_THAT(FromJson<StalenessBound>("open"),
              ::testing::Optional(::testing::Field(
                  &StalenessBound::bounded_by_open_time, true)));
  EXPECT_THAT(
      FromJson<StalenessBound>(0),
      ::testing::Optional(::testing::AllOf(
          StalenessBound{absl::UnixEpoch()},
          ::testing::Field(&StalenessBound::bounded_by_open_time, false))));
  EXPECT_THAT(
      FromJson<StalenessBound>(1),
      ::testing::Optional(::testing::AllOf(
          StalenessBound{absl::UnixEpoch() + absl::Seconds(1)},
          ::testing::Field(&StalenessBound::bounded_by_open_time, false))));
  EXPECT_THAT(
      FromJson<StalenessBound>(1u),
      ::testing::Optional(::testing::AllOf(
          StalenessBound{absl::UnixEpoch() + absl::Seconds(1)},
          ::testing::Field(&StalenessBound::bounded_by_open_time, false))));
  EXPECT_THAT(
      FromJson<StalenessBound>(1.5),
      ::testing::Optional(::testing::AllOf(
          StalenessBound{absl::UnixEpoch() + absl::Milliseconds(1500)},
          ::testing::Field(&StalenessBound::bounded_by_open_time, false))));
}

}  // namespace
