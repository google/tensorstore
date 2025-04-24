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

#include "tensorstore/serialization/json.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "tensorstore/internal/testing/json_gtest.h"
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/serialization/test_util.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::MatchesStatus;
using ::tensorstore::serialization::SerializationRoundTrip;
using ::tensorstore::serialization::TestSerializationRoundTrip;

TEST(SerializationTest, Valid) {
  TestSerializationRoundTrip(::nlohmann::json(5));
  TestSerializationRoundTrip(::nlohmann::json("abc"));
}

TEST(SerializationTest, Invalid) {
  EXPECT_THAT(SerializationRoundTrip(
                  ::nlohmann::json(::nlohmann::json::value_t::discarded)),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Cannot encode discarded json value.*"));
}

}  // namespace
