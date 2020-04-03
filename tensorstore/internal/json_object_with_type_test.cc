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

#include "tensorstore/internal/json_object_with_type.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/to_string.h"

namespace {

using tensorstore::MatchesStatus;
using tensorstore::StrCat;
using tensorstore::internal::JsonObjectWithType;

TEST(JsonObjectWithTypeTest, Comparison) {
  JsonObjectWithType a{"x", {{"a", "b"}}};
  JsonObjectWithType b{"x", {{"a", "c"}}};
  JsonObjectWithType c{"y", {{"a", "b"}}};

  EXPECT_EQ(a, a);
  EXPECT_NE(a, b);
  EXPECT_NE(a, c);
}

TEST(JsonObjectWithTypeTest, ParseSuccess) {
  EXPECT_EQ((JsonObjectWithType{"x", {{"a", "b"}}}),
            JsonObjectWithType::Parse({{"$type", "x"}, {"a", "b"}}));
}

TEST(JsonObjectWithTypeTest, ParseNotObject) {
  EXPECT_THAT(
      JsonObjectWithType::Parse(5),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          "Expected object with string \"\\$type\" member, but received: 5"));
}

TEST(JsonObjectWithTypeTest, ParseNoType) {
  EXPECT_THAT(
      JsonObjectWithType::Parse({{"x", 5}}),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Expected object with string \"\\$type\" member, but "
                    "received: \\{\"x\":5\\}"));
}

TEST(JsonObjectWithTypeTest, ParseTypeNotString) {
  EXPECT_THAT(
      JsonObjectWithType::Parse({{"$type", 5}, {"x", 5}}),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Expected object with string \"\\$type\" member, but "
                    "received: \\{\"\\$type\":5,\"x\":5\\}"));
}

TEST(JsonObjectWithTypeTest, ToJson) {
  JsonObjectWithType a{"x", {{"a", "b"}}};
  EXPECT_EQ((::nlohmann::json{{"$type", "x"}, {"a", "b"}}),
            ::nlohmann::json(a));
}

TEST(JsonObjectWithTypeTest, PrintToOstream) {
  JsonObjectWithType a{"x", {{"a", "b"}}};
  EXPECT_EQ("{\"$type\":\"x\",\"a\":\"b\"}", StrCat(a));
}

}  // namespace
