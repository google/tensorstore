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

#include "tensorstore/driver/json/json_change_map.h"

#include <string_view>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::MatchesJson;
using ::tensorstore::MatchesStatus;
using ::tensorstore::internal_json_driver::JsonChangeMap;
using ::testing::ElementsAre;
using ::testing::Optional;
using ::testing::Pair;

TEST(JsonChangeMapTest, AddChangeValid) {
  JsonChangeMap changes;
  TENSORSTORE_EXPECT_OK(changes.AddChange("/a/b/c", 42));
  EXPECT_THAT(changes.underlying_map(),
              ElementsAre(Pair("/a/b/c", MatchesJson(42))));
  TENSORSTORE_EXPECT_OK(changes.AddChange("/a/b/a", false));
  EXPECT_THAT(changes.underlying_map(),
              ElementsAre(Pair("/a/b/a", MatchesJson(false)),
                          Pair("/a/b/c", MatchesJson(42))));

  // Overwrite previous change for "/a/b/a"
  TENSORSTORE_EXPECT_OK(changes.AddChange("/a/b/a", true));
  EXPECT_THAT(changes.underlying_map(),
              ElementsAre(Pair("/a/b/a", MatchesJson(true)),
                          Pair("/a/b/c", MatchesJson(42))));

  // Change to "/a/b" overwrites previous changes
  TENSORSTORE_EXPECT_OK(changes.AddChange("/a/b", {{"d", "xyz"}}));
  EXPECT_THAT(
      changes.underlying_map(),
      ElementsAre(Pair("/a/b", MatchesJson(::nlohmann::json{{"d", "xyz"}}))));

  // Change to "/a/b/c" is merged into "/a/b" change.
  TENSORSTORE_EXPECT_OK(changes.AddChange("/a/b/c", 42));
  EXPECT_THAT(changes.underlying_map(),
              ElementsAre(Pair("/a/b", MatchesJson(::nlohmann::json{
                                           {"d", "xyz"}, {"c", 42}}))));

  // Change to "/a/b/a" is merged into "/a/b" change.
  TENSORSTORE_EXPECT_OK(changes.AddChange("/a/b/a", false));
  EXPECT_THAT(
      changes.underlying_map(),
      ElementsAre(Pair("/a/b", MatchesJson(::nlohmann::json{
                                   {"d", "xyz"}, {"c", 42}, {"a", false}}))));
}

TEST(JsonChangeMapTest, AddChangeValidIndependent) {
  JsonChangeMap changes;
  TENSORSTORE_EXPECT_OK(changes.AddChange("/a/b/c", 42));
  TENSORSTORE_EXPECT_OK(changes.AddChange("/a/e", "xx"));
  TENSORSTORE_EXPECT_OK(changes.AddChange("/a/a", "yy"));
  TENSORSTORE_EXPECT_OK(changes.AddChange("/a/b/a", false));
  TENSORSTORE_EXPECT_OK(changes.AddChange("/a/b", {{"d", "xyz"}}));
  EXPECT_THAT(
      changes.underlying_map(),
      ElementsAre(Pair("/a/a", MatchesJson("yy")),
                  Pair("/a/b", MatchesJson(::nlohmann::json{{"d", "xyz"}})),
                  Pair("/a/e", MatchesJson("xx"))));
}

TEST(JsonChangeMapTest, AddChangeInvalid) {
  JsonChangeMap changes;
  TENSORSTORE_EXPECT_OK(changes.AddChange("/a", 42));
  EXPECT_THAT(changes.AddChange("/a/b", 43),
              MatchesStatus(absl::StatusCode::kFailedPrecondition));
}

TEST(JsonChangeMapTest, ApplyEmptyChangeMap) {
  JsonChangeMap changes;
  EXPECT_THAT(changes.Apply({{"x", "y"}, {"z", "w"}}),
              Optional(MatchesJson(::nlohmann::json{{"x", "y"}, {"z", "w"}})));
  EXPECT_THAT(changes.Apply({{"x", "y"}, {"z", "w"}}, "/x"),
              Optional(MatchesJson(::nlohmann::json("y"))));
}

TEST(JsonChangeMapTest, ApplyContainingChangeMap1) {
  JsonChangeMap changes;
  TENSORSTORE_EXPECT_OK(changes.AddChange("", {{"a", {{"b", {{"c", 42}}}}}}));
  EXPECT_THAT(changes.Apply("old", "/a/b/c"), Optional(MatchesJson(42)));
}

TEST(JsonChangeMapTest, ApplyInvalidContainingChangeMap) {
  JsonChangeMap changes;
  TENSORSTORE_EXPECT_OK(changes.AddChange("/a", {{"b", {{"c", 42}}}}));
  EXPECT_THAT(changes.Apply(false, "/a/b/c"),
              MatchesStatus(absl::StatusCode::kFailedPrecondition));
}

TEST(JsonChangeMapTest, ApplyChangeMapPriorNonContaining) {
  JsonChangeMap changes;
  TENSORSTORE_EXPECT_OK(changes.AddChange("/a", 10));
  EXPECT_THAT(changes.Apply({{"b", 42}}, "/b"), Optional(MatchesJson(42)));
}

TEST(JsonChangeMapTest, ApplyContainingChangeMap2) {
  JsonChangeMap changes;
  TENSORSTORE_EXPECT_OK(changes.AddChange("/a", {{"b", {{"c", 42}}}}));
  EXPECT_THAT(changes.Apply({{"e", "f"}}, "/a/b/c"), Optional(MatchesJson(42)));
}

TEST(JsonChangeMapTest, ApplyChangeMap) {
  JsonChangeMap changes;
  TENSORSTORE_EXPECT_OK(changes.AddChange("/a", {{"b", {{"c", 42}}}}));
  TENSORSTORE_EXPECT_OK(changes.AddChange("/e", 42));
  EXPECT_THAT(changes.Apply({{"x", "y"}, {"e", "f"}}),
              Optional(MatchesJson(::nlohmann::json{
                  {"a", {{"b", {{"c", 42}}}}}, {"e", 42}, {"x", "y"}})));
}

TEST(JsonChangeMapTest, ApplyInvalidChangeMap1) {
  JsonChangeMap changes;
  TENSORSTORE_EXPECT_OK(changes.AddChange("/e", 42));
  EXPECT_THAT(changes.Apply(42),
              MatchesStatus(absl::StatusCode::kFailedPrecondition));
}

TEST(JsonChangeMapTest, ApplyInvalidChangeMap2) {
  JsonChangeMap changes;
  TENSORSTORE_EXPECT_OK(changes.AddChange("/4", 42));
  EXPECT_THAT(changes.Apply({1, 2, 3}),
              MatchesStatus(absl::StatusCode::kFailedPrecondition));
}

TEST(JsonChangeMapTest, ApplyRequestInvalidJsonPointer) {
  JsonChangeMap changes;
  TENSORSTORE_EXPECT_OK(changes.AddChange("/a/b", 42));
  EXPECT_THAT(changes.Apply(false, "/a"),
              MatchesStatus(absl::StatusCode::kFailedPrecondition));
}

TEST(JsonChangeMapTest, ApplyRequestInvalidJsonPointerNoChanges) {
  JsonChangeMap changes;
  EXPECT_THAT(changes.Apply(false, "/a"),
              MatchesStatus(absl::StatusCode::kFailedPrecondition));
}

TEST(JsonChangeMapTest, ApplyRequestNewMember) {
  JsonChangeMap changes;
  TENSORSTORE_EXPECT_OK(changes.AddChange("/a/b", 42));
  EXPECT_THAT(changes.Apply(::nlohmann::json::object_t{}, "/a"),
              Optional(MatchesJson(::nlohmann::json{{"b", 42}})));
}

TEST(JsonChangeMapTest, ApplyIncompatibleChangeExactRequest) {
  JsonChangeMap changes;
  TENSORSTORE_EXPECT_OK(changes.AddChange("/a", 42));
  EXPECT_THAT(changes.Apply(false, "/a"),
              MatchesStatus(absl::StatusCode::kFailedPrecondition));
}

TEST(JsonChangeMapTest, AddIncompatibleChanges) {
  JsonChangeMap changes;
  TENSORSTORE_EXPECT_OK(changes.AddChange("", 42));
  EXPECT_THAT(changes.AddChange("/a", 50),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            "JSON Pointer reference \"/a\" cannot be applied "
                            "to number value: 42"));
}

TEST(JsonChangeMapTest, CanApplyUnconditionally) {
  JsonChangeMap changes;
  EXPECT_FALSE(changes.CanApplyUnconditionally(""));
  EXPECT_FALSE(changes.CanApplyUnconditionally("/a/b/c"));
  TENSORSTORE_EXPECT_OK(changes.AddChange("/a/b", {{"c", 42}}));
  EXPECT_TRUE(changes.CanApplyUnconditionally("/a/b/c"));
  EXPECT_TRUE(changes.CanApplyUnconditionally("/a/b"));
  EXPECT_TRUE(changes.CanApplyUnconditionally("/a/b/d"));
  EXPECT_FALSE(changes.CanApplyUnconditionally("/a"));
  EXPECT_FALSE(changes.CanApplyUnconditionally("/a/x"));
  EXPECT_FALSE(changes.CanApplyUnconditionally(""));

  TENSORSTORE_EXPECT_OK(changes.AddChange("", {{"a", false}}));
  EXPECT_TRUE(changes.CanApplyUnconditionally(""));
  EXPECT_TRUE(changes.CanApplyUnconditionally("/a"));
}

}  // namespace
