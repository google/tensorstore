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

#include "tensorstore/internal/json_pointer.h"

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
using ::tensorstore::json_pointer::Compare;
using ::tensorstore::json_pointer::CompareResult;
using ::tensorstore::json_pointer::Dereference;
using ::tensorstore::json_pointer::EncodeReferenceToken;
using ::tensorstore::json_pointer::kCreate;
using ::tensorstore::json_pointer::kDelete;
using ::tensorstore::json_pointer::kMustExist;
using ::tensorstore::json_pointer::kSimulateCreate;
using ::tensorstore::json_pointer::Replace;
using ::tensorstore::json_pointer::Validate;
using ::testing::Optional;
using ::testing::Pointee;

TEST(ValidateTest, Valid) {
  TENSORSTORE_EXPECT_OK(Validate(""));
  TENSORSTORE_EXPECT_OK(Validate("/"));
  TENSORSTORE_EXPECT_OK(Validate("/a/"));
  TENSORSTORE_EXPECT_OK(Validate("/abc"));
  TENSORSTORE_EXPECT_OK(Validate("/abc/"));
  TENSORSTORE_EXPECT_OK(Validate("/abc/def"));
  TENSORSTORE_EXPECT_OK(Validate("/abc/def/xy~0/~1"));
}

TEST(ValidateTest, Invalid) {
  EXPECT_THAT(Validate("foo"),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "JSON Pointer does not start with '/': \"foo\""));

  EXPECT_THAT(
      Validate("/~~"),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          "JSON Pointer requires '~' to be followed by '0' or '1': \"/~~\""));

  EXPECT_THAT(
      Validate("/~"),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          "JSON Pointer requires '~' to be followed by '0' or '1': \"/~\""));
  // Verify bounds checking.
  EXPECT_THAT(
      Validate(std::string_view("/~0", 2)),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          "JSON Pointer requires '~' to be followed by '0' or '1': \"/~\""));
}

TEST(CompareTest, Basic) {
  EXPECT_EQ(Compare("", ""), CompareResult::kEqual);
  EXPECT_EQ(Compare("", "/foo"), CompareResult::kContains);
  EXPECT_EQ(Compare("/foo", ""), CompareResult::kContainedIn);
  EXPECT_EQ(Compare("/a", "/b"), CompareResult::kLessThan);
  EXPECT_EQ(Compare("/a", "/ab"), CompareResult::kLessThan);
  EXPECT_EQ(Compare("/a/b", "/acc"), CompareResult::kLessThan);
  EXPECT_EQ(Compare("/acc", "/a/b"), CompareResult::kGreaterThan);
  EXPECT_EQ(Compare("/a*c", "/a/b"), CompareResult::kGreaterThan);
  EXPECT_EQ(Compare("/ab", "/a"), CompareResult::kGreaterThan);
  EXPECT_EQ(Compare("/a~0", "/a~1"), CompareResult::kGreaterThan);
  EXPECT_EQ(Compare("/a~1", "/a~0"), CompareResult::kLessThan);
  EXPECT_EQ(Compare("/a~0", "/ax"), CompareResult::kGreaterThan);
  EXPECT_EQ(Compare("/a~1", "/ax"), CompareResult::kLessThan);
  EXPECT_EQ(Compare("/ax", "/a~0"), CompareResult::kLessThan);
  EXPECT_EQ(Compare("/ax", "/a~1"), CompareResult::kGreaterThan);
  EXPECT_EQ(Compare("/xx", "/xx/abc"), CompareResult::kContains);
  EXPECT_EQ(Compare("/xx/abc", "/xx"), CompareResult::kContainedIn);
  EXPECT_EQ(Compare("/abc", "/acc"), CompareResult::kLessThan);
  EXPECT_EQ(Compare("/b", "/a"), CompareResult::kGreaterThan);
  EXPECT_EQ(Compare("/ba", "/ab"), CompareResult::kGreaterThan);
}

TEST(EncodeReferenceTokenTest, Basic) {
  EXPECT_EQ("", EncodeReferenceToken(""));
  EXPECT_EQ("abc", EncodeReferenceToken("abc"));
  EXPECT_EQ("abc~0", EncodeReferenceToken("abc~"));
  EXPECT_EQ("abc~1", EncodeReferenceToken("abc/"));
  EXPECT_EQ("abc~1~0xyz", EncodeReferenceToken("abc/~xyz"));
}

TEST(DereferenceTest, ExamplesFromRfc6901) {
  ::nlohmann::json document = {
      {"foo", {"bar", "baz"}},
      {"", 0},
      {"a/b", 1},
      {"c%d", 2},
      {"e^f", 3},
      {"g|h", 4},
      {"i\\j", 5},
      {"k\"l", 6},
      {" ", 7},
      {"m~n", 8},
  };

  EXPECT_THAT(Dereference(document, "", kMustExist), Optional(&document));
  EXPECT_THAT(Dereference(document, "/foo", kMustExist),
              Optional(Pointee(::nlohmann::json{"bar", "baz"})));
  EXPECT_THAT(Dereference(document, "/foo/0", kMustExist),
              Optional(Pointee(::nlohmann::json("bar"))));
  EXPECT_THAT(Dereference(document, "/", kMustExist), Optional(Pointee(0)));
  EXPECT_THAT(Dereference(document, "/a~1b", kMustExist), Optional(Pointee(1)));
  EXPECT_THAT(Dereference(document, "/c%d", kMustExist), Optional(Pointee(2)));
  EXPECT_THAT(Dereference(document, "/e^f", kMustExist), Optional(Pointee(3)));
  EXPECT_THAT(Dereference(document, "/g|h", kMustExist), Optional(Pointee(4)));
  EXPECT_THAT(Dereference(document, "/i\\j", kMustExist), Optional(Pointee(5)));
  EXPECT_THAT(Dereference(document, "/k\"l", kMustExist), Optional(Pointee(6)));
  EXPECT_THAT(Dereference(document, "/ ", kMustExist), Optional(Pointee(7)));
  EXPECT_THAT(Dereference(document, "/m~0n", kMustExist), Optional(Pointee(8)));
}

TEST(DereferenceTest, ConstAccess) {
  EXPECT_THAT(Dereference(true, "", kMustExist), Optional(Pointee(true)));
  EXPECT_THAT(Dereference(true, "/", kMustExist),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            "JSON Pointer reference \"/\" cannot be applied to "
                            "boolean value: true"));
  EXPECT_THAT(
      Dereference(true, "/a/b/c", kMustExist),
      MatchesStatus(absl::StatusCode::kFailedPrecondition,
                    "JSON Pointer reference \"/a\" cannot be applied to "
                    "boolean value: true"));
  EXPECT_THAT(Dereference({1, 2, 3}, "/0", kMustExist), Optional(Pointee(1)));
  EXPECT_THAT(Dereference({1, 2, 3}, "/1", kMustExist), Optional(Pointee(2)));
  EXPECT_THAT(
      Dereference({1, 2, 3}, "/3", kMustExist),
      MatchesStatus(absl::StatusCode::kOutOfRange,
                    "JSON Pointer \"/3\" is out-of-range for array of size 3"));
  EXPECT_THAT(Dereference({1, 2, 3}, "/a", kMustExist),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            "JSON Pointer \"/a\" is invalid for array value"));
  EXPECT_THAT(Dereference({1, 2, 3}, "/ 1", kMustExist),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            "JSON Pointer \"/ 1\" is invalid for array value"));
  EXPECT_THAT(Dereference({1, 2, 3}, "/00", kMustExist),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            "JSON Pointer \"/00\" is invalid for array value"));
  EXPECT_THAT(Dereference({1, 2, 3}, "/", kMustExist),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            "JSON Pointer \"/\" is invalid for array value"));
  EXPECT_THAT(Dereference({1, 2, 3}, "/-", kMustExist),
              MatchesStatus(
                  absl::StatusCode::kFailedPrecondition,
                  "JSON Pointer \"/-\" refers to non-existent array element"));
  EXPECT_THAT(Dereference({1, {{"a", 7}, {"b", 8}}, 3}, "/1/a", kMustExist),
              Optional(Pointee(7)));
  EXPECT_THAT(
      Dereference({1, {{"a", 7}, {"b", 8}}, 3}, "/1/c", kMustExist),
      MatchesStatus(
          absl::StatusCode::kNotFound,
          "JSON Pointer \"/1/c\" refers to non-existent object member"));
  EXPECT_THAT(
      Dereference({1, {{"a", 7}, {"b", 8}}, 3}, "/1/c", kMustExist),
      MatchesStatus(
          absl::StatusCode::kNotFound,
          "JSON Pointer \"/1/c\" refers to non-existent object member"));
  EXPECT_THAT(
      Dereference(::nlohmann::json::value_t::discarded, "/a/b", kMustExist),
      MatchesStatus(absl::StatusCode::kNotFound, ""));
}

TEST(DereferenceTest, NonConstAccess) {
  {
    ::nlohmann::json doc{1, 2, 3};
    TENSORSTORE_EXPECT_OK(Dereference(doc, "/-", kCreate));
    EXPECT_THAT(doc, MatchesJson(::nlohmann::json{
                         1, 2, 3, ::nlohmann::json::value_t::discarded}));
  }

  {
    ::nlohmann::json doc(::nlohmann::json::value_t::discarded);
    EXPECT_THAT(Dereference(doc, "", kCreate), Optional(&doc));
    EXPECT_THAT(doc, MatchesJson(::nlohmann::json::value_t::discarded));
  }

  {
    ::nlohmann::json doc(::nlohmann::json::value_t::discarded);
    EXPECT_THAT(Dereference(doc, "", kMustExist),
                MatchesStatus(absl::StatusCode::kNotFound));
    EXPECT_THAT(doc, MatchesJson(::nlohmann::json::value_t::discarded));
  }

  {
    ::nlohmann::json doc(::nlohmann::json::value_t::discarded);
    EXPECT_THAT(Dereference(doc, "", kDelete), Optional(nullptr));
    EXPECT_THAT(doc, MatchesJson(::nlohmann::json::value_t::discarded));
  }

  {
    ::nlohmann::json doc(::nlohmann::json::value_t::discarded);
    EXPECT_THAT(Dereference(doc, "/a", kDelete), Optional(nullptr));
    EXPECT_THAT(doc, MatchesJson(::nlohmann::json::value_t::discarded));
  }

  {
    ::nlohmann::json doc(::nlohmann::json::value_t::discarded);
    EXPECT_THAT(Dereference(doc, "", kSimulateCreate), Optional(&doc));
    EXPECT_THAT(doc, MatchesJson(::nlohmann::json::value_t::discarded));
  }

  {
    ::nlohmann::json doc(::nlohmann::json::value_t::discarded);
    TENSORSTORE_EXPECT_OK(Dereference(doc, "/a/b/c", kCreate));
    EXPECT_THAT(
        doc,
        MatchesJson(::nlohmann::json{
            {"a", {{"b", {{"c", ::nlohmann::json::value_t::discarded}}}}}}));
  }

  {
    ::nlohmann::json doc{1, 2, 3};
    TENSORSTORE_EXPECT_OK(Dereference(doc, "/-/x", kCreate));
    EXPECT_THAT(doc,
                MatchesJson(::nlohmann::json{
                    1, 2, 3, {{"x", ::nlohmann::json::value_t::discarded}}}));
  }

  {
    ::nlohmann::json doc{1, 2, 3};
    EXPECT_THAT(
        Dereference(doc, "/-/a", kMustExist),
        MatchesStatus(
            absl::StatusCode::kFailedPrecondition,
            "JSON Pointer \"/-\" refers to non-existent array element"));
  }
}

TEST(ReplaceTest, ReplaceEntireValue) {
  ::nlohmann::json doc{1, 2, 3};
  TENSORSTORE_EXPECT_OK(Replace(doc, "", 42));
  EXPECT_THAT(doc, MatchesJson(42));
}

TEST(ReplaceTest, DeleteEntireValue) {
  ::nlohmann::json doc{1, 2, 3};
  TENSORSTORE_EXPECT_OK(Replace(doc, "", ::nlohmann::json::value_t::discarded));
  EXPECT_THAT(doc, MatchesJson(::nlohmann::json::value_t::discarded));
}

TEST(ReplaceTest, ReplaceArrayElement) {
  ::nlohmann::json doc{1, 2, 3};
  TENSORSTORE_EXPECT_OK(Replace(doc, "/1", 42));
  EXPECT_THAT(doc, MatchesJson(::nlohmann::json{1, 42, 3}));
}

TEST(ReplaceTest, ReplaceNestedWithinArrayElement) {
  ::nlohmann::json doc{1, {{"a", 2}}, 3};
  TENSORSTORE_EXPECT_OK(Replace(doc, "/1/a", 42));
  EXPECT_THAT(doc, MatchesJson(::nlohmann::json{1, {{"a", 42}}, 3}));
}

TEST(ReplaceTest, DeleteNestedWithinArrayElement) {
  ::nlohmann::json doc{1, {{"a", 2}}, 3};
  TENSORSTORE_EXPECT_OK(
      Replace(doc, "/1/a", ::nlohmann::json::value_t::discarded));
  EXPECT_THAT(
      doc, MatchesJson(::nlohmann::json{1, ::nlohmann::json::object_t(), 3}));
}

TEST(ReplaceTest, AppendNestedMember) {
  ::nlohmann::json doc{1, 2, 3};
  TENSORSTORE_EXPECT_OK(Replace(doc, "/-/a/b/c", 42));
  EXPECT_THAT(doc, MatchesJson(::nlohmann::json{
                       1, 2, 3, {{"a", {{"b", {{"c", 42}}}}}}}));
}

TEST(ReplaceTest, ReplaceNestedMember) {
  ::nlohmann::json doc{1, {{"d", false}}, 3};
  TENSORSTORE_EXPECT_OK(Replace(doc, "/1/a/b/c", 42));
  EXPECT_THAT(doc, MatchesJson(::nlohmann::json{
                       1, {{"a", {{"b", {{"c", 42}}}}}, {"d", false}}, 3}));
}

TEST(ReplaceTest, DeleteNestedMember) {
  ::nlohmann::json doc{{"a", {{"b", {{"c", 42}}}}}, {"d", false}};
  TENSORSTORE_EXPECT_OK(
      Replace(doc, "/a/b/c", ::nlohmann::json::value_t::discarded));
  EXPECT_THAT(doc,
              MatchesJson(::nlohmann::json{
                  {"a", {{"b", ::nlohmann::json::object_t()}}}, {"d", false}}));
}

TEST(ReplaceTest, DeleteMissingMember) {
  ::nlohmann::json doc{{"a", {{"b", {{"c", 42}}}}}, {"d", false}};
  TENSORSTORE_EXPECT_OK(
      Replace(doc, "/a/e", ::nlohmann::json::value_t::discarded));
  EXPECT_THAT(doc, MatchesJson(::nlohmann::json{{"a", {{"b", {{"c", 42}}}}},
                                                {"d", false}}));
}

TEST(ReplaceTest, DeleteMissingNestedMember) {
  ::nlohmann::json doc{{"a", {{"b", {{"c", 42}}}}}, {"d", false}};
  TENSORSTORE_EXPECT_OK(
      Replace(doc, "/a/e/f", ::nlohmann::json::value_t::discarded));
  EXPECT_THAT(doc, MatchesJson(::nlohmann::json{{"a", {{"b", {{"c", 42}}}}},
                                                {"d", false}}));
}

TEST(ReplaceTest, DeleteArrayElement) {
  ::nlohmann::json doc{1, 2, 3};
  TENSORSTORE_EXPECT_OK(
      Replace(doc, "/1", ::nlohmann::json::value_t::discarded));
  EXPECT_THAT(doc, MatchesJson(::nlohmann::json{1, 3}));
}

TEST(ReplaceTest, DeleteNewArrayElement) {
  ::nlohmann::json doc{1, 2, 3};
  TENSORSTORE_EXPECT_OK(
      Replace(doc, "/-", ::nlohmann::json::value_t::discarded));
  EXPECT_THAT(doc, MatchesJson(::nlohmann::json{1, 2, 3}));
}

TEST(ReplaceTest, DeleteOutOfRangeArrayElement) {
  ::nlohmann::json doc{1, 2, 3};
  TENSORSTORE_EXPECT_OK(
      Replace(doc, "/4", ::nlohmann::json::value_t::discarded));
  EXPECT_THAT(doc, MatchesJson(::nlohmann::json{1, 2, 3}));
}

TEST(ReplaceTest, DeleteInvalidElement) {
  ::nlohmann::json doc(false);
  EXPECT_THAT(Replace(doc, "/4", ::nlohmann::json::value_t::discarded),
              MatchesStatus(absl::StatusCode::kFailedPrecondition,
                            "JSON Pointer reference \"/4\" cannot be applied "
                            "to boolean value: false"));
  EXPECT_THAT(doc, MatchesJson(false));
}

}  // namespace
