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

#include "tensorstore/internal/json_array.h"

#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "tensorstore/array.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/internal/json.h"
#include "tensorstore/internal/json_fwd.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::nlohmann::json;
using tensorstore::DataTypeOf;
using tensorstore::MatchesStatus;
using tensorstore::Result;
using tensorstore::Status;
using tensorstore::internal::JsonEncodeNestedArray;
using tensorstore::internal::JsonParseNestedArray;
using tensorstore::internal::JsonValueAs;

TEST(JsonEncodeNestedArray, Rank0) {
  EXPECT_EQ((::nlohmann::json(1)),
            JsonEncodeNestedArray(tensorstore::MakeScalarArray(1),
                                  [](const int* i) { return *i; }));
}

TEST(JsonEncodeNestedArray, Rank1) {
  EXPECT_EQ((::nlohmann::json({1, 2, 3})),
            JsonEncodeNestedArray(tensorstore::MakeOffsetArray({3}, {1, 2, 3}),
                                  [](const int* i) { return *i; }));
}

TEST(JsonEncodeNestedArray, Rank1Offset) {
  EXPECT_EQ((::nlohmann::json({1, 2, 3})),
            JsonEncodeNestedArray(tensorstore::MakeArray({1, 2, 3}),
                                  [](const int* i) { return *i; }));
}

TEST(JsonEncodeNestedArray, Rank1ZeroSize) {
  EXPECT_EQ((::nlohmann::json::array()),
            JsonEncodeNestedArray(tensorstore::AllocateArray<int>({0}),
                                  [](const int* i) { return *i; }));
}

TEST(JsonEncodeNestedArray, Rank2) {
  EXPECT_EQ(
      (::nlohmann::json{{1, 2, 3}, {4, 5, 6}}),
      JsonEncodeNestedArray(tensorstore::MakeArray({{1, 2, 3}, {4, 5, 6}}),
                            [](const int* i) { return *i; }));
}

TEST(JsonEncodeNestedArray, Rank2ZeroSizeDim0) {
  EXPECT_EQ(::nlohmann::json::array(),
            JsonEncodeNestedArray(tensorstore::AllocateArray<int>({0, 2}),
                                  [](const int* i) { return *i; }));
}

TEST(JsonEncodeNestedArray, Rank2ZeroSizeDim1) {
  EXPECT_EQ(
      (::nlohmann::json{::nlohmann::json::array(), ::nlohmann::json::array()}),
      JsonEncodeNestedArray(tensorstore::AllocateArray<int>({2, 0}),
                            [](const int* i) { return *i; }));
}

TEST(JsonEncodeNestedArray, Rank2Fortran) {
  EXPECT_EQ(
      (::nlohmann::json{{1, 2, 3}, {4, 5, 6}}),
      JsonEncodeNestedArray(
          tensorstore::MakeCopy(tensorstore::MakeArray({{1, 2, 3}, {4, 5, 6}}),
                                tensorstore::fortran_order),
          [](const int* i) { return *i; }));
}

TEST(JsonEncodeNestedArray, Rank3) {
  EXPECT_EQ(
      (::nlohmann::json{{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}}),
      JsonEncodeNestedArray(tensorstore::MakeArray({{{1, 2, 3}, {4, 5, 6}},
                                                    {{7, 8, 9}, {10, 11, 12}}}),
                            [](const int* i) { return *i; }));
}

TEST(JsonEncodeNestedArray, Rank3Fortran) {
  EXPECT_EQ(
      (::nlohmann::json{{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}}),
      JsonEncodeNestedArray(
          tensorstore::MakeCopy(
              tensorstore::MakeArray(
                  {{{1, 2, 3}, {4, 5, 6}}, {{7, 8, 9}, {10, 11, 12}}}),
              tensorstore::fortran_order),
          [](const int* i) { return *i; }));
}

TEST(JsonEncodeNestedArray, DataTypeConversionInt) {
  EXPECT_THAT(
      JsonEncodeNestedArray(tensorstore::MakeArray({{1, 2, 3}, {4, 5, 6}})),
      ::testing::Optional(::nlohmann::json{{1, 2, 3}, {4, 5, 6}}));
}

TEST(JsonEncodeNestedArray, DataTypeConversionString) {
  EXPECT_THAT(
      JsonEncodeNestedArray(tensorstore::MakeArray<std::string>(
          {{"a", "b", "c"}, {"d", "e", "f"}})),
      ::testing::Optional(::nlohmann::json{{"a", "b", "c"}, {"d", "e", "f"}}));
}

TEST(JsonEncodeNestedArray, DataTypeConversionStringError) {
  EXPECT_THAT(JsonEncodeNestedArray(
                  tensorstore::MakeArray<std::string>({"a", "b\xff"})),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Invalid UTF-8 sequence encountered"));
}

TEST(JsonEncodeNestedArray, DataTypeConversionByteError) {
  EXPECT_THAT(JsonEncodeNestedArray(tensorstore::MakeArray<std::byte>(
                  {std::byte{1}, std::byte{2}})),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Conversion from byte to JSON is not implemented"));
}

Result<std::int64_t> DecodeInt64(const ::nlohmann::json& v) {
  if (auto x = JsonValueAs<std::int64_t>(v)) return *x;
  return absl::InvalidArgumentError("Invalid integer");
}

TEST(JsonParseNestedArrayTest, RankZero) {
  EXPECT_EQ(tensorstore::MakeScalarArray<std::int64_t>(1),
            JsonParseNestedArray(::nlohmann::json(1), &DecodeInt64));
}

TEST(JsonParseNestedArrayTest, RankOne) {
  EXPECT_EQ(tensorstore::MakeArray<std::int64_t>({1, 2, 3}),
            JsonParseNestedArray(::nlohmann::json{1, 2, 3}, &DecodeInt64));
}

TEST(JsonParseNestedArrayTest, RankTwo) {
  EXPECT_EQ(tensorstore::MakeArray<std::int64_t>({{1, 2, 3}, {4, 5, 6}}),
            JsonParseNestedArray(::nlohmann::json{{1, 2, 3}, {4, 5, 6}},
                                 &DecodeInt64));
}

TEST(JsonParseNestedArrayTest, DecodeElementError) {
  EXPECT_THAT(JsonParseNestedArray(::nlohmann::json{{1, 2, 3}, {4, 5, "a"}},
                                   &DecodeInt64),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error parsing array element at position \\{1, "
                            "2\\}: Invalid integer"));
}

TEST(JsonParseNestedArrayTest, TooShallow) {
  EXPECT_THAT(
      JsonParseNestedArray(::nlohmann::json{{{1, 2}, 2, 3}, {4, 5, 6}},
                           &DecodeInt64),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Expected rank-3 array, but found non-array element 2 at "
                    "position \\{0, 1\\}\\."));
}

TEST(JsonParseNestedArrayTest, TooDeep) {
  EXPECT_THAT(JsonParseNestedArray(::nlohmann::json{{1, {2, 3}, 3}, {4, 5, 6}},
                                   &DecodeInt64),
              MatchesStatus(
                  absl::StatusCode::kInvalidArgument,
                  "Expected rank-2 array, but found array element \\[2,3\\] at "
                  "position \\{0, 1\\}\\."));
}

TEST(JsonParseNestedArrayTest, Ragged) {
  EXPECT_THAT(
      JsonParseNestedArray(::nlohmann::json{{1, 2, 3}, {4, 5}}, &DecodeInt64),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Expected array of shape \\{2, 3\\}, but found array "
                    "element \\[4,5\\] of length 2 at position \\{1\\}."));
}

TEST(JsonParseNestedArrayTest, ZeroSize) {
  EXPECT_EQ(tensorstore::AllocateArray<std::int64_t>({2, 0}),
            JsonParseNestedArray(
                ::nlohmann::json::array_t{::nlohmann::json::array_t(),
                                          ::nlohmann::json::array_t()},
                &DecodeInt64));
}

TEST(JsonParseNestedArrayTest, DataTypeConversionInt) {
  EXPECT_THAT(
      JsonParseNestedArray(::nlohmann::json{{1, 2, 3}, {4, 5, 6}},
                           DataTypeOf<std::int32_t>(), 2),
      ::testing::Optional(tensorstore::MakeArray({{1, 2, 3}, {4, 5, 6}})));
}

TEST(JsonParseNestedArrayTest, DataTypeConversionIntRankError) {
  EXPECT_THAT(
      JsonParseNestedArray(::nlohmann::json{{1, 2, 3}, {4, 5, 6}},
                           DataTypeOf<std::int32_t>(), 3),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Array rank \\(2\\) does not match expected rank \\(3\\)"));
}

TEST(JsonParseNestedArrayTest, DataTypeConversionString) {
  EXPECT_THAT(
      JsonParseNestedArray(::nlohmann::json{{"a", "b", "c"}, {"d", "e", "f"}},
                           DataTypeOf<std::string>(), 2),
      ::testing::Optional(tensorstore::MakeArray<std::string>(
          {{"a", "b", "c"}, {"d", "e", "f"}})));
}

TEST(JsonParseNestedArray, DataTypeConversionStringError) {
  EXPECT_THAT(
      JsonParseNestedArray(::nlohmann::json{{"a", "b", 3}, {"d", "e", "f"}},
                           DataTypeOf<std::string>(), 2),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Error parsing array element at position \\{0, 2\\}: "
                    "Expected string, but received: 3"));
}

TEST(JsonParseNestedArray, DataTypeConversionByteError) {
  EXPECT_THAT(
      JsonParseNestedArray(::nlohmann::json{{"a", "b", 3}, {"d", "e", "f"}},
                           DataTypeOf<std::byte>(), 2),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Conversion from JSON to byte is not implemented"));
}

}  // namespace
