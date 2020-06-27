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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/driver/zarr/compressor.h"
#include "tensorstore/internal/json.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using tensorstore::MatchesStatus;
using tensorstore::Status;
using tensorstore::internal_zarr::Compressor;

// Tests that a small input round trips.
TEST(Bzip2CompressorTest, SmallRoundtrip) {
  auto compressor = Compressor::FromJson({{"id", "bz2"}, {"level", 6}}).value();
  const absl::Cord input("The quick brown fox jumped over the lazy dog.");
  absl::Cord encode_result, decode_result;
  TENSORSTORE_ASSERT_OK(compressor->Encode(input, &encode_result, 1));
  TENSORSTORE_ASSERT_OK(compressor->Decode(encode_result, &decode_result, 1));
  EXPECT_EQ(input, decode_result);
}

// Tests that round tripping works for with an input that exceeds the 16KiB
// buffer size.
TEST(Bzip2CompressorTest, LargeRoundtrip) {
  std::string input(100000, '\0');
  unsigned char x = 0;
  for (auto& v : input) {
    v = x;
    x += 7;
  }
  auto compressor = Compressor::FromJson({{"id", "bz2"}, {"level", 6}}).value();
  absl::Cord encode_result, decode_result;
  TENSORSTORE_ASSERT_OK(
      compressor->Encode(absl::Cord(input), &encode_result, 1));
  TENSORSTORE_ASSERT_OK(compressor->Decode(encode_result, &decode_result, 1));
  EXPECT_EQ(input, decode_result);
}

// Tests that specifying a level of 1 gives the same result as not specifying a
// level.
TEST(Bzip2CompressorTest, DefaultLevel) {
  auto compressor1 = Compressor::FromJson({{"id", "bz2"}}).value();
  auto compressor2 =
      Compressor::FromJson({{"id", "bz2"}, {"level", 1}}).value();
  const absl::Cord input("The quick brown fox jumped over the lazy dog.");
  absl::Cord encode_result1, encode_result2;
  TENSORSTORE_ASSERT_OK(compressor1->Encode(input, &encode_result1, 1));
  TENSORSTORE_ASSERT_OK(compressor2->Encode(input, &encode_result2, 1));
  EXPECT_EQ(encode_result1, encode_result2);
}

// Tests that specifying a level of 9 gives a result that is different from not
// specifying a level.
TEST(Bzip2CompressorTest, NonDefaultLevel) {
  auto compressor1 = Compressor::FromJson({{"id", "bz2"}}).value();
  auto compressor2 =
      Compressor::FromJson({{"id", "bz2"}, {"level", 9}}).value();
  const absl::Cord input("The quick brown fox jumped over the lazy dog.");
  absl::Cord encode_result1, encode_result2;
  TENSORSTORE_ASSERT_OK(compressor1->Encode(input, &encode_result1, 1));
  TENSORSTORE_ASSERT_OK(compressor2->Encode(input, &encode_result2, 1));
  EXPECT_NE(encode_result1, encode_result2);
  absl::Cord decode_result;
  TENSORSTORE_ASSERT_OK(compressor2->Decode(encode_result2, &decode_result, 1));
  EXPECT_EQ(input, decode_result);
}

// Tests that an invalid parameter gives an error.
TEST(Bzip2CompressorTest, InvalidParameter) {
  EXPECT_THAT(Compressor::FromJson({{"id", "bz2"}, {"level", "6"}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error parsing object member \"level\": .*"));
  EXPECT_THAT(Compressor::FromJson({{"id", "bz2"}, {"level", -1}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error parsing object member \"level\": .*"));
  EXPECT_THAT(Compressor::FromJson({{"id", "bz2"}, {"level", 10}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error parsing object member \"level\": .*"));
  EXPECT_THAT(Compressor::FromJson({{"id", "bz2"}, {"foo", 10}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Object includes extra members: \"foo\""));
}

TEST(Bzip2CompressorTest, ToJson) {
  auto compressor = Compressor::FromJson({{"id", "bz2"}, {"level", 5}}).value();
  EXPECT_EQ(nlohmann::json({{"id", "bz2"}, {"level", 5}}), compressor.ToJson());
}

}  // namespace
