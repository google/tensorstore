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
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using tensorstore::MatchesStatus;
using tensorstore::Status;
using tensorstore::internal_zarr::Compressor;

class ZlibCompressorTest : public ::testing::TestWithParam<const char*> {};

INSTANTIATE_TEST_SUITE_P(ZlibCompressorTestCases, ZlibCompressorTest,
                         ::testing::Values("zlib", "gzip"));

// Tests that a small input round trips.
TEST_P(ZlibCompressorTest, SmallRoundtrip) {
  auto compressor =
      Compressor::FromJson({{"id", GetParam()}, {"level", 6}}).value();
  const absl::Cord input("The quick brown fox jumped over the lazy dog.");
  absl::Cord encode_result, decode_result;
  TENSORSTORE_ASSERT_OK(compressor->Encode(input, &encode_result, 1));
  TENSORSTORE_ASSERT_OK(compressor->Decode(encode_result, &decode_result, 1));
  EXPECT_EQ(input, decode_result);
}

// Tests that specifying a level of 1 gives the same result as not specifying a
// level.
TEST_P(ZlibCompressorTest, DefaultLevel) {
  auto compressor1 = Compressor::FromJson({{"id", GetParam()}}).value();
  auto compressor2 =
      Compressor::FromJson({{"id", GetParam()}, {"level", 1}}).value();
  const absl::Cord input("The quick brown fox jumped over the lazy dog.");
  absl::Cord encode_result1, encode_result2;
  TENSORSTORE_ASSERT_OK(compressor1->Encode(input, &encode_result1, 1));
  TENSORSTORE_ASSERT_OK(compressor2->Encode(input, &encode_result2, 1));
  EXPECT_EQ(encode_result1, encode_result2);
}

// Tests that specifying a level of 9 gives a result that is different from not
// specifying a level.
TEST_P(ZlibCompressorTest, NonDefaultLevel) {
  auto compressor1 = Compressor::FromJson({{"id", GetParam()}}).value();
  auto compressor2 =
      Compressor::FromJson({{"id", GetParam()}, {"level", 9}}).value();
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
TEST_P(ZlibCompressorTest, InvalidParameter) {
  EXPECT_THAT(Compressor::FromJson({{"id", GetParam()}, {"level", "6"}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error parsing object member \"level\": .*"));
  EXPECT_THAT(Compressor::FromJson({{"id", GetParam()}, {"level", -1}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error parsing object member \"level\": .*"));
  EXPECT_THAT(Compressor::FromJson({{"id", GetParam()}, {"level", 10}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error parsing object member \"level\": .*"));
  EXPECT_THAT(Compressor::FromJson({{"id", GetParam()}, {"foo", 10}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Object includes extra members: \"foo\""));
}

TEST_P(ZlibCompressorTest, ToJson) {
  auto compressor =
      Compressor::FromJson({{"id", GetParam()}, {"level", 5}}).value();
  EXPECT_EQ(nlohmann::json({{"id", GetParam()}, {"level", 5}}),
            compressor.ToJson());
}

}  // namespace
