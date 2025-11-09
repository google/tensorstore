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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include <nlohmann/json.hpp>
#include "tensorstore/driver/n5/compressor.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::MatchesStatus;
using ::tensorstore::internal_n5::Compressor;

absl::Cord GetInput() {
  return absl::Cord(
      "Sed ut perspiciatis unde omnis iste natus error sit voluptatem "
      "accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae "
      "ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt "
      "explicabo. Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut "
      "odit aut fugit, sed quia consequuntur magni dolores eos qui ratione "
      "voluptatem sequi nesciunt. Neque porro quisquam est, qui dolorem ipsum "
      "quia dolor sit amet, consectetur, adipisci velit, sed quia non numquam "
      "eius modi tempora incidunt ut labore et dolore magnam aliquam quaerat "
      "voluptatem. Ut enim ad minima veniam, quis nostrum exercitationem ullam "
      "corporis suscipit laboriosam, nisi ut aliquid ex ea commodi "
      "consequatur? Quis autem vel eum iure reprehenderit qui in ea voluptate "
      "velit esse quam nihil molestiae consequatur, vel illum qui dolorem eum "
      "fugiat quo voluptas nulla pariatur?");
}

// Tests that a small input round trips.
TEST(ZstdCompressorTest, SmallRoundtrip) {
  auto compressor =
      Compressor::FromJson({{"type", "zstd"}, {"level", 6}}).value();
  const absl::Cord input = GetInput();
  absl::Cord encode_result, decode_result;
  TENSORSTORE_ASSERT_OK(compressor->Encode(GetInput(), &encode_result, 1));
  TENSORSTORE_ASSERT_OK(compressor->Decode(encode_result, &decode_result, 1));
  EXPECT_EQ(input, decode_result);
}

// Tests that specifying a level of 3 gives the same result as not specifying a
// level.
TEST(ZstdCompressorTest, DefaultLevel) {
  auto compressor1 = Compressor::FromJson({{"type", "zstd"}}).value();
  auto compressor2 =
      Compressor::FromJson({{"type", "zstd"}, {"level", 3}}).value();
  const absl::Cord input = GetInput();
  absl::Cord encode_result1, encode_result2;
  TENSORSTORE_ASSERT_OK(compressor1->Encode(input, &encode_result1, 1));
  TENSORSTORE_ASSERT_OK(compressor2->Encode(input, &encode_result2, 1));
  EXPECT_EQ(encode_result1, encode_result2);
}

// Tests that the default level is different from level 1.
TEST(ZstdCompressorTest, NonDefaultLevel) {
  auto compressor1 = Compressor::FromJson({{"type", "zstd"}}).value();
  auto compressor2 =
      Compressor::FromJson({{"type", "zstd"}, {"level", 1}}).value();
  const absl::Cord input = GetInput();
  absl::Cord encode_result1, encode_result2;
  TENSORSTORE_ASSERT_OK(compressor1->Encode(input, &encode_result1, 1));
  TENSORSTORE_ASSERT_OK(compressor2->Encode(input, &encode_result2, 1));
  EXPECT_NE(encode_result1, encode_result2);
}

// Tests that specifying a level 22 works.
TEST(ZstdCompressorTest, Level22) {
  auto compressor =
      Compressor::FromJson({{"type", "zstd"}, {"level", 22}}).value();
  const absl::Cord input = GetInput();
  absl::Cord encode_result;
  TENSORSTORE_ASSERT_OK(compressor->Encode(input, &encode_result, 1));
  absl::Cord decode_result;
  TENSORSTORE_ASSERT_OK(compressor->Decode(encode_result, &decode_result, 1));
  EXPECT_EQ(input, decode_result);
}

// Tests that an invalid parameter gives an error.
TEST(ZstdCompressorTest, InvalidParameter) {
  EXPECT_THAT(Compressor::FromJson({{"type", "zstd"}, {"level", "6"}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error parsing object member \"level\": .*"));
  EXPECT_THAT(Compressor::FromJson({{"type", "zstd"}, {"level", -131073}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error parsing object member \"level\": .*"));
  EXPECT_THAT(Compressor::FromJson({{"type", "zstd"}, {"level", 23}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error parsing object member \"level\": .*"));
  EXPECT_THAT(Compressor::FromJson({{"type", "zstd"}, {"foo", 10}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Object includes extra members: \"foo\""));
}

TEST(ZstdCompressorTest, ToJson) {
  auto compressor =
      Compressor::FromJson({{"type", "zstd"}, {"level", 5}}).value();
  EXPECT_EQ(nlohmann::json({{"type", "zstd"}, {"level", 5}}),
            compressor.ToJson());
}

}  // namespace
