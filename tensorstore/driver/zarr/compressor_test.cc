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

#include "tensorstore/driver/zarr/compressor.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using tensorstore::MatchesStatus;
using tensorstore::Status;
using tensorstore::internal_zarr::Compressor;

TEST(ParseCompressorTest, Null) {
  auto compressor_result = Compressor::FromJson(nullptr);
  EXPECT_EQ(Status(), GetStatus(compressor_result));
  EXPECT_EQ(nullptr, ::nlohmann::json(*compressor_result));
}

TEST(ParseCompressorTest, ZlibSuccess) {
  auto compressor_result = Compressor::FromJson({{"id", "zlib"}, {"level", 5}});
  EXPECT_EQ(Status(), GetStatus(compressor_result));
  EXPECT_EQ((::nlohmann::json{{"id", "zlib"}, {"level", 5}}),
            ::nlohmann::json(*compressor_result));
}

TEST(ParseCompressorTest, ZlibFailure) {
  EXPECT_THAT(
      Compressor::FromJson(::nlohmann::json{{"id", "zlib"}, {"level", "a"}}),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Error parsing object member \"level\": .*"));
}

TEST(ParseCompressorTest, UnsupportedId) {
  EXPECT_THAT(
      Compressor::FromJson(::nlohmann::json{{"id", "invalid"}, {"level", "a"}}),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Error parsing object member \"id\": "
                    "\"invalid\" is not registered"));
}

TEST(ParseCompressorTest, InvalidId) {
  EXPECT_THAT(Compressor::FromJson(::nlohmann::json{{"id", 5}, {"level", "a"}}),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Error parsing object member \"id\": "
                            "Expected string, but received: 5"));
}

}  // namespace
