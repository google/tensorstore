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

#include <stdint.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/data_type.h"
#include "tensorstore/driver/zarr3/codec/codec_chain_spec.h"
#include "tensorstore/driver/zarr3/codec/codec_spec.h"
#include "tensorstore/driver/zarr3/codec/codec_test_util.h"
#include "tensorstore/internal/json_gtest.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::dtype_v;
using ::tensorstore::MatchesJson;
using ::tensorstore::MatchesStatus;
using ::tensorstore::internal_zarr3::ArrayCodecResolveParameters;
using ::tensorstore::internal_zarr3::CodecRoundTripTestParams;
using ::tensorstore::internal_zarr3::CodecSpecRoundTripTestParams;
using ::tensorstore::internal_zarr3::GetDefaultBytesCodecJson;
using ::tensorstore::internal_zarr3::TestCodecRoundTrip;
using ::tensorstore::internal_zarr3::TestCodecSpecRoundTrip;
using ::tensorstore::internal_zarr3::ZarrCodecChainSpec;

TEST(BytesTest, SpecRoundTrip) {
  CodecSpecRoundTripTestParams p;
  p.orig_spec = {"bytes"};
  p.expected_spec = ::nlohmann::json::array_t{GetDefaultBytesCodecJson()};
  TestCodecSpecRoundTrip(p);
}

TEST(BytesTest, DuplicateArrayToBytes) {
  EXPECT_THAT(
      ZarrCodecChainSpec::FromJson({
          {{"name", "bytes"}, {"configuration", {{"endian", "little"}}}},
          {{"name", "bytes"}, {"configuration", {{"endian", "little"}}}},
      }),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Expected bytes -> bytes codec, but received: .*"));
}

TEST(BytesTest, RoundTrip) {
  CodecRoundTripTestParams p;
  p.spec = {"bytes"};
  TestCodecRoundTrip(p);
}

TEST(BytesTest, AutomaticTranspose) {
  ArrayCodecResolveParameters p;
  p.dtype = dtype_v<uint16_t>;
  p.rank = 2;
  auto& inner_order = p.inner_order.emplace();
  inner_order[0] = 1;
  inner_order[1] = 0;
  EXPECT_THAT(
      TestCodecSpecResolve(
          ::nlohmann::json::array_t{GetDefaultBytesCodecJson()}, p),
      ::testing::Optional(MatchesJson({
          {{"name", "transpose"}, {"configuration", {{"order", {1, 0}}}}},
          GetDefaultBytesCodecJson(),
      })));
}

TEST(BytesTest, EndianInvariantDataType) {
  ArrayCodecResolveParameters p;
  p.dtype = dtype_v<uint8_t>;
  p.rank = 2;
  EXPECT_THAT(
      TestCodecSpecResolve(::nlohmann::json::array_t{{{"name", "bytes"}}}, p,
                           /*constraints=*/false),
      ::testing::Optional(
          MatchesJson(::nlohmann::json::array_t{{{"name", "bytes"}}})));
}

TEST(BytesTest, MissingEndianEndianInvariantDataType) {
  ArrayCodecResolveParameters p;
  p.dtype = dtype_v<uint16_t>;
  p.rank = 2;
  EXPECT_THAT(
      TestCodecSpecResolve(::nlohmann::json::array_t{{{"name", "bytes"}}}, p,
                           /*constraints=*/false),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    ".*: \"bytes\" codec requires that \"endian\" option is "
                    "specified for data type uint16"));
}

}  // namespace
