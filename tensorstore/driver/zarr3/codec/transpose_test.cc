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

#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
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
using ::tensorstore::internal_zarr3::TestCodecMerge;
using ::tensorstore::internal_zarr3::TestCodecRoundTrip;
using ::tensorstore::internal_zarr3::TestCodecSpecResolve;
using ::tensorstore::internal_zarr3::TestCodecSpecRoundTrip;
using ::tensorstore::internal_zarr3::ZarrCodecChainSpec;

TEST(TransposeTest, Basic) {
  CodecSpecRoundTripTestParams p;
  p.orig_spec = {
      {{"name", "transpose"}, {"configuration", {{"order", {2, 1, 0}}}}},
  };
  p.resolve_params.rank = 3;
  p.expected_spec = {
      {{"name", "transpose"}, {"configuration", {{"order", {2, 1, 0}}}}},
      GetDefaultBytesCodecJson(),
  };
  TestCodecSpecRoundTrip(p);
}

TEST(TransposeTest, C) {
  CodecSpecRoundTripTestParams p;
  p.orig_spec = {
      {{"name", "transpose"}, {"configuration", {{"order", "C"}}}},
  };
  p.resolve_params.rank = 3;
  p.expected_spec = {
      {{"name", "transpose"}, {"configuration", {{"order", {0, 1, 2}}}}},
      GetDefaultBytesCodecJson(),
  };
  TestCodecSpecRoundTrip(p);
}

TEST(TransposeTest, F) {
  CodecSpecRoundTripTestParams p;
  p.orig_spec = {
      {{"name", "transpose"}, {"configuration", {{"order", "F"}}}},
  };
  p.resolve_params.rank = 3;
  p.expected_spec = {
      {{"name", "transpose"}, {"configuration", {{"order", {2, 1, 0}}}}},
      GetDefaultBytesCodecJson(),
  };
  TestCodecSpecRoundTrip(p);
}

TEST(TransposeTest, InvalidPermutation) {
  EXPECT_THAT(
      ZarrCodecChainSpec::FromJson(
          {{{"name", "transpose"}, {"configuration", {{"order", {2, 1, 2}}}}}}),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    ".*is not a valid permutation.*"));
}

TEST(TransposeTest, RoundTrip) {
  CodecRoundTripTestParams p;
  p.spec = {{{"name", "transpose"}, {"configuration", {{"order", {2, 1, 0}}}}}};
  TestCodecRoundTrip(p);
}

TEST(TransposeTest, RankMismatch) {
  ArrayCodecResolveParameters p;
  p.dtype = dtype_v<uint16_t>;
  p.rank = 2;
  EXPECT_THAT(
      TestCodecSpecResolve(
          {{{"name", "transpose"}, {"configuration", {{"order", {2, 1, 0}}}}}},
          p),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Error resolving codec spec .* is not a valid dimension "
                    "permutation for a rank 2 array"));
}

TEST(TransposeTest, AttributeMismatch) {
  ArrayCodecResolveParameters p;
  p.dtype = dtype_v<uint16_t>;
  p.rank = 2;
  EXPECT_THAT(
      TestCodecSpecResolve(
          {{{"name", "transpose"},
            {"configuration", {{"order", {0, 1}}, {"extra", 1}}}}},
          p),
      MatchesStatus(absl::StatusCode::kInvalidArgument, ".* \"extra\""));
}

TEST(TransposeTest, Merge) {
  ::nlohmann::json perm_012 = {
      {{"name", "transpose"}, {"configuration", {{"order", {0, 1, 2}}}}}};
  ::nlohmann::json perm_210 = {
      {{"name", "transpose"}, {"configuration", {{"order", {2, 1, 0}}}}}};
  ::nlohmann::json perm_C = {
      {{"name", "transpose"}, {"configuration", {{"order", "C"}}}}};
  ::nlohmann::json perm_F = {
      {{"name", "transpose"}, {"configuration", {{"order", "F"}}}}};
  EXPECT_THAT(TestCodecMerge(perm_012, perm_C,
                             /*strict=*/false),
              ::testing::Optional(MatchesJson(perm_012)));
  EXPECT_THAT(TestCodecMerge(perm_C, perm_012,
                             /*strict=*/false),
              ::testing::Optional(MatchesJson(perm_012)));
  EXPECT_THAT(TestCodecMerge(perm_210, perm_F,
                             /*strict=*/false),
              ::testing::Optional(MatchesJson(perm_210)));
  EXPECT_THAT(TestCodecMerge(perm_F, perm_210,
                             /*strict=*/false),
              ::testing::Optional(MatchesJson(perm_210)));
  EXPECT_THAT(TestCodecMerge(perm_C, perm_C,
                             /*strict=*/false),
              ::testing::Optional(MatchesJson(perm_C)));
  EXPECT_THAT(TestCodecMerge(perm_F, perm_F,
                             /*strict=*/false),
              ::testing::Optional(MatchesJson(perm_F)));
  EXPECT_THAT(TestCodecMerge(perm_012, perm_210, /*strict=*/false),
              MatchesStatus(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(TestCodecMerge(perm_C, perm_F, /*strict=*/false),
              MatchesStatus(absl::StatusCode::kFailedPrecondition));
}

}  // namespace
