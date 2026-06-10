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

#include <array>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorstore/array.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/zarr3/codec/codec_chain_spec.h"
#include "tensorstore/driver/zarr3/codec/codec_spec.h"
#include "tensorstore/driver/zarr3/codec/codec_test_util.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/testing/json_gtest.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::AllocateArray;
using ::tensorstore::c_order;
using ::tensorstore::dtype_v;
using ::tensorstore::Index;
using ::tensorstore::MatchesJson;
using ::tensorstore::StatusIs;
using ::tensorstore::value_init;
using ::tensorstore::internal_zarr3::ArrayCodecChunkLayoutInfo;
using ::tensorstore::internal_zarr3::ArrayCodecResolveParameters;
using ::tensorstore::internal_zarr3::ArrayDataTypeAndShapeInfo;
using ::tensorstore::internal_zarr3::CodecRoundTripTestParams;
using ::tensorstore::internal_zarr3::CodecSpecRoundTripTestParams;
using ::tensorstore::internal_zarr3::GetDefaultBytesCodecJson;
using ::tensorstore::internal_zarr3::TestCodecMerge;
using ::tensorstore::internal_zarr3::TestCodecRoundTrip;
using ::tensorstore::internal_zarr3::TestCodecSpecResolve;
using ::tensorstore::internal_zarr3::TestCodecSpecRoundTrip;
using ::tensorstore::internal_zarr3::ZarrCodecChainSpec;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::MatchesRegex;

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
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("is not a valid permutation")));
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
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          MatchesRegex("Error resolving codec spec .* is not a valid dimension "
                       "permutation for a rank 2 array")));
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
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("\"extra\"")));
}

// `inner_shape` (dtype-contributed trailing dims, e.g. for `open_as_void`)
// must be forwarded unchanged through every transpose propagation entry
// point: `PropagateDataTypeAndShape`, `GetDecodedChunkLayout`, and `Resolve`.
TEST(TransposeTest, PropagateDataTypeAndShapeForwardsInnerShape) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto chain_spec,
      ZarrCodecChainSpec::FromJson(
          {{{"name", "transpose"}, {"configuration", {{"order", {2, 0, 1}}}}},
           GetDefaultBytesCodecJson()}));
  ASSERT_EQ(chain_spec.array_to_array.size(), 1);

  ArrayDataTypeAndShapeInfo decoded;
  decoded.dtype = dtype_v<uint16_t>;
  decoded.rank = 3;
  decoded.shape = std::array<Index, tensorstore::kMaxRank>{};
  (*decoded.shape)[0] = 5;
  (*decoded.shape)[1] = 6;
  (*decoded.shape)[2] = 7;
  decoded.inner_shape = {4, 2};

  ArrayDataTypeAndShapeInfo encoded;
  TENSORSTORE_ASSERT_OK(chain_spec.array_to_array[0]->PropagateDataTypeAndShape(
      decoded, encoded));

  EXPECT_EQ(encoded.rank, decoded.rank);
  EXPECT_EQ(encoded.dtype, decoded.dtype);
  EXPECT_THAT(encoded.inner_shape, ElementsAre(4, 2));
  ASSERT_TRUE(encoded.shape.has_value());
  // order = {2, 0, 1}; only the leading `rank` dims are permuted.
  EXPECT_EQ((*encoded.shape)[0], (*decoded.shape)[2]);
  EXPECT_EQ((*encoded.shape)[1], (*decoded.shape)[0]);
  EXPECT_EQ((*encoded.shape)[2], (*decoded.shape)[1]);
}

// `GetDecodedChunkLayout` must accept a non-empty `inner_shape` on the
// array_info and produce the inverse-permutation `inner_order` over the
// chunked dims (the inner dims do not participate).
TEST(TransposeTest, GetDecodedChunkLayoutWithInnerShape) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto chain_spec,
      ZarrCodecChainSpec::FromJson(
          {{{"name", "transpose"}, {"configuration", {{"order", {2, 0, 1}}}}},
           GetDefaultBytesCodecJson()}));

  ArrayDataTypeAndShapeInfo array_info;
  array_info.dtype = dtype_v<uint16_t>;
  array_info.rank = 3;
  array_info.inner_shape = {4, 2};

  ArrayCodecChunkLayoutInfo decoded;
  TENSORSTORE_ASSERT_OK(chain_spec.GetDecodedChunkLayout(array_info, decoded));

  ASSERT_TRUE(decoded.inner_order.has_value());
  // Bytes codec asks for c_order = {0, 1, 2} on the encoded side; transpose
  // maps decoded[i] = order[encoded[i]] with order = {2, 0, 1}.
  EXPECT_EQ((*decoded.inner_order)[0], 2);
  EXPECT_EQ((*decoded.inner_order)[1], 0);
  EXPECT_EQ((*decoded.inner_order)[2], 1);
}

TEST(TransposeTest, ResolveForwardsInnerShape) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto chain_spec,
      ZarrCodecChainSpec::FromJson(
          {{{"name", "transpose"}, {"configuration", {{"order", {2, 0, 1}}}}},
           GetDefaultBytesCodecJson()}));
  ASSERT_EQ(chain_spec.array_to_array.size(), 1);

  ArrayCodecResolveParameters decoded;
  decoded.dtype = dtype_v<uint16_t>;
  decoded.rank = 3;
  decoded.fill_value = AllocateArray(tensorstore::span<const Index>{}, c_order,
                                     value_init, decoded.dtype);
  decoded.inner_shape = {4, 2};

  ArrayCodecResolveParameters encoded;
  TENSORSTORE_ASSERT_OK(chain_spec.array_to_array[0]->Resolve(
      std::move(decoded), encoded, /*resolved_spec=*/nullptr));

  EXPECT_EQ(encoded.rank, 3);
  EXPECT_THAT(encoded.inner_shape, ElementsAre(4, 2));
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
              StatusIs(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(TestCodecMerge(perm_C, perm_F, /*strict=*/false),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

}  // namespace
