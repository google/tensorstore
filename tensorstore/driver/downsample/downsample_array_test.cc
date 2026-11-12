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

#include "tensorstore/driver/downsample/downsample_array.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include <nlohmann/json.hpp>
#include "tensorstore/array.h"
#include "tensorstore/array_testutil.h"
#include "tensorstore/box.h"
#include "tensorstore/data_type.h"
#include "tensorstore/downsample_method.h"
#include "tensorstore/driver/downsample/downsample_nditerable.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/transformed_array.h"
#include "tensorstore/internal/arena.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/internal/nditerable.h"
#include "tensorstore/internal/nditerable_copy.h"
#include "tensorstore/internal/nditerable_transformed_array.h"
#include "tensorstore/internal/nditerable_util.h"
#include "tensorstore/internal/unique_with_intrusive_allocator.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/iterate_over_index_range.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::AllocateArray;
using ::tensorstore::Array;
using ::tensorstore::ArrayView;
using ::tensorstore::DataType;
using ::tensorstore::Dims;
using ::tensorstore::DownsampleMethod;
using ::tensorstore::dtype_v;
using ::tensorstore::EqualityComparisonKind;
using ::tensorstore::Index;
using ::tensorstore::kImplicit;
using ::tensorstore::MakeArray;
using ::tensorstore::MakeOffsetArray;
using ::tensorstore::MatchesArray;
using ::tensorstore::MatchesArrayIdentically;
using ::tensorstore::SharedOffsetArray;
using ::tensorstore::span;
using ::tensorstore::StatusIs;
using ::tensorstore::StridedLayout;
using ::tensorstore::TransformedArray;
using ::tensorstore::dtypes::bfloat16_t;
using ::tensorstore::dtypes::byte_t;
using ::tensorstore::dtypes::char_t;
using ::tensorstore::dtypes::complex128_t;
using ::tensorstore::dtypes::complex64_t;
using ::tensorstore::dtypes::float16_t;
using ::tensorstore::dtypes::float32_t;
using ::tensorstore::dtypes::float4_e2m1fn_t;
using ::tensorstore::dtypes::float64_t;
using ::tensorstore::dtypes::float8_e3m4_t;
using ::tensorstore::dtypes::float8_e4m3b11fnuz_t;
using ::tensorstore::dtypes::float8_e4m3fn_t;
using ::tensorstore::dtypes::float8_e4m3fnuz_t;
using ::tensorstore::dtypes::float8_e5m2_t;
using ::tensorstore::dtypes::float8_e5m2fnuz_t;
using ::tensorstore::dtypes::float8_e8m0fnu_t;
using ::tensorstore::dtypes::int2_t;
using ::tensorstore::dtypes::int4_t;
using ::tensorstore::dtypes::json_t;
using ::tensorstore::dtypes::string_t;
using ::tensorstore::dtypes::ustring_t;
using ::tensorstore::internal_downsample::DownsampleArray;
using ::tensorstore::internal_downsample::DownsampleNDIterable;
using ::tensorstore::internal_downsample::DownsampleTransformedArray;
using ::tensorstore::internal_downsample::IsDownsampleMethodSupported;
using ::testing::Optional;

#if 0
// Matches a 1-element `complex64_t` `SharedOffsetArray<void>` whose element has
// a NaN real or imaginary component.
MATCHER(HasSingleComplex64NaN, "") {
  if (arg.dtype() != dtype_v<complex64_t> || arg.num_elements() != 1) {
    *result_listener << "dtype=" << arg.dtype()
                     << ", num_elements=" << arg.num_elements();
    return false;
  }
  const auto& z = *static_cast<const complex64_t*>(arg.data());
  if (!std::isnan(z.real()) && !std::isnan(z.imag())) {
    *result_listener << "value=(" << z.real() << ", " << z.imag() << ")";
    return false;
  }
  return true;
}
#endif

TEST(DownsampleArrayTest, MeanRank0) {
  EXPECT_THAT(DownsampleArray(tensorstore::MakeScalarArray<float>(42.0),
                              span<const Index>(), DownsampleMethod::kMean),
              Optional(tensorstore::MakeScalarArray<float>(42.0)));
}

// Test case where original size is exact multiple of downsample factor.
TEST(DownsampleArrayTest, MeanRank1ExactMultiple) {
  EXPECT_THAT(DownsampleArray(MakeArray<float>({1, 2, 5, 7}),
                              span<const Index>({2}), DownsampleMethod::kMean),
              Optional(MakeArray<float>({1.5, 6})));
  EXPECT_THAT(DownsampleArray(MakeArray<float>({1, 2, 3, 5, 7, 12}),
                              span<const Index>({3}), DownsampleMethod::kMean),
              Optional(MakeArray<float>({2, 8})));
}

TEST(DownsampleArrayTest, MeanRoundingUint8) {
  EXPECT_THAT(DownsampleArray(MakeArray<uint8_t>({253, 254, 254}),
                              span<const Index>({3}), DownsampleMethod::kMean),
              Optional(MakeArray<uint8_t>({254})));
}

TEST(DownsampleArrayTest, MeanRoundingInt16) {
  EXPECT_THAT(DownsampleArray(MakeArray<int16_t>({-253, -254, -254}),
                              span<const Index>({3}), DownsampleMethod::kMean),
              Optional(MakeArray<int16_t>({-254})));
}

TEST(DownsampleArrayTest, MeanRoundingToEvenInt16) {
  EXPECT_THAT(DownsampleArray(MakeArray<int16_t>({3, 3, 2, 2}),
                              span<const Index>({4}), DownsampleMethod::kMean),
              Optional(MakeArray<int16_t>({2})));
  EXPECT_THAT(DownsampleArray(MakeArray<int16_t>({3, 3, 4, 4}),
                              span<const Index>({4}), DownsampleMethod::kMean),
              Optional(MakeArray<int16_t>({4})));
  EXPECT_THAT(DownsampleArray(MakeArray<int16_t>({-3, -3, -2, -2}),
                              span<const Index>({4}), DownsampleMethod::kMean),
              Optional(MakeArray<int16_t>({-2})));
  EXPECT_THAT(DownsampleArray(MakeArray<int16_t>({-3, -3, -4, -4}),
                              span<const Index>({4}), DownsampleMethod::kMean),
              Optional(MakeArray<int16_t>({-4})));
}

TEST(DownsampleArrayTest, MeanRoundingUint64) {
  EXPECT_THAT(DownsampleArray(MakeArray<uint64_t>({253, 254, 254}),
                              span<const Index>({3}), DownsampleMethod::kMean),
              Optional(MakeArray<uint64_t>({254})));
}

TEST(DownsampleArrayTest, MeanRoundingBool) {
  EXPECT_THAT(DownsampleArray(MakeArray<bool>({0, 0, 1}),
                              span<const Index>({3}), DownsampleMethod::kMean),
              Optional(MakeArray<bool>({0})));
  EXPECT_THAT(DownsampleArray(MakeArray<bool>({0, 1, 1}),
                              span<const Index>({3}), DownsampleMethod::kMean),
              Optional(MakeArray<bool>({1})));
  EXPECT_THAT(DownsampleArray(MakeArray<bool>({0, 1, 1, 0}),
                              span<const Index>({4}), DownsampleMethod::kMean),
              Optional(MakeArray<bool>({0})));
}

TEST(DownsampleArrayTest, MeanRank1Offset) {
  EXPECT_THAT(DownsampleArray(MakeOffsetArray<float>({1}, {1, 2, 5, 9}),
                              span<const Index>({2}), DownsampleMethod::kMean),
              Optional(MakeArray<float>({1, 3.5, 9})));
}

// Test case where original size is exact multiple of downsample factor.
TEST(DownsampleArrayTest, MeanRank1SingleDownsampledElement) {
  EXPECT_THAT(DownsampleArray(MakeArray<float>({1, 2}), span<const Index>({2}),
                              DownsampleMethod::kMean),
              Optional(MakeArray<float>({1.5})));
}

// Test case where original size is not exact multiple of downsample factor.
TEST(DownsampleArrayTest, MeanRank1NotExactMultiple) {
  EXPECT_THAT(DownsampleArray(MakeArray<float>({1, 2, 5, 7, 9}),
                              span<const Index>({2}), DownsampleMethod::kMean),
              Optional(MakeArray<float>({1.5, 6, 9})));
  EXPECT_THAT(DownsampleArray(MakeArray<float>({1, 2, 6, 7, 9}),
                              span<const Index>({3}), DownsampleMethod::kMean),
              Optional(MakeArray<float>({3, 8})));
}

TEST(DownsampleArrayTest, MeanRank1NoDownsampling) {
  EXPECT_THAT(DownsampleArray(MakeArray<float>({1, 2, 5, 7}),
                              span<const Index>({1}), DownsampleMethod::kMean),
              Optional(MakeArray<float>({1, 2, 5, 7})));
}

TEST(DownsampleArrayTest, MeanRank2SingleDownsampleDim1) {
  EXPECT_THAT(
      DownsampleArray(MakeArray<float>({
                          {1, 2, 5, 7},
                          {5, 6, 15, 25},
                      }),
                      span<const Index>({1, 2}), DownsampleMethod::kMean),
      Optional(MakeArray<float>({{1.5, 6}, {5.5, 20}})));
}

TEST(DownsampleArrayTest, MeanRank2SingleDownsampleDim0) {
  EXPECT_THAT(
      DownsampleArray(MakeArray<float>({
                          {1, 2, 5, 7},
                          {5, 6, 15, 25},
                      }),
                      span<const Index>({2, 1}), DownsampleMethod::kMean),
      Optional(MakeArray<float>({{3, 4, 10, 16}})));
}

TEST(DownsampleArrayTest, MeanRank2TwoDownsampleDims) {
  EXPECT_THAT(
      DownsampleArray(MakeArray<float>({
                          {1, 2, 5, 7},
                          {5, 6, 15, 25},
                      }),
                      span<const Index>({2, 2}), DownsampleMethod::kMean),
      Optional(MakeArray<float>({{3.5, 13.0}})));
}

TEST(DownsampleArrayTest, MeanRank2NotExactMultiple) {
  EXPECT_THAT(
      DownsampleArray(MakeArray<float>({
                          {1, 2, 3, 4, 5},
                          {6, 7, 8, 9, 10},
                          {11, 12, 13, 14, 15},
                      }),
                      span<const Index>({2, 2}), DownsampleMethod::kMean),
      Optional(MakeArray<float>({
          {4, 6, 7.5},
          {11.5, 13.5, 15},
      })));
}

TEST(DownsampleArrayTest, MeanRank2PartialStartBlock) {
  EXPECT_THAT(
      DownsampleArray(MakeOffsetArray<float>({3, 8}, {{1, 2, 3, 4, 5},
                                                      {6, 7, 8, 9, 10},
                                                      {11, 12, 13, 14, 15}}),
                      span<const Index>({2, 3}), DownsampleMethod::kMean),
      Optional(MakeOffsetArray<float>({1, 2}, {{1, 3, 5}, {8.5, 10.5, 12.5}})));
}

TEST(DownsampleArrayTest, MedianRank2PartialStartBlock) {
  EXPECT_THAT(
      DownsampleArray(MakeOffsetArray<float>({3, 8}, {{1, 2, 3, 4, 5},
                                                      {6, 7, 8, 9, 10},
                                                      {11, 12, 13, 14, 15}}),
                      span<const Index>({2, 3}), DownsampleMethod::kMedian),
      Optional(MakeOffsetArray<float>({1, 2}, {{1, 3, 5}, {6, 9, 10}})));
}

TEST(DownsampleArrayTest, ModeRank2PartialStartBlock) {
  EXPECT_THAT(
      DownsampleArray(MakeOffsetArray<float>({3, 8},
                                             {
                                                 {1, 2, 3, 3, 5},
                                                 {6, 4, 5, 5, 10},
                                                 {11, 6, 6, 6, 15},
                                             }),
                      span<const Index>({2, 3}), DownsampleMethod::kMode),
      Optional(MakeOffsetArray<float>({1, 2}, {{1, 3, 5}, {6, 6, 10}})));
}

TEST(DownsampleArrayTest, StrideRank2PartialEndBlock) {
  EXPECT_THAT(
      DownsampleArray(MakeOffsetArray<float>({2, 6},
                                             {
                                                 {1, 2, 3, 4, 5},
                                                 {6, 7, 8, 9, 10},
                                                 {11, 12, 13, 14, 15},
                                             }),
                      span<const Index>({2, 3}), DownsampleMethod::kStride),
      Optional(MakeOffsetArray<float>({1, 2}, {
                                                  {1, 4},
                                                  {11, 14},
                                              })));
}

TEST(DownsampleArrayTest, StrideRank2PartialStartBlock) {
  EXPECT_THAT(
      DownsampleArray(MakeOffsetArray<float>({3, 8},
                                             {
                                                 {1, 2, 3, 4, 5},
                                                 {6, 7, 8, 9, 10},
                                                 {11, 12, 13, 14, 15},
                                             }),
                      span<const Index>({2, 3}), DownsampleMethod::kStride),
      Optional(MakeOffsetArray<float>({2, 3}, {
                                                  {7, 10},
                                              })));
}

TEST(DownsampleArrayTest, MeanRank3ThreeDownsampleDims) {
  EXPECT_THAT(
      DownsampleArray(MakeArray<float>({{
                                            {1, 2, 3, 4},
                                            {5, 6, 7, 8},
                                            {9, 10, 11, 12},
                                        },
                                        {
                                            {13, 14, 15, 16},
                                            {17, 18, 19, 20},
                                            {21, 22, 23, 24},
                                        },
                                        {
                                            {25, 26, 27, 28},
                                            {29, 30, 31, 32},
                                            {33, 34, 35, 36},
                                        }}),
                      span<const Index>({2, 2, 2}), DownsampleMethod::kMean),
      Optional(MakeArray<float>({{
                                     {9.5, 11.5},
                                     {15.5, 17.5},
                                 },
                                 {
                                     {27.5, 29.5},
                                     {33.5, 35.5},
                                 }})));
}

TEST(DownsampleArrayTest, MeanRank1ReversedExactMultiple) {
  EXPECT_THAT(DownsampleTransformedArray(
                  (MakeArray<float>({1, 2, 3, 4}) |
                   Dims(0).TranslateSizedInterval(kImplicit, kImplicit, -1))
                      .value(),
                  span<const Index>({2}), DownsampleMethod::kMean),
              Optional(MakeArray<float>({3.5, 1.5})));
}

TEST(DownsampleArrayTest, MeanRank1ReversedNotExactMultiple) {
  EXPECT_THAT(DownsampleTransformedArray(
                  (MakeArray<float>({1, 2, 3, 4, 5}) |
                   Dims(0).TranslateSizedInterval(kImplicit, kImplicit, -1))
                      .value(),
                  span<const Index>({2}), DownsampleMethod::kMean),
              Optional(MakeArray<float>({4.5, 2.5, 1})));
}

TEST(DownsampleArrayTest, MeanRank2ReversedNotExactMultiple) {
  EXPECT_THAT(DownsampleTransformedArray(
                  (MakeArray<float>({
                       {1, 2, 3, 4, 5},
                       {6, 7, 8, 9, 10},
                       {11, 12, 13, 14, 15},
                   }) |
                   Dims(0, 1).TranslateSizedInterval(kImplicit, kImplicit, -1))
                      .value(),
                  span<const Index>({2, 2}), DownsampleMethod::kMean),
              Optional(MakeArray<float>({
                  {12, 10, 8.5},
                  {4.5, 2.5, 1},
              })));
}

TEST(DownsampleArrayTest, MinRank1ExactMultiple) {
  EXPECT_THAT(DownsampleArray(MakeArray<float>({2, 3, 5, 1}),
                              span<const Index>({2}), DownsampleMethod::kMin),
              Optional(MakeArray<float>({2, 1})));
  EXPECT_THAT(DownsampleArray(MakeArray<int>({2, 3, 8, 7, 1, 5}),
                              span<const Index>({3}), DownsampleMethod::kMin),
              Optional(MakeArray<int>({2, 1})));
}

TEST(DownsampleArrayTest, MaxRank1ExactMultiple) {
  EXPECT_THAT(DownsampleArray(MakeArray<float>({2, 3, 5, 1}),
                              span<const Index>({2}), DownsampleMethod::kMax),
              Optional(MakeArray<float>({3, 5})));
  EXPECT_THAT(DownsampleArray(MakeArray<int>({2, 3, 8, 7, 1, 5}),
                              span<const Index>({3}), DownsampleMethod::kMax),
              Optional(MakeArray<int>({8, 7})));
}

TEST(DownsampleArrayTest, MedianRank1ExactMultiple) {
  EXPECT_THAT(
      DownsampleArray(MakeArray<float>({100, 3, 1, 2, 99, 98, 97, 5}),
                      span<const Index>({4}), DownsampleMethod::kMedian),
      Optional(MakeArray<float>({2, 97})));
}

TEST(DownsampleArrayTest, MedianRank1Partial) {
  EXPECT_THAT(
      DownsampleArray(MakeArray<float>({100, 3, 1, 2, 99, 97, 98}),
                      span<const Index>({4}), DownsampleMethod::kMedian),
      Optional(MakeArray<float>({2, 98})));
}

TEST(DownsampleArrayTest, ModeRank1ExactMultiple) {
  EXPECT_THAT(DownsampleArray(MakeArray<float>({100, 99, 99, 99, 3, 3, 2, 2}),
                              span<const Index>({4}), DownsampleMethod::kMode),
              Optional(MakeArray<float>({99, 2})));
}

TEST(DownsampleArrayTest, ModeRank1Partial) {
  EXPECT_THAT(DownsampleArray(MakeArray<float>({100, 99, 99, 99, 3, 3, 2}),
                              span<const Index>({4}), DownsampleMethod::kMode),
              Optional(MakeArray<float>({99, 3})));
}

TEST(DownsampleArrayTest, ModeBool) {
  EXPECT_THAT(DownsampleArray(MakeArray<bool>({0, 0, 1, 1}),
                              span<const Index>({4}), DownsampleMethod::kMode),
              Optional(MakeArray<bool>({0})));
  EXPECT_THAT(DownsampleArray(MakeArray<bool>({0, 1, 1, 1}),
                              span<const Index>({4}), DownsampleMethod::kMode),
              Optional(MakeArray<bool>({1})));
  EXPECT_THAT(DownsampleArray(MakeArray<bool>({0, 0, 1, 1, 1}),
                              span<const Index>({5}), DownsampleMethod::kMode),
              Optional(MakeArray<bool>({1})));
}

TEST(DownsampleArrayTest, MeanBool) {
  EXPECT_THAT(DownsampleArray(MakeArray<bool>({0, 0, 1, 1}),
                              span<const Index>({4}), DownsampleMethod::kMean),
              Optional(MakeArray<bool>({0})));
  EXPECT_THAT(DownsampleArray(MakeArray<bool>({0, 1, 1, 1}),
                              span<const Index>({4}), DownsampleMethod::kMean),
              Optional(MakeArray<bool>({1})));
  EXPECT_THAT(DownsampleArray(MakeArray<bool>({0, 0, 1, 1, 1}),
                              span<const Index>({5}), DownsampleMethod::kMean),
              Optional(MakeArray<bool>({1})));
}

TEST(DownsampleArrayTest, MedianBool) {
  EXPECT_THAT(
      DownsampleArray(MakeArray<bool>({0, 0, 1, 1}), span<const Index>({4}),
                      DownsampleMethod::kMedian),
      Optional(MakeArray<bool>({0})));
  EXPECT_THAT(
      DownsampleArray(MakeArray<bool>({0, 1, 1, 1}), span<const Index>({4}),
                      DownsampleMethod::kMedian),
      Optional(MakeArray<bool>({1})));
  EXPECT_THAT(
      DownsampleArray(MakeArray<bool>({0, 0, 1, 1, 1}), span<const Index>({5}),
                      DownsampleMethod::kMedian),
      Optional(MakeArray<bool>({1})));
}

TEST(DownsampleArrayTest, ModeJson) {
  using ::tensorstore::dtypes::json_t;
  EXPECT_THAT(DownsampleArray(MakeArray<json_t>({"a", "a", 3.0, 3, 3u}),
                              span<const Index>({5}), DownsampleMethod::kMode),
              Optional(MakeArray<::nlohmann::json>({json_t(3)})));
}

// Tests the downsampling behaves correctly when multiple blocks along
// downsampled dimensions are needed.
TEST(DownsampleArrayTest, MultipleBlocks) {
  auto source_array = tensorstore::AllocateArray<uint8_t>({128, 128});
  // For uint8_t, accumulation is done using uint64_t.  64 * 64 * 8 == 32 *
  // 1024, which exceeds the buffer size limit of `24 * 1024`.
  auto expected_downsampled = tensorstore::AllocateArray<uint8_t>({64, 64});
  for (int i = 0; i < 128; ++i) {
    for (int j = 0; j < 128; ++j) {
      source_array(i, j) = static_cast<uint8_t>(i);
    }
  }
  for (int i = 0; i < 64; ++i) {
    for (int j = 0; j < 64; ++j) {
      expected_downsampled(i, j) = static_cast<uint8_t>(i * 2);
    }
  }
  EXPECT_THAT(DownsampleArray(source_array, {{2, 2}}, DownsampleMethod::kMean),
              Optional(tensorstore::MatchesArray(expected_downsampled)));
}

TEST(DownsampleArrayTest, MeanLowPrecisionFloat) {
  using ::tensorstore::dtypes::bfloat16_t;
  using ::tensorstore::dtypes::float4_e2m1fn_t;
  using ::tensorstore::dtypes::float8_e4m3fn_t;
  EXPECT_THAT(
      DownsampleArray(
          MakeArray<bfloat16_t>({bfloat16_t(1.0f), bfloat16_t(2.0f),
                                 bfloat16_t(5.0f), bfloat16_t(7.0f)}),
          span<const Index>({2}), DownsampleMethod::kMean),
      Optional(MakeArray<bfloat16_t>({bfloat16_t(1.5f), bfloat16_t(6.0f)})));
  EXPECT_THAT(
      DownsampleArray(MakeArray<float8_e4m3fn_t>(
                          {float8_e4m3fn_t(1.0f), float8_e4m3fn_t(2.0f),
                           float8_e4m3fn_t(5.0f), float8_e4m3fn_t(7.0f)}),
                      span<const Index>({2}), DownsampleMethod::kMean),
      Optional(MakeArray<float8_e4m3fn_t>(
          {float8_e4m3fn_t(1.5f), float8_e4m3fn_t(6.0f)})));
  EXPECT_THAT(
      DownsampleArray(MakeArray<float4_e2m1fn_t>(
                          {float4_e2m1fn_t(1.0f), float4_e2m1fn_t(2.0f),
                           float4_e2m1fn_t(2.0f), float4_e2m1fn_t(6.0f)}),
                      span<const Index>({2}), DownsampleMethod::kMean),
      Optional(MakeArray<float4_e2m1fn_t>(
          {float4_e2m1fn_t(1.5f), float4_e2m1fn_t(4.0f)})));
}

TEST(DownsampleArrayTest, IsDownsampleMethodSupportedMatrix) {
  EXPECT_FALSE(
      IsDownsampleMethodSupported(DataType(), DownsampleMethod::kMean));
  EXPECT_TRUE(
      IsDownsampleMethodSupported(DataType(), DownsampleMethod::kStride));

  const DataType kOrderedNumericTypes[] = {
      dtype_v<bool>,
      dtype_v<int2_t>,
      dtype_v<int4_t>,
      dtype_v<int8_t>,
      dtype_v<uint8_t>,
      dtype_v<int16_t>,
      dtype_v<uint16_t>,
      dtype_v<int32_t>,
      dtype_v<uint32_t>,
      dtype_v<int64_t>,
      dtype_v<uint64_t>,
      dtype_v<float4_e2m1fn_t>,
      dtype_v<float8_e3m4_t>,
      dtype_v<float8_e4m3fn_t>,
      dtype_v<float8_e4m3fnuz_t>,
      dtype_v<float8_e4m3b11fnuz_t>,
      dtype_v<float8_e5m2_t>,
      dtype_v<float8_e5m2fnuz_t>,
      dtype_v<float8_e8m0fnu_t>,
      dtype_v<bfloat16_t>,
      dtype_v<float16_t>,
      dtype_v<float>,
      dtype_v<double>,
  };
  for (DataType dtype : kOrderedNumericTypes) {
    SCOPED_TRACE(dtype.name());
    EXPECT_TRUE(IsDownsampleMethodSupported(dtype, DownsampleMethod::kStride));
    EXPECT_TRUE(IsDownsampleMethodSupported(dtype, DownsampleMethod::kMean));
    EXPECT_TRUE(IsDownsampleMethodSupported(dtype, DownsampleMethod::kMin));
    EXPECT_TRUE(IsDownsampleMethodSupported(dtype, DownsampleMethod::kMax));
    EXPECT_TRUE(IsDownsampleMethodSupported(dtype, DownsampleMethod::kMedian));
    EXPECT_TRUE(IsDownsampleMethodSupported(dtype, DownsampleMethod::kMode));
  }

  const DataType kComplexTypes[] = {
      dtype_v<complex64_t>,
      dtype_v<complex128_t>,
  };
  for (DataType dtype : kComplexTypes) {
    SCOPED_TRACE(dtype.name());
    EXPECT_TRUE(IsDownsampleMethodSupported(dtype, DownsampleMethod::kStride));
    EXPECT_TRUE(IsDownsampleMethodSupported(dtype, DownsampleMethod::kMean));
    EXPECT_FALSE(IsDownsampleMethodSupported(dtype, DownsampleMethod::kMin));
    EXPECT_FALSE(IsDownsampleMethodSupported(dtype, DownsampleMethod::kMax));
    EXPECT_FALSE(IsDownsampleMethodSupported(dtype, DownsampleMethod::kMedian));
    EXPECT_TRUE(IsDownsampleMethodSupported(dtype, DownsampleMethod::kMode));
  }

  const DataType kModeOnlyTypes[] = {
      dtype_v<char_t>,    dtype_v<byte_t>, dtype_v<string_t>,
      dtype_v<ustring_t>, dtype_v<json_t>,
  };
  for (DataType dtype : kModeOnlyTypes) {
    SCOPED_TRACE(dtype.name());
    EXPECT_TRUE(IsDownsampleMethodSupported(dtype, DownsampleMethod::kStride));
    EXPECT_FALSE(IsDownsampleMethodSupported(dtype, DownsampleMethod::kMean));
    EXPECT_FALSE(IsDownsampleMethodSupported(dtype, DownsampleMethod::kMin));
    EXPECT_FALSE(IsDownsampleMethodSupported(dtype, DownsampleMethod::kMax));
    EXPECT_FALSE(IsDownsampleMethodSupported(dtype, DownsampleMethod::kMedian));
    EXPECT_TRUE(IsDownsampleMethodSupported(dtype, DownsampleMethod::kMode));
  }

  // Ensure every canonical `DataTypeId` is covered by the three tables above.
  EXPECT_EQ(std::size(kOrderedNumericTypes) + std::size(kComplexTypes) +
                std::size(kModeOnlyTypes),
            tensorstore::kNumDataTypeIds);
}

TEST(DownsampleArrayTest, Float8WithInfinityReductions) {
  const auto e3m4_in =
      MakeArray<float8_e3m4_t>({float8_e3m4_t{1.0f}, float8_e3m4_t{3.0f},
                                float8_e3m4_t{-2.0f}, float8_e3m4_t{-6.0f}});
  EXPECT_THAT(
      DownsampleArray(e3m4_in, span<const Index>({2}), DownsampleMethod::kMean),
      Optional(MatchesArray(MakeArray<float8_e3m4_t>(
          {float8_e3m4_t{2.0f}, float8_e3m4_t{-4.0f}}))));
  EXPECT_THAT(
      DownsampleArray(e3m4_in, span<const Index>({2}), DownsampleMethod::kMin),
      Optional(MatchesArray(MakeArray<float8_e3m4_t>(
          {float8_e3m4_t{1.0f}, float8_e3m4_t{-6.0f}}))));
  EXPECT_THAT(
      DownsampleArray(e3m4_in, span<const Index>({2}), DownsampleMethod::kMax),
      Optional(MatchesArray(MakeArray<float8_e3m4_t>(
          {float8_e3m4_t{3.0f}, float8_e3m4_t{-2.0f}}))));
  EXPECT_THAT(
      DownsampleArray(e3m4_in, span<const Index>({4}),
                      DownsampleMethod::kMedian),
      Optional(MatchesArray(MakeArray<float8_e3m4_t>({float8_e3m4_t{-2.0f}}))));
  EXPECT_THAT(
      DownsampleArray(
          MakeArray<float8_e3m4_t>({float8_e3m4_t{4.0f}, float8_e3m4_t{2.0f},
                                    float8_e3m4_t{4.0f}, float8_e3m4_t{2.0f}}),
          span<const Index>({4}), DownsampleMethod::kMode),
      Optional(MatchesArray(MakeArray<float8_e3m4_t>({float8_e3m4_t{2.0f}}))));
  EXPECT_THAT(
      DownsampleArray(
          MakeArray<float8_e5m2_t>({float8_e5m2_t{2.0f}, float8_e5m2_t{6.0f}}),
          span<const Index>({2}), DownsampleMethod::kMean),
      Optional(MatchesArray(MakeArray<float8_e5m2_t>({float8_e5m2_t{4.0f}}))));
}

TEST(DownsampleArrayTest, FiniteOnlyFloat8MeanMedianMode) {
  const auto f8_in = MakeArray<float8_e4m3fn_t>(
      {float8_e4m3fn_t{1.0f}, float8_e4m3fn_t{3.0f}, float8_e4m3fn_t{-2.0f},
       float8_e4m3fn_t{-6.0f}});
  EXPECT_THAT(
      DownsampleArray(f8_in, span<const Index>({2}), DownsampleMethod::kMean),
      Optional(MatchesArray(MakeArray<float8_e4m3fn_t>(
          {float8_e4m3fn_t{2.0f}, float8_e4m3fn_t{-4.0f}}))));
  EXPECT_THAT(
      DownsampleArray(f8_in, span<const Index>({4}), DownsampleMethod::kMedian),
      Optional(
          MatchesArray(MakeArray<float8_e4m3fn_t>({float8_e4m3fn_t{-2.0f}}))));
  EXPECT_THAT(
      DownsampleArray(MakeArray<float8_e4m3fn_t>(
                          {float8_e4m3fn_t{4.0f}, float8_e4m3fn_t{2.0f},
                           float8_e4m3fn_t{4.0f}, float8_e4m3fn_t{2.0f}}),
                      span<const Index>({4}), DownsampleMethod::kMode),
      Optional(
          MatchesArray(MakeArray<float8_e4m3fn_t>({float8_e4m3fn_t{2.0f}}))));
  EXPECT_THAT(
      DownsampleArray(MakeArray<float8_e4m3fnuz_t>(
                          {float8_e4m3fnuz_t{2.0f}, float8_e4m3fnuz_t{6.0f}}),
                      span<const Index>({2}), DownsampleMethod::kMean),
      Optional(MatchesArray(
          MakeArray<float8_e4m3fnuz_t>({float8_e4m3fnuz_t{4.0f}}))));
}

#if 0
// Fails at HEAD: `kMin`/`kMax` initialize accumulators to
// `+/-std::numeric_limits<T>::infinity()`, which is `NaN` on finite-only
// `float8_*` types (`float8_e4m3fn_t`, `float8_e4m3b11fnuz_t`,
// `float8_e5m2fnuz_t`, and `float8_e8m0fnu_t`), producing `NaN`.
TEST(DownsampleArrayTest, FiniteOnlyFloat8MinAndMax) {
  const auto f8_in = MakeArray<float8_e4m3fn_t>(
      {float8_e4m3fn_t{1.0f}, float8_e4m3fn_t{3.0f}, float8_e4m3fn_t{-2.0f},
       float8_e4m3fn_t{-6.0f}});
  EXPECT_THAT(
      DownsampleArray(f8_in, span<const Index>({2}), DownsampleMethod::kMin),
      Optional(MatchesArray(MakeArray<float8_e4m3fn_t>(
          {float8_e4m3fn_t{1.0f}, float8_e4m3fn_t{-6.0f}}))));
  EXPECT_THAT(
      DownsampleArray(f8_in, span<const Index>({2}), DownsampleMethod::kMax),
      Optional(MatchesArray(MakeArray<float8_e4m3fn_t>(
          {float8_e4m3fn_t{3.0f}, float8_e4m3fn_t{-2.0f}}))));
  EXPECT_THAT(DownsampleArray(
                  MakeArray<float8_e4m3b11fnuz_t>(
                      {float8_e4m3b11fnuz_t{1.0f}, float8_e4m3b11fnuz_t{3.0f}}),
                  span<const Index>({2}), DownsampleMethod::kMin),
              Optional(MatchesArray(MakeArray<float8_e4m3b11fnuz_t>(
                  {float8_e4m3b11fnuz_t{1.0f}}))));
  EXPECT_THAT(
      DownsampleArray(MakeArray<float8_e5m2fnuz_t>(
                          {float8_e5m2fnuz_t{-4.0f}, float8_e5m2fnuz_t{-2.0f}}),
                      span<const Index>({2}), DownsampleMethod::kMax),
      Optional(MatchesArray(
          MakeArray<float8_e5m2fnuz_t>({float8_e5m2fnuz_t{-2.0f}}))));
  EXPECT_THAT(
      DownsampleArray(MakeArray<float8_e8m0fnu_t>(
                          {float8_e8m0fnu_t{1.0f}, float8_e8m0fnu_t{4.0f}}),
                      span<const Index>({2}), DownsampleMethod::kMin),
      Optional(
          MatchesArray(MakeArray<float8_e8m0fnu_t>({float8_e8m0fnu_t{1.0f}}))));
}
#endif

TEST(DownsampleArrayTest, Float4Bfloat16AndFloat16Mean) {
  EXPECT_THAT(
      DownsampleArray(MakeArray<float4_e2m1fn_t>(
                          {float4_e2m1fn_t{1.0f}, float4_e2m1fn_t{3.0f},
                           float4_e2m1fn_t{-2.0f}, float4_e2m1fn_t{-6.0f}}),
                      span<const Index>({2}), DownsampleMethod::kMean),
      Optional(MatchesArray(MakeArray<float4_e2m1fn_t>(
          {float4_e2m1fn_t{2.0f}, float4_e2m1fn_t{-4.0f}}))));
  EXPECT_THAT(DownsampleArray(
                  MakeArray<bfloat16_t>({bfloat16_t{1.0f}, bfloat16_t{3.0f},
                                         bfloat16_t{5.0f}, bfloat16_t{7.0f}}),
                  span<const Index>({2}), DownsampleMethod::kMean),
              Optional(MatchesArray(MakeArray<bfloat16_t>(
                  {bfloat16_t{2.0f}, bfloat16_t{6.0f}}))));
  EXPECT_THAT(
      DownsampleArray(MakeArray<float16_t>({float16_t{1.0f}, float16_t{3.0f},
                                            float16_t{5.0f}, float16_t{7.0f}}),
                      span<const Index>({2}), DownsampleMethod::kMean),
      Optional(MatchesArray(
          MakeArray<float16_t>({float16_t{2.0f}, float16_t{6.0f}}))));
}

TEST(DownsampleArrayTest, Int2AndInt4Reductions) {
  EXPECT_THAT(
      DownsampleArray(
          MakeArray<int2_t>({int2_t{-2}, int2_t{0}, int2_t{-1}, int2_t{1}}),
          span<const Index>({2}), DownsampleMethod::kMean),
      Optional(MatchesArray(MakeArray<int2_t>({int2_t{-1}, int2_t{0}}))));
  EXPECT_THAT(
      DownsampleArray(
          MakeArray<int4_t>({int4_t{-7}, int4_t{-3}, int4_t{2}, int4_t{6}}),
          span<const Index>({2}), DownsampleMethod::kMean),
      Optional(MatchesArray(MakeArray<int4_t>({int4_t{-5}, int4_t{4}}))));
  EXPECT_THAT(
      DownsampleArray(
          MakeArray<int4_t>({int4_t{5}, int4_t{-3}, int4_t{2}, int4_t{-7}}),
          span<const Index>({4}), DownsampleMethod::kMedian),
      Optional(MatchesArray(MakeArray<int4_t>({int4_t{-3}}))));
}

TEST(DownsampleArrayTest, CharByteAndStringMode) {
  EXPECT_THAT(DownsampleArray(MakeArray<char_t>({'d', 'b', 'd', 'a'}),
                              span<const Index>({4}), DownsampleMethod::kMode),
              Optional(MakeArray<char_t>({'d'})));
  EXPECT_THAT(DownsampleArray(MakeArray<byte_t>(
                                  {byte_t{9}, byte_t{3}, byte_t{9}, byte_t{1}}),
                              span<const Index>({4}), DownsampleMethod::kMode),
              Optional(MakeArray<byte_t>({byte_t{9}})));
  EXPECT_THAT(
      DownsampleArray(MakeArray<ustring_t>(
                          {ustring_t{"z"}, ustring_t{"a"}, ustring_t{"z"}}),
                      span<const Index>({3}), DownsampleMethod::kMode),
      Optional(MakeArray<ustring_t>({ustring_t{"z"}})));

  const std::string long_a(48, 'a');
  const std::string long_b(48, 'b');
  EXPECT_THAT(DownsampleArray(MakeArray<string_t>({long_b, long_a, long_b}),
                              span<const Index>({3}), DownsampleMethod::kMode),
              Optional(MakeArray<string_t>({long_b})));
}

TEST(DownsampleArrayTest, ComplexMean) {
  EXPECT_THAT(DownsampleArray(MakeArray<complex64_t>({complex64_t{1.0f, 4.0f},
                                                      complex64_t{3.0f, 8.0f}}),
                              span<const Index>({2}), DownsampleMethod::kMean),
              Optional(MakeArray<complex64_t>({complex64_t{2.0f, 6.0f}})));
  EXPECT_THAT(DownsampleArray(MakeArray<complex128_t>({complex128_t{1.0, 3.0},
                                                       complex128_t{2.0, 6.0},
                                                       complex128_t{6.0, 9.0}}),
                              span<const Index>({3}), DownsampleMethod::kMean),
              Optional(MakeArray<complex128_t>({complex128_t{3.0, 6.0}})));
}

TEST(DownsampleArrayTest, Int64LargeSumsAndTieRounding) {
  constexpr int64_t kLargePos = int64_t{1} << 62;
  EXPECT_THAT(
      DownsampleArray(MakeArray<int64_t>({kLargePos, kLargePos, kLargePos}),
                      span<const Index>({3}), DownsampleMethod::kMean),
      Optional(MakeArray<int64_t>({kLargePos})));

  // 128-bit negative sums with exact half-way ties (round-to-nearest-even).
  EXPECT_THAT(
      DownsampleArray(MakeArray<int64_t>({-kLargePos - 2, -kLargePos - 1}),
                      span<const Index>({2}), DownsampleMethod::kMean),
      Optional(MakeArray<int64_t>({-kLargePos - 2})));
  EXPECT_THAT(
      DownsampleArray(MakeArray<int64_t>({-kLargePos - 3, -kLargePos - 2}),
                      span<const Index>({2}), DownsampleMethod::kMean),
      Optional(MakeArray<int64_t>({-kLargePos - 2})));
  EXPECT_THAT(
      DownsampleArray(
          MakeArray<int64_t>({-kLargePos, -kLargePos, -kLargePos,
                              -kLargePos - 1, -kLargePos - 1, -kLargePos - 1}),
          span<const Index>({6}), DownsampleMethod::kMean),
      Optional(MakeArray<int64_t>({-kLargePos})));
  EXPECT_THAT(
      DownsampleArray(
          MakeArray<int64_t>({-kLargePos - 1, -kLargePos - 1, -kLargePos - 1,
                              -kLargePos - 2, -kLargePos - 2, -kLargePos - 2}),
          span<const Index>({6}), DownsampleMethod::kMean),
      Optional(MakeArray<int64_t>({-kLargePos - 2})));
}

TEST(DownsampleArrayTest, Uint64LargeSumsAndTieRounding) {
  constexpr uint64_t kLargeU64 = (uint64_t{1} << 63) + 10;
  EXPECT_THAT(DownsampleArray(MakeArray<uint64_t>({kLargeU64, kLargeU64 + 1}),
                              span<const Index>({2}), DownsampleMethod::kMean),
              Optional(MakeArray<uint64_t>({kLargeU64})));
  EXPECT_THAT(
      DownsampleArray(MakeArray<uint64_t>({kLargeU64 + 1, kLargeU64 + 2}),
                      span<const Index>({2}), DownsampleMethod::kMean),
      Optional(MakeArray<uint64_t>({kLargeU64 + 2})));
  // Non-power-of-two `uint64_t` sums exceeding 2^64 (`Uint128High64 != 0`),
  // including exact half-way round-to-nearest-even ties.
  EXPECT_THAT(DownsampleArray(
                  MakeArray<uint64_t>({kLargeU64, kLargeU64, kLargeU64 + 1}),
                  span<const Index>({3}), DownsampleMethod::kMean),
              Optional(MakeArray<uint64_t>({kLargeU64})));
  EXPECT_THAT(DownsampleArray(MakeArray<uint64_t>(
                                  {kLargeU64, kLargeU64 + 1, kLargeU64 + 1}),
                              span<const Index>({3}), DownsampleMethod::kMean),
              Optional(MakeArray<uint64_t>({kLargeU64 + 1})));
  EXPECT_THAT(
      DownsampleArray(
          MakeArray<uint64_t>({kLargeU64, kLargeU64, kLargeU64, kLargeU64 + 1,
                               kLargeU64 + 1, kLargeU64 + 1}),
          span<const Index>({6}), DownsampleMethod::kMean),
      Optional(MakeArray<uint64_t>({kLargeU64})));
  EXPECT_THAT(
      DownsampleArray(
          MakeArray<uint64_t>({kLargeU64 + 1, kLargeU64 + 1, kLargeU64 + 1,
                               kLargeU64 + 2, kLargeU64 + 2, kLargeU64 + 2}),
          span<const Index>({6}), DownsampleMethod::kMean),
      Optional(MakeArray<uint64_t>({kLargeU64 + 2})));
}

#if 0
// Fails at HEAD: when a single 1D output cell is clipped at both its start and
// end (`[1, 3)` inside `[0, 4)`), `PartitionDim` at HEAD does not clamp
// `first_block_factor` to `base_block_shape`, computing `{22}` instead of
// `{15}` for `kMean`.
TEST(DownsampleArrayTest, DoublyClippedSingleCell1D) {
  const auto arr_1d = MakeOffsetArray<int32_t>({1}, {10, 20});
  EXPECT_THAT(
      DownsampleArray(arr_1d, span<const Index>({4}), DownsampleMethod::kMean),
      Optional(MakeOffsetArray<int32_t>({0}, {15})));
  EXPECT_THAT(
      DownsampleArray(arr_1d, span<const Index>({4}), DownsampleMethod::kMin),
      Optional(MakeOffsetArray<int32_t>({0}, {10})));
  EXPECT_THAT(
      DownsampleArray(arr_1d, span<const Index>({4}), DownsampleMethod::kMax),
      Optional(MakeOffsetArray<int32_t>({0}, {20})));
  EXPECT_THAT(DownsampleArray(arr_1d, span<const Index>({4}),
                              DownsampleMethod::kMedian),
              Optional(MakeOffsetArray<int32_t>({0}, {10})));
  EXPECT_THAT(
      DownsampleArray(arr_1d, span<const Index>({4}), DownsampleMethod::kMode),
      Optional(MakeOffsetArray<int32_t>({0}, {10})));
  EXPECT_THAT(DownsampleArray(MakeOffsetArray<int32_t>({1}, {20, 10, 20}),
                              span<const Index>({6}), DownsampleMethod::kMode),
              Optional(MakeOffsetArray<int32_t>({0}, {20})));
}

// Fails at HEAD: when a single 2D output cell is clipped at both its start and
// end (`[1, 3) x [2, 5)` inside `[0, 4) x [0, 6)`), `PartitionDim` at HEAD
// reads out-of-window elements and computes `{{7.41981}}` instead of `{{7}}`
// for `kMean` and `{{0}}` / `{{2}}` for `kMin`/`kMedian`/`kMode`.
TEST(DownsampleArrayTest, DoublyClippedSingleCell2D) {
  const auto arr_2d = MakeOffsetArray<float>(
      {1, 2}, {{2.0f, 4.0f, 6.0f}, {8.0f, 10.0f, 12.0f}});
  EXPECT_THAT(DownsampleArray(arr_2d, span<const Index>({4, 6}),
                              DownsampleMethod::kMean),
              Optional(MakeOffsetArray<float>({0, 0}, {{7.0f}})));
  EXPECT_THAT(DownsampleArray(arr_2d, span<const Index>({4, 6}),
                              DownsampleMethod::kMin),
              Optional(MakeOffsetArray<float>({0, 0}, {{2.0f}})));
  EXPECT_THAT(DownsampleArray(arr_2d, span<const Index>({4, 6}),
                              DownsampleMethod::kMax),
              Optional(MakeOffsetArray<float>({0, 0}, {{12.0f}})));
  EXPECT_THAT(DownsampleArray(arr_2d, span<const Index>({4, 6}),
                              DownsampleMethod::kMedian),
              Optional(MakeOffsetArray<float>({0, 0}, {{6.0f}})));
  EXPECT_THAT(DownsampleArray(arr_2d, span<const Index>({4, 6}),
                              DownsampleMethod::kMode),
              Optional(MakeOffsetArray<float>({0, 0}, {{2.0f}})));
}
#endif

TEST(DownsampleArrayTest, FloatMinAndMaxNaN) {
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  EXPECT_THAT(
      DownsampleArray(MakeArray<float>({kNaN, 2.0f, 5.0f, kNaN}),
                      span<const Index>({2}), DownsampleMethod::kMin),
      Optional(MatchesArrayIdentically(MakeArray<float>({2.0f, 5.0f}))));
  EXPECT_THAT(
      DownsampleArray(MakeArray<float>({kNaN, 2.0f, 5.0f, kNaN}),
                      span<const Index>({2}), DownsampleMethod::kMax),
      Optional(MatchesArrayIdentically(MakeArray<float>({2.0f, 5.0f}))));
}

#if 0
// Aborts at HEAD with SIGABRT in libc++ hardening
// (`strict_weak_ordering_check.h:59: Your comparator is not a valid
// strict-weak ordering`) when `StoreReductionTraitsBase::Finalize` passes
// NaN-containing windows to `std::nth_element` / `std::sort`.
TEST(DownsampleArrayTest, FloatMedianAndModeNaN) {
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  EXPECT_THAT(
      DownsampleArray(
          MakeArray<float>({1.0f, 5.0f, kNaN, kNaN, kNaN, kNaN, kNaN, kNaN}),
          span<const Index>({4}), DownsampleMethod::kMedian),
      Optional(MatchesArrayIdentically(MakeArray<float>({1.0f, kNaN}))));
  EXPECT_THAT(
      DownsampleArray(MakeArray<float>({kNaN, 3.0f, kNaN, 2.0f, 3.0f, kNaN,
                                        kNaN, kNaN, kNaN, kNaN}),
                      span<const Index>({5}), DownsampleMethod::kMode),
      Optional(MatchesArrayIdentically(MakeArray<float>({3.0f, kNaN}))));
}

// Fails at HEAD because finite-only `float8_*` `kMin`/`kMax` initialize
// accumulators to `numeric_limits::infinity()`, which is `NaN`.
TEST(DownsampleArrayTest, Float8MinAndMaxNaN) {
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  const float8_e4m3fn_t f8_nan{kNaN};
  EXPECT_THAT(
      DownsampleArray(MakeArray<float8_e4m3fn_t>(
                          {f8_nan, float8_e4m3fn_t{2.0f}, f8_nan, f8_nan}),
                      span<const Index>({2}), DownsampleMethod::kMin),
      Optional(MatchesArrayIdentically(
          MakeArray<float8_e4m3fn_t>({float8_e4m3fn_t{2.0f}, f8_nan}))));
  EXPECT_THAT(
      DownsampleArray(MakeArray<float8_e4m3fn_t>(
                          {f8_nan, float8_e4m3fn_t{-2.0f}, f8_nan, f8_nan}),
                      span<const Index>({2}), DownsampleMethod::kMax),
      Optional(MatchesArrayIdentically(
          MakeArray<float8_e4m3fn_t>({float8_e4m3fn_t{-2.0f}, f8_nan}))));

  const float8_e8m0fnu_t e8m0_nan{kNaN};
  EXPECT_THAT(DownsampleArray(
                  MakeArray<float8_e8m0fnu_t>(
                      {e8m0_nan, float8_e8m0fnu_t{4.0f}, e8m0_nan, e8m0_nan}),
                  span<const Index>({2}), DownsampleMethod::kMin),
              Optional(MatchesArrayIdentically(MakeArray<float8_e8m0fnu_t>(
                  {float8_e8m0fnu_t{4.0f}, e8m0_nan}))));
  EXPECT_THAT(DownsampleArray(
                  MakeArray<float8_e8m0fnu_t>(
                      {e8m0_nan, float8_e8m0fnu_t{4.0f}, e8m0_nan, e8m0_nan}),
                  span<const Index>({2}), DownsampleMethod::kMax),
              Optional(MatchesArrayIdentically(MakeArray<float8_e8m0fnu_t>(
                  {float8_e8m0fnu_t{4.0f}, e8m0_nan}))));
}
#endif

TEST(DownsampleArrayTest, Strided2DTransformedArrayMeanAndMedian) {
  const auto backing = MakeArray<int32_t>({
      {10, 99, 20, 99, 30, 99, 40, 99},
      {99, 99, 99, 99, 99, 99, 99, 99},
      {50, 99, 60, 99, 70, 99, 80, 99},
      {99, 99, 99, 99, 99, 99, 99, 99},
  });
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto strided_src,
                                   backing | Dims(0, 1).Stride(2));
  EXPECT_THAT(DownsampleTransformedArray(strided_src, span<const Index>({2, 2}),
                                         DownsampleMethod::kMean),
              Optional(MakeArray<int32_t>({{35, 55}})));
  EXPECT_THAT(DownsampleTransformedArray(strided_src, span<const Index>({2, 2}),
                                         DownsampleMethod::kMedian),
              Optional(MakeArray<int32_t>({{20, 40}})));
}

TEST(DownsampleArrayTest, Indexed2DFloat8TransformedArrayMeanAndMedian) {
  const auto f8_src = MakeArray<float8_e4m3fn_t>({
      {float8_e4m3fn_t{8.0f}, float8_e4m3fn_t{2.0f}, float8_e4m3fn_t{6.0f}},
      {float8_e4m3fn_t{4.0f}, float8_e4m3fn_t{0.0f}, float8_e4m3fn_t{2.0f}},
  });
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto indexed_f8_src,
      f8_src |
          Dims(1).IndexArraySlice(MakeArray<Index>({2, 0, 1, 2})).MoveTo(1));
  auto f8_dst = AllocateArray<float8_e4m3fn_t>({1, 2});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto indexed_f8_dst,
      f8_dst | Dims(1).IndexArraySlice(MakeArray<Index>({1, 0})).MoveTo(1));
  TENSORSTORE_EXPECT_OK(DownsampleTransformedArray(
      indexed_f8_src, indexed_f8_dst, span<const Index>({2, 2}),
      DownsampleMethod::kMean));
  // Window 0 cols {2, 0} -> {6, 8, 2, 4} mean=5 -> stored at dst col 1.
  // Window 1 cols {1, 2} -> {2, 6, 0, 2} mean=2.5 -> stored at dst col 0.
  EXPECT_THAT(f8_dst, MatchesArray(MakeArray<float8_e4m3fn_t>(
                          {{float8_e4m3fn_t{2.5f}, float8_e4m3fn_t{5.0f}}})));

  TENSORSTORE_EXPECT_OK(DownsampleTransformedArray(
      indexed_f8_src, indexed_f8_dst, span<const Index>({2, 2}),
      DownsampleMethod::kMedian));
  // Window 0 {2, 4, 6, 8} median=4 -> col 1; Window 1 {0, 2, 2, 6} -> 2 col 0.
  EXPECT_THAT(f8_dst, MatchesArray(MakeArray<float8_e4m3fn_t>(
                          {{float8_e4m3fn_t{2.0f}, float8_e4m3fn_t{4.0f}}})));
}

TEST(DownsampleArrayTest, Indexed1DStringTransformedArrayMode) {
  const auto str_src = MakeArray<string_t>({"alpha", "beta", "gamma"});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto indexed_str_src,
      str_src | Dims(0).IndexArraySlice(MakeArray<Index>({2, 0, 2, 1, 1, 0})));
  auto str_dst = AllocateArray<string_t>({2});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto indexed_str_dst,
      str_dst | Dims(0).IndexArraySlice(MakeArray<Index>({1, 0})));
  TENSORSTORE_EXPECT_OK(DownsampleTransformedArray(
      indexed_str_src, indexed_str_dst, span<const Index>({3}),
      DownsampleMethod::kMode));
  EXPECT_THAT(str_dst, MatchesArray(MakeArray<string_t>({"beta", "gamma"})));
}

TEST(DownsampleArrayTest, WidePromotedTilesAndLargeDivisors) {
  // Output width 149 (> 2 * kTileWidth=64) with partial boundary windows and
  // 5x5=25 divisor (> kInsertionSortThreshold=20).
  auto src = AllocateArray<bfloat16_t>({13, 743});
  for (Index r = 0; r < 13; ++r) {
    for (Index c = 0; c < 743; ++c) {
      const Index base = ((c / 5) % 8) * 10;
      const Index offset = (r % 5) * 2 + (c % 5 > 2 ? 1 : 0);
      src(r, c) = bfloat16_t{static_cast<float>(base + offset)};
    }
  }
  auto expected_median = AllocateArray<bfloat16_t>({3, 149});
  auto expected_mode = AllocateArray<bfloat16_t>({3, 149});
  for (Index r = 0; r < 3; ++r) {
    for (Index c = 0; c < 149; ++c) {
      const Index base = (c % 8) * 10;
      // Full rows (r < 2) have 5 row pairs -> median row offset 4;
      // boundary row (r == 2) has 3 rows {0, 2, 4} -> median row offset 2.
      const Index med_off = (r < 2) ? 4 : 2;
      expected_median(r, c) = bfloat16_t{static_cast<float>(base + med_off)};
      expected_mode(r, c) = bfloat16_t{static_cast<float>(base)};
    }
  }
  EXPECT_THAT(DownsampleArray(src, span<const Index>({5, 5}),
                              DownsampleMethod::kMedian),
              Optional(MatchesArray(expected_median)));
  EXPECT_THAT(
      DownsampleArray(src, span<const Index>({5, 5}), DownsampleMethod::kMode),
      Optional(MatchesArray(expected_mode)));
}

using MisalignedComplexTestTypes = ::testing::Types<complex64_t, complex128_t>;

template <typename T>
class MisalignedComplexDownsampleTest : public ::testing::Test {};
TYPED_TEST_SUITE(MisalignedComplexDownsampleTest, MisalignedComplexTestTypes);

TYPED_TEST(MisalignedComplexDownsampleTest, StridedContiguousAndIndexed) {
  // `std::complex<Real>` has `alignof == sizeof(Real)` (half of `sizeof(T)`);
  // verify half-element-aligned (`sizeof(Real) mod sizeof(T)`) addresses and
  // odd-half-element row strides across `kContiguous`, `kStrided`, and
  // `kIndexed` paths without over-aligned UB.
  using Complex = TypeParam;
  using Real = typename Complex::value_type;
  constexpr size_t kHalfAlign = sizeof(Real);
  constexpr size_t kElemSize = sizeof(Complex);
  static_assert(alignof(Complex) == kHalfAlign);
  static_assert(kElemSize == 2 * kHalfAlign);

  // Row 0 at `+0` (`kHalfAlign mod kElemSize`): `(1, 2), (3, 4)`;
  // `kHalfAlign`-byte gap; row 1 at `+5 * kHalfAlign` (`0 mod kElemSize`):
  // `(1, 2), (11, 12)` (`row_byte_stride = 5 * kHalfAlign`).
  constexpr Index kRowByteStride = static_cast<Index>(5 * kHalfAlign);
  constexpr Index kElemByteStride = static_cast<Index>(kElemSize);
  alignas(
      kElemSize) unsigned char raw_strided[kHalfAlign + kHalfAlign * 9] = {};
  auto* strided_ptr = reinterpret_cast<Complex*>(raw_strided + kHalfAlign);
  ASSERT_EQ(reinterpret_cast<uintptr_t>(strided_ptr) % kElemSize, kHalfAlign);
  new (raw_strided + kHalfAlign + 0) Complex(Real{1}, Real{2});
  new (raw_strided + kHalfAlign + kElemSize) Complex(Real{3}, Real{4});
  new (raw_strided + kHalfAlign + kRowByteStride) Complex(Real{1}, Real{2});
  new (raw_strided + kHalfAlign + kRowByteStride + kElemSize)
      Complex(Real{11}, Real{12});

  Array<const Complex, 2> strided_arr(
      strided_ptr, StridedLayout<2>({2, 2}, {kRowByteStride, kElemByteStride}));
  EXPECT_THAT(DownsampleArray(strided_arr, span<const Index>({2, 2}),
                              DownsampleMethod::kMode),
              Optional(MakeArray<Complex>({{Complex{Real{1}, Real{2}}}})));
  EXPECT_THAT(DownsampleArray(strided_arr, span<const Index>({2, 2}),
                              DownsampleMethod::kMean),
              Optional(MakeArray<Complex>({{Complex{Real{4}, Real{5}}}})));

  alignas(kElemSize) unsigned char raw_src[kHalfAlign + kElemSize * 6];
  alignas(kElemSize) unsigned char raw_dst[kHalfAlign + kElemSize * 2];
  auto* src = reinterpret_cast<Complex*>(raw_src + kHalfAlign);
  auto* dst = reinterpret_cast<Complex*>(raw_dst + kHalfAlign);
  ASSERT_EQ(reinterpret_cast<uintptr_t>(src) % kElemSize, kHalfAlign);
  ASSERT_EQ(reinterpret_cast<uintptr_t>(dst) % kElemSize, kHalfAlign);
  const Complex kVals[6] = {
      {Real{1}, Real{2}}, {Real{3}, Real{4}}, {Real{9}, Real{9}},
      {Real{1}, Real{2}}, {Real{5}, Real{6}}, {Real{7}, Real{8}},
  };
  for (size_t i = 0; i < 6; ++i) {
    new (&src[i]) Complex(kVals[i]);
  }
  for (size_t i = 0; i < 2; ++i) {
    new (&dst[i]) Complex(Real{0}, Real{0});
  }

  // Contiguous half-element-aligned `{2, 2}` view starting at `src`.
  Array<const Complex, 2> contig_arr(
      src, StridedLayout<2>({2, 2}, {2 * kElemByteStride, kElemByteStride}));
  EXPECT_THAT(DownsampleArray(contig_arr, span<const Index>({1, 2}),
                              DownsampleMethod::kMode),
              Optional(MakeArray<Complex>(
                  {{Complex{Real{1}, Real{2}}}, {Complex{Real{1}, Real{2}}}})));

  // `kIndexed` gather and scatter on half-element-aligned base pointers.
  Array<const Complex, 1> idx_base(src,
                                   StridedLayout<1>({6}, {kElemByteStride}));
  Array<Complex, 1> idx_out_base(dst, StridedLayout<1>({2}, {kElemByteStride}));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto indexed_src,
      TransformedArray(idx_base) |
          Dims(0).IndexArraySlice(MakeArray<Index>({0, 3, 1, 4, 5, 4})));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto indexed_dst, TransformedArray(idx_out_base) |
                            Dims(0).IndexArraySlice(MakeArray<Index>({1, 0})));
  TENSORSTORE_EXPECT_OK(DownsampleTransformedArray(indexed_src, indexed_dst,
                                                   span<const Index>({3}),
                                                   DownsampleMethod::kMode));
  EXPECT_THAT(idx_out_base,
              MatchesArray(MakeArray<Complex>(
                  {Complex{Real{5}, Real{6}}, Complex{Real{1}, Real{2}}})));
  TENSORSTORE_EXPECT_OK(DownsampleTransformedArray(indexed_src, indexed_dst,
                                                   span<const Index>({3}),
                                                   DownsampleMethod::kMean));
  EXPECT_THAT(idx_out_base,
              MatchesArray(MakeArray<Complex>(
                  {Complex{Real{17} / Real{3}, Real{20} / Real{3}},
                   Complex{Real{5} / Real{3}, Real{8} / Real{3}}})));
}

#if 0
// Aborts at HEAD with SIGABRT in libc++ hardening
// (`strict_weak_ordering_check`) when sorting complex values containing NaN in
// `kMode`.
TEST(DownsampleArrayTest, ComplexModeNaNSymmetry) {
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

  EXPECT_THAT(
      DownsampleArray(MakeArray<complex64_t>(
                          {complex64_t{kNaN, 1.0f}, complex64_t{2.0f, 3.0f},
                           complex64_t{kNaN, 1.0f}, complex64_t{2.0f, 3.0f},
                           complex64_t{2.0f, 3.0f}}),
                      span<const Index>({5}), DownsampleMethod::kMode),
      Optional(MatchesArrayIdentically(
          MakeArray<complex64_t>({complex64_t{2.0f, 3.0f}}))));

  // Complex `kMode` ignores values with NaN in either component, regardless of
  // whether real or imag is NaN or whether NaNs outnumber finite elements.
  EXPECT_THAT(DownsampleArray(MakeArray<complex64_t>({complex64_t{2.0f, 0.0f},
                                                      complex64_t{0.0f, kNaN},
                                                      complex64_t{1.0f, 0.0f}}),
                              span<const Index>({3}), DownsampleMethod::kMode),
              Optional(MakeArray<complex64_t>({complex64_t{1.0f, 0.0f}})));
  EXPECT_THAT(DownsampleArray(MakeArray<complex64_t>({complex64_t{0.0f, 2.0f},
                                                      complex64_t{kNaN, 0.0f},
                                                      complex64_t{0.0f, 1.0f}}),
                              span<const Index>({3}), DownsampleMethod::kMode),
              Optional(MakeArray<complex64_t>({complex64_t{0.0f, 1.0f}})));
  EXPECT_THAT(
      DownsampleArray(MakeArray<complex64_t>(
                          {complex64_t{0.0f, kNaN}, complex64_t{0.0f, kNaN},
                           complex64_t{kNaN, 0.0f}, complex64_t{3.0f, 4.0f}}),
                      span<const Index>({4}), DownsampleMethod::kMode),
      Optional(MakeArray<complex64_t>({complex64_t{3.0f, 4.0f}})));

  // Exercise `divisor == 25 > 23` with mixed complex NaNs.
  auto large_c64 = AllocateArray<complex64_t>({25});
  for (Index i = 0; i < 23; ++i) {
    large_c64(i) = complex64_t{static_cast<float>(23 - i), 1.0f};
  }
  large_c64(10) = complex64_t{7.0f, 1.0f};
  large_c64(23) = complex64_t{0.0f, kNaN};
  large_c64(24) = complex64_t{kNaN, 0.0f};
  EXPECT_THAT(DownsampleArray(large_c64, span<const Index>({25}),
                              DownsampleMethod::kMode),
              Optional(MakeArray<complex64_t>({complex64_t{7.0f, 1.0f}})));

  // All-NaN complex windows return a NaN-containing complex value.
  EXPECT_THAT(DownsampleArray(MakeArray<complex64_t>({complex64_t{kNaN, kNaN},
                                                      complex64_t{kNaN, kNaN}}),
                              span<const Index>({2}), DownsampleMethod::kMode),
              Optional(MatchesArrayIdentically(
                  MakeArray<complex64_t>({complex64_t{kNaN, kNaN}}))));
  EXPECT_THAT(DownsampleArray(MakeArray<complex64_t>({complex64_t{0.0f, kNaN},
                                                      complex64_t{kNaN, 0.0f}}),
                              span<const Index>({2}), DownsampleMethod::kMode),
              Optional(HasSingleComplex64NaN()));
}

// Fails at HEAD: `std::complex::operator/=` produces `(inf, NaN)` under Annex G
// complex division when dividing `(inf, finite)` by a real scalar divisor.
TEST(DownsampleArrayTest, ComplexMeanInfiniteRealComponent) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  EXPECT_THAT(DownsampleArray(MakeArray<complex64_t>({complex64_t{kInf, 0.0f},
                                                      complex64_t{2.0f, 4.0f}}),
                              span<const Index>({2}), DownsampleMethod::kMean),
              Optional(MatchesArrayIdentically(
                  MakeArray<complex64_t>({complex64_t{kInf, 2.0f}}))));
  EXPECT_THAT(DownsampleArray(MakeArray<complex64_t>({complex64_t{kInf, 0.0f},
                                                      complex64_t{1.0f, 3.0f},
                                                      complex64_t{2.0f, 6.0f}}),
                              span<const Index>({3}), DownsampleMethod::kMean),
              Optional(MatchesArrayIdentically(
                  MakeArray<complex64_t>({complex64_t{kInf, 3.0f}}))));
}

// Aborts at HEAD with SIGABRT in libc++ hardening
// (`strict_weak_ordering_check`) when `std::nth_element` / `std::sort`
// encounter NaN values in `divisor == 2` windows.
TEST(DownsampleArrayTest, FloatMedianAndModeNaNDivisor2) {
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  const auto div2_in =
      MakeArray<float>({kNaN, 4.0f, 5.0f, kNaN, kNaN, kNaN, 8.0f, 2.0f});
  EXPECT_THAT(DownsampleArray(div2_in, span<const Index>({2}),
                              DownsampleMethod::kMedian),
              Optional(MatchesArrayIdentically(
                  MakeArray<float>({4.0f, 5.0f, kNaN, 2.0f}))));
  EXPECT_THAT(
      DownsampleArray(div2_in, span<const Index>({2}), DownsampleMethod::kMode),
      Optional(
          MatchesArrayIdentically(MakeArray<float>({4.0f, 5.0f, kNaN, 2.0f}))));
  EXPECT_THAT(
      DownsampleArray(
          MakeArray<float>({10.0f, 20.0f, 30.0f, 40.0f, kNaN, 7.0f}),
          span<const Index>({4}), DownsampleMethod::kMedian),
      Optional(MatchesArrayIdentically(MakeArray<float>({20.0f, 7.0f}))));
  EXPECT_THAT(
      DownsampleArray(
          MakeArray<float>({10.0f, 20.0f, 30.0f, 40.0f, kNaN, 7.0f}),
          span<const Index>({4}), DownsampleMethod::kMode),
      Optional(MatchesArrayIdentically(MakeArray<float>({10.0f, 7.0f}))));
}

// Aborts at HEAD with SIGABRT in libc++ hardening
// (`strict_weak_ordering_check`) when `std::nth_element` / `std::sort`
// encounter NaN values in `divisor == 3` windows.
TEST(DownsampleArrayTest, FloatMedianAndModeNaNDivisor3) {
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  const auto div3_in = MakeArray<float>({
      3.0f,
      1.0f,
      2.0f,  // 0 NaNs -> median 2, mode 1
      5.0f,
      kNaN,
      1.0f,  // 1 NaN  -> median 1, mode 1
      5.0f,
      kNaN,
      kNaN,  // 2 NaNs -> median 5, mode 5
      kNaN,
      kNaN,
      kNaN,  // 3 NaNs -> median NaN, mode NaN
  });
  EXPECT_THAT(DownsampleArray(div3_in, span<const Index>({3}),
                              DownsampleMethod::kMedian),
              Optional(MatchesArrayIdentically(
                  MakeArray<float>({2.0f, 1.0f, 5.0f, kNaN}))));
  EXPECT_THAT(
      DownsampleArray(div3_in, span<const Index>({3}), DownsampleMethod::kMode),
      Optional(
          MatchesArrayIdentically(MakeArray<float>({1.0f, 1.0f, 5.0f, kNaN}))));
}

// Aborts at HEAD with SIGABRT in libc++ hardening
// (`strict_weak_ordering_check`) when `std::nth_element` / `std::sort`
// encounter NaN values in `divisor == 4` windows.
TEST(DownsampleArrayTest, FloatMedianAndModeNaNDivisor4) {
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  const auto div4_in = MakeArray<float>({
      4.0f, 1.0f, 3.0f, 3.0f,  // 0 NaNs -> median 3, mode 3
      9.0f, 1.0f, kNaN, 5.0f,  // 1 NaN  -> median 5, mode 1
      9.0f, kNaN, 2.0f, kNaN,  // 2 NaNs -> median 2, mode 2
      kNaN, 7.0f, kNaN, kNaN,  // 3 NaNs -> median 7, mode 7
      kNaN, kNaN, kNaN, kNaN,  // 4 NaNs -> median NaN, mode NaN
  });
  EXPECT_THAT(DownsampleArray(div4_in, span<const Index>({4}),
                              DownsampleMethod::kMedian),
              Optional(MatchesArrayIdentically(
                  MakeArray<float>({3.0f, 5.0f, 2.0f, 7.0f, kNaN}))));
  EXPECT_THAT(
      DownsampleArray(div4_in, span<const Index>({4}), DownsampleMethod::kMode),
      Optional(MatchesArrayIdentically(
          MakeArray<float>({3.0f, 1.0f, 2.0f, 7.0f, kNaN}))));
  const auto div4_stride6 = MakeOffsetArray<float>(
      {2}, {9.0f, kNaN, 2.0f, 5.0f, 10.0f, 10.0f, 20.0f, 30.0f, 40.0f, 50.0f});
  EXPECT_THAT(DownsampleArray(div4_stride6, span<const Index>({6}),
                              DownsampleMethod::kMedian),
              Optional(MatchesArrayIdentically(
                  MakeOffsetArray<float>({0}, {5.0f, 20.0f}))));
  EXPECT_THAT(DownsampleArray(div4_stride6, span<const Index>({6}),
                              DownsampleMethod::kMode),
              Optional(MatchesArrayIdentically(
                  MakeOffsetArray<float>({0}, {2.0f, 10.0f}))));
}

// Aborts at HEAD with SIGABRT in libc++ hardening
// (`strict_weak_ordering_check`) when `std::nth_element` / `std::sort`
// encounter NaN values in `divisor == 25` windows.
TEST(DownsampleArrayTest, FloatMedianAndModeNaNDivisor25) {
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  auto div25_in = AllocateArray<float>({75});
  for (Index i = 0; i < 11; ++i) div25_in(i) = 2.0f;
  div25_in(11) = 5.0f;
  for (Index i = 12; i < 22; ++i) div25_in(i) = 8.0f;
  for (Index i = 22; i < 25; ++i) div25_in(i) = kNaN;
  for (Index i = 25; i < 37; ++i) div25_in(i) = 4.0f;
  for (Index i = 37; i < 75; ++i) div25_in(i) = kNaN;
  EXPECT_THAT(
      DownsampleArray(div25_in, span<const Index>({25}),
                      DownsampleMethod::kMedian),
      Optional(MatchesArrayIdentically(MakeArray<float>({2.0f, 4.0f, kNaN}))));
  EXPECT_THAT(
      DownsampleArray(div25_in, span<const Index>({25}),
                      DownsampleMethod::kMode),
      Optional(MatchesArrayIdentically(MakeArray<float>({2.0f, 4.0f, kNaN}))));
}
#endif

TEST(DownsampleArrayTest, MedianAndModeRank3ThreeDownsampleDims) {
  // `{3, 3, 3}` input downsampled by `{2, 2, 2}` exercises `prior_calls > 0`
  // outer-dimension staging in `StoreSegment` alongside partial windows.
  const auto src = MakeArray<int32_t>({
      {{10, 20, 30}, {20, 40, 50}, {60, 70, 80}},
      {{20, 80, 90}, {90, 95, 99}, {60, 65, 80}},
      {{11, 12, 13}, {12, 14, 15}, {16, 17, 18}},
  });
  EXPECT_THAT(DownsampleArray(src, span<const Index>({2, 2, 2}),
                              DownsampleMethod::kMedian),
              Optional(MakeArray<int32_t>({
                  {{20, 50}, {60, 80}},
                  {{12, 13}, {16, 18}},
              })));
  EXPECT_THAT(DownsampleArray(src, span<const Index>({2, 2, 2}),
                              DownsampleMethod::kMode),
              Optional(MakeArray<int32_t>({
                  {{20, 30}, {60, 80}},
                  {{12, 13}, {16, 18}},
              })));
}

TEST(DownsampleArrayTest, MedianAndModeReversedNegativeStride1DAnd2D) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto rev_1d,
      MakeArray<int32_t>({10, 20, 40, 50, 40}) |
          Dims(0).TranslateSizedInterval(kImplicit, kImplicit, -1));
  EXPECT_THAT(DownsampleTransformedArray(rev_1d, span<const Index>({3}),
                                         DownsampleMethod::kMedian),
              Optional(MakeArray<int32_t>({40, 10})));
  EXPECT_THAT(DownsampleTransformedArray(rev_1d, span<const Index>({3}),
                                         DownsampleMethod::kMode),
              Optional(MakeArray<int32_t>({40, 10})));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto rev_2d,
      MakeArray<float>(
          {{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}}) |
          Dims(0, 1).TranslateSizedInterval(kImplicit, kImplicit, -1));
  EXPECT_THAT(DownsampleTransformedArray(rev_2d, span<const Index>({2, 2}),
                                         DownsampleMethod::kMedian),
              Optional(MakeArray<float>({{10, 8, 6}, {4, 2, 1}})));
  EXPECT_THAT(DownsampleTransformedArray(rev_2d, span<const Index>({2, 2}),
                                         DownsampleMethod::kMode),
              Optional(MakeArray<float>({{9, 7, 6}, {4, 2, 1}})));
}

TEST(DownsampleArrayTest, Float8ReversedNegativeStrideMeanAndMedian) {
  // Promoted `float8_e4m3fn_t` negative-stride (`kStrided`) input.
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto rev_f8,
      MakeArray<float8_e4m3fn_t>({float8_e4m3fn_t{2.0f}, float8_e4m3fn_t{6.0f},
                                  float8_e4m3fn_t{4.0f},
                                  float8_e4m3fn_t{8.0f}}) |
          Dims(0).TranslateSizedInterval(kImplicit, kImplicit, -1));
  EXPECT_THAT(DownsampleTransformedArray(rev_f8, span<const Index>({2}),
                                         DownsampleMethod::kMean),
              Optional(MatchesArray(MakeArray<float8_e4m3fn_t>(
                  {float8_e4m3fn_t{6.0f}, float8_e4m3fn_t{4.0f}}))));
  EXPECT_THAT(DownsampleTransformedArray(rev_f8, span<const Index>({4}),
                                         DownsampleMethod::kMedian),
              Optional(MatchesArray(
                  MakeArray<float8_e4m3fn_t>({float8_e4m3fn_t{4.0f}}))));
}

TEST(DownsampleArrayTest, MinAndMaxInt32StridedAndIndexedOutputs) {
  // Non-promoted `int32_t` `kMin`/`kMax` into `kStrided` and `kIndexed` targets
  // (`OutputKind::kCopyAccumulate`).
  const auto i32_src = MakeArray<int32_t>({{8, 1, 6, 3}, {2, 9, 4, 7}});
  auto strided_i32_buf = MakeArray<int32_t>({{-1, -1, -1, -1}});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto strided_i32_dst,
                                   strided_i32_buf | Dims(1).Stride(2));
  TENSORSTORE_EXPECT_OK(DownsampleTransformedArray(
      TransformedArray(i32_src), strided_i32_dst, span<const Index>({2, 2}),
      DownsampleMethod::kMin));
  EXPECT_THAT(strided_i32_buf,
              MatchesArray(MakeArray<int32_t>({{1, -1, 3, -1}})));
  TENSORSTORE_EXPECT_OK(DownsampleTransformedArray(
      TransformedArray(i32_src), strided_i32_dst, span<const Index>({2, 2}),
      DownsampleMethod::kMax));
  EXPECT_THAT(strided_i32_buf,
              MatchesArray(MakeArray<int32_t>({{9, -1, 7, -1}})));

  auto indexed_i32_buf = AllocateArray<int32_t>({1, 2});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto indexed_i32_dst,
      indexed_i32_buf |
          Dims(1).IndexArraySlice(MakeArray<Index>({1, 0})).MoveTo(1));
  TENSORSTORE_EXPECT_OK(DownsampleTransformedArray(
      TransformedArray(i32_src), indexed_i32_dst, span<const Index>({2, 2}),
      DownsampleMethod::kMin));
  EXPECT_THAT(indexed_i32_buf, MatchesArray(MakeArray<int32_t>({{3, 1}})));
  TENSORSTORE_EXPECT_OK(DownsampleTransformedArray(
      TransformedArray(i32_src), indexed_i32_dst, span<const Index>({2, 2}),
      DownsampleMethod::kMax));
  EXPECT_THAT(indexed_i32_buf, MatchesArray(MakeArray<int32_t>({{7, 9}})));
}

#if 0
// Fails at HEAD (`{{nan, 0, nan, 0}}` / `{{-nan, 0, -nan, 0}}` /
// `{{nan, nan}}` / `{{-nan, -nan}}`) because
// `numeric_limits<float8_e4m3fn_t>::infinity()` is NaN.
TEST(DownsampleArrayTest, MinAndMaxFloat8StridedAndIndexedOutputs) {
  // Promoted `float8_e4m3fn_t` `kMin`/`kMax` into `kStrided` and `kIndexed`
  // targets (exercising `from_compute_` + `raw_row_scratch_` scatter).
  const auto f8_src = MakeArray<float8_e4m3fn_t>({
      {float8_e4m3fn_t{8.0f}, float8_e4m3fn_t{-2.0f}, float8_e4m3fn_t{6.0f},
       float8_e4m3fn_t{1.0f}},
      {float8_e4m3fn_t{4.0f}, float8_e4m3fn_t{10.0f}, float8_e4m3fn_t{-4.0f},
       float8_e4m3fn_t{2.0f}},
  });
  auto strided_f8_buf = MakeArray<float8_e4m3fn_t>(
      {{float8_e4m3fn_t{0.0f}, float8_e4m3fn_t{0.0f}, float8_e4m3fn_t{0.0f},
        float8_e4m3fn_t{0.0f}}});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto strided_f8_dst,
                                   strided_f8_buf | Dims(1).Stride(2));
  TENSORSTORE_EXPECT_OK(DownsampleTransformedArray(
      TransformedArray(f8_src), strided_f8_dst, span<const Index>({2, 2}),
      DownsampleMethod::kMin));
  EXPECT_THAT(strided_f8_buf,
              MatchesArray(MakeArray<float8_e4m3fn_t>(
                  {{float8_e4m3fn_t{-2.0f}, float8_e4m3fn_t{0.0f},
                    float8_e4m3fn_t{-4.0f}, float8_e4m3fn_t{0.0f}}})));
  TENSORSTORE_EXPECT_OK(DownsampleTransformedArray(
      TransformedArray(f8_src), strided_f8_dst, span<const Index>({2, 2}),
      DownsampleMethod::kMax));
  EXPECT_THAT(strided_f8_buf,
              MatchesArray(MakeArray<float8_e4m3fn_t>(
                  {{float8_e4m3fn_t{10.0f}, float8_e4m3fn_t{0.0f},
                    float8_e4m3fn_t{6.0f}, float8_e4m3fn_t{0.0f}}})));

  auto indexed_f8_buf = AllocateArray<float8_e4m3fn_t>({1, 2});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto indexed_f8_dst,
      indexed_f8_buf |
          Dims(1).IndexArraySlice(MakeArray<Index>({1, 0})).MoveTo(1));
  TENSORSTORE_EXPECT_OK(DownsampleTransformedArray(
      TransformedArray(f8_src), indexed_f8_dst, span<const Index>({2, 2}),
      DownsampleMethod::kMin));
  EXPECT_THAT(indexed_f8_buf,
              MatchesArray(MakeArray<float8_e4m3fn_t>(
                  {{float8_e4m3fn_t{-4.0f}, float8_e4m3fn_t{-2.0f}}})));
  TENSORSTORE_EXPECT_OK(DownsampleTransformedArray(
      TransformedArray(f8_src), indexed_f8_dst, span<const Index>({2, 2}),
      DownsampleMethod::kMax));
  EXPECT_THAT(indexed_f8_buf,
              MatchesArray(MakeArray<float8_e4m3fn_t>(
                  {{float8_e4m3fn_t{6.0f}, float8_e4m3fn_t{10.0f}}})));
}
#endif

#if 0
// Hangs / triggers signed integer overflow at HEAD when
// `kMaxFactor == INT64_MAX` on an offset domain spanning `[-1, 1)`.
TEST(DownsampleArrayTest, MaxIndexDownsampleFactorOffsetDomain) {
  // `kMaxFactor` on offset domain `[-1, 1)^3` spans `2 x 2 x 2` output cells
  // (`[-1, 1)^3`, each containing 1 input element), exercising both outer and
  // inner `orig_index == 0` and `orig_index == 1` bounds math without signed
  // overflow.
  constexpr Index kMaxFactor = std::numeric_limits<Index>::max();
  const auto offset_3d = MakeOffsetArray<float>(
      {-1, -1, -1},
      {{{1.0f, 2.0f}, {3.0f, 4.0f}}, {{5.0f, 6.0f}, {7.0f, 8.0f}}});
  for (DownsampleMethod method :
       {DownsampleMethod::kMean, DownsampleMethod::kMin, DownsampleMethod::kMax,
        DownsampleMethod::kMedian, DownsampleMethod::kMode}) {
    SCOPED_TRACE(method);
    EXPECT_THAT(
        DownsampleArray(offset_3d,
                        span<const Index>({kMaxFactor, kMaxFactor, kMaxFactor}),
                        method),
        Optional(offset_3d));
  }

  const auto f8_offset = MakeOffsetArray<float8_e4m3fn_t>(
      {-1, -1}, {{float8_e4m3fn_t{1.0f}, float8_e4m3fn_t{2.0f}},
                 {float8_e4m3fn_t{3.0f}, float8_e4m3fn_t{4.0f}}});
  EXPECT_THAT(
      DownsampleArray(f8_offset, span<const Index>({kMaxFactor, kMaxFactor}),
                      DownsampleMethod::kMedian),
      Optional(MatchesArray(f8_offset)));

#ifndef NDEBUG
  // Force `1 x 1` blocks in debug builds so inner dimensions also step through
  // `orig_index == 1` individually in `GetBlock`.
  struct ScopedUnitBlockSize {
    ScopedUnitBlockSize() {
      tensorstore::internal::SetNDIterableTestUnitBlockSize(true);
    }
    ~ScopedUnitBlockSize() {
      tensorstore::internal::SetNDIterableTestUnitBlockSize(false);
    }
  } unit_block_guard;
  EXPECT_THAT(
      DownsampleArray(offset_3d,
                      span<const Index>({kMaxFactor, kMaxFactor, kMaxFactor}),
                      DownsampleMethod::kMedian),
      Optional(offset_3d));
#endif
}

// Triggers signed integer overflow at HEAD when `product(min(factor, shape))`
// (`100000^4`) overflows `Index` in `GetWorkingMemoryBytesPerElement`.
TEST(DownsampleArrayTest, HugeDownsampleFactorWorkingMemoryOverflowSafety) {
  // Verify `GetWorkingMemoryBytesPerElement` and `NDIterationInfo` saturate
  // safely without UB or `SIGFPE` when `product(min(factor, shape))` (`10^20`)
  // exceeds `INT64_MAX`.
  tensorstore::internal::DefaultNDIterableArena arena;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto broadcast_src,
      tensorstore::MakeScalarArray<float8_e4m3fn_t>(float8_e4m3fn_t{2.0f}) |
          Dims(0, 1, 2, 3)
              .AddNew()
              .SizedInterval({0, 0, 0, 0}, {100000, 100000, 100000, 100000}));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto base_iterable,
      tensorstore::internal::GetTransformedArrayNDIterable(
          tensorstore::UnownedToShared(broadcast_src), arena));
  auto downsampled_iterable = DownsampleNDIterable(
      std::move(base_iterable), broadcast_src.domain().box(),
      span<const Index>({100000, 100000, 100000, 100000}),
      DownsampleMethod::kMedian, 4, arena);
  tensorstore::internal::NDIterationInfo<> info(
      *downsampled_iterable, span<const Index>({1, 1, 1, 1}),
      tensorstore::skip_repeated_elements);
  EXPECT_GT(downsampled_iterable->GetWorkingMemoryBytesPerElement(
                info.layout_view(), info.buffer_kind),
            0);
  EXPECT_EQ(info.block_shape,
            (tensorstore::internal::IterationBufferShape{1, 1}));
}
#endif

using IntegerAndBoolBoundaryTestTypes =
    ::testing::Types<bool, int2_t, int4_t, int8_t, uint8_t, int16_t, uint16_t,
                     int32_t, uint32_t, int64_t, uint64_t>;

template <typename T>
class DownsampleIntegerAndBoolBoundariesTest : public ::testing::Test {};
TYPED_TEST_SUITE(DownsampleIntegerAndBoolBoundariesTest,
                 IntegerAndBoolBoundaryTestTypes);

TYPED_TEST(DownsampleIntegerAndBoolBoundariesTest, MinAndMaxAtExtremes) {
  using T = TypeParam;
  const T lo = std::numeric_limits<T>::min();
  const T hi = std::numeric_limits<T>::max();
  const T zero = T{0};
  const T one = T{1};
  EXPECT_THAT(DownsampleArray(MakeArray<T>({hi, hi, lo, hi}),
                              span<const Index>({2}), DownsampleMethod::kMin),
              Optional(MakeArray<T>({hi, lo})));
  EXPECT_THAT(DownsampleArray(MakeArray<T>({lo, lo, hi, lo}),
                              span<const Index>({2}), DownsampleMethod::kMax),
              Optional(MakeArray<T>({lo, hi})));
  EXPECT_THAT(DownsampleArray(MakeArray<T>({lo, one, zero, hi}),
                              span<const Index>({2}), DownsampleMethod::kMin),
              Optional(MakeArray<T>({lo, zero})));
  EXPECT_THAT(DownsampleArray(MakeArray<T>({lo, one, zero, hi}),
                              span<const Index>({2}), DownsampleMethod::kMax),
              Optional(MakeArray<T>({one, hi})));
}

TEST(DownsampleArrayTest, Float64MinMaxAndMeanWithInfinityAndNaN) {
  constexpr double kInfD = std::numeric_limits<double>::infinity();
  constexpr double kNaND = std::numeric_limits<double>::quiet_NaN();
  const auto dbl_arr = MakeArray<double>({-kInfD, -1.5, 2.0, kNaND});
  EXPECT_THAT(
      DownsampleArray(dbl_arr, span<const Index>({2}), DownsampleMethod::kMin),
      Optional(MatchesArrayIdentically(MakeArray<double>({-kInfD, 2.0}))));
  EXPECT_THAT(
      DownsampleArray(dbl_arr, span<const Index>({2}), DownsampleMethod::kMax),
      Optional(MatchesArrayIdentically(MakeArray<double>({-1.5, 2.0}))));
  EXPECT_THAT(
      DownsampleArray(MakeArray<double>({1.0, 4.0, 2.0, 2.0, 9.0, kNaND}),
                      span<const Index>({3}), DownsampleMethod::kMean),
      Optional(MatchesArrayIdentically(MakeArray<double>({7.0 / 3.0, kNaND}))));
}

#if 0
// Aborts at HEAD with SIGABRT in libc++ hardening
// (`strict_weak_ordering_check`) when `std::nth_element` / `std::sort`
// encounter NaN values in `double` windows.
TEST(DownsampleArrayTest, Float64MedianAndModeWithNaN) {
  constexpr double kNaND = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THAT(
      DownsampleArray(MakeArray<double>({9.0, 1.0, 4.0, 2.0, kNaND, kNaND}),
                      span<const Index>({3}), DownsampleMethod::kMedian),
      Optional(MatchesArrayIdentically(MakeArray<double>({4.0, 2.0}))));
  EXPECT_THAT(
      DownsampleArray(MakeArray<double>({9.0, 4.0, 4.0, 2.0, kNaND, kNaND}),
                      span<const Index>({3}), DownsampleMethod::kMode),
      Optional(MatchesArrayIdentically(MakeArray<double>({4.0, 2.0}))));
}
#endif

// Exercises 8-bit `InsertionSort` (`5 <= divisor <= 32`) vs 256-bin histogram
// (`divisor >= 33`).
TEST(DownsampleArrayTest, EightBitMedianAndModeSmallAndLargeDivisors) {
  // Signed `int8_t` `divisor == 5` (`InsertionSort` + `^ 0x80` sign bijection).
  EXPECT_THAT(
      DownsampleArray(MakeArray<int8_t>({10, int8_t{-128}, 127, 0, -5}),
                      span<const Index>({5}), DownsampleMethod::kMedian),
      Optional(MakeArray<int8_t>({0})));
  EXPECT_THAT(DownsampleArray(MakeArray<int8_t>({10, int8_t{-128}, 127, 0, -5}),
                              span<const Index>({5}), DownsampleMethod::kMode),
              Optional(MakeArray<int8_t>({int8_t{-128}})));

  // Length `65` downsampled by `{33}` exercises `divisor == 33` (histogram) in
  // cell 0 and `divisor == 32` (`InsertionSort`) in cell 1 within a single
  // call.
  auto u8_65 = AllocateArray<uint8_t>({65});
  for (Index i = 0; i < 32; ++i) u8_65(i) = static_cast<uint8_t>(i);
  u8_65(32) = 15;
  for (Index i = 0; i < 31; ++i) u8_65(33 + i) = static_cast<uint8_t>(100 + i);
  u8_65(64) = 110;
  EXPECT_THAT(DownsampleArray(u8_65, span<const Index>({33}),
                              DownsampleMethod::kMedian),
              Optional(MakeArray<uint8_t>({15, 114})));
  EXPECT_THAT(
      DownsampleArray(u8_65, span<const Index>({33}), DownsampleMethod::kMode),
      Optional(MakeArray<uint8_t>({15, 110})));
}

TEST(DownsampleArrayTest, Int64AndUint64MeanWithClippedTail) {
  // `int64_t` and `uint64_t` `kMean` with `> 64`-bit sums in cell 0 and
  // `divisor == 1` clipped 1-element tail in cell 1.
  constexpr int64_t kI64Min = std::numeric_limits<int64_t>::min();
  constexpr int64_t kI64Max = std::numeric_limits<int64_t>::max();
  constexpr uint64_t kU64Max = std::numeric_limits<uint64_t>::max();
  EXPECT_THAT(
      DownsampleArray(MakeArray<int64_t>({kI64Max, kI64Max - 2, kI64Min}),
                      span<const Index>({2}), DownsampleMethod::kMean),
      Optional(MakeArray<int64_t>({kI64Max - 1, kI64Min})));
  EXPECT_THAT(
      DownsampleArray(MakeArray<uint64_t>({kU64Max, kU64Max - 2, kU64Max}),
                      span<const Index>({2}), DownsampleMethod::kMean),
      Optional(MakeArray<uint64_t>({kU64Max - 1, kU64Max})));
  EXPECT_THAT(DownsampleArray(MakeArray<int64_t>({kI64Min, kI64Max, -1, 2}),
                              span<const Index>({2}), DownsampleMethod::kMean),
              Optional(MakeArray<int64_t>({0, 0})));
}

TEST(DownsampleArrayTest, Json2DModeWithClippedWindows) {
  // 2-D `3 x 3` `json_t` by `{2, 2}` hits `seg_factor == 2`, `seg_factor == 1`,
  // `divisor == 4`, `divisor == 2`, and `divisor == 1` in `ComputeModeRow`.
  const auto json_3x3 = MakeArray<json_t>({
      {json_t{{"a", 1}}, json_t{{"a", 1}}, json_t{{"b", 2}}},
      {json_t{{"b", 2}}, json_t{{"a", 1}}, json_t{{"b", 2}}},
      {json_t{{"c", 3}}, json_t{{"c", 3}}, json_t{{"d", 4}}},
  });
  EXPECT_THAT(DownsampleArray(json_3x3, span<const Index>({2, 2}),
                              DownsampleMethod::kMode),
              Optional(MakeArray<json_t>({
                  {json_t{{"a", 1}}, json_t{{"b", 2}}},
                  {json_t{{"c", 3}}, json_t{{"d", 4}}},
              })));
}

TEST(DownsampleArrayTest, NonSSOStringUniformWindowMode) {
  // Non-SSO heap-allocated `string_t` uniform window (`divisor == 3`).
  const std::string long_s(64, 'z');
  EXPECT_THAT(DownsampleArray(MakeArray<string_t>({long_s, long_s, long_s}),
                              span<const Index>({3}), DownsampleMethod::kMode),
              Optional(MakeArray<string_t>({long_s})));
}

TEST(DownsampleArrayTest, ClippedHeadFullAndTailWindows1D) {
  EXPECT_THAT(
      DownsampleArray(MakeOffsetArray<int32_t>({1}, {3, 9, 2, 4, 6, 10, 20}),
                      span<const Index>({3}), DownsampleMethod::kMean),
      Optional(MakeOffsetArray<int32_t>({0}, {6, 4, 15})));
}

// Exercises simultaneous `(head, full, tail)` windows along BOTH dimensions
// (`[1, 8) x [2, 10)` by `{3, 3}` -> all 9 region combinations in one block).
TEST(DownsampleArrayTest, AllNineClippedWindowRegions2D) {
  auto f_grid = AllocateArray<float>(tensorstore::BoxView<2>({1, 2}, {7, 8}));
  auto bf_grid =
      AllocateArray<bfloat16_t>(tensorstore::BoxView<2>({1, 2}, {7, 8}));
  auto i16_grid =
      AllocateArray<int16_t>(tensorstore::BoxView<2>({1, 2}, {7, 8}));
  auto ustr_grid =
      AllocateArray<ustring_t>(tensorstore::BoxView<2>({1, 2}, {7, 8}));
  for (Index r = 1; r < 8; ++r) {
    for (Index c = 2; c < 10; ++c) {
      const int v = static_cast<int>(r * 10 + c);
      f_grid(r, c) = static_cast<float>(v);
      bf_grid(r, c) = bfloat16_t(static_cast<float>(v));
      i16_grid(r, c) = static_cast<int16_t>(v);
      ustr_grid(r, c) =
          ustring_t{(r / 3 == 1 && c / 3 == 2) ? "mid" : "border"};
    }
  }
  // Cell `(0, 0)` averages `{12, 22}` = `17`; `(1, 2)` (`[3,6) x [6,9)`)
  // averages `{36..38, 46..48, 56..58}` = `47`; `(2, 3)` (`[6,8) x [9,10)`)
  // averages `{69, 79}` = `74`.
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto f_out, DownsampleArray(f_grid, span<const Index>({3, 3}),
                                  DownsampleMethod::kMean));
  auto f_typed = tensorstore::StaticDataTypeCast<float>(f_out).value();
  EXPECT_EQ(f_typed.domain(), tensorstore::BoxView<2>({0, 0}, {3, 4}));
  EXPECT_FLOAT_EQ(f_typed(0, 0), 17.0f);
  EXPECT_FLOAT_EQ(f_typed(1, 2), 47.0f);
  EXPECT_FLOAT_EQ(f_typed(2, 3), 74.0f);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto i16_out, DownsampleArray(i16_grid, span<const Index>({3, 3}),
                                    DownsampleMethod::kMean));
  auto i16_typed = tensorstore::StaticDataTypeCast<int16_t>(i16_out).value();
  EXPECT_EQ(i16_typed(0, 0), 17);
  EXPECT_EQ(i16_typed(1, 2), 47);
  EXPECT_EQ(i16_typed(2, 3), 74);

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto bf_med, DownsampleArray(bf_grid, span<const Index>({3, 3}),
                                   DownsampleMethod::kMedian));
  auto bf_med_typed =
      tensorstore::StaticDataTypeCast<bfloat16_t>(bf_med).value();
  EXPECT_EQ(bf_med_typed(0, 0), bfloat16_t(12.0f));
  EXPECT_EQ(bf_med_typed(1, 2), bfloat16_t(47.0f));
  EXPECT_EQ(bf_med_typed(2, 3), bfloat16_t(69.0f));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto bf_mode, DownsampleArray(bf_grid, span<const Index>({3, 3}),
                                    DownsampleMethod::kMode));
  auto bf_mode_typed =
      tensorstore::StaticDataTypeCast<bfloat16_t>(bf_mode).value();
  EXPECT_EQ(bf_mode_typed(1, 2), bfloat16_t(36.0f));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto ustr_out, DownsampleArray(ustr_grid, span<const Index>({3, 3}),
                                     DownsampleMethod::kMode));
  auto ustr_typed =
      tensorstore::StaticDataTypeCast<ustring_t>(ustr_out).value();
  EXPECT_EQ(ustr_typed(0, 0), ustring_t{"border"});
  EXPECT_EQ(ustr_typed(1, 2), ustring_t{"mid"});
  EXPECT_EQ(ustr_typed(2, 3), ustring_t{"border"});
}

// Exercises multi-tile promoted `kMedian`/`kMode` (`out_w > kTileWidth == 64`)
// and `> 4 KiB` tile headroom.
TEST(DownsampleArrayTest, WideBfloat16MultiTileMedianAndMode) {
  // `2 x 151` `bfloat16_t` by `{2, 2}` -> `out_w == 76 > 64` (2 full tiles +
  // partial tile + trailing `2 x 1` column).
  auto bf_wide = AllocateArray<bfloat16_t>({2, 151});
  auto bf_expected = AllocateArray<bfloat16_t>({1, 76});
  for (Index j = 0; j < 75; ++j) {
    const bfloat16_t lo(static_cast<float>(j % 32));
    const bfloat16_t hi(static_cast<float>((j % 32) + 2));
    bf_wide(0, 2 * j) = hi;
    bf_wide(0, 2 * j + 1) = lo;
    bf_wide(1, 2 * j) = lo;
    bf_wide(1, 2 * j + 1) = hi;
    bf_expected(0, j) = lo;
  }
  bf_wide(0, 150) = bfloat16_t(9.0f);
  bf_wide(1, 150) = bfloat16_t(3.0f);
  bf_expected(0, 75) = bfloat16_t(3.0f);
  EXPECT_THAT(DownsampleArray(bf_wide, span<const Index>({2, 2}),
                              DownsampleMethod::kMedian),
              Optional(MatchesArray(bf_expected)));
  EXPECT_THAT(DownsampleArray(bf_wide, span<const Index>({2, 2}),
                              DownsampleMethod::kMode),
              Optional(MatchesArray(bf_expected)));

  // `1 x 2000` `bfloat16_t` by `{1, 500}` -> `tile_bytes_per_col = 2000 B`,
  // `4 * 2000 = 8000 > 4096` (`GetWorkingMemoryBytesPerElement` tile headroom).
  auto bf_tile_budget = AllocateArray<bfloat16_t>({1, 2000});
  std::fill_n(bf_tile_budget.data(), 2000, bfloat16_t(4.0f));
  bf_tile_budget(0, 10) = bfloat16_t(2.0f);
  EXPECT_THAT(
      DownsampleArray(bf_tile_budget, span<const Index>({1, 500}),
                      DownsampleMethod::kMedian),
      Optional(MakeArray<bfloat16_t>({{bfloat16_t(4.0f), bfloat16_t(4.0f),
                                       bfloat16_t(4.0f), bfloat16_t(4.0f)}})));
}

TEST(DownsampleArrayTest, WideBfloat16RowMean) {
  // `1 x 4096` `bfloat16_t` by `{1, 2}` `kMean` -> triggers `row_scratch`
  // folding (`2048 * 16 > 24576`) and multi-block 1-D iteration.
  auto bf_4k = AllocateArray<bfloat16_t>({1, 4096});
  for (Index j = 0; j < 2048; ++j) {
    bf_4k(0, 2 * j) = bfloat16_t(1.0f);
    bf_4k(0, 2 * j + 1) = bfloat16_t(3.0f);
  }
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto bf_4k_mean, DownsampleArray(bf_4k, span<const Index>({1, 2}),
                                       DownsampleMethod::kMean));
  auto bf_4k_typed =
      tensorstore::StaticDataTypeCast<bfloat16_t>(bf_4k_mean).value();
  EXPECT_EQ(bf_4k_typed(0, 0), bfloat16_t(2.0f));
  EXPECT_EQ(bf_4k_typed(0, 2047), bfloat16_t(2.0f));
}

TEST(DownsampleArrayTest, LargeWindowMedianAndModeHeapAllocation) {
  // `> 32 KiB` `Arena` heap fallback for trivial (`10000` `uint32_t`) and
  // non-trivial (`2000` heap `string_t` -> `48 KiB > 32 KiB` construct +
  // destroy + dealloc).
  auto u32_10k = AllocateArray<uint32_t>({10000});
  for (Index i = 0; i < 10000; ++i) u32_10k(i) = static_cast<uint32_t>(i);
  EXPECT_THAT(DownsampleArray(u32_10k, span<const Index>({10000}),
                              DownsampleMethod::kMedian),
              Optional(MakeArray<uint32_t>({4999u})));

  auto str_2k = AllocateArray<string_t>({2000});
  for (Index i = 0; i < 1999; ++i) {
    str_2k(i) = std::string(36, 'a') + std::to_string(i);
  }
  str_2k(1999) = str_2k(500);
  EXPECT_THAT(DownsampleArray(str_2k, span<const Index>({2000}),
                              DownsampleMethod::kMode),
              Optional(MakeArray<string_t>({str_2k(500)})));
}

// Exercises Rank-3 `prior_accumulate_calls > 0` on promoted (`bfloat16_t`,
// `int4_t`) and non-trivial (`string_t`) types.
TEST(DownsampleArrayTest, Rank3Bfloat16Int4AndStringReductions) {
  auto bf_3d = AllocateArray<bfloat16_t>({3, 3, 3});
  auto i4_3d = AllocateArray<int4_t>({3, 3, 3});
  auto str_3d = AllocateArray<string_t>({3, 3, 3});
  for (Index z = 0; z < 3; ++z) {
    for (Index y = 0; y < 3; ++y) {
      for (Index x = 0; x < 3; ++x) {
        bf_3d(z, y, x) = bfloat16_t(static_cast<float>(z + y + x));
        i4_3d(z, y, x) = int4_t{static_cast<int8_t>(z + y + x - 2)};
        str_3d(z, y, x) = (z < 2 && y < 2 && x < 2) ? "cell0" : "edge";
      }
    }
  }
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto bf_med_3d, DownsampleArray(bf_3d, span<const Index>({2, 2, 2}),
                                      DownsampleMethod::kMedian));
  auto bf_med_t =
      tensorstore::StaticDataTypeCast<bfloat16_t>(bf_med_3d).value();
  EXPECT_EQ(bf_med_t(0, 0, 0), bfloat16_t(1.0f));
  EXPECT_EQ(bf_med_t(1, 1, 1), bfloat16_t(6.0f));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto i4_min_3d, DownsampleArray(i4_3d, span<const Index>({2, 2, 2}),
                                      DownsampleMethod::kMin));
  auto i4_min_t = tensorstore::StaticDataTypeCast<int4_t>(i4_min_3d).value();
  EXPECT_EQ(i4_min_t(0, 0, 0), int4_t{-2});
  EXPECT_EQ(i4_min_t(1, 1, 1), int4_t{4});

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto str_mode_3d, DownsampleArray(str_3d, span<const Index>({2, 2, 2}),
                                        DownsampleMethod::kMode));
  auto str_mode_t =
      tensorstore::StaticDataTypeCast<string_t>(str_mode_3d).value();
  EXPECT_EQ(str_mode_t(0, 0, 0), "cell0");
  EXPECT_EQ(str_mode_t(1, 1, 1), "edge");
}

TEST(DownsampleArrayTest, Rank3CombinedNonDownsampledLeadingDimensions) {
  // `{1, 1, 2}` combines non-downsampled dims 0 and 1 via
  // `CanCombineDimensions`.
  auto comb_3d = AllocateArray<int32_t>({2, 3, 4});
  auto comb_expected = AllocateArray<int32_t>({2, 3, 2});
  for (Index z = 0; z < 2; ++z) {
    for (Index y = 0; y < 3; ++y) {
      for (Index x = 0; x < 2; ++x) {
        comb_3d(z, y, 2 * x) = static_cast<int32_t>(z * 10 + y * 2);
        comb_3d(z, y, 2 * x + 1) = static_cast<int32_t>(z * 10 + y * 2 + 2);
        comb_expected(z, y, x) = static_cast<int32_t>(z * 10 + y * 2 + 1);
      }
    }
  }
  EXPECT_THAT(DownsampleArray(comb_3d, span<const Index>({1, 1, 2}),
                              DownsampleMethod::kMean),
              Optional(MatchesArray(comb_expected)));
}

TEST(DownsampleArrayTest, RowPadded2DContiguousOutput) {
  const auto f_src = MakeArray<float>({
      {4.0f, 1.0f, 8.0f, 2.0f},
      {3.0f, 5.0f, 6.0f, 7.0f},
      {9.0f, 0.0f, 4.0f, 3.0f},
      {2.0f, 8.0f, 5.0f, 9.0f},
  });
  const auto bf_src = MakeArray<bfloat16_t>({
      {bfloat16_t(4.0f), bfloat16_t(1.0f), bfloat16_t(8.0f), bfloat16_t(2.0f)},
      {bfloat16_t(3.0f), bfloat16_t(5.0f), bfloat16_t(6.0f), bfloat16_t(7.0f)},
      {bfloat16_t(9.0f), bfloat16_t(0.0f), bfloat16_t(4.0f), bfloat16_t(3.0f)},
      {bfloat16_t(2.0f), bfloat16_t(8.0f), bfloat16_t(5.0f), bfloat16_t(9.0f)},
  });

  // Row-padded 2-D `kContiguous` target (`outer_byte_stride = 5 * elem_size
  // != 2 * elem_size`, exercising `!flat_contig_out`).
  auto f_padded_backing = AllocateArray<float>({2, 5});
  std::fill_n(f_padded_backing.data(), 10, -1.0f);
  Array<float, 2> f_padded_view(
      f_padded_backing.data(),
      StridedLayout<2>({2, 2}, {5 * sizeof(float), sizeof(float)}));
  TENSORSTORE_EXPECT_OK(DownsampleArray(
      f_src, f_padded_view, span<const Index>({2, 2}), DownsampleMethod::kMin));
  EXPECT_THAT(f_padded_view,
              MatchesArray(MakeArray<float>({{1.0f, 2.0f}, {0.0f, 3.0f}})));
  EXPECT_EQ(f_padded_backing(0, 2), -1.0f);

  auto bf_padded_backing = AllocateArray<bfloat16_t>({2, 5});
  Array<bfloat16_t, 2> bf_padded_view(
      bf_padded_backing.data(),
      StridedLayout<2>({2, 2}, {5 * sizeof(bfloat16_t), sizeof(bfloat16_t)}));
  TENSORSTORE_EXPECT_OK(DownsampleArray(bf_src, bf_padded_view,
                                        span<const Index>({2, 2}),
                                        DownsampleMethod::kMin));
  EXPECT_THAT(bf_padded_view, MatchesArray(MakeArray<bfloat16_t>(
                                  {{bfloat16_t(1.0f), bfloat16_t(2.0f)},
                                   {bfloat16_t(0.0f), bfloat16_t(3.0f)}})));
}

TEST(DownsampleArrayTest, StridedAndNegativeStrideOutput) {
  const auto bf_src = MakeArray<bfloat16_t>({
      {bfloat16_t(4.0f), bfloat16_t(1.0f), bfloat16_t(8.0f), bfloat16_t(2.0f)},
      {bfloat16_t(3.0f), bfloat16_t(5.0f), bfloat16_t(6.0f), bfloat16_t(7.0f)},
      {bfloat16_t(9.0f), bfloat16_t(0.0f), bfloat16_t(4.0f), bfloat16_t(3.0f)},
      {bfloat16_t(2.0f), bfloat16_t(8.0f), bfloat16_t(5.0f), bfloat16_t(9.0f)},
  });

  // Strided + negative-stride output (`out_kind_ == kStrided`) on `bfloat16_t`
  // (`kMedian`) and `string_t` (`kMode`).
  auto bf_strided_backing = AllocateArray<bfloat16_t>({2, 4});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto bf_strided_dst,
      bf_strided_backing | Dims(1).TranslateSizedInterval(3, 2, -2));
  TENSORSTORE_EXPECT_OK(DownsampleTransformedArray(bf_src, bf_strided_dst,
                                                   span<const Index>({2, 2}),
                                                   DownsampleMethod::kMedian));
  EXPECT_EQ(bf_strided_backing(0, 3), bfloat16_t(3.0f));
  EXPECT_EQ(bf_strided_backing(0, 1), bfloat16_t(6.0f));
  EXPECT_EQ(bf_strided_backing(1, 3), bfloat16_t(2.0f));
  EXPECT_EQ(bf_strided_backing(1, 1), bfloat16_t(4.0f));

  const auto str_src =
      MakeArray<string_t>({{"a", "b", "c", "c"}, {"a", "d", "e", "c"}});
  auto str_strided_backing = AllocateArray<string_t>({1, 4});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto str_strided_dst,
                                   str_strided_backing | Dims(1).Stride(2));
  TENSORSTORE_EXPECT_OK(DownsampleTransformedArray(str_src, str_strided_dst,
                                                   span<const Index>({2, 2}),
                                                   DownsampleMethod::kMode));
  EXPECT_EQ(str_strided_backing(0, 0), "a");
  EXPECT_EQ(str_strided_backing(0, 2), "c");
}

TEST(DownsampleArrayTest, IndexedOutputPromotedAndString) {
  const auto bf_src = MakeArray<bfloat16_t>({
      {bfloat16_t(4.0f), bfloat16_t(1.0f), bfloat16_t(8.0f), bfloat16_t(2.0f)},
      {bfloat16_t(3.0f), bfloat16_t(5.0f), bfloat16_t(6.0f), bfloat16_t(7.0f)},
      {bfloat16_t(9.0f), bfloat16_t(0.0f), bfloat16_t(4.0f), bfloat16_t(3.0f)},
      {bfloat16_t(2.0f), bfloat16_t(8.0f), bfloat16_t(5.0f), bfloat16_t(9.0f)},
  });

  // Indexed output (`out_kind_ == kIndexed`) on promoted `bfloat16_t`
  // (`kMin` -> `kCopyAccumulate` and `kMean` -> `kFinalizeMean`, exercising
  // `raw_row_scratch_` scatter) and non-trivial `string_t`.
  auto bf_idx_backing = AllocateArray<bfloat16_t>({2, 3});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto bf_idx_dst,
      TransformedArray(bf_idx_backing) |
          Dims(1).IndexArraySlice(MakeArray<Index>({2, 0})).MoveToBack());
  TENSORSTORE_EXPECT_OK(DownsampleTransformedArray(
      bf_src, bf_idx_dst, span<const Index>({2, 2}), DownsampleMethod::kMin));
  EXPECT_EQ(bf_idx_backing(0, 2), bfloat16_t(1.0f));
  EXPECT_EQ(bf_idx_backing(0, 0), bfloat16_t(2.0f));
  EXPECT_EQ(bf_idx_backing(1, 2), bfloat16_t(0.0f));
  EXPECT_EQ(bf_idx_backing(1, 0), bfloat16_t(3.0f));

  TENSORSTORE_EXPECT_OK(DownsampleTransformedArray(
      bf_src, bf_idx_dst, span<const Index>({2, 2}), DownsampleMethod::kMean));
  EXPECT_EQ(bf_idx_backing(0, 2), bfloat16_t(3.25f));
  EXPECT_EQ(bf_idx_backing(0, 0), bfloat16_t(5.75f));

  const auto str_src =
      MakeArray<string_t>({{"a", "b", "c", "c"}, {"a", "d", "e", "c"}});
  auto str_idx_backing = AllocateArray<string_t>({1, 3});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto str_idx_dst,
      TransformedArray(str_idx_backing) |
          Dims(1).IndexArraySlice(MakeArray<Index>({2, 0})).MoveToBack());
  TENSORSTORE_EXPECT_OK(DownsampleTransformedArray(str_src, str_idx_dst,
                                                   span<const Index>({2, 2}),
                                                   DownsampleMethod::kMode));
  EXPECT_EQ(str_idx_backing(0, 2), "a");
  EXPECT_EQ(str_idx_backing(0, 0), "c");
}

TEST(DownsampleArrayTest, BroadcastDimensionInput) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto bcast_bf,
      MakeArray<bfloat16_t>({bfloat16_t(2.0f), bfloat16_t(6.0f)}) |
          Dims(1).AddNew().SizedInterval(0, 4));
  EXPECT_THAT(
      DownsampleTransformedArray(bcast_bf, span<const Index>({2, 2}),
                                 DownsampleMethod::kMean),
      Optional(MakeArray<bfloat16_t>({{bfloat16_t(4.0f), bfloat16_t(4.0f)}})));
}

class MedianAndModeDivisorThresholdTest
    : public ::testing::TestWithParam<Index> {};

TEST_P(MedianAndModeDivisorThresholdTest, AdjacentDivisorPair) {
  const Index factor = GetParam();
  const Index n = 2 * factor - 1;  // Cell 0 has `factor`, cell 1 `factor - 1`.
  auto arr = AllocateArray<int32_t>({n});
  for (Index i = 0; i < n; ++i) arr(i) = static_cast<int32_t>(n - i);
  arr(0) = arr(factor - 1);  // Duplicate in cell 0.
  arr(factor) = arr(n - 1);  // Duplicate in cell 1.
  EXPECT_THAT(DownsampleArray(arr, span<const Index>({factor}),
                              DownsampleMethod::kMode),
              Optional(MakeArray<int32_t>({arr(factor - 1), arr(n - 1)})));
  EXPECT_THAT(
      DownsampleArray(arr, span<const Index>({factor}),
                      DownsampleMethod::kMedian),
      Optional(MakeArray<int32_t>(
          {static_cast<int32_t>(n - (factor - 1) + (factor - 1) / 2 - 1),
           static_cast<int32_t>(1 + (factor - 2) / 2 - 1)})));
}

INSTANTIATE_TEST_SUITE_P(DivisorThresholds, MedianAndModeDivisorThresholdTest,
                         ::testing::Values(Index{5}, Index{21}, Index{24}),
                         [](const ::testing::TestParamInfo<Index>& info) {
                           return absl::StrFormat("Factor%d", info.param);
                         });

TEST(DownsampleArrayTest, Uint16MeanLargeDivisor) {
  // `uint16_t` `kMean` straddling `divisor == 32768` (32-bit vs 64-bit magic
  // multiplier) in a single `65537`-element array (`32769` + `32768`).
  auto u16_big = AllocateArray<uint16_t>({65537});
  std::fill_n(u16_big.data(), 65537, uint16_t{65535});
  EXPECT_THAT(DownsampleArray(u16_big, span<const Index>({32769}),
                              DownsampleMethod::kMean),
              Optional(MakeArray<uint16_t>({65535, 65535})));
}

#if 0
// Fails at HEAD (`{+0.0f}` instead of `{-0.0f}`) because `kMean` initializes
// its floating-point accumulator to `+0.0f` (`AccumulateElement{}`) rather than
// the IEEE-754 additive identity `-0.0f` (`+0.0f + -0.0f == +0.0f`).
TEST(DownsampleArrayTest, FloatSignedZeroMeanPreservation) {
  EXPECT_THAT(DownsampleArray(MakeArray<float>({-0.0f, -0.0f}),
                              span<const Index>({2}), DownsampleMethod::kMean),
              Optional(MatchesArrayIdentically(MakeArray<float>({-0.0f}))));
}
#endif

#if 0
// Aborts at HEAD with SIGABRT in libc++ hardening
// (`strict_weak_ordering_check`) when `std::nth_element` / `std::sort`
// encounter `signaling_NaN()`.
TEST(DownsampleArrayTest, FloatSignalingNaNMedianAndMode) {
  constexpr float kSNaN = std::numeric_limits<float>::signaling_NaN();
  const auto snan_arr = MakeArray<float>({kSNaN, 3.0f, 3.0f, kSNaN});
  EXPECT_THAT(
      DownsampleArray(snan_arr, span<const Index>({2}),
                      DownsampleMethod::kMedian),
      Optional(MatchesArrayIdentically(MakeArray<float>({3.0f, 3.0f}))));
  EXPECT_THAT(
      DownsampleArray(snan_arr, span<const Index>({2}),
                      DownsampleMethod::kMode),
      Optional(MatchesArrayIdentically(MakeArray<float>({3.0f, 3.0f}))));
}
#endif

// Parameterized test case for `DownsampleArray` across data types, methods,
// and special floating-point values (`NaN`, `+/-0.0`, `+/-Inf`).
struct DownsampleSpecialValuesParam {
  std::string name;
  SharedOffsetArray<const void> input;
  std::vector<Index> factors;
  DownsampleMethod method;
  SharedOffsetArray<const void> expected;
  EqualityComparisonKind comparison_kind = EqualityComparisonKind::identical;
};

std::ostream& operator<<(std::ostream& os,
                         const DownsampleSpecialValuesParam& p) {
  return os << p.name << "_" << p.input.dtype() << "_" << p.method;
}

// Appends a 1-D test case `[x0, x1, ...]` -> `[y0, y1, ...]` for `T`.
template <typename T>
void Add1DCase(std::vector<DownsampleSpecialValuesParam>* out, std::string name,
               DownsampleMethod method, Index factor, std::vector<T> in,
               std::vector<T> exp,
               EqualityComparisonKind cmp = EqualityComparisonKind::identical) {
  auto in_arr = AllocateArray<T>({static_cast<Index>(in.size())});
  std::copy(in.begin(), in.end(), in_arr.data());
  auto exp_arr = AllocateArray<T>({static_cast<Index>(exp.size())});
  std::copy(exp.begin(), exp.end(), exp_arr.data());
  out->push_back({
      std::move(name),
      std::move(in_arr),
      {factor},
      method,
      std::move(exp_arr),
      cmp,
  });
}

template <typename T>
void AddIEEEFloatSpecialCasesForType(
    std::vector<DownsampleSpecialValuesParam>* out) {
  const T pos_zero = static_cast<T>(+0.0f);
  const T neg_zero = static_cast<T>(-0.0f);
  const T one = static_cast<T>(1.0f);
  const T two = static_cast<T>(2.0f);
  const T three = static_cast<T>(3.0f);
  const T pos_inf = std::numeric_limits<T>::infinity();
  const T neg_inf = -std::numeric_limits<T>::infinity();
  // Convert `float` quiet_NaN rather than `numeric_limits<T>::quiet_NaN()` so
  // the payload bit-pattern matches `float32`-promoted `kMean` outputs on
  // `bfloat16_t` (`0x7fc0` vs `0x7fc1`).
  const T qnan = static_cast<T>(std::numeric_limits<float>::quiet_NaN());

  // All `-0.0` window preserves `-0.0` for `kMin`, `kMax`, `kMedian`, `kMode`
  // (`kMean` is excluded here and covered by `#if 0`
  // `FloatSignedZeroMeanPreservation` because HEAD initializes the `kMean`
  // accumulator to `+0.0`).
  for (DownsampleMethod m :
       {DownsampleMethod::kMin, DownsampleMethod::kMax,
        DownsampleMethod::kMedian, DownsampleMethod::kMode}) {
    Add1DCase<T>(out, "AllNegZero", m, 2, {neg_zero, neg_zero}, {neg_zero});
    Add1DCase<T>(out, "AllPosZero", m, 2, {pos_zero, pos_zero}, {pos_zero});
  }
  Add1DCase<T>(out, "AllPosZero", DownsampleMethod::kMean, 2,
               {pos_zero, pos_zero}, {pos_zero});
  // Mixed `{-0.0, +0.0}` and `{+0.0, -0.0}` windows: IEEE-754 treats `-0.0 ==
  // +0.0`, so `kMin`, `kMax`, `kMedian`, and `kMode` may return either sign
  // depending on reduction order; compare with `EqualityComparisonKind::equal`.
  for (DownsampleMethod m :
       {DownsampleMethod::kMean, DownsampleMethod::kMin, DownsampleMethod::kMax,
        DownsampleMethod::kMedian, DownsampleMethod::kMode}) {
    Add1DCase<T>(out, "MixedSignedZeros", m, 2,
                 {neg_zero, pos_zero, pos_zero, neg_zero}, {pos_zero, pos_zero},
                 EqualityComparisonKind::equal);
  }

  // `+/-Inf` with finite values across `kMin`, `kMax`, `kMedian`, `kMode`,
  // and `kMean`.
  Add1DCase<T>(out, "InfAndFinite", DownsampleMethod::kMin, 2,
               {pos_inf, two, neg_inf, two, pos_inf, neg_inf},
               {two, neg_inf, neg_inf});
  Add1DCase<T>(out, "InfAndFinite", DownsampleMethod::kMax, 2,
               {pos_inf, two, neg_inf, two, pos_inf, neg_inf},
               {pos_inf, two, pos_inf});
  Add1DCase<T>(out, "InfAndFinite", DownsampleMethod::kMedian, 2,
               {pos_inf, two, neg_inf, two, pos_inf, neg_inf},
               {two, neg_inf, neg_inf});
  Add1DCase<T>(out, "InfAndFinite", DownsampleMethod::kMode, 3,
               {pos_inf, two, pos_inf, neg_inf, two, neg_inf},
               {pos_inf, neg_inf});
  Add1DCase<T>(out, "InfAndFinite", DownsampleMethod::kMean, 2,
               {pos_inf, two, neg_inf, two, pos_inf, pos_inf},
               {pos_inf, neg_inf, pos_inf});

  // `NaN` propagation in `kMean` (any `NaN` in window yields `NaN`).
  Add1DCase<T>(out, "NaNPropagation", DownsampleMethod::kMean, 2,
               {one, three, qnan, two, two, qnan, qnan, qnan},
               {two, qnan, qnan, qnan});
}

template <typename T>
void AddStandardFloatNaNMinMaxCasesForType(
    std::vector<DownsampleSpecialValuesParam>* out) {
  const T two = static_cast<T>(2.0);
  const T five = static_cast<T>(5.0);
  const T pos_inf = std::numeric_limits<T>::infinity();
  const T neg_inf = -std::numeric_limits<T>::infinity();
  const T qnan = std::numeric_limits<T>::quiet_NaN();
  const T snan = std::numeric_limits<T>::signaling_NaN();

  // `kMin` and `kMax` ignore single `quiet_NaN` and `signaling_NaN` when paired
  // with finite or infinite values.
  Add1DCase<T>(out, "QuietNaNIgnoredWithFinite", DownsampleMethod::kMin, 2,
               {qnan, two, five, qnan, qnan, pos_inf}, {two, five, pos_inf});
  Add1DCase<T>(out, "QuietNaNIgnoredWithFinite", DownsampleMethod::kMax, 2,
               {qnan, two, five, qnan, qnan, neg_inf}, {two, five, neg_inf});
  Add1DCase<T>(out, "SignalingNaNIgnoredWithFinite", DownsampleMethod::kMin, 2,
               {snan, two, five, snan}, {two, five});
  Add1DCase<T>(out, "SignalingNaNIgnoredWithFinite", DownsampleMethod::kMax, 2,
               {snan, two, five, snan}, {two, five});
}

std::vector<DownsampleSpecialValuesParam> GetPassingSpecialValuesParams() {
  std::vector<DownsampleSpecialValuesParam> out;
  // Types with IEEE-754 `+/-Inf`, `-0.0`, and `NaN` (finite-only `float8_*`
  // types without `+/-Inf` fail `kMin`/`kMax` at HEAD and are covered in `#if
  // 0` `FiniteOnlyFloat8MinAndMax` / `Float8MinAndMaxNaN`).
  AddIEEEFloatSpecialCasesForType<float16_t>(&out);
  AddIEEEFloatSpecialCasesForType<bfloat16_t>(&out);
  AddIEEEFloatSpecialCasesForType<float32_t>(&out);
  AddIEEEFloatSpecialCasesForType<float64_t>(&out);
  AddIEEEFloatSpecialCasesForType<float8_e3m4_t>(&out);
  AddIEEEFloatSpecialCasesForType<float8_e5m2_t>(&out);

  // `kMin`/`kMax` with `quiet_NaN` and `signaling_NaN`: at HEAD, promoted float
  // types (`float16_t`, `bfloat16_t`, `float8_*`) use `std::min`/`std::max` on
  // `float`, where `std::min(+Inf, NaN)` returns `+Inf` if `NaN` is the first
  // element (`std::min(a, b)` returns `a` when `!(b < a)`), whereas native
  // `float32_t`/`float64_t` use `<`/`>` directly (`NaN < +Inf` is false, so
  // the first finite element overwrites `+Inf`).
  AddStandardFloatNaNMinMaxCasesForType<float32_t>(&out);
  AddStandardFloatNaNMinMaxCasesForType<float64_t>(&out);
  return out;
}

class DownsampleSpecialValuesTest
    : public ::testing::TestWithParam<DownsampleSpecialValuesParam> {};

TEST_P(DownsampleSpecialValuesTest, DownsampleArray) {
  const auto& p = GetParam();
  EXPECT_THAT(DownsampleArray(p.input, p.factors, p.method),
              Optional(MatchesArray(p.expected, p.comparison_kind)));
}

INSTANTIATE_TEST_SUITE_P(
    SpecialValues, DownsampleSpecialValuesTest,
    ::testing::ValuesIn(GetPassingSpecialValuesParams()),
    [](const ::testing::TestParamInfo<DownsampleSpecialValuesParam>& info) {
      return absl::StrFormat("%s_%v_%v", info.param.name,
                             info.param.input.dtype(), info.param.method);
    });

TEST(DownsampleArrayTest, FloatPositiveAndNegativeInfinityPair) {
  // For `{+Inf, -Inf}`: lower-median and smallest-tie mode pick `-Inf`, while
  // `kMean` computes `(+Inf + -Inf) / 2 = NaN`.
  constexpr float kInf = std::numeric_limits<float>::infinity();
  const auto inf_pair = MakeArray<float>({kInf, -kInf});
  EXPECT_THAT(
      DownsampleArray(inf_pair, span<const Index>({2}), DownsampleMethod::kMin),
      Optional(MatchesArrayIdentically(MakeArray<float>({-kInf}))));
  EXPECT_THAT(
      DownsampleArray(inf_pair, span<const Index>({2}), DownsampleMethod::kMax),
      Optional(MatchesArrayIdentically(MakeArray<float>({kInf}))));
  EXPECT_THAT(DownsampleArray(inf_pair, span<const Index>({2}),
                              DownsampleMethod::kMedian),
              Optional(MatchesArrayIdentically(MakeArray<float>({-kInf}))));
  EXPECT_THAT(DownsampleArray(inf_pair, span<const Index>({2}),
                              DownsampleMethod::kMode),
              Optional(MatchesArrayIdentically(MakeArray<float>({-kInf}))));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto inf_mean, DownsampleArray(inf_pair, span<const Index>({2}),
                                     DownsampleMethod::kMean));
  EXPECT_TRUE(std::isnan(
      *tensorstore::StaticDataTypeCast<float>(inf_mean).value().data()));
}

TEST(DownsampleArrayTest, ZeroExtentShapes) {
  // Zero-extent shapes across `kStride` and all 5 reduction methods (including
  // non-trivial `string_t`).
  auto empty_2d = AllocateArray<float>(tensorstore::BoxView<2>({2, 1}, {0, 5}));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto empty_stride, DownsampleArray(empty_2d, span<const Index>({2, 2}),
                                         DownsampleMethod::kStride));
  EXPECT_EQ(empty_stride.domain(), tensorstore::BoxView<2>({1, 1}, {0, 2}));
  for (DownsampleMethod method :
       {DownsampleMethod::kMean, DownsampleMethod::kMin, DownsampleMethod::kMax,
        DownsampleMethod::kMedian, DownsampleMethod::kMode}) {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto out, DownsampleArray(empty_2d, span<const Index>({2, 2}), method));
    EXPECT_EQ(out.domain(), tensorstore::BoxView<2>({1, 0}, {0, 3}));
  }
  auto empty_str = AllocateArray<string_t>({3, 0, 4});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto empty_str_out,
      DownsampleArray(empty_str, span<const Index>({2, 2, 2}),
                      DownsampleMethod::kMode));
  EXPECT_THAT(empty_str_out.shape(), ::testing::ElementsAre(2, 0, 2));
}

// Forwarding `NDIterable` wrapper that injects `GetBlock` or `UpdateBlock`
// errors after a configurable number of calls.
class FaultInjectingNDIterator
    : public tensorstore::internal::NDIterator::Base<FaultInjectingNDIterator> {
 public:
  FaultInjectingNDIterator(tensorstore::internal::NDIterator::Ptr inner,
                           int* call_counter, int fail_get_at, bool fail_update,
                           absl::Status error,
                           tensorstore::internal::ArenaAllocator<> allocator)
      : inner_(std::move(inner)),
        call_counter_(call_counter),
        fail_get_at_(fail_get_at),
        fail_update_(fail_update),
        error_(std::move(error)),
        allocator_(allocator) {}

  tensorstore::internal::ArenaAllocator<> get_allocator() const override {
    return allocator_;
  }

  bool GetBlock(span<const Index> indices,
                tensorstore::internal::IterationBufferShape block_shape,
                tensorstore::internal::IterationBufferPointer* pointer,
                absl::Status* status) override {
    const int idx = (*call_counter_)++;
    if (idx == fail_get_at_) {
      *status = error_;
      return false;
    }
    return inner_->GetBlock(indices, block_shape, pointer, status);
  }

  bool UpdateBlock(span<const Index> indices,
                   tensorstore::internal::IterationBufferShape block_shape,
                   tensorstore::internal::IterationBufferPointer pointer,
                   absl::Status* status) override {
    if (fail_update_) {
      *status = error_;
      return false;
    }
    return inner_->UpdateBlock(indices, block_shape, pointer, status);
  }

 private:
  tensorstore::internal::NDIterator::Ptr inner_;
  int* call_counter_;
  int fail_get_at_;
  bool fail_update_;
  absl::Status error_;
  tensorstore::internal::ArenaAllocator<> allocator_;
};

class FaultInjectingNDIterable
    : public tensorstore::internal::NDIterable::Base<FaultInjectingNDIterable> {
 public:
  FaultInjectingNDIterable(tensorstore::internal::NDIterable::Ptr inner,
                           int* call_counter, int fail_get_at, bool fail_update,
                           absl::Status error,
                           tensorstore::internal::ArenaAllocator<> allocator)
      : inner_(std::move(inner)),
        call_counter_(call_counter),
        fail_get_at_(fail_get_at),
        fail_update_(fail_update),
        error_(std::move(error)),
        allocator_(allocator) {}

  tensorstore::internal::ArenaAllocator<> get_allocator() const override {
    return allocator_;
  }
  DataType dtype() const override { return inner_->dtype(); }
  int GetDimensionOrder(tensorstore::DimensionIndex a,
                        tensorstore::DimensionIndex b) const override {
    return inner_->GetDimensionOrder(a, b);
  }
  bool CanCombineDimensions(tensorstore::DimensionIndex a, int da,
                            tensorstore::DimensionIndex b, int db,
                            Index sb) const override {
    return inner_->CanCombineDimensions(a, da, b, db, sb);
  }
  void UpdateDirectionPrefs(DirectionPref* prefs) const override {
    inner_->UpdateDirectionPrefs(prefs);
  }
  IterationBufferConstraint GetIterationBufferConstraint(
      IterationLayoutView layout) const override {
    return inner_->GetIterationBufferConstraint(layout);
  }
  ptrdiff_t GetWorkingMemoryBytesPerElement(
      IterationLayoutView layout,
      tensorstore::internal::IterationBufferKind kind) const override {
    return inner_->GetWorkingMemoryBytesPerElement(layout, kind);
  }
  tensorstore::internal::NDIterator::Ptr GetIterator(
      IterationBufferKindLayoutView layout) const override {
    return tensorstore::internal::MakeUniqueWithVirtualIntrusiveAllocator<
        FaultInjectingNDIterator>(allocator_, inner_->GetIterator(layout),
                                  call_counter_, fail_get_at_, fail_update_,
                                  error_);
  }

 private:
  tensorstore::internal::NDIterable::Ptr inner_;
  int* call_counter_;
  int fail_get_at_;
  bool fail_update_;
  absl::Status error_;
  tensorstore::internal::ArenaAllocator<> allocator_;
};

// Verifies error propagation and non-trivial `string_t` cleanup when base
// `GetBlock` fails on `prior_accumulate_calls == 0` or `1`.
TEST(DownsampleArrayTest, BaseGetBlockErrorPropagation) {
  auto src_3d = AllocateArray<string_t>({2, 2, 2});
  std::fill_n(src_3d.data(), 8, std::string(40, 'q'));
  auto dst_3d = AllocateArray<string_t>({1, 1, 1});
  dst_3d(0, 0, 0) = "unchanged";

  for (int fail_at : {0, 1}) {
    tensorstore::internal::DefaultNDIterableArena arena;
    int get_calls = 0;
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto base_it,
        tensorstore::internal::GetTransformedArrayNDIterable(src_3d, arena));
    auto faulty_base =
        tensorstore::internal::MakeUniqueWithVirtualIntrusiveAllocator<
            FaultInjectingNDIterable>(
            tensorstore::internal::ArenaAllocator<>(
                static_cast<tensorstore::internal::Arena*>(arena)),
            std::move(base_it), &get_calls, /*fail_get_at=*/fail_at,
            /*fail_update=*/false, absl::DataLossError("corrupt chunk"));
    auto downsampled = DownsampleNDIterable(
        std::move(faulty_base), src_3d.domain(), span<const Index>({2, 2, 2}),
        DownsampleMethod::kMode, 3, arena);
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto dst_it,
        tensorstore::internal::GetTransformedArrayNDIterable(dst_3d, arena));
    tensorstore::internal::NDIterableCopier copier(
        *downsampled, *dst_it, span<const Index>({1, 1, 1}), arena);
    EXPECT_THAT(copier.Copy(),
                StatusIs(absl::StatusCode::kDataLoss, "corrupt chunk"))
        << "fail_at=" << fail_at;
    EXPECT_EQ(get_calls, fail_at + 1);
    EXPECT_EQ(dst_3d(0, 0, 0), "unchanged");
  }
}

TEST(DownsampleArrayTest, TargetUpdateBlockErrorPropagation) {
  auto src_3d = AllocateArray<string_t>({2, 2, 2});
  std::fill_n(src_3d.data(), 8, std::string(40, 'q'));
  auto dst_3d = AllocateArray<string_t>({1, 1, 1});

  tensorstore::internal::DefaultNDIterableArena arena;
  int target_calls = 0;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto ok_base,
      tensorstore::internal::GetTransformedArrayNDIterable(src_3d, arena));
  auto downsampled = DownsampleNDIterable(std::move(ok_base), src_3d.domain(),
                                          span<const Index>({2, 2, 2}),
                                          DownsampleMethod::kMode, 3, arena);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto inner_dst,
      tensorstore::internal::GetTransformedArrayNDIterable(dst_3d, arena));
  auto faulty_dst =
      tensorstore::internal::MakeUniqueWithVirtualIntrusiveAllocator<
          FaultInjectingNDIterable>(
          tensorstore::internal::ArenaAllocator<>(
              static_cast<tensorstore::internal::Arena*>(arena)),
          std::move(inner_dst), &target_calls, /*fail_get_at=*/-1,
          /*fail_update=*/true, absl::OutOfRangeError("target overflow"));
  tensorstore::internal::NDIterableCopier copier(
      *downsampled, *faulty_dst, span<const Index>({1, 1, 1}), arena);
  EXPECT_THAT(copier.Copy(),
              StatusIs(absl::StatusCode::kOutOfRange, "target overflow"));
}

}  // namespace
