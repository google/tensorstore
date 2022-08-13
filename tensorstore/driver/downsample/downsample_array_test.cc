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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "tensorstore/array.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/transformed_array.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::Dims;
using ::tensorstore::DownsampleMethod;
using ::tensorstore::Index;
using ::tensorstore::kImplicit;
using ::tensorstore::MakeArray;
using ::tensorstore::MakeOffsetArray;
using ::tensorstore::span;
using ::tensorstore::internal_downsample::DownsampleArray;
using ::tensorstore::internal_downsample::DownsampleTransformedArray;
using ::testing::Optional;

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
  using ::tensorstore::json_t;
  EXPECT_THAT(DownsampleArray(MakeArray<json_t>({"a", "a", 3.0, 3, 3u}),
                              span<const Index>({5}), DownsampleMethod::kMode),
              Optional(MakeArray<::nlohmann::json>({json_t(3)})));
}

}  // namespace
