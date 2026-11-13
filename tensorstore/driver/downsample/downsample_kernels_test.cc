// Copyright 2026 The TensorStore Authors
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

#include "tensorstore/driver/downsample/downsample_kernels.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/numeric/int128.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/util/span.h"

namespace {

using ::tensorstore::dtype_v;
using ::tensorstore::Index;
using ::tensorstore::span;
using ::tensorstore::dtypes::byte_t;
using ::tensorstore::dtypes::char_t;
using ::tensorstore::dtypes::complex128_t;
using ::tensorstore::dtypes::complex64_t;
using ::tensorstore::dtypes::float16_t;
using ::tensorstore::dtypes::float32_t;
using ::tensorstore::dtypes::float64_t;
using ::tensorstore::dtypes::float8_e3m4_t;
using ::tensorstore::dtypes::float8_e4m3fn_t;
using ::tensorstore::dtypes::int2_t;
using ::tensorstore::dtypes::int4_t;
using ::tensorstore::dtypes::json_t;
using ::tensorstore::dtypes::string_t;
using ::tensorstore::dtypes::ustring_t;
using ::tensorstore::internal_downsample::AccumulateRow;
using ::tensorstore::internal_downsample::ComplexLess;
using ::tensorstore::internal_downsample::ComputeMedianRow;
using ::tensorstore::internal_downsample::ComputeModeRow;
using ::tensorstore::internal_downsample::ComputeTypeTraits;
using ::tensorstore::internal_downsample::DimPartition;
using ::tensorstore::internal_downsample::FinalizeMeanBoolRow;
using ::tensorstore::internal_downsample::FinalizeMeanComplexRow;
using ::tensorstore::internal_downsample::FinalizeMeanFloatRow;
using ::tensorstore::internal_downsample::FinalizeMeanIntRow;
using ::tensorstore::internal_downsample::FindMostFrequentSorted;
using ::tensorstore::internal_downsample::FloatToOrderedKey;
using ::tensorstore::internal_downsample::IsUniformWindow;
using ::tensorstore::internal_downsample::LowerMedian4;
using ::tensorstore::internal_downsample::MakeUnsignedMagicDiv;
using ::tensorstore::internal_downsample::MakeUnsignedMagicDiv32;
using ::tensorstore::internal_downsample::MaxOp;
using ::tensorstore::internal_downsample::MeanAccumulateType;
using ::tensorstore::internal_downsample::Median3;
using ::tensorstore::internal_downsample::MinOp;
using ::tensorstore::internal_downsample::Mode3;
using ::tensorstore::internal_downsample::Mode4;
using ::tensorstore::internal_downsample::OrderedKeyToFloat;
using ::tensorstore::internal_downsample::PartitionDim;
using ::tensorstore::internal_downsample::RoundDivSignedMagic;
using ::tensorstore::internal_downsample::RoundDivUnsignedMagic;
using ::tensorstore::internal_downsample::StoreSegmentNonTrivial;
using ::tensorstore::internal_downsample::StoreSegmentTrivial;
using ::tensorstore::internal_downsample::SumOp;
using ::tensorstore::internal_downsample::UnsignedMagicDiv;
using ::tensorstore::internal_downsample::UnsignedMagicDiv32;
using ::testing::ElementsAre;
using ::testing::IsNan;

MATCHER(IsComplexNaN, "has a NaN real or imaginary component") {
  if (std::isnan(arg.real()) || std::isnan(arg.imag())) return true;
  *result_listener << "whose components (" << arg.real() << ", " << arg.imag()
                   << ") are both non-NaN";
  return false;
}

MATCHER_P6(MatchesDimPartition, head_src_len, full_out_begin, full_out_count,
           full_src_begin, tail_src_len, tail_src_begin, "") {
  if (arg.head_src_len == head_src_len &&
      arg.full_out_begin == full_out_begin &&
      arg.full_out_count == full_out_count &&
      arg.full_src_begin == full_src_begin &&
      arg.tail_src_len == tail_src_len &&
      arg.tail_src_begin == tail_src_begin) {
    return true;
  }
  *result_listener << "where partition is {head_src_len=" << arg.head_src_len
                   << ", full_out_begin=" << arg.full_out_begin
                   << ", full_out_count=" << arg.full_out_count
                   << ", full_src_begin=" << arg.full_src_begin
                   << ", tail_src_len=" << arg.tail_src_len
                   << ", tail_src_begin=" << arg.tail_src_begin << "}";
  return false;
}

// Compile-time verification of compute-type promotion, layout aliasing, and
// integer mean accumulator widening.
static_assert(std::is_same_v<ComputeTypeTraits<float8_e3m4_t>::ComputeElement,
                             float32_t>);
static_assert(ComputeTypeTraits<float8_e3m4_t>::kIsPromoted);
static_assert(std::is_same_v<ComputeTypeTraits<float8_e4m3fn_t>::ComputeElement,
                             float32_t>);
static_assert(ComputeTypeTraits<float16_t>::kIsPromoted);
static_assert(
    std::is_same_v<ComputeTypeTraits<int2_t>::ComputeElement, int8_t>);
static_assert(ComputeTypeTraits<int4_t>::kIsPromoted);
static_assert(
    std::is_same_v<ComputeTypeTraits<byte_t>::ComputeElement, uint8_t>);
static_assert(!ComputeTypeTraits<byte_t>::kIsPromoted);
static_assert(!ComputeTypeTraits<char_t>::kIsPromoted);
static_assert(
    std::is_same_v<ComputeTypeTraits<ustring_t>::ComputeElement, ustring_t>);
static_assert(!ComputeTypeTraits<ustring_t>::kIsPromoted);
static_assert(std::is_same_v<MeanAccumulateType<int64_t>::type, absl::int128>);
static_assert(
    std::is_same_v<MeanAccumulateType<uint64_t>::type, absl::uint128>);
static_assert(std::is_same_v<MeanAccumulateType<string_t>::type, void>);

TEST(PartitionDimTest, UnclippedAndClippedCases) {
  // Unclipped, exact multiple of `factor`.
  EXPECT_THAT(PartitionDim(/*out_len=*/2, /*base_len=*/6, /*offset=*/0,
                           /*factor=*/3),
              MatchesDimPartition(0, 0, 2, 0, 0, 6));
  // Unclipped on left, partial trailing window.
  EXPECT_THAT(PartitionDim(/*out_len=*/3, /*base_len=*/7, /*offset=*/0,
                           /*factor=*/3),
              MatchesDimPartition(0, 0, 2, 0, 1, 6));
  // Left-clipped, exact end boundary.
  EXPECT_THAT(PartitionDim(/*out_len=*/2, /*base_len=*/5, /*offset=*/1,
                           /*factor=*/3),
              MatchesDimPartition(2, 1, 1, 2, 0, 5));
  // Clipped at both ends across multiple output cells.
  EXPECT_THAT(PartitionDim(/*out_len=*/3, /*base_len=*/6, /*offset=*/2,
                           /*factor=*/3),
              MatchesDimPartition(1, 1, 1, 1, 2, 4));
  // Doubly-clipped single output cell (`offset + base_len < factor`).
  EXPECT_THAT(PartitionDim(/*out_len=*/1, /*base_len=*/2, /*offset=*/2,
                           /*factor=*/5),
              MatchesDimPartition(2, 1, 0, 2, 0, 2));
}

template <typename T>
void CheckPermutations3(std::array<T, 3> vals, T expected_median,
                        T expected_mode) {
  std::sort(vals.begin(), vals.end());
  do {
    EXPECT_EQ(Median3(vals[0], vals[1], vals[2]), expected_median);
    EXPECT_EQ(Mode3(vals[0], vals[1], vals[2]), expected_mode);
  } while (std::next_permutation(vals.begin(), vals.end()));
}

template <typename T>
void CheckPermutations4(std::array<T, 4> vals, T expected_median,
                        T expected_mode) {
  std::sort(vals.begin(), vals.end());
  do {
    EXPECT_EQ(LowerMedian4(vals[0], vals[1], vals[2], vals[3]),
              expected_median);
    EXPECT_EQ(Mode4(vals[0], vals[1], vals[2], vals[3]), expected_mode);
  } while (std::next_permutation(vals.begin(), vals.end()));
}

TEST(SmallSortingNetworksTest, Median3AndMode3AllPermutations) {
  CheckPermutations3<int>({-7, 0, 5}, /*expected_median=*/0,
                          /*expected_mode=*/-7);
  CheckPermutations3<int>({-4, -4, 9}, /*expected_median=*/-4,
                          /*expected_mode=*/-4);
  CheckPermutations3<int>({-4, 9, 9}, /*expected_median=*/9,
                          /*expected_mode=*/9);
  CheckPermutations3<int>({3, 3, 3}, /*expected_median=*/3,
                          /*expected_mode=*/3);
  CheckPermutations3<double>({-2.5, -0.5, 4.25}, /*expected_median=*/-0.5,
                             /*expected_mode=*/-2.5);
}

TEST(SmallSortingNetworksTest, LowerMedian4AndMode4AllPermutations) {
  // All distinct: lower median is index 1, mode breaks 4-way tie to minimum.
  CheckPermutations4<int>({-10, -2, 3, 8}, /*expected_median=*/-2,
                          /*expected_mode=*/-10);
  // Pair tie at largest values: lower median is index 1, mode is the pair.
  CheckPermutations4<int>({-5, 1, 7, 7}, /*expected_median=*/1,
                          /*expected_mode=*/7);
  // Pair tie in the middle.
  CheckPermutations4<int>({-5, 2, 2, 9}, /*expected_median=*/2,
                          /*expected_mode=*/2);
  // Two distinct pairs: mode breaks 2-vs-2 tie toward smaller pair.
  CheckPermutations4<int>({-3, -3, 6, 6}, /*expected_median=*/-3,
                          /*expected_mode=*/-3);
  // 3-of-a-kind and 4-of-a-kind.
  CheckPermutations4<int>({-1, 4, 4, 4}, /*expected_median=*/4,
                          /*expected_mode=*/4);
  CheckPermutations4<int>({-8, -8, -8, -8}, /*expected_median=*/-8,
                          /*expected_mode=*/-8);
}

TEST(SmallSortingNetworksTest, SignedZerosCompareEqualWithoutNaN) {
  // `-0.0f == +0.0f` under IEEE-754, so signed-zero mixtures must produce a
  // zero value of magnitude 0 across every input order.
  EXPECT_EQ(Median3(-0.0f, +0.0f, 1.0f), 0.0f);
  EXPECT_EQ(Mode3(-0.0f, +0.0f, 1.0f), 0.0f);
  EXPECT_EQ(LowerMedian4(-1.0f, -0.0f, +0.0f, 2.0f), 0.0f);
  EXPECT_EQ(Mode4(-1.0f, -0.0f, +0.0f, 2.0f), 0.0f);
}

TEST(SmallSortingNetworksTest, PartialAndAllNaNOrderedKeys) {
  const uint32_t kInfKey =
      FloatToOrderedKey(std::numeric_limits<float>::infinity());
  const uint32_t kNaNKey =
      FloatToOrderedKey(std::numeric_limits<float>::quiet_NaN());
  const uint32_t k1 = FloatToOrderedKey(1.0f);
  const uint32_t k5 = FloatToOrderedKey(5.0f);
  const uint32_t k9 = FloatToOrderedKey(9.0f);

  // 1 or 2 NaNs in a 3-element window: lower median and mode ignore NaNs.
  EXPECT_EQ(Median3(kNaNKey, k9, k1, kInfKey), k1);
  EXPECT_EQ(Mode3(kNaNKey, k9, k1, kInfKey), k1);
  EXPECT_EQ(Median3(kNaNKey, kNaNKey, k5, kInfKey), k5);
  EXPECT_EQ(Mode3(kNaNKey, kNaNKey, k5, kInfKey), k5);
  EXPECT_EQ(Median3(kNaNKey, kNaNKey, kNaNKey, kInfKey), kNaNKey);
  EXPECT_EQ(Mode3(kNaNKey, kNaNKey, kNaNKey, kInfKey), kNaNKey);

  // 1, 2, 3, or 4 NaNs in a 4-element window.
  EXPECT_EQ(LowerMedian4(kNaNKey, k9, k1, k5, kInfKey), k5);
  EXPECT_EQ(Mode4(kNaNKey, kNaNKey, k9, k1, kInfKey), k1);
  EXPECT_EQ(LowerMedian4(kNaNKey, kNaNKey, k9, k1, kInfKey), k1);
  EXPECT_EQ(LowerMedian4(kNaNKey, kNaNKey, kNaNKey, k5, kInfKey), k5);
  EXPECT_EQ(Mode4(kNaNKey, kNaNKey, kNaNKey, k5, kInfKey), k5);
  EXPECT_EQ(LowerMedian4(kNaNKey, kNaNKey, kNaNKey, kNaNKey, kInfKey), kNaNKey);
  EXPECT_EQ(Mode4(kNaNKey, kNaNKey, kNaNKey, kNaNKey, kInfKey), kNaNKey);
}

TEST(FloatRadixKeyAndComplexOrderTest, BijectionMonotonicityAndNaNMapping) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  constexpr float kSub = std::numeric_limits<float>::denorm_min();
  constexpr float kQNaN = std::numeric_limits<float>::quiet_NaN();
  float neg_qnan;
  const uint32_t neg_qnan_bits = 0xffc00001u;
  std::memcpy(&neg_qnan, &neg_qnan_bits, sizeof(neg_qnan));

  // Strictly ordered values from `-inf` to `+inf` (including `-0.0f < +0.0f`
  // in the unsigned key space).
  const std::array<float, 10> ordered = {-kInf, -1e20f, -1.0f, -kSub, -0.0f,
                                         +0.0f, kSub,   1.0f,  1e20f, kInf};
  for (size_t i = 0; i < ordered.size(); ++i) {
    const uint32_t ki = FloatToOrderedKey(ordered[i]);
    const float rt = OrderedKeyToFloat<float>(ki);
    EXPECT_EQ(rt, ordered[i]);
    EXPECT_EQ(std::signbit(rt), std::signbit(ordered[i]));
    if (i + 1 < ordered.size()) {
      EXPECT_LT(ki, FloatToOrderedKey(ordered[i + 1]));
    }
  }
  EXPECT_EQ(FloatToOrderedKey(-0.0f), 0x7fffffffu);
  EXPECT_EQ(FloatToOrderedKey(+0.0f), 0x80000000u);

  // Positive and negative NaNs map to `0xffc00000u` (`> Key(+inf)`), and decode
  // bit-identically to `std::numeric_limits<float>::quiet_NaN()`.
  EXPECT_EQ(FloatToOrderedKey(kQNaN), 0xffc00000u);
  EXPECT_EQ(FloatToOrderedKey(neg_qnan), 0xffc00000u);
  EXPECT_GT(FloatToOrderedKey(kQNaN), FloatToOrderedKey(kInf));
  const float decoded_qnan = OrderedKeyToFloat<float>(FloatToOrderedKey(kQNaN));
  EXPECT_EQ(std::memcmp(&decoded_qnan, &kQNaN, sizeof(float)), 0);

  // Also verify 64-bit `double` round-trip and NaN mapping.
  constexpr double kDInf = std::numeric_limits<double>::infinity();
  constexpr double kDQNaN = std::numeric_limits<double>::quiet_NaN();
  EXPECT_LT(FloatToOrderedKey(-kDInf), FloatToOrderedKey(-0.0));
  EXPECT_LT(FloatToOrderedKey(-0.0), FloatToOrderedKey(+0.0));
  EXPECT_LT(FloatToOrderedKey(+0.0), FloatToOrderedKey(kDInf));
  EXPECT_LT(FloatToOrderedKey(kDInf), FloatToOrderedKey(kDQNaN));
  EXPECT_TRUE(std::signbit(OrderedKeyToFloat<double>(FloatToOrderedKey(-0.0))));
  EXPECT_EQ(OrderedKeyToFloat<double>(FloatToOrderedKey(-3.25)), -3.25);
  EXPECT_THAT(OrderedKeyToFloat<double>(FloatToOrderedKey(kDQNaN)), IsNan());
}

TEST(FloatRadixKeyAndComplexOrderTest, ComplexLessStrictWeakOrdering) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  const std::array<complex64_t, 5> order_domain = {
      complex64_t{-kInf, 0.0f}, complex64_t{-1.0f, 2.0f},
      complex64_t{-1.0f, 3.0f}, complex64_t{0.0f, -1.0f},
      complex64_t{kInf, 1.0f},
  };
  ComplexLess<float> less;
  for (size_t i = 0; i < order_domain.size(); ++i) {
    EXPECT_FALSE(less(order_domain[i], order_domain[i]));
    for (size_t j = i + 1; j < order_domain.size(); ++j) {
      EXPECT_TRUE(less(order_domain[i], order_domain[j]));
      EXPECT_FALSE(less(order_domain[j], order_domain[i]));
    }
  }
}

TEST(FindMostFrequentSortedTest, TieBreakingAndPosition) {
  const std::array<int, 1> single = {42};
  EXPECT_EQ(FindMostFrequentSorted(single.data(), 1), 42);

  const std::array<int, 5> tie_first = {1, 1, 2, 3, 3};
  EXPECT_EQ(FindMostFrequentSorted(tie_first.data(), 5), 1);

  const std::array<int, 6> winner_last = {1, 2, 2, 5, 5, 5};
  EXPECT_EQ(FindMostFrequentSorted(winner_last.data(), 6), 5);
}

TEST(StagingTest, MisalignedStoreSegmentTrivialComplex128) {
  // Place `complex128_t` (16 bytes) at an `8 mod 16` address to verify
  // `StoreSegmentTrivial` never faults or corrupts unaligned buffers.
  alignas(16) unsigned char raw16[16 * 12];
  auto* base16 = reinterpret_cast<complex128_t*>(raw16 + 8);
  ASSERT_EQ(reinterpret_cast<uintptr_t>(base16) % 16, 8u);
  base16[0] = {1.0, -1.0};
  base16[1] = {2.0, -2.0};
  base16[2] = {3.0, -3.0};
  base16[3] = {4.0, -4.0};

  alignas(16) unsigned char stage_raw16[16 * 10];
  auto* staged16 = reinterpret_cast<complex128_t*>(stage_raw16 + 8);
  StoreSegmentTrivial(staged16, base16, /*out_x_begin=*/0, /*out_count=*/2,
                      /*src_x_begin=*/0, /*seg_factor=*/2,
                      /*max_total_elements=*/4, /*row_call_idx=*/0,
                      sizeof(complex128_t));
  const std::array<complex128_t, 4> row1_16 = {
      complex128_t{5.0, -5.0}, complex128_t{6.0, -6.0}, complex128_t{7.0, -7.0},
      complex128_t{8.0, -8.0}};
  StoreSegmentTrivial(staged16, row1_16.data(), /*out_x_begin=*/0,
                      /*out_count=*/2, /*src_x_begin=*/0, /*seg_factor=*/2,
                      /*max_total_elements=*/4, /*row_call_idx=*/1,
                      sizeof(complex128_t));
  EXPECT_THAT(span<const complex128_t>(staged16, 8),
              ElementsAre(complex128_t{1.0, -1.0}, complex128_t{2.0, -2.0},
                          complex128_t{5.0, -5.0}, complex128_t{6.0, -6.0},
                          complex128_t{3.0, -3.0}, complex128_t{4.0, -4.0},
                          complex128_t{7.0, -7.0}, complex128_t{8.0, -8.0}));
}

TEST(StagingTest, MisalignedStoreSegmentTrivialComplex64AndUint8) {
  alignas(8) unsigned char raw8[8 * 10];
  auto* base8 = reinterpret_cast<complex64_t*>(raw8 + 4);
  ASSERT_EQ(reinterpret_cast<uintptr_t>(base8) % 8, 4u);
  for (int i = 0; i < 8; ++i) {
    base8[i] = complex64_t{static_cast<float>(i + 1), -static_cast<float>(i)};
  }
  alignas(8) unsigned char stage_raw8[8 * 12] = {};
  auto* staged8 = reinterpret_cast<complex64_t*>(stage_raw8 + 4);
  StoreSegmentTrivial(staged8, base8, /*out_x_begin=*/0, /*out_count=*/1,
                      /*src_x_begin=*/0, /*seg_factor=*/1,
                      /*max_total_elements=*/5, /*row_call_idx=*/0,
                      sizeof(complex64_t));
  StoreSegmentTrivial(staged8, base8, /*out_x_begin=*/1, /*out_count=*/1,
                      /*src_x_begin=*/1, /*seg_factor=*/3,
                      /*max_total_elements=*/5, /*row_call_idx=*/0,
                      sizeof(complex64_t));
  StoreSegmentTrivial(staged8, base8, /*out_x_begin=*/0, /*out_count=*/1,
                      /*src_x_begin=*/4, /*seg_factor=*/4,
                      /*max_total_elements=*/5, /*row_call_idx=*/0,
                      sizeof(complex64_t));
  EXPECT_THAT(
      span<const complex64_t>(staged8, 8),
      ElementsAre(base8[4], base8[5], base8[6], base8[7],
                  complex64_t{0.0f, 0.0f}, base8[1], base8[2], base8[3]));
  StoreSegmentTrivial(staged8, base8, /*out_x_begin=*/0, /*out_count=*/1,
                      /*src_x_begin=*/0, /*seg_factor=*/4,
                      /*max_total_elements=*/4, /*row_call_idx=*/0,
                      sizeof(complex64_t));
  EXPECT_THAT(span<const complex64_t>(staged8, 4),
              ElementsAre(base8[0], base8[1], base8[2], base8[3]));

  // Also cover `seg_bytes` == 1, 2, and 4 (`uint8_t` with `seg_factor` 1, 2,
  // 4).
  const std::array<uint8_t, 8> u8_src = {10, 11, 12, 13, 20, 21, 22, 23};
  std::array<uint8_t, 12> u8_staged = {};
  StoreSegmentTrivial(u8_staged.data(), u8_src.data(), /*out_x_begin=*/0,
                      /*out_count=*/2, /*src_x_begin=*/0, /*seg_factor=*/4,
                      /*max_total_elements=*/6, /*row_call_idx=*/0, 1);
  StoreSegmentTrivial(u8_staged.data(), u8_src.data(), /*out_x_begin=*/0,
                      /*out_count=*/2, /*src_x_begin=*/0, /*seg_factor=*/2,
                      /*max_total_elements=*/6, /*row_call_idx=*/2, 1);
  EXPECT_THAT(u8_staged,
              ElementsAre(10, 11, 12, 13, 10, 11, 20, 21, 22, 23, 12, 13));
}

TEST(StagingTest, NonTrivialStringStoreSegment) {
  // Exceed SSO capacity so heap-allocated `std::string` assignment is tested.
  const std::string s0(40, 'a');
  const std::string s1(40, 'b');
  const std::string s2(40, 'c');
  const std::string s3(40, 'd');
  const std::array<string_t, 4> src = {s0, s1, s2, s3};

  std::array<string_t, 10> str_staged;
  StoreSegmentNonTrivial(dtype_v<string_t>, str_staged.data(), src.data(),
                         /*out_x_begin=*/0, /*out_count=*/1, /*src_x_begin=*/0,
                         /*seg_factor=*/1, /*max_total_elements=*/5,
                         /*row_call_idx=*/0);
  EXPECT_EQ(str_staged[0], s0);
  StoreSegmentNonTrivial(dtype_v<string_t>, str_staged.data(), src.data(),
                         /*out_x_begin=*/0, /*out_count=*/1, /*src_x_begin=*/1,
                         /*seg_factor=*/2, /*max_total_elements=*/5,
                         /*row_call_idx=*/0);
  EXPECT_THAT(span<const string_t>(str_staged.data(), 2), ElementsAre(s1, s2));
  StoreSegmentNonTrivial(dtype_v<string_t>, str_staged.data(), src.data(),
                         /*out_x_begin=*/0, /*out_count=*/1, /*src_x_begin=*/0,
                         /*seg_factor=*/3, /*max_total_elements=*/5,
                         /*row_call_idx=*/0);
  EXPECT_THAT(span<const string_t>(str_staged.data(), 3),
              ElementsAre(s0, s1, s2));
  StoreSegmentNonTrivial(dtype_v<string_t>, str_staged.data(), src.data(),
                         /*out_x_begin=*/1, /*out_count=*/1, /*src_x_begin=*/0,
                         /*seg_factor=*/4, /*max_total_elements=*/5,
                         /*row_call_idx=*/0);
  EXPECT_THAT(span<const string_t>(&str_staged[5], 4),
              ElementsAre(s0, s1, s2, s3));
}

TEST(StagingTest, NonTrivialUstringAndJsonStoreSegment) {
  std::array<json_t, 4> json_src = {json_t{{"k", 1}}, json_t{{"k", 2}},
                                    json_t{{"k", 3}}, json_t{{"k", 4}}};
  std::array<json_t, 4> json_staged;
  StoreSegmentNonTrivial(dtype_v<json_t>, json_staged.data(), json_src.data(),
                         /*out_x_begin=*/0, /*out_count=*/2, /*src_x_begin=*/0,
                         /*seg_factor=*/2, /*max_total_elements=*/2,
                         /*row_call_idx=*/0);
  EXPECT_THAT(json_staged, ElementsAre(json_t{{"k", 1}}, json_t{{"k", 2}},
                                       json_t{{"k", 3}}, json_t{{"k", 4}}));

  const std::array<ustring_t, 4> ustr_r0 = {
      ustring_t{"alpha"}, ustring_t{"beta"}, ustring_t{"gamma"},
      ustring_t{"gamma"}};
  const std::array<ustring_t, 4> ustr_r1 = {
      ustring_t{"beta"}, ustring_t{"alpha"}, ustring_t{"gamma"},
      ustring_t{"delta"}};
  std::array<ustring_t, 8> ustr_staged;
  StoreSegmentNonTrivial(dtype_v<ustring_t>, ustr_staged.data(), ustr_r0.data(),
                         /*out_x_begin=*/0, /*out_count=*/2, /*src_x_begin=*/0,
                         /*seg_factor=*/2, /*max_total_elements=*/4,
                         /*row_call_idx=*/0);
  StoreSegmentNonTrivial(dtype_v<ustring_t>, ustr_staged.data(), ustr_r1.data(),
                         /*out_x_begin=*/0, /*out_count=*/2, /*src_x_begin=*/0,
                         /*seg_factor=*/2, /*max_total_elements=*/4,
                         /*row_call_idx=*/1);
  std::array<ustring_t, 2> ustr_mode;
  ComputeModeRow<ustring_t>(ustr_staged.data(), /*stride=*/4, ustr_mode.data(),
                            /*count=*/2, /*divisor=*/4);
  EXPECT_THAT(ustr_mode, ElementsAre(ustring_t{"alpha"}, ustring_t{"gamma"}));
}

TEST(AccumulateRowTest, SumAndFloatMinMaxIgnoreNaN) {
  const std::array<int32_t, 6> src = {3, -2, 5, 10, -8, 4};
  std::array<int64_t, 2> sum_acc = {0, 0};
  AccumulateRow<SumOp, int64_t, int32_t>(sum_acc.data(), src.data(),
                                         /*count=*/2, /*factor=*/3);
  EXPECT_THAT(sum_acc, ElementsAre(6, 6));

  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  constexpr float kInf = std::numeric_limits<float>::infinity();
  const std::array<float, 4> fsrc = {kNaN, -2.5f, kNaN, kNaN};
  std::array<float, 2> min_acc = {kInf, kInf};
  std::array<float, 2> max_acc = {-kInf, -kInf};
  AccumulateRow<MinOp, float, float>(min_acc.data(), fsrc.data(), /*count=*/2,
                                     /*factor=*/2);
  AccumulateRow<MaxOp, float, float>(max_acc.data(), fsrc.data(), /*count=*/2,
                                     /*factor=*/2);
  // `MinOp`/`MaxOp` ignore NaNs and retain their `+inf`/`-inf` identity when an
  // entire window is NaN.
  EXPECT_EQ(min_acc[0], -2.5f);
  EXPECT_EQ(max_acc[0], -2.5f);
  EXPECT_EQ(min_acc[1], kInf);
  EXPECT_EQ(max_acc[1], -kInf);
}

TEST(FinalizeMeanBoolRowTest, TieBreakingRoundsToFalse) {
  // `bool` `FinalizeMeanBoolRow`: exact `divisor / 2` ties round to `false`
  // (0 is even).
  std::array<int64_t, 5> bool_sums = {0, 1, 2, 3, 4};
  bool bool_out[5] = {};
  FinalizeMeanBoolRow(bool_sums.data(), bool_out, /*count=*/5, /*divisor=*/4);
  EXPECT_THAT(bool_out, ElementsAre(false, false, false, true, true));
}

// Independent exact round-half-to-even oracle on unsigned 128-bit magnitudes.
absl::uint128 RefRoundDivUnsigned(absl::uint128 mag, uint64_t divisor) {
  const absl::uint128 d = divisor;
  const absl::uint128 q = mag / d;
  const absl::uint128 r = mag % d;
  const absl::uint128 two_r = r * 2;
  if (two_r > d || (two_r == d && (absl::Uint128Low64(q) & 1) != 0)) {
    return q + 1;
  }
  return q;
}

int64_t RefRoundDivSigned(absl::int128 x, int64_t divisor) {
  const uint64_t u_div = static_cast<uint64_t>(divisor);
  if (x >= 0) {
    return static_cast<int64_t>(
        RefRoundDivUnsigned(static_cast<absl::uint128>(x), u_div));
  }
  // Negate in unsigned 128-bit arithmetic to avoid overflow.
  const absl::uint128 mag = ~static_cast<absl::uint128>(x) + absl::uint128{1};
  const absl::uint128 q_mag = RefRoundDivUnsigned(mag, u_div);
  return static_cast<int64_t>(~q_mag + absl::uint128{1});
}

TEST(FinalizeMeanIntRowTest, ExhaustiveSmallDomainSignedAndUnsigned) {
  for (Index divisor : {1, 2, 3, 4, 5, 6, 7, 8, 12, 16}) {
    std::vector<uint64_t> u_in(201);
    std::vector<uint32_t> u_expected(201);
    for (size_t i = 0; i <= 200; ++i) {
      u_in[i] = i;
      u_expected[i] = static_cast<uint32_t>(
          RefRoundDivUnsigned(i, static_cast<uint64_t>(divisor)));
    }
    std::vector<uint32_t> u_out(201);
    FinalizeMeanIntRow<uint64_t, uint32_t>(u_in.data(), u_out.data(), 201,
                                           divisor);
    EXPECT_EQ(u_out, u_expected) << "divisor=" << divisor;

    std::vector<int64_t> s_in(401);
    std::vector<int32_t> s_expected(401);
    for (int v = -200; v <= 200; ++v) {
      s_in[v + 200] = v;
      s_expected[v + 200] = static_cast<int32_t>(RefRoundDivSigned(v, divisor));
    }
    std::vector<int32_t> s_out(401);
    FinalizeMeanIntRow<int64_t, int32_t>(s_in.data(), s_out.data(), 401,
                                         divisor);
    EXPECT_EQ(s_out, s_expected) << "divisor=" << divisor;
  }
}

TEST(FinalizeMeanIntRowTest, Int128AndUint128Above64Bits) {
  // Exercise both the 64-bit fast path and the >64-bit 128-bit division path
  // across power-of-two (`2, 4, 8`) and non-power-of-two (`3, 5, 6, 7`)
  // divisors around half-way tie boundaries.
  const uint64_t u_big_q_even = (uint64_t{1} << 62) + 10;
  const uint64_t u_big_q_odd = u_big_q_even + 1;
  for (Index divisor : {2, 3, 4, 5, 6, 7, 8}) {
    const absl::uint128 d = static_cast<uint64_t>(divisor);
    std::vector<absl::uint128> u_in;
    for (uint64_t base_q :
         {uint64_t{7}, uint64_t{8}, u_big_q_even, u_big_q_odd}) {
      for (uint64_t r = 0; r < static_cast<uint64_t>(divisor); ++r) {
        u_in.push_back(absl::uint128{base_q} * d + r);
      }
    }
    std::vector<uint64_t> u_expected(u_in.size());
    for (size_t i = 0; i < u_in.size(); ++i) {
      u_expected[i] = static_cast<uint64_t>(
          RefRoundDivUnsigned(u_in[i], static_cast<uint64_t>(divisor)));
    }
    std::vector<uint64_t> u_out(u_in.size());
    FinalizeMeanIntRow<absl::uint128, uint64_t>(
        u_in.data(), u_out.data(), static_cast<Index>(u_in.size()), divisor);
    EXPECT_EQ(u_out, u_expected) << "unsigned 128-bit divisor=" << divisor;

    const int64_t s_big_mag_even = (int64_t{1} << 61) + 12;
    const int64_t s_big_mag_odd = s_big_mag_even + 1;
    std::vector<absl::int128> s_in;
    for (int64_t base_q :
         {int64_t{-9}, int64_t{-8}, int64_t{8}, int64_t{9}, -s_big_mag_odd,
          -s_big_mag_even, s_big_mag_even, s_big_mag_odd}) {
      for (int64_t delta = -divisor; delta <= divisor; ++delta) {
        s_in.push_back(absl::int128{base_q} * divisor + delta);
      }
    }
    std::vector<int64_t> s_expected(s_in.size());
    for (size_t i = 0; i < s_in.size(); ++i) {
      s_expected[i] = RefRoundDivSigned(s_in[i], divisor);
    }
    std::vector<int64_t> s_out(s_in.size());
    FinalizeMeanIntRow<absl::int128, int64_t>(
        s_in.data(), s_out.data(), static_cast<Index>(s_in.size()), divisor);
    EXPECT_EQ(s_out, s_expected) << "signed 128-bit divisor=" << divisor;

    // Also exercise `count == 1` scalar division fast paths on both 64-bit and
    // >64-bit 128-bit sums.
    for (size_t i = 0; i < u_in.size(); ++i) {
      uint64_t single_u = 0;
      FinalizeMeanIntRow<absl::uint128, uint64_t>(&u_in[i], &single_u, 1,
                                                  divisor);
      EXPECT_EQ(single_u, u_expected[i]);
    }
    for (size_t i = 0; i < s_in.size(); ++i) {
      int64_t single_s = 0;
      FinalizeMeanIntRow<absl::int128, int64_t>(&s_in[i], &single_s, 1,
                                                divisor);
      EXPECT_EQ(single_s, s_expected[i]);
    }
  }

  // Exercise `divisor == 1` fast path on `absl::uint128` and `absl::int128`.
  std::array<absl::uint128, 2> u128_div1 = {
      0, std::numeric_limits<uint64_t>::max()};
  std::array<uint64_t, 2> u64_div1 = {};
  FinalizeMeanIntRow<absl::uint128, uint64_t>(u128_div1.data(), u64_div1.data(),
                                              2, /*divisor=*/1);
  EXPECT_THAT(u64_div1, ElementsAre(0, std::numeric_limits<uint64_t>::max()));

  std::array<absl::int128, 2> s128_div1 = {std::numeric_limits<int64_t>::min(),
                                           std::numeric_limits<int64_t>::max()};
  std::array<int64_t, 2> s64_div1 = {};
  FinalizeMeanIntRow<absl::int128, int64_t>(s128_div1.data(), s64_div1.data(),
                                            2, /*divisor=*/1);
  EXPECT_THAT(s64_div1, ElementsAre(std::numeric_limits<int64_t>::min(),
                                    std::numeric_limits<int64_t>::max()));
}

TEST(FinalizeMeanFloatAndComplexTest, FloatPowerOfTwoAndNonPowerOfTwoDivisors) {
  constexpr float kInf = std::numeric_limits<float>::infinity();
  std::array<float, 3> f_in = {3.0f, -6.0f, kInf};
  std::array<float, 3> f_pow2 = {};
  std::array<float, 3> f_nonpow2 = {};
  FinalizeMeanFloatRow<float>(f_in.data(), f_pow2.data(), 3, /*divisor=*/2);
  FinalizeMeanFloatRow<float>(f_in.data(), f_nonpow2.data(), 3, /*divisor=*/3);
  EXPECT_THAT(f_pow2, ElementsAre(1.5f, -3.0f, kInf));
  EXPECT_THAT(f_nonpow2, ElementsAre(1.0f, -2.0f, kInf));
}

TEST(FinalizeMeanFloatAndComplexTest,
     ComplexComponentwiseDivisionPreservesInf) {
  // Componentwise real division must preserve `(inf, finite)` and
  // `(0, -inf)` without Annex G `(inf, NaN)` corruption.
  constexpr float kInf = std::numeric_limits<float>::infinity();
  std::array<complex64_t, 3> c_in = {complex64_t{kInf, 3.0f},
                                     complex64_t{0.0f, -kInf},
                                     complex64_t{6.0f, -9.0f}};
  std::array<complex64_t, 3> c_out = {};
  FinalizeMeanComplexRow<float>(c_in.data(), c_out.data(), 3, /*divisor=*/3);
  EXPECT_THAT(c_out,
              ElementsAre(complex64_t{kInf, 1.0f}, complex64_t{0.0f, -kInf},
                          complex64_t{2.0f, -3.0f}));
}

TEST(ComputeMedianAndModeRowTest, StridedVsContiguousSmallWindows) {
  // `stride > divisor` with extreme sentinel values in `[divisor, stride)`
  // verifies that padding slots never leak into any window size branch.
  for (Index divisor : {1, 2, 3, 4}) {
    const Index stride = divisor + 2;
    std::vector<int32_t> staged(2 * stride, 999);
    // Window 0: `[10, 4, 10, 2]` prefix; Window 1: `[-3, -9, -1, -9]` prefix.
    const std::array<int32_t, 4> w0 = {10, 4, 10, 2};
    const std::array<int32_t, 4> w1 = {-3, -9, -1, -9};
    std::copy_n(w0.data(), divisor, staged.data());
    std::copy_n(w1.data(), divisor, staged.data() + stride);

    std::vector<int32_t> med_strided(2);
    std::vector<int32_t> mode_strided(2);
    auto staged_med = staged;
    auto staged_mode = staged;
    ComputeMedianRow<int32_t>(staged_med.data(), stride, med_strided.data(),
                              /*count=*/2, divisor);
    ComputeModeRow<int32_t>(staged_mode.data(), stride, mode_strided.data(),
                            /*count=*/2, divisor);

    // Compare against contiguous `stride == divisor`.
    std::vector<int32_t> contig(2 * divisor);
    std::copy_n(w0.data(), divisor, contig.data());
    std::copy_n(w1.data(), divisor, contig.data() + divisor);
    std::vector<int32_t> med_contig(2);
    std::vector<int32_t> mode_contig(2);
    auto contig_med = contig;
    auto contig_mode = contig;
    ComputeMedianRow<int32_t>(contig_med.data(), divisor, med_contig.data(),
                              /*count=*/2, divisor);
    ComputeModeRow<int32_t>(contig_mode.data(), divisor, mode_contig.data(),
                            /*count=*/2, divisor);
    EXPECT_EQ(med_strided, med_contig) << "divisor=" << divisor;
    EXPECT_EQ(mode_strided, mode_contig) << "divisor=" << divisor;
  }
}

TEST(ComputeMedianAndModeRowTest, FloatWindowsWithNaNsDivisor7And25) {
  // Test `InsertionSort` (`divisor == 7`) and `nth_element`/`ShellSort`
  // (`divisor == 25`) paths on `float` windows with NaNs.
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  {
    // `divisor == 7`:
    // - Window 0: 3 NaNs + `{9, 2, 7, 2}` -> `valid_n == 4`, sorted finite is
    //   `{2, 2, 7, 9}`, so lower median (`idx == 1`) is `2.0f` and mode is
    //   `2.0f`.
    // - Window 1: 4 NaNs + `{5, 8, 5}` -> `valid_n == 3`, sorted finite is
    //   `{5, 5, 8}`, so median (`idx == 1`) and mode are both `5.0f`.
    std::array<float, 14> w7 = {
        kNaN, 9.0f, kNaN, 2.0f, 7.0f, kNaN, 2.0f,
        5.0f, kNaN, kNaN, 8.0f, kNaN, 5.0f, kNaN,
    };
    auto w7_med = w7;
    auto w7_mode = w7;
    std::array<float, 2> med7 = {};
    std::array<float, 2> mode7 = {};
    ComputeMedianRow<float>(w7_med.data(), /*stride=*/7, med7.data(),
                            /*count=*/2, /*divisor=*/7);
    ComputeModeRow<float>(w7_mode.data(), /*stride=*/7, mode7.data(),
                          /*count=*/2, /*divisor=*/7);
    EXPECT_EQ(med7[0], 2.0f);
    EXPECT_EQ(med7[1], 5.0f);
    EXPECT_EQ(mode7[0], 2.0f);
    EXPECT_EQ(mode7[1], 5.0f);
  }
  {
    // `divisor == 25`:
    // - Window 0: 12 NaNs + 13 finite `1..13` -> `valid_n == 13`, median
    //   (`idx == 6`) is `7.0f`, mode is `1.0f` (all unique finite).
    // - Window 1: 13 NaNs + 12 finite `5.0f` -> `valid_n == 12`, median and
    //   mode both ignore NaNs and return `5.0f`.
    // - Window 2: all 25 NaNs -> both median and mode return NaN.
    std::vector<float> w(3 * 25, kNaN);
    for (int k = 0; k < 13; ++k) {
      w[2 * k] = static_cast<float>(13 - k);
    }
    for (int k = 0; k < 12; ++k) {
      w[25 + 2 * k] = 5.0f;
    }
    auto w_med = w;
    auto w_mode = w;
    std::array<float, 3> med_out = {};
    std::array<float, 3> mode_out = {};
    ComputeMedianRow<float>(w_med.data(), /*stride=*/25, med_out.data(),
                            /*count=*/3, /*divisor=*/25);
    ComputeModeRow<float>(w_mode.data(), /*stride=*/25, mode_out.data(),
                          /*count=*/3, /*divisor=*/25);
    EXPECT_EQ(med_out[0], 7.0f);
    EXPECT_EQ(med_out[1], 5.0f);
    EXPECT_THAT(med_out[2], IsNan());
    EXPECT_EQ(mode_out[0], 1.0f);
    EXPECT_EQ(mode_out[1], 5.0f);
    EXPECT_THAT(mode_out[2], IsNan());
  }
}

TEST(ComputeMedianAndModeRowTest, ComplexModeWithNaNsAndStringMode) {
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  std::vector<complex64_t> cw(2 * 25, complex64_t{kNaN, 1.0f});
  cw[3] = complex64_t{2.0f, 3.0f};
  cw[11] = complex64_t{1.0f, 9.0f};
  cw[19] = complex64_t{2.0f, 3.0f};
  std::array<complex64_t, 2> c_out = {};
  ComputeModeRow<complex64_t>(cw.data(), /*stride=*/25, c_out.data(),
                              /*count=*/2, /*divisor=*/25);
  EXPECT_EQ(c_out[0], (complex64_t{2.0f, 3.0f}));
  EXPECT_THAT(c_out[1], IsComplexNaN());

  std::array<string_t, 6> sw = {"pear", "apple", "pear",
                                "fig",  "apple", "banana"};
  std::array<string_t, 2> s_out;
  ComputeModeRow<string_t>(sw.data(), /*stride=*/3, s_out.data(), /*count=*/2,
                           /*divisor=*/3);
  EXPECT_THAT(s_out, ElementsAre("pear", "apple"));
}

TEST(FinalizeMeanIntRowTest, MagicDiv64And32ExtremeBounds) {
  constexpr uint64_t kU64Max = std::numeric_limits<uint64_t>::max();
  constexpr int64_t kI64Min = std::numeric_limits<int64_t>::min();
  constexpr int64_t kI64Max = std::numeric_limits<int64_t>::max();

  for (Index divisor :
       {Index{3}, Index{7}, Index{9}, Index{25}, Index{27}, Index{641},
        Index{6700417}, (Index{1} << 62) + 1, kI64Max}) {
    const uint64_t d = static_cast<uint64_t>(divisor);
    const UnsignedMagicDiv magic = MakeUnsignedMagicDiv(d);
    const std::array<uint64_t, 9> u_cases = {
        0, 1, d - 1, d, d + 1, 2 * d - 1, 2 * d, kU64Max - 1, kU64Max};
    for (uint64_t x : u_cases) {
      EXPECT_EQ(RoundDivUnsignedMagic(x, magic),
                static_cast<uint64_t>(RefRoundDivUnsigned(x, d)))
          << "x=" << x << " d=" << d;
    }

    const std::array<int64_t, 7> s_cases = {kI64Min, kI64Min + 1, -divisor, -1,
                                            0,       divisor,     kI64Max};
    for (int64_t x : s_cases) {
      EXPECT_EQ(RoundDivSignedMagic(x, magic), RefRoundDivSigned(x, divisor))
          << "x=" << x << " divisor=" << divisor;
    }
  }

  for (uint32_t d32 : {uint32_t{3}, uint32_t{32767}, uint32_t{0x3fffffffu}}) {
    const UnsignedMagicDiv32 m32 = MakeUnsignedMagicDiv32(d32);
    for (uint32_t x :
         {0u, 1u, d32 - 1u, d32, d32 + 1u, 0x7fffffffu, 0x80000000u}) {
      const uint32_t q =
          static_cast<uint32_t>((uint64_t{x} * m32.multiplier) >> m32.shift);
      EXPECT_EQ(q, x / d32) << "x=" << x << " d32=" << d32;
    }
  }
}

TEST(FinalizeMeanIntRowTest, NarrowIntegerOutputs8And16Bit) {
  // Verify narrow integer outputs (`int8_t`, `uint8_t`, `int16_t`, `uint16_t`),
  // including non-power-of-two 32-bit multiplier fast paths (`divisor == 3, 6,
  // 7`) and `-128 * divisor` / `255 * divisor` extrema.
  for (Index divisor : {Index{3}, Index{6}, Index{7}, Index{25}}) {
    std::vector<int64_t> i8_sums;
    std::vector<int8_t> i8_expected;
    for (int v = -128 * divisor; v <= 127 * divisor; v += 3) {
      i8_sums.push_back(v);
      i8_expected.push_back(static_cast<int8_t>(RefRoundDivSigned(v, divisor)));
    }
    std::vector<int8_t> i8_out(i8_sums.size());
    FinalizeMeanIntRow<int64_t, int8_t>(i8_sums.data(), i8_out.data(),
                                        static_cast<Index>(i8_sums.size()),
                                        divisor);
    EXPECT_EQ(i8_out, i8_expected) << "int8 divisor=" << divisor;

    std::vector<uint64_t> u8_sums;
    std::vector<uint8_t> u8_expected;
    for (uint64_t v = 0; v <= static_cast<uint64_t>(255 * divisor); v += 3) {
      u8_sums.push_back(v);
      u8_expected.push_back(static_cast<uint8_t>(
          RefRoundDivUnsigned(v, static_cast<uint64_t>(divisor))));
    }
    std::vector<uint8_t> u8_out(u8_sums.size());
    FinalizeMeanIntRow<uint64_t, uint8_t>(u8_sums.data(), u8_out.data(),
                                          static_cast<Index>(u8_sums.size()),
                                          divisor);
    EXPECT_EQ(u8_out, u8_expected) << "uint8 divisor=" << divisor;
  }

  std::array<int64_t, 4> s_sums_16 = {-15, -13, 13, 15};
  std::array<int16_t, 4> i16_out = {};
  FinalizeMeanIntRow<int64_t, int16_t>(s_sums_16.data(), i16_out.data(), 4,
                                       /*divisor=*/6);
  EXPECT_THAT(i16_out, ElementsAre(-2, -2, 2, 2));

  std::array<uint64_t, 4> u_sums_16 = {3, 9, 15, 21};
  std::array<uint16_t, 4> u16_out = {};
  FinalizeMeanIntRow<uint64_t, uint16_t>(u_sums_16.data(), u16_out.data(), 4,
                                         /*divisor=*/6);
  EXPECT_THAT(u16_out, ElementsAre(0, 2, 2, 4));
}

TEST(FinalizeMeanIntRowTest, Uint16ThresholdAnd32BitExtrema) {
  // Verify `uint16_t` straddling `0x80000000ULL / 65535 == 32768`, and 32-bit
  // `FinalizeMeanIntRow` on the 64-bit `UnsignedMagicDiv` path
  // (`max_elem_abs >= 2^31 - 1`).
  for (Index divisor : {Index{32767}, Index{32768}, Index{32769}}) {
    const uint64_t d = static_cast<uint64_t>(divisor);
    std::array<uint64_t, 3> u16_boundary = {
        uint64_t{65535} * d, uint64_t{1000} * d + d / 2, d / 2 + 1};
    std::array<uint16_t, 3> u16_expected = {
        static_cast<uint16_t>(RefRoundDivUnsigned(u16_boundary[0], d)),
        static_cast<uint16_t>(RefRoundDivUnsigned(u16_boundary[1], d)),
        static_cast<uint16_t>(RefRoundDivUnsigned(u16_boundary[2], d))};
    std::array<uint16_t, 3> u16_res = {};
    FinalizeMeanIntRow<uint64_t, uint16_t>(u16_boundary.data(), u16_res.data(),
                                           3, divisor);
    EXPECT_EQ(u16_res, u16_expected) << "divisor=" << divisor;
  }
  for (Index divisor : {Index{3}, Index{4}, Index{7}}) {
    const int64_t i32_min = std::numeric_limits<int32_t>::min();
    const int64_t i32_max = std::numeric_limits<int32_t>::max();
    std::array<int64_t, 4> i32_sums = {
        i32_min * divisor, i32_min * divisor + divisor / 2,
        i32_max * divisor - divisor / 2, i32_max * divisor};
    std::array<int32_t, 4> i32_expected = {
        static_cast<int32_t>(RefRoundDivSigned(i32_sums[0], divisor)),
        static_cast<int32_t>(RefRoundDivSigned(i32_sums[1], divisor)),
        static_cast<int32_t>(RefRoundDivSigned(i32_sums[2], divisor)),
        static_cast<int32_t>(RefRoundDivSigned(i32_sums[3], divisor))};
    std::array<int32_t, 4> i32_out = {};
    FinalizeMeanIntRow<int64_t, int32_t>(i32_sums.data(), i32_out.data(), 4,
                                         divisor);
    EXPECT_EQ(i32_out, i32_expected) << "int32 divisor=" << divisor;

    const uint64_t u32_max = std::numeric_limits<uint32_t>::max();
    const uint64_t d = static_cast<uint64_t>(divisor);
    std::array<uint64_t, 3> u32_sums = {u32_max * d, u32_max * d - d / 2,
                                        d / 2};
    std::array<uint32_t, 3> u32_expected = {
        static_cast<uint32_t>(RefRoundDivUnsigned(u32_sums[0], d)),
        static_cast<uint32_t>(RefRoundDivUnsigned(u32_sums[1], d)),
        static_cast<uint32_t>(RefRoundDivUnsigned(u32_sums[2], d))};
    std::array<uint32_t, 3> u32_out = {};
    FinalizeMeanIntRow<uint64_t, uint32_t>(u32_sums.data(), u32_out.data(), 3,
                                           divisor);
    EXPECT_EQ(u32_out, u32_expected) << "uint32 divisor=" << divisor;
  }
}

TEST(ComputeMedianAndModeRowTest, UniformWindowsFastPath) {
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  EXPECT_TRUE(IsUniformWindow(
      std::array<uint64_t, 8>{7, 7, 7, 7, 7, 7, 7, 7}.data(), 8));
  EXPECT_FALSE(IsUniformWindow(
      std::array<uint64_t, 8>{7, 7, 7, 9, 7, 7, 7, 7}.data(), 8));
  EXPECT_FALSE(IsUniformWindow(
      std::array<float, 6>{kNaN, kNaN, kNaN, kNaN, kNaN, kNaN}.data(), 6));

  // Uniform vs non-uniform `uint64_t` segmentation windows (`divisor == 8` and
  // `divisor == 27`).
  for (Index divisor : {Index{8}, Index{27}}) {
    std::vector<uint64_t> seg(3 * divisor, 42);
    seg[divisor + divisor / 2] = 99;
    for (Index k = 0; k < divisor / 2 + 1; ++k) {
      seg[2 * divisor + k] = 100;
    }
    auto seg_med = seg;
    auto seg_mode = seg;
    std::array<uint64_t, 3> med_out = {};
    std::array<uint64_t, 3> mode_out = {};
    ComputeMedianRow<uint64_t>(seg_med.data(), divisor, med_out.data(), 3,
                               divisor);
    ComputeModeRow<uint64_t>(seg_mode.data(), divisor, mode_out.data(), 3,
                             divisor);
    EXPECT_THAT(med_out, ElementsAre(42, 42, 100));
    EXPECT_THAT(mode_out, ElementsAre(42, 42, 100));
  }
}

TEST(ComputeMedianAndModeRowTest, ClippedStridedFloatNaNsSmallWindows) {
  // Clipped boundary float windows (`stride > divisor` for `divisor` 2, 3, 4)
  // with partial NaNs and sentinel values in `[divisor, stride)`.
  constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
  for (Index divisor : {Index{2}, Index{3}, Index{4}}) {
    const Index stride = 6;
    std::array<float, 12> strided_buf;
    strided_buf.fill(-999.0f);
    // Window 0: `{kNaN, 8.0f, 3.0f, 8.0f}`; Window 1: all `kNaN` in `[0..div)`.
    const std::array<float, 4> w0 = {kNaN, 8.0f, 3.0f, 8.0f};
    std::copy_n(w0.data(), divisor, strided_buf.data());
    std::fill_n(strided_buf.data() + stride, divisor, kNaN);

    std::vector<float> contig_buf(2 * divisor);
    std::copy_n(w0.data(), divisor, contig_buf.data());
    std::fill_n(contig_buf.data() + divisor, divisor, kNaN);

    auto s_med = strided_buf;
    auto s_mode = strided_buf;
    auto c_med = contig_buf;
    auto c_mode = contig_buf;
    std::array<float, 2> med_s = {}, mode_s = {}, med_c = {}, mode_c = {};
    ComputeMedianRow<float>(s_med.data(), stride, med_s.data(), 2, divisor);
    ComputeModeRow<float>(s_mode.data(), stride, mode_s.data(), 2, divisor);
    ComputeMedianRow<float>(c_med.data(), divisor, med_c.data(), 2, divisor);
    ComputeModeRow<float>(c_mode.data(), divisor, mode_c.data(), 2, divisor);
    EXPECT_EQ(med_s[0], med_c[0]) << "divisor=" << divisor;
    EXPECT_EQ(mode_s[0], mode_c[0]) << "divisor=" << divisor;
    EXPECT_THAT(med_s[1], IsNan());
    EXPECT_THAT(mode_s[1], IsNan());
  }
}

TEST(ComputeMedianAndModeRowTest, Histogram8BitSignedAndUnsigned) {
  // Exercise 8-bit 256-bin histogram path (`divisor == 64 > 32`) on signed
  // `int8_t` (verifying `^ 0x80` sign-bit bijection preserves negative ordering
  // and smallest-value tie-breaking).
  std::array<int8_t, 64> i8_w;
  for (int k = 0; k < 64; ++k) {
    i8_w[k] = static_cast<int8_t>(-100 + k * 3);
  }
  // Create a 2-way mode tie at `-70` and `20`; `-70` must win.
  i8_w[62] = -70;
  i8_w[63] = 20;
  auto i8_med_buf = i8_w;
  auto i8_mode_buf = i8_w;
  auto i8_sorted = i8_w;
  std::sort(i8_sorted.begin(), i8_sorted.end());
  int8_t i8_med = 0;
  int8_t i8_mode = 0;
  ComputeMedianRow<int8_t>(i8_med_buf.data(), 64, &i8_med, 1, 64);
  ComputeModeRow<int8_t>(i8_mode_buf.data(), 64, &i8_mode, 1, 64);
  EXPECT_EQ(i8_med, i8_sorted[31]);
  EXPECT_EQ(i8_mode, -70);

  // Verify `MedianHistogram8Bit` `return 255` boundary (`divisor == 35 > 32`,
  // 17 elements `< 255` and 18 elements `== 255` -> `median_idx == 17` lands in
  // bin `255`).
  std::array<uint8_t, 35> u8_top;
  u8_top.fill(255);
  for (int k = 0; k < 17; ++k) u8_top[k] = static_cast<uint8_t>(k);
  uint8_t u8_med_top = 0;
  ComputeMedianRow<uint8_t>(u8_top.data(), 35, &u8_med_top, 1, 35);
  EXPECT_EQ(u8_med_top, 255);
}

TEST(ComputeMedianAndModeRowTest, FloatSignedZeroCoalescingAndPreservation) {
  // `-0.0f` and `+0.0f` must coalesce as equal in `ComputeModeRow<float>`
  // across `divisor == 3` and `divisor == 7` (beating a singleton `-1.0f`) and
  // retain the sign bit of the first zero in the window.
  std::array<float, 3> zero3_neg_first = {-1.0f, -0.0f, +0.0f};
  float zero3_neg_mode = -99.0f;
  ComputeModeRow<float>(zero3_neg_first.data(), 3, &zero3_neg_mode, 1, 3);
  EXPECT_EQ(zero3_neg_mode, 0.0f);
  EXPECT_TRUE(std::signbit(zero3_neg_mode));

  std::array<float, 3> zero3_pos_first = {-1.0f, +0.0f, -0.0f};
  float zero3_pos_mode = -99.0f;
  ComputeModeRow<float>(zero3_pos_first.data(), 3, &zero3_pos_mode, 1, 3);
  EXPECT_EQ(zero3_pos_mode, 0.0f);
  EXPECT_FALSE(std::signbit(zero3_pos_mode));

  std::array<float, 7> zero7 = {-5.0f, -3.0f, -1.0f, -0.0f, +0.0f, 2.0f, 4.0f};
  float zero7_mode = -99.0f;
  ComputeModeRow<float>(zero7.data(), 7, &zero7_mode, 1, 7);
  EXPECT_EQ(zero7_mode, 0.0f);
  EXPECT_TRUE(std::signbit(zero7_mode));

  // All-`-0.0f` windows in both `ComputeMedianRow<float>` and
  // `ComputeModeRow<float>` preserve `-0.0f`.
  std::array<float, 2> neg_zeros_med = {-0.0f, -0.0f};
  float med_neg_zero = +1.0f;
  ComputeMedianRow<float>(neg_zeros_med.data(), 2, &med_neg_zero, 1, 2);
  EXPECT_EQ(med_neg_zero, 0.0f);
  EXPECT_TRUE(std::signbit(med_neg_zero));

  std::array<float, 2> neg_zeros_mode = {-0.0f, -0.0f};
  float mode_neg_zero = +1.0f;
  ComputeModeRow<float>(neg_zeros_mode.data(), 2, &mode_neg_zero, 1, 2);
  EXPECT_EQ(mode_neg_zero, 0.0f);
  EXPECT_TRUE(std::signbit(mode_neg_zero));
}

TEST(ComputeMedianAndModeRowTest, LargeWindowShellSortAndNthElement16To64Bit) {
  // Exercise `ModeWindowUnsigned` (`ShellSort` across multi-gap Ciura sequence
  // at `divisor == 150 > 132`) and `MedianWindowUnsigned` (`std::nth_element`
  // at `divisor == 150 > 20`) on 16-bit, 32-bit, and 64-bit integer and
  // floating-point types.
  constexpr Index kDiv = 150;
  std::vector<int16_t> i16_w(kDiv);
  std::vector<uint32_t> u32_w(kDiv);
  std::vector<int64_t> i64_w(kDiv);
  std::vector<double> f64_w(kDiv);
  for (Index k = 0; k < kDiv - 1; ++k) {
    const int val = static_cast<int>(100 - k);
    i16_w[k] = static_cast<int16_t>(val);
    u32_w[k] = static_cast<uint32_t>(500 - k);
    i64_w[k] = static_cast<int64_t>(val) * 1000;
    f64_w[k] = static_cast<double>(val) * 0.5;
  }
  // Duplicate one element so the mode is unique (`-17`, `383`, `-17000`,
  // `-8.5`).
  i16_w[kDiv - 1] = -17;
  u32_w[kDiv - 1] = 383;
  i64_w[kDiv - 1] = -17000;
  f64_w[kDiv - 1] = -8.5;

  auto i16_sorted = i16_w;
  std::sort(i16_sorted.begin(), i16_sorted.end());
  auto i16_med_buf = i16_w;
  auto i16_mode_buf = i16_w;
  int16_t i16_med = 0, i16_mode = 0;
  ComputeMedianRow<int16_t>(i16_med_buf.data(), kDiv, &i16_med, 1, kDiv);
  ComputeModeRow<int16_t>(i16_mode_buf.data(), kDiv, &i16_mode, 1, kDiv);
  EXPECT_EQ(i16_med, i16_sorted[(kDiv - 1) / 2]);
  EXPECT_EQ(i16_mode, -17);

  auto u32_sorted = u32_w;
  std::sort(u32_sorted.begin(), u32_sorted.end());
  auto u32_med_buf = u32_w;
  auto u32_mode_buf = u32_w;
  uint32_t u32_med = 0, u32_mode = 0;
  ComputeMedianRow<uint32_t>(u32_med_buf.data(), kDiv, &u32_med, 1, kDiv);
  ComputeModeRow<uint32_t>(u32_mode_buf.data(), kDiv, &u32_mode, 1, kDiv);
  EXPECT_EQ(u32_med, u32_sorted[(kDiv - 1) / 2]);
  EXPECT_EQ(u32_mode, 383u);

  auto i64_sorted = i64_w;
  std::sort(i64_sorted.begin(), i64_sorted.end());
  auto i64_med_buf = i64_w;
  auto i64_mode_buf = i64_w;
  int64_t i64_med = 0, i64_mode = 0;
  ComputeMedianRow<int64_t>(i64_med_buf.data(), kDiv, &i64_med, 1, kDiv);
  ComputeModeRow<int64_t>(i64_mode_buf.data(), kDiv, &i64_mode, 1, kDiv);
  EXPECT_EQ(i64_med, i64_sorted[(kDiv - 1) / 2]);
  EXPECT_EQ(i64_mode, -17000);

  auto f64_sorted = f64_w;
  std::sort(f64_sorted.begin(), f64_sorted.end());
  auto f64_med_buf = f64_w;
  auto f64_mode_buf = f64_w;
  double f64_med = 0.0, f64_mode = 0.0;
  ComputeMedianRow<double>(f64_med_buf.data(), kDiv, &f64_med, 1, kDiv);
  ComputeModeRow<double>(f64_mode_buf.data(), kDiv, &f64_mode, 1, kDiv);
  EXPECT_EQ(f64_med, f64_sorted[(kDiv - 1) / 2]);
  EXPECT_EQ(f64_mode, -8.5);
}

TEST(ComputeMedianAndModeRowTest, GenericModeLargeJsonAndUstringFastPaths) {
  // Exercise `ComputeModeGenericRow` (`ShellSort` with multi-gap Ciura sequence
  // at `divisor == 150 > 132`) on `json_t`.
  std::vector<json_t> jw;
  jw.reserve(150);
  for (int k = 0; k < 149; ++k) {
    jw.push_back(json_t{{"id", 200 - k}});
  }
  jw.push_back(json_t{{"id", 137}});
  json_t j_out;
  ComputeModeRow<json_t>(jw.data(), 150, &j_out, 1, 150);
  EXPECT_EQ(j_out, (json_t{{"id", 137}}));

  // Exercise `ComputeModeRow` `divisor == 1` and `IsUniformWindow`
  // (`divisor == 3`) on `ustring_t`.
  std::array<ustring_t, 4> ustr_staged = {ustring_t{"alpha"}, ustring_t{"beta"},
                                          ustring_t{"gamma"},
                                          ustring_t{"delta"}};
  ustring_t ustr_div1;
  ComputeModeRow<ustring_t>(ustr_staged.data(), /*stride=*/4, &ustr_div1,
                            /*count=*/1, /*divisor=*/1);
  EXPECT_EQ(ustr_div1, ustring_t{"alpha"});

  std::array<ustring_t, 3> ustr_uniform = {ustring_t{"same"}, ustring_t{"same"},
                                           ustring_t{"same"}};
  ustring_t ustr_uni_out;
  ComputeModeRow<ustring_t>(ustr_uniform.data(), /*stride=*/3, &ustr_uni_out,
                            /*count=*/1, /*divisor=*/3);
  EXPECT_EQ(ustr_uni_out, ustring_t{"same"});
}

}  // namespace
