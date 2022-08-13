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

#include "tensorstore/util/bit_span.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <type_traits>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace {
using ::tensorstore::BitIterator;
using ::tensorstore::BitSpan;
using ::tensorstore::BitVectorSizeInBlocks;

static_assert(std::is_convertible_v<BitSpan<std::uint32_t>,
                                    BitSpan<const std::uint32_t>>);

static_assert(std::is_convertible_v<BitSpan<const std::uint32_t, 3>,
                                    BitSpan<const std::uint32_t>>);

static_assert(std::is_convertible_v<BitSpan<std::uint32_t, 3>,
                                    BitSpan<const std::uint32_t>>);

TEST(BitSpanTest, Basic) {
  std::uint16_t data[2] = {0, 0};
  BitSpan<std::uint16_t> s(data, 11, 10);
  EXPECT_EQ(10, s.size());
  EXPECT_EQ(data, s.base());
  EXPECT_EQ(11, s.offset());
}

TEST(BitSpanTest, ConstructFromIterator) {
  std::uint16_t data[2] = {0, 0};
  BitSpan<std::uint16_t> s(BitIterator<std::uint16_t>(data, 11), 10);
  EXPECT_EQ(10, s.size());
  EXPECT_EQ(data, s.base());
  EXPECT_EQ(11, s.offset());
}

TEST(BitSpanTest, Iterate) {
  std::uint16_t data[2] = {0, 0};
  BitSpan<std::uint16_t> s(data, 11, 10);
  std::array<bool, 10> arr = {1, 1, 0, 0, 1, 1, 1, 0, 1, 0};
  std::copy(arr.begin(), arr.end(), s.begin());
  EXPECT_THAT(data, ::testing::ElementsAre(0x9800 /*=0b1001100000000000*/,
                                           0xb /*=0b01011*/));
  std::array<bool, 10> arr2;
  std::copy(s.begin(), s.end(), arr2.begin());
  EXPECT_EQ(arr, arr2);
  std::sort(s.begin(), s.end());
  EXPECT_THAT(s, ::testing::ElementsAre(0, 0, 0, 0, 1, 1, 1, 1, 1, 1));
}

TEST(BitSpanTest, Convert) {
  std::uint16_t data[2] = {0, 0};
  BitSpan<std::uint16_t, 10> s_static(data, 11, 10);
  BitSpan<std::uint16_t> s2 = s_static;
  BitSpan<const std::uint16_t> s2_const = s2;
  EXPECT_EQ(data, s_static.base());
  EXPECT_EQ(11, s_static.offset());
  EXPECT_EQ(10, s_static.size());
  EXPECT_EQ(data, s2.base());
  EXPECT_EQ(11, s2.offset());
  EXPECT_EQ(10, s2.size());
  EXPECT_EQ(data, s2_const.base());
  EXPECT_EQ(11, s2_const.offset());
  EXPECT_EQ(10, s2_const.size());
}

TEST(BitSpanTest, FillPartialSingleBlockTrue) {
  std::uint16_t data[2] = {0xaaaa, 0xaaaa};
  BitSpan<std::uint16_t>(data, 10, 4).fill(true);
  EXPECT_THAT(data,
              ::testing::ElementsAre(0xbeaa /*=0b1011111010101010*/, 0xaaaa));
}

TEST(BitSpanTest, FillPartialSingleBlockFalse) {
  std::uint16_t data[2] = {0xaaaa, 0xaaaa};
  BitSpan<std::uint16_t>(data, 11, 4).fill(false);
  EXPECT_THAT(data,
              ::testing::ElementsAre(0x82aa /*=0b1000001010101010*/, 0xaaaa));
}

TEST(BitSpanTest, FillPartialTwoBlocksTrue) {
  std::uint16_t data[2] = {0xaaaa, 0xaaaa};
  BitSpan<std::uint16_t>(data, 11, 10).fill(true);
  EXPECT_THAT(data, ::testing::ElementsAre(0xfaaa /*=0b1111101010101010*/,
                                           0xaabf /*=0b1010101010111111*/));
}

TEST(BitSpanTest, FillPartialTwoBlocksFalse) {
  std::uint16_t data[2] = {0xaaaa, 0xaaaa};
  BitSpan<std::uint16_t>(data, 11, 10).fill(false);
  EXPECT_THAT(data, ::testing::ElementsAre(0x02aa /*=0b0000001010101010*/,
                                           0xaaa0 /*=0b1010101010100000*/));
}

TEST(BitSpanTest, FillOneBlockExactEndTrue) {
  std::uint16_t data[2] = {0xaaaa, 0xaaaa};
  BitSpan<std::uint16_t>(data, 13, 3).fill(true);
  EXPECT_THAT(data, ::testing::ElementsAre(0xeaaa /*=0b1110101010101010*/,
                                           0xaaaa /*=0b1010101010101010*/));
}

TEST(BitSpanTest, FillOneBlockExactEndFalse) {
  std::uint16_t data[2] = {0xaaaa, 0xaaaa};
  BitSpan<std::uint16_t>(data, 13, 3).fill(false);
  EXPECT_THAT(data, ::testing::ElementsAre(0x0aaa /*=0b0000101010101010*/,
                                           0xaaaa /*=0b1010101010101010*/));
}

TEST(BitSpanTest, FillTwoBlockExactEndTrue) {
  std::uint16_t data[3] = {0xaaaa, 0xaaaa, 0xaaaa};
  BitSpan<std::uint16_t>(data, 13, 19).fill(true);
  EXPECT_THAT(data, ::testing::ElementsAre(0xeaaa /*=0b1110101010101010*/,
                                           0xffff /*=0b1111111111111111*/,
                                           0xaaaa /*=0b1010101010101010*/));
}

TEST(BitSpanTest, FillTwoBlockExactEndFalse) {
  std::uint16_t data[3] = {0xaaaa, 0xaaaa, 0xaaaa};
  BitSpan<std::uint16_t>(data, 13, 19).fill(false);
  EXPECT_THAT(data, ::testing::ElementsAre(0x0aaa /*=0b0000101010101010*/,
                                           0x0000 /*=0b0000000000000000*/,
                                           0xaaaa /*=0b1010101010101010*/));
}

TEST(BitSpanTest, FillPartialThreeBlocksTrue) {
  std::uint16_t data[3] = {0xaaaa, 0xaaaa, 0xaaaa};
  BitSpan<std::uint16_t>(data, 11, 23).fill(true);
  EXPECT_THAT(data, ::testing::ElementsAre(0xfaaa /*=0b1111101010101010*/,
                                           0xffff /*=0b1111111111111111*/,
                                           0xaaab /*=0b1010101010101011*/));
}

TEST(BitSpanTest, FillPartialThreeBlocksFalse) {
  std::uint16_t data[3] = {0xaaaa, 0xaaaa, 0xaaaa};
  BitSpan<std::uint16_t>(data, 11, 23).fill(false);
  EXPECT_THAT(data, ::testing::ElementsAre(0x02aa /*=0b0000001010101010*/,
                                           0x0000 /*=0b0000000000000000*/,
                                           0xaaa8 /*=0b1010101010101000*/));
}

TEST(BitSpanTest, DeepAssign) {
  std::uint16_t data[2] = {0x9e0e /*=0b1001111000001110*/,
                           0xe1f1 /*=0b1110000111110001*/};
  BitSpan<std::uint16_t> s1(data, 11, 10);

  std::uint16_t data2[2] = {0x1e0e /*=0b0001111000001110*/,
                            0xe1f1 /*=0b1110000111110001*/};
  BitSpan<std::uint16_t> s2(data2, 9, 10);

  s2.DeepAssign(s1);
  EXPECT_THAT(data, ::testing::ElementsAre(0x9e0e /*=0b1001111000001110*/,
                                           0xe1f1 /*0b1110000111110001*/));
  EXPECT_THAT(data2, ::testing::ElementsAre(0x660e /*=0b0110011000001110*/,
                                            0xe1f4 /*=0b1110000111110100*/));
}

static_assert(BitVectorSizeInBlocks<std::uint64_t>(0) == 0, "");
static_assert(BitVectorSizeInBlocks<std::uint64_t>(1) == 1, "");
static_assert(BitVectorSizeInBlocks<std::uint64_t>(63) == 1, "");
static_assert(BitVectorSizeInBlocks<std::uint64_t>(64) == 1, "");
static_assert(BitVectorSizeInBlocks<std::uint64_t>(65) == 2, "");
static_assert(BitVectorSizeInBlocks<std::uint32_t>(65) == 3, "");

}  // namespace
