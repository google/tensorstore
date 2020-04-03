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
using tensorstore::BitIterator;
using tensorstore::BitRef;
using tensorstore::BitSpan;
using tensorstore::BitVectorSizeInBlocks;

static_assert(std::is_convertible<BitRef<std::uint32_t>,
                                  BitRef<const std::uint32_t>>::value,
              "");

static_assert(!std::is_convertible<BitRef<const std::uint32_t>,
                                   BitRef<std::uint32_t>>::value,
              "");

static_assert(std::is_convertible<BitIterator<std::uint32_t>,
                                  BitIterator<const std::uint32_t>>::value,
              "");

static_assert(!std::is_convertible<BitIterator<const std::uint32_t>,
                                   BitIterator<std::uint32_t>>::value,
              "");

static_assert(std::is_convertible<BitSpan<std::uint32_t>,
                                  BitSpan<const std::uint32_t>>::value,
              "");

static_assert(std::is_convertible<BitSpan<const std::uint32_t, 3>,
                                  BitSpan<const std::uint32_t>>::value,
              "");

static_assert(std::is_convertible<BitSpan<std::uint32_t, 3>,
                                  BitSpan<const std::uint32_t>>::value,
              "");

static_assert(!std::is_assignable<BitRef<const std::uint32_t>, bool>::value,
              "");

TEST(BitRefTest, Basic) {
  std::uint16_t data[2] = {0, 0};
  BitRef<std::uint16_t> ref(data + 1, 19);
  BitRef<std::uint16_t> ref2(data, 2);
  BitRef<const std::uint16_t> const_ref(data, 3);
  EXPECT_EQ(false, ref);
  ref = true;
  EXPECT_EQ(true, ref);
  EXPECT_THAT(data, ::testing::ElementsAre(0, 8));
  data[0] = 0xffff /*=0b1111111111111111*/;
  data[1] = 0xffff /*=0b1111111111111111*/;
  EXPECT_EQ(true, ref);
  ref = false;
  EXPECT_EQ(false, ref);
  EXPECT_THAT(data, ::testing::ElementsAre(0xffff /*=0b1111111111111111*/,
                                           0xfff7 /*=0b1111111111110111*/));
  ref = ref2;
  EXPECT_THAT(data, ::testing::ElementsAre(0xffff /*=0b1111111111111111*/,
                                           0xffff /*=0b1111111111111111*/));
  data[0] = 0;
  ref = const_ref;
  EXPECT_THAT(data, ::testing::ElementsAre(0, 0xfff7 /*=0b1111111111110111*/));
}

TEST(BitRefTest, Swap) {
  std::uint16_t data[2] = {0, 0};
  BitRef<std::uint16_t> ref(data + 1, 19);
  BitRef<std::uint16_t> ref2(data, 2);
  std::uint32_t data2 = 0;
  ref = true;
  ref2 = false;
  EXPECT_THAT(data, ::testing::ElementsAre(0, 8));
  using std::swap;
  swap(ref, ref2);

  EXPECT_EQ(false, ref);
  EXPECT_EQ(true, ref2);
  EXPECT_THAT(data, ::testing::ElementsAre(4, 0));

  bool b = false;
  swap(b, ref2);
  EXPECT_THAT(data, ::testing::ElementsAre(0, 0));
  EXPECT_EQ(true, b);
  swap(ref2, b);
  EXPECT_THAT(data, ::testing::ElementsAre(4, 0));
  EXPECT_EQ(false, b);
  BitRef<std::uint32_t> ref3(&data2, 1);
  swap(ref2, ref3);
  EXPECT_THAT(data, ::testing::ElementsAre(0, 0));
  EXPECT_EQ(2, data2);
}

TEST(BitIteratorTest, Basic) {
  std::uint16_t data[2] = {0, 0};
  BitIterator<std::uint16_t> it(data, 19);
  BitIterator<std::uint16_t> it2(data, 2);
  BitIterator<const std::uint16_t> const_it(data, 3);
  BitIterator<const std::uint16_t> const_it2 = it;
  EXPECT_EQ(data, it.base());
  EXPECT_EQ(data, it2.base());
  EXPECT_EQ(data, const_it.base());
  EXPECT_EQ(data, const_it2.base());
  EXPECT_EQ(19, it.offset());
  EXPECT_EQ(2, it2.offset());
  EXPECT_EQ(3, const_it.offset());
  EXPECT_EQ(19, const_it2.offset());

  {
    auto ref = *it;
    static_assert(std::is_same<BitRef<std::uint16_t>, decltype(ref)>::value,
                  "");
    auto ref_subscript = it[0];
    auto ref_subscript2 = it2[17];
    static_assert(
        std::is_same<BitRef<std::uint16_t>, decltype(ref_subscript)>::value,
        "");
    EXPECT_FALSE(ref_subscript);
    EXPECT_FALSE(ref_subscript2);
    ref = true;
    EXPECT_TRUE(ref);
    EXPECT_TRUE(ref_subscript);
    EXPECT_TRUE(ref_subscript2);
    EXPECT_THAT(data, ::testing::ElementsAre(0, 0x8 /*=0b1000*/));
    ref = false;
    EXPECT_FALSE(ref);
    EXPECT_THAT(data, ::testing::ElementsAre(0, 0));
    data[1] = ~0x8;
    EXPECT_FALSE(ref);
    ref = true;
    EXPECT_TRUE(ref);
    EXPECT_THAT(data, ::testing::ElementsAre(0, 0xffff));
  }

  {
    auto ref = *const_it;
    static_assert(
        std::is_same<BitRef<const std::uint16_t>, decltype(ref)>::value, "");
    EXPECT_FALSE(ref);
    data[0] = 0x8 /*=0b1000*/;
    EXPECT_TRUE(ref);
    data[0] = ~data[0];
    EXPECT_FALSE(ref);
  }
}

TEST(BitIteratorTest, IteratorPlusOffset) {
  std::uint16_t data[2] = {0, 0};
  auto it = BitIterator<std::uint16_t>(data, 3) + 5;
  EXPECT_EQ(data, it.base());
  EXPECT_EQ(8, it.offset());
}

TEST(BitIteratorTest, OffsetPlusIterator) {
  std::uint16_t data[2] = {0, 0};
  auto it = 5 + BitIterator<std::uint16_t>(data, 3);
  EXPECT_EQ(data, it.base());
  EXPECT_EQ(8, it.offset());
}

TEST(BitIteratorTest, IteratorMinusOffset) {
  std::uint16_t data[2] = {0, 0};
  auto it = BitIterator<std::uint16_t>(data, 7) - 2;
  EXPECT_EQ(data, it.base());
  EXPECT_EQ(5, it.offset());
}

TEST(BitIteratorTest, IteratorMinusIterator) {
  std::uint16_t data[2] = {0, 0};
  EXPECT_EQ(3, BitIterator<std::uint16_t>(data, 7) -
                   BitIterator<std::uint16_t>(data, 4));
  EXPECT_EQ(3, BitIterator<std::uint16_t>(data, 7) -
                   BitIterator<const std::uint16_t>(data, 4));
  EXPECT_EQ(3, BitIterator<const std::uint16_t>(data, 7) -
                   BitIterator<std::uint16_t>(data, 4));
  EXPECT_EQ(3, BitIterator<const std::uint16_t>(data, 7) -
                   BitIterator<const std::uint16_t>(data, 4));
}

TEST(BitIteratorTest, PreIncrement) {
  std::uint16_t data[2] = {0, 0};
  BitIterator<std::uint16_t> it(data, 19);
  auto& x = ++it;
  EXPECT_EQ(&it, &x);
  EXPECT_EQ(20, it.offset());
  EXPECT_EQ(data, it.base());
}

TEST(BitIteratorTest, PreDecrement) {
  std::uint16_t data[2] = {0, 0};
  BitIterator<std::uint16_t> it(data, 19);
  auto& x = --it;
  EXPECT_EQ(&it, &x);
  EXPECT_EQ(18, it.offset());
  EXPECT_EQ(data, it.base());
}

TEST(BitIteratorTest, PostIncrement) {
  std::uint16_t data[2] = {0, 0};
  BitIterator<std::uint16_t> it(data, 19);
  EXPECT_EQ(BitIterator<std::uint16_t>(data, 19), it++);
  EXPECT_EQ(20, it.offset());
  EXPECT_EQ(data, it.base());
}

TEST(BitIteratorTest, PostDecrement) {
  std::uint16_t data[2] = {0, 0};
  BitIterator<std::uint16_t> it(data, 19);
  EXPECT_EQ(BitIterator<std::uint16_t>(data, 19), it--);
  EXPECT_EQ(18, it.offset());
  EXPECT_EQ(data, it.base());
}

TEST(BitIteratorTest, Comparison) {
  std::uint16_t data[2] = {0, 0};
  EXPECT_EQ(BitIterator<std::uint16_t>(data, 3),
            BitIterator<std::uint16_t>(data, 3));
  EXPECT_EQ(BitIterator<std::uint16_t>(data, 3),
            BitIterator<const std::uint16_t>(data, 3));
  EXPECT_NE(BitIterator<std::uint16_t>(data, 3),
            BitIterator<std::uint16_t>(data, 4));
  EXPECT_NE(BitIterator<std::uint16_t>(data, 3),
            BitIterator<const std::uint16_t>(data, 4));
  EXPECT_TRUE(BitIterator<std::uint16_t>(data, 3) <
              BitIterator<std::uint16_t>(data, 4));
  EXPECT_TRUE(BitIterator<std::uint16_t>(data, 3) <
              BitIterator<const std::uint16_t>(data, 4));
  EXPECT_FALSE(BitIterator<std::uint16_t>(data, 3) <
               BitIterator<std::uint16_t>(data, 3));
  EXPECT_TRUE(BitIterator<std::uint16_t>(data, 3) <=
              BitIterator<std::uint16_t>(data, 4));
  EXPECT_TRUE(BitIterator<std::uint16_t>(data, 3) <=
              BitIterator<std::uint16_t>(data, 3));
  EXPECT_FALSE(BitIterator<std::uint16_t>(data, 3) <=
               BitIterator<std::uint16_t>(data, 2));
  EXPECT_TRUE(BitIterator<std::uint16_t>(data, 3) >
              BitIterator<std::uint16_t>(data, 2));
  EXPECT_FALSE(BitIterator<std::uint16_t>(data, 3) >
               BitIterator<std::uint16_t>(data, 3));
  EXPECT_TRUE(BitIterator<std::uint16_t>(data, 3) >=
              BitIterator<std::uint16_t>(data, 2));
  EXPECT_TRUE(BitIterator<std::uint16_t>(data, 3) >=
              BitIterator<std::uint16_t>(data, 3));
  EXPECT_FALSE(BitIterator<std::uint16_t>(data, 2) >=
               BitIterator<std::uint16_t>(data, 3));
}

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
