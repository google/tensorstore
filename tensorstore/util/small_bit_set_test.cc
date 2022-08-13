// Copyright 2021 The TensorStore Authors
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

#include "tensorstore/util/small_bit_set.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/util/str_cat.h"

namespace {

using ::tensorstore::BitIterator;
using ::tensorstore::BitRef;
using BitSet = ::tensorstore::SmallBitSet<32>;

static_assert(
    std::is_convertible_v<BitRef<std::uint32_t>, BitRef<const std::uint32_t>>);

static_assert(
    !std::is_convertible_v<BitRef<const std::uint32_t>, BitRef<std::uint32_t>>);

static_assert(std::is_convertible_v<BitIterator<std::uint32_t>,
                                    BitIterator<const std::uint32_t>>);

static_assert(!std::is_convertible_v<BitIterator<const std::uint32_t>,
                                     BitIterator<std::uint32_t>>);

static_assert(!std::is_assignable_v<BitRef<const std::uint32_t>, bool>);

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
    static_assert(std::is_same_v<BitRef<std::uint16_t>, decltype(ref)>);
    auto ref_subscript = it[0];
    auto ref_subscript2 = it2[17];
    static_assert(
        std::is_same_v<BitRef<std::uint16_t>, decltype(ref_subscript)>);
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
    static_assert(std::is_same_v<BitRef<const std::uint16_t>, decltype(ref)>);
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

TEST(SmallBitSetTest, DefaultConstruct) {
  BitSet v;
  EXPECT_FALSE(v);
  EXPECT_EQ(0, v.bits());
  EXPECT_EQ(v, v);
  BitSet v_true = true;
  EXPECT_EQ(v_true, v_true);
  EXPECT_NE(v, v_true);
  EXPECT_THAT(v, ::testing::ElementsAreArray(std::vector<bool>(32)));
}

TEST(SmallBitSetTest, FromBits) {
  auto v = BitSet::FromBits(0b11'0111);
  EXPECT_TRUE(v);
  EXPECT_EQ(0b110111, v.bits());
  EXPECT_EQ(true, v[0]);
  EXPECT_EQ(true, v[1]);
  EXPECT_EQ(true, v[2]);
  EXPECT_EQ(false, v[3]);
  EXPECT_EQ(true, v[4]);
  EXPECT_EQ(true, v[5]);
  EXPECT_THAT(v, ::testing::ElementsAre(1, 1, 1, 0, 1, 1, 0, 0,  //
                                        0, 0, 0, 0, 0, 0, 0, 0,  //
                                        0, 0, 0, 0, 0, 0, 0, 0,  //
                                        0, 0, 0, 0, 0, 0, 0, 0));
  EXPECT_EQ(
      "11101100"
      "00000000"
      "00000000"
      "00000000",
      tensorstore::StrCat(v));
  EXPECT_EQ(0b11111111'11111111'11111111'11001000, (~v).bits());
  auto v1 = BitSet::FromBits(0b101'1100);
  EXPECT_EQ(0b111'1111, (v | v1).bits());
  EXPECT_EQ(0b001'0100, (v & v1).bits());
  EXPECT_EQ(0b110'1011, (v ^ v1).bits());
  auto v2 = v1;
  v2 |= v;
  EXPECT_EQ(0b111'1111, v2.bits());
  v2 = v1;
  v2 &= v;
  EXPECT_EQ(0b001'0100, v2.bits());
  v2 = v1;
  v2 ^= v;
  EXPECT_EQ(0b110'1011, v2.bits());
}

TEST(SmallBitSetTest, BracedList) {
  auto v = BitSet({0, 1, 1, 0, 0, 1});
  EXPECT_EQ(0b100110, v.bits());
}

TEST(SmallBitSetTest, Reference) {
  BitSet v;
  v[2] = true;
  EXPECT_TRUE(v[2]);
  EXPECT_FALSE(v[0]);
  EXPECT_EQ(0b100, v.bits());
}

TEST(SmallBitSetTest, UpTo) {
  EXPECT_EQ(0x00000000, BitSet::UpTo(0).bits());
  EXPECT_EQ(0x00000001, BitSet::UpTo(1).bits());
  EXPECT_EQ(0x0000ffff, BitSet::UpTo(16).bits());
  EXPECT_EQ(0x7fffffff, BitSet::UpTo(31).bits());
  EXPECT_EQ(0xffffffff, BitSet::UpTo(32).bits());
}

}  // namespace
