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

#include <stdint.h>

#include <type_traits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/util/str_cat.h"

namespace {

using ::tensorstore::BitIterator;
using ::tensorstore::BitRef;
using BitSet = ::tensorstore::SmallBitSet<32>;

static_assert(
    std::is_convertible_v<BitIterator<uint32_t>, BitIterator<const uint32_t>>);

static_assert(
    !std::is_convertible_v<BitIterator<const uint32_t>, BitIterator<uint32_t>>);

TEST(BitRefTest, Basic) {
  uint16_t data[2] = {0, 0};
  BitRef<uint16_t> ref(data + 1, 19);
  BitRef<uint16_t> ref2(data, 2);
  BitRef<const uint16_t> const_ref(data, 3);
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
  uint16_t data[2] = {0, 0};
  BitRef<uint16_t> ref(data + 1, 19);
  BitRef<uint16_t> ref2(data, 2);
  uint32_t data2 = 0;
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
  BitRef<uint32_t> ref3(&data2, 1);
  swap(ref2, ref3);
  EXPECT_THAT(data, ::testing::ElementsAre(0, 0));
  EXPECT_EQ(2, data2);
}

TEST(BitIteratorTest, Basic) {
  uint16_t data[2] = {0, 0};
  BitIterator<uint16_t> it(data, 19);
  BitIterator<uint16_t> it2(data, 2);
  BitIterator<const uint16_t> const_it(data, 3);
  BitIterator<const uint16_t> const_it2 = it;
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
    static_assert(std::is_same_v<BitRef<uint16_t>, decltype(ref)>);
    auto ref_subscript = it[0];
    auto ref_subscript2 = it2[17];
    static_assert(std::is_same_v<BitRef<uint16_t>, decltype(ref_subscript)>);
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
    static_assert(std::is_same_v<BitRef<const uint16_t>, decltype(ref)>);
    EXPECT_FALSE(ref);
    data[0] = 0x8 /*=0b1000*/;
    EXPECT_TRUE(ref);
    data[0] = ~data[0];
    EXPECT_FALSE(ref);
  }
}

TEST(BitIteratorTest, IteratorPlusOffset) {
  uint16_t data[2] = {0, 0};
  auto it = BitIterator<uint16_t>(data, 3) + 5;
  EXPECT_EQ(data, it.base());
  EXPECT_EQ(8, it.offset());
}

TEST(BitIteratorTest, OffsetPlusIterator) {
  uint16_t data[2] = {0, 0};
  auto it = 5 + BitIterator<uint16_t>(data, 3);
  EXPECT_EQ(data, it.base());
  EXPECT_EQ(8, it.offset());
}

TEST(BitIteratorTest, IteratorMinusOffset) {
  uint16_t data[2] = {0, 0};
  auto it = BitIterator<uint16_t>(data, 7) - 2;
  EXPECT_EQ(data, it.base());
  EXPECT_EQ(5, it.offset());
}

TEST(BitIteratorTest, IteratorMinusIterator) {
  uint16_t data[2] = {0, 0};
  EXPECT_EQ(3, BitIterator<uint16_t>(data, 7) - BitIterator<uint16_t>(data, 4));
  EXPECT_EQ(
      3, BitIterator<uint16_t>(data, 7) - BitIterator<const uint16_t>(data, 4));
  EXPECT_EQ(
      3, BitIterator<const uint16_t>(data, 7) - BitIterator<uint16_t>(data, 4));
  EXPECT_EQ(3, BitIterator<const uint16_t>(data, 7) -
                   BitIterator<const uint16_t>(data, 4));
}

TEST(BitIteratorTest, PreIncrement) {
  uint16_t data[2] = {0, 0};
  BitIterator<uint16_t> it(data, 19);
  auto& x = ++it;
  EXPECT_EQ(&it, &x);
  EXPECT_EQ(20, it.offset());
  EXPECT_EQ(data, it.base());
}

TEST(BitIteratorTest, PreDecrement) {
  uint16_t data[2] = {0, 0};
  BitIterator<uint16_t> it(data, 19);
  auto& x = --it;
  EXPECT_EQ(&it, &x);
  EXPECT_EQ(18, it.offset());
  EXPECT_EQ(data, it.base());
}

TEST(BitIteratorTest, PostIncrement) {
  uint16_t data[2] = {0, 0};
  BitIterator<uint16_t> it(data, 19);
  EXPECT_EQ(BitIterator<uint16_t>(data, 19), it++);
  EXPECT_EQ(20, it.offset());
  EXPECT_EQ(data, it.base());
}

TEST(BitIteratorTest, PostDecrement) {
  uint16_t data[2] = {0, 0};
  BitIterator<uint16_t> it(data, 19);
  EXPECT_EQ(BitIterator<uint16_t>(data, 19), it--);
  EXPECT_EQ(18, it.offset());
  EXPECT_EQ(data, it.base());
}

TEST(BitIteratorTest, Comparison) {
  uint16_t data[2] = {0, 0};
  EXPECT_EQ(BitIterator<uint16_t>(data, 3), BitIterator<uint16_t>(data, 3));
  EXPECT_EQ(BitIterator<uint16_t>(data, 3),
            BitIterator<const uint16_t>(data, 3));
  EXPECT_NE(BitIterator<uint16_t>(data, 3), BitIterator<uint16_t>(data, 4));
  EXPECT_NE(BitIterator<uint16_t>(data, 3),
            BitIterator<const uint16_t>(data, 4));
  EXPECT_TRUE(BitIterator<uint16_t>(data, 3) < BitIterator<uint16_t>(data, 4));
  EXPECT_TRUE(BitIterator<uint16_t>(data, 3) <
              BitIterator<const uint16_t>(data, 4));
  EXPECT_FALSE(BitIterator<uint16_t>(data, 3) < BitIterator<uint16_t>(data, 3));
  EXPECT_TRUE(BitIterator<uint16_t>(data, 3) <= BitIterator<uint16_t>(data, 4));
  EXPECT_TRUE(BitIterator<uint16_t>(data, 3) <= BitIterator<uint16_t>(data, 3));
  EXPECT_FALSE(BitIterator<uint16_t>(data, 3) <=
               BitIterator<uint16_t>(data, 2));
  EXPECT_TRUE(BitIterator<uint16_t>(data, 3) > BitIterator<uint16_t>(data, 2));
  EXPECT_FALSE(BitIterator<uint16_t>(data, 3) > BitIterator<uint16_t>(data, 3));
  EXPECT_TRUE(BitIterator<uint16_t>(data, 3) >= BitIterator<uint16_t>(data, 2));
  EXPECT_TRUE(BitIterator<uint16_t>(data, 3) >= BitIterator<uint16_t>(data, 3));
  EXPECT_FALSE(BitIterator<uint16_t>(data, 2) >=
               BitIterator<uint16_t>(data, 3));
}

TEST(SmallBitSetTest, DefaultConstruct) {
  BitSet v;
  EXPECT_FALSE(v);
  EXPECT_EQ(0, v.to_uint());
  EXPECT_EQ(v, v);
  BitSet v_true = true;
  EXPECT_EQ(v_true, v_true);
  EXPECT_NE(v, v_true);
  EXPECT_THAT(v.bools_view(),
              ::testing::ElementsAreArray(std::vector<bool>(32)));
}

TEST(SmallBitSetTest, FromUint) {
  auto v = BitSet::FromUint(0b11'0111);
  EXPECT_TRUE(v);
  EXPECT_EQ(0b110111, v.to_uint());
  EXPECT_EQ(true, v[0]);
  EXPECT_EQ(true, v[1]);
  EXPECT_EQ(true, v[2]);
  EXPECT_EQ(false, v[3]);
  EXPECT_EQ(true, v[4]);
  EXPECT_EQ(true, v[5]);
  EXPECT_THAT(v.bools_view(), ::testing::ElementsAre(1, 1, 1, 0, 1, 1, 0, 0,  //
                                                     0, 0, 0, 0, 0, 0, 0, 0,  //
                                                     0, 0, 0, 0, 0, 0, 0, 0,  //
                                                     0, 0, 0, 0, 0, 0, 0, 0));

  EXPECT_THAT(const_cast<const BitSet&>(v).bools_view(),
              ::testing::ElementsAre(1, 1, 1, 0, 1, 1, 0, 0,  //
                                     0, 0, 0, 0, 0, 0, 0, 0,  //
                                     0, 0, 0, 0, 0, 0, 0, 0,  //
                                     0, 0, 0, 0, 0, 0, 0, 0));

  EXPECT_EQ(
      "11101100"
      "00000000"
      "00000000"
      "00000000",
      tensorstore::StrCat(v));
  EXPECT_EQ(0b11111111'11111111'11111111'11001000, (~v).to_uint());
  auto v1 = BitSet::FromUint(0b101'1100);
  EXPECT_EQ(0b111'1111, (v | v1).to_uint());
  EXPECT_EQ(0b001'0100, (v & v1).to_uint());
  EXPECT_EQ(0b110'1011, (v ^ v1).to_uint());
  auto v2 = v1;
  v2 |= v;
  EXPECT_EQ(0b111'1111, v2.to_uint());
  v2 = v1;
  v2 &= v;
  EXPECT_EQ(0b001'0100, v2.to_uint());
  v2 = v1;
  v2 ^= v;
  EXPECT_EQ(0b110'1011, v2.to_uint());
}

TEST(SmallBitSetTest, BracedList) {
  auto v = BitSet::FromBools({0, 1, 1, 0, 0, 1});
  EXPECT_EQ(0b100110, v.to_uint());
}

TEST(SmallBitSetTest, Reference) {
  BitSet v;
  v[2] = true;
  EXPECT_TRUE(v[2]);
  EXPECT_FALSE(v[0]);
  EXPECT_EQ(0b100, v.to_uint());
}

TEST(SmallBitSetTest, UpTo) {
  EXPECT_EQ(0x00000000, BitSet::UpTo(0).to_uint());
  EXPECT_EQ(0x00000001, BitSet::UpTo(1).to_uint());
  EXPECT_EQ(0x0000ffff, BitSet::UpTo(16).to_uint());
  EXPECT_EQ(0x7fffffff, BitSet::UpTo(31).to_uint());
  EXPECT_EQ(0xffffffff, BitSet::UpTo(32).to_uint());
  EXPECT_EQ(1, BitSet::UpTo(1).count());
}

TEST(SmallBitSetTest, FromIndices) {
  BitSet v = BitSet::FromIndices({1, 3, 10});
  EXPECT_FALSE(v.none());
  EXPECT_EQ(3, v.count());
  EXPECT_EQ((static_cast<uint32_t>(1) << 1) | (static_cast<uint32_t>(1) << 3) |
                (static_cast<uint32_t>(1) << 10),
            v.to_uint());
  EXPECT_THAT(v.index_view(), ::testing::ElementsAre(1, 3, 10));
}

}  // namespace
