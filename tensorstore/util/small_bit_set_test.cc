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

using BitSet = tensorstore::SmallBitSet<32>;

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

}  // namespace
