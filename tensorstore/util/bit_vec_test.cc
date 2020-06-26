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

#include "tensorstore/util/bit_vec.h"

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/util/bit_span.h"
#include "tensorstore/util/str_cat.h"

namespace {
using tensorstore::BitSpan;
using tensorstore::BitVec;

static_assert(!std::is_convertible<BitSpan<std::uint64_t, 3>, BitVec<>>::value,
              "");

static_assert(
    std::is_constructible<BitVec<3>, BitSpan<std::uint32_t, 3>>::value, "");

static_assert(std::is_constructible<BitVec<>, BitSpan<std::uint32_t, 3>>::value,
              "");

static_assert(!std::is_constructible<BitVec<3>, BitVec<>>::value, "");

static_assert(!std::is_constructible<BitVec<3>, BitSpan<std::uint32_t>>::value,
              "");

static_assert(
    !std::is_constructible<BitVec<3>, BitSpan<std::uint32_t, 4>>::value, "");

static_assert(!std::is_constructible<BitVec<3>, BitVec<4>>::value, "");

TEST(BitVecTest, StaticDefaultConstruct) {
  BitVec<9> v;
  EXPECT_THAT(v, ::testing::ElementsAre(0, 0, 0, 0, 0, 0, 0, 0, 0));
}

TEST(BitVecTest, StaticConstructTrue) {
  BitVec<9> v({}, true);
  EXPECT_THAT(v, ::testing::ElementsAre(1, 1, 1, 1, 1, 1, 1, 1, 1));
}

TEST(BitVecTest, DynamicDefaultConstruct) {
  BitVec<> v;
  EXPECT_EQ(0, v.size());
  EXPECT_TRUE(v.empty());
  v.resize(65);
  EXPECT_FALSE(v.empty());
  EXPECT_THAT(v, ::testing::ElementsAreArray(std::vector<bool>(65, false)));
  v.fill(true);
  EXPECT_THAT(v, ::testing::ElementsAreArray(std::vector<bool>(65, true)));
}

TEST(BitVecTest, DynamicConstructFalse) {
  BitVec<> v(65);
  EXPECT_THAT(v, ::testing::ElementsAreArray(std::vector<bool>(65, false)));
}

TEST(BitVecTest, Subscript) {
  BitVec<> v(9);
  const auto& v_ref = v;
  EXPECT_FALSE(v_ref[3]);
  v[3] = true;
  EXPECT_TRUE(v_ref[3]);
  v[5] = true;
  v[6] = true;
  EXPECT_THAT(v, ::testing::ElementsAre(0, 0, 0, 1, 0, 1, 1, 0, 0));
  v[8] = true;
  EXPECT_THAT(v, ::testing::ElementsAre(0, 0, 0, 1, 0, 1, 1, 0, 1));
  v[3] = false;
  EXPECT_THAT(v, ::testing::ElementsAre(0, 0, 0, 0, 0, 1, 1, 0, 1));
}

TEST(BitVecTest, CopyConstructInline) {
  BitVec<> a(9);
  a[0] = true;
  a[3] = true;
  a[5] = true;
  a[6] = true;
  BitVec<> b(a);
  EXPECT_THAT(a, ::testing::ElementsAre(1, 0, 0, 1, 0, 1, 1, 0, 0));
  EXPECT_THAT(b, ::testing::ElementsAre(1, 0, 0, 1, 0, 1, 1, 0, 0));
}

TEST(BitVecTest, CopyConstructLarge) {
  BitVec<> a(129);
  std::vector<bool> expected(129);
  for (int i : {0, 3, 5, 6, 31, 33, 72, 128}) {
    expected[i] = true;
    a[i] = true;
  }
  BitVec<> b(a);
  EXPECT_THAT(a, ::testing::ElementsAreArray(expected));
  EXPECT_THAT(b, ::testing::ElementsAreArray(expected));
}

TEST(BitVecTest, MoveConstructInline) {
  BitVec<> a(9);
  a[0] = true;
  a[3] = true;
  a[5] = true;
  a[6] = true;
  BitVec<> b(std::move(a));
  EXPECT_THAT(b, ::testing::ElementsAre(1, 0, 0, 1, 0, 1, 1, 0, 0));
}

TEST(BitVecTest, MoveConstructLarge) {
  BitVec<> a(129);
  std::vector<bool> expected(129);
  for (int i : {0, 3, 5, 6, 31, 33, 72, 128}) {
    expected[i] = true;
    a[i] = true;
  }
  BitVec<> b(std::move(a));
  EXPECT_THAT(b, ::testing::ElementsAreArray(expected));
}

TEST(BitVecTest, CopyAssignInline) {
  BitVec<> a(9);
  a[0] = true;
  a[3] = true;
  a[5] = true;
  a[6] = true;
  BitVec<> b(9);
  b = a;
  EXPECT_THAT(a, ::testing::ElementsAre(1, 0, 0, 1, 0, 1, 1, 0, 0));
  EXPECT_THAT(b, ::testing::ElementsAre(1, 0, 0, 1, 0, 1, 1, 0, 0));
}

TEST(BitVecTest, CopyAssignLargeSameNumBlocks) {
  BitVec<> a(129);
  std::vector<bool> expected(129);
  for (int i : {0, 3, 5, 6, 31, 33, 72, 128}) {
    expected[i] = true;
    a[i] = true;
  }
  BitVec<> b(129);
  b = a;
  EXPECT_THAT(a, ::testing::ElementsAreArray(expected));
  EXPECT_THAT(b, ::testing::ElementsAreArray(expected));
}

TEST(BitVecTest, CopyAssignLargeDifferentNumBlocks) {
  BitVec<> a(129);
  std::vector<bool> expected(129);
  for (int i : {0, 3, 5, 6, 31, 33, 72, 128}) {
    expected[i] = true;
    a[i] = true;
  }
  BitVec<> b(65);
  b = a;
  EXPECT_THAT(a, ::testing::ElementsAreArray(expected));
  EXPECT_THAT(b, ::testing::ElementsAreArray(expected));
}

TEST(BitVecTest, MoveAssignInline) {
  BitVec<> a(9);
  a[0] = true;
  a[3] = true;
  a[5] = true;
  a[6] = true;
  BitVec<> b(9);
  b = std::move(a);
  EXPECT_THAT(b, ::testing::ElementsAre(1, 0, 0, 1, 0, 1, 1, 0, 0));
}

TEST(BitVecTest, MoveAssignLarge) {
  BitVec<> a(129);
  std::vector<bool> expected(129);
  for (int i : {0, 3, 5, 6, 31, 33, 72, 128}) {
    expected[i] = true;
    a[i] = true;
  }
  BitVec<> b(129);
  b = std::move(a);
  EXPECT_THAT(b, ::testing::ElementsAreArray(expected));
}

TEST(BitVecTest, BracedListConstruct) {
  BitVec<> a({1, 0, 0, 1, 1});
  EXPECT_THAT(a, ::testing::ElementsAre(1, 0, 0, 1, 1));
}

TEST(BitVecTest, DeduceBitVec) {
  auto a = BitVec({true, false, false, true, true});
  EXPECT_THAT(a, ::testing::ElementsAre(1, 0, 0, 1, 1));
  static_assert(std::is_same_v<decltype(a), BitVec<5>>);
  auto b = BitVec(a.bit_span());
  static_assert(std::is_same_v<decltype(b), BitVec<5>>);
  EXPECT_THAT(b, ::testing::ElementsAre(1, 0, 0, 1, 1));
}

TEST(BitVecTest, BitSpanConstruct) {
  BitVec<> a(37);
  a[32] = 1;
  a[17] = 1;
  a[2] = 1;
  EXPECT_THAT(a, ::testing::ElementsAre(0, 0, 1, 0, 0, 0, 0, 0,  //
                                        0, 0, 0, 0, 0, 0, 0, 0,  //
                                        0, 1, 0, 0, 0, 0, 0, 0,  //
                                        0, 0, 0, 0, 0, 0, 0, 0,  //
                                        1, 0, 0, 0, 0));
  BitVec<> b(a.bit_span());
  EXPECT_THAT(b, ::testing::ElementsAre(0, 0, 1, 0, 0, 0, 0, 0,  //
                                        0, 0, 0, 0, 0, 0, 0, 0,  //
                                        0, 1, 0, 0, 0, 0, 0, 0,  //
                                        0, 0, 0, 0, 0, 0, 0, 0,  //
                                        1, 0, 0, 0, 0));
}

TEST(BitVecTest, BitVecConvertConstruct) {
  BitVec<37> a;
  a[32] = 1;
  a[17] = 1;
  a[2] = 1;
  EXPECT_THAT(a, ::testing::ElementsAre(0, 0, 1, 0, 0, 0, 0, 0,  //
                                        0, 0, 0, 0, 0, 0, 0, 0,  //
                                        0, 1, 0, 0, 0, 0, 0, 0,  //
                                        0, 0, 0, 0, 0, 0, 0, 0,  //
                                        1, 0, 0, 0, 0));

  BitVec<> b = a;
  EXPECT_THAT(b, ::testing::ElementsAre(0, 0, 1, 0, 0, 0, 0, 0,  //
                                        0, 0, 0, 0, 0, 0, 0, 0,  //
                                        0, 1, 0, 0, 0, 0, 0, 0,  //
                                        0, 0, 0, 0, 0, 0, 0, 0,  //
                                        1, 0, 0, 0, 0));
}

TEST(BitVecTest, ComparisonShort) {
  BitVec<> a(18);
  BitVec<> b(17);
  EXPECT_NE(a, b);
  b.resize(18);
  EXPECT_EQ(a, b);
  b[2] = true;
  EXPECT_NE(a, b);
  a[2] = true;
  EXPECT_EQ(a, b);
  a[17] = true;
  EXPECT_NE(a, b);
  b[17] = true;
  EXPECT_EQ(a, b);
}

TEST(BitVecTest, ComparisonLong) {
  BitVec<> a(150);
  BitVec<> b(151);
  EXPECT_NE(a, b);
  b.resize(150);
  EXPECT_EQ(a, b);
  b[2] = true;
  EXPECT_NE(a, b);
  a[2] = true;
  EXPECT_EQ(a, b);
  a[149] = true;
  EXPECT_NE(a, b);
  b[149] = true;
  EXPECT_EQ(a, b);
}

TEST(BitVecTest, ConstIterators) {
  BitVec<> a(7);
  a[1] = 1;
  a[4] = 1;
  {
    const auto& a_ref = a;
    std::vector<bool> b(a_ref.begin(), a_ref.end());
    EXPECT_THAT(b, ::testing::ElementsAre(0, 1, 0, 0, 1, 0, 0));
  }
  {
    std::vector<bool> b(a.cbegin(), a.cend());
    EXPECT_THAT(b, ::testing::ElementsAre(0, 1, 0, 0, 1, 0, 0));
  }
}

TEST(BitVecTest, NonConstIterators) {
  BitVec<> a(7);
  a[1] = 1;
  a[4] = 1;
  std::vector<bool> b(a.begin(), a.end());
  EXPECT_THAT(b, ::testing::ElementsAre(0, 1, 0, 0, 1, 0, 0));
}

TEST(BitVecTest, NonConstIteratorsMutate) {
  BitVec<> a(7);
  std::vector<bool> b{0, 1, 0, 0, 1, 0, 0};
  std::copy(b.begin(), b.end(), a.begin());
  EXPECT_THAT(a, ::testing::ElementsAre(0, 1, 0, 0, 1, 0, 0));
}

TEST(BitVecTest, BlocksInline) {
  BitVec<> a(64);
  for (int i : {0, 5, 17, 62}) {
    a[i] = true;
  }
  EXPECT_THAT(a.blocks(), ::testing::ElementsAre(          //
                              (std::uint64_t(1) << 0) |    //
                              (std::uint64_t(1) << 5) |    //
                              (std::uint64_t(1) << 17) |   //
                              (std::uint64_t(1) << 62)));  //
}

TEST(BitVecTest, BlocksLarge) {
  BitVec<> a(128);
  for (int i : {0, 5, 17, 62, 90, 127}) {
    a[i] = true;
  }
  EXPECT_THAT(a.blocks(),
              ::testing::ElementsAre(                //
                  (std::uint64_t(1) << 0) |          //
                      (std::uint64_t(1) << 5) |      //
                      (std::uint64_t(1) << 17) |     //
                      (std::uint64_t(1) << 62),      //
                  (std::uint64_t(1) << (90 - 64)) |  //
                      (std::uint64_t(1) << (127 - 64))));
}

TEST(BitVecTest, ResizeStatic) {
  BitVec<65> b;
  std::vector<bool> expected(65);
  for (int i : {0, 3, 7, 29, 35, 64}) {
    expected[i] = true;
    b[i] = true;
  }
  EXPECT_THAT(b, ::testing::ElementsAreArray(expected));
  b.resize(std::integral_constant<std::ptrdiff_t, 65>{});
  EXPECT_THAT(b, ::testing::ElementsAreArray(expected));
}

void TestResizeDynamic(std::ptrdiff_t orig_size, std::ptrdiff_t new_size,
                       std::vector<int> bits) {
  SCOPED_TRACE(tensorstore::StrCat("orig_size=", orig_size,
                                   ", new_size=", new_size,
                                   ", bits=", ::testing::PrintToString(bits)));
  BitVec<> b(orig_size);
  std::vector<bool> expected(orig_size);
  for (int i : bits) {
    expected[i] = true;
    b[i] = true;
  }
  std::vector<bool> expected_resize_false = expected;
  expected_resize_false.resize(new_size, false);
  std::vector<bool> expected_resize_true = expected;
  expected_resize_true.resize(new_size, true);
  EXPECT_THAT(b, ::testing::ElementsAreArray(expected));
  BitVec<> b_resize_false = b;
  b_resize_false.resize(new_size, false);
  BitVec<> b_resize_true = b;
  b_resize_true.resize(new_size, true);
  EXPECT_THAT(b_resize_false,
              ::testing::ElementsAreArray(expected_resize_false));
  EXPECT_THAT(b_resize_true, ::testing::ElementsAreArray(expected_resize_true));
}

TEST(BitVecTest, ResizeDynamicLargeNoOp) {
  TestResizeDynamic(65, 65, {0, 3, 7, 29, 35, 64});
}

TEST(BitVecTest, ResizeDynamicInlineNoOp) {
  TestResizeDynamic(62, 62, {0, 3, 7, 29, 35, 61});
}

TEST(BitVecTest, ResizeDynamicInlineShrink) {
  TestResizeDynamic(62, 30, {0, 3, 7, 29, 35, 61});
}

TEST(BitVecTest, ResizeDynamicInlineExpand) {
  TestResizeDynamic(36, 41, {0, 3, 7, 29, 35});
}

TEST(BitVecTest, ResizeDynamicShrinkSameNumBlocks) {
  TestResizeDynamic(150, 132, {0, 3, 7, 29, 35, 64, 127, 131, 149});
}

TEST(BitVecTest, ResizeDynamicExpandSameNumBlocks) {
  TestResizeDynamic(150, 160, {0, 3, 7, 29, 35, 64, 127, 131, 149});
}

TEST(BitVecTest, ResizeDynamicShrinkDifferentNumBlocks) {
  TestResizeDynamic(150, 128, {0, 3, 7, 29, 35, 64, 127, 131, 149});
  TestResizeDynamic(150, 126, {0, 3, 7, 29, 35, 64, 127, 131, 149});
}

TEST(BitVecTest, ResizeDynamicExpandDifferentNumBlocks) {
  TestResizeDynamic(150, 250, {0, 3, 7, 29, 35, 64, 127, 131, 149});
}

TEST(BitVecTest, ResizeDynamicExpandFromEmpty) {
  TestResizeDynamic(0, 15, {});
  TestResizeDynamic(0, 65, {});
  TestResizeDynamic(0, 150, {});
  TestResizeDynamic(0, 0, {});
}

TEST(BitVecTest, ResizeDynamicShrinkToEmpty) {
  TestResizeDynamic(13, 0, {1, 2, 12});
  TestResizeDynamic(129, 0, {1, 2, 12, 65, 73, 128});
}

}  // namespace
