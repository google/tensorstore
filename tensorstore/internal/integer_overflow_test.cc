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

#include "tensorstore/internal/integer_overflow.h"

#include <stdint.h>

#include <limits>

#include <gtest/gtest.h>
#include "tensorstore/index.h"

namespace {

using ::tensorstore::Index;
using ::tensorstore::internal::AddOverflow;
using ::tensorstore::internal::AddSaturate;
using ::tensorstore::internal::MulOverflow;
using ::tensorstore::internal::MulSaturate;
using ::tensorstore::internal::SubOverflow;
using ::tensorstore::internal::SubSaturate;
using ::tensorstore::internal::wrap_on_overflow::Add;
using ::tensorstore::internal::wrap_on_overflow::InnerProduct;
using ::tensorstore::internal::wrap_on_overflow::Multiply;
using ::tensorstore::internal::wrap_on_overflow::Subtract;

// Compile-time verification of `constexpr` evaluation across narrow and 64-bit
// signed/unsigned types (including 16-bit promotion boundaries).
static_assert(Add(uint16_t{0xffffu}, uint16_t{2u}) == uint16_t{1u});
static_assert(Subtract(uint16_t{1u}, uint16_t{2u}) == uint16_t{0xffffu});
static_assert(Multiply(uint16_t{0xffffu}, uint16_t{0xffffu}) == uint16_t{1u});
static_assert(Multiply(int16_t{-1}, int16_t{-1}) == int16_t{1});
static_assert(AddSaturate<int8_t>(100, 50) == 127);
static_assert(AddSaturate<int8_t>(-100, -50) == -128);
static_assert(AddSaturate<uint8_t>(200, 100) == 255);
static_assert(SubSaturate<int16_t>(-30000, 10000) == -32768);
static_assert(SubSaturate<uint16_t>(5, 10) == 0);
static_assert(MulSaturate<uint16_t>(0xffffu, 0xffffu) == 0xffffu);
static_assert(MulSaturate<int16_t>(-1, -1) == 1);
static_assert(MulSaturate<int16_t>(std::numeric_limits<int16_t>::min(), -1) ==
              std::numeric_limits<int16_t>::max());
static_assert(AddSaturate<int64_t>(std::numeric_limits<int64_t>::max(), 1) ==
              std::numeric_limits<int64_t>::max());
static_assert(SubSaturate<int64_t>(std::numeric_limits<int64_t>::min(), 1) ==
              std::numeric_limits<int64_t>::min());
static_assert(MulSaturate<uint64_t>(uint64_t{1} << 40, uint64_t{1} << 30) ==
              std::numeric_limits<uint64_t>::max());

TEST(AddTest, Overflow) {
  EXPECT_EQ(int32_t{-0x80000000LL},
            Add(int32_t{0x40000000}, int32_t{0x40000000}));
}

TEST(MultiplyTest, Overflow) {
  EXPECT_EQ(int32_t{-0x80000000LL}, Multiply(int32_t{0x40000000}, int32_t{2}));
}

TEST(InnerProductTest, Basic) {
  const Index a[] = {1, 2, 3};
  const Index b[] = {4, 5, 6};
  EXPECT_EQ(1 * 4 + 2 * 5 + 3 * 6, InnerProduct<Index>(3, a, b));
}

TEST(InnerProductTest, Convert) {
  const uint32_t a[] = {0x80000000};
  const uint32_t b[] = {2};
  EXPECT_EQ(Index{0x100000000}, InnerProduct<Index>(1, a, b));
}

TEST(InnerProductTest, WrapOnOverflowMultiply) {
  const Index a[] = {Index(1) << 62, 2, 3};
  const Index b[] = {4, 5, 6};
  EXPECT_EQ(Index{2 * 5 + 3 * 6}, InnerProduct<Index>(3, a, b));
}

TEST(InnerProductTest, WrapOnOverflowAdd) {
  const Index a[] = {Index(1) << 62, Index(1) << 62};
  const Index b[] = {2, 2};
  EXPECT_EQ(Index{0}, InnerProduct<Index>(2, a, b));
}

TEST(MulOverflow, Uint32) {
  uint32_t a, b, c;

  a = 0x7fffffff;
  b = 2;
  EXPECT_EQ(false, MulOverflow(a, b, &c));
  EXPECT_EQ(uint32_t{0xfffffffe}, c);
  EXPECT_EQ(false, MulOverflow(b, a, &c));
  EXPECT_EQ(uint32_t{0xfffffffe}, c);

  a = 0x80000000;
  c = 2;
  EXPECT_EQ(true, MulOverflow(a, b, &c));
  EXPECT_EQ(uint32_t{0}, c);
  EXPECT_EQ(true, MulOverflow(b, a, &c));
  EXPECT_EQ(uint32_t{0}, c);
}

TEST(MulOverflow, Int32) {
  int32_t a, b, c;

  a = -0x40000000;
  b = 2;
  EXPECT_EQ(false, MulOverflow(a, b, &c));
  EXPECT_EQ(int32_t{-0x80000000LL}, c);
  EXPECT_EQ(false, MulOverflow(b, a, &c));
  EXPECT_EQ(int32_t{-0x80000000LL}, c);

  a = 0x40000000;
  c = 2;
  EXPECT_EQ(true, MulOverflow(a, b, &c));
  EXPECT_EQ(int32_t{-0x80000000LL}, c);
  EXPECT_EQ(true, MulOverflow(b, a, &c));
  EXPECT_EQ(int32_t{-0x80000000LL}, c);
}

TEST(MulOverflow, Int32Limits) {
  int32_t c;
  EXPECT_EQ(true,
            MulOverflow(std::numeric_limits<int32_t>::min(), int32_t{-1}, &c));
  EXPECT_EQ(c, std::numeric_limits<int32_t>::min());
  EXPECT_EQ(true,
            MulOverflow(int32_t{-1}, std::numeric_limits<int32_t>::min(), &c));
  EXPECT_EQ(c, std::numeric_limits<int32_t>::min());
}

TEST(MulOverflow, Int64Limits) {
  int64_t c;
  EXPECT_EQ(true,
            MulOverflow(std::numeric_limits<int64_t>::min(), int64_t{-1}, &c));
  EXPECT_EQ(c, std::numeric_limits<int64_t>::min());
  EXPECT_EQ(true,
            MulOverflow(int64_t{-1}, std::numeric_limits<int64_t>::min(), &c));
  EXPECT_EQ(c, std::numeric_limits<int64_t>::min());
}

TEST(AddOverflow, Uint32) {
  uint32_t a, b, c;

  a = 0x7fffffff;
  b = 0x80000000;
  EXPECT_EQ(false, AddOverflow(a, b, &c));
  EXPECT_EQ(uint32_t{0xffffffff}, c);
  EXPECT_EQ(false, AddOverflow(b, a, &c));
  EXPECT_EQ(uint32_t{0xffffffff}, c);

  a = 0x80000000;
  c = 0x80000000;
  EXPECT_EQ(true, MulOverflow(a, b, &c));
  EXPECT_EQ(uint32_t{0}, c);
}

TEST(AddOverflow, Int32) {
  int32_t a, b, c;

  a = 0x40000000;
  b = 0x3fffffff;
  EXPECT_EQ(false, AddOverflow(a, b, &c));
  EXPECT_EQ(int32_t{0x7fffffff}, c);
  EXPECT_EQ(false, AddOverflow(b, a, &c));
  EXPECT_EQ(int32_t{0x7fffffff}, c);

  a = -0x40000000;
  b = -0x40000000;
  EXPECT_EQ(false, AddOverflow(a, b, &c));
  EXPECT_EQ(int32_t{-0x80000000LL}, c);

  a = 0x40000000;
  b = 0x40000000;
  EXPECT_EQ(true, AddOverflow(a, b, &c));
  EXPECT_EQ(int32_t{-0x80000000LL}, c);
}

TEST(AddSaturate, Int32) {
  EXPECT_EQ(0x7fffffff, AddSaturate<int32_t>(0x40000000, 0x3fffffff));
  EXPECT_EQ(0x7fffffff, AddSaturate<int32_t>(0x40000000, 0x40000000));
  EXPECT_EQ(-0x80000000, AddSaturate<int32_t>(-0x40000000, -0x40000000));
  EXPECT_EQ(-0x80000000, AddSaturate<int32_t>(-0x40000000, -0x41000000));
}

TEST(AddSaturate, Uint32) {
  EXPECT_EQ(0xffffffffu, AddSaturate<uint32_t>(0x7fffffffu, 0x80000000u));
  EXPECT_EQ(0xffffffffu, AddSaturate<uint32_t>(0x80000000u, 0x80000000u));
}

TEST(SubSaturate, Int32) {
  EXPECT_EQ(0x7fffffff, SubSaturate<int32_t>(0x40000000, -0x3fffffff));
  EXPECT_EQ(0x7fffffff, SubSaturate<int32_t>(0x40000000, -0x40000000));
  EXPECT_EQ(0x7fffffff,
            SubSaturate<int32_t>(0, std::numeric_limits<int32_t>::min()));
  EXPECT_EQ(-0x80000000, SubSaturate<int32_t>(-0x40000000, 0x40000000));
  EXPECT_EQ(-0x80000000, SubSaturate<int32_t>(-0x40000001, 0x40000000));
}

TEST(SubSaturate, Uint32) {
  EXPECT_EQ(1u, SubSaturate<uint32_t>(0x80000000u, 0x7fffffffu));
  EXPECT_EQ(0u, SubSaturate<uint32_t>(0x7fffffffu, 0x80000000u));
}

TEST(MulSaturate, Int32) {
  EXPECT_EQ(0x7ffffffe, MulSaturate<int32_t>(0x3fffffff, 2));
  EXPECT_EQ(0x7fffffff, MulSaturate<int32_t>(0x40000000, 2));
  EXPECT_EQ(0x7fffffff, MulSaturate<int32_t>(-0x40000000, -2));
  EXPECT_EQ(0x7fffffff,
            MulSaturate<int32_t>(std::numeric_limits<int32_t>::min(), -1));
  EXPECT_EQ(-0x80000000, MulSaturate<int32_t>(-0x40000000, 2));
  EXPECT_EQ(-0x80000000, MulSaturate<int32_t>(0x40000000, -3));
  EXPECT_EQ(-0x80000000,
            MulSaturate<int32_t>(std::numeric_limits<int32_t>::min(), 2));
  EXPECT_EQ(0, MulSaturate<int32_t>(std::numeric_limits<int32_t>::max(), 0));
}

TEST(MulSaturate, Uint32) {
  EXPECT_EQ(0xfffffffeu, MulSaturate<uint32_t>(0x7fffffffu, 2u));
  EXPECT_EQ(0xffffffffu, MulSaturate<uint32_t>(0x80000000u, 2u));
  EXPECT_EQ(0u, MulSaturate<uint32_t>(0xffffffffu, 0u));
}

TEST(SaturateNarrow, Int8AndUint8) {
  EXPECT_EQ(int8_t{127}, AddSaturate<int8_t>(100, 40));
  EXPECT_EQ(int8_t{-128}, AddSaturate<int8_t>(-100, -40));
  EXPECT_EQ(uint8_t{255}, AddSaturate<uint8_t>(200, 100));
  EXPECT_EQ(int8_t{127}, SubSaturate<int8_t>(100, -40));
  EXPECT_EQ(int8_t{-128}, SubSaturate<int8_t>(-100, 40));
  EXPECT_EQ(uint8_t{0}, SubSaturate<uint8_t>(10, 20));
  EXPECT_EQ(int8_t{127}, MulSaturate<int8_t>(-128, -1));
  EXPECT_EQ(int8_t{-128}, MulSaturate<int8_t>(64, -3));
  EXPECT_EQ(uint8_t{255}, MulSaturate<uint8_t>(20, 20));
}

TEST(SaturateNarrow, Int16AndUint16SignedPromotionSafety) {
  // `uint16_t{0xffff} * uint16_t{0xffff}` and `int16_t{-1} * int16_t{-1}` test
  // against signed 32-bit integral-promotion overflow (`65535 * 65535 >
  // INT_MAX`).
  EXPECT_EQ(uint16_t{1}, Multiply<uint16_t>(0xffffu, 0xffffu));
  EXPECT_EQ(uint16_t{0xffff}, MulSaturate<uint16_t>(0xffffu, 0xffffu));
  EXPECT_EQ(int16_t{1}, MulSaturate<int16_t>(-1, -1));
  EXPECT_EQ(int16_t{32767},
            MulSaturate<int16_t>(std::numeric_limits<int16_t>::min(), -1));
  EXPECT_EQ(int16_t{-32768}, MulSaturate<int16_t>(256, -256));
}

TEST(Saturate64Bit, Int64AndUint64Boundaries) {
  constexpr int64_t kI64Min = std::numeric_limits<int64_t>::min();
  constexpr int64_t kI64Max = std::numeric_limits<int64_t>::max();
  constexpr uint64_t kU64Max = std::numeric_limits<uint64_t>::max();
  EXPECT_EQ(kI64Max, AddSaturate<int64_t>(kI64Max - 5, 10));
  EXPECT_EQ(kI64Min, AddSaturate<int64_t>(kI64Min + 5, -10));
  EXPECT_EQ(kU64Max, AddSaturate<uint64_t>(kU64Max - 5, 10));
  EXPECT_EQ(kI64Max, SubSaturate<int64_t>(0, kI64Min));
  EXPECT_EQ(kI64Min, SubSaturate<int64_t>(kI64Min + 5, 10));
  EXPECT_EQ(uint64_t{0}, SubSaturate<uint64_t>(5, 10));
  EXPECT_EQ(kI64Max, MulSaturate<int64_t>(kI64Min, -1));
  EXPECT_EQ(kI64Min,
            MulSaturate<int64_t>(int64_t{1} << 40, -(int64_t{1} << 30)));
  EXPECT_EQ(kU64Max,
            MulSaturate<uint64_t>(uint64_t{1} << 40, uint64_t{1} << 30));
}

TEST(SubOverflow, Uint32) {
  uint32_t a, b, c;

  a = 0x80000000;
  b = 0x7fffffff;
  EXPECT_EQ(false, SubOverflow(a, b, &c));
  EXPECT_EQ(uint32_t{1}, c);

  a = 0x7fffffff;
  b = 0x80000000;
  EXPECT_EQ(true, SubOverflow(a, b, &c));
  EXPECT_EQ(uint32_t{0xffffffff}, c);
}

TEST(SubOverflow, Int32) {
  int32_t a, b, c;

  a = -0x40000000;
  b = 0x40000000;
  EXPECT_EQ(false, SubOverflow(a, b, &c));
  EXPECT_EQ(int32_t{-0x80000000LL}, c);

  a = 0x40000000;
  b = -0x40000000;
  EXPECT_EQ(true, SubOverflow(a, b, &c));
  EXPECT_EQ(int32_t{-0x80000000LL}, c);

  a = -0x40000001;
  b = 0x40000000;
  EXPECT_EQ(true, SubOverflow(a, b, &c));
  EXPECT_EQ(int32_t{0x7fffffff}, c);
}

}  // namespace
