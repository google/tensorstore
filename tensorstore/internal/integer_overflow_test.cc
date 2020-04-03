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

#include <cstdint>

#include <gtest/gtest.h>
#include "tensorstore/index.h"

namespace {

using tensorstore::Index;
using tensorstore::internal::AddOverflow;
using tensorstore::internal::MulOverflow;
using tensorstore::internal::SubOverflow;
using tensorstore::internal::wrap_on_overflow::Add;
using tensorstore::internal::wrap_on_overflow::InnerProduct;
using tensorstore::internal::wrap_on_overflow::Multiply;

TEST(AddTest, Overflow) {
  EXPECT_EQ(std::int32_t{-0x80000000LL},
            Add(std::int32_t{0x40000000}, std::int32_t{0x40000000}));
}

TEST(MultiplyTest, Overflow) {
  EXPECT_EQ(std::int32_t{-0x80000000LL},
            Multiply(std::int32_t{0x40000000}, std::int32_t{2}));
}

TEST(InnerProductTest, Basic) {
  const Index a[] = {1, 2, 3};
  const Index b[] = {4, 5, 6};
  EXPECT_EQ(1 * 4 + 2 * 5 + 3 * 6, InnerProduct<Index>(3, a, b));
}

TEST(InnerProductTest, Convert) {
  const std::uint32_t a[] = {0x80000000};
  const std::uint32_t b[] = {2};
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
  std::uint32_t a, b, c;

  a = 0x7fffffff;
  b = 2;
  EXPECT_EQ(false, MulOverflow(a, b, &c));
  EXPECT_EQ(std::uint32_t{0xfffffffe}, c);
  EXPECT_EQ(false, MulOverflow(b, a, &c));
  EXPECT_EQ(std::uint32_t{0xfffffffe}, c);

  a = 0x80000000;
  c = 2;
  EXPECT_EQ(true, MulOverflow(a, b, &c));
  EXPECT_EQ(std::uint32_t{0}, c);
  EXPECT_EQ(true, MulOverflow(b, a, &c));
  EXPECT_EQ(std::uint32_t{0}, c);
}

TEST(MulOverflow, Int32) {
  std::int32_t a, b, c;

  a = -0x40000000;
  b = 2;
  EXPECT_EQ(false, MulOverflow(a, b, &c));
  EXPECT_EQ(std::int32_t{-0x80000000LL}, c);
  EXPECT_EQ(false, MulOverflow(b, a, &c));
  EXPECT_EQ(std::int32_t{-0x80000000LL}, c);

  a = 0x40000000;
  c = 2;
  EXPECT_EQ(true, MulOverflow(a, b, &c));
  EXPECT_EQ(std::int32_t{-0x80000000LL}, c);
  EXPECT_EQ(true, MulOverflow(b, a, &c));
  EXPECT_EQ(std::int32_t{-0x80000000LL}, c);
}

TEST(AddOverflow, Uint32) {
  std::uint32_t a, b, c;

  a = 0x7fffffff;
  b = 0x80000000;
  EXPECT_EQ(false, AddOverflow(a, b, &c));
  EXPECT_EQ(std::uint32_t{0xffffffff}, c);
  EXPECT_EQ(false, AddOverflow(b, a, &c));
  EXPECT_EQ(std::uint32_t{0xffffffff}, c);

  a = 0x80000000;
  c = 0x80000000;
  EXPECT_EQ(true, MulOverflow(a, b, &c));
  EXPECT_EQ(std::uint32_t{0}, c);
}

TEST(AddOverflow, Int32) {
  std::int32_t a, b, c;

  a = 0x40000000;
  b = 0x3fffffff;
  EXPECT_EQ(false, AddOverflow(a, b, &c));
  EXPECT_EQ(std::int32_t{0x7fffffff}, c);
  EXPECT_EQ(false, AddOverflow(b, a, &c));
  EXPECT_EQ(std::int32_t{0x7fffffff}, c);

  a = -0x40000000;
  b = -0x40000000;
  EXPECT_EQ(false, AddOverflow(a, b, &c));
  EXPECT_EQ(std::int32_t{-0x80000000LL}, c);

  a = 0x40000000;
  b = 0x40000000;
  EXPECT_EQ(true, AddOverflow(a, b, &c));
  EXPECT_EQ(std::int32_t{-0x80000000LL}, c);
}

TEST(SubOverflow, Uint32) {
  std::uint32_t a, b, c;

  a = 0x80000000;
  b = 0x7fffffff;
  EXPECT_EQ(false, SubOverflow(a, b, &c));
  EXPECT_EQ(std::uint32_t{1}, c);

  a = 0x7fffffff;
  b = 0x80000000;
  EXPECT_EQ(true, SubOverflow(a, b, &c));
  EXPECT_EQ(std::uint32_t{0xffffffff}, c);
}

TEST(SubOverflow, Int32) {
  std::int32_t a, b, c;

  a = -0x40000000;
  b = 0x40000000;
  EXPECT_EQ(false, SubOverflow(a, b, &c));
  EXPECT_EQ(std::int32_t{-0x80000000LL}, c);

  a = 0x40000000;
  b = -0x40000000;
  EXPECT_EQ(true, SubOverflow(a, b, &c));
  EXPECT_EQ(std::int32_t{-0x80000000LL}, c);

  a = -0x40000001;
  b = 0x40000000;
  EXPECT_EQ(true, SubOverflow(a, b, &c));
  EXPECT_EQ(std::int32_t{0x7fffffff}, c);
}

}  // namespace
