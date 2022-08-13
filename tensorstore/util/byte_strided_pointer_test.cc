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

#include "tensorstore/util/byte_strided_pointer.h"

#include <limits>
#include <type_traits>

#include <gtest/gtest.h>

namespace {

using ::tensorstore::ByteStridedPointer;

struct Base {};
struct Derived : Base {};

static_assert(std::is_convertible_v<int*, ByteStridedPointer<int>>);
static_assert(std::is_constructible_v<int*, ByteStridedPointer<void>>);
static_assert(!std::is_constructible_v<int*, ByteStridedPointer<const void>>);
static_assert(std::is_convertible_v<ByteStridedPointer<int>, int*>);
static_assert(std::is_convertible_v<ByteStridedPointer<int>, const int*>);
static_assert(
    std::is_convertible_v<ByteStridedPointer<int>, ByteStridedPointer<void>>);
static_assert(std::is_convertible_v<ByteStridedPointer<const int>,
                                    ByteStridedPointer<const void>>);
static_assert(!std::is_convertible_v<ByteStridedPointer<const int>,
                                     ByteStridedPointer<void>>);
static_assert(
    !std::is_convertible_v<ByteStridedPointer<void>, ByteStridedPointer<int>>);
static_assert(
    std::is_constructible_v<ByteStridedPointer<int>, ByteStridedPointer<void>>);
static_assert(!std::is_convertible_v<ByteStridedPointer<const int>,
                                     ByteStridedPointer<const float>>);
static_assert(!std::is_convertible_v<ByteStridedPointer<Derived>,
                                     ByteStridedPointer<Base>>);
static_assert(!std::is_convertible_v<ByteStridedPointer<Base>,
                                     ByteStridedPointer<Derived>>);

TEST(ByteStridedPointerTest, DefaultConstructor) {
  // Just check that it compiles.
  ByteStridedPointer<int> ptr;
  static_cast<void>(ptr);
}

TEST(ByteStridedPointerTest, ConstructFromRaw) {
  int value;
  ByteStridedPointer<int> ptr = &value;
  EXPECT_EQ(&value, ptr.get());
}

TEST(ByteStridedPointerTest, ConstructFromRawConvertImplicit) {
  int value;
  ByteStridedPointer<const int> ptr = &value;
  EXPECT_EQ(&value, ptr.get());
}

TEST(ByteStridedPointerTest, ConstructFromRawConvertExplicit) {
  int value;
  ByteStridedPointer<const int> ptr(static_cast<void*>(&value));
  EXPECT_EQ(&value, ptr.get());
}

TEST(ByteStridedPointerTest, ConstructFromOther) {
  int value;
  ByteStridedPointer<int> ptr = ByteStridedPointer<int>(&value);
  EXPECT_EQ(&value, ptr.get());
}

TEST(ByteStridedPointerTest, ConstructFromOtherConvertImplicit) {
  int value;
  ByteStridedPointer<const int> ptr = ByteStridedPointer<int>(&value);
  EXPECT_EQ(&value, ptr.get());
}

TEST(ByteStridedPointerTest, ConstructFromOtherConvertExplicit) {
  int value;
  ByteStridedPointer<const int> ptr{ByteStridedPointer<void>(&value)};
  EXPECT_EQ(&value, ptr.get());
}

TEST(ByteStridedPointerTest, ArrowOperator) {
  int value;
  ByteStridedPointer<const int> x(&value);
  EXPECT_EQ(&value, x.operator->());
}

TEST(ByteStridedPointerTest, Dereference) {
  int value = 3;
  ByteStridedPointer<const int> x(&value);
  EXPECT_EQ(3, *x);
  EXPECT_EQ(&value, &*x);
}

TEST(ByteStridedPointerTest, CastImplicit) {
  int value = 3;
  ByteStridedPointer<const int> x(&value);
  const int* p = x;
  EXPECT_EQ(&value, p);
}

TEST(ByteStridedPointerTest, CastExplicit) {
  int value = 3;
  ByteStridedPointer<void> x(&value);
  const int* p = static_cast<const int*>(x);
  EXPECT_EQ(&value, p);
}

TEST(ByteStridedPointerTest, Add) {
  int arr[] = {1, 2, 3};
  ByteStridedPointer<int> x(&arr[0]);
  x += sizeof(int);
  EXPECT_EQ(x, ByteStridedPointer<int>(&arr[0]) + sizeof(int));
  EXPECT_EQ(x, sizeof(int) + ByteStridedPointer<int>(&arr[0]));
  EXPECT_EQ(&arr[1], x.get());
}

TEST(ByteStridedPointerTest, Subtract) {
  int arr[] = {1, 2, 3};
  ByteStridedPointer<int> x(&arr[2]);
  x -= sizeof(int);
  EXPECT_EQ(x, ByteStridedPointer<int>(&arr[2]) - sizeof(int));
  EXPECT_EQ(&arr[1], x.get());
}

TEST(ByteStridedPointerTest, AddWrapOnOverflow) {
  int arr[] = {1, 2, 3};
  ByteStridedPointer<int> x(&arr[0]);
  const std::uintptr_t base_index =
      std::numeric_limits<std::uintptr_t>::max() - 99;
  x -= base_index;
  x += (base_index + sizeof(int));
  EXPECT_EQ(x, ByteStridedPointer<int>(&arr[0]) + sizeof(int));
  EXPECT_EQ(x, sizeof(int) + ByteStridedPointer<int>(&arr[0]));
  EXPECT_EQ(&arr[1], x.get());
}

TEST(ByteStridedPointerTest, Difference) {
  int arr[] = {1, 2, 3};
  ByteStridedPointer<int> x(&arr[2]);
  ByteStridedPointer<int> y(&arr[1]);
  EXPECT_EQ(4, x - y);
}

TEST(ByteStridedPointerTest, Comparison) {
  int arr[] = {1, 2, 3};
  ByteStridedPointer<int> x(&arr[2]);
  ByteStridedPointer<int> y = x;
  EXPECT_TRUE(x == y);
  x -= sizeof(int);
  EXPECT_FALSE(x == y);
  EXPECT_TRUE(x < y);
}

}  // namespace
