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

#include "tensorstore/internal/elementwise_function.h"

#include <functional>
#include <limits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/attributes.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/integer_overflow.h"
#include "tensorstore/util/byte_strided_pointer.h"
#include "tensorstore/util/status.h"

namespace {

using tensorstore::Index;
using tensorstore::Status;
using tensorstore::internal::ElementwiseClosure;
using tensorstore::internal::ElementwiseFunction;
using tensorstore::internal::IterationBufferAccessor;
using tensorstore::internal::IterationBufferKind;
using tensorstore::internal::IterationBufferPointer;
using tensorstore::internal::SimpleElementwiseFunction;

using ContiguousAccessor =
    IterationBufferAccessor<IterationBufferKind::kContiguous>;
using StridedAccessor = IterationBufferAccessor<IterationBufferKind::kStrided>;
using OffsetArrayAccessor =
    IterationBufferAccessor<IterationBufferKind::kIndexed>;

TEST(ContiguousAccessorTest, Basic) {
  int arr[3] = {1, 2, 3};
  IterationBufferPointer ptr{&arr[0], Index(0)};
  EXPECT_EQ(&arr[0], ContiguousAccessor::GetPointerAtOffset<int>(ptr, 0));
  EXPECT_EQ(&arr[1], ContiguousAccessor::GetPointerAtOffset<int>(ptr, 1));
}

TEST(ContiguousAccessorTest, WrapOnOverflow) {
  int arr[3] = {1, 2, 3};
  IterationBufferPointer ptr{&arr[0], Index(0)};
  // We want to be able to access the array elements using indices starting at
  // `std::numeric_limits<Index>::max() - 3`.
  const Index base_index = std::numeric_limits<Index>::max() - 3;
  ptr.pointer -= tensorstore::internal::wrap_on_overflow::Multiply(
      base_index, static_cast<Index>(sizeof(int)));
  EXPECT_EQ(&arr[0],
            ContiguousAccessor::GetPointerAtOffset<int>(ptr, base_index + 0));
  EXPECT_EQ(&arr[1],
            ContiguousAccessor::GetPointerAtOffset<int>(ptr, base_index + 1));
}

TEST(StridedAccessorTest, Basic) {
  int arr[3] = {1, 2, 3};
  IterationBufferPointer ptr{&arr[0], sizeof(int) * 2};
  EXPECT_EQ(&arr[0], StridedAccessor::GetPointerAtOffset<int>(ptr, 0));
  EXPECT_EQ(&arr[2], StridedAccessor::GetPointerAtOffset<int>(ptr, 1));
}

TEST(StridedAccessorTest, WrapOnOverflow) {
  int arr[3] = {1, 2, 3};
  IterationBufferPointer ptr{&arr[0], sizeof(int) * 2};
  const Index base_index = std::numeric_limits<Index>::max() - 3;
  ptr.pointer -= tensorstore::internal::wrap_on_overflow::Multiply(
      base_index, ptr.byte_stride);
  EXPECT_EQ(&arr[0],
            StridedAccessor::GetPointerAtOffset<int>(ptr, base_index + 0));
  EXPECT_EQ(&arr[2],
            StridedAccessor::GetPointerAtOffset<int>(ptr, base_index + 1));
}

TEST(OffsetArrayAccessorTest, Basic) {
  int arr[3] = {1, 2, 3};
  Index offsets[] = {0, sizeof(int) * 2};
  IterationBufferPointer ptr{&arr[0], &offsets[0]};
  EXPECT_EQ(&arr[0], OffsetArrayAccessor::GetPointerAtOffset<int>(ptr, 0));
  EXPECT_EQ(&arr[2], OffsetArrayAccessor::GetPointerAtOffset<int>(ptr, 1));
}

TEST(OffsetArrayAccessorTest, WrapOnOverflow) {
  int arr[3] = {1, 2, 3};
  const Index base_index = std::numeric_limits<Index>::max() - 100;
  Index offsets[] = {base_index + 0, base_index + sizeof(int) * 2};
  IterationBufferPointer ptr{&arr[0], &offsets[0]};
  ptr.pointer -= base_index;
  EXPECT_EQ(&arr[0], OffsetArrayAccessor::GetPointerAtOffset<int>(ptr, 0));
  EXPECT_EQ(&arr[2], OffsetArrayAccessor::GetPointerAtOffset<int>(ptr, 1));
}

TEST(SimpleElementwiseFunctionTest, ArityOne) {
  struct AddOneB {
    // Not default constructible.
    AddOneB() = delete;
    bool operator()(int* x) const {
      if (*x > 0) return false;
      *x += 1;
      return true;
    }
  };
  ElementwiseFunction<1> function = SimpleElementwiseFunction<AddOneB(int)>();

  std::vector<int> arr{-5, -6, 1, 2};

  EXPECT_EQ(2, function[IterationBufferKind::kContiguous](
                   nullptr, 2, IterationBufferPointer{&arr[0], sizeof(int)}));
  EXPECT_THAT(arr, ::testing::ElementsAre(-4, -5, 1, 2));

  EXPECT_EQ(1,
            function[IterationBufferKind::kStrided](
                nullptr, 2, IterationBufferPointer{&arr[0], sizeof(int) * 2}));
  EXPECT_THAT(arr, ::testing::ElementsAre(-3, -5, 1, 2));

  Index offsets[] = {sizeof(int), sizeof(int)};
  EXPECT_EQ(2, function[IterationBufferKind::kIndexed](
                   nullptr, 2, IterationBufferPointer{&arr[0], &offsets[0]}));
  EXPECT_THAT(arr, ::testing::ElementsAre(-3, -3, 1, 2));
}

TEST(SimpleElementwiseFunctionTest, ArityOneCaptureLessLambda) {
  const auto add_one ABSL_ATTRIBUTE_UNUSED = [](int* x) {
    if (*x > 0) return false;
    *x += 1;
    return true;
  };
  ElementwiseFunction<1> function =
      SimpleElementwiseFunction<decltype(add_one)(int)>();

  std::vector<int> arr{-5, -6, 1, 2};

  EXPECT_EQ(2, function[IterationBufferKind::kContiguous](
                   nullptr, 2, IterationBufferPointer{&arr[0], sizeof(int)}));
  EXPECT_THAT(arr, ::testing::ElementsAre(-4, -5, 1, 2));
}

TEST(SimpleElementwiseFunctionTest, NonEmptyArityOne) {
  struct AddOneC {
    int value = 0;
    bool operator()(int* x) {
      ++value;
      if (*x > 0) return false;
      *x += 1;
      return true;
    }
  };
  AddOneC add_one;
  ElementwiseClosure<1> closure =
      SimpleElementwiseFunction<AddOneC(int)>::Closure(&add_one);
  EXPECT_EQ(&add_one, closure.context);
  std::vector<int> arr{-5, -6, 1, 2};

  EXPECT_EQ(
      2, (*closure.function)[IterationBufferKind::kContiguous](
             closure.context, 2, IterationBufferPointer{&arr[0], sizeof(int)}));
  EXPECT_THAT(arr, ::testing::ElementsAre(-4, -5, 1, 2));
  EXPECT_EQ(2, add_one.value);
}

TEST(SimpleElementwiseFunctionTest, NonEmptyArityOneBind) {
  struct AddOneD {
    bool operator()(int* x, int* counter) {
      ++*counter;
      if (*x > 0) return false;
      *x += 1;
      return true;
    }
  };
  int counter = 0;
  auto add_one = std::bind(AddOneD{}, std::placeholders::_1, &counter);
  ElementwiseClosure<1> closure =
      SimpleElementwiseFunction<decltype(add_one)(int)>::Closure(&add_one);
  std::vector<int> arr{-5, -6, 1, 2};

  EXPECT_EQ(2, (*closure.function)[IterationBufferKind::kContiguous](
                   &add_one, 2, IterationBufferPointer{&arr[0], sizeof(int)}));
  EXPECT_THAT(arr, ::testing::ElementsAre(-4, -5, 1, 2));
  EXPECT_EQ(2, counter);
}

TEST(SimpleElementwiseFunctionTest, ArityTwo) {
  struct Convert {
    bool operator()(int* x, double* y) const {
      *x = static_cast<int>(*y);
      return (*x < 0);
    }
  };
  ElementwiseFunction<2> function =
      SimpleElementwiseFunction<Convert(int, double)>();

  std::vector<int> arr{0, 0, 0, 0};
  std::vector<double> arr2{-3.5, -2.5, -1.5, 2.5};

  EXPECT_EQ(2, function[IterationBufferKind::kContiguous](
                   nullptr, 2, IterationBufferPointer{&arr[0], sizeof(int)},
                   IterationBufferPointer{&arr2[0], sizeof(double)}));
  EXPECT_THAT(arr, ::testing::ElementsAre(-3, -2, 0, 0));

  EXPECT_EQ(2, function[IterationBufferKind::kStrided](
                   nullptr, 2, IterationBufferPointer{&arr[0], sizeof(int) * 2},
                   IterationBufferPointer{&arr2[0], sizeof(double)}));
  EXPECT_THAT(arr, ::testing::ElementsAre(-3, -2, -2, 0));

  Index offsets[] = {0, sizeof(int), 2 * sizeof(int)};
  Index offsets2[] = {sizeof(double), sizeof(double) * 3, 0};
  EXPECT_EQ(1, function[IterationBufferKind::kIndexed](
                   nullptr, 3, IterationBufferPointer{&arr[0], &offsets[0]},
                   IterationBufferPointer{&arr2[0], &offsets2[0]}));
  EXPECT_THAT(arr, ::testing::ElementsAre(-2, 2, -2, 0));
}

TEST(SimpleElementwiseFunctionTest, ArityOneExtraArgsIndexReturn) {
  struct AddOneA {
    bool operator()(int* x, int* sum) const {
      if (*x > 0) return false;
      *sum += *x;
      *x += 1;
      return true;
    }
  };
  ElementwiseFunction<1, int*> function =
      SimpleElementwiseFunction<AddOneA(int), int*>();

  std::vector<int> arr{-5, -6, 1, 2};

  {
    int sum = 0;
    EXPECT_EQ(
        2, function[IterationBufferKind::kContiguous](
               nullptr, 2, IterationBufferPointer{&arr[0], sizeof(int)}, &sum));
    EXPECT_EQ(-11, sum);
    EXPECT_THAT(arr, ::testing::ElementsAre(-4, -5, 1, 2));
  }

  {
    int sum = 0;
    EXPECT_EQ(1, function[IterationBufferKind::kStrided](
                     nullptr, 2,
                     IterationBufferPointer{&arr[0], sizeof(int) * 2}, &sum));
    EXPECT_THAT(arr, ::testing::ElementsAre(-3, -5, 1, 2));
    EXPECT_EQ(-4, sum);
  }

  {
    int sum = 0;
    Index offsets[] = {sizeof(int), sizeof(int)};
    EXPECT_EQ(
        2, function[IterationBufferKind::kIndexed](
               nullptr, 2, IterationBufferPointer{&arr[0], &offsets[0]}, &sum));
    EXPECT_THAT(arr, ::testing::ElementsAre(-3, -3, 1, 2));
    EXPECT_EQ(-9, sum);
  }
}

}  // namespace
