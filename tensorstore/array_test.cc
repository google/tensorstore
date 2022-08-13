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

#include "tensorstore/array.h"

#include <random>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/random/bit_gen_ref.h"
#include "absl/random/random.h"
#include <nlohmann/json.hpp>
#include "tensorstore/box.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/index_transform_testutil.h"
#include "tensorstore/internal/data_type_random_generator.h"
#include "tensorstore/internal/test_util.h"
#include "tensorstore/serialization/batch.h"
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/serialization/test_util.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

/// TENSORSTORE_EXPECT_DEATH_DEBUG_ONLY behaves similarly to
/// `EXPECT_DEBUG_DEATH` except that `stmt` is not executed when not in debug
/// mode.
///
/// This is useful in cases where `stmt` would result in undefined behavior when
/// not in debug mode, e.g. because it is intended to `assert` statements
/// designed to catch precondition violations.
#ifdef NDEBUG
#define TENSORSTORE_EXPECT_DEATH_DEBUG_ONLY(stmt, pattern)
#else
#define TENSORSTORE_EXPECT_DEATH_DEBUG_ONLY(stmt, pattern) \
  EXPECT_DEATH(stmt, pattern)
#endif

namespace {

using ::tensorstore::Array;
using ::tensorstore::ArrayIterateResult;
using ::tensorstore::ArrayOriginKind;
using ::tensorstore::ArrayView;
using ::tensorstore::BoxView;
using ::tensorstore::BroadcastArray;
using ::tensorstore::c_order;
using ::tensorstore::container;
using ::tensorstore::ContainerKind;
using ::tensorstore::ContiguousLayoutOrder;
using ::tensorstore::DimensionIndex;
using ::tensorstore::dtype_v;
using ::tensorstore::dynamic_rank;
using ::tensorstore::ElementPointer;
using ::tensorstore::fortran_order;
using ::tensorstore::Index;
using ::tensorstore::IsArrayExplicitlyConvertible;
using ::tensorstore::kInfIndex;
using ::tensorstore::kInfSize;
using ::tensorstore::MakeArray;
using ::tensorstore::MakeArrayView;
using ::tensorstore::MakeCopy;
using ::tensorstore::MakeOffsetArray;
using ::tensorstore::MakeScalarArrayView;
using ::tensorstore::MatchesStatus;
using ::tensorstore::offset_origin;
using ::tensorstore::SharedArray;
using ::tensorstore::SharedArrayView;
using ::tensorstore::SharedSubArray;
using ::tensorstore::span;
using ::tensorstore::StaticCast;
using ::tensorstore::StaticDataTypeCast;
using ::tensorstore::StaticRankCast;
using ::tensorstore::StrCat;
using ::tensorstore::StridedLayout;
using ::tensorstore::SubArray;
using ::tensorstore::SubArrayStaticRank;
using ::tensorstore::unchecked;
using ::tensorstore::ValidateShapeBroadcast;
using ::tensorstore::view;
using ::tensorstore::zero_origin;
using ::tensorstore::serialization::DecodeBatch;
using ::tensorstore::serialization::EncodeBatch;
using ::tensorstore::serialization::SerializationRoundTrip;
using ::tensorstore::serialization::TestSerializationRoundTrip;
using ::testing::ElementsAre;

namespace array_metafunctions_tests {
static_assert(IsArrayExplicitlyConvertible<int, dynamic_rank, zero_origin,
                                           const int, 2, zero_origin>);
static_assert(
    IsArrayExplicitlyConvertible<const void, dynamic_rank, zero_origin,
                                 const int, 2, zero_origin>);
static_assert(
    IsArrayExplicitlyConvertible<const void, dynamic_rank, zero_origin,
                                 const int, 2, offset_origin>);
static_assert(
    !IsArrayExplicitlyConvertible<const void, dynamic_rank, offset_origin,
                                  const int, 2, zero_origin>);
static_assert(
    !IsArrayExplicitlyConvertible<const void, dynamic_rank, offset_origin,
                                  const void, dynamic_rank, zero_origin>);
static_assert(!IsArrayExplicitlyConvertible<const void, 2, zero_origin,
                                            const void, 3, zero_origin>);
static_assert(!IsArrayExplicitlyConvertible<const int, 2, zero_origin,
                                            const float, 2, zero_origin>);

}  // namespace array_metafunctions_tests

namespace subarray_ref_tests {
static_assert(SubArrayStaticRank<dynamic_rank, span<const Index, 2>> ==
              dynamic_rank);
static_assert(SubArrayStaticRank<5, span<const Index>> == dynamic_rank);
static_assert(SubArrayStaticRank<5, span<const Index, 3>> == 2);
}  // namespace subarray_ref_tests

namespace strided_array_size_tests {
// Just a shared_ptr
static_assert(sizeof(SharedArray<float, 0>) == sizeof(void*) * 2);

// shared_ptr + 2 Index values
static_assert(sizeof(SharedArray<float, 1>) == (sizeof(void*) * 4));

// shared_ptr + DataType
static_assert(sizeof(SharedArray<void, 0>) == (sizeof(void*) * 3));

// shared_ptr + DataType + 2 Index values
static_assert(sizeof(SharedArray<void, 1>) == (sizeof(void*) * 5));

// shared_ptr + DataType + 1 DimensionIndex + 1 Index pointer
static_assert(sizeof(SharedArray<void>) == (sizeof(void*) * 5));

// Just a raw pointer.
static_assert(sizeof(ArrayView<float, 0>) == sizeof(void*));

// raw pointer + 2 Index pointers + 1 DimensionIndex
static_assert(sizeof(ArrayView<float>) == sizeof(void*) * 4);

// raw pointer + DataType + 2 Index pointers + 1 DimensionIndex
static_assert(sizeof(ArrayView<void>) == sizeof(void*) * 5, "");

// raw pointer + 2 Index pointers
static_assert(sizeof(ArrayView<float, 1>) == sizeof(void*) * 3, "");

// raw pointer + DataType + 2 Index pointers
static_assert(sizeof(ArrayView<void, 1>) == sizeof(void*) * 4, "");

// raw pointer + DataType
static_assert(sizeof(ArrayView<void, 0>) == sizeof(void*) * 2, "");
}  // namespace strided_array_size_tests

namespace make_array_ref_tests {

TEST(MakeArrayViewTest, Scalar) {
  int value = 3;
  auto result = MakeScalarArrayView(value);
  static_assert(std::is_same_v<decltype(result), ArrayView<int, 0>>);
  EXPECT_EQ(&value, result.data());
}

TEST(MakeArrayViewTest, Span) {
  std::vector<int> values{1, 2, 3};
  auto result = MakeArrayView(values);
  static_assert(std::is_same_v<decltype(result), Array<int, 1>>);
  EXPECT_EQ(values.data(), result.data());
  EXPECT_EQ(StridedLayout(ContiguousLayoutOrder::c, sizeof(int), {3}),
            result.layout());
  EXPECT_EQ(1, result(0));
  EXPECT_EQ(2, result(1));
  EXPECT_EQ(3, result(2));
}

// Test calls to MakeArrayView with a braced list.
static_assert(std::is_same_v<ArrayView<const int, 1>,
                             decltype(MakeArrayView({1, 2, 3}))>);
static_assert(std::is_same_v<ArrayView<const int, 2>,
                             decltype(MakeArrayView({{1, 2, 3}, {4, 5, 6}}))>);
static_assert(
    std::is_same_v<ArrayView<const int, 3>,
                   decltype(MakeArrayView({{{1, 2, 3}}, {{4, 5, 6}}}))>);
static_assert(
    std::is_same_v<ArrayView<const int, 4>,
                   decltype(MakeArrayView({{{{1, 2, 3}}, {{4, 5, 6}}}}))>);
static_assert(
    std::is_same_v<ArrayView<const int, 5>,
                   decltype(MakeArrayView({{{{{1, 2, 3}}, {{4, 5, 6}}}}}))>);
static_assert(
    std::is_same_v<ArrayView<const int, 6>,
                   decltype(MakeArrayView({{{{{{1, 2, 3}}, {{4, 5, 6}}}}}}))>);

TEST(MakeArrayViewTest, Rank1Array) {
  int values[] = {1, 2, 3};
  const int cvalues[] = {1, 2, 3};
  auto result = MakeArrayView(values);
  auto cresult = MakeArrayView(cvalues);
  static_assert(std::is_same_v<decltype(result), ArrayView<int, 1>>);
  static_assert(std::is_same_v<decltype(cresult), ArrayView<const int, 1>>);
  EXPECT_EQ(&values[0], result.data());
  EXPECT_EQ(&cvalues[0], cresult.data());
  auto layout = StridedLayout(ContiguousLayoutOrder::c, sizeof(int), {3});
  EXPECT_EQ(layout, result.layout());
  EXPECT_EQ(layout, cresult.layout());
  EXPECT_EQ(2, result(1));
}

TEST(MakeArrayViewTest, Rank2Array) {
  int values[2][3] = {{1, 2, 3}, {4, 5, 6}};
  const int cvalues[2][3] = {{1, 2, 3}, {4, 5, 6}};
  auto result = MakeArrayView(values);
  auto cresult = MakeArrayView(cvalues);
  static_assert(std::is_same_v<decltype(result), ArrayView<int, 2>>);
  static_assert(std::is_same_v<decltype(cresult), ArrayView<const int, 2>>);
  EXPECT_EQ(&values[0][0], result.data());
  EXPECT_EQ(&cvalues[0][0], cresult.data());
  auto layout = StridedLayout(ContiguousLayoutOrder::c, sizeof(int), {2, 3});
  EXPECT_EQ(layout, result.layout());
  EXPECT_EQ(layout, cresult.layout());
  EXPECT_EQ(6, result(1, 2));
}

TEST(MakeArrayViewTest, Rank3Array) {
  int values[1][2][3] = {{{1, 2, 3}, {4, 5, 6}}};
  const int cvalues[1][2][3] = {{{1, 2, 3}, {4, 5, 6}}};
  auto result = MakeArrayView(values);
  auto cresult = MakeArrayView(cvalues);
  static_assert(std::is_same_v<decltype(result), ArrayView<int, 3>>);
  static_assert(std::is_same_v<decltype(cresult), ArrayView<const int, 3>>);
  EXPECT_EQ(&values[0][0][0], result.data());
  EXPECT_EQ(&cvalues[0][0][0], cresult.data());
  auto layout = StridedLayout(ContiguousLayoutOrder::c, sizeof(int), {1, 2, 3});
  EXPECT_EQ(layout, result.layout());
  EXPECT_EQ(layout, cresult.layout());
  EXPECT_EQ(6, result(0, 1, 2));
}

TEST(MakeArrayViewTest, Rank4Array) {
  int values[1][1][2][3] = {{{{1, 2, 3}, {4, 5, 6}}}};
  const int cvalues[1][1][2][3] = {{{{1, 2, 3}, {4, 5, 6}}}};
  auto result = MakeArrayView(values);
  auto cresult = MakeArrayView(cvalues);
  static_assert(std::is_same_v<decltype(result), ArrayView<int, 4>>);
  static_assert(std::is_same_v<decltype(cresult), ArrayView<const int, 4>>);
  EXPECT_EQ(&values[0][0][0][0], result.data());
  auto layout =
      StridedLayout(ContiguousLayoutOrder::c, sizeof(int), {1, 1, 2, 3});
  EXPECT_EQ(layout, result.layout());
  EXPECT_EQ(layout, cresult.layout());
  EXPECT_EQ(6, result(0, 0, 1, 2));
}

TEST(MakeArrayViewTest, Rank5Array) {
  int values[1][1][1][2][3] = {{{{{1, 2, 3}, {4, 5, 6}}}}};
  const int cvalues[1][1][1][2][3] = {{{{{1, 2, 3}, {4, 5, 6}}}}};
  auto result = MakeArrayView(values);
  auto cresult = MakeArrayView(cvalues);
  static_assert(std::is_same_v<decltype(result), ArrayView<int, 5>>);
  static_assert(std::is_same_v<decltype(cresult), ArrayView<const int, 5>>);
  EXPECT_EQ(&values[0][0][0][0][0], result.data());
  auto layout =
      StridedLayout(ContiguousLayoutOrder::c, sizeof(int), {1, 1, 1, 2, 3});
  EXPECT_EQ(layout, result.layout());
  EXPECT_EQ(layout, cresult.layout());
  EXPECT_EQ(6, result(0, 0, 0, 1, 2));
}

TEST(MakeArrayViewTest, Rank6Array) {
  int values[1][1][1][2][3][1] = {{{{{{1}, {2}, {3}}, {{4}, {5}, {6}}}}}};
  const int cvalues[1][1][1][2][3][1] = {
      {{{{{1}, {2}, {3}}, {{4}, {5}, {6}}}}}};
  auto result = MakeArrayView(values);
  auto cresult = MakeArrayView(cvalues);
  static_assert(std::is_same_v<decltype(result), ArrayView<int, 6>>);
  static_assert(std::is_same_v<decltype(cresult), ArrayView<const int, 6>>);
  EXPECT_EQ(&values[0][0][0][0][0][0], result.data());
  EXPECT_EQ(&cvalues[0][0][0][0][0][0], cresult.data());
  auto layout =
      StridedLayout(ContiguousLayoutOrder::c, sizeof(int), {1, 1, 1, 2, 3, 1});
  EXPECT_EQ(layout, result.layout());
  EXPECT_EQ(layout, cresult.layout());
  EXPECT_EQ(6, result(0, 0, 0, 1, 2, 0));
}

}  // namespace make_array_ref_tests

namespace array_conversion_tests {

TEST(ArrayViewTest, ConstructDefault) {
  {
    ArrayView<int> p;
    EXPECT_EQ(0, p.rank());
    EXPECT_EQ(nullptr, p.data());
  }
  {
    ArrayView<void> p;
    EXPECT_EQ(0, p.rank());
    EXPECT_EQ(nullptr, p.data());
    EXPECT_FALSE(p.dtype().valid());
  }
}

TEST(ArrayViewTest, ConstructAndAssign) {
  static_assert(!std::is_convertible_v<ArrayView<float>, ArrayView<float, 2>>);
  static_assert(!std::is_convertible_v<ArrayView<float, 2>, ArrayView<int, 2>>);
  static_assert(
      !std::is_constructible_v<ArrayView<float, 2>, ArrayView<const float, 2>>);
  static_assert(
      !std::is_constructible_v<ArrayView<float, 2>, ArrayView<float, 3>>);
  static_assert(
      !std::is_assignable_v<ArrayView<float, 2>, ArrayView<const float, 2>>);
  static_assert(
      !std::is_assignable_v<ArrayView<float, 2>, ArrayView<float, 3>>);
  static_assert(!std::is_assignable_v<ArrayView<float, 2>, ArrayView<int, 2>>);
  static_assert(
      !std::is_convertible_v<ArrayView<void, 2>, ArrayView<float, 2>>);
  float data[2][3] = {{1, 2, 3}, {4, 5, 6}};
  ArrayView<float, 2> a = MakeArrayView(data);

  ArrayView<float> a1 = a;
  EXPECT_EQ(a.data(), a1.data());
  EXPECT_EQ(a.layout(), a1.layout());

  ArrayView<void> a2 = a1;
  EXPECT_EQ(a.data(), a2.data());
  EXPECT_EQ(a.layout(), a2.layout());
  EXPECT_EQ(dtype_v<float>, a2.dtype());

  {
    auto a3 = StaticDataTypeCast<float>(a2).value();
    static_assert(std::is_same_v<decltype(a3), ArrayView<float>>);
    EXPECT_EQ(a.data(), a3.data());
    EXPECT_EQ(a.layout(), a3.layout());
  }
  {
    ArrayView<float, 2> a4 = StaticCast<ArrayView<float, 2>>(a2).value();
    EXPECT_EQ(a.data(), a4.data());
    EXPECT_EQ(a.layout(), a4.layout());
  }

  {
    ArrayView<void, 2> a5(a.element_pointer(), a.layout());
    EXPECT_EQ(a.data(), a5.data());
    EXPECT_EQ(a.layout(), a5.layout());
    EXPECT_EQ(dtype_v<float>, a5.dtype());
  }

  {
    ArrayView<float, 2> a6;
    a6 = a;
    EXPECT_EQ(a.data(), a6.data());
    EXPECT_EQ(a.layout(), a6.layout());
    EXPECT_EQ(dtype_v<float>, a6.dtype());
  }
  static_assert(!std::is_assignable_v<ArrayView<float, 2>, ArrayView<float>>);
  static_assert(!std::is_assignable_v<ArrayView<float, 2>, ArrayView<void>>);
  {
    ArrayView<float> a6;
    a6 = a;
    EXPECT_EQ(a.data(), a6.data());
    EXPECT_EQ(a.layout(), a6.layout());
    EXPECT_EQ(dtype_v<float>, a6.dtype());
  }
  {
    ArrayView<float> a6;
    a6 = a1;
    EXPECT_EQ(a.data(), a6.data());
    EXPECT_EQ(a.layout(), a6.layout());
    EXPECT_EQ(dtype_v<float>, a6.dtype());
  }
  static_assert(!std::is_assignable_v<ArrayView<float>, ArrayView<void>>);
  {
    ArrayView<const void, 2> a6;
    a6 = a;
    EXPECT_EQ(a.data(), a6.data());
    EXPECT_EQ(a.layout(), a6.layout());
    EXPECT_EQ(dtype_v<float>, a6.dtype());
  }
  static_assert(
      !std::is_assignable_v<ArrayView<const void, 2>, ArrayView<float>>);
  static_assert(
      !std::is_assignable_v<ArrayView<const void, 2>, ArrayView<void>>);
  /// Test UnownedToShared array conversion.
  {
    SharedArray<void> a3_arr(UnownedToShared(a2));
    EXPECT_EQ(a2.element_pointer(), a3_arr.element_pointer());
    EXPECT_EQ(a2.layout(), a3_arr.layout());
    auto a4_ref = StaticCast<ArrayView<float>>(a3_arr).value();
    EXPECT_EQ(a2.element_pointer(), a4_ref.element_pointer());
    EXPECT_EQ(a2.layout(), a4_ref.layout());
  }
}

TEST(ArrayViewTest, StaticCast) {
  float data[2][3] = {{1, 2, 3}, {4, 5, 6}};
  ArrayView<void> a2 = MakeArrayView(data);

  SharedArray<void> a3(UnownedToShared(a2));

  EXPECT_THAT(
      (StaticCast<ArrayView<float, 3>>(a2)),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Cannot cast array with data type of float32 and rank of 2 "
                    "to array with data type of float32 and rank of 3"));
  EXPECT_THAT(
      (StaticCast<ArrayView<void, 3>>(a2)),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Cannot cast array with data type of float32 and rank of 2 "
                    "to array with dynamic data type and rank of 3"));

  EXPECT_THAT(
      (StaticCast<ArrayView<std::int32_t, 2>>(a2)),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Cannot cast array with data type of float32 and rank of 2 "
                    "to array with data type of int32 and rank of 2"));

  EXPECT_THAT(
      (StaticCast<ArrayView<std::int32_t>>(a2)),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Cannot cast array with data type of float32 and rank of 2 "
                    "to array with data type of int32 and dynamic rank"));

  EXPECT_THAT(
      (StaticCast<ArrayView<std::int32_t>>(a3)),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Cannot cast array with data type of float32 and rank of 2 "
                    "to array with data type of int32 and dynamic rank"));
}

TEST(SharedArrayTest, ConstructAndAssign) {
  static_assert(
      !std::is_convertible_v<SharedArray<float>, SharedArray<float, 2>>);
  static_assert(
      !std::is_convertible_v<const SharedArray<float>&, SharedArray<float, 2>>);
  static_assert(
      !std::is_convertible_v<SharedArray<float, 2>, SharedArray<int, 2>>);
  static_assert(!std::is_constructible_v<SharedArray<float, 2>,
                                         SharedArray<const float, 2>>);
  static_assert(!std::is_constructible_v<SharedArray<float, 2>,
                                         const SharedArray<const float, 2>&>);
  static_assert(
      !std::is_constructible_v<SharedArray<float, 2>, SharedArray<float, 3>>);
  static_assert(!std::is_assignable_v<SharedArray<float, 2>,
                                      SharedArray<const float, 2>>);
  static_assert(
      !std::is_assignable_v<SharedArray<float, 2>, SharedArray<float, 3>>);
  static_assert(
      !std::is_assignable_v<SharedArray<float, 2>, SharedArray<int, 2>>);
  static_assert(
      !std::is_assignable_v<SharedArray<float, 2>, const SharedArray<int, 2>&>);
  static_assert(
      !std::is_convertible_v<SharedArray<void, 2>, SharedArray<float, 2>>);
  static_assert(
      !std::is_convertible_v<SharedArray<void, 2>, SharedArray<float, 2>>);

  float data[2][3] = {{1, 2, 3}, {4, 5, 6}};
  ArrayView<float, 2> a_ref = MakeArrayView(data);

  SharedArray<float, 2> a(UnownedToShared(a_ref));
  EXPECT_EQ(a_ref.data(), a.data());
  EXPECT_EQ(a_ref.layout(), a.layout());

  SharedArray<float> a1(a);
  EXPECT_EQ(a.data(), a1.data());
  EXPECT_EQ(a.layout(), a1.layout());

  SharedArray<void> a2 = a1;
  EXPECT_EQ(a.data(), a2.data());
  EXPECT_EQ(a.layout(), a2.layout());
  EXPECT_EQ(dtype_v<float>, a2.dtype());

  {
    SharedArray<float> a3 = StaticDataTypeCast<float>(a2).value();
    EXPECT_EQ(a.data(), a3.data());
    EXPECT_EQ(a.layout(), a3.layout());
  }
  {
    SharedArray<float, 2> a4 = StaticCast<SharedArray<float, 2>>(a2).value();
    EXPECT_EQ(a.data(), a4.data());
    EXPECT_EQ(a.layout(), a4.layout());
  }

  {
    SharedArray<void, 2> a5(a.element_pointer(), a.layout());
    EXPECT_EQ(a.data(), a5.data());
    EXPECT_EQ(a.layout(), a5.layout());
    EXPECT_EQ(dtype_v<float>, a5.dtype());
  }

  {
    SharedArray<float, 2> a6;
    a6 = a;
    EXPECT_EQ(a.data(), a6.data());
    EXPECT_EQ(a.layout(), a6.layout());
    EXPECT_EQ(dtype_v<float>, a6.dtype());
  }

  static_assert(
      !std::is_assignable_v<SharedArray<float, 2>, SharedArray<float>>);
  static_assert(
      !std::is_assignable_v<SharedArray<float, 2>, SharedArray<void>>);
  {
    SharedArray<float> a6;
    a6 = a;
    EXPECT_EQ(a.data(), a6.data());
    EXPECT_EQ(a.layout(), a6.layout());
    EXPECT_EQ(dtype_v<float>, a6.dtype());
  }
  {
    SharedArray<float> a6;
    a6 = a1;
    EXPECT_EQ(a.data(), a6.data());
    EXPECT_EQ(a.layout(), a6.layout());
    EXPECT_EQ(dtype_v<float>, a6.dtype());
  }
  static_assert(!std::is_assignable_v<SharedArray<float>, SharedArray<void>>);
  {
    SharedArray<const void, 2> a6;
    a6 = a;
    EXPECT_EQ(a.data(), a6.data());
    EXPECT_EQ(a.layout(), a6.layout());
    EXPECT_EQ(dtype_v<float>, a6.dtype());
  }
  static_assert(
      !std::is_assignable_v<SharedArray<const void, 2>, SharedArray<void>>);
  static_assert(
      !std::is_assignable_v<SharedArray<const void, 2>, SharedArray<float>>);
  static_assert(
      !std::is_assignable_v<SharedArray<float, 2>, SharedArray<float>>);
  static_assert(!std::is_assignable_v<SharedArray<float, 2>, ArrayView<float>>);

  // Construct rank-0 array from just a pointer.
  {
    auto data = std::make_shared<float>(2.0f);
    SharedArray<float, 0> a7(data);
    EXPECT_EQ(data.get(), a7.data());
    EXPECT_EQ(0, a7.rank());
    EXPECT_EQ(2, a7());
  }

  // Move construct and assign.
  {
    auto data = std::make_shared<float>(2.0f);
    SharedArray<float> a7(
        data, StridedLayout(ContiguousLayoutOrder::c, sizeof(float), {1}));

    const Index* shape_ptr = a7.shape().data();
    const Index* byte_strides_ptr = a7.byte_strides().data();
    EXPECT_EQ(2, data.use_count());

    SharedArray<float> a8 = std::move(a7);
    EXPECT_EQ(nullptr, a7.data());  // NOLINT
    EXPECT_EQ(2, data.use_count());
    EXPECT_EQ(data.get(), a8.data());
    EXPECT_EQ(1, a8.rank());
    EXPECT_EQ(shape_ptr, a8.shape().data());
    EXPECT_EQ(byte_strides_ptr, a8.byte_strides().data());

    SharedArray<void> a9 = std::move(a8);
    EXPECT_EQ(nullptr, a8.data());  // NOLINT
    EXPECT_EQ(2, data.use_count());
    EXPECT_EQ(data.get(), a9.data());
    EXPECT_EQ(1, a9.rank());
    EXPECT_EQ(shape_ptr, a9.shape().data());
    EXPECT_EQ(byte_strides_ptr, a9.byte_strides().data());

    SharedArray<float> a10 = StaticDataTypeCast<float>(std::move(a9)).value();
    // Ideally, this would move the shared_ptr to avoid atomic operations.
    // However, static_pointer_cast and the shared_ptr aliasing constructor does
    // not permit this.
    EXPECT_EQ(data.get(), a10.data());
    EXPECT_EQ(1, a10.rank());
    EXPECT_EQ(shape_ptr, a10.shape().data());
    EXPECT_EQ(byte_strides_ptr, a10.byte_strides().data());

    SharedArray<void> a11;
    a11 = std::move(a10);
    EXPECT_EQ(nullptr, a10.data());  // NOLINT
    EXPECT_EQ(data.get(), a11.data());
    EXPECT_EQ(1, a11.rank());
    EXPECT_EQ(shape_ptr, a11.shape().data());
    EXPECT_EQ(byte_strides_ptr, a11.byte_strides().data());

    SharedArray<const void> a12;
    a12 = std::move(a11);
    EXPECT_EQ(nullptr, a11.data());  // NOLINT
    EXPECT_EQ(data.get(), a12.data());
    EXPECT_EQ(1, a12.rank());
    EXPECT_EQ(shape_ptr, a12.shape().data());
    EXPECT_EQ(byte_strides_ptr, a12.byte_strides().data());
  }

  // Assign from SharedArrayView via UnownedToShared.
  {
    SharedArray<void> x;
    x = UnownedToShared(a_ref);
    EXPECT_EQ(a_ref.layout(), x.layout());
    EXPECT_EQ(a_ref.element_pointer(), x.element_pointer());

    ArrayView<const void> y;
    y = x;
    EXPECT_EQ(a_ref.layout(), y.layout());
    EXPECT_EQ(a_ref.element_pointer(), y.element_pointer());
  }
}

TEST(ArrayTest, SubArray) {
  auto array = MakeArray<int>({{1, 2, 3}, {4, 5, 6}});
  EXPECT_THAT(array.shape(), ElementsAre(2, 3));

  EXPECT_EQ(1, array.pointer().use_count());
  auto sub_array = SubArray<container>(array, {1});
  static_assert(std::is_same_v<decltype(sub_array), Array<int, 1>>);
  EXPECT_THAT(sub_array.shape(), ElementsAre(3));
  EXPECT_EQ(array.data() + 3, sub_array.data());
  EXPECT_EQ(StridedLayout<1>({3}, {sizeof(int)}), sub_array.layout());

  auto sub_array_view = SubArray<view>(array, {1});
  static_assert(std::is_same_v<decltype(sub_array_view), ArrayView<int, 1>>);
  EXPECT_EQ(array.data() + 3, sub_array_view.data());
  EXPECT_EQ(StridedLayout<1>({3}, {sizeof(int)}), sub_array_view.layout());
}

TEST(ArrayTest, SharedSubArray) {
  auto array = MakeArray<int>({{1, 2, 3}, {4, 5, 6}});
  EXPECT_EQ(1, array.pointer().use_count());
  auto sub_array = SharedSubArray<container>(array, {1});
  static_assert(std::is_same_v<decltype(sub_array), SharedArray<int, 1>>);
  EXPECT_EQ(2, array.pointer().use_count());
  EXPECT_EQ(array.data() + 3, sub_array.data());
  EXPECT_EQ(StridedLayout<1>({3}, {sizeof(int)}), sub_array.layout());

  auto sub_array_view = SharedSubArray<view>(array, {1});
  static_assert(
      std::is_same_v<decltype(sub_array_view), SharedArrayView<int, 1>>);
  EXPECT_EQ(3, array.pointer().use_count());
  EXPECT_EQ(array.data() + 3, sub_array_view.data());
  EXPECT_EQ(StridedLayout<1>({3}, {sizeof(int)}), sub_array_view.layout());
}

TEST(ArrayTest, DynamicCast) {
  int data[2][3] = {{1, 2, 3}, {4, 5, 6}};
  ArrayView<int, 2> a_int_2 = MakeArrayView(data);

  auto a_int = StaticRankCast<dynamic_rank>(a_int_2).value();
  EXPECT_EQ(a_int_2.layout(), a_int.layout());
  EXPECT_EQ(a_int_2.element_pointer(), a_int.element_pointer());
  static_assert(std::is_same_v<decltype(a_int), ArrayView<int>>);
  StaticRankCast<dynamic_rank>(a_int).value();

  /// If a conversion is done, StaticRankCast returns by value.  However, for
  /// efficiency, a no-op StaticRankCast should just return an lvalue or rvalue
  /// reference to the argument.
  static_assert(
      std::is_same_v<decltype(StaticRankCast<dynamic_rank, unchecked>(a_int)),
                     ArrayView<int>&>);
  static_assert(std::is_same_v<decltype(StaticRankCast<dynamic_rank, unchecked>(
                                   std::declval<const ArrayView<int>&>())),
                               const ArrayView<int>&>);
  static_assert(std::is_same_v<decltype(StaticRankCast<dynamic_rank, unchecked>(
                                   std::declval<ArrayView<int>>())),
                               ArrayView<int>&&>);

  auto a_void_2 = StaticDataTypeCast<void>(a_int_2).value();
  EXPECT_EQ(a_int_2.layout(), a_void_2.layout());
  EXPECT_EQ(a_int_2.element_pointer(), a_void_2.element_pointer());
  static_assert(std::is_same_v<decltype(a_void_2), ArrayView<void, 2>>);

  StaticDataTypeCast<void>(a_void_2).value();
  static_assert(
      std::is_same_v<decltype(StaticDataTypeCast<void, unchecked>(a_void_2)),
                     ArrayView<void, 2>&>);
  static_assert(std::is_same_v<decltype(StaticDataTypeCast<void, unchecked>(
                                   std::declval<const ArrayView<void, 2>&>())),
                               const ArrayView<void, 2>&>);
  static_assert(std::is_same_v<decltype(StaticDataTypeCast<void, unchecked>(
                                   std::declval<ArrayView<void, 2>>())),
                               ArrayView<void, 2>&&>);

  auto a_void = StaticCast<ArrayView<void>>(a_int_2).value();
  EXPECT_EQ(a_int_2.layout(), a_void.layout());
  EXPECT_EQ(a_int_2.element_pointer(), a_void.element_pointer());
  static_assert(std::is_same_v<decltype(a_void), ArrayView<void>>);

  StaticCast<ArrayView<void>>(a_void).value();
  static_assert(
      std::is_same_v<decltype(StaticCast<ArrayView<void>, unchecked>(a_void)),
                     ArrayView<void>&>);
  static_assert(std::is_same_v<decltype(StaticCast<ArrayView<void>, unchecked>(
                                   std::declval<const ArrayView<void>&>())),
                               const ArrayView<void>&>);
  static_assert(std::is_same_v<decltype(StaticCast<ArrayView<void>, unchecked>(
                                   std::declval<ArrayView<void>>())),
                               ArrayView<void>&&>);

  auto b_int_2 =
      StaticCast<SharedArray<int, 2>>(UnownedToShared(a_void)).value();
  EXPECT_EQ(a_int_2.layout(), b_int_2.layout());
  EXPECT_EQ(a_int_2.element_pointer(), b_int_2.element_pointer());
  static_assert(std::is_same_v<decltype(b_int_2), SharedArray<int, 2>>);

  auto c_int_2 = StaticRankCast<2>(a_int).value();
  EXPECT_EQ(a_int_2.layout(), c_int_2.layout());
  EXPECT_EQ(a_int_2.element_pointer(), c_int_2.element_pointer());
  static_assert(std::is_same_v<decltype(c_int_2), ArrayView<int, 2>>);

  auto d_int_2 = StaticDataTypeCast<int>(a_void_2).value();
  EXPECT_EQ(a_int_2.layout(), d_int_2.layout());
  EXPECT_EQ(a_int_2.element_pointer(), d_int_2.element_pointer());
  static_assert(std::is_same_v<decltype(d_int_2), ArrayView<int, 2>>);
}

TEST(ArrayTest, OffsetOriginConstruct) {
  auto a = MakeOffsetArray<int>({7, 8}, {{1, 2, 3}, {4, 5, 6}});
  EXPECT_THAT(a.origin(), ElementsAre(7, 8));
  EXPECT_EQ(BoxView({7, 8}, {2, 3}), a.domain());
}

TEST(ArrayTest, ZeroOriginToOffsetOrigin) {
  SharedArray<int, 2> a = MakeArray<int>({{1, 2}, {3, 4}});
  SharedArray<int, 2, offset_origin> b(a);
  EXPECT_THAT(b.origin(), ElementsAre(0, 0));
  EXPECT_EQ(a.domain(), b.domain());
  EXPECT_EQ(a, b);
}

TEST(ArrayTest, ImplicitElementPointerConstruction) {
  int value = 5;
  // Test construction of dynamic_rank array with layout view from pointer.
  {
    ArrayView<void> a = &value;
    EXPECT_EQ(&value, a.data());
    EXPECT_EQ(0, a.rank());
  }

  // Test construction of dynamic_rank array with layout container from pointer.
  {
    tensorstore::Array<void> a = &value;
    EXPECT_EQ(&value, a.data());
    EXPECT_EQ(0, a.rank());
  }

  // Test construction of rank-0 array with layout view from pointer.
  {
    ArrayView<void, 0> a = &value;
    EXPECT_EQ(&value, a.data());
    EXPECT_EQ(0, a.rank());
  }

  // Test construction of rank-0 array with layout container from pointer.
  {
    tensorstore::Array<void, 0> a = &value;
    EXPECT_EQ(&value, a.data());
    EXPECT_EQ(0, a.rank());
  }

  // Test construction of rank-0 array from implicitly constructed
  // ElementPointer.
  {
    tensorstore::Array<void, 0> a = {
        {static_cast<void*>(&value), dtype_v<int>}};
    EXPECT_EQ(&value, a.data());
    EXPECT_EQ(0, a.rank());
  }

  // Test construction of a rank-0 array from implicitly constructed
  // ElementPointer and explicit layout.
  {
    tensorstore::Array<void, 0> a = {{static_cast<void*>(&value), dtype_v<int>},
                                     tensorstore::StridedLayout<0>()};
    EXPECT_EQ(&value, a.data());
    EXPECT_EQ(0, a.rank());
  }

  // Test assignment of a rank-0 array from implicitly constructed
  // ElementPointer.
  {
    tensorstore::Array<void, 0> a;
    a = &value;
    EXPECT_EQ(&value, a.data());
    EXPECT_EQ(0, a.rank());
  }

  // Test assignment of a rank-0 array from implicitly constructed
  // ElementPointer.
  {
    tensorstore::Array<void, 0> a;
    a = &value;
    EXPECT_EQ(&value, a.data());
    EXPECT_EQ(0, a.rank());
  }

  // Test assignment of a rank-0 array from implicitly constructed
  // ElementPointer and explicit layout.
  {
    tensorstore::Array<void, 0> a;
    a = {{static_cast<void*>(&value), dtype_v<int>},
         tensorstore::StridedLayout<0>()};
    EXPECT_EQ(&value, a.data());
    EXPECT_EQ(0, a.rank());
  }
}

TEST(ArrayTest, UnownedToSharedAliasing) {
  auto owned = std::make_shared<int>();
  int value;
  tensorstore::Array<int> arr = &value;
  EXPECT_EQ(1, owned.use_count());
  {
    auto alias_arr = UnownedToShared(owned, arr);
    static_assert(
        std::is_same_v<decltype(alias_arr), tensorstore::SharedArray<int>>);
    EXPECT_EQ(2, owned.use_count());
  }
  EXPECT_EQ(1, owned.use_count());
}

}  // namespace array_conversion_tests

namespace array_indexing_tests {

TEST(ArrayTest, Indexing) {
  const Index two = 2;
  int data[2][3] = {{1, 2, 3}, {4, 5, 6}};
  ArrayView<int, 2> a = MakeArrayView(data);
  EXPECT_EQ(1, a(0, 0));
  EXPECT_EQ(6, a(1, 2));
  EXPECT_EQ(6, a(1, two));
  EXPECT_EQ(6, a({1, 2}));
  EXPECT_EQ(6, a(span({1, 2})));
  EXPECT_EQ(MakeScalarArrayView<int>(6), a[1][2]);

  {
    auto a_sub = a[1];
    static_assert(std::is_same_v<decltype(a_sub), ArrayView<int, 1>>);
    EXPECT_EQ(&data[1][0], a_sub.data());
    EXPECT_EQ(6, a_sub(2));
    EXPECT_EQ(a.shape().data() + 1, a_sub.shape().data());
    EXPECT_EQ(a.byte_strides().data() + 1, a_sub.byte_strides().data());
  }

  {
    auto a_sub = a[span<const Index>({1, 2})];
    static_assert(std::is_same_v<decltype(a_sub), ArrayView<int>>);
    EXPECT_EQ(0, a_sub.rank());
    EXPECT_EQ(&data[1][2], a_sub.data());
    EXPECT_EQ(6, a_sub());
  }

  {
    auto a_sub = a[{1, 2}];
    static_assert(std::is_same_v<decltype(a_sub), ArrayView<int, 0>>);
    EXPECT_EQ(0, a_sub.rank());
    EXPECT_EQ(&data[1][2], a_sub.data());
    EXPECT_EQ(6, a_sub());
  }

  {
    ArrayView<int> a_d = a;
    auto a_sub = a_d[1];
    static_assert(std::is_same_v<decltype(a_sub), ArrayView<int>>);
    EXPECT_EQ(1, a_sub.rank());
    EXPECT_EQ(&data[1][0], a_sub.data());
    EXPECT_EQ(a_d.shape().data() + 1, a_sub.shape().data());
    EXPECT_EQ(a_d.byte_strides().data() + 1, a_sub.byte_strides().data());
  }
}

TEST(ArrayTest, OffsetOriginIndexing) {
  SharedArray<int, 2, offset_origin> a =
      MakeOffsetArray({7, 8}, {{1, 2, 3}, {4, 5, 6}});
  EXPECT_EQ(1, a(7, 8));
  EXPECT_EQ(5, a(8, 9));
  auto a_sub = a[7];
  EXPECT_THAT(a_sub.origin(), ElementsAre(8));
  EXPECT_THAT(a_sub.shape(), ElementsAre(3));
  EXPECT_EQ(3, a_sub(10));
}

TEST(ArrayDeathTest, Indexing) {
  int data[2][3] = {{1, 2, 3}, {4, 5, 6}};
  [[maybe_unused]] ArrayView<int, 2> a = MakeArrayView(data);
  TENSORSTORE_EXPECT_DEATH_DEBUG_ONLY(a(-1, 1), "Array index out of bounds");
  TENSORSTORE_EXPECT_DEATH_DEBUG_ONLY(a(2, 1), "Array index out of bounds");
  TENSORSTORE_EXPECT_DEATH_DEBUG_ONLY(
      (ArrayView<int>(a)[0][0][0]),
      "Length of index vector is greater than rank of array");
  TENSORSTORE_EXPECT_DEATH_DEBUG_ONLY(
      a(span<const Index>({1})),
      "Length of index vector must match rank of array");
  TENSORSTORE_EXPECT_DEATH_DEBUG_ONLY(
      a(span<const Index>({1, 2, 3})),
      "Length of index vector must match rank of array");
  TENSORSTORE_EXPECT_DEATH_DEBUG_ONLY(
      (a[span<const Index>({1, 2, 3})]),
      "Length of index vector is greater than rank of array");
}

TEST(ArrayDeathTest, OffsetOriginIndexing) {
  int data[2][3] = {{1, 2, 3}, {4, 5, 6}};
  SharedArray<int, 2, offset_origin> a = MakeOffsetArray({7, 8}, data);
  TENSORSTORE_EXPECT_DEATH_DEBUG_ONLY(a(0, 0), "Array index out of bounds");
  TENSORSTORE_EXPECT_DEATH_DEBUG_ONLY(a(7, 7), "Array index out of bounds");
  TENSORSTORE_EXPECT_DEATH_DEBUG_ONLY(
      (ArrayView<int, dynamic_rank, offset_origin>(a)[7][8][0]),
      "Length of index vector is greater than rank of array");
  TENSORSTORE_EXPECT_DEATH_DEBUG_ONLY(
      a(span<const Index>({1})),
      "Length of index vector must match rank of array");
  TENSORSTORE_EXPECT_DEATH_DEBUG_ONLY(
      a(span<const Index>({1, 2, 3})),
      "Length of index vector must match rank of array");
  TENSORSTORE_EXPECT_DEATH_DEBUG_ONLY(
      (a[span<const Index>({1, 2, 3})]),
      "Length of index vector is greater than rank of array");
}

}  // namespace array_indexing_tests

namespace allocate_and_construct_shared_elements_test {
TEST(AllocateAndConstructSharedElementsTest, StaticType) {
  auto result =
      tensorstore::internal::AllocateAndConstructSharedElements<float>(3);
  static_assert(std::is_same_v<decltype(result),
                               tensorstore::SharedElementPointer<float>>);

  EXPECT_NE(nullptr, result.data());
}

TEST(AllocateAndConstructSharedElementsTest, DynamicType) {
  auto result = tensorstore::internal::AllocateAndConstructSharedElements(
      3, tensorstore::default_init, dtype_v<float>);
  static_assert(std::is_same_v<decltype(result),
                               tensorstore::SharedElementPointer<void>>);

  EXPECT_EQ(dtype_v<float>, result.dtype());
  EXPECT_NE(nullptr, result.data());
}
}  // namespace allocate_and_construct_shared_elements_test

TEST(ToStringTest, Basic) {
  EXPECT_EQ("1", ToString(MakeScalarArrayView(1)));
  EXPECT_EQ("{1}", ToString(MakeArrayView({1})));
  EXPECT_EQ("{1, 2, 3}", ToString(MakeArrayView({1, 2, 3})));
  EXPECT_EQ("{{1, 2, 3}, {4, 5, 6}}",
            ToString(MakeArrayView({{1, 2, 3}, {4, 5, 6}})));
  EXPECT_EQ("<null>", ToString(ArrayView<const void>()));

  // Printing of type-erased arrays.
  ArrayView<const void> void_array = MakeArrayView<int>({1, 2, 3});
  EXPECT_EQ("{1, 2, 3}", ToString(void_array));

  tensorstore::ArrayFormatOptions options;
  options.summary_threshold = 10;
  options.summary_edge_items = 2;
  EXPECT_EQ("{{1, 2, 3, 4}, {5, 6, 7, 8}}",
            ToString(MakeArrayView({{1, 2, 3, 4}, {5, 6, 7, 8}}), options));
  EXPECT_EQ(
      "{{1, 2, 3, 4, 5}, {5, 6, 7, 8, 9}}",
      ToString(MakeArrayView({{1, 2, 3, 4, 5}, {5, 6, 7, 8, 9}}), options));
  options.summary_threshold = 9;
  EXPECT_EQ(
      "{{1, 2, ..., 4, 5}, {5, 6, ..., 8, 9}}",
      ToString(MakeArrayView({{1, 2, 3, 4, 5}, {5, 6, 7, 8, 9}}), options));

  std::ostringstream ostr;
  ostr << MakeScalarArrayView(3);
  EXPECT_EQ("3", ostr.str());
}

TEST(ArrayTest, Compare) {
  EXPECT_TRUE(MakeArrayView({{1, 2, 3}, {4, 5, 6}}) ==
              MakeArrayView({{1, 2, 3}, {4, 5, 6}}));

  EXPECT_FALSE(MakeArrayView({{1, 2, 3}, {4, 5, 6}}) !=
               MakeArrayView({{1, 2, 3}, {4, 5, 6}}));

  EXPECT_TRUE(MakeArrayView({{1, 5, 3}, {4, 5, 6}}) !=
              MakeArrayView({{1, 2, 3}, {4, 5, 6}}));
  EXPECT_FALSE(MakeArrayView({{1, 5, 3}, {4, 5, 6}}) ==
               MakeArrayView({{1, 2, 3}, {4, 5, 6}}));

  EXPECT_FALSE(ArrayView<void>(MakeScalarArrayView(1.0)) ==
               MakeScalarArrayView(1));
  EXPECT_TRUE(ArrayView<void>(MakeScalarArrayView(1.0)) !=
              MakeScalarArrayView(1));
  EXPECT_TRUE(MakeArrayView({1}) != MakeArrayView({1, 2}));
}

TEST(ArrayTest, SameValue) {
  EXPECT_TRUE(
      AreArraysSameValueEqual(MakeArrayView<float>({{1, 2, 3}, {4, 5, 6}}),
                              MakeArrayView<float>({{1, 2, 3}, {4, 5, 6}})));

  EXPECT_TRUE(
      AreArraysSameValueEqual(MakeArrayView<float>({{NAN, 2, 3}, {4, 5, 6}}),
                              MakeArrayView<float>({{NAN, 2, 3}, {4, 5, 6}})));

  EXPECT_FALSE(AreArraysSameValueEqual(
      MakeArrayView<float>({{NAN, 2, +0.0}, {4, 5, 6}}),
      MakeArrayView<float>({{NAN, 2, -0.0}, {4, 5, 6}})));
}

TEST(CopyArrayTest, ZeroOrigin) {
  int arr[2][3] = {{1, 2, 3}, {4, 5, 6}};
  auto arr_ref = MakeArrayView(arr);
  auto copy = MakeCopy(arr_ref);
  EXPECT_EQ(arr_ref, copy);

  InitializeArray(copy);
  EXPECT_EQ(MakeArrayView({{0, 0, 0}, {0, 0, 0}}), copy);
}

TEST(CopyArrayTest, OffsetOrigin) {
  int arr[2][3] = {{1, 2, 3}, {4, 5, 6}};
  auto source = MakeOffsetArray({7, 8}, arr);
  auto copy = MakeCopy(source);
  EXPECT_EQ(source, copy);
}

TEST(CopyConvertedArrayTest, Int32ToFloat32) {
  auto a = MakeArray<tensorstore::int32_t>({{1, 2, 3}, {4, 5, 6}});
  auto b = tensorstore::AllocateArray<tensorstore::float32_t>({2, 3});
  EXPECT_EQ(absl::OkStatus(), CopyConvertedArray(a, b));
  EXPECT_EQ(
      b, MakeArray<tensorstore::float32_t>({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}));
}

TEST(CopyConvertedArrayTest, Int32ToUint32) {
  auto a = MakeArray<tensorstore::int32_t>({{1, 2, 3}, {4, 5, 6}});
  auto b = tensorstore::AllocateArray<tensorstore::uint32_t>({2, 3});
  EXPECT_EQ(absl::OkStatus(), CopyConvertedArray(a, b));
  EXPECT_EQ(b, MakeArray<tensorstore::uint32_t>({{1, 2, 3}, {4, 5, 6}}));
}

TEST(CopyConvertedArrayTest, CopyError) {
  auto a = MakeArray<tensorstore::json_t>({3.0, "x"});
  auto b = tensorstore::AllocateArray<tensorstore::float32_t>({2});
  EXPECT_THAT(
      CopyConvertedArray(a, b),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          "Expected 64-bit floating-point number, but received: \"x\""));
}

TEST(CopyConvertedArrayTest, InvalidDataType) {
  auto a = MakeArray<tensorstore::string_t>({"x", "y"});
  auto b = tensorstore::AllocateArray<tensorstore::float32_t>({2});
  EXPECT_THAT(CopyConvertedArray(a, b),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Cannot convert string -> float32"));
}

TEST(MakeCopyTest, NoConversion) {
  const std::int32_t data[] = {1, 2, 3, 0, 4, 5, 6, 0};
  tensorstore::Array<const std::int32_t, 3> array(
      data, tensorstore::StridedLayout<3>({2, 2, 3}, {0, 4 * 4, 4}));

  auto expected = MakeArray<const std::int32_t>(
      {{{1, 2, 3}, {4, 5, 6}}, {{1, 2, 3}, {4, 5, 6}}});

  {
    // Default constraints (equivalent to {c_order, include_repeated_elements})
    auto copy = MakeCopy(array);
    EXPECT_EQ(expected, copy);
    EXPECT_THAT(copy.byte_strides(), ::testing::ElementsAre(24, 12, 4));
  }

  {
    auto copy = MakeCopy(
        array, {tensorstore::c_order, tensorstore::include_repeated_elements});
    EXPECT_EQ(expected, copy);
    EXPECT_THAT(copy.byte_strides(), ::testing::ElementsAre(24, 12, 4));
  }

  {
    auto copy = MakeCopy(array, {tensorstore::fortran_order,
                                 tensorstore::include_repeated_elements});
    EXPECT_EQ(expected, copy);
    // Shape of allocated array is {2, 2, 3}
    EXPECT_THAT(
        copy.byte_strides(),
        ::testing::ElementsAre(sizeof(std::int32_t), sizeof(std::int32_t) * 2,
                               sizeof(std::int32_t) * 2 * 2));
  }

  {
    auto copy = MakeCopy(
        array, {tensorstore::c_order, tensorstore::skip_repeated_elements});
    EXPECT_EQ(expected, copy);
    // Shape of allocated array is {1, 2, 3}
    EXPECT_THAT(copy.byte_strides(),
                ::testing::ElementsAre(0, 3 * sizeof(std::int32_t),
                                       sizeof(std::int32_t)));
  }

  {
    auto copy = MakeCopy(array, {tensorstore::fortran_order,
                                 tensorstore::skip_repeated_elements});
    EXPECT_EQ(expected, copy);
    // Shape of allocated array is {1, 2, 3}
    EXPECT_THAT(copy.byte_strides(),
                ::testing::ElementsAre(0, sizeof(std::int32_t),
                                       sizeof(std::int32_t) * 2));
  }

  {
    // Order is unspecified. `MakeCopy` determines order of dimensions (which
    // may match neither `c_order` nor `fortran_order`) that best matches the
    // input order.  In this case the chosen order matches `c_order`.
    auto copy = MakeCopy(array, {tensorstore::skip_repeated_elements});
    EXPECT_EQ(expected, copy);
    EXPECT_THAT(copy.byte_strides(), ::testing::ElementsAre(0, 12, 4));
  }

  {
    auto copy = MakeCopy(array, {tensorstore::include_repeated_elements});
    EXPECT_EQ(expected, copy);
    EXPECT_THAT(copy.byte_strides(), ::testing::ElementsAre(4, 24, 8));
  }
}

TEST(MakeCopyTest, Conversion) {
  const std::int32_t data[] = {1, 2, 3, 0, 4, 5, 6, 0};
  tensorstore::Array<const std::int32_t, 3> array(
      data, tensorstore::StridedLayout<3>({2, 2, 3}, {0, 4 * 4, 4}));

  auto expected =
      MakeArray<const float>({{{1, 2, 3}, {4, 5, 6}}, {{1, 2, 3}, {4, 5, 6}}});

  EXPECT_THAT(MakeCopy<std::byte>(array),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Cannot convert int32 -> byte"));

  {
    auto copy = MakeCopy<float>(array).value();
    EXPECT_EQ(expected, copy);
    EXPECT_THAT(copy.byte_strides(), ::testing::ElementsAre(24, 12, 4));
  }

  {
    auto copy = MakeCopy<float>(array, {tensorstore::fortran_order,
                                        tensorstore::skip_repeated_elements})
                    .value();
    EXPECT_EQ(expected, copy);
    EXPECT_THAT(copy.byte_strides(), ::testing::ElementsAre(0, 4, 8));
  }

  {
    auto copy = MakeCopy(array, {tensorstore::skip_repeated_elements},
                         tensorstore::DataType(dtype_v<float>))
                    .value();
    EXPECT_EQ(expected, copy);
    EXPECT_THAT(copy.byte_strides(), ::testing::ElementsAre(0, 12, 4));
  }

  {
    auto copy = MakeCopy<float>(array, {tensorstore::include_repeated_elements})
                    .value();
    EXPECT_EQ(expected, copy);
    EXPECT_THAT(copy.byte_strides(), ::testing::ElementsAre(4, 24, 8));
  }
}

template <tensorstore::ArrayOriginKind OriginKind>
void TestAllocateArrayLike(
    ArrayView<const int, 2, OriginKind> source,
    ArrayView<const int, 3, OriginKind> source_repeated) {
  {
    auto copy = tensorstore::AllocateArrayLike<int>(source.layout(),
                                                    ContiguousLayoutOrder::c);
    EXPECT_THAT(copy.byte_strides(),
                ::testing::ElementsAre(sizeof(int) * 3, sizeof(int)));
    EXPECT_EQ(source.domain(), copy.domain());
  }
  {
    auto copy = tensorstore::AllocateArrayLike<int>(
        source.layout(), ContiguousLayoutOrder::fortran);
    EXPECT_THAT(copy.byte_strides(),
                ::testing::ElementsAre(sizeof(int), sizeof(int) * 2));
    EXPECT_EQ(source.domain(), copy.domain());
  }
  {
    auto copy = tensorstore::AllocateArrayLike<int>(
        source_repeated.layout(),
        {ContiguousLayoutOrder::c, tensorstore::include_repeated_elements});
    EXPECT_THAT(
        copy.byte_strides(),
        ::testing::ElementsAre(sizeof(int) * 6, sizeof(int) * 3, sizeof(int)));
    EXPECT_EQ(source_repeated.domain(), copy.domain());
  }
  {
    auto copy = tensorstore::AllocateArrayLike<int>(
        source_repeated.layout(), {ContiguousLayoutOrder::fortran,
                                   tensorstore::include_repeated_elements});
    EXPECT_THAT(
        copy.byte_strides(),
        ::testing::ElementsAre(sizeof(int), sizeof(int) * 2, sizeof(int) * 4));
    EXPECT_EQ(source_repeated.domain(), copy.domain());
  }
  {
    auto copy = tensorstore::AllocateArrayLike<int>(
        source_repeated.layout(),
        {ContiguousLayoutOrder::c, tensorstore::skip_repeated_elements});
    EXPECT_THAT(copy.byte_strides(),
                ::testing::ElementsAre(sizeof(int) * 3, 0, sizeof(int)));
    EXPECT_EQ(source_repeated.domain(), copy.domain());
  }
  {
    auto copy = tensorstore::AllocateArrayLike<int>(
        source_repeated.layout(),
        {ContiguousLayoutOrder::fortran, tensorstore::skip_repeated_elements});
    EXPECT_THAT(copy.byte_strides(),
                ::testing::ElementsAre(sizeof(int), 0, sizeof(int) * 2));
    EXPECT_EQ(source_repeated.domain(), copy.domain());
  }
  {
    auto copy = tensorstore::AllocateArrayLike<int>(
        source_repeated.layout(), tensorstore::skip_repeated_elements);
    EXPECT_THAT(copy.byte_strides(),
                ::testing::ElementsAre(sizeof(int) * 3, 0, sizeof(int)));
    EXPECT_EQ(source_repeated.domain(), copy.domain());
  }
}

TEST(AllocateArrayLikeTest, ZeroOrigin) {
  int arr[2][3] = {{1, 2, 3}, {4, 5, 6}};
  auto source = MakeArray(arr);
  SharedArray<int, 3> source_repeated(
      source.element_pointer(),
      tensorstore::StridedLayoutView<3>({2, 2, 3},
                                        {sizeof(int) * 3, 0, sizeof(int)}));
  TestAllocateArrayLike<zero_origin>(source, source_repeated);
}

TEST(AllocateArrayLikeTest, OffsetOrigin) {
  int arr[2][3] = {{1, 2, 3}, {4, 5, 6}};
  auto source = MakeOffsetArray({7, 8}, arr);
  SharedArray<int, 3, offset_origin> source_repeated(
      source.element_pointer(),
      tensorstore::StridedLayoutView<3, offset_origin>(
          {7, 9, 8}, {2, 2, 3}, {sizeof(int) * 3, 0, sizeof(int)}));
  TestAllocateArrayLike<offset_origin>(source, source_repeated);
}

TEST(AllocateArrayTest, Default) {
  SharedArray<int, 2> result = tensorstore::AllocateArray<int>(
      {2, 3u}, ContiguousLayoutOrder::c, tensorstore::value_init);
  EXPECT_THAT(result.shape(), testing::ElementsAreArray({2, 3}));
  EXPECT_THAT(result.byte_strides(),
              testing::ElementsAreArray({3 * sizeof(int), sizeof(int)}));
  result(1, 2) = 1;
}

TEST(AllocateArrayElementsLikeTest, ZeroOrigin) {
  StridedLayout<2, zero_origin> source_layout({2, 3}, {1, 10});
  std::vector<Index> byte_strides(2);
  auto array_pointer = tensorstore::AllocateArrayElementsLike<int32_t>(
      source_layout, byte_strides.data(), tensorstore::skip_repeated_elements,
      tensorstore::value_init);
  ASSERT_THAT(byte_strides, testing::ElementsAre(4, 8));
  for (Index i = 0; i < source_layout.num_elements(); ++i) {
    EXPECT_EQ(0, array_pointer.data()[i]);
  }
}

TEST(AllocateArrayElementsLikeTest, ZeroOriginSkipRepeatedElements) {
  StridedLayout<2, zero_origin> source_layout({2, 3}, {0, 10});
  std::vector<Index> byte_strides(2);
  auto array_pointer = tensorstore::AllocateArrayElementsLike<int32_t>(
      source_layout, byte_strides.data(), tensorstore::skip_repeated_elements,
      tensorstore::value_init);
  ASSERT_THAT(byte_strides, testing::ElementsAre(0, 4));
  for (Index i = 0; i < 3; ++i) {
    EXPECT_EQ(0, array_pointer.data()[i]);
  }
}

TEST(AllocateArrayElementsLikeTest, OffsetOrigin) {
  StridedLayout<2, offset_origin> source_layout({1, 2}, {2, 3}, {1, 10});
  std::vector<Index> byte_strides(2);
  auto array_pointer = tensorstore::AllocateArrayElementsLike<int32_t>(
      source_layout, byte_strides.data(), tensorstore::skip_repeated_elements,
      tensorstore::value_init);
  ASSERT_THAT(byte_strides, testing::ElementsAre(4, 8));
  for (Index i = 0; i < source_layout.num_elements(); ++i) {
    EXPECT_EQ(0, array_pointer.data()[1 + 4 + i]);
  }
}

TEST(AllocateArrayElementsLikeTest, OffsetOriginSkipRepeatedElements) {
  StridedLayout<2, offset_origin> source_layout({1, 2}, {2, 3}, {0, 10});
  std::vector<Index> byte_strides(2);
  auto array_pointer = tensorstore::AllocateArrayElementsLike<int32_t>(
      source_layout, byte_strides.data(), tensorstore::skip_repeated_elements,
      tensorstore::value_init);
  EXPECT_THAT(byte_strides, testing::ElementsAre(0, 4));
  for (Index i = 0; i < 3; ++i) {
    EXPECT_EQ(0, array_pointer.data()[2 + i]);
  }
}

TEST(IterateOverArrays, VoidReturn) {
  std::vector<std::pair<int, int>> values;
  EXPECT_EQ(
      (ArrayIterateResult{true, 4}),
      IterateOverArrays(
          [&](const int* a, const int* b) { values.emplace_back(*a, *b); },
          /*constraints=*/ContiguousLayoutOrder::c,
          MakeArrayView({{1, 2}, {3, 4}}), MakeArrayView({{5, 6}, {7, 8}})));

  const std::vector<std::pair<int, int>> expected_values{
      {1, 5}, {2, 6}, {3, 7}, {4, 8}};
  EXPECT_EQ(expected_values, values);
}

TEST(IterateOverArrays, VoidReturnStatus) {
  std::vector<std::pair<int, int>> values;
  absl::Status status;
  EXPECT_EQ(
      (ArrayIterateResult{true, 4}),
      IterateOverArrays(
          [&](const int* a, const int* b, absl::Status* status_ptr) {
            values.emplace_back(*a, *b);
            EXPECT_EQ(&status, status_ptr);
          },
          &status,
          /*constraints=*/ContiguousLayoutOrder::c,
          MakeArrayView({{1, 2}, {3, 4}}), MakeArrayView({{5, 6}, {7, 8}})));

  const std::vector<std::pair<int, int>> expected_values{
      {1, 5}, {2, 6}, {3, 7}, {4, 8}};
  EXPECT_EQ(expected_values, values);
}

TEST(IterateOverArrays, BoolReturnZeroElements) {
  EXPECT_EQ((ArrayIterateResult{true, 0}),
            IterateOverArrays([&](const int* a) -> bool { return false; },
                              /*constraints=*/{},
                              tensorstore::AllocateArray<int>({0})));
}

TEST(IterateOverArrays, BoolTrueReturn) {
  std::vector<std::pair<int, int>> values;
  EXPECT_EQ(
      (ArrayIterateResult{true, 4}),
      IterateOverArrays(
          [&](const int* a, const int* b) -> bool {
            values.emplace_back(*a, *b);
            return true;
          },
          /*constraints=*/ContiguousLayoutOrder::c,
          MakeArrayView({{1, 2}, {3, 4}}), MakeArrayView({{5, 6}, {7, 8}})));

  const std::vector<std::pair<int, int>> expected_values{
      {1, 5}, {2, 6}, {3, 7}, {4, 8}};
  EXPECT_EQ(expected_values, values);
}

TEST(IterateOverArrays, BoolFalseReturn) {
  std::vector<std::pair<int, int>> values;
  EXPECT_EQ(
      (ArrayIterateResult{false, 1}),
      IterateOverArrays(
          [&](const int* a, const int* b) {
            values.emplace_back(*a, *b);
            return (*a != 2);
          },
          /*constraints=*/ContiguousLayoutOrder::c,
          MakeArrayView({{1, 2}, {3, 4}}), MakeArrayView({{5, 6}, {7, 8}})));

  const std::vector<std::pair<int, int>> expected_values{{1, 5}, {2, 6}};
  EXPECT_EQ(expected_values, values);
}

TEST(SharedArrayTest, Example) {
  using ::tensorstore::CopyArray;
  using ::tensorstore::MakeArray;
  using ::tensorstore::SharedArray;
  ///! [SharedArray usage example]

  SharedArray<int, 2> x = MakeArray<int>({{1, 2, 3}, {4, 5, 6}});
  SharedArray<int, 2> y = x;
  SharedArray<int, 2> z = MakeCopy(x);
  EXPECT_THAT(x.shape(), ::testing::ElementsAreArray({2, 3}));
  EXPECT_EQ("{{1, 2, 3}, {4, 5, 6}}", ToString(x));
  EXPECT_EQ("{1, 2, 3}", ToString(x[0]));
  EXPECT_EQ(6, x(1, 2));
  x(1, 2) = 7;
  x(0, 0) = 9;
  EXPECT_EQ("{{9, 2, 3}, {4, 5, 7}}", ToString(x));
  EXPECT_EQ("{{9, 2, 3}, {4, 5, 7}}", ToString(y));
  EXPECT_EQ("{{1, 2, 3}, {4, 5, 6}}", ToString(z));

  ///! [SharedArray usage example]
}

TEST(ArrayViewTest, Example) {
  using ::tensorstore::AllocateArray;
  using ::tensorstore::ArrayView;
  using ::tensorstore::IterateOverArrays;
  using ::tensorstore::MakeArray;
  using ::tensorstore::SharedArray;
  ///! [ArrayView usage example]

  const auto compute_sum = [&](ArrayView<int> a,
                               ArrayView<int> b) -> SharedArray<int> {
    auto c = AllocateArray<int>(a.shape());
    IterateOverArrays(
        [&](const int* a_v, const int* b_v, int* c_v) { *c_v = *a_v + *b_v; },
        /*constraints=*/{}, a, b, c);
    return c;
  };
  SharedArray<int, 2> a = MakeArray<int>({{1, 2}, {3, 4}});
  SharedArray<int, 2> b = MakeArray<int>({{5, 6}, {7, 9}});
  EXPECT_EQ(MakeArray<int>({{6, 8}, {10, 13}}), compute_sum(a, b));

  ///! [ArrayView usage example]
}

TEST(SharedArrayViewTest, Example) {
  using ::tensorstore::AllocateArray;
  using ::tensorstore::MakeArray;
  using ::tensorstore::SharedArray;
  using ::tensorstore::SharedArrayView;
  ///! [SharedArrayView usage example]

  const auto transpose = [&](SharedArrayView<int> x) -> SharedArray<int> {
    SharedArray<int> y(x);
    std::reverse(y.shape().begin(), y.shape().end());
    std::reverse(y.byte_strides().begin(), y.byte_strides().end());
    return y;
  };
  SharedArray<int, 2> a = MakeArray<int>({{1, 2, 3}, {4, 5, 6}});
  EXPECT_EQ(MakeArray<int>({{1, 4}, {2, 5}, {3, 6}}), transpose(a));

  ///! [SharedArrayView usage example]
}

TEST(DynamicArrayCastTest, Example) {
  using ::tensorstore::AllocateArray;
  using ::tensorstore::MakeArray;
  using ::tensorstore::SharedArray;
  using ::tensorstore::SharedArrayView;
  ///! [DynamicArrayCast usage example]

  SharedArray<void> a(MakeArray<int>({1, 2, 3}));
  auto b = StaticCast<SharedArray<int, 1>>(a).value();
  auto c = StaticCast<ArrayView<int, 1>>(a).value();
  auto d = StaticCast<ArrayView<const int, 1>>(a).value();

  ///! [DynamicArrayCast usage example]
  static_cast<void>(b);
  static_cast<void>(c);
  static_cast<void>(d);
}

TEST(DynamicElementCastTest, Example) {
  using ::tensorstore::AllocateArray;
  using ::tensorstore::MakeArray;
  using ::tensorstore::SharedArray;
  using ::tensorstore::SharedArrayView;
  ///! [DynamicElementCast usage example]

  SharedArray<void, 1> a = MakeArray<int>({1, 2, 3});
  auto b = StaticDataTypeCast<int>(a);  // Result<SharedArray<int, 1>>
  if (b) {
    // Conversion successful.
  }
  auto c =
      StaticDataTypeCast<const int>(a);  // Result<SharedArray<const int, 1>>
  if (c) {
    // Conversion successful.
  }

  ///! [DynamicElementCast usage example]
}

TEST(DynamicRankCastTest, Example) {
  using ::tensorstore::AllocateArray;
  using ::tensorstore::MakeArray;
  using ::tensorstore::SharedArray;
  using ::tensorstore::SharedArrayView;
  ///! [DynamicRankCast usage example]

  SharedArray<int> a(MakeArray<int>({1, 2, 3}));
  auto b = StaticRankCast<1>(a);  // Result<SharedArray<int, 1>>

  ///! [DynamicRankCast usage example]
  static_cast<void>(b);
}

TEST(SharedArrayTest, Domain) {
  SharedArray<int, 2> a = MakeArray<int>({{1, 2}, {3, 4}});
  auto box = a.domain();
  static_assert(std::is_same_v<decltype(box), tensorstore::BoxView<2>>);
  EXPECT_THAT(box.shape(), ::testing::ElementsAre(2, 2));
  EXPECT_THAT(box.origin(), ::testing::ElementsAre(0, 0));
  static_assert(tensorstore::HasBoxDomain<SharedArray<int, 2>>);
  EXPECT_EQ(box, GetBoxDomainOf(a));
}

TEST(SharedArrayTest, AllocateArrayFromDomain) {
  auto array = tensorstore::AllocateArray<int>(BoxView({1, 2}, {3, 4}),
                                               ContiguousLayoutOrder::c,
                                               tensorstore::value_init);
  EXPECT_THAT(array.byte_strides(),
              ::testing::ElementsAre(4 * sizeof(int), sizeof(int)));
  array(3, 5) = 10;
  array(1, 2) = 5;
  EXPECT_EQ(BoxView({1, 2}, {3, 4}), array.domain());
  EXPECT_EQ("{{5, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 10}} @ {1, 2}",
            ToString(array));
}

template <ContainerKind SourceLayoutCKind, ContainerKind TargetLayoutCKind>
void TestArrayOriginCastOffsetOriginToZeroOrigin() {
  auto source = MakeOffsetArray<int>({2, 3}, {{1, 2, 3}, {4, 5, 6}});
  SharedArray<int, 2, offset_origin, SourceLayoutCKind> source_copy = source;
  auto result =
      tensorstore::ArrayOriginCast<zero_origin, TargetLayoutCKind>(source);
  static_assert(std::is_same_v<tensorstore::Result<tensorstore::SharedArray<
                                   int, 2, zero_origin, TargetLayoutCKind>>,
                               decltype(result)>);
  EXPECT_EQ(MakeArray<int>({{1, 2, 3}, {4, 5, 6}}), *result);
}

template <ArrayOriginKind TargetOriginKind, ContainerKind TargetLayoutCKind,
          typename ElementTag, DimensionIndex Rank, ArrayOriginKind OriginKind,
          ContainerKind SourceLayoutCKind>
void TestArrayOriginCastImplicitCase(
    Array<ElementTag, Rank, OriginKind, SourceLayoutCKind> source) {
  auto result =
      tensorstore::ArrayOriginCast<TargetOriginKind, TargetLayoutCKind>(source);
  static_assert(
      std::is_same_v<tensorstore::Array<ElementTag, Rank, TargetOriginKind,
                                        TargetLayoutCKind>,
                     decltype(result)>);
  EXPECT_EQ(source, result);
}

TEST(ArrayOriginCastTest, OffsetOriginToZeroOriginSuccess) {
  TestArrayOriginCastOffsetOriginToZeroOrigin<container, container>();
  TestArrayOriginCastOffsetOriginToZeroOrigin<container, view>();
  TestArrayOriginCastOffsetOriginToZeroOrigin<view, container>();
  TestArrayOriginCastOffsetOriginToZeroOrigin<view, view>();
}

TEST(ArrayOriginCastTest, OffsetOriginToZeroOriginFailure) {
  int value = 1;
  Array<int, 1, offset_origin> array;
  array.element_pointer() = &value;
  array.byte_strides()[0] = 0;
  array.origin()[0] = -kInfIndex;
  array.shape()[0] = kInfSize;
  EXPECT_THAT(tensorstore::ArrayOriginCast<zero_origin>(array),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            StrCat("Cannot translate array with shape \\{",
                                   kInfSize, "\\} to have zero origin\\.")));
}

TEST(ArrayOriginCastTest, ImplicitCaseOffsetOriginSource) {
  SharedArray<int, 2, offset_origin> array =
      MakeOffsetArray<int>({2, 3}, {{1, 2, 3}, {3, 4, 5}});
  TestArrayOriginCastImplicitCase<offset_origin, container>(array);
  TestArrayOriginCastImplicitCase<offset_origin, view>(array);

  const SharedArrayView<int, 2, offset_origin> array_view = array;
  TestArrayOriginCastImplicitCase<offset_origin, container>(array_view);
  TestArrayOriginCastImplicitCase<offset_origin, view>(array_view);
}

TEST(ArrayOriginCastTest, ImplicitCaseZeroOriginSource) {
  SharedArray<int, 2> array = MakeArray<int>({{1, 2, 3}, {3, 4, 5}});
  TestArrayOriginCastImplicitCase<offset_origin, container>(array);
  TestArrayOriginCastImplicitCase<offset_origin, view>(array);
  TestArrayOriginCastImplicitCase<zero_origin, container>(array);
  TestArrayOriginCastImplicitCase<zero_origin, view>(array);

  SharedArrayView<int, 2> array_view = array;
  TestArrayOriginCastImplicitCase<offset_origin, container>(array_view);
  TestArrayOriginCastImplicitCase<offset_origin, view>(array_view);
  TestArrayOriginCastImplicitCase<zero_origin, container>(array_view);
  TestArrayOriginCastImplicitCase<zero_origin, view>(array_view);
}

TEST(ArrayTest, ConstructContiguousBracedList) {
  int data[6] = {1, 2, 3, 4, 5, 6};
  // Defaults to `c_order`.
  Array<int, 2> array(data, {2, 3});
  EXPECT_EQ(data, array.data());
  EXPECT_EQ(array, MakeArray<int>({{1, 2, 3}, {4, 5, 6}}));

  // Explicit `c_order`.
  Array<int, 2> array_c_order(data, {2, 3}, c_order);
  EXPECT_EQ(data, array_c_order.data());
  EXPECT_EQ(array_c_order, MakeArray<int>({{1, 2, 3}, {4, 5, 6}}));

  // Explicit `fortran_order`.
  Array<int, 2> array_f_order(data, {3, 2}, fortran_order);
  EXPECT_EQ(data, array_f_order.data());
  EXPECT_EQ(array_f_order, MakeArray<int>({{1, 4}, {2, 5}, {3, 6}}));

  Array<int, 2, offset_origin> array_offset_origin(data, {2, 3});
  EXPECT_EQ(data, array.data());
  EXPECT_EQ(array_offset_origin, MakeArray<int>({{1, 2, 3}, {4, 5, 6}}));
}

TEST(ArrayTest, ConstructContiguousSpan) {
  int data[6] = {1, 2, 3, 4, 5, 6};
  // Defaults to `c_order`.
  Array<int, 2> array(data, span<const Index, 2>({2, 3}));
  EXPECT_EQ(data, array.data());
  EXPECT_EQ(array, MakeArray<int>({{1, 2, 3}, {4, 5, 6}}));

  // Explicit `c_order`.
  Array<int, 2> array_c_order(data, span<const Index, 2>({2, 3}), c_order);
  EXPECT_EQ(data, array_c_order.data());
  EXPECT_EQ(array_c_order, MakeArray<int>({{1, 2, 3}, {4, 5, 6}}));

  // Explicit `fortran_order`.
  Array<int, 2> array_f_order(data, span<const Index, 2>({3, 2}),
                              fortran_order);
  EXPECT_EQ(data, array_f_order.data());
  EXPECT_EQ(array_f_order, MakeArray<int>({{1, 4}, {2, 5}, {3, 6}}));

  Array<int, 2, offset_origin> array_offset_origin(
      data, span<const Index, 2>({2, 3}));
  EXPECT_EQ(data, array.data());
  EXPECT_EQ(array_offset_origin, MakeArray<int>({{1, 2, 3}, {4, 5, 6}}));
}

TEST(ArrayTest, ConstructContiguousBox) {
  int data[6] = {1, 2, 3, 4, 5, 6};
  // Defaults to `c_order`.
  Array<int, 2, offset_origin> array(data, BoxView({1, 2}, {2, 3}));
  EXPECT_EQ(&data[0], array.byte_strided_origin_pointer());
  EXPECT_EQ(array, MakeOffsetArray<int>({1, 2}, {{1, 2, 3}, {4, 5, 6}}));

  // Explicit `c_order`.
  Array<int, 2, offset_origin> array_c_order(data, BoxView({1, 2}, {2, 3}),
                                             c_order);
  EXPECT_EQ(&data[0], array_c_order.byte_strided_origin_pointer());
  EXPECT_EQ(array_c_order,
            MakeOffsetArray<int>({1, 2}, {{1, 2, 3}, {4, 5, 6}}));

  // Explicit `fortran_order`.
  Array<int, 2, offset_origin> array_f_order(data, BoxView({1, 2}, {3, 2}),
                                             fortran_order);
  EXPECT_EQ(&data[0], array_f_order.byte_strided_origin_pointer());
  EXPECT_EQ(array_f_order,
            MakeOffsetArray<int>({1, 2}, {{1, 4}, {2, 5}, {3, 6}}));
}

TEST(ArrayTest, DeductionGuides) {
  int value = 42;
  int* raw_p = &value;
  int data[] = {1, 2, 3, 4, 5, 6};
  std::shared_ptr<int> shared_p = std::make_shared<int>(42);
  auto existing_array = MakeArray<int>({{1, 2, 3}, {4, 5, 6}});
  ElementPointer<int> int_el_p = raw_p;
  ElementPointer<void> void_el_p = raw_p;
  ElementPointer<tensorstore::Shared<void>> shared_el_p = shared_p;
  StridedLayout<3, zero_origin> layout_3_zero;
  StridedLayout<3, offset_origin> layout_3_offset;
  StridedLayout<dynamic_rank, zero_origin> layout_dynamic_zero;
  StridedLayout<dynamic_rank, offset_origin> layout_dynamic_offset;

  {
    auto a = Array(raw_p);
    static_assert(std::is_same_v<decltype(a), Array<int, 0>>);
    EXPECT_EQ(a, tensorstore::MakeScalarArray<int>(42));
  }

  {
    auto a = Array(shared_p);
    static_assert(std::is_same_v<decltype(a), SharedArray<int, 0>>);
    EXPECT_EQ(a, tensorstore::MakeScalarArray<int>(42));
  }

  {
    auto a = Array(int_el_p);
    static_assert(std::is_same_v<decltype(a), Array<int, 0>>);
    EXPECT_EQ(a, tensorstore::MakeScalarArray<int>(42));
  }

  {
    auto a = Array(void_el_p);
    static_assert(std::is_same_v<decltype(a), Array<void, 0>>);
    EXPECT_EQ(a, tensorstore::MakeScalarArray<int>(42));
  }

  {
    auto a = Array(shared_el_p);
    static_assert(std::is_same_v<decltype(a), SharedArray<void, 0>>);
    EXPECT_EQ(a, tensorstore::MakeScalarArray<int>(42));
  }

  {
    auto a = Array(raw_p, layout_3_zero);
    static_assert(std::is_same_v<decltype(a), Array<int, 3>>);
  }

  {
    auto a = Array(raw_p, layout_3_offset);
    static_assert(std::is_same_v<decltype(a), Array<int, 3, offset_origin>>);
  }

  {
    auto a = Array(shared_p, layout_3_zero);
    static_assert(std::is_same_v<decltype(a), SharedArray<int, 3>>);
  }

  {
    auto a = Array(shared_p, layout_dynamic_zero);
    static_assert(std::is_same_v<decltype(a), SharedArray<int>>);
  }

  {
    auto a = Array(shared_p, layout_dynamic_offset);
    static_assert(
        std::is_same_v<decltype(a),
                       SharedArray<int, dynamic_rank, offset_origin>>);
  }

  {
    auto a = Array(void_el_p, layout_dynamic_offset);
    static_assert(
        std::is_same_v<decltype(a), Array<void, dynamic_rank, offset_origin>>);
  }

  {
    auto a = Array(data, {2, 3});
    static_assert(std::is_same_v<decltype(a), Array<int, 2>>);
    EXPECT_EQ(existing_array, a);
  }

  {
    auto a = Array(data, {2, 3}, c_order);
    static_assert(std::is_same_v<decltype(a), Array<int, 2>>);
    EXPECT_EQ(existing_array, a);
  }
}
TEST(ValidateShapeBroadcastTest, Examples) {
  TENSORSTORE_EXPECT_OK(ValidateShapeBroadcast(span<const Index>({5}),
                                               span<const Index>({4, 5})));
  TENSORSTORE_EXPECT_OK(ValidateShapeBroadcast(span<const Index>({4, 1}),
                                               span<const Index>({4, 5})));
  TENSORSTORE_EXPECT_OK(ValidateShapeBroadcast(span<const Index>({1, 1, 5}),
                                               span<const Index>({4, 5})));
  TENSORSTORE_EXPECT_OK(ValidateShapeBroadcast(span<const Index>({1, 1, 5}),
                                               span<const Index>({4, 5})));
  EXPECT_THAT(ValidateShapeBroadcast(span<const Index>({2, 5}),
                                     span<const Index>({4, 5})),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ValidateShapeBroadcast(span<const Index>({2, 5}),
                                     span<const Index>({5, 5})),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST(ValidateShapeBroadcastTest, Basic) {
  TENSORSTORE_EXPECT_OK(
      ValidateShapeBroadcast(span<const Index>(), span<const Index>()));
  TENSORSTORE_EXPECT_OK(
      ValidateShapeBroadcast(span<const Index>(), span<const Index>({3, 4})));
  TENSORSTORE_EXPECT_OK(ValidateShapeBroadcast(span<const Index>({3, 4}),
                                               span<const Index>({3, 4})));
  TENSORSTORE_EXPECT_OK(ValidateShapeBroadcast(span<const Index>({1, 3, 4}),
                                               span<const Index>({3, 4})));
  TENSORSTORE_EXPECT_OK(ValidateShapeBroadcast(span<const Index>({1, 1, 3, 4}),
                                               span<const Index>({3, 4})));
  TENSORSTORE_EXPECT_OK(ValidateShapeBroadcast(span<const Index>({1, 1, 1, 4}),
                                               span<const Index>({3, 4})));
  TENSORSTORE_EXPECT_OK(ValidateShapeBroadcast(span<const Index>({1, 1, 3, 1}),
                                               span<const Index>({3, 4})));
  TENSORSTORE_EXPECT_OK(ValidateShapeBroadcast(span<const Index>({3, 1}),
                                               span<const Index>({3, 4})));
  TENSORSTORE_EXPECT_OK(ValidateShapeBroadcast(span<const Index>({4}),
                                               span<const Index>({3, 4})));
  TENSORSTORE_EXPECT_OK(ValidateShapeBroadcast(span<const Index>({1}),
                                               span<const Index>({3, 4})));
  EXPECT_THAT(
      ValidateShapeBroadcast(span<const Index>({5}), span<const Index>({3, 4})),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Cannot broadcast array of shape \\{5\\} to target shape "
                    "\\{3, 4\\}"));
}

TEST(BroadcastStridedLayoutTest, Basic) {
  StridedLayout<1> source_layout({3}, {5});
  StridedLayout<2> target_layout({4, 3}, {42, 42});
  TENSORSTORE_ASSERT_OK(
      tensorstore::BroadcastStridedLayout(source_layout, target_layout.shape(),
                                          target_layout.byte_strides().data()));
  EXPECT_THAT(target_layout.byte_strides(), ::testing::ElementsAre(0, 5));
}

TEST(BroadcastArrayTest, ZeroOrigin) {
  EXPECT_THAT(
      BroadcastArray(MakeArray<int>({1, 2, 3}), span<const Index>({2, 3})),
      MakeArray<int>({{1, 2, 3}, {1, 2, 3}}));
  EXPECT_THAT(BroadcastArray(MakeArray<int>({{1}, {2}, {3}}),
                             span<const Index>({3, 2})),
              MakeArray<int>({{1, 1}, {2, 2}, {3, 3}}));
  EXPECT_THAT(BroadcastArray(MakeArray<int>({{1}, {2}, {3}}),
                             span<const Index>({4, 2})),
              MatchesStatus(
                  absl::StatusCode::kInvalidArgument,
                  "Cannot broadcast array of shape \\{3, 1\\} to target shape "
                  "\\{4, 2\\}"));
}

TEST(BroadcastArrayTest, OffsetOrigin) {
  EXPECT_THAT(BroadcastArray(MakeOffsetArray<int>({3}, {1, 2, 3}),
                             BoxView<>({1, 2}, {2, 3})),
              MakeOffsetArray<int>({1, 2}, {{1, 2, 3}, {1, 2, 3}}));
  EXPECT_THAT(BroadcastArray(MakeOffsetArray<int>({3, 4}, {{1}, {2}, {3}}),
                             BoxView<>({1, 2}, {3, 2})),
              MakeOffsetArray<int>({1, 2}, {{1, 1}, {2, 2}, {3, 3}}));
  EXPECT_THAT(BroadcastArray(MakeOffsetArray<int>({3, 4}, {{1}, {2}, {3}}),
                             BoxView<>({1, 2}, {4, 2})),
              MatchesStatus(
                  absl::StatusCode::kInvalidArgument,
                  "Cannot broadcast array of shape \\{3, 1\\} to target shape "
                  "\\{4, 2\\}"));
}

TEST(UnbroadcastArrayTest, Basic) {
  auto orig_array = MakeArray<int>({{{1, 2}}, {{3, 4}}, {{5, 6}}});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto broadcast_array,
      BroadcastArray(orig_array, BoxView<>({1, 2, 3, 4}, {2, 3, 2, 2})));
  auto unbroadcast_array = UnbroadcastArray(broadcast_array);
  auto unbroadcast_array2 = UnbroadcastArray(unbroadcast_array);
  EXPECT_EQ(orig_array, unbroadcast_array);
  EXPECT_EQ(orig_array.pointer(), unbroadcast_array.pointer());
  EXPECT_EQ(orig_array.layout(), unbroadcast_array.layout());
  EXPECT_EQ(orig_array, unbroadcast_array2);
  EXPECT_EQ(orig_array.pointer(), unbroadcast_array2.pointer());
  EXPECT_EQ(orig_array.layout(), unbroadcast_array2.layout());
}

TEST(UnbroadcastArrayTest, PreserveRank) {
  auto orig_array = MakeArray<int>({{{1, 2}}, {{3, 4}}, {{5, 6}}});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto broadcast_array1,
      BroadcastArray(orig_array, BoxView<>({1, 3, 1, 2})));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto broadcast_array2,
      BroadcastArray(orig_array, BoxView<>({1, 2, 3, 4}, {2, 3, 2, 2})));
  auto unbroadcast_array2 = UnbroadcastArrayPreserveRank(broadcast_array2);
  EXPECT_EQ(unbroadcast_array2.pointer(), broadcast_array1.pointer());
  EXPECT_EQ(unbroadcast_array2.layout(), broadcast_array1.layout());
}

TEST(ArraySerializationTest, ZeroOrigin) {
  tensorstore::SharedArray<int, 2> array =
      tensorstore::MakeArray<int>({{1, 2, 3}, {4, 5, 6}});
  TestSerializationRoundTrip(array);
  TestSerializationRoundTrip(tensorstore::SharedArray<void, 2>(array));
  TestSerializationRoundTrip(tensorstore::SharedArray<int>(array));
  TestSerializationRoundTrip(tensorstore::SharedArray<void>(array));
}

TEST(ArraySerializationTest, OffsetOrigin) {
  tensorstore::SharedOffsetArray<int, 2> array =
      tensorstore::MakeOffsetArray<int>({7, 8}, {{1, 2, 3}, {4, 5, 6}});
  TestSerializationRoundTrip(array);
  TestSerializationRoundTrip(tensorstore::SharedOffsetArray<void, 2>(array));
  TestSerializationRoundTrip(tensorstore::SharedOffsetArray<int>(array));
  TestSerializationRoundTrip(tensorstore::SharedOffsetArray<void>(array));
}

// Tests that singleton dimensions (with zero stride) are correctly preserved.
TEST(ArraySerializationTest, ZeroStrides) {
  int data[] = {1, 2, 3, 4, 5, 6};
  tensorstore::SharedArray<int> array(
      std::shared_ptr<int>(std::shared_ptr<void>(), &data[0]),
      StridedLayout<>({kInfIndex + 1, 2, 3, kInfIndex + 1},
                      {0, 3 * sizeof(int), sizeof(int), 0}));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto copy, SerializationRoundTrip(array));
  ASSERT_EQ(array.layout(), copy.layout());
  EXPECT_EQ(array, copy);
}

TEST(ArraySerializationTest, DataTypeMismatch) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto encoded,
                                   EncodeBatch(MakeArray<int>({1, 2, 3})));
  SharedArray<float> array;
  EXPECT_THAT(
      DecodeBatch(encoded, array),
      MatchesStatus(absl::StatusCode::kDataLoss,
                    "Expected data type of float32 but received: int32; .*"));
}

TEST(ArraySerializationTest, RankMismatch) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto encoded,
                                   EncodeBatch(MakeArray<int>({1, 2, 3})));
  SharedArray<int, 2> array;
  EXPECT_THAT(DecodeBatch(encoded, array),
              MatchesStatus(absl::StatusCode::kDataLoss,
                            "Expected rank of 2 but received: 1; .*"));
}

class RandomDataSerializationTest
    : public ::testing::TestWithParam<tensorstore::DataType> {};

INSTANTIATE_TEST_SUITE_P(DataTypes, RandomDataSerializationTest,
                         ::testing::ValuesIn(tensorstore::kDataTypes));

TEST_P(RandomDataSerializationTest, COrder) {
  auto dtype = GetParam();
  for (int iteration = 0; iteration < 100; ++iteration) {
    std::minstd_rand gen{tensorstore::internal::GetRandomSeedForTest(
        "TENSORSTORE_ARRAY_SERIALIZATION_TEST_SEED")};
    auto box = tensorstore::internal::MakeRandomBox(gen);
    auto array =
        tensorstore::internal::MakeRandomArray(gen, box, dtype, c_order);
    TestSerializationRoundTrip(array);
    tensorstore::serialization::TestSerializationRoundTripCorrupt(array);
  }
}

TEST_P(RandomDataSerializationTest, FOrder) {
  auto dtype = GetParam();
  for (int iteration = 0; iteration < 100; ++iteration) {
    std::minstd_rand gen{tensorstore::internal::GetRandomSeedForTest(
        "TENSORSTORE_ARRAY_SERIALIZATION_TEST_SEED")};
    auto box = tensorstore::internal::MakeRandomBox(gen);
    auto array =
        tensorstore::internal::MakeRandomArray(gen, box, dtype, fortran_order);
    TestSerializationRoundTrip(array);
  }
}

}  // namespace
