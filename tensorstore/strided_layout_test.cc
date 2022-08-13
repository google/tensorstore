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

#include "tensorstore/strided_layout.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/index.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/util/span.h"
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

using ::tensorstore::Box;
using ::tensorstore::BoxView;
using ::tensorstore::ContiguousLayoutOrder;
using ::tensorstore::DimensionIndex;
using ::tensorstore::dynamic_rank;
using ::tensorstore::GetSubLayoutView;
using ::tensorstore::Index;
using ::tensorstore::IndexInnerProduct;
using ::tensorstore::IsStridedLayout;
using ::tensorstore::MatchesStatus;
using ::tensorstore::offset_origin;
using ::tensorstore::span;
using ::tensorstore::StaticCast;
using ::tensorstore::StaticRankCast;
using ::tensorstore::StrCat;
using ::tensorstore::StridedLayout;
using ::tensorstore::StridedLayoutView;
using ::tensorstore::unchecked;
using ::tensorstore::zero_origin;
using ::tensorstore::internal::remove_cvref_t;

static_assert(!IsStridedLayout<int>);
static_assert(IsStridedLayout<StridedLayout<>>);
static_assert(IsStridedLayout<StridedLayout<2, offset_origin>>);
static_assert(IsStridedLayout<StridedLayoutView<>>);
static_assert(IsStridedLayout<StridedLayoutView<2, offset_origin>>);

// Tests the no-op overload of unchecked `StaticCast`.
namespace dynamic_layout_cast_tests {
template <typename T>
constexpr inline bool NoOpCheck =
    std::is_same_v<T, decltype(StaticCast<remove_cvref_t<T>, unchecked>(
                          std::declval<T>()))>;

static_assert(NoOpCheck<const StridedLayout<2>&>);
static_assert(NoOpCheck<StridedLayout<2>&>);
static_assert(NoOpCheck<StridedLayout<2>&&>);
static_assert(NoOpCheck<const StridedLayout<2, offset_origin>&>);
static_assert(NoOpCheck<StridedLayout<2, offset_origin>&>);
static_assert(NoOpCheck<StridedLayout<2, offset_origin>&&>);
}  // namespace dynamic_layout_cast_tests

// Tests the no-op overload of `StaticRankCast`.
namespace dynamic_rank_cast_tests {
template <typename T>
constexpr inline bool NoOpCheck =
    std::is_same_v<T, decltype(StaticRankCast<remove_cvref_t<T>::static_rank,
                                              unchecked>(std::declval<T>()))>;

static_assert(NoOpCheck<const StridedLayout<2>&>);
static_assert(NoOpCheck<StridedLayout<2>&>);
static_assert(NoOpCheck<StridedLayout<2>&&>);
static_assert(NoOpCheck<const StridedLayout<2, offset_origin>&>);
static_assert(NoOpCheck<StridedLayout<2, offset_origin>&>);
static_assert(NoOpCheck<StridedLayout<2, offset_origin>&&>);
}  // namespace dynamic_rank_cast_tests

static_assert(std::is_empty_v<StridedLayout<0>>);
static_assert(std::is_empty_v<StridedLayoutView<0>>);
static_assert(sizeof(Index) * 2 == sizeof(StridedLayout<1>));
static_assert(sizeof(Index) * 4 == sizeof(StridedLayout<2>));
static_assert(sizeof(Index*) * 2 == sizeof(StridedLayout<>));
static_assert(sizeof(Index*) * 3 == sizeof(StridedLayoutView<>));
static_assert(sizeof(Index*) * 2 == sizeof(StridedLayoutView<2>));
static_assert(sizeof(Index*) * 3 ==
              sizeof(StridedLayoutView<2, offset_origin>));

TEST(IndexInnerProductTest, Basic) {
  const Index a[] = {1, 2, 3};
  const Index b[] = {4, 5, 6};
  EXPECT_EQ(1 * 4 + 2 * 5 + 3 * 6, IndexInnerProduct(3, a, b));
}

TEST(IndexInnerProductTest, WrapOnOverflowMultiply) {
  const Index a[] = {Index(1) << 62, 2, 3};
  const Index b[] = {4, 5, 6};
  EXPECT_EQ(2 * 5 + 3 * 6, IndexInnerProduct(3, a, b));
}

TEST(IndexInnerProductTest, WrapOnOverflowAdd) {
  const Index a[] = {Index(1) << 62, Index(1) << 62};
  const Index b[] = {2, 2};
  EXPECT_EQ(0, IndexInnerProduct(2, a, b));
}

TEST(IndexInnerProductTest, Span) {
  const Index a[] = {1, 2, 3};
  const Index b[] = {4, 5, 6};
  EXPECT_EQ(1 * 4 + 2 * 5 + 3 * 6, IndexInnerProduct(span(a), span(b)));
}

namespace conversion_tests {
using ::tensorstore::internal::IsOnlyExplicitlyConvertible;

// Tests that StridedLayout is explicitly but not implicitly constructible from
// any StridedLayoutView type with an implicitly convertible rank and origin
// kind.
static_assert(IsOnlyExplicitlyConvertible<  //
              StridedLayoutView<dynamic_rank, offset_origin>,
              StridedLayout<dynamic_rank, offset_origin>>);
static_assert(IsOnlyExplicitlyConvertible<  //
              StridedLayoutView<2, offset_origin>,
              StridedLayout<dynamic_rank, offset_origin>>);
static_assert(
    IsOnlyExplicitlyConvertible<  //
        StridedLayoutView<2, offset_origin>, StridedLayout<2, offset_origin>>);
static_assert(IsOnlyExplicitlyConvertible<  //
              StridedLayoutView<dynamic_rank, zero_origin>,
              StridedLayout<dynamic_rank, offset_origin>>);
static_assert(
    IsOnlyExplicitlyConvertible<  //
        StridedLayoutView<2, zero_origin>, StridedLayout<2, offset_origin>>);
static_assert(IsOnlyExplicitlyConvertible<  //
              StridedLayoutView<dynamic_rank, zero_origin>,
              StridedLayout<dynamic_rank, zero_origin>>);
static_assert(IsOnlyExplicitlyConvertible<  //
              StridedLayoutView<2, zero_origin>,
              StridedLayout<dynamic_rank, zero_origin>>);

// Tests that StridedLayout is not explicitly constructible from a
// StridedLayoutView with incompatible origin kind or rank.
static_assert(!std::is_constructible_v<  //
              StridedLayout<dynamic_rank, zero_origin>,
              StridedLayoutView<dynamic_rank, offset_origin>>);
static_assert(!std::is_constructible_v<  //
              StridedLayout<2, zero_origin>,
              StridedLayoutView<dynamic_rank, zero_origin>>);
static_assert(!std::is_constructible_v<       //
              StridedLayout<2, zero_origin>,  //
              StridedLayoutView<3, zero_origin>>);

// Tests implicit conversion of zero-rank offset_origin layout to zero_origin
// layout.
static_assert(std::is_convertible_v<  //
              StridedLayoutView<0, offset_origin>,
              StridedLayout<dynamic_rank, zero_origin>>);
static_assert(
    std::is_convertible_v<  //
        StridedLayoutView<0, offset_origin>, StridedLayout<0, zero_origin>>);

// Tests implicit conversion of a StridedLayout to any StridedLayoutView with
// compatible rank and origin kind.
static_assert(std::is_convertible_v<          //
              StridedLayout<2, zero_origin>,  //
              StridedLayoutView<2, zero_origin>>);
static_assert(std::is_convertible_v<          //
              StridedLayout<2, zero_origin>,  //
              StridedLayoutView<dynamic_rank, zero_origin>>);
static_assert(std::is_convertible_v<          //
              StridedLayout<2, zero_origin>,  //
              StridedLayoutView<2, offset_origin>>);
static_assert(std::is_convertible_v<          //
              StridedLayout<2, zero_origin>,  //
              StridedLayoutView<dynamic_rank, offset_origin>>);
static_assert(std::is_convertible_v<            //
              StridedLayout<2, offset_origin>,  //
              StridedLayoutView<dynamic_rank, offset_origin>>);
static_assert(std::is_convertible_v<                       //
              StridedLayout<dynamic_rank, offset_origin>,  //
              StridedLayoutView<dynamic_rank, offset_origin>>);

// Tests implicit conversion of a StridedLayout to any StridedLayout with
// compatible rank and origin kind.
static_assert(std::is_convertible_v<          //
              StridedLayout<2, zero_origin>,  //
              StridedLayout<2, offset_origin>>);
static_assert(std::is_convertible_v<          //
              StridedLayout<2, zero_origin>,  //
              StridedLayout<dynamic_rank, zero_origin>>);
static_assert(std::is_convertible_v<          //
              StridedLayout<2, zero_origin>,  //
              StridedLayout<2, offset_origin>>);
static_assert(std::is_convertible_v<          //
              StridedLayout<2, zero_origin>,  //
              StridedLayout<dynamic_rank, offset_origin>>);
static_assert(std::is_convertible_v<            //
              StridedLayout<2, offset_origin>,  //
              StridedLayout<dynamic_rank, offset_origin>>);

// Tests implicit conversion of a zero-rank offset_origin layout to zero_origin
// layout.
static_assert(std::is_convertible_v<            //
              StridedLayout<0, offset_origin>,  //
              StridedLayout<dynamic_rank, zero_origin>>);
static_assert(std::is_convertible_v<            //
              StridedLayout<0, offset_origin>,  //
              StridedLayout<0, zero_origin>>);
}  // namespace conversion_tests

TEST(StridedLayoutTest, DynamicRank0) {
  StridedLayout<> layout;
  EXPECT_EQ(0, layout.rank());
  EXPECT_EQ(1, layout.num_elements());
  EXPECT_TRUE(layout.shape().empty());
  EXPECT_TRUE(layout.byte_strides().empty());
  EXPECT_EQ(0, layout());
}

TEST(StridedLayoutDeathTest, DynamicRank0) {
  StridedLayout<> layout;
  TENSORSTORE_EXPECT_DEATH_DEBUG_ONLY(
      layout[{1}], "Length of index vector is greater than rank of array");
  TENSORSTORE_EXPECT_DEATH_DEBUG_ONLY(
      layout({1}), "Length of index vector must match rank of array.");
  TENSORSTORE_EXPECT_DEATH_DEBUG_ONLY(
      layout(1), "Length of index vector must match rank of array.");
}

TEST(StridedLayoutTest, DynamicRankCopyAndMove) {
  StridedLayout<> layout;
  layout.set_rank(3);
  EXPECT_EQ(3, layout.rank());
  layout.shape()[0] = 7;
  layout.shape()[1] = 8;
  layout.shape()[2] = 9;
  layout.byte_strides()[0] = 4;
  layout.byte_strides()[1] = 5;
  layout.byte_strides()[2] = 6;
  EXPECT_EQ(7 * 8 * 9, layout.num_elements());

  EXPECT_EQ(8 + 5, (layout[{2, 1}]));
  EXPECT_EQ(8 + 5 + 6, (layout[{2, 1, 1}]));
  EXPECT_EQ(8 + 5 + 6, (layout({2, 1, 1})));
  EXPECT_EQ(8 + 5 + 6, layout(span({2, 1, 1})));
  EXPECT_EQ(8 + 5 + 6, layout(2, 1, 1));

  // Test copy construction.
  auto layout2 = layout;
  EXPECT_EQ(3, layout2.rank());
  EXPECT_THAT(layout2.shape(), ::testing::ElementsAreArray({7, 8, 9}));
  EXPECT_THAT(layout2.byte_strides(), ::testing::ElementsAreArray({4, 5, 6}));
  EXPECT_TRUE(layout == layout2);
  EXPECT_FALSE(layout != layout2);

  layout.shape()[0] = 1;
  EXPECT_FALSE(layout == layout2);
  EXPECT_TRUE(layout != layout2);

  const auto* shape = layout2.shape().data();
  const auto* byte_strides = layout2.byte_strides().data();

  // Test move construction.
  auto layout3 = std::move(layout2);
  EXPECT_EQ(0, layout2.rank());  // NOLINT
  EXPECT_EQ(3, layout3.rank());
  EXPECT_EQ(shape, layout3.shape().data());
  EXPECT_EQ(byte_strides, layout3.byte_strides().data());

  // Test move assignment.
  StridedLayout<> layout4 = layout;
  layout4 = std::move(layout3);
  EXPECT_EQ(3, layout4.rank());
  EXPECT_EQ(shape, layout4.shape().data());
  EXPECT_EQ(byte_strides, layout4.byte_strides().data());
}

// Test construction from shape and byte_strides spans.
TEST(StridedLayoutTest, ConstructDynamicFromShapeAndByteStrides) {
  const Index shape_arr[] = {1, 2};
  const Index byte_strides_arr[] = {3, 4};
  span<const Index> shape(shape_arr);
  span<const Index> byte_strides(byte_strides_arr);
  StridedLayout<> layout5(shape, byte_strides);
  EXPECT_EQ(2, layout5.rank());
  EXPECT_THAT(layout5.shape(), ::testing::ElementsAreArray({1, 2}));
  EXPECT_THAT(layout5.byte_strides(), ::testing::ElementsAreArray({3, 4}));
}

// Test construction from shape and byte_strides spans.
TEST(StridedLayoutDeathTest, ConstructDynamicFromShapeAndByteStrides) {
  const Index shape_arr[] = {1, 2};
  const Index byte_strides_arr[] = {3};
  span<const Index> shape(shape_arr);
  span<const Index> byte_strides(byte_strides_arr);
  TENSORSTORE_EXPECT_DEATH_DEBUG_ONLY((StridedLayout<>(shape, byte_strides)),
                                      "shape");
}

TEST(StridedLayoutTest, ConstructDynamicFromStridedLayoutView) {
  const Index shape_arr[] = {1, 2};
  const Index byte_strides_arr[] = {3, 4};
  StridedLayoutView<> layout_ref(shape_arr, byte_strides_arr);
  StridedLayout<> layout(layout_ref);
  EXPECT_EQ(2, layout.rank());
  EXPECT_THAT(layout.shape(), ::testing::ElementsAreArray({1, 2}));
  EXPECT_THAT(layout.byte_strides(), ::testing::ElementsAreArray({3, 4}));

  // The layout and layout_ref do not have the same shape pointers.
  EXPECT_NE(layout_ref.shape().data(), layout.shape().data());
  EXPECT_NE(layout_ref.byte_strides().data(), layout.byte_strides().data());
}

TEST(StridedLayoutTest, ConstructDynamicFromStatic) {
  StridedLayout<2> layout_s({1, 2}, {3, 4});
  StridedLayout<> layout_d(layout_s);
  EXPECT_EQ(2, layout_d.rank());
  EXPECT_THAT(layout_d.shape(), ::testing::ElementsAreArray({1, 2}));
  EXPECT_THAT(layout_d.byte_strides(), ::testing::ElementsAreArray({3, 4}));
}

TEST(StridedLayoutTest, AssignDynamicFromDynamic) {
  StridedLayout<> layout1({1, 2}, {3, 4});
  StridedLayout<> layout2;
  layout2 = layout1;
  EXPECT_EQ(2, layout2.rank());
  EXPECT_THAT(layout2.shape(), ::testing::ElementsAreArray({1, 2}));
  EXPECT_THAT(layout2.byte_strides(), ::testing::ElementsAreArray({3, 4}));
}

TEST(StridedLayoutTest, AssignDynamicFromDynamicRef) {
  StridedLayout<> layout1({1, 2}, {3, 4});
  StridedLayoutView<> layout_ref = layout1;
  StridedLayout<> layout2;
  layout2 = layout_ref;
  EXPECT_EQ(2, layout2.rank());
  EXPECT_THAT(layout2.shape(), ::testing::ElementsAreArray({1, 2}));
  EXPECT_THAT(layout2.byte_strides(), ::testing::ElementsAreArray({3, 4}));
}

TEST(StridedLayoutTest, AssignDynamicFromStatic) {
  StridedLayout<2> layout_s({1, 2}, {3, 4});
  StridedLayout<> layout_d;
  layout_d = layout_s;
  EXPECT_EQ(2, layout_d.rank());
  EXPECT_THAT(layout_d.shape(), ::testing::ElementsAreArray({1, 2}));
  EXPECT_THAT(layout_d.byte_strides(), ::testing::ElementsAreArray({3, 4}));
}

TEST(StridedLayoutDeathTest, DynamicRankIndexing) {
  StridedLayout<> layout(3);
  layout.shape()[0] = 7;
  layout.shape()[1] = 8;
  layout.shape()[2] = 9;
  layout.byte_strides()[0] = 4;
  layout.byte_strides()[1] = 5;
  layout.byte_strides()[2] = 6;
  EXPECT_EQ(4 * 6, (layout[{6}]));

  TENSORSTORE_EXPECT_DEATH_DEBUG_ONLY((layout[{7}]),
                                      "Array index out of bounds");
  TENSORSTORE_EXPECT_DEATH_DEBUG_ONLY((layout[{-1}]),
                                      "Array index out of bounds");
  TENSORSTORE_EXPECT_DEATH_DEBUG_ONLY((layout[{1, 2, 10}]),
                                      "Array index out of bounds");

  TENSORSTORE_EXPECT_DEATH_DEBUG_ONLY(
      (layout[{1, 2, 3, 4}]),
      "Length of index vector is greater than rank of array");
  TENSORSTORE_EXPECT_DEATH_DEBUG_ONLY(
      layout({1, 2}), "Length of index vector must match rank of array");
  TENSORSTORE_EXPECT_DEATH_DEBUG_ONLY(
      (StridedLayout<>(span<const Index>({1}), span<const Index>({1, 2}))),
      "shape");
}

TEST(StridedLayoutTest, StaticRank0) {
  StridedLayout<0> layout;
  EXPECT_EQ(1, layout.num_elements());
  EXPECT_EQ(0, layout.rank());
  EXPECT_TRUE(layout.shape().empty());
  EXPECT_TRUE(layout.byte_strides().empty());

  static_assert(!std::is_assignable_v<StridedLayout<0>, StridedLayout<>>);
  static_assert(!std::is_assignable_v<StridedLayout<0>, StridedLayoutView<>>);
  static_assert(!std::is_constructible_v<StridedLayout<0>, StridedLayout<1>>);
  static_assert(
      !std::is_constructible_v<StridedLayout<0>, StridedLayoutView<1>>);

  // Test constructing from zero-length shape and byte_strides spans.
  StridedLayout<0> layout3(span<const Index, 0>{}, span<const Index, 0>{});

  // Test copy construction from StridedLayout<0>.
  [[maybe_unused]] StridedLayout<0> layout2 = layout;

  // Test copy assignment from StridedLayout<0>.
  layout3 = layout;

  // Explicit construction from a StridedLayoutView<>.
  StridedLayout<0> layout5{StridedLayoutView<0>{}};

  EXPECT_EQ(0, layout());
  EXPECT_EQ(0, (layout[std::array<int, 0>{}]));
  EXPECT_EQ(0, (layout(std::array<int, 0>{})));
}

TEST(StridedLayoutTest, DefaultConstructStatic) {
  StridedLayout<2> layout;
  EXPECT_EQ(2, layout.rank());
}

TEST(StridedLayoutTest, ConstructStaticFromArrays) {
  StridedLayout<2> layout({1, 2}, {3, 4});
  EXPECT_THAT(layout.shape(), ::testing::ElementsAreArray({1, 2}));
  EXPECT_THAT(layout.byte_strides(), ::testing::ElementsAreArray({3, 4}));
}

TEST(StridedLayoutTest, ConstructDynamicFromArrays) {
  StridedLayout<> layout({1, 2}, {3, 4});
  EXPECT_EQ(2, layout.rank());
  EXPECT_THAT(layout.shape(), ::testing::ElementsAreArray({1, 2}));
  EXPECT_THAT(layout.byte_strides(), ::testing::ElementsAreArray({3, 4}));
}

TEST(StridedLayoutTest, ConstructStaticFromDynamic) {
  StridedLayout<> layout_d({1, 2}, {3, 4});

  auto layout_s = StaticRankCast<2>(layout_d).value();
  static_assert(std::is_same_v<decltype(layout_s), StridedLayout<2>>);
  EXPECT_THAT(layout_s.shape(), ::testing::ElementsAreArray({1, 2}));
  EXPECT_THAT(layout_s.byte_strides(), ::testing::ElementsAreArray({3, 4}));

  static_assert(!std::is_constructible_v<StridedLayout<2>, StridedLayout<3>>);

  static_assert(!std::is_assignable_v<StridedLayout<2>, StridedLayout<3>>);

  // Test copy construct from another static layout.
  StridedLayout<2> layout_s2(layout_s);
  EXPECT_THAT(layout_s2.shape(), ::testing::ElementsAreArray({1, 2}));
  EXPECT_THAT(layout_s2.byte_strides(), ::testing::ElementsAreArray({3, 4}));

  // Check that explicit conversion from dynamic to static is not available.
  static_assert(!std::is_constructible_v<StridedLayout<2>, StridedLayout<>>);
}

TEST(StridedLayoutTest, ConstructStaticFromDynamicStridedLayoutView) {
  StridedLayout<> layout_d({1, 2}, {3, 4});
  StridedLayoutView<> layout_ref = layout_d;
  auto layout_s = StaticCast<StridedLayout<2>>(layout_ref).value();
  static_assert(std::is_same_v<decltype(layout_s), StridedLayout<2>>);

  EXPECT_THAT(layout_s.shape(), ::testing::ElementsAreArray({1, 2}));
  EXPECT_THAT(layout_s.byte_strides(), ::testing::ElementsAreArray({3, 4}));

  auto layout_ref2 = StaticCast<StridedLayoutView<2>>(layout_d).value();
  StridedLayout<2> layout_s2(layout_ref2);
  EXPECT_THAT(layout_s2.shape(), ::testing::ElementsAreArray({1, 2}));
  EXPECT_THAT(layout_s2.byte_strides(), ::testing::ElementsAreArray({3, 4}));

  static_assert(
      !std::is_constructible_v<StridedLayout<2>, StridedLayoutView<3>>);
}

TEST(StridedLayoutTest, AssignStatic) {
  StridedLayout<> layout_d({1, 2}, {3, 4});

  static_assert(!std::is_assignable_v<StridedLayout<2>, StridedLayout<>>);
  static_assert(!std::is_assignable_v<StridedLayout<2>, StridedLayoutView<>>);

  // Assign from StridedLayout<2>
  {
    StridedLayout<2> layout_s;
    layout_s = StaticRankCast<2>(layout_d).value();
    EXPECT_THAT(layout_s.shape(), ::testing::ElementsAreArray({1, 2}));
    EXPECT_THAT(layout_s.byte_strides(), ::testing::ElementsAreArray({3, 4}));
  }

  // Assign from StridedLayoutView<2>
  {
    StridedLayout<2> layout_s;
    layout_s = StaticCast<StridedLayoutView<2>>(layout_d).value();
    EXPECT_THAT(layout_s.shape(), ::testing::ElementsAreArray({1, 2}));
    EXPECT_THAT(layout_s.byte_strides(), ::testing::ElementsAreArray({3, 4}));
  }
}

TEST(StridedLayoutTest, StaticIndexing) {
  StridedLayout<2> layout({3, 5}, {3, 4});
  EXPECT_EQ(6 + 4, layout(2, 1));
}

TEST(StridedLayoutViewTest, StaticConstructDefault) {
  StridedLayoutView<2> ref;
  EXPECT_EQ(2, ref.rank());
  EXPECT_EQ(0, ref.shape()[0]);
  EXPECT_EQ(0, ref.shape()[1]);
  EXPECT_EQ(0, ref.byte_strides()[0]);
  EXPECT_EQ(0, ref.byte_strides()[1]);
}

TEST(StridedLayoutViewTest, StaticConstructFromSpans) {
  const Index shape[] = {5, 3};
  const Index byte_strides[] = {3, 4};
  StridedLayoutView<2> ref(shape, byte_strides);
  EXPECT_EQ(&shape[0], ref.shape().data());
  EXPECT_EQ(&byte_strides[0], ref.byte_strides().data());
}

TEST(StridedLayoutViewTest, StaticConstructAndAssign) {
  const Index shape[] = {5, 3};
  const Index byte_strides[] = {3, 4};
  StridedLayoutView<2> ref(shape, byte_strides);

  // Copy construct from another StridedLayoutView<2>.
  {
    StridedLayoutView<2> ref2 = ref;
    EXPECT_EQ(&shape[0], ref2.shape().data());
    EXPECT_EQ(&byte_strides[0], ref2.byte_strides().data());
  }

  // Copy construct from a StridedLayoutView<>.
  {
    StridedLayoutView<2> ref2 =
        StaticRankCast<2>(StridedLayoutView<>{ref}).value();
    EXPECT_EQ(&shape[0], ref2.shape().data());
    EXPECT_EQ(&byte_strides[0], ref2.byte_strides().data());
  }

  static_assert(
      !std::is_convertible_v<StridedLayoutView<>, StridedLayoutView<2>>);
  static_assert(!std::is_convertible_v<StridedLayout<>, StridedLayoutView<2>>);
  static_assert(
      !std::is_constructible_v<StridedLayoutView<2>, StridedLayoutView<3>>);
  static_assert(
      !std::is_constructible_v<StridedLayoutView<2>, StridedLayoutView<>>);
  static_assert(
      !std::is_constructible_v<StridedLayoutView<2>, StridedLayout<>>);
  static_assert(
      !std::is_assignable_v<StridedLayoutView<2>, StridedLayoutView<>>);
  static_assert(!std::is_assignable_v<StridedLayoutView<2>, StridedLayout<>>);
  static_assert(!std::is_assignable_v<StridedLayoutView<2>, StridedLayout<3>>);
  static_assert(
      !std::is_assignable_v<StridedLayoutView<2>, StridedLayoutView<3>>);

  // Assign from another StridedLayoutView<2>.
  {
    StridedLayoutView<2> ref2;
    ref2 = ref;
    EXPECT_EQ(&shape[0], ref2.shape().data());
    EXPECT_EQ(&byte_strides[0], ref2.byte_strides().data());
  }

  // Assign from a StridedLayout<2>.
  {
    StridedLayout<2> layout(ref);
    StridedLayoutView<2> ref2;
    ref2 = layout;
    EXPECT_EQ(layout.shape().data(), ref2.shape().data());
    EXPECT_EQ(layout.byte_strides().data(), ref2.byte_strides().data());
  }

  StridedLayout<2> layout(std::integral_constant<DimensionIndex, 2>{});
}

TEST(StridedLayoutViewTest, CastError) {
  const Index shape[] = {5, 3};
  const Index byte_strides[] = {3, 4};
  StridedLayoutView<> ref(shape, byte_strides);

  EXPECT_THAT(StaticCast<StridedLayout<1>>(ref),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Cannot cast strided layout with rank of 2 to "
                            "strided layout with rank of 1"));
}

TEST(StridedLayoutViewTest, DynamicConsructAndAssign) {
  const Index shape[] = {5, 3};
  const Index byte_strides[] = {3, 4};
  StridedLayoutView<2> ref(shape, byte_strides);

  // Default construct.
  {
    StridedLayoutView<> r;
    EXPECT_EQ(0, r.rank());
    EXPECT_TRUE(r.shape().empty());
    EXPECT_TRUE(r.byte_strides().empty());
  }

  {
    // Construct from shape and byte_strides arrays.
    StridedLayoutView<> r(shape, byte_strides);
    EXPECT_EQ(2, r.rank());
    EXPECT_EQ(&shape[0], r.shape().data());
    EXPECT_EQ(&byte_strides[0], r.byte_strides().data());
    EXPECT_EQ(2, r.shape().size());
    EXPECT_EQ(2, r.byte_strides().size());

    // Copy construct from another StridedLayoutView<>.
    {
      StridedLayoutView<> r2 = r;
      EXPECT_EQ(2, r2.rank());
      EXPECT_EQ(&shape[0], r2.shape().data());
      EXPECT_EQ(&byte_strides[0], r2.byte_strides().data());
    }

    // Assign from another StridedLayoutView<>.
    {
      StridedLayoutView<> r2;
      r2 = r;
      EXPECT_EQ(2, r2.rank());
      EXPECT_EQ(&shape[0], r2.shape().data());
      EXPECT_EQ(&byte_strides[0], r2.byte_strides().data());
    }
  }

  // Copy construct from StridedLayoutView<2>.
  {
    StridedLayoutView<> r = ref;
    EXPECT_EQ(2, r.rank());
    EXPECT_EQ(&shape[0], r.shape().data());
    EXPECT_EQ(&byte_strides[0], r.byte_strides().data());
  }

  // Assign from StridedLayoutView<2>.
  {
    StridedLayoutView<> r;
    r = ref;
    EXPECT_EQ(2, r.rank());
    EXPECT_EQ(&shape[0], r.shape().data());
    EXPECT_EQ(&byte_strides[0], r.byte_strides().data());
  }

  {
    StridedLayout<> layout(ref);

    // Copy construct from StridedLayout<>.
    {
      StridedLayoutView<> r = layout;
      EXPECT_EQ(2, r.rank());
      EXPECT_EQ(layout.shape().data(), r.shape().data());
      EXPECT_EQ(layout.byte_strides().data(), r.byte_strides().data());
    }

    // Assign from StridedLayout<>.
    {
      StridedLayoutView<> r;
      r = layout;
      EXPECT_EQ(2, r.rank());
      EXPECT_EQ(layout.shape().data(), r.shape().data());
      EXPECT_EQ(layout.byte_strides().data(), r.byte_strides().data());
    }
  }

  {
    StridedLayout<2> layout(ref);

    // Copy construct from StridedLayout<2>.
    {
      StridedLayoutView<> r = layout;
      EXPECT_EQ(2, r.rank());
      EXPECT_EQ(layout.shape().data(), r.shape().data());
      EXPECT_EQ(layout.byte_strides().data(), r.byte_strides().data());
    }

    // Assign from StridedLayout<2>.
    {
      StridedLayoutView<> r;
      r = layout;
      EXPECT_EQ(2, r.rank());
      EXPECT_EQ(layout.shape().data(), r.shape().data());
      EXPECT_EQ(layout.byte_strides().data(), r.byte_strides().data());
    }
  }
}

TEST(StridedLayoutViewTest, Static0) {
  // Default construct.
  {
    StridedLayoutView<0> r;
    EXPECT_EQ(0, r.rank());
    EXPECT_EQ(nullptr, r.shape().data());
    EXPECT_EQ(nullptr, r.byte_strides().data());
  }

  // Construct from another StridedLayoutView<0>.
  {
    StridedLayoutView<0> r;
    [[maybe_unused]] StridedLayoutView<0> r2 = r;
  }

  // Construct from zero-length shape and byte_strides.
  { StridedLayoutView<0> r(span<const Index, 0>{}, span<const Index, 0>{}); }

  {
    // Construct from a StridedLayout<0>.
    StridedLayout<0> layout;
    StridedLayoutView<0> r = layout;

    // Assign from StridedLayout<0>.
    r = layout;
  }
}

TEST(StridedLayoutViewDeathTest, DynamicConstruct) {
  [[maybe_unused]] const Index shape[] = {5, 3};
  [[maybe_unused]] const Index byte_strides[] = {3};
  TENSORSTORE_EXPECT_DEATH_DEBUG_ONLY(
      (StridedLayoutView<>(shape, byte_strides)), "shape");

  StridedLayout<> x;
  x.set_rank(2);

  EXPECT_THAT(StaticCast<StridedLayoutView<0>>(StridedLayoutView<>(x)),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(StaticCast<StridedLayoutView<0>>(x),
              MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST(StridedLayoutViewTest, Compare) {
  StridedLayout<> r1(span<const Index>({1, 2}), span<const Index>({3, 4}));

  StridedLayout<> r2(span<const Index>({1, 2}), span<const Index>({3, 4}));

  StridedLayout<> r3(span<const Index>({1, 2, 3}),
                     span<const Index>({3, 4, 5}));

  EXPECT_TRUE(r1 == r2);
  EXPECT_FALSE(r1 != r2);

  r1.shape()[0] = 2;
  EXPECT_FALSE(r1 == r2);
  EXPECT_TRUE(r1 != r2);

  EXPECT_FALSE(r1 == StridedLayoutView<>{});
  EXPECT_TRUE(r1 != StridedLayoutView<>{});

  EXPECT_TRUE(StridedLayout<0>() == StridedLayoutView<0>());

  EXPECT_FALSE(r3 == r2);
  EXPECT_FALSE(r2 == r3);
  EXPECT_TRUE(r2 != r3);
}

TEST(StridedLayoutViewTest, SubLayout) {
  // Test static Rank, static SubRank.
  {
    StridedLayout<3> r({1, 2, 3}, {3, 4, 5});
    {
      auto s = GetSubLayoutView<0>(r);
      static_assert(std::is_same_v<decltype(s), StridedLayoutView<3>>);
      EXPECT_EQ(r.rank(), s.rank());
      EXPECT_EQ(r.shape().data(), s.shape().data());
      EXPECT_EQ(r.byte_strides().data(), s.byte_strides().data());
    }
    {
      auto s = GetSubLayoutView<1>(r);
      static_assert(std::is_same_v<decltype(s), StridedLayoutView<2>>);
      EXPECT_EQ(2, s.rank());
      EXPECT_EQ(r.shape().data() + 1, s.shape().data());
      EXPECT_EQ(r.byte_strides().data() + 1, s.byte_strides().data());
    }
    {
      auto s = GetSubLayoutView(r, tensorstore::StaticRank<1>{});
      static_assert(std::is_same_v<decltype(s), StridedLayoutView<2>>);
      EXPECT_EQ(2, s.rank());
      EXPECT_EQ(r.shape().data() + 1, s.shape().data());
      EXPECT_EQ(r.byte_strides().data() + 1, s.byte_strides().data());
    }
    {
      auto s = GetSubLayoutView<2>(r);
      static_assert(std::is_same_v<decltype(s), StridedLayoutView<1>>);

      EXPECT_EQ(1, s.rank());
      EXPECT_EQ(r.shape().data() + 2, s.shape().data());
      EXPECT_EQ(r.byte_strides().data() + 2, s.byte_strides().data());
    }
    {
      auto s = GetSubLayoutView<3>(r);
      static_assert(std::is_same_v<decltype(s), StridedLayoutView<0>>);

      EXPECT_EQ(0, s.rank());
    }
  }

  // Test static Rank, dynamic SubRank.
  {
    StridedLayout<3> r({1, 2, 3}, {3, 4, 5});
    {
      auto s = GetSubLayoutView(r, 0);
      static_assert(std::is_same_v<decltype(s), StridedLayoutView<>>);
      EXPECT_EQ(r.rank(), s.rank());
      EXPECT_EQ(r.shape().data(), s.shape().data());
      EXPECT_EQ(r.byte_strides().data(), s.byte_strides().data());
    }
    {
      auto s = GetSubLayoutView(r, 1);
      EXPECT_EQ(2, s.rank());
      EXPECT_EQ(r.shape().data() + 1, s.shape().data());
      EXPECT_EQ(r.byte_strides().data() + 1, s.byte_strides().data());
    }
    {
      auto s = GetSubLayoutView(r, 2);
      EXPECT_EQ(1, s.rank());
      EXPECT_EQ(r.shape().data() + 2, s.shape().data());
      EXPECT_EQ(r.byte_strides().data() + 2, s.byte_strides().data());
    }
    {
      auto s = GetSubLayoutView(r, 3);
      EXPECT_EQ(0, s.rank());
    }
  }

  // Test dynamic Rank, static SubRank.
  {
    StridedLayout<> r({1, 2, 3}, {3, 4, 5});
    {
      auto s = GetSubLayoutView<0>(r);
      static_assert(std::is_same_v<decltype(s), StridedLayoutView<>>);
      EXPECT_EQ(r.rank(), s.rank());
      EXPECT_EQ(r.shape().data(), s.shape().data());
      EXPECT_EQ(r.byte_strides().data(), s.byte_strides().data());
    }
    {
      auto s = GetSubLayoutView<1>(r);
      static_assert(std::is_same_v<decltype(s), StridedLayoutView<>>);
      EXPECT_EQ(2, s.rank());
      EXPECT_EQ(r.shape().data() + 1, s.shape().data());
      EXPECT_EQ(r.byte_strides().data() + 1, s.byte_strides().data());
    }
    {
      auto s = GetSubLayoutView<2>(r);
      static_assert(std::is_same_v<decltype(s), StridedLayoutView<>>);

      EXPECT_EQ(1, s.rank());
      EXPECT_EQ(r.shape().data() + 2, s.shape().data());
      EXPECT_EQ(r.byte_strides().data() + 2, s.byte_strides().data());
    }
    {
      auto s = GetSubLayoutView<3>(r);
      static_assert(std::is_same_v<decltype(s), StridedLayoutView<>>);
      EXPECT_EQ(0, s.rank());
    }
  }

  // Test dynamic Rank, dynamic SubRank.
  {
    StridedLayout<> r({1, 2, 3}, {3, 4, 5});
    {
      auto s = GetSubLayoutView(r, 0);
      static_assert(std::is_same_v<decltype(s), StridedLayoutView<>>);
      EXPECT_EQ(r.rank(), s.rank());
      EXPECT_EQ(r.shape().data(), s.shape().data());
      EXPECT_EQ(r.byte_strides().data(), s.byte_strides().data());
    }
    {
      auto s = GetSubLayoutView(r, 1);
      EXPECT_EQ(2, s.rank());
      EXPECT_EQ(r.shape().data() + 1, s.shape().data());
      EXPECT_EQ(r.byte_strides().data() + 1, s.byte_strides().data());
    }
    {
      auto s = GetSubLayoutView(r, 2);
      EXPECT_EQ(1, s.rank());
      EXPECT_EQ(r.shape().data() + 2, s.shape().data());
      EXPECT_EQ(r.byte_strides().data() + 2, s.byte_strides().data());
    }
    {
      auto s = GetSubLayoutView(r, 3);
      EXPECT_EQ(0, s.rank());
    }
  }
}

TEST(StridedLayoutViewDeathTest, SubLayout) {
  StridedLayout<> r({1, 2, 3}, {3, 4, 5});
  TENSORSTORE_EXPECT_DEATH_DEBUG_ONLY(GetSubLayoutView(r, -1), "sub_rank");
  TENSORSTORE_EXPECT_DEATH_DEBUG_ONLY(GetSubLayoutView(r, 4), "sub_rank");
  TENSORSTORE_EXPECT_DEATH_DEBUG_ONLY(GetSubLayoutView<4>(r), "sub_rank");
}

TEST(StridedLayoutTest, COrderStatic) {
  auto layout = StridedLayout(ContiguousLayoutOrder::c, 2,
                              span<const Index, 3>({3, 4, 5}));
  static_assert(std::is_same_v<decltype(layout), StridedLayout<3>>);
  EXPECT_EQ(StridedLayout<3>({3, 4, 5}, {4 * 5 * 2, 5 * 2, 2}), layout);

  StridedLayout<3, offset_origin> layout_offset_origin(ContiguousLayoutOrder::c,
                                                       2, {3, 4, 5});
  EXPECT_EQ((StridedLayout<3, offset_origin>({0, 0, 0}, {3, 4, 5},
                                             {4 * 5 * 2, 5 * 2, 2})),
            layout_offset_origin);
}

TEST(StridedLayoutTest, COrderDynamic) {
  auto layout =
      StridedLayout(ContiguousLayoutOrder::c, 2, span<const Index>({3, 4, 5}));
  static_assert(std::is_same_v<decltype(layout), StridedLayout<>>);
  EXPECT_EQ(StridedLayout<3>({3, 4, 5}, {4 * 5 * 2, 5 * 2, 2}), layout);
}

TEST(StridedLayoutTest, COrderVector) {
  auto layout =
      StridedLayout(ContiguousLayoutOrder::c, 2, std::vector<Index>{3, 4, 5});
  static_assert(std::is_same_v<decltype(layout), StridedLayout<>>);
  EXPECT_EQ(StridedLayout<3>({3, 4, 5}, {4 * 5 * 2, 5 * 2, 2}), layout);

  StridedLayout<3, offset_origin> layout_offset_origin(
      ContiguousLayoutOrder::c, 2, std::vector<Index>{3, 4, 5});
  EXPECT_EQ((StridedLayout<3, offset_origin>({0, 0, 0}, {3, 4, 5},
                                             {4 * 5 * 2, 5 * 2, 2})),
            layout_offset_origin);
}

TEST(StridedLayoutTest, FortranOrderStatic) {
  auto layout = StridedLayout(ContiguousLayoutOrder::fortran, 2, {3, 4, 5});
  static_assert(std::is_same_v<decltype(layout), StridedLayout<3>>);
  EXPECT_EQ(StridedLayout<3>({3, 4, 5}, {2, 3 * 2, 3 * 4 * 2}), layout);
}

TEST(StridedLayoutTest, FortranOrderDynamic) {
  auto layout = StridedLayout(ContiguousLayoutOrder::fortran, 2,
                              span<const Index>({3, 4, 5}));
  static_assert(std::is_same_v<decltype(layout), StridedLayout<>>);
  EXPECT_EQ(StridedLayout<3>({3, 4, 5}, {2, 3 * 2, 3 * 4 * 2}), layout);
}

TEST(StridedLayoutTest, PrintToOstream) {
  auto layout = StridedLayout(ContiguousLayoutOrder::fortran, 2, {3, 4, 5});
  EXPECT_EQ(
      "{domain={origin={0, 0, 0}, shape={3, 4, 5}}, byte_strides={2, 6, 24}}",
      StrCat(layout));
}

TEST(StridedLayoutViewTest, PrintToOstream) {
  auto layout = StridedLayout(ContiguousLayoutOrder::fortran, 2, {3, 4, 5});
  EXPECT_EQ(
      "{domain={origin={0, 0, 0}, shape={3, 4, 5}}, byte_strides={2, 6, 24}}",
      StrCat(StridedLayoutView<>(layout)));
}

TEST(StridedLayoutTest, Domain) {
  auto layout = StridedLayout(ContiguousLayoutOrder::fortran, 2, {3, 4, 5});
  auto box = layout.domain();
  static_assert(std::is_same_v<decltype(box), tensorstore::BoxView<3>>);
  EXPECT_THAT(box.shape(), ::testing::ElementsAreArray({3, 4, 5}));
  EXPECT_THAT(box.origin(), ::testing::ElementsAreArray({0, 0, 0}));
  EXPECT_EQ(box, GetBoxDomainOf(layout));
}

TEST(StridedLayoutTest, OffsetOrigin) {
  auto domain = Box({1, 2, 3}, {4, 5, 6});
  auto layout = StridedLayout(ContiguousLayoutOrder::c, 2, domain);
  EXPECT_EQ(domain, layout.domain());
  EXPECT_THAT(layout.byte_strides(), ::testing::ElementsAreArray({60, 12, 2}));
}

TEST(StridedLayoutTest, ConstructOffsetFromRankAndThreePointers) {
  const Index origin[] = {1, 2, 3};
  const Index shape[] = {4, 5, 6};
  const Index byte_strides[] = {7, 8, 9};
  StridedLayout<dynamic_rank, offset_origin> layout(3, origin, shape,
                                                    byte_strides);
  EXPECT_EQ(layout.domain(), BoxView(origin, shape));
  EXPECT_THAT(layout.byte_strides(), ::testing::ElementsAreArray(byte_strides));
}

TEST(StridedLayoutTest, ConstructOffsetFromThreeSpans) {
  const Index origin[] = {1, 2, 3};
  const Index shape[] = {4, 5, 6};
  const Index byte_strides[] = {7, 8, 9};
  StridedLayout<dynamic_rank, offset_origin> layout{span(origin), span(shape),
                                                    span(byte_strides)};
  EXPECT_EQ(layout.domain(), BoxView(origin, shape));
  EXPECT_THAT(layout.byte_strides(), ::testing::ElementsAreArray(byte_strides));
}

TEST(StridedLayoutTest, ConstructOffsetFromTwoSpans) {
  const Index shape[] = {4, 5, 6};
  const Index byte_strides[] = {7, 8, 9};
  StridedLayout<dynamic_rank, offset_origin> layout{span(shape),
                                                    span(byte_strides)};
  EXPECT_EQ(layout.domain(), BoxView(shape));
  EXPECT_THAT(layout.byte_strides(), ::testing::ElementsAreArray(byte_strides));
}

TEST(StridedLayoutTest, ConstructOffsetFromBoxAndByteStrides) {
  const Index origin[] = {1, 2, 3};
  const Index shape[] = {4, 5, 6};
  const Index byte_strides[] = {7, 8, 9};
  StridedLayout<dynamic_rank, offset_origin> layout{BoxView(origin, shape),
                                                    span(byte_strides)};
  EXPECT_EQ(layout.domain(), BoxView(origin, shape));
  EXPECT_THAT(layout.byte_strides(), ::testing::ElementsAreArray(byte_strides));
}

TEST(StridedLayoutTest, AssignOffsetOriginFromZeroOrigin) {
  const Index shape[] = {4, 5, 6};
  const Index byte_strides[] = {7, 8, 9};
  StridedLayout<dynamic_rank, offset_origin> layout;
  layout = StridedLayout<>(span(shape), span(byte_strides));
  EXPECT_EQ(layout.domain(), BoxView(shape));
  EXPECT_THAT(layout.byte_strides(), ::testing::ElementsAreArray(byte_strides));
}

TEST(StridedLayoutTest, AssignOffsetOriginFromStaticOffsetOrigin) {
  const Index origin[] = {1, 2, 3};
  const Index shape[] = {4, 5, 6};
  const Index byte_strides[] = {7, 8, 9};
  StridedLayout<dynamic_rank, offset_origin> layout;
  layout = StridedLayout<3, offset_origin>(origin, shape, byte_strides);
  EXPECT_EQ(BoxView(origin, shape), layout.domain());
  EXPECT_THAT(layout.byte_strides(), ::testing::ElementsAreArray(byte_strides));
}

TEST(StridedLayoutTest, OffsetOriginGetSubLayoutRef) {
  const Index origin[] = {1, 2, 3};
  const Index shape[] = {4, 5, 6};
  const Index byte_strides[] = {7, 8, 9};
  StridedLayout<dynamic_rank, offset_origin> layout;
  layout = StridedLayout<3, offset_origin>(origin, shape, byte_strides);
  auto layout2 = GetSubLayoutView(layout, 1);
  EXPECT_EQ((StridedLayout<dynamic_rank, offset_origin>(
                2, origin + 1, shape + 1, byte_strides + 1)),
            layout2);
}

TEST(StridedLayoutTest, Contains) {
  const Index origin[] = {1, 2, 3};
  const Index shape[] = {4, 5, 6};
  const Index byte_strides[] = {7, 8, 9};
  StridedLayout<dynamic_rank, offset_origin> layout(origin, shape,
                                                    byte_strides);
  EXPECT_TRUE(Contains(layout, span({1, 2, 3})));
  EXPECT_FALSE(Contains(layout, span({0, 2, 3})));
  EXPECT_FALSE(Contains(layout, span({1, 2, 3, 4})));
  EXPECT_FALSE(Contains(layout, span({1, 2})));
}

TEST(StridedLayoutTest, ContainsPartial) {
  const Index origin[] = {1, 2, 3};
  const Index shape[] = {4, 5, 6};
  const Index byte_strides[] = {7, 8, 9};
  StridedLayout<dynamic_rank, offset_origin> layout(origin, shape,
                                                    byte_strides);
  EXPECT_TRUE(ContainsPartial(layout, span({1, 2, 3})));
  EXPECT_FALSE(ContainsPartial(layout, span({0, 2, 3})));
  EXPECT_FALSE(ContainsPartial(layout, span({1, 2, 3, 4})));
  EXPECT_TRUE(ContainsPartial(layout, span({1, 2})));
  EXPECT_FALSE(ContainsPartial(layout, span({0, 2})));
}

TEST(StridedLayoutTest, RankCastNoOp) {
  const Index origin[] = {1, 2, 3};
  const Index shape[] = {4, 5, 6};
  const Index byte_strides[] = {7, 8, 9};
  StridedLayout<dynamic_rank, offset_origin> layout(origin, shape,
                                                    byte_strides);
  auto layout2 = StaticRankCast<dynamic_rank>(layout).value();
  EXPECT_EQ(layout, layout2);
}

TEST(StridedLayoutTest, RankCastOffsetOrigin) {
  const Index origin[] = {1, 2, 3};
  const Index shape[] = {4, 5, 6};
  const Index byte_strides[] = {7, 8, 9};
  StridedLayout<dynamic_rank, offset_origin> layout(origin, shape,
                                                    byte_strides);
  auto layout2 = StaticRankCast<3>(layout).value();
  static_assert(
      std::is_same_v<decltype(layout2), StridedLayout<3, offset_origin>>);
  EXPECT_EQ(layout, layout2);
}

TEST(StridedLayoutTest, ZeroOriginByteOffset) {
  StridedLayout<dynamic_rank> layout({1, 2}, {3, 4});
  EXPECT_EQ(0, layout.origin_byte_offset());
}

TEST(StridedLayoutTest, OffsetOriginByteOffset) {
  StridedLayout<dynamic_rank, offset_origin> layout({1, 2}, {3, 4}, {5, 6});
  EXPECT_EQ(1 * 5 + 2 * 6, layout.origin_byte_offset());
}

TEST(StridedLayoutTest, DynamicLayoutCastOffsetOrigin) {
  const Index origin[] = {1, 2, 3};
  const Index shape[] = {4, 5, 6};
  const Index byte_strides[] = {7, 8, 9};
  StridedLayout<dynamic_rank, offset_origin> layout(origin, shape,
                                                    byte_strides);
  auto layout2 = StaticCast<StridedLayout<3, offset_origin>>(layout).value();
  EXPECT_EQ(layout, layout2);
}

TEST(StridedLayoutTest, DynamicLayoutCastNoOp) {
  const Index origin[] = {1, 2, 3};
  const Index shape[] = {4, 5, 6};
  const Index byte_strides[] = {7, 8, 9};
  StridedLayout<dynamic_rank, offset_origin> layout(origin, shape,
                                                    byte_strides);
  auto layout2 =
      StaticCast<StridedLayout<dynamic_rank, offset_origin>>(layout).value();
  EXPECT_EQ(layout, layout2);
}

TEST(ArrayOriginKindTest, PrintToOstream) {
  EXPECT_EQ("zero", StrCat(zero_origin));
  EXPECT_EQ("offset", StrCat(offset_origin));
}

}  // namespace
