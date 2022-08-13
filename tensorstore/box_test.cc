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

#include "tensorstore/box.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/box.h"
#include "tensorstore/rank.h"
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/serialization/test_util.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

namespace {

using ::tensorstore::Box;
using ::tensorstore::BoxView;
using ::tensorstore::dynamic_rank;
using ::tensorstore::HasBoxDomain;
using ::tensorstore::Index;
using ::tensorstore::IndexInterval;
using ::tensorstore::IsStaticCastConstructible;
using ::tensorstore::kInfIndex;
using ::tensorstore::kInfSize;
using ::tensorstore::MatchesStatus;
using ::tensorstore::MutableBoxView;
using ::tensorstore::span;
using ::tensorstore::StaticRankCast;
using ::tensorstore::unchecked;
using ::tensorstore::serialization::TestSerializationRoundTrip;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

static_assert(std::is_convertible_v<BoxView<3>, BoxView<>>);
static_assert(!std::is_constructible_v<BoxView<3>, BoxView<>>);
static_assert(!std::is_assignable_v<BoxView<3>, BoxView<>>);
static_assert(!std::is_assignable_v<Box<3>, Box<>>);
static_assert(!std::is_constructible_v<Box<3>, Box<>>);
static_assert(!std::is_constructible_v<BoxView<3>, Box<>>);
static_assert(!std::is_constructible_v<MutableBoxView<3>, MutableBoxView<>>);
static_assert(!std::is_constructible_v<MutableBoxView<3>, Box<>>);
static_assert(std::is_constructible_v<MutableBoxView<3>, Box<3>&>);

static_assert(IsStaticCastConstructible<BoxView<3>, BoxView<>>);
static_assert(IsStaticCastConstructible<Box<3>, BoxView<>>);
static_assert(IsStaticCastConstructible<Box<3>, Box<>>);
static_assert(IsStaticCastConstructible<BoxView<>, BoxView<3>>);
static_assert(IsStaticCastConstructible<MutableBoxView<3>, Box<3>&>);
static_assert(!IsStaticCastConstructible<MutableBoxView<>, const Box<3>&>);
static_assert(!IsStaticCastConstructible<BoxView<2>, BoxView<3>>);
static_assert(!IsStaticCastConstructible<BoxView<2>, Box<3>>);
static_assert(!IsStaticCastConstructible<Box<3>, Box<2>>);

TEST(BoxTest, DefaultConstructDynamic) {
  Box<> box;
  EXPECT_EQ(0, box.rank());
}

TEST(BoxTest, DefaultConstructStatic) {
  Box<3> box;
  EXPECT_THAT(box.origin(), ElementsAre(-kInfIndex, -kInfIndex, -kInfIndex));
  EXPECT_THAT(box.shape(), ElementsAre(kInfSize, kInfSize, kInfSize));
}

TEST(BoxTest, RankPointersConstruct) {
  const Index origin[] = {1, 2, 3};
  const Index shape[] = {4, 5, 6};
  Box<> box(3, origin, shape);
  EXPECT_THAT(box.origin(), ElementsAre(1, 2, 3));
  EXPECT_THAT(box.shape(), ElementsAre(4, 5, 6));
}

TEST(BoxTest, SizeConstruct) {
  Box<> box(3);
  EXPECT_THAT(box.origin(), ElementsAre(-kInfIndex, -kInfIndex, -kInfIndex));
  EXPECT_THAT(box.shape(), ElementsAre(kInfSize, kInfSize, kInfSize));
}

TEST(BoxTest, ShapeArrayConstruct) {
  std::array<Index, 3> shape{{1, 2, 3}};
  Box<> box(shape);
  EXPECT_THAT(box.origin(), ElementsAre(0, 0, 0));
  EXPECT_THAT(box.shape(), ElementsAre(1, 2, 3));
}

TEST(BoxTest, DynamicRankSpanConstruct) {
  const Index origin[] = {1, 2, 3};
  const Index shape[] = {3, 4, 5};
  Box<> box{span(origin), span(shape)};
  EXPECT_EQ(3, box.rank());
  EXPECT_THAT(origin, ElementsAreArray(origin));
  EXPECT_THAT(shape, ElementsAreArray(shape));
}

TEST(BoxTest, ConstructFromArrays) {
  Box<> box({1, 2, 3}, {4, 5, 6});
  EXPECT_THAT(box.origin(), ElementsAre(1, 2, 3));
  EXPECT_THAT(box.shape(), ElementsAre(4, 5, 6));
}

TEST(BoxTest, ConstructFromBoxView) {
  const Index origin[] = {1, 2, 3};
  const Index shape[] = {4, 5, 6};
  BoxView<> view(origin, shape);
  Box<> box(view);
  EXPECT_EQ(3, box.rank());
  EXPECT_THAT(box.shape(), ElementsAreArray(shape));
  EXPECT_THAT(box.origin(), ElementsAreArray(origin));
}

TEST(BoxTest, DeduceFromShapeArray) {
  const Index shape[] = {3, 4, 5};
  auto box = Box(shape);
  static_assert(std::is_same_v<decltype(box), Box<3>>);
  EXPECT_THAT(box.origin(), ElementsAre(0, 0, 0));
  EXPECT_THAT(box.shape(), ElementsAre(3, 4, 5));
}

TEST(BoxTest, DeduceFromShapeSpanStatic) {
  const Index shape[] = {3, 4, 5};
  auto box = Box(span(shape));
  static_assert(std::is_same_v<decltype(box), Box<3>>);
  EXPECT_THAT(box.origin(), ElementsAre(0, 0, 0));
  EXPECT_THAT(box.shape(), ElementsAre(3, 4, 5));
}

TEST(BoxTest, DeduceFromShapeSpanDynamic) {
  const Index shape[] = {3, 4, 5};
  auto box = Box(span<const Index>(shape));
  static_assert(std::is_same_v<decltype(box), Box<>>);
  EXPECT_THAT(box.origin(), ElementsAre(0, 0, 0));
  EXPECT_THAT(box.shape(), ElementsAre(3, 4, 5));
}

TEST(BoxTest, DeduceFromOriginAndShapeArrays) {
  const Index origin[] = {1, 2, 3};
  const Index shape[] = {3, 4, 5};
  auto box = Box(origin, shape);
  static_assert(std::is_same_v<decltype(box), Box<3>>);
  EXPECT_THAT(box.origin(), ElementsAre(1, 2, 3));
  EXPECT_THAT(box.shape(), ElementsAre(3, 4, 5));
}

TEST(BoxTest, DeduceFromOriginAndShapeSpansStatic) {
  const Index origin[] = {1, 2, 3};
  const Index shape[] = {3, 4, 5};
  auto box = Box(span(origin), span(shape));
  static_assert(std::is_same_v<decltype(box), Box<3>>);
  EXPECT_THAT(box.origin(), ElementsAre(1, 2, 3));
  EXPECT_THAT(box.shape(), ElementsAre(3, 4, 5));
}

TEST(BoxTest, DeduceFromOriginAndShapeDynamic) {
  const Index origin[] = {1, 2, 3};
  const Index shape[] = {3, 4, 5};
  auto box = Box(span<const Index>(origin), span<const Index>(shape));
  static_assert(std::is_same_v<decltype(box), Box<>>);
  EXPECT_THAT(box.origin(), ElementsAre(1, 2, 3));
  EXPECT_THAT(box.shape(), ElementsAre(3, 4, 5));
}

TEST(BoxTest, DeduceFromBoxView) {
  const Index origin[] = {1, 2, 3};
  const Index shape[] = {3, 4, 5};
  BoxView<3> box(origin, shape);
  auto box2 = Box(box);
  static_assert(std::is_same_v<decltype(box2), Box<3>>);
  EXPECT_THAT(box2.shape(), ElementsAreArray(shape));
  EXPECT_THAT(box2.origin(), ElementsAreArray(origin));
}

TEST(BoxTest, DeduceFromBox) {
  const Index origin[] = {1, 2, 3};
  const Index shape[] = {3, 4, 5};
  Box<3> box(origin, shape);
  auto box2 = Box(box);
  static_assert(std::is_same_v<decltype(box2), Box<3>>);
  EXPECT_THAT(box2.shape(), ElementsAreArray(shape));
  EXPECT_THAT(box2.origin(), ElementsAreArray(origin));
}

TEST(BoxTest, AssignFromBoxView) {
  const Index origin[] = {1, 2, 3};
  const Index shape[] = {4, 5, 6};
  BoxView<> view(origin, shape);
  Box<> box;
  box = view;
  EXPECT_EQ(3, box.rank());
  EXPECT_THAT(box.shape(), ElementsAreArray(shape));
  EXPECT_THAT(box.origin(), ElementsAreArray(origin));
}

TEST(BoxTest, AssignFromBox) {
  const Index origin[] = {1, 2, 3};
  const Index shape[] = {4, 5, 6};
  Box<> other(origin, shape);
  Box<> box;
  box = other;
  EXPECT_EQ(3, box.rank());
  EXPECT_THAT(box.shape(), ElementsAreArray(shape));
  EXPECT_THAT(box.origin(), ElementsAreArray(origin));
}

TEST(BoxTest, AssignDynamicBoxFromStaticBox) {
  const Index origin[] = {1, 2, 3};
  const Index shape[] = {4, 5, 6};
  Box<3> other(origin, shape);
  Box<> box;
  box = other;
  EXPECT_EQ(3, box.rank());
  EXPECT_THAT(box.shape(), ElementsAreArray(shape));
  EXPECT_THAT(box.origin(), ElementsAreArray(origin));

  box.Fill();
  box = BoxView<3>(other);
  EXPECT_EQ(3, box.rank());
  EXPECT_THAT(box.shape(), ElementsAreArray(shape));
  EXPECT_THAT(box.origin(), ElementsAreArray(origin));
}

TEST(BoxTest, AssignStaticBoxFromDynamic) {
  const Index origin[] = {1, 2, 3};
  const Index shape[] = {4, 5, 6};
  Box<> other(origin, shape);
  Box<3> box;
  box = StaticRankCast<3, unchecked>(other);
  EXPECT_EQ(3, box.rank());
  EXPECT_THAT(box.shape(), ElementsAreArray(shape));
  EXPECT_THAT(box.origin(), ElementsAreArray(origin));
}

TEST(BoxTest, SetRank) {
  Box<> box;
  box.set_rank(3);
  EXPECT_EQ(3, box.rank());
  EXPECT_THAT(box.origin(), ElementsAre(-kInfIndex, -kInfIndex, -kInfIndex));
  EXPECT_THAT(box.shape(), ElementsAre(kInfSize, kInfSize, kInfSize));
}

TEST(BoxTest, Accessors) {
  Box<> box({1, 2, 3}, {4, 5, 6});
  EXPECT_EQ(4 * 5 * 6, box.num_elements());
  EXPECT_EQ(IndexInterval::UncheckedSized(1, 4), box[0]);
  EXPECT_EQ(IndexInterval::UncheckedSized(2, 5), box[1]);
  EXPECT_EQ(IndexInterval::UncheckedSized(3, 6), box[2]);
}

TEST(BoxTest, ConstAccessors) {
  const Box<> box({1, 2, 3}, {4, 5, 6});
  EXPECT_EQ(4 * 5 * 6, box.num_elements());
  EXPECT_EQ(IndexInterval::UncheckedSized(1, 4), box[0]);
  EXPECT_EQ(IndexInterval::UncheckedSized(2, 5), box[1]);
  EXPECT_EQ(IndexInterval::UncheckedSized(3, 6), box[2]);
}

TEST(BoxTest, SubscriptAssignment) {
  Box<> box(2);
  box[1] = IndexInterval::UncheckedSized(1, 5);
  EXPECT_THAT(box.origin(), ElementsAre(-kInfIndex, 1));
  EXPECT_THAT(box.shape(), ElementsAre(kInfSize, 5));
}

TEST(BoxTest, Fill) {
  Box<> box(2);
  box.Fill(IndexInterval::UncheckedSized(1, 5));
  EXPECT_THAT(box.origin(), ElementsAre(1, 1));
  EXPECT_THAT(box.shape(), ElementsAre(5, 5));
}

TEST(BoxTest, IsEmpty) {
  Box<> box(3);
  EXPECT_FALSE(box.is_empty());

  box.Fill(IndexInterval::UncheckedSized(0, 0));
  EXPECT_TRUE(box.is_empty());
}

TEST(BoxViewTest, StaticRankDefaultConstruct) {
  BoxView<3> box;
  EXPECT_THAT(box.origin(), ElementsAre(-kInfIndex, -kInfIndex, -kInfIndex));
  EXPECT_THAT(box.shape(), ElementsAre(kInfSize, kInfSize, kInfSize));
}

TEST(BoxViewTest, DynamicRankDefaultConstruct) {
  BoxView<> box;
  EXPECT_EQ(0, box.rank());
}

TEST(BoxViewTest, DynamicRankSizeConstruct) {
  BoxView<> box(3);
  EXPECT_EQ(3, box.rank());
  EXPECT_THAT(box.origin(), ElementsAre(-kInfIndex, -kInfIndex, -kInfIndex));
  EXPECT_THAT(box.shape(), ElementsAre(kInfSize, kInfSize, kInfSize));
}

TEST(BoxViewTest, DynamicRankSpanConstruct) {
  const Index origin[] = {1, 2, 3};
  const Index shape[] = {3, 4, 5};
  BoxView<> box{span(origin), span(shape)};
  EXPECT_EQ(3, box.rank());
  EXPECT_EQ(&origin[0], box.origin().data());
  EXPECT_EQ(&shape[0], box.shape().data());
}

TEST(BoxViewTest, DeduceFromShapeArray) {
  const Index shape[] = {3, 4, 5};
  auto box = BoxView(shape);
  static_assert(std::is_same_v<decltype(box), BoxView<3>>);
  EXPECT_THAT(box.origin(), ElementsAre(0, 0, 0));
  EXPECT_THAT(box.shape(), ElementsAre(3, 4, 5));
}

TEST(BoxViewTest, DeduceFromShapeSpanStatic) {
  const Index shape[] = {3, 4, 5};
  auto box = BoxView(span(shape));
  static_assert(std::is_same_v<decltype(box), BoxView<3>>);
  EXPECT_THAT(box.origin(), ElementsAre(0, 0, 0));
  EXPECT_THAT(box.shape(), ElementsAre(3, 4, 5));
}

TEST(BoxViewTest, DeduceFromShapeSpanDynamic) {
  const Index shape[] = {3, 4, 5};
  auto box = BoxView(span<const Index>(shape));
  static_assert(std::is_same_v<decltype(box), BoxView<>>);
  EXPECT_THAT(box.origin(), ElementsAre(0, 0, 0));
  EXPECT_THAT(box.shape(), ElementsAre(3, 4, 5));
}

TEST(BoxViewTest, DeduceFromOriginAndShapeArrays) {
  const Index origin[] = {1, 2, 3};
  const Index shape[] = {3, 4, 5};
  auto box = BoxView(origin, shape);
  static_assert(std::is_same_v<decltype(box), BoxView<3>>);
  EXPECT_THAT(box.origin(), ElementsAre(1, 2, 3));
  EXPECT_THAT(box.shape(), ElementsAre(3, 4, 5));
}

TEST(BoxViewTest, DeduceFromOriginAndShapeSpansStatic) {
  const Index origin[] = {1, 2, 3};
  const Index shape[] = {3, 4, 5};
  auto box = BoxView(span(origin), span(shape));
  static_assert(std::is_same_v<decltype(box), BoxView<3>>);
  EXPECT_THAT(box.origin(), ElementsAre(1, 2, 3));
  EXPECT_THAT(box.shape(), ElementsAre(3, 4, 5));
}

TEST(BoxViewTest, DeduceFromOriginAndShapeDynamic) {
  const Index origin[] = {1, 2, 3};
  const Index shape[] = {3, 4, 5};
  auto box = BoxView(span<const Index>(origin), span<const Index>(shape));
  static_assert(std::is_same_v<decltype(box), BoxView<>>);
  EXPECT_THAT(box.origin(), ElementsAre(1, 2, 3));
  EXPECT_THAT(box.shape(), ElementsAre(3, 4, 5));
}

TEST(BoxViewTest, DeduceFromBoxView) {
  const Index origin[] = {1, 2, 3};
  const Index shape[] = {3, 4, 5};
  BoxView<3> box(origin, shape);
  auto box2 = BoxView(box);
  static_assert(std::is_same_v<decltype(box2), BoxView<3>>);
  EXPECT_THAT(box2.shape(), ElementsAreArray(shape));
  EXPECT_THAT(box2.origin(), ElementsAreArray(origin));
}

TEST(BoxViewTest, DeduceFromBox) {
  const Index origin[] = {1, 2, 3};
  const Index shape[] = {3, 4, 5};
  const Box<3> box(origin, shape);
  auto box2 = BoxView(box);
  static_assert(std::is_same_v<decltype(box2), BoxView<3>>);
  EXPECT_THAT(box2.shape(), ElementsAreArray(shape));
  EXPECT_THAT(box2.origin(), ElementsAreArray(origin));
}

TEST(BoxViewTest, Subscript) {
  const Index origin[] = {1, 2, 3};
  const Index shape[] = {3, 4, 5};
  BoxView<> box(origin, shape);
  EXPECT_EQ(IndexInterval::UncheckedSized(1, 3), box[0]);
  EXPECT_EQ(IndexInterval::UncheckedSized(2, 4), box[1]);
  EXPECT_EQ(IndexInterval::UncheckedSized(3, 5), box[2]);
}

TEST(BoxViewTest, NumElements) {
  const Index origin[] = {1, 2, 3};
  const Index shape[] = {3, 4, 5};
  BoxView<> box(origin, shape);
  EXPECT_EQ(3 * 4 * 5, box.num_elements());
}

TEST(BoxViewTest, StaticToDynamicConversion) {
  const Index origin[] = {1, 2, 3};
  const Index shape[] = {3, 4, 5};
  BoxView<3> box(origin, shape);
  BoxView<> dynamic_box = box;
  EXPECT_EQ(3, dynamic_box.rank());
  EXPECT_THAT(dynamic_box.shape(), ElementsAreArray(shape));
  EXPECT_THAT(dynamic_box.origin(), ElementsAreArray(origin));
}

TEST(BoxViewTest, DefaultAssignment) {
  const Index origin[] = {1, 2, 3};
  const Index shape[] = {3, 4, 5};
  BoxView<3> box(origin, shape);
  BoxView<3> box2;
  box2 = box;
  EXPECT_EQ(3, box2.rank());
  EXPECT_THAT(box2.shape(), ElementsAreArray(shape));
  EXPECT_THAT(box2.origin(), ElementsAreArray(origin));
}

TEST(BoxViewTest, DefaultAssignmentStaticToDynamic) {
  const Index origin[] = {1, 2, 3};
  const Index shape[] = {3, 4, 5};
  BoxView<3> box(origin, shape);
  BoxView<> box2;
  box2 = box;
  EXPECT_EQ(3, box2.rank());
  EXPECT_THAT(box2.shape(), ElementsAreArray(shape));
  EXPECT_THAT(box2.origin(), ElementsAreArray(origin));
}

TEST(BoxViewTest, StaticRankCast) {
  const Index origin[] = {1, 2, 3};
  const Index shape[] = {3, 4, 5};
  BoxView<> box(origin, shape);
  auto box2 = StaticRankCast<3, unchecked>(box);
  EXPECT_THAT(
      StaticRankCast<2>(box),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Cannot cast box with rank of 3 to box with rank of 2"));
  static_assert(std::is_same_v<decltype(box2), BoxView<3>>);
  EXPECT_THAT(box2.shape(), ElementsAreArray(shape));
  EXPECT_THAT(box2.origin(), ElementsAreArray(origin));
}

TEST(BoxViewTest, ConstructFromDynamicBox) {
  Box<> box({1, 2}, {3, 4});
  BoxView<> box_view = box;
  EXPECT_EQ(2, box_view.rank());
  EXPECT_EQ(box.shape().data(), box_view.shape().data());
  EXPECT_EQ(box.origin().data(), box_view.origin().data());
}

TEST(BoxViewTest, ConstructFromStaticBox) {
  Box<2> box({1, 2}, {3, 4});
  BoxView<> box_view = box;
  EXPECT_EQ(2, box_view.rank());
  EXPECT_EQ(box.shape().data(), box_view.shape().data());
  EXPECT_EQ(box.origin().data(), box_view.origin().data());
}

TEST(MutableBoxViewTest, RankPointersConstruct) {
  Index origin[] = {1, 2, 3};
  Index shape[] = {4, 5, 6};
  MutableBoxView<> box(3, origin, shape);
  EXPECT_EQ(3, box.rank());
  EXPECT_EQ(box.origin().data(), origin);
  EXPECT_EQ(box.shape().data(), shape);
}

TEST(MutableBoxViewTest, DynamicRankSpanConstruct) {
  Index origin[] = {1, 2, 3};
  Index shape[] = {3, 4, 5};
  MutableBoxView<> box{span(origin), span(shape)};
  EXPECT_EQ(3, box.rank());
  EXPECT_EQ(box.origin().data(), origin);
  EXPECT_EQ(box.shape().data(), shape);
}

TEST(MutableBoxViewTest, DeduceFromOriginAndShapeArrays) {
  Index origin[] = {1, 2, 3};
  Index shape[] = {3, 4, 5};
  auto box = BoxView(origin, shape);
  static_assert(std::is_same_v<decltype(box), MutableBoxView<3>>);
  EXPECT_EQ(3, box.rank());
  EXPECT_EQ(box.origin().data(), origin);
  EXPECT_EQ(box.shape().data(), shape);
}

TEST(MutableBoxViewTest, DeduceFromOriginAndShapeSpansStatic) {
  Index origin[] = {1, 2, 3};
  Index shape[] = {3, 4, 5};
  auto box = BoxView(span(origin), span(shape));
  static_assert(std::is_same_v<decltype(box), MutableBoxView<3>>);
  EXPECT_EQ(3, box.rank());
  EXPECT_EQ(box.origin().data(), origin);
  EXPECT_EQ(box.shape().data(), shape);
}

TEST(MutableBoxViewTest, DeduceFromOriginAndShapeDynamic) {
  Index origin[] = {1, 2, 3};
  Index shape[] = {3, 4, 5};
  auto box = BoxView(span<Index>(origin), span<Index>(shape));
  static_assert(std::is_same_v<decltype(box), MutableBoxView<>>);
  EXPECT_EQ(3, box.rank());
  EXPECT_EQ(box.origin().data(), origin);
  EXPECT_EQ(box.shape().data(), shape);
}

TEST(MutableBoxViewTest, DeduceFromBox) {
  const Index origin[] = {1, 2, 3};
  const Index shape[] = {3, 4, 5};
  Box<3> box(origin, shape);
  auto box2 = BoxView(box);
  static_assert(std::is_same_v<decltype(box2), MutableBoxView<3>>);
  EXPECT_EQ(box2.shape().data(), box.shape().data());
  EXPECT_EQ(box2.origin().data(), box.origin().data());
}

TEST(MutableBoxViewTest, DeduceFromMutableBoxView) {
  Index origin[] = {1, 2, 3};
  Index shape[] = {3, 4, 5};
  MutableBoxView<3> box(origin, shape);
  auto box2 = BoxView(box);
  static_assert(std::is_same_v<decltype(box2), MutableBoxView<3>>);
  EXPECT_EQ(box2.shape().data(), box.shape().data());
  EXPECT_EQ(box2.origin().data(), box.origin().data());
}

TEST(MutableBoxViewTest, AssignFromBoxView) {
  Index origin1[] = {1, 2, 3};
  Index shape1[] = {4, 5, 6};
  const Index origin2[] = {10, 20, 30};
  const Index shape2[] = {40, 50, 60};
  MutableBoxView<> box(origin1, shape1);
  box.DeepAssign(BoxView(origin2, shape2));
  EXPECT_EQ(3, box.rank());
  EXPECT_THAT(origin1, ElementsAreArray(origin2));
  EXPECT_THAT(shape1, ElementsAreArray(shape2));
}

TEST(MutableBoxViewTest, AssignFromBox) {
  Index origin1[] = {1, 2, 3};
  Index shape1[] = {4, 5, 6};
  const Index origin2[] = {10, 20, 30};
  const Index shape2[] = {40, 50, 60};
  MutableBoxView<> box(origin1, shape1);
  box.DeepAssign(Box(origin2, shape2));
  EXPECT_EQ(3, box.rank());
  EXPECT_THAT(origin1, ElementsAreArray(origin2));
  EXPECT_THAT(shape1, ElementsAreArray(shape2));
}

TEST(MutableBoxViewTest, CopyAssign) {
  Index origin1[] = {1, 2, 3};
  Index shape1[] = {4, 5, 6};
  Index origin2[] = {10, 20, 30};
  Index shape2[] = {40, 50, 60};
  MutableBoxView<> box(origin1, shape1);
  box.DeepAssign(MutableBoxView<>(origin2, shape2));
  EXPECT_EQ(3, box.rank());
  EXPECT_THAT(origin1, ElementsAreArray(origin2));
  EXPECT_THAT(shape1, ElementsAreArray(shape2));
}

TEST(MutableBoxViewTest, SubscriptAssignment) {
  Index origin[] = {1, 2, 3};
  Index shape[] = {4, 5, 6};
  MutableBoxView<> box(origin, shape);
  box[1] = IndexInterval::UncheckedSized(1, 7);
  EXPECT_THAT(origin, ElementsAre(1, 1, 3));
  EXPECT_THAT(shape, ElementsAre(4, 7, 6));
}

TEST(MutableBoxViewTest, Fill) {
  Index origin[] = {1, 2, 3};
  Index shape[] = {4, 5, 6};
  MutableBoxView<> box(origin, shape);
  box.Fill(IndexInterval::UncheckedSized(1, 5));
  EXPECT_THAT(box.origin(), ElementsAre(1, 1, 1));
  EXPECT_THAT(box.shape(), ElementsAre(5, 5, 5));
  box.Fill();
  EXPECT_THAT(box.origin(), ElementsAre(-kInfIndex, -kInfIndex, -kInfIndex));
  EXPECT_THAT(box.shape(), ElementsAre(kInfSize, kInfSize, kInfSize));
}

TEST(MutableBoxViewTest, StaticRankCast) {
  Index origin[] = {1, 2, 3};
  Index shape[] = {3, 4, 5};
  MutableBoxView<> box(origin, shape);
  auto box2 = StaticRankCast<3, unchecked>(box);
  static_assert(std::is_same_v<decltype(box2), MutableBoxView<3>>);
  EXPECT_THAT(box2.shape(), ElementsAreArray(shape));
  EXPECT_THAT(box2.origin(), ElementsAreArray(origin));
}

TEST(BoxTest, Comparison) {
  const Index origin1[] = {1, 2, 3};
  const Index shape1[] = {4, 5, 6};

  const Index origin2[] = {1, 2, 3};
  const Index shape2[] = {4, 5, 6};

  const Index origin3[] = {1, 2, 4};
  const Index shape3[] = {4, 5, 7};

  const Index origin4[] = {1, 2};
  const Index shape4[] = {4, 5};

  BoxView<> view1(origin1, shape1);
  Box<> box1(view1);
  BoxView<> view2(origin2, shape2);
  Box<> box2(view2);
  BoxView<> view3(origin3, shape3);
  Box<> box3(view3);
  BoxView<> view4(origin4, shape4);
  Box<> box4(view4);

  EXPECT_EQ(box1, view1);
  EXPECT_EQ(box2, view2);
  EXPECT_EQ(box3, view3);
  EXPECT_EQ(box4, view4);

  EXPECT_EQ(view1, view2);
  EXPECT_EQ(view1, box2);
  EXPECT_EQ(box1, view2);
  EXPECT_EQ(box1, box2);
  EXPECT_NE(view1, view3);
  EXPECT_NE(view1, box3);
  EXPECT_NE(box1, view3);
  EXPECT_NE(box1, box3);
  EXPECT_NE(view1, view4);
  EXPECT_NE(view1, box4);
  EXPECT_NE(box1, view4);
  EXPECT_NE(box1, box4);
}

TEST(BoxTest, Print) {
  Index origin[] = {1, 2, 3};
  Index shape[] = {3, 4, 5};
  EXPECT_EQ("{origin={1, 2, 3}, shape={3, 4, 5}}",
            tensorstore::StrCat(BoxView<>(origin, shape)));
  EXPECT_EQ("{origin={1, 2, 3}, shape={3, 4, 5}}",
            tensorstore::StrCat(Box<>(origin, shape)));
  EXPECT_EQ("{origin={1, 2, 3}, shape={3, 4, 5}}",
            tensorstore::StrCat(MutableBoxView<>(origin, shape)));
}

TEST(BoxTest, Contains) {
  const Index origin1[] = {1, 2};
  const Index shape1[] = {4, 5};

  const Index origin2[] = {2, 2};
  const Index shape2[] = {3, 5};

  const Index origin3[] = {1, 2};
  const Index shape3[] = {4, 6};

  const Index origin4[] = {1};
  const Index shape4[] = {4};

  const Index indices1[] = {2, 3};
  const Index indices2[] = {0, 3};
  const Index indices3[] = {0};
  Index indices4[] = {2};

  auto span1 = span(indices1);
  auto span2 = span(indices2);
  auto span3 = span(indices3);
  auto span4 = span(indices4);

  BoxView<> view1(origin1, shape1);
  BoxView<> view2(origin2, shape2);
  BoxView<> view3(origin3, shape3);
  BoxView<> view4(origin4, shape4);

  Box<> box1(origin1, shape1);
  Box<> box2(origin2, shape2);
  Box<> box3(origin3, shape3);
  Box<> box4(origin4, shape4);

  EXPECT_TRUE(Contains(view1, indices1));
  EXPECT_TRUE(ContainsPartial(view1, indices1));
  EXPECT_TRUE(ContainsPartial(view1, indices4));
  EXPECT_FALSE(Contains(view1, indices2));
  EXPECT_FALSE(Contains(view1, indices3));
  EXPECT_FALSE(ContainsPartial(view1, indices2));
  EXPECT_FALSE(ContainsPartial(view1, indices3));

  EXPECT_TRUE(Contains(view1, span1));
  EXPECT_TRUE(ContainsPartial(view1, span1));
  EXPECT_FALSE(Contains(view1, span2));
  EXPECT_FALSE(ContainsPartial(view1, span2));
  EXPECT_FALSE(Contains(view1, span3));
  EXPECT_FALSE(ContainsPartial(view1, span3));
  EXPECT_TRUE(ContainsPartial(view1, span4));

  EXPECT_TRUE(Contains(box1, indices1));
  EXPECT_TRUE(ContainsPartial(box1, indices1));
  EXPECT_FALSE(Contains(box1, indices2));
  EXPECT_FALSE(Contains(box1, indices3));

  EXPECT_TRUE(Contains(box1, span1));
  EXPECT_FALSE(Contains(box1, span2));
  EXPECT_FALSE(Contains(box1, span3));

  EXPECT_TRUE(Contains(view1, view2));
  EXPECT_FALSE(Contains(view1, view3));
  EXPECT_FALSE(Contains(view1, view4));

  EXPECT_TRUE(Contains(view1, box2));
  EXPECT_FALSE(Contains(view1, box3));
  EXPECT_FALSE(Contains(view1, box4));

  EXPECT_TRUE(Contains(box1, view2));
  EXPECT_FALSE(Contains(box1, view3));
  EXPECT_FALSE(Contains(box1, view4));

  EXPECT_TRUE(Contains(box1, box2));
  EXPECT_FALSE(Contains(box1, box3));
  EXPECT_FALSE(Contains(box1, box4));
}

TEST(BoxTest, GetBoxDomainOf) {
  static_assert(!HasBoxDomain<int>);
  static_assert(HasBoxDomain<BoxView<>>);
  static_assert(HasBoxDomain<Box<>>);
  static_assert(HasBoxDomain<MutableBoxView<>>);
  Box<> box({1, 2}, {3, 4});
  BoxView<> view = box;
  EXPECT_EQ(box, GetBoxDomainOf(box));
  EXPECT_EQ(box, GetBoxDomainOf(view));
}

TEST(BoxTest, InlineSize) {
  Box<dynamic_rank(2)> box({1, 2}, {3, 4});
  BoxView<dynamic_rank> v = box;
  EXPECT_EQ(v, box);
  MutableBoxView<dynamic_rank> v2 = box;
  EXPECT_EQ(v2, box);
}

TEST(BoxTest, DeductionGuides) {
  auto box = Box({1, 2}, {3, 4});
  static_assert(std::is_same_v<decltype(box), Box<2>>);

  static_assert(std::is_same_v<decltype(BoxView({1, 2}, {3, 4})), BoxView<2>>);

  static_assert(decltype(box)::static_rank == 2);

  auto box_view = BoxView(box);
  static_assert(std::is_same_v<decltype(box_view), MutableBoxView<2>>);
}

TEST(BoxTest, IsFinite) {
  EXPECT_TRUE(IsFinite(Box<>()));
  EXPECT_TRUE(IsFinite(BoxView<>()));
  EXPECT_FALSE(IsFinite(Box<>(1)));
  EXPECT_FALSE(IsFinite(Box<1>()));
  EXPECT_FALSE(IsFinite(BoxView<>(1)));
  EXPECT_FALSE(IsFinite(BoxView<>(2)));
  EXPECT_FALSE(IsFinite(BoxView<2>()));
  EXPECT_TRUE(IsFinite(Box<3>({1, 2, 3}, {4, 5, 6})));
  EXPECT_TRUE(IsFinite(BoxView<3>({1, 2, 3}, {4, 5, 6})));
  EXPECT_TRUE(IsFinite(Box<>({1, 2, 3}, {4, 5, 6})));
  EXPECT_TRUE(IsFinite(BoxView<>({1, 2, 3}, {4, 5, 6})));
  EXPECT_TRUE(IsFinite(Box<1>({1}, {4})));
  EXPECT_FALSE(IsFinite(Box<3>({1, -kInfIndex, 3}, {4, 5, 6})));
  EXPECT_FALSE(IsFinite(Box<3>({1, kInfIndex - 5, 3}, {4, 6, 6})));
}

TEST(BoxSerializationTest, StaticRank) {
  TestSerializationRoundTrip(Box<0>());
  TestSerializationRoundTrip(Box<3>({1, 2, 3}, {4, 5, 6}));
}

TEST(BoxSerializationTest, DynamicRank) {
  TestSerializationRoundTrip(Box<>());
  TestSerializationRoundTrip(Box({1, 2, 3}, {4, 5, 6}));
}

}  // namespace
