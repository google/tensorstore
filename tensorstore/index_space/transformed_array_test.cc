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

#include "tensorstore/index_space/transformed_array.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/array.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

namespace {

using tensorstore::dynamic_rank;
using tensorstore::Index;
using tensorstore::IndexInterval;
using tensorstore::kImplicit;
using tensorstore::kInfIndex;
using tensorstore::kInfSize;
using tensorstore::MakeArray;
using tensorstore::MakeOffsetArray;
using tensorstore::MatchesStatus;
using tensorstore::NormalizedTransformedArray;
using tensorstore::Result;
using tensorstore::Shared;
using tensorstore::StaticDataTypeCast;
using tensorstore::StaticRankCast;
using tensorstore::Status;
using tensorstore::TransformedArray;

static_assert(
    std::is_convertible<tensorstore::TransformedSharedArray<int, 1>,
                        tensorstore::TransformedArrayView<int, 1>>::value,
    "");

static_assert(
    !std::is_convertible<tensorstore::TransformedArrayView<int, 1>,
                         tensorstore::TransformedSharedArray<int, 1>>::value,
    "");
static_assert(
    !std::is_convertible<tensorstore::TransformedArrayView<int, 1>,
                         tensorstore::TransformedArray<int, 1>>::value,
    "");

static_assert(
    std::is_constructible<tensorstore::TransformedArray<int, 1>,
                          tensorstore::TransformedArrayView<int, 1>>::value,
    "");
static_assert(
    std::is_same<typename tensorstore::TransformedArrayView<int, 1>::
                     template RebindContainerKind<tensorstore::container>,
                 tensorstore::TransformedArray<int, 1>>::value,
    "");

static_assert(
    tensorstore::HasBoxDomain<tensorstore::TransformedArray<int, 1>>::value,
    "");

template <typename TA>
std::vector<const typename TA::Element*> GetPointers(const TA& a) {
  using Element = const typename TA::Element;
  std::vector<Element*> pointers;
  auto result = IterateOverTransformedArrays(
      [&](Element* x) { pointers.push_back(x); },
      /*constraints=*/tensorstore::skip_repeated_elements, a);
  EXPECT_TRUE(result);
  return pointers;
}
using TransformedArrayTestTypes =
    ::testing::Types<tensorstore::TransformedSharedArray<int>,
                     tensorstore::TransformedSharedArray<int, 1>,
                     tensorstore::TransformedSharedArrayView<int>,
                     tensorstore::TransformedSharedArrayView<int, 1>,
                     tensorstore::TransformedArrayView<int>,
                     tensorstore::TransformedArrayView<int, 1>>;

template <typename T>
class TransformedArrayConstructorTest : public ::testing::Test {};

TYPED_TEST_SUITE(TransformedArrayConstructorTest, TransformedArrayTestTypes);

template <typename TransformedArrayType, typename SourceArray>
void TestCopyAndMove(SourceArray&& source,
                     std::vector<const int*> expected_pointers) {
  // Test copy construction.
  {
    TransformedArrayType tb(source);
    EXPECT_EQ(source.domain(), tb.domain());
    EXPECT_EQ(source.domain(), GetBoxDomainOf(tb.domain()));
    EXPECT_EQ(expected_pointers, GetPointers(tb));
  }

  // Test move construction.
  {
    auto source_copy = source;
    TransformedArrayType tc(std::move(source_copy));
    EXPECT_EQ(source.domain(), tc.domain());
    EXPECT_EQ(expected_pointers, GetPointers(tc));
  }

  // Test copy assignment.
  {
    TransformedArrayType td;
    td = source;
    EXPECT_EQ(source.domain(), td.domain());
    EXPECT_EQ(expected_pointers, GetPointers(td));
  }

  // Test move assignment.
  {
    auto source_copy = source;
    TransformedArrayType td;
    td = std::move(source_copy);
    EXPECT_EQ(expected_pointers, GetPointers(td));
    EXPECT_EQ(source.domain(), td.domain());
  }
}

TYPED_TEST(TransformedArrayConstructorTest, DefaultConstruct) {
  TypeParam ta;
  EXPECT_FALSE(ta.has_transform());
  EXPECT_FALSE(ta.has_base_array());
  EXPECT_TRUE(ta.has_untransformed_array());
  EXPECT_FALSE(ta.transform());
  EXPECT_EQ(nullptr, ta.untransformed_array().data());
}

template <typename TransformedArrayType, typename Array>
void TestConstructFromArray(Array&& array,
                            std::vector<const int*> expected_pointers) {
  auto array_copy = array;
  TransformedArrayType ta(std::forward<Array>(array));
  EXPECT_EQ(array_copy.domain(), ta.domain());
  EXPECT_EQ(array_copy.domain(), GetBoxDomainOf(ta));
  EXPECT_FALSE(ta.has_transform());
  EXPECT_FALSE(ta.has_base_array());
  EXPECT_TRUE(ta.has_untransformed_array());
  EXPECT_FALSE(ta.transform());
  EXPECT_EQ(array_copy, ta.untransformed_array());
  EXPECT_EQ(array_copy.layout(), ta.untransformed_strided_layout());
  auto pointers = GetPointers(ta);
  EXPECT_EQ(expected_pointers, pointers);

  TestCopyAndMove<TransformedArrayType>(ta, expected_pointers);
  TestCopyAndMove<typename TransformedArrayType::template RebindContainerKind<
      tensorstore::container>>(ta, expected_pointers);
}

TYPED_TEST(TransformedArrayConstructorTest, ConstructFromZeroOriginArray) {
  auto a = MakeArray<int>({1, 2, 3});
  const std::vector<const int*> expected_pointers{&a(0), &a(1), &a(2)};
  TestConstructFromArray<TypeParam>(a, expected_pointers);
  TestConstructFromArray<TypeParam>(tensorstore::SharedArrayView<int, 1>(a),
                                    expected_pointers);
}

TYPED_TEST(TransformedArrayConstructorTest, ConstructFromOffsetOriginArray) {
  auto a = MakeOffsetArray<int>({3}, {1, 2, 3});
  const std::vector<const int*> expected_pointers{&a(3), &a(4), &a(5)};
  TestConstructFromArray<TypeParam>(a, expected_pointers);
  TestConstructFromArray<TypeParam>(
      tensorstore::SharedOffsetArrayView<int, 1>(a), expected_pointers);
}

template <typename TransformedArrayType, typename ElementPointer,
          typename Transform>
void TestConstructFromElementPointerAndTransform(
    ElementPointer&& element_pointer, Transform&& transform,
    std::vector<const int*> expected_pointers) {
  auto element_pointer_copy = element_pointer;
  auto transform_copy = transform;
  TransformedArrayType ta(std::forward<ElementPointer>(element_pointer),
                          std::forward<Transform>(transform));
  EXPECT_EQ(transform_copy.input_domain().box(), ta.domain());
  EXPECT_EQ(transform_copy.input_domain().box(), GetBoxDomainOf(ta));
  EXPECT_TRUE(ta.has_transform());
  EXPECT_FALSE(ta.has_base_array());
  EXPECT_FALSE(ta.has_untransformed_array());
  EXPECT_EQ(transform_copy, ta.transform());
  EXPECT_EQ(element_pointer_copy, ta.base_array().element_pointer());
  EXPECT_THAT(ta.base_array().shape(), ::testing::ElementsAre(kInfSize));
  EXPECT_THAT(ta.base_array().origin(), ::testing::ElementsAre(-kInfIndex));
  EXPECT_THAT(ta.base_array().byte_strides(), ::testing::ElementsAre(1));
  auto pointers = GetPointers(ta);
  EXPECT_EQ(expected_pointers, pointers);

  TestCopyAndMove<TransformedArrayType>(ta, expected_pointers);
  TestCopyAndMove<typename TransformedArrayType::template RebindContainerKind<
      tensorstore::container>>(ta, expected_pointers);
}

TYPED_TEST(TransformedArrayConstructorTest,
           ConstructFromElementPointerAndTransform) {
  auto a = MakeArray<int>({1, 2, 3});
  const std::vector<const int*> expected_pointers{&a(0), &a(1), &a(2)};
  auto t = tensorstore::IndexTransformBuilder<1, 1>()
               .input_origin({0})
               .input_shape({3})
               .output_single_input_dimension(0, 0, sizeof(int), 0)
               .Finalize()
               .value();
  TestConstructFromElementPointerAndTransform<TypeParam>(a.element_pointer(), t,
                                                         expected_pointers);
  auto element_pointer = a.element_pointer();
  auto t_copy = t;
  TestConstructFromElementPointerAndTransform<TypeParam>(
      std::move(element_pointer), std::move(t_copy), expected_pointers);

  tensorstore::IndexTransformView<1, 1> t_view = t;
  TestConstructFromElementPointerAndTransform<TypeParam>(
      a.element_pointer(), t_view, expected_pointers);
}

template <typename TransformedArrayType, typename Array, typename Transform>
void TestConstructFromArrayAndTransform(
    Array&& array, Transform&& transform,
    std::vector<const int*> expected_pointers) {
  auto array_copy = array;
  auto transform_copy = transform;
  TransformedArrayType ta{std::forward<Array>(array),
                          std::forward<Transform>(transform)};
  EXPECT_TRUE(ta.has_transform());
  EXPECT_TRUE(ta.has_base_array());
  EXPECT_FALSE(ta.has_untransformed_array());
  EXPECT_TRUE(ta.transform());
  EXPECT_EQ(array_copy, ta.base_array());
  EXPECT_EQ(array_copy.layout(), ta.base_strided_layout());
  EXPECT_EQ(array_copy.domain(), ta.domain());
  EXPECT_EQ(array_copy.domain(), GetBoxDomainOf(ta));
  EXPECT_EQ(transform_copy, ta.transform());
  EXPECT_EQ(expected_pointers, GetPointers(ta));
  TestCopyAndMove<TransformedArrayType>(ta, expected_pointers);
}

TYPED_TEST(TransformedArrayConstructorTest,
           ConstructFromZeroOriginArrayAndTransform) {
  auto a = MakeArray<int>({1, 2, 3});
  tensorstore::SharedArrayView<int, 1> a_view = a;
  auto t = tensorstore::IdentityTransform(a.domain());
  auto t_view = tensorstore::IndexTransformView<1, 1>(t);
  const std::vector<const int*> expected_pointers{&a(0), &a(1), &a(2)};
  TestConstructFromArrayAndTransform<TypeParam>(a, t, expected_pointers);
  TestConstructFromArrayAndTransform<TypeParam>(a, t_view, expected_pointers);
  TestConstructFromArrayAndTransform<TypeParam>(a_view, t_view,
                                                expected_pointers);
  auto t_copy = t;
  auto array_copy = a;
  TestConstructFromArrayAndTransform<TypeParam>(
      std::move(array_copy), std::move(t_copy), expected_pointers);
}

TYPED_TEST(TransformedArrayConstructorTest,
           ConstructFromOffsetOriginArrayAndTransform) {
  auto a = MakeOffsetArray<int>({3}, {1, 2, 3});
  auto t = tensorstore::IdentityTransform(a.domain());
  auto t_view = tensorstore::IndexTransformView<1, 1>(t);
  const std::vector<const int*> expected_pointers{&a(3), &a(4), &a(5)};
  TestConstructFromArrayAndTransform<TypeParam>(a, t, expected_pointers);
  TestConstructFromArrayAndTransform<TypeParam>(a, t_view, expected_pointers);
  TestConstructFromArrayAndTransform<TypeParam>(a, std::move(t),
                                                expected_pointers);
}

TEST(TransformedArrayTest, Array) {
  auto a = MakeOffsetArray<int>({3}, {1, 2, 3});
  auto ta = tensorstore::TransformedArray(a);
  static_assert(
      std::is_same<decltype(ta),
                   tensorstore::TransformedSharedArray<int, 1>>::value,
      "");
  EXPECT_EQ(a, ta.untransformed_array());
  auto a_copy = a;
  EXPECT_EQ(3, a.pointer().use_count());
  auto tb = tensorstore::TransformedArray(std::move(a_copy));
  static_assert(
      std::is_same<decltype(tb),
                   tensorstore::TransformedSharedArray<int, 1>>::value,
      "");
  EXPECT_EQ(3, a.pointer().use_count());
  EXPECT_FALSE(a_copy.valid());  // NOLINT
}

TEST(TransformedArrayTest, TransformedArray) {
  auto a = MakeOffsetArray<int>({3}, {1, 2, 3});
  auto ta = tensorstore::TransformedArray(a);
  auto tb = tensorstore::TransformedArray(ta);
  static_assert(
      std::is_same<decltype(tb),
                   tensorstore::TransformedSharedArray<int, 1>>::value,
      "");
  EXPECT_EQ(a, tb.untransformed_array());
  auto ta_copy = ta;
  EXPECT_EQ(4, a.pointer().use_count());
  auto tc = tensorstore::TransformedArray(std::move(ta_copy));
  static_assert(
      std::is_same<decltype(tc),
                   tensorstore::TransformedSharedArray<int, 1>>::value,
      "");
  EXPECT_EQ(a.element_pointer(), tc.element_pointer());
  EXPECT_EQ(a, tc.untransformed_array());
  EXPECT_EQ(4, a.pointer().use_count());
  EXPECT_FALSE(ta_copy.element_pointer());  // NOLINT
}

TEST(TransformedArrayTest, MapTransform) {
  auto array = MakeArray<int>({1, 2, 3});
  tensorstore::TransformedArray<int, 1> tarray(array);
  auto tarray2 =
      ChainResult(tarray, tensorstore::Dims(0).SizedInterval(1, 2)).value();
  EXPECT_EQ(MakeOffsetArray<int>({1}, {2, 3}), tarray2.Materialize().value());
}

TEST(TransformedArrayTest, ArrayAndTransform) {
  auto a = MakeOffsetArray<int>({3}, {1, 2, 3});
  auto t = tensorstore::IndexTransformBuilder<1, 1>()
               .input_origin({0})
               .input_shape({3})
               .input_labels({"a"})
               .output_single_input_dimension(0, 3, 1, 0)
               .Finalize()
               .value();
  auto ta = tensorstore::TransformedArray(a, t);
  static_assert(
      std::is_same<decltype(ta),
                   tensorstore::TransformedSharedArray<int, 1>>::value,
      "");
  EXPECT_EQ(a, ta.base_array());
  EXPECT_EQ(t, ta.transform());
  EXPECT_THAT(ta.labels(), ::testing::ElementsAre("a"));
}

TEST(TransformedArrayTest, DefaultLabels) {
  auto a = MakeOffsetArray<int>({3}, {1, 2, 3});
  tensorstore::TransformedSharedArray<int, 1> ta(a);
  EXPECT_EQ(a, ta.untransformed_array());
  EXPECT_THAT(ta.labels(), ::testing::ElementsAre(""));
}

TEST(TransformedArrayTest, DimExpression) {
  auto a = MakeOffsetArray<int>({10, 20}, {{1, 2, 3}, {4, 5, 6}});
  auto ta = ChainResult(a,
                        tensorstore::Dims(0, 1).IndexVectorArraySlice(
                            MakeArray<Index>({{10, 22}, {11, 21}, {11, 22}})),
                        tensorstore::Dims(0).Label("a"))
                .value();
  EXPECT_EQ(ta.transform(),
            (tensorstore::IndexTransformBuilder<1, 2>()
                 .input_origin({0})
                 .input_shape({3})
                 .input_labels({"a"})
                 .output_index_array(0, 0, sizeof(int) * 3,
                                     MakeArray<Index>({10, 11, 11}),
                                     IndexInterval::Sized(10, 2))
                 .output_index_array(1, 0, sizeof(int),
                                     MakeArray<Index>({22, 21, 22}),
                                     IndexInterval::Sized(20, 3))
                 .Finalize()
                 .value()));
  EXPECT_EQ(a.element_pointer(), ta.element_pointer());
  EXPECT_EQ(ta.domain().box(), tensorstore::BoxView<1>({3}));
}

TEST(TransformedArrayTest, MaterializeWithOffsetOrigin) {
  EXPECT_EQ(MakeOffsetArray<int>({2}, {3, 5, 6}),
            ChainResult(MakeOffsetArray<int>({10, 20}, {{1, 2, 3}, {4, 5, 6}}),
                        tensorstore::Dims(0, 1)
                            .IndexVectorArraySlice(MakeArray<Index>(
                                {{10, 22}, {11, 21}, {11, 22}}))
                            .TranslateTo(2))
                .value()
                .Materialize());
}

TEST(TransformedArrayTest, MaterializeWithZeroOrigin) {
  EXPECT_EQ(MakeArray<int>({3, 5, 6}),
            ChainResult(MakeOffsetArray<int>({10, 20}, {{1, 2, 3}, {4, 5, 6}}),
                        tensorstore::Dims(0, 1)
                            .IndexVectorArraySlice(MakeArray<Index>(
                                {{10, 22}, {11, 21}, {11, 22}}))
                            .TranslateTo(2))
                .value()
                .template Materialize<tensorstore::zero_origin>()
                .value());
}

TEST(TransformedArrayTest, MaterializeConstraints) {
  auto array = MakeOffsetArray<int>({2, 3}, {{3, 4, 5}, {6, 7, 8}});
  auto transformed_array =
      ChainResult(array,
                  tensorstore::Dims(1)
                      .ClosedInterval(kImplicit, kImplicit, 2)
                      .MoveToFront(),
                  tensorstore::Dims(2).AddNew().SizedInterval(5, 3))
          .value();
  auto expected_array = MakeOffsetArray<int>(
      {1, 2, 5}, {{{3, 3, 3}, {6, 6, 6}}, {{5, 5, 5}, {8, 8, 8}}});
  {
    auto new_array = transformed_array.Materialize().value();
    EXPECT_EQ(GetPointers(transformed_array), GetPointers(new_array));
  }

  const auto ValidateCopy =
      [&](const Result<tensorstore::SharedOffsetArray<const int, 3>>& new_array,
          const std::vector<Index>& expected_byte_strides) {
        ASSERT_EQ(Status(), GetStatus(new_array));
        EXPECT_NE(GetPointers(transformed_array), GetPointers(*new_array));
        EXPECT_EQ(expected_array, *new_array);
        EXPECT_THAT(new_array->byte_strides(),
                    ::testing::ElementsAreArray(expected_byte_strides));
      };

  const auto TestCopyAndMaterialize =
      [&](tensorstore::TransformArrayConstraints constraints,
          std::vector<Index> expected_byte_strides) {
        SCOPED_TRACE(tensorstore::StrCat("TestCopyAndMaterialize: constraints=",
                                         constraints.value()));
        // Test Materialize
        {
          SCOPED_TRACE("Materialize");
          auto new_array = transformed_array.Materialize(constraints);
          static_assert(
              std::is_same<
                  decltype(new_array),
                  Result<tensorstore::SharedOffsetArray<const int, 3>>>::value,
              "");
          ValidateCopy(new_array, expected_byte_strides);
        }
        // Test MakeCopy
        {
          SCOPED_TRACE("MakeCopy");
          auto new_array =
              MakeCopy(transformed_array, constraints.iteration_constraints());
          static_assert(
              std::is_same<
                  decltype(new_array),
                  Result<tensorstore::SharedOffsetArray<int, 3>>>::value,
              "");
          ValidateCopy(new_array, expected_byte_strides);
        }
      };
  TestCopyAndMaterialize(
      {tensorstore::skip_repeated_elements, tensorstore::must_allocate},
      {sizeof(int), sizeof(int) * 2, 0});
  TestCopyAndMaterialize(
      {tensorstore::c_order, tensorstore::skip_repeated_elements,
       tensorstore::must_allocate},
      {sizeof(int) * 2, sizeof(int), 0});
  TestCopyAndMaterialize(
      {tensorstore::fortran_order, tensorstore::skip_repeated_elements,
       tensorstore::must_allocate},
      {sizeof(int), sizeof(int) * 2, 0});
  TestCopyAndMaterialize(
      {tensorstore::fortran_order, tensorstore::include_repeated_elements,
       tensorstore::must_allocate},
      {sizeof(int), sizeof(int) * 2, sizeof(int) * 2 * 2});
  TestCopyAndMaterialize(
      {tensorstore::c_order, tensorstore::include_repeated_elements,
       tensorstore::must_allocate},
      {sizeof(int) * 2 * 3, sizeof(int) * 3, sizeof(int)});
}

TEST(TransformedArrayTest, MaterializeError) {
  EXPECT_THAT(
      ChainResult(MakeArray<int>({1, 2}), tensorstore::Dims(0).IndexArraySlice(
                                              MakeArray<Index>({3, 4})))
          .value()
          .Materialize(),
      MatchesStatus(absl::StatusCode::kOutOfRange));
}

TEST(TransformedArrayTest, MakeCopy) {
  EXPECT_THAT(MakeCopy(ChainResult(MakeArray<int>({1, 2}),
                                   tensorstore::Dims(0).IndexArraySlice(
                                       MakeArray<Index>({3, 4})))
                           .value()),
              MatchesStatus(absl::StatusCode::kOutOfRange));
}

/// Tests that move constructing a TransformedArray view from a TransformedArray
/// container does not destroy the transform until the end of the complete
/// expression.
TEST(TransformedArrayTest, MoveConstructViewFromContainer) {
  [](tensorstore::TransformedSharedArrayView<const void> x) {
    EXPECT_EQ(tensorstore::BoxView({2, 3}, {2, 2}), x.domain());
  }(tensorstore::TransformedArray(
      tensorstore::MakeOffsetArray<int>({2, 3}, {{1, 2}, {3, 4}}),
      tensorstore::IdentityTransform(tensorstore::BoxView({2, 3}, {2, 2}))));
}

/// Tests that ComposeLayoutAndTransformTest works when a transform is not
/// specified (i.e. a transform for which `valid() == false`).
TEST(ComposeLayoutAndTransformTest, NoTransform) {
  tensorstore::StridedLayout<tensorstore::dynamic_rank,
                             tensorstore::offset_origin>
      layout({1, 2}, {3, 4}, {5, 6});
  auto transform = tensorstore::ComposeLayoutAndTransform(
                       layout, tensorstore::IndexTransform<>())
                       .value();
  EXPECT_EQ(transform, tensorstore::IndexTransformBuilder<>(2, 2)
                           .input_origin({1, 2})
                           .input_shape({3, 4})
                           .output_single_input_dimension(0, 0, 5, 0)
                           .output_single_input_dimension(1, 0, 6, 1)
                           .Finalize()
                           .value());
}

TEST(ComposeLayoutAndTransformTest, ExistingTransform) {
  tensorstore::StridedLayout<tensorstore::dynamic_rank,
                             tensorstore::offset_origin>
      layout({1, 2}, {3, 4}, {5, 6});
  // When a transform is specified, `ComposeLayoutAndTransform` verifies that
  // the range of the transform is contained within the domain of `layout`.
  auto transform = tensorstore::ComposeLayoutAndTransform(
                       layout, tensorstore::IndexTransformBuilder<>(2, 2)
                                   .input_origin({11, 12})
                                   .input_shape({3, 2})
                                   .input_labels({"x", "y"})
                                   .output_single_input_dimension(0, -10, 1, 0)
                                   .output_single_input_dimension(1, -22, 2, 1)
                                   .Finalize()
                                   .value())
                       .value();
  EXPECT_EQ(transform, tensorstore::IndexTransformBuilder<>(2, 2)
                           .input_origin({11, 12})
                           .input_shape({3, 2})
                           .input_labels({"x", "y"})
                           .output_single_input_dimension(0, -10 * 5, 1 * 5, 0)
                           .output_single_input_dimension(1, -22 * 6, 2 * 6, 1)
                           .Finalize()
                           .value());
}

TEST(ComposeLayoutAndTransformTest, RankMismatch) {
  tensorstore::StridedLayout<tensorstore::dynamic_rank,
                             tensorstore::offset_origin>
      layout({1, 2}, {3, 4}, {5, 6});
  EXPECT_THAT(tensorstore::ComposeLayoutAndTransform(
                  layout, tensorstore::IdentityTransform(3)),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Transform output rank \\(3\\) does not equal "
                            "array rank \\(2\\)"));
}

TEST(MakeNormalizedTransformedArrayTest, UntransformedArray) {
  auto array = MakeOffsetArray<int>({2, 3}, {{3, 4, 5}, {6, 7, 8}});
  auto result = tensorstore::MakeNormalizedTransformedArray(
      tensorstore::TransformedArray(array));
  ASSERT_TRUE(result);
  EXPECT_EQ(array.element_pointer(), result->element_pointer());
  EXPECT_THAT(result->origin(), ::testing::ElementsAre(2, 3));
  EXPECT_THAT(result->shape(), ::testing::ElementsAre(2, 3));
  EXPECT_THAT(result->labels(), ::testing::ElementsAre("", ""));
  EXPECT_EQ(GetBoxDomainOf(array), GetBoxDomainOf(*result));
  EXPECT_EQ(tensorstore::IndexTransformBuilder<>(2, 2)
                .input_origin({2, 3})
                .input_shape({2, 3})
                .output_single_input_dimension(0, 0, sizeof(int) * 3, 0)
                .output_single_input_dimension(1, 0, sizeof(int), 1)
                .Finalize()
                .value(),
            result->transform());
}

TEST(MakeNormalizedTransformedArrayTest, BaseArrayAndTransform) {
  auto array = MakeOffsetArray<int>({2, 3}, {{3, 4, 5}, {6, 7, 8}});
  auto t = tensorstore::IndexTransformBuilder<1, 2>()
               .implicit_lower_bounds({1})
               .implicit_upper_bounds({1})
               .output_single_input_dimension(0, 1, 1, 0)
               .output_single_input_dimension(1, 2, 1, 0)
               .Finalize()
               .value();
  auto result = tensorstore::MakeNormalizedTransformedArray(
      tensorstore::TransformedArray(array, t));
  ASSERT_EQ(Status(), GetStatus(result));
  EXPECT_EQ(array.element_pointer(), result->element_pointer());
  EXPECT_EQ(
      tensorstore::IndexTransformBuilder<>(1, 2)
          .input_origin({1})
          .input_shape({2})
          .output_single_input_dimension(0, sizeof(int) * 3, sizeof(int) * 3, 0)
          .output_single_input_dimension(1, sizeof(int) * 2, sizeof(int), 0)
          .Finalize()
          .value(),
      result->transform());
}

TEST(MakeNormalizedTransformedArrayTest, TwoArgumentBaseArrayAndTransform) {
  auto array = MakeOffsetArray<int>({2, 3}, {{3, 4, 5}, {6, 7, 8}});
  auto t = tensorstore::IndexTransformBuilder<1, 2>()
               .implicit_lower_bounds({1})
               .implicit_upper_bounds({1})
               .output_single_input_dimension(0, 1, 1, 0)
               .output_single_input_dimension(1, 2, 1, 0)
               .Finalize()
               .value();
  auto result = tensorstore::MakeNormalizedTransformedArray(array, t);
  ASSERT_TRUE(result);
  EXPECT_EQ(array.element_pointer(), result->element_pointer());
  EXPECT_EQ(
      tensorstore::IndexTransformBuilder<>(1, 2)
          .input_origin({1})
          .input_shape({2})
          .output_single_input_dimension(0, sizeof(int) * 3, sizeof(int) * 3, 0)
          .output_single_input_dimension(1, sizeof(int) * 2, sizeof(int), 0)
          .Finalize()
          .value(),
      result->transform());
}

TEST(MakeNormalizedTransformedArrayTest, ElementPointerAndTransform) {
  auto array = MakeOffsetArray<int>({2, 3}, {{3, 4, 5}, {6, 7, 8}});
  tensorstore::TransformedArray<int, 2> transformed_array =
      tensorstore::MakeNormalizedTransformedArray(
          tensorstore::TransformedArray(array))
          .value();
  auto result = tensorstore::MakeNormalizedTransformedArray(transformed_array);
  ASSERT_TRUE(result);
  EXPECT_EQ(array.element_pointer(), result->element_pointer());
  EXPECT_EQ(transformed_array.transform(), result->transform());

  // Test `NormalizedTransformedArray` converting `operator=`.
  tensorstore::NormalizedTransformedArray<int, 2, tensorstore::view> t_view;
  t_view = *result;
  EXPECT_EQ(array.element_pointer(), t_view.element_pointer());
  EXPECT_EQ(transformed_array.transform(), t_view.transform());
}

TEST(GetUnboundedLayoutTest, Basic) {
  EXPECT_EQ((tensorstore::StridedLayout<tensorstore::dynamic_rank,
                                        tensorstore::offset_origin>(
                {-kInfIndex, -kInfIndex}, {kInfSize, kInfSize}, {1, 1})),
            tensorstore::internal_index_space::GetUnboundedLayout(2));
}

TEST(TransformedArrayTest, StaticDataTypeCast) {
  TransformedArray<std::int32_t, 1> ta_orig = MakeArray<std::int32_t>({3, 4});
  TransformedArray<void, 1> ta = ta_orig;
  auto ta_int = StaticDataTypeCast<std::int32_t>(ta);
  static_assert(
      std::is_same<decltype(ta_int), Result<TransformedArray<int, 1>>>::value,
      "");
  ASSERT_TRUE(ta_int);
  EXPECT_THAT(GetPointers(*ta_int),
              ::testing::ElementsAreArray(GetPointers(ta_orig)));
}

TEST(TransformedArrayTest, StaticRankCast) {
  TransformedArray<Shared<std::int32_t>, dynamic_rank> ta =
      MakeArray<std::int32_t>({3, 4});
  auto ta1 = StaticRankCast<1>(ta);
  static_assert(
      std::is_same<decltype(ta1),
                   Result<TransformedArray<Shared<std::int32_t>, 1>>>::value,
      "");
  ASSERT_TRUE(ta1);
  EXPECT_THAT(GetPointers(*ta1), ::testing::ElementsAreArray(GetPointers(ta)));
  EXPECT_THAT(
      StaticRankCast<2>(ta),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          "Cannot cast transformed array with data type of int32 and rank of 1 "
          "to transformed array with data type of int32 and rank of 2"));
}

// Tests cast from Array of dynamic rank to TransformedArray of static rank.
TEST(TransformedArrayTest, CastArrayToTransformedArray) {
  tensorstore::SharedArray<std::int32_t> a = MakeArray<std::int32_t>({1, 2});
  auto ta_result = tensorstore::StaticCast<
      tensorstore::TransformedArrayView<std::int32_t, 1>>(a);
  ASSERT_EQ(Status(), GetStatus(ta_result));
  EXPECT_THAT(GetPointers(*ta_result), ::testing::ElementsAre(&a(0), &a(1)));
}

TEST(NormalizedTransformedArrayTest, StaticDataTypeCast) {
  auto ta_orig = tensorstore::MakeNormalizedTransformedArray(
      MakeArray<std::int32_t>({3, 4}));
  NormalizedTransformedArray<Shared<void>, 1> ta = ta_orig;
  auto ta_int = StaticDataTypeCast<std::int32_t>(ta);
  static_assert(
      std::is_same<
          decltype(ta_int),
          Result<NormalizedTransformedArray<Shared<std::int32_t>, 1>>>::value,
      "");
  ASSERT_TRUE(ta_int);
  EXPECT_THAT(GetPointers(*ta_int),
              ::testing::ElementsAreArray(GetPointers(ta_orig)));
}

TEST(NormalizedTransformedArrayTest, StaticRankCast) {
  NormalizedTransformedArray<Shared<std::int32_t>, dynamic_rank> ta =
      tensorstore::MakeNormalizedTransformedArray(
          MakeArray<std::int32_t>({3, 4}));
  auto ta1 = StaticRankCast<1>(ta);
  static_assert(
      std::is_same<
          decltype(ta1),
          Result<NormalizedTransformedArray<Shared<std::int32_t>, 1>>>::value,
      "");
  ASSERT_TRUE(ta1);
  EXPECT_THAT(GetPointers(*ta1), ::testing::ElementsAreArray(GetPointers(ta)));
  EXPECT_THAT(
      StaticRankCast<2>(ta),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          "Cannot cast transformed array with data type of int32 and rank of 1 "
          "to transformed array with data type of int32 and rank of 2"));
}

TEST(TransformedArrayTest, ApplyIndexTransform) {
  auto array = MakeArray<int>({{1, 2, 3}, {4, 5, 6}});
  auto result = ChainResult(array, tensorstore::IdentityTransform<2>());
  ASSERT_EQ(Status(), GetStatus(result));
  EXPECT_EQ(array, MakeCopy(*result));
}

TEST(CopyTransformedArrayTest, Int32ToUint32) {
  auto a = MakeArray<tensorstore::int32_t>({{1, 2, 3}, {4, 5, 6}});
  auto b = tensorstore::AllocateArray<tensorstore::uint32_t>({3, 2});
  EXPECT_EQ(Status(),
            CopyTransformedArray(
                a, ChainResult(b, tensorstore::Dims(1, 0).Transpose())));
  EXPECT_EQ(b, MakeArray<tensorstore::uint32_t>({{1, 4}, {2, 5}, {3, 6}}));
}

TEST(CopyTransformedArrayTest, Int32ToInt32) {
  auto a = MakeArray<tensorstore::int32_t>({{1, 2, 3}, {4, 5, 6}});
  auto b = tensorstore::AllocateArray<tensorstore::int32_t>({3, 2});
  EXPECT_EQ(Status(),
            CopyTransformedArray(
                a, ChainResult(b, tensorstore::Dims(1, 0).Transpose())));
  EXPECT_EQ(b, MakeArray<tensorstore::int32_t>({{1, 4}, {2, 5}, {3, 6}}));
}

TEST(CopyTransformedArrayTest, Int32ToFloat32) {
  auto a = MakeArray<tensorstore::int32_t>({{1, 2, 3}, {4, 5, 6}});
  auto b = tensorstore::AllocateArray<tensorstore::float32_t>({3, 2});
  EXPECT_EQ(Status(),
            CopyTransformedArray(
                ChainResult(a, tensorstore::Dims(1, 0).Transpose()), b));
  EXPECT_EQ(b, MakeArray<tensorstore::float32_t>(
                   {{1.0, 4.0}, {2.0, 5.0}, {3.0, 6.0}}));
}

TEST(CopyTransformedArrayTest, InvalidDataType) {
  auto a = MakeArray<tensorstore::string_t>({"x", "y"});
  auto b = tensorstore::AllocateArray<tensorstore::float32_t>({2});
  EXPECT_THAT(CopyTransformedArray(a, b),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Cannot convert string -> float32"));
}

TEST(TransformedArrayTest, UnownedToShared) {
  auto a = MakeArray<int>({1, 2, 3});
  TransformedArray<int> ta = a;
  auto shared_ta = UnownedToShared(ta);
  static_assert(
      std::is_same_v<decltype(shared_ta), TransformedArray<Shared<int>>>);
}

TEST(TransformedArrayTest, UnownedToSharedAliasing) {
  auto a = MakeArray<int>({1, 2, 3});
  TransformedArray<int> ta = a;
  EXPECT_EQ(1, a.pointer().use_count());
  {
    auto shared_ta = UnownedToShared(a.pointer(), ta);
    EXPECT_EQ(2, a.pointer().use_count());
    static_assert(
        std::is_same_v<decltype(shared_ta), TransformedArray<Shared<int>>>);
    auto shared_ta_copy = UnownedToShared(shared_ta);
    static_assert(
        std::is_same_v<decltype(shared_ta), TransformedArray<Shared<int>>>);
    EXPECT_EQ(3, a.pointer().use_count());
  }
  EXPECT_EQ(1, a.pointer().use_count());
}

TEST(NormalizedTransformedArrayTest, UnownedToShared) {
  auto a = MakeArray<int>({1, 2, 3});
  NormalizedTransformedArray<int> ta = MakeNormalizedTransformedArray(a);
  auto shared_ta = UnownedToShared(ta);
  static_assert(std::is_same_v<decltype(shared_ta),
                               NormalizedTransformedArray<Shared<int>>>);
}

TEST(NormalizedTransformedArrayTest, UnownedToSharedAliasing) {
  auto a = MakeArray<int>({1, 2, 3});
  NormalizedTransformedArray<int> ta = MakeNormalizedTransformedArray(a);
  EXPECT_EQ(1, a.pointer().use_count());
  {
    auto shared_ta = UnownedToShared(a.pointer(), ta);
    EXPECT_EQ(2, a.pointer().use_count());
    static_assert(std::is_same_v<decltype(shared_ta),
                                 NormalizedTransformedArray<Shared<int>>>);
    auto shared_ta_copy = UnownedToShared(shared_ta);
    static_assert(std::is_same_v<decltype(shared_ta),
                                 NormalizedTransformedArray<Shared<int>>>);
    EXPECT_EQ(3, a.pointer().use_count());
  }
  EXPECT_EQ(1, a.pointer().use_count());
}

}  // namespace
