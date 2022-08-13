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

#include "tensorstore/util/internal/iterate.h"

#include <array>
#include <tuple>
#include <type_traits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/inlined_vector.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/index.h"
#include "tensorstore/util/internal/iterate_impl.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/str_cat.h"

namespace {

using ::tensorstore::ArrayIterateResult;
using ::tensorstore::c_order;
using ::tensorstore::ContiguousLayoutOrder;
using ::tensorstore::DimensionIndex;
using ::tensorstore::fortran_order;
using ::tensorstore::include_repeated_elements;
using ::tensorstore::Index;
using ::tensorstore::IterationConstraints;
using ::tensorstore::LayoutOrderConstraint;
using ::tensorstore::skip_repeated_elements;
using ::tensorstore::span;
using ::tensorstore::internal::AdvanceIndices;
using ::tensorstore::internal::DefaultIterationResult;
using ::tensorstore::internal_iterate::
    ComputeStridedLayoutDimensionIterationOrder;
using ::tensorstore::internal_iterate::ExtractInnerShapeAndStrides;
using ::tensorstore::internal_iterate::InnerShapeAndStrides;
using ::tensorstore::internal_iterate::PermuteAndSimplifyStridedIterationLayout;
using ::tensorstore::internal_iterate::SimplifyStridedIterationLayout;
using ::tensorstore::internal_iterate::StridedIterationLayout;
using ::testing::ElementsAre;

TEST(LayoutOrderConstraint, Basic) {
  static_assert(!LayoutOrderConstraint{}, "");
  static_assert(!LayoutOrderConstraint(tensorstore::unspecified_order), "");
  static_assert(LayoutOrderConstraint(ContiguousLayoutOrder::c), "");
  static_assert(LayoutOrderConstraint(ContiguousLayoutOrder::fortran), "");
  static_assert(
      0 == LayoutOrderConstraint(tensorstore::unspecified_order).value(), "");
  static_assert(2 == LayoutOrderConstraint(ContiguousLayoutOrder::c).value(),
                "");
  static_assert(
      3 == LayoutOrderConstraint(ContiguousLayoutOrder::fortran).value(), "");
  static_assert(ContiguousLayoutOrder::c ==
                    LayoutOrderConstraint(ContiguousLayoutOrder::c).order(),
                "");
  static_assert(
      ContiguousLayoutOrder::fortran ==
          LayoutOrderConstraint(ContiguousLayoutOrder::fortran).order(),
      "");
}

TEST(IterationConstraintsTest, Basic) {
  static_assert(!IterationConstraints().order_constraint(), "");
  static_assert(
      !IterationConstraints(tensorstore::unspecified_order).order_constraint(),
      "");
  static_assert(
      IterationConstraints(ContiguousLayoutOrder::c).order_constraint(), "");
  static_assert(
      IterationConstraints(ContiguousLayoutOrder::fortran).order_constraint(),
      "");
  static_assert(
      ContiguousLayoutOrder::c == IterationConstraints(ContiguousLayoutOrder::c)
                                      .order_constraint()
                                      .order(),
      "");
  static_assert(ContiguousLayoutOrder::fortran ==
                    IterationConstraints(ContiguousLayoutOrder::fortran)
                        .order_constraint()
                        .order(),
                "");
  static_assert(include_repeated_elements ==
                    IterationConstraints().repeated_elements_constraint(),
                "");
  static_assert(include_repeated_elements ==
                    IterationConstraints(include_repeated_elements)
                        .repeated_elements_constraint(),
                "");
  static_assert(
      skip_repeated_elements == IterationConstraints(skip_repeated_elements)
                                    .repeated_elements_constraint(),
      "");
  static_assert(
      skip_repeated_elements ==
          IterationConstraints(ContiguousLayoutOrder::c, skip_repeated_elements)
              .repeated_elements_constraint(),
      "");
  static_assert(include_repeated_elements ==
                    IterationConstraints(ContiguousLayoutOrder::c,
                                         include_repeated_elements)
                        .repeated_elements_constraint(),
                "");
  static_assert(ContiguousLayoutOrder::c ==
                    IterationConstraints(ContiguousLayoutOrder::c,
                                         include_repeated_elements)
                        .order_constraint()
                        .order(),
                "");
  static_assert(ContiguousLayoutOrder::fortran ==
                    IterationConstraints(ContiguousLayoutOrder::fortran,
                                         include_repeated_elements)
                        .order_constraint()
                        .order(),
                "");
  static_assert(ContiguousLayoutOrder::fortran ==
                    IterationConstraints(ContiguousLayoutOrder::fortran,
                                         include_repeated_elements)
                        .order_constraint()
                        .order(),
                "");
  static_assert(3 == IterationConstraints(ContiguousLayoutOrder::fortran,
                                          include_repeated_elements)
                         .value(),
                "");
}

TEST(PermuteAndSimplifyStridedIterationLayoutTest, Fortran1D) {
  const Index shape[] = {3, 4, 5};
  const DimensionIndex dimension_order[] = {2, 1, 0};
  const Index strides[] = {1, 3, 12};
  auto layout = PermuteAndSimplifyStridedIterationLayout<1>(
      shape, dimension_order, {{strides}});
  StridedIterationLayout<1> expected_layout{{60, {{1}}}};
  EXPECT_EQ(expected_layout, layout);
}

TEST(PermuteAndSimplifyStridedIterationLayoutTest, C1D) {
  const Index shape[] = {3, 4, 5};
  const DimensionIndex dimension_order[] = {0, 1, 2};
  const Index strides[] = {20, 5, 1};
  auto layout = PermuteAndSimplifyStridedIterationLayout<1>(
      shape, dimension_order, {{strides}});
  StridedIterationLayout<1> expected_layout{{60, {{1}}}};
  EXPECT_EQ(expected_layout, layout);
}

TEST(PermuteAndSimplifyStridedIterationLayoutTest, C2D) {
  const Index shape[] = {3, 4, 5};
  const DimensionIndex dimension_order[] = {0, 1, 2};
  const Index strides[] = {40, 5, 1};
  auto layout = PermuteAndSimplifyStridedIterationLayout<1>(
      shape, dimension_order, {{strides}});
  StridedIterationLayout<1> expected_layout{{3, {{40}}}, {20, {{1}}}};
  EXPECT_EQ(expected_layout, layout);
}

TEST(PermuteAndSimplifyStridedIterationLayoutTest, C2D2Layouts) {
  const Index shape[] = {3, 4, 5};
  const ptrdiff_t dimension_order[] = {0, 1, 2};
  const Index strides0[] = {40, 5, 1};
  const Index strides1[] = {40, 10, 2};
  auto layout = PermuteAndSimplifyStridedIterationLayout<2>(
      shape, dimension_order, {{strides0, strides1}});
  StridedIterationLayout<2> expected_layout{{3, {{40, 40}}}, {20, {{1, 2}}}};
  EXPECT_EQ(expected_layout, layout);
}

TEST(PermuteAndSimplifyStridedIterationLayoutTest, C3D2Layouts) {
  const Index shape[] = {3, 4, 5};
  const ptrdiff_t dimension_order[] = {0, 1, 2};
  const Index strides0[] = {40, 5, 1};
  const Index strides1[] = {40, 10, 1};
  auto layout = PermuteAndSimplifyStridedIterationLayout<2>(
      shape, dimension_order, {{strides0, strides1}});
  StridedIterationLayout<2> expected_layout{
      {3, {{40, 40}}}, {4, {{5, 10}}}, {5, {{1, 1}}}};
  EXPECT_EQ(expected_layout, layout);
}

TEST(ComputeStridedLayoutDimensionIterationOrderTest, Unconstrained1D1Layout) {
  const Index shape[] = {3, 4, 5};
  const Index strides0[] = {20, 1, 4};
  EXPECT_THAT(ComputeStridedLayoutDimensionIterationOrder(
                  include_repeated_elements, shape, span({strides0})),
              ElementsAre(0, 2, 1));
}

TEST(ComputeStridedLayoutDimensionIterationOrderTest,
     Unconstrained1D1LayoutSkipRepeated) {
  const Index shape[] = {3, 5, 4, 5};
  const Index strides0[] = {20, 0, 1, 4};

  EXPECT_THAT(ComputeStridedLayoutDimensionIterationOrder(
                  include_repeated_elements, shape, span({strides0})),
              ElementsAre(0, 3, 2, 1));

  EXPECT_THAT(ComputeStridedLayoutDimensionIterationOrder(
                  skip_repeated_elements, shape, span({strides0})),
              ElementsAre(0, 3, 2));
}

TEST(ComputeStridedLayoutDimensionIterationOrderTest,
     Unconstrained1D1LayoutSingletonDims) {
  const Index shape[] = {3, 1, 4, 5};
  const Index strides0[] = {20, 5, 1, 4};
  EXPECT_THAT(ComputeStridedLayoutDimensionIterationOrder(
                  include_repeated_elements, shape, span({strides0})),
              ElementsAre(0, 3, 2));
}

TEST(ComputeStridedLayoutDimensionIterationOrderTest, Unconstrained1D2Layouts) {
  const Index shape[] = {3, 4, 5};
  const Index strides0[] = {20, 1, 4};
  const Index strides1[] = {40, 2, 8};
  EXPECT_THAT(ComputeStridedLayoutDimensionIterationOrder(
                  include_repeated_elements, shape, span({strides0, strides1})),
              ElementsAre(0, 2, 1));
}

TEST(ComputeStridedLayoutDimensionIterationOrderTest,
     Unconstrained1D2LayoutsSkipRepeated) {
  const Index shape[] = {3, 5, 4, 5, 2};
  const Index strides0[] = {20, 0, 1, 4, 71};
  const Index strides1[] = {40, 0, 2, 8, 0};
  EXPECT_THAT(ComputeStridedLayoutDimensionIterationOrder(
                  include_repeated_elements, shape, span({strides0, strides1})),
              ElementsAre(4, 0, 3, 2, 1));
  EXPECT_THAT(ComputeStridedLayoutDimensionIterationOrder(
                  skip_repeated_elements, shape, span({strides0, strides1})),
              ElementsAre(4, 0, 3, 2));
  EXPECT_THAT(ComputeStridedLayoutDimensionIterationOrder(
                  skip_repeated_elements, shape, span({strides1, strides0})),
              ElementsAre(0, 3, 2, 4));
}

TEST(ComputeStridedLayoutDimensionIterationOrderTest, Fortran1D) {
  const Index shape[] = {3, 4, 5};
  const Index strides[] = {1, 3, 12};
  EXPECT_THAT(ComputeStridedLayoutDimensionIterationOrder(
                  {ContiguousLayoutOrder::fortran, include_repeated_elements},
                  shape, span({strides})),
              ElementsAre(2, 1, 0));
}

TEST(ComputeStridedLayoutDimensionIterationOrderTest, Fortran1DSkipRepeated) {
  const Index shape[] = {3, 4, 2, 5};
  const Index strides[] = {1, 3, 0, 12};
  EXPECT_THAT(ComputeStridedLayoutDimensionIterationOrder(
                  {ContiguousLayoutOrder::fortran, include_repeated_elements},
                  shape, span({strides})),
              ElementsAre(3, 2, 1, 0));
  EXPECT_THAT(ComputeStridedLayoutDimensionIterationOrder(
                  {ContiguousLayoutOrder::fortran, skip_repeated_elements},
                  shape, span({strides})),
              ElementsAre(3, 1, 0));
}

TEST(ComputeStridedLayoutDimensionIterationOrderTest, C3D) {
  const Index shape[] = {3, 4, 5};
  const Index strides[] = {1, 3, 12};
  EXPECT_THAT(ComputeStridedLayoutDimensionIterationOrder(
                  {ContiguousLayoutOrder::c, include_repeated_elements}, shape,
                  span({strides})),
              ElementsAre(0, 1, 2));
}

TEST(SimplifyStridedIterationLayoutTest, Unconstrained1D1Layout) {
  const Index shape[] = {3, 4, 5};
  const Index strides0[] = {20, 1, 4};
  auto layout = SimplifyStridedIterationLayout<1>(include_repeated_elements,
                                                  shape, {{strides0}});
  StridedIterationLayout<1> expected_layout{{60, {{1}}}};
  EXPECT_EQ(expected_layout, layout);
}

TEST(SimplifyStridedIterationLayoutTest, Unconstrained1D1LayoutSkipRepeated) {
  const Index shape[] = {3, 5, 4, 5};
  const Index strides0[] = {20, 0, 1, 4};

  {
    auto layout = SimplifyStridedIterationLayout<1>(include_repeated_elements,
                                                    shape, {{strides0}});
    StridedIterationLayout<1> expected_layout{{{60, {{1}}}, {5, {{0}}}}};
    EXPECT_EQ(expected_layout, layout);
  }

  {
    auto layout = SimplifyStridedIterationLayout<1>(skip_repeated_elements,
                                                    shape, {{strides0}});
    StridedIterationLayout<1> expected_layout{{60, {{1}}}};
    EXPECT_EQ(expected_layout, layout);
  }
}

TEST(SimplifyStridedIterationLayoutTest, Unconstrained1D1LayoutSingletonDims) {
  const Index shape[] = {3, 1, 4, 5};
  const Index strides0[] = {20, 5, 1, 4};
  auto layout = SimplifyStridedIterationLayout<1>(include_repeated_elements,
                                                  shape, {{strides0}});
  StridedIterationLayout<1> expected_layout{{60, {{1}}}};
  EXPECT_EQ(expected_layout, layout);
}

TEST(SimplifyStridedIterationLayoutTest, Unconstrained1D2Layouts) {
  const Index shape[] = {3, 4, 5};
  const Index strides0[] = {20, 1, 4};
  const Index strides1[] = {40, 2, 8};
  auto layout = SimplifyStridedIterationLayout<2>(
      include_repeated_elements, shape, {{strides0, strides1}});
  StridedIterationLayout<2> expected_layout{{60, {{1, 2}}}};
  EXPECT_EQ(expected_layout, layout);
}

TEST(SimplifyStridedIterationLayoutTest, Unconstrained1D2LayoutsSkipRepeated) {
  const Index shape[] = {3, 5, 4, 5, 2};
  const Index strides0[] = {20, 0, 1, 4, 71};
  const Index strides1[] = {40, 0, 2, 8, 0};
  {
    auto layout = SimplifyStridedIterationLayout<2>(
        include_repeated_elements, shape, {{strides0, strides1}});
    StridedIterationLayout<2> expected_layout{
        {2, {{71, 0}}}, {60, {{1, 2}}}, {5, {{0, 0}}}};
    EXPECT_EQ(expected_layout, layout);
  }
  {
    auto layout = SimplifyStridedIterationLayout<2>(
        skip_repeated_elements, shape, {{strides0, strides1}});
    StridedIterationLayout<2> expected_layout{{2, {{71, 0}}}, {60, {{1, 2}}}};
    EXPECT_EQ(expected_layout, layout);
  }
  {
    auto layout = SimplifyStridedIterationLayout<2>(
        skip_repeated_elements, shape, {{strides1, strides0}});
    StridedIterationLayout<2> expected_layout{{60, {{2, 1}}}, {2, {{0, 71}}}};
    EXPECT_EQ(expected_layout, layout);
  }
}

TEST(SimplifyStridedIterationLayoutTest, Fortran1D) {
  const Index shape[] = {3, 4, 5};
  const Index strides[] = {1, 3, 12};
  auto layout = SimplifyStridedIterationLayout<1>(
      {ContiguousLayoutOrder::fortran, include_repeated_elements}, shape,
      {{strides}});
  StridedIterationLayout<1> expected_layout{{60, {{1}}}};
  EXPECT_EQ(expected_layout, layout);
}

TEST(SimplifyStridedIterationLayoutTest, Fortran1DSkipRepeated) {
  const Index shape[] = {3, 4, 2, 5};
  const Index strides[] = {1, 3, 0, 12};
  {
    auto layout = SimplifyStridedIterationLayout<1>(
        {ContiguousLayoutOrder::fortran, include_repeated_elements}, shape,
        {{strides}});
    StridedIterationLayout<1> expected_layout{
        {5, {{12}}}, {2, {{0}}}, {12, {{1}}}};
    EXPECT_EQ(expected_layout, layout);
  }
  {
    auto layout = SimplifyStridedIterationLayout<1>(
        {ContiguousLayoutOrder::fortran, skip_repeated_elements}, shape,
        {{strides}});
    StridedIterationLayout<1> expected_layout{{60, {{1}}}};
    EXPECT_EQ(expected_layout, layout);
  }
}

TEST(SimplifyStridedIterationLayoutTest, C3D) {
  const Index shape[] = {3, 4, 5};
  const Index strides[] = {1, 3, 12};
  auto layout = SimplifyStridedIterationLayout<1>(
      {ContiguousLayoutOrder::c, include_repeated_elements}, shape,
      {{strides}});
  StridedIterationLayout<1> expected_layout{
      {3, {{1}}}, {4, {{3}}}, {5, {{12}}}};
  EXPECT_EQ(expected_layout, layout);
}

TEST(ExtractInnerShapeAndStridesTest, N2Rank2Inner0) {
  StridedIterationLayout<2> iteration_layout{{3, {{1, 2}}}, {4, {{4, 5}}}};
  auto inner_layout = ExtractInnerShapeAndStrides<0>(&iteration_layout);
  InnerShapeAndStrides<2, 0> expected_inner;
  StridedIterationLayout<2> expected_outer{{3, {{1, 2}}}, {4, {{4, 5}}}};
  EXPECT_EQ(expected_inner, inner_layout);
  EXPECT_EQ(expected_outer, iteration_layout);
}

TEST(ExtractInnerShapeAndStridesTest, N2Rank2Inner1) {
  StridedIterationLayout<2> iteration_layout{{3, {{1, 2}}}, {4, {{4, 5}}}};
  auto inner_layout = ExtractInnerShapeAndStrides<1>(&iteration_layout);
  InnerShapeAndStrides<2, 1> expected_inner{{{4}}, {{{{4}}, {{5}}}}};
  StridedIterationLayout<2> expected_outer{{3, {{1, 2}}}};
  EXPECT_EQ(expected_inner, inner_layout);
  EXPECT_EQ(expected_outer, iteration_layout);
}

TEST(ExtractInnerShapeAndStridesTest, N2Rank2Inner2) {
  StridedIterationLayout<2> iteration_layout{{3, {{1, 2}}}, {4, {{4, 5}}}};
  auto inner_layout = ExtractInnerShapeAndStrides<2>(&iteration_layout);
  InnerShapeAndStrides<2, 2> expected_inner{{{3, 4}}, {{{{1, 4}}, {{2, 5}}}}};
  StridedIterationLayout<2> expected_outer;
  EXPECT_EQ(expected_inner, inner_layout);
  EXPECT_EQ(expected_outer, iteration_layout);
}

TEST(ExtractInnerShapeAndStridesTest, N2Rank2Inner3) {
  StridedIterationLayout<2> iteration_layout{{3, {{1, 2}}}, {4, {{4, 5}}}};
  auto inner_layout = ExtractInnerShapeAndStrides<3>(&iteration_layout);
  InnerShapeAndStrides<2, 3> expected_inner{{{1, 3, 4}},
                                            {{{{0, 1, 4}}, {{0, 2, 5}}}}};
  StridedIterationLayout<2> expected_outer;
  EXPECT_EQ(expected_inner, inner_layout);
  EXPECT_EQ(expected_outer, iteration_layout);
}

template <typename Func, typename... Pointer>
std::invoke_result_t<Func&, Pointer...> IterateOverStridedLayouts(
    span<const Index> shape,
    std::array<const Index*, sizeof...(Pointer)> strides, Func&& func,
    tensorstore::IterationConstraints constraints, Pointer... pointer) {
  auto iteration_layout =
      SimplifyStridedIterationLayout(constraints, shape, strides);
  return tensorstore::internal_iterate::IterateHelper<Func&, Pointer...>::Start(
      func, iteration_layout, pointer...);
}

TEST(IterateOverStridedLayoutsTest, InnerRank0ContiguousC) {
  const Index shape[] = {2, 3};
  const Index strides0[] = {3, 1};
  const Index strides1[] = {3 * 4, 1 * 4};

  using R = std::tuple<int, int>;
  std::vector<R> result;
  auto func = [&](int a, int b) {
    result.emplace_back(a, b);
    return true;
  };
  IterateOverStridedLayouts(shape, {{strides0, strides1}}, func,
                            ContiguousLayoutOrder::c, 0, 0);
  std::vector<R> expected_result{R{0, 0},  R{1, 4},  R{2, 8},
                                 R{3, 12}, R{4, 16}, R{5, 20}};
  EXPECT_EQ(expected_result, result);
}

TEST(IterateOverStridedLayoutsTest, EmptyDomain) {
  const Index shape[] = {0, 3};
  const Index strides[] = {0, 1};

  std::vector<int> result;
  auto func = [&](int a) {
    result.emplace_back(a);
    return true;
  };
  IterateOverStridedLayouts(shape, {{strides}}, func,
                            {ContiguousLayoutOrder::c, skip_repeated_elements},
                            0);
  EXPECT_THAT(result, ::testing::ElementsAre());
}

TEST(IterateOverStridedLayoutsTest, InnerRank0ContiguousCStop) {
  const Index shape[] = {2, 3};
  const Index strides0[] = {3, 1};
  const Index strides1[] = {3 * 4, 1 * 4};

  using R = std::tuple<int, int>;
  std::vector<R> result;
  auto func = [&](int a, int b) {
    result.emplace_back(a, b);
    return a != 2;
  };
  EXPECT_EQ(false,
            IterateOverStridedLayouts(shape, {{strides0, strides1}}, func,
                                      ContiguousLayoutOrder::c, 0, 0));
  std::vector<R> expected_result{R{0, 0}, R{1, 4}, R{2, 8}};
  EXPECT_EQ(expected_result, result);
}

TEST(IterateOverStridedLayoutsTest, InnerRank0NonContiguousFortran) {
  const Index shape[] = {2, 3};
  const Index strides0[] = {3, 1};
  const Index strides1[] = {3 * 4, 1 * 4};

  using R = std::tuple<int, int>;
  std::vector<R> result;
  auto func = [&](int a, int b) {
    result.emplace_back(a, b);
    return true;
  };
  IterateOverStridedLayouts(shape, {{strides0, strides1}}, func,
                            ContiguousLayoutOrder::fortran, 0, 0);
  std::vector<R> expected_result{R{0, 0},  R{3, 12}, R{1, 4},
                                 R{4, 16}, R{2, 8},  R{5, 20}};
  EXPECT_EQ(expected_result, result);
}

TEST(IterateOverStridedLayoutsTest, InnerRank0NonContiguousFortranStop) {
  const Index shape[] = {2, 3};
  const Index strides0[] = {3, 1};
  const Index strides1[] = {3 * 4, 1 * 4};

  using R = std::tuple<int, int>;
  std::vector<R> result;
  auto func = [&](int a, int b) {
    result.emplace_back(a, b);
    return a != 3;
  };
  IterateOverStridedLayouts(shape, {{strides0, strides1}}, func,
                            ContiguousLayoutOrder::fortran, 0, 0);
  std::vector<R> expected_result{R{0, 0}, R{3, 12}};
  EXPECT_EQ(expected_result, result);
}

TEST(ArrayIterateResultTest, Comparison) {
  ArrayIterateResult r0{true, 3};
  ArrayIterateResult r1{true, 4};
  ArrayIterateResult r2{false, 4};
  EXPECT_EQ(r0, r0);
  EXPECT_EQ(r1, r1);
  EXPECT_EQ(r2, r2);
  EXPECT_NE(r0, r1);
  EXPECT_NE(r0, r2);
  EXPECT_NE(r1, r2);
}

TEST(ArrayIterateResultTest, PrintToOstream) {
  EXPECT_EQ("{success=1, count=3}",
            tensorstore::StrCat(ArrayIterateResult{true, 3}));
}

template <ContiguousLayoutOrder Order>
std::vector<std::vector<int>> GetIndexVectors(std::vector<int> shape) {
  std::vector<std::vector<int>> result;

  std::vector<int> indices(shape.size());
  do {
    result.push_back(indices);
  } while (AdvanceIndices<Order>(indices.size(), indices.data(), shape.data()));
  return result;
}

template <ContiguousLayoutOrder Order>
std::vector<std::vector<int>> GetIndexVectors(std::vector<int> inclusive_min,
                                              std::vector<int> exclusive_max) {
  std::vector<std::vector<int>> result;

  std::vector<int> indices = inclusive_min;
  do {
    result.push_back(indices);
  } while (AdvanceIndices<Order>(indices.size(), indices.data(),
                                 inclusive_min.data(), exclusive_max.data()));
  return result;
}

TEST(AdvanceIndicesTest, COrderRank0) {
  EXPECT_THAT(GetIndexVectors<c_order>({}), ElementsAre(ElementsAre()));
}

TEST(AdvanceIndicesTest, FortranOrderRank0) {
  EXPECT_THAT(GetIndexVectors<fortran_order>({}), ElementsAre(ElementsAre()));
}

TEST(AdvanceIndicesTest, COrderShape) {
  EXPECT_THAT(GetIndexVectors<c_order>({2, 3}),
              ElementsAre(  //
                  ElementsAre(0, 0), ElementsAre(0, 1), ElementsAre(0, 2),
                  ElementsAre(1, 0), ElementsAre(1, 1), ElementsAre(1, 2)));
}

TEST(AdvanceIndicesTest, FortranOrderShape) {
  EXPECT_THAT(GetIndexVectors<fortran_order>({2, 3}),
              ElementsAre(ElementsAre(0, 0), ElementsAre(1, 0),  //
                          ElementsAre(0, 1), ElementsAre(1, 1),  //
                          ElementsAre(0, 2), ElementsAre(1, 2)));
}

TEST(AdvanceIndicesTest, COrderInclusiveMinExclusiveMax) {
  EXPECT_THAT(
      GetIndexVectors<c_order>({1, 2}, {3, 5}),
      ElementsAre(ElementsAre(1, 2), ElementsAre(1, 3), ElementsAre(1, 4),
                  ElementsAre(2, 2), ElementsAre(2, 3), ElementsAre(2, 4)));
}

TEST(AdvanceIndicesTest, FortranOrderInclusiveMinExclusiveMax) {
  EXPECT_THAT(GetIndexVectors<fortran_order>({1, 2}, {3, 5}),
              ElementsAre(ElementsAre(1, 2), ElementsAre(2, 2),  //
                          ElementsAre(1, 3), ElementsAre(2, 3),  //
                          ElementsAre(1, 4), ElementsAre(2, 4)));
}

// Test bool specialization.
static_assert(DefaultIterationResult<bool>::value() == true, "");

// Test default definition.
static_assert(DefaultIterationResult<int>::value() == 0, "");

}  // namespace
