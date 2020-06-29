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

/// Tests for core functionality of IndexTransform.

#include "tensorstore/index_space/index_transform.h"

#include <array>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/array.h"
#include "tensorstore/index_space/index_domain_builder.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using tensorstore::AllocateArray;
using tensorstore::Box;
using tensorstore::DimensionIndex;
using tensorstore::IdentityTransform;
using tensorstore::Index;
using tensorstore::IndexDomain;
using tensorstore::IndexDomainBuilder;
using tensorstore::IndexDomainDimension;
using tensorstore::IndexDomainView;
using tensorstore::IndexInterval;
using tensorstore::IndexTransform;
using tensorstore::IndexTransformBuilder;
using tensorstore::IndexTransformView;
using tensorstore::IsIndexDomain;
using tensorstore::kImplicit;
using tensorstore::kInfIndex;
using tensorstore::kMaxFiniteIndex;
using tensorstore::kMinFiniteIndex;
using tensorstore::MakeArray;
using tensorstore::MakeOffsetArray;
using tensorstore::MatchesStatus;
using tensorstore::Result;
using tensorstore::span;
using tensorstore::StaticCast;
using tensorstore::StaticRankCast;
using tensorstore::Status;
using tensorstore::StrCat;
using tensorstore::unchecked;
using tensorstore::view;
using tensorstore::internal_index_space::TransformAccess;

TEST(IndexTransformTest, Equality) {
  // Two invalid transforms are equal.
  EXPECT_EQ(IndexTransform<>(), IndexTransform<>());

  // Check that initial values are equal.
  EXPECT_EQ(IndexTransformBuilder<>(2, 3).Finalize().value(),
            IndexTransformBuilder<>(2, 3).Finalize().value());

  // Check with different output ranks.
  EXPECT_NE(IndexTransformBuilder<>(2, 3).Finalize().value(),
            IndexTransformBuilder<>(2, 2).Finalize().value());

  // Check with different input ranks.
  EXPECT_NE(IndexTransformBuilder<>(3, 2).Finalize().value(),
            IndexTransformBuilder<>(2, 2).Finalize().value());

  // Check with non-initial input_shape values.
  EXPECT_EQ(
      IndexTransformBuilder<>(3, 2).input_shape({2, 3, 4}).Finalize().value(),
      IndexTransformBuilder<>(3, 2).input_shape({2, 3, 4}).Finalize().value());

  // Check with different input_shape values.
  EXPECT_NE(
      IndexTransformBuilder<>(3, 2).input_shape({2, 3, 4}).Finalize().value(),
      IndexTransformBuilder<>(3, 2).input_shape({2, 3, 3}).Finalize().value());

  // Check with non-initial input_origin values.
  EXPECT_EQ(IndexTransformBuilder<>(3, 2)
                .input_origin({1, 2, 3})
                .input_shape({3, 4, 5})
                .Finalize()
                .value(),
            IndexTransformBuilder<>(3, 2)
                .input_origin({1, 2, 3})
                .input_shape({3, 4, 5})
                .Finalize()
                .value());

  // Check with different input_origin values.
  EXPECT_NE(IndexTransformBuilder<>(3, 2)
                .input_origin({1, 2, 3})
                .input_shape({3, 4, 5})
                .Finalize()
                .value(),
            IndexTransformBuilder<>(3, 2)
                .input_origin({1, 2, 2})
                .input_shape({3, 4, 5})
                .Finalize()
                .value());

  // Check with non-initial input_labels values.
  EXPECT_EQ(IndexTransformBuilder<>(3, 2)
                .input_labels({"x", "y", "z"})
                .Finalize()
                .value(),
            IndexTransformBuilder<>(3, 2)
                .input_labels({"x", "y", "z"})
                .Finalize()
                .value());

  // Check with different input_labels values.
  EXPECT_NE(IndexTransformBuilder<>(3, 2)
                .input_labels({"a", "b", "c"})
                .Finalize()
                .value(),
            IndexTransformBuilder<>(3, 2)
                .input_labels({"a", "b", "d"})
                .Finalize()
                .value());

  // Check with different output index map methods.
  EXPECT_NE(IndexTransformBuilder<>(3, 2).Finalize().value(),
            IndexTransformBuilder<>(3, 2)
                .output_single_input_dimension(0, 0, 1, 0)
                .Finalize()
                .value());

  // Check with different output index map input_dimension values.
  EXPECT_NE(IndexTransformBuilder<>(3, 2)
                .output_single_input_dimension(0, 0, 1, 1)
                .Finalize()
                .value(),
            IndexTransformBuilder<>(3, 2)
                .output_single_input_dimension(0, 0, 1, 0)
                .Finalize()
                .value());

  // Check with non-initial constant output offset values.
  EXPECT_EQ(
      IndexTransformBuilder<>(3, 2).output_constant(0, 2).Finalize().value(),
      IndexTransformBuilder<>(3, 2).output_constant(0, 2).Finalize().value());

  // Check with different constant output offset values.
  EXPECT_NE(
      IndexTransformBuilder<>(3, 2).output_constant(0, 1).Finalize().value(),
      IndexTransformBuilder<>(3, 2).output_constant(0, 2).Finalize().value());

  // Check with non-initial output stride values.
  EXPECT_EQ(IndexTransformBuilder<>(3, 2)
                .output_single_input_dimension(1, 0, 2, 1)
                .Finalize()
                .value(),
            IndexTransformBuilder<>(3, 2)
                .output_single_input_dimension(1, 0, 2, 1)
                .Finalize()
                .value());

  // Check with different output stride values.
  EXPECT_NE(IndexTransformBuilder<>(3, 2)
                .output_single_input_dimension(1, 0, 2, 1)
                .Finalize()
                .value(),
            IndexTransformBuilder<>(3, 2)
                .output_single_input_dimension(1, 0, 1, 1)
                .Finalize()
                .value());

  // Check with different index arrays.
  EXPECT_NE(IndexTransformBuilder<>(3, 2)
                .input_origin({1, 2, 3})
                .input_shape({2, 2, 3})
                .output_index_array(0, 0, 1, MakeArray<Index>({{{1, 1, 1}}}))
                .Finalize()
                .value(),
            IndexTransformBuilder<>(3, 2)
                .input_origin({1, 2, 3})
                .input_shape({2, 2, 3})
                .output_index_array(0, 0, 1, MakeArray<Index>({{{1, 1, 2}}}))
                .Finalize()
                .value());

  // Check with different index array bounds.
  EXPECT_NE(IndexTransformBuilder<>(3, 2)
                .input_origin({1, 2, 3})
                .input_shape({2, 2, 3})
                .output_index_array(0, 0, 1, MakeArray<Index>({{{1, 1, 2}}}),
                                    IndexInterval::Closed(1, 4))
                .Finalize()
                .value(),
            IndexTransformBuilder<>(3, 2)
                .input_origin({1, 2, 3})
                .input_shape({2, 2, 3})
                .output_index_array(0, 0, 1, MakeArray<Index>({{{1, 1, 2}}}),
                                    IndexInterval::Closed(1, 5))
                .Finalize()
                .value());
}

TEST(IndexTransformTest, ImplicitConversion) {
  IndexTransform<2, 2> t = IdentityTransform<2>();
  IndexTransform<> t_labeled = t;
  EXPECT_EQ(IdentityTransform(2), t_labeled);
}

TEST(IndexTransformTest, Assign) {
  auto make_labeled_transform = [] {
    return IndexTransformBuilder<3, 3>()
        .input_origin({0, 1, 2})
        .input_shape({2, 2, 3})
        .input_labels({"x", "y", "z"})
        .output_index_array(0, 1, 4, MakeArray<Index>({{{1, 1, 2}}}),
                            IndexInterval::Closed(1, 4))
        .output_single_input_dimension(1, 2, 5, 1)
        .output_constant(2, 3)
        .Finalize()
        .value();
  };
  auto make_transform = [] {
    return IndexTransformBuilder<3, 3>()
        .input_origin({0, 1, 2})
        .input_shape({2, 2, 3})
        .output_index_array(0, 1, 4, MakeArray<Index>({{{1, 1, 2}}}),
                            IndexInterval::Closed(1, 4))
        .output_single_input_dimension(1, 2, 5, 1)
        .output_constant(2, 3)
        .Finalize()
        .value();
  };
  auto make_identity = [] { return IdentityTransform(2); };
  auto make_labeled_identity = [] {
    return IdentityTransform(span<const absl::string_view>({"x", "y"}));
  };

  auto unlabeled_t = make_identity();

  // Move assign from same type.
  {
    auto unlabeled_t2 = make_identity();
    unlabeled_t2 = make_transform();
    auto* rep_t2 = TransformAccess::rep(unlabeled_t2);
    unlabeled_t = std::move(unlabeled_t2);
    EXPECT_EQ(rep_t2, TransformAccess::rep(unlabeled_t));
    EXPECT_EQ(nullptr, TransformAccess::rep(unlabeled_t2));
  }

  // Move assign with different static ranks.
  unlabeled_t = make_transform();
  EXPECT_EQ(make_transform(), unlabeled_t);

  // Move assign from default (invalid) transform.
  unlabeled_t = IndexTransform<2, 2>();
  EXPECT_FALSE(unlabeled_t.valid());

  auto labeled_t = make_labeled_identity();

  // Move assign labeled from same index space type (but different static
  // ranks).
  labeled_t = make_labeled_transform();
  EXPECT_EQ(make_labeled_transform(), labeled_t);

  {
    auto labeled_t2 = make_labeled_transform();
    // Copy assign from same index space type (but different static ranks).
    labeled_t = labeled_t2;
    EXPECT_EQ(labeled_t, make_labeled_transform());
    labeled_t = make_labeled_identity();
  }

  {
    // Copy assign from same type.
    auto labeled_t3 = make_labeled_identity();
    labeled_t3 = make_labeled_transform();
    labeled_t = labeled_t3;
    EXPECT_EQ(make_labeled_transform(), labeled_t);
  }

  // Copy assign from default (invalid) transform.
  {
    IndexTransform<2, 2> invalid_t;
    labeled_t = invalid_t;
    EXPECT_FALSE(labeled_t.valid());
  }
}

TEST(IndexTransformTest, ToString) {
  EXPECT_EQ("<Invalid index space transform>",
            StrCat(IndexTransformView<1, 1>()));
  EXPECT_EQ(
      R"s(Rank 3 -> 4 index space transform:
  Input domain:
    0: [1*, 3) "x"
    1: [2, 4*) "y"
    2: [3, 7) "z"
  Output index maps:
    out[0] = 4
    out[1] = 5 + 7 * in[2]
    out[2] = 6
    out[3] = 7 + 9 * bounded([0, 4), array(in)), where array =
      {{{1, 0, 2, 2}}}
)s",
      StrCat(IndexTransformBuilder<>(3, 4)
                 .input_origin({1, 2, 3})
                 .input_shape({2, 2, 4})
                 .implicit_lower_bounds({1, 0, 0})
                 .implicit_upper_bounds({0, 1, 0})
                 .input_labels({"x", "y", "z"})
                 .output_constant(0, 4)
                 .output_single_input_dimension(1, 5, 7, 2)
                 .output_constant(2, 6)
                 .output_index_array(3, 7, 9,
                                     MakeArray<Index>({{{1, 0, 2, 2}}}),
                                     IndexInterval::Closed(0, 3))
                 .Finalize()
                 .value()));
}

// Verify that GoogleTest printing works.
TEST(IndexTransformTest, GTestToString) {
  EXPECT_EQ(
      R"s(Rank 3 -> 4 index space transform:
  Input domain:
    0: [1, 3) "x"
    1: [2, 4) "y"
    2: [3, 7) "z"
  Output index maps:
    out[0] = 4
    out[1] = 5 + 7 * in[2]
    out[2] = 6
    out[3] = 7 + 9 * bounded([0, 4), array(in)), where array =
      {{{1, 0, 2, 2}}}
)s",
      ::testing::PrintToString(
          IndexTransformBuilder<>(3, 4)
              .input_origin({1, 2, 3})
              .input_shape({2, 2, 4})
              .input_labels({"x", "y", "z"})
              .output_constant(0, 4)
              .output_single_input_dimension(1, 5, 7, 2)
              .output_constant(2, 6)
              .output_index_array(3, 7, 9, MakeArray<Index>({{{1, 0, 2, 2}}}),
                                  IndexInterval::Closed(0, 3))
              .Finalize()
              .value()));
}

TEST(IndexTransformTest, Constant) {
  auto t = IndexTransformBuilder<>(1, 1)
               .input_origin({1})
               .input_shape({4})
               .output_constant(0, 10)
               .Finalize()
               .value();
  std::array<Index, 1> output_indices;
  ASSERT_EQ(Status(),
            t.TransformIndices(span<const Index, 1>({3}), output_indices));
  EXPECT_THAT(output_indices, ::testing::ElementsAre(10));
}

TEST(IndexTransformTest, SingleInputDimension) {
  auto t = IndexTransformBuilder<>(1, 1)
               .input_origin({1})
               .input_shape({20})
               .output_single_input_dimension(0, 5, 2, 0)
               .Finalize()
               .value();
  std::array<Index, 1> output_indices;
  ASSERT_EQ(Status(),
            t.TransformIndices(span<const Index, 1>({6}), output_indices));
  EXPECT_THAT(output_indices, ::testing::ElementsAre(5 + 2 * 6));
}

TEST(IndexTransformTest, IndexArray) {
  auto t = IndexTransformBuilder<>(1, 1)
               .input_origin({1})
               .input_shape({3})
               .output_index_array(0, 5, 2, MakeArray<Index>({4, 5, 6}))
               .Finalize()
               .value();
  std::array<Index, 1> output_indices;
  ASSERT_EQ(Status(),
            t.TransformIndices(span<const Index, 1>({1}), output_indices));
  EXPECT_THAT(output_indices, ::testing::ElementsAre(5 + 2 * 4));
  ASSERT_EQ(Status(),
            t.TransformIndices(span<const Index, 1>({2}), output_indices));
  EXPECT_THAT(output_indices, ::testing::ElementsAre(5 + 2 * 5));
  ASSERT_EQ(Status(),
            t.TransformIndices(span<const Index, 1>({3}), output_indices));
  EXPECT_THAT(output_indices, ::testing::ElementsAre(5 + 2 * 6));
}

TEST(TransformIndicesTest, ConstantAndSingleInputDimensionAndIndexArray) {
  auto t = IndexTransformBuilder<>(3, 3)
               .input_origin({1, 2, 3})
               .input_shape({4, 4, 3})
               .output_constant(0, 10)
               .output_single_input_dimension(1, 20, 2, 2)
               .output_index_array(2, 30, 3,
                                   MakeArray<Index>({{{5}, {6}, {7}, {8}}}))
               .Finalize()
               .value();
  std::array<Index, 3> output_indices;
  ASSERT_EQ(Status(), t.TransformIndices(span<const Index, 3>({2, 4, 5}),
                                         output_indices));
  EXPECT_THAT(output_indices,
              ::testing::ElementsAre(10, 20 + 2 * 5, 30 + 3 * 7));
}

TEST(TransformIndicesTest, Implicit) {
  auto t = IndexTransformBuilder<>(1, 1)
               .input_origin({1})
               .implicit_lower_bounds({1})
               .input_shape({3})
               .output_single_input_dimension(0, 0)
               .Finalize()
               .value();
  std::array<Index, 1> output_indices;
  EXPECT_EQ(Status(),
            t.TransformIndices(span<const Index, 1>({-3}), output_indices));
  EXPECT_THAT(output_indices, ::testing::ElementsAre(-3));
  EXPECT_THAT(t.TransformIndices(span<const Index, 1>({10}), output_indices),
              MatchesStatus(absl::StatusCode::kOutOfRange,
                            "Index 10 is not contained in the domain "
                            "\\[1\\*, 4\\) for input dimension 0"));
}

TEST(TransformIndicesTest, IndexRangeError) {
  auto t = IndexTransformBuilder<>(1, 1)
               .input_origin({1})
               .input_shape({3})
               .output_index_array(0, 0, 1, MakeArray<Index>({5, 6, 7}),
                                   IndexInterval::Closed(6, 7))
               .Finalize()
               .value();
  std::array<Index, 1> output_indices;
  EXPECT_EQ(Status(),
            t.TransformIndices(span<const Index, 1>({2}), output_indices));
  EXPECT_THAT(output_indices, ::testing::ElementsAre(6));
  EXPECT_THAT(t.TransformIndices(span<const Index, 1>({1}), output_indices),
              MatchesStatus(absl::StatusCode::kOutOfRange,
                            "Computing index for output dimension 0: "
                            "Checking result of index array output index map: "
                            "Index 5 is outside valid range \\[6, 8\\)"));
}

TEST(IndexTransformTest, ConstructMove) {
  auto t = IdentityTransform(2);
  auto* data = TransformAccess::rep(t);
  IndexTransform<> t2(std::move(t));
  EXPECT_EQ(data, TransformAccess::rep(t2));
}

TEST(IndexTransformTest, AssignMove) {
  auto t = IdentityTransform(2);
  auto* data = TransformAccess::rep(t);
  IndexTransform<> t2;
  t2 = std::move(t);
  EXPECT_EQ(data, TransformAccess::rep(t2));
}

TEST(IndexDomainTest, DefaultConstruct) {
  IndexDomainView<> d;
  EXPECT_FALSE(d.valid());
}

TEST(IndexDomainTest, ConstructFromTransform) {
  auto d = IndexDomainBuilder<2>()
               .origin({1, 2})
               .shape({3, 4})
               .implicit_lower_bounds({1, 0})
               .implicit_upper_bounds({0, 1})
               .labels({"x", "y"})
               .Finalize()
               .value();
  ASSERT_TRUE(d.valid());
  EXPECT_EQ(2, d.rank());
  EXPECT_THAT(d.origin(), ::testing::ElementsAre(1, 2));
  EXPECT_THAT(d.shape(), ::testing::ElementsAre(3, 4));
  EXPECT_THAT(d.implicit_lower_bounds(), ::testing::ElementsAre(1, 0));
  EXPECT_THAT(d.implicit_upper_bounds(), ::testing::ElementsAre(0, 1));
  EXPECT_THAT(d.labels(), ::testing::ElementsAre("x", "y"));
  EXPECT_EQ(IndexDomainDimension<view>(
                {IndexInterval::UncheckedSized(1, 3), true, false}, "x"),
            d[0]);
  EXPECT_EQ(IndexDomainDimension<view>(
                {IndexInterval::UncheckedSized(2, 4), false, true}, "y"),
            d[1]);
  EXPECT_EQ(12, d.num_elements());
}

TEST(IndexDomainTest, CompareEqual) {
  IndexDomain<2> d1;
  auto d2 = IndexDomainBuilder<2>()
                .origin({1, 2})
                .shape({3, 4})
                .implicit_lower_bounds({1, 0})
                .implicit_upper_bounds({0, 1})
                .labels({"x", "y"})
                .Finalize()
                .value();
  // Differs from d2 only in output rank, which does not affect domain.
  IndexDomain<2> d3(IndexTransformBuilder<2, 1>()
                        .input_origin({1, 2})
                        .input_shape({3, 4})
                        .implicit_lower_bounds({1, 0})
                        .implicit_upper_bounds({0, 1})
                        .input_labels({"x", "y"})
                        .output_constant(0, 1)
                        .Finalize()
                        .value());
  // Differs from d2 only in `input_origin`.
  auto d4 = IndexDomainBuilder<2>()
                .origin({1, 3})
                .shape({3, 4})
                .implicit_lower_bounds({1, 0})
                .implicit_upper_bounds({0, 1})
                .labels({"x", "y"})
                .Finalize()
                .value();
  // Differs from d2 only in `input_shape`.
  auto d5 = IndexDomainBuilder<2>()
                .origin({1, 2})
                .shape({3, 5})
                .implicit_lower_bounds({1, 0})
                .implicit_upper_bounds({0, 1})
                .labels({"x", "y"})
                .Finalize()
                .value();
  // Differs from d2 only in `implicit_lower_bounds`.
  auto d6 = IndexDomainBuilder<2>()
                .origin({1, 2})
                .shape({3, 4})
                .implicit_lower_bounds({1, 1})
                .implicit_upper_bounds({0, 1})
                .labels({"x", "y"})
                .Finalize()
                .value();
  // Differs from d2 only in `implicit_upper_bounds`.
  auto d7 = IndexDomainBuilder<2>()
                .origin({1, 2})
                .shape({3, 4})
                .implicit_lower_bounds({1, 0})
                .implicit_upper_bounds({1, 1})
                .labels({"x", "y"})
                .Finalize()
                .value();
  // Differs from d2 only in `input_labels`.
  auto d8 = IndexDomainBuilder<2>()
                .origin({1, 2})
                .shape({3, 4})
                .implicit_lower_bounds({1, 0})
                .implicit_upper_bounds({0, 1})
                .labels({"z", "y"})
                .Finalize()
                .value();
  EXPECT_EQ(d1, d1);
  EXPECT_EQ(d2, d2);
  EXPECT_EQ(d3, d3);
  EXPECT_EQ(d4, d4);
  EXPECT_EQ(d5, d5);
  EXPECT_EQ(d6, d6);
  EXPECT_EQ(d7, d7);
  EXPECT_EQ(d8, d8);

  EXPECT_NE(d1, d2);

  EXPECT_EQ(d2, d3);
  EXPECT_NE(d2, d4);
  EXPECT_NE(d2, d5);
  EXPECT_NE(d2, d6);
  EXPECT_NE(d2, d7);
  EXPECT_NE(d2, d8);
}

TEST(IndexDomainTest, ConvertRank) {
  auto d2 = IndexDomainBuilder<2>()
                .origin({1, 2})
                .shape({3, 4})
                .implicit_lower_bounds({1, 0})
                .implicit_upper_bounds({0, 1})
                .labels({"x", "y"})
                .Finalize()
                .value();

  // Test implicit conversion from lvalue
  IndexDomain<> d_dynamic = d2;
  EXPECT_EQ(d_dynamic, d2);

  // Test implicit conversion from rvalue
  IndexDomain<> d_dynamic_from_rvalue = IndexDomain<2>(d2);
  EXPECT_EQ(d_dynamic_from_rvalue, d2);

  // Test cast conversion from lvalue
  auto d2_cast = StaticRankCast<2>(d_dynamic);
  static_assert(std::is_same<decltype(d2_cast), Result<IndexDomain<2>>>::value,
                "");
  EXPECT_EQ(d2_cast, d2);

  // Test cast conversion from rvalue
  auto d2_cast_rvalue = StaticRankCast<2>(IndexDomain<>(d_dynamic));
  static_assert(
      std::is_same<decltype(d2_cast_rvalue), Result<IndexDomain<2>>>::value,
      "");
  EXPECT_EQ(d2_cast_rvalue, d2);

  // Test failed conversion.
  EXPECT_THAT(StaticRankCast<3>(d_dynamic),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Cannot cast index domain with rank of 2 "
                            "to index domain with rank of 3"));
}

TEST(IndexDomainTest, SubDomain) {
  auto d2 = IndexDomainBuilder<2>()
                .origin({1, 2})
                .shape({3, 4})
                .implicit_lower_bounds({1, 0})
                .implicit_upper_bounds({0, 1})
                .labels({"x", "y"})
                .Finalize()
                .value();

  auto d3 = IndexDomainBuilder<2>()
                .origin({2, 1})
                .shape({4, 3})
                .implicit_lower_bounds({0, 1})
                .implicit_upper_bounds({1, 0})
                .labels({"y", "x"})
                .Finalize()
                .value();
  EXPECT_EQ(d3, (d2[span<const DimensionIndex, 2>({1, 0})]));
}

TEST(IndexDomainTest, PrintToOstream) {
  EXPECT_EQ("<invalid index domain>", StrCat(IndexDomain<2>()));
  auto d2 = IndexDomainBuilder<2>()
                .origin({1, 2})
                .shape({3, 4})
                .implicit_lower_bounds({1, 0})
                .implicit_upper_bounds({0, 1})
                .labels({"x", "y"})
                .Finalize()
                .value();

  EXPECT_EQ(R"({ "x": [1*, 4), "y": [2, 6*) })", StrCat(d2));
}

static_assert(IsIndexDomain<bool>::value == false, "");
static_assert(IsIndexDomain<IndexDomain<3>>::value == true, "");
static_assert(IsIndexDomain<IndexDomainView<3>>::value == true, "");

TEST(CastTest, IndexTransform) {
  auto t = IdentityTransform(span<const Index>({2, 3}));
  auto t2 = StaticCast<IndexTransform<2, 2>, unchecked>(t);
  EXPECT_EQ(IndexTransformBuilder<>(2, 2)
                .input_origin({0, 0})
                .input_shape({2, 3})
                .output_single_input_dimension(0, 0)
                .output_single_input_dimension(1, 1)
                .Finalize()
                .value(),
            t2);

  EXPECT_THAT((StaticCast<IndexTransformView<2, 2>>(t)),
              ::testing::Optional(t));

  EXPECT_THAT(
      (StaticCast<IndexTransform<2, 3>>(t)),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          "Cannot cast "
          "index transform with input rank of 2 and output rank of 2 to "
          "index transform with input rank of 2 and output rank of 3"));

  EXPECT_THAT(
      (tensorstore::StaticRankCast<3>(t)),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          "Cannot cast "
          "index transform with input rank of 2 and output rank of 2 to "
          "index transform with input rank of 3 and output dynamic rank"));
}

TEST(CastTest, IndexTransformView) {
  auto t = IdentityTransform(span<const Index>({2, 3}));
  IndexTransformView<> t_ref = t;
  auto t2 = StaticCast<IndexTransformView<2, 2>>(t_ref);
  EXPECT_EQ(IndexTransformBuilder<>(2, 2)
                .input_origin({0, 0})
                .input_shape({2, 3})
                .output_single_input_dimension(0, 0)
                .output_single_input_dimension(1, 1)
                .Finalize()
                .value(),
            t2);

  EXPECT_THAT((StaticCast<IndexTransform<2, 2>>(t_ref)),
              ::testing::Optional(t));

  EXPECT_THAT(
      (StaticCast<IndexTransformView<2, 3>>(t_ref)),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          "Cannot cast "
          "index transform with input rank of 2 and output rank of 2 to "
          "index transform with input rank of 2 and output rank of 3"));

  EXPECT_THAT(
      (tensorstore::StaticRankCast<3>(t_ref)),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          "Cannot cast "
          "index transform with input rank of 2 and output rank of 2 to "
          "index transform with input rank of 3 and output dynamic rank"));
}

}  // namespace
