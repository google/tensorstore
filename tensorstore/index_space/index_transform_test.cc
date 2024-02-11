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
#include <string_view>
#include <type_traits>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorstore/array.h"
#include "tensorstore/container_kind.h"
#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/index_space/index_domain.h"
#include "tensorstore/index_space/index_domain_builder.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/index_space/internal/transform_rep.h"
#include "tensorstore/rank.h"
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/serialization/test_util.h"
#include "tensorstore/static_cast.h"
#include "tensorstore/util/dimension_set.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::DimensionIndex;
using ::tensorstore::DimensionSet;
using ::tensorstore::HullIndexDomains;
using ::tensorstore::IdentityTransform;
using ::tensorstore::Index;
using ::tensorstore::IndexDomain;
using ::tensorstore::IndexDomainBuilder;
using ::tensorstore::IndexDomainDimension;
using ::tensorstore::IndexDomainView;
using ::tensorstore::IndexInterval;
using ::tensorstore::IndexTransform;
using ::tensorstore::IndexTransformBuilder;
using ::tensorstore::IndexTransformView;
using ::tensorstore::IntersectIndexDomains;
using ::tensorstore::IsIndexDomain;
using ::tensorstore::kInfIndex;
using ::tensorstore::kMaxFiniteIndex;
using ::tensorstore::kMinFiniteIndex;
using ::tensorstore::MakeArray;
using ::tensorstore::MatchesStatus;
using ::tensorstore::MergeIndexDomains;
using ::tensorstore::Result;
using ::tensorstore::span;
using ::tensorstore::StaticCast;
using ::tensorstore::StaticRankCast;
using ::tensorstore::StrCat;
using ::tensorstore::unchecked;
using ::tensorstore::view;
using ::tensorstore::internal::ComputeInputDimensionReferenceCounts;
using ::tensorstore::internal::GetInputDimensionsForOutputDimension;
using ::tensorstore::internal_index_space::TransformAccess;
using ::tensorstore::serialization::TestSerializationRoundTrip;

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
                .output_single_input_dimension(0, 0)
                .Finalize()
                .value());

  // Check with different output index map input_dimension values.
  EXPECT_NE(IndexTransformBuilder<>(3, 2)
                .output_single_input_dimension(0, 1)
                .Finalize()
                .value(),
            IndexTransformBuilder<>(3, 2)
                .output_single_input_dimension(0, 0)
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
                .output_single_input_dimension(1, 1)
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
    return IdentityTransform(span<const std::string_view>({"x", "y"}));
  };

  auto unlabeled_t = make_identity();

  // Move assign from same type.
  {
    auto unlabeled_t2 = make_identity();
    unlabeled_t2 = make_transform();
    auto* rep_t2 = TransformAccess::rep(unlabeled_t2);
    unlabeled_t = std::move(unlabeled_t2);
    EXPECT_EQ(rep_t2, TransformAccess::rep(unlabeled_t));
    // NOLINTNEXTLINE(bugprone-use-after-move)
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
  ASSERT_EQ(absl::OkStatus(),
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
  ASSERT_EQ(absl::OkStatus(),
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
  ASSERT_EQ(absl::OkStatus(),
            t.TransformIndices(span<const Index, 1>({1}), output_indices));
  EXPECT_THAT(output_indices, ::testing::ElementsAre(5 + 2 * 4));
  ASSERT_EQ(absl::OkStatus(),
            t.TransformIndices(span<const Index, 1>({2}), output_indices));
  EXPECT_THAT(output_indices, ::testing::ElementsAre(5 + 2 * 5));
  ASSERT_EQ(absl::OkStatus(),
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
  ASSERT_EQ(
      absl::OkStatus(),
      t.TransformIndices(span<const Index, 3>({2, 4, 5}), output_indices));
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
  EXPECT_EQ(absl::OkStatus(),
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
  EXPECT_EQ(absl::OkStatus(),
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
  EXPECT_THAT(d.implicit_lower_bounds(), DimensionSet::FromBools({1, 0}));
  EXPECT_THAT(d.implicit_upper_bounds(), DimensionSet::FromBools({0, 1}));
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
                        .value()
                        .domain());
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
  static_assert(std::is_same_v<decltype(d2_cast), Result<IndexDomain<2>>>);
  EXPECT_EQ(d2_cast, d2);

  // Test cast conversion from rvalue
  auto d2_cast_rvalue = StaticRankCast<2>(IndexDomain<>(d_dynamic));
  static_assert(
      std::is_same_v<decltype(d2_cast_rvalue), Result<IndexDomain<2>>>);
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

static_assert(IsIndexDomain<bool> == false);
static_assert(IsIndexDomain<IndexDomain<3>> == true);
static_assert(IsIndexDomain<IndexDomainView<3>> == true);

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

TEST(MergeIndexDomainsTest, Basic) {
  EXPECT_THAT(MergeIndexDomains(IndexDomain<>(), IndexDomain<>()),
              ::testing::Optional(IndexDomain<>()));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto domain1,
                                   IndexDomainBuilder(3)
                                       .implicit_lower_bounds({0, 1, 0})
                                       .implicit_upper_bounds({0, 0, 1})
                                       .origin({0, -kInfIndex, 2})
                                       .inclusive_max({10, 11, kInfIndex})
                                       .labels({"x", "", ""})
                                       .Finalize());

  EXPECT_THAT(MergeIndexDomains(IndexDomain<>(), domain1),
              ::testing::Optional(domain1));
  EXPECT_THAT(MergeIndexDomains(domain1, IndexDomain<>()),
              ::testing::Optional(domain1));
  EXPECT_THAT(MergeIndexDomains(domain1, domain1),
              ::testing::Optional(domain1));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto domain2,
                                   IndexDomainBuilder(4).Finalize());
  EXPECT_THAT(
      MergeIndexDomains(domain1, domain2),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          "Cannot merge index domain \\{ .* \\} with index domain \\{ .* \\}: "
          "Ranks do not match"));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto domain3,
                                   IndexDomainBuilder(3)
                                       .implicit_lower_bounds({0, 1, 0})
                                       .implicit_upper_bounds({0, 0, 1})
                                       .origin({0, 5, 2})
                                       .inclusive_max({10, 11, kInfIndex})
                                       .labels({"x", "y", ""})
                                       .Finalize());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto domain4,
                                   IndexDomainBuilder(3)
                                       .implicit_lower_bounds({0, 1, 0})
                                       .implicit_upper_bounds({0, 0, 0})
                                       .origin({0, -kInfIndex, 2})
                                       .inclusive_max({10, 11, 12})
                                       .labels({"", "y", ""})
                                       .Finalize());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto domain4_merged,
                                   IndexDomainBuilder(3)
                                       .implicit_lower_bounds({0, 1, 0})
                                       .implicit_upper_bounds({0, 0, 0})
                                       .origin({0, -kInfIndex, 2})
                                       .inclusive_max({10, 11, 12})
                                       .labels({"x", "y", ""})
                                       .Finalize());
  EXPECT_THAT(MergeIndexDomains(domain1, domain3),
              ::testing::Optional(domain3));
  EXPECT_THAT(MergeIndexDomains(domain1, domain4),
              ::testing::Optional(domain4_merged));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto domain5,
                                   IndexDomainBuilder(3)
                                       .implicit_lower_bounds({0, 1, 0})
                                       .implicit_upper_bounds({0, 0, 1})
                                       .origin({0, -kInfIndex, 2})
                                       .inclusive_max({10, 11, kInfIndex})
                                       .labels({"z", "", ""})
                                       .Finalize());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto domain6,
                                   IndexDomainBuilder(3)
                                       .implicit_lower_bounds({0, 1, 0})
                                       .implicit_upper_bounds({0, 0, 1})
                                       .origin({2, -kInfIndex, 2})
                                       .inclusive_max({10, 11, kInfIndex})
                                       .labels({"x", "", ""})
                                       .Finalize());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto domain7,
                                   IndexDomainBuilder(3)
                                       .implicit_lower_bounds({0, 1, 0})
                                       .implicit_upper_bounds({0, 0, 1})
                                       .origin({0, -kInfIndex, 2})
                                       .inclusive_max({10, 12, kInfIndex})
                                       .labels({"x", "", ""})
                                       .Finalize());

  EXPECT_THAT(MergeIndexDomains(domain1, domain5),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Cannot merge .*: "
                            "Mismatch in dimension 0: "
                            "Dimension labels do not match"));
  EXPECT_THAT(MergeIndexDomains(domain1, domain6),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Cannot merge .*: "
                            "Mismatch in dimension 0: "
                            "Lower bounds do not match"));
  EXPECT_THAT(MergeIndexDomains(domain1, domain7),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Cannot merge .*: "
                            "Mismatch in dimension 1: "
                            "Upper bounds do not match"));
}

TEST(HullIndexDomains, Basic) {
  EXPECT_THAT(HullIndexDomains(IndexDomain<>(), IndexDomain<>()),
              ::testing::Optional(IndexDomain<>()));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto domain1, IndexDomainBuilder(3)
                        .implicit_lower_bounds({0, 0, 0})
                        .implicit_upper_bounds({0, 0, 1})
                        .origin({1, kMinFiniteIndex, -kInfIndex})
                        .inclusive_max({10, kInfIndex, kMaxFiniteIndex})
                        .labels({"x", "", ""})
                        .Finalize());

  EXPECT_THAT(HullIndexDomains(IndexDomain<>(), domain1),
              ::testing::Optional(domain1));
  EXPECT_THAT(HullIndexDomains(domain1, IndexDomain<>()),
              ::testing::Optional(domain1));
  EXPECT_THAT(HullIndexDomains(domain1, domain1), ::testing::Optional(domain1));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto domain2,
                                   IndexDomainBuilder(4).Finalize());

  EXPECT_THAT(
      HullIndexDomains(domain1, domain2),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          "Cannot hull index domain \\{ .* \\} with index domain \\{ .* \\}: "
          "Ranks do not match"));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto domain3, IndexDomainBuilder(3)
                        .implicit_lower_bounds({0, 1, 1})
                        .implicit_upper_bounds({1, 1, 1})
                        .origin({0, -kInfIndex, kMinFiniteIndex})
                        .inclusive_max({9, kMaxFiniteIndex, kInfIndex})
                        .labels({"x", "y", ""})
                        .Finalize());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto domain4, IndexDomainBuilder(3)
                        .implicit_lower_bounds({0, 1, 0})
                        .implicit_upper_bounds({0, 0, 1})
                        .origin({0, -kInfIndex, -kInfIndex})
                        .inclusive_max({10, kInfIndex, kInfIndex})
                        .labels({"x", "y", ""})
                        .Finalize());

  EXPECT_THAT(HullIndexDomains(domain1, domain3), ::testing::Optional(domain4));
}

TEST(IntersectIndexDomains, Basic) {
  EXPECT_THAT(IntersectIndexDomains(IndexDomain<>(), IndexDomain<>()),
              ::testing::Optional(IndexDomain<>()));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto domain1, IndexDomainBuilder(3)
                        .implicit_lower_bounds({0, 0, 0})
                        .implicit_upper_bounds({0, 0, 1})
                        .origin({1, kMinFiniteIndex, -kInfIndex})
                        .inclusive_max({10, kInfIndex, kMaxFiniteIndex})
                        .labels({"x", "", ""})
                        .Finalize());

  EXPECT_THAT(IntersectIndexDomains(IndexDomain<>(), domain1),
              ::testing::Optional(domain1));
  EXPECT_THAT(IntersectIndexDomains(domain1, IndexDomain<>()),
              ::testing::Optional(domain1));
  EXPECT_THAT(IntersectIndexDomains(domain1, domain1),
              ::testing::Optional(domain1));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto domain2,
                                   IndexDomainBuilder(4).Finalize());

  EXPECT_THAT(IntersectIndexDomains(domain1, domain2),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Cannot intersect index domain \\{ .* \\} with "
                            "index domain \\{ .* \\}: "
                            "Ranks do not match"));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto domain3, IndexDomainBuilder(3)
                        .implicit_lower_bounds({0, 1, 1})
                        .implicit_upper_bounds({1, 1, 1})
                        .origin({0, -kInfIndex, kMinFiniteIndex})
                        .inclusive_max({9, kMaxFiniteIndex, kInfIndex})
                        .labels({"x", "y", ""})
                        .Finalize());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto domain4, IndexDomainBuilder(3)
                        .implicit_lower_bounds({0, 0, 1})
                        .implicit_upper_bounds({1, 1, 1})
                        .origin({1, kMinFiniteIndex, kMinFiniteIndex})
                        .inclusive_max({9, kMaxFiniteIndex, kMaxFiniteIndex})
                        .labels({"x", "y", ""})
                        .Finalize());

  EXPECT_THAT(IntersectIndexDomains(domain1, domain3),
              ::testing::Optional(domain4));

  // Somewhat surprising: implicit vs. explicit bounds make a difference.
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto domain5, IndexDomainBuilder(3)
                        .implicit_lower_bounds({0, 0, 0})
                        .implicit_upper_bounds({1, 1, 1})
                        .origin({0, -kInfIndex, kMinFiniteIndex})
                        .inclusive_max({9, kMaxFiniteIndex, kInfIndex})
                        .labels({"x", "y", ""})
                        .Finalize());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto domain6, IndexDomainBuilder(3)
                        .implicit_lower_bounds({0, 0, 0})
                        .implicit_upper_bounds({1, 1, 1})
                        .origin({1, kMinFiniteIndex, kMinFiniteIndex})
                        .inclusive_max({9, kMaxFiniteIndex, kMaxFiniteIndex})
                        .labels({"x", "y", ""})
                        .Finalize());

  EXPECT_THAT(IntersectIndexDomains(domain1, domain5),
              ::testing::Optional(domain6));
}

TEST(ConstrainIndexDomain, Basic) {
  using ::tensorstore::ConstrainIndexDomain;
  EXPECT_THAT(ConstrainIndexDomain(IndexDomain<>(), IndexDomain<>()),
              ::testing::Optional(IndexDomain<>()));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto domain1, IndexDomainBuilder(3)
                        .implicit_lower_bounds({0, 0, 0})
                        .implicit_upper_bounds({0, 0, 1})
                        .origin({1, kMinFiniteIndex, -kInfIndex})
                        .inclusive_max({10, kInfIndex, kMaxFiniteIndex})
                        .labels({"x", "", ""})
                        .Finalize());

  EXPECT_THAT(ConstrainIndexDomain(IndexDomain<>(), domain1),
              ::testing::Optional(domain1));
  EXPECT_THAT(ConstrainIndexDomain(domain1, IndexDomain<>()),
              ::testing::Optional(domain1));
  EXPECT_THAT(ConstrainIndexDomain(domain1, domain1),
              ::testing::Optional(domain1));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto domain2,
                                   IndexDomainBuilder(4).Finalize());

  EXPECT_THAT(ConstrainIndexDomain(domain1, domain2),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Cannot constrain index domain \\{ .* \\} with "
                            "index domain \\{ .* \\}: "
                            "Ranks do not match"));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto domain3, IndexDomainBuilder(3)
                        .implicit_lower_bounds({0, 1, 1})
                        .implicit_upper_bounds({1, 1, 1})
                        .origin({0, -kInfIndex, -100})
                        .inclusive_max({9, kMaxFiniteIndex, kInfIndex})
                        .labels({"x", "y", ""})
                        .Finalize());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto domain4, IndexDomainBuilder(3)
                        .implicit_lower_bounds({0, 0, 1})
                        .implicit_upper_bounds({1, 1, 1})
                        .origin({0, kMinFiniteIndex, -100})
                        .inclusive_max({9, kMaxFiniteIndex, kMaxFiniteIndex})
                        .labels({"x", "y", ""})
                        .Finalize());

  EXPECT_THAT(ConstrainIndexDomain(domain3, domain1),
              ::testing::Optional(domain4));
}

TEST(IndexTransformTest, WithImplicitDimensions) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto expected_transform,
                                   IndexTransformBuilder(3, 3)
                                       .implicit_lower_bounds({0, 1, 1})
                                       .implicit_upper_bounds({1, 0, 1})
                                       .output_identity_transform()
                                       .Finalize());
  EXPECT_EQ(expected_transform,
            WithImplicitDimensions(IdentityTransform(3),
                                   DimensionSet::FromBools({0, 1, 1}),
                                   DimensionSet::FromBools({1, 0, 1})));
}

// Tests that input dimensions used by index array output index maps remain
// explicit.
TEST(IndexTransformTest, WithImplicitDimensionsIndexArray) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto expected_transform,
      IndexTransformBuilder(1, 1)
          .input_shape({3})
          .output_index_array(0, 0, 1, MakeArray<Index>({0, 1, 2}))
          .Finalize());
  EXPECT_EQ(
      expected_transform,
      WithImplicitDimensions(expected_transform, DimensionSet::FromBools({1}),
                             DimensionSet::FromBools({1})));

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto expected_domain,
                                   IndexDomainBuilder(1)
                                       .shape({3})
                                       .implicit_lower_bounds({1})
                                       .implicit_upper_bounds({1})
                                       .Finalize());
  // Verify that bounds are marked implicit when applied to just the domain.
  EXPECT_EQ(expected_domain,
            WithImplicitDimensions(expected_transform.domain(),
                                   DimensionSet::FromBools({1}),
                                   DimensionSet::FromBools({1})));
}

TEST(IndexTransformTest, WithImplicitDimensionsStaticRank) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto expected_transform,
                                   (IndexTransformBuilder<3, 3>()
                                        .implicit_lower_bounds({0, 1, 1})
                                        .implicit_upper_bounds({1, 0, 1})
                                        .output_identity_transform()
                                        .Finalize()));
  EXPECT_EQ(expected_transform,
            WithImplicitDimensions(IdentityTransform<3>(),
                                   DimensionSet::FromBools({0, 1, 1}),
                                   DimensionSet::FromBools({1, 0, 1})));
}

TEST(IndexDomainTest, WithImplicitDimensions) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto expected_domain,
                                   IndexDomainBuilder(3)
                                       .implicit_lower_bounds({0, 1, 1})
                                       .implicit_upper_bounds({1, 0, 1})
                                       .Finalize());
  EXPECT_EQ(
      expected_domain,
      WithImplicitDimensions(IndexDomain(3), DimensionSet::FromBools({0, 1, 1}),
                             DimensionSet::FromBools({1, 0, 1})));
}

TEST(IndexDomainTest, WithImplicitDimensionsStaticRank) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto expected_domain,
                                   IndexDomainBuilder<3>()
                                       .implicit_lower_bounds({0, 1, 1})
                                       .implicit_upper_bounds({1, 0, 1})
                                       .Finalize());
  EXPECT_EQ(expected_domain,
            WithImplicitDimensions(IndexDomain<3>(tensorstore::StaticRank<3>{}),
                                   DimensionSet::FromBools({0, 1, 1}),
                                   DimensionSet::FromBools({1, 0, 1})));
}

TEST(IndexDomainTest, ApplyIndexTransform) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto domain,
      IndexDomainBuilder<3>().origin({1, 2, 3}).shape({5, 5, 5}).Finalize());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto transform, (IndexTransformBuilder<4, 3>()
                           // output_dim, offset, stride, input_dim
                           .output_single_input_dimension(0, 5, 1, 3)
                           .output_single_input_dimension(1, -7, 1, 0)
                           .output_single_input_dimension(2, 3, 1, 1)
                           .Finalize()));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto expected_domain,
                                   IndexDomainBuilder<4>()
                                       .origin({9, 0, -kInfIndex, -4})
                                       .shape({5, 5, tensorstore::kInfSize, 5})
                                       .implicit_lower_bounds({0, 0, 1, 0})
                                       .implicit_upper_bounds({0, 0, 1, 0})
                                       .Finalize());
  EXPECT_THAT(domain | transform, ::testing::Optional(expected_domain));
}

TEST(IndexTransformSerializationTest, Basic) {
  TestSerializationRoundTrip(tensorstore::IndexTransform<>());
  TestSerializationRoundTrip(tensorstore::IdentityTransform(5));
}

TEST(IndexDomainSerializationTest, Basic) {
  TestSerializationRoundTrip(tensorstore::IndexDomain<>());
  TestSerializationRoundTrip(
      tensorstore::IndexDomain<>(tensorstore::IdentityTransform(5).domain()));
}

TEST(ComputeInputDimensionReferenceCountsTest, Identity) {
  DimensionIndex reference_counts[3];
  ComputeInputDimensionReferenceCounts(IdentityTransform(3), reference_counts);
  EXPECT_THAT(reference_counts, ::testing::ElementsAre(1, 1, 1));
}

TEST(ComputeInputDimensionReferenceCountsTest, IndexArray) {
  DimensionIndex reference_counts[3];
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto transform,
      IndexTransformBuilder(3, 1)
          .input_shape({2, 2, 2})
          .output_index_array(0, 0, 1, MakeArray<Index>({{{1, 2}, {3, 4}}}))
          .Finalize());
  ComputeInputDimensionReferenceCounts(transform, reference_counts);
  EXPECT_THAT(reference_counts, ::testing::ElementsAre(0, 1, 1));
}

TEST(GetInputDimensionsForOutputDimensionTest, Basic) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto transform,
      IndexTransformBuilder(3, 3)
          .input_shape({2, 2, 2})
          .output_constant(0, 42)
          .output_single_input_dimension(1, 0, 1, 1)
          .output_index_array(2, 0, 1, MakeArray<Index>({{{1, 2}, {3, 4}}}))
          .Finalize());
  EXPECT_THAT(GetInputDimensionsForOutputDimension(transform, 0),
              ::testing::Pair(DimensionSet(), false));
  EXPECT_THAT(GetInputDimensionsForOutputDimension(transform, 1),
              ::testing::Pair(DimensionSet::FromBools({0, 1, 0}), false));
  EXPECT_THAT(GetInputDimensionsForOutputDimension(transform, 2),
              ::testing::Pair(DimensionSet::FromBools({0, 1, 1}), true));
}

TEST(TranslateOutputDimensionsByTest, Basic) {
  auto orig_transform = IdentityTransform(3);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto expected_transform, IndexTransformBuilder(3, 3)
                                   .output_single_input_dimension(0, 1, 1, 0)
                                   .output_single_input_dimension(1, 2, 1, 1)
                                   .output_single_input_dimension(2, 3, 1, 2)
                                   .Finalize());
  EXPECT_THAT(TranslateOutputDimensionsBy(orig_transform, {{1, 2, 3}}),
              ::testing::Optional(expected_transform));
}

}  // namespace
