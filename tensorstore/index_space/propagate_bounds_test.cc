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

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/index_space/internal/transform_rep.h"
#include "tensorstore/util/bit_vec.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using tensorstore::BitVec;
using tensorstore::Box;
using tensorstore::BoxView;
using tensorstore::DimensionIndex;
using tensorstore::Index;
using tensorstore::IndexInterval;
using tensorstore::IndexTransform;
using tensorstore::IndexTransformBuilder;
using tensorstore::kInfIndex;
using tensorstore::kMinFiniteIndex;
using tensorstore::MakeArray;
using tensorstore::MatchesStatus;
using tensorstore::PropagateBounds;
using tensorstore::PropagateBoundsToTransform;
using tensorstore::PropagateExplicitBounds;
using tensorstore::PropagateExplicitBoundsToTransform;

// Tests that a default-constructed (invalid) IndexTransform is treated as an
// identity transform.
TEST(PropagateExplicitBoundsTest, IdentityTransform) {
  DimensionIndex rank = 2;
  const Box<> b({2, 3}, {4, 5});
  Box<> a(rank);
  EXPECT_EQ(absl::OkStatus(),
            PropagateExplicitBounds(b, IndexTransform<>(), a));
  EXPECT_EQ(a, b);
}

TEST(PropagateBoundsTest, IdentityTransform) {
  auto b = Box({2, 3}, {4, 5});
  Box<2> a;
  BitVec<2> a_implicit_lower_bounds, a_implicit_upper_bounds;
  auto b_implicit_lower_bounds = BitVec({0, 1});
  auto b_implicit_upper_bounds = BitVec({1, 0});
  EXPECT_EQ(absl::OkStatus(),
            PropagateBounds(b, b_implicit_lower_bounds, b_implicit_upper_bounds,
                            IndexTransform<2, 2>(), a, a_implicit_lower_bounds,
                            a_implicit_upper_bounds));
  EXPECT_EQ(a, b);
  EXPECT_EQ(b_implicit_lower_bounds, a_implicit_lower_bounds);
  EXPECT_EQ(b_implicit_upper_bounds, a_implicit_upper_bounds);
}

TEST(PropagateBoundsTest, ValidateOnly) {
  auto transform = IndexTransformBuilder<2, 3>()
                       .input_origin({2, 3})
                       .input_shape({5, 10})
                       .output_single_input_dimension(0, 15, 2, 0)
                       .output_single_input_dimension(1, 30, 3, 1)
                       .output_single_input_dimension(2, 45, 4, 1)
                       .Finalize()
                       .value();
  const Box<3> b({2, 3, 4}, {50, 66, 100});
  Box<2> a;
  BitVec<3> b_implicit_lower_bounds({0, 1, 0});
  BitVec<3> b_implicit_upper_bounds({1, 0, 0});
  BitVec<2> a_implicit_lower_bounds, a_implicit_upper_bounds;

  EXPECT_EQ(absl::OkStatus(),
            PropagateBounds(b, b_implicit_lower_bounds, b_implicit_upper_bounds,
                            transform, a, a_implicit_lower_bounds,
                            a_implicit_upper_bounds));
  // Check that the propagated bounds match the transform input domain.
  EXPECT_EQ(BoxView({2, 3}, {5, 10}), a);
  EXPECT_THAT(a_implicit_lower_bounds, ::testing::ElementsAre(0, 0));
  EXPECT_THAT(a_implicit_upper_bounds, ::testing::ElementsAre(0, 0));
}

TEST(PropagateBoundsTest, Constant) {
  auto transform = IndexTransformBuilder<0, 2>()
                       .output_constant(0, 1)
                       .output_constant(1, 2)
                       .Finalize()
                       .value();
  Box<0> a;
  EXPECT_EQ(absl::OkStatus(),
            PropagateBounds(/*b_domain=*/Box({2, 1}, {2, 3}),
                            /*b_implicit_lower_bounds=*/BitVec({1, 0}),
                            /*b_implicit_upper_bounds=*/BitVec({0, 0}),
                            transform, a));
}

TEST(PropagateBoundsTest, ConstantError) {
  auto transform = IndexTransformBuilder<0, 2>()
                       .output_constant(0, 1)
                       .output_constant(1, 2)
                       .Finalize()
                       .value();
  Box<0> a;
  EXPECT_THAT(
      PropagateBounds(/*b_domain=*/Box({2, 1}, {2, 3}),
                      /*b_implicit_lower_bounds=*/BitVec({0, 1}),
                      /*b_implicit_upper_bounds=*/BitVec({0, 0}), transform, a),
      MatchesStatus(absl::StatusCode::kOutOfRange,
                    "Checking bounds of constant output index map for "
                    "dimension 0: Index 1 is outside valid range .*"));
}

TEST(PropagateBoundsTest, Propagate0Upper1Lower) {
  auto transform = IndexTransformBuilder<2, 3>()
                       .input_origin({2, 3})
                       .implicit_lower_bounds({0, 1})
                       .implicit_upper_bounds({1, 0})
                       .input_shape({5, 10})
                       .output_single_input_dimension(0, 15, 2, 0)
                       .output_single_input_dimension(1, 30, 3, 1)
                       .output_single_input_dimension(2, 45, 4, 1)
                       .Finalize()
                       .value();
  Box<2> a;
  BitVec<2> a_implicit_lower_bounds, a_implicit_upper_bounds;
  EXPECT_EQ(
      absl::OkStatus(),
      PropagateBounds(/*b_domain=*/Box({2, 3, 4}, {50, 66, 100}),
                      /*b_implicit_lower_bounds=*/BitVec({0, 1, 1}),
                      /*b_implicit_upper_bounds=*/BitVec({1, 0, 1}), transform,
                      a, a_implicit_lower_bounds, a_implicit_upper_bounds));

  // a dim 0:
  //   Initial bounds:     [2, 7*)
  //   From b dim 0:       [-6, 19*)

  // a dim 1:
  //   Initial bounds:     [3*, 13)
  //   From b dim 1:       [-9*, 13)
  //   From b dim 2:       [-10*, 15*)

  // Check that implicit initial bounds are updated.
  EXPECT_EQ(BoxView({2, -9}, {19 - 2, 13 - -9}), a);

  // Check that the propagated implicit bits are equal to:
  // `initial && any(dependencies)`.
  EXPECT_THAT(a_implicit_lower_bounds, ::testing::ElementsAre(0, 1));
  EXPECT_THAT(a_implicit_upper_bounds, ::testing::ElementsAre(1, 0));
}

// Tests that implicit bounds from a single dimension of `b` don't constrain the
// transform.
TEST(PropagateBoundsTest, PropagateImplicitConstraints1) {
  const auto transform = IndexTransformBuilder<1, 1>()
                             .input_origin({-1})
                             .input_exclusive_max({2})
                             .implicit_upper_bounds({1})
                             .output_single_input_dimension(0, 0)
                             .Finalize()
                             .value();
  Box<1> a;
  BitVec<1> a_implicit_lower_bounds, a_implicit_upper_bounds;
  EXPECT_EQ(
      absl::OkStatus(),
      PropagateBounds(/*b_domain=*/Box({0}, {4}),
                      /*b_implicit_lower_bounds=*/BitVec({1}),
                      /*b_implicit_upper_bounds=*/BitVec({0}), transform, a,
                      a_implicit_lower_bounds, a_implicit_upper_bounds));

  // a dim 0:
  //   Initial bounds:     [-1, 2*)
  //   From b dim 0:       [0*, 4)

  // Check that implicit initial bounds are updated.
  EXPECT_EQ(BoxView({-1}, {5}), a);

  // Check that the propagated implicit bits are equal to:
  // `initial && any(dependencies)`.
  EXPECT_THAT(a_implicit_lower_bounds, ::testing::ElementsAre(0));
  EXPECT_THAT(a_implicit_upper_bounds, ::testing::ElementsAre(0));
}

// Tests that implicit bounds from two dimensions of `b` don't constrain the
// transform.
TEST(PropagateBoundsTest, PropagateImplicitConstraints2) {
  const auto transform = IndexTransformBuilder<1, 2>()
                             .input_origin({-1})
                             .input_exclusive_max({2})
                             .implicit_upper_bounds({1})
                             .output_single_input_dimension(0, 0)
                             .output_single_input_dimension(1, 0)
                             .Finalize()
                             .value();
  Box<1> a;
  BitVec<1> a_implicit_lower_bounds, a_implicit_upper_bounds;
  EXPECT_EQ(absl::OkStatus(),
            PropagateBounds(
                /*b_domain=*/Box({-1, 0}, {3, 4}),
                /*b_implicit_lower_bounds=*/BitVec({1, 1}),
                /*b_implicit_upper_bounds=*/BitVec({1, 0}), transform, a,
                a_implicit_lower_bounds, a_implicit_upper_bounds));

  // a dim 0:
  //   Initial bounds:     [-1, 2*)
  //   From b dim 0:       [-1*, 2*)
  //   From b dim 1:       [0*, 4)

  // Check that implicit initial bounds are updated.
  EXPECT_EQ(BoxView({-1}, {5}), a);

  // Check that the propagated implicit bits are equal to:
  // `initial && any(dependencies)`.
  EXPECT_THAT(a_implicit_lower_bounds, ::testing::ElementsAre(0));
  EXPECT_THAT(a_implicit_upper_bounds, ::testing::ElementsAre(0));
}

TEST(PropagateBoundsTest, PropagateNegativeStride) {
  auto transform = IndexTransformBuilder<2, 1>()
                       .input_origin({2, 3})
                       .implicit_lower_bounds({0, 1})
                       .implicit_upper_bounds({1, 0})
                       .input_shape({4, 10})
                       .output_single_input_dimension(0, 15, -2, 0)
                       .Finalize()
                       .value();
  const Box<1> b({2}, {50});
  Box<2> a;
  BitVec<1> b_implicit_lower_bounds({0}), b_implicit_upper_bounds({1});
  BitVec<2> a_implicit_lower_bounds, a_implicit_upper_bounds;

  EXPECT_EQ(absl::OkStatus(),
            PropagateBounds(b, b_implicit_lower_bounds, b_implicit_upper_bounds,
                            transform, a, a_implicit_lower_bounds,
                            a_implicit_upper_bounds));

  // a dim 0:
  //   Initial bounds:     [2, 6*)
  //   From b dim 0:       [-18*, 7)

  // a dim 1:
  //   Initial bounds:     [3*, 13)

  // Check that implicit initial bounds are updated.
  EXPECT_EQ(BoxView({2, 3}, {7 - 2, 10}), a);

  // Check that the propagated implicit bits are equal to:
  // `initial && any(dependencies)`.
  EXPECT_THAT(a_implicit_lower_bounds, ::testing::ElementsAre(0, 1));
  EXPECT_THAT(a_implicit_upper_bounds, ::testing::ElementsAre(0, 0));
}

TEST(PropagateExplicitBoundsTest, Propagate0Upper1Upper) {
  auto transform = IndexTransformBuilder<2, 3>()
                       .input_origin({2, 10})
                       .input_shape({5, 11})
                       .implicit_lower_bounds({0, 1})
                       .implicit_upper_bounds({0, 1})
                       .output_single_input_dimension(0, 15, 2, 0)
                       .output_single_input_dimension(1, 30, 3, 1)
                       .output_single_input_dimension(2, 45, 4, 1)
                       .Finalize()
                       .value();
  const Box<3> b({2, 3, 4}, {50, 66, 100});
  Box<2> a;
  EXPECT_EQ(absl::OkStatus(), PropagateExplicitBounds(b, transform, a));
  EXPECT_EQ(Box<>({2, -9}, {5, 22}), a);
}

TEST(PropagateExplicitBoundsTest, PropagateExtraExplicit) {
  auto transform = IndexTransformBuilder<3, 3>()
                       .input_origin({2, 10, 7})
                       .input_shape({5, 11, 8})
                       .implicit_lower_bounds({0, 1, 0})
                       .implicit_upper_bounds({0, 1, 0})
                       .output_single_input_dimension(0, 15, 2, 0)
                       .output_single_input_dimension(1, 30, 3, 1)
                       .output_single_input_dimension(2, 45, 4, 1)
                       .Finalize()
                       .value();
  const Box<3> b({2, 3, 4}, {50, 66, 100});
  Box<3> a;
  EXPECT_EQ(absl::OkStatus(), PropagateExplicitBounds(b, transform, a));
  EXPECT_EQ(Box<>({2, -9, 7}, {5, 22, 8}), a);
}

TEST(PropagateExplicitBoundsTest, PropagateExtraImplicitLower) {
  auto transform = IndexTransformBuilder<3, 3>()
                       .input_origin({2, 10, 7})
                       .input_shape({5, 11, 8})
                       .implicit_lower_bounds({0, 1, 1})
                       .implicit_upper_bounds({0, 1, 0})
                       .output_single_input_dimension(0, 15, 2, 0)
                       .output_single_input_dimension(1, 30, 3, 1)
                       .output_single_input_dimension(2, 45, 4, 1)
                       .Finalize()
                       .value();
  const Box<3> b({2, 3, 4}, {50, 66, 100});
  Box<3> a;
  EXPECT_EQ(absl::OkStatus(), PropagateExplicitBounds(b, transform, a));
  EXPECT_EQ(Box<>({2, -9, 7}, {5, 22, 8}), a);
}

TEST(PropagateExplicitBoundsTest, PropagateExtraImplicitUpper) {
  auto transform = IndexTransformBuilder<3, 3>()
                       .input_origin({2, 10, 7})
                       .input_shape({5, 11, 8})
                       .implicit_lower_bounds({0, 1, 0})
                       .implicit_upper_bounds({0, 1, 1})
                       .output_single_input_dimension(0, 15, 2, 0)
                       .output_single_input_dimension(1, 30, 3, 1)
                       .output_single_input_dimension(2, 45, 4, 1)
                       .Finalize()
                       .value();
  const Box<3> b({2, 3, 4}, {50, 66, 100});
  Box<3> a;
  EXPECT_EQ(absl::OkStatus(), PropagateExplicitBounds(b, transform, a));
  EXPECT_EQ(Box<>({2, -9, 7}, {5, 22, 8}), a);
}

TEST(PropagateExplicitBoundsTest, OutOfBounds) {
  auto transform = IndexTransformBuilder<2, 3>()
                       .input_origin({2, 3})
                       .input_shape({5, 10})
                       .output_single_input_dimension(0, 15, 2, 0)
                       .output_single_input_dimension(1, 30, 3, 1)
                       .output_single_input_dimension(2, 45, 4, 1)
                       .Finalize()
                       .value();
  const Box<3> b({2, 3, 4}, {50, 60, 100});
  // To be contained within `b`, an `input` vector to `transform` must satisfy:
  //
  //     2 <= input[0] * 2 + 15 < 52
  //     3 <= input[1] * 3 + 30 < 63
  //     4 <= input[1] * 4 + 45 < 104
  //
  // which implies:
  //
  //     -13 / 2 <= input[0] < 37 / 2
  //     -27 / 3 <= input[1] < 33 / 3
  //     -41 / 4 <= input[1] < 59 / 4
  //
  // which implies:
  //
  //     -6 <= input[0] <= 18
  //     -9 <= input[1] <= 10
  //     -10 <= input[1] <= 14
  Box<2> a;
  EXPECT_THAT(
      PropagateExplicitBounds(b, transform, a),
      MatchesStatus(absl::StatusCode::kOutOfRange,
                    "Propagated bounds \\[-9, 11\\) for dimension 1 are "
                    "incompatible with existing bounds \\[3, 13\\)\\."));
}

TEST(PropagateExplicitBoundsTest, OutOfBoundsInfLower) {
  auto transform = IndexTransformBuilder<2, 3>()
                       .input_origin({2, -kInfIndex})
                       .input_shape({5, kInfIndex + 4})
                       .output_single_input_dimension(0, 15, 2, 0)
                       .output_single_input_dimension(1, 30, 3, 1)
                       .output_single_input_dimension(2, 45, 4, 1)
                       .Finalize()
                       .value();
  const Box<3> b({2, 3, 4}, {50, 60, 100});
  Box<2> a;
  EXPECT_THAT(
      PropagateExplicitBounds(b, transform, a),
      MatchesStatus(absl::StatusCode::kOutOfRange,
                    "Propagated bounds \\[-9, 11\\) for dimension 1 are "
                    "incompatible with existing bounds \\(-inf, 4\\)\\."));
}

TEST(PropagateExplicitBoundsTest, OutOfBoundsInfUpper) {
  auto transform = IndexTransformBuilder<2, 3>()
                       .input_origin({2, 2})
                       .input_shape({5, kInfIndex + 1 - 2})
                       .output_single_input_dimension(0, 15, 2, 0)
                       .output_single_input_dimension(1, 30, 3, 1)
                       .output_single_input_dimension(2, 45, 4, 1)
                       .Finalize()
                       .value();
  const Box<3> b({2, 3, 4}, {50, 60, 100});
  Box<2> a;
  EXPECT_THAT(
      PropagateExplicitBounds(b, transform, a),
      MatchesStatus(absl::StatusCode::kOutOfRange,
                    "Propagated bounds \\[-9, 11\\) for dimension 1 are "
                    "incompatible with existing bounds \\[2, \\+inf\\)\\."));
}

TEST(PropagateExplicitBoundsTest, Overflow) {
  auto transform = IndexTransformBuilder<2, 3>()
                       .input_origin({2, -kInfIndex})
                       .input_shape({5, kInfIndex + 10})
                       .output_single_input_dimension(0, 15, 2, 0)
                       .output_single_input_dimension(1, 30, 1, 1)
                       .output_single_input_dimension(2, 45, 4, 1)
                       .Finalize()
                       .value();
  // b is: [2,51] * [kMinFiniteIndex,68] * [4,103]
  const Box<3> b({2, kMinFiniteIndex, 4}, {50, -kMinFiniteIndex + 69, 100});
  Box<2> a;
  EXPECT_THAT(PropagateExplicitBounds(b, transform, a),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Propagating bounds from dimension 1 to input "
                            "dimension 1: Integer overflow propagating .*"));
}

TEST(PropagateExplicitBoundsTest, ZeroSize) {
  auto transform = IndexTransformBuilder<2, 3>()
                       .input_origin({2, 3})
                       .input_shape({5, 0})
                       .output_single_input_dimension(0, 15, 2, 0)
                       .output_single_input_dimension(1, 30, 3, 1)
                       .output_single_input_dimension(2, 45, 4, 1)
                       .Finalize()
                       .value();
  const Box<3> b({2, 3, 4}, {50, 66, 100});
  Box<2> a;
  EXPECT_EQ(absl::OkStatus(), PropagateExplicitBounds(b, transform, a));
  // Check that the propagated bounds match the transform input domain.
  EXPECT_EQ(BoxView({2, 3}, {5, 0}), a);
}

// Tests that an invalid transform is correctly treated as an identity transform
// with the default all-false `b_implicit_{lower,upper}_bounds` vectors.
TEST(PropagateExplicitBoundsToTransformTest,
     InvalidTransformTreatedAsIdentityTransformDefaultImplicit) {
  IndexTransform<2, 2> t;
  Box<2> output_domain({1, 2}, {3, 4});
  auto t_result = PropagateExplicitBoundsToTransform(output_domain, t);
  ASSERT_TRUE(t_result);
  EXPECT_EQ(IndexTransformBuilder<>(2, 2)
                .input_bounds(output_domain)
                .implicit_lower_bounds({0, 0})
                .implicit_upper_bounds({0, 0})
                .output_single_input_dimension(0, 0)
                .output_single_input_dimension(1, 1)
                .Finalize()
                .value(),
            *t_result);
}

// Tests that an invalid transform is correctly treated as an identity transform
// with the specified `b_implicit_{lower,upper}_bounds` vectors.
TEST(PropagateBoundsToTransformTest,
     InvalidTransformTreatedAsIdentityTransformImplicit) {
  IndexTransform<2, 2> t;
  Box<2> output_domain({1, 2}, {3, 4});
  auto t_result = PropagateBoundsToTransform(output_domain, BitVec({1, 0}),
                                             BitVec({0, 1}), t);
  ASSERT_TRUE(t_result);
  EXPECT_EQ(IndexTransformBuilder<>(2, 2)
                .input_bounds(output_domain)
                .implicit_lower_bounds({1, 0})
                .implicit_upper_bounds({0, 1})
                .output_single_input_dimension(0, 0)
                .output_single_input_dimension(1, 1)
                .Finalize()
                .value(),
            *t_result);
}

// Tests that calling PropagateExplicitBoundsToTransform with an index transform
// with an index array output index map with a fully specified index_range just
// returns the transform unmodified.
TEST(PropagateExplicitBoundsToTransformTest, IndexArrayNoPropagationNeeded) {
  Box<1> output_domain({1}, {10});
  auto t = IndexTransformBuilder<1, 1>()
               .input_origin({11})
               .input_shape({3})
               .output_index_array(0, 2, 3, MakeArray<Index>({1, 2, 1}),
                                   IndexInterval::Closed(1, 2))
               .Finalize()
               .value();
  auto t_result = PropagateExplicitBoundsToTransform(output_domain, t);
  ASSERT_TRUE(t_result);
  EXPECT_EQ(t, *t_result);
}

// Tests that calling PropagateExplicitBoundsToTransform with an index transform
// with a single_input_dimension output index map with an output range contained
// within the output_domain just returns the transform unmodified.
TEST(PropagateExplicitBoundsToTransformTest,
     SingleInputDimensionNoPropagationNeeded) {
  Box<1> output_domain({1}, {10});
  auto t = IndexTransformBuilder<1, 1>()
               .input_origin({11})
               .input_shape({3})
               .output_single_input_dimension(0, -32, 3, 0)
               .Finalize()
               .value();
  auto t_result = PropagateExplicitBoundsToTransform(output_domain, t);
  ASSERT_TRUE(t_result);
  EXPECT_EQ(t, *t_result);
}

/// Tests that the index_range for an index array output index map is correctly
/// computed using GetAffineTransformDomain.
TEST(PropagateExplicitBoundsToTransformTest, PropagateToIndexRange) {
  Box<1> output_domain({1}, {10});
  auto t = IndexTransformBuilder<1, 1>()
               .input_origin({11})
               .input_shape({3})
               .output_index_array(0, 2, 3, MakeArray<Index>({1, 2, 1}))
               .Finalize()
               .value();
  auto t_result = PropagateExplicitBoundsToTransform(output_domain, t);
  ASSERT_TRUE(t_result);
  auto t_expected =
      IndexTransformBuilder<>(1, 1)
          .input_origin({11})
          .input_shape({3})
          .output_index_array(0, 2, 3, MakeArray<Index>({1, 2, 1}),
                              IndexInterval::Closed(0, 2))
          .Finalize()
          .value();

  EXPECT_EQ(t_expected, *t_result);
}

/// Tests that implicit bounds do not propagate to the index range of index
/// array output index maps.
TEST(PropagateBoundsToTransformTest, PropagateToIndexRange) {
  Box<1> output_domain({1}, {10});

  const auto get_transform =
      [](tensorstore::Result<IndexInterval> index_range) {
        return IndexTransformBuilder<1, 1>()
            .input_origin({11})
            .input_shape({3})
            .output_index_array(0, 2, 3, MakeArray<Index>({1, 2, 1}),
                                index_range)
            .Finalize()
            .value();
      };
  EXPECT_THAT(  //
      PropagateBoundsToTransform(output_domain, BitVec({0}), BitVec({0}),
                                 get_transform(IndexInterval())),
      ::testing::Optional(get_transform(IndexInterval::Closed(0, 2))));

  EXPECT_THAT(  //
      PropagateBoundsToTransform(output_domain, BitVec({1}), BitVec({0}),
                                 get_transform(IndexInterval())),
      ::testing::Optional(get_transform(IndexInterval::Closed(-kInfIndex, 2))));

  EXPECT_THAT(  //
      PropagateBoundsToTransform(output_domain, BitVec({0}), BitVec({1}),
                                 get_transform(IndexInterval())),
      ::testing::Optional(get_transform(IndexInterval::Closed(0, kInfIndex))));

  EXPECT_THAT(  //
      PropagateBoundsToTransform(output_domain, BitVec({1}), BitVec({1}),
                                 get_transform(IndexInterval())),
      ::testing::Optional(get_transform(IndexInterval())));
}

TEST(PropagateBoundsToTransformTest, PropagateToInputDomain) {
  Box<1> output_domain({1}, {10});
  auto t = IndexTransformBuilder<1, 1>()
               .implicit_lower_bounds({1})
               .implicit_upper_bounds({1})
               .output_single_input_dimension(0, -32, 3, 0)
               .Finalize()
               .value();
  auto t_result =
      PropagateBoundsToTransform(output_domain, BitVec({1}), BitVec({0}), t);
  ASSERT_TRUE(t_result);
  // Since `t` has an implicit input domain, the new input domain is computed
  // from the `output_domain` using GetAffineTransformDomain:
  //   {x : 1 <= (x * 3 - 32) <= 10} = {11, 12, 13, 14}
  auto t_expected = IndexTransformBuilder<1, 1>()
                        .input_origin({11})
                        .input_shape({4})
                        .implicit_lower_bounds({1})
                        .implicit_upper_bounds({0})
                        .output_single_input_dimension(0, -32, 3, 0)
                        .Finalize()
                        .value();
  EXPECT_EQ(t_expected, *t_result);
}

TEST(PropagateExplicitBoundsToTransformTest, OutOfBounds) {
  auto t = IndexTransformBuilder<2, 3>()
               .input_origin({2, 3})
               .input_shape({5, 10})
               .output_single_input_dimension(0, 15, 2, 0)
               .output_single_input_dimension(1, 30, 3, 1)
               .output_single_input_dimension(2, 45, 4, 1)
               .Finalize()
               .value();
  const Box<3> output_domain({2, 3, 4}, {50, 60, 100});
  // Output domain is [2, 51] * [3, 62] * [4, 103].
  // For output dimension 0, {x : 2 <= (x * 2 + 15) <= 51} = [-6, 18].
  // For output dimension 1, {x : 3 <= (x * 3 + 30) <= 62} = [-9, 10].
  // For output dimension 2, {x : 4 <= (x * 4 + 45) <= 103} = [-10, 14].
  EXPECT_THAT(
      PropagateExplicitBoundsToTransform(output_domain, t),
      MatchesStatus(absl::StatusCode::kOutOfRange,
                    "Propagated bounds \\[-9, 11\\) for dimension 1 are "
                    "incompatible with existing bounds \\[3, 13\\)\\."));
}

TEST(PropagateExplicitBoundsToTransformTest, Overflow) {
  auto t = IndexTransformBuilder<2, 3>()
               .input_origin({2, -kInfIndex})
               .input_shape({5, kInfIndex + 10})
               .output_single_input_dimension(0, 15, 2, 0)
               .output_single_input_dimension(1, 30, 1, 1)
               .output_single_input_dimension(2, 45, 4, 1)
               .Finalize()
               .value();
  // output_domain is: [2,51] * [kMinFiniteIndex,68] * [4,103]
  const Box<3> output_domain({2, kMinFiniteIndex, 4},
                             {50, -kMinFiniteIndex + 69, 100});
  EXPECT_THAT(PropagateExplicitBoundsToTransform(output_domain, t),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Propagating bounds from dimension 1 to input "
                            "dimension 1: Integer overflow propagating .*"));
}

}  // namespace
