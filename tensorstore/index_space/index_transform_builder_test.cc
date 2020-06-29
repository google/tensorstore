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

#include "tensorstore/index_space/index_transform_builder.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/array.h"
#include "tensorstore/box.h"
#include "tensorstore/index_space/index_domain_builder.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using tensorstore::Index;
using tensorstore::IndexDomainBuilder;
using tensorstore::IndexInterval;
using tensorstore::IndexTransform;
using tensorstore::IndexTransformBuilder;
using tensorstore::kInfIndex;
using tensorstore::kInfSize;
using tensorstore::MakeArray;
using tensorstore::MatchesStatus;
using tensorstore::OutputIndexMethod;
using tensorstore::span;
using tensorstore::Status;
using tensorstore::internal_index_space::TransformAccess;

TEST(IndexTransformTest, BuilderValid) {
  auto index_array = MakeArray<Index>({{{1, 0, 2, 2}}});
  auto t =
      IndexTransformBuilder<3, 4>()
          .input_origin({1, 2, 3})
          .input_shape({2, 2, 4})
          .implicit_lower_bounds({0, 1, 0})
          .implicit_upper_bounds({1, 0, 0})
          .input_labels({"x", "y", "z"})
          .output_constant(0, 4)
          .output_single_input_dimension(1, 5, 7, 2)
          .output_constant(2, 6)
          .output_index_array(3, 7, 9, index_array, IndexInterval::Closed(0, 3))
          .Finalize()
          .value();
  static_assert(std::is_same<decltype(t), IndexTransform<3, 4>>::value, "");
  EXPECT_THAT(t.input_origin(), ::testing::ElementsAre(1, 2, 3));
  EXPECT_THAT(t.input_shape(), ::testing::ElementsAre(2, 2, 4));
  EXPECT_THAT(t.input_labels(), ::testing::ElementsAre("x", "y", "z"));
  EXPECT_THAT(t.implicit_lower_bounds(), ::testing::ElementsAre(0, 1, 0));
  EXPECT_THAT(t.implicit_upper_bounds(), ::testing::ElementsAre(1, 0, 0));
  EXPECT_EQ(IndexInterval::UncheckedSized(1, 2),
            t.input_domain()[0].interval());
  EXPECT_EQ(IndexInterval::UncheckedSized(2, 2),
            t.input_domain()[1].interval());
  EXPECT_EQ(IndexInterval::UncheckedSized(3, 4),
            t.input_domain()[2].interval());
  {
    auto map = t.output_index_map(0);
    EXPECT_EQ(OutputIndexMethod::constant, map.method());
    EXPECT_EQ(4, map.offset());
    EXPECT_EQ(0, map.stride());
  }
  {
    auto map = t.output_index_map(1);
    EXPECT_EQ(OutputIndexMethod::single_input_dimension, map.method());
    EXPECT_EQ(2, map.input_dimension());
    EXPECT_EQ(5, map.offset());
    EXPECT_EQ(7, map.stride());
  }
  {
    auto map = t.output_index_map(2);
    EXPECT_EQ(OutputIndexMethod::constant, map.method());
    EXPECT_EQ(6, map.offset());
    EXPECT_EQ(0, map.stride());
  }
  {
    auto map = t.output_index_map(3);
    EXPECT_EQ(OutputIndexMethod::array, map.method());
    EXPECT_EQ(7, map.offset());
    EXPECT_EQ(9, map.stride());
    auto index_array_ref = map.index_array();
    EXPECT_EQ(&index_array(0, 0, 0), &index_array_ref.array_ref()(1, 2, 3));
    EXPECT_THAT(index_array_ref.layout().byte_strides(),
                ::testing::ElementsAre(0, 0, sizeof(Index)));
  }
  {
    std::array<Index, 4> output_indices;
    ASSERT_EQ(Status(), t.TransformIndices(span<const Index, 3>({1, 2, 3}),
                                           output_indices));
    EXPECT_THAT(output_indices, ::testing::ElementsAre(4, 26, 6, 16));
  }
}

TEST(IndexTransformBuilderTest, Nullptr) {
  IndexTransformBuilder<> builder(nullptr);
  EXPECT_FALSE(builder.valid());

  {
    IndexTransformBuilder<> other_builder(builder);
    EXPECT_FALSE(other_builder.valid());
  }

  {
    IndexTransformBuilder<> other_builder(nullptr);
    other_builder = builder;
    EXPECT_FALSE(other_builder.valid());
  }
}

TEST(IndexTransformBuilderTest, Move) {
  IndexTransformBuilder<> builder(1, 1);
  EXPECT_TRUE(builder.valid());

  builder.input_origin({1});

  auto builder2 = std::move(builder);

  EXPECT_TRUE(builder2.valid());

  // Check that moved-from value is not valid.
  EXPECT_FALSE(builder.valid());  // NOLINT

  builder2.output_constant(0, 5);

  EXPECT_THAT(builder2.Finalize().value(), IndexTransformBuilder<>(1, 1)
                                               .input_origin({1})
                                               .output_constant(0, 5)
                                               .Finalize()
                                               .value());
}

TEST(IndexTransformBuilderTest, Copy) {
  IndexTransformBuilder<> builder(1, 1);
  EXPECT_TRUE(builder.valid());

  builder.input_origin({1});

  auto builder2 = builder;

  EXPECT_TRUE(builder.valid());
  EXPECT_TRUE(builder2.valid());

  builder.output_constant(0, 4);
  builder2.output_constant(0, 5);
  EXPECT_THAT(builder.Finalize().value(), IndexTransformBuilder<>(1, 1)
                                              .input_origin({1})
                                              .output_constant(0, 4)
                                              .Finalize()
                                              .value());

  EXPECT_THAT(builder2.Finalize().value(), IndexTransformBuilder<>(1, 1)
                                               .input_origin({1})
                                               .output_constant(0, 5)
                                               .Finalize()
                                               .value());
}

TEST(IndexTransformBuilderTest, Default) {
  auto t = IndexTransformBuilder<>(2, 1).Finalize().value();
  EXPECT_THAT(t.input_origin(), ::testing::ElementsAre(-kInfIndex, -kInfIndex));
  EXPECT_THAT(t.input_shape(), ::testing::ElementsAre(kInfSize, kInfSize));
  EXPECT_THAT(t.implicit_lower_bounds(), ::testing::ElementsAre(1, 1));
  EXPECT_THAT(t.implicit_upper_bounds(), ::testing::ElementsAre(1, 1));
  EXPECT_THAT(t.input_labels(), ::testing::ElementsAre("", ""));
  auto map = t.output_index_map(0);
  EXPECT_EQ(0, map.offset());
  EXPECT_EQ(0, map.stride());
  EXPECT_EQ(OutputIndexMethod::constant, map.method());
}

// Tests that lower bounds default to "explicit" if `input_origin` is called.
TEST(IndexTransformBuilderTest, InputOriginSpecified) {
  auto t =
      IndexTransformBuilder<>(2, 0).input_origin({1, 2}).Finalize().value();
  EXPECT_EQ(t.domain()[0].interval(),
            IndexInterval::UncheckedClosed(1, kInfIndex));
  EXPECT_EQ(t.domain()[1].interval(),
            IndexInterval::UncheckedClosed(2, kInfIndex));
  EXPECT_THAT(t.implicit_lower_bounds(), ::testing::ElementsAre(0, 0));
  EXPECT_THAT(t.implicit_upper_bounds(), ::testing::ElementsAre(1, 1));
  EXPECT_THAT(t.input_labels(), ::testing::ElementsAre("", ""));
}

// Tests that calling `implicit_lower_bounds` overrides the default value even
// if `input_origin` isn't called.
TEST(IndexTransformBuilderTest, ImplicitLowerBoundsSpecified) {
  auto t = IndexTransformBuilder<>(2, 0)
               .implicit_lower_bounds({1, 0})
               .Finalize()
               .value();
  EXPECT_EQ(t.domain()[0].interval(),
            IndexInterval::UncheckedClosed(-kInfIndex, kInfIndex));
  EXPECT_EQ(t.domain()[1].interval(),
            IndexInterval::UncheckedClosed(-kInfIndex, kInfIndex));
  EXPECT_THAT(t.implicit_lower_bounds(), ::testing::ElementsAre(1, 0));
  EXPECT_THAT(t.implicit_upper_bounds(), ::testing::ElementsAre(1, 1));
  EXPECT_THAT(t.input_labels(), ::testing::ElementsAre("", ""));
}

// Tests that upper bounds default to "explicit" if `input_shape` is called.
TEST(IndexTransformBuilderTest, InputShapeSpecified) {
  auto t =
      IndexTransformBuilder<>(2, 0).input_shape({5, 10}).Finalize().value();
  EXPECT_EQ(t.domain()[0].interval(), IndexInterval::UncheckedSized(0, 5));
  EXPECT_EQ(t.domain()[1].interval(), IndexInterval::UncheckedSized(0, 10));
  EXPECT_THAT(t.implicit_lower_bounds(), ::testing::ElementsAre(0, 0));
  EXPECT_THAT(t.implicit_upper_bounds(), ::testing::ElementsAre(0, 0));
  EXPECT_THAT(t.input_labels(), ::testing::ElementsAre("", ""));
}

// Tests that upper bounds default to "explicit" if `input_inclusive_max` is
// called.
TEST(IndexTransformBuilderTest, InputInclusiveMaxSpecified) {
  auto t = IndexTransformBuilder<>(2, 0)
               .input_inclusive_max({5, 10})
               .Finalize()
               .value();
  EXPECT_EQ(t.domain()[0].interval(),
            IndexInterval::UncheckedClosed(-kInfIndex, 5));
  EXPECT_EQ(t.domain()[1].interval(),
            IndexInterval::UncheckedClosed(-kInfIndex, 10));
  EXPECT_THAT(t.implicit_lower_bounds(), ::testing::ElementsAre(1, 1));
  EXPECT_THAT(t.implicit_upper_bounds(), ::testing::ElementsAre(0, 0));
  EXPECT_THAT(t.input_labels(), ::testing::ElementsAre("", ""));
}

// Tests that upper bounds default to "explicit" if `input_exclusive_max` is
// called.
TEST(IndexTransformBuilderTest, InputExclusiveMaxSpecified) {
  auto t = IndexTransformBuilder<>(2, 0)
               .input_exclusive_max({5, 10})
               .Finalize()
               .value();
  EXPECT_EQ(t.domain()[0].interval(),
            IndexInterval::UncheckedHalfOpen(-kInfIndex, 5));
  EXPECT_EQ(t.domain()[1].interval(),
            IndexInterval::UncheckedHalfOpen(-kInfIndex, 10));
  EXPECT_THAT(t.implicit_lower_bounds(), ::testing::ElementsAre(1, 1));
  EXPECT_THAT(t.implicit_upper_bounds(), ::testing::ElementsAre(0, 0));
  EXPECT_THAT(t.input_labels(), ::testing::ElementsAre("", ""));
}

// Tests that calling `implicit_upper_bounds` overrides the default value even
// if `input_shape`, `input_inclusive_max`, and `input_exclusive_max` aren't
// called.
TEST(IndexTransformBuilderTest, ImplicitUpperBoundsSpecified) {
  auto t = IndexTransformBuilder<>(2, 0)
               .implicit_upper_bounds({1, 0})
               .Finalize()
               .value();
  EXPECT_EQ(t.domain()[0].interval(),
            IndexInterval::UncheckedClosed(-kInfIndex, kInfIndex));
  EXPECT_EQ(t.domain()[1].interval(),
            IndexInterval::UncheckedClosed(-kInfIndex, kInfIndex));
  EXPECT_THAT(t.implicit_lower_bounds(), ::testing::ElementsAre(1, 1));
  EXPECT_THAT(t.implicit_upper_bounds(), ::testing::ElementsAre(1, 0));
  EXPECT_THAT(t.input_labels(), ::testing::ElementsAre("", ""));
}

TEST(IndexTransformBuilderTest, SingleInputDimensionDefaults) {
  EXPECT_EQ(IndexTransformBuilder<>(3, 1)
                .output_single_input_dimension(0, 2)
                .Finalize()
                .value(),
            IndexTransformBuilder<>(3, 1)
                .output_single_input_dimension(0, 0, 1, 2)
                .Finalize()
                .value());
}

TEST(IndexTransformBuilderTest, ErrorHandling) {
  // input_origin out of range
  EXPECT_THAT(IndexTransformBuilder<>(2, 1)
                  .input_origin({-kInfIndex - 1, -kInfIndex})
                  .Finalize(),
              MatchesStatus(absl::StatusCode::kInvalidArgument));

  // input_shape out of range
  EXPECT_THAT(IndexTransformBuilder<>(2, 1).input_shape({1, -1}).Finalize(),
              MatchesStatus(absl::StatusCode::kInvalidArgument));

  // invalid input dimension
  EXPECT_THAT(IndexTransformBuilder<>(2, 1)
                  .output_single_input_dimension(0, 0, 1, -1)
                  .Finalize(),
              MatchesStatus(absl::StatusCode::kInvalidArgument));

  // invalid input dimension
  EXPECT_THAT(IndexTransformBuilder<>(2, 1)
                  .output_single_input_dimension(0, 0, 1, 2)
                  .Finalize(),
              MatchesStatus(absl::StatusCode::kInvalidArgument));

  // invalid index array rank
  EXPECT_THAT(IndexTransformBuilder<>(2, 1)
                  .output_index_array(0, 0, 1, MakeArray<Index>({1}))
                  .Finalize(),
              MatchesStatus(absl::StatusCode::kInvalidArgument));

  // invalid index array shape
  EXPECT_THAT(IndexTransformBuilder<>(2, 1)
                  .input_shape({2, 2})
                  .output_index_array(
                      0, 0, 1, MakeArray<Index>({{1, 2}, {3, 4}, {5, 6}}))
                  .Finalize(),
              MatchesStatus(absl::StatusCode::kInvalidArgument));

  // index array depends on input dimension with implicit lower bound
  EXPECT_THAT(IndexTransformBuilder<>(2, 1)
                  .input_shape({3, 2})
                  .implicit_lower_bounds({1, 0})
                  .output_index_array(
                      0, 0, 1, MakeArray<Index>({{1, 2}, {3, 4}, {5, 6}}))
                  .Finalize(),
              MatchesStatus(absl::StatusCode::kInvalidArgument));

  // index array depends on input dimension with implicit upper bound
  EXPECT_THAT(IndexTransformBuilder<>(2, 1)
                  .input_shape({3, 2})
                  .implicit_upper_bounds({1, 0})
                  .output_index_array(
                      0, 0, 1, MakeArray<Index>({{1, 2}, {3, 4}, {5, 6}}))
                  .Finalize(),
              MatchesStatus(absl::StatusCode::kInvalidArgument));

  // invalid index array bounds
  EXPECT_THAT(
      IndexTransformBuilder<>(2, 1)
          .input_shape({2, 2})
          .output_index_array(0, 0, 1, MakeArray<Index>({{1, 2}, {3, 4}}),
                              IndexInterval::Sized(3, -1))
          .Finalize(),
      MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST(IndexTransformBuilderDeathTest, InvalidArguments) {
  EXPECT_DEATH((IndexTransformBuilder<>(2, 1).input_origin({1, 2, 3})),
               "range size mismatch");

  EXPECT_DEATH((IndexTransformBuilder<>(2, 1).input_shape({1, 2, 3})),
               "range size mismatch");

  EXPECT_DEATH((IndexTransformBuilder<>(2, 1).implicit_lower_bounds({1, 1, 0})),
               "range size mismatch");

  EXPECT_DEATH((IndexTransformBuilder<>(2, 1).implicit_upper_bounds({1, 1, 0})),
               "range size mismatch");

  EXPECT_DEATH((IndexTransformBuilder<>(2, 1).input_labels({"a"})),
               "range size mismatch");

  EXPECT_DEATH((IndexTransformBuilder<>(2, 1).output_constant(1, 0)),
               "invalid output dimension");
}

// Test the special allocation-free code path for rank zero.
TEST(IndexTransformBuilderTest, RankZero) {
  auto transform = IndexTransformBuilder<>(0, 0).Finalize().value();
  EXPECT_TRUE(transform.input_origin().empty());
  EXPECT_TRUE(transform.input_shape().empty());
  EXPECT_TRUE(transform.input_labels().empty());
  EXPECT_TRUE(transform.output_index_maps().empty());

  auto transform2 = IndexTransformBuilder<>(0, 0).Finalize().value();
  EXPECT_EQ(TransformAccess::rep(transform), TransformAccess::rep(transform2));
}

TEST(IndexTransformBuilderTest, OutputStrideZero) {
  // Setting an output stride of zero should make the output index map
  // constant.
  auto t = IndexTransformBuilder<>(1, 1)
               .output_single_input_dimension(0, 1, 0, 0)
               .Finalize()
               .value();
  auto map = t.output_index_map(0);
  EXPECT_EQ(1, map.offset());
  EXPECT_EQ(0, map.stride());
  EXPECT_EQ(OutputIndexMethod::constant, map.method());
}

// Tests that the input domain upper bound can be set using the
// `input_inclusive_max` method.
TEST(IndexTransformBuilderTest, InclusiveMax) {
  auto t = IndexTransformBuilder<>(2, 2)
               .input_origin({1, 2})
               .input_inclusive_max({3, 5})
               .Finalize()
               .value();
  EXPECT_THAT(t.input_origin(), ::testing::ElementsAre(1, 2));
  EXPECT_THAT(t.input_shape(), ::testing::ElementsAre(3, 4));
}

// Tests that the `input_shape` setter handles `kInfSize`.
TEST(IndexTransformBuilderTest, InputShapeInfSize) {
  auto t = IndexTransformBuilder<>(2, 2)
               .input_origin({1, 2})
               .input_shape({3, kInfSize})
               .Finalize()
               .value();
  EXPECT_THAT(t.input_origin(), ::testing::ElementsAre(1, 2));
  EXPECT_THAT(t.input_shape(), ::testing::ElementsAre(3, kInfIndex + 1 - 2));
}

// Tests that the input domain upper bound can be set using the
// `input_exclusive_max` method.
TEST(IndexTransformBuilderTest, ExclusiveMax) {
  auto t = IndexTransformBuilder<>(2, 2)
               .input_origin({1, 2})
               .input_exclusive_max({3, 5})
               .Finalize()
               .value();
  EXPECT_THAT(t.input_origin(), ::testing::ElementsAre(1, 2));
  EXPECT_THAT(t.input_shape(), ::testing::ElementsAre(2, 3));
}

// Tests that the input domain upper bound can be set using the
// `input_exclusive_max` method to override a previously set `input_shape`.
TEST(IndexTransformBuilderTest, ExclusiveMaxAfterShape) {
  auto t = IndexTransformBuilder<>(2, 2)
               .input_origin({1, 2})
               .input_shape({15, 16})
               .input_exclusive_max({3, 5})
               .Finalize()
               .value();
  EXPECT_THAT(t.input_origin(), ::testing::ElementsAre(1, 2));
  EXPECT_THAT(t.input_shape(), ::testing::ElementsAre(2, 3));
}

// Tests that the input domain bounds can be set from a box using the
// `input_bounds` method.
TEST(IndexTransformBuilderTest, InputDomainBox) {
  auto t = IndexTransformBuilder<>(2, 2)
               .input_bounds(tensorstore::BoxView({1, 2}, {2, 3}))
               .Finalize()
               .value();
  EXPECT_THAT(t.input_origin(), ::testing::ElementsAre(1, 2));
  EXPECT_THAT(t.input_shape(), ::testing::ElementsAre(2, 3));
}

// Tests that the input domain can be set from an IndexDomain using the
// `input_domain` method.
TEST(IndexTransformBuilderTest, InputDomain) {
  tensorstore::IndexDomain<2> domain(IndexTransformBuilder<2, 0>()
                                         .input_origin({1, 2})
                                         .input_shape({3, 4})
                                         .implicit_lower_bounds({0, 1})
                                         .implicit_upper_bounds({1, 0})
                                         .input_labels({"x", "y"})
                                         .Finalize()
                                         .value());

  auto t =
      IndexTransformBuilder<>(2, 2).input_domain(domain).Finalize().value();
  EXPECT_EQ(domain, t.domain());
}

TEST(InitializeTransformRepForBuilder, Basic) {
  auto source = tensorstore::internal_index_space::TransformRep::Allocate(1, 2);
  source->output_rank = 2;
  tensorstore::internal_index_space::InitializeTransformRepForBuilder(
      source.get());
  EXPECT_EQ(0, source->output_index_maps()[0].offset());
  EXPECT_EQ(0, source->output_index_maps()[0].stride());
  EXPECT_EQ(0, source->output_index_maps()[1].offset());
  EXPECT_EQ(0, source->output_index_maps()[1].stride());
}

TEST(IndexTransformBuilder, NonUniqueLabels) {
  EXPECT_THAT(
      IndexTransformBuilder<>(3, 0).input_labels({"a", "", "a"}).Finalize(),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Dimension label\\(s\\) \"a\" not unique"));
}

TEST(IndexDomainBuilderTest, Null) {
  IndexDomainBuilder builder(nullptr);
  EXPECT_FALSE(builder.valid());
}

TEST(IndexDomainBuilderTest, Basic) {
  IndexDomainBuilder builder(3);
  EXPECT_EQ(3, builder.rank());
  builder.origin(span<const Index, 3>({1, 2, 3}));
  EXPECT_THAT(builder.origin(), ::testing::ElementsAre(1, 2, 3));
  builder.shape(span<const Index, 3>({4, 5, 6}));
  EXPECT_THAT(builder.shape(), ::testing::ElementsAre(4, 5, 6));
  builder.exclusive_max(span<const Index, 3>({4, 5, 6}));
  EXPECT_THAT(builder.exclusive_max(), ::testing::ElementsAre(4, 5, 6));
  builder.inclusive_max(span<const Index, 3>({4, 5, 6}));
  EXPECT_THAT(builder.inclusive_max(), ::testing::ElementsAre(4, 5, 6));
  builder.implicit_lower_bounds(span<const bool, 3>({0, 1, 1}));
  builder.implicit_upper_bounds(span<const bool, 3>({1, 0, 1}));
  EXPECT_THAT(builder.implicit_lower_bounds(), ::testing::ElementsAre(0, 1, 1));
  EXPECT_THAT(builder.implicit_upper_bounds(), ::testing::ElementsAre(1, 0, 1));
  builder.labels(std::vector<std::string>{"x", "y", "z"});
  EXPECT_THAT(builder.labels(), ::testing::ElementsAre("x", "y", "z"));
}

// Tests that the labels can be set using the `labels` method.
TEST(IndexDomainBuilderTest, Labels) {
  auto d = IndexDomainBuilder(2).labels({"x", "y"}).Finalize().value();
  EXPECT_THAT(d.labels(), ::testing::ElementsAre("x", "y"));
}

// Tests that the upper bound can be set using the `inclusive_max` method.
TEST(IndexDomainBuilderTest, InclusiveMax) {
  auto d = IndexDomainBuilder(2)
               .origin({1, 2})
               .inclusive_max({3, 5})
               .Finalize()
               .value();
  EXPECT_THAT(d.origin(), ::testing::ElementsAre(1, 2));
  EXPECT_THAT(d.shape(), ::testing::ElementsAre(3, 4));
}

// Tests that the upper bound can be set using the `shape` method.
TEST(IndexDomainBuilderTest, Shape) {
  auto d =
      IndexDomainBuilder(2).origin({1, 2}).shape({3, 5}).Finalize().value();
  EXPECT_THAT(d.origin(), ::testing::ElementsAre(1, 2));
  EXPECT_THAT(d.shape(), ::testing::ElementsAre(3, 5));
}

// Tests that the upper bound can be set using the `exclusive_max` method.
TEST(IndexDomainBuilderTest, ExclusiveMax) {
  auto d = IndexDomainBuilder(2)
               .origin({1, 2})
               .exclusive_max({3, 5})
               .Finalize()
               .value();
  EXPECT_THAT(d.origin(), ::testing::ElementsAre(1, 2));
  EXPECT_THAT(d.shape(), ::testing::ElementsAre(2, 3));
}

// Tests that the bounds can be set from a box using the `bounds` method.
TEST(IndexDomainBuilderTest, InputDomainBox) {
  auto d = IndexDomainBuilder(2)
               .bounds(tensorstore::BoxView({1, 2}, {2, 3}))
               .Finalize()
               .value();
  EXPECT_THAT(d.origin(), ::testing::ElementsAre(1, 2));
  EXPECT_THAT(d.shape(), ::testing::ElementsAre(2, 3));
}

// Tests that the domain can be set from an existing IndexDomain using the
// `domain` method.
TEST(IndexDomainBuilderTest, InputDomain) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(tensorstore::IndexDomain<2> domain,
                                   IndexDomainBuilder<2>()
                                       .origin({1, 2})
                                       .shape({3, 4})
                                       .implicit_lower_bounds({0, 1})
                                       .implicit_upper_bounds({1, 0})
                                       .labels({"x", "y"})
                                       .Finalize());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto d, IndexDomainBuilder<>(2).domain(domain).Finalize());
  EXPECT_EQ(domain, d);
}

}  // namespace
