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

#include "tensorstore/driver/downsample/downsample_util.h"

#include <random>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/random/bit_gen_ref.h"
#include "absl/random/random.h"
#include "tensorstore/driver/downsample/downsample_array.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/index_space/index_transform_testutil.h"
#include "tensorstore/index_space/transformed_array.h"
#include "tensorstore/internal/data_type_random_generator.h"
#include "tensorstore/internal/test_util.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::Box;
using ::tensorstore::BoxView;
using ::tensorstore::DimensionIndex;
using ::tensorstore::Dims;
using ::tensorstore::DownsampleMethod;
using ::tensorstore::Index;
using ::tensorstore::IndexInterval;
using ::tensorstore::IndexTransformBuilder;
using ::tensorstore::kInfIndex;
using ::tensorstore::MakeArray;
using ::tensorstore::MatchesStatus;
using ::tensorstore::span;
using ::tensorstore::internal_downsample::CanDownsampleIndexTransform;
using ::tensorstore::internal_downsample::DownsampleArray;
using ::tensorstore::internal_downsample::DownsampleBounds;
using ::tensorstore::internal_downsample::DownsampleInterval;
using ::tensorstore::internal_downsample::DownsampleTransformedArray;
using ::tensorstore::internal_downsample::PropagatedIndexTransformDownsampling;
using ::tensorstore::internal_downsample::PropagateIndexTransformDownsampling;
using ::testing::Optional;

TEST(PropagateIndexTransformDownsamplingTest, Rank0) {
  EXPECT_THAT(PropagateIndexTransformDownsampling(
                  tensorstore::IdentityTransform(0), {}, {}),
              Optional(PropagatedIndexTransformDownsampling{
                  tensorstore::IdentityTransform(0), {}}));
}

TEST(PropagateIndexTransformDownsamplingTest, Rank1SingleInputDimension) {
  EXPECT_THAT(PropagateIndexTransformDownsampling(
                  tensorstore::IdentityTransform(BoxView({1}, {3})),
                  BoxView<1>({7}), span<const Index>({2})),
              Optional(PropagatedIndexTransformDownsampling{
                  tensorstore::IdentityTransform(BoxView({2}, {5})), {2}}));
}

TEST(PropagateIndexTransformDownsamplingTest, InvalidRank) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto downsampled_transform,
      tensorstore::IdentityTransform(32) | Dims(0).Stride(2));
  EXPECT_THAT(PropagateIndexTransformDownsampling(
                  downsampled_transform, Box(32), std::vector<Index>(32, 2)),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Rank 33 is outside valid range \\[0, 32\\]"));
}

TEST(PropagateIndexTransformDownsamplingTest, Rank1Constant) {
  EXPECT_THAT(
      PropagateIndexTransformDownsampling(
          IndexTransformBuilder(0, 1).output_constant(0, 2).Finalize().value(),
          BoxView({7}, {2}), span<const Index>({3})),
      Optional(PropagatedIndexTransformDownsampling{
          IndexTransformBuilder(1, 1)
              .input_origin({1})
              .input_exclusive_max({3})
              .output_single_input_dimension(0, 6, 1, 0)
              .Finalize()
              .value(),
          {3}}));
}

TEST(PropagateIndexTransformDownsamplingTest,
     Rank1SingleInputDimensionPartialStartBlock) {
  EXPECT_THAT(PropagateIndexTransformDownsampling(
                  tensorstore::IdentityTransform(BoxView({0}, {4})),
                  BoxView({1}, {6}), span<const Index>({2})),
              Optional(PropagatedIndexTransformDownsampling{
                  tensorstore::IdentityTransform(BoxView({1}, {6})), {2}}));
}

TEST(PropagateIndexTransformDownsamplingTest, Rank2WithIgnoredDimension) {
  EXPECT_THAT(
      PropagateIndexTransformDownsampling(
          tensorstore::IdentityTransform(BoxView({1, 2}, {3, 5})),
          BoxView({7, 10}), span<const Index>({2, 1})),
      Optional(PropagatedIndexTransformDownsampling{
          tensorstore::IdentityTransform(BoxView({2, 2}, {5, 5})), {2, 1}}));
}

TEST(PropagateIndexTransformDownsamplingTest, Rank1IndexArray) {
  EXPECT_THAT(PropagateIndexTransformDownsampling(
                  IndexTransformBuilder(1, 1)
                      .input_shape({3})
                      .output_index_array(0, 0, 1, MakeArray<Index>({4, 7, 3}))
                      .Finalize()
                      .value(),
                  BoxView<1>({50}), span<const Index>({4})),
              Optional(PropagatedIndexTransformDownsampling{
                  IndexTransformBuilder(2, 1)
                      .input_shape({3, 4})
                      .output_index_array(0, 0, 1,
                                          MakeArray<Index>({{16, 17, 18, 19},
                                                            {28, 29, 30, 31},
                                                            {12, 13, 14, 15}}),
                                          IndexInterval::Sized(0, 50))
                      .Finalize()
                      .value(),
                  {1, 4}}));
}

TEST(PropagateIndexTransformDownsamplingTest,
     Rank3IndexArrayConstantNoDownsampling) {
  EXPECT_THAT(
      PropagateIndexTransformDownsampling(
          IndexTransformBuilder(2, 3)
              .input_shape({3, 4})
              .output_index_array(0, 0, 1, MakeArray<Index>({{4}, {7}, {3}}))
              .output_single_input_dimension(1, 1)
              .output_constant(2, 42)
              .Finalize()
              .value(),
          BoxView({30, 50, 55}), span<const Index>({1, 2, 1})),
      Optional(PropagatedIndexTransformDownsampling{
          IndexTransformBuilder(2, 3)
              .input_shape({3, 8})
              .output_index_array(0, 0, 1, MakeArray<Index>({{4}, {7}, {3}}))
              .output_single_input_dimension(1, 1)
              .output_constant(2, 42)
              .Finalize()
              .value(),
          {1, 2}}));
}

TEST(PropagateIndexTransformDownsamplingTest, Rank2IndexArray) {
  EXPECT_THAT(
      PropagateIndexTransformDownsampling(
          IndexTransformBuilder(2, 1)
              .input_shape({2, 3})
              .output_index_array(0, 0, 1,
                                  MakeArray<Index>({{1, 2, 3}, {4, 5, 6}}))
              .Finalize()
              .value(),
          BoxView<1>({50}), span<const Index>({4})),
      Optional(PropagatedIndexTransformDownsampling{
          IndexTransformBuilder(3, 1)
              .input_shape({2, 3, 4})
              .output_index_array(
                  0, 0, 1,
                  MakeArray<Index>(
                      {{{4, 5, 6, 7}, {8, 9, 10, 11}, {12, 13, 14, 15}},
                       {{16, 17, 18, 19}, {20, 21, 22, 23}, {24, 25, 26, 27}}}),
                  IndexInterval::Sized(0, 50))
              .Finalize()
              .value(),
          {1, 1, 4}}));
}

TEST(PropagateIndexTransformDownsamplingTest,
     Rank1SingleInputDimensionStrided) {
  EXPECT_THAT(PropagateIndexTransformDownsampling(
                  IndexTransformBuilder(1, 1)
                      .input_shape({3})
                      .output_single_input_dimension(0, 1, 5, 0)
                      .Finalize()
                      .value(),
                  BoxView<1>({50}), span<const Index>({4})),
              Optional(PropagatedIndexTransformDownsampling{
                  IndexTransformBuilder(2, 1)
                      .input_shape({3, 4})
                      .output_index_array(0, 0, 1,
                                          MakeArray<Index>({{4, 5, 6, 7},
                                                            {24, 25, 26, 27},
                                                            {44, 45, 46, 47}}),
                                          IndexInterval::Sized(0, 50))
                      .Finalize()
                      .value(),
                  {1, 4}}));
}

TEST(PropagateIndexTransformDownsamplingTest, ErrorRank1ConstantOverflow) {
  EXPECT_THAT(
      PropagateIndexTransformDownsampling(
          IndexTransformBuilder(0, 1)
              .output_constant(0, tensorstore::kMaxFiniteIndex)
              .Finalize()
              .value(),
          BoxView<1>({0}, {kInfIndex}), span<const Index>({1000})),
      MatchesStatus(absl::StatusCode::kOutOfRange, ".*Integer overflow.*"));
}

TEST(PropagateIndexTransformDownsamplingTest, ErrorRank1ConstantOutOfBounds) {
  TENSORSTORE_EXPECT_OK(PropagateIndexTransformDownsampling(
      IndexTransformBuilder(0, 1).output_constant(0, 4).Finalize().value(),
      BoxView<1>({0}, {15}), span<const Index>({3})));
  TENSORSTORE_EXPECT_OK(PropagateIndexTransformDownsampling(
      IndexTransformBuilder(0, 1).output_constant(0, 4).Finalize().value(),
      BoxView<1>({0}, {14}), span<const Index>({3})));
  TENSORSTORE_EXPECT_OK(PropagateIndexTransformDownsampling(
      IndexTransformBuilder(0, 1).output_constant(0, 4).Finalize().value(),
      BoxView<1>({0}, {13}), span<const Index>({3})));
  TENSORSTORE_EXPECT_OK(PropagateIndexTransformDownsampling(
      IndexTransformBuilder(0, 1).output_constant(0, 0).Finalize().value(),
      BoxView<1>({1}, {13}), span<const Index>({3})));
  TENSORSTORE_EXPECT_OK(PropagateIndexTransformDownsampling(
      IndexTransformBuilder(0, 1).output_constant(0, 0).Finalize().value(),
      BoxView<1>({2}, {13}), span<const Index>({3})));
  EXPECT_THAT(
      PropagateIndexTransformDownsampling(
          IndexTransformBuilder(0, 1).output_constant(0, 5).Finalize().value(),
          BoxView<1>({0}, {15}), span<const Index>({3})),
      MatchesStatus(absl::StatusCode::kOutOfRange,
                    ".*Propagated bounds interval .* does not contain .*"));
  EXPECT_THAT(
      PropagateIndexTransformDownsampling(
          IndexTransformBuilder(0, 1).output_constant(0, 0).Finalize().value(),
          BoxView<1>({3}, {15}), span<const Index>({3})),
      MatchesStatus(absl::StatusCode::kOutOfRange,
                    ".*Propagated bounds interval .* does not contain .*"));
}

TEST(PropagateIndexTransformDownsamplingTest,
     ErrorSingleInputDimensionStridedNonFiniteDomain) {
  EXPECT_THAT(PropagateIndexTransformDownsampling(
                  IndexTransformBuilder(1, 1)
                      .input_origin({0})
                      .output_single_input_dimension(0, 0, 2, 0)
                      .Finalize()
                      .value(),
                  BoxView<1>({0}, {kInfIndex}), span<const Index>({1000})),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            ".*Input domain .* is not finite"));
}

TEST(PropagateIndexTransformDownsamplingTest,
     ErrorSingleInputDimensionSize1StridedOverflow) {
  EXPECT_THAT(
      PropagateIndexTransformDownsampling(
          IndexTransformBuilder(1, 1)
              .input_origin({100})
              .input_shape({1})
              .output_single_input_dimension(
                  0, std::numeric_limits<Index>::max(), 2, 0)
              .Finalize()
              .value(),
          BoxView<1>({0}, {kInfIndex}), span<const Index>({1000})),
      MatchesStatus(absl::StatusCode::kOutOfRange, ".*Integer overflow.*"));
  EXPECT_THAT(
      PropagateIndexTransformDownsampling(
          IndexTransformBuilder(1, 1)
              .input_origin({100})
              .input_shape({1})
              .output_single_input_dimension(
                  0, 0, std::numeric_limits<Index>::max(), 0)
              .Finalize()
              .value(),
          BoxView<1>({0}, {kInfIndex}), span<const Index>({1000})),
      MatchesStatus(absl::StatusCode::kOutOfRange, ".*Integer overflow.*"));
}

TEST(PropagateIndexTransformDownsamplingTest,
     ErrorSingleInputDimStridedInvalidDownsampleFactor) {
  EXPECT_THAT(PropagateIndexTransformDownsampling(
                  IndexTransformBuilder(1, 1)
                      .input_shape({100})
                      .output_single_input_dimension(0, 0, 2, 0)
                      .Finalize()
                      .value(),
                  BoxView<1>({0}, {1000}),
                  span<const Index>({std::numeric_limits<Index>::max()})),
              MatchesStatus(absl::StatusCode::kOutOfRange,
                            ".*Downsample factor is out of range"));
}

TEST(PropagateIndexTransformDownsamplingTest,
     ErrorSingleInputDimStridedOverflowMultiplyingStrideAndDownsampleFactor) {
  EXPECT_THAT(
      PropagateIndexTransformDownsampling(
          IndexTransformBuilder(1, 1)
              .input_shape({100})
              .output_single_input_dimension(0, 0, 100, 0)
              .Finalize()
              .value(),
          BoxView<1>({0}, {1000}), span<const Index>({kInfIndex})),
      MatchesStatus(absl::StatusCode::kOutOfRange, ".*Integer overflow.*"));
}

TEST(PropagateIndexTransformDownsamplingTest,
     ErrorSingleInputDimStridedOverflowMultiplyingOffsetAndDownsampleFactor) {
  EXPECT_THAT(
      PropagateIndexTransformDownsampling(
          IndexTransformBuilder(1, 1)
              .input_shape({100})
              .output_single_input_dimension(
                  0, std::numeric_limits<Index>::max(), 2, 0)
              .Finalize()
              .value(),
          BoxView<1>({0}, {1000}), span<const Index>({0xfffffffffffff})),
      MatchesStatus(absl::StatusCode::kOutOfRange, ".*Integer overflow.*"));
}

TEST(PropagateIndexTransformDownsamplingTest,
     ErrorSingleInputDimStridedOutOfRange) {
  EXPECT_THAT(PropagateIndexTransformDownsampling(
                  IndexTransformBuilder(1, 1)
                      .input_shape({100})
                      .output_single_input_dimension(0, 0, 2, 0)
                      .Finalize()
                      .value(),
                  BoxView<1>({0}, {199}), span<const Index>({2})),
              MatchesStatus(absl::StatusCode::kOutOfRange,
                            ".*Output bounds interval .* does not contain "
                            "output range interval .*"));
}

TEST(PropagateIndexTransformDownsamplingTest,
     ErrorIndexArrayInvalidDownsampleFactor) {
  EXPECT_THAT(PropagateIndexTransformDownsampling(
                  IndexTransformBuilder(1, 1)
                      .input_shape({3})
                      .output_index_array(0, 0, 1, MakeArray<Index>({3, 4, 5}))
                      .Finalize()
                      .value(),
                  BoxView<1>({0}, {100}),
                  span<const Index>({std::numeric_limits<Index>::max()})),
              MatchesStatus(absl::StatusCode::kOutOfRange,
                            ".*Downsample factor is out of range"));
}

TEST(PropagateIndexTransformDownsamplingTest,
     ErrorIndexArrayOverflowMultiplyingStrideAndDownsampleFactor) {
  EXPECT_THAT(
      PropagateIndexTransformDownsampling(
          IndexTransformBuilder(1, 1)
              .input_shape({3})
              .output_index_array(0, 0, 100, MakeArray<Index>({3, 4, 5}))
              .Finalize()
              .value(),
          BoxView<1>({0}, {100}), span<const Index>({kInfIndex})),
      MatchesStatus(absl::StatusCode::kOutOfRange, ".*Integer overflow.*"));
}

TEST(PropagateIndexTransformDownsamplingTest,
     ErrorIndexArrayOverflowMultiplyingOffsetAndDownsampleFactor) {
  EXPECT_THAT(
      PropagateIndexTransformDownsampling(
          IndexTransformBuilder(1, 1)
              .input_shape({3})
              .output_index_array(0, 100, 1, MakeArray<Index>({3, 4, 5}))
              .Finalize()
              .value(),
          BoxView<1>({0}, {100}), span<const Index>({kInfIndex})),
      MatchesStatus(absl::StatusCode::kOutOfRange, ".*Integer overflow.*"));
}

TEST(PropagateIndexTransformDownsamplingTest, ErrorIndexArrayOutOfRange) {
  EXPECT_THAT(
      PropagateIndexTransformDownsampling(
          IndexTransformBuilder(1, 1)
              .input_shape({3})
              .output_index_array(0, 0, 1, MakeArray<Index>({3, 4, 5}))
              .Finalize()
              .value(),
          BoxView<1>({0}, {9}), span<const Index>({2})),
      MatchesStatus(
          absl::StatusCode::kOutOfRange,
          "Propagating downsampling factor 2 through output dimension 0: "
          "Index 5 is outside valid range \\[0, 5\\)"));
}

TEST(CanDownsampleIndexTransformTest, Rank0) {
  EXPECT_TRUE(
      CanDownsampleIndexTransform(tensorstore::IdentityTransform(0), {}, {}));
}

TEST(CanDownsampleIndexTransformTest, Constant) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto transform,
      tensorstore::IdentityTransform(1) | Dims(0).IndexSlice(42));
  EXPECT_TRUE(CanDownsampleIndexTransform(transform, BoxView<1>({42}, {1}),
                                          span<const Index>({3})));
  EXPECT_FALSE(CanDownsampleIndexTransform(transform, BoxView<1>({42}, {2}),
                                           span<const Index>({3})));
  EXPECT_FALSE(CanDownsampleIndexTransform(transform, BoxView<1>({41}, {3}),
                                           span<const Index>({3})));
  EXPECT_TRUE(CanDownsampleIndexTransform(transform, BoxView<1>({41}, {2}),
                                          span<const Index>({3})));
  EXPECT_FALSE(CanDownsampleIndexTransform(transform, BoxView<1>({100}),
                                           span<const Index>({3})));
  // No downsampling
  EXPECT_TRUE(CanDownsampleIndexTransform(transform, BoxView<1>({100}),
                                          span<const Index>({1})));
}

TEST(CanDownsampleIndexTransformTest, SingleInputDimension) {
  EXPECT_TRUE(CanDownsampleIndexTransform(
      (tensorstore::IdentityTransform(1) | Dims(0).SizedInterval(9, 3)).value(),
      BoxView<1>({9}, {10}), span<const Index>({3})));
  EXPECT_TRUE(CanDownsampleIndexTransform(
      (tensorstore::IdentityTransform(1) | Dims(0).SizedInterval(18, 1))
          .value(),
      BoxView<1>({9}, {10}), span<const Index>({3})));
  // Not aligned to a downsample block.
  EXPECT_FALSE(CanDownsampleIndexTransform(
      (tensorstore::IdentityTransform(1) | Dims(0).SizedInterval(9, 2)).value(),
      BoxView<1>({9}, {10}), span<const Index>({3})));
  // Stride of -1 is supported.
  EXPECT_FALSE(CanDownsampleIndexTransform(
      (tensorstore::IdentityTransform(1) | Dims(0).SizedInterval(9, 3, -1))
          .value(),
      BoxView<1>({9}, {10}), span<const Index>({3})));
  // Not aligned to a downsample block.
  EXPECT_FALSE(CanDownsampleIndexTransform(
      (tensorstore::IdentityTransform(1) | Dims(0).SizedInterval(10, 2))
          .value(),
      BoxView<1>({9}, {10}), span<const Index>({3})));
  // Non-unit striding not supported.
  EXPECT_FALSE(CanDownsampleIndexTransform(
      (tensorstore::IdentityTransform(1) | Dims(0).SizedInterval(9, 3, 2))
          .value(),
      BoxView<1>({9}, {10}), span<const Index>({3})));
}

TEST(CanDownsampleIndexTransformTest, IndexArray) {
  // Index array maps are not supported.
  EXPECT_FALSE(CanDownsampleIndexTransform(
      (tensorstore::IdentityTransform(1) |
       Dims(0).IndexArraySlice(MakeArray<Index>({2, 5, 3})))
          .value(),
      BoxView<1>({0}, {100}), span<const Index>({2})));
}

void TestPropagateIndexTransformDownsamplingInvariance(DimensionIndex rank) {
  std::minstd_rand gen{tensorstore::internal::GetRandomSeedForTest(
      "TENSORSTORE_DOWNSAMPLE_PROPAGATE_INVARIANCE_SEED")};
  tensorstore::internal::MakeRandomBoxParameters box_p;
  box_p.min_rank = box_p.max_rank = rank;
  auto base_bounds = tensorstore::internal::MakeRandomBox(gen, box_p);
  SCOPED_TRACE(tensorstore::StrCat("base_bounds=", base_bounds));
  auto base_data = tensorstore::internal::MakeRandomArray(
      gen, base_bounds, tensorstore::dtype_v<uint8_t>);
  SCOPED_TRACE(tensorstore::StrCat("base_data=", base_data));
  std::vector<Index> downsample_factors(rank);
  for (DimensionIndex i = 0; i < rank; ++i) {
    downsample_factors[i] =
        absl::Uniform<Index>(absl::IntervalClosedClosed, gen, 1, 2);
  }
  SCOPED_TRACE(tensorstore::StrCat("downsample_factors=",
                                   tensorstore::span(downsample_factors)));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto downsampled_data,
      DownsampleArray(base_data, downsample_factors, DownsampleMethod::kMean));
  Box<> downsampled_bounds(rank);
  DownsampleBounds(base_bounds, downsampled_bounds, downsample_factors,
                   DownsampleMethod::kMean);
  SCOPED_TRACE(tensorstore::StrCat("downsampled_bounds=", downsampled_bounds));
  auto downsampled_transform = tensorstore::internal::MakeRandomIndexTransform(
      gen, downsampled_bounds, rank * 2);
  SCOPED_TRACE(
      tensorstore::StrCat("downsampled_transform=", downsampled_transform));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto propagated,
      PropagateIndexTransformDownsampling(downsampled_transform, base_bounds,
                                          downsample_factors));
  SCOPED_TRACE(tensorstore::StrCat("propagated=", propagated));
  SCOPED_TRACE(tensorstore::StrCat("downsampled_data=", downsampled_data));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto downsample_then_transform,
      downsampled_data | downsampled_transform | tensorstore::Materialize());

  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto transformed_base,
                                   base_data | propagated.transform);

  tensorstore::SharedOffsetArray<const void> transform_then_downsample;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      transform_then_downsample,
      DownsampleTransformedArray(transformed_base,
                                 propagated.input_downsample_factors,
                                 DownsampleMethod::kMean));
  if (downsampled_transform.input_rank() < propagated.transform.input_rank()) {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        transform_then_downsample,
        transform_then_downsample |
            tensorstore::DynamicDims(
                {tensorstore::DimRangeSpec{downsampled_transform.input_rank()}})
                .IndexSlice(0) |
            tensorstore::Materialize());
  }
  EXPECT_EQ(transform_then_downsample, downsample_then_transform);
}

constexpr size_t kNumRandomTests = 1000;

TEST(PropagateIndexTransformDownsamplingTest, InvarianceRank0) {
  for (size_t i = 0; i < kNumRandomTests; ++i) {
    TestPropagateIndexTransformDownsamplingInvariance(/*rank=*/0);
  }
}

TEST(PropagateIndexTransformDownsamplingTest, InvarianceRank1) {
  for (size_t i = 0; i < kNumRandomTests; ++i) {
    TestPropagateIndexTransformDownsamplingInvariance(/*rank=*/1);
  }
}

TEST(PropagateIndexTransformDownsamplingTest, InvarianceRank2) {
  for (size_t i = 0; i < kNumRandomTests; ++i) {
    TestPropagateIndexTransformDownsamplingInvariance(/*rank=*/2);
  }
}

TEST(PropagateIndexTransformDownsamplingTest, InvarianceRank3) {
  for (size_t i = 0; i < kNumRandomTests; ++i) {
    TestPropagateIndexTransformDownsamplingInvariance(/*rank=*/3);
  }
}

TEST(DownsampleIntervalTest, UnboundedLower) {
  EXPECT_EQ(IndexInterval::Closed(-kInfIndex, 10),
            DownsampleInterval(IndexInterval::UncheckedClosed(-kInfIndex, 30),
                               3, DownsampleMethod::kMean));
}

TEST(DownsampleIntervalTest, UnboundedUpper) {
  EXPECT_EQ(IndexInterval::Closed(-10, kInfIndex),
            DownsampleInterval(IndexInterval::UncheckedClosed(-30, kInfIndex),
                               3, DownsampleMethod::kMean));
}

}  // namespace
