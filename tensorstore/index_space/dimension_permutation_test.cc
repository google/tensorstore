// Copyright 2021 The TensorStore Authors
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

#include "tensorstore/index_space/dimension_permutation.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/array.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::DimensionIndex;
using ::tensorstore::Dims;
using ::tensorstore::IsValidPermutation;
using ::tensorstore::PermutationMatchesOrder;
using ::tensorstore::span;

TEST(TransformOutputDimensionOrderTest, Rank0) {
  std::vector<DimensionIndex> source;
  std::vector<DimensionIndex> dest;
  tensorstore::TransformOutputDimensionOrder(tensorstore::IdentityTransform(0),
                                             source, dest);
  EXPECT_THAT(dest, ::testing::IsEmpty());
  tensorstore::TransformInputDimensionOrder(tensorstore::IdentityTransform(0),
                                            dest, source);
  EXPECT_THAT(source, ::testing::IsEmpty());
}

TEST(TransformOutputDimensionOrderTest, Rank1Identity) {
  std::vector<DimensionIndex> source{0};
  std::vector<DimensionIndex> dest(1, 42);
  tensorstore::TransformOutputDimensionOrder(tensorstore::IdentityTransform(1),
                                             source, dest);
  EXPECT_THAT(dest, ::testing::ElementsAre(0));
}

TEST(TransformOutputDimensionOrderTest, Rank2COrderIdentity) {
  std::vector<DimensionIndex> source{0, 1};
  std::vector<DimensionIndex> dest(2, 42);
  std::vector<DimensionIndex> source2(2, 42);
  auto transform = tensorstore::IdentityTransform(2);
  tensorstore::TransformOutputDimensionOrder(transform, source, dest);
  EXPECT_THAT(dest, ::testing::ElementsAre(0, 1));
  tensorstore::TransformInputDimensionOrder(transform, dest, source2);
  EXPECT_EQ(source, source2);
}

TEST(TransformOutputDimensionOrderTest, Rank2FortranOrderIdentity) {
  std::vector<DimensionIndex> source{1, 0};
  std::vector<DimensionIndex> dest(2, 42);
  std::vector<DimensionIndex> source2(2, 42);
  auto transform = tensorstore::IdentityTransform(2);
  tensorstore::TransformOutputDimensionOrder(transform, source, dest);
  EXPECT_THAT(dest, ::testing::ElementsAre(1, 0));
  tensorstore::TransformInputDimensionOrder(transform, dest, source2);
  EXPECT_EQ(source, source2);
}

TEST(TransformOutputDimensionOrderTest, Rank2COrderTranspose) {
  std::vector<DimensionIndex> source{0, 1};
  std::vector<DimensionIndex> dest(2, 42);
  std::vector<DimensionIndex> source2(2, 42);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto transform,
      tensorstore::IdentityTransform(2) | Dims(1, 0).Transpose());
  tensorstore::TransformOutputDimensionOrder(transform, source, dest);
  EXPECT_THAT(dest, ::testing::ElementsAre(1, 0));
  tensorstore::TransformInputDimensionOrder(transform, dest, source2);
  EXPECT_EQ(source, source2);
}

TEST(TransformOutputDimensionOrderTest, Rank2FortranOrderTranspose) {
  std::vector<DimensionIndex> source{1, 0};
  std::vector<DimensionIndex> dest(2, 42);
  std::vector<DimensionIndex> source2(2, 42);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto transform,
      tensorstore::IdentityTransform(2) | Dims(1, 0).Transpose());
  tensorstore::TransformOutputDimensionOrder(transform, source, dest);
  EXPECT_THAT(dest, ::testing::ElementsAre(0, 1));
  tensorstore::TransformInputDimensionOrder(transform, dest, source2);
  EXPECT_EQ(source, source2);
}

TEST(TransformOutputDimensionOrderTest, NonBijective) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto transform,
                                   tensorstore::IndexTransformBuilder<>(2, 3)
                                       .output_single_input_dimension(0, 1)
                                       .output_constant(1, 0)
                                       .output_single_input_dimension(2, 1)
                                       .Finalize());
  std::vector<DimensionIndex> output_perm{2, 1, 0};
  std::vector<DimensionIndex> input_perm(2, 42);
  tensorstore::TransformOutputDimensionOrder(transform, output_perm,
                                             input_perm);
  EXPECT_THAT(input_perm, ::testing::ElementsAre(1, 0));

  std::vector<DimensionIndex> round_trip_output_perm(3, 42);
  tensorstore::TransformInputDimensionOrder(transform, input_perm,
                                            round_trip_output_perm);
  EXPECT_THAT(round_trip_output_perm, ::testing::ElementsAre(0, 2, 1));
}

TEST(TransformOutputDimensionOrderTest, NonBijectiveArrayAndTieBreaking) {
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto transform,
      tensorstore::IndexTransformBuilder<>(4, 4)
          .input_shape({2, 2, 2, 2})
          .output_single_input_dimension(0, 3)
          .output_constant(1, 0)
          .output_single_input_dimension(2, 1)
          .output_index_array(
              3, 0, 1, tensorstore::MakeArray<tensorstore::Index>({{{{0}}}}))
          .Finalize());
  std::vector<DimensionIndex> output_perm{2, 1, 3, 0};
  std::vector<DimensionIndex> input_perm(4, 42);
  tensorstore::TransformOutputDimensionOrder(transform, output_perm,
                                             input_perm);
  // Input dim 1 maps from output_perm[0]=2; input dim 3 maps from
  // output_perm[3]=0; input dims 0 and 2 are unmapped and ordered ascending.
  EXPECT_THAT(input_perm, ::testing::ElementsAre(1, 3, 0, 2));

  std::vector<DimensionIndex> input_perm_in{1, 3, 2, 0};
  std::vector<DimensionIndex> output_perm_out(4, 42);
  tensorstore::TransformInputDimensionOrder(transform, input_perm_in,
                                            output_perm_out);
  // Output dim 2 maps to input dim 1 (ordinal 0); output dim 0 maps to input
  // dim 3 (ordinal 1); output dims 1 (constant) and 3 (array) are ordered last
  // and ascending by dimension index.
  EXPECT_THAT(output_perm_out, ::testing::ElementsAre(2, 0, 1, 3));
}

}  // namespace
