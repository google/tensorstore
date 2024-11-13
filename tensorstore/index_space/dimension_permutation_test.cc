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
#include "tensorstore/index.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_transform.h"
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

}  // namespace
