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

#include "tensorstore/index_space/transform_broadcastable_array.h"

#include <random>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/array.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/index_space/index_transform_testutil.h"
#include "tensorstore/internal/test_util.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

namespace {

using ::tensorstore::DimensionIndex;
using ::tensorstore::Index;
using ::tensorstore::IndexDomain;
using ::tensorstore::IndexDomainView;
using ::tensorstore::IndexTransformBuilder;
using ::tensorstore::IndexTransformView;
using ::tensorstore::MakeArray;
using ::tensorstore::MakeScalarArray;
using ::tensorstore::MatchesStatus;
using ::tensorstore::SharedArray;
using ::tensorstore::SharedArrayView;
using ::tensorstore::span;
using ::tensorstore::TransformInputBroadcastableArray;
using ::tensorstore::TransformOutputBroadcastableArray;

/// Tests round tripping, with both the `input_array` and `output_array`
/// specified.
void TestRoundTrip(IndexTransformView<> transform,
                   SharedArrayView<const void> input_array,
                   SharedArrayView<const void> output_array,
                   IndexDomainView<> output_domain) {
  SCOPED_TRACE(tensorstore::StrCat(
      "transform=", transform, ", output_domain=", output_domain,
      ", input_array.shape=", input_array.shape(),
      ", output_array.shape=", output_array.shape()));
  EXPECT_THAT(
      TransformOutputBroadcastableArray(transform, output_array, output_domain),
      ::testing::Optional(input_array));
  EXPECT_THAT(TransformInputBroadcastableArray(transform, input_array),
              ::testing::Optional(output_array));
}

/// Tests round tripping, with only the `output_array` specified.
void TestRoundTrip(IndexTransformView<> transform,
                   SharedArrayView<const void> output_array,
                   IndexDomainView<> output_domain = IndexDomainView<>(),
                   bool test_inverse = false) {
  SCOPED_TRACE(tensorstore::StrCat(
      "transform=", transform, ", output_domain=", output_domain,
      ", output_array.shape=", output_array.shape()));
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto input_array,
                                   TransformOutputBroadcastableArray(
                                       transform, output_array, output_domain));
  EXPECT_THAT(TransformInputBroadcastableArray(transform, input_array),
              ::testing::Optional(output_array));
  if (test_inverse) {
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto inverse_transform,
                                     tensorstore::InverseTransform(transform));
    EXPECT_THAT(
        TransformInputBroadcastableArray(inverse_transform, output_array),
        ::testing::Optional(input_array));
  }
}

SharedArray<int> MakeTestArray(span<const Index> shape) {
  auto array = tensorstore::AllocateArray<int>(shape);
  for (Index i = 0, num_elements = array.num_elements(); i < num_elements;
       ++i) {
    array.data()[i] = i;
  }
  return array;
}

TEST(RoundTripTest, IdentityTransform) {
  for (DimensionIndex rank = 0; rank <= 3; ++rank) {
    SCOPED_TRACE(tensorstore::StrCat("rank=", rank));
    std::vector<Index> shape(rank);
    for (DimensionIndex dim = 0; dim < rank; ++dim) {
      shape[dim] = dim + 2;
    }
    auto array = MakeTestArray(shape);
    TestRoundTrip(tensorstore::IdentityTransform(shape), array, array,
                  tensorstore::IndexDomain<>());
    TestRoundTrip(tensorstore::IdentityTransform(rank), array, array,
                  tensorstore::IndexDomain<>());
    TestRoundTrip(tensorstore::IdentityTransform(shape), array, array,
                  tensorstore::IdentityTransform(shape).domain());
  }
}

TEST(RoundTripTest, RandomInvertibleTransform) {
  constexpr size_t kNumIterations = 100;
  for (size_t i = 0; i < kNumIterations; ++i) {
    std::minstd_rand gen{tensorstore::internal::GetRandomSeedForTest(
        "TENSORSTORE_INTERNAL_TRANSFORM_BROADCASTABLE_ARRAY_TEST_SEED")};
    auto box = tensorstore::internal::MakeRandomBox(gen);
    auto array = tensorstore::UnbroadcastArray(MakeTestArray(box.shape()));
    auto domain = IndexDomain(box);
    auto transform =
        tensorstore::internal::MakeRandomStridedIndexTransformForOutputSpace(
            gen, domain);
    TestRoundTrip(transform, array);
    TestRoundTrip(transform, array, domain);
  }
}

TEST(RoundTripTest, RandomInvertibleTransformNoNewDims) {
  constexpr size_t kNumIterations = 100;
  for (size_t i = 0; i < kNumIterations; ++i) {
    std::minstd_rand gen{tensorstore::internal::GetRandomSeedForTest(
        "TENSORSTORE_INTERNAL_TRANSFORM_BROADCASTABLE_ARRAY_TEST_SEED")};
    auto box = tensorstore::internal::MakeRandomBox(gen);
    auto array = tensorstore::UnbroadcastArray(MakeTestArray(box.shape()));
    auto domain = IndexDomain(box);
    tensorstore::internal::MakeStridedIndexTransformForOutputSpaceParameters p;
    p.max_new_dims = 0;
    auto transform =
        tensorstore::internal::MakeRandomStridedIndexTransformForOutputSpace(
            gen, domain, p);
    TestRoundTrip(transform, array, IndexDomain(), /*test_inverse=*/true);
    TestRoundTrip(transform, array, domain, /*test_inverse=*/true);
  }
}

TEST(TransformOutputBroadcastableArrayTest, ConstantMap) {
  auto array = MakeArray<int>({{1}, {2}, {3}});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto transform, IndexTransformBuilder(1, 2)
                          .output_single_input_dimension(0, 5, -1, 0)
                          .output_constant(1, 42)
                          .Finalize());
  EXPECT_THAT(
      TransformOutputBroadcastableArray(transform, array, IndexDomain()),
      ::testing::Optional(MakeArray<int>({3, 2, 1})));
}

TEST(TransformOutputBroadcastableArrayTest, NonUnitStrideMap) {
  auto array = MakeArray<int>({{1}, {2}, {3}});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto transform, IndexTransformBuilder(2, 2)
                          .output_single_input_dimension(0, 5, -1, 0)
                          .output_single_input_dimension(1, 42, 2, 1)
                          .Finalize());
  EXPECT_THAT(
      TransformOutputBroadcastableArray(transform, array, IndexDomain()),
      ::testing::Optional(MakeArray<int>({{3}, {2}, {1}})));
}

TEST(TransformOutputBroadcastableArrayTest, ArrayMap) {
  auto array = MakeArray<int>({{1}, {2}, {3}});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto transform,
      IndexTransformBuilder(1, 2)
          .input_shape({3})
          .output_single_input_dimension(0, 5, -1, 0)
          .output_index_array(1, 20, 1, MakeArray<Index>({0, 5, 10}))
          .Finalize());
  EXPECT_THAT(
      TransformOutputBroadcastableArray(transform, array, IndexDomain()),
      ::testing::Optional(MakeArray<int>({3, 2, 1})));
}

TEST(TransformInputBroadcastableArrayTest, ConstantMap) {
  auto array = MakeScalarArray<int>(42);
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto transform,
      IndexTransformBuilder(0, 1).output_constant(0, 42).Finalize());
  EXPECT_THAT(
      TransformInputBroadcastableArray(transform, array),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          "Cannot transform input array through constant output index map"));
}

TEST(TransformInputBroadcastableArrayTest, NonUnitStrideMap) {
  auto array = MakeArray<int>({1, 2, 3});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto transform, IndexTransformBuilder(1, 1)
                          .output_single_input_dimension(0, 5, 2, 0)
                          .Finalize());
  EXPECT_THAT(TransformInputBroadcastableArray(transform, array),
              MatchesStatus(absl::StatusCode::kInvalidArgument,
                            "Cannot transform input array through "
                            "non-unit-stride output index map"));
}

TEST(TransformInputBroadcastableArrayTest, ArrayMap) {
  auto array = MakeArray<int>({1, 2, 3});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto transform,
      IndexTransformBuilder(1, 1)
          .input_shape({3})
          .output_index_array(0, 20, 1, MakeArray<Index>({0, 5, 10}))
          .Finalize());
  EXPECT_THAT(
      TransformInputBroadcastableArray(transform, array),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          "Cannot transform input array through array output index map"));
}

TEST(TransformInputBroadcastableArrayTest, Diagonal) {
  auto array = MakeArray<int>({1, 2, 3});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto transform,
                                   IndexTransformBuilder(1, 2)
                                       .output_single_input_dimension(0, 0)
                                       .output_single_input_dimension(1, 0)
                                       .Finalize());
  EXPECT_THAT(
      TransformInputBroadcastableArray(transform, array),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Cannot transform input array with multiple "
                    "output dimensions mapping to the same input dimension"));
}

TEST(TransformInputBroadcastableArrayTest, UnmappedNoError) {
  auto array = MakeArray<int>({1, 2, 3});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto transform,
                                   IndexTransformBuilder(2, 1)
                                       .output_single_input_dimension(0, 1)
                                       .Finalize());
  EXPECT_THAT(TransformInputBroadcastableArray(transform, array),
              ::testing::Optional(array));
}

TEST(TransformInputBroadcastableArrayTest, UnmappedError) {
  auto array = MakeArray<int>({1, 2, 3});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto transform,
                                   IndexTransformBuilder(2, 1)
                                       .output_single_input_dimension(0, 0)
                                       .Finalize());
  EXPECT_THAT(
      TransformInputBroadcastableArray(transform, array),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          "Cannot transform input array; dimension 0 cannot be mapped"));
}

TEST(TransformInputBroadcastableArrayTest, ExtraDimensionError) {
  auto array = MakeArray<int>({{1, 2, 3}, {4, 5, 6}});
  EXPECT_THAT(
      TransformInputBroadcastableArray(tensorstore::IdentityTransform(1),
                                       array),
      MatchesStatus(
          absl::StatusCode::kInvalidArgument,
          "Cannot transform input array; dimension 0 cannot be mapped"));
}

TEST(TransformInputBroadcastableArrayTest, ExtraDimensionNoError) {
  auto array = MakeArray<int>({{1, 2, 3}});
  EXPECT_THAT(TransformInputBroadcastableArray(
                  tensorstore::IdentityTransform(1), array),
              ::testing::Optional(MakeArray<int>({1, 2, 3})));
}

}  // namespace
