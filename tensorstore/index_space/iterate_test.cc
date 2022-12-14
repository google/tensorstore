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

/// Tests for iteration over transformed arrays.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/index_interval.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/index_space/internal/iterate_impl.h"
#include "tensorstore/index_space/transformed_array.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

namespace {

using ::tensorstore::DimensionIndex;
using ::tensorstore::Dims;
using ::tensorstore::Index;
using ::tensorstore::IndexInterval;
using ::tensorstore::IterationConstraints;
using ::tensorstore::kInfIndex;
using ::tensorstore::MakeArray;
using ::tensorstore::MatchesStatus;
using ::tensorstore::span;
using ::tensorstore::internal_index_space::TransformAccess;

TEST(ValidateIndexArrayBoundsTest, Basic) {
  EXPECT_EQ(absl::OkStatus(),
            ValidateIndexArrayBounds(IndexInterval::UncheckedClosed(5, 8),
                                     MakeArray<Index>({5, 6, 7, 8})));
  EXPECT_THAT(ValidateIndexArrayBounds(IndexInterval::UncheckedClosed(5, 8),
                                       MakeArray<Index>({5, 6, 7, 8, 9})),
              MatchesStatus(absl::StatusCode::kOutOfRange,
                            "Index 9 is outside valid range \\[5, 9\\)"));
  // +/-kInfIndex are not valid even if the bounds include them.
  EXPECT_THAT(
      ValidateIndexArrayBounds(IndexInterval(), MakeArray<Index>({kInfIndex})),
      MatchesStatus(absl::StatusCode::kOutOfRange));
}

TEST(InitializeSingleArrayIterationStateTest, Basic) {
  namespace flags =
      tensorstore::internal_index_space::input_dimension_iteration_flags;
  std::vector<flags::Bitmask> input_dimension_flags(2, 0);
  auto array = tensorstore::MakeOffsetArray<int>(/*origin=*/{5, 6},
                                                 {{1, 2, 3, 4}, {5, 6, 7, 8}});
  auto index_array = MakeArray<Index>({{0, 1, 1, 0}});
  auto transform = tensorstore::IndexTransformBuilder<2, 2>()
                       .input_origin({1, 2})
                       .input_shape({2, 4})
                       .output_single_input_dimension(0, 7, -1, 0)
                       .output_index_array(1, 7, 2, index_array)
                       .Finalize()
                       .value();

  tensorstore::internal_index_space::SingleArrayIterationState
      single_array_state(2, 2);
  EXPECT_EQ(
      absl::OkStatus(),
      tensorstore::internal_index_space::InitializeSingleArrayIterationState(
          array, TransformAccess::rep(transform),
          transform.input_origin().data(), transform.input_shape().data(),
          &single_array_state, input_dimension_flags.data()));
  EXPECT_EQ(1, single_array_state.num_array_indexed_output_dimensions);
  EXPECT_THAT(input_dimension_flags,
              ::testing::ElementsAre(flags::strided, flags::array_indexed));
  EXPECT_EQ(index_array.data(), single_array_state.index_array_pointers[0]);
  EXPECT_EQ(transform.output_index_map(1)
                .index_array()
                .layout()
                .byte_strides()
                .data(),
            single_array_state.index_array_byte_strides[0]);
  EXPECT_EQ(&array(6, 7), single_array_state.base_pointer);
  // Stride is equal to the stride (2) specified for the output index map for
  // output dimension 1 of `transform`, multiplied by the byte stride
  // (sizeof(int)) of dimension 1 of `array`.
  EXPECT_THAT(single_array_state.index_array_output_byte_strides_span(),
              ::testing::ElementsAre(2 * sizeof(int)));
  EXPECT_THAT(single_array_state.input_byte_strides,
              ::testing::ElementsAre(-4 * static_cast<Index>(sizeof(int)), 0));
}

TEST(ComputeDimensionIterationOrderTest, Basic) {
  namespace flags =
      tensorstore::internal_index_space::input_dimension_iteration_flags;
  using ::tensorstore::internal_index_space::ComputeDimensionIterationOrder;
  const flags::Bitmask input_dimension_flags[] = {
      flags::can_skip,      flags::strided,  flags::array_indexed,
      flags::array_indexed, flags::can_skip, flags::strided};

  {
    const int order[] = {0, 3, 5, 4, 2, 1};
    auto layout =
        tensorstore::internal_index_space::ComputeDimensionIterationOrder(
            input_dimension_flags, /*order_constraint=*/{},
            [&](DimensionIndex a, DimensionIndex b) {
              return order[a] < order[b];
            });
    EXPECT_EQ(2, layout.pure_strided_start_dim);
    EXPECT_EQ(4, layout.pure_strided_end_dim);
    EXPECT_THAT(span(layout.input_dimension_order).first(4),
                ::testing::ElementsAre(3, 2, 5, 1));
  }

  {
    auto layout =
        tensorstore::internal_index_space::ComputeDimensionIterationOrder(
            input_dimension_flags,
            /*order_constraint=*/tensorstore::ContiguousLayoutOrder::c,
            [](DimensionIndex a, DimensionIndex b) { return false; });
    EXPECT_EQ(3, layout.pure_strided_start_dim);
    EXPECT_EQ(4, layout.pure_strided_end_dim);
    EXPECT_THAT(span(layout.input_dimension_order).first(4),
                ::testing::ElementsAre(1, 2, 3, 5));
  }

  {
    auto layout = ComputeDimensionIterationOrder(
        input_dimension_flags,
        /*order_constraint=*/tensorstore::ContiguousLayoutOrder::fortran,
        [](DimensionIndex a, DimensionIndex b) { return false; });
    EXPECT_EQ(3, layout.pure_strided_start_dim);
    EXPECT_EQ(4, layout.pure_strided_end_dim);
    EXPECT_THAT(span(layout.input_dimension_order).first(4),
                ::testing::ElementsAre(5, 3, 2, 1));
  }
}

TEST(SimplifyDimensionIterationOrderTest, Rank5) {
  tensorstore::internal_index_space::DimensionIterationOrder original_layout(5);
  original_layout.input_dimension_order[0] = 7;
  original_layout.input_dimension_order[1] = 9;
  original_layout.input_dimension_order[2] = 3;
  original_layout.input_dimension_order[3] = 1;
  original_layout.input_dimension_order[4] = 5;
  original_layout.pure_strided_start_dim = 3;
  original_layout.pure_strided_end_dim = 5;
  const Index input_shape[] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

  {
    auto result = SimplifyDimensionIterationOrder(
        original_layout, input_shape, /*can_combine_dimensions=*/
        [&](DimensionIndex a, DimensionIndex b, DimensionIndex size) {
          EXPECT_EQ(input_shape[b], size);
          switch (b) {
            case 9:
              EXPECT_EQ(7, a);
              return true;
            case 3:
              EXPECT_EQ(9, a);
              return false;
            default:
              ADD_FAILURE();
              return false;
          }
        });
    EXPECT_THAT(span(result.input_dimension_order).first(4),
                ::testing::ElementsAre(9, 3, 1, 5));
    EXPECT_THAT(span(result.simplified_shape).first(4),
                ::testing::ElementsAre(9 * 11, 5, 3, 7));
    EXPECT_THAT(2, result.pure_strided_start_dim);
    EXPECT_THAT(4, result.pure_strided_end_dim);
  }
}

TEST(SimplifyDimensionIterationOrderTest, Rank1) {
  tensorstore::internal_index_space::DimensionIterationOrder original_layout(1);
  original_layout.input_dimension_order[0] = 0;
  original_layout.pure_strided_start_dim = 1;
  original_layout.pure_strided_end_dim = 1;
  const Index input_shape[] = {5};

  {
    auto result = SimplifyDimensionIterationOrder(
        original_layout, input_shape, /*can_combine_dimensions=*/
        [&](DimensionIndex a, DimensionIndex b, DimensionIndex size) {
          return false;
        });
    EXPECT_THAT(result.input_dimension_order, ::testing::ElementsAre(0));
    EXPECT_THAT(result.simplified_shape, ::testing::ElementsAre(5));
    EXPECT_THAT(1, result.pure_strided_start_dim);
    EXPECT_THAT(1, result.pure_strided_end_dim);
  }
}

TEST(IterateOverTransformedArraysTest, StridedOnly) {
  auto source_array = MakeArray<const float>({{1, 2, 3, 4}, {5, 6, 7, 8}});
  auto dest_array = tensorstore::AllocateArray<float>(
      {2, 4, 2}, tensorstore::c_order, tensorstore::value_init);
  TENSORSTORE_CHECK_OK(IterateOverTransformedArrays(
      [&](const float* source_ptr, float* dest_ptr) {
        *dest_ptr = *source_ptr;
      },
      /*constraints=*/{},  //
      Dims(1).TranslateClosedInterval(1, 2)(source_array),
      Dims(1).IndexSlice(0)(dest_array)));
  EXPECT_EQ(MakeArray<float>({{{2, 3}, {0, 0}, {0, 0}, {0, 0}},
                              {{6, 7}, {0, 0}, {0, 0}, {0, 0}}}),
            dest_array);
}

TEST(IterateOverTransformedArraysTest, ErrorHandling) {
  EXPECT_THAT(IterateOverTransformedArrays(
                  [&](const float* source_ptr, float* dest_ptr) {},
                  /*constraints=*/{},
                  tensorstore::ArrayView<const float>(
                      tensorstore::MakeScalarArray<float>(1)),
                  MakeArray<float>({1}))
                  .status(),
              MatchesStatus(
                  absl::StatusCode::kInvalidArgument,
                  "Transformed array input ranks \\{0, 1\\} do not all match"));
}

TEST(IterateOverTransformedArrayTest, EarlyStoppingWithoutStatus) {
  auto array_a = MakeArray<float>({5, 6, 7, 9});
  auto array_b = MakeArray<float>({5, 6, 8, 9});
  auto result = IterateOverTransformedArrays(
      [&](const float* a_ptr, float* b_ptr) {
        if (*a_ptr != *b_ptr) {
          return false;
        }
        *b_ptr = 0;
        return true;
      },
      /*constraints=*/{}, array_a, array_b);
  TENSORSTORE_ASSERT_OK(result);
  EXPECT_FALSE(result->success);
  EXPECT_EQ(2, result->count);
  EXPECT_EQ(MakeArray<float>({5, 6, 7, 9}), array_a);
  EXPECT_EQ(MakeArray<float>({0, 0, 8, 9}), array_b);
}

TEST(IterateOverTransformedArrayTest, EarlyStoppingWithStatus) {
  auto array_a = MakeArray<float>({5, 6, 7, 9});
  auto array_b = MakeArray<float>({5, 6, 8, 9});
  absl::Status status;
  auto result = IterateOverTransformedArrays(
      [&](const float* a_ptr, float* b_ptr, absl::Status* status) {
        if (*a_ptr != *b_ptr) {
          *status =
              absl::UnknownError(tensorstore::StrCat(*a_ptr, " ", *b_ptr));
          // The status pointer is just passed through by
          // `IterateOverTransformedArrays`, but is not otherwise used.  We have
          // to return `false` to cause iteration to stop.
          return false;
        }
        *b_ptr = 0;
        return true;
      },
      &status,
      /*constraints=*/{}, array_a, array_b);
  TENSORSTORE_ASSERT_OK(result);
  EXPECT_THAT(status, MatchesStatus(absl::StatusCode::kUnknown, "7 8"));
  EXPECT_FALSE(result->success);
  EXPECT_EQ(2, result->count);
  EXPECT_EQ(MakeArray<float>({5, 6, 7, 9}), array_a);
  EXPECT_EQ(MakeArray<float>({0, 0, 8, 9}), array_b);
}

TEST(IterateOverTransformedArraysTest, IndexArrays) {
  auto source_array = MakeArray<const float>({{1, 2, 3, 4}, {5, 6, 7, 8}});
  auto dest_array = tensorstore::AllocateArray<float>(
      {2, 4, 2}, tensorstore::c_order, tensorstore::value_init);
  TENSORSTORE_CHECK_OK(IterateOverTransformedArrays(
      [&](const float* source_ptr, float* dest_ptr) {
        *dest_ptr = *source_ptr;
      },
      /*constraints=*/{},                  //
      Dims(0).MoveToBack()(source_array),  //
      Dims(0, 1)
          .IndexArraySlice(MakeArray<Index>({1, 0, 1, 1}),
                           MakeArray<Index>({0, 1, 2, 3}))
          .MoveToFront()(dest_array)));
  EXPECT_EQ(MakeArray<float>({{{0, 0}, {2, 6}, {0, 0}, {0, 0}},
                              {{1, 5}, {0, 0}, {3, 7}, {4, 8}}}),
            dest_array);
}

TEST(IterateOverTransformedArraysTest, SingleElementIndexArray) {
  EXPECT_EQ(tensorstore::TransformArray(
                MakeArray<float>({1, 2, 3}),
                tensorstore::IndexTransformBuilder<1, 1>()
                    .input_origin({0})
                    .input_shape({1})
                    .output_index_array(0, 0, 1, MakeArray<Index>({2}))
                    .Finalize()
                    .value())
                .value(),
            MakeArray<float>({3}));
}

// Tests iteration when two index array dimensions can be combined.
TEST(IterateOverTransformedArraysTest, CombineDimensions) {
  EXPECT_EQ(
      tensorstore::TransformArray(
          MakeArray<float>({1, 2, 3, 4}),
          tensorstore::IndexTransformBuilder<2, 1>()
              .input_origin({0, 0})
              .input_shape({2, 2})
              .output_index_array(0, 0, 1, MakeArray<Index>({{0, 1}, {1, 3}}))
              .Finalize()
              .value())
          .value(),
      MakeArray<float>({{1, 2}, {2, 4}}));
}

// Tests iteration when there are non-index array dimensions that cannot be
// combined.
TEST(IterateOverTransformedArraysTest, NotCombinableNonIndexedDimensions) {
  EXPECT_EQ(tensorstore::TransformArray(
                MakeArray<float>({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}),
                tensorstore::IndexTransformBuilder<3, 3>()
                    .input_origin({0, 0, 0})
                    .input_shape({2, 2, 2})
                    .output_single_input_dimension(0, 0)
                    .output_index_array(1, 0, 1, MakeArray<Index>({{{0}, {1}}}))
                    .output_single_input_dimension(2, 2)
                    .Finalize()
                    .value())
                .value(),
            MakeArray<float>({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}));
}

}  // namespace
