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

#include "tensorstore/internal/nditerable_util.h"

#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/index.h"
#include "tensorstore/util/span.h"

namespace {

using ::tensorstore::Index;
using ::tensorstore::span;
using ::tensorstore::internal::GetNDIterationBlockSize;
using ::tensorstore::internal::NDIterationPositionStepper;
using ::tensorstore::internal::ResetBufferPositionAtBeginning;
using ::tensorstore::internal::ResetBufferPositionAtEnd;
using ::tensorstore::internal::StepBufferPositionBackward;
using ::tensorstore::internal::StepBufferPositionForward;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;

TEST(GetNDIterationBlockSize, Basic) {
#ifndef TENSORSTORE_INTERNAL_NDITERABLE_TEST_UNIT_BLOCK_SIZE
  constexpr auto expected_block_size = [](Index block_size) {
    return block_size;
  };
#else
  constexpr auto expected_block_size = [](Index block_size) { return 1; };
#endif
  // If no temporary buffer is required, uses the full extent of the last
  // dimension.
  EXPECT_EQ(expected_block_size(1000000),
            GetNDIterationBlockSize(/*working_memory_bytes_per_element=*/0,
                                    span<const Index>({3, 4, 1000000})));

  // Block size is limited by the extent of the last dimension.
  EXPECT_EQ(expected_block_size(15),
            GetNDIterationBlockSize(/*working_memory_bytes_per_element=*/1,
                                    span<const Index>({3, 4, 15})));

  EXPECT_EQ(expected_block_size(24 * 1024),
            GetNDIterationBlockSize(/*working_memory_bytes_per_element=*/1,
                                    span<const Index>({3, 4, 1000000})));

  EXPECT_EQ(expected_block_size(768),
            GetNDIterationBlockSize(/*working_memory_bytes_per_element=*/32,
                                    span<const Index>({3, 4, 1000000})));

  EXPECT_EQ(expected_block_size(384),
            GetNDIterationBlockSize(/*working_memory_bytes_per_element=*/64,
                                    span<const Index>({3, 4, 1000000})));
}

TEST(ResetBufferPositionTest, OneDimensional) {
  std::vector<Index> shape{10};
  std::vector<Index> position{42};
  ResetBufferPositionAtBeginning(position);
  EXPECT_THAT(position, ElementsAre(0));
  ResetBufferPositionAtEnd(shape, /*step=*/1, position.data());
  EXPECT_THAT(position, ElementsAre(9));
  ResetBufferPositionAtEnd(shape, /*step=*/4, position.data());
  EXPECT_THAT(position, ElementsAre(6));
}

TEST(ResetBufferPositionTest, TwoDimensional) {
  std::vector<Index> shape{10, 15};
  std::vector<Index> position{42, 43};
  ResetBufferPositionAtBeginning(position);
  EXPECT_THAT(position, ElementsAre(0, 0));
  ResetBufferPositionAtEnd(shape, /*step=*/4, position.data());
  EXPECT_THAT(position, ElementsAre(9, 11));
}

TEST(ResetBufferPositionTest, ThreeDimensional) {
  std::vector<Index> shape{10, 15, 19};
  std::vector<Index> position{42, 43, 44};
  ResetBufferPositionAtBeginning(position);
  EXPECT_THAT(position, ElementsAre(0, 0, 0));
  ResetBufferPositionAtEnd(shape, /*step=*/4, position.data());
  EXPECT_THAT(position, ElementsAre(9, 14, 15));
}

TEST(StepBufferPositionForwardTest, OneDimensional) {
  std::vector<Index> shape{10};
  std::vector<Index> position{0};
  EXPECT_EQ(4, StepBufferPositionForward(
                   shape, /*step=*/4, /*max_buffer_size=*/4, position.data()));
  EXPECT_THAT(position, ElementsAre(4));
  EXPECT_EQ(2, StepBufferPositionForward(
                   shape, /*step=*/4, /*max_buffer_size=*/4, position.data()));
  EXPECT_THAT(position, ElementsAre(8));
  EXPECT_EQ(0, StepBufferPositionForward(
                   shape, /*step=*/2, /*max_buffer_size=*/4, position.data()));
  EXPECT_THAT(position, ElementsAre(10));
}

TEST(StepBufferPositionForwardTest, TwoDimensional) {
  std::vector<Index> shape{2, 10};
  std::vector<Index> position{0, 0};
  EXPECT_EQ(4, StepBufferPositionForward(
                   shape, /*step=*/4, /*max_buffer_size=*/4, position.data()));
  EXPECT_THAT(position, ElementsAre(0, 4));
  EXPECT_EQ(2, StepBufferPositionForward(
                   shape, /*step=*/4, /*max_buffer_size=*/4, position.data()));
  EXPECT_THAT(position, ElementsAre(0, 8));
  EXPECT_EQ(4, StepBufferPositionForward(
                   shape, /*step=*/2, /*max_buffer_size=*/4, position.data()));
  EXPECT_THAT(position, ElementsAre(1, 0));
  EXPECT_EQ(4, StepBufferPositionForward(
                   shape, /*step=*/4, /*max_buffer_size=*/4, position.data()));
  EXPECT_THAT(position, ElementsAre(1, 4));
  EXPECT_EQ(2, StepBufferPositionForward(
                   shape, /*step=*/4, /*max_buffer_size=*/4, position.data()));
  EXPECT_THAT(position, ElementsAre(1, 8));
  EXPECT_EQ(0, StepBufferPositionForward(
                   shape, /*step=*/2, /*max_buffer_size=*/4, position.data()));
  EXPECT_THAT(position, ElementsAre(2, 0));
}

TEST(StepBufferPositionForwardTest, ThreeDimensional) {
  std::vector<Index> shape{2, 2, 6};
  std::vector<Index> position{0, 0, 0};
  EXPECT_EQ(2, StepBufferPositionForward(
                   shape, /*step=*/4, /*max_buffer_size=*/4, position.data()));
  EXPECT_THAT(position, ElementsAre(0, 0, 4));
  EXPECT_EQ(4, StepBufferPositionForward(
                   shape, /*step=*/2, /*max_buffer_size=*/4, position.data()));
  EXPECT_THAT(position, ElementsAre(0, 1, 0));
  EXPECT_EQ(2, StepBufferPositionForward(
                   shape, /*step=*/4, /*max_buffer_size=*/4, position.data()));
  EXPECT_THAT(position, ElementsAre(0, 1, 4));
  EXPECT_EQ(4, StepBufferPositionForward(
                   shape, /*step=*/2, /*max_buffer_size=*/4, position.data()));
  EXPECT_THAT(position, ElementsAre(1, 0, 0));
  EXPECT_EQ(2, StepBufferPositionForward(
                   shape, /*step=*/4, /*max_buffer_size=*/4, position.data()));
  EXPECT_THAT(position, ElementsAre(1, 0, 4));
  EXPECT_EQ(4, StepBufferPositionForward(
                   shape, /*step=*/2, /*max_buffer_size=*/4, position.data()));
  EXPECT_THAT(position, ElementsAre(1, 1, 0));
  EXPECT_EQ(2, StepBufferPositionForward(
                   shape, /*step=*/4, /*max_buffer_size=*/4, position.data()));
  EXPECT_THAT(position, ElementsAre(1, 1, 4));
  EXPECT_EQ(0, StepBufferPositionForward(
                   shape, /*step=*/2, /*max_buffer_size=*/4, position.data()));
  EXPECT_THAT(position, ElementsAre(2, 0, 0));
}

TEST(StepBufferPositionBackwardTest, OneDimensional) {
  std::vector<Index> shape{10};
  std::vector<Index> position{6};
  EXPECT_EQ(4, StepBufferPositionBackward(shape, /*max_buffer_size=*/4,
                                          position.data()));
  EXPECT_THAT(position, ElementsAre(2));
  EXPECT_EQ(2, StepBufferPositionBackward(shape, /*max_buffer_size=*/4,
                                          position.data()));
  EXPECT_THAT(position, ElementsAre(0));
  EXPECT_EQ(0, StepBufferPositionBackward(shape, /*max_buffer_size=*/4,
                                          position.data()));
  EXPECT_THAT(position, ElementsAre(0));
}

TEST(StepBufferPositionBackwardTest, TwoDimensional) {
  std::vector<Index> shape{2, 10};
  std::vector<Index> position{1, 6};
  EXPECT_EQ(4, StepBufferPositionBackward(shape, /*max_buffer_size=*/4,
                                          position.data()));
  EXPECT_THAT(position, ElementsAre(1, 2));
  EXPECT_EQ(2, StepBufferPositionBackward(shape, /*max_buffer_size=*/4,
                                          position.data()));
  EXPECT_THAT(position, ElementsAre(1, 0));
  EXPECT_EQ(4, StepBufferPositionBackward(shape, /*max_buffer_size=*/4,
                                          position.data()));
  EXPECT_THAT(position, ElementsAre(0, 6));
  EXPECT_EQ(4, StepBufferPositionBackward(shape, /*max_buffer_size=*/4,
                                          position.data()));
  EXPECT_THAT(position, ElementsAre(0, 2));
  EXPECT_EQ(2, StepBufferPositionBackward(shape, /*max_buffer_size=*/4,
                                          position.data()));
  EXPECT_THAT(position, ElementsAre(0, 0));
  EXPECT_EQ(0, StepBufferPositionBackward(shape, /*max_buffer_size=*/4,
                                          position.data()));
  EXPECT_THAT(position, ElementsAre(0, 0));
}

TEST(StepBufferPositionBackwardTest, ThreeDimensional) {
  std::vector<Index> shape{2, 2, 10};
  std::vector<Index> position{1, 1, 6};
  EXPECT_EQ(4, StepBufferPositionBackward(shape, /*max_buffer_size=*/4,
                                          position.data()));
  EXPECT_THAT(position, ElementsAre(1, 1, 2));
  EXPECT_EQ(2, StepBufferPositionBackward(shape, /*max_buffer_size=*/4,
                                          position.data()));
  EXPECT_THAT(position, ElementsAre(1, 1, 0));
  EXPECT_EQ(4, StepBufferPositionBackward(shape, /*max_buffer_size=*/4,
                                          position.data()));
  EXPECT_THAT(position, ElementsAre(1, 0, 6));
  EXPECT_EQ(4, StepBufferPositionBackward(shape, /*max_buffer_size=*/4,
                                          position.data()));
  EXPECT_THAT(position, ElementsAre(1, 0, 2));
  EXPECT_EQ(2, StepBufferPositionBackward(shape, /*max_buffer_size=*/4,
                                          position.data()));
  EXPECT_THAT(position, ElementsAre(1, 0, 0));
  EXPECT_EQ(4, StepBufferPositionBackward(shape, /*max_buffer_size=*/4,
                                          position.data()));
  EXPECT_THAT(position, ElementsAre(0, 1, 6));
  EXPECT_EQ(4, StepBufferPositionBackward(shape, /*max_buffer_size=*/4,
                                          position.data()));
  EXPECT_THAT(position, ElementsAre(0, 1, 2));
  EXPECT_EQ(2, StepBufferPositionBackward(shape, /*max_buffer_size=*/4,
                                          position.data()));
  EXPECT_THAT(position, ElementsAre(0, 1, 0));
  EXPECT_EQ(4, StepBufferPositionBackward(shape, /*max_buffer_size=*/4,
                                          position.data()));
  EXPECT_THAT(position, ElementsAre(0, 0, 6));
  EXPECT_EQ(4, StepBufferPositionBackward(shape, /*max_buffer_size=*/4,
                                          position.data()));
  EXPECT_THAT(position, ElementsAre(0, 0, 2));
  EXPECT_EQ(2, StepBufferPositionBackward(shape, /*max_buffer_size=*/4,
                                          position.data()));
  EXPECT_THAT(position, ElementsAre(0, 0, 0));
  EXPECT_EQ(0, StepBufferPositionBackward(shape, /*max_buffer_size=*/4,
                                          position.data()));
  EXPECT_THAT(position, ElementsAre(0, 0, 0));
}

TEST(NDIterationPositionStepperTest, Forward) {
  std::vector<Index> shape({2, 3, 7});
  NDIterationPositionStepper stepper(shape, 4);
  EXPECT_THAT(stepper.shape(), ElementsAreArray(shape));

  using PositionsAndBlockSizes =
      std::vector<std::pair<std::vector<Index>, Index>>;

  PositionsAndBlockSizes expected_results{
      {{0, 0, 0}, 4}, {{0, 0, 4}, 3},  //
      {{0, 1, 0}, 4}, {{0, 1, 4}, 3},  //
      {{0, 2, 0}, 4}, {{0, 2, 4}, 3},  //
      {{1, 0, 0}, 4}, {{1, 0, 4}, 3},  //
      {{1, 1, 0}, 4}, {{1, 1, 4}, 3},  //
      {{1, 2, 0}, 4}, {{1, 2, 4}, 3},  //
  };

  PositionsAndBlockSizes results;
  for (Index block_size = stepper.ResetAtBeginning(); block_size;
       block_size = stepper.StepForward(block_size)) {
    results.emplace_back(
        std::vector(stepper.position().begin(), stepper.position().end()),
        block_size);
  }
  EXPECT_THAT(results, ElementsAreArray(expected_results));
}

TEST(NDIterationPositionStepperTest, Backward) {
  std::vector<Index> shape({2, 3, 7});
  NDIterationPositionStepper stepper(shape, 4);
  EXPECT_THAT(stepper.shape(), ElementsAreArray(shape));

  using PositionsAndBlockSizes =
      std::vector<std::pair<std::vector<Index>, Index>>;

  PositionsAndBlockSizes expected_results{
      {{1, 2, 3}, 4}, {{1, 2, 0}, 3},  //
      {{1, 1, 3}, 4}, {{1, 1, 0}, 3},  //
      {{1, 0, 3}, 4}, {{1, 0, 0}, 3},  //
      {{0, 2, 3}, 4}, {{0, 2, 0}, 3},  //
      {{0, 1, 3}, 4}, {{0, 1, 0}, 3},  //
      {{0, 0, 3}, 4}, {{0, 0, 0}, 3},  //
  };

  PositionsAndBlockSizes results;
  for (Index block_size = stepper.ResetAtEnd(); block_size;
       block_size = stepper.StepBackward()) {
    results.emplace_back(
        std::vector(stepper.position().begin(), stepper.position().end()),
        block_size);
  }
  EXPECT_THAT(results, ElementsAreArray(expected_results));
}

}  // namespace
