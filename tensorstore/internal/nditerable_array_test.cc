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

#include "tensorstore/internal/nditerable_array.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/array.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/arena.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/internal/nditerable.h"
#include "tensorstore/internal/nditerable_buffer_management.h"
#include "tensorstore/internal/nditerable_util.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/byte_strided_pointer.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

namespace {

using tensorstore::Array;
using tensorstore::DimensionIndex;
using tensorstore::Index;
using tensorstore::span;
using tensorstore::Status;
using tensorstore::StridedLayout;
using tensorstore::internal::Arena;
using tensorstore::internal::GetArrayNDIterable;
using tensorstore::internal::IterationBufferKind;
using tensorstore::internal::IterationBufferPointer;
using tensorstore::internal::MultiNDIterator;
using tensorstore::internal::NDIterable;

using DirectionPref = NDIterable::DirectionPref;

// Directly tests the `NDIterable` implementation returned by
// `GetArrayNDIterable`.
TEST(NDIterableArrayTest, Direct) {
  std::uint8_t data[1000];
  // Dimension 0 is contiguous, dimension 2 can be skipped, and dimensions 0 and
  // 1 are reversed.
  Array<std::uint8_t> array(data + 500,
                            StridedLayout<>({6, 3, 4, 5}, {-1, -6, 0, 3}));
  Arena arena;
  auto iterable = GetArrayNDIterable(UnownedToShared(array), &arena);
  {
    std::vector<DirectionPref> direction_prefs(4, DirectionPref::kCanSkip);
    iterable->UpdateDirectionPrefs(direction_prefs.data());
    EXPECT_THAT(direction_prefs,
                ::testing::ElementsAre(
                    DirectionPref::kBackward, DirectionPref::kBackward,
                    DirectionPref::kCanSkip, DirectionPref::kForward));
  }
  // Dimension order is determined from the magnitude of the byte strides.
  EXPECT_GT(iterable->GetDimensionOrder(0, 1), 0);
  EXPECT_LT(iterable->GetDimensionOrder(0, 2), 0);
  EXPECT_GT(iterable->GetDimensionOrder(0, 3), 0);
  EXPECT_LT(iterable->GetDimensionOrder(1, 0), 0);
  EXPECT_LT(iterable->GetDimensionOrder(1, 2), 0);
  EXPECT_LT(iterable->GetDimensionOrder(1, 3), 0);
  EXPECT_GT(iterable->GetDimensionOrder(2, 0), 0);
  EXPECT_GT(iterable->GetDimensionOrder(2, 1), 0);
  EXPECT_LT(iterable->GetDimensionOrder(1, 3), 0);
  EXPECT_LT(iterable->GetDimensionOrder(3, 0), 0);
  EXPECT_GT(iterable->GetDimensionOrder(3, 1), 0);
  EXPECT_LT(iterable->GetDimensionOrder(3, 2), 0);
  // Dimensions `i` and `j` can be combined iff `byte_strides[i] * direction[i]`
  // is equal to `byte_strides[j] * direction[j] * shape[j]`.
  EXPECT_TRUE(iterable->CanCombineDimensions(/*dim_i=*/1, /*dir_i=*/1,
                                             /*dim_j=*/0, /*dir_j=*/1,
                                             /*size_j=*/6));
  EXPECT_TRUE(iterable->CanCombineDimensions(/*dim_i=*/1, /*dir_i=*/-1,
                                             /*dim_j=*/0, /*dir_j=*/-1,
                                             /*size_j=*/6));
  // Not combinable due to incompatible directions
  EXPECT_FALSE(iterable->CanCombineDimensions(/*dim_i=*/1, /*dir_i=*/1,
                                              /*dim_j=*/0, /*dir_j=*/-1,
                                              /*size_j=*/6));
  // Not combinable due to size.
  EXPECT_FALSE(iterable->CanCombineDimensions(/*dim_i=*/1, /*dir_i=*/1,
                                              /*dim_j=*/0, /*dir_j=*/1,
                                              /*size_j=*/5));
  EXPECT_TRUE(iterable->CanCombineDimensions(/*dim_i=*/3, /*dir_i=*/1,
                                             /*dim_j=*/0, /*dir_j=*/-1,
                                             /*size_j=*/3));
  EXPECT_TRUE(iterable->CanCombineDimensions(/*dim_i=*/3, /*dir_i=*/-1,
                                             /*dim_j=*/0, /*dir_j=*/1,
                                             /*size_j=*/3));
  EXPECT_TRUE(iterable->CanCombineDimensions(/*dim_i=*/1, /*dir_i=*/-1,
                                             /*dim_j=*/3, /*dir_j=*/1,
                                             /*size_j=*/2));
  {
    auto c = iterable->GetIterationBufferConstraint(
        {/*.shape=*/span<const Index>({6, 3, 4, 5}),
         /*.directions=*/span<const int>({1, 1, 1, 1}),
         /*.iteration_dimensions=*/span<const DimensionIndex>({0, 1, 2, 3}),
         /*.iteration_shape=*/span<const Index>({6, 3, 4, 5})});
    // `kStrided` because the last iteration dimension has a byte stride of
    // `array.byte_strides()[3] == 3`, which is not equal to the element size of
    // `1`.
    EXPECT_EQ(IterationBufferKind::kStrided, c.min_buffer_kind);
    EXPECT_FALSE(c.external);
  }

  {
    auto c = iterable->GetIterationBufferConstraint(
        {/*.shape=*/span<const Index>({6, 3, 4, 5}),
         /*.directions=*/span<const int>({1, 1, 1, 1}),
         /*.iteration_dimensions=*/span<const DimensionIndex>({1, 3, 0}),
         /*.iteration_shape=*/span<const Index>({3, 5, 6})});
    // `kStrided` because the last iteration dimension has a byte stride of
    // `array.byte_strides()[0] == -1`, which is not equal to the element size
    // of `1`.
    EXPECT_EQ(IterationBufferKind::kStrided, c.min_buffer_kind);
    EXPECT_FALSE(c.external);
  }

  {
    auto c = iterable->GetIterationBufferConstraint(
        {/*.shape=*/span<const Index>({6, 3, 4, 5}),
         /*.directions=*/span<const int>({-1, -1, 0, 1}),
         /*.iteration_dimensions=*/span<const DimensionIndex>({1, 3, 0}),
         /*.iteration_shape=*/span<const Index>({3, 5, 6})});
    // `kContiguous` because the last iteration dimension has a byte stride of
    // `-1 * array.byte_strides()[0] == 1`, which is equal to the element size
    // of `1`.
    EXPECT_EQ(IterationBufferKind::kContiguous, c.min_buffer_kind);
    EXPECT_FALSE(c.external);
  }

  EXPECT_EQ(
      0, iterable->GetWorkingMemoryBytesPerElement(
             {/*.shape=*/span<const Index>({6, 3, 4, 5}),
              /*.directions=*/span<const int>({-1, -1, 0, 1}),
              /*.iteration_dimensions=*/span<const DimensionIndex>({1, 3, 0}),
              /*.iteration_shape=*/span<const Index>({3, 5, 6})},
             IterationBufferKind::kContiguous));
  EXPECT_EQ(
      0, iterable->GetWorkingMemoryBytesPerElement(
             {/*.shape=*/span<const Index>({6, 3, 4, 5}),
              /*.directions=*/span<const int>({-1, -1, 0, 1}),
              /*.iteration_dimensions=*/span<const DimensionIndex>({1, 3, 0}),
              /*.iteration_shape=*/span<const Index>({3, 5, 6})},
             IterationBufferKind::kStrided));
  EXPECT_EQ(
      sizeof(Index),
      iterable->GetWorkingMemoryBytesPerElement(
          {/*.shape=*/span<const Index>({6, 3, 4, 5}),
           /*.directions=*/span<const int>({-1, -1, 0, 1}),
           /*.iteration_dimensions=*/span<const DimensionIndex>({1, 3, 0}),
           /*.iteration_shape=*/span<const Index>({3, 5, 6})},
          IterationBufferKind::kIndexed));

  {
    auto iterator = iterable->GetIterator(
        {{{/*.shape=*/span<const Index>({6, 3, 4, 5}),
           /*.directions=*/span<const int>({-1, -1, 0, 1}),
           /*.iteration_dimensions=*/span<const DimensionIndex>({1, 3, 0}),
           /*.iteration_shape=*/span<const Index>({3, 5, 6})},
          /*.block_size=*/3},
         /*.buffer_kind=*/IterationBufferKind::kContiguous});
    IterationBufferPointer pointer;
    Status status;
    EXPECT_EQ(3, iterator->GetBlock(span<const Index>({2, 3, 1}), 3, &pointer,
                                    &status));
    EXPECT_EQ(&array((6 - 1) - 1, (3 - 1) - 2, 0, 3), pointer.pointer.get());
    EXPECT_EQ(1, pointer.byte_stride);
    EXPECT_EQ(Status(), status);
  }

  {
    auto iterator = iterable->GetIterator(
        {{{/*.shape=*/span<const Index>({6, 3, 4, 5}),
           /*.directions=*/span<const int>({-1, -1, 0, 1}),
           /*.iteration_dimensions=*/span<const DimensionIndex>({1, 3, 0}),
           /*.iteration_shape=*/span<const Index>({3, 5, 6})},
          /*.block_size=*/3},
         /*.buffer_kind=*/IterationBufferKind::kIndexed});
    IterationBufferPointer pointer;
    Status status;
    EXPECT_EQ(3, iterator->GetBlock(span<const Index>({2, 3, 1}), 3, &pointer,
                                    &status));
    EXPECT_EQ(&array((6 - 1) - 1, (3 - 1) - 2, 0, 3), pointer.pointer.get());
    EXPECT_THAT(span<const Index>(pointer.byte_offsets, 3),
                ::testing::ElementsAre(0, 1, 2));
    EXPECT_EQ(Status(), status);
  }
}

TEST(NDIterableArrayTest, RankZero) {
  auto array = tensorstore::MakeScalarArray<int>(5);

  Arena arena;
  auto iterable = GetArrayNDIterable(array, &arena);
  MultiNDIterator<1, /*Full=*/true> multi_iterator(span<const Index>{}, {},
                                                   {{iterable.get()}}, &arena);

  EXPECT_THAT(multi_iterator.iteration_dimensions, ::testing::ElementsAre(-1));
  EXPECT_THAT(multi_iterator.directions, ::testing::ElementsAre());
  EXPECT_THAT(multi_iterator.shape, ::testing::ElementsAre());
  EXPECT_THAT(multi_iterator.iteration_shape, ::testing::ElementsAre(1));
  EXPECT_THAT(multi_iterator.full_iteration_dimensions,
              ::testing::ElementsAre());
  EXPECT_EQ(IterationBufferKind::kContiguous, multi_iterator.buffer_kind);
  EXPECT_EQ(false, multi_iterator.empty);
  EXPECT_EQ(1, multi_iterator.block_size);

  EXPECT_EQ(1, multi_iterator.ResetAtBeginning());
  Status status;
  EXPECT_TRUE(multi_iterator.GetBlock(1, &status));
  EXPECT_THAT(multi_iterator.position(), ::testing::ElementsAre(0));
  EXPECT_EQ(Status(), status);
  EXPECT_EQ(array.data(), multi_iterator.block_pointers()[0].pointer);
  EXPECT_EQ(0, multi_iterator.block_pointers()[0].byte_stride);
  EXPECT_EQ(0, multi_iterator.StepForward(1));
  EXPECT_THAT(multi_iterator.position(), ::testing::ElementsAre(1));
}

TEST(NDIterableArrayTest, RankOne) {
  auto array = tensorstore::MakeArray<int>({1, 2, 3, 4, 5});

  Arena arena;
  auto iterable = GetArrayNDIterable(array, &arena);
  MultiNDIterator<1, /*Full=*/true> multi_iterator(span<const Index>({5}), {},
                                                   {{iterable.get()}}, &arena);

  EXPECT_THAT(multi_iterator.shape, ::testing::ElementsAre(5));
  EXPECT_THAT(multi_iterator.iteration_dimensions, ::testing::ElementsAre(0));
  EXPECT_THAT(multi_iterator.directions, ::testing::ElementsAre(1));
  EXPECT_THAT(multi_iterator.iteration_shape, ::testing::ElementsAre(5));
  EXPECT_THAT(multi_iterator.full_iteration_dimensions,
              ::testing::ElementsAre(0));
  EXPECT_EQ(IterationBufferKind::kContiguous, multi_iterator.buffer_kind);
  EXPECT_EQ(false, multi_iterator.empty);
  EXPECT_EQ(5, multi_iterator.block_size);

  EXPECT_EQ(5, multi_iterator.ResetAtBeginning());
  Status status;
  EXPECT_TRUE(multi_iterator.GetBlock(5, &status));
  EXPECT_THAT(multi_iterator.position(), ::testing::ElementsAre(0));
  EXPECT_EQ(Status(), status);
  EXPECT_EQ(array.data(), multi_iterator.block_pointers()[0].pointer);
  EXPECT_EQ(sizeof(int), multi_iterator.block_pointers()[0].byte_stride);
  EXPECT_EQ(0, multi_iterator.StepForward(5));
  EXPECT_THAT(multi_iterator.position(), ::testing::ElementsAre(5));
}

TEST(NDIterableArrayTest, RankTwoContiguous) {
  auto array = tensorstore::MakeArray<int>({{1, 2, 3}, {4, 5, 6}});

  Arena arena;
  auto iterable = GetArrayNDIterable(array, &arena);
  MultiNDIterator<1, /*Full=*/true> multi_iterator(array.shape(), {},
                                                   {{iterable.get()}}, &arena);

  EXPECT_THAT(multi_iterator.shape, ::testing::ElementsAre(2, 3));
  EXPECT_THAT(multi_iterator.iteration_dimensions, ::testing::ElementsAre(1));
  EXPECT_THAT(multi_iterator.directions, ::testing::ElementsAre(1, 1));
  EXPECT_THAT(multi_iterator.iteration_shape, ::testing::ElementsAre(6));
  EXPECT_THAT(multi_iterator.full_iteration_dimensions,
              ::testing::ElementsAre(0, 1));
  EXPECT_EQ(IterationBufferKind::kContiguous, multi_iterator.buffer_kind);
  EXPECT_EQ(false, multi_iterator.empty);
  EXPECT_EQ(6, multi_iterator.block_size);

  EXPECT_EQ(6, multi_iterator.ResetAtBeginning());
  Status status;
  EXPECT_TRUE(multi_iterator.GetBlock(6, &status));
  EXPECT_THAT(multi_iterator.position(), ::testing::ElementsAre(0));
  EXPECT_EQ(Status(), status);
  EXPECT_EQ(array.data(), multi_iterator.block_pointers()[0].pointer);
  EXPECT_EQ(sizeof(int), multi_iterator.block_pointers()[0].byte_stride);
  EXPECT_EQ(0, multi_iterator.StepForward(6));
  EXPECT_THAT(multi_iterator.position(), ::testing::ElementsAre(6));
}

TEST(NDIterableArrayTest, RankTwoTranspose) {
  auto array = tensorstore::MakeArray<int>({{1, 2, 3}, {4, 5, 6}});

  Arena arena;
  auto iterable = GetArrayNDIterable(array, &arena);
  MultiNDIterator<1, /*Full=*/true> multi_iterator(
      array.shape(), tensorstore::fortran_order, {{iterable.get()}}, &arena);

  EXPECT_THAT(multi_iterator.shape, ::testing::ElementsAre(2, 3));
  EXPECT_THAT(multi_iterator.iteration_dimensions,
              ::testing::ElementsAre(1, 0));
  EXPECT_THAT(multi_iterator.directions, ::testing::ElementsAre(1, 1));
  EXPECT_THAT(multi_iterator.iteration_shape, ::testing::ElementsAre(3, 2));
  EXPECT_THAT(multi_iterator.full_iteration_dimensions,
              ::testing::ElementsAre(1, 0));
  EXPECT_EQ(IterationBufferKind::kStrided, multi_iterator.buffer_kind);
  EXPECT_EQ(false, multi_iterator.empty);
  EXPECT_EQ(2, multi_iterator.block_size);

  EXPECT_EQ(2, multi_iterator.ResetAtBeginning());
  Status status;
  EXPECT_THAT(multi_iterator.position(), ::testing::ElementsAre(0, 0));
  EXPECT_TRUE(multi_iterator.GetBlock(2, &status));
  EXPECT_EQ(Status(), status);
  EXPECT_EQ(&array(0, 0), multi_iterator.block_pointers()[0].pointer);
  EXPECT_EQ(sizeof(int) * 3, multi_iterator.block_pointers()[0].byte_stride);
  EXPECT_EQ(2, multi_iterator.StepForward(2));
  EXPECT_THAT(multi_iterator.position(), ::testing::ElementsAre(1, 0));
  EXPECT_TRUE(multi_iterator.GetBlock(2, &status));
  EXPECT_EQ(Status(), status);
  EXPECT_EQ(&array(0, 1), multi_iterator.block_pointers()[0].pointer);
  EXPECT_EQ(sizeof(int) * 3, multi_iterator.block_pointers()[0].byte_stride);
  EXPECT_EQ(2, multi_iterator.StepForward(2));
  EXPECT_THAT(multi_iterator.position(), ::testing::ElementsAre(2, 0));
  EXPECT_TRUE(multi_iterator.GetBlock(2, &status));
  EXPECT_EQ(Status(), status);
  EXPECT_EQ(&array(0, 2), multi_iterator.block_pointers()[0].pointer);
  EXPECT_EQ(sizeof(int) * 3, multi_iterator.block_pointers()[0].byte_stride);
  EXPECT_EQ(0, multi_iterator.StepForward(2));
  EXPECT_THAT(multi_iterator.position(), ::testing::ElementsAre(3, 0));
}

// Tests that dimensions with size of 1 are skipped even if the `byte_stride` is
// non-zero.
TEST(NDIterableArrayTest, SkipSize1Dimension) {
  unsigned char data[300];

  Arena arena;

  Array<unsigned char> array = {&data[150],
                                StridedLayout<>({2, 1, 3}, {5, 10, -20})};
  auto iterable = GetArrayNDIterable(UnownedToShared(array), &arena);

  MultiNDIterator<1, /*Full=*/true> multi_iterator(array.shape(), {},
                                                   {{iterable.get()}}, &arena);
  EXPECT_THAT(multi_iterator.shape, ::testing::ElementsAre(2, 1, 3));
  EXPECT_THAT(multi_iterator.iteration_dimensions,
              ::testing::ElementsAre(2, 0));
  EXPECT_THAT(multi_iterator.directions, ::testing::ElementsAre(1, 0, -1));
  EXPECT_THAT(multi_iterator.iteration_shape, ::testing::ElementsAre(3, 2));
  EXPECT_THAT(multi_iterator.full_iteration_dimensions,
              ::testing::ElementsAre(1, 2, 0));
}

TEST(NDIterableArrayTest, SkipZeroByteStride) {
  unsigned char data[300];

  Arena arena;

  Array<unsigned char> array = {&data[150], StridedLayout<>({2, 3}, {5, 0})};
  auto iterable = GetArrayNDIterable(UnownedToShared(array), &arena);

  MultiNDIterator<1, /*Full=*/true> multi_iterator(
      array.shape(), tensorstore::skip_repeated_elements, {{iterable.get()}},
      &arena);
  EXPECT_THAT(multi_iterator.shape, ::testing::ElementsAre(2, 3));
  EXPECT_THAT(multi_iterator.iteration_dimensions, ::testing::ElementsAre(0));
  EXPECT_THAT(multi_iterator.directions, ::testing::ElementsAre(1, 0));
  EXPECT_THAT(multi_iterator.iteration_shape, ::testing::ElementsAre(2));
  EXPECT_THAT(multi_iterator.full_iteration_dimensions,
              ::testing::ElementsAre(1, 0));
}

TEST(NDIterableArrayTest, FortranOrderArray) {
  auto array =
      tensorstore::AllocateArray<int>({2, 3}, tensorstore::fortran_order);

  Arena arena;
  auto iterable = GetArrayNDIterable(array, &arena);
  MultiNDIterator<1, /*Full=*/true> multi_iterator(
      array.shape(), tensorstore::skip_repeated_elements, {{iterable.get()}},
      &arena);
  EXPECT_THAT(multi_iterator.shape, ::testing::ElementsAre(2, 3));
  EXPECT_THAT(multi_iterator.iteration_dimensions, ::testing::ElementsAre(0));
  EXPECT_THAT(multi_iterator.directions, ::testing::ElementsAre(1, 1));
  EXPECT_THAT(multi_iterator.iteration_shape, ::testing::ElementsAre(6));
  EXPECT_THAT(multi_iterator.full_iteration_dimensions,
              ::testing::ElementsAre(1, 0));
}

TEST(NDIterableArrayTest, ReversedDimensions) {
  auto orig_array = tensorstore::AllocateArray<int>({3, 4, 5});
  auto orig_shape = orig_array.shape();
  auto orig_strides = orig_array.byte_strides();
  Array<int> array(
      &orig_array(0, 4 - 1, 5 - 1),
      StridedLayout<>({orig_shape[2], orig_shape[0], orig_shape[1]},
                      {-orig_strides[2], orig_strides[0], -orig_strides[1]}));
  Arena arena;
  auto iterable = GetArrayNDIterable(UnownedToShared(array), &arena);
  MultiNDIterator<1, /*Full=*/true> multi_iterator(
      array.shape(), tensorstore::skip_repeated_elements, {{iterable.get()}},
      &arena);

  EXPECT_THAT(multi_iterator.shape, ::testing::ElementsAre(5, 3, 4));
  EXPECT_THAT(multi_iterator.iteration_dimensions, ::testing::ElementsAre(0));
  EXPECT_THAT(multi_iterator.directions, ::testing::ElementsAre(-1, 1, -1));
  EXPECT_THAT(multi_iterator.iteration_shape,
              ::testing::ElementsAre(3 * 4 * 5));
  EXPECT_THAT(multi_iterator.full_iteration_dimensions,
              ::testing::ElementsAre(1, 2, 0));
  EXPECT_EQ(IterationBufferKind::kContiguous, multi_iterator.buffer_kind);
  EXPECT_EQ(false, multi_iterator.empty);
  EXPECT_EQ(3 * 4 * 5, multi_iterator.block_size);

  EXPECT_EQ(3 * 4 * 5, multi_iterator.ResetAtBeginning());
  Status status;
  EXPECT_THAT(multi_iterator.position(), ::testing::ElementsAre(0));
  EXPECT_TRUE(multi_iterator.GetBlock(3 * 4 * 5, &status));
  EXPECT_EQ(Status(), status);
  EXPECT_EQ(orig_array.byte_strided_pointer(),
            multi_iterator.block_pointers()[0].pointer);
  EXPECT_EQ(sizeof(int), multi_iterator.block_pointers()[0].byte_stride);
}

TEST(NDIterableArrayTest, MultipleArrays) {
  auto array_a = tensorstore::AllocateArray<int>({2, 3}, tensorstore::c_order);
  auto array_b =
      tensorstore::AllocateArray<int>({2, 3}, tensorstore::fortran_order);
  Arena arena;
  auto iterable_a = GetArrayNDIterable(array_a, &arena);
  auto iterable_b = GetArrayNDIterable(array_b, &arena);
  MultiNDIterator<2, /*Full=*/true> multi_iterator(
      array_a.shape(), tensorstore::skip_repeated_elements,
      {{iterable_a.get(), iterable_b.get()}}, &arena);

  EXPECT_THAT(multi_iterator.shape, ::testing::ElementsAre(2, 3));
  EXPECT_THAT(multi_iterator.iteration_dimensions,
              ::testing::ElementsAre(0, 1));
  EXPECT_THAT(multi_iterator.directions, ::testing::ElementsAre(1, 1));
  EXPECT_THAT(multi_iterator.iteration_shape, ::testing::ElementsAre(2, 3));
  EXPECT_THAT(multi_iterator.full_iteration_dimensions,
              ::testing::ElementsAre(0, 1));
  EXPECT_EQ(IterationBufferKind::kStrided, multi_iterator.buffer_kind);
  EXPECT_EQ(false, multi_iterator.empty);
  EXPECT_EQ(3, multi_iterator.block_size);
  EXPECT_EQ(3, multi_iterator.ResetAtBeginning());
  Status status;
  EXPECT_THAT(multi_iterator.position(), ::testing::ElementsAre(0, 0));
  EXPECT_TRUE(multi_iterator.GetBlock(3, &status));
  EXPECT_EQ(Status(), status);
  EXPECT_EQ(&array_a(0, 0), multi_iterator.block_pointers()[0].pointer);
  EXPECT_EQ(&array_b(0, 0), multi_iterator.block_pointers()[1].pointer);
  EXPECT_EQ(sizeof(int), multi_iterator.block_pointers()[0].byte_stride);
  EXPECT_EQ(sizeof(int) * 2, multi_iterator.block_pointers()[1].byte_stride);

  EXPECT_EQ(3, multi_iterator.StepForward(3));
  EXPECT_THAT(multi_iterator.position(), ::testing::ElementsAre(1, 0));
  EXPECT_TRUE(multi_iterator.GetBlock(3, &status));
  EXPECT_EQ(Status(), status);
  EXPECT_EQ(&array_a(1, 0), multi_iterator.block_pointers()[0].pointer);
  EXPECT_EQ(&array_b(1, 0), multi_iterator.block_pointers()[1].pointer);
  EXPECT_EQ(sizeof(int), multi_iterator.block_pointers()[0].byte_stride);
  EXPECT_EQ(sizeof(int) * 2, multi_iterator.block_pointers()[1].byte_stride);
}

}  // namespace
