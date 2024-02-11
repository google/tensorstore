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

#include "tensorstore/internal/nditerable_transformed_array.h"

#include <stddef.h>

#include <array>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorstore/array.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/index_space/transformed_array.h"
#include "tensorstore/internal/arena.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/internal/nditerable.h"
#include "tensorstore/internal/nditerable_buffer_management.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/element_pointer.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::AllocateArray;
using ::tensorstore::Index;
using ::tensorstore::IndexTransformBuilder;
using ::tensorstore::kImplicit;
using ::tensorstore::MakeArray;
using ::tensorstore::MatchesStatus;
using ::tensorstore::Result;
using ::tensorstore::Shared;
using ::tensorstore::SharedArray;
using ::tensorstore::skip_repeated_elements;
using ::tensorstore::StridedLayout;
using ::tensorstore::TransformedArray;
using ::tensorstore::internal::Arena;
using ::tensorstore::internal::GetTransformedArrayNDIterable;
using ::tensorstore::internal::IterationBufferKind;
using ::tensorstore::internal::IterationBufferShape;
using ::tensorstore::internal::MultiNDIterator;
using ::tensorstore::internal::NDIterable;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::FieldsAre;
using ::testing::Pair;

using IterationTrace = std::vector<void*>;

template <typename... Element>
std::pair<std::array<IterationTrace, sizeof...(Element)>, absl::Status>
GetIterationTrace(
    MultiNDIterator<sizeof...(Element), /*Full=*/true>* multi_iterator) {
  std::pair<std::array<IterationTrace, sizeof...(Element)>, absl::Status>
      result;
  for (auto block_shape = multi_iterator->ResetAtBeginning();
       block_shape[0] && block_shape[1];
       block_shape = multi_iterator->StepForward(block_shape)) {
    if (!multi_iterator->GetBlock(block_shape, &result.second)) {
      break;
    }
    ptrdiff_t i = 0;
    const auto unused = {(
        [&] {
          const auto get_trace_func = [](void* ptr, IterationTrace* trace) {
            trace->push_back(ptr);
          };
          tensorstore::internal::ElementwiseFunction<1, IterationTrace*> func =
              tensorstore::internal::SimpleElementwiseFunction<
                  decltype(get_trace_func)(Element), IterationTrace*>();
          func[multi_iterator->buffer_kind](nullptr, block_shape,
                                            multi_iterator->block_pointers()[i],
                                            &result.first[i]);
          ++i;
        }(),
        0)...};
    (void)unused;
  }
  return result;
}

template <size_t N>
using BlockTrace =
    std::vector<std::tuple<std::vector<Index>, IterationBufferShape,
                           std::array<IterationTrace, N>>>;

template <typename... Element>
std::pair<BlockTrace<sizeof...(Element)>, absl::Status> GetBlockTrace(
    MultiNDIterator<sizeof...(Element), /*Full=*/true>* multi_iterator) {
  std::pair<BlockTrace<sizeof...(Element)>, absl::Status> result;
  for (auto block_shape = multi_iterator->ResetAtBeginning();
       block_shape[0] && block_shape[1];
       block_shape = multi_iterator->StepForward(block_shape)) {
    if (!multi_iterator->GetBlock(block_shape, &result.second)) {
      break;
    }
    auto& [position, shape, traces] = result.first.emplace_back();
    position.assign(multi_iterator->position().begin(),
                    multi_iterator->position().end());
    shape = block_shape;
    ptrdiff_t i = 0;
    const auto unused = {(
        [&, traces_ptr = &traces[i]] {
          const auto get_trace_func = [](void* ptr, IterationTrace* trace) {
            trace->push_back(ptr);
          };
          tensorstore::internal::ElementwiseFunction<1, IterationTrace*> func =
              tensorstore::internal::SimpleElementwiseFunction<
                  decltype(get_trace_func)(Element), IterationTrace*>();
          func[multi_iterator->buffer_kind](nullptr, block_shape,
                                            multi_iterator->block_pointers()[i],
                                            traces_ptr);
          ++i;
        }(),
        0)...};
    (void)unused;
  }
  return result;
}

// The parameter determines whether the array is specified as a
// `TransformedArray` directly or as an ``(array, transform)`` pair of
// arguments, in order to test both variants.
class MaybeDirectTest : public ::testing::TestWithParam<bool> {
 protected:
  Arena arena;

  Result<NDIterable::Ptr> GetMaybeDirectTransformedArrayNDIterable(
      tensorstore::SharedOffsetArrayView<const void> array,
      tensorstore::IndexTransformView<> transform) {
    if (GetParam()) {
      TENSORSTORE_ASSIGN_OR_RETURN(auto transformed_array,
                                   MakeTransformedArray(array, transform));
      return GetTransformedArrayNDIterable(std::move(transformed_array),
                                           &arena);
    } else {
      return GetTransformedArrayNDIterable(std::move(array), transform, &arena);
    }
  }
};

INSTANTIATE_TEST_SUITE_P(Indirect, MaybeDirectTest, ::testing::Values(true));
INSTANTIATE_TEST_SUITE_P(Direct, MaybeDirectTest, ::testing::Values(false));

// Test the case of an array with no index array input dimensions (simply
// forwards to `GetArrayNDIterable`).
TEST(NDIterableTransformedArrayTest, Strided) {
  Arena arena;
  auto a = AllocateArray<int>({2, 3});
  auto ta = (a | tensorstore::Dims(1).SizedInterval(0, 2, 2)).value();

  auto iterable = GetTransformedArrayNDIterable(ta, &arena).value();
  MultiNDIterator<1, /*Full=*/true> multi_iterator(
      ta.shape(), skip_repeated_elements, {{iterable.get()}}, &arena);
  EXPECT_EQ(IterationBufferKind::kStrided, multi_iterator.buffer_kind);
  EXPECT_THAT(multi_iterator.iteration_dimensions, ElementsAre(0, 1));
  EXPECT_THAT(
      GetIterationTrace<int>(&multi_iterator),
      Pair(ElementsAre(ElementsAre(&a(0, 0), &a(0, 2), &a(1, 0), &a(1, 2))),
           absl::OkStatus()));
}

TEST(NDIterableTransformedArrayTest, SingleIndexedDimension) {
  Arena arena;
  auto a = AllocateArray<int>({4});
  auto ta = (a | tensorstore::Dims(0).OuterIndexArraySlice(
                     MakeArray<Index>({1, 2, 3, 0})))
                .value();
  auto iterable = GetTransformedArrayNDIterable(ta, &arena).value();
  EXPECT_EQ(tensorstore::dtype_v<int>, iterable->dtype());
  MultiNDIterator<1, /*Full=*/true> multi_iterator(
      ta.shape(), skip_repeated_elements, {{iterable.get()}}, &arena);
  EXPECT_EQ(IterationBufferKind::kIndexed, multi_iterator.buffer_kind);
  EXPECT_THAT(multi_iterator.iteration_dimensions, ElementsAre(-1, 0));
  EXPECT_THAT(GetIterationTrace<int>(&multi_iterator),
              Pair(ElementsAre(ElementsAre(&a(1), &a(2), &a(3), &a(0))),
                   absl::OkStatus()));
}

// Test the case of an array with both index array input dimensions and purely
// strided input dimensions.  The inner loop dimension will not be an index
// array input dimension, but it will still use an indexed buffer.
TEST(NDIterableTransformedArrayTest,
     OneStridedOneIndexedDimensionIndexedBuffer) {
  Arena arena;
  auto a = AllocateArray<int>({2, 3});
  auto ta = (a | tensorstore::Dims(1).OuterIndexArraySlice(
                     MakeArray<Index>({0, 2, 1, 1})))
                .value();

  auto iterable = GetTransformedArrayNDIterable(ta, &arena).value();
  MultiNDIterator<1, /*Full=*/true> multi_iterator(
      ta.shape(), skip_repeated_elements, {{iterable.get()}}, &arena);
  EXPECT_THAT(multi_iterator.iteration_dimensions, ElementsAre(1, 0));
  EXPECT_THAT(multi_iterator.iteration_shape, ElementsAre(4, 2));
  EXPECT_EQ(IterationBufferKind::kIndexed, multi_iterator.buffer_kind);
  EXPECT_THAT(
      GetIterationTrace<int>(&multi_iterator),
      Pair(ElementsAre(ElementsAre(&a(0, 0), &a(1, 0), &a(0, 2), &a(1, 2),
                                   &a(0, 1), &a(1, 1), &a(0, 1), &a(1, 1))),
           absl::OkStatus()));
}

// Test the case of an array with 1 index array dimension and 2 strided
// dimensions, the innermost of which will be contiguous.
TEST(NDIterableTransformedArrayTest,
     TwoStridedOneIndexedDimensionContiguousBuffer) {
  Arena arena;
  auto a = AllocateArray<int>({2, 3, 2});
  auto ta = (a | tensorstore::Dims(1).OuterIndexArraySlice(
                     MakeArray<Index>({0, 2, 1, 1})))
                .value();

  auto iterable = GetTransformedArrayNDIterable(ta, &arena).value();
  MultiNDIterator<1, /*Full=*/true> multi_iterator(
      ta.shape(), skip_repeated_elements, {{iterable.get()}}, &arena);
  EXPECT_THAT(multi_iterator.iteration_dimensions, ElementsAre(1, 0, 2));
  EXPECT_THAT(multi_iterator.iteration_shape, ElementsAre(4, 2, 2));
  EXPECT_EQ(IterationBufferKind::kContiguous, multi_iterator.buffer_kind);
  EXPECT_THAT(
      GetIterationTrace<int>(&multi_iterator),
      Pair(ElementsAre(ElementsAreArray(  //
               {
                   &a(0, 0, 0), &a(0, 0, 1), &a(1, 0, 0), &a(1, 0, 1),  //
                   &a(0, 2, 0), &a(0, 2, 1), &a(1, 2, 0), &a(1, 2, 1),  //
                   &a(0, 1, 0), &a(0, 1, 1), &a(1, 1, 0), &a(1, 1, 1),  //
                   &a(0, 1, 0), &a(0, 1, 1), &a(1, 1, 0), &a(1, 1, 1)   //
               })),
           absl::OkStatus()));
}

// Test the case of an array with 1 index array dimension and 2 strided
// dimensions, the innermost of which will be non-contiguous.
TEST(NDIterableTransformedArrayTest,
     TwoStridedOneIndexedDimensionStridedBuffer) {
  Arena arena;
  auto a = AllocateArray<int>({2, 3, 4});
  auto ta = (a | tensorstore::Dims(2).Stride(2) |
             tensorstore::Dims(1).OuterIndexArraySlice(
                 MakeArray<Index>({0, 2, 1, 1})))
                .value();

  auto iterable = GetTransformedArrayNDIterable(ta, &arena).value();
  MultiNDIterator<1, /*Full=*/true> multi_iterator(
      ta.shape(), skip_repeated_elements, {{iterable.get()}}, &arena);
  EXPECT_THAT(multi_iterator.iteration_dimensions, ElementsAre(1, 0, 2));
  EXPECT_THAT(multi_iterator.iteration_shape, ElementsAre(4, 2, 2));
  EXPECT_EQ(IterationBufferKind::kStrided, multi_iterator.buffer_kind);
  EXPECT_THAT(
      GetIterationTrace<int>(&multi_iterator),
      Pair(ElementsAre(ElementsAreArray(  //
               {
                   &a(0, 0, 0), &a(0, 0, 2), &a(1, 0, 0), &a(1, 0, 2),  //
                   &a(0, 2, 0), &a(0, 2, 2), &a(1, 2, 0), &a(1, 2, 2),  //
                   &a(0, 1, 0), &a(0, 1, 2), &a(1, 1, 0), &a(1, 1, 2),  //
                   &a(0, 1, 0), &a(0, 1, 2), &a(1, 1, 0), &a(1, 1, 2)   //
               })),
           absl::OkStatus()));
}

TEST(NDIterableTransformedArrayTest,
     TwoStridedOneIndexedDimensionIndexedBuffer) {
  Arena arena;
  auto a = AllocateArray<int>({2, 3, 2});
  auto ta = (a | tensorstore::Dims(1).OuterIndexArraySlice(
                     MakeArray<Index>({0, 2, 1, 1})))
                .value();
  auto tb =
      (a | tensorstore::Dims(0).OuterIndexArraySlice(MakeArray<Index>({0, 1})) |
       tensorstore::Dims(1).OuterIndexArraySlice(
           MakeArray<Index>({0, 2, 1, 1})))
          .value();

  auto iterable1 = GetTransformedArrayNDIterable(ta, &arena).value();
  auto iterable2 = GetTransformedArrayNDIterable(tb, &arena).value();
  MultiNDIterator<2, /*Full=*/true> multi_iterator(
      ta.shape(), skip_repeated_elements, {{iterable1.get(), iterable2.get()}},
      &arena);
  EXPECT_THAT(multi_iterator.iteration_dimensions, ElementsAre(1, 0, 2));
  EXPECT_THAT(multi_iterator.iteration_shape, ElementsAre(4, 2, 2));
  EXPECT_EQ(IterationBufferKind::kIndexed, multi_iterator.buffer_kind);
  auto element_matcher = ElementsAreArray(  //
      {
          &a(0, 0, 0), &a(0, 0, 1), &a(1, 0, 0), &a(1, 0, 1),  //
          &a(0, 2, 0), &a(0, 2, 1), &a(1, 2, 0), &a(1, 2, 1),  //
          &a(0, 1, 0), &a(0, 1, 1), &a(1, 1, 0), &a(1, 1, 1),  //
          &a(0, 1, 0), &a(0, 1, 1), &a(1, 1, 0), &a(1, 1, 1)   //
      });
  EXPECT_THAT(
      (GetIterationTrace<int, int>(&multi_iterator)),
      Pair(ElementsAre(element_matcher, element_matcher), absl::OkStatus()));
}

// Test the case of an array with both an index array input dimension and a
// purely strided input dimension, where the strided input dimension is
// reversed.  The inner loop dimension will not be an index array input
// dimension.
TEST(NDIterableTransformedArrayTest, IndexedAndReversedStrided) {
  Arena arena;
  auto a = AllocateArray<int>({2, 3});
  auto ta = (a |
             tensorstore::Dims(1).OuterIndexArraySlice(
                 MakeArray<Index>({0, 2, 1, 1})) |
             tensorstore::Dims(0).SizedInterval(kImplicit, kImplicit, -1))
                .value();

  auto iterable = GetTransformedArrayNDIterable(ta, &arena).value();
  MultiNDIterator<1, /*Full=*/true> multi_iterator(
      ta.shape(), skip_repeated_elements, {{iterable.get()}}, &arena);
  EXPECT_THAT(multi_iterator.iteration_dimensions, ElementsAre(1, 0));
  EXPECT_THAT(multi_iterator.directions, ElementsAre(-1, 1));
  EXPECT_THAT(multi_iterator.iteration_shape, ElementsAre(4, 2));
  EXPECT_THAT(
      GetIterationTrace<int>(&multi_iterator),
      Pair(ElementsAre(ElementsAre(&a(0, 0), &a(1, 0), &a(0, 2), &a(1, 2),
                                   &a(0, 1), &a(1, 1), &a(0, 1), &a(1, 1))),
           absl::OkStatus()));
}

// Tests that input dimensions on which index array maps depend can be combined.
TEST(NDIterableTransformedArrayTest, IndexedCombine) {
  Arena arena;
  auto a = AllocateArray<int>({2, 3});
  auto ta = (a | tensorstore::Dims(1).OuterIndexArraySlice(
                     MakeArray<Index>({{0, 2}, {2, 0}})))
                .value();

  auto iterable = GetTransformedArrayNDIterable(ta, &arena).value();
  MultiNDIterator<1, /*Full=*/true> multi_iterator(
      ta.shape(), skip_repeated_elements, {{iterable.get()}}, &arena);
  EXPECT_THAT(multi_iterator.iteration_dimensions, ElementsAre(2, 0));
  EXPECT_THAT(
      GetIterationTrace<int>(&multi_iterator),
      Pair(ElementsAre(ElementsAre(&a(0, 0), &a(1, 0), &a(0, 2), &a(1, 2),
                                   &a(0, 2), &a(1, 2), &a(0, 0), &a(1, 0))),
           absl::OkStatus()));
}

// Tests that index array maps depending on reversed dimensions are handled
// correctly.
TEST(NDIterableTransformedArrayTest, IndexedCombinePartiallyReversed) {
  Arena arena;
  auto a = AllocateArray<int>({2, 3});
  auto ta = (a | tensorstore::Dims(1)
                     .OuterIndexArraySlice(MakeArray<Index>({{0, 2}, {2, 0}}))
                     .SizedInterval(kImplicit, kImplicit, {1, -1}))
                .value();

  auto iterable = GetTransformedArrayNDIterable(ta, &arena).value();
  MultiNDIterator<1, /*Full=*/true> multi_iterator(
      ta.shape(), skip_repeated_elements, {{iterable.get()}}, &arena);
  EXPECT_THAT(multi_iterator.iteration_dimensions, ElementsAre(2, 0));
  EXPECT_THAT(multi_iterator.directions, ElementsAre(1, 1, -1));
  EXPECT_THAT(
      GetIterationTrace<int>(&multi_iterator),
      Pair(ElementsAre(ElementsAre(&a(0, 0), &a(1, 0), &a(0, 2), &a(1, 2),
                                   &a(0, 2), &a(1, 2), &a(0, 0), &a(1, 0))),
           absl::OkStatus()));
}

// Same as above, but with both index array dimensions reversed.
TEST(NDIterableTransformedArrayTest, IndexedCombineBothReversed) {
  Arena arena;
  auto a = AllocateArray<int>({2, 3});
  auto ta = (a | tensorstore::Dims(1)
                     .OuterIndexArraySlice(MakeArray<Index>({{0, 2}, {2, 0}}))
                     .SizedInterval(kImplicit, kImplicit, -1))
                .value();

  auto iterable = GetTransformedArrayNDIterable(ta, &arena).value();
  MultiNDIterator<1, /*Full=*/true> multi_iterator(
      ta.shape(), skip_repeated_elements, {{iterable.get()}}, &arena);
  EXPECT_THAT(multi_iterator.iteration_dimensions, ElementsAre(2, 0));
  EXPECT_THAT(multi_iterator.directions, ElementsAre(1, -1, -1));
  EXPECT_THAT(
      GetIterationTrace<int>(&multi_iterator),
      Pair(ElementsAre(ElementsAre(&a(0, 0), &a(1, 0), &a(0, 2), &a(1, 2),
                                   &a(0, 2), &a(1, 2), &a(0, 0), &a(1, 0))),
           absl::OkStatus()));
}

// Tests that the preference for input dimensions on which index arrays to come
// first take precedence over the preference for a higher-magnitude byte_stride
// dimension to come first.
TEST(NDIterableTransformedArrayTest, IndexedVsStrided) {
  Arena arena;
  auto a = AllocateArray<int>({2, 2});
  auto b = AllocateArray<int>({2, 3});

  auto tb =
      (b | tensorstore::Dims(1).OuterIndexArraySlice(MakeArray<Index>({0, 2})))
          .value();

  auto iterable_a = GetTransformedArrayNDIterable(a, &arena).value();
  auto iterable_b = GetTransformedArrayNDIterable(tb, &arena).value();
  MultiNDIterator<2, /*Full=*/true> multi_iterator(
      tb.shape(), skip_repeated_elements,
      {{iterable_a.get(), iterable_b.get()}}, &arena);
  EXPECT_THAT(multi_iterator.iteration_dimensions, ElementsAre(1, 0));
  EXPECT_THAT(
      (GetIterationTrace<int, int>(&multi_iterator)),
      Pair(ElementsAre(ElementsAre(&a(0, 0), &a(1, 0), &a(0, 1), &a(1, 1)),
                       ElementsAre(&b(0, 0), &b(1, 0), &b(0, 2), &b(1, 2))),
           absl::OkStatus()));
}

TEST(NDIterableTransformedArrayTest, IndexedWith2StridedDims) {
  Arena arena;
  auto a = AllocateArray<int>({2, 2, 3});

  auto ta =
      (a | tensorstore::Dims(1).MoveToFront() |
       tensorstore::Dims(2).OuterIndexArraySlice(MakeArray<Index>({0, 2, 1})))
          .value();

  auto iterable = GetTransformedArrayNDIterable(ta, &arena).value();
  MultiNDIterator<1, /*Full=*/true> multi_iterator(
      ta.shape(), skip_repeated_elements, {{iterable.get()}}, &arena);
  EXPECT_THAT(multi_iterator.iteration_dimensions, ElementsAre(2, 0));
  EXPECT_THAT(GetIterationTrace<int>(&multi_iterator),
              Pair(ElementsAre(ElementsAre(
                       &a(0, 0, 0), &a(0, 1, 0), &a(1, 0, 0), &a(1, 1, 0),
                       &a(0, 0, 2), &a(0, 1, 2), &a(1, 0, 2), &a(1, 1, 2),
                       &a(0, 0, 1), &a(0, 1, 1), &a(1, 0, 1), &a(1, 1, 1))),
                   absl::OkStatus()));
}

TEST(NDIterableTransformedArrayTest, TwoIndexedDims) {
  Arena arena;
  auto a = AllocateArray<int>({2, 3});

  auto ta =
      (a |
       tensorstore::Dims(0).OuterIndexArraySlice(MakeArray<Index>({0, 1, 1})) |
       tensorstore::Dims(1).OuterIndexArraySlice(MakeArray<Index>({0, 2})))
          .value();

  auto iterable = GetTransformedArrayNDIterable(ta, &arena).value();
  MultiNDIterator<1, /*Full=*/true> multi_iterator(
      ta.shape(), skip_repeated_elements, {{iterable.get()}}, &arena);
  EXPECT_THAT(multi_iterator.iteration_dimensions, ElementsAre(0, 1));
  EXPECT_THAT(GetIterationTrace<int>(&multi_iterator),
              Pair(ElementsAre(ElementsAre(&a(0, 0), &a(0, 2), &a(1, 0),
                                           &a(1, 2), &a(1, 0), &a(1, 2))),
                   absl::OkStatus()));
}

TEST(NDIterableTransformedArrayTest, FourIndexedDims) {
  Arena arena;
  auto a = AllocateArray<int>({2, 3});

  auto ta = (a |
             tensorstore::Dims(0).OuterIndexArraySlice(
                 MakeArray<Index>({{0, 1}, {1, 1}})) |
             tensorstore::Dims(-1).OuterIndexArraySlice(
                 MakeArray<Index>({{0, 2}, {1, 0}})))
                .value();

  auto b = AllocateArray<int>({2, 2, 2, 2});

  auto iterable_a = GetTransformedArrayNDIterable(ta, &arena).value();
  auto iterable_b = GetTransformedArrayNDIterable(b, &arena).value();
  MultiNDIterator<2, /*Full=*/true> multi_iterator(
      ta.shape(), skip_repeated_elements,
      {{iterable_a.get(), iterable_b.get()}}, &arena);
  EXPECT_THAT(multi_iterator.iteration_dimensions, ElementsAre(1, 3));
  EXPECT_THAT(  //
      (GetIterationTrace<int, int>(&multi_iterator)),
      Pair(                                                        //
          ElementsAre(                                             //
              ElementsAre(&a(0, 0), &a(0, 2), &a(0, 1), &a(0, 0),  //
                          &a(1, 0), &a(1, 2), &a(1, 1), &a(1, 0),  //
                          &a(1, 0), &a(1, 2), &a(1, 1), &a(1, 0),  //
                          &a(1, 0), &a(1, 2), &a(1, 1), &a(1, 0)),
              ElementsAre(  //
                  b.data() + 0, b.data() + 1, b.data() + 2, b.data() + 3,
                  b.data() + 4, b.data() + 5, b.data() + 6, b.data() + 7,
                  b.data() + 8, b.data() + 9, b.data() + 10, b.data() + 11,
                  b.data() + 12, b.data() + 13, b.data() + 14, b.data() + 15)),
          absl::OkStatus()));
}

TEST(NDIterableTransformedArrayTest, LastTwoDimsStrided) {
  Arena arena;
  auto a = AllocateArray<int>({2, 3});

  auto ta = (a |
             tensorstore::Dims(0).OuterIndexArraySlice(
                 MakeArray<Index>({{0, 1}, {1, 1}})) |
             tensorstore::Dims(-1).OuterIndexArraySlice(
                 MakeArray<Index>({{0, 2}, {1, 0}})))
                .value();

  auto b = AllocateArray<int>({2, 2, 2, 2});

  auto iterable_a = GetTransformedArrayNDIterable(ta, &arena).value();
  auto iterable_b = GetTransformedArrayNDIterable(b, &arena).value();
  MultiNDIterator<2, /*Full=*/true> multi_iterator(
      ta.shape(), skip_repeated_elements,
      {{iterable_a.get(), iterable_b.get()}}, &arena);
  EXPECT_THAT(multi_iterator.iteration_dimensions, ElementsAre(1, 3));
  EXPECT_THAT(  //
      (GetIterationTrace<int, int>(&multi_iterator)),
      Pair(                                                        //
          ElementsAre(                                             //
              ElementsAre(&a(0, 0), &a(0, 2), &a(0, 1), &a(0, 0),  //
                          &a(1, 0), &a(1, 2), &a(1, 1), &a(1, 0),  //
                          &a(1, 0), &a(1, 2), &a(1, 1), &a(1, 0),  //
                          &a(1, 0), &a(1, 2), &a(1, 1), &a(1, 0)),
              ElementsAre(  //
                  b.data() + 0, b.data() + 1, b.data() + 2, b.data() + 3,
                  b.data() + 4, b.data() + 5, b.data() + 6, b.data() + 7,
                  b.data() + 8, b.data() + 9, b.data() + 10, b.data() + 11,
                  b.data() + 12, b.data() + 13, b.data() + 14, b.data() + 15)),
          absl::OkStatus()));
}

TEST(NDIterableTransformedArrayTest, TwoTransformedArrays) {
  Arena arena;
  auto a = AllocateArray<int>({2, 3});
  auto b = AllocateArray<int>({2, 3});
  auto ta =
      (a | tensorstore::Dims(0).OuterIndexArraySlice(MakeArray<Index>({0, 1})))
          .value();
  auto tb = (b | tensorstore::Dims(1).OuterIndexArraySlice(
                     MakeArray<Index>({0, 1, 2})))
                .value();
  auto iterable_a = GetTransformedArrayNDIterable(ta, &arena).value();
  auto iterable_b = GetTransformedArrayNDIterable(tb, &arena).value();
  MultiNDIterator<2, /*Full=*/true> multi_iterator(
      ta.shape(), skip_repeated_elements,
      {{iterable_a.get(), iterable_b.get()}}, &arena);
  EXPECT_THAT(multi_iterator.iteration_dimensions, ElementsAre(0, 1));
  EXPECT_THAT((GetIterationTrace<int, int>(&multi_iterator)),
              Pair(ElementsAre(ElementsAre(&a(0, 0), &a(0, 1), &a(0, 2),
                                           &a(1, 0), &a(1, 1), &a(1, 2)),
                               ElementsAre(&b(0, 0), &b(0, 1), &b(0, 2),
                                           &b(1, 0), &b(1, 1), &b(1, 2))),
                   absl::OkStatus()));
}

TEST(NDIterableTransformedArrayTest, ZeroRankIndexArray) {
  Arena arena;
  SharedArray<const Index> index_array{std::make_shared<Index>(3),
                                       StridedLayout<>({5}, {0})};
  int data[100];
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto transform,
      IndexTransformBuilder(1, 1)
          .input_shape({5})
          .output_index_array(0, sizeof(int) * 2, sizeof(int) * 4, index_array)
          .Finalize());
  auto iterable_a = GetTransformedArrayNDIterable(
                        {tensorstore::UnownedToShared(
                             tensorstore::ElementPointer<int>(&data[0])),
                         transform},
                        &arena)
                        .value();
  MultiNDIterator<1, /*Full=*/true> multi_iterator(
      transform.input_shape(), skip_repeated_elements, {{iterable_a.get()}},
      &arena);
  EXPECT_THAT(multi_iterator.iteration_dimensions, ElementsAre(-1, -1));
  EXPECT_THAT(
      (GetIterationTrace<int>(&multi_iterator)),
      Pair(ElementsAre(ElementsAre(&data[4 * 3 + 2])), absl::OkStatus()));
}

TEST(NDIterableTransformedArrayTest, OutOfBoundsConstant) {
  Arena arena;
  auto a = AllocateArray<int>({5});
  auto transform = IndexTransformBuilder<1, 1>()
                       .input_shape({5})
                       .output_constant(0, 8)
                       .Finalize()
                       .value();
  EXPECT_THAT(
      GetTransformedArrayNDIterable(a, transform, &arena),
      MatchesStatus(absl::StatusCode::kOutOfRange,
                    "Checking bounds of constant output index map for "
                    "dimension 0: Index 8 is outside valid range \\[0, 5\\)"));
}

TEST(NDIterableTransformedArrayTest, NullTransform) {
  Arena arena;
  auto a = AllocateArray<int>({5});
  auto iterable_a = GetTransformedArrayNDIterable(a, {}, &arena).value();
  EXPECT_EQ(tensorstore::dtype_v<int>, iterable_a->dtype());
  MultiNDIterator<1, /*Full=*/true> multi_iterator(
      a.shape(), skip_repeated_elements, {{iterable_a.get()}}, &arena);
  EXPECT_THAT(multi_iterator.iteration_dimensions, ElementsAre(-1, 0));
  EXPECT_THAT((GetIterationTrace<int>(&multi_iterator)),
              Pair(ElementsAre(ElementsAre(&a(0), &a(1), &a(2), &a(3), &a(4))),
                   absl::OkStatus()));
}

TEST(NDIterableTransformedArrayTest, IdentityTransform) {
  Arena arena;
  auto a = AllocateArray<int>({5});
  auto iterable_a =
      GetTransformedArrayNDIterable(
          a,
          tensorstore::IdentityTransform(tensorstore::span<const Index>({5})),
          &arena)
          .value();
  EXPECT_EQ(tensorstore::dtype_v<int>, iterable_a->dtype());
  MultiNDIterator<1, /*Full=*/true> multi_iterator(
      a.shape(), skip_repeated_elements, {{iterable_a.get()}}, &arena);
  EXPECT_THAT(multi_iterator.iteration_dimensions, ElementsAre(-1, 0));
  EXPECT_THAT((GetIterationTrace<int>(&multi_iterator)),
              Pair(ElementsAre(ElementsAre(&a(0), &a(1), &a(2), &a(3), &a(4))),
                   absl::OkStatus()));
}

TEST(NDIterableTransformedArrayTest, OutOfBoundsSingleInputDimension) {
  Arena arena;
  auto a = AllocateArray<int>({5});
  auto transform = IndexTransformBuilder<1, 1>()
                       .input_shape({5})
                       .output_single_input_dimension(0, 2, 1, 0)
                       .Finalize()
                       .value();
  EXPECT_THAT(GetTransformedArrayNDIterable(a, transform, &arena),
              MatchesStatus(absl::StatusCode::kOutOfRange,
                            "Output dimension 0 range of \\[2, 7\\) is not "
                            "contained within array domain of \\[0, 5\\)"));
}

TEST_P(MaybeDirectTest, OutOfBoundsIndexArray) {
  auto a = AllocateArray<int>({5});
  auto transform =
      IndexTransformBuilder<1, 1>()
          .input_shape({5})
          .output_index_array(0, 2, 1, MakeArray<Index>({0, 0, 0, 0, 42}))
          .Finalize()
          .value();
  EXPECT_THAT(GetMaybeDirectTransformedArrayNDIterable(a, transform),
              MatchesStatus(absl::StatusCode::kOutOfRange,
                            ".*Index 42 is outside valid range \\[-2, 3\\)"));
}

TEST_P(MaybeDirectTest, OutOfBoundsSingletonIndexArray) {
  SharedArray<const Index> index_array{std::make_shared<Index>(42),
                                       StridedLayout<>({5}, {0})};
  auto a = AllocateArray<int>({5});
  auto transform = IndexTransformBuilder<1, 1>()
                       .input_shape({5})
                       .output_index_array(0, 2, 1, index_array)
                       .Finalize()
                       .value();
  EXPECT_THAT(GetMaybeDirectTransformedArrayNDIterable(a, transform),
              MatchesStatus(absl::StatusCode::kOutOfRange,
                            ".*Index 42 is outside valid range \\[-2, 3\\)"));
}

// Test the case of an array with no index array input dimensions (simply
// forwards to `GetArrayNDIterable`).
TEST(NDIterableTransformedArrayTest, BlockTraceThreeStridedDimensions) {
  Arena arena;
  auto a = AllocateArray<int>({2, 5, 3});
  auto ta = (a | tensorstore::Dims(1).SizedInterval(0, 2, 2)).value();

  auto iterable = GetTransformedArrayNDIterable(ta, &arena).value();
  MultiNDIterator<1, /*Full=*/true> multi_iterator(
      ta.shape(), skip_repeated_elements, {{iterable.get()}}, &arena);
  EXPECT_EQ(IterationBufferKind::kContiguous, multi_iterator.buffer_kind);
  EXPECT_THAT(multi_iterator.iteration_dimensions, ElementsAre(0, 1, 2));
  EXPECT_THAT(
      GetBlockTrace<int>(&multi_iterator),
      Pair(ElementsAre(FieldsAre(ElementsAre(0, 0, 0), ElementsAre(2, 3),
                                 ElementsAre(ElementsAreArray({
                                     &a(0, 0, 0),
                                     &a(0, 0, 1),
                                     &a(0, 0, 2),
                                     &a(0, 2, 0),
                                     &a(0, 2, 1),
                                     &a(0, 2, 2),
                                 }))),
                       FieldsAre(ElementsAre(1, 0, 0), ElementsAre(2, 3),
                                 ElementsAre(ElementsAreArray({
                                     &a(1, 0, 0),
                                     &a(1, 0, 1),
                                     &a(1, 0, 2),
                                     &a(1, 2, 0),
                                     &a(1, 2, 1),
                                     &a(1, 2, 2),
                                 })))),
           absl::OkStatus()));
}

}  // namespace
