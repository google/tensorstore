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

#include "tensorstore/internal/async_write_array.h"

#include <cstddef>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/array.h"
#include "tensorstore/box.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/arena.h"
#include "tensorstore/internal/nditerable_array.h"
#include "tensorstore/internal/nditerable_copy.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using tensorstore::StorageGeneration;
using tensorstore::internal::AsyncWriteArray;
using Spec = AsyncWriteArray::Spec;
using MaskedArray = AsyncWriteArray::MaskedArray;
using tensorstore::Index;
using tensorstore::kInfIndex;
using tensorstore::kInfSize;
using tensorstore::MakeArray;
using tensorstore::span;
using tensorstore::internal::Arena;

tensorstore::SharedArray<void> CopyNDIterable(
    tensorstore::internal::NDIterable::Ptr source_iterable,
    span<const Index> shape, Arena* arena) {
  auto dest_array = tensorstore::AllocateArray(shape, tensorstore::c_order,
                                               tensorstore::default_init,
                                               source_iterable->data_type());
  auto dest_iterable =
      tensorstore::internal::GetArrayNDIterable(dest_array, arena);
  tensorstore::internal::NDIterableCopier copier(*source_iterable,
                                                 *dest_iterable, shape, arena);
  TENSORSTORE_EXPECT_OK(copier.Copy());
  return dest_array;
}

template <typename Target>
void TestWrite(Target* target, const Spec& spec, span<const Index> origin,
               tensorstore::SharedOffsetArrayView<const void> source_array,
               bool expected_modified) {
  Arena arena;
  auto transform = tensorstore::IdentityTransform(source_array.domain());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto dest_iterable, target->BeginWrite(spec, origin, transform, &arena));
  auto source_iterable =
      tensorstore::internal::GetArrayNDIterable(source_array, &arena);
  tensorstore::internal::NDIterableCopier copier(
      *source_iterable, *dest_iterable, source_array.shape(), &arena);
  TENSORSTORE_EXPECT_OK(copier.Copy());
  EXPECT_EQ(expected_modified,
            target->EndWrite(spec, origin, transform,
                             copier.layout_info().layout_view(),
                             copier.stepper().position(), &arena));
}

TEST(SpecTest, Basic) {
  auto fill_value = MakeArray<int32_t>({{1, 2, 3}, {4, 5, 6}});
  tensorstore::Box<> component_bounds({-1, -kInfIndex}, {3, kInfSize});
  Spec spec(fill_value, component_bounds);
  EXPECT_EQ(fill_value, spec.fill_value);
  EXPECT_EQ(component_bounds, spec.component_bounds);
  EXPECT_THAT(spec.shape(), ::testing::ElementsAre(2, 3));
  EXPECT_EQ(6, spec.num_elements());

  // Chunk [0,2), [0, 3) is fully contained within bounds [-1, 2), (-inf,+inf)
  EXPECT_EQ(6, spec.chunk_num_elements(span<const Index>({0, 0})));

  // Chunk [-2,0), [0, 3) is half contained within bounds [-1, 2), (-inf,+inf)
  EXPECT_EQ(3, spec.chunk_num_elements(span<const Index>({-2, 0})));

  EXPECT_EQ(2, spec.rank());
  EXPECT_EQ(tensorstore::DataTypeOf<int32_t>(), spec.data_type());
  EXPECT_THAT(spec.c_order_byte_strides,
              ::testing::ElementsAre(3 * sizeof(int32_t), sizeof(int32_t)));

  EXPECT_EQ(tensorstore::StridedLayout<>({2, 3}, {3 * 4, 4}),
            spec.write_layout());

  EXPECT_EQ(0, spec.EstimateReadStateSizeInBytes(/*valid=*/false));
  EXPECT_EQ(2 * 3 * sizeof(int32_t),
            spec.EstimateReadStateSizeInBytes(/*valid=*/true));

  {
    auto read_array = MakeArray<int32_t>({{7, 8, 9}, {10, 11, 12}});
    Arena arena;
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto iterable,
        spec.GetReadNDIterable(
            read_array, /*origin=*/span<const Index>({2, 6}),
            tensorstore::IdentityTransform(tensorstore::Box<>({2, 6}, {2, 2})),
            &arena));
    EXPECT_EQ(
        MakeArray<int32_t>({{7, 8}, {10, 11}}),
        CopyNDIterable(std::move(iterable), span<const Index>({2, 2}), &arena));
  }
}

TEST(MaskedArrayTest, Basic) {
  auto fill_value = MakeArray<int32_t>({{1, 2, 3}, {4, 5, 6}});
  auto fill_value_copy = MakeCopy(fill_value);
  tensorstore::Box<> component_bounds({-1, -kInfIndex}, {3, kInfSize});
  Spec spec(fill_value, component_bounds);
  MaskedArray write_state(2);
  std::vector<Index> origin{0, 0};
  EXPECT_EQ(0, write_state.EstimateSizeInBytes(spec));
  EXPECT_TRUE(write_state.IsUnmodified());
  EXPECT_FALSE(write_state.data);

  auto read_array = MakeArray<int32_t>({{11, 12, 13}, {14, 15, 16}});

  {
    auto writeback_data = write_state.GetArrayForWriteback(
        spec, origin, /*read_array=*/{},
        /*read_state_already_integrated=*/false);
    EXPECT_EQ(spec.fill_value, writeback_data.array);
    EXPECT_TRUE(writeback_data.equals_fill_value);
  }

  {
    auto writeback_data = write_state.GetArrayForWriteback(
        spec, origin, /*read_array=*/fill_value_copy,
        /*read_state_already_integrated=*/false);
    EXPECT_EQ(spec.fill_value, writeback_data.array);
    EXPECT_TRUE(writeback_data.equals_fill_value);
  }

  {
    auto writeback_data = write_state.GetArrayForWriteback(
        spec, origin, read_array,
        /*read_state_already_integrated=*/false);
    EXPECT_EQ(read_array, writeback_data.array);
    EXPECT_FALSE(writeback_data.equals_fill_value);
  }

  // Write a zero-size region to test handling of a write that does not modify
  // the array.
  TestWrite(&write_state, spec, origin,
            tensorstore::AllocateArray<int32_t>(
                tensorstore::BoxView<>({1, 1}, {0, 0})),
            /*expected_modified=*/false);
  EXPECT_TRUE(write_state.data);
  EXPECT_TRUE(write_state.IsUnmodified());
  EXPECT_FALSE(write_state.IsFullyOverwritten(spec, origin));
  // Data array has been allocated, but mask array has not.
  EXPECT_EQ(2 * 3 * sizeof(int32_t), write_state.EstimateSizeInBytes(spec));
  // Zero initialize `write_state.data` to simplify testing.
  std::fill_n(static_cast<int32_t*>(write_state.data.get()),
              spec.num_elements(), 0);
  TestWrite(&write_state, spec, origin,
            tensorstore::MakeOffsetArray<int32_t>({1, 1}, {{7, 8}}),
            /*expected_modified=*/true);
  EXPECT_EQ(MakeArray<int32_t>({{0, 0, 0}, {0, 7, 8}}),
            write_state.shared_array_view(spec));
  EXPECT_FALSE(write_state.IsUnmodified());
  EXPECT_FALSE(write_state.IsFullyOverwritten(spec, origin));

  // Make write region non-rectangular to force mask allocation.
  TestWrite(&write_state, spec, origin,
            tensorstore::MakeOffsetArray<int32_t>({0, 0}, {{9}}),
            /*expected_modified=*/true);
  EXPECT_EQ(MakeArray<int32_t>({{9, 0, 0}, {0, 7, 8}}),
            write_state.shared_array_view(spec));
  EXPECT_EQ(MakeArray<bool>({{1, 0, 0}, {0, 1, 1}}),
            tensorstore::Array(write_state.mask.mask_array.get(), {2, 3}));
  EXPECT_FALSE(write_state.IsUnmodified());
  EXPECT_FALSE(write_state.IsFullyOverwritten(spec, origin));
  // Both data array and mask array have been allocated.
  EXPECT_EQ(2 * 3 * (sizeof(int32_t) + sizeof(bool)),
            write_state.EstimateSizeInBytes(spec));

  {
    auto writeback_data = write_state.GetArrayForWriteback(
        spec, origin, /*read_array=*/{},
        /*read_state_already_integrated=*/false);
    EXPECT_FALSE(writeback_data.equals_fill_value);
    EXPECT_EQ(MakeArray<int32_t>({{9, 2, 3}, {4, 7, 8}}), writeback_data.array);
  }

  {
    auto writeback_data = write_state.GetArrayForWriteback(
        spec, origin, read_array,
        /*read_state_already_integrated=*/true);
    EXPECT_FALSE(writeback_data.equals_fill_value);
    // Data is not updated due to `read_state_already_integrated`.
    EXPECT_EQ(MakeArray<int32_t>({{9, 2, 3}, {4, 7, 8}}), writeback_data.array);
  }

  {
    auto writeback_data = write_state.GetArrayForWriteback(
        spec, origin, read_array,
        /*read_state_already_integrated=*/false);
    EXPECT_FALSE(writeback_data.equals_fill_value);
    EXPECT_EQ(MakeArray<int32_t>({{9, 12, 13}, {14, 7, 8}}),
              writeback_data.array);
  }

  // Overwrite fully.
  TestWrite(&write_state, spec, origin,
            tensorstore::MakeOffsetArray<int32_t>({0, 0}, {{9}, {9}}),
            /*expected_modified=*/true);
  TestWrite(&write_state, spec, origin,
            tensorstore::MakeOffsetArray<int32_t>({0, 0}, {{10, 10, 10}}),
            /*expected_modified=*/true);
  EXPECT_FALSE(write_state.IsUnmodified());
  EXPECT_TRUE(write_state.IsFullyOverwritten(spec, origin));

  {
    auto writeback_data = write_state.GetArrayForWriteback(
        spec, origin, read_array,
        /*read_state_already_integrated=*/false);
    EXPECT_FALSE(writeback_data.equals_fill_value);
    EXPECT_EQ(MakeArray<int32_t>({{10, 10, 10}, {9, 7, 8}}),
              writeback_data.array);
  }

  // Overwrite with the fill value via writing.
  TestWrite(&write_state, spec, origin, fill_value_copy,
            /*expected_modified=*/true);
  // Data array is still allocated.
  EXPECT_TRUE(write_state.data);
  {
    auto writeback_data = write_state.GetArrayForWriteback(
        spec, origin, read_array,
        /*read_state_already_integrated=*/false);
    EXPECT_TRUE(writeback_data.equals_fill_value);
    EXPECT_EQ(spec.fill_value, writeback_data.array);
    // Data array no longer allocated.
    EXPECT_FALSE(write_state.data);
  }

  // Reset.
  write_state.Clear();
  EXPECT_TRUE(write_state.IsUnmodified());

  // Partially overwrite then call `WriteFillValue`.
  TestWrite(&write_state, spec, origin,
            tensorstore::MakeOffsetArray<int32_t>({1, 1}, {{7, 8}}),
            /*expected_modified=*/true);
  write_state.WriteFillValue(spec, origin);
  EXPECT_FALSE(write_state.IsUnmodified());
  EXPECT_FALSE(write_state.data);
  EXPECT_EQ(0, write_state.EstimateSizeInBytes(spec));
  EXPECT_TRUE(write_state.IsFullyOverwritten(spec, origin));

  {
    auto writeback_data = write_state.GetArrayForWriteback(
        spec, origin, read_array,
        /*read_state_already_integrated=*/false);
    EXPECT_TRUE(writeback_data.equals_fill_value);
    EXPECT_EQ(spec.fill_value, writeback_data.array);
    // Data array still not allocated.
    EXPECT_FALSE(write_state.data);
  }

  // Partially overwrite.
  TestWrite(&write_state, spec, origin,
            tensorstore::MakeOffsetArray<int32_t>({1, 1}, {{7, 8}}),
            /*expected_modified=*/true);
  EXPECT_EQ(MakeArray<int32_t>({{1, 2, 3}, {4, 7, 8}}),
            write_state.shared_array_view(spec));
}

// Tests that `IsFullyOverwritten` correctly handles chunks that are partially
// outside `component_bounds`.
TEST(MaskedArrayTest, PartialChunk) {
  auto fill_value = MakeArray<int32_t>({{1, 2, 3}, {4, 5, 6}});
  tensorstore::Box<> component_bounds({-1, -kInfIndex}, {3, kInfSize});
  Spec spec(fill_value, component_bounds);
  std::vector<Index> origin{-2, 0};
  MaskedArray write_state(2);
  // Fully overwrite the portion within `component_bounds`.
  TestWrite(&write_state, spec, origin,
            tensorstore::MakeOffsetArray<int32_t>({-1, 0}, {{7, 8, 9}}),
            /*expected_modified=*/true);
  EXPECT_TRUE(write_state.IsFullyOverwritten(spec, origin));
}

TEST(AsyncWriteArrayTest, Basic) {
  AsyncWriteArray async_write_array(2);
  auto fill_value = MakeArray<int32_t>({{1, 2, 3}, {4, 5, 6}});
  tensorstore::Box<> component_bounds({-1, -kInfIndex}, {3, kInfSize});
  Spec spec(fill_value, component_bounds);
  std::vector<Index> origin{0, 0};

  // Test that `GetReadNDIterable` correctly handles the case of an unmodified
  // write state.
  {
    Arena arena;
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto iterable,
        async_write_array.GetReadNDIterable(
            spec, origin, /*read_array=*/{},
            /*read_generation=*/StorageGeneration::FromString("a"),
            tensorstore::IdentityTransform(tensorstore::Box<>({0, 1}, {2, 2})),
            &arena));
    EXPECT_EQ(
        MakeArray<int32_t>({{2, 3}, {5, 6}}),
        CopyNDIterable(std::move(iterable), span<const Index>({2, 2}), &arena));
  }

  // Test that `GetArrayForWriteback` handles an unmodified write state.
  {
    auto read_array = MakeArray<int32_t>({{21, 22, 23}, {24, 25, 26}});
    Arena arena;
    auto writeback_data = async_write_array.GetArrayForWriteback(
        spec, origin, read_array,
        /*read_generation=*/StorageGeneration::FromString("b"));
    EXPECT_FALSE(writeback_data.equals_fill_value);
    // Writeback reflects updated `read_array`.
    EXPECT_EQ(read_array, writeback_data.array);
    EXPECT_EQ(StorageGeneration::Invalid(), async_write_array.read_generation);
  }

  // Test that `GetReadNDIterable` correctly handles the case of a
  // partially-modified write state.
  TestWrite(&async_write_array, spec, origin,
            tensorstore::MakeOffsetArray<int32_t>({0, 0}, {{8}}),
            /*expected_modified=*/true);

  // Test that writing does not normally copy the data array.
  {
    auto* data_ptr = async_write_array.write_state.data.get();
    TestWrite(&async_write_array, spec, origin,
              tensorstore::MakeOffsetArray<int32_t>({0, 0}, {{7}}),
              /*expected_modified=*/true);
    EXPECT_EQ(data_ptr, async_write_array.write_state.data.get());
  }

  {
    Arena arena;
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto iterable,
        async_write_array.GetReadNDIterable(
            spec, origin, /*read_array=*/{},
            /*read_generation=*/StorageGeneration::FromString("a"),
            tensorstore::IdentityTransform(tensorstore::Box<>({0, 0}, {2, 3})),
            &arena));
    EXPECT_EQ(
        MakeArray<int32_t>({{7, 2, 3}, {4, 5, 6}}),
        CopyNDIterable(std::move(iterable), span<const Index>({2, 3}), &arena));
  }

  // Test that `GetReadNDIterable` handles a non-updated read array.
  {
    auto read_array = MakeArray<int32_t>({{11, 12, 13}, {14, 15, 16}});
    Arena arena;
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto iterable,
        async_write_array.GetReadNDIterable(
            spec, origin, read_array,
            /*read_generation=*/StorageGeneration::FromString("a"),
            tensorstore::IdentityTransform(tensorstore::Box<>({0, 0}, {2, 3})),
            &arena));
    // Result should not reflect updated `read_array`.
    EXPECT_EQ(
        MakeArray<int32_t>({{7, 2, 3}, {4, 5, 6}}),
        CopyNDIterable(std::move(iterable), span<const Index>({2, 3}), &arena));
  }

  // Test that `GetArrayForWriteback` handles a non-updated read array.
  {
    auto read_array = MakeArray<int32_t>({{11, 12, 13}, {14, 15, 16}});
    Arena arena;
    auto writeback_data = async_write_array.GetArrayForWriteback(
        spec, origin, read_array,
        /*read_generation=*/StorageGeneration::FromString("a"));
    EXPECT_FALSE(writeback_data.equals_fill_value);
    // Writeback does not reflect updated `read_array`.
    EXPECT_EQ(MakeArray<int32_t>({{7, 2, 3}, {4, 5, 6}}), writeback_data.array);
  }

  // Test that `GetReadNDIterable` handles an updated read array.
  {
    auto read_array = MakeArray<int32_t>({{11, 12, 13}, {14, 15, 16}});
    Arena arena;
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto iterable,
        async_write_array.GetReadNDIterable(
            spec, origin, read_array,
            /*read_generation=*/StorageGeneration::FromString("b"),
            tensorstore::IdentityTransform(tensorstore::Box<>({0, 0}, {2, 3})),
            &arena));
    EXPECT_EQ(
        MakeArray<int32_t>({{7, 12, 13}, {14, 15, 16}}),
        CopyNDIterable(std::move(iterable), span<const Index>({2, 3}), &arena));
  }

  // Test that `GetArrayForWriteback` handles an updated read array.
  {
    auto read_array = MakeArray<int32_t>({{21, 22, 23}, {24, 25, 26}});
    Arena arena;
    auto writeback_data = async_write_array.GetArrayForWriteback(
        spec, origin, read_array,
        /*read_generation=*/StorageGeneration::FromString("c"));
    EXPECT_FALSE(writeback_data.equals_fill_value);
    // Writeback reflects updated `read_array`.
    EXPECT_EQ(MakeArray<int32_t>({{7, 22, 23}, {24, 25, 26}}),
              writeback_data.array);
    EXPECT_EQ(StorageGeneration::FromString("c"),
              async_write_array.read_generation);
  }

  // Test that `GetReadNDIterable` handles a write state fully-overwritten with
  // the fill value.
  async_write_array.write_state.WriteFillValue(spec, origin);
  {
    auto read_array = MakeArray<int32_t>({{11, 12, 13}, {14, 15, 16}});
    Arena arena;
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto iterable,
        async_write_array.GetReadNDIterable(
            spec, origin, read_array,
            /*read_generation=*/StorageGeneration::FromString("b"),
            tensorstore::IdentityTransform(tensorstore::Box<>({0, 0}, {2, 3})),
            &arena));
    EXPECT_EQ(fill_value, CopyNDIterable(std::move(iterable),
                                         span<const Index>({2, 3}), &arena));
  }

  // Test that `GetReadNDIterable` handles a write state fully-overwritten not
  // with the fill value.
  TestWrite(&async_write_array, spec, origin,
            tensorstore::MakeOffsetArray<int32_t>({0, 0}, {{9}}),
            /*expected_modified=*/true);
  {
    auto read_array = MakeArray<int32_t>({{11, 12, 13}, {14, 15, 16}});
    Arena arena;
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto iterable,
        async_write_array.GetReadNDIterable(
            spec, origin, read_array,
            /*read_generation=*/StorageGeneration::FromString("b"),
            tensorstore::IdentityTransform(tensorstore::Box<>({0, 0}, {2, 3})),
            &arena));
    EXPECT_EQ(
        MakeArray<int32_t>({{9, 2, 3}, {4, 5, 6}}),
        CopyNDIterable(std::move(iterable), span<const Index>({2, 3}), &arena));
  }
}

}  // namespace
