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

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorstore/array.h"
#include "tensorstore/array_testutil.h"
#include "tensorstore/box.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/index_space/index_domain.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_transform_testutil.h"
#include "tensorstore/index_space/transformed_array.h"
#include "tensorstore/internal/arena.h"
#include "tensorstore/internal/nditerable.h"
#include "tensorstore/internal/nditerable_array.h"
#include "tensorstore/internal/nditerable_copy.h"
#include "tensorstore/internal/testing/random_seed.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

namespace {

using ::tensorstore::Box;
using ::tensorstore::BoxView;
using ::tensorstore::Index;
using ::tensorstore::kInfIndex;
using ::tensorstore::kInfSize;
using ::tensorstore::MakeArray;
using ::tensorstore::MakeOffsetArray;
using ::tensorstore::MakeScalarArray;
using ::tensorstore::ReferencesSameDataAs;
using ::tensorstore::StorageGeneration;
using ::tensorstore::internal::Arena;
using ::tensorstore::internal::AsyncWriteArray;

using MaskedArray = AsyncWriteArray::MaskedArray;
using Spec = AsyncWriteArray::Spec;

tensorstore::SharedArray<void> CopyNDIterable(
    tensorstore::internal::NDIterable::Ptr source_iterable,
    tensorstore::span<const Index> shape, Arena* arena) {
  auto dest_array = tensorstore::AllocateArray(shape, tensorstore::c_order,
                                               tensorstore::default_init,
                                               source_iterable->dtype());
  auto dest_iterable =
      tensorstore::internal::GetArrayNDIterable(dest_array, arena);
  tensorstore::internal::NDIterableCopier copier(*source_iterable,
                                                 *dest_iterable, shape, arena);
  TENSORSTORE_EXPECT_OK(copier.Copy());
  return dest_array;
}

template <typename Target>
void TestWrite(Target* target, const Spec& spec, BoxView<> domain,
               tensorstore::SharedOffsetArrayView<const void> source_array) {
  Arena arena;
  auto transform = tensorstore::IdentityTransform(source_array.domain());
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto dest_iterable, target->BeginWrite(spec, domain, transform, &arena));
  auto source_iterable =
      tensorstore::internal::GetArrayNDIterable(source_array, &arena);
  tensorstore::internal::NDIterableCopier copier(
      *source_iterable, *dest_iterable, source_array.shape(), &arena);
  TENSORSTORE_EXPECT_OK(copier.Copy());
  if constexpr (std::is_same_v<Target, AsyncWriteArray>) {
    target->EndWrite(spec, domain, transform, true, &arena);
  } else {
    target->EndWrite(spec, domain, transform, &arena);
  }
}

TEST(SpecTest, Basic) {
  auto overall_fill_value = MakeOffsetArray<int32_t>(
      {-2, 0}, {
                   {1, 2, 3, 4, 5, 6, 7, 8, 9},
                   {11, 12, 13, 14, 15, 16, 17, 18, 19},
                   {21, 22, 23, 24, 25, 26, 27, 28, 29},
                   {31, 32, 33, 34, 35, 36, 37, 38, 39},
               });
  tensorstore::Box<> component_bounds({-1, -kInfIndex}, {3, kInfSize});
  Spec spec{overall_fill_value, component_bounds};

  // Chunk [0,2), [0, 3) is fully contained within bounds [-1, 2), (-inf,+inf)
  EXPECT_EQ(6, spec.GetNumInBoundsElements(BoxView<>({0, 0}, {2, 3})));

  // Chunk [-2,0), [0, 3) is half contained within bounds [-1, 2), (-inf,+inf)
  EXPECT_EQ(3, spec.GetNumInBoundsElements(BoxView<>({-2, 0}, {2, 3})));

  EXPECT_EQ(2, spec.rank());
  EXPECT_EQ(tensorstore::dtype_v<int32_t>, spec.dtype());

  EXPECT_EQ(0, spec.EstimateReadStateSizeInBytes(
                   /*valid=*/false, tensorstore::span<const Index>({2, 3})));
  EXPECT_EQ(2 * 3 * sizeof(int32_t),
            spec.EstimateReadStateSizeInBytes(
                /*valid=*/true, tensorstore::span<const Index>({2, 3})));

  {
    auto read_array = MakeArray<int32_t>({{7, 8, 9}, {10, 11, 12}});
    Arena arena;
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto iterable,
        spec.GetReadNDIterable(
            read_array, /*domain=*/BoxView<>({2, 6}, {2, 3}),
            tensorstore::IdentityTransform(tensorstore::Box<>({2, 6}, {2, 2})),
            &arena));
    EXPECT_EQ(MakeArray<int32_t>({{7, 8}, {10, 11}}),
              CopyNDIterable(std::move(iterable),
                             tensorstore::span<const Index>({2, 2}), &arena));
  }
}

TEST(MaskedArrayTest, Basic) {
  auto overall_fill_value = MakeOffsetArray<int32_t>(
      {-2, 0}, {
                   {1, 2, 3, 4, 5, 6, 7, 8, 9},
                   {11, 12, 13, 14, 15, 16, 17, 18, 19},
                   {21, 22, 23, 24, 25, 26, 27, 28, 29},
                   {31, 32, 33, 34, 35, 36, 37, 38, 39},
               });
  auto fill_value_copy = MakeArray<int32_t>({{21, 22, 23}, {31, 32, 33}});
  tensorstore::Box<> component_bounds({-1, -kInfIndex}, {3, kInfSize});
  Spec spec{overall_fill_value, component_bounds};
  MaskedArray write_state(2);
  Box<> domain({0, 0}, {2, 3});
  EXPECT_EQ(0, write_state.EstimateSizeInBytes(spec, domain.shape()));
  EXPECT_TRUE(write_state.IsUnmodified());
  EXPECT_FALSE(write_state.array.valid());

  auto read_array = MakeArray<int32_t>({{11, 12, 13}, {14, 15, 16}});

  {
    auto writeback_data = write_state.GetArrayForWriteback(
        spec, domain, /*read_array=*/{},
        /*read_state_already_integrated=*/false);
    EXPECT_EQ(spec.GetFillValueForDomain(domain), writeback_data.array);
    EXPECT_FALSE(writeback_data.must_store);
  }

  {
    auto writeback_data = write_state.GetArrayForWriteback(
        spec, domain, /*read_array=*/fill_value_copy,
        /*read_state_already_integrated=*/false);
    EXPECT_EQ(spec.GetFillValueForDomain(domain), writeback_data.array);
    EXPECT_FALSE(writeback_data.must_store);
  }

  {
    auto writeback_data = write_state.GetArrayForWriteback(
        spec, domain, read_array,
        /*read_state_already_integrated=*/false);
    EXPECT_EQ(read_array, writeback_data.array);
    EXPECT_TRUE(writeback_data.must_store);
  }

  // Write a zero-size region to test handling of a write that does not modify
  // the array.
  TestWrite(&write_state, spec, domain,
            tensorstore::AllocateArray<int32_t>(
                tensorstore::BoxView<>({1, 1}, {0, 0})));
  EXPECT_TRUE(write_state.array.valid());
  EXPECT_TRUE(write_state.IsUnmodified());
  EXPECT_FALSE(write_state.IsFullyOverwritten(spec, domain));
  // Data array has been allocated, but mask array has not.
  EXPECT_EQ(2 * 3 * sizeof(int32_t),
            write_state.EstimateSizeInBytes(spec, domain.shape()));
  // Zero initialize `write_state.array` to simplify testing.
  std::fill_n(static_cast<int32_t*>(write_state.array.data()),
              domain.num_elements(), 0);
  TestWrite(&write_state, spec, domain,
            tensorstore::MakeOffsetArray<int32_t>({1, 1}, {{7, 8}}));
  EXPECT_EQ(MakeArray<int32_t>({{0, 0, 0}, {0, 7, 8}}),
            write_state.shared_array_view(spec));
  EXPECT_FALSE(write_state.IsUnmodified());
  EXPECT_FALSE(write_state.IsFullyOverwritten(spec, domain));

  // Make write region non-rectangular to force mask allocation.
  TestWrite(&write_state, spec, domain,
            tensorstore::MakeOffsetArray<int32_t>({0, 0}, {{9}}));
  EXPECT_EQ(MakeArray<int32_t>({{9, 0, 0}, {0, 7, 8}}),
            write_state.shared_array_view(spec));
  EXPECT_EQ(MakeArray<bool>({{1, 0, 0}, {0, 1, 1}}),
            write_state.mask.mask_array);
  EXPECT_FALSE(write_state.IsUnmodified());
  EXPECT_FALSE(write_state.IsFullyOverwritten(spec, domain));
  // Both data array and mask array have been allocated.
  EXPECT_EQ(2 * 3 * (sizeof(int32_t) + sizeof(bool)),
            write_state.EstimateSizeInBytes(spec, domain.shape()));

  {
    auto writeback_data = write_state.GetArrayForWriteback(
        spec, domain, /*read_array=*/{},
        /*read_state_already_integrated=*/false);
    EXPECT_TRUE(writeback_data.must_store);
    EXPECT_EQ(MakeArray<int32_t>({{9, 22, 23}, {31, 7, 8}}),
              writeback_data.array);
  }

  {
    auto writeback_data = write_state.GetArrayForWriteback(
        spec, domain, read_array,
        /*read_state_already_integrated=*/true);
    EXPECT_TRUE(writeback_data.must_store);
    // Data is not updated due to `read_state_already_integrated`.
    EXPECT_EQ(MakeArray<int32_t>({{9, 22, 23}, {31, 7, 8}}),
              writeback_data.array);
  }

  {
    auto writeback_data = write_state.GetArrayForWriteback(
        spec, domain, read_array,
        /*read_state_already_integrated=*/false);
    EXPECT_TRUE(writeback_data.must_store);
    EXPECT_EQ(MakeArray<int32_t>({{9, 12, 13}, {14, 7, 8}}),
              writeback_data.array);
  }

  // Overwrite fully.
  TestWrite(&write_state, spec, domain,
            tensorstore::MakeOffsetArray<int32_t>({0, 0}, {{9}, {9}}));
  TestWrite(&write_state, spec, domain,
            tensorstore::MakeOffsetArray<int32_t>({0, 0}, {{10, 10, 10}}));
  EXPECT_FALSE(write_state.IsUnmodified());
  EXPECT_TRUE(write_state.IsFullyOverwritten(spec, domain));

  {
    auto writeback_data = write_state.GetArrayForWriteback(
        spec, domain, read_array,
        /*read_state_already_integrated=*/false);
    EXPECT_TRUE(writeback_data.must_store);
    EXPECT_EQ(MakeArray<int32_t>({{10, 10, 10}, {9, 7, 8}}),
              writeback_data.array);
  }

  // Overwrite with the fill value via writing.
  TestWrite(&write_state, spec, domain, fill_value_copy);
  // Data array is still allocated.
  EXPECT_TRUE(write_state.array.valid());
  {
    auto writeback_data = write_state.GetArrayForWriteback(
        spec, domain, read_array,
        /*read_state_already_integrated=*/false);
    EXPECT_FALSE(writeback_data.must_store);
    EXPECT_EQ(spec.GetFillValueForDomain(domain), writeback_data.array);
    // Data array no longer allocated.
    EXPECT_FALSE(write_state.array.valid());
  }

  // Reset.
  write_state.Clear();
  EXPECT_TRUE(write_state.IsUnmodified());

  // Partially overwrite then call `WriteFillValue`.
  TestWrite(&write_state, spec, domain,
            tensorstore::MakeOffsetArray<int32_t>({1, 1}, {{7, 8}}));
  write_state.WriteFillValue(spec, domain);
  EXPECT_FALSE(write_state.IsUnmodified());
  EXPECT_FALSE(write_state.array.valid());
  EXPECT_EQ(0, write_state.EstimateSizeInBytes(spec, domain.shape()));
  EXPECT_TRUE(write_state.IsFullyOverwritten(spec, domain));

  {
    auto writeback_data = write_state.GetArrayForWriteback(
        spec, domain, read_array,
        /*read_state_already_integrated=*/false);
    EXPECT_FALSE(writeback_data.must_store);
    EXPECT_EQ(spec.GetFillValueForDomain(domain), writeback_data.array);
    // Data array still not allocated.
    EXPECT_FALSE(write_state.array.valid());
  }

  // Partially overwrite.
  TestWrite(&write_state, spec, domain,
            tensorstore::MakeOffsetArray<int32_t>({1, 1}, {{7, 8}}));
  EXPECT_EQ(MakeArray<int32_t>({{21, 22, 23}, {31, 7, 8}}),
            write_state.shared_array_view(spec));
}

// Tests that `IsFullyOverwritten` correctly handles chunks that are partially
// outside `component_bounds`.
TEST(MaskedArrayTest, PartialChunk) {
  auto overall_fill_value = MakeOffsetArray<int32_t>(
      {-2, 0}, {
                   {1, 2, 3, 4, 5, 6, 7, 8, 9},
                   {11, 12, 13, 14, 15, 16, 17, 18, 19},
                   {21, 22, 23, 24, 25, 26, 27, 28, 29},
                   {31, 32, 33, 34, 35, 36, 37, 38, 39},
               });
  tensorstore::Box<> component_bounds({-1, -kInfIndex}, {3, kInfSize});
  Spec spec{overall_fill_value, component_bounds};
  Box<> domain{{-2, 0}, {2, 3}};
  MaskedArray write_state(2);
  // Fully overwrite the portion within `component_bounds`.
  TestWrite(&write_state, spec, domain,
            tensorstore::MakeOffsetArray<int32_t>({-1, 0}, {{7, 8, 9}}));
  EXPECT_TRUE(write_state.IsFullyOverwritten(spec, domain));
}

// Tests that `store_if_equal_to_fill_value==true` is correctly handled.
TEST(MaskedArrayTest, StoreIfEqualToFillValue) {
  auto overall_fill_value = MakeScalarArray<int32_t>(42);
  tensorstore::Box<> component_bounds;
  Spec spec{overall_fill_value, component_bounds};
  MaskedArray write_state(0);
  write_state.store_if_equal_to_fill_value = true;
  // Fully overwrite the portion within `component_bounds`.
  TestWrite(&write_state, spec, {}, tensorstore::MakeScalarArray<int32_t>(42));
  {
    auto writeback_data = write_state.GetArrayForWriteback(
        spec, /*domain=*/{}, /*read_array=*/{},
        /*read_state_already_integrated=*/false);
    EXPECT_EQ(overall_fill_value, writeback_data.array);
    EXPECT_TRUE(writeback_data.must_store);
  }

  auto read_array = MakeScalarArray<int32_t>(50);
  {
    auto writeback_data = write_state.GetArrayForWriteback(
        spec, /*domain=*/{}, read_array,
        /*read_state_already_integrated=*/false);
    EXPECT_EQ(overall_fill_value, writeback_data.array);
    EXPECT_TRUE(writeback_data.must_store);
  }
}

// Tests that `fill_value_comparison_kind==EqualityComparisonKind::identical` is
// correctly handled.
TEST(MaskedArrayTest, CompareFillValueIdenticallyEqual) {
  auto fill_value =
      MakeScalarArray<float>(std::numeric_limits<float>::quiet_NaN());
  tensorstore::Box<> component_bounds;
  Spec spec{fill_value, component_bounds};
  spec.fill_value_comparison_kind =
      tensorstore::EqualityComparisonKind::identical;
  MaskedArray write_state(0);
  // Fully overwrite the portion within `component_bounds`.
  TestWrite(&write_state, spec, {},
            tensorstore::MakeScalarArray<float>(
                std::numeric_limits<float>::signaling_NaN()));
  {
    auto writeback_data = write_state.GetArrayForWriteback(
        spec, /*domain=*/{}, /*read_array=*/{},
        /*read_state_already_integrated=*/false);
    EXPECT_TRUE(AreArraysIdenticallyEqual(
        tensorstore::MakeScalarArray<float>(
            std::numeric_limits<float>::signaling_NaN()),
        writeback_data.array));
    EXPECT_TRUE(writeback_data.must_store);
  }

  TestWrite(&write_state, spec, {},
            tensorstore::MakeScalarArray<float>(
                std::numeric_limits<float>::quiet_NaN()));
  {
    auto writeback_data = write_state.GetArrayForWriteback(
        spec, /*domain=*/{}, /*read_array=*/{},
        /*read_state_already_integrated=*/false);
    EXPECT_TRUE(
        AreArraysIdenticallyEqual(tensorstore::MakeScalarArray<float>(
                                      std::numeric_limits<float>::quiet_NaN()),
                                  writeback_data.array));
    EXPECT_FALSE(writeback_data.must_store);
    EXPECT_EQ(fill_value.data(), writeback_data.array.data());
  }
}

TEST(AsyncWriteArrayTest, Basic) {
  AsyncWriteArray async_write_array(2);
  auto overall_fill_value = MakeOffsetArray<int32_t>(
      {-2, 0}, {
                   {1, 2, 3, 4, 5, 6, 7, 8, 9},
                   {11, 12, 13, 14, 15, 16, 17, 18, 19},
                   {21, 22, 23, 24, 25, 26, 27, 28, 29},
                   {31, 32, 33, 34, 35, 36, 37, 38, 39},
               });
  tensorstore::Box<> component_bounds({-1, -kInfIndex}, {3, kInfSize});
  Spec spec{overall_fill_value, component_bounds};
  Box<> domain{{0, 0}, {2, 3}};
  auto fill_value_copy = MakeArray<int32_t>({{21, 22, 23}, {31, 32, 33}});

  // Test that `GetReadNDIterable` correctly handles the case of an unmodified
  // write state.
  {
    Arena arena;
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto iterable,
        async_write_array.GetReadNDIterable(
            spec, domain, /*read_array=*/{},
            /*read_generation=*/StorageGeneration::FromString("a"),
            tensorstore::IdentityTransform(tensorstore::Box<>({0, 1}, {2, 2})),
            &arena));
    EXPECT_EQ(MakeArray<int32_t>({{22, 23}, {32, 33}}),
              CopyNDIterable(std::move(iterable),
                             tensorstore::span<const Index>({2, 2}), &arena));
  }

  // Test that `GetArrayForWriteback` handles an unmodified write state.
  {
    auto read_array = MakeArray<int32_t>({{21, 22, 23}, {24, 25, 26}});
    Arena arena;
    auto writeback_data = async_write_array.GetArrayForWriteback(
        spec, domain, read_array,
        /*read_generation=*/StorageGeneration::FromString("b"));
    EXPECT_TRUE(writeback_data.must_store);
    // Writeback reflects updated `read_array`.
    EXPECT_EQ(read_array, writeback_data.array);
    EXPECT_EQ(StorageGeneration::Invalid(), async_write_array.read_generation);
  }

  // Test that `GetReadNDIterable` correctly handles the case of a
  // partially-modified write state.
  TestWrite(&async_write_array, spec, domain,
            tensorstore::MakeOffsetArray<int32_t>({0, 0}, {{8}}));

  // Test that writing does not normally copy the data array.
  {
    auto* data_ptr = async_write_array.write_state.array.data();
    TestWrite(&async_write_array, spec, domain,
              tensorstore::MakeOffsetArray<int32_t>({0, 0}, {{7}}));
    EXPECT_EQ(data_ptr, async_write_array.write_state.array.data());
  }

  {
    Arena arena;
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto iterable,
        async_write_array.GetReadNDIterable(
            spec, domain, /*read_array=*/{},
            /*read_generation=*/StorageGeneration::FromString("a"),
            tensorstore::IdentityTransform(tensorstore::Box<>({0, 0}, {2, 3})),
            &arena));
    EXPECT_EQ(MakeArray<int32_t>({{7, 22, 23}, {31, 32, 33}}),
              CopyNDIterable(std::move(iterable),
                             tensorstore::span<const Index>({2, 3}), &arena));
  }

  // Test that `GetReadNDIterable` handles a non-updated read array.
  {
    auto read_array = MakeArray<int32_t>({{11, 12, 13}, {14, 15, 16}});
    Arena arena;
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto iterable,
        async_write_array.GetReadNDIterable(
            spec, domain, read_array,
            /*read_generation=*/StorageGeneration::FromString("a"),
            tensorstore::IdentityTransform(tensorstore::Box<>({0, 0}, {2, 3})),
            &arena));
    // Result should not reflect updated `read_array`.
    EXPECT_EQ(MakeArray<int32_t>({{7, 22, 23}, {31, 32, 33}}),
              CopyNDIterable(std::move(iterable),
                             tensorstore::span<const Index>({2, 3}), &arena));
  }

  // Test that `GetArrayForWriteback` handles a non-updated read array.
  tensorstore::SharedArray<const void> prev_writeback_array;
  {
    auto read_array = MakeArray<int32_t>({{11, 12, 13}, {14, 15, 16}});
    Arena arena;
    auto writeback_data = async_write_array.GetArrayForWriteback(
        spec, domain, read_array,
        /*read_generation=*/StorageGeneration::FromString("a"));
    EXPECT_TRUE(writeback_data.must_store);
    prev_writeback_array = writeback_data.array;
    // Writeback does not reflect updated `read_array`.
    EXPECT_EQ(MakeArray<int32_t>({{7, 22, 23}, {31, 32, 33}}),
              writeback_data.array);
  }

  // Test that `GetReadNDIterable` handles an updated read array.
  {
    auto read_array = MakeArray<int32_t>({{11, 12, 13}, {14, 15, 16}});
    Arena arena;
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto iterable,
        async_write_array.GetReadNDIterable(
            spec, domain, read_array,
            /*read_generation=*/StorageGeneration::FromString("b"),
            tensorstore::IdentityTransform(tensorstore::Box<>({0, 0}, {2, 3})),
            &arena));
    EXPECT_EQ(MakeArray<int32_t>({{7, 12, 13}, {14, 15, 16}}),
              CopyNDIterable(std::move(iterable),
                             tensorstore::span<const Index>({2, 3}), &arena));
  }

  // Test that `GetArrayForWriteback` handles an updated read array.
  {
    auto read_array = MakeArray<int32_t>({{21, 22, 23}, {24, 25, 26}});
    Arena arena;
    auto writeback_data = async_write_array.GetArrayForWriteback(
        spec, domain, read_array,
        /*read_generation=*/StorageGeneration::FromString("c"));
    EXPECT_TRUE(writeback_data.must_store);
    // Writeback reflects updated `read_array`.
    EXPECT_EQ(MakeArray<int32_t>({{7, 22, 23}, {24, 25, 26}}),
              writeback_data.array);
    EXPECT_EQ(StorageGeneration::FromString("c"),
              async_write_array.read_generation);

    // Check that previous writeback array (which is supposed to be immutable)
    // remains unmodified.
    EXPECT_NE(prev_writeback_array, writeback_data.array);
  }

  // Test that `GetReadNDIterable` handles a write state fully-overwritten with
  // the fill value.
  async_write_array.write_state.WriteFillValue(spec, domain);
  {
    auto read_array = MakeArray<int32_t>({{11, 12, 13}, {14, 15, 16}});
    Arena arena;
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto iterable,
        async_write_array.GetReadNDIterable(
            spec, domain, read_array,
            /*read_generation=*/StorageGeneration::FromString("b"),
            tensorstore::IdentityTransform(tensorstore::Box<>({0, 0}, {2, 3})),
            &arena));
    EXPECT_EQ(fill_value_copy,
              CopyNDIterable(std::move(iterable),
                             tensorstore::span<const Index>({2, 3}), &arena));
  }

  // Test that `GetReadNDIterable` handles a write state fully-overwritten not
  // with the fill value.
  TestWrite(&async_write_array, spec, domain,
            tensorstore::MakeOffsetArray<int32_t>({0, 0}, {{9}}));
  {
    auto read_array = MakeArray<int32_t>({{11, 12, 13}, {14, 15, 16}});
    Arena arena;
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto iterable,
        async_write_array.GetReadNDIterable(
            spec, domain, read_array,
            /*read_generation=*/StorageGeneration::FromString("b"),
            tensorstore::IdentityTransform(tensorstore::Box<>({0, 0}, {2, 3})),
            &arena));
    EXPECT_EQ(MakeArray<int32_t>({{9, 22, 23}, {31, 32, 33}}),
              CopyNDIterable(std::move(iterable),
                             tensorstore::span<const Index>({2, 3}), &arena));
  }
}

// https://github.com/google/tensorstore/issues/144
TEST(AsyncWriteArrayTest, Issue144) {
  AsyncWriteArray async_write_array(1);
  auto overall_fill_value = MakeArray<int32_t>({0, 0});
  tensorstore::Box<> component_bounds(1);
  Spec spec{overall_fill_value, component_bounds};
  Box<> domain{{0}, {2}};
  // Overwrite position 1 with the fill value.
  TestWrite(&async_write_array, spec, domain,
            tensorstore::MakeOffsetArray<int32_t>({1}, {0}));

  // Get writeback data, assuming no existing `read_array`.
  {
    auto writeback_data = async_write_array.GetArrayForWriteback(
        spec, domain, /*read_array=*/{},
        /*read_generation=*/StorageGeneration::FromString("c"));
    EXPECT_EQ(spec.GetFillValueForDomain(domain), writeback_data.array);
    EXPECT_FALSE(writeback_data.must_store);
  }

  //  `GetArrayForWriteback` causes the data array to be destroyed.
  EXPECT_EQ(1, async_write_array.write_state.mask.num_masked_elements);
  EXPECT_FALSE(async_write_array.write_state.array.data());

  // Get writeback data again, assuming no existing `read_array`, to confirm
  // that the case of a partially-filled mask with no data array is handled
  // correctly.
  for (int i = 0; i < 2; ++i) {
    auto writeback_data = async_write_array.GetArrayForWriteback(
        spec, domain, /*read_array=*/{},
        /*read_generation=*/StorageGeneration::FromString("d"));
    EXPECT_EQ(spec.GetFillValueForDomain(domain), writeback_data.array);
    EXPECT_FALSE(writeback_data.must_store);
    EXPECT_EQ(1, async_write_array.write_state.mask.num_masked_elements);
    EXPECT_FALSE(async_write_array.write_state.array.data());
  }

  // Get writeback data with a new read array that is not equal to the fill
  // value.
  {
    auto writeback_data = async_write_array.GetArrayForWriteback(
        spec, domain, /*read_array=*/MakeArray<int32_t>({2, 2}),
        /*read_generation=*/StorageGeneration::FromString("e"));
    EXPECT_EQ(MakeArray<int32_t>({2, 0}), writeback_data.array);
    EXPECT_TRUE(writeback_data.must_store);
    EXPECT_EQ(1, async_write_array.write_state.mask.num_masked_elements);
    EXPECT_TRUE(async_write_array.write_state.array.data());
  }

  // Get writeback data with a new read array where the resultant array is equal
  // to the fill value.
  {
    auto writeback_data = async_write_array.GetArrayForWriteback(
        spec, domain, /*read_array=*/MakeArray<int32_t>({0, 2}),
        /*read_generation=*/StorageGeneration::FromString("f"));
    EXPECT_EQ(spec.GetFillValueForDomain(domain), writeback_data.array);
    EXPECT_FALSE(writeback_data.must_store);
    EXPECT_EQ(1, async_write_array.write_state.mask.num_masked_elements);
    EXPECT_FALSE(async_write_array.write_state.array.data());
  }
}

using WriteArraySourceCapabilities =
    AsyncWriteArray::WriteArraySourceCapabilities;
using ArrayCapabilities = AsyncWriteArray::MaskedArray::ArrayCapabilities;

void TestWriteArraySuccess(
    WriteArraySourceCapabilities source_capabilities,
    ArrayCapabilities expected_array_capabilities, bool may_retain_writeback,
    bool zero_copy, tensorstore::IndexTransformView<> chunk_transform,
    tensorstore::TransformedSharedArray<const void> source_array) {
  SCOPED_TRACE(tensorstore::StrCat("chunk_transform=", chunk_transform));
  AsyncWriteArray async_write_array(chunk_transform.output_rank());
  tensorstore::Box<> output_range(chunk_transform.output_rank());
  ASSERT_THAT(tensorstore::GetOutputRange(chunk_transform, output_range),
              ::testing::Optional(true));
  auto origin = output_range.origin();
  SCOPED_TRACE(tensorstore::StrCat("origin=", origin));
  auto fill_value =
      tensorstore::AllocateArray(output_range, tensorstore::c_order,
                                 tensorstore::value_init, source_array.dtype());
  tensorstore::Box<> component_bounds(chunk_transform.output_rank());
  Spec spec{fill_value, component_bounds};
  size_t orig_use_count = source_array.element_pointer().pointer().use_count();

  TENSORSTORE_ASSERT_OK(async_write_array.WriteArray(
      spec, output_range, chunk_transform,
      [&] { return std::pair{source_array, source_capabilities}; }));

  auto validate_zero_copy = [&](const auto& target_array,
                                size_t orig_use_count) {
    EXPECT_EQ((zero_copy ? orig_use_count + 1 : orig_use_count),
              source_array.element_pointer().pointer().use_count());
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto materialized_target_array,
        target_array |
            tensorstore::AllDims().TranslateTo(output_range.origin()) |
            chunk_transform | tensorstore::TryConvertToArray());

    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto materialized_source_array,
        source_array | tensorstore::TryConvertToArray());

    EXPECT_THAT(
        materialized_target_array,
        ::testing::Conditional(
            zero_copy, ReferencesSameDataAs(materialized_source_array),
            ::testing::Not(ReferencesSameDataAs(materialized_source_array))));
  };

  {
    SCOPED_TRACE(
        "Checking async_write_array.write_state.array before calling "
        "GetArrayForWriteback");
    validate_zero_copy(async_write_array.write_state.array, orig_use_count);
  }
  EXPECT_EQ(expected_array_capabilities,
            async_write_array.write_state.array_capabilities);

  {
    SCOPED_TRACE("Checking writeback_data");
    orig_use_count = source_array.element_pointer().pointer().use_count();
    auto writeback_data = async_write_array.GetArrayForWriteback(
        spec, output_range, /*read_array=*/{},
        /*read_generation=*/StorageGeneration::Invalid());

    validate_zero_copy(writeback_data.array, orig_use_count);

    EXPECT_EQ(may_retain_writeback,
              writeback_data.may_retain_reference_to_array_indefinitely);
    EXPECT_EQ(expected_array_capabilities,
              async_write_array.write_state.array_capabilities);
  }
}

absl::Status TestWriteArrayError(
    WriteArraySourceCapabilities source_capabilities, tensorstore::Box<> box,
    tensorstore::IndexTransformView<> chunk_transform,
    tensorstore::TransformedSharedArray<const void> source_array) {
  AsyncWriteArray async_write_array(chunk_transform.output_rank());
  auto fill_value = tensorstore::AllocateArray(
      box, tensorstore::c_order, tensorstore::value_init, source_array.dtype());
  tensorstore::Box<> component_bounds(chunk_transform.output_rank());
  Spec spec{fill_value, component_bounds};

  return async_write_array.WriteArray(spec, box, chunk_transform, [&] {
    return std::pair{source_array, source_capabilities};
  });
}

void TestWriteArrayIdentityTransformSuccess(
    WriteArraySourceCapabilities source_capabilities,
    ArrayCapabilities expected_array_capabilities, bool may_retain_writeback,
    bool zero_copy) {
  auto source_array = MakeArray<int32_t>({{7, 8, 9}, {10, 11, 12}});
  auto chunk_transform = tensorstore::IdentityTransform(source_array.shape());
  TestWriteArraySuccess(source_capabilities, expected_array_capabilities,
                        may_retain_writeback, zero_copy, chunk_transform,
                        source_array);
}

TEST(WriteArrayIdentityTransformSuccessTest, kCannotRetain) {
  TestWriteArrayIdentityTransformSuccess(
      WriteArraySourceCapabilities::kCannotRetain,
      AsyncWriteArray::MaskedArray::kMutableArray,
      /*may_retain_writeback=*/true,
      /*zero_copy=*/false);
}
TEST(WriteArrayIdentityTransformSuccessTest,
     kImmutableAndCanRetainIndefinitely) {
  TestWriteArrayIdentityTransformSuccess(
      WriteArraySourceCapabilities::kImmutableAndCanRetainIndefinitely,
      AsyncWriteArray::MaskedArray::kImmutableAndCanRetainIndefinitely,
      /*may_retain_writeback=*/true,
      /*zero_copy=*/true);
}
TEST(WriteArrayIdentityTransformSuccessTest,
     kImmutableAndCanRetainUntilCommit) {
  TestWriteArrayIdentityTransformSuccess(
      WriteArraySourceCapabilities::kImmutableAndCanRetainUntilCommit,
      AsyncWriteArray::MaskedArray::kImmutableAndCanRetainUntilCommit,
      /*may_retain_writeback=*/false,
      /*zero_copy=*/true);
}

TEST(WriteArrayIdentityTransformSuccessTest, kMutable) {
  TestWriteArrayIdentityTransformSuccess(
      WriteArraySourceCapabilities::kMutable,
      AsyncWriteArray::MaskedArray::kMutableArray,
      /*may_retain_writeback=*/true,
      /*zero_copy=*/true);
}

TEST(WriteArrayNonIdentityTransformSuccess, kMutable) {
  std::minstd_rand gen{tensorstore::internal_testing::GetRandomSeedForTest(
      "TENSORSTORE_INTERNAL_ASYNC_WRITE_ARRAY")};
  tensorstore::SharedArray<const void> base_source_array =
      MakeArray<int32_t>({{7, 8, 9}, {10, 11, 12}});

  constexpr size_t kNumIterations = 10;
  for (size_t iter_i = 0; iter_i < kNumIterations; ++iter_i) {
    tensorstore::IndexTransform<> source_transform;
    {
      tensorstore::internal::MakeStridedIndexTransformForOutputSpaceParameters
          p;
      p.max_stride = 2;
      source_transform =
          tensorstore::internal::MakeRandomStridedIndexTransformForOutputSpace(
              gen, tensorstore::IndexDomain<>(base_source_array.domain()), p);
    }
    SCOPED_TRACE(tensorstore::StrCat("source_transform=", source_transform));
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto source_array,
                                     base_source_array | source_transform);
    auto chunk_transform =
        tensorstore::internal::MakeRandomStridedIndexTransformForInputSpace(
            gen, source_array.domain());
    TestWriteArraySuccess(WriteArraySourceCapabilities::kMutable,
                          AsyncWriteArray::MaskedArray::kMutableArray,
                          /*may_retain_writeback=*/true,
                          /*zero_copy=*/true, chunk_transform, source_array);
  }
}

TEST(WriteArrayErrorTest, SourceArrayIndexArrayMap) {
  tensorstore::SharedArray<const void> base_source_array =
      MakeArray<int32_t>({{7, 8, 9}, {10, 11, 12}});
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto source_array,
      base_source_array | tensorstore::Dims(1).OuterIndexArraySlice(
                              tensorstore::MakeArray<Index>({1, 0, 1, 1, 2})));
  auto chunk_transform = tensorstore::IdentityTransform(source_array.domain());
  EXPECT_THAT(TestWriteArrayError(WriteArraySourceCapabilities::kMutable,
                                  tensorstore::Box<>({2, 5}), chunk_transform,
                                  source_array),
              tensorstore::MatchesStatus(absl::StatusCode::kCancelled));
}

TEST(WriteArrayErrorTest, ChunkTransformIndexArrayMap) {
  tensorstore::SharedArray<const void> base_source_array =
      MakeArray<int32_t>({{7, 8, 9}, {10, 11, 12}});
  tensorstore::TransformedSharedArray<const void> source_array =
      base_source_array;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(
      auto chunk_transform,
      tensorstore::IdentityTransform(source_array.domain()) |
          tensorstore::Dims(1).OuterIndexArraySlice(
              tensorstore::MakeArray<Index>({0, 1, 2})));
  EXPECT_THAT(TestWriteArrayError(WriteArraySourceCapabilities::kMutable,
                                  tensorstore::Box<>({2, 3}), chunk_transform,
                                  source_array),
              tensorstore::MatchesStatus(absl::StatusCode::kCancelled));
}

}  // namespace
