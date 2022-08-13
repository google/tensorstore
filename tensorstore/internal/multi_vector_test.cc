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

#include "tensorstore/internal/multi_vector.h"

#include <cstddef>
#include <type_traits>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/rank.h"
#include "tensorstore/util/span.h"

namespace {

using ::tensorstore::dynamic_rank;
using ::tensorstore::span;
using ::tensorstore::internal::MultiVectorAccess;
using ::tensorstore::internal::MultiVectorStorage;
using ::tensorstore::internal::MultiVectorStorageImpl;
using ::tensorstore::internal_multi_vector::GetAlignedOffset;
using ::tensorstore::internal_multi_vector::PackStorageOffsets;
using ::testing::ElementsAre;

static_assert(
    MultiVectorAccess<MultiVectorStorage<3, int, float>>::static_extent == 3);
static_assert(
    MultiVectorAccess<MultiVectorStorage<3, int, float>>::num_vectors == 2);

static_assert(GetAlignedOffset(0, 4, 4) == 0);
static_assert(GetAlignedOffset(4, 4, 4) == 4);
static_assert(GetAlignedOffset(4, 4, 8) == 8);
static_assert(GetAlignedOffset(4, 4, 8) == 8);

template <std::size_t Len, std::size_t Align>
using Aligned = typename std::aligned_storage<Len, Align>::type;

static_assert(PackStorageOffsets<Aligned<4, 4>>::GetVectorOffset(5, 0) == 0);
static_assert(PackStorageOffsets<Aligned<4, 4>>::GetVectorOffset(5, 1) ==
              5 * 4);
static_assert(PackStorageOffsets<Aligned<4, 4>>::GetTotalSize(5) == 5 * 4);

static_assert(PackStorageOffsets<Aligned<4, 4>, Aligned<4, 4>>::GetVectorOffset(
                  5, 0) == 0);
static_assert(PackStorageOffsets<Aligned<4, 4>, Aligned<4, 4>>::GetVectorOffset(
                  5, 1) == 5 * 4);
static_assert(PackStorageOffsets<Aligned<4, 4>, Aligned<4, 4>>::GetVectorOffset(
                  5, 2) == 2 * 5 * 4);
static_assert(PackStorageOffsets<Aligned<4, 4>, Aligned<4, 4>>::GetTotalSize(
                  5) == 2 * 5 * 4);

static_assert(PackStorageOffsets<Aligned<4, 4>, Aligned<4, 4>,
                                 Aligned<4, 4>>::GetVectorOffset(5, 0) == 0);
static_assert(PackStorageOffsets<Aligned<4, 4>, Aligned<4, 4>,
                                 Aligned<4, 4>>::GetVectorOffset(5, 1) ==
              5 * 4);
static_assert(PackStorageOffsets<Aligned<4, 4>, Aligned<4, 4>,
                                 Aligned<4, 4>>::GetVectorOffset(5, 2) ==
              2 * 5 * 4);
static_assert(PackStorageOffsets<Aligned<4, 4>, Aligned<4, 4>,
                                 Aligned<4, 4>>::GetVectorOffset(5, 3) ==
              3 * 5 * 4);

static_assert(PackStorageOffsets<Aligned<4, 4>, Aligned<8, 8>,
                                 Aligned<4, 4>>::GetTotalSize(5) ==
              4 * 6 + 8 * 5 + 4 * 5);

static_assert(PackStorageOffsets<Aligned<8, 8>, Aligned<4, 4>>::GetVectorOffset(
                  5, 0) == 0);
static_assert(PackStorageOffsets<Aligned<8, 8>, Aligned<4, 4>>::GetVectorOffset(
                  5, 1) == 8 * 5);
static_assert(PackStorageOffsets<Aligned<8, 8>, Aligned<4, 4>>::GetVectorOffset(
                  5, 2) == 8 * 5 + 4 * 5);

template <typename StorageType>
class MultiVectorDynamicTest : public ::testing::Test {};

using DynamicStorageTypes =
    ::testing::Types<MultiVectorStorage<dynamic_rank, int, int>,
                     MultiVectorStorage<dynamic_rank, int, float>,
                     MultiVectorStorage<dynamic_rank, int, double>,
                     MultiVectorStorage<dynamic_rank, double, int>,

                     MultiVectorStorage<dynamic_rank(2), int, int>,
                     MultiVectorStorage<dynamic_rank(2), int, float>,
                     MultiVectorStorage<dynamic_rank(2), int, double>,
                     MultiVectorStorage<dynamic_rank(2), double, int>,

                     MultiVectorStorage<dynamic_rank(3), int, int>,
                     MultiVectorStorage<dynamic_rank(3), int, float>,
                     MultiVectorStorage<dynamic_rank(3), int, double>,
                     MultiVectorStorage<dynamic_rank(3), double, int>,

                     MultiVectorStorage<dynamic_rank(4), int, int>,
                     MultiVectorStorage<dynamic_rank(4), int, float>,
                     MultiVectorStorage<dynamic_rank(4), int, double>,
                     MultiVectorStorage<dynamic_rank(4), double, int>>;

TYPED_TEST_SUITE(MultiVectorDynamicTest, DynamicStorageTypes);

template <typename T>
struct Decompose;

template <std::ptrdiff_t Extent, std::ptrdiff_t InlineSize, typename T0,
          typename T1>
struct Decompose<MultiVectorStorageImpl<Extent, InlineSize, T0, T1>> {
  constexpr static std::ptrdiff_t inline_size = InlineSize;
  constexpr static std::ptrdiff_t extent = Extent;
  using Element0 = T0;
  using Element1 = T1;
};

TYPED_TEST(MultiVectorDynamicTest, Basic) {
  using Container = TypeParam;
  using D = Decompose<Container>;
  using T0 = typename D::Element0;
  using T1 = typename D::Element1;
  using Access = MultiVectorAccess<Container>;
  static_assert(std::is_same_v<T0, typename Access::template ElementType<0>>);
  static_assert(std::is_same_v<T1, typename Access::template ElementType<1>>);
  static_assert(
      std::is_same_v<const T0, typename Access::template ConstElementType<0>>);
  static_assert(
      std::is_same_v<const T1, typename Access::template ConstElementType<1>>);

  // Test default construction.
  Container vec;
  EXPECT_EQ(0, Access::GetExtent(vec));

  Access::Resize(&vec, 3);
  EXPECT_EQ(3, Access::GetExtent(vec));

  // Test (rank, pointer...) assignment.
  const T0 a0[] = {1, 2, 3, 4};
  const T1 a1[] = {5, 6, 7, 8};
  Access::Assign(&vec, 4, a0, a1);
  EXPECT_THAT(Access::template get<0>(&vec), ElementsAre(1, 2, 3, 4));
  EXPECT_THAT(Access::template get<1>(&vec), ElementsAre(5, 6, 7, 8));

  // Test (span...) assignment.
  Access::Assign(&vec, span<const T0>({4, 5, 6}), span<const T1>({7, 8, 9}));
  EXPECT_THAT(Access::template get<0>(&vec), ElementsAre(4, 5, 6));
  EXPECT_THAT(Access::template get<1>(&vec), ElementsAre(7, 8, 9));

  // Test copy construction.
  {
    Container vec2 = vec;
    EXPECT_THAT(Access::template get<0>(&vec2), ElementsAre(4, 5, 6));
    EXPECT_THAT(Access::template get<1>(&vec2), ElementsAre(7, 8, 9));
    EXPECT_THAT(Access::template get<0>(&vec), ElementsAre(4, 5, 6));
    EXPECT_THAT(Access::template get<1>(&vec), ElementsAre(7, 8, 9));
    EXPECT_NE(Access::template get<0>(&vec2).data(),
              Access::template get<0>(&vec).data());

    // Test move construction.
    {
      T0* ptr0 = Access::template get<0>(&vec2).data();
      T1* ptr1 = Access::template get<1>(&vec2).data();
      Container vec3 = std::move(vec2);
      EXPECT_THAT(Access::template get<0>(&vec3), ElementsAre(4, 5, 6));
      EXPECT_THAT(Access::template get<1>(&vec3), ElementsAre(7, 8, 9));
      EXPECT_EQ(0, Access::GetExtent(vec2));  // NOLINT
      if (D::inline_size < 3) {
        EXPECT_EQ(ptr0, Access::template get<0>(&vec3).data());
        EXPECT_EQ(ptr1, Access::template get<1>(&vec3).data());
      }
    }
  }

  // Test copy assignment.
  {
    Container vec4;
    vec4 = vec;
    EXPECT_THAT(Access::template get<0>(&vec), ElementsAre(4, 5, 6));
    EXPECT_THAT(Access::template get<1>(&vec), ElementsAre(7, 8, 9));
    EXPECT_THAT(Access::template get<0>(&vec4), ElementsAre(4, 5, 6));
    EXPECT_THAT(Access::template get<1>(&vec4), ElementsAre(7, 8, 9));
    EXPECT_NE(Access::template get<0>(&vec).data(),
              Access::template get<0>(&vec4).data());

    // Test move assignment.
    {
      T0* ptr0 = Access::template get<0>(&vec4).data();
      T1* ptr1 = Access::template get<1>(&vec4).data();
      Container vec5;
      vec5 = std::move(vec4);
      EXPECT_THAT(Access::template get<0>(&vec5), ElementsAre(4, 5, 6));
      EXPECT_THAT(Access::template get<1>(&vec5), ElementsAre(7, 8, 9));
      EXPECT_EQ(0, Access::GetExtent(vec4));  // NOLINT
      if (D::inline_size < 3) {
        EXPECT_EQ(ptr0, Access::template get<0>(&vec5).data());
        EXPECT_EQ(ptr1, Access::template get<1>(&vec5).data());
      }
    }
  }
}

template <typename StorageType>
class MultiVectorStaticTest : public ::testing::Test {};

using StaticStorageTypes = ::testing::Types<
    MultiVectorStorage<2, int, int>, MultiVectorStorage<2, int, float>,
    MultiVectorStorage<2, int, double>, MultiVectorStorage<2, double, int>>;

TYPED_TEST_SUITE(MultiVectorStaticTest, StaticStorageTypes);

TYPED_TEST(MultiVectorStaticTest, Basic) {
  using Container = TypeParam;
  using Access = MultiVectorAccess<Container>;
  using D = Decompose<Container>;
  using T0 = typename D::Element0;
  using T1 = typename D::Element1;
  static_assert(std::is_same_v<T0, typename Access::template ElementType<0>>);
  static_assert(std::is_same_v<T1, typename Access::template ElementType<1>>);
  static_assert(
      std::is_same_v<const T0, typename Access::template ConstElementType<0>>);
  static_assert(
      std::is_same_v<const T1, typename Access::template ConstElementType<1>>);

  // Test default construction.
  Container vec;
  static_assert(std::is_same_v<std::integral_constant<std::ptrdiff_t, 2>,
                               decltype(Access::GetExtent(vec))>);
  EXPECT_EQ(2, Access::GetExtent(vec));

  // Test Resize (no op).
  Access::Resize(&vec, std::integral_constant<std::ptrdiff_t, 2>());

  // Test (rank, pointer...) assignment.
  const T0 a0[] = {1, 2};
  const T1 a1[] = {5, 6};
  Access::Assign(&vec, std::integral_constant<std::ptrdiff_t, 2>(), a0, a1);
  EXPECT_THAT(Access::template get<0>(&vec), ElementsAre(1, 2));
  EXPECT_THAT(Access::template get<1>(&vec), ElementsAre(5, 6));

  // Test (span...) assignment.
  Access::Assign(&vec, span<const T0, 2>({4, 5}), span<const T1, 2>({7, 8}));
  EXPECT_THAT(Access::template get<0>(&vec), ElementsAre(4, 5));
  EXPECT_THAT(Access::template get<1>(&vec), ElementsAre(7, 8));

  // Test copy construction.
  {
    Container vec2 = vec;
    EXPECT_THAT(Access::template get<0>(&vec2), ElementsAre(4, 5));
    EXPECT_THAT(Access::template get<1>(&vec2), ElementsAre(7, 8));
    EXPECT_THAT(Access::template get<0>(&vec), ElementsAre(4, 5));
    EXPECT_THAT(Access::template get<1>(&vec), ElementsAre(7, 8));
    EXPECT_NE(Access::template get<0>(&vec2).data(),
              Access::template get<0>(&vec).data());

    // Test move construction.
    {
      Container vec3 = std::move(vec2);
      EXPECT_THAT(Access::template get<0>(&vec3), ElementsAre(4, 5));
      EXPECT_THAT(Access::template get<1>(&vec3), ElementsAre(7, 8));
    }
  }

  // Test copy assignment.
  {
    Container vec4;
    vec4 = vec;
    EXPECT_THAT(Access::template get<0>(&vec), ElementsAre(4, 5));
    EXPECT_THAT(Access::template get<1>(&vec), ElementsAre(7, 8));
    EXPECT_THAT(Access::template get<0>(&vec4), ElementsAre(4, 5));
    EXPECT_THAT(Access::template get<1>(&vec4), ElementsAre(7, 8));

    // Test move assignment.
    {
      Container vec5;
      vec5 = std::move(vec4);
      EXPECT_THAT(Access::template get<0>(&vec5), ElementsAre(4, 5));
      EXPECT_THAT(Access::template get<1>(&vec5), ElementsAre(7, 8));
    }
  }
}

}  // namespace
