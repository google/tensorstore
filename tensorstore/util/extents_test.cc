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

#include "tensorstore/util/extents.h"

#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>
#include "tensorstore/index.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/span.h"

namespace {
using tensorstore::dynamic_extent;
using tensorstore::Index;
using tensorstore::IsCompatibleFullIndexVector;
using tensorstore::IsCompatiblePartialIndexVector;
using tensorstore::IsImplicitlyCompatibleFullIndexVector;
using tensorstore::IsIndexConvertibleVector;
using tensorstore::IsIndexVector;
using tensorstore::IsMutableIndexVector;
using tensorstore::ProductOfExtents;
using tensorstore::span;
using tensorstore::SpanStaticExtent;

static_assert(IsCompatibleFullIndexVector<3, int (&)[3]>::value, "");
static_assert(IsCompatibleFullIndexVector<dynamic_extent, int (&)[3]>::value,
              "");
static_assert(IsCompatibleFullIndexVector<3, span<int, 3>>::value, "");
static_assert(IsCompatibleFullIndexVector<3, span<int>>::value, "");
static_assert(IsCompatibleFullIndexVector<dynamic_extent, span<int>>::value,
              "");
static_assert(IsCompatibleFullIndexVector<dynamic_extent, span<int, 3>>::value,
              "");
static_assert(!IsCompatibleFullIndexVector<3, span<int, 2>>::value, "");
static_assert(!IsCompatibleFullIndexVector<3, span<float, 3>>::value, "");
static_assert(!IsCompatibleFullIndexVector<3, span<float, 2>>::value, "");

static_assert(IsCompatiblePartialIndexVector<3, int (&)[3]>::value, "");
static_assert(IsCompatiblePartialIndexVector<4, int (&)[3]>::value, "");
static_assert(IsCompatiblePartialIndexVector<dynamic_extent, int (&)[3]>::value,
              "");
static_assert(IsCompatiblePartialIndexVector<3, span<int, 3>>::value, "");
static_assert(IsCompatiblePartialIndexVector<4, span<int, 3>>::value, "");
static_assert(IsCompatiblePartialIndexVector<3, span<int>>::value, "");
static_assert(IsCompatiblePartialIndexVector<dynamic_extent, span<int>>::value,
              "");
static_assert(
    IsCompatiblePartialIndexVector<dynamic_extent, span<int, 3>>::value, "");
static_assert(!IsCompatiblePartialIndexVector<3, span<int, 4>>::value, "");
static_assert(!IsCompatiblePartialIndexVector<3, span<float, 3>>::value, "");
static_assert(!IsCompatiblePartialIndexVector<3, span<float, 2>>::value, "");

static_assert(IsImplicitlyCompatibleFullIndexVector<3, int (&)[3]>::value, "");
static_assert(
    IsImplicitlyCompatibleFullIndexVector<dynamic_extent, int (&)[3]>::value,
    "");
static_assert(IsImplicitlyCompatibleFullIndexVector<3, span<int, 3>>::value,
              "");
static_assert(
    IsImplicitlyCompatibleFullIndexVector<dynamic_extent, span<int>>::value,
    "");
static_assert(!IsImplicitlyCompatibleFullIndexVector<3, span<int>>::value, "");
static_assert(!IsImplicitlyCompatibleFullIndexVector<3, span<float, 3>>::value,
              "");
static_assert(!IsImplicitlyCompatibleFullIndexVector<3, span<float, 2>>::value,
              "");

static_assert(IsIndexConvertibleVector<span<int>>::value, "");
static_assert(IsIndexConvertibleVector<span<int, 3>>::value, "");
static_assert(IsIndexConvertibleVector<std::vector<int>>::value, "");
static_assert(!IsIndexConvertibleVector<span<float, 3>>::value, "");

static_assert(IsIndexVector<span<Index>>::value, "");
static_assert(IsIndexVector<span<Index, 3>>::value, "");
static_assert(IsIndexVector<span<const Index>>::value, "");
static_assert(IsIndexVector<span<const Index, 3>>::value, "");
static_assert(IsIndexVector<span<const Index>>::value, "");
static_assert(IsIndexVector<std::vector<Index>>::value, "");
static_assert(IsIndexVector<const std::vector<Index>>::value, "");
static_assert(!IsIndexVector<span<int, 3>>::value, "");
static_assert(!IsIndexVector<span<float>>::value, "");

static_assert(IsMutableIndexVector<span<Index>>::value, "");
static_assert(IsMutableIndexVector<span<Index, 3>>::value, "");
static_assert(!IsMutableIndexVector<span<const Index>>::value, "");
static_assert(!IsMutableIndexVector<span<const Index, 3>>::value, "");
static_assert(IsMutableIndexVector<std::vector<Index>&>::value, "");
static_assert(!IsMutableIndexVector<const std::vector<Index>>::value, "");
static_assert(!IsMutableIndexVector<span<int, 3>>::value, "");
static_assert(!IsMutableIndexVector<span<float>>::value, "");

static_assert(SpanStaticExtent<std::vector<int>>() == dynamic_extent, "");
static_assert(SpanStaticExtent<span<int, 3>>() == 3, "");
static_assert(SpanStaticExtent<span<int>>() == dynamic_extent, "");

static_assert(SpanStaticExtent<std::vector<int>, span<int>>() == dynamic_extent,
              "");
static_assert(SpanStaticExtent<span<int, 3>, span<float, 3>>() == 3, "");

TEST(ProductOfExtentsTest, Basic) {
  EXPECT_EQ(1, ProductOfExtents(span<int, 0>()));
  EXPECT_EQ(20, ProductOfExtents(span({4, 5})));
}

TEST(ProductOfExtentsTest, Overflow) {
  EXPECT_EQ(0, ProductOfExtents(span<const int>(
                   {5, std::numeric_limits<int>::max() - 1, 0})));
  EXPECT_EQ(std::numeric_limits<int>::max(),
            ProductOfExtents(
                span<const int>({5, std::numeric_limits<int>::max() - 1})));
  EXPECT_EQ(std::numeric_limits<std::int64_t>::max(),
            ProductOfExtents(
                span<const std::int64_t>({32768, 32768, 32768, 32768, 32768})));
  EXPECT_EQ(0, ProductOfExtents(span<const int>(
                   {5, std::numeric_limits<int>::max() - 1, 0})));
}
}  // namespace
