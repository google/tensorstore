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

#include "tensorstore/index_space/index_vector_or_scalar.h"

#include <stdint.h>

#include <system_error>  // NOLINT
#include <type_traits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorstore/index.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status_testutil.h"

namespace {

using ::tensorstore::dynamic_extent;
using ::tensorstore::Index;
using ::tensorstore::IsIndexVectorOrScalar;
using ::tensorstore::span;
using ::tensorstore::internal_index_space::CheckIndexVectorSize;
using ::tensorstore::internal_index_space::IndexVectorOrScalarView;

static_assert(IsIndexVectorOrScalar<Index>::value == true);

// Scalar types permit conversions.
static_assert(IsIndexVectorOrScalar<int32_t>::value == true);
static_assert(IsIndexVectorOrScalar<float>::value == false);
static_assert(std::is_same_v<
              typename IsIndexVectorOrScalar<int32_t>::normalized_type, Index>);
static_assert(IsIndexVectorOrScalar<int32_t>::extent == dynamic_extent);

// std::vector<int32_t> is not convertible to span<Index>.
static_assert(IsIndexVectorOrScalar<std::vector<int32_t>>::value == false);
static_assert(IsIndexVectorOrScalar<const std::vector<Index>>::value == true);
static_assert(std::is_same_v<typename IsIndexVectorOrScalar<
                                 const std::vector<Index>>::normalized_type,
                             span<const Index>>);
static_assert(IsIndexVectorOrScalar<const std::vector<Index>>::extent ==
              dynamic_extent);

static_assert(IsIndexVectorOrScalar<span<const Index>>::value == true);
static_assert(
    std::is_same_v<typename IsIndexVectorOrScalar<span<Index>>::normalized_type,
                   span<const Index>>);
static_assert(IsIndexVectorOrScalar<span<const Index>>::extent ==
              dynamic_extent);

static_assert(IsIndexVectorOrScalar<span<const Index, 5>>::value == true);
static_assert(std::is_same_v<
              typename IsIndexVectorOrScalar<span<Index, 5>>::normalized_type,
              span<const Index, 5>>);
static_assert(IsIndexVectorOrScalar<span<Index, 5>>::extent == 5);

TEST(IndexVectorOrScalarTest, Scalar) {
  IndexVectorOrScalarView v(5);
  EXPECT_EQ(5, v.size_or_scalar);
  EXPECT_EQ(nullptr, v.pointer);
  EXPECT_EQ(5, v[0]);
  EXPECT_EQ(5, v[1]);
  EXPECT_TRUE(CheckIndexVectorSize(v, 3).ok());
}

TEST(IndexVectorOrScalarTest, Vector) {
  const Index arr[] = {1, 2, 3};
  IndexVectorOrScalarView v{span(arr)};
  EXPECT_EQ(3, v.size_or_scalar);
  EXPECT_EQ(&arr[0], v.pointer);
  EXPECT_EQ(1, v[0]);
  EXPECT_EQ(2, v[1]);
  EXPECT_EQ(3, v[2]);
  EXPECT_TRUE(CheckIndexVectorSize(v, 3).ok());
  EXPECT_THAT(CheckIndexVectorSize(v, 5),
              tensorstore::MatchesStatus(absl::StatusCode::kInvalidArgument));
}

}  // namespace
