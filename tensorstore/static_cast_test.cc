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

#include "tensorstore/static_cast.h"

#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

namespace {

using ::tensorstore::dynamic_extent;
using ::tensorstore::IsStaticCastConstructible;
using ::tensorstore::MatchesStatus;
using ::tensorstore::Result;
using ::tensorstore::StaticCast;
using ::tensorstore::unchecked;
using ::tensorstore::unchecked_t;

/// Define a type that follows the `unchecked_t` construction convention.
template <ptrdiff_t Extent>
struct X {
  X(tensorstore::span<int, Extent> data) : data(data) {}

  template <ptrdiff_t OtherExtent,
            std::enable_if_t<(OtherExtent == Extent ||
                              OtherExtent == dynamic_extent ||
                              Extent == dynamic_extent)>* = nullptr>
  explicit X(unchecked_t, X<OtherExtent> other)
      : data(other.data.data(), other.data.size()) {}
  tensorstore::span<int, Extent> data;
};

/// Define a type that does not follow the `unchecked_t` construction
/// convention.
template <ptrdiff_t Extent>
struct Y {
  Y(tensorstore::span<int, Extent> data) : data(data) {}
  tensorstore::span<int, Extent> data;
};

}  // namespace

namespace tensorstore {
/// Specialize `StaticCastTraits` for `X<Extent>`, using
/// `DefaultStaticCastTraits<X<Extent>>` as a base class.
template <ptrdiff_t Extent>
struct StaticCastTraits<X<Extent>> : public DefaultStaticCastTraits<X<Extent>> {
  template <typename Other>
  static bool IsCompatible(const Other& other) {
    return other.data.size() == Extent || Extent == tensorstore::dynamic_extent;
  }

  static std::string Describe() { return StrCat("X with extent of ", Extent); }

  static std::string Describe(const X<Extent>& value) {
    return StrCat("X with extent of ", value.data.size());
  }
};

/// Specialize `StaticCastTraits` for `Y<Extent>`.
template <ptrdiff_t Extent>
struct StaticCastTraits<Y<Extent>> {
  /// Define custom `Construct` function.
  template <ptrdiff_t OtherExtent,
            std::enable_if_t<(OtherExtent == Extent ||
                              OtherExtent == dynamic_extent ||
                              Extent == dynamic_extent)>* = nullptr>
  static Y<Extent> Construct(Y<OtherExtent> other) {
    return Y<Extent>(
        tensorstore::span<int, Extent>(other.data.data(), other.data.size()));
  }

  template <typename Other>
  static bool IsCompatible(const Other& other) {
    return other.data.size() == Extent || Extent == tensorstore::dynamic_extent;
  }

  static std::string Describe() { return StrCat("Y with extent of ", Extent); }

  static std::string Describe(const Y<Extent>& value) {
    return StrCat("Y with extent of ", value.data.size());
  }
};

}  // namespace tensorstore

namespace {

// Test IsStaticCastConstructible
static_assert(IsStaticCastConstructible<X<3>, X<dynamic_extent>>);
static_assert(IsStaticCastConstructible<X<dynamic_extent>, X<3>>);
static_assert(IsStaticCastConstructible<X<3>, X<3>>);
static_assert(!IsStaticCastConstructible<X<3>, X<2>>);

static_assert(IsStaticCastConstructible<Y<3>, Y<dynamic_extent>>);
static_assert(IsStaticCastConstructible<Y<dynamic_extent>, Y<3>>);
static_assert(IsStaticCastConstructible<Y<3>, Y<3>>);
static_assert(!IsStaticCastConstructible<Y<3>, Y<2>>);

// Test unchecked no-op casting result type.
static_assert(std::is_same_v<const X<3>&, decltype(StaticCast<X<3>, unchecked>(
                                              std::declval<const X<3>&>()))>);
static_assert(std::is_same_v<X<3>&, decltype(StaticCast<X<3>, unchecked>(
                                        std::declval<X<3>&>()))>);
static_assert(std::is_same_v<X<3>&&, decltype(StaticCast<X<3>, unchecked>(
                                         std::declval<X<3>&&>()))>);

// Test unchecked regular result type.
static_assert(
    std::is_same_v<X<3>, decltype(StaticCast<X<3>, unchecked>(
                             std::declval<const X<dynamic_extent>&>()))>);
static_assert(std::is_same_v<X<3>, decltype(StaticCast<X<3>, unchecked>(
                                       std::declval<X<dynamic_extent>&>()))>);

// Test checked no-op casting result type.
static_assert(std::is_same_v<Result<X<3>>, decltype(StaticCast<X<3>>(
                                               std::declval<const X<3>&>()))>);
static_assert(std::is_same_v<
              Result<X<3>>, decltype(StaticCast<X<3>>(std::declval<X<3>&>()))>);

// Test checked regular result type.
static_assert(std::is_same_v<Result<X<3>>,
                             decltype(StaticCast<X<3>>(
                                 std::declval<const X<dynamic_extent>&>()))>);
static_assert(
    std::is_same_v<Result<X<3>>, decltype(StaticCast<X<3>>(
                                     std::declval<X<dynamic_extent>&>()))>);

TEST(DefaultCastTraitsTest, Success) {
  std::vector<int> vec{1, 2, 3};
  X<dynamic_extent> x(vec);
  auto cast_result = StaticCast<X<3>>(x);
  static_assert(std::is_same_v<decltype(cast_result), Result<X<3>>>);
  ASSERT_TRUE(cast_result);
  EXPECT_EQ(vec.data(), cast_result->data.data());

  auto& noop_cast_result = StaticCast<X<dynamic_extent>, unchecked>(x);
  EXPECT_EQ(&noop_cast_result, &x);

  auto unchecked_cast_result = StaticCast<X<3>, unchecked>(x);
  static_assert(std::is_same_v<decltype(unchecked_cast_result), X<3>>);
}

TEST(DefaultCastTraitsTest, CheckedFailure) {
  std::vector<int> vec{1, 2, 3};
  X<dynamic_extent> x(vec);
  EXPECT_THAT(
      StaticCast<X<2>>(x),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Cannot cast X with extent of 3 to X with extent of 2"));
}

TEST(DefaultCastTraitsDeathTest, UncheckedFailure) {
  std::vector<int> vec{1, 2, 3};
  X<dynamic_extent> x(vec);
  EXPECT_DEBUG_DEATH((StaticCast<X<2>, unchecked>(x)),
                     "StaticCast is not valid");
}

TEST(CustomTraitsTest, Success) {
  std::vector<int> vec{1, 2, 3};
  Y<dynamic_extent> x(vec);
  auto cast_result = StaticCast<Y<3>>(x);
  static_assert(std::is_same_v<decltype(cast_result), Result<Y<3>>>);
  ASSERT_TRUE(cast_result);
  EXPECT_EQ(vec.data(), cast_result->data.data());

  auto& noop_cast_result = StaticCast<Y<dynamic_extent>, unchecked>(x);
  EXPECT_EQ(&noop_cast_result, &x);

  auto unchecked_cast_result = StaticCast<Y<3>, unchecked>(x);
  static_assert(std::is_same_v<decltype(unchecked_cast_result), Y<3>>);
}

TEST(CustomTraitsTest, CheckedFailure) {
  std::vector<int> vec{1, 2, 3};
  Y<dynamic_extent> x(vec);
  EXPECT_THAT(
      StaticCast<Y<2>>(x),
      MatchesStatus(absl::StatusCode::kInvalidArgument,
                    "Cannot cast Y with extent of 3 to Y with extent of 2"));
}

TEST(CustomTraitsDeathTest, UncheckedFailure) {
  std::vector<int> vec{1, 2, 3};
  Y<dynamic_extent> x(vec);
  EXPECT_DEBUG_DEATH((StaticCast<Y<2>, unchecked>(x)),
                     "StaticCast is not valid");
}

}  // namespace
