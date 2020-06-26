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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/str_cat.h"

namespace {
using tensorstore::dynamic_extent;
using tensorstore::IsCastConstructible;
using tensorstore::MatchesStatus;
using tensorstore::Result;
using tensorstore::span;
using tensorstore::StaticCast;
using tensorstore::unchecked;
using tensorstore::unchecked_t;

/// Define a type that follows the `unchecked_t` construction convention.
template <std::ptrdiff_t Extent>
struct X {
  X(span<int, Extent> data) : data(data) {}

  template <std::ptrdiff_t OtherExtent,
            absl::enable_if_t<(OtherExtent == Extent ||
                               OtherExtent == dynamic_extent ||
                               Extent == dynamic_extent)>* = nullptr>
  explicit X(unchecked_t, X<OtherExtent> other)
      : data(other.data.data(), other.data.size()) {}
  span<int, Extent> data;
};

/// Define a type that does not follow the `unchecked_t` construction
/// convention.
template <std::ptrdiff_t Extent>
struct Y {
  Y(span<int, Extent> data) : data(data) {}
  span<int, Extent> data;
};

}  // namespace

namespace tensorstore {
/// Specialize `StaticCastTraits` for `X<Extent>`, using
/// `DefaultStaticCastTraits<X<Extent>>` as a base class.
template <std::ptrdiff_t Extent>
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
template <std::ptrdiff_t Extent>
struct StaticCastTraits<Y<Extent>> {
  /// Define custom `Construct` function.
  template <std::ptrdiff_t OtherExtent,
            absl::enable_if_t<(OtherExtent == Extent ||
                               OtherExtent == dynamic_extent ||
                               Extent == dynamic_extent)>* = nullptr>
  static Y<Extent> Construct(Y<OtherExtent> other) {
    return Y<Extent>(span<int, Extent>(other.data.data(), other.data.size()));
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

// Test IsCastConstructible
static_assert(IsCastConstructible<X<3>, X<dynamic_extent>>::value, "");
static_assert(IsCastConstructible<X<dynamic_extent>, X<3>>::value, "");
static_assert(IsCastConstructible<X<3>, X<3>>::value, "");
static_assert(!IsCastConstructible<X<3>, X<2>>::value, "");

static_assert(IsCastConstructible<Y<3>, Y<dynamic_extent>>::value, "");
static_assert(IsCastConstructible<Y<dynamic_extent>, Y<3>>::value, "");
static_assert(IsCastConstructible<Y<3>, Y<3>>::value, "");
static_assert(!IsCastConstructible<Y<3>, Y<2>>::value, "");

// Test unchecked no-op casting result type.
static_assert(
    std::is_same<const X<3>&, decltype(StaticCast<X<3>, unchecked>(
                                  std::declval<const X<3>&>()))>::value,
    "");
static_assert(std::is_same<X<3>&, decltype(StaticCast<X<3>, unchecked>(
                                      std::declval<X<3>&>()))>::value,
              "");
static_assert(std::is_same<X<3>&&, decltype(StaticCast<X<3>, unchecked>(
                                       std::declval<X<3>&&>()))>::value,
              "");

// Test unchecked regular result type.
static_assert(
    std::is_same<X<3>, decltype(StaticCast<X<3>, unchecked>(
                           std::declval<const X<dynamic_extent>&>()))>::value,
    "");
static_assert(
    std::is_same<X<3>, decltype(StaticCast<X<3>, unchecked>(
                           std::declval<X<dynamic_extent>&>()))>::value,
    "");

// Test checked no-op casting result type.
static_assert(
    std::is_same<Result<X<3>>, decltype(StaticCast<X<3>>(
                                   std::declval<const X<3>&>()))>::value,
    "");
static_assert(std::is_same<Result<X<3>>, decltype(StaticCast<X<3>>(
                                             std::declval<X<3>&>()))>::value,
              "");

// Test checked regular result type.
static_assert(
    std::is_same<Result<X<3>>,
                 decltype(StaticCast<X<3>>(
                     std::declval<const X<dynamic_extent>&>()))>::value,
    "");
static_assert(
    std::is_same<Result<X<3>>, decltype(StaticCast<X<3>>(
                                   std::declval<X<dynamic_extent>&>()))>::value,
    "");

TEST(DefaultCastTraitsTest, Success) {
  std::vector<int> vec{1, 2, 3};
  X<dynamic_extent> x(vec);
  auto cast_result = StaticCast<X<3>>(x);
  static_assert(std::is_same<decltype(cast_result), Result<X<3>>>::value, "");
  ASSERT_TRUE(cast_result);
  EXPECT_EQ(vec.data(), cast_result->data.data());

  auto& noop_cast_result = StaticCast<X<dynamic_extent>, unchecked>(x);
  EXPECT_EQ(&noop_cast_result, &x);

  auto unchecked_cast_result = StaticCast<X<3>, unchecked>(x);
  static_assert(std::is_same<decltype(unchecked_cast_result), X<3>>::value, "");
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
  static_assert(std::is_same<decltype(cast_result), Result<Y<3>>>::value, "");
  ASSERT_TRUE(cast_result);
  EXPECT_EQ(vec.data(), cast_result->data.data());

  auto& noop_cast_result = StaticCast<Y<dynamic_extent>, unchecked>(x);
  EXPECT_EQ(&noop_cast_result, &x);

  auto unchecked_cast_result = StaticCast<Y<3>, unchecked>(x);
  static_assert(std::is_same<decltype(unchecked_cast_result), Y<3>>::value, "");
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
