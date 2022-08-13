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

#include "tensorstore/internal/type_traits.h"

#include <tuple>
#include <type_traits>

#include <gtest/gtest.h>
#include "absl/base/attributes.h"

namespace {

using ::tensorstore::internal::CopyQualifiers;
using ::tensorstore::internal::FirstNonVoidType;
using ::tensorstore::internal::FirstType;
using ::tensorstore::internal::GetLValue;
using ::tensorstore::internal::IsConstConvertible;
using ::tensorstore::internal::IsConstConvertibleOrVoid;
using ::tensorstore::internal::IsEqualityComparable;
using ::tensorstore::internal::PossiblyEmptyObjectGetter;
using ::tensorstore::internal::remove_cvref_t;
using ::tensorstore::internal::type_identity_t;
using ::tensorstore::internal::TypePackElement;

static_assert(std::is_same_v<FirstNonVoidType<void, int>, int>);
static_assert(std::is_same_v<FirstNonVoidType<float, void>, float>);
static_assert(std::is_same_v<FirstNonVoidType<float, int>, float>);

namespace equality_comparable_tests {
struct X {};

static_assert(IsEqualityComparable<float, float>);
static_assert(IsEqualityComparable<float, int>);
static_assert(!IsEqualityComparable<X, X>);
}  // namespace equality_comparable_tests

static_assert(std::is_same_v<remove_cvref_t<const int&>, int>);
static_assert(std::is_same_v<remove_cvref_t<int&&>, int>);
static_assert(std::is_same_v<remove_cvref_t<const int&&>, int>);
static_assert(std::is_same_v<remove_cvref_t<const volatile int&&>, int>);

static_assert(std::is_same_v<CopyQualifiers<float, int>, int>);
static_assert(std::is_same_v<CopyQualifiers<const float, int>, const int>);
static_assert(std::is_same_v<CopyQualifiers<const float&, int>, const int&>);
static_assert(std::is_same_v<CopyQualifiers<const float, int&>, const int>);
static_assert(std::is_same_v<CopyQualifiers<float&&, const int&>, int&&>);

static_assert(std::is_same_v<int&, decltype(GetLValue(3))>);
static_assert(std::is_same_v<int*, decltype(&GetLValue(3))>);

static_assert(std::is_same_v<FirstType<void>, void>);
static_assert(std::is_same_v<FirstType<int, void>, int>);

static_assert(IsConstConvertible<int, int>);
static_assert(IsConstConvertible<void, void>);
static_assert(IsConstConvertible<void, const void>);
static_assert(IsConstConvertible<int, const int>);
static_assert(!IsConstConvertible<const int, int>);
static_assert(!IsConstConvertible<int, float>);
static_assert(!IsConstConvertible<int, const float>);
static_assert(!IsConstConvertible<int, const void>);
static_assert(!IsConstConvertible<const int, void>);
static_assert(!IsConstConvertible<int, void>);

static_assert(IsConstConvertibleOrVoid<int, int>);
static_assert(IsConstConvertibleOrVoid<int, const int>);
static_assert(IsConstConvertibleOrVoid<int, void>);
static_assert(IsConstConvertibleOrVoid<const int, void>);
static_assert(IsConstConvertibleOrVoid<int, const void>);
static_assert(!IsConstConvertibleOrVoid<const int, int>);
static_assert(!IsConstConvertibleOrVoid<int, float>);
static_assert(!IsConstConvertibleOrVoid<int, const float>);

static_assert(std::is_same_v<TypePackElement<0, int, float>, int>);
static_assert(std::is_same_v<TypePackElement<1, int, float>, float>);

template <std::size_t I, typename... Ts>
using NonBuiltinTypePackElement =
    typename std::tuple_element<I, std::tuple<Ts...>>::type;

static_assert(std::is_same_v<NonBuiltinTypePackElement<0, int, float>, int>);
static_assert(std::is_same_v<NonBuiltinTypePackElement<1, int, float>, float>);

TEST(PossiblyEmptyObjectGetterTest, Basic) {
  struct Empty {
    Empty() = delete;
    int foo() { return 3; }
  };

  {
    PossiblyEmptyObjectGetter<Empty> helper;
    Empty& obj = helper.get(nullptr);
    EXPECT_EQ(3, obj.foo());
  }
  {
    auto lambda = [](int x, int y) { return x + y; };
    using Lambda = decltype(lambda);
    PossiblyEmptyObjectGetter<Lambda> helper;
    Lambda& obj = helper.get(nullptr);
    EXPECT_EQ(3, obj(1, 2));
  }
  {
    int value = 3;
    PossiblyEmptyObjectGetter<int> helper;
    auto& obj = helper.get(&value);
    EXPECT_EQ(&value, &obj);
  }
}

static_assert(std::is_same_v<int, type_identity_t<int>>);


namespace explict_conversion_tests {
using ::tensorstore::internal::IsOnlyExplicitlyConvertible;
using ::tensorstore::internal::IsPairExplicitlyConvertible;
using ::tensorstore::internal::IsPairImplicitlyConvertible;
using ::tensorstore::internal::IsPairOnlyExplicitlyConvertible;

struct X {
  X(int) {}
  explicit X(float*) {}
};

static_assert(IsOnlyExplicitlyConvertible<float*, X>);
static_assert(std::is_convertible_v<int, X>);
static_assert(std::is_constructible_v<X, int>);
static_assert(!IsOnlyExplicitlyConvertible<int, X>);

struct Y {
  Y(int*) {}
  explicit Y(double*) {}
};

static_assert(IsPairImplicitlyConvertible<int, int*, X, Y>);
static_assert(IsPairExplicitlyConvertible<int, int*, X, Y>);
static_assert(IsPairExplicitlyConvertible<int, double*, X, Y>);
static_assert(IsPairExplicitlyConvertible<float*, int*, X, Y>);
static_assert(IsPairExplicitlyConvertible<float*, double*, X, Y>);
static_assert(!IsPairImplicitlyConvertible<int, double*, X, Y>);
static_assert(!IsPairImplicitlyConvertible<float*, int*, X, Y>);
static_assert(!IsPairImplicitlyConvertible<float*, double*, X, Y>);
static_assert(IsPairOnlyExplicitlyConvertible<int, double*, X, Y>);
static_assert(IsPairOnlyExplicitlyConvertible<float*, int*, X, Y>);
static_assert(IsPairOnlyExplicitlyConvertible<float*, double*, X, Y>);

}  // namespace explict_conversion_tests

TEST(DefaultConstructibleFunctionIfEmptyTest, Basic) {
  auto fn = [](int x) { return x + 1; };
  using Wrapper =
      tensorstore::internal::DefaultConstructibleFunctionIfEmpty<decltype(fn)>;
  static_assert(std::is_default_constructible_v<Wrapper>);
  EXPECT_EQ(4, Wrapper()(3));
  EXPECT_EQ(4, Wrapper(fn)(3));
}

}  // namespace
