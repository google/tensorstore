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

using tensorstore::internal::CopyQualifiers;
using tensorstore::internal::FirstNonVoidType;
using tensorstore::internal::FirstType;
using tensorstore::internal::GetLValue;
using tensorstore::internal::IsConstConvertible;
using tensorstore::internal::IsConstConvertibleOrVoid;
using tensorstore::internal::IsEqualityComparable;
using tensorstore::internal::PossiblyEmptyObjectGetter;
using tensorstore::internal::remove_cvref_t;
using tensorstore::internal::type_identity_t;
using tensorstore::internal::TypePackElement;

static_assert(std::is_same<FirstNonVoidType<void, int>, int>::value, "");
static_assert(std::is_same<FirstNonVoidType<float, void>, float>::value, "");
static_assert(std::is_same<FirstNonVoidType<float, int>, float>::value, "");

namespace equality_comparable_tests {
struct X {};

static_assert(IsEqualityComparable<float, float>::value, "");
static_assert(IsEqualityComparable<float, int>::value, "");
static_assert(!IsEqualityComparable<X, X>::value, "");
}  // namespace equality_comparable_tests

static_assert(std::is_same<remove_cvref_t<const int&>, int>::value, "");
static_assert(std::is_same<remove_cvref_t<int&&>, int>::value, "");
static_assert(std::is_same<remove_cvref_t<const int&&>, int>::value, "");
static_assert(std::is_same<remove_cvref_t<const volatile int&&>, int>::value,
              "");

static_assert(std::is_same<CopyQualifiers<float, int>, int>::value, "");
static_assert(std::is_same<CopyQualifiers<const float, int>, const int>::value,
              "");
static_assert(
    std::is_same<CopyQualifiers<const float&, int>, const int&>::value, "");
static_assert(std::is_same<CopyQualifiers<const float, int&>, const int>::value,
              "");
static_assert(std::is_same<CopyQualifiers<float&&, const int&>, int&&>::value,
              "");

static_assert(std::is_same<int&, decltype(GetLValue(3))>::value, "");
static_assert(std::is_same<int*, decltype(&GetLValue(3))>::value, "");

static_assert(std::is_same<FirstType<void>, void>::value, "");
static_assert(std::is_same<FirstType<int, void>, int>::value, "");

static_assert(IsConstConvertible<int, int>::value, "");
static_assert(IsConstConvertible<void, void>::value, "");
static_assert(IsConstConvertible<void, const void>::value, "");
static_assert(IsConstConvertible<int, const int>::value, "");
static_assert(!IsConstConvertible<const int, int>::value, "");
static_assert(!IsConstConvertible<int, float>::value, "");
static_assert(!IsConstConvertible<int, const float>::value, "");
static_assert(!IsConstConvertible<int, const void>::value, "");
static_assert(!IsConstConvertible<const int, void>::value, "");
static_assert(!IsConstConvertible<int, void>::value, "");

static_assert(IsConstConvertibleOrVoid<int, int>::value, "");
static_assert(IsConstConvertibleOrVoid<int, const int>::value, "");
static_assert(IsConstConvertibleOrVoid<int, void>::value, "");
static_assert(IsConstConvertibleOrVoid<const int, void>::value, "");
static_assert(IsConstConvertibleOrVoid<int, const void>::value, "");
static_assert(!IsConstConvertibleOrVoid<const int, int>::value, "");
static_assert(!IsConstConvertibleOrVoid<int, float>::value, "");
static_assert(!IsConstConvertibleOrVoid<int, const float>::value, "");

static_assert(std::is_same<TypePackElement<0, int, float>, int>::value, "");
static_assert(std::is_same<TypePackElement<1, int, float>, float>::value, "");

template <std::size_t I, typename... Ts>
using NonBuiltinTypePackElement =
    typename std::tuple_element<I, std::tuple<Ts...>>::type;

static_assert(
    std::is_same<NonBuiltinTypePackElement<0, int, float>, int>::value, "");
static_assert(
    std::is_same<NonBuiltinTypePackElement<1, int, float>, float>::value, "");

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

static_assert(std::is_same<int, type_identity_t<int>>::value, "");

namespace has {
TENSORSTORE_INTERNAL_DEFINE_HAS_METHOD(Foo)
TENSORSTORE_INTERNAL_DEFINE_HAS_ADL_FUNCTION(Bar)
}  // namespace has

struct HasFooStruct {
  int* Foo(int);
  float* Foo(int, int);
};

struct MissingStruct {
  void Bar(int, int);
};

struct HasBarStruct {
  friend int* Bar(HasBarStruct) ABSL_ATTRIBUTE_UNUSED;
  friend float* Bar(HasBarStruct, int, int) ABSL_ATTRIBUTE_UNUSED;
};

static_assert(has::HasMethodFoo<int*, HasFooStruct, int>::value, "");
static_assert(has::HasMethodFoo<const int*, HasFooStruct, int>::value, "");
static_assert(has::HasMethodFoo<void, HasFooStruct, int>::value, "");
static_assert(has::HasMethodFoo<void, HasFooStruct, int, int>::value, "");
static_assert(has::HasMethodFoo<float*, HasFooStruct, int, int>::value, "");
static_assert(!has::HasMethodFoo<void, HasFooStruct, int, int, int>::value, "");
static_assert(!has::HasMethodFoo<void, HasFooStruct>::value, "");
static_assert(!has::HasMethodFoo<void, MissingStruct, int>::value, "");
static_assert(has::HasAdlFunctionBar<void, HasBarStruct>::value, "");
static_assert(!has::HasAdlFunctionBar<void, HasBarStruct, int>::value, "");
static_assert(has::HasAdlFunctionBar<float*, HasBarStruct, int, int>::value,
              "");
static_assert(
    has::HasAdlFunctionBar<const float*, HasBarStruct, int, int>::value, "");
static_assert(has::HasAdlFunctionBar<void, HasBarStruct, int, int>::value, "");
static_assert(!has::HasAdlFunctionBar<void, MissingStruct, int, int>::value,
              "");

namespace explict_conversion_tests {
using tensorstore::internal::IsOnlyExplicitlyConvertible;
using tensorstore::internal::IsPairExplicitlyConvertible;
using tensorstore::internal::IsPairImplicitlyConvertible;
using tensorstore::internal::IsPairOnlyExplicitlyConvertible;

struct X {
  X(int) {}
  explicit X(float*) {}
};

static_assert(IsOnlyExplicitlyConvertible<float*, X>::value, "");
static_assert(std::is_convertible<int, X>::value, "");
static_assert(std::is_constructible<X, int>::value, "");
static_assert(!IsOnlyExplicitlyConvertible<int, X>::value, "");

struct Y {
  Y(int*) {}
  explicit Y(double*) {}
};

static_assert(IsPairImplicitlyConvertible<int, int*, X, Y>::value, "");
static_assert(IsPairExplicitlyConvertible<int, int*, X, Y>::value, "");
static_assert(IsPairExplicitlyConvertible<int, double*, X, Y>::value, "");
static_assert(IsPairExplicitlyConvertible<float*, int*, X, Y>::value, "");
static_assert(IsPairExplicitlyConvertible<float*, double*, X, Y>::value, "");
static_assert(!IsPairImplicitlyConvertible<int, double*, X, Y>::value, "");
static_assert(!IsPairImplicitlyConvertible<float*, int*, X, Y>::value, "");
static_assert(!IsPairImplicitlyConvertible<float*, double*, X, Y>::value, "");
static_assert(IsPairOnlyExplicitlyConvertible<int, double*, X, Y>::value, "");
static_assert(IsPairOnlyExplicitlyConvertible<float*, int*, X, Y>::value, "");
static_assert(IsPairOnlyExplicitlyConvertible<float*, double*, X, Y>::value,
              "");

}  // namespace explict_conversion_tests

}  // namespace
