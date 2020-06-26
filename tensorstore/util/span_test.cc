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

/// Tests for span.h.

#include "tensorstore/util/span.h"

#include <array>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/attributes.h"
#include "absl/types/span.h"
#include "tensorstore/util/str_cat.h"

using tensorstore::span;
using tensorstore::StrCat;
using tensorstore::internal::ConstSpanType;
using tensorstore::internal::SpanType;
using tensorstore::internal_span::IsSpanImplicitlyConvertible;

static_assert(
    std::is_same_v<ConstSpanType<const std::vector<int>&>, span<const int>>);
static_assert(std::is_same_v<ConstSpanType<std::vector<int>>, span<const int>>);
static_assert(std::is_same_v<ConstSpanType<int (&)[3]>, span<const int, 3>>);
static_assert(std::is_same_v<SpanType<int (&)[3]>, span<int, 3>>);
static_assert(std::is_same_v<ConstSpanType<span<int, 3>>, span<const int, 3>>);
static_assert(
    std::is_same_v<ConstSpanType<std::array<int, 3>>, span<const int, 3>>);
static_assert(
    std::is_same_v<ConstSpanType<span<const int, 3>>, span<const int, 3>>);

namespace {

MATCHER_P(DataIs, data,
          StrCat("data() ", negation ? "is " : "isn't ",
                 testing::PrintToString(data))) {
  return arg.data() == data;
}

// GMock native testing::SizeIs relies on type::size_type, which span does
// not have.
MATCHER_P(SizeIs, size,
          StrCat("size() ", negation ? "is " : "isn't ",
                 testing::PrintToString(size))) {
  return arg.size() == size;
}

template <typename T>
auto SpanIs(T data, std::ptrdiff_t size)
    -> decltype(testing::AllOf(DataIs(data), SizeIs(size))) {
  return testing::AllOf(DataIs(data), SizeIs(size));
}

template <typename Container>
auto SpanIs(const Container& c) -> decltype(SpanIs(c.data(), c.size())) {
  return SpanIs(c.data(), c.size());
}

std::vector<int> MakeRamp(int len, int offset = 0) {
  std::vector<int> v(len);
  std::iota(v.begin(), v.end(), offset);
  return v;
}

namespace compile_time_convertibility_tests {
static_assert(IsSpanImplicitlyConvertible<int, 3, const int, 3>::value, "");
static_assert(span<int, 3>::size() == 3, "");
static_assert(std::is_convertible<span<int, 3>, span<const int, 3>>::value, "");
static_assert(!std::is_convertible<span<const int, 3>, span<int, 3>>::value,
              "");
static_assert(!std::is_convertible<span<int, 3>, span<float, 3>>::value, "");
static_assert(!std::is_convertible<span<int, 3>, span<int, 2>>::value, "");
static_assert(!std::is_convertible<span<int, 3>, span<float, 2>>::value, "");
static_assert(!std::is_convertible<span<int, 3>, span<const float, 3>>::value,
              "");
static_assert(std::is_convertible<std::array<int, 3>&, span<int, 3>>::value,
              "");
static_assert(std::is_convertible<std::array<int, 3>&, span<int>>::value, "");
static_assert(
    std::is_convertible<std::array<int, 3>, span<const int, 3>>::value, "");
static_assert(std::is_convertible<std::array<int, 3>, span<const int>>::value,
              "");
static_assert(
    std::is_convertible<const std::array<int, 3>, span<const int, 3>>::value,
    "");
static_assert(
    std::is_convertible<const std::array<int, 3>, span<const int>>::value, "");
static_assert(
    !std::is_convertible<const std::array<int, 3>, span<int, 3>>::value, "");
static_assert(
    !std::is_convertible<std::array<const int, 3>, span<int, 3>>::value, "");
static_assert(!std::is_convertible<std::array<float, 3>, span<int, 3>>::value,
              "");
static_assert(!std::is_convertible<std::array<float, 3>, span<int, 4>>::value,
              "");
}  // namespace compile_time_convertibility_tests

namespace deduction_guide_tests {
static_assert(
    std::is_same<decltype(span(std::declval<int*>(), 3)), span<int>>::value,
    "");
static_assert(
    std::is_same<decltype(span(std::declval<int*>(), std::declval<int*>())),
                 span<int>>::value,
    "");
static_assert(std::is_same<decltype(span(std::declval<std::vector<int>>())),
                           span<const int>>::value,
              "");
static_assert(std::is_same<decltype(span(std::declval<std::vector<int>&>())),
                           span<int>>::value,
              "");
static_assert(std::is_same<decltype(span(std::declval<std::string>())),
                           span<const char>>::value,
              "");
static_assert(std::is_same<decltype(span(std::declval<std::array<int, 3>&>())),
                           span<int, 3>>::value,
              "");
static_assert(std::is_same<decltype(span(std::declval<std::array<int, 3>>())),
                           span<const int, 3>>::value,
              "");
static_assert(
    std::is_same<decltype(span(std::declval<const std::array<int, 3>>())),
                 span<const int, 3>>::value,
    "");
static_assert(std::is_same<decltype(span(std::declval<const int (&)[3]>())),
                           span<const int, 3>>::value,
              "");
static_assert(std::is_same<decltype(span(std::declval<span<int, 4>>())),
                           span<int, 4>>::value,
              "");
}  // namespace deduction_guide_tests

namespace span_constructor_tests {

TEST(SpanTest, ConstructDefault) {
  span<int> s;
  EXPECT_THAT(s, SpanIs(nullptr, 0));
}

TEST(SpanTest, ConstructStaticPointerSize) {
  int arr[] = {1, 2, 3};
  span<int, 3> s(&arr[0], 3);
  EXPECT_THAT(s, SpanIs(arr, 3));
  EXPECT_EQ(arr + 3, s.end());
}

TEST(SpanTest, ConstructDynamicPointerSize) {
  int arr[] = {1, 2, 3};
  span<int> s(&arr[0], 3);
  EXPECT_THAT(s, SpanIs(arr, 3));
  EXPECT_EQ(arr + 3, s.end());
}

TEST(SpanTest, ConstructStaticPointerPointer) {
  int arr[] = {1, 2, 3};
  span<int, 3> s(&arr[0], &arr[0] + 3);
  EXPECT_THAT(s, SpanIs(arr, 3));
  EXPECT_EQ(arr + 3, s.end());
}

TEST(SpanTest, ConstructDynamicPointerPointer) {
  int arr[] = {1, 2, 3};
  span<int> s(&arr[0], &arr[0] + 3);
  EXPECT_THAT(s, SpanIs(arr, 3));
  EXPECT_EQ(arr + 3, s.end());
}

TEST(SpanTest, ConstructStaticStdArray) {
  std::array<int, 3> arr = {1, 2, 3};
  span<int, 3> s(arr);
  EXPECT_THAT(s, SpanIs(arr));
}

TEST(SpanTest, ConstructDynamicStdArray) {
  std::array<int, 3> arr = {1, 2, 3};
  span<int> s(arr);
  EXPECT_THAT(s, SpanIs(arr));
}

TEST(SpanTest, ConstructStaticContainer) {
  std::vector<int> vec = {1, 2, 3};
  span<int, 3> s(vec);
  EXPECT_THAT(s, SpanIs(vec));
}

TEST(SpanTest, ConstructDynamicContainer) {
  std::vector<int> vec = {1, 2, 3};
  span<int> s(vec);
  EXPECT_THAT(s, SpanIs(vec));
}

TEST(SpanTest, ConstructStaticContainerEmpty) {
  std::vector<int> empty;
  span<int, 0> s(empty);
  EXPECT_THAT(s, SpanIs(empty));
}

TEST(SpanTest, ConstructDynamicContainerEmpty) {
  std::vector<int> empty;
  span<int> s(empty);
  EXPECT_THAT(s, SpanIs(empty));
}

TEST(SpanTest, ConstructBracedList) {
  [](auto s) {
    EXPECT_EQ(1, s[0]);
    EXPECT_EQ(2, s[1]);
  }(span<const int, 2>({1, 2}));

  [](auto s2) {
    EXPECT_EQ(3, s2.size());
    EXPECT_EQ(1, s2[0]);
    EXPECT_EQ(2, s2[1]);
    EXPECT_EQ(3, s2[2]);
  }(span<const int>({1, 2, 3}));
}

TEST(SpanTest, ConstructPointerZero) {
  int arr[] = {1, 2, 3};
  span<int> s(&arr[0], 0);
  EXPECT_THAT(s, SpanIs(arr, 0));
}

// A struct supplying shallow data() const.
struct ContainerWithShallowConstData {
  std::vector<int> storage;
  int* data() const { return const_cast<int*>(storage.data()); }
  int size() const { return storage.size(); }
};

TEST(SpanTest, ShallowConstness) {
  const ContainerWithShallowConstData c{MakeRamp(20)};
  span<int> s(c);  // We should be able to do this even though data() is const.
  s[0] = -1;
  EXPECT_EQ(c.storage[0], -1);
}

TEST(CharSpan, StringCtor) {
  std::string empty = "";
  span<char> s_empty(empty);
  EXPECT_THAT(s_empty, SpanIs(empty));

  std::string abc = "abc";
  span<char> s_abc(abc);
  EXPECT_THAT(s_abc, SpanIs(abc));

  span<const char> s_const_abc = abc;
  EXPECT_THAT(s_const_abc, SpanIs(abc));
}

}  // namespace span_constructor_tests

TEST(SpanTest, SizeBytes) {
  int arr[] = {1, 2, 3};
  span<int> s(arr);
  EXPECT_EQ(sizeof(int) * 3, s.size_bytes());
}

TEST(SpanTest, Empty) {
  span<int, 0> s0;
  EXPECT_TRUE(s0.empty());

  int arr[] = {1, 2, 3};
  span<int, 3> s1(arr);
  EXPECT_FALSE(s1.empty());

  span<int> s2;
  EXPECT_TRUE(s2.empty());
  s2 = span<int>(&arr[0], 0);
  EXPECT_TRUE(s2.empty());
  s2 = s1;
  EXPECT_FALSE(s2.empty());
}

TEST(SpanTest, Iterators) {
  int arr[] = {1, 2, 3};
  span<int, 3> s1(arr);
  EXPECT_EQ(&arr[0], s1.begin());
  static_assert(std::is_same<int*, decltype(s1.begin())>::value, "");
  static_assert(std::is_same<int*, decltype(s1.end())>::value, "");
  EXPECT_EQ(&arr[0], s1.cbegin());
  static_assert(std::is_same<const int*, decltype(s1.cend())>::value, "");
  static_assert(std::is_same<const int*, decltype(s1.cbegin())>::value, "");
  EXPECT_EQ(arr + 3, s1.end());
  EXPECT_EQ(arr + 3, s1.cend());
}

TEST(SpanTest, Subscript) {
  int arr[] = {1, 2, 3};
  const span<int, 3> s(arr);
  const span<const int, 3> s2(arr);
  static_assert(std::is_same<int&, decltype(s[0])>::value, "");
  static_assert(std::is_same<const int&, decltype(s2[0])>::value, "");
  EXPECT_EQ(2, s[1]);
}

namespace first_tests {
TEST(SpanTest, FirstStaticStatic) {
  int arr[] = {1, 2, 3};
  span<int, 3> s(arr);
  auto s2 = s.first<2>();
  static_assert(std::is_same<span<int, 2>, decltype(s2)>::value, "");
  EXPECT_THAT(s2, SpanIs(arr, 2));
}

TEST(SpanTest, FirstStaticDynamic) {
  int arr[] = {1, 2, 3};
  span<int, 3> s(arr);
  auto s2 = s.first(2);
  static_assert(std::is_same<span<int>, decltype(s2)>::value, "");
  EXPECT_THAT(s2, SpanIs(arr, 2));
}

TEST(SpanTest, FirstDynamicStatic) {
  int arr[] = {1, 2, 3};
  span<int> s(arr);
  auto s2 = s.first<2>();
  static_assert(std::is_same<span<int, 2>, decltype(s2)>::value, "");
  EXPECT_THAT(s2, SpanIs(arr, 2));
}

TEST(SpanTest, FirstDynamicDynamic) {
  int arr[] = {1, 2, 3};
  span<int> s(arr);
  auto s2 = s.first(2);
  static_assert(std::is_same<span<int>, decltype(s2)>::value, "");
  EXPECT_THAT(s2, SpanIs(arr, 2));
}
}  // namespace first_tests
namespace last_tests {

TEST(SpanTest, LastStaticStatic) {
  int arr[] = {1, 2, 3};
  span<int, 3> s(arr);
  auto s2 = s.last<2>();
  static_assert(std::is_same<span<int, 2>, decltype(s2)>::value, "");
  EXPECT_THAT(s2, SpanIs(arr + 1, 2));
}

TEST(SpanTest, LastStaticDynamic) {
  int arr[] = {1, 2, 3};
  span<int, 3> s(arr);
  auto s2 = s.last(2);
  static_assert(std::is_same<span<int>, decltype(s2)>::value, "");
  EXPECT_THAT(s2, SpanIs(arr + 1, 2));
}

TEST(SpanTest, LastDynamicStatic) {
  int arr[] = {1, 2, 3};
  span<int> s(arr);
  auto s2 = s.last<2>();
  static_assert(std::is_same<span<int, 2>, decltype(s2)>::value, "");
  EXPECT_THAT(s2, SpanIs(arr + 1, 2));
}

TEST(SpanTest, LastDynamicDynamic) {
  int arr[] = {1, 2, 3};
  span<int> s(arr);
  auto s2 = s.last(2);
  static_assert(std::is_same<span<int>, decltype(s2)>::value, "");
  EXPECT_THAT(s2, SpanIs(arr + 1, 2));
}
}  // namespace last_tests

TEST(SpanTest, IteratorsAndReferences) {
  auto accept_pointer = [](int*) {};
  auto accept_reference = [](int&) {};
  auto accept_iterator = [](span<int>::iterator) {};
  auto accept_const_iterator = [](span<int>::const_iterator) {};
  auto accept_reverse_iterator = [](span<int>::reverse_iterator) {};
  auto accept_const_reverse_iterator = [](span<int>::const_reverse_iterator) {};

  int arr[] = {1, 2, 3};
  {
    span<int> s = arr;
    accept_pointer(s.data());
    accept_iterator(s.begin());
    accept_const_iterator(s.begin());
    accept_const_iterator(s.cbegin());
    accept_iterator(s.end());
    accept_const_iterator(s.end());
    accept_const_iterator(s.cend());
    accept_reverse_iterator(s.rbegin());
    accept_const_reverse_iterator(s.rbegin());
    accept_const_reverse_iterator(s.crbegin());
    accept_reverse_iterator(s.rend());
    accept_const_reverse_iterator(s.rend());
    accept_const_reverse_iterator(s.crend());

    accept_reference(s[0]);
    accept_reference(s.at(0));
    accept_reference(s.front());
    accept_reference(s.back());
  }

  {
    span<int, 3> s = arr;
    accept_pointer(s.data());
    accept_iterator(s.begin());
    accept_const_iterator(s.begin());
    accept_const_iterator(s.cbegin());
    accept_iterator(s.end());
    accept_const_iterator(s.end());
    accept_const_iterator(s.cend());
    accept_reverse_iterator(s.rbegin());
    accept_const_reverse_iterator(s.rbegin());
    accept_const_reverse_iterator(s.crbegin());
    accept_reverse_iterator(s.rend());
    accept_const_reverse_iterator(s.rend());
    accept_const_reverse_iterator(s.crend());

    accept_reference(s[0]);
    accept_reference(s.at(0));
    accept_reference(s.front());
    accept_reference(s.back());
  }
}

TEST(SpanTest, NoexceptTests) {
  int arr[] = {1, 2, 3};
  std::vector<int> vec;

  static_assert(noexcept(span<const int>()));
  static_assert(noexcept(span<const int>(arr)));
  static_assert(noexcept(span<const int>({1, 2, 3})));

  static_assert(noexcept(span<const int>(arr, 2)));

  span<int> s = arr;

  static_assert(noexcept(s.data()));
  static_assert(noexcept(s.size()));
  static_assert(noexcept(s.empty()));
  static_assert(noexcept(s[0]));

  static_assert(!noexcept(s.at(0)));
  static_assert(!noexcept(s.front()));
  static_assert(!noexcept(s.back()));
  static_assert(!noexcept(s.first(1)));
  static_assert(!noexcept(s.last(1)));
  static_assert(!noexcept(s.template first<1>()));
  static_assert(!noexcept(s.template last<1>()));

  static_assert(noexcept(s.begin()));
  static_assert(noexcept(s.cbegin()));
  static_assert(noexcept(s.end()));
  static_assert(noexcept(s.cend()));
  static_assert(noexcept(s.rbegin()));
  static_assert(noexcept(s.crbegin()));
  static_assert(noexcept(s.rend()));
  static_assert(noexcept(s.crend()));
}

// ConstexprTester exercises expressions in a constexpr context. Simply placing
// the expression in a constexpr function is not enough, as some compilers will
// simply compile the constexpr function as runtime code. Using template
// parameters forces compile-time execution.
template <int i>
struct ConstexprTester {};

#define TENSORSTORE_TEST_CONSTEXPR(expr)                \
  do {                                                  \
    ABSL_ATTRIBUTE_UNUSED ConstexprTester<(expr, 1)> t; \
  } while (0)

TEST(SpanTest, ConstexprTest) {
  static constexpr int a[] = {1, 2, 3};
  static constexpr int sized_arr[2] = {1, 2};

  TENSORSTORE_TEST_CONSTEXPR(span<const int>());
  TENSORSTORE_TEST_CONSTEXPR(span<const int>(a, 2));
  TENSORSTORE_TEST_CONSTEXPR(span<const int>(sized_arr));

  constexpr span<const int> span = a;
  TENSORSTORE_TEST_CONSTEXPR(span.data());
  TENSORSTORE_TEST_CONSTEXPR(span.size());
  TENSORSTORE_TEST_CONSTEXPR(span.empty());
  TENSORSTORE_TEST_CONSTEXPR(span.begin());
  TENSORSTORE_TEST_CONSTEXPR(span.cbegin());
  TENSORSTORE_TEST_CONSTEXPR(span.subspan(0, 0));
  TENSORSTORE_TEST_CONSTEXPR(span[0]);
}

namespace subspan_tests {

TEST(SpanTest, SubspanStaticStaticStatic) {
  int arr[] = {1, 2, 3};
  span<int, 3> s(arr);
  auto s2 = s.subspan<1, 1>();
  static_assert(std::is_same<span<int, 1>, decltype(s2)>::value, "");
  EXPECT_THAT(s2, SpanIs(arr + 1, 1));
}

TEST(SpanTest, SubspanStaticStaticDynamic) {
  int arr[] = {1, 2, 3};
  span<int, 3> s(arr);
  auto s2 = s.subspan<1>();
  static_assert(std::is_same<span<int, 2>, decltype(s2)>::value, "");
  EXPECT_THAT(s2, SpanIs(arr + 1, 2));
}

TEST(SpanTest, SubspanStaticDynamicDynamic) {
  int arr[] = {1, 2, 3};
  span<int, 3> s(arr);
  auto s2 = s.subspan(1, 1);
  static_assert(std::is_same<span<int>, decltype(s2)>::value, "");
  EXPECT_THAT(s2, SpanIs(arr + 1, 1));
}

TEST(SpanTest, SubspanStaticDynamic) {
  int arr[] = {1, 2, 3};
  span<int, 3> s(arr);
  auto s2 = s.subspan(1);
  static_assert(std::is_same<span<int>, decltype(s2)>::value, "");
  EXPECT_THAT(s2, SpanIs(arr + 1, 2));
}

TEST(SpanTest, SubspanDynamicStaticStatic) {
  int arr[] = {1, 2, 3};
  span<int> s(arr);
  auto s2 = s.subspan<1, 1>();
  static_assert(std::is_same<span<int, 1>, decltype(s2)>::value, "");
  EXPECT_THAT(s2, SpanIs(arr + 1, 1));
}

TEST(SpanTest, SubspanDynamicStaticDynamic) {
  int arr[] = {1, 2, 3};
  span<int> s(arr);
  auto s2 = s.subspan<1>();
  static_assert(std::is_same<span<int>, decltype(s2)>::value, "");
  EXPECT_THAT(s2, SpanIs(arr + 1, 2));
}

TEST(SpanTest, SubspanDynamicDynamicDynamic) {
  int arr[] = {1, 2, 3};
  span<int> s(arr);
  auto s2 = s.subspan(1, 1);
  static_assert(std::is_same<span<int>, decltype(s2)>::value, "");
  EXPECT_THAT(s2, SpanIs(arr + 1, 1));
}

TEST(SpanTest, SubspanDynamicDynamic) {
  int arr[] = {1, 2, 3};
  span<int> s(arr);
  auto s2 = s.subspan(1);
  static_assert(std::is_same<span<int>, decltype(s2)>::value, "");
  EXPECT_THAT(s2, SpanIs(arr + 1, 2));
}
}  // namespace subspan_tests

namespace deduction_tests {

TEST(DeduceSpanTest, PointerSize) {
  int arr[] = {1, 2, 3};
  auto s = span(&arr[0], 3);
  EXPECT_THAT(s, SpanIs(arr, 3));
}

TEST(DeduceSpanTest, PointerPointer) {
  int arr[] = {1, 2, 3};
  auto s = span(&arr[0], &arr[2]);
  EXPECT_THAT(s, SpanIs(arr, 2));
}

TEST(DeduceSpanTest, PointerZero) {
  int arr[] = {1, 2, 3};
  auto s = span(&arr[0], 0);
  EXPECT_THAT(s, SpanIs(arr, 0));
}

TEST(DeduceSpanTest, Array) {
  int arr[] = {1, 2, 3};
  auto s = span(arr);
  EXPECT_THAT(s, SpanIs(arr, 3));
}

TEST(DeduceSpanTest, StdArray) {
  std::array<int, 3> arr = {{1, 2, 3}};
  auto s = span(arr);
  EXPECT_THAT(s, SpanIs(arr));
}

TEST(DeduceSpanTest, ConstStdArray) {
  const std::array<int, 3> arr = {{1, 2, 3}};
  auto s = span(arr);
  EXPECT_THAT(s, SpanIs(arr));
}

TEST(DeduceSpanTest, Vector) {
  std::vector<int> arr = {1, 2, 3};
  auto s = span(arr);
  EXPECT_THAT(s, SpanIs(arr));
}

TEST(DeduceSpanTest, ConstVector) {
  const std::vector<int> arr = {1, 2, 3};
  auto s = span(arr);
  EXPECT_THAT(s, SpanIs(arr));
}

TEST(DeduceSpanTest, Span) {
  int arr[] = {1, 2, 3};
  span<int> s1(arr);
  auto s2 = span(s1);
  EXPECT_THAT(s2, SpanIs(arr, 3));
}
}  // namespace deduction_tests

namespace compare_tests {
TEST(SpanTest, RangesEqual) {
  using tensorstore::internal::RangesEqual;

  EXPECT_TRUE(RangesEqual(span<const int>(), span<const int>()));
  EXPECT_TRUE(RangesEqual(span<const int>(), span<int>()));

  EXPECT_TRUE(RangesEqual(span({1, 2, 3}), span({1, 2, 3})));

  EXPECT_FALSE(RangesEqual(span({1, 2}), span({1, 2, 3})));
  EXPECT_FALSE(RangesEqual(span({1, 2, 3}), span({1, 2, 4})));

  int arr1[] = {1, 2, 3};
  const int arr2[] = {1, 2, 3, 4};
  EXPECT_TRUE(RangesEqual(span(arr1), span(arr1)));
  EXPECT_TRUE(RangesEqual(span(arr1), span(arr2, 3)));
  EXPECT_FALSE(RangesEqual(span(arr1), span(arr2)));
}

}  // namespace compare_tests

namespace tuple_tests {
static_assert(std::tuple_size<span<int, 3>>::value == 3, "");
static_assert(
    std::is_same<int,
                 typename std::tuple_element<1, span<int, 3>>::type>::value,
    "");

TEST(SpanTest, Get) {
  int arr[] = {1, 2, 3};
  span<int, 3> x(arr);
  EXPECT_EQ(&arr[1], &tensorstore::get<1>(x));
}

}  // namespace tuple_tests

namespace static_or_dynamic_extent_tests {
TEST(GetStaticOrDynamicExtentTest, Static) {
  int arr[] = {1, 2, 3};
  auto extent = GetStaticOrDynamicExtent(span(arr));
  static_assert(std::is_same<decltype(extent),
                             std::integral_constant<std::ptrdiff_t, 3>>::value,
                "");
  EXPECT_EQ(3, extent);
}
TEST(GetStaticOrDynamicExtentTest, Dynamic) {
  int arr[] = {1, 2, 3};
  auto extent = GetStaticOrDynamicExtent(span<const int>(arr));
  static_assert(std::is_same<decltype(extent), std::ptrdiff_t>::value, "");
  EXPECT_EQ(3, extent);
}
}  // namespace static_or_dynamic_extent_tests

}  // namespace
