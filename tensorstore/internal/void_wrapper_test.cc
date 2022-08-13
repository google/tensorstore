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

#include "tensorstore/internal/void_wrapper.h"

#include <type_traits>

#include <gtest/gtest.h>
#include "tensorstore/internal/type_traits.h"

namespace {

using ::tensorstore::internal::Void;

// Tests for WrappedType.
static_assert(std::is_same_v<Void, Void::WrappedType<void>>);
static_assert(std::is_same_v<int, Void::WrappedType<int>>);

// Tests for UnwrappedType.
static_assert(std::is_same_v<void, Void::UnwrappedType<Void>>);
static_assert(std::is_same_v<int, Void::UnwrappedType<int>>);

TEST(VoidWrapperTest, BoolConversion) {
  // Void converts to `true`.
  EXPECT_EQ(true, static_cast<bool>(Void{}));
}

TEST(VoidWrapperTest, Unwrap) {
  EXPECT_EQ(3, Void::Unwrap(3));
  Void::Unwrap(Void{});
}

TEST(VoidWrapperTest, CallAndWrap) {
  int value;
  const auto void_func = [&](int arg) -> void {
    value = arg;
    return;
  };
  const auto int_func = [&](int arg) {
    value = arg;
    return 3;
  };

  auto result = Void::CallAndWrap(void_func, 4);
  static_assert(std::is_same_v<decltype(result), Void>);
  EXPECT_EQ(4, value);

  EXPECT_EQ(3, Void::CallAndWrap(int_func, 5));
  EXPECT_EQ(5, value);
}

///! [Repeat example]

/// Calls `func(args...)` up to `n` times, stopping as soon as `func` returns a
/// non-void value that equals `false` when converted to `bool`.  Returns the
/// last value returned by `func`, or value-initialized return value if `n < 1`.
template <typename Func, typename... Args,
          typename ResultType = std::invoke_result_t<Func, Args...>>
ResultType Repeat(int n, Func func, Args... args) {
  Void::WrappedType<ResultType> result = {};
  for (int i = 0; i < n; ++i) {
    result = Void::CallAndWrap(func, args...);
    if (!result) break;
  }
  return Void::Unwrap(result);
}
///! [Repeat example]

TEST(RepeatTest, VoidReturn) {
  int num = 0;
  Repeat(
      3, [&](int k) { num += k; }, 2);
  EXPECT_EQ(6, num);
}

TEST(RepeatTest, NonVoidReturn) {
  {
    int num = 0;
    EXPECT_EQ(true, Repeat(
                        3,
                        [&](int k) {
                          num += k;
                          return true;
                        },
                        2));
    EXPECT_EQ(6, num);
  }
  {
    int num = 0;
    EXPECT_EQ(false, Repeat(
                         3,
                         [&](int k) {
                           num += k;
                           return num < 4;
                         },
                         2));
    EXPECT_EQ(4, num);
  }
}

}  // namespace
