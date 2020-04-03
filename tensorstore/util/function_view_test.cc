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

#include "tensorstore/util/function_view.h"

#include <functional>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>
#include "tensorstore/util/assert_macros.h"

namespace {
using tensorstore::FunctionView;

template <typename Signature>
struct Callable;

template <typename R, typename... Arg>
struct Callable<R(Arg...)> {
  R operator()(Arg...) { TENSORSTORE_UNREACHABLE; }
};

template <typename R, typename... Arg>
struct Callable<R(Arg...) const> {
  R operator()(Arg...) const { TENSORSTORE_UNREACHABLE; }
};

static_assert(std::is_constructible<FunctionView<int(float, int*)>,
                                    Callable<int(float, int*)>>::value,
              "");
static_assert(std::is_constructible<FunctionView<void(float, int*)>,
                                    Callable<int(float, int*)>>::value,
              "");
static_assert(std::is_constructible<FunctionView<int(double, int*)>,
                                    Callable<int(float, const int*)>>::value,
              "");
static_assert(!std::is_constructible<FunctionView<int(float, int*)>,
                                     Callable<void(float, int*)>>::value,
              "");
static_assert(!std::is_constructible<FunctionView<int(float, int*)>,
                                     Callable<void(float*, int*)>>::value,
              "");

TEST(FunctionViewTest, DefaultConstruct) {
  FunctionView<int()> f;
  EXPECT_FALSE(f);
}

TEST(FunctionViewTest, NullptrConstruct) {
  FunctionView<int()> f = nullptr;
  EXPECT_FALSE(f);
}

TEST(FunctionViewTest, ObjConstruct) {
  auto const func = [&](int x, int y) { return x + y; };
  FunctionView<int(int, int)> f = func;
  EXPECT_TRUE(f);
  EXPECT_EQ(5, f(2, 3));
}

TEST(FunctionViewTest, StdFunctionConstruct) {
  auto const func = [&](int x, int y) { return x + y; };
  std::function<int(int, int)> func1 = func;
  FunctionView<int(int, int)> f = func1;
  EXPECT_TRUE(f);
  EXPECT_EQ(5, f(2, 3));
}

void ForEach(const std::vector<int>& v, FunctionView<void(int)> callback) {
  for (int x : v) callback(x);
}

TEST(FunctionViewTest, Example) {
  int sum = 0;
  ForEach({1, 2, 3}, [&](int x) { sum += x; });
  EXPECT_EQ(6, sum);
}

}  // namespace
