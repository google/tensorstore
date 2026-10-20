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

#include "tensorstore/util/executor.h"

#include <functional>
#include <memory>
#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

namespace {
using ::tensorstore::Executor;
using ::tensorstore::InlineExecutor;
using ::tensorstore::WithExecutor;

TEST(InlineExecutorTest, Basic) {
  Executor executor = InlineExecutor{};
  bool invoked = false;
  executor([&] { invoked = true; });
  EXPECT_TRUE(invoked);
}

TEST(WithExecutorTest, NonConst) {
  struct Func {
    void operator()(bool* x) const = delete;
    void operator()(bool* x) { *x = true; }
  };
  Executor executor = InlineExecutor{};
  bool invoked = false;
  auto with_executor = WithExecutor(executor, Func{});
  with_executor(&invoked);
  EXPECT_TRUE(invoked);
}

TEST(WithExecutorTest, Const) {
  struct Func {
    void operator()(bool* x) const { *x = true; }
    void operator()(bool*) = delete;
  };
  Executor executor = InlineExecutor{};
  bool invoked = false;
  const auto with_executor = WithExecutor(executor, Func{});
  with_executor(&invoked);
  EXPECT_TRUE(invoked);
}

TEST(WithExecutorTest, ConstCallUsesConstLvalue) {
  struct Func {
    void operator()(int* out) const& { *out = 1; }
    void operator()(int* out) && { *out = 2; }
  };
  Executor executor = InlineExecutor{};
  const auto f = WithExecutor(executor, Func{});
  int v = 0;
  f(&v);
  EXPECT_EQ(v, 1);
}

TEST(WithExecutorTest, LvalueCallTwice) {
  Executor executor = InlineExecutor{};
  auto count = std::make_shared<int>(0);
  auto f = WithExecutor(executor, [count] { ++*count; });
  f();
  f();
  EXPECT_EQ(*count, 2);
}

TEST(ExecutorTest, MoveOnly) {
  Executor executor = InlineExecutor{};
  int value = 0;
  executor(std::bind([&](const std::unique_ptr<int>& ptr) { value = *ptr; },
                     std::make_unique<int>(3)));
  EXPECT_EQ(3, value);
}

TEST(WithExecutorTest, MoveOnly) {
  Executor executor = InlineExecutor{};
  int value = 0;
  auto with_executor = WithExecutor(
      executor,
      std::bind([&](const std::unique_ptr<int>& ptr) { value = *ptr; },
                std::make_unique<int>(3)));
  static_assert(!std::is_invocable_v<decltype(with_executor)&>);
  static_assert(!std::is_invocable_v<const decltype(with_executor)&>);
  std::move(with_executor)();
  EXPECT_EQ(3, value);
}

TEST(WithExecutorTest, MoveOnlyArgument) {
  Executor executor = InlineExecutor{};
  int value = 0;
  auto f = WithExecutor(executor, [&](std::unique_ptr<int> p) { value = *p; });
  f(std::make_unique<int>(3));
  EXPECT_EQ(value, 3);
}

TEST(WithExecutorTest, RvalueOnly) {
  struct Func {
    std::unique_ptr<int> ptr;
    void operator()(int* out) && { *out = *ptr; }
  };
  Executor executor = InlineExecutor{};
  int value = 0;
  auto with_executor = WithExecutor(executor, Func{std::make_unique<int>(3)});
  static_assert(!std::is_invocable_v<decltype(with_executor)&, int*>);
  std::move(with_executor)(&value);
  EXPECT_EQ(3, value);
}

}  // namespace
