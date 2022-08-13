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
  InlineExecutor executor;
  bool invoked = false;
  struct Func {
    void operator()(bool* x) const = delete;
    void operator()(bool* x) { *x = true; }
  };
  auto with_executor = WithExecutor(executor, Func{});
  with_executor(&invoked);
  EXPECT_TRUE(invoked);
}

TEST(WithExecutorTest, Const) {
  InlineExecutor executor;
  bool invoked = false;
  struct Func {
    void operator()(bool* x) const { *x = true; }
    void operator()(bool*) = delete;
  };
  const auto with_executor = WithExecutor(executor, Func{});
  with_executor(&invoked);
  EXPECT_TRUE(invoked);
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
  with_executor();
  EXPECT_EQ(3, value);
}

}  // namespace
