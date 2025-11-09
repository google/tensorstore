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

#include "tensorstore/internal/testing/dynamic.h"

#include <functional>
#include <string>
#include <utility>

#include <gtest/gtest.h>
#include "tensorstore/internal/source_location.h"

namespace tensorstore {
namespace internal_testing {
namespace {

struct Fixture : public ::testing::Test {};

class Test : public Fixture {
 public:
  Test(const std::function<void()>& test_func) : test_func_(test_func) {}
  void TestBody() override { test_func_(); }

 private:
  std::function<void()> test_func_;
};

}  // namespace

void RegisterGoogleTestCaseDynamically(std::string test_suite_name,
                                       std::string test_name,
                                       std::function<void()> test_func,
                                       SourceLocation loc) {
  ::testing::RegisterTest(test_suite_name.c_str(), test_name.c_str(),
                          /*type_param=*/nullptr,
                          /*value_param=*/nullptr, loc.file_name(), loc.line(),
                          [test_func = std::move(test_func)]() -> Fixture* {
                            return new Test(test_func);
                          });
}

}  // namespace internal_testing
}  // namespace tensorstore
