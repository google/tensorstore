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

#include "tensorstore/util/result_util.h"

#include <functional>
#include <string>
#include <type_traits>

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace {

using tensorstore::ChainResult;
using tensorstore::FlatMapResultType;
using tensorstore::FlatResult;
using tensorstore::Result;
using tensorstore::Status;
using tensorstore::UnwrapQualifiedResultType;
using tensorstore::UnwrapResultType;

/// FIXME: Is FlatMapResultType pulling it's weight?

static_assert(std::is_same<UnwrapResultType<int>, int>::value);
static_assert(std::is_same<UnwrapResultType<Result<int>>, int>::value);
static_assert(std::is_same<UnwrapResultType<Status>, void>::value);
static_assert(std::is_same<UnwrapQualifiedResultType<Status>, void>::value);
static_assert(std::is_same<UnwrapQualifiedResultType<Result<int>>, int>::value);
static_assert(
    std::is_same<UnwrapQualifiedResultType<Result<int>&>, int&>::value);
static_assert(std::is_same<UnwrapQualifiedResultType<const Result<int>&>,
                           const int&>::value);

static_assert(
    std::is_same<UnwrapQualifiedResultType<Result<int>&&>, int&&>::value);

/// FIXME: Typically a meta-function like FlatResult would be named MakeResult<>
/// or similar.

static_assert(std::is_same<FlatResult<Result<int>>, Result<int>>::value);

static_assert(std::is_same<FlatResult<int>, Result<int>>::value);

TEST(ChainResultTest, Example) {
  auto func1 = [](int x) -> float { return 1.0f + x; };
  auto func2 = [](float x) -> Result<std::string> {
    return absl::StrCat("fn.", x);
  };
  auto func3 = [](absl::string_view x) -> bool { return x.length() > 4; };

  Result<bool> y1 = ChainResult(Result<int>(3), func1, func2, func3);
  Result<bool> y2 = ChainResult(3, func1, func2, func3);

  EXPECT_TRUE(y1.has_value());
  EXPECT_TRUE(y2.has_value());

  EXPECT_EQ(y1.value(), y2.value());
}

TEST(ChainResultTest, Basic) {
  EXPECT_EQ(Result<int>(2), ChainResult(2));
  EXPECT_EQ(Result<int>(2), ChainResult(Result<int>(2)));

  EXPECT_EQ(Result<int>(3),
            ChainResult(Result<int>(2), [](int x) { return x + 1; }));

  EXPECT_EQ(Result<float>(1.5), ChainResult(
                                    Result<int>(2), [](int x) { return x + 1; },
                                    [](int x) { return x / 2.0f; }));

  EXPECT_EQ(Result<int>(absl::UnknownError("A")),
            ChainResult(Result<int>(absl::UnknownError("A")),
                        [](int x) { return x + 1; }));
}

TEST(MapResultTest, Basic) {
  tensorstore::Status status;

  EXPECT_EQ(Result<int>(absl::UnknownError("A")),
            tensorstore::MapResult(std::plus<int>(),
                                   Result<int>(absl::UnknownError("A")),
                                   Result<int>(absl::UnknownError("B"))));
  EXPECT_EQ(Result<int>(absl::UnknownError("B")),
            tensorstore::MapResult(std::plus<int>(), 1,
                                   Result<int>(absl::UnknownError("B"))));
  EXPECT_EQ(Result<int>(3), tensorstore::MapResult(std::plus<int>(), 1, 2));
  EXPECT_EQ(
      Result<int>(absl::UnknownError("C")),
      tensorstore::MapResult(
          [](int a, int b) { return Result<int>(absl::UnknownError("C")); }, 1,
          2));
}

}  // namespace
