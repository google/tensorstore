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

#include "tensorstore/util/status_testutil.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"

namespace {

using ::tensorstore::Future;
using ::tensorstore::IsOk;
using ::tensorstore::IsOkAndHolds;
using ::tensorstore::MatchesStatus;
using ::tensorstore::Result;
using ::tensorstore::StatusIs;

// Returns the reason why x matches, or doesn't match, m.
template <typename MatcherType, typename Value>
std::string Explain(const MatcherType& m, const Value& x) {
  testing::StringMatchResultListener listener;
  ExplainMatchResult(m, x, &listener);
  return listener.str();
}

TEST(StatusTestutilTest, IsOk) {
  EXPECT_THAT([]() -> Future<void> { return absl::OkStatus(); }(), IsOk());
  EXPECT_THAT([]() -> Result<void> { return absl::OkStatus(); }(), IsOk());

  EXPECT_THAT(absl::OkStatus(), IsOk());
  EXPECT_THAT(Result<int>{1}, IsOk());
  EXPECT_THAT(Future<int>{2}, IsOk());
  EXPECT_THAT(absl::InternalError(""), ::testing::Not(IsOk()));

  // IsOk() doesn't explain; EXPECT logs the value.
  EXPECT_THAT(Explain(IsOk(), absl::InternalError("")), testing::IsEmpty());
  EXPECT_THAT(Explain(IsOk(), absl::OkStatus()), testing::IsEmpty());

  // Our macros
  TENSORSTORE_EXPECT_OK(absl::OkStatus());
  TENSORSTORE_ASSERT_OK(absl::OkStatus());
  TENSORSTORE_EXPECT_OK([]() -> Future<void> { return absl::OkStatus(); }());
  TENSORSTORE_ASSERT_OK([]() -> Result<void> { return absl::OkStatus(); }());
}

TEST(StatusTestutilTest, Optional) {
  // For Result<T>, ::testing::Optional is very similar to IsOkAndHolds.
  EXPECT_THAT(Result<int>{1}, ::testing::Optional(1));
  EXPECT_THAT(Result<int>{absl::InternalError("")},
              ::testing::Not(::testing::Optional(1)));
  EXPECT_THAT(Result<int>{1}, ::testing::Optional(::testing::_));

  // Negations
  EXPECT_THAT(Result<int>{2}, ::testing::Optional(::testing::Not(1)));
  EXPECT_THAT(Result<int>{absl::InternalError("")},
              ::testing::Not(::testing::Optional(1)));

  EXPECT_THAT(
      Explain(::testing::Optional(1), Result<int>(absl::InternalError(""))),
      testing::HasSubstr("which is not engaged"));
  EXPECT_THAT(Explain(::testing::Optional(1), Result<int>(2)),
              testing::HasSubstr("whose value 2 doesn't match"));

  // Consider adding a death test:
  // EXPECT_THAT(Result<int>(absl::InternalError("")), ::testing::Optional(1));
}

TEST(StatusTestutilTest, IsOkAndHolds) {
  EXPECT_THAT(Result<int>{1}, IsOkAndHolds(1));
  EXPECT_THAT(Future<int>{2}, IsOkAndHolds(2));
  EXPECT_THAT(Result<int>{1}, IsOkAndHolds(::testing::_));

  // Negations
  EXPECT_THAT(Result<int>{2}, IsOkAndHolds(::testing::Not(1)));
  EXPECT_THAT(Result<int>{absl::InternalError("")},
              ::testing::Not(IsOkAndHolds(1)));

  int result;
  TENSORSTORE_ASSERT_OK_AND_ASSIGN(result, []() -> Result<int> { return 2; }());
  EXPECT_EQ(2, result);

  EXPECT_THAT(Explain(IsOkAndHolds(1), Result<int>(absl::InternalError(""))),
              testing::HasSubstr("whose status code is INTERNAL"));

  EXPECT_THAT(Explain(IsOkAndHolds(1), Result<int>(2)),
              testing::HasSubstr("whose value 2 doesn't match"));

  // Consider adding a death test:
  // EXPECT_THAT(Result<int>(absl::InternalError("")),
  //             IsOkAndHolds(1));
}

TEST(StatusTestutilTest, StatusIs) {
  EXPECT_THAT(Result<void>{absl::InternalError("")},
              StatusIs(absl::StatusCode::kInternal));
  EXPECT_THAT(Future<void>{absl::InternalError("")},
              StatusIs(absl::StatusCode::kInternal));
  EXPECT_THAT(absl::InternalError(""), StatusIs(absl::StatusCode::kInternal));

  EXPECT_THAT(absl::OkStatus(),
              ::testing::Not(StatusIs(absl::StatusCode::kInternal)));
  EXPECT_THAT(absl::OkStatus(), StatusIs(absl::StatusCode::kOk));

  EXPECT_THAT(Explain(StatusIs(absl::StatusCode::kOk), absl::InternalError("")),
              testing::HasSubstr("whose status code INTERNAL doesn't match"));

  // Consider adding a death test:
  // EXPECT_THAT(Result<int>(absl::InternalError("")),
  //             StatusIs(absl::StatusCode::kInternal,
  //                                     ::testing::HasSubstr("foo")));
}

TEST(StatusTestutilTest, StatusIs_WithMessage) {
  EXPECT_THAT(Result<void>{absl::InternalError("strongbad")},
              StatusIs(::testing::_, ::testing::HasSubstr("bad")));
  EXPECT_THAT(Future<void>{absl::InternalError("strongbad")},
              StatusIs(::testing::_, ::testing::HasSubstr("bad")));

  EXPECT_THAT(absl::InternalError("strongbad"),
              StatusIs(::testing::_, ::testing::HasSubstr("bad")));
  EXPECT_THAT(
      absl::InternalError("strongbad"),
      StatusIs(::testing::_, ::testing::Not(::testing::HasSubstr("good"))));

  EXPECT_THAT(absl::Status{absl::InternalError("strongbad")},
              StatusIs(::testing::Not(absl::StatusCode::kAborted),
                       ::testing::Not(::testing::HasSubstr("good"))));
}

TEST(StatusTestutilTest, MatchesStatus) {
  EXPECT_THAT(Result<void>{absl::InternalError("")},
              StatusIs(absl::StatusCode::kInternal));
  EXPECT_THAT(Future<void>{absl::InternalError("")},
              StatusIs(absl::StatusCode::kInternal));

  EXPECT_THAT(absl::InternalError(""), StatusIs(absl::StatusCode::kInternal));
  EXPECT_THAT(absl::OkStatus(), StatusIs(absl::StatusCode::kOk));
}

TEST(StatusTestutilTest, MatchesStatus_Pattern) {
  EXPECT_THAT(Result<void>{absl::InternalError("a")},
              MatchesStatus(absl::StatusCode::kInternal, "a"));
  EXPECT_THAT(Future<void>{absl::InternalError("a")},
              MatchesStatus(absl::StatusCode::kInternal, "a"));

  EXPECT_THAT(absl::InternalError("a"),
              MatchesStatus(absl::StatusCode::kInternal, "a"));
  EXPECT_THAT(absl::InternalError("a"),
              ::testing::Not(MatchesStatus(absl::StatusCode::kInternal, "b")));
  EXPECT_THAT(absl::InternalError("a"),
              ::testing::Not(MatchesStatus(absl::StatusCode::kCancelled, "a")));
}

}  // namespace
