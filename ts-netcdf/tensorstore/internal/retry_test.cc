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

#include "tensorstore/internal/retry.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/time/time.h"

namespace {

using ::tensorstore::internal::BackoffForAttempt;

TEST(RetryTest, BackoffForAttempt) {
  // first attempt ==
  EXPECT_EQ(absl::Microseconds(1),
            BackoffForAttempt(0, absl::Microseconds(1), absl::Microseconds(100),
                              /*jitter=*/absl::ZeroDuration()));

  EXPECT_EQ(absl::Microseconds(2),
            BackoffForAttempt(1, absl::Microseconds(1), absl::Microseconds(100),
                              /*jitter=*/absl::ZeroDuration()));

  EXPECT_EQ(absl::Microseconds(4),
            BackoffForAttempt(2, absl::Microseconds(1), absl::Microseconds(100),
                              /*jitter=*/absl::ZeroDuration()));

  // Some long attempt later.
  EXPECT_EQ(
      absl::Microseconds(100),
      BackoffForAttempt(66, absl::Microseconds(1), absl::Microseconds(100),
                        /*jitter=*/absl::ZeroDuration()));

  // 4ms +0-100 jitter
  EXPECT_THAT(absl::ToInt64Microseconds(BackoffForAttempt(
                  2, absl::Microseconds(1), absl::Microseconds(200),
                  /*jitter=*/absl::Microseconds(100))),
              ::testing::AllOf(::testing::Ge(2), testing::Le(104)));
}

}  // namespace
