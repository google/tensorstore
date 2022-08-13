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

#include <deque>
#include <functional>
#include <string>
#include <string_view>
#include <utility>

#include <gtest/gtest.h>
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorstore/util/status.h"

namespace {

using ::tensorstore::internal::RetryWithBackoff;

TEST(RetryTest, ImmediateSuccess) {
  std::deque<absl::Status> results({absl::OkStatus()});

  std::function<absl::Status()> f = [&results]() {
    auto result = results[0];
    results.erase(results.begin());
    return result;
  };

  auto status =
      RetryWithBackoff(f, 1, absl::Microseconds(1), absl::Microseconds(1),
                       /*jitter=*/absl::ZeroDuration());
  EXPECT_TRUE(status.ok());
  EXPECT_TRUE(results.empty());
}

TEST(RetryTest, NeverSuccess) {
  int calls = 0;
  std::function<absl::Status()> f = [&calls]() {
    calls++;
    return absl::UnavailableError("Not available.");
  };

  auto status =
      RetryWithBackoff(f, 100, absl::Microseconds(1), absl::Microseconds(1),
                       /*jitter=*/absl::ZeroDuration());
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(100, calls);
}

TEST(RetryTest, EventualSuccess_NoSleep) {
  std::deque<absl::Status> results({absl::UnavailableError("Failed."),
                                    absl::UnavailableError("Failed again."),
                                    absl::OkStatus()});

  std::function<absl::Status()> f = [&results]() {
    auto result = std::move(results[0]);
    results.erase(results.begin());
    return result;
  };

  auto status =
      RetryWithBackoff(f, 10, absl::Microseconds(1), absl::Microseconds(1),
                       /*jitter=*/absl::ZeroDuration());
  EXPECT_TRUE(status.ok());
  EXPECT_TRUE(results.empty());
}

TEST(RetryTest, EventualSuccess_Sleep) {
  std::deque<absl::Status> results({absl::UnavailableError("Failed."),
                                    absl::UnavailableError("Failed again."),
                                    absl::OkStatus()});

  std::function<absl::Status()> f = [&results]() {
    auto result = std::move(results[0]);
    results.erase(results.begin());
    return result;
  };
  auto before = absl::Now();
  auto status =
      RetryWithBackoff(f, 10, absl::Microseconds(10), absl::Microseconds(100),
                       /*jitter=*/absl::Microseconds(1));
  auto after = absl::Now();

  EXPECT_TRUE(status.ok());
  EXPECT_TRUE(results.empty());

  // We should have waited several us...
  EXPECT_LT(absl::Microseconds(10), after - before);
}

}  // namespace
