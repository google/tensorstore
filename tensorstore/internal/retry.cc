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

#include <algorithm>
#include <cassert>
#include <functional>
#include <string>
#include <system_error>  // NOLINT

#include "absl/log/absl_log.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal {

bool DefaultIsRetriable(const absl::Status& status) {
  return (status.code() == absl::StatusCode::kUnknown ||
          status.code() == absl::StatusCode::kDeadlineExceeded ||
          status.code() == absl::StatusCode::kUnavailable);
}

absl::Duration BackoffForAttempt(int attempt, absl::Duration initial_delay,
                                 absl::Duration max_delay,
                                 absl::Duration jitter) {
  assert(initial_delay > absl::ZeroDuration());
  assert(max_delay >= initial_delay);
  assert(attempt >= 0);

  int64_t multiple = int64_t{1} << (attempt > 63 ? 63 : attempt);
  auto delay = initial_delay * multiple;
  if (jitter >= absl::Microseconds(1)) {
    delay += absl::Microseconds(absl::Uniform(
        absl::InsecureBitGen{}, 0, absl::ToInt64Microseconds(jitter)));
  }
  if (delay > max_delay) delay = max_delay;
  return delay;
}

absl::Status RetryWithBackoff(
    std::function<absl::Status()> function, int max_retries,
    absl::Duration initial_delay, absl::Duration max_delay,
    absl::Duration jitter,
    std::function<bool(const absl::Status&)> is_retriable) {
  absl::Status status;
  for (int attempt = 0; attempt < max_retries; attempt++) {
    status = function();
    if (status.ok() || !is_retriable(status)) {
      return status;
    }

    // Compute backoff.
    auto delay = BackoffForAttempt(attempt, initial_delay, max_delay, jitter);

    // NOTE: Figure out a way to enable better logging when we want it.
    if (false) {
      ABSL_LOG(INFO)
          << "The operation failed and will be automatically retried in "
          << absl::ToDoubleSeconds(delay) << " seconds (attempt " << attempt + 1
          << " out of " << max_retries << "), caused by: " << status;
    }

    absl::SleepFor(delay);
  }

  // Return AbortedError, so that it doesn't get retried again somewhere
  // at a higher level.
  return absl::AbortedError(tensorstore::StrCat(
      "All ", max_retries, " retry attempts failed: ", status));
}

}  // namespace internal
}  // namespace tensorstore
