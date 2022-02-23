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
#include <optional>
#include <string>
#include <system_error>  // NOLINT

#include "absl/random/random.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorstore/internal/logging.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal {

bool DefaultIsRetriable(const absl::Status& status) {
  return (status.code() == absl::StatusCode::kUnknown ||
          status.code() == absl::StatusCode::kDeadlineExceeded ||
          status.code() == absl::StatusCode::kUnavailable);
}

absl::Status RetryWithBackoff(
    std::function<absl::Status()> function, int max_retries,
    absl::Duration initial_delay, absl::Duration max_delay,
    absl::Duration jitter,
    std::function<bool(const absl::Status&)> is_retriable) {
  assert(initial_delay > absl::ZeroDuration());
  assert(max_delay >= initial_delay);
  assert(max_retries >= 0);

  std::optional<absl::BitGen> rng;
  absl::Status status;

  int64_t multiple = 1;
  for (int retries = 0; retries < max_retries; retries++) {
    status = function();
    if (status.ok() || !is_retriable(status)) {
      return status;
    }

    // Compute backoff.
    auto delay = initial_delay * multiple;
    multiple <<= 1;
    if (jitter >= absl::Microseconds(1)) {
      if (!rng) rng.emplace();
      delay += absl::Microseconds(
          absl::Uniform(*rng, 0, absl::ToInt64Microseconds(jitter)));
    }
    if (delay > max_delay) delay = max_delay;

    // NOTE: Figure out a way to enable better logging when we want it.
    if (false) {
      TENSORSTORE_LOG(
          "The operation failed and will be automatically retried in ",
          absl::ToDoubleSeconds(delay), " seconds (attempt ", retries + 1,
          " out of ", max_retries, "), caused by: ", status);
    }

    absl::SleepFor(delay);
  }

  // Return AbortedError, so that it doesn't get retried again somewhere
  // at a higher level.
  return absl::AbortedError(
      StrCat("All ", max_retries, " retry attempts failed: ", status));
}

}  // namespace internal
}  // namespace tensorstore
