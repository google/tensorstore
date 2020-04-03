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
#include <functional>
#include <optional>
#include <string>
#include <system_error>  // NOLINT

#include "absl/base/macros.h"
#include "absl/random/random.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorstore/internal/logging.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal {

bool DefaultIsRetriable(const tensorstore::Status& status) {
  return (status.code() == absl::StatusCode::kUnknown ||
          status.code() == absl::StatusCode::kDeadlineExceeded ||
          status.code() == absl::StatusCode::kUnavailable);
}

Status RetryWithBackoff(std::function<Status()> function, int max_retries,
                        absl::Duration initial_delay_time,
                        absl::Duration max_delay_time,
                        std::function<bool(const Status&)> is_retriable) {
  ABSL_ASSERT(initial_delay_time >= absl::ZeroDuration());
  ABSL_ASSERT(max_delay_time >= initial_delay_time);
  ABSL_ASSERT(max_retries >= 0);

  std::optional<absl::BitGen> rng;
  Status status;
  for (int retries = 0; retries < max_retries; retries++) {
    status = function();
    if (status.ok() || !is_retriable(status)) {
      return status;
    }

    // Compute backoff.
    auto delay = absl::ZeroDuration();
    if (initial_delay_time > absl::ZeroDuration()) {
      int64_t jitter = ToInt64Microseconds(initial_delay_time) *
                       ((retries > 1) ? (1 << (retries - 1)) : 1);
      jitter = std::max(jitter, static_cast<int64_t>(1000));
      delay = initial_delay_time * (1 << retries);
      if (!rng) rng.emplace();
      delay += absl::Microseconds(absl::Uniform(*rng, 0, jitter));
      if (delay > max_delay_time) delay = max_delay_time;
    }

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
