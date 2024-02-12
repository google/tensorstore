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

#include <stdint.h>

#include <cassert>

#include "absl/random/random.h"
#include "absl/time/time.h"

namespace tensorstore {
namespace internal {

absl::Duration BackoffForAttempt(int attempt, absl::Duration initial_delay,
                                 absl::Duration max_delay,
                                 absl::Duration jitter) {
  assert(initial_delay > absl::ZeroDuration());
  assert(max_delay >= initial_delay);
  assert(attempt >= 0);

  int64_t multiple = int64_t{1} << (attempt > 62 ? 62 : attempt);
  auto delay = initial_delay * multiple;
  int64_t jitter_us = absl::ToInt64Microseconds(jitter);
  if (jitter_us > 0) {
    delay += absl::Microseconds(absl::Uniform(
        absl::IntervalClosed, absl::InsecureBitGen{}, 0, jitter_us));
  }
  if (delay > max_delay) delay = max_delay;
  return delay;
}

}  // namespace internal
}  // namespace tensorstore
