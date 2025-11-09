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

#ifndef TENSORSTORE_INTERNAL_RETRY_H_
#define TENSORSTORE_INTERNAL_RETRY_H_

#include "absl/time/time.h"

namespace tensorstore {
namespace internal {

/// BackoffForAttempt computes a backoff to use after a retry attempt.
/// Example:
///   for (int i = 0; i < max_retries; i++) {
///     if (function()) return absl::OkStatus();
///     auto delay = BackoffForAttempt(i);
///     thread::Sleep(*delay);
///   }
absl::Duration BackoffForAttempt(
    int attempt,
    absl::Duration initial_delay,  // GCS recommends absl::Seconds(1)
    absl::Duration max_delay,      // GCS recommends absl::Seconds(32)
    absl::Duration jitter          // GCS recommends absl::Seconds(1)
);

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_RETRY_H_
