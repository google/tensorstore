// Copyright 2024 The TensorStore Authors
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

#ifndef TENSORSTORE_INTERNAL_RATE_LIMITER_TOKEN_BUCKET_RATE_LIMITER_H_
#define TENSORSTORE_INTERNAL_RATE_LIMITER_TOKEN_BUCKET_RATE_LIMITER_H_

#include <functional>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tensorstore/internal/rate_limiter/rate_limiter.h"

namespace tensorstore {
namespace internal {

/// TokenBucketRateLimiter is a base class for implementing Leaky-bucket rate
/// limiter algorithms.
class TokenBucketRateLimiter : public RateLimiter {
 public:
  explicit TokenBucketRateLimiter(double max_tokens);

  // Test constructor.
  explicit TokenBucketRateLimiter(double max_tokens,
                                  std::function<absl::Time()> clock);

  ~TokenBucketRateLimiter() override;

  absl::Time start_time() const { return start_time_; }

  absl::Time last_update() const {
    absl::MutexLock l(&mutex_);
    return last_update_;
  }

  double available() const {
    absl::MutexLock l(&mutex_);
    return available_;
  }

  // Admit one operation; each admitted node costs 1.0.
  void Admit(RateLimiterNode* node, RateLimiterNode::StartFn fn) final;
  void Finish(RateLimiterNode* node) final;

  // Returns the number of tokens to add to the rate-limiter for
  // the time between start and end.
  virtual double TokensToAdd(absl::Time current, absl::Time previous) const = 0;

  // Returns the delay for next work unit.
  virtual absl::Duration GetSchedulerDelay() const;

  // Allows test-based manipulation of PerformWork.
  void PeriodicCallForTesting() ABSL_LOCKS_EXCLUDED(mutex_);

 protected:
  void PerformWork() ABSL_LOCKS_EXCLUDED(mutex_);
  void PerformWorkLocked() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // Intermediate state values.
  std::function<absl::Time()> clock_;
  const double max_tokens_;
  const absl::Time start_time_;
  absl::Time last_update_ ABSL_GUARDED_BY(mutex_);

  // Available buckets. Each request subtracts 1.0.
  double available_ ABSL_GUARDED_BY(mutex_) = 0;
  bool scheduled_ ABSL_GUARDED_BY(mutex_) = false;
  bool allow_schedule_at_ ABSL_GUARDED_BY(mutex_) = true;
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_RATE_LIMITER_TOKEN_BUCKET_RATE_LIMITER_H_
