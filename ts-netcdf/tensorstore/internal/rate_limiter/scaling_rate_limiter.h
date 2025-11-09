// Copyright 2022 The TensorStore Authors
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

#ifndef TENSORSTORE_INTERNAL_RATE_LIMITER_SCALING_RATE_LIMITER_H_
#define TENSORSTORE_INTERNAL_RATE_LIMITER_SCALING_RATE_LIMITER_H_

#include <functional>

#include "absl/time/time.h"
#include "tensorstore/internal/rate_limiter/token_bucket_rate_limiter.h"

namespace tensorstore {
namespace internal {

/// DoublingRateLimiter implements a leaky-bucket rate-limiter with time-based
/// growth scaling. GCS best practices suggest that reads should be limited to
/// 5000/s and ramp up to double that over 20 minutes.
///
/// https://cloud.google.com/storage/docs/request-rate#ramp-up
///
/// The DoublingRateLimiter accepts an initial_rate and a doubling_time. The
/// doubling time is computed over the life of the DoublingRateLimiter, and so
/// is best used for short-duration processes.
class DoublingRateLimiter : public TokenBucketRateLimiter {
 public:
  /// Constructs a DoublingRateLimiter
  DoublingRateLimiter(double initial_rate, absl::Duration doubling_time);

  // Test constructor.
  DoublingRateLimiter(double initial_rate, absl::Duration doubling_time,
                      std::function<absl::Time()> clock);

  ~DoublingRateLimiter() override = default;

  /// Accessors.
  double initial_rate() const { return initial_rate_; }
  absl::Duration doubling_time() const { return doubling_time_; }

  double TokensToAdd(absl::Time current, absl::Time previous) const override;

  // Returns the delay for next work unit.
  absl::Duration GetSchedulerDelay() const override;

 private:
  const double initial_rate_;
  const absl::Duration doubling_time_;
  const double a_;  // ln(2)/Tdouble.
};

/// ConstantRateLimiter implements a simple linear rate-limiter which is similar
/// to a typical Leaky-Bucket rate limiting algorithm.
class ConstantRateLimiter : public TokenBucketRateLimiter {
 public:
  /// Constructs a DoublingRateLimiter
  explicit ConstantRateLimiter(double initial_rate);

  // Test constructor.
  ConstantRateLimiter(double initial_rate, std::function<absl::Time()> clock);

  ~ConstantRateLimiter() override = default;

  /// Accessors.
  double initial_rate() const { return initial_rate_; }

  double TokensToAdd(absl::Time current, absl::Time previous) const override;

  // Returns the delay for next work unit.
  absl::Duration GetSchedulerDelay() const override;

 private:
  const double initial_rate_;
  const absl::Duration r_;  // interval at which new tokens are added.
};

}  // namespace internal
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_RATE_LIMITER_SCALING_RATE_LIMITER_H_
