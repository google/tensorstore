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

#ifndef TENSORSTORE_KVSTORE_GCS_SCALING_RATE_LIMITER_H_
#define TENSORSTORE_KVSTORE_GCS_SCALING_RATE_LIMITER_H_

#include <assert.h>
#include <stddef.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tensorstore/internal/poly/poly.h"
#include "tensorstore/kvstore/gcs/rate_limiter.h"

namespace tensorstore {
namespace internal_storage_gcs {

/// ScalingRateLimiter implements a rate-limiter which is similar to a typical
/// Leaky-Bucket rate limiting algorithm with optional time-based growth
/// scaling. GCS best practices suggest that reads should be limited to 5000/s
/// and ramp up to double that over 20 minutes.
///
/// https://cloud.google.com/storage/docs/request-rate#ramp-up
///
/// The ScalingRateLimiter accepts an initial_rate and a doubling_time. The
/// doubling time is computed over the life of the ScalingRateLimiter, and so
/// is best used for short-duration processes.
class ScalingRateLimiter : public RateLimiter {
 public:
  using ClockPoly = poly::Poly<sizeof(void*), true, absl::Time()>;

  /// Constructs a ScalingRateLimiter
  //
  ScalingRateLimiter(double initial_rate, double max_available,
                     absl::Duration doubling_time);

  ~ScalingRateLimiter() override;

  /// Accessors.
  double initial_rate() const { return initial_rate_; }
  absl::Duration doubling_time() const { return doubling_time_; }
  absl::Time start_time() const { return start_time_; }

  double available() const {
    absl::MutexLock l(&mutex_);
    return available_;
  }

  // Admit one operation; each admitted node costs 1.0.
  void Admit(RateLimiterNode* node, RateLimiterNode::StartFn fn) override;
  void Finish(RateLimiterNode* node) override;

  // For testing; set a fake clock and issue a periodic call.
  void SetClockForTesting(ClockPoly clock_now) {
    clock_now_ = std::move(clock_now);
  }
  void PeriodicCallForTesting() { StartAvailableNodes(kUpdateBit); }

 private:
  void UpdateCapacity(absl::Time now) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  static constexpr uint32_t kUpdateBit = 1;
  static constexpr uint32_t kClearSchedulerBit = 2;

  // Starts all available nodes
  void StartAvailableNodes(uint32_t state);

  // Rate-limiter algorithm values.
  const double initial_rate_;
  const double max_available_;
  const absl::Duration doubling_time_;
  const absl::Time start_time_;
  const double a_;  // ln(2)/Tdouble; 0 = constant rate.

  // Intermediate state values.
  absl::Time last_update_ ABSL_GUARDED_BY(mutex_);
  double last_a_t_ ABSL_GUARDED_BY(mutex_) = 1.0;

  // Available buckets. Each request subtracts 1.0.
  double available_ ABSL_GUARDED_BY(mutex_) = 0;
  bool running_in_scheduler_ ABSL_GUARDED_BY(mutex_) = false;

  ClockPoly clock_now_;
};

}  // namespace internal_storage_gcs
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_GCS_SCALING_RATE_LIMITER_H_
