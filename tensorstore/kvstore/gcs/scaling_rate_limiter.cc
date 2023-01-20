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
#include "tensorstore/kvstore/gcs/scaling_rate_limiter.h"

#include <stddef.h>

#include <limits>
#include <memory>
#include <optional>

#include "absl/log/absl_check.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tensorstore/data_type.h"
#include "tensorstore/internal/schedule_at.h"
#include "tensorstore/kvstore/gcs/rate_limiter.h"

namespace tensorstore {
namespace internal_storage_gcs {
namespace {

double GetLogA(absl::Duration doubling_time) {
  if (doubling_time <= absl::ZeroDuration() ||
      doubling_time == absl::InfiniteDuration()) {
    return 0;
  }
  // Given a doubling time, Tdouble, a is determined by the equation:
  // Tdouble = ln(2) / ln(a). So ln(a) = ln(2) / Tdouble.
  return 0.69314718055994530941723212145817656 /
         absl::ToDoubleSeconds(doubling_time);
}

double GetReasonableMaxAvailable(double max_available) {
  if (max_available < 2) {
    return (max_available <= 0) ? 10000.0 : 2.0;
  }
  return std::min(10000.0, max_available);
}

}  // namespace

ScalingRateLimiter::ScalingRateLimiter(double initial_rate,
                                       double max_available,
                                       absl::Duration doubling_time)
    : initial_rate_(initial_rate),
      max_available_(GetReasonableMaxAvailable(max_available)),
      doubling_time_(doubling_time),
      start_time_(absl::Now()),
      a_(GetLogA(doubling_time)),
      last_update_(start_time_),
      clock_now_([]() { return absl::Now(); }) {
  ABSL_CHECK_GT(initial_rate, std::numeric_limits<double>::min());
  absl::MutexLock l(&mutex_);
  internal::intrusive_linked_list::Initialize(RateLimiterNodeAccessor{},
                                              &head_);
  UpdateCapacity(start_time_);
}

ScalingRateLimiter::~ScalingRateLimiter() {
  absl::MutexLock l(&mutex_);
  mutex_.Await(absl::Condition(
      +[](ScalingRateLimiter* self) ABSL_EXCLUSIVE_LOCKS_REQUIRED(
           self->mutex_) { return !self->running_in_scheduler_; },
      this));
}

void ScalingRateLimiter::UpdateCapacity(absl::Time now) {
  auto time_delta = now - last_update_;
  if (time_delta < absl::Milliseconds(10)) {
    // Avoid updating capacity too frequently.
    return;
  }
  last_update_ = now;

  // Take the integral since the last call.
  if (a_ > 0) {
    // Using an exponential growth model, so take the integral is:
    // integral[t0..t1] of e^ax dx.
    // which evaluates to 1/a * [e^(a*t1) - e^(a*t2)].
    double a_t = std::exp(a_ * absl::ToDoubleSeconds(now - start_time_));
    double integral = (a_t - last_a_t_) / a_;
    last_a_t_ = a_t;
    available_ += initial_rate_ * integral;
  } else {
    // No growth, so the integral is simply Tdelta.
    available_ += initial_rate_ * absl::ToDoubleSeconds(time_delta);
  }
  if (available_ > max_available_) {
    available_ = max_available_;
  }
}

void ScalingRateLimiter::StartAvailableNodes(uint32_t state) {
  RateLimiterNode* next_node = nullptr;
  while (true) {
    {
      absl::MutexLock lock(&mutex_);
      if (state & kUpdateBit) {
        UpdateCapacity(clock_now_());
        state ^= kUpdateBit;
      }

      next_node = head_.next_;
      if (next_node == &head_) {
        // No more nodes, so clear the scheduler bit if requested.
        if (state & kClearSchedulerBit) {
          running_in_scheduler_ = false;
        }
        return;
      }
      if (available_ < 1.0) {
        // No more capacity, maybe reschedule.
        if (state & kClearSchedulerBit) {
          running_in_scheduler_ = false;
        }
        if (!running_in_scheduler_) {
          running_in_scheduler_ = true;
          internal::ScheduleAt(absl::Now() + absl::Milliseconds(100), [this] {
            StartAvailableNodes(kUpdateBit | kClearSchedulerBit);
          });
        }
        return;
      }

      available_ -= 1.0;
      internal::intrusive_linked_list::Remove(RateLimiterNodeAccessor{},
                                              next_node);
    }

    RunStartFunction(next_node);
  }
}

void ScalingRateLimiter::Admit(RateLimiterNode* node,
                               RateLimiterNode::StartFn fn) {
  assert(node->next_ == nullptr);
  assert(node->prev_ == nullptr);
  assert(node->start_fn_ == nullptr);
  node->start_fn_ = fn;

  // Admit to the queue.
  {
    absl::MutexLock lock(&mutex_);
    UpdateCapacity(clock_now_());
    internal::intrusive_linked_list::InsertBefore(RateLimiterNodeAccessor{},
                                                  &head_, node);
  }

  StartAvailableNodes(0);
}

void ScalingRateLimiter::Finish(RateLimiterNode* node) {
  assert(node->next_ == nullptr);
  StartAvailableNodes(kUpdateBit);
}

}  // namespace internal_storage_gcs
}  // namespace tensorstore
