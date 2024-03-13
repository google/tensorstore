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

#include "tensorstore/internal/rate_limiter/token_bucket_rate_limiter.h"

#include <cassert>
#include <functional>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/thread_annotations.h"
#include "absl/log/absl_log.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorstore/internal/container/intrusive_linked_list.h"
#include "tensorstore/internal/log/verbose_flag.h"
#include "tensorstore/internal/rate_limiter/rate_limiter.h"
#include "tensorstore/internal/thread/schedule_at.h"

using ::tensorstore::internal::intrusive_linked_list::OnlyContainsNode;

namespace tensorstore {
namespace internal {
namespace {

ABSL_CONST_INIT internal_log::VerboseFlag rate_limiter_logging("rate_limiter");

}  // namespace

TokenBucketRateLimiter::TokenBucketRateLimiter(double max_tokens)
    : clock_(absl::Now),
      max_tokens_(max_tokens),
      start_time_(clock_()),
      last_update_(start_time_),
      allow_schedule_at_(true) {}

TokenBucketRateLimiter::TokenBucketRateLimiter(
    double max_tokens, std::function<absl::Time()> clock)
    : clock_(std::move(clock)),
      max_tokens_(max_tokens),
      start_time_(clock_()),
      last_update_(start_time_),
      allow_schedule_at_(false) {}

TokenBucketRateLimiter::~TokenBucketRateLimiter() {
  absl::MutexLock l(&mutex_);
  mutex_.Await(absl::Condition(
      +[](TokenBucketRateLimiter* self) ABSL_EXCLUSIVE_LOCKS_REQUIRED(
           self->mutex_) { return !self->scheduled_; },
      this));
}

void TokenBucketRateLimiter::Admit(RateLimiterNode* node,
                                   RateLimiterNode::StartFn fn) {
  assert(node->next_ == nullptr);
  assert(node->prev_ == nullptr);
  assert(node->start_fn_ == nullptr);
  node->start_fn_ = fn;

  // Admit to the queue.
  {
    absl::MutexLock lock(&mutex_);
    internal::intrusive_linked_list::InsertBefore(RateLimiterNodeAccessor{},
                                                  &head_, node);
    PerformWorkLocked();
  }
}

void TokenBucketRateLimiter::Finish(RateLimiterNode* node) {
  assert(node->next_ == nullptr);
  assert(node->prev_ == nullptr);
}

absl::Duration TokenBucketRateLimiter::GetSchedulerDelay() const {
  return absl::Milliseconds(10);
}

void TokenBucketRateLimiter::PeriodicCallForTesting() {
  absl::MutexLock lock(&mutex_);
  assert(!allow_schedule_at_);
  PerformWorkLocked();
}

void TokenBucketRateLimiter::PerformWork() {
  absl::MutexLock lock(&mutex_);
  PerformWorkLocked();
}

void TokenBucketRateLimiter::PerformWorkLocked() {
  RateLimiterNode local;
  internal::intrusive_linked_list::Initialize(RateLimiterNodeAccessor{},
                                              &local);

  // Attempt to fill the available_ tokens based on the timestamp.
  auto now = clock_();
  if (now > last_update_ && now > start_time_) {
    double to_add = TokensToAdd(now, last_update_);
    if (to_add > 0.5) {
      last_update_ = now;
      available_ += to_add;
      if (available_ > max_tokens_) {
        available_ = max_tokens_;
      }
      ABSL_LOG_IF(INFO, rate_limiter_logging)
          << "Fill " << to_add << " => " << available_;
    }
  }

  // Start all nodes which can be started.
  int count = 0;
  RateLimiterNodeAccessor accessor;
  while (available_ >= 1.0 && !OnlyContainsNode(accessor, &head_)) {
    available_ -= 1.0;
    auto* n = accessor.GetNext(&head_);
    internal::intrusive_linked_list::Remove(accessor, n);
    internal::intrusive_linked_list::InsertBefore(accessor, &local, n);
    count++;
  }

  // Maybe enqueue on scheduler.
  if (allow_schedule_at_ && !scheduled_ &&
      !OnlyContainsNode(accessor, &head_)) {
    // Non-empty list; maybe schedule.
    auto delay = GetSchedulerDelay();
    if (delay > absl::ZeroDuration()) {
      ABSL_LOG_IF(INFO, rate_limiter_logging.Level(1))
          << "ScheduleAt delay=" << delay;
      scheduled_ = true;
      internal::ScheduleAt(absl::Now() + delay, [this] {
        absl::MutexLock lock(&mutex_);
        scheduled_ = false;
        PerformWorkLocked();
      });
    }
  }

  if (count == 0) {
    return;
  }

  // Run all nodes without locks.
  mutex_.Unlock();
  ABSL_LOG_IF(INFO, rate_limiter_logging.Level(1)) << "Starting " << count;
  for (int i = 0; i < count; ++i) {
    auto* n = accessor.GetNext(&local);
    internal::intrusive_linked_list::Remove(accessor, n);
    RunStartFunction(n);
  }
  mutex_.Lock();
}

}  // namespace internal
}  // namespace tensorstore