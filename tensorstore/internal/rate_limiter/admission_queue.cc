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

#include "tensorstore/internal/rate_limiter/admission_queue.h"

#include <stddef.h>

#include <cassert>
#include <limits>

#include "absl/synchronization/mutex.h"
#include "tensorstore/internal/container/intrusive_linked_list.h"
#include "tensorstore/internal/rate_limiter/rate_limiter.h"

namespace tensorstore {
namespace internal {

AdmissionQueue::AdmissionQueue(size_t limit)
    : limit_(limit == 0 ? std::numeric_limits<size_t>::max() : limit) {
  internal::intrusive_linked_list::Initialize(RateLimiterNodeAccessor{},
                                              &head_);
}

AdmissionQueue::~AdmissionQueue() {
  absl::MutexLock l(&mutex_);
  assert(head_.next_ == &head_);
}

void AdmissionQueue::Admit(RateLimiterNode* node, RateLimiterNode::StartFn fn) {
  assert(node->next_ == nullptr);
  assert(node->prev_ == nullptr);
  assert(node->start_fn_ == nullptr);
  node->start_fn_ = fn;

  {
    absl::MutexLock lock(&mutex_);
    if (in_flight_ + 1 > limit_) {
      internal::intrusive_linked_list::InsertBefore(RateLimiterNodeAccessor{},
                                                    &head_, node);
      return;
    }
    in_flight_++;
  }

  RunStartFunction(node);
}

void AdmissionQueue::Finish(RateLimiterNode* node) {
  assert(node->next_ == nullptr);

  absl::MutexLock lock(&mutex_);
  in_flight_--;

  // Typically this loop will admit only a single node at a time.
  RateLimiterNode* next_node = nullptr;
  while (true) {
    next_node = head_.next_;
    if (next_node == &head_) return;
    if (in_flight_ + 1 > limit_) return;
    in_flight_++;
    internal::intrusive_linked_list::Remove(RateLimiterNodeAccessor{},
                                            next_node);

    // Next node gets a chance to run after clearing admission queue state.
    mutex_.Unlock();
    RunStartFunction(next_node);
    mutex_.Lock();
  }
}

}  // namespace internal
}  // namespace tensorstore
