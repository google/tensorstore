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

#include "tensorstore/kvstore/gcs/admission_queue.h"

#include <assert.h>
#include <stddef.h>

#include <memory>
#include <optional>

namespace tensorstore {
namespace internal_storage_gcs {

AdmissionQueue::AdmissionQueue(size_t limit)
    : limit_(limit == 0 ? std::numeric_limits<size_t>::max() : limit) {}

void AdmissionQueue::Admit(RateLimiterNode* node, RateLimiterNode::StartFn fn) {
  assert(node->next_ == nullptr);
  assert(node->prev_ == nullptr);
  assert(node->start_fn_ == nullptr);
  node->start_fn_ = fn;

  {
    absl::MutexLock lock(&mutex_);
    if (in_flight_++ >= limit_) {
      internal::intrusive_linked_list::InsertBefore(RateLimiterNodeAccessor{},
                                                    &head_, node);
      return;
    }
  }

  RunStartFunction(node);
}

void AdmissionQueue::Finish(RateLimiterNode* node) {
  assert(node->next_ == nullptr);

  RateLimiterNode* next_node = nullptr;
  {
    absl::MutexLock lock(&mutex_);
    in_flight_--;
    next_node = head_.next_;
    if (next_node == &head_) return;
    internal::intrusive_linked_list::Remove(RateLimiterNodeAccessor{},
                                            next_node);
  }

  // Next node gets a chance to run after clearing admission queue state.
  RunStartFunction(next_node);
}

}  // namespace internal_storage_gcs
}  // namespace tensorstore
