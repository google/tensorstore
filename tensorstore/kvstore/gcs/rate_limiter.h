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

#ifndef TENSORSTORE_KVSTORE_GCS_RATE_LIMITED_NODE_H_
#define TENSORSTORE_KVSTORE_GCS_RATE_LIMITED_NODE_H_

#include <cassert>
#include <clocale>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "tensorstore/internal/intrusive_linked_list.h"

namespace tensorstore {
namespace internal_storage_gcs {

// RateLimiter is an interface which supports rate-limiting for an operation.
// Pending operations use the `RateLimiterNode` base class, and are managed
// via `RateLimiter::Admit` and `RateLimiter::Finish` calls.
//
// Generally, a RateLimiterNode will also be reference counted, however neither
// the RateLimiterNode nor the RateLimiter class manage any reference counts.
// Callers should manage reference counts externally.
//
struct RateLimiterNode {
  using StartFn = void (*)(void*);

  RateLimiterNode* next_ = nullptr;
  RateLimiterNode* prev_ = nullptr;
  StartFn start_fn_ = nullptr;
};

using RateLimiterNodeAccessor = internal::intrusive_linked_list::MemberAccessor<
    RateLimiterNode, &RateLimiterNode::prev_, &RateLimiterNode::next_>;

/// RateLimiter interface.
class RateLimiter {
 public:
  RateLimiter();
  virtual ~RateLimiter();

  /// Add a task to  the rate limiter. Will arrange for `fn(node)` to be called
  /// at some (possible future) point.
  virtual void Admit(RateLimiterNode* node, RateLimiterNode::StartFn fn) = 0;

  /// Cleanup a task from the rate limiter.
  virtual void Finish(RateLimiterNode* node) = 0;

 protected:
  void RunStartFunction(RateLimiterNode* node);

  mutable absl::Mutex mutex_;
  RateLimiterNode head_ ABSL_GUARDED_BY(mutex_);
};

/// RateLimiter interface.
class NoRateLimiter : public RateLimiter {
 public:
  NoRateLimiter() = default;
  ~NoRateLimiter() override = default;

  /// Add a task to  the rate limiter. Will arrange for `fn(node)` to be called
  /// at some (possible future) point.
  void Admit(RateLimiterNode* node, RateLimiterNode::StartFn fn) override;

  /// Cleanup a task from the rate limiter.
  void Finish(RateLimiterNode* node) override;
};

}  // namespace internal_storage_gcs
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_GCS_RATE_LIMITED_NODE_H_
