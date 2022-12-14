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

#ifndef TENSORSTORE_KVSTORE_GCS_ADMISSION_QUEUE_H_
#define TENSORSTORE_KVSTORE_GCS_ADMISSION_QUEUE_H_

#include <stddef.h>

#include <limits>
#include <memory>
#include <optional>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "tensorstore/kvstore/gcs/rate_limiter.h"

namespace tensorstore {
namespace internal_storage_gcs {

/// AdmissionQueue implements a `RateLimiter` which restricts operation
/// parallelism to a pre-specified limit. When the requested number of in-flight
/// operations exceeds the limit, future operations will be gated on prior
/// operation completion.
///
/// AdmissionQueue maintains a list of pending operations managed via
/// `AdmissionQueue::Admit` and `AdmissionQueue::Finish` methods. `Admit` must
/// be called when an operation starts, and `Finish` must be called when an
/// operation completes. Operations are enqueued if limit is reached, to be
/// started once the number of parallel operations are below limit.
class AdmissionQueue : public RateLimiter {
 public:
  /// Construct an AdmissionQueue with `limit` parallelism.
  AdmissionQueue(size_t limit);
  ~AdmissionQueue() override = default;

  size_t limit() const { return limit_; }
  size_t in_flight() const {
    absl::MutexLock l(&mutex_);
    return in_flight_;
  }

  /// Admit a task node to the queue. Admit ensures that at most `limit`
  /// operations are running concurrently.  When the node is admitted the start
  /// function, `fn(node)`, which may happen immediately or when space is
  /// available.
  void Admit(RateLimiterNode* node, RateLimiterNode::StartFn fn) override;

  /// Mark a task node for completion. When a node finishes in this way, a
  /// queued node will have it's start function invoked.
  void Finish(RateLimiterNode* node) override;

 private:
  const size_t limit_;
  size_t in_flight_ ABSL_GUARDED_BY(mutex_) = 0;
};

}  // namespace internal_storage_gcs
}  // namespace tensorstore

#endif  // TENSORSTORE_KVSTORE_GCS_ADMISSION_QUEUE_H_
