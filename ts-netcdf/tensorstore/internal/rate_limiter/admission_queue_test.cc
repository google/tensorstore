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

#include <atomic>
#include <utility>

#include <gtest/gtest.h>
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/rate_limiter/rate_limiter.h"
#include "tensorstore/util/executor.h"

namespace {

using ::tensorstore::ExecutorTask;
using ::tensorstore::internal::AdmissionQueue;
using ::tensorstore::internal::AtomicReferenceCount;
using ::tensorstore::internal::IntrusivePtr;
using ::tensorstore::internal::MakeIntrusivePtr;
using ::tensorstore::internal::RateLimiter;
using ::tensorstore::internal::RateLimiterNode;

/// This class holds a reference count on itself while held by a RateLimiter,
/// and upon start will call the `task_` function.
struct Task : public RateLimiterNode, public AtomicReferenceCount<Task> {
  RateLimiter* rate_limiter_;
  ExecutorTask task_;

  Task(RateLimiter* rate_limiter, ExecutorTask task)
      : rate_limiter_(rate_limiter), task_(std::move(task)) {
    // NOTE: Do not call Admit in the constructor as the task may complete
    // and try to delete self before MakeIntrusivePtr completes.
  }

  ~Task() { rate_limiter_->Finish(this); }

  void Admit() {
    intrusive_ptr_increment(this);  // adopted by RateLimiterTask::Start.
    rate_limiter_->Admit(this, &Task::Start);
  }

  static void Start(RateLimiterNode* task) {
    IntrusivePtr<Task> self(static_cast<Task*>(task),
                            tensorstore::internal::adopt_object_ref);
    std::move(self->task_)();
  }
};

TEST(AdmissionQueueTest, Basic) {
  AdmissionQueue queue(1);
  std::atomic<size_t> done{0};

  EXPECT_EQ(1, queue.limit());
  EXPECT_EQ(0, queue.in_flight());

  {
    for (int i = 0; i < 100; i++) {
      auto task = MakeIntrusivePtr<Task>(&queue, [&done] { done++; });
      task->Admit();
    }
  }

  EXPECT_EQ(100, done);
}

}  // namespace
