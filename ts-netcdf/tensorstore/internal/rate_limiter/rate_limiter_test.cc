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

#include "tensorstore/internal/rate_limiter/rate_limiter.h"

#include <stddef.h>

#include <atomic>
#include <type_traits>

#include <gtest/gtest.h>
#include "tensorstore/internal/intrusive_ptr.h"

namespace {

using ::tensorstore::internal::AtomicReferenceCount;
using ::tensorstore::internal::IntrusivePtr;
using ::tensorstore::internal::MakeIntrusivePtr;
using ::tensorstore::internal::NoRateLimiter;
using ::tensorstore::internal::RateLimiter;
using ::tensorstore::internal::RateLimiterNode;

/// This class holds a reference count on itself while held by a RateLimiter,
/// and upon start will call the `task_` function.
template <typename Fn>
struct RateLimiterTask : public AtomicReferenceCount<RateLimiterTask<Fn>>,
                         public RateLimiterNode {
  RateLimiter* rate_limiter_;
  Fn task_;

  RateLimiterTask(RateLimiter* rate_limiter, Fn task)
      : rate_limiter_(rate_limiter), task_(std::move(task)) {
    // NOTE: Do not call Admit in the constructor as the task may complete
    // and try to delete self before MakeIntrusivePtr completes.
  }

  ~RateLimiterTask() { rate_limiter_->Finish(this); }

  void Admit() {
    intrusive_ptr_increment(this);  // adopted by RateLimiterTask::Start.
    rate_limiter_->Admit(this, &RateLimiterTask::Start);
  }

  static void Start(RateLimiterNode* task) {
    IntrusivePtr<RateLimiterTask> self(static_cast<RateLimiterTask<Fn>*>(task),
                                       tensorstore::internal::adopt_object_ref);
    std::move(self->task_)();
  }
};

TEST(AdmissionQueueTest, Basic) {
  NoRateLimiter queue;
  std::atomic<size_t> done{0};

  auto increment = [&done] { done++; };
  using Node = RateLimiterTask<std::remove_reference_t<decltype(increment)>>;

  {
    for (int i = 0; i < 100; i++) {
      MakeIntrusivePtr<Node>(&queue, increment)->Admit();
    }
  }

  EXPECT_EQ(100, done);
}
}  // namespace
