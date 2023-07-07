// Copyright 2023 The TensorStore Authors
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

#include "tensorstore/kvstore/gcs_http/admission_queue.h"

#include <gtest/gtest.h>
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/kvstore/gcs_http/rate_limiter.h"
#include "tensorstore/util/executor.h"

namespace {

using ::tensorstore::Executor;
using ::tensorstore::ExecutorTask;
using ::tensorstore::internal::adopt_object_ref;
using ::tensorstore::internal::AtomicReferenceCount;
using ::tensorstore::internal::IntrusivePtr;
using ::tensorstore::internal::MakeIntrusivePtr;
using ::tensorstore::internal_kvstore_gcs_http::AdmissionQueue;
using ::tensorstore::internal_kvstore_gcs_http::RateLimiterNode;

struct Node : public RateLimiterNode, public AtomicReferenceCount<Node> {
  AdmissionQueue* queue_;
  ExecutorTask task_;

  Node(AdmissionQueue* queue, ExecutorTask task)
      : queue_(queue), task_(std::move(task)) {}

  ~Node() { queue_->Finish(this); }

  static void Start(void* task) {
    IntrusivePtr<Node> self(reinterpret_cast<Node*>(task), adopt_object_ref);
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
      auto node = MakeIntrusivePtr<Node>(&queue, [&done] { done++; });

      intrusive_ptr_increment(node.get());  // adopted by Node::Start.
      queue.Admit(node.get(), &Node::Start);
    }
  }

  EXPECT_EQ(100, done);
}

}  // namespace
