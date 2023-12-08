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

#include "tensorstore/internal/thread/schedule_at.h"

#include <stdint.h>

#include <algorithm>
#include <atomic>
#include <iterator>
#include <memory>
#include <utility>

#include "absl/base/optimization.h"
#include "absl/base/thread_annotations.h"
#include "absl/functional/any_invocable.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorstore/internal/attributes.h"
#include "tensorstore/internal/container/intrusive_red_black_tree.h"
#include "tensorstore/internal/metrics/gauge.h"
#include "tensorstore/internal/metrics/histogram.h"
#include "tensorstore/internal/metrics/value.h"
#include "tensorstore/internal/no_destructor.h"
#include "tensorstore/internal/tagged_ptr.h"
#include "tensorstore/internal/thread/thread.h"
#include "tensorstore/internal/tracing/tracing.h"
#include "tensorstore/util/stop_token.h"

namespace tensorstore {
namespace internal {
namespace {

using ScheduleAtTask = absl::AnyInvocable<void() &&>;

auto& schedule_at_queued_ops = internal_metrics::Gauge<int64_t>::New(
    "/tensorstore/internal/thread/schedule_at/queued_ops",
    "Operations in flight on the schedule_at thread");

auto& schedule_at_next_event = internal_metrics::Value<absl::Time>::New(
    "/tensorstore/internal/thread/schedule_at/next_event",
    "Time of the next in-flight schedule_at operation");

auto& schedule_at_insert_histogram_ms =
    internal_metrics::Histogram<internal_metrics::DefaultBucketer>::New(
        "/tensorstore/internal/thread/schedule_at/insert_histogram_ms",
        "Histogram of schedule_at insert delays (ms)");

class DeadlineTaskQueue;

using TaggedQueuePointer = TaggedPtr<DeadlineTaskQueue, 1>;

struct DeadlineTaskNode;

using DeadlineTaskTree = intrusive_red_black_tree::Tree<DeadlineTaskNode>;

struct DeadlineTaskStopCallback {
  DeadlineTaskNode& node;
  void operator()() const;
};

struct DeadlineTaskNode : public DeadlineTaskTree::NodeBase {
  DeadlineTaskNode(absl::Time deadline, ScheduleAtTask&& task,
                   const StopToken& token)
      : deadline(deadline),
        task(std::move(task)),
        trace_context(internal_tracing::TraceContext::kThread),
        queue(TaggedQueuePointer{}),
        stop_callback(token, DeadlineTaskStopCallback{*this}) {}

  // Runs `task` if cancellation was not already requested, and then deletes
  // `this`.
  void RunAndDelete();

  absl::Time deadline;
  ScheduleAtTask task;
  TENSORSTORE_ATTRIBUTE_NO_UNIQUE_ADDRESS internal_tracing::TraceContext
      trace_context;

  // The raw `DeadlineTaskQueue` pointer is non-null once the task has been
  // added to the tree.  The tag bit is 1 if cancellation of the task has been
  // requested.
  //
  // If the stop request occurs while the `DeadlineTaskQueue` pointer is null
  // (meaning it is in the process of being added to the queue by another
  // thread), then the tag bit is set to 1 and nothing else is done.  The task
  // will be cancelled by the thread that is adding it.
  //
  // If the stop request occurs while the `DeadlineTaskQueue` point is
  // non-null, then it is removed from the queue directly.
  std::atomic<TaggedQueuePointer> queue;
  StopCallback<DeadlineTaskStopCallback> stop_callback;
};

using RunImmediatelyQueueAccessor =
    intrusive_red_black_tree::LinkedListAccessor<DeadlineTaskNode>;

class DeadlineTaskQueue {
 public:
  explicit DeadlineTaskQueue()
      : run_immediately_queue_(nullptr),
        next_wakeup_(absl::InfinitePast()),
        woken_up_(absl::InfinitePast()),
        thread_({"TensorstoreScheduleAt"}, &DeadlineTaskQueue::Run, this) {}

  ~DeadlineTaskQueue() { ABSL_UNREACHABLE(); }  // COV_NF_LINE

  void ScheduleAt(absl::Time target_time, ScheduleAtTask task,
                  const StopToken& stop_token);

  void Run();

 private:
  friend struct DeadlineTaskNode;
  friend struct DeadlineTaskStopCallback;

  // Called from `DeadlineTaskStopCallback::operator()` to attempt to remove a
  // node if it is not already exiting.
  //
  // We can safely assume that `node` remains valid despite not initially
  // holding a lock on `mutex_` because the `DeadlineTaskStopCallback`
  // destructor blocks the destruction of the `DeadlineTaskNode` until any
  // concurrently running stop callback completes.
  void TryRemove(DeadlineTaskNode& node);

  absl::Mutex mutex_;
  absl::CondVar cond_var_;
  DeadlineTaskTree tree_ ABSL_GUARDED_BY(mutex_);

  // Additional circular linked list of tasks to run immediately.
  DeadlineTaskNode* run_immediately_queue_ ABSL_GUARDED_BY(mutex_);

  absl::Time next_wakeup_ ABSL_GUARDED_BY(mutex_);
  absl::Time woken_up_ ABSL_GUARDED_BY(mutex_);
  Thread thread_;
};

void DeadlineTaskQueue::ScheduleAt(absl::Time target_time, ScheduleAtTask task,
                                   const StopToken& stop_token) {
  schedule_at_queued_ops.Increment();
  schedule_at_insert_histogram_ms.Observe(
      absl::ToInt64Milliseconds(target_time - absl::Now()));

  auto node = std::make_unique<DeadlineTaskNode>(target_time, std::move(task),
                                                 stop_token);

  // Enqueue the task.
  absl::MutexLock l(&mutex_);

  auto tagged_queue_ptr = node->queue.exchange(TaggedQueuePointer(this));
  if (tagged_queue_ptr.tag()) {
    // Stop was requested already.
    //
    // Note: `return` destroys the task with the mutex *unlocked* because `node`
    // is declared before `l`.
    return;
  }
  if (target_time <= woken_up_) {
    // Target time is in the past, schedule to run immediately.
    RunImmediatelyQueueAccessor{}.SetNext(node.get(), nullptr);
    if (run_immediately_queue_) {
      RunImmediatelyQueueAccessor{}.SetNext(
          RunImmediatelyQueueAccessor{}.GetPrev(run_immediately_queue_),
          node.get());
      RunImmediatelyQueueAccessor{}.SetPrev(run_immediately_queue_, node.get());
    } else {
      run_immediately_queue_ = node.get();
      RunImmediatelyQueueAccessor{}.SetPrev(node.get(), node.get());
    }
    if (next_wakeup_ != absl::InfinitePast()) {
      next_wakeup_ = absl::InfinitePast();
      // Wake up thread immediately due to earlier deadline.
      cond_var_.Signal();
    }
    node.release();
    return;
  }

  // Schedule to run normally.
  tree_.FindOrInsert(
      [&](DeadlineTaskNode& other) {
        return target_time < other.deadline ? -1 : 1;
      },
      [&] { return node.release(); });
  if (target_time < next_wakeup_) {
    next_wakeup_ = target_time;
    // Wake up thread immediately due to earlier deadline.
    cond_var_.Signal();
  }
}

void DeadlineTaskQueue::Run() {
  while (true) {
    DeadlineTaskTree runnable;
    DeadlineTaskNode* run_immediately = nullptr;
    {
      absl::MutexLock l(&mutex_);
      do {
        run_immediately = std::exchange(run_immediately_queue_, nullptr);

        if (!run_immediately) {
          next_wakeup_ =
              tree_.empty() ? absl::InfiniteFuture() : tree_.begin()->deadline;

          // Sleep until our next deadline.
          schedule_at_next_event.Set(next_wakeup_);
          cond_var_.WaitWithDeadline(&mutex_, next_wakeup_);
        }

        // Consume the queue.
        auto woken_up = woken_up_ = std::max(woken_up_, absl::Now());

        auto split_result = tree_.FindSplit([&](DeadlineTaskNode& node) {
          return node.deadline <= woken_up ? 1 : -1;
        });

        runnable = std::move(split_result.trees[0]);
        tree_ = std::move(split_result.trees[1]);
      } while (runnable.empty() && !run_immediately);

      next_wakeup_ = absl::InfinitePast();
    }  // MutexLock

    // Execute functions without lock

    internal_tracing::TraceContext base =
        internal_tracing::TraceContext(internal_tracing::TraceContext::kThread);

    // First run any tasks in `run_immediately` list.
    while (run_immediately) {
      auto* next = RunImmediatelyQueueAccessor{}.GetNext(run_immediately);
      run_immediately->RunAndDelete();
      run_immediately = next;
    }

    // Run any tasks in `runnable` tree.
    for (DeadlineTaskTree::iterator it = runnable.begin(), next;
         it != runnable.end(); it = next) {
      next = std::next(it);
      runnable.Remove(*it);
      it->RunAndDelete();
    }

    internal_tracing::SwapCurrentTraceContext(&base);
  }
}

void DeadlineTaskNode::RunAndDelete() {
  schedule_at_queued_ops.Decrement();
  if (queue.load(std::memory_order_relaxed).tag()) {
    // Cancellation requested.
  } else {
    internal_tracing::SwapCurrentTraceContext(&trace_context);
    std::move(task)();
  }
  delete this;
}

void DeadlineTaskStopCallback::operator()() const {
  auto tagged_queue_ptr = node.queue.exchange(TaggedQueuePointer{nullptr, 1});
  auto* queue_ptr = tagged_queue_ptr.get();
  if (!queue_ptr) {
    // Task is still being added to the queue.  The thread that is adding it
    // will handle cancellation.
    return;
  }
  queue_ptr->TryRemove(node);
}

void DeadlineTaskQueue::TryRemove(DeadlineTaskNode& node) {
  {
    absl::MutexLock lock(&mutex_);
    // If deadline is met after the exchange operation above, the node will be
    // removed from the queue but not destroyed.
    if (node.deadline <= woken_up_) {
      // Task is being executed now.  Too late to cancel.
      return;
    }
    tree_.Remove(node);
    // No need to recompute `queue_ptr->next_wakeup_` here, since it can only
    // get later.
  }
  delete &node;
  schedule_at_queued_ops.Decrement();
}

}  // namespace

void ScheduleAt(absl::Time target_time, ScheduleAtTask task,
                const StopToken& stop_token) {
  static internal::NoDestructor<DeadlineTaskQueue> g_queue;
  g_queue->ScheduleAt(std::move(target_time), std::move(task), stop_token);
}

}  // namespace internal
}  // namespace tensorstore
