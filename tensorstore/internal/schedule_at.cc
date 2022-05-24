#include "tensorstore/internal/schedule_at.h"

#include <algorithm>
#include <cassert>
#include <queue>
#include <thread>  // NOLINT
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/logging.h"
#include "tensorstore/internal/metrics/gauge.h"
#include "tensorstore/internal/mutex.h"
#include "tensorstore/internal/no_destructor.h"
#include "tensorstore/util/assert_macros.h"
#include "tensorstore/util/executor.h"

namespace tensorstore {
namespace internal {
namespace {

auto& executor_queued_ops = internal_metrics::Gauge<int64_t>::New(
    "/tensorstore/internal/schedule_at/queued_ops",
    "Operations in flight on the managed thread pool");

struct DeadlineTask {
  absl::Time deadline;
  ExecutorTask task;
};

/// Comparison for constructing a min-heap.
struct Compare {
  bool operator()(const DeadlineTask& a, const DeadlineTask& b) {
    return b.deadline < a.deadline;
  }
};

class DeadlineTaskQueue {
 public:
  explicit DeadlineTaskQueue()
      : next_wakeup_(absl::InfiniteFuture()),
        thread_(&DeadlineTaskQueue::Run, this) {}

  ~DeadlineTaskQueue() { TENSORSTORE_UNREACHABLE; }

  void ScheduleAt(absl::Time target_time, ExecutorTask task);

  void Run();

 private:
  static bool Wakeup(DeadlineTaskQueue* self)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(self->mutex_) {
    return self->next_wakeup_ > self->heap_min_deadline();
  }

  absl::Time heap_min_deadline() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mutex_) {
    return heap_.empty() ? absl::InfiniteFuture() : heap_.begin()->deadline;
  }

  absl::Mutex mutex_;
  std::vector<DeadlineTask> heap_ ABSL_GUARDED_BY(mutex_);
  absl::Time next_wakeup_ ABSL_GUARDED_BY(mutex_);

  std::thread thread_;
};

void DeadlineTaskQueue::ScheduleAt(absl::Time target_time, ExecutorTask task) {
  executor_queued_ops.Increment();

  {
    absl::MutexLock l(&mutex_);
    heap_.emplace_back(DeadlineTask{std::move(target_time), std::move(task)});
    std::push_heap(heap_.begin(), heap_.end(), Compare{});
  }
}

void DeadlineTaskQueue::Run() {
  std::vector<DeadlineTask> runnable;
  runnable.reserve(1000);
  absl::Time minimum_wakeup;

  while (true) {
    {
      absl::MutexLock l(&mutex_);

      // Sleep until our next deadline.
      next_wakeup_ = heap_min_deadline();
      mutex_.AwaitWithDeadline(
          absl::Condition(&DeadlineTaskQueue::Wakeup, this), next_wakeup_);

      // Consume the queue.
      minimum_wakeup = absl::Now();
      while (!heap_.empty() && heap_.begin()->deadline <= minimum_wakeup) {
        std::pop_heap(heap_.begin(), heap_.end(), Compare{});
        runnable.emplace_back(std::move(heap_.back()));
        heap_.pop_back();
      }
    }  // MutexLock

    // Execute functions without lock
    for (auto& r : runnable) {
      executor_queued_ops.Decrement();
      r.task();
    }
    runnable.clear();
  }
}

}  // namespace

void ScheduleAt(absl::Time target_time, ExecutorTask task) {
  static internal::NoDestructor<DeadlineTaskQueue> g_queue;
  g_queue->ScheduleAt(std::move(target_time), std::move(task));
}

}  // namespace internal
}  // namespace tensorstore
