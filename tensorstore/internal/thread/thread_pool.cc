// Copyright 2020 The TensorStore Authors
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

#include "tensorstore/internal/thread/thread_pool.h"

#include <stddef.h>

#include <cassert>
#include <limits>
#include <memory>
#include <thread>  // NOLINT
#include <utility>

#include "absl/log/absl_check.h"
#include "absl/log/absl_log.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/no_destructor.h"
#include "tensorstore/internal/thread/pool_impl.h"
#include "tensorstore/internal/thread/task.h"
#include "tensorstore/internal/thread/task_group_impl.h"
#include "tensorstore/util/executor.h"

namespace tensorstore {
namespace internal {
namespace {

Executor DefaultThreadPool(size_t num_threads) {
  static internal::NoDestructor<internal_thread_impl::SharedThreadPool> pool_;
  intrusive_ptr_increment(pool_.get());
  if (num_threads == 0 || num_threads == std::numeric_limits<size_t>::max()) {
    // Threads are "unbounded"; that doesn't work so well, so put a bound on it.
    num_threads = std::thread::hardware_concurrency() * 16;
    if (num_threads == 0) num_threads = 1024;
    ABSL_LOG_FIRST_N(INFO, 1)
        << "DetachedThreadPool should specify num_threads; using "
        << num_threads;
  }

  auto task_group = internal_thread_impl::TaskGroup::Make(
      internal::IntrusivePtr<internal_thread_impl::SharedThreadPool>(
          pool_.get()),
      num_threads);
  return [task_group = std::move(task_group)](ExecutorTask task) {
    task_group->AddTask(
        std::make_unique<internal_thread_impl::InFlightTask>(std::move(task)));
  };
}

}  // namespace

Executor DetachedThreadPool(size_t num_threads) {
  return DefaultThreadPool(num_threads);
}

}  // namespace internal
}  // namespace tensorstore
