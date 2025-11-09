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

#ifndef TENSORSTORE_INTERNAL_THREAD_TASK_PROVIDER_H_
#define TENSORSTORE_INTERNAL_THREAD_TASK_PROVIDER_H_

#include <stdint.h>

#include "tensorstore/internal/intrusive_ptr.h"

namespace tensorstore {
namespace internal_thread_impl {

/// In conjunction with SharedThreadPool
class TaskProvider : public internal::AtomicReferenceCount<TaskProvider> {
 public:
  virtual ~TaskProvider() = default;

  /// Returns an instantaneous estimate of additional work threads required.
  virtual int64_t EstimateThreadsRequired() = 0;

  /// Worker Method: Assign a thread to this task provider.
  virtual void DoWorkOnThread() = 0;
};

}  // namespace internal_thread_impl
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_THREAD_TASK_PROVIDER_H_
