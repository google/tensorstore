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

#ifndef TENSORSTORE_INTERNAL_THREAD_TASK_H_
#define TENSORSTORE_INTERNAL_THREAD_TASK_H_

#include <stdint.h>

#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/time/clock.h"
#include "tensorstore/internal/attributes.h"
#include "tensorstore/internal/tracing/tracing.h"

namespace tensorstore {
namespace internal_thread_impl {

/// An in-flight task. Implementation detail of thread_pool.
struct InFlightTask {
  InFlightTask(absl::AnyInvocable<void() &&> callback)
      : callback_(std::move(callback)),
        tc_(internal_tracing::TraceContext(
            internal_tracing::TraceContext::kThread)),
        start_nanos(absl::GetCurrentTimeNanos()) {}

  void Run() {
    internal_tracing::SwapCurrentTraceContext(&tc_);
    std::move(callback_)();
    callback_ = {};
    internal_tracing::SwapCurrentTraceContext(&tc_);
  }

  absl::AnyInvocable<void() &&> callback_;
  TENSORSTORE_ATTRIBUTE_NO_UNIQUE_ADDRESS internal_tracing::TraceContext tc_;
  int64_t start_nanos;
};

}  // namespace internal_thread_impl
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_THREAD_TASK_H_
