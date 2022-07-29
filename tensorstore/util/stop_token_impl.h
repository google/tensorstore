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

#ifndef TENSORSTORE_UTIL_STOP_TOKEN_IMPL_H_
#define TENSORSTORE_UTIL_STOP_TOKEN_IMPL_H_

#include <stdint.h>

#include <atomic>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"

namespace tensorstore {
namespace internal_stop_token {

struct StopState;
struct StopCallbackInvocationState;

struct StopCallbackBase {
  /// If `next` equals `nullptr`, then the callback is no longer registered
  /// and `invocation_state` is the active member of the union.
  /// If `invocation_state` is also null, then the callback has completed.
  ///
  /// Otherwise, `next` and `prev` behave normally for a doubly-linked list.
  StopCallbackBase* next;

  union {
    /// Only valid if `next` does not equal `nullptr`.
    StopCallbackBase* prev;

    /// While this callback is being invoked, this points to a member on the
    /// stack of thread executing the callback, and is guarded by the
    /// StopState::mutex_.
    StopCallbackInvocationState* invocation_state;
  };

  /// Callback pointer.
  using CallbackInvoker = void (*)(StopCallbackBase&);
  CallbackInvoker invoker_;

  /// State pointer.
  std::atomic<StopState*> state_{nullptr};
};

struct StopState {
  absl::Mutex mutex_;
  StopCallbackBase* callbacks_ ABSL_GUARDED_BY(mutex_) = nullptr;
  bool stop_requested_ ABSL_GUARDED_BY(mutex_) = false;
  mutable std::atomic<std::uint32_t> ref_count_{0};

  StopState() = default;

  // true if a stop has been requested.
  bool stop_requested() ABSL_LOCKS_EXCLUDED(mutex_) {
    absl::MutexLock l(&mutex_);
    return stop_requested_;
  }

  friend void intrusive_ptr_increment(const StopState* p) noexcept {
    p->ref_count_.fetch_add(1, std::memory_order_acq_rel);
  }

  friend void intrusive_ptr_decrement(const StopState* p) noexcept {
    if (p->ref_count_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      delete p;
    }
  }

  /// Request a stop, running all currently registered callbacks.
  bool RequestStop() ABSL_LOCKS_EXCLUDED(mutex_);

  /// Adds the stop callback to the callback list, or, when stop has been
  /// requested immediately invokes the callback. May assign callback.state_
  /// to this, also incrementing the reference count.
  void RegisterImpl(StopCallbackBase& callback) ABSL_LOCKS_EXCLUDED(mutex_);

  /// Removes the stop callback from the callback list. May block if the
  /// callback is currently being run by a separate process.
  /// Call intrusive_ptr_increment(this) before returning, so the StopState
  /// may be deallocated immediately afterwards.
  void UnregisterImpl(StopCallbackBase& callback) ABSL_LOCKS_EXCLUDED(mutex_);
};

}  // namespace internal_stop_token
}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_STOP_TOKEN_IMPL_H_
