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

#include <atomic>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "tensorstore/internal/intrusive_ptr.h"

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

struct StopState : public internal::AtomicReferenceCount<StopState> {
  absl::Mutex mutex_;
  StopCallbackBase* callbacks_ ABSL_GUARDED_BY(mutex_) = nullptr;

  /// Only transitions from `false` to `true`, and only while holding `mutex_`,
  /// so that `RegisterImpl` and `RequestStop` observe a consistent value under
  /// the lock.  May be read without holding `mutex_`.
  std::atomic<bool> stop_requested_{false};

  StopState() = default;

  // true if a stop has been requested.
  bool stop_requested() const noexcept {
    return stop_requested_.load(std::memory_order_acquire);
  }

  /// Request a stop, running all currently registered callbacks.
  bool RequestStop() ABSL_LOCKS_EXCLUDED(mutex_);

  /// Adds the stop callback to the callback list, or, when stop has been
  /// requested immediately invokes the callback.  If registered, assigns
  /// `callback.state_` to `this`, which owns one reference to `this`.
  ///
  /// If `adopt_reference` is `true`, the caller transfers ownership of one
  /// existing reference to `this`, which is either assigned to
  /// `callback.state_` or released.  Otherwise a new reference is acquired if
  /// needed.
  void RegisterImpl(StopCallbackBase& callback, bool adopt_reference)
      ABSL_LOCKS_EXCLUDED(mutex_);

  /// Removes the stop callback from the callback list. May block if the
  /// callback is currently being run by a separate process.
  /// Calls intrusive_ptr_decrement(this) before returning, so the StopState
  /// may be deallocated immediately afterwards.
  void UnregisterImpl(StopCallbackBase& callback) ABSL_LOCKS_EXCLUDED(mutex_);
};

}  // namespace internal_stop_token
}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_STOP_TOKEN_IMPL_H_
