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

#include "tensorstore/util/stop_token_impl.h"

#include <assert.h>
#include <stdint.h>

#include <atomic>
#include <thread>

#include "absl/synchronization/mutex.h"
#include "tensorstore/internal/intrusive_linked_list.h"

namespace tensorstore {
namespace internal_stop_token {

struct StopCallbackInvocationState {
  std::thread::id thread_id;
  bool callback_destroyed;
};

using CallbackListAccessor =
    tensorstore::internal::intrusive_linked_list::MemberAccessor<
        StopCallbackBase>;

bool StopState::RequestStop() {
  StopCallbackInvocationState invocation_state{
      /*thread_id=*/std::this_thread::get_id(), /*callback_destroyed=*/false};

  absl::MutexLock lock(&mutex_);
  if (stop_requested_) return false;
  stop_requested_ = true;

  if (!callbacks_) return true;

  while (callbacks_) {
    StopCallbackBase* cur_callback = callbacks_;

    invocation_state.callback_destroyed = false;

    // Remove the current callback.
    if (internal::intrusive_linked_list::OnlyContainsNode(
            CallbackListAccessor{}, cur_callback)) {
      callbacks_ = nullptr;
    } else {
      callbacks_ = CallbackListAccessor::GetNext(cur_callback);
      internal::intrusive_linked_list::Remove(CallbackListAccessor{},
                                              cur_callback);
    }

    // Mark callback as being invoked.
    cur_callback->next = nullptr;
    cur_callback->invocation_state = &invocation_state;

    // Invoke the callback with the mutex unlocked.
    {
      mutex_.Unlock();
      cur_callback->invoker_(*cur_callback);
      mutex_.Lock();
    }

    if (invocation_state.callback_destroyed) {
      // The callback unregistered itself.  `cur_callback` no longer points to
      // valid data.
      continue;
    }
    if (cur_callback->state_.exchange(nullptr, std::memory_order_acq_rel)) {
      // `Unregister` was not called, and won't be called.  We are responsible
      // for decrementing `ref_count_`. At this point, refcount cannot free the
      // state object because a caller holds a pointer.
      intrusive_ptr_decrement(this);
    }

    // Another thread is waiting in `Unregister` for the callback invocation to
    // complete.  Notify it that the callback has completed.
    cur_callback->invocation_state = nullptr;
  }

  return true;
}

void StopState::UnregisterImpl(StopCallbackBase& callback) {
  {
    absl::MutexLock lock(&mutex_);

    if (callback.next != nullptr) {
      // Callback has not yet been invoked; it's safe to remove.
      if (internal::intrusive_linked_list::OnlyContainsNode(
              CallbackListAccessor{}, &callback)) {
        callbacks_ = nullptr;
      } else {
        if (callbacks_ == &callback) {
          callbacks_ = CallbackListAccessor::GetNext(&callback);
        }
        internal::intrusive_linked_list::Remove(CallbackListAccessor{},
                                                &callback);
      }
    } else if (callback.invocation_state != nullptr) {
      // Callback is currently being invoked.
      if (callback.invocation_state->thread_id == std::this_thread::get_id()) {
        // The callback is being invoked from this thread, i.e. `Unregister` was
        // (transitively) called by the callback itself.
        callback.invocation_state->callback_destroyed = true;
      } else {
        // Callback is being invoked on another thread.  Wait until it
        // completes.
        mutex_.Await(absl::Condition(
            +[](StopCallbackBase* callback) {
              return callback->invocation_state == nullptr;
            },
            &callback));
      }
    }  // else callback has been invoked
  }

  intrusive_ptr_decrement(this);
}

void StopState::RegisterImpl(StopCallbackBase& callback) {
  bool stop_requested;
  {
    absl::MutexLock lock(&mutex_);
    stop_requested = stop_requested_;
    if (!stop_requested) {
      intrusive_ptr_increment(this);
      callback.state_.store(this, std::memory_order_relaxed);

      internal::intrusive_linked_list::Initialize(CallbackListAccessor{},
                                                  &callback);
      if (callbacks_ != nullptr) {
        internal::intrusive_linked_list::InsertBefore(CallbackListAccessor{},
                                                      callbacks_, &callback);
      }
      callbacks_ = &callback;
    }
  }

  /// Stop requested; execute callback immediately.
  if (stop_requested) {
    callback.invoker_(callback);
  }
}

}  // namespace internal_stop_token
}  // namespace tensorstore
