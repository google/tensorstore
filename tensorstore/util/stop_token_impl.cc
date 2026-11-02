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

#include <stdint.h>

#include <atomic>
#include <cassert>
#include <thread>

#include "absl/synchronization/mutex.h"
#include "tensorstore/internal/container/intrusive_linked_list.h"
#include "tensorstore/internal/intrusive_ptr.h"

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
  // Fast path: `stop_requested_` never reverts to `false`, so a stop that has
  // already been requested can be detected without acquiring `mutex_`.
  if (stop_requested_.load(std::memory_order_acquire)) return false;

  StopCallbackInvocationState invocation_state{
      /*thread_id=*/std::this_thread::get_id(), /*callback_destroyed=*/false};

  absl::MutexLock lock(mutex_);
  if (stop_requested_.load(std::memory_order_relaxed)) return false;
  stop_requested_.store(true, std::memory_order_release);

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
      mutex_.unlock();
      cur_callback->invoker_(*cur_callback);
      mutex_.lock();
    }

    if (invocation_state.callback_destroyed) {
      // The callback unregistered itself.  `cur_callback` no longer points to
      // valid data.
      continue;
    }
    if (cur_callback->state_.exchange(nullptr, std::memory_order_acq_rel)) {
      // `Unregister` was not called, and won't be called.  We are responsible
      // for releasing the reference owned by `state_`. At this point, refcount
      // cannot free the state object because a caller holds a pointer.
      //
      // Once the exchange above completes, `~StopCallback` may run on another
      // thread without acquiring `mutex_` and free `cur_callback`, so it must
      // not be accessed again.  No thread can be waiting in `Unregister`,
      // since that requires winning the exchange.
      intrusive_ptr_decrement(this);
      continue;
    }

    // Another thread won the exchange in `~StopCallback` and is (or will be)
    // in `Unregister`, which cannot return until it acquires `mutex_`; hence
    // `cur_callback` remains valid here.  Notify it that the callback has
    // completed.
    cur_callback->invocation_state = nullptr;
  }

  return true;
}

void StopState::UnregisterImpl(StopCallbackBase& callback) {
  {
    absl::MutexLock lock(mutex_);

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

void StopState::RegisterImpl(StopCallbackBase& callback, bool adopt_reference) {
  // Fast path: `stop_requested_` never reverts to `false`, so once a stop has
  // been requested the callback can be invoked without acquiring `mutex_`.
  if (!stop_requested_.load(std::memory_order_acquire)) {
    absl::MutexLock lock(mutex_);
    if (!stop_requested_.load(std::memory_order_relaxed)) {
      // `callback.state_` owns a reference, released by either
      // `UnregisterImpl` or `RequestStop`.
      if (!adopt_reference) intrusive_ptr_increment(this);
      callback.state_.store(this, std::memory_order_relaxed);

      internal::intrusive_linked_list::Initialize(CallbackListAccessor{},
                                                  &callback);
      if (callbacks_ != nullptr) {
        internal::intrusive_linked_list::InsertBefore(CallbackListAccessor{},
                                                      callbacks_, &callback);
      }
      callbacks_ = &callback;
      return;
    }
  }

  // Stop requested; execute callback immediately.  The callback is never
  // registered, so release the adopted reference, if any.
  callback.invoker_(callback);
  if (adopt_reference) intrusive_ptr_decrement(this);
}

}  // namespace internal_stop_token
}  // namespace tensorstore
