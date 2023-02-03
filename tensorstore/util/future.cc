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

#include "tensorstore/util/future.h"

#include <atomic>
#include <cassert>
#include <new>
#include <thread>  // NOLINT

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/base/optimization.h"
#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tensorstore/internal/intrusive_linked_list.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/metrics/counter.h"
#include "tensorstore/internal/metrics/gauge.h"
#include "tensorstore/internal/no_destructor.h"
#include "tensorstore/util/future_impl.h"

namespace tensorstore {
namespace internal_future {
namespace {

auto& live_futures = internal_metrics::Gauge<int64_t>::New(
    "/tensorstore/futures/live", "Live futures");

#if 1
auto& future_ready_callbacks = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/futures/ready_callbacks", "Ready callbacks");
auto& future_not_needed_callbacks = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/futures/not_needed_callbacks", "Not needed callbacks");
auto& future_force_callbacks = internal_metrics::Counter<int64_t>::New(
    "/tensorstore/futures/force_callbacks", "Force callbacks");
#else
struct MockCounter {
  void Increment() {}
};
static MockCounter future_ready_callbacks;
static MockCounter future_not_needed_callbacks;
static MockCounter future_force_callbacks;
#endif

}  // namespace

/// Special value to which CallbackListNode::next points to indicate that
/// unregistration was requested.
static CallbackListNode unregister_requested;

struct ABSL_CACHELINE_ALIGNED CacheLineAlignedMutex {
  absl::Mutex mutex{absl::kConstInit};
};

constexpr size_t kNumMutexes = 64;

absl::Mutex* GetMutex(FutureStateBase* ptr) {
  ABSL_CONST_INIT static CacheLineAlignedMutex mutexes[kNumMutexes];
  return &mutexes[absl::HashOf(ptr) % kNumMutexes].mutex;
}

using CallbackListAccessor =
    internal::intrusive_linked_list::MemberAccessor<CallbackListNode>;
namespace {

/// Returns a CallbackPointer corresponding to a callback that was never added
/// to the list of callbacks, because the condition under which it should be
/// unregistered was satisfied at the time
/// Register{Future,Promise,NotNeeded}Callback was invoked.
///
/// \param callback Non-null pointer to a callback.  Implicitly, the caller of
///     `MakeUnregisteredCallbackPointer` transfers ownership of 2
///     `callback->reference_count_` references to
///     `MakeUnregisteredCallbackPointer`.
/// \dchecks `callback->reference_count_.load() >= 2`.
/// \returns A reference to `callback`.
CallbackPointer MakeUnregisteredCallbackPointer(CallbackBase* callback) {
  assert(callback->reference_count_.load(std::memory_order_relaxed) >= 2);
  callback->next = callback->prev = callback;
  callback->reference_count_.fetch_sub(1, std::memory_order_relaxed);
  return CallbackPointer(callback, internal::adopt_object_ref);
}

}  // namespace

CallbackPointer FutureStateBase::RegisterReadyCallback(
    ReadyCallbackBase* callback) {
  assert(callback->reference_count_.load(std::memory_order_relaxed) >= 2);
  {
    absl::MutexLock lock(GetMutex(this));
    future_ready_callbacks.Increment();
    if (!this->ready()) {
      InsertBefore(CallbackListAccessor{}, &ready_callbacks_, callback);
      return CallbackPointer(callback, internal::adopt_object_ref);
    }
  }
  callback->OnReady();
  return MakeUnregisteredCallbackPointer(callback);
}

CallbackPointer FutureStateBase::RegisterNotNeededCallback(
    ResultNotNeededCallbackBase* callback) {
  assert(callback->reference_count_.load(std::memory_order_relaxed) >= 2);
  {
    absl::MutexLock lock(GetMutex(this));
    future_not_needed_callbacks.Increment();
    if (result_needed()) {
      InsertBefore(CallbackListAccessor{}, &promise_callbacks_, callback);
      return CallbackPointer(callback, internal::adopt_object_ref);
    }
  }
  callback->OnResultNotNeeded();
  return MakeUnregisteredCallbackPointer(callback);
}

CallbackPointer FutureStateBase::RegisterForceCallback(
    ForceCallbackBase* callback) {
  assert(callback->reference_count_.load(std::memory_order_relaxed) >= 2);
  auto* mutex = GetMutex(this);
  {
    absl::MutexLock lock(mutex);
    future_force_callbacks.Increment();
    const auto state = state_.load(std::memory_order_acquire);
    if ((state & kResultLocked) != 0 || !has_future()) {
      // Handle result-not-needed case after unlocking the mutex.
      goto destroy_callback;
    }
    if (state & kForcing) {
      // Handling already-forced case after unlocking the mutex.
      goto already_forced;
    }
    InsertBefore(CallbackListAccessor{}, &promise_callbacks_, callback);
    return CallbackPointer(callback, internal::adopt_object_ref);
  }

already_forced:
  callback->OnForced();
  if (callback->callback_type() == CallbackBase::kLinkCallback) {
    absl::MutexLock lock(mutex);
    // For a `kLinkCallback` callback, if the result is still needed, register
    // the callback.
    if (result_needed()) {
      InsertBefore(CallbackListAccessor{}, &promise_callbacks_, callback);
      return CallbackPointer(callback, internal::adopt_object_ref);
    }
    // Otherwise, proceed with the destroy_callback case.
  } else {
    // For a `kForceCallback` callback, we don't call `OnUregistered` after
    // OnForced.
    return MakeUnregisteredCallbackPointer(callback);
  }

destroy_callback:
  callback->OnUnregistered();
  return MakeUnregisteredCallbackPointer(callback);
}

CallbackBase::~CallbackBase() {}

void CallbackBase::Unregister(bool block) noexcept {
  auto* shared_state = this->shared_state();
  auto* mutex = GetMutex(shared_state);
  {
    absl::MutexLock lock(mutex);
    if (next == this) {
      // Already unregistered, do nothing.
      return;
    }

    if (next == nullptr || next == &unregister_requested) {
      next = &unregister_requested;
      // Callback is currently being invoked.
      if (!block || running_callback_thread == std::this_thread::get_id()) {
        // The unregistration call was invoked from the callback itself.  Do
        // nothing.
        return;
      }

      // The callback is being invoked in another thread.  Wait for it
      // to finish.
      const auto is_done = [&] { return this->next != &unregister_requested; };
      mutex->Await(absl::Condition(&is_done));
      // Callback has finished running.
      return;
    }

    // Remove callback from list and decrement reference count.
    Remove(CallbackListAccessor{}, this);
    next = this;
  }

  this->OnUnregistered();
  CallbackPointerTraits::decrement(this);
}

FutureStateBase::FutureStateBase()
    : state_(kInitial),
      combined_reference_count_(2),
      promise_reference_count_(1),
      future_reference_count_(1) {
  Initialize(CallbackListAccessor{}, &ready_callbacks_);
  Initialize(CallbackListAccessor{}, &promise_callbacks_);
  live_futures.Increment();
}

namespace {

void NoMorePromiseReferences(FutureStateBase* shared_state) {
  if (shared_state->LockResult()) {
    shared_state->MarkResultWrittenAndCommitResult();
  } else {
    shared_state->CommitResult();
  }
  shared_state->ReleaseCombinedReference();
}

template <typename BeforeUnregisterFunc, typename AfterUnregisterFunc>
inline void RunAndReleaseCallbacks(FutureStateBase* shared_state,
                                   CallbackListNode* head,
                                   BeforeUnregisterFunc before_func,
                                   AfterUnregisterFunc after_func) {
  const auto thread_id = std::this_thread::get_id();
  auto* mutex = GetMutex(shared_state);
  // Pointer to callback that was just run.
  CallbackPointer prev_node;

  while (true) {
    CallbackListNode* next_node;
    {
      absl::MutexLock lock(mutex);
      if (prev_node != nullptr) {
        // Reset prev_node->next pointer, which marks that the callback has
        // finished running.
        using Id = std::thread::id;
        prev_node->running_callback_thread.~Id();
        prev_node->next = prev_node.get();
      }
      // Run callbacks in the order that they were added.
      next_node = head->next;
      if (next_node == head) {
        // No more callbacks to run.
        break;
      }
      Remove(CallbackListAccessor{}, next_node);
      next_node->next = nullptr;
      new (&next_node->running_callback_thread) std::thread::id(thread_id);
    }
    // Call after_func on the previous callback if this is not the first
    // iteration of the loop.
    if (prev_node) after_func(prev_node.get());
    prev_node.reset(static_cast<CallbackBase*>(next_node),
                    internal::adopt_object_ref);
    before_func(prev_node.get());
  }
  // Call after_func on the last callback processed by the loop, if the loop
  // processed at least one callback.
  if (prev_node) after_func(prev_node.get());
}

void RunReadyCallbacks(FutureStateBase* shared_state) {
  RunAndReleaseCallbacks(
      shared_state, &shared_state->ready_callbacks_,
      /*before_func=*/
      [](CallbackBase* callback) {
        static_cast<ReadyCallbackBase*>(callback)->OnReady();
      },
      /*after_func=*/[](CallbackBase* callback) {});
}

void DestroyPromiseCallbacks(FutureStateBase* shared_state) {
  RunAndReleaseCallbacks(
      shared_state, &shared_state->promise_callbacks_,
      /*before_func=*/
      [](CallbackBase* callback) {
        if (callback->callback_type() ==
            CallbackBase::kResultNotNeededCallback) {
          static_cast<ResultNotNeededCallbackBase*>(callback)
              ->OnResultNotNeeded();
        }
      },
      /*after_func=*/
      [](CallbackBase* callback) {
        if (callback->callback_type() !=
            CallbackBase::kResultNotNeededCallback) {
          // Call OnUnregistered after the callback has been marked
          // unregistered to prevent a possible deadlock, since OnUnregistered
          // can itself unregister other callbacks, which may be blocked
          // waiting for this callback to be unregistered by another thread.
          callback->OnUnregistered();
        }
      });
}

void RunForceCallbacks(FutureStateBase* shared_state) {
  const auto thread_id = std::this_thread::get_id();
  auto* mutex = GetMutex(shared_state);

  // Pointer to callback that was just run.
  CallbackPointer prev_node;

  CallbackListNode temp_head;
  CallbackListNode* const head = &shared_state->promise_callbacks_;

  while (true) {
    CallbackListNode* next_node;
    {
      absl::MutexLock lock(mutex);
      if (prev_node) {
        // Reset prev_node->next pointer, which marks that the callback has
        // finished running.
        using Id = std::thread::id;
        if (prev_node->callback_type() == CallbackBase::kLinkCallback) {
          // A `kLinkCallback` remains registered after the call to `OnForced`
          // unless it was explicitly unregistered during the call.
          if (prev_node->next == &unregister_requested) {
            // Notify Unregister that unregistration is complete. This is done
            // before calling OnUnregistered to prevent a possible deadlock,
            // since OnUnregistered can itself unregister other callbacks, which
            // may be blocked waiting for this callback to be unregistered by
            // another thread.
            prev_node->next = prev_node.get();
            mutex->Unlock();
            static_cast<CallbackBase*>(prev_node.get())->OnUnregistered();
            mutex->Lock();
          } else {
            // Add back to promise_callbacks_ list.
            prev_node->running_callback_thread.~Id();
            InsertBefore(CallbackListAccessor{}, head, prev_node.release());
          }
        } else {
          assert(prev_node->callback_type() == CallbackBase::kForceCallback);
          // A `kForceCallback` is always unregistered immediately after calling
          // `OnForced`.
          prev_node->next = prev_node.get();
        }
      } else {
        temp_head.next = head->next;
        temp_head.next->prev = &temp_head;
        temp_head.prev = head->prev;
        temp_head.prev->next = &temp_head;
        head->next = head->prev = head;
        shared_state->state_.fetch_or(FutureStateBase::kForcing);
      }
      // Run callbacks in the order that they were added.
      while (true) {
        next_node = temp_head.next;
        if (next_node == &temp_head) return;
        Remove(CallbackListAccessor{}, next_node);
        if (static_cast<CallbackBase*>(next_node)->callback_type() ==
            CallbackBase::kResultNotNeededCallback) {
          InsertBefore(CallbackListAccessor{}, head, next_node);
          continue;
        }
        next_node->next = nullptr;
        new (&next_node->running_callback_thread) std::thread::id(thread_id);
        break;
      }
    }
    prev_node.reset(static_cast<CallbackBase*>(next_node),
                    internal::adopt_object_ref);
    static_cast<ForceCallbackBase*>(prev_node.get())->OnForced();
  }
}

void NoMoreFutureReferences(FutureStateBase* shared_state) {
  // We can safely destroy the Promise callbacks.  We know that a Force is not
  // in progress because Force can only be called with a Future reference, and
  // by definition there are none when this function is called.
  DestroyPromiseCallbacks(shared_state);
  shared_state->ReleaseCombinedReference();
}

}  // namespace

void FutureStateBase::Force() noexcept {
  StateValue prior_state = kInitial;
  if (!state_.compare_exchange_strong(prior_state, kPreparingToForce)) {
    // Wasn't in `kInitial` state, do nothing.
    return;
  }

  RunForceCallbacks(this);
  prior_state = state_.fetch_or(kForced);
  if (prior_state & kResultLocked) {
    // kResultLocked state was set before forced was set.  It is our
    // responsibility to unregister promise callbacks.
    DestroyPromiseCallbacks(this);
  }
}

void FutureStateBase::ReleaseFutureReference() {
  if (--future_reference_count_ == 0) {
    NoMoreFutureReferences(this);
  }
}

void FutureStateBase::ReleasePromiseReference() {
  if (--promise_reference_count_ == 0) {
    NoMorePromiseReferences(this);
  }
}

void FutureStateBase::ReleaseCombinedReference() {
  if (--combined_reference_count_ == 0) {
    delete this;
  }
}

bool FutureStateBase::AcquireFutureReference() noexcept {
  auto existing = future_reference_count_.load(std::memory_order_relaxed);
  while (true) {
    if (existing == 0) {
      if ((state_.load(std::memory_order_acquire) & kResultLocked) == 0) {
        return false;
      }
      if (future_reference_count_.fetch_add(1, std::memory_order_acq_rel) ==
          0) {
        combined_reference_count_.fetch_add(1, std::memory_order_relaxed);
      }
      return true;
    }
    if (future_reference_count_.compare_exchange_weak(
            existing, existing + 1, std::memory_order_acq_rel)) {
      return true;
    }
  }
}

bool FutureStateBase::LockResult() noexcept {
  const StateValue prior_state = state_.fetch_or(kResultLocked);
  if (prior_state & kResultLocked) return false;
  if ((prior_state & kForced) != 0 || (prior_state & kPreparingToForce) == 0) {
    // Forcing was not in progress when kResultLocked state was set.  It is
    // our responsibility to unregister promise callbacks.
    DestroyPromiseCallbacks(this);
  } else {
    // Forcing is in progress.  Force is responsible for destroying the
    // promise callbacks.
  }
  return true;
}

void FutureStateBase::MarkResultWritten() noexcept {
  const StateValue prior_state = state_.fetch_or(kResultWritten);
  assert(prior_state & kResultLocked);
  assert((prior_state & kResultWritten) == 0);
  if (prior_state & kReady) {
    RunReadyCallbacks(this);
  }
}

bool FutureStateBase::CommitResult() noexcept {
  const StateValue prior_state = state_.fetch_or(kReady);
  if (prior_state & kReady) return false;
  if (prior_state & kResultWritten) {
    RunReadyCallbacks(this);
  }
  return true;
}

void FutureStateBase::MarkResultWrittenAndCommitResult() noexcept {
  [[maybe_unused]] const StateValue prior_state =
      state_.fetch_or(kResultWrittenAndReady);
  assert(prior_state & kResultLocked);
  assert((prior_state & kResultWritten) == 0);
  // MarkResultWrittenAndCommitResult must only be called after `LockResult()`
  // returned true, so at least one of {kReady, kWrittenResult} was set by the
  // thread and it takes ownership of RunReadyCallbacks(this).
  RunReadyCallbacks(this);
}

bool FutureStateBase::WaitFor(absl::Duration duration) noexcept {
  if (ready()) return true;
  Force();
  absl::Mutex* mutex = GetMutex(this);
  bool is_ready = mutex->LockWhenWithTimeout(
      absl::Condition(this, &FutureStateBase::ready), duration);
  mutex->Unlock();
  return is_ready;
}

bool FutureStateBase::WaitUntil(absl::Time deadline) noexcept {
  if (ready()) return true;
  Force();
  absl::Mutex* mutex = GetMutex(this);
  bool is_ready = mutex->LockWhenWithDeadline(
      absl::Condition(this, &FutureStateBase::ready), deadline);
  mutex->Unlock();
  return is_ready;
}

void FutureStateBase::Wait() noexcept {
  if (ready()) return;
  Force();
  absl::Mutex* mutex = GetMutex(this);
  mutex->LockWhen(absl::Condition(this, &FutureStateBase::ready));
  mutex->Unlock();
}

FutureStateBase::~FutureStateBase() {
  assert(promise_callbacks_.next == &promise_callbacks_);
  assert(ready_callbacks_.next == &ready_callbacks_);
  live_futures.Decrement();
}

}  // namespace internal_future

ReadyFuture<const void> MakeReadyFuture() {
  static internal::NoDestructor<ReadyFuture<const void>> future{
      MakeReadyFuture<void>(MakeResult())};
  return *future;
}

Future<void> WaitAllFuture(tensorstore::span<const AnyFuture> futures) {
  auto& f = futures;
  switch (f.size()) {
    case 0:
      return MakeReadyFuture<void>(absl::OkStatus());
    case 1:
      return PromiseFuturePair<void>::LinkError(absl::OkStatus(), f[0]).future;
    case 2:
      return PromiseFuturePair<void>::LinkError(absl::OkStatus(), f[0], f[1])
          .future;
    case 3:
      return PromiseFuturePair<void>::LinkError(absl::OkStatus(), f[0], f[1],
                                                f[2])
          .future;
    case 4:
      return PromiseFuturePair<void>::LinkError(absl::OkStatus(), f[0], f[1],
                                                f[2], f[3])
          .future;
    case 5:
      return PromiseFuturePair<void>::LinkError(absl::OkStatus(), f[0], f[1],
                                                f[2], f[3], f[4])
          .future;
    case 6:
      return PromiseFuturePair<void>::LinkError(absl::OkStatus(), f[0], f[1],
                                                f[2], f[3], f[4], f[5])
          .future;
    case 7:
      return PromiseFuturePair<void>::LinkError(absl::OkStatus(), f[0], f[1],
                                                f[2], f[3], f[4], f[5], f[6])
          .future;
    default:
      break;
  }

  // 8 or more...
  auto [promise, result] = PromiseFuturePair<void>::LinkError(
      absl::OkStatus(), f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]);
  f = f.subspan(8);
  while (f.size() > 8) {
    LinkError(promise, f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]);
    f = f.subspan(8);
  }

  switch (f.size()) {
    case 0:
      return std::move(result);
    case 1:
      LinkError(std::move(promise), f[0]);
      return std::move(result);
    case 2:
      LinkError(std::move(promise), f[0], f[1]);
      return std::move(result);
    case 3:
      LinkError(std::move(promise), f[0], f[1], f[2]);
      return std::move(result);
    case 4:
      LinkError(std::move(promise), f[0], f[1], f[2], f[3]);
      return std::move(result);
    case 5:
      LinkError(std::move(promise), f[0], f[1], f[2], f[3], f[4]);
      return std::move(result);
    case 6:
      LinkError(std::move(promise), f[0], f[1], f[2], f[3], f[4], f[5]);
      return std::move(result);
    case 7:
      LinkError(std::move(promise), f[0], f[1], f[2], f[3], f[4], f[5], f[6]);
      return std::move(result);
    case 8:
      LinkError(std::move(promise), f[0], f[1], f[2], f[3], f[4], f[5], f[6],
                f[7]);
      return std::move(result);
  }
  ABSL_UNREACHABLE();  // COV_NF_LINE
}

}  // namespace tensorstore
