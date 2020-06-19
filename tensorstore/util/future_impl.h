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

#ifndef TENSORSTORE_UTIL_FUTURE_IMPL_H_
#define TENSORSTORE_UTIL_FUTURE_IMPL_H_

// IWYU pragma: private, include "third_party/tensorstore/util/future.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <thread>  // NOLINT
#include <type_traits>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/macros.h"
#include "absl/meta/type_traits.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/utility/utility.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/tagged_ptr.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace tensorstore {

template <typename T>
class [[nodiscard]] Future;
template <typename T>
class [[nodiscard]] Promise;
template <typename T>
class [[nodiscard]] ReadyFuture;

class FutureCallbackRegistration;

namespace internal_future {

template <typename T>
class FutureState;

class FutureStateBase;
class CallbackBase;
class ReadyCallbackBase;
class ForceCallbackBase;
class ResultNotNeededCallbackBase;

struct CallbackPointerTraits;
using CallbackPointer =
    internal::IntrusivePtr<CallbackBase, CallbackPointerTraits>;

/// FutureAccess mediates access to internal implementation details for
/// some parts of Future/Promise which allows constructing and converting
/// Future/Promise data.
class FutureAccess {
 public:
  template <typename T, typename... U>
  static T Construct(U&&... u) {
    return T(std::forward<U>(u)...);
  }

  template <typename T>
  static auto rep(T&& x) -> decltype(std::declval<T>().rep()) {
    return std::forward<T>(x).rep();
  }

  template <typename T>
  static auto rep_pointer(T&& x) -> decltype((std::declval<T>().rep_)) {
    return std::forward<T>(x).rep_;
  }
};

/// Returns a pointer to the Mutex guarding the callback lists for `ptr`.
///
/// The mutex isn't stored as a member of FutureStateBase, because a
/// CallbackBase may outlast the FutureStateBase.
absl::Mutex* GetMutex(FutureStateBase* ptr);

/// Base class representing an element of a doubly-linked list of callbacks.
///
/// In addition to representing an element of a callback list, this type is also
/// used to represent the special head node.
///
/// The `next`, `prev`, and `running_callback_thread` fields are all protected
/// by the mutex used for the associated FutureStateBase.
struct CallbackListNode {
  CallbackListNode() noexcept {}

  /// If `next` to points to `this`, then this callback has already been
  /// unregistered.
  ///
  /// If `next` equals `nullptr` or `&unregister_requested`, then the callback
  /// is currently being invoked, `running_callback_thread` is the active member
  /// of the union, and `prev` cannot be accessed.
  ///
  /// Otherwise, `next` and `prev` behave normally for a doubly-linked list.
  CallbackListNode* next;

  union {
    /// Only valid if `next` does not equal `nullptr` or
    /// `&unregister_requested`.
    CallbackListNode* prev;

    /// While this callback is being invoked, this is the thread in which it is
    /// invoked.  Only valid if `next` is `nullptr` or `&unregister_requested`.
    std::thread::id running_callback_thread;
  };
};

/// The shared state base class of a Promise/Future.
///
/// This object owns the collection of callbacks associated with a future
/// as well as the state machine representing the operational flow
/// of a Promise/Future.
///
/// \remarks Instances of this class must not be constructed directly.  Instead,
///     an instance of the `FutureState` class template should be constructed.
class FutureStateBase {
 public:
  FutureStateBase();
  virtual ~FutureStateBase();

  /// Registers a ready callback.
  ///
  /// The caller of `RegisterReadyCallback` implicitly transfers 1
  /// `future_reference_count_` to `RegisterReadyCallback`.
  ///
  /// \param callback Non-null pointer to the callback to register.  The caller
  ///     of `RegisterReadyCallback` implicitly transfers 2
  ///     `callback->reference_count_` reference counts to
  ///     RegisterReadyCallback.
  /// \dchecks `callback->reference_count_.load() >= 2`.
  /// \returns A CallbackPointer that references `callback`.
  /// \threadsafety Thread safe.
  /// \pre `GetMutex(this)` is not held by the current thread.
  CallbackPointer RegisterReadyCallback(ReadyCallbackBase* callback);

  /// Registers a force callback, which may either be `kForceCallback` or a
  /// `kLinkCallback`.
  ///
  /// The caller of `RegisterForceCallback` implicitly transfers 1
  /// `promise_reference_count_` to `RegisterForceCallback`.
  ///
  /// \param callback Non-null pointer to the callback to register.  The caller
  ///     of `RegisterForceCallback` implicitly transfers 2
  ///     `callback->reference_count_` reference counts to
  ///     `RegisterForceCallback`.
  /// \dchecks `callback->reference_count_.load() >= 2`.
  /// \returns A CallbackPointer that references `callback`.
  /// \threadsafety Thread safe.
  /// \pre `GetMutex(this)` is not held by the current thread.
  CallbackPointer RegisterForceCallback(ForceCallbackBase* callback);

  /// Registers a result-not-needed callback.
  ///
  /// Unlike for `RegisterPromiseCallback` and `RegisterForceCallback`, the
  /// caller of `RegisterNotNeededCallback` does not transfer any promise
  /// references.
  ///
  /// \param callback Non-null pointer to the callback to register.  The caller
  ///     of `RegisterNotNeededCallback` implicitly transfers 2
  ///     `callback->reference_count_` reference counts to
  ///     `RegisterNotNeededCallback`.
  /// \dchecks `callback->reference_count_.load() >= 2`.
  /// \returns A CallbackPointer that references `callback`.
  /// \threadsafety Thread safe.
  /// \pre `GetMutex(this)` is not held by the current thread.
  CallbackPointer RegisterNotNeededCallback(
      ResultNotNeededCallbackBase* callback);

  /// Attempts to acquire an exclusive lock for setting the result (as indicated
  /// by the `kResultLocked` bit in `state_`).
  ///
  /// The first call to this method returns `true`, all subsequent calls return
  /// `false`.
  ///
  /// This unregisters all promise callbacks, and prevents any new promise
  /// callbacks from being registered.
  ///
  /// \threadsafety Thread safe.
  /// \post `result_needed() == false`.
  /// \returns `true` if the lock was acquired.
  /// \remarks The result is not actually considered ready until
  ///     `CommitResult()` is called.
  bool LockResult() noexcept;

  /// Release the exclusive lock used for setting the result (as indicated by
  /// the `kReady` bit in `state_`).
  ///
  /// The first time this function is called, the future callbacks are invoked.
  /// Subsequent calls have no effect.
  ///
  /// \dchecks `LockResult()` must have been called previously.
  /// \post `ready() == true`.
  /// \pre `GetMutex(this)` is not held by the current thread.
  /// \returns `true` if this was the first call to this function.
  bool CommitResult() noexcept;

  /// Marks the result as ready without modifying it.
  bool SetReady() noexcept {
    if (!this->LockResult()) return false;
    this->CommitResult();
    return true;
  }

  /// If `state_` is `kInitial`, atomically changes the state to
  /// `kPreparingToForce` and then begins the Force operation, which invokes the
  /// promise callbacks.
  ///
  /// \threadsafety Thread safe.
  /// \pre `GetMutex(this)` is not held by the current thread.
  void Force() noexcept;

  /// Waits for the result to be ready.
  /// \post `ready()`
  void Wait() noexcept;

  /// Waits until the specified time for the result to be ready.
  /// \returns `ready()`.
  bool WaitUntil(absl::Time deadline) noexcept;

  /// Waits for up to the specified duration for the result to be ready.
  /// \returns `ready()`.
  bool WaitFor(absl::Duration duration) noexcept;

  /// Returns true if the result is ready.
  ///
  /// \threadsafety Thread safe.
  bool ready() const noexcept {
    return (state_.load(std::memory_order_acquire) & kReady) != 0;
  }

  /// Returns `true` if there are any remaining `Future` objects or ready
  /// callbacks, and setting of the result has not yet been started.
  ///
  /// \threadsafety Thread safe.
  bool result_needed() const noexcept {
    return (state_.load(std::memory_order_acquire) & kResultLocked) == 0 &&
           has_future();
  }

  /// Returns `true` if there are any remaining `Future` objects referencing
  /// this shared state.
  bool has_future() const noexcept {
    return future_reference_count_.load(std::memory_order_acquire) != 0;
  }

  /// \pre `GetMutex(this)` is not held by the current thread.
  void ReleaseFutureReference();

  /// \pre `GetMutex(this)` is not held by the current thread.
  void ReleasePromiseReference();

  void ReleaseCombinedReference();

  /// Attempts to acquire an additional future reference from assuming that a
  /// promise reference is held.
  ///
  /// This is only possible if it will not affect the value of
  /// `result_needed()`, as `result_needed()` must not transition from `false`
  /// to `true`.
  ///
  /// If `future_reference_count_ > 0`, atomically increments it and returns
  /// `true`.
  ///
  /// Otherwise if `future_reference_count_` is equal to `0`:
  ///
  ///   if `(state_ & kResultLocked) != 0`, atomically increments both
  ///   `future_reference_count_` and `combined_reference_count_`, and returns
  ///   `true`.
  ///
  ///   if `(state_ & kResultLocked) == 0`, returns `false`.
  ///
  /// \returns `true` on success, `false` on failure (i.e. `result_needed()`
  ///     would be affected by the additional future reference).
  bool AcquireFutureReference() noexcept;

  /// Callbacks to be invoked when `ready()` becomes `true`.  These callbacks
  /// must inherit from ReadyCallbackBase.  Once the state is ready, this list
  /// will remain empty.
  CallbackListNode ready_callbacks_;

  /// Callbacks to be invoked when the `Force` is called, and to be destroyed
  /// when `result_needed()` becomes `false`.  These callbacks must inherit from
  /// ForceCallbackBase.
  CallbackListNode promise_callbacks_;

  /// State bitvector representation.  State bits (corresponding to the
  /// following constants) are added to but never removed from the state.
  using StateValue = std::uint32_t;

  /// Initial state, neither LockResult nor Force have been called.
  constexpr static StateValue kInitial = 0;

  /// Force called.  Subsequent calls to Force are ignored, but new promise
  /// callbacks will still be invoked.  This state can only be transitioned into
  /// from `kInitial`.
  constexpr static uint32_t kPreparingToForce = 1;

  /// The Force operation is in progress (Force callbacks are being invoked).
  /// The list of callbacks to invoke has been "snapshotted".  New promise
  /// callbacks added once this bit has been set will not be invoked by the
  /// Force operation.  Therefore, they should be invoked prior to being
  /// registered.  Promise callbacks that have not yet been invoked may still be
  /// unregistered, however.  If `kResultLocked` is set before `kForced` is set,
  /// then the Force operation itself is responsible for destroying the promise
  /// callbacks after it completes.
  constexpr static uint32_t kForcing = 2;

  /// The Force operation is complete.
  constexpr static uint32_t kForced = 4;

  /// A thread has acquired an exclusive lock for setting the result.
  /// Subsequent attempts to set the result will be ignored.  Additionally,
  /// calls to Force will also be ignored, though it is possible for a prior
  /// Force call to still be in progress.  If `kResultLocked` is set while a
  /// Force operation is in progress (meaning `kPreparingToForce` is set but
  /// `kForced` is not set), the promise callbacks are not destroyed; instead,
  /// the `Force` operation is responsible for destroying them.  Otherwise, the
  /// thread that sets the result is responsible for destroying the promise
  /// callbacks.
  constexpr static uint32_t kResultLocked = 8;

  /// The result has been set.  The thread that sets kReady is responsible for
  /// running the future callbacks.
  constexpr static uint32_t kReady = 16;

  /// Contains one of the state values.
  std::atomic<uint32_t> state_;

  // !!promise_reference_count_ + !!future_reference_count_
  std::atomic<uint32_t> combined_reference_count_;

  // Promise-only reference count.
  std::atomic<uint32_t> promise_reference_count_;

  // Future-only reference count.
  std::atomic<uint32_t> future_reference_count_;
};

struct FuturePointerTraits {
  template <typename>
  using pointer = FutureStateBase*;

  static void increment(FutureStateBase* pointer) {
    ++pointer->future_reference_count_;
  }

  static void decrement(FutureStateBase* pointer) {
    pointer->ReleaseFutureReference();
  }
};

struct PromisePointerTraits {
  template <typename>
  using pointer = FutureStateBase*;

  static void increment(FutureStateBase* pointer) {
    ++pointer->promise_reference_count_;
  }

  static void decrement(FutureStateBase* pointer) {
    pointer->ReleasePromiseReference();
  }
};

using FutureStatePointer =
    internal::IntrusivePtr<FutureStateBase, FuturePointerTraits>;
using PromiseStatePointer =
    internal::IntrusivePtr<FutureStateBase, PromisePointerTraits>;

/// Base class representing a registered callback in the
/// FutureStateBase::ready_callbacks_ or FutureStateBase::promise_callbacks_
/// list.
class CallbackBase : public CallbackListNode {
 public:
  /// Stores a pointer to the FutureStateBase along with tag bits that specify
  /// the type of callback.
  using SharedStatePointer = internal::TaggedPtr<FutureStateBase, 2>;

  /// Callback type indicators, stored in the tag bits of `SharedStatePointer`.
  constexpr static std::uintptr_t
      /// Inherits from `ReadyCallbackBase`.
      kReadyCallback = 0,
      /// Inherits from `ForceCallbackBase`.  When the promise is forced,
      /// `OnForce` is called and then the callback is automatically
      /// unregistered.
      kForceCallback = 1,
      /// Inherits from `ResultNotNeededCallbackBase`.
      kResultNotNeededCallback = 2,
      /// Inherits from `ForceCallbackBase`.  When the promise is forced,
      /// `OnForce` is called.  The callback is automatically unregistered when
      /// the result is not needed.  This type of callback is attached to the
      /// promise associated with a FutureLink, and is similar to a combined
      /// kForceCallback and kResultNotNeededCallback.
      kLinkCallback = 3;

  explicit CallbackBase(SharedStatePointer shared_state)
      : shared_state_(shared_state), reference_count_(2) {}

  virtual ~CallbackBase();

  /// Called at most once when the callback is unregistered.
  ///
  /// It is always called when `Unregister` is called while the callback is
  /// still registered.  Depending on the callback type, it may also be called
  /// in other cases:
  ///
  /// 1. For force callbacks, also called when the result is not needed and
  ///    Force has not yet been called.
  ///
  /// 2. For link callbacks, called in all cases when the callback is
  ///    unregistered.
  virtual void OnUnregistered() noexcept = 0;

  /// Called once there are no more references to this registration, and it has
  /// been unregistered.
  ///
  /// This method will be called exactly once.
  ///
  /// `this` may not be valid after calling this method.
  virtual void DestroyCallback() noexcept = 0;

  /// Unregisters the callback if it is currently registered.
  ///
  /// \param block If `true`, if the callback is being invoked concurrently by
  ///     another thread, block until it is done executing.  This can cause
  ///     deadlock.
  void Unregister(bool block) noexcept;

  /// Returns a pointer to the FutureStateBase with which this registration is
  /// associated.
  ///
  /// This pointer is only valid if `this->next != this`.
  FutureStateBase* shared_state() { return shared_state_.get(); }
  std::uintptr_t callback_type() { return shared_state_.tag(); }

  SharedStatePointer shared_state_;

  /// Number of references to this callback.  The FutureStateBase object itself
  /// owns a reference while this callback is registered.  Each
  /// FutureCallbackRegistration handle also owns a reference.  This object is
  /// destroyed once the reference count reaches zero.  However, the callback
  /// function object itself (contained in a derived class) may be destroyed
  /// before this object itself is destroyed.
  std::atomic<std::size_t> reference_count_;
};

struct CallbackPointerTraits {
  template <typename>
  using pointer = CallbackBase*;

  static void increment(CallbackBase* r) noexcept { ++r->reference_count_; }

  static void decrement(CallbackBase* r) noexcept {
    if (--r->reference_count_ == 0) {
      r->DestroyCallback();
    }
  }
};

/// Base class representing a callback added to the
/// FutureStateBase::ready_callbacks_ list to be invoked when the future state
/// becomes ready.
///
/// Owns a reference to the future to which it is registered.
class ReadyCallbackBase : public CallbackBase {
 public:
  explicit ReadyCallbackBase(FutureStateBase* shared_state)
      : CallbackBase({shared_state, CallbackBase::kReadyCallback}) {}

  /// If the future becomes ready while this callback is registered, `OnReady()`
  /// is called and then the callback is immediately unregistered automatically.
  ///
  /// If the callback is explicitly unregistered before the future becomes
  /// ready, `OnUnregistered` is called instead.
  virtual void OnReady() noexcept = 0;

  /// Transfers the future reference owned by this callback to the returned
  /// FutureStatePointer.  Must be called exactly once.
  FutureStatePointer TakeStatePointer() {
    return FutureStatePointer(this->shared_state(), internal::adopt_object_ref);
  }
};

/// Base class for ExecuteWhenForced callbacks added to the
/// FutureStateBase::promise_callbacks_ list.
///
/// Owns a reference to the promise to which it is registered.
class ForceCallbackBase : public CallbackBase {
 public:
  using CallbackBase::CallbackBase;

  /// Called at most once, when the promise is forced.
  ///
  /// If `this->callback_type()` is `CallbaseBase::kForceCallback`, then the
  /// callback is unregistered immediately after `OnForced` is called. and
  /// `OnUnregistered` is called if, and only if, the promise is not forced
  /// before the promise result is not needed or the callback is unregistered.
  ///
  /// If `this->callback_type()` is `CallbackBase::kLinkCallback`, then
  /// `OnUnregistered` is always called before the callback is unregistered, but
  /// unregistering the callback is deferred until the promise result is not
  /// needed or the callback is explicitly unregistered.
  virtual void OnForced() noexcept = 0;

  /// Transfers the promise reference owned by this callback to the returned
  /// PromiseStatePointer.  Must be called exactly once.
  PromiseStatePointer TakeStatePointer() {
    return PromiseStatePointer(this->shared_state(),
                               internal::adopt_object_ref);
  }
};

/// Base class for ExecuteWhenNotNeeded callbacks added to the
/// FutureStateBase::promise_callbacks_ list.
///
/// Does not own a reference to the promise to which it is registered.
class ResultNotNeededCallbackBase : public CallbackBase {
 public:
  explicit ResultNotNeededCallbackBase(FutureStateBase* shared_state)
      : CallbackBase({shared_state, CallbackBase::kResultNotNeededCallback}) {}
  virtual void OnResultNotNeeded() noexcept = 0;
};

/// Implements a future callback for use with ExecuteWhenReady.
/// \tparam T The value type of the future.
/// \tparam Callback Type of unary function object called with a
///     `ReadyFuture<T>` when the future becomes ready.
template <typename T, typename Callback>
class ReadyCallback final : public ReadyCallbackBase {
 public:
  template <typename U>
  explicit ReadyCallback(FutureStateBase* state, U&& u)
      : ReadyCallbackBase(state), callback_(std::forward<U>(u)) {}

  void OnReady() noexcept override {
    std::move(callback_)(
        FutureAccess::Construct<ReadyFuture<T>>(TakeStatePointer()));
    callback_.~Callback();
  }

  void OnUnregistered() noexcept override {
    TakeStatePointer();
    callback_.~Callback();
  }
  void DestroyCallback() noexcept override { delete this; }

  ~ReadyCallback() override {}

 private:
  // We store the Callback in a union in order to disable the automatic
  // invocation of the constructor and destructor.
  union {
    Callback callback_;
  };
};

/// Implements a promise callback for use with ExecuteWhenForced.
/// \tparam T The value type of the promise.
/// \tparam Callback Type of unary function object called with a `Promise<T>`
///     when the promise is forced.
template <typename T, typename Callback>
class ForceCallback final : public ForceCallbackBase {
 public:
  template <typename U>
  explicit ForceCallback(FutureStateBase* state, U&& u)
      : ForceCallbackBase({state, CallbackBase::kForceCallback}),
        callback_(std::forward<U>(u)) {}

  void OnForced() noexcept override {
    std::move(callback_)(
        FutureAccess::Construct<Promise<T>>(TakeStatePointer()));
    callback_.~Callback();
  }

  void OnUnregistered() noexcept override {
    TakeStatePointer();
    callback_.~Callback();
  }
  void DestroyCallback() noexcept override { delete this; }

  ~ForceCallback() override {}

 private:
  // We store the Callback in a union in order to disable the automatic
  // invocation of the constructor and destructor.
  union {
    Callback callback_;
  };
};

/// Implements a promise callback for use with ExecuteWhenNotNeeded.
/// \tparam Callback Type of nullary function object called when the promise
///     result is not needed.
template <typename Callback>
struct ResultNotNeededCallback final : public ResultNotNeededCallbackBase {
 public:
  template <typename U>
  explicit ResultNotNeededCallback(FutureStateBase* state, U&& u)
      : ResultNotNeededCallbackBase(state), callback_(std::forward<U>(u)) {}

  void OnResultNotNeeded() noexcept override {
    std::move(callback_)();
    callback_.~Callback();
  }
  void OnUnregistered() noexcept override { callback_.~Callback(); }
  void DestroyCallback() noexcept override { delete this; }

  ~ResultNotNeededCallback() override {}

 private:
  // We store the callback in a union so that we can control its construction
  // and destruction.
  union {
    Callback callback_;
  };
};

/// Class template that extends `FutureStateBase` with a concrete value type
/// `T`.
template <typename T>
class FutureState : public FutureStateBase {
 public:
  template <typename... Args>
  bool SetResult(Args&&... args) noexcept {
    if (!this->LockResult()) return false;
    result.Construct(std::forward<Args>(args)...);
    // FIXME: Handle exceptions thrown by `Construct`.
    this->CommitResult();
    return true;
  }

  explicit FutureState() : result{absl::UnknownError("")} {}

  template <typename... Args>
  explicit FutureState(Args&&... args) : result(std::forward<Args>(args)...) {}

  ~FutureState() override {}

  Result<T> result;
};

template <typename T>
using FutureStateType = FutureState<absl::remove_const_t<T>>;

template <typename T>
using ResultType = internal::CopyQualifiers<T, Result<absl::remove_const_t<T>>>;

/// A FutureLinkReadyCallback is created for each future associated with a
/// FutureLink, and is contained (as a base class) within the FutureLink.  It is
/// registered as an ExecuteWhenReady callback in order to trigger the
/// FutureLink::OnFutureReady method when the future becomes ready (which
/// ultimately leads to the user-specified callback being invoked, or an error
/// propagating the promise associated with the FutureLink).
///
/// \tparam LinkType The instance of the FutureLink class template that inherits
///     from this type.
/// \tparam T The value type of the Future to which this callback is bound.
/// \tparam I Unique identifier within the FutureLink.
template <typename LinkType, typename T, std::size_t I>
class FutureLinkReadyCallback : public ReadyCallbackBase {
 public:
  using SharedState = FutureStateType<T>;

  explicit FutureLinkReadyCallback(FutureStateBase* shared_state)
      : ReadyCallbackBase(shared_state) {
  }

  void OnReady() noexcept override {
    GetParent()->OnFutureReady(shared_state());
  }
  void OnUnregistered() noexcept override { GetParent()->EnsureCancelled(); }
  void DestroyCallback() noexcept override {
    GetParent()->MarkLinkCallbackDestroyed(/*promise=*/false);
  }
  LinkType* GetParent() { return static_cast<LinkType*>(this); }
  SharedState* shared_state() {
    return static_cast<SharedState*>(CallbackBase::shared_state());
  }
};

/// A FutureLinkForceCallback is contained (as a base class) within a FutureLink
/// and registered as a ExecuteWhenForced/ExecuteWhenNotNeeded callback on the
/// promise associated with the FutureLink.
///
/// As an ExecuteWhenForced callback, it serves to force all of the associated
/// futures when the promise is forced.  As an ExecuteWhenNotNeeded callback, it
/// serves to unregister the FutureLink when the promise result is no longer
/// needed.
///
/// \tparam LinkType The instance of the FutureLink class template that inherits
///     from this type.
/// \tparam T The value type of the Promise to which this callback is bound.
template <typename LinkType, typename T>
class FutureLinkForceCallback : public ForceCallbackBase {
 public:
  using SharedState = FutureStateType<T>;
  explicit FutureLinkForceCallback(FutureStateBase* shared_state)
      : ForceCallbackBase({shared_state, CallbackBase::kLinkCallback}) {}

  void OnForced() noexcept override { GetParent()->OnPromiseForced(); }
  void OnUnregistered() noexcept override {
    GetParent()->OnForceCallbackUnregistered();
  }
  void DestroyCallback() noexcept override {
    GetParent()->MarkLinkCallbackDestroyed(/*promise=*/true);
  }
  LinkType* GetParent() { return static_cast<LinkType*>(this); }
  SharedState* shared_state() {
    return static_cast<SharedState*>(CallbackBase::shared_state());
  }
};

/// Base class for FutureLink that holds an atomic bitvector used to track the
/// state of the FutureLink.
class FutureLinkBase {
 public:
  /// LinkState is a bit vector type for the atomic `link_state_` field that
  /// packs together several different values, described below:
  using LinkState = std::uint32_t;

  /// Maximum number of futures (number of futures in the FutureLink parameter
  /// pack) supported, due to the use of a 14-bit reference count within
  /// `LinkState`.  This is certainly well above any practical limit imposed by
  /// the compiler.
  constexpr static LinkState kMaxNumFutures = (1 << 14);

  /// Bit 0: Set if at least one ready callback has been unregistered
  /// without being invoked.
  constexpr static LinkState kCancelled = 1;

  /// Bit 1: Set after the FutureLinkForceCallback has been registered.  This
  /// needs to be tracked because futures may become ready before the FutureLink
  /// constructor has completed.
  constexpr static LinkState kRegistered = 2;

  /// Bit 2: Set to 1 to indicate that FutureLinkForceCallback::DestroyCallback
  /// has not been called.
  constexpr static LinkState kLiveForceCallbackMultiplier = 1 << 2;

  /// Bits [3, 16]: Number of FutureLinkReadyCallback bases for which
  /// FutureLinkReadyCallback::DestroyCallback has not been called.
  constexpr static LinkState kLiveReadyCallbackMultiplier = 1 << 3;

  /// Bits [2, 16]: Bits that must be zero to indicate that all
  /// FutureLinkForceCallback and FutureLinkReadyCallback objects have had their
  /// DestroyCallback method called.
  constexpr static LinkState kLiveCallbackMask =
      (kMaxNumFutures - 1) * kLiveReadyCallbackMultiplier +
      kLiveForceCallbackMultiplier;

  /// Bits [17, 30]: Number of futures that are not ready.
  constexpr static std::uint32_t kNotReadyFutureMultiplier = 1 << 17;
  constexpr static std::uint32_t kNotReadyFutureMask =
      (kMaxNumFutures - 1) * kNotReadyFutureMultiplier;

  explicit constexpr FutureLinkBase(std::size_t num_futures)
      : link_state_(
            // Initially, no DestroyCallback methods have been called, and no
            // futures are ready.
            kLiveForceCallbackMultiplier +
            (kLiveReadyCallbackMultiplier + kNotReadyFutureMultiplier) *
                num_futures) {}

  /// Marks the FutureLink as cancelled (meaning the callback won't be invoked).
  /// Returns the previous value of `link_state_`, which can be used to
  /// determine whether the FutureLink was already cancelled (in which case
  /// nothing should be done).
  LinkState MarkCancelled() { return link_state_.fetch_or(kCancelled); }

  /// Releases a reference to the FutureLink owned by the
  /// FutureLinkForceCallback (if `promise == true`) or by a
  /// FutureLinkReadyCallback (if `promise == false`).
  ///
  /// \returns `true` if the FutureLink should be destroyed.
  bool ReleaseLinkReferenceFromCallback(bool promise) {
    const LinkState new_count =
        (link_state_ -= (promise ? kLiveForceCallbackMultiplier
                                 : kLiveReadyCallbackMultiplier));
    if ((new_count & kLiveCallbackMask) == 0) {
      // The FutureLink must either have completed successfully, or been
      // cancelled.
      ABSL_ASSERT((new_count & kNotReadyFutureMask) == 0 ||
                  (new_count & kCancelled));
      return true;
    }
    return false;
  }

  std::atomic<LinkState> link_state_;
};

/// A class satisfies the FutureLinkPolicy concept if it defines:
///
///     template <typename T, typename U>
///     static bool OnFutureReady(FutureState<T>* future_state,
///                               FutureState<U>* promise_state);
///
/// may optionally mark the promise_state ready, and returns `true` if the link
/// should not be cancelled.

/// Policy that causes the callback to be invoked when all the linked futures
/// become ready.
struct FutureLinkAllReadyPolicy {
  static bool OnFutureReady(void* future_state, void* promise_state) {
    return true;
  }
};

/// Policy that causes the callback to be invoked when all the linked futures
/// become ready in a success state.  If any future becomes ready in an error
/// state, the promise result is set to the error and the FutureLink is
/// unregistered.
struct FutureLinkPropagateFirstErrorPolicy {
  template <typename FutureValue, typename PromiseValue>
  static bool OnFutureReady(FutureState<FutureValue>* future_state,
                            FutureState<PromiseValue>* promise_state) {
    if (future_state->result) return true;
    promise_state->SetResult(tensorstore::GetStatus(future_state->result));
    return false;
  }
};

template <typename Policy, typename Deleter, typename Callback,
          typename PromiseValue, typename IndexSequence,
          typename... FutureValue>
class FutureLink;

/// Default Deleter for use with FutureLink that simply calls `delete`.
class DefaultFutureLinkDeleter {
 public:
  template <typename T>
  void operator()(T* x) const {
    delete x;
  }
};

template <typename T>
class EmptyCallbackHolder {
 public:
  /// Allow construction from any arguments.
  template <typename... U>
  EmptyCallbackHolder(U&&...) {}

  class Getter : public internal::EmptyObject<T> {
   public:
    Getter(EmptyCallbackHolder) {}
  };

  void Destroy() {}
};

template <typename T>
class NonEmptyCallbackHolder {
 public:
  template <typename... U>
  NonEmptyCallbackHolder(U&&... u) : value_(std::forward<U>(u)...) {}

  class Getter {
   public:
    Getter(NonEmptyCallbackHolder& holder) : holder_(holder) {}
    T& get() { return holder_.value_; }
    NonEmptyCallbackHolder& holder_;
  };
  void Destroy() { value_.~T(); }
  ~NonEmptyCallbackHolder() {}
  union {
    T value_;
  };
};

template <typename T>
using CallbackHolder =
    absl::conditional_t<std::is_empty<T>::value, EmptyCallbackHolder<T>,
                        NonEmptyCallbackHolder<T>>;

/// Alias that supplies the absl::index_sequence corresponding to the
/// `FutureValue...` pack.
template <typename Policy, typename Deleter, typename Callback,
          typename PromiseValue, typename... FutureValue>
using FutureLinkType =
    FutureLink<Policy, Deleter, Callback, PromiseValue,
               absl::make_index_sequence<sizeof...(FutureValue)>,
               FutureValue...>;

/// A FutureLink ties a `Promise<PromiseValue>` to one or more
/// `Future<FutureValue>...` objects and a `Callback`.
///
/// 1. If all of the futures become ready, the `Callback` is invoked, and the
///    FutureLink is unregistered.
///
/// 2. Depending on the `Policy`, if a future becomes ready in an error state,
///    the policy may propagate the error to the promise and unregister the
///    FutureLink.
///
/// 3. If the promise is forced before the FutureLink is unregistered, all of
///    the futures are forced.
///
/// 4. The future link may be manually unregistered through the use of the
///    registration handle returned by `GetCallbackPointer()`.  It is guaranteed
///    that no effects of the link (namely invocation of the `Callback`,
///    possibly setting an error on the promise, and forcing the futures) occur
///    after the call to unregister the link has returned (except that
///    unregistering the link from within the `Callback` itself is a no-op).
///
/// 5. If promise's result is no longer needed (i.e. `result_needed()` returns
///    `false`), the link is automatically unregistered.
///
/// To accomplish this, a `FutureLinkReadyCallback` (contained as a base class
/// of the FutureLink) is registered as a callback on each future, and a
/// `FutureLinkForceCallback` (also contained as a base class of the FutureLink)
/// is registered as a callback on the promise.
///
/// \tparam Policy A type that models the FutureLinkPolicy concept.
/// \tparam Deleter A function object type that can be called with a pointer to
///     this `FutureLink` object.
/// \tparam Callback A function object type that can be called with
///     `(Promise<PromiseValue>, ReadyFuture<FutureValue>...)`.
/// \tparam PromiseValue Value type of the promise.
/// \tparam FutureValue Value type of the future.
/// \tparam Is Must equal `0, ..., sizeof...(FutureValue)-1`.
template <typename Policy, typename Deleter, typename Callback,
          typename PromiseValue, typename... FutureValue, std::size_t... Is>
class FutureLink<Policy, Deleter, Callback, PromiseValue,
                 absl::index_sequence<Is...>, FutureValue...>
    : public FutureLinkBase,
      /// Inherit from the CallbackHolder, which holds the callback (if
      /// non-empty), in order to take advantage of empty base optimization,
      /// allow it to be initialized before the `FutureLinkReadyCallback` bases,
      /// and to allow destruction to be done manually.
      public CallbackHolder<Callback>,
      /// Inherit from the Deleter in order to take advantage of empty base
      /// optimization.
      public Deleter,
      /// Inherit from a single FutureLinkForceCallback and from a unique
      /// FutureLinkReadyCallback for each `Future<FutureValue)...` (uniquely
      /// identified by `Is...`).  Compared to using a std::tuple, this allows
      /// converting from a pointer to the
      /// `FutureLinkForceCallback`/`FutureLinkReadyCallback` to a pointer to
      /// the `FutureLink` without the need to store an extra pointer and
      /// without resorting to offsetof tricks.
      public FutureLinkForceCallback<
          FutureLinkType<Policy, Deleter, Callback, PromiseValue,
                         FutureValue...>,
          PromiseValue>,
      public FutureLinkReadyCallback<
          FutureLinkType<Policy, Deleter, Callback, PromiseValue,
                         FutureValue...>,
          FutureValue, Is>... {
  static_assert(sizeof...(FutureValue) <= kMaxNumFutures, "");

  // The promise and each of the futures have reference counts.  Additionally,
  // the FutureLinkForceCallback and each FutureLinkReadyCallback also have
  // reference counts.  Externally, ownership of references to these types is
  // managed using the reference counting smart pointer types
  // `FutureStatePointer`, `PromiseStatePointer`, and `CallbackPointer`.
  // Internally to FutureLink, however, ownership of these references is
  // implicit, and is managed as follows:
  //
  // The constructor is explicitly given 1 reference to the promise and 1
  // reference to each future due to passing of the Promise and Future objects
  // by value.
  //
  // 1. The FutureLinkForceCallback owns 1 reference to each future (so that it
  //    can ensure they remain valid until it calls Force on them, even if the
  //    futures become ready concurrently with calls to Force) and 1 reference
  //    to each FutureLinkReadyCallback (because it is responsible for
  //    unregistering the FutureLink).  Additionally, like any promise callback,
  //    it owns 1 reference to the promise.
  //
  // 2. Each FutureLinkReadyCallback, like any future callback, owns 1 reference
  //    to the future to which it is registered.  When the future becomes ready,
  //    or the callback is unregistered without being invoked, ownership of this
  //    reference to the future is implicitly transferred to the FutureLink
  //    object itself.
  //
  // 3. The FutureLink owns 1 reference to the promise (which is transferred to
  //    the callback if it is invoked), and 1 reference to the
  //    FutureLinkForceCallback (used to unregister it in response to ready
  //    state changes of the futures, depending on the Policy).
  //
  // 4. The caller of the FutureLink constructor implicitly owns one reference
  //    to the FutureLinkForceCallback (used to manually unregister the
  //    FutureLink), which it must obtain by calling `GetCallbackPointer()`.
  //
  // 5. If all of the futures become ready before the FutureLink is
  //    unregistered, the `Callback` is invoked and the FutureLink transfers to
  //    the callback its reference to the promise and a reference to each future
  //    (which it obtained from the FutureLinkReadyCallback when those futures
  //    became ready).  The FutureLink unregisters and releases its reference to
  //    the FutureLinkForceCallback.
  //
  // 6. If the FutureLink is unregistered before all of the futures become
  //    ready, it is considered to have been "cancelled".  The FutureLink
  //    unregisters and releases its reference to the FutureLinkForceCallback.
  //    This ensures that each FutureLinkReadyCallback has been unregistered,
  //    and that the future references they own have been transferred to the
  //    FutureLink.  The FutureLink then releases its reference to the promise
  //    and the references to the futures acquired from each
  //    FutureLinkReadyCallback.
  //
  // 7. Additionally, the FutureLink object itself has a reference count (stored
  //    in FutureLinkBase) that tracks whether the
  //    FutureLinkForceCallback::DestroyCallback and the
  //    FutureLinkReadyCallback::DestroyCallback methods have been called.  The
  //    `Deleter` is called to destroy the FutureLink once the last
  //    DestroyCallback methods is called.
 public:
  template <typename... CallbackInit>
  explicit FutureLink(Promise<PromiseValue> promise,
                      Future<FutureValue>... future, Deleter deleter,
                      CallbackInit&&... arg)
      : FutureLinkBase(sizeof...(FutureValue)),
        CallbackHolder<Callback>(std::forward<CallbackInit>(arg)...),
        Deleter(std::move(deleter)),
        // The FutureLinkForceCallback constructor initializes the stored
        // `shared_state` pointer but does not register the callback with its
        // promise.  That is done in the `RegisterLink` method below after the
        // FutureLinkReadyCallback objects have been initialized and registered.
        FutureLinkForceCallback<FutureLink, PromiseValue>(
            // Detach the promise reference to transfer ownership to this
            // FutureLink.
            FutureAccess::rep_pointer(promise).release()),
        // The FutureLinkReadyCallback constructor initializes the stores
        // `shared_state` pointer but does not register the callback with its
        // future.  This is done in the `RegisterLink` method below.
        FutureLinkReadyCallback<FutureLink, FutureValue, Is>(
            // Detach the future reference to transfer ownership to the
            // FutureLinkForceCallback.
            FutureAccess::rep_pointer(future).release())... {}

  /// Must be called exactly once after construction to finish setting up the
  /// link.  This must be called before the destructor.  This is done separately
  /// from the constructor to avoid a potential data race in accesses to the
  /// vtable pointer (in the case that `FutureLink` is the base class of another
  /// class), because as soon as the future link is established virtual methods
  /// may be called concurrently from other threads.
  void RegisterLink() {
    // Register each FutureLinkReadyCallback with its future.
    //
    // The link_state_ in FutureLinkBase, the Callback, and the
    // FutureLinkForceCallback were previously initialized (though the
    // `FutureLinkForceCallback` is not registered) by the `FutureLink`
    // constructor, because FutureLinkReadyCallback::OnReady, which depends on
    // these objects, may be called before the constructor returns.
    ForEachReadyCallback([](ReadyCallbackBase* ready_callback) {
      // Add a reference to `shared_state` and detach it in order to transfer
      // the reference to RegisterReadyCallback.
      FutureStatePointer(ready_callback->shared_state())
          .release()
          ->RegisterReadyCallback(ready_callback)
          // Detach the reference to the registered ready callback, implicitly
          // transferring ownership of the callback reference to the
          // FutureLinkForceCallback.
          .release();
    });

    // Increment the reference count of the force callback to account for the
    // reference that will be returned by GetCallbackPointer to the caller of
    // this constructor.  We increment the reference count before calling
    // RegisterForceCallback, to ensure the callback remains valid.
    ForceCallback()->reference_count_.fetch_add(1, std::memory_order_relaxed);

    // Create and detach a new promise reference that will be transferred to
    // `RegisterForceCallback`.
    PromiseStatePointer(ForceCallback()->shared_state())
        .release()
        // Register the FutureLinkForceCallback with the promise.  This must be
        // done after the FutureLinkReadyCallback objects have been registered,
        // because FutureLinkForceCallback::PreDestroy, which may be invoked
        // before RegisterForceCallback returns, assumes the ready callbacks are
        // registered.
        ->RegisterForceCallback(ForceCallback())
        // Detach the reference to the FutureLinkForceCallback, implicitly
        // transferring ownership to this FutureLink.
        .release();

    // Mark that the FutureLinkForceCallback has been registered.
    LinkState reference_count = link_state_.fetch_or(kRegistered);
    if (reference_count & kCancelled) {
      // The FutureLink was cancelled before the FutureLinkForceCallback was
      // registered.
      Cancel();
    } else if ((reference_count & kNotReadyFutureMask) == 0) {
      // All of the futures became ready before the FutureLinkForceCallback was
      // registered.
      InvokeCallback();
    }
  }

  /// Must be called exactly once by the caller of the FutureLink constructor to
  /// obtain the CallbackPointer that may be used to unregister this link.
  CallbackPointer GetCallbackPointer() {
    return CallbackPointer(ForceCallback(), internal::adopt_object_ref);
  }

  /// Returns a non-null pointer to the force callback.
  FutureLinkForceCallback<FutureLink, PromiseValue>* ForceCallback() {
    return this;
  }

  template <typename T, std::size_t I>
  FutureLinkReadyCallback<FutureLink, T, I>* ReadyCallback() {
    return this;
  }

  /// Invokes `func` with a pointer to each of the ready callbacks.
  template <typename Func>
  ABSL_ATTRIBUTE_ALWAYS_INLINE void ForEachReadyCallback(Func func) {
    const auto unused ABSL_ATTRIBUTE_UNUSED = {
        (static_cast<void>(func(ReadyCallback<FutureValue, Is>())), 0)...};
  }

  /// Called by the FutureLinkForceCallback::OnForced method when the linked
  /// promise is forced.
  void OnPromiseForced() {
    ForEachReadyCallback([](ReadyCallbackBase* ready_callback) {
      ready_callback->shared_state()->Force();
    });
  }

  /// Called by FutureLinkForceCallback::OnUnregistered.
  void OnForceCallbackUnregistered() {
    ForceCallback()->shared_state()->ReleasePromiseReference();
    ForEachReadyCallback([](ReadyCallbackBase* ready_callback) {
      ready_callback->shared_state()->ReleaseFutureReference();
      ready_callback->Unregister(/*block=*/true);
      CallbackPointerTraits::decrement(ready_callback);
    });
  }

  /// Called when one of the linked futures becomes ready.
  template <typename T>
  void OnFutureReady(FutureState<T>* future_state) {
    if (Policy::OnFutureReady(future_state, ForceCallback()->shared_state())) {
      OnFutureReadyForCallback();
    } else {
      EnsureCancelled();
    }
  }

  /// Called by `OnFutureReady` when one of the linked futures becomes ready and
  /// a condition that would disqualify the callback from being invoked has not
  /// occurred.
  void OnFutureReadyForCallback() {
    // Update `link_state_` to reflect that another future is ready.
    const LinkState new_reference_count =
        (link_state_ -= kNotReadyFutureMultiplier);
    if ((new_reference_count & (kNotReadyFutureMask | kRegistered)) ==
        kRegistered) {
      // All futures are ready, and the FutureLinkForceCallback has been
      // registered.
      InvokeCallback();
    }
  }

  /// Destroys the contained callback.  This is done manually exactly once.
  void DestroyUserCallback() { this->CallbackHolder<Callback>::Destroy(); }

  /// Called when all the futures are ready.  This is called before the last
  /// call to OnReady returns, and therefore is guaranteed not to be run after
  /// this FutureLink has been unregistered.
  ///
  /// There must be exactly one call to InvokeCallback or Cancel prior to the
  /// destructor being called.
  void InvokeCallback() noexcept {
    typename CallbackHolder<Callback>::Getter callback_getter(*this);
    callback_getter.get()(
        FutureAccess::Construct<Promise<PromiseValue>>(PromiseStatePointer(
            ForceCallback()->shared_state(), internal::adopt_object_ref)),
        FutureAccess::Construct<ReadyFuture<FutureValue>>(
            FutureStatePointer(ReadyCallback<FutureValue, Is>()->shared_state(),
                               internal::adopt_object_ref))...);
    DestroyUserCallback();

    // Any concurrent attempts from another thread to unregister the link while
    // this function is executing will block until `UnregisterForceCallback()`
    // returns.  Therefore, we call `UnregisterForceCallback()` after invoking
    // the user-specified callback, rather than before, to ensure that any
    // concurrent unregister attempt blocks until after the user-specified
    // callback has run.
    UnregisterForceCallback();
  }

  /// Called when one of the ready callbacks is unregistered without being
  /// invoked.
  ///
  /// 1. If the FutureLink was already cancelled, this does nothing.
  ///
  /// 2. Otherwise:
  ///
  ///    a. If the FutureLinkForceCallback has already been registered, then
  ///       call Cancel() immediately.
  ///
  ///    b. Otherwise, the FutureLink constructor will call `Cancel()` after the
  ///       FutureLinkForceCallback has been registered.
  void EnsureCancelled() {
    // If the FutureLink is not already cancelled, and the force callback has
    // been registered, call `Cancel()`.
    if ((MarkCancelled() & (kCancelled | kRegistered)) == kRegistered) {
      Cancel();
    }
  }

  /// Called when the first ready callback has been unregistered without being
  /// invoked.
  void Cancel() {
    DestroyUserCallback();

    // Unregister the force callback.  This causes all of the ready callbacks to
    // be unregistered, which results in the ownership of a reference to their
    // futures being transferred to this FutureLink, which allows us to release
    // those references.
    UnregisterForceCallback();
    ForEachReadyCallback([](ReadyCallbackBase* ready_callback) {
      ready_callback->shared_state()->ReleaseFutureReference();
    });

    // We release the promise reference only after unregistering all of the
    // callbacks.  This ensures that a concurrent call to OnFutureReady with
    // `Policy == FutureLinkPropagateFirstErrorPolicy` is done accessing the
    // promise (in the case of an error) before we release ownership of it.
    ForceCallback()->shared_state()->ReleasePromiseReference();
  }

  /// Unregisters the force callback if it has not already been unregistered,
  /// and releases the reference to it owned by the FutureLink.
  void UnregisterForceCallback() {
    ForceCallback()->Unregister(/*block=*/false);
    CallbackPointerTraits::decrement(ForceCallback());
  }

  /// Called from FutureLinkForceCallback::DestroyCallback (if `promise` is
  /// `true`) or FutureLinkReadyCallback::DestroyCallback (if `promise` is
  /// `false`) to indicate the FutureLinkForceCallback or
  /// FutureLinkReadyCallback has no more references.  This decrements the
  /// FutureLink's reference count, and the Deleter is invoked when the
  /// reference count reaches zero.
  void MarkLinkCallbackDestroyed(bool promise) {
    if (ReleaseLinkReferenceFromCallback(promise)) {
      (*static_cast<Deleter*>(this))(this);
    }
  }
};

enum class FutureErrorPropagationResult { kReady, kNotReady, kError };

template <typename Policy>
FutureErrorPropagationResult PropagateFutureError(void* promise) {
  return FutureErrorPropagationResult::kReady;
}

template <typename Policy, typename T, typename U>
FutureErrorPropagationResult PropagateFutureError(FutureState<T>* promise,
                                                  FutureState<U>* future) {
  if (!future->ready()) return FutureErrorPropagationResult::kNotReady;
  return Policy::OnFutureReady(future, promise)
             ? FutureErrorPropagationResult::kReady
             : FutureErrorPropagationResult::kError;
}

template <typename Policy, typename T, typename U0, typename... U>
FutureErrorPropagationResult PropagateFutureError(FutureState<T>* promise,
                                                  FutureState<U0>* future0,
                                                  FutureState<U>*... future) {
  const auto result = PropagateFutureError<Policy>(promise, future0);
  if (result == FutureErrorPropagationResult::kError) return result;
  return std::max(result, PropagateFutureError<Policy>(promise, future...));
}

/// Creates a FutureLink and returns the CallbackPointer that can be used to
/// unregister it.
///
/// \tparam Policy A type that models the FutureLinkPolicy concept.
/// \tparam Callback A function object type that can be called with
///     `(Promise<PromiseValue>, ReadyFuture<FutureValue>...)`.
/// \tparam PromiseValue Value type of the promise.
/// \tparam FutureValue Value type of the future.
template <typename Policy, typename Callback, typename PromiseValue,
          typename... FutureValue>
CallbackPointer MakeLink(Callback&& callback, Promise<PromiseValue> promise,
                         Future<FutureValue>... future) {
  if (!promise.result_needed()) return {};
  switch (PropagateFutureError<Policy>(&FutureAccess::rep(promise),
                                       &FutureAccess::rep(future)...)) {
    case FutureErrorPropagationResult::kReady:
      std::forward<Callback>(callback)(
          std::move(promise), ReadyFuture<FutureValue>(std::move(future))...);
      return {};
    case FutureErrorPropagationResult::kError:
      return {};
    case FutureErrorPropagationResult::kNotReady: {
      auto link = new internal_future::FutureLinkType<
          Policy, DefaultFutureLinkDeleter, internal::remove_cvref_t<Callback>,
          PromiseValue, FutureValue...>(std::move(promise),
                                        std::move(future)..., {},
                                        std::forward<Callback>(callback));
      link->RegisterLink();
      return link->GetCallbackPointer();
    }
  }
  TENSORSTORE_UNREACHABLE;  // COV_NF_LINE
}

/// Overload for the case of zero futures.
///
/// This simply invokes the callback immediately and returns a nullptr
/// `CallbackPointer`.
///
/// \tparam Policy Ignored.
/// \tparam Callback A function object type that can be called with
///     `(Promise<PromiseValue>)`.
/// \tparam PromiseValue Value type of the promise.
template <typename Policy, typename Callback, typename PromiseValue>
CallbackPointer MakeLink(Callback&& callback, Promise<PromiseValue> promise) {
  std::forward<Callback>(callback)(std::move(promise));
  return {};
}

/// No-op callback that may be used with `Link`.
struct NoOpCallback {
  template <typename... X>
  void operator()(X&&...) const {}
};

template <typename Policy, typename Callback, typename PromiseValue,
          typename... FutureValue>
class LinkedFutureState;

class LinkedFutureStateDeleter {
 public:
  template <typename Policy, typename Callback, typename PromiseValue,
            typename... FutureValue>
  void operator()(FutureLinkType<Policy, LinkedFutureStateDeleter, Callback,
                                 PromiseValue, FutureValue...>* x) const {
    static_cast<
        LinkedFutureState<Policy, Callback, PromiseValue, FutureValue...>*>(x)
        ->DestroyLink();
  }
};

/// Class that inherits from both FutureState and FutureLink that allows both
/// the shared FutureState and a FutureLink to be created with a single dynamic
/// memory allocation.
///
/// \tparam Policy A type that models the FutureLinkPolicy concept.
/// \tparam Callback A function object type that can be called with
///     `(Promise<PromiseValue>, ReadyFuture<FutureValue>...)`.
/// \tparam PromiseValue Value type of the promise.
/// \tparam FutureValue Value type of the future.
template <typename Policy, typename Callback, typename PromiseValue,
          typename... FutureValue>
class LinkedFutureState
    : public FutureState<PromiseValue>,
      public FutureLinkType<Policy, LinkedFutureStateDeleter, Callback,
                            PromiseValue, FutureValue...> {
  static_assert(sizeof...(FutureValue) > 0,
                "LinkedFutureState requires at least one Future.");
  using FutureStateType = FutureState<PromiseValue>;
  using FutureLink = FutureLinkType<Policy, LinkedFutureStateDeleter, Callback,
                                    PromiseValue, FutureValue...>;

 public:
  /// Constructs the LinkedFutureState with the callback initialized from
  /// `callback_init` and the result initialized from `result_init...`.
  template <typename CallbackInit, typename... ResultInit>
  explicit LinkedFutureState(Future<FutureValue>... future,
                             CallbackInit&& callback_init,
                             ResultInit&&... result_init)
      : FutureStateType(std::forward<ResultInit>(result_init)...),
        FutureLink(FutureAccess::Construct<Promise<PromiseValue>>((
                       // Increment the combined reference count, which is
                       // released by the LinkedFutureStateDeleter.
                       FutureStateType::combined_reference_count_.fetch_add(
                           1, std::memory_order_relaxed),
                       // Add a promise reference to transfer to the FutureLink.
                       PromiseStatePointer(this))),
                   std::move(future)..., /*deleter=*/{},
                   std::forward<CallbackInit>(callback_init)) {
    this->RegisterLink();
    // Release the callback pointer.
    this->GetCallbackPointer();
  }

  void DestroyLink() noexcept {
    static_cast<FutureStateType*>(this)->ReleaseCombinedReference();
  }
};

/// Interface for making a FutureState that is linked to zero or more futures
/// according to `Policy`.
///
/// This is defined as a class with a nested `Make` function in order to allow
/// partial specialization and to accommodate both the `FutureValue...` and
/// `ResultInit...` template parameter packs.
///
/// \tparam Policy A type that models the FutureLinkPolicy concept.
/// \tparam PromiseValue Value type of the promise.
/// \tparam FutureValue Value type of the future.
template <typename Policy, typename PromiseValue, typename... FutureValue>
struct MakeLinkedFutureState {
  /// Returns a new `FutureState<PromiseValue>` where the promise has been
  /// linked via `Callback` to `future...`.
  ///
  /// The promise result is initialized with `result_init...`.
  ///
  /// \tparam Callback A function object type that can be called with
  ///     `(Promise<PromiseValue>, ReadyFuture<FutureValue>...)`.
  template <typename Callback, typename... ResultInit>
  static FutureState<PromiseValue>* Make(Future<FutureValue>... future,
                                         Callback&& callback,
                                         ResultInit&&... result_init) {
    return new internal_future::LinkedFutureState<
        Policy, internal::remove_cvref_t<Callback>, PromiseValue,
        FutureValue...>(std::move(future)..., std::forward<Callback>(callback),
                        std::forward<ResultInit>(result_init)...);
  }
};

/// Partial specialization for the case of no linked futures.
///
/// This just invokes the callback immediately.
template <typename Policy, typename PromiseValue>
struct MakeLinkedFutureState<Policy, PromiseValue> {
  template <typename Callback, typename... ResultInit>
  static FutureState<PromiseValue>* Make(Callback&& callback,
                                         ResultInit&&... result_init) {
    auto* state =
        new FutureState<PromiseValue>(std::forward<ResultInit>(result_init)...);
    std::forward<Callback>(callback)(
        FutureAccess::Construct<Promise<PromiseValue>>(
            PromiseStatePointer(state)));
    return state;
  }
};

}  // namespace internal_future
}  // namespace tensorstore

#endif  // TENSORSTORE_UTIL_FUTURE_IMPL_H_
