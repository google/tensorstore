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

#ifndef TENSORSTORE_UTIL_FUTURE_H_
#define TENSORSTORE_UTIL_FUTURE_H_

/// \file
/// Implements
/// `Future<T>` and `Promise<T>` which provide an asynchronous one-time channel
/// between a producer and a consumer.
///
/// `Future<T>` provides an interface for a consumer to asynchronously receive a
/// value of type `T` or an error through a `Result<T>`.
///
/// In typical use, a consumer uses the `Future::ExecuteWhenReady` method to
/// register a callback to be invoked when the shared state becomes ready.
/// Alternatively, `Future::result` may be called to synchronously wait for
/// result, blocking the current thread.
///
/// `Promise<T>` provides an interface for a producer to asynchronously send a
/// value of type `T` or an error through a `Result<T>`.
///
/// In typical usage, an asynchronous operation initiator will create paired
/// Future<T> and Promise<T> objects by invoking `PromiseFuturePair<T>::Make()`.
/// From there, the asynchronous operation which returns a T can be installed on
/// the Promise<>, typically putting the operation and the promise onto an
/// executor or other thread pool, and storing the result of the operation on
/// the promise, while returning the Future<> to the caller.
///
/// Both `Promise<void>` and `Future<void>` are supported and return a
/// Result<void>.
///
/// `Promise<T>` and `Future<T>` behave like reference counted pointers to a
/// shared state that stores the actual value.  In normal usage, there will be
/// one or more instances of `Promise<T>`, and one or more instances of
/// `Future<T>`, all referring to the same shared state.  The shared state is
/// initially marked "not ready", and transitions to the "ready" state when a
/// producer sets the result.  Once the state becomes "ready", it cannot become
/// "not ready", and any further attempts to set the value have no effect.
///
/// A limited form of cancellation is supported. Cancellation is signalled when
/// all `Future<T>` instances referencing the shared state are destroyed or
/// go out of scope before the operation completes. Using the
/// `Promise::ExecuteWhenNotNeeded` method, a producer may register callbacks to
/// be invoked when either the shared state becomes ready or there are no more
/// futures referencing the shared state.
///
/// In addition to the producer -> consumer communication of a value or error, a
/// limited form of consumer -> producer communication is also supported: a
/// producer may defer performing certain work until the result is actually
/// required.  For example, this is useful if the deferred work may never be
/// needed, or multiple deferred work items may be more efficiently handled
/// together as a batch.  On the consumer side, this is handled by the
/// Future::Force method that may be invoked manually to indicate that the
/// result is needed, and is also invoked automatically when a thread
/// synchronously waits for the result.  This calls any callbacks that a
/// producer has registered using the `Promise::ExecuteWhenForced` method.
///
/// This Promise/Future implementation is executor-agnostic: all callbacks that
/// are registered are either invoked immediately before the registration
/// function returns (if the condition is already satisfied), or immediately by
/// the thread that causes the condition to be satisfied (e.g. calling
/// `Promise::SetResult`, `Future::Force`, or destroying the last Future or
/// Promise referencing a given shared state).  Therefore, the callbacks that
/// are registered should not block or perform expensive computations; instead,
/// any blocking or expensive computations should either be somehow deferred or
/// run asynchronously on another thread.  In addition, actions such as calling
/// `Promise::SetResult`, `Future::Force`, and destroying a Promise or Future
/// object should not be done while holding any exclusive locks that may be
/// needed by a callback.
///
/// Examples:
///
///     // Runs `function` asynchronously using `executor`, and returns a Future
///     // representing the return value of `function`.
///     template <typename T>
///     Future<T> RunAsync(std::function<T()> function, Executor executor) {
///       auto pair = PromiseFuturePair<T>::Make();
///       executor([promise = std::move(pair.promise),
///                 function = std::move(function)] {
///         promise.SetResult(function());
///       });
///       return std::move(pair.future);
///     }
///     Future<void> future = RunAsync([]->void {
///         std::count << " Async! " << std::endl;
///     }, executor);
///     future.Wait();
///
///     // Like `RunAsync`, except execution of `function` is deferred until
///     // Force is called on the returned Future.
///     template <typename T>
///     Future<T> RunDeferred(std::function<T()> function, Executor executor) {
///       auto pair = PromiseFuturePair<T>::Make();
///       pair.promise.ExecuteWhenForced([promise = std::move(pair.promise),
///                                       function = std::move(function),
///                                       executor = std::move(executor)] {
///         executor([promise = std::move(promise),
///                   function = std::move(function)] {
///           promise.SetResult(function());
///         });
///       });
///       return std::move(pair.future);
///     }
///
///     // Returns a Future representing the asynchronous addition of
///     // `a.value()` and `b.value()`.
///     Future<int> AddAsync1(Future<int> a, Future<int> b) {
///       return PromiseFuturePair<int>::LinkValue(
///         [](Promise<int> c, Future<int> a, Future<int> b) {
///           c.SetResult(MapResult(std::plus<int>{}, a.result(), b.result()));
///         }, std::move(a), std::move(b)).future;
///     }
///
///     // Equivalent to `AddAsync1`, showing usage of `MapFutureValue`.
///     Future<int> AddAsync2(Future<int> a, Future<int> b) {
///       return MapFutureValue(InlineExecutor{},
///                             std::plus<int>{}, std::move(a), std::move(b));
///     }
///
///     // Synchronously adds `a.value()` and `b.value()`.
///     Result<int> AddSync1(Future<int> a, Future<int> b) {
///       // In case `a` and `b` represent deferred computations, call Force on
///       // both before blocking via `result()` to allow both computations to
///       // run in parallel.
///       a.Force();
///       b.Force();
///       return MapResult(std::plus<int>{}, a.result(), b.result());
///     }
///
///     // Equivalent to `AddSync1`, showing how a Future-returning function can
///     // be used synchronously.
///     Result<int> AddSync2(Future<int> a, Future<int> b) {
///       return AddAsync2(a, b).result();
///     }

#include <atomic>
#include <tuple>
#include <type_traits>
#include <utility>

#include "absl/base/macros.h"
#include "absl/meta/type_traits.h"
#include "absl/time/time.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/meta.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/internal/void_wrapper.h"
#include "tensorstore/util/assert_macros.h"
#include "tensorstore/util/execution.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future_impl.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

namespace tensorstore {

template <typename T>
class [[nodiscard]] Future;
template <typename T>
class [[nodiscard]] Promise;
template <typename T>
class [[nodiscard]] PromiseFuturePair;
template <typename T>
class [[nodiscard]] ReadyFuture;

/// Bool-valued metafunction equal to `true` if, and only if, `Future<SourceT>`
/// is convertible to `Future<DestT>`.
template <typename SourceT, typename DestT>
using IsFutureConvertible = internal::IsConstConvertible<SourceT, DestT>;

/// Handle to a registered Future or Promise callback, that may be used to
/// unregister it.
///
/// This type has shared weak ownership semantics, and may be copied to create
/// an equivalent handle, either of which may be used to unregister the
/// callback.
///
/// A handle may be either live, meaning it refers to a callback, or null.  Live
/// handles to a callback prevent its memory from being freed, even after it has
/// finished running or been unregistered using a different handle, but live
/// handles do not prevent the destructors of callback function objects from
/// running.
///
/// Move construction and move assignment leave the moved-from handle in the
/// null state.
///
/// \threadsafety Multiple handle objects referring to the same callback may
///     safely be used concurrently from multiple threads, but non-`const`
///     methods of the same handle object must not be called concurrently from
///     multiple threads.
class FutureCallbackRegistration {
 public:
  /// Constructs a null handle.
  FutureCallbackRegistration() = default;

  /// Unregisters the associated callback.
  ///
  /// 1. If the associated callback has already finished running, or has already
  ///    been unregistered, this has no effect.
  ///
  /// 2. If the associated callback is currently running in another thread, this
  ///    blocks until the callback finishes running.  This has the potential to
  ///    cause deadlock if not used carefully.  Use `UnregisterNonBlocking`
  ///    instead to reduce the risk of deadlock if this blocking behavior is not
  ///    required.
  ///
  /// 3. If the associated callback is currently running in the current thread,
  ///    i.e. the callback has directly or indirectly attempted to unregister
  ///    itself, then this function returns immediately (it does not deadlock).
  ///
  /// In all cases, after this function returns, the handle is null.
  ///
  /// \note In cases 1 and 2, it is guaranteed that no code within the callback
  ///     will execute once `Unregister` returns.  The only exception is case 3,
  ///     where this guarantee is impossible.
  ///
  /// \threadsafety This method is NOT safe to call concurrently from multiple
  ///     threads on the same handle, but is safe to call concurrently from
  ///     multiple threads on different handles that refer to the same callback.
  void Unregister() noexcept {
    if (!rep_) {
      return;
    }
    rep_->Unregister(/*block=*/true);
    rep_.reset();
  }

  /// Same as `Unregister`, except that in the case that the callback is
  /// executing concurrently in another thread, this method does not block until
  /// it completes.
  void UnregisterNonBlocking() noexcept {
    if (!rep_) {
      return;
    }
    rep_->Unregister(/*block=*/false);
    rep_.reset();
  }

  /// Equivalent to `Unregister()`.
  void operator()() noexcept { this->Unregister(); }

 private:
  friend class internal_future::FutureAccess;
  explicit FutureCallbackRegistration(internal_future::CallbackPointer pointer)
      : rep_(std::move(pointer)) {}
  internal_future::CallbackPointer rep_;
};

/// "Producer" interface to a one-time channel.
///
/// \tparam T Specifies the type of the value to be transmitted.  The actual
///     result value type is `Result<remove_const_t<T>>`.
template <typename T>
class Promise {
  static_assert(!std::is_reference<T>::value,
                "T must not be a reference type.");
  using SharedState = internal_future::FutureStateType<T>;

 public:
  /// The result type transmitted by the channel.
  using result_type = internal_future::ResultType<T>;

  /// The value type contained in the result type.
  using value_type = T;

  /// Constructs an invalid `Promise`.
  /// \post `!valid()`.
  Promise() = default;

  /// Constructs from a compatible `Promise`.
  /// \tparam U The source value type.
  /// \requires `IsFutureConvertible<U, T>::value`.
  template <typename U,
            std::enable_if_t<IsFutureConvertible<U, T>::value>* = nullptr>
  Promise(Promise<U> x) noexcept
      : rep_(std::move(internal_future::FutureAccess::rep_pointer(x))) {}

  /// Assigns from a compatible `Promise`.
  /// \tparam U The source value type.
  /// \requires `IsFutureConvertible<U, T>::value`.
  template <typename U>
  std::enable_if_t<IsFutureConvertible<U, T>::value, Promise&> operator=(
      Promise<U> x) noexcept {
    rep_ = std::move(internal_future::FutureAccess::rep_pointer(x));
    return *this;
  }

  /// Resets this Promise to be invalid.
  /// \post `!valid()`.
  void reset() noexcept { rep_.reset(); }

  /// Returns `true` if this `Promise` refers to a valid shared state.
  bool valid() const noexcept { return static_cast<bool>(rep_); }

  /// Returns `true` if the result is ready.
  ///
  /// \dchecks `valid()`
  /// \remark Once this returns `true` for a given shared state, it will never
  ///     return `false`.
  bool ready() const noexcept { return rep().ready(); }

  /// Returns `true` if this future has not been cancelled.
  /// Once there are no remaining `Future` objects or ready callbacks
  /// associated with the shared state, the future will be cancelled.
  ///
  /// \remark Once this returns `false` for a given shared state, it will never
  ///     return `true`.
  bool result_needed() const noexcept { return rep().result_needed(); }

  /// Returns a reference to the result value stored in the shared state.
  ///
  /// No locking/synchronization is used.  The caller must ensure that
  /// concurrent access to the referenced value are safe, including any
  /// concurrent accesses that may result from calls to SetResult or
  /// SetResultWith.
  ///
  /// \dchecks `valid()`
  std::add_lvalue_reference_t<result_type> raw_result() const {
    return rep().result;
  }

  /// Sets the result if it has not already been set.
  ///
  /// The arguments are forwarded to the `result_type` constructor.
  /// \returns `true` if the result was not already set.
  template <typename... U>
  std::enable_if_t<(!std::is_const<T>::value &&
                    (std::is_constructible<result_type, U...>::value)),
                   bool>
  SetResult(U&&... u) const noexcept {
    return rep().SetResult(std::forward<U>(u)...);
  }

  /// Marks result as ready without modifying it.
  ///
  /// The result may have previously been modified directly using the
  /// `raw_result()` accessor.
  ///
  /// \requires `T` is not `const`.
  /// \returns `true` if there were no prior calls to `SetResult` or `SetReady`.
  template <typename U = T>
  std::enable_if_t<!std::is_const<U>::value, bool> SetReady() const noexcept {
    return rep().SetReady();
  }

  /// Registers a callback to be invoked when `Force` is called on an associated
  /// `Future`.
  ///
  /// If `Force` has already been called, the callback is invoked immediately.
  ///
  /// \param callback A function object to be invoked with a copy of `*this`.
  ///     The return value of the callback is ignored.
  /// \returns A handle that may be used to unregister the callback.
  /// \remark Typically, the specified `callback` function starts deferred work
  ///     that ultimately leads to the result being set.
  template <typename Callback>
  FutureCallbackRegistration ExecuteWhenForced(Callback&& callback) const {
    auto& rep = this->rep();
    // If there are no future references, there is no need to force.
    if (rep.has_future()) {
      const auto value = rep.state_.load(std::memory_order_acquire);
      if ((value & (internal_future::FutureStateBase::kReady |
                    internal_future::FutureStateBase::kForcing)) == 0) {
        // The state has not been marked ready and the force callbacks have not
        // yet been invoked.  We just add the callback to the list normally.
        using Impl =
            internal_future::ForceCallback<T,
                                           internal::remove_cvref_t<Callback>>;
        return internal_future::FutureAccess::Construct<
            FutureCallbackRegistration>(
            // Create and detach a new promise reference to transfer to
            // RegisterForceCallback.  The reference detached by the call to
            // `release` is implicitly transferred via the `this` pointer to
            // `RegisterForceCallback`.
            internal_future::PromiseStatePointer(rep_)
                .release()
                ->RegisterForceCallback(
                    new Impl(&rep, std::forward<Callback>(callback))));
      }
      if ((value & (internal_future::FutureStateBase::kReady |
                    internal_future::FutureStateBase::kForcing)) ==
          internal_future::FutureStateBase::kForcing) {
        // The state has not been marked ready, but the invocation of force
        // callbacks has already started (and possibly finished).  We just
        // invoke it immediately.
        std::forward<Callback>(callback)(*this);
      }
    }
    return {};
  }

  /// Registers a callback to be invoked when `result_needed()` becomes `false`.
  ///
  /// If `result_needed()` is `false` upon invocation, the callback is executed
  /// immediately.
  ///
  /// \param callback A function object to be invoked with no arguments.  The
  ///     return value of the callback is ignored.
  /// \returns A handle that may be used to unregister the callback.
  /// \remark Typically, the specified `callback` function cancels pending work
  ///     when invoked.
  template <typename Callback>
  FutureCallbackRegistration ExecuteWhenNotNeeded(Callback&& callback) const {
    auto& rep = this->rep();
    if (rep.result_needed()) {
      using Impl = internal_future::ResultNotNeededCallback<
          internal::remove_cvref_t<Callback>>;
      return internal_future::FutureAccess::Construct<
          FutureCallbackRegistration>(rep.RegisterNotNeededCallback(
          new Impl(&rep, std::forward<Callback>(callback))));
    }
    std::forward<Callback>(callback)();
    return {};
  }

  /// Returns an associated Future, if one can be obtained without affecting the
  /// value of `result_needed()` (which would violate an invariant).
  ///
  /// Specifically, this returns a valid Future if there is at least one
  /// remaining future reference, or `SetResult()` or `SetReady()` has been
  /// called.
  ///
  /// Otherwise, this returns an invalid Future.
  ///
  /// \dchecks `valid()`.
  Future<T> future() const {
    auto& rep = this->rep();
    if (!rep.AcquireFutureReference()) return {};
    return internal_future::FutureAccess::Construct<Future<T>>(
        internal_future::FutureStatePointer(&rep, internal::adopt_object_ref));
  }

  /// The `set_value`, `set_error`, and `set_cancel` functions defined below
  /// make `Promise<T>` model the `Receiver<Status, T>` concept.  Calling any of
  /// these methods has no effect if the promise is already in a ready state.
  /// This implies that calling any of these functions after they have already
  /// been called on a given Promise has no effect.

  /// Implements the Receiver `set_value` operation.
  template <typename... V>
  friend std::enable_if_t<
      (!std::is_const<T>::value &&
       std::is_constructible<result_type, std::in_place_t, V...>::value)>
  set_value(const Promise& promise, V&&... v) {
    promise.SetResult(std::in_place, std::forward<V>(v)...);
  }

  /// Implements the Receiver `set_error` operation.
  template <typename... V>
  friend void set_error(const Promise& promise, Status error) {
    promise.SetResult(std::move(error));
  }

  /// Implements the Receiver `set_cancel` operation.
  template <typename... V>
  friend void set_cancel(const Promise& promise) {
    promise.SetResult(absl::CancelledError(""));
  }

 private:
  explicit Promise(internal_future::PromiseStatePointer rep)
      : rep_(std::move(rep)) {}
  friend class internal_future::FutureAccess;
  constexpr SharedState& rep() const {
    ABSL_ASSERT(rep_);
    return static_cast<SharedState&>(*rep_);
  }
  internal_future::PromiseStatePointer rep_;
};

/// Creates a `Future` in a ready state.
///
/// The contained result of type `Result<T>` is initialized using the specified
/// arguments.
///
/// \tparam T The value type.
template <typename T, typename... U>
std::enable_if_t<std::is_constructible<Result<T>, U...>::value, ReadyFuture<T>>
MakeReadyFuture(U&&... u) {
  auto pair = PromiseFuturePair<T>::Make(std::forward<U>(u)...);
  // Release the reference to the promise, which makes the future ready.
  pair.promise.reset();
  return ReadyFuture<T>(pair.future);
}

/// Returns a ready `Future<const void>`.  This simply returns a global value
/// and avoids allocations.
ReadyFuture<const void> MakeReadyFuture();

/// "Consumer" interface to a one-time channel.
///
/// \tparam T Specifies the type of the value to be transmitted.  The actual
///     result value type is `Result<remove_const_t<T>>`.
template <typename T>
class Future {
  static_assert(!std::is_reference<T>::value,
                "T must not be a reference type.");
  using SharedState = internal_future::FutureStateType<T>;

 public:
  /// The result type transmitted by the channel.
  using result_type = internal_future::ResultType<T>;

  /// The value type contained in the result type.
  using value_type = T;

  /// Constructs an invalid `Future`.
  /// \post `!valid()`.
  Future() = default;

  /// Constructs a ready Future from any value that can be implicitly converted
  /// to `result_type`.
  template <
      typename U,
      std::enable_if_t<std::is_convertible<U&&, result_type>::value>* = nullptr>
  Future(U&& u)
      : Future(MakeReadyFuture<std::remove_const_t<T>>(std::forward<U>(u))) {}

  /// Constructs a Future from a `Result<Future<U>>`, where `U` is `T` or
  /// `const U` is `T`.
  ///
  /// If `result` is in an error state, constructs a ready future from
  /// `result.status()`.  Otherwise, constructs from `*result`.
  ///
  /// \remark Passing the argument by value doesn't work due to a circular
  ///     dependency in overload resolution leading `std::is_convertible` to
  ///     break.  There is not much advantage in a separate rvalue overload
  ///     because it would only save copying the error status of the result.
  template <typename U,
            std::enable_if_t<IsFutureConvertible<U, T>::value>* = nullptr>
  Future(const Result<Future<U>>& result) {
    if (result) {
      *this = *result;
    } else {
      *this = MakeReadyFuture<std::remove_const_t<T>>(result.status());
    }
  }

  /// Constructs from a compatible `Future`.
  /// \tparam U The source value type.
  /// \requires `IsFutureConvertible<U, T>::value`.
  template <typename U,
            std::enable_if_t<IsFutureConvertible<U, T>::value>* = nullptr>
  Future(Future<U> x) noexcept
      : rep_(std::move(internal_future::FutureAccess::rep_pointer(x))) {}

  /// Assigns from a compatible `Future`.
  /// \tparam U The source value type.
  /// \requires `IsFutureConvertible<U, T>::value`.
  template <typename U>
  std::enable_if_t<IsFutureConvertible<U, T>::value, Future&> operator=(
      Future<U> x) noexcept {
    rep_ = std::move(internal_future::FutureAccess::rep_pointer(x));
    return *this;
  }

  /// Ignores the future. This method signals intent to ignore the result
  /// to suppress compiler warnings from [[nodiscard]].
  void IgnoreFuture() const {}

  /// Resets this Future to be invalid.
  /// \post `!valid()`.
  void reset() noexcept { rep_.reset(); }

  /// Returns `true` if this `Future` refers to a valid shared state.
  bool valid() const noexcept { return static_cast<bool>(rep_); }

  /// Returns `true` if the result is ready.
  /// \dchecks `valid()`
  /// \remark Once this returns `true` for a given shared state, it will never
  ///     return `false`.
  bool ready() const noexcept { return rep().ready(); }

  /// Calls `Force()`, and waits until `ready() == true`.
  ///
  /// \dchecks `valid()`
  void Wait() const noexcept { rep().Wait(); }

  /// Waits for up to the specified duration for the result to be ready.
  /// \dchecks `valid()`
  /// \returns `ready()`.
  bool WaitFor(absl::Duration duration) const noexcept {
    return rep().WaitFor(duration);
  }

  /// Waits until the specified time for the result to be ready.
  /// \dchecks `valid()`
  /// \returns `ready()`.
  bool WaitUntil(absl::Time deadline) const noexcept {
    return rep().WaitUntil(deadline);
  }

  /// Calls any `Force` callbacks registered on the associated `Promise`, if
  /// they have not already been called.
  ///
  /// Commonly, this will trigger deferred work to begin.
  ///
  /// \dchecks `valid()`
  void Force() const noexcept { rep().Force(); }

  /// Registers a callback to invoke when `ready()` becomes `true`.
  ///
  /// \dchecks `valid()`.
  /// \param callback A function object to be invoked with a `ReadyFuture<T>`
  ///     referring to the same shared state as `*this`.  The return value of
  ///     the callback is ignored.
  /// \returns A handle that may be used to unregister the callback.
  /// \remark If this `Future` corresponds to a deferred operation, it may be
  ///     necessary to call `Force()` directly or indirectly in order to ensure
  ///     the registered callback is ever actually invoked.
  template <class Callback>
  FutureCallbackRegistration ExecuteWhenReady(Callback&& callback) && {
    if (!rep_->ready()) {
      using Impl =
          internal_future::ReadyCallback<T, internal::remove_cvref_t<Callback>>;
      return internal_future::FutureAccess::Construct<
          FutureCallbackRegistration>(rep_->RegisterReadyCallback(
          new Impl(rep_.release(), std::forward<Callback>(callback))));
    }
    std::forward<Callback>(callback)(ReadyFuture<T>(std::move(*this)));
    return FutureCallbackRegistration();
  }

  template <class Callback>
  FutureCallbackRegistration ExecuteWhenReady(Callback&& callback) const& {
    // Call the rvalue-reference overload on a copy of `*this` (with the same
    // shared state).
    return Future<T>(*this).ExecuteWhenReady(std::forward<Callback>(callback));
  }

  /// Calls `Force()`, waits for the result to be ready, and returns a reference
  /// to the result.
  ///
  /// \dchecks `valid()`
  std::add_lvalue_reference_t<result_type> result() const {
    this->Wait();
    return rep().result;
  }

  /// Equivalent to `result().value()`.
  std::add_lvalue_reference_t<T> value() const { return result().value(); }

  /// Makes `Future<T>` model the `Sender<Status, T>` concept.
  ///
  /// The `set_value`, `set_error` or `set_cancel` function is called on the
  /// specified `receiver` once the future becomes ready.  It is valid to call
  /// `submit` multiple times on the same `Future`.
  template <typename Receiver>
  friend std::void_t<decltype(execution::set_value, std::declval<Receiver&>(),
                              std::declval<T>()),
                     decltype(execution::set_error, std::declval<Receiver&>(),
                              std::declval<Status>()),
                     decltype(execution::set_cancel, std::declval<Receiver&>())>
  submit(const Future& future, Receiver receiver) {
    struct Callback {
      Receiver receiver;
      void operator()(ReadyFuture<T> future) {
        execution::submit(future.result(), std::move(receiver));
      }
    };
    future.Force();
    future.ExecuteWhenReady(Callback{std::move(receiver)});
  }

 private:
  explicit Future(internal_future::FutureStatePointer rep)
      : rep_(std::move(rep)) {}
  friend class internal_future::FutureAccess;
  constexpr SharedState& rep() const {
    ABSL_ASSERT(rep_);
    return static_cast<SharedState&>(*rep_);
  }
  internal_future::FutureStatePointer rep_;
};

/// Returns `true` if both futures refer to the same shared state, or are both
/// invalid.
template <typename T, typename U>
inline bool HaveSameSharedState(const Future<T>& a, const Future<U>& b) {
  return internal_future::FutureAccess::rep_pointer(a).get() ==
         internal_future::FutureAccess::rep_pointer(b).get();
}

template <typename T, typename U>
inline bool HaveSameSharedState(const Promise<T>& a, const Future<U>& b) {
  return internal_future::FutureAccess::rep_pointer(a).get() ==
         internal_future::FutureAccess::rep_pointer(b).get();
}

template <typename T, typename U>
inline bool HaveSameSharedState(const Future<T>& a, const Promise<U>& b) {
  return internal_future::FutureAccess::rep_pointer(a).get() ==
         internal_future::FutureAccess::rep_pointer(b).get();
}

template <typename T, typename U>
inline bool HaveSameSharedState(const Promise<T>& a, const Promise<U>& b) {
  return internal_future::FutureAccess::rep_pointer(a).get() ==
         internal_future::FutureAccess::rep_pointer(b).get();
}

/// Future that is guaranteed to be ready.
///
/// This type is effectively just a shared-ownership pointer to the result, and
/// is used as the parameter type for `ExecuteWhenReady` and `Link` callbacks.
template <typename T>
class ReadyFuture : public Future<T> {
 public:
  using result_type = typename Future<T>::result_type;

  /// Constructs an invalid ReadyFuture.
  ReadyFuture() = default;

  /// Constructs a ReadyFuture from an existing Future, which must either be
  /// invalid or ready.
  ///
  /// \dchecks `!future.valid() || future.ready()`.
  explicit ReadyFuture(Future<T> future) : Future<T>(std::move(future)) {
    if (this->valid()) {
      ABSL_ASSERT(this->Future<T>::ready());
    }
  }

  /// Constructs a ReadyFuture from an existing ReadyFuture.
  template <typename SourceT,
            std::enable_if_t<IsFutureConvertible<SourceT, T>::value>* = nullptr>
  ReadyFuture(ReadyFuture<SourceT> other) : Future<T>(std::move(other)) {}

  /// Assigns a ReadyFuture from an existing ReadyFuture.
  template <typename SourceT>
  std::enable_if_t<IsFutureConvertible<SourceT, T>::value, ReadyFuture&>
  operator=(ReadyFuture<SourceT> other) {
    Future<T>::operator=(std::move(other));
    return *this;
  }

  /// Returns a reference to the result, guaranteed not to block.
  result_type& result() const {
    return internal_future::FutureAccess::rep(*this).result;
  }

  /// Returns a reference to the value contained in the result, guaranteed not
  /// to block.
  std::add_lvalue_reference_t<T> value() const { return result().value(); }

 private:
  friend class internal_future::FutureAccess;
  explicit ReadyFuture(internal_future::FutureStatePointer rep)
      : ReadyFuture(internal_future::FutureAccess::Construct<Future<T>>(
            std::move(rep))) {}
};

/// Creates a "link", which ties a `promise` to one or more `future` objects and
/// a `callback`.
///
/// While this link remains in effect, invokes:
/// `callback(promise, ReadyFuture<FutureValue>(future)...)`
/// when all of the futures become ready.  If `future.ready()` is true upon
/// invocation of this function for all `future` objects, `callback` will be
/// invoked from the current thread before this function returns.
///
/// Additionally, forcing the future associated with `promise` will result in
/// all of the `future` objects being forced.
///
/// If `promise.result_needed()` becomes `false`, the link is automatically
/// removed.
///
/// \param callback The function to be called when the `future` objects are
///     ready.  This function will be invoked either from the current thread,
///     before this function returns, or from the thread that causes the last
///     `future` to be ready.  It must not throw exceptions, and in general it
///     should not block or take a long time to execute.  The return value is
///     ignored.
/// \param promise The promise to be linked.
/// \param future The futures to be linked.
/// \returns A FutureCallbackRegistration handle that can be used to remove this
///     link.
/// \remark A common use case is to call `promise.SetResult` within the callback
///     function, but this is not required.
template <typename Callback, typename PromiseValue, typename... FutureValue>
std::enable_if_t<std::is_invocable_v<Callback, Promise<PromiseValue>,
                                     ReadyFuture<FutureValue>...>,
                 FutureCallbackRegistration>
Link(Callback&& callback, Promise<PromiseValue> promise,
     Future<FutureValue>... future) {
  return internal_future::FutureAccess::Construct<FutureCallbackRegistration>(
      internal_future::MakeLink<internal_future::FutureLinkAllReadyPolicy>(
          std::forward<Callback>(callback), std::move(promise),
          std::move(future)...));
}

/// Same as `Link`, except that the `callback` is called only if the `future`
/// objects become ready with a non-error result.  The first error result
/// encountered among the `future` objects will be automatically propagated to
/// the `promise`.
template <typename Callback, typename PromiseValue, typename... FutureValue>
std::enable_if_t<std::is_invocable_v<Callback, Promise<PromiseValue>,
                                     ReadyFuture<FutureValue>...>,
                 FutureCallbackRegistration>
LinkValue(Callback&& callback, Promise<PromiseValue> promise,
          Future<FutureValue>... future) {
  return internal_future::FutureAccess::Construct<FutureCallbackRegistration>(
      internal_future::MakeLink<
          internal_future::FutureLinkPropagateFirstErrorPolicy>(
          std::forward<Callback>(callback), std::move(promise),
          std::move(future)...));
}

/// Same as `LinkValue`, except that no callback function is called in the case
/// that all `future` objects are successfully resolved.
template <typename PromiseValue, typename... FutureValue>
FutureCallbackRegistration LinkError(Promise<PromiseValue> promise,
                                     Future<FutureValue>... future) {
  return LinkValue(internal_future::NoOpCallback{}, std::move(promise),
                   std::move(future)...);
}

/// Creates a link that moves a future result to a promise.
///
/// While this link remains in effect, invokes:
/// `promise.SetResult(future.result())` when `future` becomes ready.  If
/// `future.ready()` is true upon invocation of this function,
/// `promise.SetResult(future.result())` will be invoked from the current thread
/// before this function returns.
///
/// Additionally, forcing the future associated with `promise` will result in
/// the `future` object being forced.
///
/// If `promise.result_needed()` becomes `false`, the link is automatically
/// removed.
///
/// \param promise The promise to be linked.
/// \param future The futures to be linked.
/// \returns A FutureCallbackRegistration handle that can be used to remove this
///     link.
template <typename PromiseValue, typename FutureValue>
std::enable_if_t<
    std::is_constructible_v<internal_future::ResultType<PromiseValue>,
                            internal_future::ResultType<FutureValue>>,
    FutureCallbackRegistration>
Link(Promise<PromiseValue> promise, Future<FutureValue> future) {
  return Link(
      [](Promise<PromiseValue> promise, ReadyFuture<FutureValue> future) {
        promise.SetResult(std::move(future.result()));
      },
      std::move(promise), std::move(future));
}

/// Pairs a Promise with a Future.
///
/// This is the primary interface intended to be used by producers.
///
/// \tparam T The contained value type.  The actual result type of the Promise
///     and Future is `Result<T>`.
template <typename T>
class PromiseFuturePair {
  using StateType = internal_future::FutureState<T>;

 public:
  using PromiseType = Promise<T>;
  using FutureType = Future<T>;
  PromiseType promise;
  FutureType future;

  /// Makes a new Promise/Future pair.
  ///
  /// The result is initialized using `result_init...`.  The Future resolves to
  /// this initial result if the result is not set to a different value (e.g. by
  /// calling `Promise::SetResult`) before the last Promise reference is
  /// released.
  template <typename... ResultInit>
  static std::enable_if_t<
      (sizeof...(ResultInit) > 0 &&
       std::is_constructible<Result<T>, ResultInit...>::value),
      PromiseFuturePair>
  Make(ResultInit&&... init) {
    auto* state = new StateType(std::forward<ResultInit>(init)...);
    return MakeFromState(state);
  }

  /// Makes a new Promise/Future pair, with the result initialized to an error
  /// status of `absl::StatusCode::kUnknown` (which can be overridden by a call
  /// to `Promise::SetResult`).
  static PromiseFuturePair Make() { return MakeFromState(new StateType()); }

  /// Creates a new PromiseFuturePair and links the newly created promise with
  /// the specified future objects as if by calling `tensorstore::Link`.
  ///
  /// Equivalent to:
  ///
  ///     auto pair = PromiseFuturePair<T>::Make();
  ///     tensorstore::Link(callback, pair.promise, future...);
  ///     return pair;
  ///
  /// except that only a single allocation is used for both the promise/future
  /// pair and the link.
  ///
  /// \param callback The callback to be invoked with the newly created
  ///     `promise` and the specified `future` objects once they become ready.
  /// \param future The future objects to link.
  /// \returns The promise/future pair.
  template <typename Callback, typename... FutureValue>
  static std::enable_if_t<
      std::is_invocable_v<Callback, PromiseType, ReadyFuture<FutureValue>...>,
      PromiseFuturePair>
  Link(Callback&& callback, Future<FutureValue>... future) {
    return MakeFromState(
        internal_future::MakeLinkedFutureState<
            internal_future::FutureLinkAllReadyPolicy, T,
            FutureValue...>::Make(std::move(future)...,
                                  std::forward<Callback>(callback)));
  }

  /// Same as the `Link` function defined above, but initializes the Future
  /// result using `result_init`.
  template <typename ResultInit, typename Callback, typename... FutureValue>
  static std::enable_if_t<
      std::is_invocable_v<Callback, PromiseType, ReadyFuture<FutureValue>...> &&
          std::is_constructible_v<Result<T>, ResultInit>,
      PromiseFuturePair>
  Link(ResultInit&& result_init, Callback&& callback,
       Future<FutureValue>... future) {
    return MakeFromState(
        internal_future::MakeLinkedFutureState<
            internal_future::FutureLinkAllReadyPolicy, T,
            FutureValue...>::Make(std::move(future)...,
                                  std::forward<Callback>(callback),
                                  std::forward<ResultInit>(result_init)));
  }

  /// Same as `Link`, except that the behavior matches `tensorstore::LinkValue`
  /// instead of `tensorstore::Link`.
  ///
  /// Equivalent to:
  ///
  ///     auto pair = PromiseFuturePair<T>::Make();
  ///     tensorstore::LinkValue(callback, pair.promise, future...);
  ///     return pair;
  ///
  /// \returns The promise/future pair.
  template <typename Callback, typename... FutureValue>
  static std::enable_if_t<
      std::is_invocable_v<Callback, PromiseType, ReadyFuture<FutureValue>...>,
      PromiseFuturePair>
  LinkValue(Callback&& callback, Future<FutureValue>... future) {
    return MakeFromState(
        internal_future::MakeLinkedFutureState<
            internal_future::FutureLinkPropagateFirstErrorPolicy, T,
            FutureValue...>::Make(std::move(future)...,
                                  std::forward<Callback>(callback)));
  }

  /// Same as `LinkValue` function defined above, but initializes the Future
  /// result using `result_init`.
  template <typename ResultInit, typename Callback, typename... FutureValue>
  static std::enable_if_t<
      std::is_invocable_v<Callback, PromiseType, ReadyFuture<FutureValue>...> &&
          std::is_constructible_v<Result<T>, ResultInit>,
      PromiseFuturePair>
  LinkValue(ResultInit&& result_init, Callback&& callback,
            Future<FutureValue>... future) {
    return MakeFromState(
        internal_future::MakeLinkedFutureState<
            internal_future::FutureLinkPropagateFirstErrorPolicy, T,
            FutureValue...>::Make(std::move(future)...,
                                  std::forward<Callback>(callback),
                                  std::forward<ResultInit>(result_init)));
  }

  /// Creates a new PromiseFuturePair with the Future result initialized using
  /// `result_init`.  Links the specified `future` objects to the newly created
  /// promise as if by calling `tensorstore::LinkError`.
  ///
  /// Equivalent to:
  ///
  ///     auto pair = PromiseFuturePair<T>::Make();
  ///     pair.raw_result() = initial_result;
  ///     tensorstore::LinkError(pair.promise, future...);
  ///     return pair;
  ///
  /// \returns The promise/future pair.
  template <typename ResultInit, typename... FutureValue>
  static std::enable_if_t<std::is_constructible<Result<T>, ResultInit>::value,
                          PromiseFuturePair>
  LinkError(ResultInit&& result_init, Future<FutureValue>... future) {
    return LinkValue(std::forward<ResultInit>(result_init),
                     internal_future::NoOpCallback{}, std::move(future)...);
  }

 private:
  static PromiseFuturePair MakeFromState(StateType* state) {
    return {internal_future::FutureAccess::Construct<PromiseType>(
                internal_future::PromiseStatePointer(
                    state, internal::adopt_object_ref)),
            internal_future::FutureAccess::Construct<FutureType>(
                internal_future::FutureStatePointer(
                    state, internal::adopt_object_ref))};
  }
};

/// Returns a `Future` that resolves to the result of calling
/// `callback(future.result()...)` when all of the specified `future` objects
/// become ready.  The `callback` is invoked using the specified `executor`.
///
/// \param executor Executor with which to invoke the `callback` callback.
/// \param callback Callback function to be invoked as
///     `callback(future.result()...)` when all of the `future` objects become
///     ready.  The callback is invoked immediately if all of the `future`
///     objects are already ready.
/// \param future The `Future` objects to link.
/// \dchecks `future.valid() && ...`
/// \returns A `Future<UnwrapResultType<remove_cvref_t<U>>>`, where `U` is the
///     return type of the specified `callback` function.
template <typename Executor, typename Callback, typename... FutureValue>
Future<UnwrapResultType<internal::remove_cvref_t<std::invoke_result_t<
    Callback, internal_future::ResultType<FutureValue>&...>>>>
MapFuture(Executor&& executor, Callback&& callback,
          Future<FutureValue>... future) {
  using PromiseValue =
      UnwrapResultType<internal::remove_cvref_t<std::invoke_result_t<
          Callback, internal_future::ResultType<FutureValue>&...>>>;
  struct SetPromiseFromCallback {
    internal::remove_cvref_t<Callback> callback;
    void operator()(Promise<PromiseValue> promise,
                    Future<FutureValue>... future) {
      promise.SetResult(callback(future.result()...));
    }
  };
  return PromiseFuturePair<PromiseValue>::Link(
             WithExecutor(
                 std::forward<Executor>(executor),
                 SetPromiseFromCallback{std::forward<Callback>(callback)}),
             std::move(future)...)
      .future;
}

/// Returns a `Future` that resolves to `callback(future.value()...)` when all
/// of the specified `future` objects become ready with non-error results.  The
/// `callback` is invoked using the specified `executor`.
///
/// If any of the `future` objects become ready with an error result, the error
/// propagates to the returned `Future`.
///
/// Example:
///
///     Future<int> future_a = ...;
///     Future<int> future_b = ...;
///     Future<int> mapped_future = MapFutureValue(
///         InlineExecutor{}, [](int a, int b) { return a + b; },
///         future_a, future_b);
///
/// \param executor Executor with which to invoke the `callback` callback.
/// \param callback Callback function to be invoked as
///     `callback(future.result().value()...)` when all of the `future` objects
///     become ready with non-error results.
/// \param future The `Future` objects to link.
/// \dchecks `future.valid() && ...`
/// \returns A `Future<UnwrapResultType<remove_cvref_t<U>>>`, where `U` is the
///     return type of the specified `callback` function.
template <typename Executor, typename Callback, typename... FutureValue>
Future<UnwrapResultType<
    internal::remove_cvref_t<std::invoke_result_t<Callback, FutureValue&...>>>>
MapFutureValue(Executor&& executor, Callback&& callback,
               Future<FutureValue>... future) {
  using PromiseValue = UnwrapResultType<internal::remove_cvref_t<
      std::invoke_result_t<Callback, FutureValue&...>>>;
  struct SetPromiseFromCallback {
    internal::remove_cvref_t<Callback> callback;
    void operator()(Promise<PromiseValue> promise,
                    Future<FutureValue>... future) {
      promise.SetResult(callback(future.result().value()...));
    }
  };
  return PromiseFuturePair<PromiseValue>::LinkValue(
             WithExecutor(
                 std::forward<Executor>(executor),
                 SetPromiseFromCallback{std::forward<Callback>(callback)}),
             std::move(future)...)
      .future;
}

/// Transforms the error status of a Future.
///
/// Example:
///
///     Future<int> future = ...;
///     auto mapped_future = MapFutureError(
///         InlineExecutor{},
///         [](const Status& status) -> Result<int> {
///           return status.Annotate("Error doing xxx");
///         }, future);
///
/// \param executor Executor to use to run `func`.
/// \param func Unary function to apply to the error status of `future`.  Must
///     have a signature compatible with `Result<T>(Status)`.
/// \param future The future to transform.
/// \returns A future that becomes ready when `future` is ready.  If
///     `future.result()` is in a success state, the result of the returned
///     future is a copy of `future.result()`.  Otherwise, the result is equal
///     to `func(future.result().status())`.
template <typename Executor, typename T, typename Func>
std::enable_if_t<
    std::is_convertible_v<std::invoke_result_t<Func&&, Status>, Result<T>>,
    Future<T>>
MapFutureError(Executor&& executor, Func func, Future<T> future) {
  struct Callback : public std::tuple<Func> {
    Result<T> operator()(Result<T> result) {
      if (result) return result;
      return std::get<0>(static_cast<std::tuple<Func>&&>(*this))(
          result.status());
    }
  };
  return MapFuture(std::forward<Executor>(executor),
                   Callback{std::make_tuple(std::move(func))},
                   std::move(future));
}

/// Converts an arbitrary `Sender<Status, T>` into a `Future<T>`.
template <typename T, typename Sender>
Future<T> MakeSenderFuture(Sender sender) {
  auto pair = PromiseFuturePair<T>::Make();
  struct Callback {
    Sender sender;
    void operator()(Promise<T> promise) {
      execution::submit(sender, std::move(promise));
    }
  };
  pair.promise.ExecuteWhenForced(Callback{std::move(sender)});
  return pair.future;
}

/// If `promise` does not already have a result set, sets its result to `result`
/// and sets `promise.result_needed() = false`.
///
/// This does not cause `promise.ready()` to become `true`.
template <typename T, typename U>
void SetDeferredResult(const Promise<T>& promise, U&& result) {
  if (internal_future::FutureAccess::rep(promise).LockResult()) {
    promise.raw_result() = std::forward<U>(result);
  }
}

/// Waits for the future to be ready and returns the status.
template <typename T>
inline absl::Status GetStatus(const Future<T>& future) {
  return tensorstore::GetStatus(future.result());
}

}  // namespace tensorstore

#endif  // TENSORSTORE_FUTURE_H_
