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

#ifndef THIRD_PARTY_PY_TENSORSTORE_FUTURE_H_
#define THIRD_PARTY_PY_TENSORSTORE_FUTURE_H_

/// \file
///
/// Defines the `tensorstore.Future` and `tensorstore.Promise` Python classes
/// (as wrappers around `tensorstore::Future` and `tensorstore::Promise`,
/// respectively).
///
/// This is used to expose all of the Future-based TensorStore APIs to Python.

#include <algorithm>
#include <functional>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include "absl/functional/function_ref.h"
#include "python/tensorstore/gil_safe.h"
#include "python/tensorstore/python_value_or_exception.h"
#include "python/tensorstore/result_type_caster.h"
#include "python/tensorstore/status.h"
#include "python/tensorstore/type_name_override.h"
#include "pybind11/pybind11.h"
#include "tensorstore/internal/intrusive_linked_list.h"
#include "tensorstore/internal/logging.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_python {

/// Throws an exception that maps to the Python `asyncio.CancelledError`.
[[noreturn]] void ThrowCancelledError();

/// Throws an exception that maps to the Python `TimeoutError`.
[[noreturn]] void ThrowTimeoutError();

/// Returns a new Python object of type `asyncio.CancelledError`.
pybind11::object GetCancelledError();

/// Converts an optional timeout or deadline in seconds to an `absl::Time`
/// deadline.
///
/// If neither `timeout` nor `deadline` is specified, returns
/// `absl::InfiniteFuture()`.
absl::Time GetWaitDeadline(std::optional<double> timeout,
                           std::optional<double> deadline);

/// Type intended for use as a pybind11 function parameter type.
///
/// It simply holds a `pybind11::object` but displays as
/// `tensorstore.FutureLike[T]`.
template <typename T>
struct FutureLike {
  pybind11::object value;

  constexpr static auto tensorstore_pybind11_type_name_override =
      pybind11::detail::_("tensorstore.FutureLike[") +
      pybind11::detail::make_caster<T>::name + pybind11::detail::_("]");
};

/// Type intended for use as a pybind11 function parameter type.
///
/// It simply holds a `pybind11::object` but displays as
/// `tensorstore.FutureLike`.
struct UntypedFutureLike {
  pybind11::object value;

  constexpr static auto tensorstore_pybind11_type_name_override =
      pybind11::detail::_("tensorstore.FutureLike");
};

/// Returns the current thread's asyncio event loop.
///
/// If there is none, or an error occurs, returns None.
///
/// Never throws an exception or sets the Python error indicator.
pybind11::object GetCurrentThreadAsyncioEventLoop();

struct AbstractEventLoopParameter {
  pybind11::object value;

  constexpr static auto tensorstore_pybind11_type_name_override =
      pybind11::detail::_("asyncio.AbstractEventLoop");
};

/// Base class that represents a Future exposed to Python.
///
/// This provides an interface similar to the `concurrent.futures.Future` Python
/// type, but also is directly compatible with asyncio (via `await`).
///
/// The value type is erased.
///
/// For each concrete value type `T`, the actual run time type is the derived
/// class `PythonFuture<T>`.
class PythonFutureBase : public std::enable_shared_from_this<PythonFutureBase> {
 public:
  PythonFutureBase();

  /// Returns `true` if the underlying Future is ready (either with a value or
  /// an error) or already cancelled.
  virtual bool done() const = 0;

  /// Calls `Force` on the underlying `Future`.
  virtual void force() = 0;

  /// Waits for the Future to be done (interruptible by `KeyboardInterrupt`).
  /// If it has finished with a value, returns `None`.  Otherwise, returns the
  /// Python exception object representing the error.
  ///
  /// If the deadline is exceeded, raises `TimeoutError`.
  virtual pybind11::object exception(absl::Time deadline) = 0;

  /// Waits for the Future to be done (interruptible by `KeyboardInterrupt`).
  /// Returns the value if the Future completed successfully, otherwise throws
  /// an exception that maps to the corresponding Python exception.
  ///
  /// If the deadline is exceeded, raises `TimeoutError`.
  virtual pybind11::object result(absl::Time deadline) = 0;

  /// Returns `true` if the Future was cancelled.
  virtual bool cancelled() const = 0;

  /// Attempts to cancel the `Future`.  Returns `true` if the `Future` is not
  /// already done.  It is possible that any computation corresponding to the
  /// Future may still continue, however.
  virtual bool cancel() = 0;

  /// Adds a nullary callback to be invoked when the Future is done.
  virtual void add_done_callback(pybind11::object callback) = 0;

  /// Removes any previously-registered callback identical to `callback`.
  /// Returns the number of callbacks removed.
  virtual std::size_t remove_done_callback(pybind11::object callback);

  /// Returns a corresponding `asyncio`-compatible future object.
  pybind11::object get_await_result();

  /// Returns a Future that resolves directly to the Python value or exception.
  virtual Future<const GilSafePythonValueOrException>
  GetPythonValueOrExceptionFuture() = 0;

  virtual ~PythonFutureBase();

  struct CancelCallbackBase {
    CancelCallbackBase* next;
    CancelCallbackBase* prev;
  };

  struct CancelCallback : public CancelCallbackBase {
    using Accessor =
        internal::intrusive_linked_list::MemberAccessor<CancelCallbackBase>;
    explicit CancelCallback(PythonFutureBase* base,
                            absl::FunctionRef<void()> callback)
        : callback(callback) {
      internal::intrusive_linked_list::InsertBefore(
          Accessor{}, &base->cancel_callbacks_, this);
    }
    ~CancelCallback() {
      internal::intrusive_linked_list::Remove(Accessor{}, this);
    }
    absl::FunctionRef<void()> callback;
  };

 protected:
  void RunCallbacks();
  void RunCancelCallbacks();

  /// Callbacks to be invoked when the future becomes ready.  Guarded by the
  /// GIL.
  std::vector<pybind11::object> callbacks_;
  /// Registration of `ExecuteWhenReady` callback used when `callbacks_` is
  /// non-empty.  Guarded by the GIL.
  FutureCallbackRegistration registration_;
  /// Linked list of callbacks to be invoked when cancelled.  Guarded by the
  /// GIL.
  CancelCallbackBase cancel_callbacks_;
};

/// Waits for an event to occur, but supports interruption due to a Python
/// signal handler throwing a Python exception.
///
/// Invokes the specified `register_listener` function, passing in a
/// `notify_done` callback that the `register_listener` function should either:
///
/// 1. call immediately (if the event has already occurred), or;
///
/// 2. arrange for another thread to call asynchronously when the event occurs.
///
/// In either case, `register_listener` must return a
/// `FutureCallbackRegistration` that can be used to cancel the registration of
/// the `notify_done` callback.
///
/// The following events terminate the wait:
///
/// 1. If `notify_done` is called, this function returns normally.
///
/// 2. If an operating system signal results in a Python signal handler throwing
///    an exception (e.g. KeyboardInterrupt), this function stops waiting
///    immediately and throws `pybind11::error_already_set`.
///
/// 3. If the deadline is reached, this functions throws
///    `pybind11::error_already_set`, with a `TimeoutError` set.
///
/// 4. If `python_future` is non-null and is cancelled, this function throws
///    `pybind11::error_already_set`, with an `asyncio.CancelledError` set.
///
/// This function factors out the type-independent, platform-dependent logic
/// from the `PythonFuture<T>::WaitForResult` method defined below.
void InterruptibleWaitImpl(absl::FunctionRef<FutureCallbackRegistration(
                               absl::FunctionRef<void()> notify_done)>
                               register_listener,
                           absl::Time deadline,
                           PythonFutureBase* python_future);

/// Waits for the Future to be ready, but supports interruption by operating
/// system signals.
///
/// This allows the user to use Control+C to stop waiting on "stuck"
/// asynchronous operations.
///
/// We can't simply use the normal `tensorstore::Future<T>::Wait` method, since
/// that does not support interruption or cancellation.
///
/// \pre GIL must be held.
template <typename T>
typename Future<T>::result_type& InterruptibleWait(
    const Future<T>& future, absl::Time deadline = absl::InfiniteFuture(),
    PythonFutureBase* python_future = nullptr) {
  assert(future.valid());
  if (!future.ready()) {
    {
      GilScopedRelease gil_release;
      future.Force();
    }
    internal_python::InterruptibleWaitImpl(
        [&](auto signal) {
          return future.ExecuteWhenReady(
              [signal = std::move(signal)](ReadyFuture<const T> f) {
                signal();
              });
        },
        deadline, python_future);
  }
  return future.result();
}

template <typename T>
class PythonFuture : public PythonFutureBase {
 public:
  PythonFuture(Future<const T> future) : future_(std::move(future)) {}

  bool done() const override { return !future_.valid() || future_.ready(); }

  void force() override {
    if (!done()) {
      // Use copy of `future_`, since `future_` may be modified by another
      // thread calling `PythonFuture::cancel` once GIL is released.
      auto future_copy = future_;
      GilScopedRelease gil_release;
      future_copy.Force();
    }
  }

  bool cancelled() const override { return !future_.valid(); }

  Future<const T> WaitForResult(absl::Time deadline) {
    // Copy `future_`, since `future_` may be modified by another threading
    // calling `PythonFuture::cancel` once GIL is released.
    auto future_copy = future_;
    internal_python::InterruptibleWait(future_copy, deadline, this);
    return future_copy;
  }

  pybind11::object exception(absl::Time deadline) override {
    if (!future_.valid()) return GetCancelledError();
    auto future = WaitForResult(deadline);
    auto& result = future.result();
    if (result.has_value()) {
      if constexpr (std::is_same_v<T, GilSafePythonValueOrException>) {
        if (!(**result).value.ptr()) {
          return (**result).error_value;
        }
      }
      return pybind11::none();
    }
    return GetStatusPythonException(result.status());
  }

  pybind11::object result(absl::Time deadline) override {
    if (!future_.valid()) ThrowCancelledError();
    auto future = WaitForResult(deadline);
    return pybind11::cast(future.result());
  }

  bool cancel() override {
    if (!future_.valid() || future_.ready()) {
      return false;
    }
    future_ = Future<const T>{};
    registration_.Unregister();
    RunCancelCallbacks();
    RunCallbacks();
    return true;
  }

  void add_done_callback(pybind11::object callback) override {
    if (done()) {
      callback(pybind11::cast(shared_from_this()));
      return;
    }
    callbacks_.push_back(callback);
    if (callbacks_.size() == 1) {
      registration_.Unregister();
      auto self = std::static_pointer_cast<PythonFuture<T>>(shared_from_this());
      registration_ = future_.ExecuteWhenReady([self](Future<const T> future) {
        ExitSafeGilScopedAcquire gil;
        if (gil.acquired()) {
          self->RunCallbacks();
        }
      });
      // Set up `ExecuteWhenReady` registration before calling `force`, since
      // `force` releases the GIL.
      force();
    }
  }

  Future<const GilSafePythonValueOrException> GetPythonValueOrExceptionFuture()
      override {
    if constexpr (std::is_same_v<GilSafePythonValueOrException, T>) {
      return future_;
    } else {
      return MapFuture(
          InlineExecutor{},
          [](const Result<T>& result) -> Result<GilSafePythonValueOrException> {
            if (!result.ok()) return result.status();
            ExitSafeGilScopedAcquire gil;
            if (!gil.acquired()) {
              return PythonExitingError();
            }
            // Convert `result` rather than `*result` to account
            // for `T=void`.
            return GilSafePythonValueOrException(
                PythonValueOrException::FromValue(result));
          },
          future_);
    }
  }

  ~PythonFuture() override = default;

 private:
  Future<const T> future_;
};

void RegisterFutureBindings(pybind11::module m, Executor defer);

/// Attempts to convert a `FutureLike` Python object to a `Future`.
///
/// \param src Source python object.  Supported types are: a
///     `tensorstore.Future` object, or a coroutine.
/// \param loop Python object of type `asyncio.AbstractEventLoop` or `None`.  If
///     `None`, an exception is thrown if `src` is a coroutine.  Otherwise, if
///     `src` is a coroutine, it is run using `loop`.
/// \param future Set to the future on success.
/// \returns `true` if `src` could be converted to a `Future`, or `false`
///     otherwise.  The error indicator is never set upon return.
/// \throws An exception if `src` if an error occurs in invoking `asyncio`
///     (unlikely).
///
/// If `src` resolves to an exception, the future resolves to an error.  Python
/// exceptions are stored via pickling if possible.
bool TryConvertToFuture(pybind11::handle src, pybind11::handle loop,
                        std::shared_ptr<PythonFutureBase>& future);

/// Converts a `FutureLike` Python object to a `Future` with the specified
/// result type.
///
/// \param src Source python object.  Supported types are: null pointer, an
///     immediate value convertible to `T`, a `tensorstore.Future` object, or a
///     coroutine.
/// \param loop Python object of type `asyncio.AbstractEventLoop` or `None`.  If
///     `None`, it is an error to specify a coroutine for `src`.  Otherwise, if
///     `src` is a coroutine, it is run using `loop`.
///
/// If `src` is a null pointer, the returned future resolves to the exception
/// set as the current error indicator.  (The error indicator must be set.)
///
/// Otherwise, if `src` resolves to an exception, or the result cannot be
/// converted to `T`, the future resolves to an error.  Python exceptions are
/// stored via pickling if possible.
template <typename T>
Future<T> ConvertToFuture(pybind11::handle src, pybind11::handle loop) {
  if (!src.ptr()) return internal_python::GetStatusFromPythonException();
  std::shared_ptr<PythonFutureBase> python_future_base;
  Future<T> future;
  if (CallAndSetErrorIndicator([&] {
        if (!TryConvertToFuture(src, loop, python_future_base)) {
          // Attempt to convert the value directly.
          future = pybind11::cast<T>(src);
        }
      })) {
    return internal_python::GetStatusFromPythonException();
  }
  if (future.valid()) return future;
  return MapFutureValue(
      InlineExecutor{},
      [](const GilSafePythonValueOrException& v) -> Result<T> {
        ExitSafeGilScopedAcquire gil;
        if (!gil.acquired()) return PythonExitingError();
        return Result<T>(*v);
      },
      python_future_base->GetPythonValueOrExceptionFuture());
}

}  // namespace internal_python
}  // namespace tensorstore

namespace pybind11 {
namespace detail {

/// Defines automatic mapping of `tensorstore::Future<T>` to
/// `tensorstore::internal_python::PythonFuture<T>`.
template <typename T>
struct type_caster<tensorstore::Future<T>> {
  using FutureType = tensorstore::Future<T>;
  using value_conv = make_caster<typename FutureType::result_type>;

  PYBIND11_TYPE_CASTER(FutureType,
                       _("tensorstore.Future[") + value_conv::name + _("]"));

  static handle cast(const FutureType& future, return_value_policy policy,
                     handle parent) {
    return pybind11::cast(
               std::shared_ptr<tensorstore::internal_python::PythonFutureBase>(
                   std::make_shared<tensorstore::internal_python::PythonFuture<
                       std::remove_const_t<T>>>(future)))
        .release();
  }
};

}  // namespace detail
}  // namespace pybind11

#endif  // THIRD_PARTY_PY_TENSORSTORE_FUTURE_H_
