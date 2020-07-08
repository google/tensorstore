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

#include "python/tensorstore/result_type_caster.h"
#include "python/tensorstore/status.h"
#include "pybind11/pybind11.h"
#include "tensorstore/internal/logging.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace internal_python {

/// Throws an exception that maps to the Python `asyncio.CancelledError`.
void ThrowCancelledError();

/// Returns a new Python object of type `asyncio.CancelledError`.
pybind11::object GetCancelledError();

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
/// If an operating system signal results in a Python signal handler throwing an
/// exception (e.g. KeyboardInterrupt), this function stops waiting immediately
/// and throws `pybind11::error_already_set`.
///
/// Otherwise, this function waits until `notify_done` is called, and returns
/// normally.
///
/// This function factors out the type-independent, platform-dependent logic
/// from the `PythonFuture<T>::WaitForResult` method defined below.
void InterruptibleWaitImpl(
    std::function<FutureCallbackRegistration(std::function<void()> notify_done)>
        register_listener);

/// Waits for the Future to be ready, but supports interruption by operating
/// system signals.
///
/// This allows the user to use Control+C to stop waiting on "stuck"
/// asynchronous operations.
///
/// We can't simply use the normal `tensorstore::Future<T>::Wait` method, since
/// that does not support interruption.
template <typename T>
typename Future<T>::result_type& InterruptibleWait(const Future<T>& future) {
  assert(future.valid());
  if (!future.ready() && _PyOS_IsMainThread()) {
    // If on main thread and not already ready, use "interruptible" wait that
    // may throw a KeyboardInterrupt exception if SIGINT is received.
    internal_python::InterruptibleWaitImpl([&](auto signal) {
      return future.ExecuteWhenReady(
          [signal = std::move(signal)](ReadyFuture<const T> f) { signal(); });
    });
  }
  return future.result();
}

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
  /// Returns `true` if the underlying Future is ready (either with a value or
  /// an error) or already cancelled.
  virtual bool done() const = 0;

  /// Waits for the Future to be done (interruptible by `KeyboardInterrupt`).
  /// If it has finished with a value, returns `None`.  Otherwise, returns the
  /// Python exception object representing the error.
  virtual pybind11::object exception() = 0;

  /// Waits for the Future to be done (interruptible by `KeyboardInterrupt`).
  /// Returns the value if the Future completed successfully, otherwise throws
  /// an exception that maps to the corresponding Python exception.
  virtual pybind11::object result() = 0;

  /// Returns `true` if the Future completed with
  /// `absl::StatusCode::kCancelled`.
  virtual bool cancelled() const = 0;

  /// Attempts to cancel the `Future`.  Returns `true` if the `Future` is not
  /// already done.  It is possible that any computation corresponding to the
  /// Future may still continue, however.
  virtual bool cancel() = 0;

  /// Adds a nullary callback to be invoked when the Future is done.
  virtual void add_done_callback(pybind11::object callback) = 0;

  /// Removes any previously-registered callback identical to `callback`.
  /// Returns the number of callbacks removed.
  virtual std::size_t remove_done_callback(pybind11::object callback) = 0;

  /// Returns a corresponding `asyncio`-compatible future object.
  pybind11::object get_await_result();

  virtual ~PythonFutureBase();
};

/// Special type capable of holding any Python value or exception.  This is used
/// as the result type for `Promise`/`Future` pairs created by Python.
struct PythonValueOrException {
  pybind11::object value;
  pybind11::object error_type;
  pybind11::object error_value;
  pybind11::object error_traceback;
};

template <typename T>
class PythonFuture : public PythonFutureBase {
 public:
  PythonFuture(Future<const T> future) : future_(std::move(future)) {}

  bool done() const override { return !future_.valid() || future_.ready(); }

  bool cancelled() const override { return !future_.valid(); }

  const Result<T>& WaitForResult() {
    return internal_python::InterruptibleWait(future_);
  }

  pybind11::object exception() override {
    if (!future_.valid()) return GetCancelledError();
    auto& result = WaitForResult();
    if (result.has_value()) {
      if constexpr (std::is_same_v<T, PythonValueOrException>) {
        if (!result->value.ptr()) {
          return result->error_value;
        }
      }
      return pybind11::none();
    }
    return GetStatusPythonException(result.status());
  }

  pybind11::object result() override {
    if (!future_.valid()) ThrowCancelledError();
    auto& result = WaitForResult();
    return pybind11::cast(result);
  }

  bool cancel() override {
    if (!future_.valid() || future_.ready()) {
      return false;
    }
    future_ = Future<const T>{};
    registration_.Unregister();
    RunCallbacks();
    return true;
  }

  void RunCallbacks() {
    auto callbacks = std::move(callbacks_);
    auto py_self = pybind11::cast(shared_from_this());
    for (const auto& callback : callbacks) {
      try {
        callback(py_self);
      } catch (pybind11::error_already_set& e) {
        e.restore();
        PyErr_WriteUnraisable(nullptr);
        PyErr_Clear();
      } catch (...) {
        TENSORSTORE_LOG("Unexpected exception thrown by python callback");
      }
    }
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
      future_.Force();
      registration_ = future_.ExecuteWhenReady([self](Future<const T> future) {
        pybind11::gil_scoped_acquire gil_acquire;
        self->RunCallbacks();
      });
    }
  }

  std::size_t remove_done_callback(pybind11::object callback) override {
    auto it = std::remove_if(
        callbacks_.begin(), callbacks_.end(),
        [&](pybind11::handle h) { return h.ptr() == callback.ptr(); });
    const size_t num_removed = callbacks_.end() - it;
    callbacks_.erase(it, callbacks_.end());
    if (callbacks_.empty()) {
      registration_.Unregister();
    }
    return num_removed;
  }

  ~PythonFuture() override = default;

 private:
  Future<const T> future_;
  std::vector<pybind11::object> callbacks_;
  FutureCallbackRegistration registration_;
};

void RegisterFutureBindings(pybind11::module m);

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

  PYBIND11_TYPE_CASTER(FutureType, _("Future[") + value_conv::name + _("]"));

  static handle cast(const FutureType& future, return_value_policy policy,
                     handle parent) {
    return pybind11::cast(
               std::shared_ptr<tensorstore::internal_python::PythonFutureBase>(
                   std::make_shared<tensorstore::internal_python::PythonFuture<
                       std::remove_const_t<T>>>(future)))
        .release();
  }
};

/// Defines automatic mapping of
/// `tensorstore::internal_python::PythonValueOrException` to the contained
/// Python value or exception.
template <>
struct type_caster<tensorstore::internal_python::PythonValueOrException> {
  PYBIND11_TYPE_CASTER(tensorstore::internal_python::PythonValueOrException,
                       _("Any"));
  static handle cast(
      tensorstore::internal_python::PythonValueOrException result,
      return_value_policy policy, handle parent) {
    if (!result.value.ptr()) {
      ::PyErr_Restore(result.error_type.release().ptr(),
                      result.error_value.release().ptr(),
                      result.error_traceback.release().ptr());
      throw error_already_set();
    }
    return result.value.release();
  }
};

}  // namespace detail
}  // namespace pybind11

#endif  // THIRD_PARTY_PY_TENSORSTORE_FUTURE_H_
