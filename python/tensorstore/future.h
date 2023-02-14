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
///
/// Note regarding garbage collection:
///
/// `tensorstore::Future<T>` inherently supports shared ownership of the `T`
/// object.  Consequently, as noted in `garbage_collection.h`, strong references
/// to Python objects must not be held by C++ types with shared ownership.
/// Therefore, the type `T` must not hold strong references to Python objects,
/// except where the lifetime of the `tensorstore::Future` object is strictly
/// managed.
///
/// Types like `tensorstore::TensorStore<>` and `tensorstore::Spec` *are* safe
/// to use with `tensorstore::Future` because they hold only weak references to
/// Python objects (via `PythonWeakRef`).  Additionally, any type `T` used with
/// `tensorstore::Future` must be safe to destroy without holding the GIL.
///
/// To ensure weak references to Python objects remain valid,
/// `PythonFutureObject` holds a `PythonObjectReferenceManager` which maintains
/// the necessary references.  Before the associated `Future` becomes ready, it
/// holds any references needed by the asynchronous operation responsible for
/// computing the result.  For example, the `PythonFutureObject` returned by
/// `tensorstore.open` holds any references needed by the `tensorstore.Spec`.
/// Once the `Future` becomes ready, the `PythonFutureObject` discards any
/// references needed by the asynchronous operation, and instead holds any
/// references needed by the result.

#include <pybind11/pybind11.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include <algorithm>
#include <functional>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include "absl/functional/function_ref.h"
#include "python/tensorstore/define_heap_type.h"
#include "python/tensorstore/garbage_collection.h"
#include "python/tensorstore/gil_safe.h"
#include "python/tensorstore/python_value_or_exception.h"
#include "python/tensorstore/result_type_caster.h"
#include "python/tensorstore/status.h"
#include "python/tensorstore/type_name_override.h"
#include "tensorstore/internal/intrusive_linked_list.h"
#include "tensorstore/serialization/fwd.h"
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

/// Holds an `asyncio.AbstractEventLoop` object.
///
/// Since the event loop is a property of the current thread environment, it
/// does not make sense to actually serialize the event loop in any way.
/// Instead, serialization is a no-op, and deserialization just returns the
/// current thread's event loop, or None if there is none.
///
/// This is normally held via `GilSafeHolder`.
struct SerializableAbstractEventLoop {
  PythonWeakRef obj;

  constexpr static auto ApplyMembers = [](auto&& x, auto f) {
    return f(x.obj);
  };
};

/// Base class that represents a Future exposed to Python.
/// Python wrapper object type for `tensorstore::Future`.
///
/// This provides an interface similar to the `concurrent.futures.Future` Python
/// type, but also is directly compatible with asyncio (via `await`).
///
/// The value type is erased.
///
/// This class is defined using the Python C API directly rather than using
/// pybind11, because the additional indirection used by pybind11 makes it
/// difficult to manage the lifetime and garbage collection correctly.
struct PythonFutureObject {
  /// Python type object corresponding to this object type.
  ///
  /// This is initialized during the tensorstore module initialization by
  /// `RegisterFutureBindings`.
  static PyTypeObject* python_type;

  constexpr static const char python_type_name[] = "tensorstore.Future";

  /// Defines Python-related operations specific to a particular value type.
  struct Vtable {
    /// Converts a successful result to a Python object.
    ///
    /// Throws a pybind11-translatable exception if conversion fails or if the
    /// result is an error.
    ///
    /// \pre The future is ready.
    using GetResult = pybind11::object (*)(internal_future::FutureStateBase&);

    /// Converts an error result to a Python object.
    ///
    /// Returns `None` if there was a successful result.
    ///
    /// \pre The future is ready.
    using GetException =
        pybind11::object (*)(internal_future::FutureStateBase&);

    /// Maps this Future by converting the result to a Python object.
    using GetPythonValueFuture =
        Future<GilSafePythonHandle> (*)(internal_future::FutureStateBase&);

    GetResult get_result;
    GetException get_exception;
    GetPythonValueFuture get_python_value_future;
  };

  /// Base class for node in linked list of cancel callbacks.  By having this
  /// separate base class rather than just using `CancelCallback`, we avoid
  /// storing a useless `callback` in the head node.
  struct CancelCallbackBase {
    CancelCallbackBase* next;
    CancelCallbackBase* prev;
  };

  struct CancelCallback : public CancelCallbackBase {
    using Accessor =
        internal::intrusive_linked_list::MemberAccessor<CancelCallbackBase>;
    explicit CancelCallback(PythonFutureObject* base,
                            absl::FunctionRef<void()> callback)
        : callback(callback) {
      internal::intrusive_linked_list::InsertBefore(
          Accessor{}, &base->cpp_data.cancel_callbacks, this);
    }
    ~CancelCallback() {
      internal::intrusive_linked_list::Remove(Accessor{}, this);
    }
    absl::FunctionRef<void()> callback;
  };

  struct CppData {
    /// Operations specified to the value type.
    const Vtable* vtable;

    internal_future::FutureStatePointer state;
    /// Callbacks to be invoked when the future becomes ready.  Guarded by the
    /// GIL.
    std::vector<pybind11::object> callbacks;
    /// Registration of `ExecuteWhenReady` callback used when `callbacks_` is
    /// non-empty.  Guarded by the GIL.
    FutureCallbackRegistration registration;
    /// Linked list of callbacks to be invoked when cancelled.  Guarded by the
    /// GIL.
    CancelCallbackBase cancel_callbacks;
    /// Holds strong references to objects weakly referenced by either the value
    /// that has been set (if done), or by the asynchronous operation
    /// responsible for setting the value (if not yet done).
    PythonObjectReferenceManager reference_manager;
  };

  // clang-format off
  PyObject_HEAD
  CppData cpp_data;
  PyObject *weakrefs;
  // clang-format on

  void RunCallbacks();
  void RunCancelCallbacks();

  /// Attempts to cancel the `Future`.  Returns `true` if the `Future` is not
  /// already done.  It is possible that any computation corresponding to the
  /// Future may still continue, however.
  bool Cancel();

  /// Calls `Force` on the underlying `Future`.
  void Force();

  /// Returns a corresponding `asyncio`-compatible future object.
  pybind11::object GetAwaitable();

  /// Adds a nullary callback to be invoked when the Future is done.
  void AddDoneCallback(pybind11::handle callback);

  /// Removes any previously-registered callback identical to `callback`.
  /// Returns the number of callbacks removed.
  size_t RemoveDoneCallback(pybind11::handle callback);

  /// Waits for the Future to be done (interruptible by `KeyboardInterrupt`).
  /// Returns the value if the Future completed successfully, otherwise throws
  /// an exception that maps to the corresponding Python exception.
  ///
  /// If the deadline is exceeded, raises `TimeoutError`.
  pybind11::object GetResult(absl::Time deadline);

  /// Waits for the Future to be done (interruptible by `KeyboardInterrupt`).
  /// If it has finished with a value, returns `None`.  Otherwise, returns the
  /// Python exception object representing the error.
  ///
  /// If the deadline is exceeded, raises `TimeoutError`.
  pybind11::object GetException(absl::Time deadline);

  /// Returns `true` if the Future was cancelled.
  bool cancelled() { return !cpp_data.state; }

  /// Returns `true` if the underlying Future is ready (either with a value or
  /// an error) or already cancelled.
  bool done() const { return !cpp_data.state || cpp_data.state->ready(); }

  /// Returns a Future that resolves directly to the Python value.
  Future<GilSafePythonHandle> GetPythonValueFuture();

  /// Invokes the visitor on each Python object directly owned by this object,
  /// as required by the `tp_traverse` protocol.
  ///
  /// This is invoked by the `tp_traverse` method for `tensorstore.Future`,
  /// which is called by the garbage collector to determine which objects are
  /// reachable from this object.
  int TraversePythonReferences(visitproc visit, void* arg);

  /// Clears Python references directly owned by this object, as required by the
  /// `tp_clear` protocol.
  ///
  /// This is invoked by the `tp_clear` method for `tensorstore.Future`, which
  /// is called by the garbage collector to break a reference cycle that
  /// contains this object.
  ///
  /// This leaves the object in a valid state, but all callbacks are
  /// unregistered and the accessor methods will behave as if the future was
  /// cancelled.
  int ClearPythonReferences();

  /// Creates a PythonFutureObject wrapper for the given `future`.
  ///
  /// \param future The future to wrap.
  /// \param manager Specifies initial object references to hold.  These are
  ///     dropped when `future` becomes ready.
  template <typename T>
  static pybind11::object Make(Future<T> future,
                               PythonObjectReferenceManager manager = {}) {
    return MakeInternal<std::remove_cv_t<T>>(std::move(future),
                                             std::move(manager));
  }

  template <typename T>
  static pybind11::object MakeInternal(
      Future<const T> future, PythonObjectReferenceManager manager = {}) {
    static constexpr Vtable vtable = {
        /*.get_result=*/[](internal_future::FutureStateBase& state)
                            -> pybind11::object {
          return pybind11::cast(
              static_cast<internal_future::FutureStateType<T>&>(state).result);
        },
        /*.get_exception=*/
        [](internal_future::FutureStateBase& state) -> pybind11::object {
          auto& result =
              static_cast<internal_future::FutureStateType<T>&>(state).result;
          if (result.has_value()) {
            if constexpr (std::is_same_v<
                              T, GilSafePythonValueOrExceptionWeakRef>) {
              auto& value = **result;
              if (!value.value) {
                return pybind11::reinterpret_borrow<pybind11::object>(
                    value.error_value.get_value_or_none());
              }
            }
            return pybind11::none();
          }
          return GetStatusPythonException(result.status());
        },
        /*.get_python_value_future=*/
        [](internal_future::FutureStateBase& state)
            -> Future<GilSafePythonHandle> {
          return MapFuture(
              InlineExecutor{},
              [](const Result<T>& result) -> Result<GilSafePythonHandle> {
                if (!result.ok()) return result.status();
                ExitSafeGilScopedAcquire gil;
                if (!gil.acquired()) {
                  return PythonExitingError();
                }
                GilSafePythonHandle obj;

                // Convert `result` rather than `*result` to account
                // for `T=void`.
                if (internal_python::CallAndSetErrorIndicator([&] {
                      obj = GilSafePythonHandle(
                          pybind11::cast(result).release().ptr(),
                          internal::adopt_object_ref);
                    })) {
                  return internal_python::GetStatusFromPythonException();
                }
                return obj;
              },
              internal_future::FutureAccess::Construct<Future<const T>>(
                  internal_future::FutureStatePointer(&state)));
        },
    };
    assert(!future.null());
    pybind11::object self = pybind11::reinterpret_steal<pybind11::object>(
        python_type->tp_alloc(python_type, 0));
    if (!self) throw pybind11::error_already_set();
    auto& obj = *reinterpret_cast<PythonFutureObject*>(self.ptr());
    auto& cpp_data = obj.cpp_data;
    cpp_data.vtable = &vtable;
    cpp_data.state = internal_future::FutureAccess::rep_pointer(future);
    cpp_data.reference_manager = std::move(manager);
    cpp_data.registration = std::move(future).ExecuteWhenReady(
        [&obj](ReadyFuture<const T> future) mutable {
          ExitSafeGilScopedAcquire gil;
          if (!gil.acquired()) return;
          if (!obj.cpp_data.state) return;
          assert(Py_REFCNT(reinterpret_cast<PyObject*>(&obj)) > 0);
          auto keep_alive = pybind11::reinterpret_borrow<pybind11::object>(
              reinterpret_cast<PyObject*>(&obj));
          auto& r = future.result();
          if constexpr (!std::is_void_v<T>) {
            if (r.ok()) {
              obj.cpp_data.reference_manager.Update(*r);
            }
          }
          obj.RunCallbacks();
        });
    PyObject_GC_Track(self.ptr());
    return self;
  }
};

/// Python wrapper object type for `tensorstore::Promise`.
struct PythonPromiseObject {
  /// Python type object corresponding to this object type.
  ///
  /// This is initialized during the tensorstore module initialization by
  /// `RegisterFutureBindings`.
  static PyTypeObject* python_type;

  constexpr static const char python_type_name[] = "tensorstore.Promise";

  struct CppData {
    Promise<GilSafePythonValueOrExceptionWeakRef> promise;

    /// Holds strong references to objects weakly referenced by the value that
    /// has been set (if done).
    PythonObjectReferenceManager reference_manager;
  };

  // clang-format off
  PyObject_HEAD
  CppData cpp_data;
  PyObject *weakrefs;
  // clang-format on
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
void InterruptibleWaitImpl(internal_future::FutureStateBase& future,
                           absl::Time deadline,
                           PythonFutureObject* python_future);

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
    PythonFutureObject* python_future = nullptr) {
  internal_python::InterruptibleWaitImpl(
      *internal_future::FutureAccess::rep_pointer(future), deadline,
      python_future);
  return future.result();
}

/// Attempts to convert a `FutureLike` Python object to a `Future`.
///
/// \param src Source python object.  Supported types are: a
///     `tensorstore.Future` object, or a coroutine.
/// \param loop Python object of type `asyncio.AbstractEventLoop` or `None`.  If
///     `None`, an exception is thrown if `src` is a coroutine.  Otherwise, if
///     `src` is a coroutine, it is run using `loop`.
/// \returns `pybind11::object` pointing to a `PythonFutureObject` if `src`
///     could be converted to a `Future`, or `nullptr` otherwise.  The error
///     indicator is never set upon return.
/// \throws An exception if `src` if an error occurs in invoking `asyncio`
///     (unlikely).
///
/// If `src` resolves to an exception, the future resolves to an error.  Python
/// exceptions are stored via pickling if possible.
pybind11::object TryConvertToFuture(pybind11::handle src,
                                    pybind11::handle loop);

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
  pybind11::object python_future;
  Future<T> future;
  if (CallAndSetErrorIndicator([&] {
        if (!(python_future = TryConvertToFuture(src, loop))) {
          // Attempt to convert the value directly.
          future = pybind11::cast<T>(src);
        }
      })) {
    return internal_python::GetStatusFromPythonException();
  }
  if (!future.null()) return future;
  auto python_value_future =
      reinterpret_cast<PythonFutureObject*>(python_future.ptr())
          ->GetPythonValueFuture();

  if constexpr (std::is_same_v<T, GilSafePythonHandle>) {
    return python_value_future;
  } else {
    return MapFutureValue(
        InlineExecutor{},
        [](const GilSafePythonHandle& v) -> Result<T> {
          ExitSafeGilScopedAcquire gil;
          if (!gil.acquired()) return PythonExitingError();
          Result<T> obj;
          if (internal_python::CallAndSetErrorIndicator([&] {
                obj = pybind11::cast<T>(pybind11::handle(v.get()));
              })) {
            obj = GetStatusFromPythonException();
          }
          return obj;
        },
        std::move(python_value_future));
  }
}

/// Wrapper that holds a `pybind11::object` but which displays in
/// pybind11-generated type signatures as `tensorstore.Future[T]`.
///
/// Provides convenient interface for creating a newly-allocated
/// `PythonFutureObject`.
template <typename T>
struct PythonFutureWrapper {
  pybind11::object value;

  PythonFutureWrapper() = default;
  explicit PythonFutureWrapper(pybind11::object value)
      : value(std::move(value)) {}
  explicit PythonFutureWrapper(Future<const T> future,
                               PythonObjectReferenceManager manager)
      : value(PythonFutureObject::Make(std::move(future), std::move(manager))) {
  }

  constexpr static auto tensorstore_pybind11_type_name_override =
      pybind11::detail::_("tensorstore.Future[") +
      pybind11::detail::make_caster<std::conditional_t<
          std::is_void_v<T>, pybind11::detail::void_type, T>>::name +
      pybind11::detail::_("]");
};

using UntypedFutureWrapper = StaticHeapTypeWrapper<PythonFutureObject>;
using PromiseWrapper = StaticHeapTypeWrapper<PythonPromiseObject>;

}  // namespace internal_python
}  // namespace tensorstore

TENSORSTORE_DECLARE_SERIALIZER_SPECIALIZATION(
    tensorstore::internal_python::SerializableAbstractEventLoop)

namespace pybind11 {
namespace detail {

/// Defines automatic mapping of `tensorstore::Future<T>` to
/// `tensorstore::internal_python::PythonFuture<T>`.
///
/// Note that this must not be used if Python objects need to be kept alive via
/// a `PythonObjectReferenceManager` before the future becomes ready.  In that
/// case use `PythonFutureWrapper` instead.
template <typename T>
struct type_caster<tensorstore::Future<T>> {
  using FutureType = tensorstore::Future<T>;
  using value_conv = make_caster<typename FutureType::result_type>;

  PYBIND11_TYPE_CASTER(FutureType,
                       tensorstore::internal_python::PythonFutureWrapper<
                           T>::tensorstore_pybind11_type_name_override);

  static handle cast(const FutureType& future, return_value_policy policy,
                     handle parent) {
    return tensorstore::internal_python::PythonFutureObject::Make(future)
        .release();
  }
};

template <>
struct type_caster<tensorstore::internal_python::PythonFutureObject>
    : public tensorstore::internal_python::StaticHeapTypeCaster<
          tensorstore::internal_python::PythonFutureObject> {};

template <>
struct type_caster<tensorstore::internal_python::PythonPromiseObject>
    : public tensorstore::internal_python::StaticHeapTypeCaster<
          tensorstore::internal_python::PythonPromiseObject> {};

}  // namespace detail
}  // namespace pybind11

#endif  // THIRD_PARTY_PY_TENSORSTORE_FUTURE_H_
