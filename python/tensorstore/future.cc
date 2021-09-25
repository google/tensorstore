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

#include "python/tensorstore/future.h"

#include <functional>
#include <memory>
#include <new>
#include <string>
#include <utility>

#include "python/tensorstore/python_imports.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"

#ifdef _WIN32
#include <windows.h>
#elif defined(__APPLE__)
#include <pthread.h>
#else
#include <semaphore.h>
#endif

namespace tensorstore {
namespace internal_python {
namespace py = ::pybind11;

PythonFutureBase::~PythonFutureBase() = default;

namespace {
enum class ScopedEventWaitResult {
  kSuccess,
  kInterrupt,
  kTimeout,
};
// Define platform-dependent `ScopedEvent` class that supports waiting that is
// interrupted if the process receives a signal.
//
// Initially, the event is in the "unset" state.  The `Set` method changes the
// event to the "set" state.  The `Wait` method waits until the "set" state is
// reached, the process receives a signal, or the deadline is reached.
#ifdef _WIN32
class ScopedEvent {
 public:
  ScopedEvent() {
    sigint_event = _PyOS_SigintEvent();
    assert(sigint_event != nullptr);

    handle = ::CreateEventA(/*lpEventAttributes=*/nullptr,
                            /*bManualReset=*/TRUE,
                            /*bInitialState=*/FALSE,
                            /*lpName=*/nullptr);
    assert(handle != nullptr);
  }
  ~ScopedEvent() { ::CloseHandle(handle); }
  void Set() { ::SetEvent(handle); }
  ScopedEventWaitResult Wait(absl::Time deadline) {
    const HANDLE handles[2] = {handle, sigint_event};
    DWORD timeout;
    if (deadline == absl::InfiniteFuture()) {
      timeout = INFINITE;
    } else {
      int64_t ms = absl::ToInt64Milliseconds(deadline - absl::Now());
      ms = std::max(int64_t(0), ms);
      timeout =
          static_cast<DWORD>(std::min(ms, static_cast<int64_t>(INFINITE)));
    }
    DWORD res = ::WaitForMultipleObjectsEx(2, handles, /*bWaitAll=*/FALSE,
                                           /*dwMilliseconds=*/timeout,
                                           /*bAlertable=*/FALSE);
    if (res == WAIT_OBJECT_0 + 1) {
      ::ResetEvent(sigint_event);
      return ScopedEventWaitResult::kInterrupt;
    } else if (res == WAIT_OBJECT_0) {
      return ScopedEventWaitResult::kSuccess;
    } else {
      assert(res == WAIT_TIMEOUT);
      return ScopedEventWaitResult::kTimeout;
    }
  }
  HANDLE handle;
  HANDLE sigint_event;
};
#elif defined(__APPLE__)
// POSIX unnamed semaphores are not implemented on Mac OS.  Use
// `pthread_cond_wait`/`pthread_cond_timedwait` instead, as it is also
// interruptible by signals.
class ScopedEvent {
 public:
  ScopedEvent() {
    {
      [[maybe_unused]] int err = ::pthread_mutex_init(&mutex, nullptr);
      assert(err == 0);
    }
    {
      [[maybe_unused]] int err = ::pthread_cond_init(&cond, nullptr);
      assert(err == 0);
    }
  }
  ~ScopedEvent() {
    {
      [[maybe_unused]] int err = ::pthread_cond_destroy(&cond);
      assert(err == 0);
    }
    {
      [[maybe_unused]] int err = ::pthread_mutex_destroy(&mutex);
      assert(err == 0);
    }
  }
  void Set() {
    {
      [[maybe_unused]] int err = ::pthread_mutex_lock(&mutex);
      assert(err == 0);
    }
    set = true;
    {
      [[maybe_unused]] int err = ::pthread_mutex_unlock(&mutex);
      assert(err == 0);
    }
    ::pthread_cond_signal(&cond);
  }
  ScopedEventWaitResult Wait(absl::Time deadline) {
    {
      [[maybe_unused]] int err = ::pthread_mutex_lock(&mutex);
      assert(err == 0);
    }
    bool set_value = set;
    bool timeout = false;
    if (!set_value) {
      if (deadline == absl::InfiniteFuture()) {
        ::pthread_cond_wait(&cond, &mutex);
      } else {
        const auto tspec = ToTimespec(deadline);
        timeout = ::pthread_cond_timedwait(&cond, &mutex, &tspec) == ETIMEDOUT;
      }
      set_value = set;
    }
    {
      [[maybe_unused]] int err = ::pthread_mutex_unlock(&mutex);
      assert(err == 0);
    }
    return set_value ? ScopedEventWaitResult::kSuccess
                     : (timeout ? ScopedEventWaitResult::kTimeout
                                : ScopedEventWaitResult::kInterrupt);
  }
  bool set{false};
  ::pthread_mutex_t mutex;
  ::pthread_cond_t cond;
};
#else
// Use POSIX semaphores
class ScopedEvent {
 public:
  ScopedEvent() {
    [[maybe_unused]] int err = ::sem_init(&sem, /*pshared=*/0, 0);
    assert(err == 0);
  }
  ~ScopedEvent() {
    [[maybe_unused]] int err = ::sem_destroy(&sem);
    assert(err == 0);
  }
  void Set() {
    [[maybe_unused]] int err = ::sem_post(&sem);
    assert(err == 0);
  }
  ScopedEventWaitResult Wait(absl::Time deadline) {
    if (deadline == absl::InfiniteFuture()) {
      if (::sem_wait(&sem) == 0) return ScopedEventWaitResult::kSuccess;
      assert(errno == EINTR);
    } else {
      const auto tspec = absl::ToTimespec(deadline);
      if (::sem_timedwait(&sem, &tspec) == 0)
        return ScopedEventWaitResult::kSuccess;
      assert(errno == EINTR || errno == ETIMEDOUT);
      if (errno == ETIMEDOUT) return ScopedEventWaitResult::kTimeout;
    }
    return ScopedEventWaitResult::kInterrupt;
  }
  ::sem_t sem;
};
#endif

class ScopedFutureCallbackRegistration {
 public:
  ScopedFutureCallbackRegistration(FutureCallbackRegistration registration)
      : registration_(std::move(registration)) {}

  ~ScopedFutureCallbackRegistration() { registration_.Unregister(); }

 private:
  FutureCallbackRegistration registration_;
};
}  // namespace

[[noreturn]] void ThrowCancelledError() {
  PyErr_SetNone(python_imports.asyncio_cancelled_error_class.ptr());
  throw py::error_already_set();
}

[[noreturn]] void ThrowTimeoutError() {
  PyErr_SetNone(python_imports.builtins_timeout_error_class.ptr());
  throw py::error_already_set();
}

pybind11::object GetCancelledError() {
  return python_imports.asyncio_cancelled_error_class(py::none());
}

void InterruptibleWaitImpl(absl::FunctionRef<FutureCallbackRegistration(
                               absl::FunctionRef<void()> notify_done)>
                               register_listener,
                           absl::Time deadline,
                           PythonFutureBase* python_future) {
  ScopedEvent event;
  const auto notify_done = [&event] { event.Set(); };
  std::optional<PythonFutureBase::CancelCallback> cancel_callback;
  if (python_future) {
    cancel_callback.emplace(python_future, notify_done);
  }
  ScopedFutureCallbackRegistration registration{register_listener(notify_done)};
  while (true) {
    ScopedEventWaitResult wait_result;
    {
      GilScopedRelease gil_release;
      wait_result = event.Wait(deadline);
    }
    switch (wait_result) {
      case ScopedEventWaitResult::kSuccess:
        if (python_future && python_future->cancelled()) {
          ThrowCancelledError();
        }
        return;
      case ScopedEventWaitResult::kInterrupt:
        break;
      case ScopedEventWaitResult::kTimeout:
        ThrowTimeoutError();
    }
    if (PyErr_CheckSignals() == -1) {
      throw py::error_already_set();
    }
  }
}

pybind11::object PythonFutureBase::get_await_result() {
  auto self = shared_from_this();
  py::object loop = python_imports.asyncio_get_event_loop_function();
  py::object awaitable_future = loop.attr("create_future")();

  self->add_done_callback(py::cpp_function([awaitable_future,
                                            loop](py::object source_future) {
    loop.attr("call_soon_threadsafe")(
        py::cpp_function([](py::object source_future,
                            py::object awaitable_future) {
          if (awaitable_future.attr("done")().ptr() == Py_True) {
            return;
          }
          if (source_future.attr("cancelled")().ptr() == Py_True) {
            awaitable_future.attr("cancel")();
            return;
          }
          auto exc = source_future.attr("exception")();
          if (!exc.is_none()) {
            awaitable_future.attr("set_exception")(std::move(exc));
          } else {
            awaitable_future.attr("set_result")(source_future.attr("result")());
          }
        }),
        source_future, awaitable_future);
  }));
  awaitable_future.attr("add_done_callback")(
      py::cpp_function([self](py::object) { self->cancel(); }));
  return awaitable_future.attr("__await__")();
}

std::size_t PythonFutureBase::remove_done_callback(pybind11::object callback) {
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

PythonFutureBase::PythonFutureBase() {
  internal::intrusive_linked_list::Initialize(CancelCallback::Accessor{},
                                              &cancel_callbacks_);
}

void PythonFutureBase::RunCancelCallbacks() {
  for (CancelCallbackBase* callback = cancel_callbacks_.next;
       callback != &cancel_callbacks_;) {
    auto* next = callback->next;
    static_cast<CancelCallback*>(callback)->callback();
    callback = next;
  }
}

void PythonFutureBase::RunCallbacks() {
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

absl::Time GetWaitDeadline(std::optional<double> timeout,
                           std::optional<double> deadline) {
  absl::Time deadline_time = absl::InfiniteFuture();
  if (deadline) {
    deadline_time = absl::UnixEpoch() + absl::Seconds(*deadline);
  }
  if (timeout) {
    deadline_time =
        std::min(deadline_time, absl::Now() + absl::Seconds(*timeout));
  }
  return deadline_time;
}

bool TryConvertToFuture(pybind11::handle src, pybind11::handle loop,
                        std::shared_ptr<PythonFutureBase>& future) {
  if (py::isinstance<PythonFutureBase>(src)) {
    future = pybind11::cast<std::shared_ptr<PythonFutureBase>>(src);
    return true;
  }

  if (python_imports.asyncio_iscoroutine_function(src).ptr() != Py_True) {
    return false;
  }

  if (loop.is_none()) {
    throw py::value_error(
        "no event loop specified and thread does not have a default event "
        "loop");
  }

  auto asyncio_future =
      python_imports.asyncio_run_coroutine_threadsafe_function(src, loop);
  auto pair = PromiseFuturePair<GilSafePythonValueOrException>::Make();

  // Register done callback.
  py::object done_callback =
      py::cpp_function([promise = pair.promise](py::object future_obj) {
        py::object result;
        if (py::object method = py::reinterpret_steal<py::object>(
                PyObject_GetAttrString(future_obj.ptr(), "result"));
            method.ptr()) {
          result = py::reinterpret_steal<py::object>(
              PyObject_CallFunctionObjArgs(method.ptr(), nullptr));
        }
        GilSafePythonValueOrException gil_safe_value(
            result ? PythonValueOrException{std::move(result)}
                   : PythonValueOrException::FromErrorIndicator());
        // Release the GIL when invoking `promise.SetResult` in order to avoid
        // blocking other Python threads while arbitrary C++ callbacks are run.
        {
          GilScopedRelease gil_release;
          promise.SetResult(std::move(gil_safe_value));
        }
      });

  asyncio_future.attr("add_done_callback")(done_callback);

  // Register cancellation handler.
  pair.promise.ExecuteWhenNotNeeded(
      [asyncio_future = asyncio_future.release().ptr()] {
        ExitSafeGilScopedAcquire gil;
        if (!gil.acquired()) return;

        // Invoke `cancel` method.
        if (auto method = py::reinterpret_steal<py::object>(
                PyObject_GetAttrString(asyncio_future, "cancel"));
            !method.ptr()) {
          // Ignore error obtaining `cancel` method.
          PyErr_WriteUnraisable(nullptr);
          PyErr_Clear();
        } else if (!py::reinterpret_steal<py::object>(
                        PyObject_CallFunctionObjArgs(method.ptr(), nullptr))
                        .ptr()) {
          // Ignore error calling `cancel` method.
          PyErr_WriteUnraisable(nullptr);
          PyErr_Clear();
        }
        Py_DECREF(asyncio_future);
      });

  future = std::make_shared<PythonFuture<GilSafePythonValueOrException>>(
      std::move(pair.future));
  return true;
}

namespace {
using FutureCls =
    py::class_<PythonFutureBase, std::shared_ptr<PythonFutureBase>>;
using PromiseCls = py::class_<Promise<GilSafePythonValueOrException>>;

/// Metaclass that forwards __call__ to a static `_class_call_` method defined
/// on the class, similar to how `__class_getitem__` works.
///
/// Note: We use `_class_call_` rather than `__class_call__` since this is not
/// an official special method.
///
/// This metaclass is used by `Future` to allow `Future(f)`, where `f` is an
/// existing `Future` object, to return it unchanged, rather than a copy
/// referring to the same `std::shared_ptr<PythonFutureBase>`.
///
/// This is a workaround for
/// https://github.com/pybind/pybind11/issues/3253
PyTypeObject* GetClassCallMetaclass() {
  static auto* metaclass = [] {
    PyTypeObject* base_metaclass =
        pybind11::detail::get_internals().default_metaclass;
    PyType_Slot slots[] = {
        {Py_tp_base, base_metaclass},
        {Py_tp_call,
         (void*)+[](PyObject* self, PyObject* args,
                    PyObject* kwargs) -> PyObject* {
           auto method = py::reinterpret_steal<py::object>(
               PyObject_GetAttrString(self, "_class_call_"));
           if (!method.ptr()) return nullptr;
           return PyObject_Call(method.ptr(), args, kwargs);
         }},
        {0},
    };
    PyType_Spec spec = {};
    spec.name = "tensorstore._ClassCallMetaclass";
    spec.basicsize = base_metaclass->tp_basicsize;
    spec.flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    spec.slots = slots;
    PyTypeObject* metaclass = (PyTypeObject*)PyType_FromSpec(&spec);
    if (!metaclass) throw py::error_already_set();
    return metaclass;
  }();
  return metaclass;
}

FutureCls MakeFutureClass(pybind11::module m) {
  return FutureCls(
      m, "Future",
      py::metaclass(reinterpret_cast<PyObject*>(GetClassCallMetaclass())), R"(
Handle for *consuming* the result of an asynchronous operation.

This type supports several different patterns for consuming results:

- Asynchronously with :py:mod:`asyncio`, using the `await<python:await>` keyword:

      >>> future = ts.open({
      ...     'driver': 'array',
      ...     'array': [1, 2, 3],
      ...     'dtype': 'uint32'
      ... })
      >>> await future
      TensorStore({
        'array': [1, 2, 3],
        'context': {'data_copy_concurrency': {}},
        'driver': 'array',
        'dtype': 'uint32',
        'transform': {'input_exclusive_max': [3], 'input_inclusive_min': [0]},
      })

- Synchronously blocking the current thread, by calling :py:meth:`.result()`.

      >>> future = ts.open({
      ...     'driver': 'array',
      ...     'array': [1, 2, 3],
      ...     'dtype': 'uint32'
      ... })
      >>> future.result()
      TensorStore({
        'array': [1, 2, 3],
        'context': {'data_copy_concurrency': {}},
        'driver': 'array',
        'dtype': 'uint32',
        'transform': {'input_exclusive_max': [3], 'input_inclusive_min': [0]},
      })

- Asynchronously, by registering a callback using :py:meth:`.add_done_callback`:

      >>> future = ts.open({
      ...     'driver': 'array',
      ...     'array': [1, 2, 3],
      ...     'dtype': 'uint32'
      ... })
      >>> future.add_done_callback(
      ...     lambda f: print(f'Callback: {f.result().domain}'))
      ... future.force()  # ensure the operation is started
      ... # wait for completion (for testing only)
      ... result = future.result()
      Callback: { [0, 3) }

If an error occurs, instead of returning a value, :py:obj:`.result()` or
`python:await<await>` will raise an exception.

This type supports a subset of the interfaces of
:py:class:`python:concurrent.futures.Future` and
:py:class:`python:asyncio.Future`.  Unlike those types, however,
:py:class:`Future` provides only the *consumer* interface.  The corresponding
*producer* interface is provided by :py:class:`Promise`.

See also:
  - :py:class:`WriteFutures`

Group:
  Asynchronous support
)");
}

void DefineFutureAttributes(FutureCls& cls) {
  // Define the constructor as both both `_class_call_` and `__init__`.  The
  // `__init__` method won't normally be used, but is useful for documentation
  // purposes.
  const auto define_constructor = [&](auto func, auto... arg) {
    cls.def_static("_class_call_", func, arg...);
    cls.def(py::init(func), arg...);
  };
  define_constructor(
      [](UntypedFutureLike python_future,
         std::optional<AbstractEventLoopParameter> loop)
          -> std::shared_ptr<PythonFutureBase> {
        if (!loop) loop.emplace().value = GetCurrentThreadAsyncioEventLoop();
        if (std::shared_ptr<PythonFutureBase> future;
            TryConvertToFuture(python_future.value, loop->value, future)) {
          return future;
        }
        return std::make_shared<PythonFuture<GilSafePythonValueOrException>>(
            Future<GilSafePythonValueOrException>(GilSafePythonValueOrException{
                PythonValueOrException{std::move(python_future.value)}}));
      },
      R"(
Converts a :py:obj:`.FutureLike` object to a :py:obj:`.Future`.

Example:

    >>> await ts.Future(3)
    3

    >>> x = ts.Future(3)
    >>> assert x is ts.Future(x)

    >>> async def get_value():
    ...     return 42
    >>> x = ts.Future(get_value())
    >>> x.done()
    False
    >>> await x
    >>> x.result()
    42

Args:
  future: Specifies the immediate or asynchronous result.

  loop: Event loop on which to run :py:param:`.future` if it is a
  :ref:`coroutine<async>`.  If not specified (or :py:obj:`None` is specified),
  defaults to the loop returned by :py:obj:`asyncio.get_running_loop`.  If
  :py:param:`.loop` is not specified and there is no running event loop, it is
  an error for :py:param:`.future` to be a coroutine.

Returns:

  - If :py:param:`.future` is a :py:obj:`.Future`, it is simply returned as is.

  - If :py:param:`.future` is a :ref:`coroutine<async>`, it is run using
    :py:param:`.loop` and the returned :py:obj:`.Future` corresponds to the
    asynchronous result.

  - Otherwise, :py:param:`.future` is treated as an immediate result, and the
    returned :py:obj:`.Future` resolves immediately to :py:param:`.future`.

Warning:

  If :py:param:`.future` is a :ref:`coroutine<async>`, a blocking call to
  :py:obj:`Future.result` or :py:obj:`Future.exception` in the thread running
  the associated event loop may lead to deadlock.  Blocking calls should be
  avoided when using an event loop.

)",
      py::arg("future"), py::kw_only(), py::arg("loop") = std::nullopt);

  cls.def("__await__", &PythonFutureBase::get_await_result);

  cls.def("add_done_callback", &PythonFutureBase::add_done_callback,
          py::arg("callback"),
          R"(
Registers a callback to be invoked upon completion of the asynchronous operation.

Group:
  Callback interface
)");
  cls.def("remove_done_callback", &PythonFutureBase::remove_done_callback,
          py::arg("callback"),
          R"(
Unregisters a previously-registered callback.

Group:
  Callback interface
)");
  cls.def(
      "result",
      [](PythonFutureBase& self, std::optional<double> timeout,
         std::optional<double> deadline) -> py::object {
        return self.result(GetWaitDeadline(timeout, deadline));
      },
      py::arg("timeout") = std::nullopt, py::arg("deadline") = std::nullopt,
      R"(
Blocks until the asynchronous operation completes, and returns the result.

If the asynchronous operation completes unsuccessfully, raises the error that
was produced.

Args:
  timeout: Maximum number of seconds to block.
  deadline: Deadline in seconds since the Unix epoch.

Returns:
  The result of the asynchronous operation, if successful.

Raises:

  TimeoutError: If the result did not become ready within the specified
    :py:param:`.timeout` or :py:param:`.deadline`.

  KeyboardInterrupt: If running on the main thread and a keyboard interrupt is
    received.

Group:
  Blocking interface
)");
  cls.def(
      "exception",
      [](PythonFutureBase& self, std::optional<double> timeout,
         std::optional<double> deadline) -> py::object {
        return self.exception(GetWaitDeadline(timeout, deadline));
      },
      py::arg("timeout") = std::nullopt, py::arg("deadline") = std::nullopt,
      R"(
Blocks until asynchronous operation completes, and returns the error if any.

Args:
  timeout: Maximum number of seconds to block.
  deadline: Deadline in seconds since the Unix epoch.

Returns:

  The error that was produced by the asynchronous operation, or :py:obj:`None`
  if the operation completed successfully.

Raises:

  TimeoutError: If the result did not become ready within the specified
    :py:param:`.timeout` or :py:param:`.deadline`.

  KeyboardInterrupt: If running on the main thread and a keyboard interrupt is
    received.

Group:
  Blocking interface
)");

  cls.def("done", &PythonFutureBase::done,
          R"(
Queries whether the asynchronous operation has completed or been cancelled.

Group:
  Accessors
)");
  cls.def("force", &PythonFutureBase::force,
          R"(
Ensures the asynchronous operation begins executing.

This is called automatically by :py:obj:`.result` and :py:obj:`.exception`, but
must be called explicitly when using :py:obj:`.add_done_callback`.
)");
  cls.def("cancelled", &PythonFutureBase::cancelled,
          R"(
Queries whether the asynchronous operation has been cancelled.

Example:

    >>> promise, future = ts.Promise.new()
    >>> future.cancelled()
    False
    >>> future.cancel()
    >>> future.cancelled()
    True
    >>> future.exception()
    CancelledError(...)

Group:
  Accessors
)");
  cls.def("cancel", &PythonFutureBase::cancel,
          R"(
Requests cancellation of the asynchronous operation.

If the operation has not already completed, it is marked as unsuccessfully
completed with an instance of :py:obj:`asyncio.CancelledError`.
)");
}

PromiseCls MakePromiseClass(pybind11::module m) {
  return PromiseCls(m, "Promise", R"(
Handle for *producing* the result of an asynchronous operation.

A promise represents the producer interface corresponding to a
:py:class:`Future`, and may be used to signal the completion of an asynchronous
operation.

    >>> promise, future = ts.Promise.new()
    >>> future.done()
    False
    >>> promise.set_result(5)
    >>> future.done()
    True
    >>> future.result()
    5

See also:
  - :py:class:`Future`

Group:
  Asynchronous support
)");
}

void DefinePromiseAttributes(PromiseCls& cls) {
  using Self = Promise<GilSafePythonValueOrException>;
  cls.def(
      "set_result",
      [](const Self& self, py::object result) {
        self.SetResult(GilSafePythonValueOrException{
            PythonValueOrException{std::move(result)}});
      },
      py::arg("result"), R"(
Marks the linked future as successfully completed with the specified result.

Example:

    >>> promise, future = ts.Promise.new()
    >>> future.done()
    False
    >>> promise.set_result(5)
    >>> future.done()
    True
    >>> future.result()
    5

)");
  cls.def(
      "set_exception",
      [](const Self& self, py::object exception) {
        PyErr_SetObject(reinterpret_cast<PyObject*>(exception.ptr()->ob_type),
                        exception.ptr());
        self.SetResult(GilSafePythonValueOrException(
            PythonValueOrException::FromErrorIndicator()));
      },
      py::arg("exception"), R"(
Marks the linked future as unsuccessfully completed with the specified error.

Example:

    >>> promise, future = ts.Promise.new()
    >>> future.done()
    False
    >>> promise.set_exception(Exception(5))
    >>> future.done()
    True
    >>> future.result()
    Traceback (most recent call last):
        ...
    Exception: 5

)");
  cls.def_static(
      "new",
      [] {
        py::tuple result(2);
        auto [promise, future] =
            PromiseFuturePair<GilSafePythonValueOrException>::Make();
        result[0] = py::cast(std::move(promise));
        result[1] = py::cast(std::move(future));
        return result;
      },
      R"(
Creates a linked promise and future pair.

Group:
  Constructors
)");
}
}  // namespace

void RegisterFutureBindings(pybind11::module m, Executor defer) {
  defer([cls = MakeFutureClass(m)]() mutable { DefineFutureAttributes(cls); });
  defer(
      [cls = MakePromiseClass(m)]() mutable { DefinePromiseAttributes(cls); });
}

py::object GetCurrentThreadAsyncioEventLoop() {
  if (auto loop =
          py::reinterpret_steal<py::object>(PyObject_CallFunctionObjArgs(
              python_imports.asyncio__get_running_loop_function.ptr(),
              nullptr))) {
    return loop;
  }
  PyErr_Clear();
  return py::none();
}

}  // namespace internal_python
}  // namespace tensorstore
