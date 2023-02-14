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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include <functional>
#include <memory>
#include <new>
#include <string>
#include <utility>

#include "python/tensorstore/define_heap_type.h"
#include "python/tensorstore/future.h"
#include "python/tensorstore/gil_safe.h"
#include "python/tensorstore/python_imports.h"
#include "python/tensorstore/tensorstore_module_components.h"
#include "tensorstore/internal/global_initializer.h"
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

// FIXME: Change to AnyFuture.
void InterruptibleWaitImpl(internal_future::FutureStateBase& future,
                           absl::Time deadline,
                           PythonFutureObject* python_future) {
  if (future.ready()) return;
  {
    GilScopedRelease gil_release;
    future.Force();
  }

  ScopedEvent event;
  const auto notify_done = [&event] { event.Set(); };
  std::optional<PythonFutureObject::CancelCallback> cancel_callback;
  if (python_future) {
    cancel_callback.emplace(python_future, notify_done);
  }
  ScopedFutureCallbackRegistration registration{
      internal_future::FutureAccess::Construct<AnyFuture>(
          internal_future::FutureStatePointer(&future))
          .UntypedExecuteWhenReady([&event](auto future) { event.Set(); })};

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

bool PythonFutureObject::Cancel() {
  if (done()) return false;
  cpp_data.state = {};
  cpp_data.registration.Unregister();
  RunCancelCallbacks();
  RunCallbacks();
  return true;
}

void PythonFutureObject::Force() {
  if (done()) return;
  // Use copy of `state`, since `state` may be modified by another thread
  // calling `Cancel` once GIL is released.
  auto state = cpp_data.state;
  GilScopedRelease gil_release;
  state->Force();
}

pybind11::object PythonFutureObject::GetAwaitable() {
  // Logically, `done_callback` needs to capture `awaitable_future`.  However,
  // lambda captures don't interoperate with Python garbage collection.
  // Instead, we create it as a capture-less function and then use a
  // `PyMethod` object to capture `awaitable_future` as the `self` argument.
  auto done_callback = py::cpp_function([](py::handle awaitable_future,
                                           py::handle source_future) {
    awaitable_future.attr("get_loop")().attr("call_soon_threadsafe")(
        py::cpp_function([](py::handle source_future,
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
  });

  py::object awaitable_future =
      python_imports.asyncio_get_event_loop_function().attr("create_future")();

  // Ensure the PythonFutureObject is cancelled if the awaitable future is
  // cancelled.
  auto cancel_callback = py::cpp_function(
      [](py::handle source_future, py::handle awaitable_future) {
        reinterpret_cast<PythonFutureObject*>(source_future.ptr())->Cancel();
      });
  auto bound_cancel_callback = py::reinterpret_steal<py::object>(
      PyMethod_New(cancel_callback.ptr(), reinterpret_cast<PyObject*>(this)));
  if (!bound_cancel_callback) throw py::error_already_set();

  awaitable_future.attr("add_done_callback")(bound_cancel_callback);

  // Ensure the awaitable future is marked ready once the PythonFutureObject
  // becomes ready.
  auto bound_done_callback = py::reinterpret_steal<py::object>(
      PyMethod_New(done_callback.ptr(), awaitable_future.ptr()));
  if (!bound_done_callback) throw py::error_already_set();

  AddDoneCallback(bound_done_callback);

  return awaitable_future.attr("__await__")();
}

internal_future::FutureStatePointer WaitForResult(PythonFutureObject& obj,
                                                  absl::Time deadline) {
  auto state = obj.cpp_data.state;
  internal_python::InterruptibleWaitImpl(*state, deadline, &obj);
  return state;
}

pybind11::object PythonFutureObject::GetResult(absl::Time deadline) {
  if (!cpp_data.state) ThrowCancelledError();
  auto state = WaitForResult(*this, deadline);
  return cpp_data.vtable->get_result(*state);
}

pybind11::object PythonFutureObject::GetException(absl::Time deadline) {
  if (!cpp_data.state) ThrowCancelledError();
  auto state = WaitForResult(*this, deadline);
  return cpp_data.vtable->get_exception(*state);
}

void PythonFutureObject::AddDoneCallback(pybind11::handle callback) {
  if (done()) {
    callback(py::handle(reinterpret_cast<PyObject*>(this)));
    return;
  }
  cpp_data.callbacks.push_back(py::reinterpret_borrow<py::object>(callback));
  if (cpp_data.callbacks.size() == 1) {
    Force();
  }
}

std::size_t PythonFutureObject::RemoveDoneCallback(pybind11::handle callback) {
  auto& callbacks = cpp_data.callbacks;
  // Since caller owns a reference to `callback`, we can be sure that removing
  // `callback` from `callbacks` does not result in any reference counts
  // reaching zero, and therefore we can be sure that `*this` is not modified.
  auto it = std::remove_if(
      callbacks.begin(), callbacks.end(),
      [&](pybind11::handle h) { return h.ptr() == callback.ptr(); });
  const size_t num_removed = callbacks.end() - it;
  callbacks.erase(it, callbacks.end());
  return num_removed;
}

Future<GilSafePythonHandle> PythonFutureObject::GetPythonValueFuture() {
  if (!cpp_data.state) return absl::CancelledError("");
  return cpp_data.vtable->get_python_value_future(*cpp_data.state);
}

int PythonFutureObject::TraversePythonReferences(visitproc visit, void* arg) {
  for (const auto& obj : cpp_data.callbacks) {
    Py_VISIT(obj.ptr());
  }
  return cpp_data.reference_manager.Traverse(visit, arg);
}

int PythonFutureObject::ClearPythonReferences() {
  cpp_data.state = {};
  cpp_data.registration.Unregister();
  // Don't modify `cpp_data.callbacks` in place, since clearing callbacks can
  // result in calls back into Python code, and `std::vector` does not support
  // re-entrant mutation.
  auto callbacks = std::move(cpp_data.callbacks);
  callbacks.clear();
  cpp_data.reference_manager.Clear();
  return 0;
}

void PythonFutureObject::RunCancelCallbacks() {
  for (CancelCallbackBase* callback = cpp_data.cancel_callbacks.next;
       callback != &cpp_data.cancel_callbacks;) {
    auto* next = callback->next;
    static_cast<CancelCallback*>(callback)->callback();
    callback = next;
  }
}

void PythonFutureObject::RunCallbacks() {
  auto callbacks = std::move(cpp_data.callbacks);
  if (callbacks.empty()) return;
  for (py::handle callback : callbacks) {
    if (PyObject* callback_result = PyObject_CallFunctionObjArgs(
            callback.ptr(), reinterpret_cast<PyObject*>(this), nullptr)) {
      Py_DECREF(callback_result);
      continue;
    }
    PyErr_WriteUnraisable(nullptr);
    PyErr_Clear();
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

pybind11::object TryConvertToFuture(pybind11::handle src,
                                    pybind11::handle loop) {
  if (Py_TYPE(src.ptr()) == PythonFutureObject::python_type) {
    return py::reinterpret_borrow<py::object>(src);
  }

  if (python_imports.asyncio_iscoroutine_function(src).ptr() != Py_True) {
    return {};
  }

  if (loop.is_none()) {
    throw py::value_error(
        "no event loop specified and thread does not have a default event "
        "loop");
  }

  auto asyncio_future =
      python_imports.asyncio_run_coroutine_threadsafe_function(src, loop);
  auto pair = PromiseFuturePair<GilSafePythonValueOrExceptionWeakRef>::Make();

  // Create Python future wrapper before adding `done_callback`, to ensure that
  // any references are retained by the `PythonObjectReferenceManager` of
  // `future`.
  auto future = PythonFutureObject::Make(std::move(pair.future));

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
        PythonValueOrException value =
            result ? PythonValueOrException(std::move(result))
                   : PythonValueOrException::FromErrorIndicator();
        PythonObjectReferenceManager manager;
        PythonValueOrExceptionWeakRef value_weak_ref(manager, value);

        // Release the GIL when invoking `promise.SetResult` in order to avoid
        // blocking other Python threads while arbitrary C++ callbacks are run.
        {
          GilScopedRelease gil_release;
          promise.SetResult(std::move(value_weak_ref));
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
  return future;
}

namespace {
using FutureCls = py::class_<PythonFutureObject>;
using PromiseCls = py::class_<PythonPromiseObject>;

PyObject* FutureAlloc(PyTypeObject* type, Py_ssize_t nitems) {
  PyObject* ptr = PyType_GenericAlloc(type, nitems);
  if (!ptr) return nullptr;
  // Immediately untrack by the garbage collector since the object is not yet
  // fully constructed.  Once it is fully constructed,
  // `PythonFutureObject::Make` marks it as tracked again.  There is no race
  // condition here because there are no intervening Python API calls
  // `PyType_GenericAlloc` marks it as tracked.
  PyObject_GC_UnTrack(ptr);
  auto& cpp_data = reinterpret_cast<PythonFutureObject*>(ptr)->cpp_data;
  new (&cpp_data) PythonFutureObject::CppData;
  internal::intrusive_linked_list::Initialize(
      PythonFutureObject::CancelCallback::Accessor{},
      &cpp_data.cancel_callbacks);
  return ptr;
}

void FutureDealloc(PyObject* self) {
  auto& obj = *reinterpret_cast<PythonFutureObject*>(self);
  auto& cpp_data = obj.cpp_data;
  // Ensure object is not tracked by garbage collector before invalidating
  // invariants during destruction.
  PyObject_GC_UnTrack(self);

  if (obj.weakrefs) PyObject_ClearWeakRefs(self);

  // Clear `state`: this ensures that the callback corresponding to
  // `registration` does not run with the reference count equal to 0.
  cpp_data.state = {};
  {
    GilScopedRelease gil_release;
    cpp_data.registration.Unregister();
  }
  cpp_data.~CppData();
  PyTypeObject* type = Py_TYPE(self);
  type->tp_free(self);
  Py_DECREF(type);
}

int FutureTraverse(PyObject* self, visitproc visit, void* arg) {
  return reinterpret_cast<PythonFutureObject*>(self)->TraversePythonReferences(
      visit, arg);
}

int FutureClear(PyObject* self) {
  return reinterpret_cast<PythonFutureObject*>(self)->ClearPythonReferences();
}

FutureCls MakeFutureClass(py::module m) {
  const char* doc = R"(
Handle for *consuming* the result of an asynchronous operation.

This type supports several different patterns for consuming results:

- Asynchronously with :py:mod:`asyncio`, using the :ref:`await<python:await>`
  keyword:

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
:ref:`await<python:await>` will raise an exception.

This type supports a subset of the interfaces of
:py:class:`python:concurrent.futures.Future` and
:py:class:`python:asyncio.Future`.  Unlike those types, however,
:py:class:`Future` provides only the *consumer* interface.  The corresponding
*producer* interface is provided by :py:class:`Promise`.

See also:
  - :py:class:`WriteFutures`

Group:
  Asynchronous support
)";
  PyType_Slot slots[] = {
      {Py_tp_doc, const_cast<char*>(doc)},
      {Py_tp_alloc, reinterpret_cast<void*>(&FutureAlloc)},
      {Py_tp_dealloc, reinterpret_cast<void*>(&FutureDealloc)},
      {Py_tp_traverse, reinterpret_cast<void*>(&FutureTraverse)},
      {Py_tp_clear, reinterpret_cast<void*>(&FutureClear)},
      {0, nullptr},
  };
  PyType_Spec spec = {};
  spec.flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC;
  spec.slots = slots;
  auto cls = DefineHeapType<PythonFutureObject>(spec);
  PythonFutureObject::python_type->tp_weaklistoffset =
      offsetof(PythonFutureObject, weakrefs);
  m.attr("Future") = cls;
  return cls;
}

void DefineFutureAttributes(FutureCls& cls) {
  cls.def(
      "__new__",
      [](py::handle cls_unused, UntypedFutureLike python_future,
         std::optional<AbstractEventLoopParameter> loop)
          -> UntypedFutureWrapper {
        if (!loop) loop.emplace().value = GetCurrentThreadAsyncioEventLoop();
        if (py::object future =
                TryConvertToFuture(python_future.value, loop->value)) {
          return {future};
        }
        PythonObjectReferenceManager manager;
        return {PythonFutureObject::Make(
            Future<GilSafePythonValueOrExceptionWeakRef>(
                GilSafePythonValueOrExceptionWeakRef{
                    PythonValueOrExceptionWeakRef(
                        manager, PythonValueOrException{
                                     std::move(python_future.value)})}))};
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

  cls.def("__await__",
          [](PythonFutureObject& self) { return self.GetAwaitable(); });

  cls.def(
      "add_done_callback",
      [](PythonFutureObject& self,
         Callable<void, PythonFutureObject> callback) {
        self.AddDoneCallback(callback.value);
      },
      py::arg("callback"),
      R"(
Registers a callback to be invoked upon completion of the asynchronous operation.

Group:
  Callback interface
)");

  cls.def(
      "remove_done_callback",
      [](PythonFutureObject& self,
         Callable<void, PythonFutureObject> callback) {
        return self.RemoveDoneCallback(callback.value);
      },
      py::arg("callback"),
      R"(
Unregisters a previously-registered callback.

Group:
  Callback interface
)");

  cls.def(
      "result",
      [](PythonFutureObject& self, std::optional<double> timeout,
         std::optional<double> deadline) -> py::object {
        return self.GetResult(GetWaitDeadline(timeout, deadline));
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
      [](PythonFutureObject& self, std::optional<double> timeout,
         std::optional<double> deadline) -> py::object {
        return self.GetException(GetWaitDeadline(timeout, deadline));
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

  cls.def(
      "done", [](PythonFutureObject& self) { return self.done(); },
      R"(
Queries whether the asynchronous operation has completed or been cancelled.

Group:
  Accessors
)");

  cls.def(
      "force", [](PythonFutureObject& self) { return self.Force(); },
      R"(
Ensures the asynchronous operation begins executing.

This is called automatically by :py:obj:`.result` and :py:obj:`.exception`, but
must be called explicitly when using :py:obj:`.add_done_callback`.
)");

  cls.def(
      "cancelled", [](PythonFutureObject& self) { return self.cancelled(); },
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
    Traceback (most recent call last):
        ...
    ...CancelledError...

Group:
  Accessors
)");

  cls.def(
      "cancel", [](PythonFutureObject& self) { return self.Cancel(); },
      R"(
Requests cancellation of the asynchronous operation.

If the operation has not already completed, it is marked as unsuccessfully
completed with an instance of :py:obj:`asyncio.CancelledError`.
)");
}

PyObject* PromiseAlloc(PyTypeObject* type, Py_ssize_t nitems) {
  PyObject* ptr = PyType_GenericAlloc(type, nitems);
  if (!ptr) return nullptr;
  auto& cpp_data = reinterpret_cast<PythonPromiseObject*>(ptr)->cpp_data;
  new (&cpp_data) PythonPromiseObject::CppData;
  return ptr;
}

void PromiseDealloc(PyObject* self) {
  auto& obj = *reinterpret_cast<PythonPromiseObject*>(self);
  auto& cpp_data = obj.cpp_data;
  // Ensure object is not tracked by garbage collector before invalidating
  // invariants during destruction.
  PyObject_GC_UnTrack(self);

  if (obj.weakrefs) PyObject_ClearWeakRefs(self);

  cpp_data.~CppData();
  PyTypeObject* type = Py_TYPE(self);
  type->tp_free(self);
  Py_DECREF(type);
}

int PromiseTraverse(PyObject* self, visitproc visit, void* arg) {
  auto& cpp_data = reinterpret_cast<PythonPromiseObject*>(self)->cpp_data;
  return cpp_data.reference_manager.Traverse(visit, arg);
}

int PromiseClear(PyObject* self) {
  auto& cpp_data = reinterpret_cast<PythonPromiseObject*>(self)->cpp_data;
  cpp_data.reference_manager.Clear();
  return 0;
}

PromiseCls MakePromiseClass(pybind11::module m) {
  const char* doc = R"(
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
)";
  PyType_Slot slots[] = {
      {Py_tp_doc, const_cast<char*>(doc)},
      {Py_tp_alloc, reinterpret_cast<void*>(&PromiseAlloc)},
      {Py_tp_dealloc, reinterpret_cast<void*>(&PromiseDealloc)},
      {Py_tp_traverse, reinterpret_cast<void*>(&PromiseTraverse)},
      {Py_tp_clear, reinterpret_cast<void*>(&PromiseClear)},
      {0, nullptr},
  };
  PyType_Spec spec = {};
  spec.flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC;
  spec.slots = slots;
  auto cls = DefineHeapType<PythonPromiseObject>(spec);
  PythonPromiseObject::python_type->tp_weaklistoffset =
      offsetof(PythonPromiseObject, weakrefs);
  DisallowInstantiationFromPython(cls);
  m.attr("Promise") = cls;
  return cls;
}

void DefinePromiseAttributes(PromiseCls& cls) {
  using Self = PythonPromiseObject;
  cls.def(
      "set_result",
      [](Self& self, py::object result) {
        self.cpp_data.promise.SetResult(
            GilSafePythonValueOrExceptionWeakRef{PythonValueOrExceptionWeakRef(
                self.cpp_data.reference_manager,
                PythonValueOrException{std::move(result)})});
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
      [](Self& self, py::object exception) {
        PyErr_SetObject(reinterpret_cast<PyObject*>(exception.ptr()->ob_type),
                        exception.ptr());
        self.cpp_data.promise.SetResult(
            GilSafePythonValueOrExceptionWeakRef{PythonValueOrExceptionWeakRef(
                self.cpp_data.reference_manager,
                PythonValueOrException::FromErrorIndicator())});
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
        auto [promise, future] =
            PromiseFuturePair<GilSafePythonValueOrExceptionWeakRef>::Make();
        pybind11::object self = pybind11::reinterpret_steal<pybind11::object>(
            PythonPromiseObject::python_type->tp_alloc(
                PythonPromiseObject::python_type, 0));
        if (!self) throw pybind11::error_already_set();
        auto& cpp_data =
            reinterpret_cast<PythonPromiseObject*>(self.ptr())->cpp_data;
        cpp_data.promise = std::move(promise);
        return std::make_pair(PromiseWrapper{std::move(self)},
                              std::move(future));
      },
      R"(
Creates a linked promise and future pair.

Group:
  Constructors
)");
}

void RegisterFutureBindings(pybind11::module m, Executor defer) {
  defer([cls = MakeFutureClass(m)]() mutable { DefineFutureAttributes(cls); });
  defer(
      [cls = MakePromiseClass(m)]() mutable { DefinePromiseAttributes(cls); });
}

TENSORSTORE_GLOBAL_INITIALIZER {
  RegisterPythonComponent(RegisterFutureBindings, /*priority=*/-450);
}

}  // namespace

PyTypeObject* PythonFutureObject::python_type = nullptr;
PyTypeObject* PythonPromiseObject::python_type = nullptr;

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

namespace serialization {

bool Serializer<internal_python::SerializableAbstractEventLoop>::Encode(
    EncodeSink& sink,
    const internal_python::SerializableAbstractEventLoop& value) {
  // Serialization is a no-op.
  return true;
}

bool Serializer<internal_python::SerializableAbstractEventLoop>::Decode(
    DecodeSource& source,
    internal_python::SerializableAbstractEventLoop& value) {
  internal_python::GilScopedAcquire gil_acquire;
  value.obj = internal_python::GetCurrentThreadAsyncioEventLoop();
  return true;
}

}  // namespace serialization

}  // namespace tensorstore
