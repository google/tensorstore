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

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
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
  auto cancelled_error = py::module::import("asyncio").attr("CancelledError");
  PyErr_SetNone(cancelled_error.ptr());
  throw py::error_already_set();
}

[[noreturn]] void ThrowTimeoutError() {
  auto timeout_error = py::module::import("builtins").attr("TimeoutError");
  PyErr_SetNone(timeout_error.ptr());
  throw py::error_already_set();
}

pybind11::object GetCancelledError() {
  return py::module::import("asyncio").attr("CancelledError")(py::none());
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
      pybind11::gil_scoped_release gil_release;
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
  py::object loop =
      py::module::import("asyncio.events").attr("get_event_loop")();
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

void RegisterFutureBindings(pybind11::module m) {
  py::class_<PythonFutureBase, std::shared_ptr<PythonFutureBase>> cls_future(
                                                                             m, "Future", R"(
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

  py::class_<Promise<PythonValueOrException>> cls_promise(m, "Promise", R"(
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


  cls_future.def("__await__", &PythonFutureBase::get_await_result);
  cls_future.def("add_done_callback", &PythonFutureBase::add_done_callback,
                 py::arg("callback"));
  cls_future.def("remove_done_callback",
                 &PythonFutureBase::remove_done_callback, py::arg("callback"));
  cls_future.def(
      "result",
      [](PythonFutureBase& self, std::optional<double> timeout,
         std::optional<double> deadline) -> py::object {
        return self.result(GetWaitDeadline(timeout, deadline));
      },
      py::arg("timeout") = std::nullopt, py::arg("deadline") = std::nullopt);
  cls_future.def(
      "exception",
      [](PythonFutureBase& self, std::optional<double> timeout,
         std::optional<double> deadline) -> py::object {
        return self.exception(GetWaitDeadline(timeout, deadline));
      },
      py::arg("timeout") = std::nullopt, py::arg("deadline") = std::nullopt);
  cls_future.def("done", &PythonFutureBase::done);
  cls_future.def("force", &PythonFutureBase::force);
  cls_future.def("cancelled", &PythonFutureBase::cancelled);
  cls_future.def("cancel", &PythonFutureBase::cancel);

  cls_promise.def("set_result", [](const Promise<PythonValueOrException>& self,
                                   py::object result) {
    self.SetResult(PythonValueOrException{std::move(result)});
  });
  cls_promise.def(
      "set_exception",
      [](const Promise<PythonValueOrException>& self, py::object exception) {
        PyErr_SetObject(nullptr, exception.ptr());
        PythonValueOrException v;
        PyErr_Fetch(&v.error_type.ptr(), &v.error_value.ptr(),
                    &v.error_traceback.ptr());
        self.SetResult(std::move(v));
      });
  cls_promise.def_static("new", [] {
    py::tuple result(2);
    auto [promise, future] = PromiseFuturePair<PythonValueOrException>::Make();
    result[0] = py::cast(std::move(promise));
    result[1] = py::cast(std::move(future));
    return result;
  });
}

}  // namespace internal_python
}  // namespace tensorstore
