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
// Define platform-dependent `ScopedEvent` class that supports waiting that is
// interrupted if the process receives a signal.
//
// Initially, the event is in the "unset" state.  The `Set` method changes the
// event to the "set" state.  The `Wait` method waits until either the "set"
// state is reached (in which case it returns `true`) or the process receives a
// signal (in which case it returns "false").
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
  bool Wait() {
    const HANDLE handles[2] = {handle, sigint_event};
    DWORD res = ::WaitForMultipleObjectsEx(2, handles, /*bWaitAll=*/FALSE,
                                           /*dwMilliseconds=*/INFINITE,
                                           /*bAlertable=*/FALSE);
    if (res == WAIT_OBJECT_0 + 1) {
      ::ResetEvent(sigint_event);
      return false;
    }
    assert(res == WAIT_OBJECT_0);
    return true;
  }
  HANDLE handle;
  HANDLE sigint_event;
};
#elif defined(__APPLE__)
// POSIX unnamed semaphores are not implemented on Mac OS.  Use
// `pthread_cond_wait` instead, as it is also interruptible by signals.
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
    set.store(true);
    ::pthread_cond_signal(&cond);
  }
  bool Wait() {
    {
      [[maybe_unused]] int err = ::pthread_mutex_lock(&mutex);
      assert(err == 0);
    }
    ::pthread_cond_wait(&cond, &mutex);
    {
      [[maybe_unused]] int err = ::pthread_mutex_unlock(&mutex);
      assert(err == 0);
    }
    return set.load();
  }
  std::atomic<bool> set{false};
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
  bool Wait() {
    if (::sem_wait(&sem) == 0) return true;
    assert(errno == EINTR);
    return false;
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

void ThrowCancelledError() {
  auto cancelled_error = py::module::import("asyncio").attr("CancelledError");
  PyErr_SetNone(cancelled_error.ptr());
}

pybind11::object GetCancelledError() {
  return py::module::import("asyncio").attr("CancelledError")(py::none());
}

void InterruptibleWaitImpl(
    std::function<FutureCallbackRegistration(std::function<void()> notify_done)>
        register_listener) {
  ScopedEvent event;
  ScopedFutureCallbackRegistration registration{
      register_listener([&event] { event.Set(); })};
  while (true) {
    {
      pybind11::gil_scoped_release gil_release;
      if (event.Wait()) return;
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

void RegisterFutureBindings(pybind11::module m) {
  py::class_<PythonFutureBase, std::shared_ptr<PythonFutureBase>> cls_future(
      m, "Future");

  cls_future.def("__await__", &PythonFutureBase::get_await_result);
  cls_future.def("add_done_callback", &PythonFutureBase::add_done_callback,
                 py::arg("callback"));
  cls_future.def("remove_done_callback",
                 &PythonFutureBase::remove_done_callback, py::arg("callback"));
  cls_future.def("result", &PythonFutureBase::result);
  cls_future.def("exception", &PythonFutureBase::exception);
  cls_future.def("done", &PythonFutureBase::done);
  cls_future.def("cancelled", &PythonFutureBase::cancelled);
  cls_future.def("cancel", &PythonFutureBase::cancel);

  py::class_<Promise<PythonValueOrException>> cls_promise(m, "Promise");
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
