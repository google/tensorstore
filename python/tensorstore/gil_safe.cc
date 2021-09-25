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
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include <thread>

#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "python/tensorstore/gil_safe.h"
#include "python/tensorstore/python_imports.h"

namespace tensorstore {
namespace internal_python {

namespace py = ::pybind11;

namespace {
/// Serves to block the main Python thread from exiting while there are pending
/// calls to Python APIs in other threads by Tensorstore code.
ABSL_CONST_INIT absl::Mutex exit_block_mutex{absl::kConstInit};
std::atomic<bool> python_exiting{false};
const std::thread::id main_thread_id = std::this_thread::get_id();

/// Checks if the current thread is the main thread.
///
/// We can't use `_PyOS_IsMainThread()` because that requires the GIL on some
/// versions of Python, and we use this function to check whether we can safely
/// acquire the GIL.
bool IsMainThread() { return main_thread_id == std::this_thread::get_id(); }
}  // namespace

void GilSafeIncref(PyObject* p) {
  if (!TryAcquireExitBlock()) return;
  GilScopedAcquire gil;
  Py_INCREF(p);
  ReleaseExitBlock();
}

void GilSafeDecref(PyObject* p) {
  if (!TryAcquireExitBlock()) return;
  GilScopedAcquire gil;
  Py_DECREF(p);
  ReleaseExitBlock();
}

void SetupExitHandler() {
  python_imports.atexit_register_function(
      py::cpp_function([]() ABSL_NO_THREAD_SAFETY_ANALYSIS {
        python_exiting.store(true, std::memory_order_release);
        // Release GIL before acquiring `exit_block_mutex`, in order to avoid
        // deadlock while we wait for other threads to complete pending
        // operations.
        {
          GilScopedRelease gil;
          exit_block_mutex.Lock();
        }
      }));
}

bool TryAcquireExitBlock() noexcept ABSL_NO_THREAD_SAFETY_ANALYSIS {
  while (!exit_block_mutex.ReaderTryLock()) {
    if (python_exiting.load(std::memory_order_acquire)) {
      return false;
    }
  }
  return true;
}

void ReleaseExitBlock() noexcept ABSL_NO_THREAD_SAFETY_ANALYSIS {
  exit_block_mutex.ReaderUnlock();
}

ExitSafeGilScopedAcquire::ExitSafeGilScopedAcquire() {
  acquired_ = IsMainThread() || TryAcquireExitBlock();
  if (acquired_) {
    gil_state_ = PyGILState_Ensure();
  }
}

ExitSafeGilScopedAcquire::~ExitSafeGilScopedAcquire() {
  if (acquired_) {
    PyGILState_Release(gil_state_);
    if (!IsMainThread()) {
      ReleaseExitBlock();
    }
  }
}

absl::Status PythonExitingError() {
  return absl::CancelledError("Python interpreter is exiting");
}

}  // namespace internal_python
}  // namespace tensorstore
