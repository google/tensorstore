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

#ifndef THIRD_PARTY_PY_TENSORSTORE_GIL_SAFE_H_
#define THIRD_PARTY_PY_TENSORSTORE_GIL_SAFE_H_

/// \file
///
/// Utilities for safely handling the Python GIL (global interpreter lock).

#include <pybind11/pybind11.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include <utility>

#include "absl/status/status.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/util/garbage_collection/garbage_collection.h"

namespace tensorstore {
namespace internal_python {

/// RAII type that ensures the GIL is held by the current thread.
///
/// If this is used from a non-main thread without holding an exit block (see
/// `TryAcquireExitBlock`), if Python finalization occurs then the program may
/// crash due to https://bugs.python.org/issue42969.
class GilScopedAcquire {
 public:
  GilScopedAcquire() : gil_state_(PyGILState_Ensure()) {}
  ~GilScopedAcquire() { PyGILState_Release(gil_state_); }
  GilScopedAcquire(const GilScopedAcquire&) = delete;

 private:
  PyGILState_STATE gil_state_;
};

/// RAII type that ensures the GIL is released by the current thread.
class GilScopedRelease {
 public:
  GilScopedRelease() : save_(PyEval_SaveThread()) {}
  ~GilScopedRelease() { PyEval_RestoreThread(save_); }
  GilScopedRelease(const GilScopedRelease&) = delete;

 private:
  PyThreadState* save_;
};

/// Attempts to acquire a block on the Python interpreter exiting.
///
/// If successful, Python will not proceed to finalization until
/// `ReleaseExitBlock` is called.
///
/// The GIL can safely be acquired in non-main threads while this block is in
/// place.  This is a workaround for https://bugs.python.org/issue42969.
///
/// \returns `true` if the block was successfully acquired, or `false` if Python
///     is exiting.
bool TryAcquireExitBlock() noexcept;

/// Releases the block acquired by a successful call to `TryAcquireExitBlock`.
void ReleaseExitBlock() noexcept;

/// RAII type that safely acquires the GIL without risk of crashing due to
/// Python exiting.
///
/// This is a workaround for https://bugs.python.org/issue42969.
class ExitSafeGilScopedAcquire {
 public:
  /// Attempts to safely acquire the GIL.
  ///
  /// If called from the main thread, the GIL can always be acquired.
  ///
  /// Otherwise, attempts to first acquire an exit block (via
  /// `TryAcquireExitBlock`), and then if successful, acquires the GIL.
  ///
  /// The result of `acquired()` must always be checked to determine whether the
  /// GIL was successfully acquired.
  ExitSafeGilScopedAcquire();

  /// Releases the GIL (and exit block, if not on the main thread) if it was
  /// successfully acquired.
  ~ExitSafeGilScopedAcquire();
  ExitSafeGilScopedAcquire(const ExitSafeGilScopedAcquire&) = delete;

  /// Indicates whether the GIL was successfully acquired.
  bool acquired() const { return acquired_; }

 private:
  bool acquired_;
  PyGILState_STATE gil_state_;
};

/// Attempts to increment the reference count of a Python object.
///
/// The calling thread need not hold the GIL.
///
/// If Python is exiting, does nothing.
void GilSafeIncref(PyObject* p);

/// Attempts to decrement the reference count of a Python object.
///
/// The calling thread need not hold the GIL.
///
/// If Python is exiting, just leaks the reference instead.
///
/// Technically, if running on the main thread, we could safely decrement the
/// reference count even while Python is exiting, assuming the reference count
/// is valid.  However, if the reference was obtained from a call to
/// `GilSafeIncref` on another thread, also while Python was exiting, then the
/// reference count was never incremented in the first place and therefore must
/// not be decremented.
void GilSafeDecref(PyObject* p);

struct GilSafePythonHandleTraits {
  template <typename>
  using pointer = PyObject*;

  static void increment(PyObject* p) { GilSafeIncref(p); }
  static void decrement(PyObject* p) { GilSafeDecref(p); }
};

/// Reference-counted PyObject smart pointer that is safe to copy/destroy
/// without holding the GIL.
using GilSafePythonHandle =
    internal::IntrusivePtr<PyObject, GilSafePythonHandleTraits>;

/// `std::shared_ptr`-compatible deleter for a PyObject.
struct GilSafePythonObjectDeleter {
  void operator()(PyObject* p) const { GilSafeDecref(p); }
};

/// Returns a `shared_ptr` that points to `ptr` and keeps `obj` alive.
template <typename T>
std::shared_ptr<T> PythonObjectOwningSharedPtr(T* ptr, pybind11::object obj) {
  return std::shared_ptr<T>(
      std::shared_ptr<PyObject>(obj.release().ptr(),
                                GilSafePythonObjectDeleter{}),
      ptr);
}

/// Contains an object of type `T` and ensures the GIL is held when destroying.
///
/// Copying is disallowed and move construction is assumed to be safe without
/// the GIL held.
template <typename T>
struct GilSafeHolder {
  /// Default constructs the contained object.
  ///
  /// Does not ensure the GIL is held.
  GilSafeHolder() { new (&value_) T; }

  /// Constructs the contained object from the specified arguments.
  ///
  /// Does not ensure the GIL is held.
  template <typename... U>
  explicit GilSafeHolder(U&&... arg) {
    new (&value_) T(std::forward<U>(arg)...);
  }

  /// Move constructs.
  ///
  /// Does not ensure the GIL is held.
  GilSafeHolder(GilSafeHolder&& other) {
    new (&value_) T(std::move(other.value_));
  }

  T* operator->() { return &value_; }
  const T* operator->() const { return &value_; }

  T& operator*() { return value_; }
  const T& operator*() const { return value_; }

  /// Destroys the contained object with the GIL held.
  ///
  /// If the GIL cannot be acquired, the destructor is not called.
  ~GilSafeHolder() {
    ExitSafeGilScopedAcquire gil;
    if (gil.acquired()) {
      value_.~T();
    }
  }

 private:
  union {
    T value_;
  };
};

/// Returns a cancelled error indicating that Python is exiting.
///
/// This may be used when `TryAcquireExitBlock` returns `false`.
absl::Status PythonExitingError();

/// Called during module initialization (with GIL held) to install the
/// tensorstore `atexit` handler which protects from memory corruption/crashes
/// during interpreter exit.
void SetupExitHandler();

}  // namespace internal_python

namespace garbage_collection {
template <typename T>
struct GarbageCollection<internal_python::GilSafeHolder<T>> {
  static void Visit(GarbageCollectionVisitor& visitor,
                    const internal_python::GilSafeHolder<T>& value) {
    garbage_collection::GarbageCollectionVisit(visitor, *value);
  }
};
}  // namespace garbage_collection
}  // namespace tensorstore

namespace pybind11 {
namespace detail {

template <typename T>
struct type_caster<tensorstore::internal_python::GilSafeHolder<T>> {
  using value_conv = make_caster<T>;
  PYBIND11_TYPE_CASTER(tensorstore::internal_python::GilSafeHolder<T>,
                       value_conv::name);

  static handle cast(
      const tensorstore::internal_python::GilSafeHolder<T>& value,
      return_value_policy policy, handle parent) {
    return value_conv::cast(*value, policy, parent);
  }
};

}  // namespace detail
}  // namespace pybind11

#endif  // THIRD_PARTY_PY_TENSORSTORE_GIL_SAFE_H_
