// Copyright 2025 The TensorStore Authors
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

#ifndef PYTHON_TENSORSTORE_CRITICAL_SECTION_H_
#define PYTHON_TENSORSTORE_CRITICAL_SECTION_H_

#include <pybind11/pybind11.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.  We actually only need the normal Python
// C API, not pybind11, but we rely on pybind11 to define appropriate macros to
// avoid build failures when including `<Python.h>`.

#include <cassert>
#include <type_traits>
#include <utility>

#include "python/tensorstore/with_handle.h"

namespace tensorstore {
namespace internal_python {

/// A no-op implementation of `ScopedPyCriticalSection`.
class NoOpScopedPyCriticalSection {
 public:
  explicit NoOpScopedPyCriticalSection(const PyObject* ptr) {}

  template <typename T, std::enable_if_t<is_with_handle_v<T>, int> = 0>
  explicit NoOpScopedPyCriticalSection(const T& t)
      : NoOpScopedPyCriticalSection(t.handle.ptr()) {}

  ~NoOpScopedPyCriticalSection() = default;

  NoOpScopedPyCriticalSection(NoOpScopedPyCriticalSection&&) = delete;
  NoOpScopedPyCriticalSection& operator=(NoOpScopedPyCriticalSection&&) =
      delete;
  NoOpScopedPyCriticalSection(const NoOpScopedPyCriticalSection&) = delete;
  NoOpScopedPyCriticalSection& operator=(const NoOpScopedPyCriticalSection&) =
      delete;
};

class NoOpScopedPyCriticalSection2 {
 public:
  explicit NoOpScopedPyCriticalSection2(const PyObject* a, const PyObject* b) {
    assert(a);
    assert(b);
  }

  ~NoOpScopedPyCriticalSection2() = default;

  NoOpScopedPyCriticalSection2(NoOpScopedPyCriticalSection2&&) = delete;
  NoOpScopedPyCriticalSection2& operator=(NoOpScopedPyCriticalSection2&&) =
      delete;
  NoOpScopedPyCriticalSection2(const NoOpScopedPyCriticalSection2&) = delete;
  NoOpScopedPyCriticalSection2& operator=(const NoOpScopedPyCriticalSection2&) =
      delete;
};

/// Replacement for `Py_BEGIN_CRITICAL_SECTION` and `Py_END_CRITICAL_SECTION`.
/// See https://peps.python.org/pep-0703/#python-critical-sections
///
/// This is a no-op when `Py_GIL_DISABLED` is not defined.
/// Also, this uses the python critical section API directly.
/// Note that the PyCriticalSection records the address of the previous
/// object, so this type cannot be copied or moved.
#if defined(Py_GIL_DISABLED)
class ScopedPyCriticalSection {
 public:
  explicit ScopedPyCriticalSection(const PyObject* ptr)
      : ScopedPyCriticalSection(const_cast<PyObject*>(ptr)) {}

  explicit ScopedPyCriticalSection(PyObject* ptr) {
    assert(ptr);
    PyCriticalSection_Begin(&py_cs_, ptr);
  }

  ~ScopedPyCriticalSection() { PyCriticalSection_End(&py_cs_); }

  ScopedPyCriticalSection(ScopedPyCriticalSection&&) = delete;
  ScopedPyCriticalSection& operator=(ScopedPyCriticalSection&&) = delete;
  ScopedPyCriticalSection(const ScopedPyCriticalSection&) = delete;
  ScopedPyCriticalSection& operator=(const ScopedPyCriticalSection&) = delete;

 private:
  ::PyCriticalSection py_cs_;
};
#else
/// A no-op implementation of `ScopedPyCriticalSection`.
using ScopedPyCriticalSection = NoOpScopedPyCriticalSection;
#endif

/// Replacement for `Py_BEGIN_CRITICAL_SECTION` and `Py_END_CRITICAL_SECTION`.
/// See https://peps.python.org/pep-0703/#python-critical-sections
///
/// This is a no-op when `Py_GIL_DISABLED` is not defined.
/// Note that the PyCriticalSection records the address of the previous
/// object, so this type cannot be copied or moved.
#if defined(Py_GIL_DISABLED)
class ScopedPyCriticalSection2 {
 public:
  ScopedPyCriticalSection2(const PyObject* a, const PyObject* b)
      : ScopedPyCriticalSection2(const_cast<PyObject*>(a),
                                 const_cast<PyObject*>(b)) {}

  ScopedPyCriticalSection2(PyObject* a, PyObject* b) {
    assert(a);
    assert(b);
    PyCriticalSection2_Begin(&py_cs2_, a, b);
  }

  ~ScopedPyCriticalSection2() { PyCriticalSection2_End(&py_cs2_); }

  ScopedPyCriticalSection2(ScopedPyCriticalSection2&&) = delete;
  ScopedPyCriticalSection2& operator=(ScopedPyCriticalSection2&&) = delete;
  ScopedPyCriticalSection2(const ScopedPyCriticalSection2&) = delete;
  ScopedPyCriticalSection2& operator=(const ScopedPyCriticalSection2&) = delete;

 private:
  ::PyCriticalSection2 py_cs2_;
};
#else
/// A no-op implementation of `ScopedPyCriticalSection`.
using ScopedPyCriticalSection2 = NoOpScopedPyCriticalSection2;
#endif

// Convenience alias to allow for templated code to use
// `ScopedPyCriticalSection` when locking is required.
template <bool Enable>
using MaybeScopedPyCriticalSection =
    std::conditional_t<Enable, ScopedPyCriticalSection,
                       NoOpScopedPyCriticalSection>;

// Convenience alias to allow for templated code to use
// `ScopedPyCriticalSection2` when locking is required.
template <bool Enable>
using MaybeScopedPyCriticalSection2 =
    std::conditional_t<Enable, ScopedPyCriticalSection2,
                       NoOpScopedPyCriticalSection2>;

}  // namespace internal_python
}  // namespace tensorstore

#endif  // PYTHON_TENSORSTORE_CRITICAL_SECTION_H_
