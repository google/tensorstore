// Copyright 2021 The TensorStore Authors
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

#include "python/tensorstore/garbage_collection.h"

// Other headers
#include <atomic>
#include <cassert>
#include <typeinfo>
#include <utility>

#include "python/tensorstore/tensorstore_module_components.h"
#include "tensorstore/internal/global_initializer.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/garbage_collection/garbage_collection.h"

namespace py = ::pybind11;

namespace tensorstore {
namespace internal_python {

PythonObjectReferenceManager::PythonObjectReferenceManager() = default;

PythonObjectReferenceManager::PythonObjectReferenceManager(
    const PythonObjectReferenceManager& other)
    : python_refs_(other.python_refs_) {
  if (python_refs_.empty()) return;
  for (auto* ptr : python_refs_) {
    Py_INCREF(ptr);
  }
}

PythonObjectReferenceManager::PythonObjectReferenceManager(
    PythonObjectReferenceManager&& other) {
  Clear();
  python_refs_.swap(other.python_refs_);
}

PythonObjectReferenceManager::~PythonObjectReferenceManager() {
  if (python_refs_.empty()) return;
  Clear();
}

PythonObjectReferenceManager& PythonObjectReferenceManager::operator=(
    const PythonObjectReferenceManager& other) {
  if (python_refs_.empty() && other.python_refs_.empty()) return *this;
  Clear();
  python_refs_ = other.python_refs_;
  for (auto* ptr : python_refs_) {
    Py_INCREF(ptr);
  }
  return *this;
}

PythonObjectReferenceManager& PythonObjectReferenceManager::operator=(
    PythonObjectReferenceManager&& other) noexcept {
  python_refs_.swap(other.python_refs_);
  other.Clear();
  return *this;
}

void PythonObjectReferenceManager::Clear() {
  auto python_refs = std::exchange(python_refs_, {});
  for (auto* ptr : python_refs) {
    Py_DECREF(ptr);
  }
}

int PythonObjectReferenceManager::Traverse(visitproc visit, void* arg) {
  for (auto* ptr : python_refs_) {
    Py_VISIT(ptr);
  }
  return 0;
}

void PythonObjectReferenceManager::Visitor::DoIndirect(
    const std::type_info& type, ErasedVisitFunction visit, const void* ptr) {
  if (type == typeid(PythonWeakRef)) {
    auto& obj = *static_cast<const PythonWeakRef*>(ptr);
    auto weak_ref = obj.weak_ref_.load(std::memory_order_acquire);
    if (!weak_ref || weak_ref.tag() == 1) {
      // Null or permanent strong reference to non-GC object.
      return;
    }

    // Actual weak reference (tag == 0 or tag == 2).
#if PY_VERSION_HEX >= 0x030D0000
    PyObject* python_obj = nullptr;
    if (PyWeakref_GetRef(weak_ref.get(), &python_obj) <= 0) {
      PyErr_Clear();
      return;
    }
    if (!manager_.python_refs_.insert(python_obj).second) {
      Py_DECREF(python_obj);
    }
#else
    PyObject* python_obj = PyWeakref_GET_OBJECT(weak_ref.get());
    if (python_obj == Py_None) return;

    if (manager_.python_refs_.insert(python_obj).second) {
      Py_INCREF(python_obj);
    }
#endif
    if (weak_ref.tag() == 2) {
      if (obj.weak_ref_.compare_exchange_strong(
              weak_ref, PythonWeakRef::TaggedObjectPtr(weak_ref.get(), 0),
              std::memory_order_acq_rel)) {
        // Release the initial strong reference owned by tag == 2 now that
        // `manager_.python_refs_` holds a strong reference to `python_obj`.
        Py_DECREF(python_obj);
      }
    }
    return;
  }
  if (seen_indirect_objects_.insert(ptr).second) {
    visit(*this, ptr);
  }
}

/// Object representation for `tensorstore._WeakRefAdapter`, which is used to
/// create weak references to Python objects that don't directly support weak
/// references.
///
/// Objects of this type should not normally be exposed to external Python code;
/// they should only be accessible to the Python garbage collector.
struct WeakRefAdapterObject {
  // clang-format off
  PyObject_HEAD
  PyObject *owned_object;
  PyObject *weakrefs;
  // clang-format on
};

PyTypeObject WeakRefAdapterType = [] {
  PyTypeObject t = {PyVarObject_HEAD_INIT(nullptr, 0)};
  t.tp_name = "tensorstore._WeakRefAdapter";
  t.tp_basicsize = sizeof(WeakRefAdapterObject);
  t.tp_itemsize = 0;
  t.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC;
  t.tp_weaklistoffset = offsetof(WeakRefAdapterObject, weakrefs);
  t.tp_dealloc = [](PyObject* self) {
    Py_DECREF(reinterpret_cast<WeakRefAdapterObject*>(self)->owned_object);
    Py_TYPE(self)->tp_free(self);
  };
  t.tp_traverse = [](PyObject* self, visitproc visit, void* arg) -> int {
    return visit(reinterpret_cast<WeakRefAdapterObject*>(self)->owned_object,
                 arg);
  };
  return t;
}();

PythonWeakRef::TaggedObjectPtr PythonWeakRef::AcquireRef(
    TaggedObjectPtr p) noexcept {
  if (!p) return nullptr;
  Py_INCREF(p.get());
  if (p.tag() == 2) {
#if PY_VERSION_HEX >= 0x030D0000
    PyObject* target = nullptr;
    if (PyWeakref_GetRef(p.get(), &target) <= 0) {
      PyErr_Clear();
      return TaggedObjectPtr(p.get(), 0);
    }
#else
    PyObject* target = PyWeakref_GET_OBJECT(p.get());
    if (target == Py_None) {
      return TaggedObjectPtr(p.get(), 0);
    }
    Py_INCREF(target);
#endif
  }
  return p;
}

void PythonWeakRef::ReleaseRef(TaggedObjectPtr p) noexcept {
  if (!p) return;
  if (p.tag() == 2) {
#if PY_VERSION_HEX >= 0x030D0000
    PyObject* target = nullptr;
    if (PyWeakref_GetRef(p.get(), &target) > 0) {
      // Release both the reference returned by `PyWeakref_GetRef` and the
      // unclaimed strong reference owned by tag == 2.
      Py_DECREF(target);
      Py_DECREF(target);
    } else {
      PyErr_Clear();
    }
#else
    PyObject* target = PyWeakref_GET_OBJECT(p.get());
    if (target != Py_None) {
      Py_DECREF(target);
    }
#endif
  }
  Py_DECREF(p.get());
}

PythonWeakRef::PythonWeakRef(const PythonWeakRef& other)
    : weak_ref_(AcquireRef(other.weak_ref_.load(std::memory_order_acquire))) {}

PythonWeakRef::PythonWeakRef(PythonWeakRef&& other) noexcept
    : weak_ref_(other.weak_ref_.exchange(nullptr, std::memory_order_acq_rel)) {}

PythonWeakRef& PythonWeakRef::operator=(const PythonWeakRef& other) {
  if (this != &other) {
    TaggedObjectPtr new_ref =
        AcquireRef(other.weak_ref_.load(std::memory_order_acquire));
    ReleaseRef(weak_ref_.exchange(new_ref, std::memory_order_acq_rel));
  }
  return *this;
}

PythonWeakRef& PythonWeakRef::operator=(PythonWeakRef&& other) noexcept {
  if (this != &other) {
    TaggedObjectPtr new_ref =
        other.weak_ref_.exchange(nullptr, std::memory_order_acq_rel);
    ReleaseRef(weak_ref_.exchange(new_ref, std::memory_order_acq_rel));
  }
  return *this;
}

PythonWeakRef::~PythonWeakRef() {
  ReleaseRef(weak_ref_.load(std::memory_order_relaxed));
}

PythonWeakRef::PythonWeakRef(pybind11::object obj) {
  if (!obj) return;
  if (!PyObject_IS_GC(obj.ptr())) {
    assert(!PyWeakref_CheckRefExact(obj.ptr()));
    weak_ref_.store(TaggedObjectPtr(obj.release().ptr(), 1),
                    std::memory_order_relaxed);
    return;
  }
  pybind11::object target;
  if (!PyType_SUPPORTS_WEAKREFS(Py_TYPE(obj.ptr()))) {
    auto* wrapper_type = &WeakRefAdapterType;
    PyObject* wrapper_object = wrapper_type->tp_alloc(wrapper_type, 0);
    if (!wrapper_object) throw py::error_already_set();
    reinterpret_cast<WeakRefAdapterObject*>(wrapper_object)->owned_object =
        obj.release().ptr();
    target = pybind11::reinterpret_steal<pybind11::object>(wrapper_object);
  } else {
    target = std::move(obj);
  }
  PyObject* weak_ref = PyWeakref_NewRef(target.ptr(), nullptr);
  if (!weak_ref) {
    throw py::error_already_set();
  }
  (void)target.release();
  weak_ref_.store(TaggedObjectPtr(weak_ref, 2), std::memory_order_relaxed);
}

PythonWeakRef::PythonWeakRef(PythonObjectReferenceManager& manager,
                             pybind11::handle obj) {
  if (!obj) return;
  if (!PyObject_IS_GC(obj.ptr())) {
    // Since `obj` does not implement the garbage collection protocol, we can
    // just store a strong reference to it directly, and save the cost of using
    // a weak reference.

    // If `obj` is itself a weak reference, it would be ambiguous whether we are
    // supposed to dereference the weak reference.  Fortunately that case cannot
    // occur since weak references implement the garbage collection protocol.
    assert(!PyWeakref_CheckRefExact(obj.ptr()));

    weak_ref_.store(TaggedObjectPtr(obj.inc_ref().ptr(), 1),
                    std::memory_order_relaxed);
    return;
  }
  if (!PyType_SUPPORTS_WEAKREFS(Py_TYPE(obj.ptr()))) {
    // Since `obj` does not directly support weak references, we wrap it with a
    // new `WeakRefAdapter` object.
    auto* wrapper_type = &WeakRefAdapterType;
    PyObject* wrapper_object = wrapper_type->tp_alloc(wrapper_type, 0);
    if (!wrapper_object) throw py::error_already_set();
    reinterpret_cast<WeakRefAdapterObject*>(wrapper_object)->owned_object =
        obj.inc_ref().ptr();
    // Transfer ownership of `wrapper_object` to `manager`.
    manager.python_refs_.insert(wrapper_object);
    obj = wrapper_object;
  } else {
    // Add `obj` to `manager`.
    if (manager.python_refs_.insert(obj.ptr()).second) {
      obj.inc_ref();
    }
  }
  PyObject* weak_ref = PyWeakref_NewRef(obj.ptr(), nullptr);
  if (!weak_ref) {
    // Should not happen
    throw py::error_already_set();
  }
  weak_ref_.store(TaggedObjectPtr(weak_ref, 0), std::memory_order_relaxed);
}

pybind11::object PythonWeakRef::get_value_or_null() const {
  auto weak_ref = weak_ref_.load(std::memory_order_acquire);
  if (!weak_ref) return {};
  if (weak_ref.tag() == 1) {
    // Actually a strong reference.
    return pybind11::reinterpret_borrow<pybind11::object>(weak_ref.get());
  }
#if PY_VERSION_HEX >= 0x030D0000
  PyObject* obj = nullptr;
  if (PyWeakref_GetRef(weak_ref.get(), &obj) < 0) {
    throw pybind11::error_already_set();
  }
  if (obj == nullptr) {
    return {};
  }
  auto ref = pybind11::reinterpret_steal<pybind11::object>(obj);
#else
  PyObject* obj = PyWeakref_GET_OBJECT(weak_ref.get());
  if (obj == Py_None) {
    // Since `None` does not support weak references, we know that the actual
    // value was not `None`.
    return {};
  }
  auto ref = pybind11::reinterpret_borrow<pybind11::object>(obj);
#endif
  auto* python_type = Py_TYPE(ref.ptr());
  if (python_type == &WeakRefAdapterType) {
    return pybind11::reinterpret_borrow<pybind11::object>(
        reinterpret_cast<WeakRefAdapterObject*>(ref.ptr())->owned_object);
  }
  return ref;
}

pybind11::object PythonWeakRef::get_value_or_none() const {
  auto h = get_value_or_null();
  if (!h) return pybind11::none();
  return h;
}

pybind11::object PythonWeakRef::get_value_or_throw() const {
  auto h = get_value_or_null();
  if (!h) throw pybind11::value_error("Expired weak reference");
  return h;
}

namespace {

void RegisterGarbageCollectionBindings(pybind11::module_ m, Executor defer) {
  if (PyType_Ready(&WeakRefAdapterType) != 0) {
    throw py::error_already_set();
  }
}

TENSORSTORE_GLOBAL_INITIALIZER {
  RegisterPythonComponent(RegisterGarbageCollectionBindings,
                          /*priority=*/-2000);
}

}  // namespace

}  // namespace internal_python

namespace garbage_collection {
void GarbageCollection<internal_python::PythonWeakRef>::Visit(
    GarbageCollectionVisitor& visitor,
    const internal_python::PythonWeakRef& value) {
  using internal_python::PythonWeakRef;
  if (!value) return;
  visitor.DoIndirect(
      typeid(PythonWeakRef), [](GarbageCollectionVisitor&, const void*) {},
      static_cast<const void*>(&value));
}
}  // namespace garbage_collection

}  // namespace tensorstore
