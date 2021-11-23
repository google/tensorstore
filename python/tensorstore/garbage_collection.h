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

#ifndef THIRD_PARTY_PY_TENSORSTORE_GARBAGE_COLLECTION_H_
#define THIRD_PARTY_PY_TENSORSTORE_GARBAGE_COLLECTION_H_

/// \file
///
/// Support for interoperating with the Python garbage collector.
///
/// The Python C API includes support for extension classes to interoperate with
/// the garbage collector.  Specifically, extension classes can opt into garbage
/// collection tracking and define a `tp_traverse` method that visits all Python
/// objects directly owned by the extension class.  Extension classes that own
/// mutable references to Python objects can additionally define a `tp_clear`
/// method that eliminates references to owned Python objects.
///
/// This mechanism does not, however, support extension classes that have shared
/// ownership of a C++ object (such as a `tensorstore::internal::DriverPtr`),
/// which in turn owns a Python object.  The problem is that a `tp_traverse`
/// function cannot be defined in such a case, since it is not possible to
/// attribute the ownership of the Python object to a single extension class
/// object.
///
/// To avoid C++ shared ownership problems, we ensure that C++ objects with
/// shared ownership only maintain weak references, rather than strong
/// references, to Python objects.  To ensure those weak references remain
/// valid, extension classes that may transitively reference Python objects also
/// have a `PythonObjectReferenceManager`, which holds a strong reference to all
/// Python objects referenced by the C++ object.  This does impose a small
/// additional cost when copying such objects, though, as the
/// `PythonObjectReferenceManager` must be copied as well.

#include <pybind11/pybind11.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include "absl/container/flat_hash_set.h"
#include "python/tensorstore/define_heap_type.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/tagged_ptr.h"
#include "tensorstore/util/garbage_collection/garbage_collection.h"

namespace tensorstore {
namespace internal_python {

/// Holds strong references to a collection of Python objects weakly referenced
/// by another object.
///
/// \threadsafety Must only be used with the GIL held.
class PythonObjectReferenceManager {
 public:
  PythonObjectReferenceManager();
  PythonObjectReferenceManager(const PythonObjectReferenceManager& other);
  PythonObjectReferenceManager(PythonObjectReferenceManager&& other);
  ~PythonObjectReferenceManager();

  PythonObjectReferenceManager& operator=(
      PythonObjectReferenceManager&& other) noexcept;
  PythonObjectReferenceManager& operator=(
      const PythonObjectReferenceManager& other);

  /// Modifies this reference manager to hold references to exactly the Python
  /// objects referenced by `obj`, and no others.
  ///
  /// \tparam T Type for which
  ///     `tensorstore::garbage_collection::GarbageCollection` has been
  ///     specialized.
  /// \threadsafety The GIL must be held.
  template <typename T>
  void Update(const T& obj) {
    PythonObjectReferenceManager new_manager;
    PythonObjectReferenceManager::Visitor visitor(new_manager);
    garbage_collection::GarbageCollectionVisit(visitor, obj);
    *this = std::move(new_manager);
  }

  /// Releases all strong references.
  ///
  /// Intended to be called from a `tp_clear` function for mutable objects.
  ///
  /// Note that this is always safe, but may result in weak references becoming
  /// invalidated.
  ///
  /// \threadsafety The GIL must be held.
  void Clear();

  /// Invokes the visitor on all strong references.
  ///
  /// Intended to be called from a `tp_traverse` function; the meaning of the
  /// return value and parameters is the same as for a `tp_traverse` function.
  ///
  /// \threadsafety The GIL must be held.
  int Traverse(visitproc visit, void* arg);

  /// Garbage collection visitor used by `Update`.
  class Visitor final : public garbage_collection::GarbageCollectionVisitor {
   public:
    explicit Visitor(PythonObjectReferenceManager& manager)
        : manager_(manager) {}
    void DoIndirect(const std::type_info& type, ErasedVisitFunction visit,
                    const void* ptr);

   private:
    PythonObjectReferenceManager& manager_;

    /// Indirect objects previously seen, to avoid traversing them more than
    /// once.
    absl::flat_hash_set<const void*> seen_indirect_objects_;
  };

  /// Strong Python references managed by this object.  Note that for types that
  /// do no support weak references, this will actually hold a wrapper object
  /// rather than the object directly.
  absl::flat_hash_set<PyObject*> python_refs_;
};

/// Holds either a weak or strong reference to a Python object.
///
/// This is used by C++ objects with shared ownership that support garbage
/// collection to reference Python objects.
class PythonWeakRef {
 public:
  /// Constructs a null reference.
  PythonWeakRef() = default;

  /// Creates a `PythonWeakRef` that actually holds a strong reference.
  ///
  /// It will be converted automatically to a weak reference when it is first
  /// encountered by `PythonObjectReferenceManager::Update`.  This is used
  /// during deserialization and construction when a
  /// `PythonObjectReferenceManager` may not be available at the call site.
  PythonWeakRef(pybind11::object obj)
      : weak_ref_(TaggedObjectPtr(obj.release().ptr(), 1),
                  internal::adopt_object_ref) {}

  /// Creates a weak reference to `obj`.
  ///
  /// The object is kept alive by `manager`.
  ///
  /// \threadsafety The GIL must be held by the calling thread.
  explicit PythonWeakRef(PythonObjectReferenceManager& manager,
                         pybind11::handle obj);

  /// Returns `true` if this is non-null.
  explicit operator bool() const { return static_cast<bool>(weak_ref_); }

  /// Returns a borrowed reference to the reference.  Any subsequent Python API
  /// calls, other than increasing a reference count, may invalidate it.
  ///
  /// \threadsafety The GIL must be held by the calling thread.
  pybind11::handle get_value_or_none() const;
  pybind11::handle get_value_or_null() const;
  pybind11::handle get_value_or_throw() const;

 private:
  // Tag bit indicates if this is a strong reference rather than a weak
  // reference.
  using TaggedObjectPtr = internal::TaggedPtr<PyObject, 1>;

  struct TaggedObjectTraits {
    template <typename>
    using pointer = TaggedObjectPtr;

    static void increment(PyObject* p) { Py_INCREF(p); }
    static void decrement(PyObject* p) { Py_DECREF(p); }
  };

  using TaggedHandle = internal::IntrusivePtr<PyObject, TaggedObjectTraits>;

  mutable TaggedHandle weak_ref_;

  friend class PythonObjectReferenceManager;
};

template <typename Derived>
struct GarbageCollectedPythonObjectHandle;

/// Python object representation that holds both an object of type `T`, and a
/// `PythonObjectReferenceManager` that manages references held by the object of
/// type `T`.
///
/// This is intended to be used as a CRTP base class, where the derived class is
/// responsible for defining the type name.
///
/// The corresponding Python heap type is defined by calling `Define` (which
/// calls `DefineHeapType<DerivedPythonObject>`).
///
/// \tparam DerivedPythonObject Derived class that inherits from this class,
///     must define `constexpr static const char python_type_name[]` member.
/// \tparam T Type for which
///     `tensorstore::garbage_collection::GarbageCollection` has been
///     specialized.
template <typename DerivedPythonObject, typename T>
struct GarbageCollectedPythonObject {
  using aligned_reference_manager_t =
      std::aligned_storage_t<sizeof(PythonObjectReferenceManager),
                             alignof(PythonObjectReferenceManager)>;
  using ContainedValue = T;

  // The constructor is never called since this is allocated and
  // zero-initialized by Python.
  GarbageCollectedPythonObject() = delete;
  ~GarbageCollectedPythonObject() = delete;
  GarbageCollectedPythonObject(const GarbageCollectedPythonObject&) = delete;

  /// Python heap type corresponding to this object representation, initialized
  /// by calling `Define`.
  inline static PyTypeObject* python_type;

  /// Handle type that may be used to construct objects of this type.
  using Handle = GarbageCollectedPythonObjectHandle<DerivedPythonObject>;

  // clang-format off
  PyObject_HEAD
  PyObject *weakrefs;
  aligned_reference_manager_t aligned_reference_manager;
  T value;
  // clang-format on

  PythonObjectReferenceManager& reference_manager() {
    return *reinterpret_cast<PythonObjectReferenceManager*>(
        &aligned_reference_manager);
  }

  /// Updates the `reference_manager` to hold strong references to all Python
  /// objects referenced by the contained object.
  ///
  /// This should be called any time the set of Python objects referenced by the
  /// contained object changes.
  ///
  /// \threadsafety The GIL must be held.
  void UpdatePythonRefs() { reference_manager().Update(value); }

  /// Defines the Python type corresponding to `DerivedPythonObject`.
  ///
  /// \param doc Docstring for the Python type.
  static pybind11::class_<DerivedPythonObject> Define(const char* doc) {
    static_assert(
        sizeof(DerivedPythonObject) == sizeof(GarbageCollectedPythonObject),
        "Derived class must not contain any additional members.");
    PyType_Spec spec = {};
    spec.flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC;
    PyType_Slot slots[] = {
        {Py_tp_doc, const_cast<char*>(doc)},
        {Py_tp_alloc,
         reinterpret_cast<void*>(&GarbageCollectedPythonObject::Alloc)},
        {Py_tp_dealloc,
         reinterpret_cast<void*>(&GarbageCollectedPythonObject::Dealloc)},
        {Py_tp_traverse,
         reinterpret_cast<void*>(&GarbageCollectedPythonObject::Traverse)},
        {Py_tp_clear,
         reinterpret_cast<void*>(&GarbageCollectedPythonObject::Clear)},
        {0, nullptr},
    };
    spec.slots = slots;
    auto cls = DefineHeapType<DerivedPythonObject>(spec);
    python_type->tp_weaklistoffset = offsetof(DerivedPythonObject, weakrefs);
    return cls;
  }

 private:
  static PyObject* Alloc(PyTypeObject* type, Py_ssize_t nitems) {
    PyObject* ptr = PyType_GenericAlloc(type, nitems);
    if (!ptr) return nullptr;
    auto& obj = *reinterpret_cast<GarbageCollectedPythonObject*>(ptr);
    new (&obj.aligned_reference_manager) PythonObjectReferenceManager;
    new (&obj.value) T;
    return ptr;
  }

  static void Dealloc(PyObject* self) {
    auto& obj = *reinterpret_cast<GarbageCollectedPythonObject*>(self);
    // Ensure object is not tracked by garbage collector before invalidating
    // invariants during destruction.
    PyObject_GC_UnTrack(self);

    if (obj.weakrefs) PyObject_ClearWeakRefs(self);

    obj.value.~T();
    obj.reference_manager().~PythonObjectReferenceManager();
    PyTypeObject* type = Py_TYPE(self);
    type->tp_free(self);
    Py_DECREF(type);
  }

  static int Traverse(PyObject* self, visitproc visit, void* arg) {
    auto& obj = *reinterpret_cast<GarbageCollectedPythonObject*>(self);
    return obj.reference_manager().Traverse(visit, arg);
  }

  static int Clear(PyObject* self) {
    auto& obj = *reinterpret_cast<GarbageCollectedPythonObject*>(self);
    obj.reference_manager().Clear();
    return 0;
  }
};

/// Interface for constructing newly-allocated Python garbage
/// collection-supporting wrapper objects of type `DerivedPythonObject`.
///
/// This supports implicit conversion from objects of
/// `DerivedPythonObject::ContainedValue`, which simplifies use as the return
/// type of pybind11-bound functions.
///
/// \tparam DerivedPythonObject Python object type, must inherit from
///     `GarbageCollectedPythonObject<DerivedPythonObject>`.
template <typename DerivedPythonObject>
struct GarbageCollectedPythonObjectHandle {
  using PythonObjectType = DerivedPythonObject;
  using ContainedValue = typename DerivedPythonObject::ContainedValue;

  /// Reference to object of type `DerivedPythonObject::python_type`.
  pybind11::object value;

  /// Constructs a null reference.
  GarbageCollectedPythonObjectHandle() = default;

  /// Constructs from an existing Python object, which is assumed to be of type
  /// `DerivedPythonObject::python_type`.
  explicit GarbageCollectedPythonObjectHandle(pybind11::object value)
      : value(std::move(value)) {}

  /// Constructs from the specified value, and acquires strong references to any
  /// referenced Python objects.
  GarbageCollectedPythonObjectHandle(const ContainedValue& contained_value) {
    value = pybind11::reinterpret_steal<pybind11::object>(
        DerivedPythonObject::python_type->tp_alloc(
            DerivedPythonObject::python_type, 0));
    if (!value) throw pybind11::error_already_set();
    auto& obj = *reinterpret_cast<DerivedPythonObject*>(value.ptr());
    obj.value = contained_value;
    obj.UpdatePythonRefs();
  }

  /// Same as above, but move constructs.
  GarbageCollectedPythonObjectHandle(ContainedValue&& contained_value) {
    value = pybind11::reinterpret_steal<pybind11::object>(
        DerivedPythonObject::python_type->tp_alloc(
            DerivedPythonObject::python_type, 0));
    if (!value) throw pybind11::error_already_set();
    auto& obj = *reinterpret_cast<DerivedPythonObject*>(value.ptr());
    obj.value = std::move(contained_value);
    obj.UpdatePythonRefs();
  }

  DerivedPythonObject& obj() const {
    return *reinterpret_cast<DerivedPythonObject*>(value.ptr());
  }

  /// Supports implicit conversion to `DerivedPythonObject&` to simplify use
  /// with `DefineIndexTransformOperations`.
  operator DerivedPythonObject&() const { return obj(); }
};

/// pybind11 `type_caster` that allows a C++ type
/// `DerivedPythonObject::ContainedValue` (e.g. `Spec`) to be converted to a
/// newly-allocated Python wrapper object of type `DerivedPythonObject`
/// (e.g. `PythonSpecObject`).
///
/// For each `DerivedPythonObject` class, you must manually define a
/// `pybind11::detail::type_caster<DerivedPythonObject::ContainedValue>`
/// specialization that inherits from this class.
template <typename DerivedPythonObject>
struct GarbageCollectedObjectCaster {
  constexpr static auto name =
      pybind11::detail::_(DerivedPythonObject::python_type_name);

  using Handle = GarbageCollectedPythonObjectHandle<DerivedPythonObject>;

  using ContainedValue = typename DerivedPythonObject::ContainedValue;

  static pybind11::handle cast(const ContainedValue& arg,
                               pybind11::return_value_policy policy,
                               pybind11::handle parent) {
    return Handle(ContainedValue(arg)).value.release();
  }

  static pybind11::handle cast(ContainedValue&& arg,
                               pybind11::return_value_policy policy,
                               pybind11::handle parent) {
    return Handle(ContainedValue(std::move(arg))).value.release();
  }
};

/// Must be called exactly once during tensorstore module initialization.
void RegisterGarbageCollectionBindings();

}  // namespace internal_python
}  // namespace tensorstore

TENSORSTORE_DECLARE_GARBAGE_COLLECTION_SPECIALIZATION(
    tensorstore::internal_python::PythonWeakRef)

namespace pybind11 {
namespace detail {

/// Defines a `cast`-only pybind11 type_caster for
/// `tensorstore::internal_python::GarbageCollectedPythonObjectHandle`.
///
/// This is intended to allow `GarbageCollectedPythonObjectHandle` to be used as
/// a return type.  `load` is not supported.  For parameter types, use
/// `DerivedPythonObject&` or `DerivedPythonObject*` instead.
template <typename DerivedPythonObject>
struct type_caster<tensorstore::internal_python::
                       GarbageCollectedPythonObjectHandle<DerivedPythonObject>>
    : public tensorstore::internal_python::StaticHeapTypeWrapperCaster<
          tensorstore::internal_python::GarbageCollectedPythonObjectHandle<
              DerivedPythonObject>> {};

}  // namespace detail
}  // namespace pybind11

#endif  // THIRD_PARTY_PY_TENSORSTORE_GARBAGE_COLLECTION_H_
