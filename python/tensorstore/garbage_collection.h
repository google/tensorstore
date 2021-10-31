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

/// Must be called exactly once during tensorstore module initialization.
void RegisterGarbageCollectionBindings();

}  // namespace internal_python
}  // namespace tensorstore

TENSORSTORE_DECLARE_GARBAGE_COLLECTION_SPECIALIZATION(
    tensorstore::internal_python::PythonWeakRef)

#endif  // THIRD_PARTY_PY_TENSORSTORE_GARBAGE_COLLECTION_H_
