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

#ifndef THIRD_PARTY_PY_TENSORSTORE_DEFINE_HEAP_TYPE_H_
#define THIRD_PARTY_PY_TENSORSTORE_DEFINE_HEAP_TYPE_H_

/// \file
///
/// Utilities for using pybind11 with custom-defined Python heap types.  Custom
/// types allow for full control of object lifetime and garbage collection.
///
/// The Python C API supports two ways of defining Python types: "static types"
/// where the `PyTypeObject` itself is a global variable, and "heap types" where
/// the `PyTypeObject` is heap-allocated by calling `PyType_FromSpec` (or
/// related function).  Heap types generally behave similarly to classes defined
/// using Python code: attributes can be added to the type's `__dict__`, and
/// adding special members to the type's `__dict__` automatically results in
/// forwarding slot functions, like `tp_call`, `tp_new`, etc., being defined.
/// Heap types are what pybind11 always creates.
///
/// See https://docs.python.org/3/c-api/typeobj.html#heap-types for details.
///
/// In general any number of Python heap types could correspond to the same C
/// object representation.  However, the utilities defined here assume that
/// there is a unique C++ struct object representation for each heap type that
/// is defined using `DefineHeapType`, and it is assume to have the following
/// members:
///
///     struct MyPythonObject {
///       constexpr static const char python_type_name[] = "module.MyPython";
///       static PyTypeObject *python_type;
///
///       PyObject_HEAD
///       // ... additional members ...
///     };
///
/// The `DefineHeapType` function stores a leaked pointer to the newly-allocated
/// heap type object in the static `python_type` member; this allows the type to
/// be accessed, and in particular allows the type to be accessed from pybind11
/// type_caster implementations, which don't have access to any state.  We refer
/// to these heap types as "static heap types" since their lifetime is managed
/// similarly to a Python "static type", and we are just using a heap type for
/// the benefit of the automatic slot assignment.

#include <pybind11/pybind11.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

namespace tensorstore {
namespace internal_python {

/// Type for use as a return type in a pybind11-bound function that displays as
/// `T::python_type_name`.
///
/// This is intended to be used with object representation types `T` for which a
/// heap type is defined by calling `DefineHeapType<T>`.
///
/// This should normally only be used for return types, not parameter types.
template <typename T>
struct StaticHeapTypeWrapper {
  using PythonObjectType = T;
  pybind11::object value;
};

/// `cast`-only pybind11 type_caster for `StaticHeapTypeWrapper`.
///
/// \tparam Wrapper Must have the same "shape" as `StaticHeapTypeWrapper<T>`,
///     i.e. must define a `PythonObjectType` alias and a
///     `pybind1::object value` member.
template <typename Wrapper>
struct StaticHeapTypeWrapperCaster {
  using PythonObjectType = typename Wrapper::PythonObjectType;
  static constexpr auto name =
      pybind11::detail::_(PythonObjectType::python_type_name);

  static pybind11::handle cast(Wrapper value,
                               pybind11::return_value_policy policy,
                               pybind11::handle parent) {
    return value.value.release();
  }
};

/// Base class for use in defining `load`-only specializations of
/// `pybind11::detail::type_caster` for types defined using `DefineHeapType`.
///
/// For every heap type defined using `DefineHeapType<T>`, you should define a
/// specialization of `pybind11::detail::type_caster<T>` that inherits from
/// `StaticHeapTypeCaster<T>`: that allows you to use `T&` and `T*` as parameter
/// types in pybind11-bound functions.
template <typename T>
struct StaticHeapTypeCaster {
  static constexpr auto name = pybind11::detail::_(T::python_type_name);

  bool load(pybind11::handle src, bool convert) {
    if (Py_TYPE(src.ptr()) != T::python_type) {
      return false;
    }
    value = reinterpret_cast<T*>(src.ptr());
    return true;
  }

  operator T&() { return *value; }

  operator T*() { return value; }

  template <typename U>
  using cast_op_type = pybind11::detail::cast_op_type<U>;

 private:
  T* value = nullptr;
};

/// Creates a Python "heap type" from the specified `spec`.
///
/// The `name` is set from `T::python_type_name`, the `basicsize` is set from
/// `sizeof(T)`, and `T::python_type` is set to a leaked reference to the newly
/// created type.
///
/// The returned `pybind11::class_` object (which simply refers to
/// `T::python_type`) may be used to define methods, but `pybind11::init`,
/// `pybind11::pickle`, `def_buffer`, or anything else that depends on a
/// `pybind11::detail::type_record`, must not be used.
///
/// If you define a specialization of `pybind11::detail::type_caster<T>` that
/// inherits from `StaticHeapTypeCaster<T>`, then you can use `T&` and `T*` as a
/// pybind11 parameter type.
///
/// \tparam T Python object type, must define `static PyTypeObject *python_type`
///     member, and `static constexpr const char python_type_name[]` member.
///     The definition of `T` must start with `PyObject_HEAD`, and note that the
///     constructor of any other members will have to be called manually.
template <typename T>
pybind11::class_<T> DefineHeapType(PyType_Spec& spec) {
  spec.basicsize = sizeof(T);
  spec.name = T::python_type_name;
  T::python_type = reinterpret_cast<PyTypeObject*>(PyType_FromSpec(&spec));
  if (!T::python_type) {
    throw pybind11::error_already_set();
  }
  return pybind11::class_<T>(pybind11::reinterpret_borrow<pybind11::object>(
      pybind11::handle(reinterpret_cast<PyObject*>(T::python_type))));
}

/// Prevents objects of the specified type from being instantiated via a call to
/// the type object.
///
/// By default heap types inherit the `__new__` method from their parent class
/// if it is not explicitly overridden.  This can lead to an improperly
/// initialized object being exposed to Python.
///
/// https://bugs.python.org/msg391598
void DisallowInstantiationFromPython(pybind11::handle type_object);

}  // namespace internal_python
}  // namespace tensorstore

namespace pybind11 {
namespace detail {

template <typename T>
struct type_caster<tensorstore::internal_python::StaticHeapTypeWrapper<T>>
    : public tensorstore::internal_python::StaticHeapTypeWrapperCaster<
          tensorstore::internal_python::StaticHeapTypeWrapper<T>> {};

}  // namespace detail
}  // namespace pybind11

#endif  // THIRD_PARTY_PY_TENSORSTORE_DEFINE_HEAP_TYPE_H_
