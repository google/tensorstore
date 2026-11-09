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

#ifndef PYTHON_TENSORSTORE_SUBSCRIPT_METHOD_H_
#define PYTHON_TENSORSTORE_SUBSCRIPT_METHOD_H_

/// \file
///
/// Facility for conveniently defining special wrapper properties in order to
/// support syntax like `x.property[expr]`, used for the `vindex` and `oindex`
/// indexing methods as well as other "dimension expression" operations.

#include <pybind11/pybind11.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include <stddef.h>

#include <string>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/strings/str_format.h"
#include "python/tensorstore/define_heap_type.h"
#include "tensorstore/internal/meta/type_traits.h"

namespace tensorstore {
namespace internal_python {

/// Shared Python object representation for subscript helper descriptors created
/// by `DefineSubscriptMethod`. Each subscript helper creates its own Python
/// heap type via `DefineHeapType`, while sharing `GetItemHelper` and
/// `pybind11::class_<GetItemHelper>` across all `Parent`/`Tag` instantiations.
struct GetItemHelper {
  constexpr static const char python_type_name[] = "typing.Any";

  // clang-format off
  PyObject_HEAD
  PyObject* weakrefs;
  PyObject* parent;
  // clang-format on

  static void Dealloc(PyObject* self) {
    PyObject_GC_UnTrack(self);
    auto& obj = *reinterpret_cast<GetItemHelper*>(self);
    if (obj.weakrefs) PyObject_ClearWeakRefs(self);
    Py_XDECREF(obj.parent);
    PyTypeObject* type = Py_TYPE(self);
    type->tp_free(self);
    Py_DECREF(type);
  }

  static int Traverse(PyObject* self, visitproc visit, void* arg) {
    auto& obj = *reinterpret_cast<GetItemHelper*>(self);
    Py_VISIT(obj.parent);
    return 0;
  }
};

/// Metafunction for inferring the `N`th parameter type of a function type.
template <size_t N, typename T>
struct FunctionArgType;

template <size_t N, typename R, typename... Arg>
struct FunctionArgType<N, R(Arg...)> {
  using type = internal::TypePackElement<N, Arg...>;
};

template <typename Self, typename Func,
          typename Sig = pybind11::detail::function_signature_t<Func>>
struct ParentForwardingFunc;

template <typename Self, typename Func, typename R, typename OrigSelf,
          typename... Arg>
struct ParentForwardingFunc<Self, Func, R(OrigSelf, Arg...)> {
  R operator()(Self self, Arg... arg) {
    pybind11::detail::make_caster<OrigSelf> conv;
    return func(pybind11::detail::cast_op<OrigSelf>(pybind11::detail::load_type(
                    conv, pybind11::handle(self.parent))),
                std::forward<Arg>(arg)...);
  }
  Func func;
};

/// Result type of `DefineSubscriptMethod` used to define methods on the wrapper
/// class.
struct GetItemHelperClass {
  pybind11::class_<GetItemHelper> class_wrapper;

  /// Defines a method (typically `__getitem__` or `__setitem__`) on the wrapper
  /// class.  This function may be used the same way as `pybind11::class_::def`.
  ///
  /// \param func Implementation of the method, called with a `self` parameter
  ///     of type `Parent` that refers to the parent object (not the wrapper
  ///     class).
  template <typename Func, typename... Options>
  GetItemHelperClass& def(const char* name, Func func,
                          const Options&... options) {
    class_wrapper.def(
        name, ParentForwardingFunc<const GetItemHelper&, Func>{std::move(func)},
        options...);
    return *this;
  }
};

ABSL_ATTRIBUTE_NOINLINE inline GetItemHelperClass DefineSubscriptMethodImpl(
    pybind11::detail::generic_type scope, const char* method_name,
    const char* helper_class_name) {
  std::string module_name = pybind11::str(scope.attr("__module__"));
  std::string qualname = absl::StrFormat(
      "%s.%s", pybind11::str(scope.attr("__qualname__")), helper_class_name);
  std::string full_name =
      absl::StrFormat("%s.%s", module_name, helper_class_name);
  PyType_Slot slots[] = {
      {Py_tp_dealloc, reinterpret_cast<void*>(&GetItemHelper::Dealloc)},
      {Py_tp_traverse, reinterpret_cast<void*>(&GetItemHelper::Traverse)},
      {0, nullptr},
  };
  PyType_Spec spec = {};
  spec.name = full_name.c_str();
  spec.basicsize = sizeof(GetItemHelper);
  spec.flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC;
  spec.slots = slots;
  PyTypeObject* helper_type = DefineHeapType(spec);
  helper_type->tp_weaklistoffset = offsetof(GetItemHelper, weakrefs);
  pybind11::class_<GetItemHelper> helper_class(
      pybind11::reinterpret_steal<pybind11::object>(
          reinterpret_cast<PyObject*>(helper_type)));
  DisallowInstantiationFromPython(helper_class);
  helper_class.attr("__qualname__") = pybind11::str(qualname);
  helper_class.attr("__module__") = pybind11::str(module_name);
  scope.attr(helper_class_name) = helper_class;
  {
    pybind11::options options;
    options.disable_function_signatures();
    pybind11::class_<GetItemHelper>(scope).def_property_readonly(
        method_name,
        pybind11::cpp_function(
            [helper_type](pybind11::object self) -> pybind11::object {
              PyObject* ptr = helper_type->tp_alloc(helper_type, 0);
              if (!ptr) throw pybind11::error_already_set();
              auto& obj = *reinterpret_cast<GetItemHelper*>(ptr);
              obj.parent = self.release().ptr();
              return pybind11::reinterpret_steal<pybind11::object>(ptr);
            },
            absl::StrFormat("(arg0: object) -> %s.%s\n", module_name, qualname)
                .c_str()),
        "");
  }
  helper_class.def("__repr__", [method_name](const GetItemHelper& self) {
    return absl::StrFormat(
        "%s.%s", pybind11::repr(pybind11::handle(self.parent)), method_name);
  });
  helper_class.attr("__iter__") = pybind11::none();
  return {std::move(helper_class)};
}

/// Defines a new wrapper property on the specified class.
///
/// \tparam Parent Self type to use with `c`.
/// \tparam Tag Unique tag type for this wrapper property.  A unique combination
///     of `Parent` and `Tag` must be used for each call to
///     `DefineSubscriptMethod`.
/// \param c The pybind11 class for which to define the wrapper property.
/// \param method_name The name of the wrapper property.
/// \param helper_class_name The name of the class used to represent the wrapper
///     property, defined as a nested type within `c`.
/// \returns A `GetItemHelperClass` object that may be used to define
///     `__getitem__` and `__setitem__` methods on the wrapper type.
template <typename Parent = void, typename Tag = void>
GetItemHelperClass DefineSubscriptMethod(pybind11::detail::generic_type* c,
                                         const char* method_name,
                                         const char* helper_class_name) {
  return DefineSubscriptMethodImpl(*c, method_name, helper_class_name);
}

}  // namespace internal_python
}  // namespace tensorstore

namespace pybind11 {
namespace detail {

template <>
struct type_caster<tensorstore::internal_python::GetItemHelper>
    : public tensorstore::internal_python::HeapTypeCaster<
          tensorstore::internal_python::GetItemHelper> {
  bool load(handle src, bool /*convert*/) {
    if (Py_TYPE(src.ptr())->tp_dealloc !=
        &tensorstore::internal_python::GetItemHelper::Dealloc) {
      return false;
    }
    value = reinterpret_cast<tensorstore::internal_python::GetItemHelper*>(
        src.ptr());
    return true;
  }
};

}  // namespace detail
}  // namespace pybind11

#endif  // PYTHON_TENSORSTORE_SUBSCRIPT_METHOD_H_
